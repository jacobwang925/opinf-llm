#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parametric Burgers: Step 2 - Train OpInf Models

Loads pre-generated FOM dataset and trains OpInf models for each ν.
Uses joint POD basis across all ν values.

Usage:
    python parametric_burgers_2_train_model.py --dataset burgers_data.pkl.gz --output burgers_model.pkl

Author: Jacob Wang
"""

import numpy as np
import pickle
import gzip
import argparse
import os

# ----------------------------
# Command-line arguments
# ----------------------------
parser = argparse.ArgumentParser(description="Train OpInf models from FOM dataset")
parser.add_argument("--dataset", type=str, required=True,
                    help="Input dataset file (from step 1)")
parser.add_argument("--output", type=str, default="burgers_model.pkl",
                    help="Output model file")
parser.add_argument("--n_modes", type=int, default=10,
                    help="Number of POD modes")
parser.add_argument("--ridge_alpha", type=float, default=0.5,
                    help="Ridge regularization parameter (default: 0.5)")
parser.add_argument("--quad_ridge", action="store_false", dest="quad_ridge", default=False,
                    help="Apply ridge to ALL terms (default: False for all, use --quad_ridge to enable quad-only)")
args = parser.parse_args()

print("=" * 60)
print("Parametric Burgers: OpInf Training")
print("=" * 60)
print(f"Loading dataset from: {args.dataset}")

# ----------------------------
# Load dataset
# ----------------------------
# Auto-detect gzip compression by checking magic bytes
def load_pickle_auto(filepath):
    """Load pickle file, auto-detecting gzip compression."""
    with open(filepath, 'rb') as f:
        # Check for gzip magic number (1f 8b)
        magic = f.read(2)
        f.seek(0)

        if magic == b'\x1f\x8b':
            # File is gzip-compressed
            with gzip.open(filepath, 'rb') as gz:
                return pickle.load(gz)
        else:
            # Regular pickle file
            return pickle.load(f)

dataset = load_pickle_auto(args.dataset)

# Handle both old and new unified formats
if "config" in dataset:
    # Old format
    config = dataset["config"]
    nu_list = config["nu_list"]
    t_eval = dataset["t_eval"]
    x_fine = dataset["x_fine"]
    per_nu_data = dataset["per_nu_data"]
    all_train_Y = dataset["all_train_Y"]
else:
    # New unified format
    metadata = dataset["metadata"]
    grid = dataset["grid"]
    per_nu_data_unified = dataset["per_nu_data"]

    # Extract data
    nu_list = [item['nu'] for item in per_nu_data_unified]
    t_eval = grid['t']
    x_fine = grid['x']

    # Convert unified format to expected format
    per_nu_data = []
    all_train_Y = []
    for item in per_nu_data_unified:
        # Keep the structure with 'train', 'val', 'test' keys
        per_nu_data.append(item)
        all_train_Y.append(item['train']['Y'])

print(f"✓ Dataset loaded")
print(f"  Nu values: {nu_list}")
print(f"  Training snapshots per nu: ~{all_train_Y[0].shape[1] if all_train_Y else 0}")
print(f"  Spatial points: {len(x_fine)}")
print(f"  Time steps: {len(t_eval)}")
print("=" * 60)

# ----------------------------
# Joint POD basis
# ----------------------------
print("\nBuilding joint POD basis...")
Ysnap_joint = np.hstack(all_train_Y)
dx = x_fine[1] - x_fine[0]

Cmat = (Ysnap_joint @ Ysnap_joint.T) / Ysnap_joint.shape[1]
eigv, eigvec = np.linalg.eigh(Cmat)
modes = np.argsort(eigv)[::-1][:args.n_modes]
phi = eigvec[:, modes] / np.sqrt(dx)

print(f"✓ POD basis: {phi.shape}")
print(f"  Captured energy: {eigv[modes].sum() / eigv.sum() * 100:.2f}%")

# Projection/lifting functions
def project(Y):
    return phi.T @ (Y * dx)

def lift(a):
    return phi @ a

# ----------------------------
# Time derivatives (5-point stencil)
# ----------------------------
dt = t_eval[1] - t_eval[0]

def five_point_dot(arr):
    d = np.zeros_like(arr)
    d[0]    = (-25*arr[0]+48*arr[1]-36*arr[2]+16*arr[3]-3*arr[4])/(12*dt)
    d[1]    = (-3*arr[0]-10*arr[1]+18*arr[2]-6*arr[3]+arr[4])/(12*dt)
    d[2:-2] = (arr[:-4]-8*arr[1:-3]+8*arr[3:-1]-arr[4:])/(12*dt)
    d[-2]   = (-arr[-5]+6*arr[-4]-18*arr[-3]+10*arr[-2]+3*arr[-1])/(12*dt)
    d[-1]   = (3*arr[-5]-16*arr[-4]+36*arr[-3]-48*arr[-2]+25*arr[-1])/(12*dt)
    return d

def five_point_dot_multi(A):
    return np.vstack([five_point_dot(A[i]) for i in range(args.n_modes)])

# ----------------------------
# OpInf feature matrix
# ----------------------------
def build_feature_matrix(a_all, w1_all, w2_all, w3_all):
    """Build feature matrix: [vec(a⊗a), a, w1, w2, w3, 1]"""
    Mtot = a_all.shape[1]
    n_states = args.n_modes
    nF = n_states*n_states + n_states + 3 + 1

    D = np.zeros((Mtot, nF))
    for k in range(Mtot):
        xi = a_all[:, k]
        # quadratic
        D[k, :n_states*n_states] = np.outer(xi, xi).ravel()
        # linear
        idx = n_states*n_states
        D[k, idx:idx+n_states] = xi
        # inputs
        idx += n_states
        D[k, idx+0] = w1_all[k]
        D[k, idx+1] = w2_all[k]
        D[k, idx+2] = w3_all[k]
        # bias
        D[k, -1] = 1.0
    return D

# ----------------------------
# Fit OpInf for one nu
# ----------------------------
def fit_opinf_quadratic(Y, w1, w2, w3):
    """Fit quadratic OpInf: ȧ = C + A*a + H(a⊗a) + B*[w1,w2,w3]"""
    if Y.size == 0:
        raise RuntimeError("Empty training Y passed to OpInf.")

    a_all = project(Y)
    adot_all = five_point_dot_multi(a_all).T
    D = build_feature_matrix(a_all, w1, w2, w3)

    # Ridge-regularized normal equations (NO normalization - matches notebook)
    G = D.T @ D

    # Check conditioning before regularization
    cond_before = np.linalg.cond(G)

    if args.quad_ridge:
        qdim = args.n_modes * args.n_modes
        # Apply regularization to quadratic terms only
        G[:qdim, :qdim] += args.ridge_alpha**2 * np.eye(qdim)
    else:
        # Apply regularization to ALL terms
        G += args.ridge_alpha**2 * np.eye(G.shape[0])

    # Check conditioning after regularization
    cond_after = np.linalg.cond(G)

    Dt = D.T
    H = np.zeros((args.n_modes, args.n_modes, args.n_modes))
    A = np.zeros((args.n_modes, args.n_modes))
    Bin = np.zeros((args.n_modes, 3))
    C = np.zeros(args.n_modes)

    for i in range(args.n_modes):
        rhs_i = Dt @ adot_all[:, i]
        theta = np.linalg.solve(G, rhs_i)

        idx = 0
        H[i] = theta[idx: idx+args.n_modes*args.n_modes].reshape(args.n_modes, args.n_modes)
        idx += args.n_modes * args.n_modes
        A[i] = theta[idx: idx+args.n_modes]
        idx += args.n_modes
        Bin[i, 0:3] = theta[idx: idx+3]
        idx += 3
        C[i] = theta[idx]

    return H, A, Bin, C, cond_before, cond_after

# ----------------------------
# Train models for all nu
# ----------------------------
print("\nTraining OpInf models...")
per_nu_models = []

for nu, data in zip(nu_list, per_nu_data):
    print(f"\n[ν={nu:g}]")
    Ytr = data["train"]["Y"]

    # Handle both old and unified format for inputs
    if "w1" in data["train"]:
        # Old format
        w1 = data["train"]["w1"]
        w2 = data["train"]["w2"]
        w3 = data["train"]["w3"]
    else:
        # Unified format: inputs stored in U dict
        U = data["train"]["U"]
        w1 = U["w1_bc"]
        w2 = U["w2_bc"]
        w3 = U["source"]

    if Ytr.size == 0:
        print(f"  ⚠ No training data, skipping")
        continue

    H, A, B_in, C, cond_before, cond_after = fit_opinf_quadratic(Ytr, w1, w2, w3)

    eigvals = np.linalg.eigvals(A)
    print(f"  ✓ Trained")
    print(f"    Gram matrix cond: {cond_before:.2e} → {cond_after:.2e} (after ridge)")
    print(f"    A eigenvalues: Re(min)={eigvals.real.min():.3e}, Re(max)={eigvals.real.max():.3e}")
    print(f"    max|A|={abs(A).max():.3e}, max|H|={abs(H).max():.3e}")

    per_nu_models.append({
        "nu": nu,
        "H": H,
        "A": A,
        "B": B_in,
        "C": C
    })

# ----------------------------
# Save model
# ----------------------------
model_data = {
    "phi": phi,
    "per_nu_models": per_nu_models,
    "config": {
        "nu_list": nu_list,
        "N": config["N"] if "config" in dataset else 64,  # Default for unified format
        "T": config["T"] if "config" in dataset else (t_eval[-1] - t_eval[0]),
        "M": config["M"] if "config" in dataset else len(t_eval),
        "n_modes": args.n_modes,
        "ridge_alpha": args.ridge_alpha,
        "quad_ridge": args.quad_ridge,
    },
    "x_fine": x_fine,
    "t_eval": t_eval,
    "n_modes": args.n_modes,
    "ridge_alpha": args.ridge_alpha,
}

print("\n" + "=" * 60)
print("Saving model...")
with open(args.output, 'wb') as f:
    pickle.dump(model_data, f, protocol=pickle.HIGHEST_PROTOCOL)

file_size = os.path.getsize(args.output) / 1024
print(f"✓ Model saved to {args.output} ({file_size:.1f} KB)")
print("\nModel Summary:")
print(f"  POD basis: {phi.shape}")
print(f"  Trained on {len(per_nu_models)} nu values: {[m['nu'] for m in per_nu_models]}")
print(f"  Ridge alpha: {args.ridge_alpha}")
print(f"  Quad ridge only: {args.quad_ridge}")
print("\n✓ Done!")
