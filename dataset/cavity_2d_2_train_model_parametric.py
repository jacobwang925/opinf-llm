#!/usr/bin/env python3
"""
Parametric 2D Cavity Flow: Step 2 - Train OpInf Models

Loads pre-generated FOM dataset and trains OpInf models for each Re.
Uses joint POD basis across all Re values.

Usage:
    python cavity_2d_2_train_model_parametric.py --dataset cavity_dataset_train.pkl.gz --output cavity_model.pkl

Author: Jacob Wang
"""

import numpy as np
import pickle
import gzip
import argparse

# ----------------------------
# Command-line arguments
# ----------------------------
parser = argparse.ArgumentParser(description="Train parametric OpInf models from FOM dataset")
parser.add_argument("--dataset", type=str, required=True,
                    help="Input dataset file (from step 1)")
parser.add_argument("--output", type=str, default="cavity_model.pkl",
                    help="Output model file")
parser.add_argument("--n_modes", type=int, default=20,
                    help="Number of POD modes")
parser.add_argument("--alpha", type=float, default=1.0,
                    help="Ridge regularization for linear terms (default: 1.0)")
parser.add_argument("--quad_alpha", type=float, default=10.0,
                    help="Ridge regularization for quadratic terms (default: 10.0)")
args = parser.parse_args()

print("=" * 70)
print("Parametric 2D Cavity Flow: OpInf Training")
print("=" * 70)
print(f"Loading dataset from: {args.dataset}")

# ----------------------------
# Load dataset
# ----------------------------

def load_pickle_auto(filepath):
    """Load pickle file, auto-detecting gzip compression."""
    with open(filepath, 'rb') as f:
        magic = f.read(2)
        f.seek(0)

        if magic == b'\x1f\x8b':
            with gzip.open(filepath, 'rb') as gz:
                return pickle.load(gz)
        else:
            return pickle.load(f)

dataset = load_pickle_auto(args.dataset)

# Extract data
config = dataset["config"]
Re_list = config["Re_list"]
N = config["N"]
t_eval = dataset["t_eval"]
x = dataset["x"]
y = dataset["y"]
dx = dataset["dx"]
per_Re_data = dataset["per_Re_data"]
all_train_Y = dataset["all_train_Y"]

print(f"✓ Dataset loaded")
print(f"  Reynolds numbers: {Re_list}")
print(f"  Grid: {N+2}×{N+2} = {(N+2)**2} DOFs per field")
print(f"  Time steps: {len(t_eval)}")
print(f"  Training data for {len(Re_list)} Re values")
print("=" * 70)


# =============================================================================
# Joint POD Basis (across all Re values)
# =============================================================================

print("\nComputing joint POD basis...")

# Concatenate all training data horizontally
Y_joint = np.hstack(all_train_Y)  # (2*n_spatial, total_snapshots)

# 2D spatial integration weight
dA = dx * dx

# POD via eigendecomposition
Cmat = (Y_joint @ Y_joint.T) / Y_joint.shape[1]
eigv, eigvec = np.linalg.eigh(Cmat)
modes = np.argsort(eigv)[::-1][:args.n_modes]
phi = eigvec[:, modes] / np.sqrt(dA)  # L2-normalized

energy_captured = eigv[modes].sum() / eigv.sum()

print(f"✓ POD basis: {phi.shape}")
print(f"  Modes: {args.n_modes}")
print(f"  Captured energy: {energy_captured * 100:.2f}%")


# =============================================================================
# Projection and Time Derivatives
# =============================================================================

def project(Y):
    """Project snapshot matrix onto POD basis."""
    return phi.T @ (Y * dA)  # (r, M)

def lift(a):
    """Lift modal coefficients to full space."""
    return phi @ a  # (2*n_spatial, M)

def time_derivative_fd5(a, dt):
    """5-point finite difference for time derivatives - EXACT copy from single-Re."""
    r, M = a.shape
    adot = np.zeros_like(a)

    for i in range(r):
        a_i = a[i, :]
        adot[i, 0] = (-25*a_i[0] + 48*a_i[1] - 36*a_i[2] + 16*a_i[3] - 3*a_i[4]) / (12*dt)
        adot[i, 1] = (-3*a_i[0] - 10*a_i[1] + 18*a_i[2] - 6*a_i[3] + a_i[4]) / (12*dt)
        adot[i, 2:-2] = (a_i[:-4] - 8*a_i[1:-3] + 8*a_i[3:-1] - a_i[4:]) / (12*dt)
        adot[i, -2] = (3*a_i[-1] + 10*a_i[-2] - 18*a_i[-3] + 6*a_i[-4] - a_i[-5]) / (12*dt)
        adot[i, -1] = (25*a_i[-1] - 48*a_i[-2] + 36*a_i[-3] - 16*a_i[-4] + 3*a_i[-5]) / (12*dt)

    return adot


# =============================================================================
# OpInf Feature Matrix (Quadratic)
# =============================================================================

def build_feature_matrix_quadratic(a, U_lid):
    """
    Build feature matrix for quadratic OpInf: ȧ = C + A·a + H(a⊗a) + B·U_lid

    Args:
        a: (r, M) modal coefficients
        U_lid: (M,) lid velocity inputs

    Returns:
        D: (M, r²+r+2) feature matrix
    """
    r, M = a.shape
    D = np.zeros((M, r*r + r + 2))

    for k in range(M):
        a_k = a[:, k]
        # Quadratic terms: vec(a⊗a)
        D[k, :r*r] = np.outer(a_k, a_k).ravel()
        # Linear terms
        D[k, r*r:r*r+r] = a_k
        # Input
        D[k, r*r+r] = U_lid[k]
        # Bias
        D[k, -1] = 1.0

    return D


# =============================================================================
# Ridge Regression with Differential Regularization
# =============================================================================

def fit_opinf_quadratic(a_all, adot_all, U_lid_all, alpha, quad_alpha):
    """
    Fit quadratic OpInf: ȧ = C + A·a + H(a⊗a) + B·U_lid

    Uses differential regularization:
    - Stronger regularization (quad_alpha) on quadratic terms
    - Moderate regularization (alpha) on linear/input/bias terms

    Args:
        a_all: (r, M) modal coefficients
        adot_all: (r, M) time derivatives
        U_lid_all: (M,) lid velocity inputs

    Returns:
        H: (r, r, r) quadratic operator
        A: (r, r) linear operator
        B: (r,) input operator
        C: (r,) constant term
    """
    r = a_all.shape[0]
    D = build_feature_matrix_quadratic(a_all, U_lid_all)  # (M, features)

    # Differential regularization matrix
    reg_matrix = np.eye(D.shape[1])
    reg_matrix[:r*r, :r*r] *= quad_alpha**2  # Quadratic terms
    reg_matrix[r*r:, r*r:] *= alpha**2        # Linear, input, bias

    # Solve least squares with regularization
    G = D.T @ D + reg_matrix

    H = np.zeros((r, r, r))
    A = np.zeros((r, r))
    B = np.zeros(r)
    C = np.zeros(r)

    for i in range(r):
        theta = np.linalg.solve(G, D.T @ adot_all[i, :])
        H[i] = theta[:r*r].reshape(r, r)
        A[i] = theta[r*r:r*r+r]
        B[i] = theta[r*r+r]
        C[i] = theta[-1]

    return H, A, B, C


# =============================================================================
# Train OpInf Models for Each Re
# =============================================================================

print("\nTraining OpInf models for each Re...")
print("=" * 70)

dt = t_eval[1] - t_eval[0]
M = len(t_eval)

per_Re_models = []

for idx, Re_data in enumerate(per_Re_data):
    Re = Re_data["Re"]
    print(f"\nReynolds number: Re = {Re}")
    print("-" * 70)

    # Extract training data for this Re
    train_omega = Re_data["train"]["Y_omega"]
    train_psi = Re_data["train"]["Y_psi"]
    train_U = Re_data["train"]["U_lid"]

    if train_omega is None or train_omega.size == 0:
        print("  ✗ No training data available for this Re")
        continue

    # Combine omega and psi
    Y_combined = np.vstack([train_omega, train_psi])

    print(f"  Training snapshots: {Y_combined.shape[1]}")

    # CRITICAL: Process each trajectory SEPARATELY to avoid incorrect time derivatives
    # at trajectory boundaries
    num_traj = train_omega.shape[1] // M
    print(f"  Processing {num_traj} trajectories separately...")

    a_list = []
    adot_list = []

    for i in range(num_traj):
        # Extract single trajectory
        Y_omega_i = train_omega[:, i*M:(i+1)*M]
        Y_psi_i = train_psi[:, i*M:(i+1)*M]
        Y_i = np.vstack([Y_omega_i, Y_psi_i])

        # Project and compute derivatives for this trajectory
        a_i = project(Y_i)
        adot_i = time_derivative_fd5(a_i, dt)

        a_list.append(a_i)
        adot_list.append(adot_i)

    # Concatenate all trajectories
    a_train = np.hstack(a_list)
    adot_train = np.hstack(adot_list)

    print(f"  Modal coefficients: {a_train.shape}")
    print(f"  Time derivatives: {adot_train.shape}")

    # Fit OpInf operators
    print(f"  Fitting OpInf operators (α={args.alpha}, quad_α={args.quad_alpha})...")
    H, A, B, C = fit_opinf_quadratic(
        a_train,
        adot_train,
        train_U,
        args.alpha,
        args.quad_alpha
    )

    print(f"  ✓ Operators fitted")
    print(f"    ||H|| = {np.linalg.norm(H):.2e}")
    print(f"    ||A|| = {np.linalg.norm(A):.2e}")
    print(f"    ||B|| = {np.linalg.norm(B):.2e}")
    print(f"    ||C|| = {np.linalg.norm(C):.2e}")

    # Store model for this Re
    per_Re_models.append({
        "Re": Re,
        "H": H,
        "A": A,
        "B": B,
        "C": C
    })


# =============================================================================
# Save Model
# =============================================================================

model = {
    "config": {
        "Re_list": Re_list,
        "N": N,
        "grid_size": N + 2,
        "n_modes": args.n_modes,
        "alpha": args.alpha,
        "quad_alpha": args.quad_alpha,
        "energy_captured": energy_captured
    },
    "phi": phi,
    "per_Re_models": per_Re_models,
    "x": x,
    "y": y,
    "dx": dx,
    "t_eval": t_eval
}

print("\n" + "=" * 70)
print(f"Saving model to: {args.output}")

with open(args.output, 'wb') as f:
    pickle.dump(model, f, protocol=4)

print("✓ Model saved successfully")
print("=" * 70)
print("\nModel Summary:")
print("-" * 70)
print(f"  POD modes: {args.n_modes} ({energy_captured*100:.2f}% energy)")
print(f"  Feature matrix size: {args.n_modes**2 + args.n_modes + 2} features")
print(f"  Regularization: α={args.alpha}, quad_α={args.quad_alpha}")
print(f"  Trained models for {len(per_Re_models)} Re values:")
for model_data in per_Re_models:
    Re = model_data["Re"]
    H = model_data["H"]
    A = model_data["A"]
    print(f"    Re={Re}: ||H||={np.linalg.norm(H):.2e}, ||A||={np.linalg.norm(A):.2e}")
print("=" * 70)
print("\n✓ Training complete!")
