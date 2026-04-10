#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parametric Heat Equation: Step 2 - Train OpInf Models

Trains parametric ROM using Operator Inference with:
- Joint POD basis across all ν values
- Per-parameter linear OpInf models (C, A, B)

Heat ROM form: ȧ = C + A·a + B·u

Usage:
    python parametric_heat_2_train_model.py \\
        --dataset heat_dataset_train.pkl.gz \\
        --n_modes 6 \\
        --ridge_alpha 1.0 \\
        --output heat_model.pkl

Author: Jacob Wang
"""

import numpy as np
import pickle
import gzip
import argparse

# ----------------------------
# Helper functions
# ----------------------------
def load_dataset(filepath):
    """Load dataset (handles gzip automatically)."""
    with gzip.open(filepath, 'rb') as f:
        return pickle.load(f)

def five_point_dot(y, t):
    """Five-point finite difference for time derivative."""
    k = len(t)
    dt = np.diff(t)
    dydt = np.zeros_like(y)

    # Forward difference for first two points
    dydt[:, 0] = (-3*y[:, 0] + 4*y[:, 1] - y[:, 2]) / (2*dt[0])
    dydt[:, 1] = (-3*y[:, 1] + 4*y[:, 2] - y[:, 3]) / (2*dt[1])

    # Central differences for middle points
    for j in range(2, k-2):
        dydt[:, j] = (
            (-y[:, j+2] + 8*y[:, j+1] - 8*y[:, j-1] + y[:, j-2]) /
            (12*dt[j])
        )

    # Backward difference for last two points
    dydt[:, -2] = (3*y[:, -2] - 4*y[:, -3] + y[:, -4]) / (2*dt[-2])
    dydt[:, -1] = (3*y[:, -1] - 4*y[:, -2] + y[:, -3]) / (2*dt[-1])

    return dydt

# ----------------------------
# POD
# ----------------------------
def compute_joint_pod(snapshot_lists, r, dx):
    """Compute joint POD basis from concatenated snapshots."""
    # Concatenate all snapshots
    all_snapshots = np.hstack(snapshot_lists)

    # Spatial covariance matrix (method from reference code)
    n_snaps = all_snapshots.shape[1]
    C = (all_snapshots @ all_snapshots.T) / n_snaps

    # Eigenvalue decomposition
    eigvals, eigvecs = np.linalg.eigh(C)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    # POD modes (L2-orthonormal: phi.T @ phi * dx = I)
    Phi = eigvecs[:, :r] / np.sqrt(dx)

    # Energy capture
    total_energy = np.sum(eigvals)
    captured_energy = np.sum(eigvals[:r])
    energy_fraction = captured_energy / total_energy

    return Phi, energy_fraction

# ----------------------------
# OpInf training
# ----------------------------
def train_opinf_linear(A_reduced, Adot_reduced, U_inputs, ridge_alpha=1.0):
    """
    Train linear OpInf model: ȧ = C + A·a + B·u

    Args:
        A_reduced: (r, K_total) reduced states
        Adot_reduced: (r, K_total) reduced time derivatives
        U_inputs: (K_total,) boundary input values
        ridge_alpha: regularization parameter

    Returns:
        C: (r,) constant term
        A: (r, r) linear operator
        B: (r, 1) input operator
    """
    r = A_reduced.shape[0]
    K = A_reduced.shape[1]

    # Build data matrix D = [1, a, u]
    ones = np.ones((1, K))
    u_row = U_inputs.reshape(1, -1)

    D = np.vstack([
        ones,           # constant
        A_reduced,      # state
        u_row          # input
    ])  # Shape: (1 + r + 1, K)

    # Ridge-regularized least squares (matching reference implementation)
    G = D @ D.T  # (1+r+1, 1+r+1)
    if ridge_alpha > 0:
        G += (ridge_alpha**2) * np.eye(G.shape[0])

    # Solve: adot = Theta @ D, so Theta = adot @ D.T @ inv(D @ D.T)
    Theta = (Adot_reduced @ D.T) @ np.linalg.inv(G)  # (r, 1+r+1)

    # Extract operators
    C = Theta[:, 0]  # (r,)
    A = Theta[:, 1:1+r]  # (r, r)
    B = Theta[:, 1+r:]  # (r, 1)

    return C, A, B

# ----------------------------
# Main training workflow
# ----------------------------
def main():
    parser = argparse.ArgumentParser(description="Train parametric heat OpInf model")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Path to dataset from step 1")
    parser.add_argument("--n_modes", type=int, default=6,
                        help="Number of POD modes")
    parser.add_argument("--ridge_alpha", type=float, default=1.0,
                        help="Ridge regularization parameter")
    parser.add_argument("--output", type=str, default="heat_model.pkl",
                        help="Output model file")
    args = parser.parse_args()

    print("=" * 60)
    print("Parametric Heat Equation: Training")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"POD modes: {args.n_modes}")
    print(f"Ridge alpha: {args.ridge_alpha}")
    print("=" * 60)

    # Load dataset
    print("\nLoading dataset...")
    dataset = load_dataset(args.dataset)

    # Support both legacy heat dataset format and unified metadata/grid format.
    if "config" in dataset and "x_grid" in dataset and "t_eval" in dataset:
        config = dataset["config"]
        nu_values = config["nu_values"]
        x_grid = dataset["x_grid"]
        t_eval = dataset["t_eval"]
        dx = dataset["dx"]

        normalized_per_nu_data = []
        for nu_data in dataset["per_nu_data"]:
            normalized_per_nu_data.append({
                "nu": nu_data["nu"],
                "Y_train": nu_data["Y_train"],
                "U_train": nu_data["U_train"],
            })
    else:
        # Unified format:
        # {
        #   "metadata": {...},
        #   "grid": {"x","t","dx","dt"},
        #   "per_nu_data": [{"nu", "train", ...}, ...]
        # }
        grid = dataset["grid"]
        metadata = dataset.get("metadata", {})
        per_nu_raw = dataset["per_nu_data"]

        x_grid = grid["x"]
        t_eval = grid["t"]
        dx = grid["dx"]
        nu_values = [item["nu"] for item in per_nu_raw]

        input_names = metadata.get("input_names", [])
        default_u_name = input_names[0] if input_names else None

        normalized_per_nu_data = []
        for item in per_nu_raw:
            train = item["train"]
            Y_train = train["Y"]
            U_dict = train["U"]

            if isinstance(Y_train, list):
                Y_train_list = Y_train
            else:
                Y_train_list = [Y_train]

            if default_u_name and default_u_name in U_dict:
                U_train = U_dict[default_u_name]
            else:
                # Fallback to first available input key.
                u_key = next(iter(U_dict.keys()))
                U_train = U_dict[u_key]

            if isinstance(U_train, list):
                U_train_list = U_train
            else:
                U_train_list = [U_train]

            normalized_per_nu_data.append({
                "nu": item["nu"],
                "Y_train": Y_train_list,
                "U_train": U_train_list,
            })

    print(f"✓ Loaded {len(nu_values)} ν values: {nu_values}")
    print(f"  Spatial points: {len(x_grid)}")
    print(f"  Time points: {len(t_eval)}")

    # Step 1: Compute joint POD basis
    print(f"\nComputing joint POD basis ({args.n_modes} modes)...")

    all_train_snapshots = []
    for nu_data in normalized_per_nu_data:
        for Y_train in nu_data["Y_train"]:
            all_train_snapshots.append(Y_train)

    phi, energy_frac = compute_joint_pod(all_train_snapshots, args.n_modes, dx)

    print(f"✓ POD basis computed")
    print(f"  Energy captured: {energy_frac*100:.2f}%")

    # Step 2: Train per-ν OpInf models
    print(f"\nTraining OpInf models...")

    per_nu_models = []

    for nu_data in normalized_per_nu_data:
        nu = nu_data["nu"]
        Y_train_list = nu_data["Y_train"]
        U_train_list = nu_data["U_train"]

        print(f"\n  [ν={nu:g}]")

        # Project training data to reduced space
        A_reduced_list = []
        Adot_reduced_list = []
        U_concat_list = []

        for Y_train, U_train in zip(Y_train_list, U_train_list):
            # Project to reduced space
            A_reduced = phi.T @ (Y_train * dx)  # (r, K)

            # Compute time derivative
            Adot_reduced = five_point_dot(A_reduced, t_eval)

            A_reduced_list.append(A_reduced)
            Adot_reduced_list.append(Adot_reduced)
            U_concat_list.append(U_train)

        # Concatenate all training trajectories
        A_reduced_all = np.hstack(A_reduced_list)
        Adot_reduced_all = np.hstack(Adot_reduced_list)
        U_all = np.concatenate(U_concat_list)

        print(f"    Training data shape: {A_reduced_all.shape}")

        # Train OpInf
        C, A, B = train_opinf_linear(
            A_reduced_all, Adot_reduced_all, U_all,
            ridge_alpha=args.ridge_alpha
        )

        print(f"    ✓ Operators trained")
        print(f"      C: {C.shape}, norm={np.linalg.norm(C):.3e}")
        print(f"      A: {A.shape}, Frobenius norm={np.linalg.norm(A):.3e}")
        print(f"      B: {B.shape}, norm={np.linalg.norm(B):.3e}")

        per_nu_models.append({
            "nu": nu,
            "C": C,
            "A": A,
            "B": B,
        })

    # Step 3: Save model
    print(f"\nSaving model to: {args.output}")

    model_data = {
        "phi": phi,
        "per_nu_models": per_nu_models,
        "config": {
            "nu_values": nu_values,
            "n_modes": args.n_modes,
            "ridge_alpha": args.ridge_alpha,
            "energy_fraction": energy_frac,
        },
        "x_grid": x_grid,
        "t_eval": t_eval,
        "dx": dx,
        "n_modes": args.n_modes,
    }

    with open(args.output, 'wb') as f:
        pickle.dump(model_data, f)

    print("✓ Model saved")
    print("=" * 60)
    print("Training complete!")
    print("=" * 60)
    print(f"\nModel summary:")
    print(f"  Joint POD modes: {args.n_modes}")
    print(f"  Energy captured: {energy_frac*100:.2f}%")
    print(f"  ν values: {nu_values}")
    print(f"  Ridge alpha: {args.ridge_alpha}")
    print(f"\nNext: Test the model with")
    print(f"  python parametric_heat_3_test_model.py --model {args.output}")

if __name__ == "__main__":
    main()
