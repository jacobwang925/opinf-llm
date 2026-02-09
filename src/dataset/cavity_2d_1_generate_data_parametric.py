#!/usr/bin/env python3
"""
Parametric 2D Cavity Flow: Step 1 - Generate FOM Dataset

Generates and saves FOM snapshot data for multiple Reynolds numbers.
Separates training and testing data into different files.

Usage:
    # Generate both train and test
    python cavity_2d_1_generate_data_parametric.py --mode both

    # Generate only training data
    python cavity_2d_1_generate_data_parametric.py --mode train

    # Generate test data for interpolation/extrapolation
    python cavity_2d_1_generate_data_parametric.py --mode test \
        --test_Re_values 250 700 1500

Author: Jacob Wang
"""

import numpy as np
import numpy.random as rnd
from scipy.sparse import diags, kron
from scipy.sparse.linalg import spsolve
import pickle
import gzip
import argparse
from datetime import datetime
from pathlib import Path

# ----------------------------
# Command-line arguments
# ----------------------------
parser = argparse.ArgumentParser(description="Generate FOM dataset for parametric 2D cavity (separated train/test)")
parser.add_argument("--mode", type=str, default="both", choices=["train", "test", "both"],
                    help="Generate 'train', 'test', or 'both' datasets")
parser.add_argument("--output_train", type=str, default="cavity_dataset_train.pkl.gz",
                    help="Output file for training dataset")
parser.add_argument("--output_test", type=str, default="cavity_dataset_test.pkl.gz",
                    help="Output file for test dataset")
parser.add_argument("--train_Re_values", nargs='+', type=float, default=[50, 75, 100, 125, 150],
                    help="List of Reynolds numbers for training")
parser.add_argument("--test_Re_values", nargs='+', type=float, default=[40, 60, 80, 90, 110, 120, 140, 160],
                    help="List of Reynolds numbers for testing (interpolation/extrapolation)")
parser.add_argument("--num_train", type=int, default=8,
                    help="Number of training trajectories per Re")
parser.add_argument("--num_val", type=int, default=2,
                    help="Number of validation trajectories per Re")
parser.add_argument("--num_test", type=int, default=2,
                    help="Number of test trajectories per Re")
parser.add_argument("--test_T_factor", type=float, default=2.0,
                    help="Factor to extend test time horizon (default: 2.0 for [0, 2T])")
parser.add_argument("--seed", type=int, default=42,
                    help="Base random seed")
args = parser.parse_args()

# FIXED SEED for reproducibility - EXACTLY as in single-Re script
rnd.seed(args.seed)
np.random.seed(args.seed)

# ----------------------------
# Global config - EXACTLY as in single-Re script
# ----------------------------
N = 32          # Interior points (34×34 total grid)
T_final = 2.0
dt_fom = 0.001  # FOM timestep

print("=" * 70)
print("Parametric 2D Cavity Flow: FOM Data Generation (SEPARATED TRAIN/TEST)")
print("=" * 70)
print(f"Mode: {args.mode}")
if args.mode in ["train", "both"]:
    print(f"Training Re values: {args.train_Re_values}")
    print(f"Training trajectories per Re: {args.num_train}")
    print(f"Validation trajectories per Re: {args.num_val}")
if args.mode in ["test", "both"]:
    print(f"Test Re values: {args.test_Re_values}")
    print(f"Test trajectories per Re: {args.num_test}")
print(f"Grid: {N+2}×{N+2} = {(N+2)**2} points per field")
print(f"Time horizon: T = {T_final}")
print(f"FOM timestep: dt = {dt_fom}")
print("=" * 70)
print()


# =============================================================================
# 2D Finite Difference Operators - EXACT COPY from single-Re script
# =============================================================================

def build_2d_operators_fd(N, L=1.0):
    """Build 2D finite difference operators using Kronecker products."""
    x = np.linspace(0, L, N + 2)
    y = np.linspace(0, L, N + 2)
    dx = L / (N + 1)

    # 1D Laplacian (interior points only)
    main_diag = -2 * np.ones(N)
    off_diag = np.ones(N - 1)
    D2_1d = diags([off_diag, main_diag, off_diag], [-1, 0, 1], format='csr') / dx**2

    # 2D Laplacian via Kronecker sum
    I_N = diags([np.ones(N)], [0], format='csr')
    Laplacian = kron(I_N, D2_1d) + kron(D2_1d, I_N)

    return x, y, dx, Laplacian


# =============================================================================
# FOM Solver - EXACT COPY from single-Re script
# =============================================================================

def solve_cavity_fom(
    Re,
    N,
    T_final,
    dt,
    lid_profile=None,
    f_t=None,
    omega_ic=None,
    dt_out=None,
    verbose=False,
):
    """
    Solve 2D cavity FOM with a fixed lid profile scaled by a time-varying input.

    Returns:
        omega_snaps: (N+2)×(N+2)×M vorticity snapshots
        psi_snaps: (N+2)×(N+2)×M streamfunction snapshots
        t_array: Time array
        f_array: (M,) scalar input at each snapshot time
    """
    x, y, dx, Laplacian_interior = build_2d_operators_fd(N)

    nu = 1.0 / Re
    n_steps = int(T_final / dt)
    if dt_out is None:
        save_freq = max(1, n_steps // 100)
    else:
        save_freq = max(1, int(round(dt_out / dt)))

    if lid_profile is None:
        lid_profile = np.ones_like(x)
    if f_t is None:
        f_t = 1.0
    f_t_is_callable = callable(f_t)

    # Initialize
    if omega_ic is None:
        omega = np.zeros((N + 2, N + 2))
    else:
        omega = omega_ic.copy()

    psi = np.zeros((N + 2, N + 2))

    omega_history = []
    psi_history = []
    t_array = []
    f_history = []

    for step in range(n_steps):
        t = step * dt

        f_val = f_t(t) if f_t_is_callable else f_t
        U_lid = lid_profile * f_val

        if step % save_freq == 0:
            omega_history.append(omega.copy())
            psi_history.append(psi.copy())
            t_array.append(t)
            f_history.append(f_val)

        # Solve Poisson: ∇²ψ = -ω
        omega_interior = omega[1:-1, 1:-1].ravel(order='C')
        psi_interior = spsolve(Laplacian_interior, -omega_interior)
        psi[1:-1, 1:-1] = psi_interior.reshape(N, N, order='C')
        psi[[0,-1],:] = 0
        psi[:,[0,-1]] = 0

        # Velocities: u = ∂ψ/∂y, v = -∂ψ/∂x
        u = np.zeros_like(psi)
        v = np.zeros_like(psi)
        u[1:-1, 1:-1] = (psi[1:-1, 2:] - psi[1:-1, :-2]) / (2 * dx)
        v[1:-1, 1:-1] = -(psi[2:, 1:-1] - psi[:-2, 1:-1]) / (2 * dx)

        # Velocity BCs (lid profile scaled by time-varying input)
        u[-1, :] = U_lid
        u[[0],:] = 0
        u[:,[0,-1]] = 0
        v[:,:] = 0

        # Wall vorticity (no-slip with time-varying lid)
        omega[0, 1:-1] = -2 * psi[1, 1:-1] / dx**2
        omega[-1, 1:-1] = -2 * (psi[-2, 1:-1] - U_lid[1:-1] * dx) / dx**2
        omega[1:-1, 0] = -2 * psi[1:-1, 1] / dx**2
        omega[1:-1, -1] = -2 * psi[1:-1, -2] / dx**2
        omega[[0,0,-1,-1],[0,-1,0,-1]] = 0  # Corners

        # Time step vorticity
        omega_new = omega.copy()
        omega_xx = (omega[1:-1, 2:] - 2*omega[1:-1, 1:-1] + omega[1:-1, :-2]) / dx**2
        omega_yy = (omega[2:, 1:-1] - 2*omega[1:-1, 1:-1] + omega[:-2, 1:-1]) / dx**2
        diffusion = nu * (omega_xx + omega_yy)

        omega_x = (omega[1:-1, 2:] - omega[1:-1, :-2]) / (2 * dx)
        omega_y = (omega[2:, 1:-1] - omega[:-2, 1:-1]) / (2 * dx)
        advection = -(u[1:-1, 1:-1] * omega_x + v[1:-1, 1:-1] * omega_y)

        omega_new[1:-1, 1:-1] += dt * (diffusion + advection)
        omega = omega_new

        # Stability check
        if np.max(np.abs(omega)) > 1e6:
            if verbose:
                print(f"  FOM unstable at t={t:.3f}")
            return None, None, None, None

    omega_snaps = np.stack(omega_history, axis=-1)
    psi_snaps = np.stack(psi_history, axis=-1)
    t_array = np.array(t_array)
    f_array = np.array(f_history)

    return omega_snaps, psi_snaps, t_array, f_array


# =============================================================================
# Random Initial Conditions - Dummy for RNG state consistency
# =============================================================================

def random_vorticity_ic(N):
    """Generate random initial vorticity - NOT USED, just advances RNG state."""
    x = np.linspace(0, 1, N + 2)
    y = np.linspace(0, 1, N + 2)
    X, Y = np.meshgrid(x, y, indexing='ij')

    omega = np.zeros((N + 2, N + 2))

    # Add random Gaussian vortex blobs
    n_blobs = rnd.randint(2, 6)
    for _ in range(n_blobs):
        x_c = rnd.uniform(0.2, 0.8)
        y_c = rnd.uniform(0.2, 0.8)
        strength = rnd.uniform(-10, 10)
        sigma = rnd.uniform(0.05, 0.15)

        omega += strength * np.exp(-((X - x_c)**2 + (Y - y_c)**2) / (2 * sigma**2))

    return omega


# =============================================================================
# Time-Varying Input Generator
# =============================================================================

def make_random_input_signal():
    """
    Generate a random time-varying scalar input f(t).

    Returns callable f(t) with sinusoidal variation:
        f(t) = base + amplitude * sin(2π * frequency * t + phase)

    Distribution matches train/test:
    - base: [0.7, 1.3]
    - amplitude: [0.1, 0.4]
    - frequency: [0.5, 2.0] Hz
    - phase: [0, 2π]
    """
    base = rnd.uniform(0.7, 1.3)
    amplitude = rnd.uniform(0.1, 0.4)
    frequency = rnd.uniform(0.5, 2.0)
    phase = rnd.uniform(0, 2*np.pi)

    def f_t(t):
        return base + amplitude * np.sin(2 * np.pi * frequency * t + phase)

    f_t.params = {
        'base': base,
        'amplitude': amplitude,
        'frequency': frequency,
        'phase': phase
    }

    return f_t


# =============================================================================
# Data Generation Loop - EXACT LOGIC from single-Re script
# =============================================================================

def generate_dataset_for_split(
    Re,
    N,
    T_final,
    dt_fom,
    num_traj,
    split_name,
    lid_profile,
    dt_out=None,
):
    """Generate FOM data with a fixed lid profile and time-varying scalar input."""
    print(f"  {split_name.capitalize()} for Re={Re}:")
    print("  " + "-" * 66)

    omega_list = []
    psi_list = []
    f_list = []  # Store f(t) arrays for each trajectory

    for i in range(num_traj):
        f_t_fn = make_random_input_signal()
        params = f_t_fn.params

        print(
            f"    Trajectory {i+1}/{num_traj} "
            f"(base={params['base']:.2f}, amp={params['amplitude']:.2f}, "
            f"f={params['frequency']:.2f}Hz)...",
            end=' '
        )

        # ZERO initial condition for reproducibility
        # (RNG still advances for consistency)
        _ = random_vorticity_ic(N)  # Advance RNG state but don't use it

        # Solve FOM with fixed lid profile and time-varying input, ZERO IC
        omega_snaps, psi_snaps, t_array, f_array = solve_cavity_fom(
            Re,
            N,
            T_final,
            dt_fom,
            lid_profile=lid_profile,
            f_t=f_t_fn,
            dt_out=dt_out,
        )

        if omega_snaps is not None:
            # Flatten spatial dimensions (2D -> 1D for OpInf)
            omega_flat = omega_snaps.reshape(-1, omega_snaps.shape[-1], order='C')
            psi_flat = psi_snaps.reshape(-1, psi_snaps.shape[-1], order='C')

            omega_list.append(omega_flat)
            psi_list.append(psi_flat)
            f_list.append(f_array)  # Store time-varying scalar input array

            # Quick check
            max_omega = np.max(np.abs(omega_flat))
            f_mean = np.mean(f_array)
            f_std = np.std(f_array)
            print(f"✓ (|ω|_max = {max_omega:.2e}, f(t): {f_mean:.2f}±{f_std:.2f})")
        else:
            print(f"FAILED")

    print(f"  Generated {len(omega_list)}/{num_traj} trajectories")

    # Concatenate all trajectories
    if len(omega_list) > 0:
        Y_omega = np.hstack(omega_list)  # (n_spatial, M * num_successful)
        Y_psi = np.hstack(psi_list)

        # f(t) values - time-varying per trajectory
        U_lid_all = np.concatenate(f_list)
        t_out = t_array  # Use the time array from FOM
    else:
        Y_omega = None
        Y_psi = None
        U_lid_all = None
        t_out = None

    return Y_omega, Y_psi, U_lid_all, t_out


# =============================================================================
# Main Dataset Generation - PARAMETRIC VERSION
# =============================================================================

def generate_parametric_dataset(
    Re_list,
    num_train,
    num_val,
    num_test,
    mode_name,
    lid_profile,
    T_final_local,
    dt_out=None,
):
    """Generate FOM data for multiple Re values."""

    print(f"\nGenerating {mode_name.upper()} data...")
    print("=" * 70)

    per_Re_data = []
    all_train_Y = []  # For joint POD
    t_final = None  # Will be set from first successful split

    for Re in Re_list:
        print(f"\nReynolds number: Re = {Re}")
        print("-" * 70)

        # Generate train/val/test for this Re
        if num_train > 0:
            train_omega, train_psi, train_U, t_train = generate_dataset_for_split(
                Re,
                N,
                T_final_local,
                dt_fom,
                num_train,
                "train",
                lid_profile,
                dt_out=dt_out,
            )
        else:
            train_omega, train_psi, train_U, t_train = None, None, None, None

        if num_val > 0:
            val_omega, val_psi, val_U, t_val = generate_dataset_for_split(
                Re,
                N,
                T_final_local,
                dt_fom,
                num_val,
                "validation",
                lid_profile,
                dt_out=dt_out,
            )
        else:
            val_omega, val_psi, val_U, t_val = None, None, None, None

        if num_test > 0:
            test_omega, test_psi, test_U, t_test = generate_dataset_for_split(
                Re,
                N,
                T_final_local,
                dt_fom,
                num_test,
                "test",
                lid_profile,
                dt_out=dt_out,
            )
        else:
            test_omega, test_psi, test_U, t_test = None, None, None, None

        # Get time array from whichever split succeeded
        if t_final is None:
            t_final = t_train if t_train is not None else (t_val if t_val is not None else t_test)

        # Store per-Re data
        per_Re_data.append({
            "Re": Re,
            "train": {
                "Y_omega": train_omega,
                "Y_psi": train_psi,
                "U_lid": train_U
            },
            "validation": {
                "Y_omega": val_omega,
                "Y_psi": val_psi,
                "U_lid": val_U
            },
            "test": {
                "Y_omega": test_omega,
                "Y_psi": test_psi,
                "U_lid": test_U
            }
        })

        # Accumulate training data for joint POD
        if train_omega is not None:
            Y_combined = np.vstack([train_omega, train_psi])
            all_train_Y.append(Y_combined)

    return per_Re_data, all_train_Y, t_final


# Build dataset structure
x, y, dx, _ = build_2d_operators_fd(N)

# Fixed non-symmetric lid profile a(x), scaled by time-varying f(t)
lid_profile = 1.0 + 0.3 * np.sin(2 * np.pi * x) + 0.2 * x
dt_out = T_final / 100.0

# Generate training data
if args.mode in ["train", "both"]:
    train_per_Re, train_all_Y, t_eval = generate_parametric_dataset(
        args.train_Re_values,
        args.num_train,
        args.num_val,
        args.num_test,
        "training",
        lid_profile,
        T_final,
        dt_out=dt_out,
    )

    train_dataset = {
        "config": {
            "Re_list": args.train_Re_values,
            "N": N,
            "grid_size": N + 2,
            "T": T_final,
            "dt_fom": dt_fom,
            "dt_out": dt_out,
            "num_train": args.num_train,
            "num_val": args.num_val,
            "num_test": args.num_test,
            "seed": args.seed,
            "generated_on": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        },
        "t_eval": t_eval,
        "x": x,
        "y": y,
        "dx": dx,
        "per_Re_data": train_per_Re,
        "all_train_Y": train_all_Y  # For joint POD
    }

    # Save training dataset
    print("\n" + "=" * 70)
    print(f"Saving training dataset to: {args.output_train}")
    with gzip.open(args.output_train, 'wb') as f:
        pickle.dump(train_dataset, f, protocol=4)
    print("✓ Training dataset saved")

    # Summary
    print("\nTraining Dataset Summary:")
    print("-" * 70)
    for item in train_per_Re:
        Re = item["Re"]
        if item["train"]["Y_omega"] is not None:
            M = t_eval.shape[0]
            n_traj = item["train"]["Y_omega"].shape[1] // M
            print(f"  Re={Re}: {n_traj} train trajectories × {M} time steps")
    print("=" * 70)

# Generate test data
if args.mode in ["test", "both"]:
    test_T_final = T_final * args.test_T_factor
    test_per_Re, test_all_Y, t_eval_test = generate_parametric_dataset(
        args.test_Re_values,
        0,
        0,
        args.num_test,
        "test",
        lid_profile,
        test_T_final,
        dt_out=dt_out,
    )

    test_dataset = {
        "config": {
            "Re_list": args.test_Re_values,
            "N": N,
            "grid_size": N + 2,
            "T": test_T_final,
            "dt_fom": dt_fom,
            "dt_out": dt_out,
            "num_test": args.num_test,
            "seed": args.seed + 1000,  # Different seed for test
            "generated_on": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        },
        "t_eval": t_eval_test,
        "x": x,
        "y": y,
        "dx": dx,
        "per_Re_data": test_per_Re
    }

    # Save test dataset
    print("\n" + "=" * 70)
    print(f"Saving test dataset to: {args.output_test}")
    with gzip.open(args.output_test, 'wb') as f:
        pickle.dump(test_dataset, f, protocol=4)
    print("✓ Test dataset saved")

    # Summary
    print("\nTest Dataset Summary:")
    print("-" * 70)
    for item in test_per_Re:
        Re = item["Re"]
        if item["test"]["Y_omega"] is not None:
            M = t_eval_test.shape[0]
            n_traj = item["test"]["Y_omega"].shape[1] // M
            print(f"  Re={Re}: {n_traj} test trajectories × {M} time steps")
    print("=" * 70)

print("\n✓ Data generation complete!")
