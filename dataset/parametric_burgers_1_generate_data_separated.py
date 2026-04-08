#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parametric Burgers: Step 1 - Generate FOM Dataset (SEPARATED TRAIN/TEST)

Generates and saves FOM snapshot data for multiple ν values.
Separates training and testing data into different files for clarity.

Burgers equation: u_t + u*u_x = nu*u_xx + s(x)*w3(t)
BCs: u(0,t) = w1(t), u(1,t) = w2(t)

Usage:
    # Generate both train and test
    python parametric_burgers_1_generate_data_separated.py --mode both

    # Generate only training data
    python parametric_burgers_1_generate_data_separated.py --mode train

    # Generate only test data (for training nu values)
    python parametric_burgers_1_generate_data_separated.py --mode test

    # Generate test data for additional nu values (interpolation/extrapolation)
    python parametric_burgers_1_generate_data_separated.py --mode test \
        --test_nu_values 0.03 0.07 0.12 \
        --output_test burgers_dataset_test_interp.pkl.gz

Author: Jacob Wang
"""

import numpy as np
import numpy.random as rnd
from scipy.integrate import solve_ivp
from numpy.polynomial.chebyshev import Chebyshev
import pickle
import gzip
import argparse
from pathlib import Path

# ----------------------------
# Command-line arguments
# ----------------------------
parser = argparse.ArgumentParser(description="Generate FOM dataset for parametric Burgers (separated train/test)")
parser.add_argument("--mode", type=str, default="both", choices=["train", "test", "both"],
                    help="Generate 'train', 'test', or 'both' datasets")
parser.add_argument("--output_train", type=str, default="burgers_dataset_train.pkl.gz",
                    help="Output file for training dataset")
parser.add_argument("--output_test", type=str, default="burgers_dataset_test.pkl.gz",
                    help="Output file for test dataset")
parser.add_argument("--train_nu_values", nargs='+', type=float, default=[0.01, 0.02, 0.05, 0.1],
                    help="List of viscosity values for training")
parser.add_argument("--test_nu_values", nargs='+', type=float, default=[0.03, 0.07, 0.12],
                    help="List of viscosity values for testing (default: [0.03, 0.07, 0.12] for interpolation/extrapolation)")
parser.add_argument("--num_train", type=int, default=100,
                    help="Number of training trajectories per nu")
parser.add_argument("--num_val", type=int, default=20,
                    help="Number of validation trajectories per nu")
parser.add_argument("--num_test", type=int, default=20,
                    help="Number of test trajectories per nu")
parser.add_argument("--T_test", type=float, default=None,
                    help="Time horizon for test data (default: 2.0, or use this to extend horizon)")
parser.add_argument("--M_test", type=int, default=None,
                    help="Time steps for test data (default: 1001, or specify for extended horizon)")
parser.add_argument("--seed", type=int, default=42,
                    help="Base random seed")
args = parser.parse_args()

# Set test time parameters (can be different from training)
if args.T_test is None:
    args.T_test = 2.0  # Default same as training
if args.M_test is None:
    args.M_test = 1001  # Default same as training

# ----------------------------
# Global config
# ----------------------------
N      = 64            # Chebyshev degree
T      = 2.0           # Time horizon for training
M      = 1001          # Time steps for training
t_eval = np.linspace(0.0, T, M)
dt     = t_eval[1] - t_eval[0]
x_fine = np.linspace(0.0, 1.0, 1001)

# Test time grid (can be extended)
T_test = args.T_test
M_test = args.M_test
t_eval_test = np.linspace(0.0, T_test, M_test)
dt_test = t_eval_test[1] - t_eval_test[0]

print("=" * 70)
print("Parametric Burgers: FOM Data Generation (SEPARATED TRAIN/TEST)")
print("=" * 70)
print(f"Mode: {args.mode}")
if args.mode in ["train", "both"]:
    print(f"Training Nu values: {args.train_nu_values}")
    print(f"Training trajectories per nu: {args.num_train}")
    print(f"Validation trajectories per nu: {args.num_val}")
    print(f"Training time horizon: T = {T}, M = {M}")
if args.mode in ["test", "both"]:
    print(f"Test Nu values: {args.test_nu_values}")
    print(f"Test trajectories per nu: {args.num_test}")
    print(f"Test time horizon: T = {T_test}, M = {M_test}")
    if T_test > T:
        print(f"  ⚠ Extended horizon: {T_test/T:.1f}× training time")
print(f"Spatial points: {len(x_fine)}")
print("=" * 70)

# ----------------------------
# Chebyshev matrices
# ----------------------------
def cheb_matrices(N):
    k = np.arange(N+1)
    s = np.cos(np.pi*k/N)
    c = np.ones_like(s); c[0]=c[-1]=2; c *= (-1)**k
    S = np.tile(s,(N+1,1)); dS = S - S.T
    D = np.outer(c,1/c)/(dS + np.eye(N+1))
    D -= np.diag(np.sum(D,axis=1))
    Dx = 2.0*D
    D2x = 4.0*(D@D)
    order = slice(None,None,-1)
    x    = 0.5*(s+1.0)[order]
    Dx   = -Dx[order,:][:,order]
    D2x  = D2x[order,:][:,order]
    return x, Dx, D2x

x_cheb, Dx, D2x = cheb_matrices(N)

# ----------------------------
# Burgers RHS
# ----------------------------
def burgers_rhs(t, u, Dx, D2x, nu, s_vec, w1, w2, w3):
    u = u.copy()
    u[0], u[-1] = w1(t), w2(t)
    ux  = Dx @ u
    uxx = D2x @ u
    return -u * ux + nu * uxx + s_vec * w3(t)

# ----------------------------
# Random multi-sine factories
# ----------------------------
num_components = 3
def make_multisine_fn_normal():
    amps   = rnd.uniform(0.1, 1.0, num_components)
    freqs  = rnd.uniform(0.2, 5.0, num_components)
    phases = rnd.uniform(0, 2*np.pi, num_components)
    bias   = rnd.uniform(-0.5, 0.5)
    def fn(t):
        return bias + sum(a*np.sin(2*np.pi*f*t + p) for a, f, p in zip(amps, freqs, phases))
    def dfn(t):
        return sum(a*2*np.pi*f*np.cos(2*np.pi*f*t + p) for a, f, p in zip(amps, freqs, phases))
    return fn, dfn

def make_multisine_fn_highmag():
    amps   = rnd.uniform(0.1, 3.0, num_components)
    freqs  = rnd.uniform(0.2, 1.0, num_components)
    phases = rnd.uniform(0, 2*np.pi, num_components)
    bias   = rnd.uniform(-1.0, 1.0)
    def fn(t):
        return bias + sum(a*np.sin(2*np.pi*f*t + p) for a, f, p in zip(amps, freqs, phases))
    def dfn(t):
        return sum(a*2*np.pi*f*np.cos(2*np.pi*f*t + p) for a, f, p in zip(amps, freqs, phases))
    return fn, dfn

# ----------------------------
# Interpolation to fine grid
# ----------------------------
def interp_to_fine(sol, x_nodes=x_cheb, x_out=x_fine):
    Y = np.zeros((x_out.size, sol.y.shape[1]))
    for j in range(sol.y.shape[1]):
        p = Chebyshev.fit(x_nodes, sol.y[:, j], deg=len(x_nodes)-1)
        Y[:, j] = p(x_out)
    return Y

# ----------------------------
# Initial condition
# ----------------------------
def init_cond(w1, w2):
    return w2(0) + 0.5*(w1(0)-w2(0))*(1 - np.tanh((x_cheb-0.3)/0.1))

s_vec = np.cosh((x_cheb-0.5)/0.05)**(-1)

# ----------------------------
# Generate dataset for one nu (TRAINING)
# ----------------------------
def generate_train_dataset_for_nu(nu_value, seed_offset=0):
    print(f"\n[ν={nu_value:g}] Generating training dataset...")
    rnd.seed(args.seed + seed_offset)

    def gen_and_solve_trajectory():
        """Generate and solve a single trajectory."""
        w1, dot_w1 = make_multisine_fn_normal()
        w2, dot_w2 = make_multisine_fn_normal()
        w3, _      = make_multisine_fn_highmag()

        sol = solve_ivp(
            burgers_rhs, [0, T], init_cond(w1, w2),
            args=(Dx, D2x, nu_value, s_vec, w1, w2, w3),
            t_eval=t_eval, method="RK45", rtol=1e-6, atol=1e-8
        )

        if sol.t.size != M or (not sol.success) or (not np.all(np.isfinite(sol.y))):
            return None

        Y = interp_to_fine(sol)
        w1_vals = w1(t_eval)
        w2_vals = w2(t_eval)
        w3_vals = w3(t_eval)

        return Y, w1_vals, w2_vals, w3_vals

    # Training data
    print(f"  Generating {args.num_train} training cases...", end='', flush=True)
    train_Y_list = []
    train_w1_list = []
    train_w2_list = []
    train_w3_list = []
    skipped = 0

    for i in range(args.num_train):
        result = gen_and_solve_trajectory()
        if result is not None:
            Y, w1_vals, w2_vals, w3_vals = result
            train_Y_list.append(Y)
            train_w1_list.append(w1_vals)
            train_w2_list.append(w2_vals)
            train_w3_list.append(w3_vals)
        else:
            skipped += 1

    print(f" done! ({len(train_Y_list)} successful, {skipped} skipped)")

    # Combine training snapshots
    if len(train_Y_list) > 0:
        train_Y_combined = np.hstack(train_Y_list)  # (space, time_snapshots)
        train_w1_combined = np.hstack(train_w1_list)
        train_w2_combined = np.hstack(train_w2_list)
        train_w3_combined = np.hstack(train_w3_list)
    else:
        train_Y_combined = np.empty((len(x_fine), 0))
        train_w1_combined = np.empty((0,))
        train_w2_combined = np.empty((0,))
        train_w3_combined = np.empty((0,))

    train_data = {
        'Y': train_Y_combined,
        'w1': train_w1_combined,
        'w2': train_w2_combined,
        'w3': train_w3_combined
    }

    # Validation data
    val_data = None
    if args.num_val > 0:
        print(f"  Generating {args.num_val} validation cases...", end='', flush=True)
        val_Y_list = []
        val_w1_list = []
        val_w2_list = []
        val_w3_list = []
        skipped = 0

        for i in range(args.num_val):
            result = gen_and_solve_trajectory()
            if result is not None:
                Y, w1_vals, w2_vals, w3_vals = result
                val_Y_list.append(Y)
                val_w1_list.append(w1_vals)
                val_w2_list.append(w2_vals)
                val_w3_list.append(w3_vals)
            else:
                skipped += 1

        print(f" done! ({len(val_Y_list)} successful, {skipped} skipped)")

        if len(val_Y_list) > 0:
            val_Y_combined = np.hstack(val_Y_list)
            val_w1_combined = np.hstack(val_w1_list)
            val_w2_combined = np.hstack(val_w2_list)
            val_w3_combined = np.hstack(val_w3_list)
            val_data = {
                'Y': val_Y_combined,
                'w1': val_w1_combined,
                'w2': val_w2_combined,
                'w3': val_w3_combined
            }

    # Test data - stored as lists for easy individual access
    # Use extended time horizon for test data
    def gen_and_solve_trajectory_test():
        """Generate and solve test trajectory with extended time horizon."""
        w1, dot_w1 = make_multisine_fn_normal()
        w2, dot_w2 = make_multisine_fn_normal()
        w3, _      = make_multisine_fn_highmag()

        # Use extended time horizon for test data
        sol = solve_ivp(
            burgers_rhs, [0, T_test], init_cond(w1, w2),
            args=(Dx, D2x, nu_value, s_vec, w1, w2, w3),
            t_eval=t_eval_test, method="RK45", rtol=1e-6, atol=1e-8
        )

        if sol.t.size != M_test or (not sol.success) or (not np.all(np.isfinite(sol.y))):
            return None

        Y = interp_to_fine(sol)
        w1_vals = w1(t_eval_test)
        w2_vals = w2(t_eval_test)
        w3_vals = w3(t_eval_test)

        return Y, w1_vals, w2_vals, w3_vals

    print(f"  Generating {args.num_test} test cases (extended T={T_test:.1f})...", end='', flush=True)
    test_Y_list = []
    test_w1_list = []
    test_w2_list = []
    test_w3_list = []
    skipped = 0

    for i in range(args.num_test):
        result = gen_and_solve_trajectory_test()
        if result is not None:
            Y, w1_vals, w2_vals, w3_vals = result
            test_Y_list.append(Y)
            test_w1_list.append(w1_vals)
            test_w2_list.append(w2_vals)
            test_w3_list.append(w3_vals)
        else:
            skipped += 1

    print(f" done! ({len(test_Y_list)} successful, {skipped} skipped)")

    # Create entry
    entry_data = {
        'train': train_data,
        'lists': {
            'Y_test': test_Y_list,
            'w1_test': test_w1_list,
            'w2_test': test_w2_list,
            'w3_test': test_w3_list
        }
    }
    if val_data is not None:
        entry_data['val'] = val_data

    return entry_data

# ----------------------------
# Generate dataset for one nu (TEST ONLY)
# ----------------------------
def generate_test_dataset_for_nu(nu_value, seed_offset=0):
    print(f"\n[ν={nu_value:g}] Generating test dataset...")
    rnd.seed(args.seed + seed_offset)

    def gen_and_solve_trajectory():
        """Generate and solve a single trajectory with extended time horizon."""
        w1, dot_w1 = make_multisine_fn_normal()
        w2, dot_w2 = make_multisine_fn_normal()
        w3, _      = make_multisine_fn_highmag()

        # Use extended time horizon for test data
        sol = solve_ivp(
            burgers_rhs, [0, T_test], init_cond(w1, w2),
            args=(Dx, D2x, nu_value, s_vec, w1, w2, w3),
            t_eval=t_eval_test, method="RK45", rtol=1e-6, atol=1e-8
        )

        if sol.t.size != M_test or (not sol.success) or (not np.all(np.isfinite(sol.y))):
            return None

        Y = interp_to_fine(sol)
        w1_vals = w1(t_eval_test)
        w2_vals = w2(t_eval_test)
        w3_vals = w3(t_eval_test)

        return Y, w1_vals, w2_vals, w3_vals

    # Test data - stored as lists for easy individual access
    print(f"  Generating {args.num_test} test cases...", end='', flush=True)
    Y_test_list = []
    w1_test_list = []
    w2_test_list = []
    w3_test_list = []
    skipped = 0

    for i in range(args.num_test):
        result = gen_and_solve_trajectory()
        if result is not None:
            Y, w1_vals, w2_vals, w3_vals = result
            Y_test_list.append(Y)
            w1_test_list.append(w1_vals)
            w2_test_list.append(w2_vals)
            w3_test_list.append(w3_vals)
        else:
            skipped += 1

    print(f" done! ({len(Y_test_list)} successful, {skipped} skipped)")

    # Create entry with lists for test data
    entry_data = {
        'lists': {
            'Y_test_list': Y_test_list,
            'w1_test_list': w1_test_list,
            'w2_test_list': w2_test_list,
            'w3_test_list': w3_test_list
        }
    }

    return entry_data

# ----------------------------
# Generate training dataset
# ----------------------------
if args.mode in ["train", "both"]:
    print("\n" + "=" * 70)
    print("GENERATING TRAINING DATASET")
    print("=" * 70)

    per_nu_data = []
    all_train_Y = []

    for j, nu in enumerate(args.train_nu_values):
        entry = generate_train_dataset_for_nu(nu, seed_offset=j*1000)
        per_nu_data.append(entry)
        if entry['train']['Y'].size > 0:
            all_train_Y.append(entry['train']['Y'])

    train_dataset = {
        'config': {
            'nu_list': args.train_nu_values,
            'N': N,
            'T': T,
            'M': M,
            'T_test': T_test,  # Extended time horizon for test data
            'M_test': M_test,  # Extended time steps for test data
            'x_fine_len': len(x_fine),
            'num_train': args.num_train,
            'num_val': args.num_val,
            'num_test': args.num_test,
            'seed': args.seed
        },
        't_eval': t_eval,  # Training time grid
        't_eval_test': t_eval_test,  # Extended test time grid
        'x_fine': x_fine,
        'per_nu_data': per_nu_data,
        'all_train_Y': all_train_Y
    }

    # Save training dataset
    print("\n" + "=" * 70)
    print("Saving training dataset...")
    print("=" * 70)

    with gzip.open(args.output_train, 'wb') as f:
        pickle.dump(train_dataset, f, protocol=pickle.HIGHEST_PROTOCOL)

    size_mb = Path(args.output_train).stat().st_size / (1024 * 1024)
    print(f"✓ Saved to {args.output_train} ({size_mb:.1f} MB)")

    # Print summary
    print("\nTraining Dataset Summary:")
    print(f"  Nu values: {args.train_nu_values}")
    total_size = 0
    for i, nu in enumerate(args.train_nu_values):
        entry = per_nu_data[i]
        train_size = entry['train']['Y'].nbytes / 1e6
        val_size = entry['val']['Y'].nbytes / 1e6 if 'val' in entry and entry['val'] is not None else 0
        total = train_size + val_size
        total_size += total
        print(f"  ν={nu:g}: train={train_size:.1f}MB, val={val_size:.1f}MB (total={total:.1f}MB)")
    print(f"  Total uncompressed: {total_size:.1f} MB")

# ----------------------------
# Generate test dataset
# ----------------------------
if args.mode in ["test", "both"]:
    print("\n" + "=" * 70)
    print("GENERATING TEST DATASET")
    print("=" * 70)

    per_nu_data = []

    for j, nu in enumerate(args.test_nu_values):
        # Use different seed offset for test data to ensure different ICBC
        # For nu in training set: use offset 10000 + j*1000
        # For nu not in training set: use offset 100 + j*1000
        if nu in args.train_nu_values:
            # Test data for training nu - use large offset to ensure different from training
            seed_offset = 10000 + args.train_nu_values.index(nu) * 1000
        else:
            # Test data for unseen nu - use standard offset
            seed_offset = 100 + j * 1000

        entry = generate_test_dataset_for_nu(nu, seed_offset=seed_offset)
        per_nu_data.append(entry)

    test_dataset = {
        'config': {
            'nu_list': args.test_nu_values,
            'N': N,
            'T': T_test,  # Extended time horizon for test data
            'M': M_test,  # Extended time steps for test data
            'T_train': T,  # Original training time horizon
            'M_train': M,  # Original training time steps
            'x_fine_len': len(x_fine),
            'num_test': args.num_test,
            'seed': args.seed
        },
        't_eval': t_eval_test,  # Extended time grid
        't_eval_train': t_eval,  # Original training time grid
        'x_fine': x_fine,
        'per_nu_data': per_nu_data
    }

    # Save test dataset
    print("\n" + "=" * 70)
    print("Saving test dataset...")
    print("=" * 70)

    with gzip.open(args.output_test, 'wb') as f:
        pickle.dump(test_dataset, f, protocol=pickle.HIGHEST_PROTOCOL)

    size_mb = Path(args.output_test).stat().st_size / (1024 * 1024)
    print(f"✓ Saved to {args.output_test} ({size_mb:.1f} MB)")

    # Print summary
    print("\nTest Dataset Summary:")
    print(f"  Nu values: {args.test_nu_values}")
    total_size = 0
    for i, nu in enumerate(args.test_nu_values):
        entry = per_nu_data[i]
        test_size = sum(Y.nbytes for Y in entry['lists']['Y_test_list']) / 1e6
        total_size += test_size
        print(f"  ν={nu:g}: test={test_size:.1f}MB ({len(entry['lists']['Y_test_list'])} cases)")
    print(f"  Total uncompressed: {total_size:.1f} MB")

# ----------------------------
# Final summary
# ----------------------------
print("\n" + "=" * 70)
print("COMPLETE")
print("=" * 70)
if args.mode in ["train", "both"]:
    print(f"✓ Training dataset: {args.output_train}")
    print("  Next: Train model with")
    print(f"    python parametric_burgers_2_train_model.py --dataset {args.output_train}")
if args.mode in ["test", "both"]:
    print(f"✓ Test dataset: {args.output_test}")
    print("  Next: Test model with")
    print(f"    python parametric_burgers_3_test_model.py --model burgers_model.pkl --dataset {args.output_test}")
print("=" * 70)
