#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parametric Heat Equation: Step 1 - Generate FOM Dataset (SEPARATED TRAIN/TEST)

Generates and saves FOM snapshot data for multiple ν values.
Separates training and testing data into different files for clarity.

Heat equation: u_t = nu*u_xx, x in [0,1]
Dirichlet BC: u(0,t) = u(1,t) = u_bc(t)

Usage:
  python parametric_heat_1_generate_data_separated.py --mode both
  python parametric_heat_1_generate_data_separated.py --mode test \\
      --test_nu_values 0.5 1.0 3.0 --T_test 2.0 --M_test 2001
"""

import numpy as np
import numpy.random as rnd
from scipy.integrate import solve_ivp
from scipy.sparse import diags
import pickle
import gzip
import argparse
from pathlib import Path


parser = argparse.ArgumentParser(description="Generate FOM dataset for parametric heat (separated train/test)")
parser.add_argument("--mode", type=str, default="both", choices=["train", "test", "both"],
                    help="Generate 'train', 'test', or 'both' datasets")
parser.add_argument("--output_train", type=str, default="heat_dataset_train.pkl.gz",
                    help="Output file for training dataset")
parser.add_argument("--output_test", type=str, default="heat_dataset_test.pkl.gz",
                    help="Output file for test dataset")
parser.add_argument("--train_nu_values", nargs="+", type=float, default=[0.1, 0.5, 2.0],
                    help="List of diffusivity values for training")
parser.add_argument("--test_nu_values", nargs="+", type=float, default=[0.5, 1.0, 3.0],
                    help="List of diffusivity values for testing")
parser.add_argument("--num_train", type=int, default=100,
                    help="Number of training trajectories per ν")
parser.add_argument("--num_val", type=int, default=20,
                    help="Number of validation trajectories per ν")
parser.add_argument("--num_test", type=int, default=20,
                    help="Number of test trajectories per ν")
parser.add_argument("--T_test", type=float, default=2.0,
                    help="Time horizon for test data (default: 2.0)")
parser.add_argument("--M_test", type=int, default=2001,
                    help="Time steps for test data (default: 2001)")
parser.add_argument("--seed", type=int, default=42,
                    help="Random seed")
args = parser.parse_args()

rnd.seed(args.seed)

# Spatial grid
L = 1.0
n = 2**10 - 1
x_all = np.linspace(0, L, n + 2)
x_grid = x_all[1:-1]
dx = x_grid[1] - x_grid[0]

# Training time grid
T_train = 1.0
M_train = 1001
t_eval_train = np.linspace(0.0, T_train, M_train)
dt_train = t_eval_train[1] - t_eval_train[0]

# Test time grid
T_test = args.T_test
M_test = args.M_test
t_eval_test = np.linspace(0.0, T_test, M_test)

# Fixed exponential IC
alpha_ic = 100.0
u0_fixed = np.exp(alpha_ic * (x_grid - 1)) + np.exp(-alpha_ic * x_grid) - np.exp(-alpha_ic)

# FD matrix for d2/dx2
dx_val = 1.0 / n
main = np.full(n, -2.0 / dx_val**2)
upper = np.full(n-1, 1.0 / dx_val**2)
lower = np.full(n-1, 1.0 / dx_val**2)
D2 = diags([lower, main, upper], offsets=[-1, 0, 1]).tolil()
D2[0, 0] = 1.0 / dx_val**2
D2[0, 1] = -2.0 / dx_val**2
D2[0, 2] = 1.0 / dx_val**2
D2[-1, -3] = 1.0 / dx_val**2
D2[-1, -2] = -2.0 / dx_val**2
D2[-1, -1] = 1.0 / dx_val**2
D2x = D2.tocsr()


def generate_random_bc_params():
    amps = rnd.uniform(0.1, 0.5, 3)
    freqs = rnd.uniform(0.2, 5.0, 3)
    phases = rnd.uniform(0, 2*np.pi, 3)
    bias = rnd.uniform(0.8, 1.2)
    return amps, freqs, phases, bias


def make_bc_func(amps, freqs, phases, bias):
    def bc_func(t):
        val = bias
        for a, f, p in zip(amps, freqs, phases):
            val += a * np.sin(2 * np.pi * f * t + p)
        return val
    return bc_func


def solve_heat_trajectory(nu, bc_func, u0, t_eval):
    def heat_rhs(t, u):
        u[0] = bc_func(t)
        u[-1] = bc_func(t)
        return nu * (D2x @ u)

    sol = solve_ivp(
        heat_rhs,
        (t_eval[0], t_eval[-1]),
        u0.copy(),
        t_eval=t_eval,
        method="BDF",
        rtol=1e-6,
        atol=1e-8,
    )
    return sol.y if sol.success else None


def generate_split(nu_value, t_eval, n_cases, label):
    Y_list = []
    U_list = []
    skipped = 0
    for _ in range(n_cases):
        amps, freqs, phases, bias = generate_random_bc_params()
        bc_func = make_bc_func(amps, freqs, phases, bias)
        Y = solve_heat_trajectory(nu_value, bc_func, u0_fixed, t_eval)
        if Y is None:
            skipped += 1
            continue
        U = np.array([bc_func(t) for t in t_eval])
        Y_list.append(Y)
        U_list.append(U)
    print(f"  {label}: {len(Y_list)}/{n_cases} successful, {skipped} skipped")
    return Y_list, U_list


if args.mode in ["train", "both"]:
    per_nu_data = []
    print("=" * 60)
    print("Generating TRAIN dataset")
    print("=" * 60)
    for nu in args.train_nu_values:
        print(f"\n[ν={nu:g}]")
        y_train, u_train = generate_split(nu, t_eval_train, args.num_train, "train")
        y_val, u_val = generate_split(nu, t_eval_train, args.num_val, "val")
        y_test, u_test = generate_split(nu, t_eval_train, args.num_test, "test")

        entry = {
            "nu": nu,
            "lists": {
                "Y_train_list": y_train,
                "U_train_list": u_train,
                "Y_val_list": y_val,
                "U_val_list": u_val,
                "Y_test_list": y_test,
                "U_test_list": u_test,
            },
        }
        per_nu_data.append(entry)

    train_dataset = {
        "config": {
            "nu_list": args.train_nu_values,
            "T": T_train,
            "M": M_train,
            "x_len": len(x_grid),
            "num_train": args.num_train,
            "num_val": args.num_val,
            "num_test": args.num_test,
            "seed": args.seed,
        },
        "x_grid": x_grid,
        "t_eval": t_eval_train,
        "per_nu_data": per_nu_data,
    }

    with gzip.open(args.output_train, "wb") as f:
        pickle.dump(train_dataset, f, protocol=4)
    size_mb = Path(args.output_train).stat().st_size / (1024 * 1024)
    print(f"\n✓ Saved training dataset to {args.output_train} ({size_mb:.2f} MB)")


if args.mode in ["test", "both"]:
    per_nu_data = []
    print("\n" + "=" * 60)
    print("Generating TEST dataset")
    print("=" * 60)
    for nu in args.test_nu_values:
        print(f"\n[ν={nu:g}]")
        y_test, u_test = generate_split(nu, t_eval_test, args.num_test, "test")

        entry = {
            "nu": nu,
            "lists": {
                "Y_test_list": y_test,
                "U_test_list": u_test,
            },
        }
        per_nu_data.append(entry)

    test_dataset = {
        "config": {
            "nu_list": args.test_nu_values,
            "T": T_test,
            "M": M_test,
            "T_train": T_train,
            "M_train": M_train,
            "x_len": len(x_grid),
            "num_test": args.num_test,
            "seed": args.seed,
        },
        "x_grid": x_grid,
        "t_eval": t_eval_test,
        "t_eval_train": t_eval_train,
        "per_nu_data": per_nu_data,
    }

    with gzip.open(args.output_test, "wb") as f:
        pickle.dump(test_dataset, f, protocol=4)
    size_mb = Path(args.output_test).stat().st_size / (1024 * 1024)
    print(f"\n✓ Saved test dataset to {args.output_test} ({size_mb:.2f} MB)")
