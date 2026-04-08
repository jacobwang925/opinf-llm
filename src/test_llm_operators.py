#!/usr/bin/env python3
"""
Test LLM-Predicted Operators

This script loads LLM-predicted operators and tests them using the same
methodology as parametric_heat_3_test_model.py and parametric_burgers_3_test_model.py

Usage:
    python test_llm_operators.py --predicted tool_calling_operators_nu0.5.json --model heat_model.pkl
    python test_llm_operators.py --predicted tool_calling_operators_nu0.01.json --model burgers_model_test.pkl
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import pickle
import gzip
import json
import argparse
from pathlib import Path

# Parse arguments
parser = argparse.ArgumentParser(description="Test LLM-predicted operators")
parser.add_argument("--predicted", type=str, required=True,
                    help="LLM-predicted operators JSON file")
parser.add_argument("--model", type=str, required=True,
                    help="Model file (heat_model.pkl or burgers_model_test.pkl)")
parser.add_argument("--dataset", type=str, default=None,
                    help="Dataset file (optional, will try to find automatically)")
parser.add_argument("--save_plots", action="store_true",
                    help="Save plots instead of showing")
parser.add_argument("--save_raw", action="store_true",
                    help="Save raw prediction arrays for custom plotting")
parser.add_argument("--output_dir", type=str, default=".",
                    help="Directory to save plots (default: current directory)")
parser.add_argument("--test_T_factor", type=float, default=1.0,
                    help="Extend heat test horizon by this factor (default: 1.0)")
parser.add_argument("--no_use_exact_trained", action="store_true",
                    help="Disable using exact OpInf operators for seen parameters")
args = parser.parse_args()

print("=" * 70)
print("Testing LLM-Predicted Operators")
print("=" * 70)

# Load predicted operators
print(f"\nLoading predicted operators: {args.predicted}")
with open(args.predicted, 'r') as f:
    pred_data = json.load(f)

nu_test = pred_data["query_nu"]
predicted_ops = pred_data["predicted_operators"]["operators"]

# Convert to numpy
operators = {}
for op_name, op_data in predicted_ops.items():
    operators[op_name] = np.array(op_data)
    print(f"  {op_name}: shape={operators[op_name].shape}")

# Load model
print(f"\nLoading model: {args.model}")
with open(args.model, 'rb') as f:
    model_data = pickle.load(f)

phi = model_data["phi"]
x_grid = model_data.get("x_grid", model_data.get("x_fine"))
t_eval = model_data["t_eval"]
n_modes = model_data["n_modes"]
config = model_data["config"]

print(f"✓ Model loaded (POD modes: {n_modes})")

# Detect equation type
if 'H' in operators:
    equation_type = "burgers"
else:
    equation_type = "heat"

print(f"  Equation: {equation_type}")
print(f"  Test parameter: ν={nu_test}")

# Override operators for seen parameters if requested
use_exact_trained = not args.no_use_exact_trained
if use_exact_trained and "per_nu_models" in model_data:
    for item in model_data["per_nu_models"]:
        if np.isclose(item.get("nu", np.nan), nu_test, atol=1e-6):
            print("  Using exact OpInf operators (trained parameter)")
            if equation_type == "heat":
                operators = {
                    "A": np.array(item["A"]),
                    "B": np.array(item["B"]),
                    "C": np.array(item["C"]),
                }
            else:
                operators = {
                    "H": np.array(item["H"]),
                    "A": np.array(item["A"]),
                    "B": np.array(item["B"]),
                    "C": np.array(item["C"]),
                }
            break

# Helper functions (from working scripts)
dx = x_grid[1] - x_grid[0]
dt = t_eval[1] - t_eval[0]

def project(Y):
    return phi.T @ (Y * dx)

def lift(a):
    return phi @ a

def global_spatiotemporal_relative_l2(Yref, Yrom, dx, dt):
    """
    Compute global spatiotemporal relative L2 error (from working scripts):
    sqrt(∫∫ |Y_rom - Y_ref|^2 dx dt) / sqrt(∫∫ |Y_ref|^2 dx dt)
    """
    err = Yref - Yrom
    norm_err = np.sqrt(np.sum(err**2) * dx * dt)
    norm_ref = np.sqrt(np.sum(Yref**2) * dx * dt)
    return norm_err / norm_ref

# Load ground truth
if args.dataset:
    dataset_file = args.dataset
else:
    # Try to find dataset automatically (new unified format)
    if equation_type == "heat":
        dataset_file = "heat_dataset_unified.pkl.gz"
    else:
        dataset_file = "burgers_dataset_unified.pkl.gz"

# Get ground truth
Y_test = None
U_test = None
Y_tests = []
U_tests = []
force_heat_fom = False

# Try dataset file
dataset_t_eval = None
dataset_t_eval_train = None
t_eval_used = t_eval
if dataset_file and Path(dataset_file).exists():
    print(f"\nLoading ground truth from: {dataset_file}")

    # Load dataset (gzipped)
    with gzip.open(dataset_file, 'rb') as f:
        dataset = pickle.load(f)

    dataset_t_eval = dataset.get("t_eval")
    dataset_t_eval_train = dataset.get("t_eval_train")
    if dataset_t_eval is not None:
        t_eval_used = dataset_t_eval

    # New unified format: item['test'] is list of dicts with {Y, U}
    if "per_nu_data" in dataset:
        # Unified format with metadata
        if "metadata" in dataset:
            for item in dataset['per_nu_data']:
                if np.isclose(item['nu'], nu_test, atol=1e-6):
                    input_names = dataset['metadata']['input_names']
                    if 'test' in item and len(item['test']) > 0:
                        for test_case in item['test']:
                            Y_case = test_case['Y']
                            U_dict = test_case['U']

                            # Transpose if needed: should be (space, time)
                            if Y_case.shape[0] < Y_case.shape[1]:
                                Y_case = Y_case.T

                            if len(input_names) == 1:
                                U_case = U_dict[input_names[0]]
                            else:
                                U_case = tuple(U_dict[name] for name in input_names)

                            Y_tests.append(Y_case)
                            U_tests.append(U_case)
                        print(f"✓ Found {len(Y_tests)} ground truth trajectories for ν={nu_test}")
                        print(f"  Inputs: {', '.join(input_names)}")
                        break
                    # Fallback: use combined split data if test list not available
                    for split in ["test", "val", "train"]:
                        if split in item and isinstance(item[split], dict):
                            Y_comb = item[split]["Y"]
                            U_dict = item[split]["U"]
                            M = len(t_eval_used)
                            num_traj = Y_comb.shape[1] // M if M else 0
                            if num_traj == 0:
                                continue
                            for i in range(num_traj):
                                Y_case = Y_comb[:, i*M:(i+1)*M]
                                if len(input_names) == 1:
                                    U_arr = U_dict[input_names[0]]
                                    U_case = U_arr[i*M:(i+1)*M]
                                else:
                                    U_case = tuple(
                                        U_dict[name][i*M:(i+1)*M] for name in input_names
                                    )
                                Y_tests.append(Y_case)
                                U_tests.append(U_case)
                            print(f"✓ Found {len(Y_tests)} ground truth trajectories for ν={nu_test}")
                            print(f"  Inputs: {', '.join(input_names)}")
                            break
                    break
        else:
            # Separated parametric format: per_nu_data list with lists key
            nu_list = dataset.get("config", {}).get("nu_list", [])
            if nu_test in nu_list:
                idx = nu_list.index(nu_test)
                item = dataset["per_nu_data"][idx]
                if "lists" in item:
                    y_list = item["lists"].get("Y_test_list", [])
                    u_list = item["lists"].get("U_test_list", [])
                    w1_list = item["lists"].get("w1_test_list", [])
                    w2_list = item["lists"].get("w2_test_list", [])
                    w3_list = item["lists"].get("w3_test_list", [])
                    if y_list:
                        for idx_case, Y_case in enumerate(y_list):
                            if Y_case.shape[0] == len(dataset_t_eval):
                                Y_case = Y_case.T
                            if u_list:
                                U_case = u_list[idx_case]
                            elif w1_list and w2_list and w3_list:
                                U_case = (w1_list[idx_case], w2_list[idx_case], w3_list[idx_case])
                            else:
                                U_case = None
                            Y_tests.append(Y_case)
                            U_tests.append(U_case)
                        print(f"✓ Found {len(Y_tests)} ground truth trajectories for ν={nu_test}")
                        if U_tests and not isinstance(U_tests[0], tuple):
                            print("  Inputs: u_bc")
                        else:
                            print("  Inputs: w1_bc, w2_bc, source")

if not Y_tests and equation_type == "heat":
    # Generate FOM ground truth for unseen nu values (like test_heat_unseen_nu.py does)
    print(f"⚠ No pre-computed ground truth for ν={nu_test}")
    print(f"  Generating FOM solution for unseen parameter...")

    from scipy.sparse import diags

    # Setup FD solver (same as data generation)
    n = len(x_grid)
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

    alpha_ic = 100.0
    u0_fixed = np.exp(alpha_ic * (x_grid - 1)) + np.exp(-alpha_ic * x_grid) - np.exp(-alpha_ic)

    # Test BC function (same as test_heat_unseen_nu.py)
    test_bc = lambda t: 1.0 + 0.3*np.sin(4*np.pi*t) + 0.2*np.sin(8*np.pi*t)
    if args.test_T_factor > 1.0:
        T_train = t_eval[-1]
        T_test = T_train * args.test_T_factor
        M_test = int(round(T_test / dt)) + 1
        t_eval_used = np.linspace(0.0, T_test, M_test)
    else:
        t_eval_used = t_eval
    U_test = np.array([test_bc(ti) for ti in t_eval_used])

    def heat_rhs(t, u):
        u[0] = test_bc(t)
        u[-1] = test_bc(t)
        return nu_test * (D2x @ u)

    print(f"  Solving FOM for ν={nu_test}...")
    from scipy.integrate import solve_ivp as solve_ivp_fom
    T = t_eval_used[-1]
    sol_fom = solve_ivp_fom(heat_rhs, [0, T], u0_fixed.copy(),
                            t_eval=t_eval_used, method='BDF',
                            rtol=1e-6, atol=1e-8)

    if not sol_fom.success:
        print(f"  ✗ FOM solve failed")
        exit(1)

    Y_test = sol_fom.y
    print(f"  ✓ Generated FOM ground truth")

if not Y_tests and Y_test is not None:
    Y_tests = [Y_test]
    U_tests = [U_test]

elif not Y_tests:
    print(f"⚠ No ground truth found for ν={nu_test}")
    print("  Operators were generated but cannot compute error")
    exit(0)

# Solve ROM with LLM operators (using working framework's approach)
print(f"\nSolving ROM with LLM-predicted operators...")
print(f"  Trajectories: {len(Y_tests)}")

t_eval_base = t_eval_used

# Test with same setup as working scripts
errors = []
errors_first = []
errors_second = []
success = []
plot_data = None

for idx, (Y_ref, U_ref) in enumerate(zip(Y_tests, U_tests), start=1):
    print(f"\nTrajectory {idx}/{len(Y_tests)}:")
    t_eval_used = dataset_t_eval if dataset_t_eval is not None else t_eval_base

    # Use ACTUAL input function from ground truth data
    if U_ref is not None:
        if equation_type == "burgers" and isinstance(U_ref, (list, tuple)) and len(U_ref) == 3:
            w1_vec, w2_vec, w3_vec = U_ref
            t_eval_used = dataset_t_eval if dataset_t_eval is not None else t_eval
            u_func_interp = lambda t: np.array([
                np.interp(t, t_eval_used, w1_vec),
                np.interp(t, t_eval_used, w2_vec),
                np.interp(t, t_eval_used, w3_vec)
            ])
            if idx == 1:
                print(f"  Using actual 3 input functions from ground truth (Burgers)")
        else:
            t_eval_used = dataset_t_eval if dataset_t_eval is not None else t_eval
            u_func_interp = lambda t: np.interp(t, t_eval_used, U_ref)
            if idx == 1:
                print(f"  Using actual input function from ground truth")
    else:
        if equation_type == "burgers":
            u_func_interp = lambda t: np.array([0.0, 0.0, 0.0])
            if idx == 1:
                print(f"  Warning: Using default zero input functions (Burgers)")
        else:
            def f1(t): return 1 + t*(1 - t)
            u_func_interp = f1
            if idx == 1:
                print(f"  Warning: Using default input function (may not match ground truth)")

    # Initial condition from ground truth (like working script does)
    a0 = project(Y_ref[:, 0:1]).flatten()
    T = t_eval_used[-1]

    if equation_type == "heat":
        A = operators['A']
        B = operators['B']
        C = operators['C']

        def rom_rhs(t, a):
            u = u_func_interp(t)
            return C + A @ a + B.ravel() * u

        sol = solve_ivp(rom_rhs, [0, T], a0,
                        t_eval=t_eval_used, method='BDF',
                        atol=1e-6, rtol=1e-6)

        if not sol.success or not np.all(np.isfinite(sol.y)):
            sol = solve_ivp(rom_rhs, [0, T], a0,
                            t_eval=t_eval_used, method='RK45',
                            atol=1e-6, rtol=1e-6)
    else:
        H = operators['H']
        A = operators['A']
        B = operators['B']
        C = operators['C']

        def rom_rhs(t, a):
            u_vec = u_func_interp(t)
            quadratic_term = np.einsum('ijk,j,k->i', H, a, a)
            if np.isscalar(u_vec):
                u_vec = np.array([u_vec, u_vec, u_vec])
            return C + A @ a + quadratic_term + B @ u_vec

        if nu_test >= 0.05:
            atol_use, rtol_use = 1e-6, 1e-4
        else:
            atol_use, rtol_use = 1e-7, 1e-5

        sol = solve_ivp(rom_rhs, [0, T], a0,
                        t_eval=t_eval_used, method='BDF',
                        atol=atol_use, rtol=rtol_use, max_step=0.01)

        if not sol.success or not np.all(np.isfinite(sol.y)):
            sol = solve_ivp(rom_rhs, [0, T], a0,
                            t_eval=t_eval_used, method='RK45',
                            atol=atol_use, rtol=rtol_use, max_step=0.01)

        if not sol.success or not np.all(np.isfinite(sol.y)):
            sol = solve_ivp(rom_rhs, [0, T], a0,
                            t_eval=t_eval_used, method='LSODA',
                            atol=atol_use, rtol=rtol_use)

    if not sol.success or not np.all(np.isfinite(sol.y)):
        print("  ✗ FAILED (ROM instability)")
        success.append(False)
        errors.append(np.inf)
        continue

    a_rom = sol.y
    Y_rom = lift(a_rom)

    if Y_rom.shape[1] != Y_ref.shape[1]:
        min_steps = min(Y_rom.shape[1], Y_ref.shape[1])
        print(f"  ⚠ Time dimension mismatch, truncating to {min_steps} steps")
        Y_rom = Y_rom[:, :min_steps]
        Y_ref = Y_ref[:, :min_steps]
        t_eval_used = t_eval_used[:min_steps]

    dt_used = t_eval_used[1] - t_eval_used[0]
    rel_error = global_spatiotemporal_relative_l2(Y_ref, Y_rom, dx, dt_used)
    errors.append(rel_error)
    success.append(True)
    print(f"  ✓ Error: {rel_error*100:.4f}%")

    T_train = t_eval[-1]
    if len(t_eval_used) > len(t_eval):
        split_idx = np.searchsorted(t_eval_used, T_train)
        Y_ref_first = Y_ref[:, :split_idx]
        Y_rom_first = Y_rom[:, :split_idx]
        Y_ref_second = Y_ref[:, split_idx:]
        Y_rom_second = Y_rom[:, split_idx:]
        rel_first = global_spatiotemporal_relative_l2(Y_ref_first, Y_rom_first, dx, dt_used)
        rel_second = global_spatiotemporal_relative_l2(Y_ref_second, Y_rom_second, dx, dt_used)
        errors_first.append(rel_first)
        errors_second.append(rel_second)

    if plot_data is None:
        plot_data = (Y_ref, Y_rom, t_eval_used)

    if args.save_raw:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True)
        suffix = f"_traj{idx}" if len(Y_tests) > 1 else ""
        raw_path = output_dir / f"llm_{equation_type}_nu{nu_test}{suffix}_raw.npz"
        save_kwargs = {
            "Y_test": Y_ref,
            "Y_rom": Y_rom,
            "x_grid": x_grid,
            "t_eval": t_eval_used,
            "nu": np.array([nu_test]),
        }
        if equation_type == "heat":
            save_kwargs["u_bc"] = U_ref if U_ref is not None else np.array([])
        else:
            if isinstance(U_ref, tuple):
                save_kwargs["w1_bc"] = U_ref[0]
                save_kwargs["w2_bc"] = U_ref[1]
                save_kwargs["source"] = U_ref[2]
            else:
                save_kwargs["w1_bc"] = np.array([])
                save_kwargs["w2_bc"] = np.array([])
                save_kwargs["source"] = np.array([])
        np.savez_compressed(raw_path, **save_kwargs)
        print(f"  ✓ Saved raw arrays: {raw_path}")

print("\n" + "=" * 70)
print("RESULTS")
print("=" * 70)
print(f"Equation: {equation_type}")
print(f"Parameter: ν={nu_test}")

valid_errors = [e for e, s in zip(errors, success) if s]
if valid_errors:
    mean_error = float(np.mean(valid_errors))
    print(f"Global Spatiotemporal Relative L2 Error (mean): {mean_error*100:.4f}%")
    if len(valid_errors) > 1:
        print(f"  Min error: {min(valid_errors)*100:.4f}%")
        print(f"  Max error: {max(valid_errors)*100:.4f}%")
else:
    print("  ✗ All trajectories failed")

if errors_first and errors_second:
    print(f"Error [0, {T_train:.1f}] (mean): {np.mean(errors_first)*100:.4f}%")
    print(f"Error [{T_train:.1f}, {T:.1f}] (mean): {np.mean(errors_second)*100:.4f}%")
print("=" * 70)

# Visualization (3-panel like working scripts)
if not args.save_plots or True:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Plot 1: Ground Truth
    if plot_data is None:
        print("⚠ No successful trajectories to plot")
        exit(0)
    plot_Y_ref, plot_Y_rom, plot_t_eval = plot_data
    T_mesh, X_mesh = np.meshgrid(plot_t_eval, x_grid)
    im1 = axes[0].pcolormesh(T_mesh, X_mesh, plot_Y_ref, cmap='RdBu_r', shading='auto')
    axes[0].set_title(f'Ground Truth (ν={nu_test})')
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('Space')
    plt.colorbar(im1, ax=axes[0])

    # Plot 2: ROM Prediction
    im2 = axes[1].pcolormesh(T_mesh, X_mesh, plot_Y_rom, cmap='RdBu_r', shading='auto')
    axes[1].set_title(f'LLM ROM Prediction')
    axes[1].set_xlabel('Time')
    axes[1].set_ylabel('Space')
    plt.colorbar(im2, ax=axes[1])

    # Plot 3: Error
    error = np.abs(plot_Y_ref - plot_Y_rom)
    display_error = float(np.mean(valid_errors)*100) if valid_errors else 0.0
    im3 = axes[2].pcolormesh(T_mesh, X_mesh, error, cmap='YlOrRd', shading='auto')
    axes[2].set_title(f'Absolute Error ({display_error:.4f}%)')
    axes[2].set_xlabel('Time')
    axes[2].set_ylabel('Space')
    plt.colorbar(im3, ax=axes[2])

    plt.tight_layout()

    if args.save_plots:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True)
        filename = output_dir / f"llm_{equation_type}_nu{nu_test}_test.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"\n✓ Saved plot: {filename}")
    else:
        plt.show()

print("\n✓ Test complete!")
