#!/usr/bin/env python3
"""
Parametric 2D Cavity Flow: Test ROM Using LLM-Derived Operators

For each requested Re value, asks an LLM (via tool calling) to compute
OpInf operators using linear regression. Then integrates the ROM and
computes spatiotemporal L2 errors.
"""

import argparse
import json
import os
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def load_pickle_auto(filepath):
    """Load pickle file, auto-detecting gzip compression."""
    import gzip

    with open(filepath, "rb") as f:
        magic = f.read(2)
        f.seek(0)
        if magic == b"\x1f\x8b":
            with gzip.open(filepath, "rb") as gz:
                return pickle.load(gz)
        return pickle.load(f)


def rom_rhs_quadratic(t, a, H, A, B, C, U_lid):
    """Quadratic ROM RHS: ȧ = C + A·a + H(a⊗a) + B·U_lid"""
    r = len(a)
    da = np.zeros(r)
    for i in range(r):
        quad = a @ (H[i] @ a)
        quad = np.clip(quad, -1e6, 1e6)
        da[i] = quad + A[i] @ a + B[i] * U_lid + C[i]
    return da


def integrate_rom_rk4(a0, t_span, dt, H, A, B, C, U_lid_array, n_steps):
    """Integrate ROM with RK4 time stepping."""
    r = len(a0)
    a_traj = np.zeros((r, n_steps))
    a_traj[:, 0] = a0
    for k in range(n_steps - 1):
        t = t_span[0] + k * dt
        a_k = a_traj[:, k]
        U_lid_k = U_lid_array[k]
        k1 = rom_rhs_quadratic(t, a_k, H, A, B, C, U_lid_k)
        k2 = rom_rhs_quadratic(t + dt/2, a_k + dt*k1/2, H, A, B, C, U_lid_k)
        k3 = rom_rhs_quadratic(t + dt/2, a_k + dt*k2/2, H, A, B, C, U_lid_k)
        k4 = rom_rhs_quadratic(t + dt, a_k + dt*k3, H, A, B, C, U_lid_k)
        a_traj[:, k+1] = a_k + dt * (k1 + 2*k2 + 2*k3 + k4) / 6
        if np.max(np.abs(a_traj[:, k+1])) > 1e6:
            return None, False
    return a_traj, True


def spatiotemporal_l2_error(Y_ref, Y_rom, dx, dt):
    """Compute spatiotemporal relative L2 error."""
    dA = dx * dx
    err = Y_ref - Y_rom
    err_norm_sq = np.sum(err**2) * dA * dt
    ref_norm_sq = np.sum(Y_ref**2) * dA * dt
    return np.sqrt(err_norm_sq / (ref_norm_sq + 1e-14))


def load_llm_operators(output_path):
    """Load operators from LLM output JSON."""
    with open(output_path, "r") as f:
        data = json.load(f)
    ops = data["predicted_operators"]["operators"]
    H = np.array(ops["H"]["values"])
    A = np.array(ops["A"]["values"])
    B = np.array(ops["B"]["values"])
    C = np.array(ops["C"]["values"])
    return H, A, B, C


def find_llm_output_path(output_dir, Re_query):
    """Find the LLM output JSON path for a given Re (tolerant of formatting)."""
    target = float(Re_query)
    candidates = list(Path(output_dir).glob("llm_cavity_operators_Re*.json"))
    for path in candidates:
        stem = path.stem
        try:
            val_str = stem.split("_Re", 1)[1]
            val = float(val_str)
        except (IndexError, ValueError):
            continue
        if abs(val - target) < 1e-6:
            return str(path)
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Test parametric cavity ROM using LLM-derived operators"
    )
    parser.add_argument("--model", type=str, required=True,
                        help="Parametric cavity model .pkl file")
    parser.add_argument("--test_data", type=str, required=True,
                        help="Dataset file (train or test)")
    parser.add_argument("--mode", type=str, default="all",
                        choices=["trained", "interpolation", "extrapolation", "all"],
                        help="Test mode")
    parser.add_argument("--Re_test", nargs="+", type=float, default=None,
                        help="Specific Re values to test (overrides mode)")
    parser.add_argument("--provider", type=str, default="openai",
                        choices=["openai", "anthropic", "gemini", "deepseek", "qwen"],
                        help="LLM provider")
    parser.add_argument("--model_name", type=str, default="gpt-4o",
                        help="LLM model name")
    parser.add_argument("--llm_output_dir", type=str, default="llm_cavity_predictions",
                        help="Where to store LLM operator outputs")
    parser.add_argument("--llm_mode", type=str, default="tool",
                        choices=["tool", "codegen"],
                        help="LLM mode: tool calling or code generation")
    parser.add_argument("--llm_method", type=str, default="regression",
                        choices=["interpolation", "regression"],
                        help="LLM operator prediction method")
    parser.add_argument("--reuse_operators", action="store_true",
                        help="Reuse existing LLM operator files if available")
    parser.add_argument("--no_use_exact_trained", action="store_true",
                        help="Disable using exact trained operators for Re in training set")
    parser.add_argument("--llm_batch_size", type=int, default=0,
                        help="Batch size for LLM tool-calling. 0 means all at once.")
    parser.add_argument("--llm_sleep_secs", type=float, default=0.0,
                        help="Sleep between LLM batches (seconds)")
    parser.add_argument("--save_plots", action="store_true",
                        help="Save visualization plots")
    parser.add_argument("--save_raw", action="store_true",
                        help="Save raw prediction arrays for custom plotting")
    parser.add_argument("--output_dir", type=str, default="cavity_test_results_timevarying_llm",
                        help="Directory for output plots")
    args = parser.parse_args()

    model = load_pickle_auto(args.model)
    config = model["config"]
    phi = model["phi"]
    x = model["x"]
    y = model["y"]
    dx = model["dx"]
    t_eval_model = model["t_eval"]

    test_dataset = load_pickle_auto(args.test_data)
    test_per_Re_data = test_dataset["per_Re_data"]
    t_eval_test = test_dataset["t_eval"]

    Re_trained = [m["Re"] for m in model["per_Re_models"]]
    Re_list_test = [item["Re"] for item in test_per_Re_data]

    dA = dx * dx
    dt = t_eval_test[1] - t_eval_test[0]
    M = len(t_eval_test)

    def project(Y):
        return phi.T @ (Y * dA)

    def lift(a):
        return phi @ a

    if args.Re_test is not None:
        Re_values_to_test = args.Re_test
    elif args.mode == "trained":
        Re_values_to_test = Re_trained
    elif args.mode == "interpolation":
        Re_min, Re_max = min(Re_trained), max(Re_trained)
        Re_values_to_test = [Re for Re in Re_list_test if Re_min < Re < Re_max]
    elif args.mode == "extrapolation":
        Re_min, Re_max = min(Re_trained), max(Re_trained)
        Re_values_to_test = [Re for Re in Re_list_test if Re < Re_min or Re > Re_max]
    else:
        Re_values_to_test = sorted(set(Re_trained + Re_list_test))

    print(f"\nTesting on Re values: {Re_values_to_test}")

    if args.save_plots:
        Path(args.output_dir).mkdir(exist_ok=True)
    Path(args.llm_output_dir).mkdir(exist_ok=True)

    results = {
        "trained": {"Re": [], "errors": [], "success": []},
        "interpolation": {"Re": [], "errors": [], "success": []},
        "extrapolation": {"Re": [], "errors": [], "success": []},
    }

    # Precompute all LLM operators in one call (single LLM session).
    use_exact_trained = not args.no_use_exact_trained
    missing = []
    if args.reuse_operators:
        for Re_query in Re_values_to_test:
            if use_exact_trained and Re_query in Re_trained:
                continue
            target = os.path.join(
                args.llm_output_dir, f"llm_cavity_operators_Re{Re_query}.json"
            )
            if not os.path.exists(target) and not find_llm_output_path(args.llm_output_dir, Re_query):
                missing.append(Re_query)
    else:
        missing = [
            Re_query for Re_query in Re_values_to_test
            if not (use_exact_trained and Re_query in Re_trained)
        ]

    if missing:
        if args.llm_mode == "codegen" and args.llm_method != "regression":
            raise ValueError("Codegen mode only supports regression for now.")
        if args.llm_mode == "tool":
            subprocess.run(
                [
                    sys.executable,
                    "src/opinf-llm/llm_tool_calling_cavity_parametric.py",
                    "--model",
                    args.model,
                    "--query_Re_values",
                    *[str(Re) for Re in missing],
                    "--provider",
                    args.provider,
                    "--model_name",
                    args.model_name,
                    "--method",
                    args.llm_method,
                    "--output_dir",
                    args.llm_output_dir,
                    "--batch_size",
                    str(args.llm_batch_size),
                    "--sleep_secs",
                    str(args.llm_sleep_secs),
                ],
                check=True,
            )
        else:
            subprocess.run(
                [
                    "python",
                    "llm_code_generation_cavity_regression.py",
                    "--model_path",
                    args.model,
                    "--re_queries",
                    *[str(Re) for Re in missing],
                    "--provider",
                    args.provider,
                    "--model_name",
                    args.model_name,
                    "--output_dir",
                    args.llm_output_dir,
                ],
                check=True,
            )

    for Re_test in Re_values_to_test:
        print(f"\n{'='*70}")
        print(f"Reynolds number: Re = {Re_test}")
        print("=" * 70)

        # Classify for reporting
        Re_min, Re_max = min(Re_trained), max(Re_trained)
        if Re_test in Re_trained:
            op_type = "trained"
        elif Re_min <= Re_test <= Re_max:
            op_type = "interpolation"
        else:
            op_type = "extrapolation"

        # Operators for this Re
        if use_exact_trained and Re_test in Re_trained:
            model_idx = Re_trained.index(Re_test)
            exact = model["per_Re_models"][model_idx]
            H, A, B, C = exact["H"], exact["A"], exact["B"], exact["C"]
            print("  Using exact OpInf operators (trained Re)")
        else:
            llm_output = os.path.join(
                args.llm_output_dir, f"llm_cavity_operators_Re{Re_test}.json"
            )
            if not os.path.exists(llm_output):
                resolved = find_llm_output_path(args.llm_output_dir, Re_test)
                if resolved:
                    llm_output = resolved
            if not os.path.exists(llm_output):
                print(f"⚠ Missing LLM operators for Re={Re_test}; skipping.")
                continue
            H, A, B, C = load_llm_operators(llm_output)

        test_data_Re = None
        for item in test_per_Re_data:
            if abs(item["Re"] - Re_test) < 1e-6:
                test_data_Re = item
                break

        if test_data_Re is None:
            print(f"✗ No test data available for Re={Re_test}")
            continue

        if test_data_Re["test"]["Y_omega"] is not None and test_data_Re["test"]["Y_omega"].size > 0:
            test_omega = test_data_Re["test"]["Y_omega"]
            test_psi = test_data_Re["test"]["Y_psi"]
            test_U = test_data_Re["test"]["U_lid"]
            split_name = "test"
        elif test_data_Re["validation"]["Y_omega"] is not None and test_data_Re["validation"]["Y_omega"].size > 0:
            test_omega = test_data_Re["validation"]["Y_omega"]
            test_psi = test_data_Re["validation"]["Y_psi"]
            test_U = test_data_Re["validation"]["U_lid"]
            split_name = "validation"
        else:
            test_omega = test_data_Re["train"]["Y_omega"]
            test_psi = test_data_Re["train"]["Y_psi"]
            test_U = test_data_Re["train"]["U_lid"]
            split_name = "train"

        num_traj = test_omega.shape[1] // M
        print(f"\nTesting on {num_traj} {split_name} trajectories (LLM operators)")

        traj_errors = []
        traj_success = []

        for i in range(num_traj):
            print(f"\nTrajectory {i+1}/{num_traj}:")
            print("-" * 70)

            Y_omega_fom = test_omega[:, i*M:(i+1)*M]
            Y_psi_fom = test_psi[:, i*M:(i+1)*M]
            Y_fom = np.vstack([Y_omega_fom, Y_psi_fom])

            a0 = project(Y_fom[:, 0:1]).ravel()
            U_lid_traj = test_U[i*M:(i+1)*M]

            print(f"  Initial condition: ||a0|| = {np.linalg.norm(a0):.2e}")
            print(f"  Input f(t): mean={np.mean(U_lid_traj):.3f}, std={np.std(U_lid_traj):.3f}")
            print("  Integrating ROM...")

            a_rom, success = integrate_rom_rk4(
                a0, [0, t_eval_test[-1]], dt, H, A, B, C, U_lid_traj, M
            )

            if not success or a_rom is None:
                print("  ✗ FAILED (ROM instability)")
                traj_errors.append(np.inf)
                traj_success.append(False)
                continue

            Y_rom = lift(a_rom)
            error = spatiotemporal_l2_error(Y_fom, Y_rom, dx, dt)
            traj_errors.append(error)
            traj_success.append(True)

            print("  ✓ Success")
            print(f"  Spatiotemporal L2 error: {error*100:.2f}%")

            if M % 2 == 0:
                mid = M // 2
                err_first = spatiotemporal_l2_error(Y_fom[:, :mid], Y_rom[:, :mid], dx, dt)
                err_second = spatiotemporal_l2_error(Y_fom[:, mid:], Y_rom[:, mid:], dx, dt)
                print(f"  Error [0, T]:   {err_first*100:.2f}%")
                print(f"  Error [T, 2T]:  {err_second*100:.2f}%")

            if args.save_raw:
                output_dir = Path(args.output_dir)
                output_dir.mkdir(exist_ok=True)
                n_omega = Y_omega_fom.shape[0]
                raw_path = output_dir / f"cavity_Re{Re_test}_traj{i+1}_raw.npz"
                np.savez_compressed(
                    raw_path,
                    Y_omega_fom=Y_omega_fom,
                    Y_psi_fom=Y_psi_fom,
                    Y_omega_rom=Y_rom[:n_omega, :],
                    Y_psi_rom=Y_rom[n_omega:, :],
                    U_lid=U_lid_traj,
                    x=x,
                    y=y,
                    t_eval=t_eval_test,
                    Re=np.array([Re_test]),
                )
                print(f"  Raw data saved: {raw_path}")

            if i == 0 and args.save_plots:
                omega_fom = Y_omega_fom.reshape(len(y), len(x), M, order="C")
                omega_rom = Y_rom[:len(Y_omega_fom)].reshape(len(y), len(x), M, order="C")

                X, Y_grid = np.meshgrid(x, y, indexing="ij")
                fig, axs = plt.subplots(2, 5, figsize=(24, 10))
                times_idx = [0, M//4, M//2, 3*M//4, M-1]
                times_val = [t_eval_test[idx] for idx in times_idx]

                for col, (t_idx, t_val) in enumerate(zip(times_idx, times_val)):
                    im0 = axs[0, col].contourf(X, Y_grid, omega_fom[:, :, t_idx], 50, cmap="RdBu_r")
                    axs[0, col].set_title(f"FOM ω at t={t_val:.2f}s")
                    axs[0, col].set_xlabel("x")
                    axs[0, col].set_ylabel("y")
                    axs[0, col].set_aspect("equal")
                    plt.colorbar(im0, ax=axs[0, col])

                    im1 = axs[1, col].contourf(X, Y_grid, omega_rom[:, :, t_idx], 50, cmap="RdBu_r")
                    axs[1, col].set_title(f"ROM ω at t={t_val:.2f}s")
                    axs[1, col].set_xlabel("x")
                    axs[1, col].set_ylabel("y")
                    axs[1, col].set_aspect("equal")
                    plt.colorbar(im1, ax=axs[1, col])

                fig.suptitle(f"Re={Re_test} (LLM), Error={error*100:.2f}%", fontsize=16, fontweight="bold")
                plt.tight_layout()

                plot_file = Path(args.output_dir) / f"cavity_Re{Re_test}_llm_traj{i+1}.png"
                plt.savefig(plot_file, dpi=150, bbox_inches="tight")
                plt.close()
                print(f"  Plot saved: {plot_file}")

        print(f"\n{'-'*70}")
        print(f"Summary for Re={Re_test} (LLM):")
        n_success = sum(traj_success)
        if n_success > 0:
            valid_errors = [e for e, s in zip(traj_errors, traj_success) if s]
            mean_error = np.mean(valid_errors)
            print(f"  Successful trajectories: {n_success}/{num_traj}")
            print(f"  Mean error: {mean_error*100:.2f}%")
            print(f"  Min error:  {min(valid_errors)*100:.2f}%")
            print(f"  Max error:  {max(valid_errors)*100:.2f}%")
            results[op_type]["Re"].append(Re_test)
            results[op_type]["errors"].append(mean_error)
            results[op_type]["success"].append(True)
        else:
            print("  ✗ All trajectories failed")
            results[op_type]["Re"].append(Re_test)
            results[op_type]["errors"].append(np.inf)
            results[op_type]["success"].append(False)

    print("\n" + "=" * 70)
    print("FINAL RESULTS SUMMARY (LLM OPERATORS)")
    print("=" * 70)

    for category in ["trained", "interpolation", "extrapolation"]:
        if len(results[category]["Re"]) > 0:
            print(f"\n{category.upper()} Re VALUES:")
            print("-" * 70)
            for Re, error, success in zip(
                results[category]["Re"],
                results[category]["errors"],
                results[category]["success"],
            ):
                if success:
                    print(f"  Re={Re:>6.1f}: {error*100:>6.2f}% error")
                else:
                    print(f"  Re={Re:>6.1f}: FAILED")

            valid_errors = [e for e, s in zip(results[category]["errors"], results[category]["success"]) if s]
            if valid_errors:
                print("\n  Statistics:")
                print(f"    Success rate: {len(valid_errors)}/{len(results[category]['Re'])}")
                print(f"    Mean error:   {np.mean(valid_errors)*100:.2f}%")
                print(f"    Median error: {np.median(valid_errors)*100:.2f}%")
                print(f"    Min error:    {min(valid_errors)*100:.2f}%")
                print(f"    Max error:    {max(valid_errors)*100:.2f}%")

    print("\n" + "=" * 70)
    print("✓ Testing complete!")
    if args.save_plots:
        print(f"✓ Plots saved to: {args.output_dir}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
