#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified LLM workflow for Heat, Burgers, and Cavity equations.

Regenerates LLM operator predictions by default, runs tests, and saves
all plots/results under a single output folder.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
import json
import pickle
import re
import time


DEFAULT_MODEL_BY_PROVIDER = {
    "openai": "gpt-4o",
    "gemini": "gemini-2.0-flash-exp",
    "deepseek": "deepseek-chat",
    "anthropic": "claude-sonnet-4-20250514",
    "qwen": "qwen-plus",
}

DEFAULT_HEAT_NUS = [0.5, 1.0, 3.0]
DEFAULT_BURGERS_NUS = [0.03, 0.07]
DEFAULT_CAVITY_RES = [60.0, 80.0, 90.0, 110.0, 120.0, 140.0]


def run(cmd, desc):
    print(f"\n==> {desc}")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def resolve_operator_path(base_dir: Path, prefix: str, nu: float) -> Path:
    """Resolve operator JSON filename produced by tool calling."""
    preferred = base_dir / f"{prefix}_nu{nu}.json"
    if preferred.exists():
        return preferred
    fallback = base_dir / f"tool_calling_operators_nu{nu}.json"
    if fallback.exists():
        return fallback
    return preferred


def to_serializable(value):
    """Convert numpy arrays to lists for JSON output."""
    return value.tolist() if hasattr(value, "tolist") else value


def compute_split_errors(raw_files, area_scale, T_train):
    """Compute mean errors over trajectories for full and split windows."""
    import numpy as np

    def rel_l2(y_ref, y_rom, scale, dt_val):
        err = y_ref - y_rom
        norm_err = np.sqrt(np.sum(err**2) * scale * dt_val)
        norm_ref = np.sqrt(np.sum(y_ref**2) * scale * dt_val)
        return norm_err / (norm_ref + 1e-14)

    errors_full = []
    errors_first = []
    errors_second = []

    for raw_path in raw_files:
        data = np.load(raw_path)
        y_ref = data["Y_test"]
        y_rom = data["Y_rom"]
        t_eval_used = data["t_eval"]
        dt_val = t_eval_used[1] - t_eval_used[0]
        err_full = rel_l2(y_ref, y_rom, area_scale, dt_val)
        errors_full.append(float(err_full))

        if t_eval_used[-1] > T_train:
            split_idx = np.searchsorted(t_eval_used, T_train)
            if split_idx > 1 and split_idx < y_ref.shape[1]:
                err_first = rel_l2(y_ref[:, :split_idx], y_rom[:, :split_idx], area_scale, dt_val)
                err_second = rel_l2(y_ref[:, split_idx:], y_rom[:, split_idx:], area_scale, dt_val)
                errors_first.append(float(err_first))
                errors_second.append(float(err_second))

    summary = {
        "n_traj": len(errors_full),
        "mean_full": float(np.mean(errors_full)) if errors_full else None,
        "min_full": float(np.min(errors_full)) if errors_full else None,
        "max_full": float(np.max(errors_full)) if errors_full else None,
        "mean_first": float(np.mean(errors_first)) if errors_first else None,
        "mean_second": float(np.mean(errors_second)) if errors_second else None,
    }
    return summary


def compute_split_errors_cavity(raw_files, area_scale, T_train):
    """Compute mean errors over trajectories for cavity raw files."""
    import numpy as np

    def rel_l2(y_ref, y_rom, scale, dt_val):
        err = y_ref - y_rom
        norm_err = np.sqrt(np.sum(err**2) * scale * dt_val)
        norm_ref = np.sqrt(np.sum(y_ref**2) * scale * dt_val)
        return norm_err / (norm_ref + 1e-14)

    errors_full = []
    errors_first = []
    errors_second = []

    for raw_path in raw_files:
        data = np.load(raw_path)
        y_ref = np.vstack([data["Y_omega_fom"], data["Y_psi_fom"]])
        y_rom = np.vstack([data["Y_omega_rom"], data["Y_psi_rom"]])
        t_eval_used = data["t_eval"]
        dt_val = t_eval_used[1] - t_eval_used[0]
        err_full = rel_l2(y_ref, y_rom, area_scale, dt_val)
        errors_full.append(float(err_full))

        if t_eval_used[-1] > T_train:
            split_idx = np.searchsorted(t_eval_used, T_train)
            if split_idx > 1 and split_idx < y_ref.shape[1]:
                err_first = rel_l2(y_ref[:, :split_idx], y_rom[:, :split_idx], area_scale, dt_val)
                err_second = rel_l2(y_ref[:, split_idx:], y_rom[:, split_idx:], area_scale, dt_val)
                errors_first.append(float(err_first))
                errors_second.append(float(err_second))

    summary = {
        "n_traj": len(errors_full),
        "mean_full": float(np.mean(errors_full)) if errors_full else None,
        "min_full": float(np.min(errors_full)) if errors_full else None,
        "max_full": float(np.max(errors_full)) if errors_full else None,
        "mean_first": float(np.mean(errors_first)) if errors_first else None,
        "mean_second": float(np.mean(errors_second)) if errors_second else None,
    }
    return summary


def write_summary_split_errors(base_dir, heat_model, burgers_model, cavity_model):
    """Write split-error summary JSON using raw .npz files."""
    import numpy as np
    import json
    from pathlib import Path

    base_dir = Path(base_dir)
    summary = {"heat": {}, "burgers": {}, "cavity": {}}
    # Align interpolation reporting with codegen defaults; regression reports all params.
    interpolation_report = {
        "heat": [0.5, 1.0, 3.0],
        "burgers": [0.03, 0.07],
        "cavity": [60.0, 80.0, 90.0, 110.0, 120.0, 140.0],
    }

    def in_report_set(eq: str, value: float) -> bool:
        return any(abs(float(value) - float(v)) < 1e-12 for v in interpolation_report[eq])

    # Heat/Burgers
    for eq, model_path in [("heat", heat_model), ("burgers", burgers_model)]:
        with open(model_path, "rb") as f:
            model_data = pickle.load(f)
        t_train = model_data["t_eval"][-1]
        x_grid = model_data.get("x_grid", model_data.get("x_fine"))
        dx = x_grid[1] - x_grid[0]
        for method in ["interpolation", "regression"]:
            folder_candidates = [
                base_dir / eq / method / "results",
                base_dir / f"{eq}_test_results" / method,  # backward compatibility
            ]
            folder = next((p for p in folder_candidates if p.exists()), None)
            if folder is None:
                continue
            by_param = {}
            for raw_path in sorted(folder.glob(f"llm_{eq}_nu*_raw.npz")):
                name = raw_path.name
                nu_match = re.search(r"nu([0-9.]+)", name)
                if not nu_match:
                    continue
                nu_val = float(nu_match.group(1))
                by_param.setdefault(nu_val, []).append(raw_path)
            for nu_val, files in by_param.items():
                if not in_report_set(eq, nu_val):
                    continue
                summary[eq].setdefault(method, {})[str(nu_val)] = compute_split_errors(
                    files, dx, t_train
                )

    # Cavity
    with open(cavity_model, "rb") as f:
        cav_model = pickle.load(f)
    t_train = cav_model["t_eval"][-1]
    x = cav_model["x"]
    dx = x[1] - x[0]
    dA = dx * dx
    for method in ["interpolation", "regression"]:
        folder_candidates = [
            base_dir / "cavity" / method / "results",
            base_dir / "cavity_test_results" / method,  # backward compatibility
        ]
        folder = next((p for p in folder_candidates if p.exists()), None)
        if folder is None:
            continue
        by_re = {}
        for raw_path in sorted(folder.glob("cavity_Re*_traj*_raw.npz")):
            match = re.search(r"Re([0-9.]+)_traj", raw_path.name)
            if not match:
                continue
            re_val = float(match.group(1))
            by_re.setdefault(re_val, []).append(raw_path)
        for re_val in sorted(by_re.keys()):
            files = by_re[re_val]
            if not in_report_set("cavity", re_val):
                continue
            # Build combined Y to use the same rel_l2 helper
            summary["cavity"].setdefault(method, {})[str(re_val)] = compute_split_errors_cavity(
                files, dA, t_train
            )

    output_path = base_dir / "summary_split_errors.json"
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved split-error summary to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Unified LLM workflow for 3 equations")
    parser.add_argument("--provider", type=str, default="gemini",
                        choices=["openai", "gemini", "deepseek", "anthropic", "qwen"],
                        help="LLM provider")
    parser.add_argument("--model_name", type=str, default=None,
                        help="LLM model name (default depends on provider)")
    parser.add_argument("--run_id", type=str, default=None,
                        help="Run identifier. Default: timestamp YYYYMMDD-HHMMSS")
    parser.add_argument("--output_dir", type=str, default="tool_call_runs",
                        help="Base output root. Results saved under <output_dir>/<model>/<run_id>/")
    parser.add_argument("--equations", nargs="+", default=["heat", "burgers", "cavity"],
                        choices=["heat", "burgers", "cavity"],
                        help="Equations to run (default: all)")
    parser.add_argument("--heat_nus", nargs="+", type=float, default=DEFAULT_HEAT_NUS,
                        help="Override heat nu values (comma-separated list)")
    parser.add_argument("--burgers_nus", nargs="+", type=float, default=DEFAULT_BURGERS_NUS,
                        help="Override burgers nu values (comma-separated list)")
    parser.add_argument("--cavity_res", nargs="+", type=float, default=DEFAULT_CAVITY_RES,
                        help="Override cavity Re values (comma-separated list)")
    parser.add_argument("--reuse_operators", action="store_true",
                        help="Reuse existing operator JSONs when available")
    parser.add_argument("--save_raw", action="store_true",
                        help="Save raw prediction arrays for plotting")
    parser.add_argument("--no_write_summary", action="store_true",
                        help="Disable writing split-error summary JSON")
    args = parser.parse_args()

    model_name = args.model_name or DEFAULT_MODEL_BY_PROVIDER[args.provider]
    run_id = args.run_id or time.strftime("%Y%m%d-%H%M%S")

    base_dir = Path(args.output_dir) / model_name / run_id
    print(f"[tool-call] run_id={run_id}")
    print(f"[tool-call] run_dir={base_dir}")
    equations = set(args.equations)
    def method_dirs(eq_name: str, method_name: str):
        run_dir = base_dir / eq_name / method_name
        ops = run_dir / "operators"
        results = run_dir / "results"
        return run_dir, ops, results

    # Heat equation parameters (same as individual workflow)
    heat_nus = args.heat_nus
    heat_model = "src/heat_model.pkl"

    # Burgers equation parameters (same as individual workflow)
    burgers_train_nus = [0.02, 0.05]
    burgers_test_nus = [0.03, 0.07]
    burgers_model = "src/burgers_model.pkl"
    burgers_dataset_train = "dataset/burgers_dataset_unified.pkl.gz"
    burgers_dataset_test = "dataset/burgers_dataset_test.pkl.gz"

    # Cavity parameters (same as individual workflow)
    cavity_model = "src/cavity_model.pkl"
    cavity_test_data = "dataset/cavity_dataset_test.pkl.gz"

    methods = ["interpolation", "regression"]

    # Heat: operator generation + test (with 2T horizon)
    if "heat" in equations:
        with open(heat_model, "rb") as f:
            heat_model_data = pickle.load(f)
        heat_train_nus = [item["nu"] for item in heat_model_data["per_nu_models"]]
        heat_unseen = [nu for nu in heat_nus if nu not in heat_train_nus]

        for method in methods:
            _, heat_ops_method, heat_results_method = method_dirs("heat", method)
            heat_ops_method.mkdir(parents=True, exist_ok=True)
            heat_results_method.mkdir(parents=True, exist_ok=True)

            heat_missing = [
                nu for nu in heat_unseen
                if not resolve_operator_path(heat_ops_method, "llm_heat", nu).exists()
            ]
            if heat_missing and not args.reuse_operators:
                run(
                    [
                        sys.executable,
                        "src/llm_tool_calling_interpolation.py",
                        "--model_pkl",
                        heat_model,
                        "--query_nu_values",
                        *[str(nu) for nu in heat_unseen],
                        "--provider",
                        args.provider,
                        "--model",
                        model_name,
                        "--method",
                        method,
                        "--output",
                        str(heat_ops_method / f"llm_heat_{method}_batch.json"),
                    ],
                    f"Generate heat operators (batch, {method})",
                )

            # Populate exact operators for seen heat nus if needed.
            for item in heat_model_data["per_nu_models"]:
                nu = item["nu"]
                out_path = heat_ops_method / f"llm_heat_nu{nu}.json"
                if not out_path.exists():
                    output_data = {
                        "query_nu": nu,
                        "predicted_operators": {
                            "nu": nu,
                            "operators": {
                                "A": to_serializable(item["A"]),
                                "B": to_serializable(item["B"]),
                                "C": to_serializable(item["C"]),
                            },
                            "method": "exact",
                        },
                        "provider": "exact",
                        "model": "opinf",
                        "equation_type": "heat",
                        "conversation_length": 0,
                    }
                    with open(out_path, "w") as f:
                        json.dump(output_data, f, indent=2)

            for nu in heat_nus:
                out_path = resolve_operator_path(heat_ops_method, "llm_heat", nu)
                heat_dataset_test = "dataset/heat_dataset_test.pkl.gz"
                test_cmd = [
                    sys.executable,
                    "src/test_llm_operators.py",
                    "--predicted",
                    str(out_path),
                    "--model",
                    heat_model,
                    "--save_plots",
                    "--output_dir",
                    str(heat_results_method),
                    "--test_T_factor",
                    "2.0",
                ]
                if args.save_raw:
                    test_cmd.append("--save_raw")
                if Path(heat_dataset_test).exists():
                    test_cmd += ["--dataset", heat_dataset_test]
                run(
                    test_cmd,
                    f"Test heat operators for nu={nu} ({method})",
                )

    # Burgers: operator generation
    if "burgers" in equations:
        if args.burgers_nus:
            burgers_all = list(args.burgers_nus)
        with open(burgers_model, "rb") as f:
            burgers_model_data = pickle.load(f)
        burgers_train_all = [item["nu"] for item in burgers_model_data["per_nu_models"]]
        if args.burgers_nus:
            burgers_train_nus = [nu for nu in burgers_all if nu in burgers_train_all]
            burgers_test_nus = [nu for nu in burgers_all if nu not in burgers_train_all]
        burgers_unseen = [nu for nu in burgers_test_nus if nu not in burgers_train_all]

        for method in methods:
            _, burgers_ops_method, burgers_results_method = method_dirs("burgers", method)
            burgers_ops_method.mkdir(parents=True, exist_ok=True)
            burgers_results_method.mkdir(parents=True, exist_ok=True)

            burgers_missing = [
                nu for nu in burgers_unseen
                if not resolve_operator_path(burgers_ops_method, "llm_burgers", nu).exists()
            ]
            if burgers_missing and not args.reuse_operators:
                run(
                    [
                        sys.executable,
                        "src/llm_tool_calling_interpolation.py",
                        "--model_pkl",
                        burgers_model,
                        "--query_nu_values",
                        *[str(nu) for nu in burgers_unseen],
                        "--provider",
                        args.provider,
                        "--model",
                        model_name,
                        "--method",
                        method,
                        "--output",
                        str(burgers_ops_method / f"llm_burgers_{method}_batch.json"),
                    ],
                    f"Generate burgers operators (batch, {method})",
                )

            # Populate exact operators for seen burgers nus if needed.
            for item in burgers_model_data["per_nu_models"]:
                nu = item["nu"]
                out_path = burgers_ops_method / f"llm_burgers_nu{nu}.json"
                if not out_path.exists():
                    output_data = {
                        "query_nu": nu,
                        "predicted_operators": {
                            "nu": nu,
                            "operators": {
                                "H": to_serializable(item["H"]),
                                "A": to_serializable(item["A"]),
                                "B": to_serializable(item["B"]),
                                "C": to_serializable(item["C"]),
                            },
                            "method": "exact",
                        },
                        "provider": "exact",
                        "model": "opinf",
                        "equation_type": "burgers",
                        "conversation_length": 0,
                    }
                    with open(out_path, "w") as f:
                        json.dump(output_data, f, indent=2)

            # Burgers: training nu tests (no extended horizon)
            if Path(burgers_dataset_train).exists():
                for nu in burgers_train_nus:
                    out_path = resolve_operator_path(burgers_ops_method, "llm_burgers", nu)
                    run(
                        [
                            sys.executable,
                            "src/test_llm_operators.py",
                            "--predicted",
                            str(out_path),
                            "--model",
                            burgers_model,
                            "--dataset",
                            burgers_dataset_train,
                            "--save_plots",
                            *(["--save_raw"] if args.save_raw else []),
                            "--output_dir",
                            str(burgers_results_method),
                        ],
                        f"Test burgers operators for nu={nu} (train, {method})",
                    )
            else:
                print(f"⚠ Skipping burgers train-nu tests; dataset not found: {burgers_dataset_train}")

            # Burgers: unseen nu tests (dataset includes 2T horizon)
            for nu in burgers_test_nus:
                out_path = resolve_operator_path(burgers_ops_method, "llm_burgers", nu)
                run(
                    [
                        sys.executable,
                        "src/test_llm_operators.py",
                        "--predicted",
                        str(out_path),
                        "--model",
                        burgers_model,
                        "--dataset",
                        burgers_dataset_test,
                        "--save_plots",
                        *(["--save_raw"] if args.save_raw else []),
                        "--output_dir",
                        str(burgers_results_method),
                    ],
                    f"Test burgers operators for nu={nu} (test, {method})",
                )

    # Cavity: LLM operators + test (single script handles both)
    if "cavity" in equations:
        for method in methods:
            _, cavity_ops_method, cavity_results_method = method_dirs("cavity", method)
            cavity_ops_method.mkdir(parents=True, exist_ok=True)
            cavity_results_method.mkdir(parents=True, exist_ok=True)
            cavity_cmd = [
                sys.executable,
                "src/cavity_test_utility.py",
                "--model",
                cavity_model,
                "--test_data",
                cavity_test_data,
                "--mode",
                "all",
                "--provider",
                args.provider,
                "--model_name",
                model_name,
                "--llm_mode",
                "tool",
                "--llm_output_dir",
                str(cavity_ops_method),
                "--output_dir",
                str(cavity_results_method),
                "--save_plots",
            ]
            if args.save_raw:
                cavity_cmd.append("--save_raw")
            if args.reuse_operators:
                cavity_cmd.append("--reuse_operators")
            if args.cavity_res:
                cavity_cmd += ["--Re_test", *[str(v) for v in args.cavity_res]]
            cavity_cmd += ["--llm_method", method]
            run(cavity_cmd, f"Test cavity operators ({method})")

    print("\nAll results saved under:")
    print(f"  {base_dir}")

    if not args.no_write_summary:
        write_summary_split_errors(
            base_dir=base_dir,
            heat_model=heat_model,
            burgers_model=burgers_model,
            cavity_model=cavity_model,
        )


if __name__ == "__main__":
    main()
