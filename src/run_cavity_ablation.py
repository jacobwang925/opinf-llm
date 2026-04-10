#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cavity-only ablation runner for POD modes and OpInf regularization.

Runs two ablations:
1) POD modes sweep with fixed alpha/quad_alpha.
2) Alpha sweep (quad_alpha tied to alpha) with fixed POD modes.
"""

from __future__ import annotations

import argparse
import json
import pickle
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List

import numpy as np
import re


DEFAULT_CAVITY_RES = [60, 80, 90, 110, 120, 140]


def run(cmd: List[str], desc: str, allow_fail: bool = False, log_path: Path | None = None) -> None:
    print(f"\n==> {desc}")
    if not cmd:
        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(log_path, "a") as log_file:
                log_file.write(f"\n==> {desc}\n")
                log_file.write("exit_code=skipped\n")
        return
    print(" ".join(cmd))
    if allow_fail:
        result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    else:
        result = subprocess.run(cmd, check=not allow_fail)
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a") as log_file:
            log_file.write(f"\n==> {desc}\n")
            log_file.write(" ".join(cmd) + "\n")
            log_file.write(f"exit_code={result.returncode}\n")
            if allow_fail and result.returncode != 0:
                stderr_tail = (result.stderr or "").strip().splitlines()[-30:]
                if stderr_tail:
                    log_file.write("stderr_tail:\n")
                    log_file.write("\n".join(stderr_tail) + "\n")
    if allow_fail and result.returncode != 0:
        print(f"⚠ Command failed (exit {result.returncode}); continuing.")


def safe_tag(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def write_exact_cavity_operators(model_path: Path, output_dir: Path) -> None:
    """Write exact trained operators to match LLM output format."""
    import numpy as np

    with open(model_path, "rb") as f:
        model_data = pickle.load(f)

    for item in model_data["per_Re_models"]:
        re_val = item["Re"]
        out_path = output_dir / f"llm_cavity_operators_Re{re_val}.json"
        if out_path.exists():
            continue

        operators = {}
        for name in ["H", "A", "B", "C"]:
            arr = np.array(item[name])
            operators[name] = {
                "values": arr.tolist(),
                "shape": list(arr.shape),
                "norm": float(np.linalg.norm(arr)),
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
            }

        output_data = {
            "query_Re": re_val,
            "llm_provider": "exact",
            "llm_model": "opinf",
            "predicted_operators": {
                "operators": operators,
                "method": "exact",
                "Re_query": re_val,
                "success": True,
            },
        }
        with open(out_path, "w") as f:
            json.dump(output_data, f, indent=2)


def compute_split_errors_cavity(raw_files, area_scale, T_train):
    """Compute mean errors over trajectories for cavity raw files."""
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


def summarize_cavity_results(model_path: Path, base_dir: Path) -> None:
    with open(model_path, "rb") as f:
        cav_model = pickle.load(f)
    t_train = cav_model["t_eval"][-1]
    x = cav_model["x"]
    dx = x[1] - x[0]
    dA = dx * dx

    summary = {"cavity": {}}
    for method in ["interpolation", "regression"]:
        folder = base_dir / "cavity_test_results" / method
        if not folder.exists():
            continue
        by_re = {}
        for raw_path in sorted(folder.glob("cavity_Re*_traj*_raw.npz")):
            match = re.search(r"Re([0-9.]+)_traj", raw_path.name)
            if not match:
                continue
            re_val = float(match.group(1))
            by_re.setdefault(re_val, []).append(raw_path)
        for re_val, files in by_re.items():
            summary["cavity"].setdefault(method, {})[str(re_val)] = compute_split_errors_cavity(
                files, dA, t_train
            )

    out_path = base_dir / "summary_split_errors.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved split-error summary to: {out_path}")


def _model_energy_and_norm(model_path: Path) -> tuple[float | None, float | None]:
    """Return (energy_percent, mean_operator_norm) for a trained model."""
    with open(model_path, "rb") as f:
        model_data = pickle.load(f)

    config = model_data.get("config", {})
    energy_value = config.get("energy_fraction", config.get("energy_captured"))
    energy_percent = None
    if energy_value is not None:
        energy_value = float(energy_value)
        energy_percent = float(100.0 * energy_value) if energy_value <= 1.0 else energy_value

    per_re = model_data.get("per_Re_models", [])
    if not per_re:
        return energy_percent, None

    norms = []
    for item in per_re:
        h_norm = np.linalg.norm(item["H"])
        a_norm = np.linalg.norm(item["A"])
        b_norm = np.linalg.norm(item["B"])
        c_norm = np.linalg.norm(item["C"])
        norms.append(float(np.sqrt(h_norm**2 + a_norm**2 + b_norm**2 + c_norm**2)))
    mean_norm = float(np.mean(norms)) if norms else None
    return energy_percent, mean_norm


def _aggregate_cavity_method_errors(summary_obj: dict, method: str) -> dict:
    """Aggregate per-Re split errors for one method."""
    method_block = summary_obj.get("cavity", {}).get(method, {})
    first_vals = []
    second_vals = []
    n_traj_total = 0
    for _, rec in method_block.items():
        n_traj_total += int(rec.get("n_traj", 0) or 0)
        mf = rec.get("mean_first")
        ms = rec.get("mean_second")
        if mf is not None:
            first_vals.append(float(mf))
        if ms is not None:
            second_vals.append(float(ms))
    return {
        "n_traj_total": n_traj_total,
        "mean_first": float(np.mean(first_vals)) if first_vals else None,
        "mean_second": float(np.mean(second_vals)) if second_vals else None,
    }


def write_cavity_ablation_tables(output_base: Path, pod_modes: List[int], alpha_values: List[float]) -> None:
    """Write compact table-like summaries for POD and alpha sweeps."""
    rows_pod: list[dict] = []
    rows_alpha: list[dict] = []

    for r in pod_modes:
        setting_dir = output_base / "pod_modes" / f"pod_{r}"
        model_path = setting_dir / "cavity_model.pkl"
        summary_path = setting_dir / "summary_split_errors.json"
        if not (model_path.exists() and summary_path.exists()):
            continue
        with open(summary_path, "r") as f:
            s = json.load(f)
        energy_pct, mean_norm = _model_energy_and_norm(model_path)
        for method in ["interpolation", "regression"]:
            agg = _aggregate_cavity_method_errors(s, method)
            if agg["mean_first"] is None and agg["mean_second"] is None:
                continue
            rows_pod.append({
                "equation": "cavity",
                "method": method,
                "pod": r,
                "energy_percent": energy_pct,
                "mean_operator_norm": mean_norm,
                "error_first": agg["mean_first"],
                "error_second": agg["mean_second"],
                "n_traj_total": agg["n_traj_total"],
                "setting_dir": str(setting_dir),
            })

    for alpha in alpha_values:
        setting_dir = output_base / "alpha" / f"alpha_{safe_tag(alpha)}"
        model_path = setting_dir / "cavity_model.pkl"
        summary_path = setting_dir / "summary_split_errors.json"
        if not (model_path.exists() and summary_path.exists()):
            continue
        with open(summary_path, "r") as f:
            s = json.load(f)
        energy_pct, mean_norm = _model_energy_and_norm(model_path)
        for method in ["interpolation", "regression"]:
            agg = _aggregate_cavity_method_errors(s, method)
            if agg["mean_first"] is None and agg["mean_second"] is None:
                continue
            rows_alpha.append({
                "equation": "cavity",
                "method": method,
                "lambda": alpha,
                "energy_percent": energy_pct,
                "mean_operator_norm": mean_norm,
                "error_first": agg["mean_first"],
                "error_second": agg["mean_second"],
                "n_traj_total": agg["n_traj_total"],
                "setting_dir": str(setting_dir),
            })

    pod_json = output_base / "pod_modes" / "summary_table.json"
    alpha_json = output_base / "alpha" / "summary_table.json"
    pod_json.parent.mkdir(parents=True, exist_ok=True)
    alpha_json.parent.mkdir(parents=True, exist_ok=True)
    with open(pod_json, "w") as f:
        json.dump(rows_pod, f, indent=2)
    with open(alpha_json, "w") as f:
        json.dump(rows_alpha, f, indent=2)

    print(f"Saved POD summary table: {pod_json}")
    print(f"Saved alpha summary table: {alpha_json}")


def run_pipeline(
    dataset: Path,
    test_dataset: Path,
    output_dir: Path,
    n_modes: int,
    alpha: float,
    quad_alpha: float,
    provider: str,
    model_name: str,
    cavity_res: Iterable[float],
    reuse_operators: bool,
    save_raw: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "cavity_model.pkl"
    log_path = output_dir / "ablation_run.log"

    if not model_path.exists():
        run(
            [
                sys.executable,
                "dataset/cavity_2d_2_train_model_parametric.py",
                "--dataset",
                str(dataset),
                "--n_modes",
                str(n_modes),
                "--alpha",
                str(alpha),
                "--quad_alpha",
                str(quad_alpha),
                "--output",
                str(model_path),
            ],
            f"Train cavity model (r={n_modes}, alpha={alpha}, quad_alpha={quad_alpha})",
            allow_fail=True,
            log_path=log_path,
        )
        if not model_path.exists():
            run([], "Training failed; skipping this setting", allow_fail=True, log_path=log_path)
            return
    else:
        print(f"✓ Using existing model: {model_path}")

    for method in ["interpolation", "regression"]:
        results_dir = output_dir / "cavity_test_results" / method
        results_dir.mkdir(parents=True, exist_ok=True)
        # Save operators directly under results.
        ops_dir = results_dir
        write_exact_cavity_operators(model_path, ops_dir)

        cmd = [
            sys.executable,
            "src/test_utility_2d.py",
            "--model",
            str(model_path),
            "--test_data",
            str(test_dataset),
            "--mode",
            "all",
            "--provider",
            provider,
            "--model_name",
            model_name,
            "--llm_mode",
            "tool",
            "--llm_output_dir",
            str(ops_dir),
            "--output_dir",
            str(results_dir),
            "--save_plots",
            "--llm_method",
            method,
            "--Re_test",
            *[str(Re) for Re in cavity_res],
        ]
        if reuse_operators:
            cmd.append("--reuse_operators")
        if save_raw:
            cmd.append("--save_raw")
        run(cmd, f"Test cavity operators ({method})", allow_fail=True, log_path=log_path)

    if save_raw:
        summarize_cavity_results(model_path, output_dir)


def parse_list(values: str) -> List[float]:
    return [float(v) for v in values.split(",") if v.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run cavity-only ablation studies.")
    parser.add_argument("--dataset", type=str, default="dataset/cavity_dataset_train.pkl.gz")
    parser.add_argument("--test_dataset", type=str, default="dataset/cavity_dataset_test.pkl.gz")
    parser.add_argument("--provider", type=str, default="openai")
    parser.add_argument("--model_name", type=str, default="gpt-4o")
    parser.add_argument("--output_dir_base", type=str, default="cavity_ablation_results")
    parser.add_argument("--cavity_res", type=str, default="60,80,90,110,120,140")
    parser.add_argument("--reuse_operators", action="store_true")
    parser.add_argument("--save_raw", action="store_true")
    parser.add_argument("--pod_modes", type=str, default="4,5,6,7,8,9,10")
    parser.add_argument("--alpha_values", type=str, default="0,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1,10")
    parser.add_argument("--default_alpha", type=float, default=3.0)
    parser.add_argument("--default_pod", type=int, default=6)
    args = parser.parse_args()

    dataset = Path(args.dataset)
    test_dataset = Path(args.test_dataset)
    output_base = Path(args.output_dir_base)
    cavity_res = parse_list(args.cavity_res)

    pod_modes = [int(v) for v in args.pod_modes.split(",") if v.strip()]
    alpha_values = parse_list(args.alpha_values)

    # Ablation A: POD modes sweep (alpha/quad_alpha fixed)
    for r in pod_modes:
        out_dir = output_base / "pod_modes" / f"pod_{r}"
        run_pipeline(
            dataset=dataset,
            test_dataset=test_dataset,
            output_dir=out_dir,
            n_modes=r,
            alpha=args.default_alpha,
            quad_alpha=args.default_alpha,
            provider=args.provider,
            model_name=args.model_name,
            cavity_res=cavity_res,
            reuse_operators=args.reuse_operators,
            save_raw=args.save_raw,
        )

    # Ablation B: alpha sweep (POD fixed, quad_alpha tied to alpha)
    for alpha in alpha_values:
        out_dir = output_base / "alpha" / f"alpha_{safe_tag(alpha)}"
        run_pipeline(
            dataset=dataset,
            test_dataset=test_dataset,
            output_dir=out_dir,
            n_modes=args.default_pod,
            alpha=alpha,
            quad_alpha=alpha,
            provider=args.provider,
            model_name=args.model_name,
            cavity_res=cavity_res,
            reuse_operators=args.reuse_operators,
            save_raw=args.save_raw,
        )

    # Aggregate table-like summaries across settings for paper/reporting.
    write_cavity_ablation_tables(output_base, pod_modes, alpha_values)


if __name__ == "__main__":
    main()
