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
                "cavity_2d_2_train_model_parametric.py",
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
        ops_dir = output_dir / "operators" / "cavity" / method
        results_dir = output_dir / "cavity_test_results" / method
        ops_dir.mkdir(parents=True, exist_ok=True)
        results_dir.mkdir(parents=True, exist_ok=True)
        write_exact_cavity_operators(model_path, ops_dir)

        cmd = [
            sys.executable,
            "cavity_test_utility.py",
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
    parser.add_argument("--dataset", type=str, default="cavity_dataset_train.pkl.gz")
    parser.add_argument("--test_dataset", type=str, default="cavity_dataset_test.pkl.gz")
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


if __name__ == "__main__":
    main()
