#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Heat-only ablation runner for POD modes and OpInf ridge alpha.

Runs two ablations:
1) POD modes sweep with fixed ridge alpha.
2) Ridge alpha sweep with fixed POD modes.
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

from run_three_equations_workflow import compute_split_errors


DEFAULT_HEAT_NUS = [0.5, 1.0, 3.0]


def run(cmd: List[str], desc: str, allow_fail: bool = False, log_path: Path | None = None) -> None:
    print(f"\n==> {desc}")
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


def resolve_operator_path(base_dir: Path, prefix: str, nu: float) -> Path:
    preferred = base_dir / f"{prefix}_nu{nu}.json"
    if preferred.exists():
        return preferred
    fallback = base_dir / f"tool_calling_operators_nu{nu}.json"
    if fallback.exists():
        return fallback
    return preferred


def write_exact_operators(model_path: Path, output_dir: Path) -> None:
    with open(model_path, "rb") as f:
        model_data = pickle.load(f)

    for item in model_data["per_nu_models"]:
        nu = item["nu"]
        out_path = output_dir / f"llm_heat_nu{nu}.json"
        if out_path.exists():
            continue
        output_data = {
            "query_nu": nu,
            "predicted_operators": {
                "nu": nu,
                "operators": {
                    "A": item["A"].tolist(),
                    "B": item["B"].tolist(),
                    "C": item["C"].tolist(),
                },
            },
        }
        with open(out_path, "w") as f:
            json.dump(output_data, f, indent=2)


def run_heat_tests(
    model_path: Path,
    test_dataset: Path | None,
    output_dir: Path,
    ops_dir: Path,
    nus: Iterable[float],
    save_raw: bool,
    log_path: Path | None,
) -> None:
    for nu in nus:
        pred_path = resolve_operator_path(ops_dir, "llm_heat", nu)
        cmd = [
            sys.executable,
            "test_llm_operators_properly.py",
            "--predicted",
            str(pred_path),
            "--model",
            str(model_path),
            "--save_plots",
            "--output_dir",
            str(output_dir),
            "--test_T_factor",
            "2.0",
        ]
        if save_raw:
            cmd.append("--save_raw")
        if test_dataset and test_dataset.exists():
            cmd += ["--dataset", str(test_dataset)]
        run(cmd, f"Test heat operators for nu={nu}", allow_fail=True, log_path=log_path)


def summarize_heat_results(model_path: Path, base_dir: Path) -> None:
    with open(model_path, "rb") as f:
        model_data = pickle.load(f)

    t_train = model_data["t_eval"][-1]
    x_grid = model_data["x_grid"]
    dx = x_grid[1] - x_grid[0]

    summary = {"heat": {}}
    for method in ["interpolation", "regression"]:
        folder = base_dir / "heat_test_results" / method
        if not folder.exists():
            continue
        by_param = {}
        for raw_path in sorted(folder.glob("llm_heat_nu*_raw.npz")):
            name = raw_path.name
            match = re.search(r"nu([0-9.]+)", name)
            if not match:
                continue
            try:
                nu_val = float(match.group(1))
            except ValueError:
                continue
            by_param.setdefault(nu_val, []).append(raw_path)
        for nu_val, files in by_param.items():
            summary["heat"].setdefault(method, {})[str(nu_val)] = compute_split_errors(
                files, dx, t_train
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
    ridge_alpha: float,
    provider: str,
    model_name: str,
    heat_nus: Iterable[float],
    reuse_operators: bool,
    save_raw: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "heat_model.pkl"
    log_path = output_dir / "ablation_run.log"

    if not model_path.exists():
        run(
            [
                sys.executable,
                "parametric_heat_2_train_model.py",
                "--dataset",
                str(dataset),
                "--n_modes",
                str(n_modes),
                "--ridge_alpha",
                str(ridge_alpha),
                "--output",
                str(model_path),
            ],
            f"Train heat model (r={n_modes}, alpha={ridge_alpha})",
            log_path=log_path,
        )
    else:
        print(f"✓ Using existing model: {model_path}")

    with open(model_path, "rb") as f:
        model_data = pickle.load(f)
    train_nus = [p["nu"] for p in model_data["per_nu_models"]]
    unseen_nus = [nu for nu in heat_nus if nu not in train_nus]

    for method in ["interpolation", "regression"]:
        ops_dir = output_dir / "operators" / "heat" / method
        results_dir = output_dir / "heat_test_results" / method
        ops_dir.mkdir(parents=True, exist_ok=True)
        results_dir.mkdir(parents=True, exist_ok=True)

        if unseen_nus and not reuse_operators:
            run(
                [
                    sys.executable,
                    "llm_tool_calling_interpolation.py",
                    "--model_pkl",
                    str(model_path),
                    "--query_nu_values",
                    *[str(nu) for nu in unseen_nus],
                    "--provider",
                    provider,
                    "--model",
                    model_name,
                    "--method",
                    method,
                    "--output",
                    str(ops_dir / f"llm_heat_{method}_batch.json"),
                ],
                f"Generate heat operators (batch, {method})",
                log_path=log_path,
            )
        write_exact_operators(model_path, ops_dir)
        run_heat_tests(model_path, test_dataset, results_dir, ops_dir, heat_nus, save_raw, log_path)

    if save_raw:
        summarize_heat_results(model_path, output_dir)


def parse_list(values: str) -> List[float]:
    return [float(v) for v in values.split(",") if v.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run heat-only ablation studies.")
    parser.add_argument("--dataset", type=str, default="heat_dataset.pkl.gz")
    parser.add_argument("--test_dataset", type=str, default="heat_dataset_test.pkl.gz")
    parser.add_argument("--provider", type=str, default="openai")
    parser.add_argument("--model_name", type=str, default="gpt-4o")
    parser.add_argument("--output_dir_base", type=str, default="heat_ablation_results")
    parser.add_argument("--heat_nus", type=str, default="0.5,1.0,3.0")
    parser.add_argument("--reuse_operators", action="store_true")
    parser.add_argument("--save_raw", action="store_true")
    parser.add_argument("--pod_modes", type=str, default="4,5,6,7,8,9,10")
    parser.add_argument("--alpha_values", type=str, default="0,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1,10")
    parser.add_argument("--default_alpha", type=float, default=1e-4)
    parser.add_argument("--default_pod", type=int, default=6)
    args = parser.parse_args()

    dataset = Path(args.dataset)
    test_dataset = Path(args.test_dataset)
    output_base = Path(args.output_dir_base)
    heat_nus = parse_list(args.heat_nus)

    pod_modes = [int(v) for v in args.pod_modes.split(",") if v.strip()]
    alpha_values = parse_list(args.alpha_values)

    # Ablation A: POD modes sweep (alpha fixed)
    for r in pod_modes:
        out_dir = output_base / "pod_modes" / f"pod_{r}"
        run_pipeline(
            dataset=dataset,
            test_dataset=test_dataset,
            output_dir=out_dir,
            n_modes=r,
            ridge_alpha=args.default_alpha,
            provider=args.provider,
            model_name=args.model_name,
            heat_nus=heat_nus,
            reuse_operators=args.reuse_operators,
            save_raw=args.save_raw,
        )

    # Ablation B: alpha sweep (POD fixed)
    for alpha in alpha_values:
        out_dir = output_base / "alpha" / f"alpha_{safe_tag(alpha)}"
        run_pipeline(
            dataset=dataset,
            test_dataset=test_dataset,
            output_dir=out_dir,
            n_modes=args.default_pod,
            ridge_alpha=alpha,
            provider=args.provider,
            model_name=args.model_name,
            heat_nus=heat_nus,
            reuse_operators=args.reuse_operators,
            save_raw=args.save_raw,
        )


if __name__ == "__main__":
    main()
