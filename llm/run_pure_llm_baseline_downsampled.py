#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pure-LLM baseline: downsampled trajectories with compact JSON output.

One prompt per parameter (single trajectory). LLM outputs fixed-size arrays
with 3-decimal precision. Errors are computed against downsampled FOM data.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt

from load_env import load_env
from llm_tool_calling_provider import call_llm_text


HEAT_NUS = [0.5, 1.0, 3.0]
BURGERS_NUS = [0.03, 0.07]
CAVITY_RES = [60, 80, 90, 110, 120, 140]

HEAT_NX = 16
HEAT_NT = 41
BURGERS_NX = 16
BURGERS_NT = 41
CAVITY_NXY = 6
CAVITY_NT = 21


@dataclass
class CaseResult:
    equation: str
    param: str
    method: str
    success: bool
    error: str | None
    mean_full: float | None
    mean_first: float | None
    mean_second: float | None


def load_pickle_auto(path: Path) -> Any:
    import gzip
    import pickle

    with open(path, "rb") as f:
        magic = f.read(2)
        f.seek(0)
        if magic == b"\x1f\x8b":
            with gzip.open(path, "rb") as gz:
                return pickle.load(gz)
        return pickle.load(f)


def downsample_indices(n: int, target: int) -> np.ndarray:
    if target <= 1:
        return np.array([0], dtype=int)
    return np.linspace(0, n - 1, target).round().astype(int)


def rel_l2(y_ref: np.ndarray, y_pred: np.ndarray, space_scale: float, dt: float) -> float:
    err = y_ref - y_pred
    norm_err = math.sqrt(float(np.sum(err**2) * space_scale * dt))
    norm_ref = math.sqrt(float(np.sum(y_ref**2) * space_scale * dt))
    return norm_err / (norm_ref + 1e-14)


def split_errors(y_ref: np.ndarray, y_pred: np.ndarray, space_scale: float, t: np.ndarray, t_train: float) -> Dict[str, float | None]:
    if len(t) < 2:
        return {"mean_full": None, "mean_first": None, "mean_second": None}
    dt = float(t[1] - t[0])
    full = rel_l2(y_ref, y_pred, space_scale, dt)
    split_idx = int(np.searchsorted(t, t_train))
    if split_idx <= 1 or split_idx >= y_ref.shape[1]:
        return {"mean_full": full, "mean_first": None, "mean_second": None}
    first = rel_l2(y_ref[:, :split_idx], y_pred[:, :split_idx], space_scale, dt)
    second = rel_l2(y_ref[:, split_idx:], y_pred[:, split_idx:], space_scale, dt)
    return {"mean_full": full, "mean_first": first, "mean_second": second}


def build_prompt_header() -> str:
    return (
        "You are given a PDE, full IC/BC specs, parameter values, and input signals. "
        "Return ONLY a valid JSON object with numeric arrays at 3-decimal precision. "
        "No extra text, no code fences.\n"
    )


def format_list(values: np.ndarray) -> str:
    return "[" + ",".join(f"{v:.3f}" for v in values.tolist()) + "]"


def format_matrix(mat: np.ndarray) -> str:
    rows = [format_list(row) for row in mat]
    return "[" + ",".join(rows) + "]"


def format_tensor(tensor: np.ndarray) -> str:
    mats = [format_matrix(tensor[:, :, k]) for k in range(tensor.shape[2])]
    return "[" + ",".join(mats) + "]"


def format_re_value(re_value: float) -> str:
    return f"{re_value:.1f}"


def write_prompt(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def parse_json_from_text(text: str) -> Dict[str, Any]:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in response.")
    payload = cleaned[start:end + 1]
    return json.loads(payload)


def ensure_shape(array: np.ndarray, shape: Tuple[int, ...], name: str) -> None:
    if array.shape != shape:
        raise ValueError(f"{name} has shape {array.shape}, expected {shape}")


def normalize_time_space(u: np.ndarray, nx: int, nt: int, name: str) -> np.ndarray:
    if u.shape == (nx, nt):
        return u
    if u.shape == (nt, nx):
        return u.T
    if u.shape[1] == nx and u.shape[0] >= nt:
        return u[:nt, :].T
    raise ValueError(f"{name} has shape {u.shape}, expected ({nx}, {nt}) or ({nt}, {nx})")


def normalize_cavity(field: np.ndarray, nxy: int, nt: int, name: str) -> np.ndarray:
    if field.shape == (nxy, nxy, nt):
        return field
    if field.shape == (nt, nxy, nxy):
        return np.transpose(field, (1, 2, 0))
    if field.shape[0] >= nt and field.shape[1] == nxy and field.shape[2] == nxy:
        trimmed = field[:nt, :, :]
        return np.transpose(trimmed, (1, 2, 0))
    if field.shape[0] < nt and field.shape[1] == nxy and field.shape[2] == nxy:
        pad_count = nt - field.shape[0]
        pad = np.repeat(field[-1:, :, :], pad_count, axis=0)
        padded = np.concatenate([field, pad], axis=0)
        return np.transpose(padded, (1, 2, 0))
    raise ValueError(f"{name} has shape {field.shape}, expected ({nxy}, {nxy}, {nt}) or ({nt}, {nxy}, {nxy})")


def downsample_heat_case(data: Dict[str, Any], nu: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    t_eval = data["t_eval"]
    t_train = data["t_eval_train"][-1]
    x = data["x_grid"]
    idx_nu = next(i for i, item in enumerate(data["per_nu_data"]) if item["nu"] == nu)
    item = data["per_nu_data"][idx_nu]
    y_list = item["lists"]["Y_test_list"]
    u_list = item["lists"]["U_test_list"]
    y = y_list[0]
    u = u_list[0]

    x_idx = downsample_indices(len(x), HEAT_NX)
    t_idx = downsample_indices(len(t_eval), HEAT_NT)
    x_ds = x[x_idx]
    t_ds = t_eval[t_idx]
    y_ds = y[np.ix_(x_idx, t_idx)]
    u_ds = u[t_idx]
    dx = float(x_ds[1] - x_ds[0])
    return x_ds, t_ds, y_ds, u_ds, t_train, dx


def downsample_burgers_case(data: Dict[str, Any], nu: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray], float, float]:
    t_eval = data["t_eval"]
    t_train = data["t_eval_train"][-1]
    x = data["x_fine"]
    nu_list = (data.get("config", {}) or {}).get("nu_list")
    if not nu_list or nu not in nu_list:
        raise ValueError(f"nu={nu} not found in burgers test dataset nu_list")
    idx_nu = list(nu_list).index(nu)
    item = data["per_nu_data"][idx_nu]
    y = item["lists"]["Y_test_list"][0]
    w1 = item["lists"]["w1_test_list"][0]
    w2 = item["lists"]["w2_test_list"][0]
    w3 = item["lists"]["w3_test_list"][0]

    x_idx = downsample_indices(len(x), BURGERS_NX)
    t_idx = downsample_indices(len(t_eval), BURGERS_NT)
    x_ds = x[x_idx]
    t_ds = t_eval[t_idx]
    y_ds = y[np.ix_(x_idx, t_idx)]
    inputs = {"w1": w1[t_idx], "w2": w2[t_idx], "w3": w3[t_idx]}
    dx = float(x_ds[1] - x_ds[0])
    return x_ds, t_ds, y_ds, inputs, t_train, dx


def downsample_cavity_case(data: Dict[str, Any], Re: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    t_eval = data["t_eval"]
    x = data["x"]
    y = data["y"]
    t_train = 2.0

    idx_re = next(i for i, item in enumerate(data["per_Re_data"]) if item["Re"] == Re)
    item = data["per_Re_data"][idx_re]
    y_omega = item["test"]["Y_omega"]
    y_psi = item["test"]["Y_psi"]
    u_lid = item["test"]["U_lid"]

    grid_size = int(data["config"]["grid_size"])
    nt = y_omega.shape[1]
    y_omega = y_omega.reshape(grid_size, grid_size, nt)
    y_psi = y_psi.reshape(grid_size, grid_size, nt)

    x_idx = downsample_indices(len(x), CAVITY_NXY)
    y_idx = downsample_indices(len(y), CAVITY_NXY)
    t_idx = downsample_indices(len(t_eval), CAVITY_NT)

    x_ds = x[x_idx]
    y_ds = y[y_idx]
    t_ds = t_eval[t_idx]

    omega_ds = y_omega[np.ix_(x_idx, y_idx, t_idx)]
    psi_ds = y_psi[np.ix_(x_idx, y_idx, t_idx)]
    u_ds = u_lid[t_idx]

    dx = float(x_ds[1] - x_ds[0])
    dy = float(y_ds[1] - y_ds[0])
    return x_ds, y_ds, t_ds, omega_ds, psi_ds, u_ds, t_train, dx * dy


def build_heat_prompt(nu: float, x: np.ndarray, t: np.ndarray, u0: np.ndarray, u_bc: np.ndarray) -> str:
    header = build_prompt_header()
    body = (
        "Task: Heat equation u_t = nu * u_xx on x in [0,1].\\n"
        "Boundary conditions: u(0,t)=u(1,t)=u_bc(t).\\n"
        f"Parameter: nu = {nu}.\\n"
        f"Output grid: {HEAT_NX} spatial points, {HEAT_NT} time steps.\\n"
        "Return JSON with fields: equation, param, grid_shape, t_steps, x, t, u0, u.\\n"
        "All numeric values must be rounded to 3 decimals.\\n"
        f"x = {format_list(x)}\\n"
        f"t = {format_list(t)}\\n"
        f"u0(x) = {format_list(u0)}\\n"
        f"u_bc(t) = {format_list(u_bc)}\\n"
        "JSON schema:\\n"
        "{\"equation\":\"heat\",\"param\":{\"nu\":...},\"grid_shape\":[16],\"t_steps\":41,"
        "\"x\":[...],\"t\":[...],\"u0\":[...],\"u\":[[...]]}\\n"
    )
    return header + body


def build_burgers_prompt(nu: float, x: np.ndarray, t: np.ndarray, u0: np.ndarray, inputs: Dict[str, np.ndarray]) -> str:
    header = build_prompt_header()
    body = (
        "Task: Burgers equation u_t + u*u_x = nu*u_xx + s(x)*w3(t), x in [0,1].\\n"
        "Boundary: u(0,t)=w1(t), u(1,t)=w2(t).\\n"
        "Forcing shape: s(x)=cosh((x-0.5)/0.05)^(-1).\\n"
        f"Parameter: nu = {nu}.\\n"
        f"Output grid: {BURGERS_NX} spatial points, {BURGERS_NT} time steps.\\n"
        "Return JSON with fields: equation, param, grid_shape, t_steps, x, t, u0, w1, w2, w3, u.\\n"
        "All numeric values must be rounded to 3 decimals.\\n"
        f"x = {format_list(x)}\\n"
        f"t = {format_list(t)}\\n"
        f"u0(x) = {format_list(u0)}\\n"
        f"w1(t) = {format_list(inputs['w1'])}\\n"
        f"w2(t) = {format_list(inputs['w2'])}\\n"
        f"w3(t) = {format_list(inputs['w3'])}\\n"
        "JSON schema:\\n"
        "{\"equation\":\"burgers\",\"param\":{\"nu\":...},\"grid_shape\":[16],\"t_steps\":41,"
        "\"x\":[...],\"t\":[...],\"u0\":[...],\"w1\":[...],\"w2\":[...],\"w3\":[...],\"u\":[[...]]}\\n"
    )
    return header + body


def build_cavity_prompt(Re: float, x: np.ndarray, y: np.ndarray, t: np.ndarray, u_lid: np.ndarray) -> str:
    header = build_prompt_header()
    body = (
        "Task: 2D lid-driven cavity (vorticity/streamfunction).\\n"
        "Equations: omega_t + u*omega_x + v*omega_y = (1/Re)*(omega_xx+omega_yy), "
        "Laplacian(psi) = -omega, u=psi_y, v=-psi_x.\\n"
        "No-slip walls. Lid velocity u_lid(x,t)=a(x)*f(t), v=0.\\n"
        "a(x)=1+0.3*sin(2*pi*x)+0.2*x (fixed).\\n"
        "Initial condition: omega=0, psi=0.\\n"
        f"Parameter: Re = {Re}.\\n"
        f"Output grid: {CAVITY_NXY}x{CAVITY_NXY}, {CAVITY_NT} time steps.\\n"
        "Return JSON with fields: equation, param, grid_shape, t_steps, x, y, t, u_lid, omega, psi.\\n"
        "All numeric values must be rounded to 3 decimals.\\n"
        f"x = {format_list(x)}\\n"
        f"y = {format_list(y)}\\n"
        f"t = {format_list(t)}\\n"
        f"u_lid(t) = {format_list(u_lid)}\\n"
        "JSON schema:\\n"
        f"{{\"equation\":\"cavity\",\"param\":{{\"Re\":...}},\"grid_shape\":[{CAVITY_NXY},{CAVITY_NXY}],\"t_steps\":{CAVITY_NT},"
        "\"x\":[...],\"y\":[...],\"t\":[...],\"u_lid\":[...],\"omega\":[[[...]]],\"psi\":[[[...]]]}\\n"
    )
    return header + body


def run_case(
    prompt_path: Path,
    provider: str,
    model: str,
    raw_out: Path,
    retries: int,
    backoff: float,
    retry_until_success: bool,
    max_attempts_per_case: int,
    attempt_log: List[Dict[str, Any]],
    case_info: Dict[str, Any],
) -> Dict[str, Any]:
    prompt = prompt_path.read_text()
    messages = [{"role": "user", "content": prompt}]
    last_exc: Exception | None = None
    attempt = 0
    if retry_until_success:
        attempt_limit = max_attempts_per_case
    else:
        attempt_limit = max(1, retries)
    while True:
        attempt += 1
        start = time.time()
        attempt_total = "∞" if retry_until_success and max_attempts_per_case == 0 else str(attempt_limit)
        print(f"[pure-llm] {case_info['equation']} {case_info['param']} attempt {attempt}/{attempt_total}...")
        try:
            response = call_llm_text(provider, messages, model)
            payload = parse_json_from_text(response)
            attempt_log.append(
                {
                    **case_info,
                    "attempt": attempt,
                    "status": "success",
                    "error": None,
                    "elapsed_s": round(time.time() - start, 3),
                }
            )
            raw_out.parent.mkdir(parents=True, exist_ok=True)
            with open(raw_out, "w") as f:
                json.dump(payload, f, indent=2)
            return payload
        except Exception as exc:
            last_exc = exc
            attempt_log.append(
                {
                    **case_info,
                    "attempt": attempt,
                    "status": "failed",
                    "error": str(exc),
                    "elapsed_s": round(time.time() - start, 3),
                }
            )
            print(f"[pure-llm] {case_info['equation']} {case_info['param']} failed: {exc}")
            if retry_until_success:
                if max_attempts_per_case > 0 and attempt >= max_attempts_per_case:
                    raise
            else:
                if attempt >= attempt_limit:
                    raise
            time.sleep(backoff * (2 ** (attempt - 1)))
    raise last_exc  # pragma: no cover


def count_attempts(attempt_log: List[Dict[str, Any]], equation: str, param: str) -> int:
    return sum(
        1
        for item in attempt_log
        if item.get("equation") == equation and item.get("param") == param
    )


def save_heat_burgers_plot(
    y_ref: np.ndarray,
    y_pred: np.ndarray,
    x: np.ndarray,
    t: np.ndarray,
    out_path: Path,
    title_prefix: str,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    t_mesh, x_mesh = np.meshgrid(t, x)

    im0 = axes[0].pcolormesh(t_mesh, x_mesh, y_ref, cmap="RdBu_r", shading="auto")
    axes[0].set_title(f"{title_prefix} FOM")
    axes[0].set_xlabel("t")
    axes[0].set_ylabel("x")
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].pcolormesh(t_mesh, x_mesh, y_pred, cmap="RdBu_r", shading="auto")
    axes[1].set_title(f"{title_prefix} LLM")
    axes[1].set_xlabel("t")
    axes[1].set_ylabel("x")
    plt.colorbar(im1, ax=axes[1])

    err = np.abs(y_ref - y_pred)
    im2 = axes[2].pcolormesh(t_mesh, x_mesh, err, cmap="YlOrRd", shading="auto")
    axes[2].set_title(f"{title_prefix} |Error|")
    axes[2].set_xlabel("t")
    axes[2].set_ylabel("x")
    plt.colorbar(im2, ax=axes[2])

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_cavity_plot(
    omega_fom: np.ndarray,
    omega_llm: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
    error_pct: float,
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    x_grid, y_grid = np.meshgrid(x, y, indexing="ij")
    fig, axs = plt.subplots(2, 5, figsize=(24, 10))
    m = omega_fom.shape[2]
    times_idx = [0, m // 4, m // 2, 3 * m // 4, m - 1]
    times_val = [t[idx] for idx in times_idx]

    for col, (t_idx, t_val) in enumerate(zip(times_idx, times_val)):
        im0 = axs[0, col].contourf(x_grid, y_grid, omega_fom[:, :, t_idx], 50, cmap="RdBu_r")
        axs[0, col].set_title(f"FOM ω at t={t_val:.2f}s")
        axs[0, col].set_xlabel("x")
        axs[0, col].set_ylabel("y")
        axs[0, col].set_aspect("equal")
        plt.colorbar(im0, ax=axs[0, col])

        im1 = axs[1, col].contourf(x_grid, y_grid, omega_llm[:, :, t_idx], 50, cmap="RdBu_r")
        axs[1, col].set_title(f"LLM ω at t={t_val:.2f}s")
        axs[1, col].set_xlabel("x")
        axs[1, col].set_ylabel("y")
        axs[1, col].set_aspect("equal")
        plt.colorbar(im1, ax=axs[1, col])

    fig.suptitle(f"Re={out_path.stem.split('_')[1][2:]} (LLM), Error={error_pct:.2f}%", fontsize=16, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run pure-LLM baseline with downsampled outputs.")
    parser.add_argument("--provider", type=str, default="openai")
    parser.add_argument("--model_name", type=str, default="gpt-4o")
    parser.add_argument("--run_id", type=str, default=None,
                        help="Run identifier. Default: timestamp YYYYMMDD-HHMMSS")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--output_dir_base", type=str, default="llm_runs")
    parser.add_argument("--prompt_dir", type=str, default=None)
    parser.add_argument("--prompt_dir_base", type=str, default="pure_llm_baseline_prompts")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--save_plots", action="store_true")
    parser.add_argument("--reuse_raw", action="store_true",
                        help="Reuse existing raw outputs instead of calling the API")
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry_backoff", type=float, default=2.0)
    parser.add_argument("--retry_until_success", action="store_true",
                        help="Keep retrying a case until it succeeds (bounded by --max_attempts_per_case).")
    parser.add_argument("--max_attempts_per_case", type=int, default=10,
                        help="Max attempts per case when --retry_until_success is set. Use 0 for unlimited.")
    parser.add_argument("--retry_on_validation_failure", action="store_true",
                        help="Retry the API call if validation/shape checks fail.")
    parser.add_argument("--request_timeout", type=float, default=None,
                        help="Per-request timeout (seconds) for LLM calls")
    parser.add_argument("--heat_nus", nargs="+", type=float, default=None)
    parser.add_argument("--burgers_nus", nargs="+", type=float, default=None)
    parser.add_argument("--cavity_res", nargs="+", type=float, default=None)
    args = parser.parse_args()

    load_env()
    if args.request_timeout is not None:
        import os
        os.environ["LLM_REQUEST_TIMEOUT"] = str(args.request_timeout)
    safe_model = re.sub(r"[^a-zA-Z0-9._-]+", "_", args.model_name)
    run_id = args.run_id or time.strftime("%Y%m%d-%H%M%S")
    out_dir = Path(args.output_dir) if args.output_dir else Path(args.output_dir_base) / safe_model / run_id
    prompt_dir = Path(args.prompt_dir) if args.prompt_dir else (out_dir / "prompts")
    raw_dir = out_dir / "raw"
    plot_dir = out_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[pure-llm] run_id={run_id}")
    print(f"[pure-llm] run_dir={out_dir}")

    results: List[CaseResult] = []
    failures: List[Dict[str, Any]] = []
    attempt_log: List[Dict[str, Any]] = []

    heat_data = load_pickle_auto(Path("dataset/heat_dataset_test.pkl.gz"))
    burgers_data = load_pickle_auto(Path("dataset/burgers_dataset_test.pkl.gz"))
    cavity_data = load_pickle_auto(Path("dataset/cavity_dataset_test.pkl.gz"))

    heat_nus = args.heat_nus or HEAT_NUS
    burgers_nus = args.burgers_nus or BURGERS_NUS
    cavity_res = args.cavity_res or CAVITY_RES

    # Heat cases
    for nu in heat_nus:
        print(f"[pure-llm] heat nu={nu} start")
        x_ds, t_ds, y_ds, u_bc, t_train, dx = downsample_heat_case(heat_data, nu)
        u0 = y_ds[:, 0]
        prompt = build_heat_prompt(nu, x_ds, t_ds, u0, u_bc)
        prompt_path = prompt_dir / f"heat_nu{nu}_traj1.txt"
        write_prompt(prompt_path, prompt)
        raw_out = raw_dir / f"heat_nu{nu}_traj1.json"
        payload = None
        if args.execute and args.reuse_raw and raw_out.exists():
            try:
                payload = json.loads(raw_out.read_text())
                u = normalize_time_space(np.array(payload["u"], dtype=float), HEAT_NX, HEAT_NT, "u")
                errs = split_errors(y_ds, u, dx, t_ds, t_train)
                if args.save_plots:
                    save_heat_burgers_plot(y_ds, u, x_ds, t_ds, plot_dir / f"llm_heat_nu{nu}_test.png", f"Heat ν={nu}")
                results.append(CaseResult("heat", f"nu={nu}", "llm", True, None,
                                          errs["mean_full"], errs["mean_first"], errs["mean_second"]))
            except Exception as exc:
                failures.append({"equation": "heat", "param": nu, "error": str(exc)})
                results.append(CaseResult("heat", f"nu={nu}", "llm", False, str(exc),
                                          None, None, None))
        elif args.execute:
            done = False
            while True:
                attempts_used = count_attempts(attempt_log, "heat", f"nu={nu}")
                if args.max_attempts_per_case > 0 and attempts_used >= args.max_attempts_per_case:
                    failures.append({"equation": "heat", "param": nu, "error": "max_attempts_per_case_exceeded"})
                    results.append(CaseResult("heat", f"nu={nu}", "llm", False, "max_attempts_per_case_exceeded",
                                              None, None, None))
                    done = True
                    break
                remaining_attempts = args.max_attempts_per_case - attempts_used if args.max_attempts_per_case > 0 else 0
                try:
                    payload = run_case(
                        prompt_path,
                        args.provider,
                        args.model_name,
                        raw_out,
                        args.retries,
                        args.retry_backoff,
                        args.retry_until_success,
                        remaining_attempts,
                        attempt_log,
                        {"equation": "heat", "param": f"nu={nu}", "raw_path": str(raw_out)},
                    )
                except Exception as exc:
                    failures.append({"equation": "heat", "param": nu, "error": str(exc)})
                    results.append(CaseResult("heat", f"nu={nu}", "llm", False, str(exc),
                                              None, None, None))
                    done = True
                    break
                try:
                    u = normalize_time_space(np.array(payload["u"], dtype=float), HEAT_NX, HEAT_NT, "u")
                    errs = split_errors(y_ds, u, dx, t_ds, t_train)
                    if args.save_plots:
                        save_heat_burgers_plot(y_ds, u, x_ds, t_ds, plot_dir / f"llm_heat_nu{nu}_test.png", f"Heat ν={nu}")
                    results.append(CaseResult("heat", f"nu={nu}", "llm", True, None,
                                              errs["mean_full"], errs["mean_first"], errs["mean_second"]))
                    done = True
                    break
                except Exception as exc:
                    attempt_log.append(
                        {
                            "equation": "heat",
                            "param": f"nu={nu}",
                            "raw_path": str(raw_out),
                            "attempt": attempts_used + 1,
                            "status": "validation_failed",
                            "error": str(exc),
                        }
                    )
                    print(f"[pure-llm] heat nu={nu} validation failed: {exc}")
                    if args.retry_on_validation_failure:
                        if raw_out.exists():
                            raw_out.unlink()
                        continue
                    failures.append({"equation": "heat", "param": nu, "error": str(exc)})
                    results.append(CaseResult("heat", f"nu={nu}", "llm", False, str(exc),
                                              None, None, None))
                    done = True
                    break
            if not done:
                results.append(CaseResult("heat", f"nu={nu}", "llm", False, "validation_failed",
                                          None, None, None))
        else:
            results.append(CaseResult("heat", f"nu={nu}", "llm", False, "not_executed", None, None, None))

    # Burgers cases
    for nu in burgers_nus:
        print(f"[pure-llm] burgers nu={nu} start")
        x_ds, t_ds, y_ds, inputs, t_train, dx = downsample_burgers_case(burgers_data, nu)
        u0 = y_ds[:, 0]
        prompt = build_burgers_prompt(nu, x_ds, t_ds, u0, inputs)
        prompt_path = prompt_dir / f"burgers_nu{nu}_traj1.txt"
        write_prompt(prompt_path, prompt)
        raw_out = raw_dir / f"burgers_nu{nu}_traj1.json"
        payload = None
        if args.execute and args.reuse_raw and raw_out.exists():
            try:
                payload = json.loads(raw_out.read_text())
                u = normalize_time_space(np.array(payload["u"], dtype=float), BURGERS_NX, BURGERS_NT, "u")
                errs = split_errors(y_ds, u, dx, t_ds, t_train)
                if args.save_plots:
                    save_heat_burgers_plot(y_ds, u, x_ds, t_ds, plot_dir / f"llm_burgers_nu{nu}_test.png", f"Burgers ν={nu}")
                results.append(CaseResult("burgers", f"nu={nu}", "llm", True, None,
                                          errs["mean_full"], errs["mean_first"], errs["mean_second"]))
            except Exception as exc:
                failures.append({"equation": "burgers", "param": nu, "error": str(exc)})
                results.append(CaseResult("burgers", f"nu={nu}", "llm", False, str(exc),
                                          None, None, None))
        elif args.execute:
            done = False
            while True:
                attempts_used = count_attempts(attempt_log, "burgers", f"nu={nu}")
                if args.max_attempts_per_case > 0 and attempts_used >= args.max_attempts_per_case:
                    failures.append({"equation": "burgers", "param": nu, "error": "max_attempts_per_case_exceeded"})
                    results.append(CaseResult("burgers", f"nu={nu}", "llm", False, "max_attempts_per_case_exceeded",
                                              None, None, None))
                    done = True
                    break
                remaining_attempts = args.max_attempts_per_case - attempts_used if args.max_attempts_per_case > 0 else 0
                try:
                    payload = run_case(
                        prompt_path,
                        args.provider,
                        args.model_name,
                        raw_out,
                        args.retries,
                        args.retry_backoff,
                        args.retry_until_success,
                        remaining_attempts,
                        attempt_log,
                        {"equation": "burgers", "param": f"nu={nu}", "raw_path": str(raw_out)},
                    )
                except Exception as exc:
                    failures.append({"equation": "burgers", "param": nu, "error": str(exc)})
                    results.append(CaseResult("burgers", f"nu={nu}", "llm", False, str(exc),
                                              None, None, None))
                    done = True
                    break
                try:
                    u = normalize_time_space(np.array(payload["u"], dtype=float), BURGERS_NX, BURGERS_NT, "u")
                    errs = split_errors(y_ds, u, dx, t_ds, t_train)
                    if args.save_plots:
                        save_heat_burgers_plot(y_ds, u, x_ds, t_ds, plot_dir / f"llm_burgers_nu{nu}_test.png", f"Burgers ν={nu}")
                    results.append(CaseResult("burgers", f"nu={nu}", "llm", True, None,
                                              errs["mean_full"], errs["mean_first"], errs["mean_second"]))
                    done = True
                    break
                except Exception as exc:
                    attempt_log.append(
                        {
                            "equation": "burgers",
                            "param": f"nu={nu}",
                            "raw_path": str(raw_out),
                            "attempt": attempts_used + 1,
                            "status": "validation_failed",
                            "error": str(exc),
                        }
                    )
                    print(f"[pure-llm] burgers nu={nu} validation failed: {exc}")
                    if args.retry_on_validation_failure:
                        if raw_out.exists():
                            raw_out.unlink()
                        continue
                    failures.append({"equation": "burgers", "param": nu, "error": str(exc)})
                    results.append(CaseResult("burgers", f"nu={nu}", "llm", False, str(exc),
                                              None, None, None))
                    done = True
                    break
            if not done:
                results.append(CaseResult("burgers", f"nu={nu}", "llm", False, "validation_failed",
                                          None, None, None))
        else:
            results.append(CaseResult("burgers", f"nu={nu}", "llm", False, "not_executed", None, None, None))

    # Cavity cases
    for Re in cavity_res:
        print(f"[pure-llm] cavity Re={Re} start")
        x_ds, y_ds, t_ds, omega_ds, psi_ds, u_lid, t_train, dA = downsample_cavity_case(cavity_data, Re)
        prompt = build_cavity_prompt(Re, x_ds, y_ds, t_ds, u_lid)
        re_label = format_re_value(Re)
        prompt_path = prompt_dir / f"cavity_Re{re_label}_traj1.txt"
        write_prompt(prompt_path, prompt)
        raw_out = raw_dir / f"cavity_Re{re_label}_traj1.json"
        payload = None
        if args.execute and args.reuse_raw and raw_out.exists():
            try:
                payload = json.loads(raw_out.read_text())
                omega = normalize_cavity(np.array(payload["omega"], dtype=float), CAVITY_NXY, CAVITY_NT, "omega")
                psi = normalize_cavity(np.array(payload["psi"], dtype=float), CAVITY_NXY, CAVITY_NT, "psi")
                y_ref = np.vstack([omega_ds.reshape(-1, CAVITY_NT), psi_ds.reshape(-1, CAVITY_NT)])
                y_pred = np.vstack([omega.reshape(-1, CAVITY_NT), psi.reshape(-1, CAVITY_NT)])
                errs = split_errors(y_ref, y_pred, dA, t_ds, t_train)
                if args.save_plots:
                    error_pct = errs["mean_full"] * 100 if errs["mean_full"] is not None else float("nan")
                    save_cavity_plot(
                        omega_ds,
                        omega,
                        x_ds,
                        y_ds,
                        t_ds,
                        error_pct,
                        plot_dir / f"cavity_Re{re_label}_llm_traj1.png",
                    )
                results.append(CaseResult("cavity", f"Re={Re}", "llm", True, None,
                                          errs["mean_full"], errs["mean_first"], errs["mean_second"]))
            except Exception as exc:
                failures.append({"equation": "cavity", "param": Re, "error": str(exc)})
                results.append(CaseResult("cavity", f"Re={Re}", "llm", False, str(exc),
                                          None, None, None))
        elif args.execute:
            done = False
            while True:
                attempts_used = count_attempts(attempt_log, "cavity", f"Re={Re}")
                if args.max_attempts_per_case > 0 and attempts_used >= args.max_attempts_per_case:
                    failures.append({"equation": "cavity", "param": Re, "error": "max_attempts_per_case_exceeded"})
                    results.append(CaseResult("cavity", f"Re={Re}", "llm", False, "max_attempts_per_case_exceeded",
                                              None, None, None))
                    done = True
                    break
                remaining_attempts = args.max_attempts_per_case - attempts_used if args.max_attempts_per_case > 0 else 0
                try:
                    payload = run_case(
                        prompt_path,
                        args.provider,
                        args.model_name,
                        raw_out,
                        args.retries,
                        args.retry_backoff,
                        args.retry_until_success,
                        remaining_attempts,
                        attempt_log,
                        {"equation": "cavity", "param": f"Re={Re}", "raw_path": str(raw_out)},
                    )
                except Exception as exc:
                    failures.append({"equation": "cavity", "param": Re, "error": str(exc)})
                    results.append(CaseResult("cavity", f"Re={Re}", "llm", False, str(exc),
                                              None, None, None))
                    done = True
                    break
                try:
                    omega = normalize_cavity(np.array(payload["omega"], dtype=float), CAVITY_NXY, CAVITY_NT, "omega")
                    psi = normalize_cavity(np.array(payload["psi"], dtype=float), CAVITY_NXY, CAVITY_NT, "psi")
                    y_ref = np.vstack([omega_ds.reshape(-1, CAVITY_NT), psi_ds.reshape(-1, CAVITY_NT)])
                    y_pred = np.vstack([omega.reshape(-1, CAVITY_NT), psi.reshape(-1, CAVITY_NT)])
                    errs = split_errors(y_ref, y_pred, dA, t_ds, t_train)
                    if args.save_plots:
                        error_pct = errs["mean_full"] * 100 if errs["mean_full"] is not None else float("nan")
                        save_cavity_plot(
                            omega_ds,
                            omega,
                            x_ds,
                            y_ds,
                            t_ds,
                            error_pct,
                            plot_dir / f"cavity_Re{re_label}_llm_traj1.png",
                        )
                    results.append(CaseResult("cavity", f"Re={Re}", "llm", True, None,
                                              errs["mean_full"], errs["mean_first"], errs["mean_second"]))
                    done = True
                    break
                except Exception as exc:
                    attempt_log.append(
                        {
                            "equation": "cavity",
                            "param": f"Re={Re}",
                            "raw_path": str(raw_out),
                            "attempt": attempts_used + 1,
                            "status": "validation_failed",
                            "error": str(exc),
                        }
                    )
                    print(f"[pure-llm] cavity Re={Re} validation failed: {exc}")
                    if args.retry_on_validation_failure:
                        if raw_out.exists():
                            raw_out.unlink()
                        continue
                    failures.append({"equation": "cavity", "param": Re, "error": str(exc)})
                    results.append(CaseResult("cavity", f"Re={Re}", "llm", False, str(exc),
                                              None, None, None))
                    done = True
                    break
            if not done:
                results.append(CaseResult("cavity", f"Re={Re}", "llm", False, "validation_failed",
                                          None, None, None))
        else:
            results.append(CaseResult("cavity", f"Re={Re}", "llm", False, "not_executed", None, None, None))

    # Save summary
    summary = {}
    for r in results:
        summary.setdefault(r.equation, {}).setdefault(r.param, {})
        summary[r.equation][r.param] = {
            "success": r.success,
            "error": r.error,
            "mean_full": r.mean_full,
            "mean_first": r.mean_first,
            "mean_second": r.mean_second,
        }
    with open(out_dir / "summary_split_errors.json", "w") as f:
        json.dump(summary, f, indent=2)

    failures_path = out_dir / "failures.json"
    if failures:
        with open(failures_path, "w") as f:
            json.dump(failures, f, indent=2)
    else:
        failures_path.write_text("[]")

    attempts_path = out_dir / "attempts.jsonl"
    if attempt_log:
        with open(attempts_path, "w") as f:
            for item in attempt_log:
                f.write(json.dumps(item) + "\n")
    else:
        attempts_path.write_text("")

    total_cases = len(results)
    failed_cases = len([r for r in results if not r.success])
    failure_rate = failed_cases / total_cases if total_cases else 0.0
    reason_counts: Dict[str, int] = {}
    for item in failures:
        reason = item.get("error", "unknown")
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
    attempt_total = len(attempt_log)
    attempt_failed = len([a for a in attempt_log if a.get("status") != "success"])
    attempt_failure_rate = attempt_failed / attempt_total if attempt_total else 0.0
    failure_summary = {
        "total_cases": total_cases,
        "failed_cases": failed_cases,
        "failure_rate": failure_rate,
        "attempt_total": attempt_total,
        "attempt_failed": attempt_failed,
        "attempt_failure_rate": attempt_failure_rate,
        "reason_counts": reason_counts,
    }
    with open(out_dir / "failure_summary.json", "w") as f:
        json.dump(failure_summary, f, indent=2)

    print(f"Saved summary to: {out_dir / 'summary_split_errors.json'}")


if __name__ == "__main__":
    main()
