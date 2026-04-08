#!/usr/bin/env python3
"""
Codegen-based LLM workflow for Heat, Burgers, and Cavity.

This script asks an LLM to generate local Python code that:
1) Performs linear regression/interpolation over OpInf operators.
2) Integrates the ROM to produce a trajectory.

Outputs are saved under: codegen/<model>/<equation>/<method>/
"""

import argparse
import gzip
import json
import os
import pickle
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import matplotlib.pyplot as plt

from llm_tool_calling_provider import call_llm_text

# Load API keys from .env
try:
    from load_env import load_env
    load_env()
except ImportError:
    pass

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


@dataclass
class AttemptResult:
    success: bool
    error: str | None = None
    code_path: Path | None = None
    output_path: Path | None = None
    failure_status: str | None = None
    failure_stage: str | None = None


def load_pickle_auto(path: str):
    with open(path, "rb") as f:
        magic = f.read(2)
        f.seek(0)
        if magic == b"\x1f\x8b":
            with gzip.open(path, "rb") as gz:
                return pickle.load(gz)
        return pickle.load(f)


def load_coeff_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def extract_code(response_text: str) -> str:
    if "```python" in response_text:
        start = response_text.find("```python") + len("```python")
        end = response_text.find("```", start)
        return response_text[start:end].strip()
    if "```" in response_text:
        start = response_text.find("```") + 3
        end = response_text.find("```", start)
        return response_text[start:end].strip()
    return response_text.strip()


def inject_paths(
    code: str,
    model_path: str,
    data_path: str,
    output_path: str,
    coeff_path: str | None = None,
) -> str:
    # Rewrite direct variable assignments first so reused code can retarget paths
    # even when it does not inline paths in np.load()/np.savez() calls.
    code = re.sub(
        r'(?m)^(\s*)model_path\s*=\s*[\'"].*?[\'"]',
        rf'\1model_path = "{model_path}"',
        code,
    )
    code = re.sub(
        r'(?m)^(\s*)data_path\s*=\s*[\'"].*?[\'"]',
        rf'\1data_path = "{data_path}"',
        code,
    )
    code = re.sub(
        r'(?m)^(\s*)output_path\s*=\s*[\'"].*?[\'"]',
        rf'\1output_path = "{output_path}"',
        code,
    )
    if coeff_path is not None:
        code = re.sub(
            r'(?m)^(\s*)(coeff_path|coeff_json_path)\s*=\s*[\'"].*?[\'"]',
            rf'\1coeff_path = "{coeff_path}"',
            code,
        )
    code = re.sub(r'load_pickle_auto\(".*?"\)', f'load_pickle_auto("{model_path}")', code)
    code = re.sub(r'np\.load\(".*?\.npz"\)', f'np.load("{data_path}")', code)
    code = re.sub(r'np\.savez_compressed\(".*?\.npz"', f'np.savez_compressed("{output_path}"', code)
    code = re.sub(r'np\.savez\(".*?\.npz"', f'np.savez("{output_path}"', code)
    return code


def aggregate_case_stats(case_outcomes: list[dict]) -> dict:
    summary: dict[str, Any] = {"overall": {}, "by_equation_method": {}}
    if not case_outcomes:
        summary["overall"] = {
            "n_cases": 0,
            "opinf_success": 0,
            "pipeline_success": 0,
            "opinf_success_rate": None,
            "pipeline_success_rate": None,
            "failure_stage_counts": {},
            "failure_status_counts": {},
        }
        return summary

    def build_bucket(records: list[dict]) -> dict:
        n_cases = len(records)
        opinf_success = sum(1 for r in records if r.get("opinf_success"))
        pipeline_success = sum(1 for r in records if r.get("pipeline_success"))
        failure_stage_counts: dict[str, int] = {}
        failure_status_counts: dict[str, int] = {}
        for rec in records:
            stage = rec.get("failure_stage")
            status = rec.get("failure_status")
            if stage:
                failure_stage_counts[stage] = failure_stage_counts.get(stage, 0) + 1
            if status:
                failure_status_counts[status] = failure_status_counts.get(status, 0) + 1
        return {
            "n_cases": n_cases,
            "opinf_success": opinf_success,
            "pipeline_success": pipeline_success,
            "opinf_success_rate": float(opinf_success / n_cases),
            "pipeline_success_rate": float(pipeline_success / n_cases),
            "failure_stage_counts": failure_stage_counts,
            "failure_status_counts": failure_status_counts,
        }

    summary["overall"] = build_bucket(case_outcomes)
    grouped: dict[tuple[str, str], list[dict]] = {}
    for rec in case_outcomes:
        key = (rec["equation"], rec["method"])
        grouped.setdefault(key, []).append(rec)
    for (equation, method), records in grouped.items():
        summary["by_equation_method"].setdefault(equation, {})[method] = build_bucket(records)
    return summary


def rel_l2(Y_ref, Y_rom, dx, dt):
    err = Y_ref - Y_rom
    norm_err = np.sqrt(np.sum(err ** 2) * dx * dt)
    norm_ref = np.sqrt(np.sum(Y_ref ** 2) * dx * dt)
    return float(norm_err / (norm_ref + 1e-14))


def spatiotemporal_l2_error(Y_fom, Y_rom, dx, dt):
    err = Y_fom - Y_rom
    norm_err = np.sqrt(np.sum(err ** 2) * dx * dt)
    norm_ref = np.sqrt(np.sum(Y_fom ** 2) * dx * dt)
    return float(norm_err / (norm_ref + 1e-14))


def get_heat_burgers_case(dataset_path: str, nu: float, traj_index: int = 0):
    dataset = load_pickle_auto(dataset_path)
    t_eval = dataset.get("t_eval")
    if t_eval is None:
        raise ValueError("Missing t_eval in dataset")

    if "per_nu_data" not in dataset:
        raise ValueError("Dataset missing per_nu_data")

    input_names = None
    if "metadata" in dataset:
        input_names = dataset["metadata"].get("input_names")
    if not input_names:
        input_names = ["u"]

    # Some test datasets store per-nu lists without explicit nu fields.
    per_nu_data = dataset["per_nu_data"]
    config_nus = None
    if isinstance(dataset.get("config"), dict):
        config_nus = dataset["config"].get("nu_list")

    def extract_from_lists(item):
        lists = item.get("lists", {})
        if "Y_test_list" not in lists:
            raise ValueError(f"Missing Y_test_list for nu={nu}")
        Y_list = lists["Y_test_list"]
        U_list = lists.get("U_test_list")
        if U_list is None:
            # Burgers test set uses w1/w2/w3 lists.
            w1_list = lists.get("w1_test_list", [])
            w2_list = lists.get("w2_test_list", [])
            w3_list = lists.get("w3_test_list", [])
            if not (w1_list and w2_list and w3_list):
                raise ValueError(f"Missing input lists for nu={nu}")
            U_list = list(zip(w1_list, w2_list, w3_list))

        if not Y_list:
            raise ValueError(f"Empty Y_test_list for nu={nu}")
        if traj_index >= len(Y_list):
            raise ValueError(f"Trajectory index {traj_index} out of range for nu={nu}")
        Y_ref = np.array(Y_list[traj_index]).copy()
        if traj_index >= len(U_list):
            raise ValueError(f"Input trajectory index {traj_index} out of range for nu={nu}")
        U_raw = U_list[traj_index]
        if Y_ref.shape[0] < Y_ref.shape[1]:
            Y_ref = Y_ref.T
        if isinstance(U_raw, tuple):
            U_ref = np.stack([np.array(u) for u in U_raw], axis=0)
        else:
            U_ref = np.array(U_raw).reshape(1, -1)
        return Y_ref, U_ref

    for idx, item in enumerate(per_nu_data):
        item_nu = item.get("nu")
        if item_nu is None and config_nus and idx < len(config_nus):
            item_nu = config_nus[idx]
        if item_nu is None or not np.isclose(item_nu, nu, atol=1e-6):
            continue

        if "lists" in item:
            Y_ref, U_ref = extract_from_lists(item)
            return Y_ref, U_ref, t_eval, input_names

        if "test" in item and item["test"]:
            test_case = item["test"][0]
            Y_ref = test_case["Y"].copy()
            U_dict = test_case["U"]
        else:
            split = None
            for key in ("test", "val", "train"):
                if key in item and isinstance(item[key], dict):
                    split = item[key]
                    break
            if split is None:
                raise ValueError(f"No test/val/train split for nu={nu}")
            Y_comb = split["Y"]
            U_dict = split["U"]
            M = len(t_eval)
            if Y_comb.shape[1] < M:
                raise ValueError(f"Not enough snapshots for nu={nu}")
            Y_ref = Y_comb[:, :M]
        if Y_ref.shape[0] < Y_ref.shape[1]:
            Y_ref = Y_ref.T
        if len(input_names) == 1:
            U_ref = np.array(U_dict[input_names[0]]).reshape(1, -1)
        else:
            U_ref = np.stack([U_dict[name] for name in input_names], axis=0)
        return Y_ref, U_ref, t_eval, input_names

    raise ValueError(f"No data found for nu={nu}")


def get_cavity_case(dataset_path: str, Re: float, traj_index: int = 0):
    dataset = load_pickle_auto(dataset_path)
    t_eval = dataset["t_eval"]
    per_Re_data = dataset["per_Re_data"]
    for item in per_Re_data:
        if not np.isclose(item["Re"], Re, atol=1e-6):
            continue
        split = None
        for key in ("test", "validation", "train"):
            if key in item and item[key]["Y_omega"] is not None and item[key]["Y_omega"].size > 0:
                split = item[key]
                break
        if split is None:
            raise ValueError(f"No data for Re={Re}")
        Y_omega = split["Y_omega"]
        Y_psi = split["Y_psi"]
        U_lid = split["U_lid"]
        M = len(t_eval)

        def select_traj(arr, name: str):
            if isinstance(arr, list):
                if traj_index >= len(arr):
                    raise ValueError(f"{name} traj_index {traj_index} out of range")
                return np.array(arr[traj_index])
            if hasattr(arr, "ndim") and arr.ndim == 3:
                if arr.shape[1] == M and arr.shape[0] != M:
                    return arr[:, :, traj_index]
                if arr.shape[2] == M and arr.shape[0] != M:
                    return arr[:, traj_index, :]
                if arr.shape[0] == M:
                    out = arr[:, :, traj_index]
                    return out.T
                return arr[:, :, traj_index]
            return arr

        Y_omega = select_traj(Y_omega, "Y_omega")
        Y_psi = select_traj(Y_psi, "Y_psi")
        U_lid = select_traj(U_lid, "U_lid")
        if Y_omega.shape[1] < M:
            raise ValueError(f"Not enough snapshots for Re={Re}")
        Y_omega = Y_omega[:, :M]
        Y_psi = Y_psi[:, :M]
        U_lid = U_lid[:M]
        return Y_omega, Y_psi, U_lid, t_eval

    raise ValueError(f"No data found for Re={Re}")


def write_attempt(attempt_log: Path, payload: dict):
    attempt_log.parent.mkdir(parents=True, exist_ok=True)
    with attempt_log.open("a") as f:
        f.write(json.dumps(payload) + "\n")


def build_prompt_heat_burgers(
    model_path: str,
    coeff_path: str | None,
    data_path: str,
    output_path: str,
    method: str,
    equation: str,
    op_shapes: dict,
    phi_shape: tuple,
    error_context: str | None,
    use_json: bool,
):
    extra = ""
    if error_context:
        extra = f"\nPrevious attempt failed with error:\n{error_context}\nFix the issue and regenerate the code."

    coeff_block = ""
    if use_json and coeff_path:
        coeff_block = f"""
COEFFICIENT JSON STRUCTURE (IMPORTANT):
- JSON file at: {coeff_path}
- Top-level keys: equation, n_modes, pod_basis_shape, parameters, pod_basis
- parameters is a list; each item has:
  * nu (float)
  * operators: {{A,B,C}} for heat or {{H,A,B,C}} for burgers
  * operators.<name>.values holds the array (always read from this nested key)
- pod_basis.values is the POD basis (space x r) to use as phi

When use_json=True:
- Load operators from the JSON, NOT from per_nu_models in the PKL.
- Use phi from coeff JSON (pod_basis.values).
- Use the PKL model ONLY to read x_grid/x_fine for dx.
- Convert loaded operator lists to numeric arrays with np.asarray(..., dtype=float).
"""
    path_contract = f"""
PATH VARIABLE CONTRACT (must follow exactly):
- Define these exact variables at top-level:
  model_path = "{model_path}"
  data_path = "{data_path}"
  output_path = "{output_path}"
"""
    if use_json and coeff_path:
        path_contract += f'  coeff_path = "{coeff_path}"\n'
        path_contract += "- Use `coeff_path` consistently for coefficient JSON. Do NOT invent `coeff_json_path` or `COEFF_JSON_PATH`.\n"
    path_contract += "- Do NOT reference any undefined path variable names.\n"

    return f"""You are an expert Python programmer.

Generate Python code to:
1) Load model from: {model_path}
2) Load case data from: {data_path}
3) Compute operators for the query parameter using method={method}
4) Integrate the ROM and save output to: {output_path}

USE_JSON={str(use_json)}
{coeff_block}
{path_contract}

MODEL STRUCTURE (IMPORTANT):
- pickle with keys: per_nu_models (list of dicts), phi, x_grid or x_fine, t_eval
- Each per_nu_models item has keys: "nu", "A", "B", "C" and for Burgers also "H"
- There is NO nested "operators" key. Operators are top-level per entry.
- Operator shapes (must match exactly): {op_shapes}
- phi shape: {phi_shape} (space x r)

CASE DATA (.npz) contains:
- Y_ref (may be saved as time x space); if Y_ref.shape[0] != phi.shape[0], transpose.
- U_ref (may be time x n_inputs); ensure U_ref is (n_inputs x time).
- t_eval (time grid)
- nu (float)

REQUIREMENTS:
- Always read inputs via: case_data = np.load(data_path)
- Always save outputs via this exact call (do not hard-code filenames):
  np.savez(output_path, Y_ref=Y_ref, Y_rom=Y_rom, t_eval=t_eval, nu=nu_query)
- Do NOT hard-code any filenames or paths.
- OUTPUT PATH RULE:
  The runner sets output_path per case/trajectory; you must use output_path directly.
  Example only (do NOT hard-code):
    output_path = "codegen/gpt-4o/burgers/regression/llm_codegen_burgers_nu0.07_traj12_raw.npz"
- If nu matches a training nu exactly, use that operator without regression.
- For method=regression: per-entry linear regression y = a*nu + b across training nus.
  Use numpy only (np.linalg.lstsq or closed form). Do NOT use sklearn.
- Regression implementation (stable): flatten operator to shape (n_train, n_flat),
  build X = [nu_train, ones], solve X @ coeffs = op_flat using lstsq.
  Then op_query_flat = coeffs[0]*nu_query + coeffs[1], reshape to op_shape.
- For method=interpolation: linear interpolation per entry.
  Use flatten-interp-reshape for arrays (np.interp expects 1D values).
- Heat ROM: a' = C + A a + B*u (B is (r,), u is scalar)
- Burgers ROM: a' = C + A a + H(a,a) + B @ u (B is (r x 3), u is 3-vector)
- For H(a,a), use: quad = np.einsum('ijk,j,k->i', H, a, a)
- Use solve_ivp(method='BDF', vectorized=False), fallback to RK45; for burgers, if needed fallback to LSODA.
- Always use t_eval from CASE DATA (do not use model t_eval).
- Projection: a0 = phi.T @ (Y_ref[:,0] * dx), after ensuring Y_ref is (space x time).
- Ensure a is a 1D vector of length r (e.g., a = np.asarray(a).ravel()) and rom_rhs returns shape (r,).
- In rom_rhs: start with a = np.asarray(a).ravel(); if a.size != r, set a = a[:r]; never reshape to (r,1) or (r,r).
- After computing a0, if a0.size != r, set a0 = a0[:r].
- Enforce r = phi.shape[1]; assert A.shape[0] == r and A.shape[1] == r before integration.
- Do NOT reference `model` inside helper functions. If a helper needs x_grid/dx/phi, pass it as an argument.
- After solve_ivp, set a_rom = sol.y; if a_rom.shape[0] != r, set a_rom = a_rom.T.
- Lift: Y_rom = phi @ a_rom.
- Save output with EXACT statement:
  np.savez(output_path, Y_ref=Y_ref, Y_rom=Y_rom, t_eval=t_eval, nu=nu_query)
- Before returning code, self-check:
  * every path variable used is defined
  * no use of coeff_json_path / COEFF_JSON_PATH
  * np.load(...) uses data_path and np.savez(...) uses output_path
- Do not print large arrays.
- Heat-specific shapes: B may be (r,) or (r,1) in training data. Canonicalize with:
  B = np.asarray(B, dtype=float).reshape(-1), and assert B.shape == (r,).
- Burgers-specific shapes: B is (r,3). u = [w1(t), w2(t), w3(t)].
- If Y_ref time length != len(t_eval), resample each spatial row to t_eval using np.interp.
- U_ref handling:
  * If U_ref.ndim == 1, reshape to (1, -1).
  * If U_ref.shape[0] == len(t_eval) and U_ref.shape[1] != len(t_eval), transpose.
  * After reshape, enforce U_ref.shape[1] == len(t_eval). If not, resample U_ref to t_eval
    using np.interp with a linearly spaced time grid over [t_eval[0], t_eval[-1]].
  * For heat, use u = float(np.interp(t, t_eval, U_ref[0])).
  * For burgers, build u vector by interp each row; ensure u has shape (3,).

SANITY CHECKS (must pass before saving):
- assert Y_ref.shape[1] == len(t_eval)
- assert Y_rom.shape == Y_ref.shape
- In rom_rhs, return a 1D array (shape (r,)); do NOT return (r,1) or (r,r).

CODE TEMPLATE (use this structure, adjust details):
1) Load model, extract nu_train and operator arrays per entry.
2) For each operator tensor:
   flat = arr.reshape(-1); stack across nu; interp/regress each entry; reshape to op_shapes.
3) Load Y_ref/U_ref; if Y_ref.shape[0] != phi.shape[0] or Y_ref.shape[0] == len(t_eval), transpose.
   Ensure U_ref is (n_inputs x time).
4) Integrate ROM; save output.

Return only the complete Python code (no explanations).{extra}
"""


def build_prompt_cavity(
    model_path: str,
    data_path: str,
    output_path: str,
    method: str,
    op_shapes: dict,
    phi_shape: tuple,
    error_context: str | None,
):
    extra = ""
    if error_context:
        extra = f"\nPrevious attempt failed with error:\n{error_context}\nFix the issue and regenerate the code."
    path_contract = f"""
PATH VARIABLE CONTRACT (must follow exactly):
- Define these exact variables at top-level:
  model_path = "{model_path}"
  data_path = "{data_path}"
  output_path = "{output_path}"
- Do NOT reference any undefined path variable names.
"""

    return f"""You are an expert Python programmer.

Generate Python code to:
1) Load cavity model from: {model_path}
2) Load case data from: {data_path}
3) Compute operators for the query Re using method={method}
4) Integrate ROM with RK4 and save output to: {output_path}
{path_contract}

MODEL STRUCTURE (IMPORTANT):
- pickle with keys: per_Re_models (list of dicts), phi, x, y, t_eval
- Each per_Re_models item has keys: "Re", "H", "A", "B", "C"
- There is NO nested "operators" key.
- Operator shapes (must match exactly): {op_shapes}
- phi shape: {phi_shape} (state x r)

CASE DATA (.npz) contains:
- Y_omega (n x time)
- Y_psi (n x time)
- U_lid (time)
- t_eval (time grid)
- Re (float)

REQUIREMENTS:
- Always read inputs via: case_data = np.load(data_path)
- Always save outputs via this exact call (do not hard-code filenames):
  np.savez(output_path, Y_omega_fom=Y_omega, Y_psi_fom=Y_psi,
           Y_omega_rom=Y_omega_rom, Y_psi_rom=Y_psi_rom,
           U_lid=U_lid, x=x, y=y, t_eval=t_eval, Re=Re_query)
- Do NOT hard-code any filenames or paths.
- OUTPUT PATH RULE:
  The runner sets output_path per case/trajectory; you must use output_path directly.
  Example only (do NOT hard-code):
    output_path = "codegen/gpt-4o/cavity/regression/llm_codegen_cavity_Re120.0_traj2_raw.npz"
- If Re matches a training Re exactly, use that operator without regression.
- For method=regression: per-entry linear regression y = a*Re + b.
  Use numpy only (np.linalg.lstsq or closed form). Do NOT use sklearn.
- Regression implementation (stable): flatten operator to shape (n_train, n_flat),
  build X = [Re_train, ones], solve X @ coeffs = op_flat using lstsq.
  Then op_query_flat = coeffs[0]*Re_query + coeffs[1], reshape to op_shape.
- For method=interpolation: linear interpolation per entry (flatten-interp-reshape).
- Modal state dimension is r = phi.shape[1]. Integrate a(t) in R^r.
- Use operators H, A, B, C with quadratic term H(a,a) via einsum:
  quad = np.einsum('ijk,j,k->i', H, a, a)
- Build Y_fom by stacking omega and psi.
- Project with phi.T @ (Y_fom * dA) where dA = dx*dx.
- Integrate with fixed-step RK4 using dt from t_eval.
- Ensure Y_omega, Y_psi are (n x time) and time length == len(t_eval); do NOT downsample.
- If Y_omega/Y_psi time length != len(t_eval), resample each spatial row to t_eval using np.interp.
- Save Y_omega_rom/Y_psi_rom with the same shape as Y_omega/Y_psi.
- Save output with EXACT statement:
  np.savez(output_path, Y_omega_fom=Y_omega, Y_psi_fom=Y_psi,
           Y_omega_rom=Y_omega_rom, Y_psi_rom=Y_psi_rom,
           U_lid=U_lid, x=x, y=y, t_eval=t_eval, Re=Re_query)
- Before returning code, self-check:
  * every path variable used is defined
  * np.load(...) uses data_path and np.savez(...) uses output_path
- Do not print large arrays.
- Use op_shapes exactly; do not reshape to other sizes.
- U_lid handling: if U_lid is not length len(t_eval), resample to t_eval using np.interp
  on a linear time grid over [t_eval[0], t_eval[-1]].
- Ensure a is 1D (r,) throughout; do not use column vectors.

SANITY CHECKS (must pass before saving):
- assert Y_omega_rom.shape == Y_omega.shape
- assert Y_psi_rom.shape == Y_psi.shape

CODE TEMPLATE (use this structure, adjust details):
1) Load model, extract Re_train and operator arrays per entry.
2) For each operator tensor:
   flat = arr.reshape(-1); stack across Re; interp/regress each entry; reshape to op_shapes.
3) Load Y_omega/Y_psi; build Y_fom; project with phi to get a0 (r x 1).
4) Integrate a(t) with RK4 in r-dim; then Y_rom = phi @ a_traj.
5) Split Y_rom into omega/psi using n = Y_omega.shape[0].
6) Save outputs.

Return only the complete Python code (no explanations).{extra}
"""


def run_codegen_case(
    provider: str,
    model: str,
    equation: str,
    method: str,
    case_id: str,
    model_path: str,
    coeff_path: str | None,
    data_path: str,
    output_path: str,
    attempts_dir: Path,
    max_attempts: int,
    sleep_secs: float,
    attempt_log: Path,
    run_id: str,
    cached_code: str | None,
    reuse_code: bool,
    op_shapes: dict,
    phi_shape: tuple,
    use_json: bool,
) -> AttemptResult:
    error_context = None
    last_failure_status = None
    last_failure_stage = None

    if reuse_code and cached_code:
        attempts_dir.mkdir(parents=True, exist_ok=True)
        code_path = attempts_dir / "codegen_reuse.py"
        code = inject_paths(cached_code, model_path, data_path, output_path, coeff_path)
        code_path.write_text(code)
        try:
            subprocess.run([sys.executable, str(code_path)], check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as exc:
            error_context = f"Code exec error (reused): {exc.stderr.strip() or exc.stdout.strip()}"
            last_failure_status = "reuse_exec_error"
            last_failure_stage = "opinf_execution"
            write_attempt(attempt_log, {
                "equation": equation,
                "case": case_id,
                "method": method,
                "attempt": 0,
                "run_id": run_id,
                "status": last_failure_status,
                "failure_stage": last_failure_stage,
                "error": error_context,
            })
        else:
            if os.path.exists(output_path):
                try:
                    data = np.load(output_path)
                    if equation == "cavity":
                        _ = data["Y_omega_rom"]
                        _ = data["Y_psi_rom"]
                    else:
                        Y_ref = data["Y_ref"]
                        Y_rom = data["Y_rom"]
                        if Y_ref.shape != Y_rom.shape:
                            raise ValueError(f"Shape mismatch: {Y_ref.shape} vs {Y_rom.shape}")
                    if not np.all(np.isfinite(data[list(data.files)[0]])):
                        raise ValueError("Non-finite values in output")
                except Exception as exc:
                    error_context = f"Validation error (reused): {exc}"
                    last_failure_status = "reuse_validation_error"
                    last_failure_stage = "opinf_validation"
                    write_attempt(attempt_log, {
                        "equation": equation,
                        "case": case_id,
                        "method": method,
                        "attempt": 0,
                        "run_id": run_id,
                        "status": last_failure_status,
                        "failure_stage": last_failure_stage,
                        "error": error_context,
                    })
                else:
                    write_attempt(attempt_log, {
                        "equation": equation,
                        "case": case_id,
                        "method": method,
                        "attempt": 0,
                        "run_id": run_id,
                        "status": "reuse_success",
                        "failure_stage": None,
                    })
                    return AttemptResult(True, code_path=code_path, output_path=Path(output_path))
            else:
                error_context = f"Expected output not found (reused): {output_path}"
                last_failure_status = "reuse_missing_output"
                last_failure_stage = "pipeline_io"
                write_attempt(attempt_log, {
                    "equation": equation,
                    "case": case_id,
                    "method": method,
                    "attempt": 0,
                    "run_id": run_id,
                    "status": last_failure_status,
                    "failure_stage": last_failure_stage,
                    "error": error_context,
                })
        return AttemptResult(
            False,
            error=error_context,
            code_path=code_path,
            failure_status=last_failure_status,
            failure_stage=last_failure_stage,
        )

    for attempt in range(1, max_attempts + 1):
        print(f"[codegen] {equation} {case_id} {method} attempt {attempt}/{max_attempts}")
        if equation == "cavity":
            prompt = build_prompt_cavity(
                model_path, data_path, output_path, method, op_shapes, phi_shape, error_context
            )
        else:
            prompt = build_prompt_heat_burgers(
                model_path,
                coeff_path,
                data_path,
                output_path,
                method,
                equation,
                op_shapes,
                phi_shape,
                error_context,
                use_json,
            )

        messages = [
            {"role": "system", "content": "You are a precise Python code generator."},
            {"role": "user", "content": prompt},
        ]

        try:
            response_text = call_llm_text(provider, messages, model)
            code = extract_code(response_text)
            if reuse_code and cached_code is None:
                cached_code = code
        except Exception as exc:
            error_context = f"LLM call error: {exc}"
            last_failure_status = "llm_error"
            last_failure_stage = "pipeline_llm"
            write_attempt(attempt_log, {
                "equation": equation,
                "case": case_id,
                "method": method,
                "attempt": attempt,
                "run_id": run_id,
                "status": last_failure_status,
                "failure_stage": last_failure_stage,
                "error": str(exc),
            })
            if sleep_secs > 0:
                wait = sleep_secs * attempt
                print(f"[codegen] {equation} {case_id} {method} sleeping {wait:.1f}s after LLM error")
                time.sleep(wait)
            continue

        if "np.load(data_path)" not in code or "np.savez(" not in code or "output_path" not in code:
            error_context = (
                "Code must use the provided data_path for np.load(...) and output_path for np.savez(...). "
                "Do not hard-code filenames."
            )
            last_failure_status = "code_contract_error"
            last_failure_stage = "pipeline_contract"
            write_attempt(attempt_log, {
                "equation": equation,
                "case": case_id,
                "method": method,
                "attempt": attempt,
                "run_id": run_id,
                "status": last_failure_status,
                "failure_stage": last_failure_stage,
                "error": error_context,
            })
            continue

        attempts_dir.mkdir(parents=True, exist_ok=True)
        code_path = attempts_dir / f"codegen_attempt_{attempt}.py"
        code_path.write_text(code)

        try:
            subprocess.run([sys.executable, str(code_path)], check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as exc:
            error_context = f"Code exec error: {exc.stderr.strip() or exc.stdout.strip()}"
            last_failure_status = "exec_error"
            last_failure_stage = "opinf_execution"
            write_attempt(attempt_log, {
                "equation": equation,
                "case": case_id,
                "method": method,
                "attempt": attempt,
                "run_id": run_id,
                "status": last_failure_status,
                "failure_stage": last_failure_stage,
                "error": error_context,
            })
            continue

        if not os.path.exists(output_path):
            error_context = f"Expected output not found: {output_path}"
            last_failure_status = "missing_output"
            last_failure_stage = "pipeline_io"
            write_attempt(attempt_log, {
                "equation": equation,
                "case": case_id,
                "method": method,
                "attempt": attempt,
                "run_id": run_id,
                "status": last_failure_status,
                "failure_stage": last_failure_stage,
                "error": error_context,
            })
            continue

        try:
            data = np.load(output_path)
            if equation == "cavity":
                _ = data["Y_omega_rom"]
                _ = data["Y_psi_rom"]
            else:
                Y_ref = data["Y_ref"]
                Y_rom = data["Y_rom"]
                if Y_ref.shape != Y_rom.shape:
                    raise ValueError(f"Shape mismatch: {Y_ref.shape} vs {Y_rom.shape}")
            if not np.all(np.isfinite(data[list(data.files)[0]])):
                raise ValueError("Non-finite values in output")
        except Exception as exc:
            error_context = f"Validation error: {exc}"
            last_failure_status = "validation_error"
            last_failure_stage = "opinf_validation"
            write_attempt(attempt_log, {
                "equation": equation,
                "case": case_id,
                "method": method,
                "attempt": attempt,
                "run_id": run_id,
                "status": last_failure_status,
                "failure_stage": last_failure_stage,
                "error": error_context,
            })
            continue

        write_attempt(attempt_log, {
            "equation": equation,
            "case": case_id,
            "method": method,
            "attempt": attempt,
            "run_id": run_id,
            "status": "success",
            "failure_stage": None,
        })
        return AttemptResult(True, code_path=code_path, output_path=Path(output_path))

    return AttemptResult(
        False,
        error=error_context,
        failure_status=last_failure_status,
        failure_stage=last_failure_stage,
    )


def plot_heat_burgers(output_path: Path, output_dir: Path, nu: float, equation: str, traj_index: int = 0):
    data = np.load(output_path)
    Y_ref = data["Y_ref"]
    Y_rom = data["Y_rom"]
    t_eval = data["t_eval"]

    # Ensure (space x time) for plotting.
    if Y_ref.shape[0] == len(t_eval):
        Y_ref = Y_ref.T
    if Y_rom.shape[0] == len(t_eval):
        Y_rom = Y_rom.T

    T_mesh, X_mesh = np.meshgrid(t_eval, np.arange(Y_ref.shape[0]))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    im1 = axes[0].pcolormesh(T_mesh, X_mesh, Y_ref, cmap='RdBu_r', shading='auto')
    axes[0].set_title(f'Ground Truth (ν={nu})')
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('Space')
    plt.colorbar(im1, ax=axes[0])

    im2 = axes[1].pcolormesh(T_mesh, X_mesh, Y_rom, cmap='RdBu_r', shading='auto')
    axes[1].set_title('Codegen ROM Prediction')
    axes[1].set_xlabel('Time')
    axes[1].set_ylabel('Space')
    plt.colorbar(im2, ax=axes[1])

    error = np.abs(Y_ref - Y_rom)
    im3 = axes[2].pcolormesh(T_mesh, X_mesh, error, cmap='YlOrRd', shading='auto')
    axes[2].set_title('Absolute Error')
    axes[2].set_xlabel('Time')
    axes[2].set_ylabel('Space')
    plt.colorbar(im3, ax=axes[2])

    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_file = output_dir / f"llm_codegen_{equation}_nu{nu}_traj{traj_index + 1}_test.png"
    plt.savefig(plot_file, dpi=150, bbox_inches="tight")
    plt.close()


def plot_cavity(output_path: Path, output_dir: Path, Re: float, traj_index: int = 0):
    data = np.load(output_path)
    Y_omega_fom = data["Y_omega_fom"]
    Y_omega_rom = data["Y_omega_rom"]
    t_eval = data["t_eval"]

    n = Y_omega_fom.shape[0]
    M = Y_omega_fom.shape[1]
    side = int(np.sqrt(n))
    if side * side != n:
        return

    omega_fom = Y_omega_fom.reshape(side, side, M, order="C")
    omega_rom = Y_omega_rom.reshape(side, side, M, order="C")
    X, Y_grid = np.meshgrid(np.arange(side), np.arange(side), indexing="ij")

    fig, axs = plt.subplots(2, 5, figsize=(24, 10))
    times_idx = [0, M//4, M//2, 3*M//4, M-1]
    times_val = [t_eval[idx] for idx in times_idx]

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

    fig.suptitle(f"Re={Re} (codegen)", fontsize=16, fontweight="bold")
    plt.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    plot_file = output_dir / f"cavity_Re{Re}_codegen_traj{traj_index + 1}.png"
    plt.savefig(plot_file, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="LLM codegen workflow for 3 equations")
    parser.add_argument("--provider", type=str, default="openai",
                        choices=list(DEFAULT_MODEL_BY_PROVIDER.keys()))
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="codegen_runs")
    parser.add_argument("--run_id", type=str, default=None,
                        help="Run identifier. Default: timestamp YYYYMMDD-HHMMSS")
    parser.add_argument("--methods", nargs="+", default=["interpolation", "regression"],
                        choices=["interpolation", "regression"])
    parser.add_argument("--equations", nargs="+", default=["heat", "burgers", "cavity"],
                        choices=["heat", "burgers", "cavity"])
    parser.add_argument("--use_json_coeffs", action="store_true",
                        help="Use coefficient JSON files for heat/burgers instead of PKL models.")
    parser.add_argument("--use_pkl_only", action="store_true",
                        help="Deprecated: PKL-only is already the default.")
    parser.add_argument("--heat_coeff", type=str, default="heat_coeff_FIXED.json",
                        help="Heat coefficient JSON (used only with --use_json_coeffs).")
    parser.add_argument("--burgers_coeff", type=str, default="burgers_coeff_FIXED.json",
                        help="Burgers coefficient JSON (used only with --use_json_coeffs).")
    parser.add_argument("--max_attempts_per_case", type=int, default=5)
    parser.add_argument("--sleep_secs", type=float, default=2.0,
                        help="Base sleep (s) after LLM errors; multiplied by attempt index.")
    parser.add_argument("--save_raw", action="store_true")
    parser.add_argument("--save_plots", action="store_true")
    parser.add_argument("--reuse_code_per_equation", action="store_true",
                        help="Reuse the first successful code per equation+method for other parameters.")
    parser.add_argument("--generate_once_per_equation", action="store_true",
                        help="Generate code once per equation+method, then reuse for all cases.")
    parser.add_argument("--n_traj_per_case", type=int, default=None,
                        help="Number of test trajectories per parameter case (uses trajectory lists when available).")
    parser.add_argument("--heat_traj_per_case", type=int, default=20,
                        help="Number of heat trajectories per parameter case.")
    parser.add_argument("--burgers_traj_per_case", type=int, default=20,
                        help="Number of burgers trajectories per parameter case.")
    parser.add_argument("--cavity_traj_per_case", type=int, default=2,
                        help="Number of cavity trajectories per parameter case.")
    parser.add_argument("--heat_nus", nargs="+", type=float, default=DEFAULT_HEAT_NUS)
    parser.add_argument("--burgers_nus", nargs="+", type=float, default=DEFAULT_BURGERS_NUS)
    parser.add_argument("--cavity_res", nargs="+", type=float, default=DEFAULT_CAVITY_RES)
    args = parser.parse_args()

    model_name = args.model_name or DEFAULT_MODEL_BY_PROVIDER[args.provider]
    use_json = bool(args.use_json_coeffs)
    if args.use_pkl_only:
        print("[codegen] --use_pkl_only is deprecated; PKL-only is already default.")
    heat_coeff_path = args.heat_coeff
    burgers_coeff_path = args.burgers_coeff
    if use_json:
        missing_coeffs = [p for p in (heat_coeff_path, burgers_coeff_path) if not Path(p).exists()]
        if missing_coeffs:
            print(f"[codegen] Coefficient JSON missing ({', '.join(missing_coeffs)}); falling back to PKL-only mode.")
            use_json = False
    run_id = args.run_id or time.strftime("%Y%m%d-%H%M%S")
    base_dir = Path(args.output_dir) / model_name / run_id
    attempt_log = base_dir / "attempts.jsonl"
    print(f"[codegen] run_id={run_id}")
    print(f"[codegen] run_dir={base_dir}")

    summary = {"heat": {}, "burgers": {}, "cavity": {}}
    case_outcomes: list[dict[str, Any]] = []
    cached_codes: dict[tuple[str, str], str] = {}
    cached_case_codes: dict[tuple[str, str, str], str] = {}

    # Heat/Burgers
    for equation in args.equations:
        if equation in ("heat", "burgers"):
            model_path = "src/heat_model.pkl" if equation == "heat" else "src/burgers_model.pkl"
            dataset_path = "dataset/heat_dataset_test.pkl.gz" if equation == "heat" else "dataset/burgers_dataset_test.pkl.gz"
            nus = args.heat_nus if equation == "heat" else args.burgers_nus

            model_data = load_pickle_auto(model_path)
            t_train = model_data["t_eval"][-1]
            x_grid = model_data.get("x_grid", model_data.get("x_fine"))
            dx = x_grid[1] - x_grid[0]
            if use_json:
                coeff_path = heat_coeff_path if equation == "heat" else burgers_coeff_path
                if not Path(coeff_path).exists():
                    raise FileNotFoundError(f"Coefficient JSON not found: {coeff_path}")
                coeff_data = load_coeff_json(coeff_path)
                phi_shape = tuple(coeff_data["pod_basis_shape"])
                ops = coeff_data["parameters"][0]["operators"]
                if equation == "heat":
                    op_shapes = {
                        "A": np.array(ops["A"]["values"]).shape,
                        "B": np.array(ops["B"]["values"]).shape,
                        "C": np.array(ops["C"]["values"]).shape,
                    }
                else:
                    op_shapes = {
                        "H": np.array(ops["H"]["values"]).shape,
                        "A": np.array(ops["A"]["values"]).shape,
                        "B": np.array(ops["B"]["values"]).shape,
                        "C": np.array(ops["C"]["values"]).shape,
                    }
            else:
                coeff_path = None
                phi_shape = model_data["phi"].shape
                if equation == "heat":
                    op_shapes = {
                        "A": np.array(model_data["per_nu_models"][0]["A"]).shape,
                        "B": np.array(model_data["per_nu_models"][0]["B"]).shape,
                        "C": np.array(model_data["per_nu_models"][0]["C"]).shape,
                    }
                else:
                    op_shapes = {
                        "H": np.array(model_data["per_nu_models"][0]["H"]).shape,
                        "A": np.array(model_data["per_nu_models"][0]["A"]).shape,
                        "B": np.array(model_data["per_nu_models"][0]["B"]).shape,
                        "C": np.array(model_data["per_nu_models"][0]["C"]).shape,
                    }

            for method in args.methods:
                if args.generate_once_per_equation and (equation, method) not in cached_codes:
                    seed_nu = nus[0]
                    seed_suffix = f"nu{seed_nu}_seed"
                    seed_out_dir = base_dir / equation / method
                    seed_out_dir.mkdir(parents=True, exist_ok=True)
                    seed_data_path = seed_out_dir / f"case_{seed_suffix}.npz"
                    seed_output_path = seed_out_dir / f"llm_codegen_{equation}_{seed_suffix}_raw.npz"
                    Y_ref, U_ref, t_eval, _ = get_heat_burgers_case(
                        dataset_path, seed_nu, 0
                    )
                    np.savez_compressed(
                        seed_data_path, Y_ref=Y_ref, U_ref=U_ref, t_eval=t_eval, nu=np.array([seed_nu])
                    )
                    seed_attempts_dir = seed_out_dir / f"attempts_{seed_suffix}"
                    seed_result = run_codegen_case(
                        args.provider, model_name, equation, method, seed_suffix,
                        model_path, coeff_path, str(seed_data_path), str(seed_output_path),
                        seed_attempts_dir, args.max_attempts_per_case, args.sleep_secs, attempt_log, run_id,
                        None, False, op_shapes, phi_shape, use_json
                    )
                    if not seed_result.success:
                        print(f"[codegen] seed failed for {equation} {method}; skipping.")
                        continue
                    try:
                        seed_code = Path(seed_result.code_path).read_text()
                        cached_codes[(equation, method)] = seed_code
                    except Exception:
                        print(f"[codegen] failed to cache seed code for {equation} {method}")
                        continue

                for nu in nus:
                    full_errs = []
                    first_errs = []
                    second_errs = []
                    n_traj = args.n_traj_per_case or args.heat_traj_per_case
                    n_traj = max(1, n_traj)
                    case_key = (equation, method, f"nu{nu}")

                    for traj_index in range(n_traj):
                        case_suffix = f"nu{nu}_traj{traj_index + 1}" if n_traj > 1 else f"nu{nu}"
                        out_dir = base_dir / equation / method
                        out_dir.mkdir(parents=True, exist_ok=True)
                        data_path = out_dir / f"case_{case_suffix}.npz"
                        output_path = out_dir / f"llm_codegen_{equation}_{case_suffix}_raw.npz"

                        Y_ref, U_ref, t_eval, input_names = get_heat_burgers_case(
                            dataset_path, nu, traj_index
                        )
                        np.savez_compressed(
                            data_path, Y_ref=Y_ref, U_ref=U_ref, t_eval=t_eval, nu=np.array([nu])
                        )

                        attempts_dir = out_dir / f"attempts_{case_suffix}"
                        cached_case = cached_case_codes.get(case_key)
                        cached_equation = cached_codes.get((equation, method))
                        cached_code = cached_case or cached_equation
                        reuse_flag = args.reuse_code_per_equation or args.generate_once_per_equation or cached_code is not None

                        result = run_codegen_case(
                            args.provider, model_name, equation, method, case_suffix,
                            model_path, coeff_path, str(data_path), str(output_path),
                            attempts_dir, args.max_attempts_per_case, args.sleep_secs, attempt_log, run_id,
                            cached_code, reuse_flag, op_shapes, phi_shape, use_json
                        )
                        if result.success:
                            try:
                                code_text = Path(result.code_path).read_text()
                                cached_case_codes[case_key] = code_text
                                if (args.reuse_code_per_equation or args.generate_once_per_equation) and (equation, method) not in cached_codes:
                                    cached_codes[(equation, method)] = code_text
                            except Exception:
                                pass
                        if not result.success:
                            case_outcomes.append({
                                "run_id": run_id,
                                "equation": equation,
                                "method": method,
                                "parameter": float(nu),
                                "trajectory_index": traj_index + 1,
                                "case_id": case_suffix,
                                "opinf_success": False,
                                "pipeline_success": False,
                                "failure_stage": result.failure_stage,
                                "failure_status": result.failure_status,
                                "error": result.error,
                            })
                            continue

                        try:
                            data = np.load(output_path)
                            Y_ref = data["Y_ref"]
                            Y_rom = data["Y_rom"]
                            t_eval_used = data["t_eval"]
                            dt = t_eval_used[1] - t_eval_used[0]

                            full_err = rel_l2(Y_ref, Y_rom, dx, dt)
                            split_idx = np.searchsorted(t_eval_used, t_train)
                            err_first = None
                            err_second = None
                            if 1 < split_idx < Y_ref.shape[1]:
                                err_first = rel_l2(Y_ref[:, :split_idx], Y_rom[:, :split_idx], dx, dt)
                                err_second = rel_l2(Y_ref[:, split_idx:], Y_rom[:, split_idx:], dx, dt)

                            full_errs.append(full_err)
                            first_errs.append(err_first)
                            second_errs.append(err_second)

                            if args.save_plots:
                                plot_heat_burgers(output_path, out_dir, nu, equation, traj_index)

                            case_outcomes.append({
                                "run_id": run_id,
                                "equation": equation,
                                "method": method,
                                "parameter": float(nu),
                                "trajectory_index": traj_index + 1,
                                "case_id": case_suffix,
                                "opinf_success": True,
                                "pipeline_success": True,
                                "failure_stage": None,
                                "failure_status": None,
                                "error": None,
                            })
                        except Exception as exc:
                            write_attempt(attempt_log, {
                                "equation": equation,
                                "case": case_suffix,
                                "method": method,
                                "attempt": 0,
                                "run_id": run_id,
                                "status": "postproc_error",
                                "failure_stage": "pipeline_postproc",
                                "error": f"Postproc error: {exc}",
                            })
                            case_outcomes.append({
                                "run_id": run_id,
                                "equation": equation,
                                "method": method,
                                "parameter": float(nu),
                                "trajectory_index": traj_index + 1,
                                "case_id": case_suffix,
                                "opinf_success": True,
                                "pipeline_success": False,
                                "failure_stage": "pipeline_postproc",
                                "failure_status": "postproc_error",
                                "error": str(exc),
                            })
                            continue

                    first_vals = [v for v in first_errs if v is not None]
                    second_vals = [v for v in second_errs if v is not None]
                    summary[equation].setdefault(method, {})[str(nu)] = {
                        "n_traj": len(full_errs),
                        "mean_full": float(np.mean(full_errs)) if full_errs else None,
                        "min_full": float(np.min(full_errs)) if full_errs else None,
                        "max_full": float(np.max(full_errs)) if full_errs else None,
                        "mean_first": float(np.mean(first_vals)) if first_vals else None,
                        "mean_second": float(np.mean(second_vals)) if second_vals else None,
                    }

        if equation == "cavity":
            model_path = "src/cavity_model.pkl"
            dataset_path = "dataset/cavity_dataset_test.pkl.gz"
            model_data = load_pickle_auto(model_path)
            x = model_data["x"]
            dx = x[1] - x[0]
            dA = dx * dx
            t_train = model_data["t_eval"][-1]
            phi_shape = model_data["phi"].shape
            op_shapes = {
                "H": np.array(model_data["per_Re_models"][0]["H"]).shape,
                "A": np.array(model_data["per_Re_models"][0]["A"]).shape,
                "B": np.array(model_data["per_Re_models"][0]["B"]).shape,
                "C": np.array(model_data["per_Re_models"][0]["C"]).shape,
            }

            for method in args.methods:
                if args.generate_once_per_equation and ("cavity", method) not in cached_codes:
                    seed_Re = args.cavity_res[0]
                    seed_suffix = f"Re{seed_Re}_seed"
                    seed_out_dir = base_dir / "cavity" / method
                    seed_out_dir.mkdir(parents=True, exist_ok=True)
                    seed_data_path = seed_out_dir / f"case_{seed_suffix}.npz"
                    seed_output_path = seed_out_dir / f"cavity_{seed_suffix}_raw.npz"
                    Y_omega, Y_psi, U_lid, t_eval = get_cavity_case(dataset_path, seed_Re, 0)
                    np.savez_compressed(
                        seed_data_path,
                        Y_omega=Y_omega,
                        Y_psi=Y_psi,
                        U_lid=U_lid,
                        t_eval=t_eval,
                        Re=np.array([seed_Re]),
                    )
                    seed_attempts_dir = seed_out_dir / f"attempts_{seed_suffix}"
                    seed_result = run_codegen_case(
                        args.provider, model_name, "cavity", method, seed_suffix,
                        model_path, None, str(seed_data_path), str(seed_output_path),
                        seed_attempts_dir, args.max_attempts_per_case, args.sleep_secs, attempt_log, run_id,
                        None, False, op_shapes, phi_shape, False
                    )
                    if not seed_result.success:
                        print(f"[codegen] seed failed for cavity {method}; skipping.")
                        continue
                    try:
                        seed_code = Path(seed_result.code_path).read_text()
                        cached_codes[("cavity", method)] = seed_code
                    except Exception:
                        print(f"[codegen] failed to cache seed code for cavity {method}")
                        continue

                for Re in args.cavity_res:
                    full_errs = []
                    first_errs = []
                    second_errs = []
                    n_traj = args.n_traj_per_case or args.cavity_traj_per_case
                    n_traj = max(1, n_traj)
                    case_key = ("cavity", method, f"Re{Re}")

                    for traj_index in range(n_traj):
                        case_suffix = f"Re{Re}_traj{traj_index + 1}" if n_traj > 1 else f"Re{Re}"
                        out_dir = base_dir / "cavity" / method
                        out_dir.mkdir(parents=True, exist_ok=True)
                        data_path = out_dir / f"case_{case_suffix}.npz"
                        output_path = out_dir / f"cavity_{case_suffix}_raw.npz"

                        Y_omega, Y_psi, U_lid, t_eval = get_cavity_case(dataset_path, Re, traj_index)
                        np.savez_compressed(
                            data_path, Y_omega=Y_omega, Y_psi=Y_psi, U_lid=U_lid, t_eval=t_eval, Re=np.array([Re])
                        )

                        attempts_dir = out_dir / f"attempts_{case_suffix}"
                        cached_case = cached_case_codes.get(case_key)
                        cached_equation = cached_codes.get(("cavity", method))
                        cached_code = cached_case or cached_equation
                        reuse_flag = args.reuse_code_per_equation or args.generate_once_per_equation or cached_code is not None

                        result = run_codegen_case(
                            args.provider, model_name, "cavity", method, case_suffix,
                            model_path, None, str(data_path), str(output_path),
                            attempts_dir, args.max_attempts_per_case, args.sleep_secs, attempt_log, run_id,
                            cached_code, reuse_flag, op_shapes, phi_shape, False
                        )
                        if result.success:
                            try:
                                code_text = Path(result.code_path).read_text()
                                cached_case_codes[case_key] = code_text
                                if (args.reuse_code_per_equation or args.generate_once_per_equation) and ("cavity", method) not in cached_codes:
                                    cached_codes[("cavity", method)] = code_text
                            except Exception:
                                pass
                        if not result.success:
                            case_outcomes.append({
                                "run_id": run_id,
                                "equation": "cavity",
                                "method": method,
                                "parameter": float(Re),
                                "trajectory_index": traj_index + 1,
                                "case_id": case_suffix,
                                "opinf_success": False,
                                "pipeline_success": False,
                                "failure_stage": result.failure_stage,
                                "failure_status": result.failure_status,
                                "error": result.error,
                            })
                            continue

                        try:
                            data = np.load(output_path)
                            Y_omega_fom = data["Y_omega_fom"]
                            Y_psi_fom = data["Y_psi_fom"]
                            Y_omega_rom = data["Y_omega_rom"]
                            Y_psi_rom = data["Y_psi_rom"]
                            t_eval_used = data["t_eval"]
                            dt = t_eval_used[1] - t_eval_used[0]

                            Y_fom = np.vstack([Y_omega_fom, Y_psi_fom])
                            Y_rom = np.vstack([Y_omega_rom, Y_psi_rom])
                            if Y_fom.shape != Y_rom.shape:
                                raise ValueError(f"Shape mismatch: {Y_fom.shape} vs {Y_rom.shape}")
                            full_err = spatiotemporal_l2_error(Y_fom, Y_rom, dA, dt)
                            split_idx = np.searchsorted(t_eval_used, t_train)
                            err_first = None
                            err_second = None
                            if 1 < split_idx < Y_fom.shape[1]:
                                err_first = spatiotemporal_l2_error(Y_fom[:, :split_idx], Y_rom[:, :split_idx], dA, dt)
                                err_second = spatiotemporal_l2_error(Y_fom[:, split_idx:], Y_rom[:, split_idx:], dA, dt)

                            full_errs.append(full_err)
                            first_errs.append(err_first)
                            second_errs.append(err_second)

                            if args.save_plots:
                                plot_cavity(output_path, out_dir, Re, traj_index)
                            case_outcomes.append({
                                "run_id": run_id,
                                "equation": "cavity",
                                "method": method,
                                "parameter": float(Re),
                                "trajectory_index": traj_index + 1,
                                "case_id": case_suffix,
                                "opinf_success": True,
                                "pipeline_success": True,
                                "failure_stage": None,
                                "failure_status": None,
                                "error": None,
                            })
                        except Exception as exc:
                            write_attempt(attempt_log, {
                                "equation": "cavity",
                                "case": case_suffix,
                                "method": method,
                                "attempt": 0,
                                "run_id": run_id,
                                "status": "postproc_error",
                                "failure_stage": "pipeline_postproc",
                                "error": f"Postproc error: {exc}",
                            })
                            case_outcomes.append({
                                "run_id": run_id,
                                "equation": "cavity",
                                "method": method,
                                "parameter": float(Re),
                                "trajectory_index": traj_index + 1,
                                "case_id": case_suffix,
                                "opinf_success": True,
                                "pipeline_success": False,
                                "failure_stage": "pipeline_postproc",
                                "failure_status": "postproc_error",
                                "error": str(exc),
                            })
                            continue

                    first_vals = [v for v in first_errs if v is not None]
                    second_vals = [v for v in second_errs if v is not None]
                    summary["cavity"].setdefault(method, {})[str(Re)] = {
                        "n_traj": len(full_errs),
                        "mean_full": float(np.mean(full_errs)) if full_errs else None,
                        "min_full": float(np.min(full_errs)) if full_errs else None,
                        "max_full": float(np.max(full_errs)) if full_errs else None,
                        "mean_first": float(np.mean(first_vals)) if first_vals else None,
                        "mean_second": float(np.mean(second_vals)) if second_vals else None,
                    }

    summary_path = base_dir / "summary_split_errors.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved split-error summary to: {summary_path}")

    outcomes_path = base_dir / f"case_outcomes_{run_id}.jsonl"
    with outcomes_path.open("w") as f:
        for rec in case_outcomes:
            f.write(json.dumps(rec) + "\n")
    print(f"Saved case outcomes to: {outcomes_path}")

    success_summary = aggregate_case_stats(case_outcomes)
    success_summary["run_id"] = run_id
    success_summary_path = base_dir / "summary_success_rates.json"
    with success_summary_path.open("w") as f:
        json.dump(success_summary, f, indent=2)
    print(f"Saved success-rate summary to: {success_summary_path}")


if __name__ == "__main__":
    main()
