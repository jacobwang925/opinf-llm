#!/usr/bin/env python3
"""
LLM Tool Calling for Parametric Cavity OpInf Operators.

Uses LLM tool calling to request interpolation or linear regression of
OpInf operators at target Reynolds numbers. The tool executes the
numeric step and returns precise operators.

Default provider: OpenAI (gpt-4o). Provider hooks are structured to
allow adding other LLMs later.
"""

import argparse
import json
import os
import pickle
import re
import time
from typing import Any, Dict, List

import numpy as np
from scipy.interpolate import interp1d

from llm_tool_calling_provider import call_llm_with_tools

# Load API keys from .env
try:
    from load_env import load_env
    load_env()
except ImportError:
    pass

# Global cache for tool access
_GLOBAL_MODEL_DATA = None


def _parse_json_from_text(text: Any) -> Any:
    """Best-effort JSON extraction from plain-text LLM responses."""
    if text is None:
        return None
    if not isinstance(text, str):
        text = str(text)
    text = text.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"```(?:json)?\\s*(\\{.*?\\})\\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start:end + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            return None
    return None


# =============================================================================
# Data Loading
# =============================================================================

def load_pickle_auto(filepath: str) -> Any:
    """Load pickle file, auto-detecting gzip compression."""
    import gzip

    with open(filepath, "rb") as f:
        magic = f.read(2)
        f.seek(0)

        if magic == b"\x1f\x8b":
            with gzip.open(filepath, "rb") as gz:
                return pickle.load(gz)
        return pickle.load(f)


def load_cavity_model(model_path: str) -> Dict[str, Any]:
    """Load cavity parametric model and extract training data."""
    model = load_pickle_auto(model_path)
    per_Re_models = model["per_Re_models"]

    Re_train = [m["Re"] for m in per_Re_models]
    operators_train = {
        "H": [m["H"] for m in per_Re_models],
        "A": [m["A"] for m in per_Re_models],
        "B": [m["B"] for m in per_Re_models],
        "C": [m["C"] for m in per_Re_models],
    }

    return {
        "Re_train": Re_train,
        "operators_train": operators_train,
    }


# =============================================================================
# Tool Functions (LLM can call these)
# =============================================================================

def interpolate_operators(
    Re_train: List[float],
    operators_train: Dict[str, List[List[float]]],
    Re_query: float,
    method: str = "linear",
) -> Dict[str, Any]:
    """Interpolate operators vs Re using scipy interp1d."""
    Re_arr = np.asarray(Re_train, dtype=float)
    operators_out: Dict[str, Any] = {}

    if method not in ["linear", "quadratic", "cubic"]:
        method = "linear"
    if method == "quadratic" and len(Re_arr) < 3:
        method = "linear"
    if method == "cubic" and len(Re_arr) < 4:
        method = "quadratic" if len(Re_arr) >= 3 else "linear"

    for op_name, op_list in operators_train.items():
        op_stack = np.array(op_list)
        orig_shape = op_stack.shape[1:]
        op_flat = op_stack.reshape(len(Re_arr), -1)

        interp_func = interp1d(
            Re_arr,
            op_flat,
            axis=0,
            kind=method,
            fill_value="extrapolate",
        )
        op_q = interp_func(Re_query).reshape(orig_shape)

        operators_out[op_name] = {
            "values": op_q.tolist(),
            "shape": list(op_q.shape),
            "norm": float(np.linalg.norm(op_q)),
            "mean": float(np.mean(op_q)),
            "std": float(np.std(op_q)),
            "min": float(np.min(op_q)),
            "max": float(np.max(op_q)),
        }

    return {
        "operators": operators_out,
        "method": method,
        "Re_query": Re_query,
        "success": True,
    }


def interpolate_operators_batch(
    Re_train: List[float],
    operators_train: Dict[str, List[List[float]]],
    Re_queries: List[float],
    method: str = "linear",
) -> Dict[str, Any]:
    """Interpolate operators for multiple Re queries."""
    batch_outputs = {}
    for Re_query in Re_queries:
        batch_outputs[str(Re_query)] = interpolate_operators(
            Re_train, operators_train, Re_query, method
        )
    return {
        "predictions": batch_outputs,
        "method": method,
        "Re_queries": Re_queries,
        "success": True,
    }


def linear_regress_operators(
    Re_train: List[float],
    operators_train: Dict[str, List[List[float]]],
    Re_query: float,
) -> Dict[str, Any]:
    """
    Perform per-entry linear regression vs normalized Re.

    Fits y = a*z + b, where z = (Re-mean)/std.
    """
    Re_arr = np.asarray(Re_train, dtype=float)
    Re_mean = Re_arr.mean()
    Re_std = Re_arr.std() if Re_arr.std() > 1e-12 else 1.0
    z = (Re_arr - Re_mean) / Re_std
    z_q = (Re_query - Re_mean) / Re_std

    operators_out: Dict[str, Any] = {}

    for op_name, op_list in operators_train.items():
        op_stack = np.array(op_list)  # (n_Re, ...)
        orig_shape = op_stack.shape[1:]
        op_flat = op_stack.reshape(len(Re_arr), -1)

        A_mat = np.vstack([z, np.ones_like(z)]).T
        coeffs, _, _, _ = np.linalg.lstsq(A_mat, op_flat, rcond=None)
        a = coeffs[0]
        b = coeffs[1]
        op_q = (a * z_q + b).reshape(orig_shape)

        operators_out[op_name] = {
            "values": op_q.tolist(),
            "shape": list(op_q.shape),
            "norm": float(np.linalg.norm(op_q)),
            "mean": float(np.mean(op_q)),
            "std": float(np.std(op_q)),
            "min": float(np.min(op_q)),
            "max": float(np.max(op_q)),
        }

    return {
        "operators": operators_out,
        "method": "linear_regression",
        "Re_query": Re_query,
        "success": True,
    }


def linear_regress_operators_batch(
    Re_train: List[float],
    operators_train: Dict[str, List[List[float]]],
    Re_queries: List[float],
) -> Dict[str, Any]:
    """Compute linear regression operators for multiple Re queries."""
    batch_outputs = {}
    for Re_query in Re_queries:
        batch_outputs[str(Re_query)] = linear_regress_operators(
            Re_train, operators_train, Re_query
        )
    return {
        "predictions": batch_outputs,
        "method": "linear_regression",
        "Re_queries": Re_queries,
        "success": True,
    }


def analyze_parameter_range(Re_train: List[float], Re_query: float) -> Dict[str, Any]:
    """Analyze whether Re_query is interpolation or extrapolation."""
    Re_min = min(Re_train)
    Re_max = max(Re_train)
    is_interpolation = Re_min <= Re_query <= Re_max

    if is_interpolation and Re_max != Re_min:
        relative_position = (Re_query - Re_min) / (Re_max - Re_min)
    else:
        relative_position = None

    if Re_query < Re_min:
        extrapolation_distance = abs(Re_query - Re_min)
        extrapolation_direction = "below"
    elif Re_query > Re_max:
        extrapolation_distance = abs(Re_query - Re_max)
        extrapolation_direction = "above"
    else:
        extrapolation_distance = 0.0
        extrapolation_direction = None

    return {
        "Re_train": Re_train,
        "Re_query": Re_query,
        "Re_range": [Re_min, Re_max],
        "is_interpolation": is_interpolation,
        "relative_position": relative_position,
        "extrapolation_distance": extrapolation_distance,
        "extrapolation_direction": extrapolation_direction,
        "confidence": "high" if is_interpolation else "low",
        "recommendation": "Use interpolation" if is_interpolation else
                         "Extrapolation detected - accuracy may degrade",
    }


def validate_operators(operators: Dict[str, Dict]) -> Dict[str, Any]:
    """Basic validation for NaN/Inf and max magnitude."""
    validations = {}

    for op_name, op_data in operators.items():
        op_array = np.array(op_data["values"])
        validations[op_name] = {
            "has_nan": bool(np.any(np.isnan(op_array))),
            "has_inf": bool(np.any(np.isinf(op_array))),
            "is_finite": bool(np.all(np.isfinite(op_array))),
            "max_abs_value": float(np.max(np.abs(op_array))),
        }

    all_valid = all(
        not v["has_nan"] and not v["has_inf"] and v["is_finite"]
        for v in validations.values()
    )

    return {
        "is_valid": all_valid,
        "operator_checks": validations,
    }


def simple_interpolate(Re_query: float, method: str = "linear") -> Dict[str, Any]:
    """Simplified interpolation using globally loaded data."""
    global _GLOBAL_MODEL_DATA
    if _GLOBAL_MODEL_DATA is None:
        return {"error": "No model data loaded"}

    Re_train = _GLOBAL_MODEL_DATA["Re_train"]
    operators_train = _GLOBAL_MODEL_DATA["operators_train"]
    return interpolate_operators(Re_train, operators_train, Re_query, method)


def simple_interpolate_batch(Re_queries: List[float], method: str = "linear") -> Dict[str, Any]:
    """Simplified interpolation for multiple Re values using global data."""
    global _GLOBAL_MODEL_DATA
    if _GLOBAL_MODEL_DATA is None:
        return {"error": "No model data loaded"}

    Re_train = _GLOBAL_MODEL_DATA["Re_train"]
    operators_train = _GLOBAL_MODEL_DATA["operators_train"]
    return interpolate_operators_batch(Re_train, operators_train, Re_queries, method)


def simple_linear_regress(Re_query: float) -> Dict[str, Any]:
    """Simplified regression using globally loaded data."""
    global _GLOBAL_MODEL_DATA
    if _GLOBAL_MODEL_DATA is None:
        return {"error": "No model data loaded"}

    Re_train = _GLOBAL_MODEL_DATA["Re_train"]
    operators_train = _GLOBAL_MODEL_DATA["operators_train"]
    return linear_regress_operators(Re_train, operators_train, Re_query)


def simple_linear_regress_batch(Re_queries: List[float]) -> Dict[str, Any]:
    """Simplified regression for multiple Re values using global data."""
    global _GLOBAL_MODEL_DATA
    if _GLOBAL_MODEL_DATA is None:
        return {"error": "No model data loaded"}

    Re_train = _GLOBAL_MODEL_DATA["Re_train"]
    operators_train = _GLOBAL_MODEL_DATA["operators_train"]
    return linear_regress_operators_batch(Re_train, operators_train, Re_queries)


# =============================================================================
# Tool Definitions for LLM
# =============================================================================

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "analyze_parameter_range",
            "description": "Check if Re is within training range (interpolation) or outside (extrapolation).",
            "parameters": {
                "type": "object",
                "properties": {
                    "Re_train": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Training Re values"
                    },
                    "Re_query": {
                        "type": "number",
                        "description": "Query Re value"
                    }
                },
                "required": ["Re_train", "Re_query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "simple_interpolate",
            "description": "Interpolate operators for a query Re using pre-loaded data.",
            "parameters": {
                "type": "object",
                "properties": {
                    "Re_query": {
                        "type": "number",
                        "description": "Target Reynolds number"
                    },
                    "method": {
                        "type": "string",
                        "description": "Interpolation method (linear, quadratic, cubic)"
                    }
                },
                "required": ["Re_query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "simple_interpolate_batch",
            "description": "Interpolate operators for multiple Re values using pre-loaded data.",
            "parameters": {
                "type": "object",
                "properties": {
                    "Re_queries": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Target Reynolds numbers"
                    },
                    "method": {
                        "type": "string",
                        "description": "Interpolation method (linear, quadratic, cubic)"
                    }
                },
                "required": ["Re_queries"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "interpolate_operators",
            "description": "Interpolate OpInf operators vs Re (per entry).",
            "parameters": {
                "type": "object",
                "properties": {
                    "Re_train": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Training Re values"
                    },
                    "operators_train": {
                        "type": "object",
                        "description": "Dict mapping operator names to lists of matrices at each Re.",
                        "additionalProperties": {
                            "type": "array",
                            "items": {
                                "type": "array",
                                "items": {"type": "number"}
                            }
                        }
                    },
                    "Re_query": {
                        "type": "number",
                        "description": "Target Reynolds number"
                    },
                    "method": {
                        "type": "string",
                        "description": "Interpolation method (linear, quadratic, cubic)"
                    }
                },
                "required": ["Re_train", "operators_train", "Re_query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "interpolate_operators_batch",
            "description": "Interpolate OpInf operators vs Re for multiple queries.",
            "parameters": {
                "type": "object",
                "properties": {
                    "Re_train": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Training Re values"
                    },
                    "operators_train": {
                        "type": "object",
                        "description": "Dict mapping operator names to lists of matrices at each Re.",
                        "additionalProperties": {
                            "type": "array",
                            "items": {
                                "type": "array",
                                "items": {"type": "number"}
                            }
                        }
                    },
                    "Re_queries": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Target Reynolds numbers"
                    },
                    "method": {
                        "type": "string",
                        "description": "Interpolation method (linear, quadratic, cubic)"
                    }
                },
                "required": ["Re_train", "operators_train", "Re_queries"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "simple_linear_regress",
            "description": "Compute linear-regression operators for a query Re using pre-loaded data.",
            "parameters": {
                "type": "object",
                "properties": {
                    "Re_query": {
                        "type": "number",
                        "description": "Target Reynolds number"
                    }
                },
                "required": ["Re_query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "simple_linear_regress_batch",
            "description": "Compute linear-regression operators for multiple Re values using pre-loaded data.",
            "parameters": {
                "type": "object",
                "properties": {
                    "Re_queries": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Target Reynolds numbers"
                    }
                },
                "required": ["Re_queries"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "linear_regress_operators",
            "description": "Linear regression of OpInf operators vs normalized Re (per entry).",
            "parameters": {
                "type": "object",
                "properties": {
                    "Re_train": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Training Re values"
                    },
                    "operators_train": {
                        "type": "object",
                        "description": "Dict mapping operator names to lists of matrices at each Re.",
                        "additionalProperties": {
                            "type": "array",
                            "items": {
                                "type": "array",
                                "items": {"type": "number"}
                            }
                        }
                    },
                    "Re_query": {
                        "type": "number",
                        "description": "Target Reynolds number"
                    }
                },
                "required": ["Re_train", "operators_train", "Re_query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "linear_regress_operators_batch",
            "description": "Linear regression of OpInf operators vs normalized Re for multiple queries.",
            "parameters": {
                "type": "object",
                "properties": {
                    "Re_train": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Training Re values"
                    },
                    "operators_train": {
                        "type": "object",
                        "description": "Dict mapping operator names to lists of matrices at each Re.",
                        "additionalProperties": {
                            "type": "array",
                            "items": {
                                "type": "array",
                                "items": {"type": "number"}
                            }
                        }
                    },
                    "Re_queries": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Target Reynolds numbers"
                    }
                },
                "required": ["Re_train", "operators_train", "Re_queries"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "validate_operators",
            "description": "Validate interpolated operators (finite values, no NaN/Inf).",
            "parameters": {
                "type": "object",
                "properties": {
                    "operators": {
                        "type": "object",
                        "description": "Operator dict with 'values' for each operator"
                    }
                },
                "required": ["operators"]
            }
        }
    }
]


# =============================================================================
# LLM Interaction
# =============================================================================

def execute_tool_call(tool_call):
    """Execute a tool call from the LLM."""
    function_name = tool_call.function.name
    try:
        arguments = json.loads(tool_call.function.arguments)
    except json.JSONDecodeError as exc:
        return {"error": f"JSON decode error: {exc}"}

    function_map = {
        "simple_interpolate": simple_interpolate,
        "simple_interpolate_batch": simple_interpolate_batch,
        "interpolate_operators": interpolate_operators,
        "interpolate_operators_batch": interpolate_operators_batch,
        "simple_linear_regress": simple_linear_regress,
        "simple_linear_regress_batch": simple_linear_regress_batch,
        "linear_regress_operators": linear_regress_operators,
        "linear_regress_operators_batch": linear_regress_operators_batch,
        "analyze_parameter_range": analyze_parameter_range,
        "validate_operators": validate_operators,
    }

    if function_name not in function_map:
        return {"error": f"Unknown function: {function_name}"}

    try:
        return function_map[function_name](**arguments)
    except Exception as exc:
        return {"error": str(exc)}


def run_tool_calling_workflow(
    model_data: Dict[str, Any],
    Re_query: float,
    provider: str,
    model_name: str,
    method: str,
) -> Dict[str, Any]:
    """Run the complete workflow with LLM tool calling."""
    global _GLOBAL_MODEL_DATA
    _GLOBAL_MODEL_DATA = model_data

    Re_train = model_data["Re_train"]
    operator_names = list(model_data["operators_train"].keys())

    if method == "regression":
        step_2 = f"2) simple_linear_regress(Re_query)"
        instruction_2 = f"2) Then, call simple_linear_regress({Re_query}) to get the operators"
    else:
        step_2 = f"2) simple_interpolate(Re_query, method)"
        instruction_2 = f"2) Then, call simple_interpolate({Re_query}, \"linear\") to get the operators"

    prompt = f"""You are an expert in reduced-order modeling and operator inference.

Task: Predict OpInf operators for cavity flow at Re = {Re_query}

Available data (already loaded in the system):
- Training Re values: {Re_train}
- Operators at each Re: {operator_names}

Use these tools in order:
1) analyze_parameter_range(Re_train, Re_query)
{step_2}
3) validate_operators(operators)

Please solve step-by-step using the tools.
{instruction_2}
"""

    messages = [
        {
            "role": "system",
            "content": "You are an expert in scientific computing. Use tools for numeric results.",
        },
        {"role": "user", "content": prompt},
    ]

    max_iterations = 10
    final_result = None

    for _ in range(max_iterations):
        response = call_llm_with_tools(provider, messages, TOOLS, model_name)
        assistant_message = response.choices[0].message
        messages.append(assistant_message)

        if assistant_message.tool_calls:
            for tool_call in assistant_message.tool_calls:
                result = execute_tool_call(tool_call)
                if tool_call.function.name in [
                    "simple_linear_regress",
                    "linear_regress_operators",
                    "simple_interpolate",
                    "interpolate_operators",
                ]:
                    final_result = result
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": tool_call.function.name,
                    "content": json.dumps(result),
                })
        else:
            parsed = _parse_json_from_text(getattr(assistant_message, "content", None))
            if isinstance(parsed, dict) and parsed.get("operators"):
                return parsed
            break

    if final_result is None:
        if method == "regression":
            print("⚠ LLM returned no tool calls; falling back to local regression.")
            return simple_linear_regress(Re_query)
        print("⚠ LLM returned no tool calls; falling back to local interpolation.")
        return simple_interpolate(Re_query, method=method)
    return final_result


def run_tool_calling_workflow_batch(
    model_data: Dict[str, Any],
    Re_queries: List[float],
    provider: str,
    model_name: str,
    method: str,
) -> Dict[str, Any]:
    """Run tool calling once to compute operators for multiple Re values."""
    global _GLOBAL_MODEL_DATA
    _GLOBAL_MODEL_DATA = model_data

    Re_train = model_data["Re_train"]
    operator_names = list(model_data["operators_train"].keys())

    if method == "regression":
        step_2 = "2) simple_linear_regress_batch(Re_queries) to compute all operators in one call"
    else:
        step_2 = "2) simple_interpolate_batch(Re_queries, method) to compute all operators in one call"

    prompt = f"""You are an expert in reduced-order modeling and operator inference.

Task: Predict OpInf operators for cavity flow at multiple Re values.

Re queries: {Re_queries}
Training Re values: {Re_train}
Operators: {operator_names}

Use these tools in order:
1) analyze_parameter_range(Re_train, Re_query) for any queries that look borderline
{step_2}
3) validate_operators(operators) if needed

Please solve step-by-step using the tools.
"""

    messages = [
        {
            "role": "system",
            "content": "You are an expert in scientific computing. Use tools for numeric results.",
        },
        {"role": "user", "content": prompt},
    ]

    max_iterations = 3

    for _ in range(max_iterations):
        response = call_llm_with_tools(provider, messages, TOOLS, model_name)
        assistant_message = response.choices[0].message
        messages.append(assistant_message)

        if assistant_message.tool_calls:
            for tool_call in assistant_message.tool_calls:
                result = execute_tool_call(tool_call)
                if tool_call.function.name in [
                    "simple_linear_regress_batch",
                    "linear_regress_operators_batch",
                    "simple_interpolate_batch",
                    "interpolate_operators_batch",
                ]:
                    # Return immediately to avoid adding huge tool outputs to context.
                    return result
                # For non-batch tools, keep a minimal trace (no large payloads).
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": tool_call.function.name,
                    "content": json.dumps({"ok": True}),
                })
        else:
            parsed = _parse_json_from_text(getattr(assistant_message, "content", None))
            if isinstance(parsed, dict) and parsed.get("predictions"):
                return parsed
            break

    if method == "regression":
        print("⚠ LLM returned no tool calls; falling back to local batch regression.")
        return simple_linear_regress_batch(Re_queries)
    print("⚠ LLM returned no tool calls; falling back to local batch interpolation.")
    return simple_interpolate_batch(Re_queries, method=method)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="LLM tool calling for cavity operator prediction"
    )
    parser.add_argument("--model", type=str, required=True,
                        help="Parametric cavity model .pkl file")
    parser.add_argument("--query_Re", type=float,
                        help="Target Reynolds number")
    parser.add_argument("--query_Re_values", nargs="+", type=float,
                        help="List of target Reynolds numbers")
    parser.add_argument("--provider", type=str, default="openai",
                        choices=["openai", "anthropic", "gemini", "deepseek", "qwen"],
                        help="LLM provider")
    parser.add_argument("--model_name", type=str, default="gpt-4o",
                        help="LLM model name")
    parser.add_argument("--method", type=str, default="regression",
                        choices=["interpolation", "regression"],
                        help="Operator prediction method")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON file for predicted operators")
    parser.add_argument("--output_dir", type=str, default="llm_cavity_predictions",
                        help="Directory for per-Re outputs when using --query_Re_values")
    parser.add_argument("--batch_size", type=int, default=0,
                        help="Batch size for tool-calling. 0 means all at once.")
    parser.add_argument("--sleep_secs", type=float, default=0.0,
                        help="Sleep between batches (seconds)")

    args = parser.parse_args()

    if args.query_Re is None and not args.query_Re_values:
        parser.error("Specify --query_Re or --query_Re_values.")

    model_data = load_cavity_model(args.model)
    global _GLOBAL_MODEL_DATA
    _GLOBAL_MODEL_DATA = model_data

    if args.query_Re_values:
        os.makedirs(args.output_dir, exist_ok=True)
        Re_values = list(args.query_Re_values)
        if args.batch_size and args.batch_size > 0:
            batches = [
                Re_values[i:i + args.batch_size]
                for i in range(0, len(Re_values), args.batch_size)
            ]
        else:
            batches = [Re_values]

        for batch_idx, batch in enumerate(batches, start=1):
            print(f"Batch {batch_idx}/{len(batches)}: Re values = {batch}")
            try:
                result = run_tool_calling_workflow_batch(
                    model_data,
                    batch,
                    args.provider,
                    args.model_name,
                    args.method,
                )
            except Exception as exc:
                print(f"⚠ Tool calling failed ({exc}); falling back to local batch {args.method}.")
                if args.method == "regression":
                    result = simple_linear_regress_batch(batch)
                else:
                    result = simple_interpolate_batch(batch, method=args.method)

            if not isinstance(result, dict) or not result.get("predictions"):
                print("⚠ No predictions returned; falling back to local batch operators.")
                if args.method == "regression":
                    result = simple_linear_regress_batch(batch)
                else:
                    result = simple_interpolate_batch(batch, method=args.method)

            predictions = result.get("predictions", {})
            normalized_predictions = {}
            for key, value in predictions.items():
                normalized_predictions[key] = value
                try:
                    normalized_predictions[str(float(key))] = value
                except (TypeError, ValueError):
                    pass
            for Re_query in batch:
                key = str(Re_query)
                if key not in normalized_predictions:
                    alt_key = str(float(Re_query))
                    if alt_key in normalized_predictions:
                        key = alt_key
                    else:
                        if args.provider == "deepseek":
                            print(f"⚠ Missing DeepSeek prediction for Re={Re_query}; skipping.")
                            continue
                        raise RuntimeError(f"Missing prediction for Re={Re_query}")
                output = {
                    "query_Re": Re_query,
                    "llm_provider": args.provider,
                    "llm_model": args.model_name,
                    "predicted_operators": normalized_predictions[key],
                }
                output_path = os.path.join(
                    args.output_dir, f"llm_cavity_operators_Re{Re_query}.json"
                )
                with open(output_path, "w") as f:
                    json.dump(output, f, indent=2)
                print(f"✓ Saved predicted operators to: {output_path}")

            if args.sleep_secs > 0 and batch_idx < len(batches):
                time.sleep(args.sleep_secs)
    else:
        result = run_tool_calling_workflow(
            model_data,
            args.query_Re,
            args.provider,
            args.model_name,
            args.method,
        )
        if result is None:
            if args.provider == "deepseek":
                print("⚠ No operators returned by DeepSeek for single query.")
                return
            raise RuntimeError("No operators were returned by tools.")

        output = {
            "query_Re": args.query_Re,
            "llm_provider": args.provider,
            "llm_model": args.model_name,
            "predicted_operators": result,
        }
        if args.output is None:
            args.output = f"llm_cavity_operators_Re{args.query_Re}.json"
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)
        print(f"✓ Saved predicted operators to: {args.output}")


if __name__ == "__main__":
    main()
