#!/usr/bin/env python3
"""
LLM Tool Calling for OpInf Operator Interpolation

Uses function calling to let the LLM directly call
interpolation functions to compute OpInf operators at new parameter values.

The LLM can:
1. Call interpolate_operators() to perform scipy interpolation
2. Call validate_operators() to check physical constraints
3. Call analyze_parameter_range() to determine if interpolation/extrapolation
4. Use multiple tool calls to reason through the problem

This gives the LLM actual code execution capability while maintaining
numerical precision.

Author: Jacob Wang
"""

import numpy as np
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Literal
from scipy.interpolate import interp1d

from llm_tool_calling_provider import call_llm_with_tools

# Load API keys from .env
try:
    from load_env import load_env
    load_env()
except ImportError:
    pass

# Load API keys from .env file
try:
    from load_env import load_env
    load_env()
except ImportError:
    pass  # load_env.py not available, use environment variables


# ============================================================================
# Tool Functions (LLM can call these)
# ============================================================================

def interpolate_operators(
    nu_train: List[float],
    operators_train: Dict[str, List[List[float]]],
    nu_query: float,
    method: Literal["linear", "quadratic", "cubic"] = "linear"
) -> Dict[str, Any]:
    """
    Interpolate OpInf operators to a new parameter value.

    Args:
        nu_train: Training parameter values (e.g., [0.1, 0.5, 2.0])
        operators_train: Dict of operator matrices at each nu_train
                        Format: {"A": [matrix_at_nu1, matrix_at_nu2, ...],
                                "B": [...], "C": [...]}
        nu_query: Target parameter value
        method: Interpolation method (linear, quadratic, cubic)

    Returns:
        Dict with interpolated operators and metadata
    """
    nu_array = np.array(nu_train)
    operator_names = list(operators_train.keys())

    # Validate method against number of points
    if method == "quadratic" and len(nu_train) < 3:
        method = "linear"
    if method == "cubic" and len(nu_train) < 4:
        method = "quadratic" if len(nu_train) >= 3 else "linear"

    interpolated = {}

    for op_name in operator_names:
        # Stack operator matrices
        op_stack = np.array(operators_train[op_name])  # (n_params, ...)

        # Interpolate element-wise
        interp_func = interp1d(
            nu_array, op_stack, axis=0,
            kind=method,
            fill_value='extrapolate'
        )

        op_interp = interp_func(nu_query)

        interpolated[op_name] = {
            "values": op_interp.tolist(),
            "shape": list(op_interp.shape),
            "norm": float(np.linalg.norm(op_interp)),
            "mean": float(np.mean(op_interp)),
            "std": float(np.std(op_interp)),
            "min": float(np.min(op_interp)),
            "max": float(np.max(op_interp))
        }

    return {
        "operators": interpolated,
        "method": method,
        "nu_query": nu_query,
        "success": True
    }


def linear_regress_operators(
    nu_train: List[float],
    operators_train: Dict[str, List[List[float]]],
    nu_query: float,
) -> Dict[str, Any]:
    """
    Linear regression of operators vs normalized nu.
    Fits y = a*z + b per entry, where z = (nu - mean) / std.
    """
    nu_array = np.array(nu_train, dtype=float)
    nu_mean = float(nu_array.mean())
    nu_std = float(nu_array.std()) if float(nu_array.std()) > 1e-12 else 1.0
    z = (nu_array - nu_mean) / nu_std
    z_q = (float(nu_query) - nu_mean) / nu_std

    operator_names = list(operators_train.keys())
    outputs = {}
    for op_name in operator_names:
        op_stack = np.array(operators_train[op_name])  # (n_params, ...)
        flat = op_stack.reshape(len(nu_array), -1)
        X = np.vstack([z, np.ones_like(z)]).T
        coeffs, _, _, _ = np.linalg.lstsq(X, flat, rcond=None)
        a = coeffs[0]
        b = coeffs[1]
        pred_flat = a * z_q + b
        pred = pred_flat.reshape(op_stack.shape[1:])
        outputs[op_name] = {
            "values": pred.tolist(),
            "shape": list(pred.shape),
            "norm": float(np.linalg.norm(pred)),
            "mean": float(np.mean(pred)),
            "std": float(np.std(pred)),
            "min": float(np.min(pred)),
            "max": float(np.max(pred)),
        }

    return {
        "operators": outputs,
        "method": "regression",
        "nu_query": nu_query,
        "success": True,
    }


def linear_regress_operators_batch(
    nu_train: List[float],
    operators_train: Dict[str, List[List[float]]],
    nu_queries: List[float],
) -> Dict[str, Any]:
    """Batch linear regression for multiple nu values."""
    outputs = {}
    for nu_query in nu_queries:
        outputs[str(nu_query)] = linear_regress_operators(nu_train, operators_train, nu_query)
    return {
        "nu_queries": nu_queries,
        "method": "regression",
        "predictions": outputs,
        "success": True,
    }


def analyze_parameter_range(
    nu_train: List[float],
    nu_query: float
) -> Dict[str, Any]:
    """
    Analyze whether query is interpolation or extrapolation.

    Args:
        nu_train: Training parameter values
        nu_query: Query parameter value

    Returns:
        Analysis of parameter range
    """
    nu_min = min(nu_train)
    nu_max = max(nu_train)

    is_interpolation = nu_min <= nu_query <= nu_max

    # Calculate relative position
    if is_interpolation:
        if nu_max != nu_min:
            relative_position = (nu_query - nu_min) / (nu_max - nu_min)
        else:
            relative_position = 0.5
    else:
        relative_position = None

    # Determine extrapolation distance
    if nu_query < nu_min:
        extrapolation_distance = abs(nu_query - nu_min)
        extrapolation_direction = "below"
    elif nu_query > nu_max:
        extrapolation_distance = abs(nu_query - nu_max)
        extrapolation_direction = "above"
    else:
        extrapolation_distance = 0
        extrapolation_direction = None

    return {
        "nu_train": nu_train,
        "nu_query": nu_query,
        "nu_range": [nu_min, nu_max],
        "is_interpolation": is_interpolation,
        "relative_position": relative_position,
        "extrapolation_distance": extrapolation_distance,
        "extrapolation_direction": extrapolation_direction,
        "confidence": "high" if is_interpolation else "low",
        "recommendation": "Use linear interpolation" if is_interpolation else
                         "Extrapolation detected - results may be less accurate"
    }


def validate_operators(
    operators: Dict[str, Dict],
    equation_type: str = "heat"
) -> Dict[str, Any]:
    """
    Validate physical constraints on operators.

    Args:
        operators: Interpolated operators with 'values' field
        equation_type: Type of equation (heat, burgers, etc.)

    Returns:
        Validation results
    """
    validations = {}

    for op_name, op_data in operators.items():
        op_array = np.array(op_data["values"])

        checks = {
            "has_nan": bool(np.any(np.isnan(op_array))),
            "has_inf": bool(np.any(np.isinf(op_array))),
            "is_finite": bool(np.all(np.isfinite(op_array))),
            "max_abs_value": float(np.max(np.abs(op_array)))
        }

        # Equation-specific checks
        if equation_type == "heat":
            # For heat equation, A should have negative eigenvalues (stable)
            if op_name == "A" and len(op_array.shape) == 2 and op_array.shape[0] == op_array.shape[1]:
                try:
                    eigvals = np.linalg.eigvals(op_array)
                    checks["max_real_eigenvalue"] = float(np.max(np.real(eigvals)))
                    checks["is_stable"] = checks["max_real_eigenvalue"] < 0
                except:
                    checks["eigenvalue_check"] = "failed"

        elif equation_type == "burgers":
            # For Burgers equation, check H tensor shape
            if op_name == "H":
                checks["shape_info"] = f"H tensor: {op_array.shape}"

        validations[op_name] = checks

    # Overall validation
    all_valid = all(
        not v["has_nan"] and not v["has_inf"] and v["is_finite"]
        for v in validations.values()
    )

    return {
        "is_valid": all_valid,
        "operator_checks": validations,
        "equation_type": equation_type
    }


# ============================================================================
# Tool Definitions for LLM
# ============================================================================

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "simple_interpolate",
            "description": "Interpolate OpInf operators to a query parameter using pre-loaded training data. This is the easiest way to get interpolated operators - just provide the target nu value.",
            "parameters": {
                "type": "object",
                "properties": {
                    "nu_query": {
                        "type": "number",
                        "description": "Target parameter value to interpolate to"
                    },
                    "method": {
                        "type": "string",
                        "enum": ["linear", "quadratic", "cubic"],
                        "description": "Interpolation method to use (default: linear)",
                        "default": "linear"
                    }
                },
                "required": ["nu_query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "linear_regress_operators",
            "description": "Linear regression of OpInf operators vs normalized nu (per entry).",
            "parameters": {
                "type": "object",
                "properties": {
                    "nu_train": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Training parameter values"
                    },
                    "operators_train": {
                        "type": "object",
                        "description": "Dict mapping operator names to lists of matrices at each nu"
                    },
                    "nu_query": {
                        "type": "number",
                        "description": "Target parameter value"
                    }
                },
                "required": ["nu_train", "operators_train", "nu_query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "linear_regress_operators_batch",
            "description": "Linear regression of OpInf operators vs normalized nu for multiple queries.",
            "parameters": {
                "type": "object",
                "properties": {
                    "nu_train": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Training parameter values"
                    },
                    "operators_train": {
                        "type": "object",
                        "description": "Dict mapping operator names to lists of matrices at each nu"
                    },
                    "nu_queries": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Target parameter values"
                    }
                },
                "required": ["nu_train", "operators_train", "nu_queries"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "simple_interpolate_batch",
            "description": "Interpolate OpInf operators for multiple nu queries using pre-loaded training data.",
            "parameters": {
                "type": "object",
                "properties": {
                    "nu_queries": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Target nu values"
                    },
                    "method": {
                        "type": "string",
                        "enum": ["linear", "quadratic", "cubic"],
                        "default": "linear"
                    }
                },
                "required": ["nu_queries"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "simple_linear_regress",
            "description": "Linear regression of OpInf operators for a single nu query using pre-loaded data.",
            "parameters": {
                "type": "object",
                "properties": {
                    "nu_query": {
                        "type": "number",
                        "description": "Target nu value"
                    }
                },
                "required": ["nu_query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "simple_linear_regress_batch",
            "description": "Linear regression of OpInf operators for multiple nu queries using pre-loaded data.",
            "parameters": {
                "type": "object",
                "properties": {
                    "nu_queries": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Target nu values"
                    }
                },
                "required": ["nu_queries"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "interpolate_operators",
            "description": "Interpolate OpInf operator matrices to a new parameter value using scipy. Returns high-precision numerical interpolation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "nu_train": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Training parameter values (e.g., [0.1, 0.5, 2.0])"
                    },
                    "operators_train": {
                        "type": "object",
                        "description": "Dict mapping operator names (A, B, C) to lists of matrices at each training nu. Each operator is a list where each element is a matrix (nested list).",
                        "additionalProperties": {
                            "type": "array",
                            "items": {
                                "type": "array",
                                "items": {"type": "number"}
                            }
                        }
                    },
                    "nu_query": {
                        "type": "number",
                        "description": "Target parameter value to interpolate to"
                    },
                    "method": {
                        "type": "string",
                        "enum": ["linear", "quadratic", "cubic"],
                        "description": "Interpolation method to use",
                        "default": "linear"
                    }
                },
                "required": ["nu_train", "operators_train", "nu_query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_parameter_range",
            "description": "Analyze whether the query parameter is within the training range (interpolation) or outside (extrapolation). Returns confidence assessment.",
            "parameters": {
                "type": "object",
                "properties": {
                    "nu_train": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Training parameter values"
                    },
                    "nu_query": {
                        "type": "number",
                        "description": "Query parameter value"
                    }
                },
                "required": ["nu_train", "nu_query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "validate_operators",
            "description": "Validate that interpolated operators satisfy physical constraints (no NaN, no Inf, stable eigenvalues for heat equation)",
            "parameters": {
                "type": "object",
                "properties": {
                    "operators": {
                        "type": "object",
                        "description": "Operator dict with 'values' field for each operator"
                    },
                    "equation_type": {
                        "type": "string",
                        "enum": ["heat", "burgers", "wave"],
                        "description": "Type of PDE",
                        "default": "heat"
                    }
                },
                "required": ["operators"]
            }
        }
    }
]


def execute_tool_call(tool_call):
    """Execute a tool call from the LLM."""
    function_name = tool_call.function.name

    # Try to parse arguments, handle malformed JSON
    try:
        arguments = json.loads(tool_call.function.arguments)
    except (json.JSONDecodeError, RecursionError) as e:
        print(f"    ⚠ JSON parsing error: {str(e)[:100]}")
        print(f"    Attempting to extract arguments...")
        # For simple_interpolate, we can extract nu_query manually
        if function_name == "simple_interpolate":
            import re
            match = re.search(r'"nu_query":\s*([\d.]+)', tool_call.function.arguments)
            if match:
                arguments = {"nu_query": float(match.group(1))}
                # Try to extract method too
                method_match = re.search(r'"method":\s*"(\w+)"', tool_call.function.arguments)
                if method_match:
                    arguments["method"] = method_match.group(1)
                print(f"    ✓ Extracted arguments: {arguments}")
            else:
                return {"error": f"Could not parse arguments: {str(e)}"}
        elif function_name == "validate_operators":
            return {"error": "Skipped validate_operators due to oversized arguments"}
        else:
            return {"error": f"JSON decode error: {str(e)}"}

    # Map function names to actual functions
    function_map = {
        "simple_interpolate": simple_interpolate,
        "simple_interpolate_batch": simple_interpolate_batch,
        "simple_linear_regress": simple_linear_regress,
        "simple_linear_regress_batch": simple_linear_regress_batch,
        "interpolate_operators": interpolate_operators,
        "analyze_parameter_range": analyze_parameter_range,
        "validate_operators": validate_operators,
        "linear_regress_operators": linear_regress_operators,
        "linear_regress_operators_batch": linear_regress_operators_batch,
    }

    if function_name not in function_map:
        return {"error": f"Unknown function: {function_name}"}

    try:
        result = function_map[function_name](**arguments)
        return result
    except Exception as e:
        return {"error": str(e)}


# Global variable to store data for simpler tool access
_GLOBAL_COEFF_DATA = None

def simple_interpolate(nu_query: float, method: str = "linear") -> Dict[str, Any]:
    """
    Simplified interpolation function that uses global data.
    This makes it easier for LLM to call (just needs nu_query).

    Args:
        nu_query: Target parameter value
        method: Interpolation method (linear, quadratic, cubic)

    Returns:
        Dict with interpolated operators
    """
    global _GLOBAL_COEFF_DATA

    if _GLOBAL_COEFF_DATA is None:
        return {"error": "No data loaded"}

    parameters = _GLOBAL_COEFF_DATA["parameters"]
    nu_train = [p["nu"] for p in parameters]

    # Extract operators
    operator_names = list(parameters[0]["operators"].keys())
    operators_train = {op_name: [] for op_name in operator_names}

    for param in parameters:
        for op_name in operator_names:
            operators_train[op_name].append(param["operators"][op_name]["values"])

    # Call the main interpolation function
    return interpolate_operators(nu_train, operators_train, nu_query, method)


def simple_interpolate_batch(nu_queries: List[float], method: str = "linear") -> Dict[str, Any]:
    """
    Batch interpolation using global data.
    Returns a dict keyed by query value.
    """
    global _GLOBAL_COEFF_DATA

    if _GLOBAL_COEFF_DATA is None:
        return {"error": "No data loaded"}

    parameters = _GLOBAL_COEFF_DATA["parameters"]
    nu_train = [p["nu"] for p in parameters]

    operator_names = list(parameters[0]["operators"].keys())
    operators_train = {op_name: [] for op_name in operator_names}

    for param in parameters:
        for op_name in operator_names:
            operators_train[op_name].append(param["operators"][op_name]["values"])

    outputs = {}
    for nu_query in nu_queries:
        outputs[str(nu_query)] = interpolate_operators(nu_train, operators_train, nu_query, method)

    return {
        "nu_queries": nu_queries,
        "method": method,
        "predictions": outputs,
        "success": True,
    }


def simple_linear_regress(nu_query: float) -> Dict[str, Any]:
    """Simplified linear regression using global data."""
    global _GLOBAL_COEFF_DATA

    if _GLOBAL_COEFF_DATA is None:
        return {"error": "No data loaded"}

    parameters = _GLOBAL_COEFF_DATA["parameters"]
    nu_train = [p["nu"] for p in parameters]
    operator_names = list(parameters[0]["operators"].keys())
    operators_train = {op_name: [] for op_name in operator_names}

    for param in parameters:
        for op_name in operator_names:
            operators_train[op_name].append(param["operators"][op_name]["values"])

    return linear_regress_operators(nu_train, operators_train, nu_query)


def simple_linear_regress_batch(nu_queries: List[float]) -> Dict[str, Any]:
    """Simplified batch regression using global data."""
    global _GLOBAL_COEFF_DATA

    if _GLOBAL_COEFF_DATA is None:
        return {"error": "No data loaded"}

    parameters = _GLOBAL_COEFF_DATA["parameters"]
    nu_train = [p["nu"] for p in parameters]
    operator_names = list(parameters[0]["operators"].keys())
    operators_train = {op_name: [] for op_name in operator_names}

    for param in parameters:
        for op_name in operator_names:
            operators_train[op_name].append(param["operators"][op_name]["values"])

    return linear_regress_operators_batch(nu_train, operators_train, nu_queries)


def run_tool_calling_workflow(coeff_data, query_nu, equation_type="heat", provider="openai", model="gpt-4o", method="interpolation"):
    """
    Run the complete workflow with LLM tool calling.
    """
    global _GLOBAL_COEFF_DATA
    _GLOBAL_COEFF_DATA = coeff_data  # Store data globally for simple_interpolate

    print("\n" + "=" * 70)
    print("LLM Tool Calling Workflow")
    print("=" * 70)

    # Prepare data for LLM
    parameters = coeff_data["parameters"]
    nu_train = [p["nu"] for p in parameters]
    operator_names = list(parameters[0]["operators"].keys())

    # Create initial prompt (simplified - no need to pass complex data)
    if method == "regression":
        step_2 = f"2. `simple_linear_regress(nu_query)`: Get regressed operators (just pass nu_query={query_nu})"
        instruction_2 = f"2. Then, call simple_linear_regress({query_nu}) to get the operators"
    else:
        step_2 = f"2. `simple_interpolate(nu_query, method)`: Get interpolated operators (EASIEST - just pass nu_query={query_nu})"
        instruction_2 = f"2. Then, call simple_interpolate({query_nu}, \"linear\") to get the operators"

    initial_prompt = f"""You are an expert in reduced-order modeling and operator inference.

**Task**: Predict OpInf operators for a {equation_type} equation at parameter ν = {query_nu}

**Available Data** (already loaded in the system):
- Training parameters: ν = {nu_train}
- Operators at each ν: {operator_names}

**Available Tools** (USE THESE IN ORDER):
1. `analyze_parameter_range(nu_train, nu_query)`: Check if ν={query_nu} is interpolation or extrapolation
{step_2}
3. `validate_operators(operators, equation_type)`: Check physical constraints

**Instructions**:
1. First, call analyze_parameter_range({nu_train}, {query_nu}) to understand the problem
{instruction_2}
3. Finally, call validate_operators with the returned operators and equation_type="{equation_type}"

Please solve this step-by-step using the tools.
"""

    messages = [
        {"role": "system", "content": "You are an expert in scientific computing and reduced-order modeling. Use the provided tools to solve problems accurately."},
        {"role": "user", "content": initial_prompt}
    ]

    # Run conversation with tool calling
    max_iterations = 10
    iteration = 0
    final_result = None

    while iteration < max_iterations:
        iteration += 1
        print(f"\n--- Iteration {iteration} ---")

        response = call_llm_with_tools(provider, messages, TOOLS, model)
        assistant_message = response.choices[0].message

        # Add assistant message to conversation
        messages.append(assistant_message)

        # Check if LLM wants to call tools
        if assistant_message.tool_calls:
            print(f"LLM is calling {len(assistant_message.tool_calls)} tool(s)...")

            for tool_call in assistant_message.tool_calls:
                function_name = tool_call.function.name
                print(f"  → {function_name}")

                # Execute the tool
                result = execute_tool_call(tool_call)

                # Store result if it's interpolation
                if function_name in ["interpolate_operators", "simple_interpolate", "linear_regress_operators", "simple_linear_regress"]:
                    final_result = result

                # Add tool result to conversation
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": function_name,
                    "content": json.dumps(result)
                })
        else:
            # LLM is done, no more tool calls
            print("\nLLM Response:")
            print(assistant_message.content)
            break

    return final_result, messages


def run_tool_calling_workflow_batch(coeff_data, query_nus, equation_type="heat", provider="openai", model="gpt-4o", method="interpolation"):
    """
    Run the workflow once to compute operators for multiple nu values.
    """
    global _GLOBAL_COEFF_DATA
    _GLOBAL_COEFF_DATA = coeff_data

    parameters = coeff_data["parameters"]
    nu_train = [p["nu"] for p in parameters]
    operator_names = list(parameters[0]["operators"].keys())

    if method == "regression":
        step_2 = "2. `simple_linear_regress_batch(nu_queries)` to compute all operators in one call"
    else:
        step_2 = "2. `simple_interpolate_batch(nu_queries, method)` to compute all operators in one call"

    initial_prompt = f"""You are an expert in reduced-order modeling and operator inference.

**Task**: Predict OpInf operators for a {equation_type} equation at multiple parameter values.

**Queries**: ν = {query_nus}
**Training parameters**: ν = {nu_train}
**Operators**: {operator_names}

**Available Tools** (USE THESE IN ORDER):
1. `analyze_parameter_range(nu_train, nu_query)` for any borderline queries
{step_2}
3. `validate_operators(operators, equation_type)` if needed

Please solve this step-by-step using the tools.
"""

    messages = [
        {"role": "system", "content": "You are an expert in scientific computing and reduced-order modeling. Use the provided tools to solve problems accurately."},
        {"role": "user", "content": initial_prompt}
    ]

    max_iterations = 6
    for _ in range(max_iterations):
        response = call_llm_with_tools(provider, messages, TOOLS, model)
        assistant_message = response.choices[0].message
        messages.append(assistant_message)

        if assistant_message.tool_calls:
            for tool_call in assistant_message.tool_calls:
                function_name = tool_call.function.name
                result = execute_tool_call(tool_call)
                if function_name in ["simple_interpolate_batch", "simple_linear_regress_batch"]:
                    return result, messages
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": function_name,
                    "content": json.dumps(result),
                })
        else:
            break

    return None, messages


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="LLM tool calling for OpInf operator interpolation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use GPT-4 with function calling
  python3 llm_tool_calling_interpolation.py --query_nu 1.0

  # Specify different model
  python3 llm_tool_calling_interpolation.py --query_nu 1.0 --model gpt-4o-mini

  # Different equation
  python3 llm_tool_calling_interpolation.py --coefficients burgers_coefficients.json --query_nu 0.03

  # Use Gemini
  python3 llm_tool_calling_interpolation.py --query_nu 1.0 --provider gemini --model gemini-2.0-flash-exp

  # Batch mode
  python3 llm_tool_calling_interpolation.py --query_nu_values 0.5 1.0 3.0 --provider gemini --model gemini-2.0-flash-exp
"""
    )

    parser.add_argument("--coefficients", type=str, default="heat_coefficients.json",
                       help="Path to ROM coefficients JSON")
    parser.add_argument("--query_nu", type=float,
                       help="Parameter value to interpolate to")
    parser.add_argument("--query_nu_values", nargs="+", type=float,
                       help="List of parameter values to interpolate to")
    parser.add_argument("--provider", type=str, default="openai",
                        choices=["openai", "gemini", "deepseek", "anthropic", "qwen"],
                        help="LLM provider (default: openai)")
    parser.add_argument("--model", type=str, default="gpt-4o",
                       help="LLM model name (default: gpt-4o)")
    parser.add_argument("--method", type=str, default="interpolation",
                       choices=["interpolation", "regression"],
                       help="Operator prediction method (default: interpolation)")
    parser.add_argument("--output", type=str, default=None,
                       help="Output file for results")
    args = parser.parse_args()

    if args.query_nu is None and not args.query_nu_values:
        parser.error("Specify --query_nu or --query_nu_values.")

    if args.output is None:
        if args.query_nu_values:
            args.output = "tool_calling_operators_batch.json"
        else:
            args.output = f"tool_calling_operators_nu{args.query_nu}.json"

    print("=" * 70)
    print("LLM Tool Calling for OpInf Operator Interpolation")
    print("=" * 70)
    print(f"Coefficients: {args.coefficients}")
    if args.query_nu_values:
        print(f"Query ν values: {args.query_nu_values}")
    else:
        print(f"Query ν: {args.query_nu}")
    print(f"Provider: {args.provider}")
    print(f"Model: {args.model}")
    print(f"Output: {args.output}")

    # Load coefficient data
    print(f"\nLoading coefficients...")
    with open(args.coefficients, 'r') as f:
        coeff_data = json.load(f)

    equation_type = coeff_data.get("equation", "heat")
    nu_train = [p["nu"] for p in coeff_data["parameters"]]
    print(f"  Equation: {equation_type}")
    print(f"  Training ν: {nu_train}")

    # Run tool calling workflow
    try:
        if args.query_nu_values:
            result, conversation = run_tool_calling_workflow_batch(
                coeff_data, args.query_nu_values, equation_type, args.provider, args.model, args.method
            )
            if result is None:
                print("\n✗ LLM did not call simple_interpolate_batch")
                return

            predictions = result.get("predictions", {})
            for nu_query in args.query_nu_values:
                key = str(nu_query)
                pred = predictions.get(key)
                if pred is None:
                    # Be tolerant of integer/float string mismatches (e.g., "1" vs "1.0").
                    for alt_key, alt_pred in predictions.items():
                        try:
                            if float(alt_key) == float(nu_query):
                                pred = alt_pred
                                break
                        except (TypeError, ValueError):
                            continue
                if pred is None:
                    raise RuntimeError(f"Missing prediction for nu={nu_query}")
                operators_for_test = {}
                for op_name, op_data in pred["operators"].items():
                    operators_for_test[op_name] = op_data["values"]

                output_data = {
                    "query_nu": nu_query,
                    "predicted_operators": {
                        "nu": nu_query,
                        "operators": operators_for_test,
                        "method": pred["method"],
                    },
                    "provider": args.provider,
                    "model": args.model,
                    "equation_type": equation_type,
                    "conversation_length": len(conversation),
                }

                output_path = Path(args.output) if args.output else Path(f"tool_calling_operators_nu{nu_query}.json")
                if args.output and len(args.query_nu_values) > 1:
                    output_path = Path(args.output).parent / f"tool_calling_operators_nu{nu_query}.json"
                with open(output_path, 'w') as f:
                    json.dump(output_data, f, indent=2)
                print(f"✓ Results saved to: {output_path}")
        else:
            if args.query_nu is None:
                raise ValueError("Specify --query_nu or --query_nu_values.")
            result, conversation = run_tool_calling_workflow(
                coeff_data, args.query_nu, equation_type, args.provider, args.model, args.method
            )

            if result is None:
                print("\n✗ LLM did not call interpolate_operators")
                return

            # Save results in format compatible with test_llm_predicted_rom.py
            operators_for_test = {}
            for op_name, op_data in result["operators"].items():
                operators_for_test[op_name] = op_data["values"]

            output_data = {
                "query_nu": args.query_nu,
                "predicted_operators": {
                    "nu": args.query_nu,
                    "operators": operators_for_test,
                    "method": result["method"]
                },
                "provider": args.provider,
                "model": args.model,
                "equation_type": equation_type,
                "conversation_length": len(conversation)
            }

            with open(args.output, 'w') as f:
                json.dump(output_data, f, indent=2)

            print("\n" + "=" * 70)
            print(f"✓ Results saved to: {args.output}")
            print("=" * 70)
            print("\nSummary:")
            print(f"  Method used: {result['method']}")
            print(f"  Operators computed: {list(result['operators'].keys())}")
            for op_name, op_data in result['operators'].items():
                print(f"    {op_name}: shape {op_data['shape']}, norm={op_data['norm']:.4e}")

            print("\nNext steps:")
            print(f"  python3 test_llm_predicted_rom.py --predicted {args.output}")
            print("  Expected error: ~0.4% (same as numerical interpolation)")

    except Exception as e:
        print(f"\n✗ Error: {e}")
        print("\nMake sure the provider API key is set:")
        print("  OPENAI_API_KEY (openai)")
        print("  GEMINI_API_KEY (gemini)")
        print("  DEEPSEEK_API_KEY (deepseek)")
        print("  ANTHROPIC_API_KEY (anthropic)")
        print("  QWEN_API_KEY (qwen)")
        raise


if __name__ == "__main__":
    main()
