#!/usr/bin/env python3
"""
Ablation: static parser vs LLM parser on diverse natural-language instructions.

Goal:
- Show whether a naive static parser can robustly interpret diverse/unstructured prompts.
- Compare against the existing LLM parser used in run_three_equations_workflow_nl.py.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nl_task_parser import normalize_config, parse_prompt_with_llm


EQUATION_CHOICES = {"heat", "burgers", "cavity"}
DEFAULT_PROVIDER = "openai"
DEFAULT_MODEL = "gpt-4o"
DEFAULT_OUTPUT_BASE = "ablation_nl_parser_runs"


@dataclass
class Case:
    case_id: str
    prompt: str
    expected: dict[str, Any]


CASES: list[Case] = [
    Case(
        "c01",
        "Can you run only heat and burgers with OpenAI gpt-4o, save raw outputs, and reuse operators?",
        {"equations": ["heat", "burgers"], "provider": "openai", "save_raw": True, "reuse_operators": True},
    ),
    Case(
        "c02",
        "Need cavity only. Use gemini with gemini-2.0-flash-exp. Don't save raw data.",
        {"equations": ["cavity"], "provider": "gemini", "save_raw": False},
    ),
    Case(
        "c03",
        "all 3 PDEs please. deepseek chat model. put results under runs/expA and don't reuse operators",
        {"equations": ["heat", "burgers", "cavity"], "provider": "deepseek", "reuse_operators": False, "output_dir": "runs/expA"},
    ),
    Case(
        "c04",
        "just burgers @ nu 0.03 0.07 0.12, openai gpt-4.1, raw on",
        {"equations": ["burgers"], "provider": "openai", "burgers_nus": [0.03, 0.07, 0.12], "save_raw": True},
    ),
    Case(
        "c05",
        "heat at nu = 0.5, 1.0, 3.0; reuse pls; output folder should be tmp/heat_only",
        {"equations": ["heat"], "heat_nus": [0.5, 1.0, 3.0], "reuse_operators": True, "output_dir": "tmp/heat_only"},
    ),
    Case(
        "c06",
        "Cavity test on Re 60 80 90 110 120 140 with anthropic claude, save raw",
        {"equations": ["cavity"], "provider": "anthropic", "cavity_res": [60, 80, 90, 110, 120, 140], "save_raw": True},
    ),
    Case(
        "c07",
        "pls do heat+burgers+cavity, qwen, no raw, yes reuse",
        {"equations": ["heat", "burgers", "cavity"], "provider": "qwen", "save_raw": False, "reuse_operators": True},
    ),
    Case(
        "c08",
        "I only care about 2D cavity flow. maybe openai is fine",
        {"equations": ["cavity"], "provider": "openai"},
    ),
    Case(
        "c09",
        "run all equations but for heat use nus 0.2 and 2.5. save RAW outputs.",
        {"equations": ["heat", "burgers", "cavity"], "heat_nus": [0.2, 2.5], "save_raw": True},
    ),
    Case(
        "c10",
        "burgers + heat, no cavity, provider gemini. output dir = ablations/nlp_10",
        {"equations": ["heat", "burgers"], "provider": "gemini", "output_dir": "ablations/nlp_10"},
    ),
    Case(
        "c11",
        "all equations. keep operators fresh every run.",
        {"equations": ["heat", "burgers", "cavity"], "reuse_operators": False},
    ),
    Case(
        "c12",
        "all equations. operator reuse ON.",
        {"equations": ["heat", "burgers", "cavity"], "reuse_operators": True},
    ),
    Case(
        "c13",
        "for cavity use Re=75,95,135 only; deepseek",
        {"equations": ["cavity"], "provider": "deepseek", "cavity_res": [75, 95, 135]},
    ),
    Case(
        "c14",
        "heat only, model gpt-4o-mini (if available), save raw false",
        {"equations": ["heat"], "save_raw": False},
    ),
    Case(
        "c15",
        "Need Burgers equation extrapolation-ish at nu 0.2 and 0.5 with OpenAI.",
        {"equations": ["burgers"], "provider": "openai", "burgers_nus": [0.2, 0.5]},
    ),
    Case(
        "c16",
        "Let's do hEat and burGERs (typos intentional) and dump in runs/noisy_case",
        {"equations": ["heat", "burgers"], "output_dir": "runs/noisy_case"},
    ),
    Case(
        "c17",
        "cavity + heat. Re values 100, 150. heat nus 0.1 0.9",
        {"equations": ["cavity", "heat"], "cavity_res": [100, 150], "heat_nus": [0.1, 0.9]},
    ),
    Case(
        "c18",
        "Just do everything with Anthropic and keep raw snapshots.",
        {"equations": ["heat", "burgers", "cavity"], "provider": "anthropic", "save_raw": True},
    ),
    Case(
        "c19",
        "OpenAI, all pdes, but don't save raw and don't reuse",
        {"equations": ["heat", "burgers", "cavity"], "provider": "openai", "save_raw": False, "reuse_operators": False},
    ),
    Case(
        "c20",
        "I want quick run: burgers only, nu=0.03, output quick/burg",
        {"equations": ["burgers"], "burgers_nus": [0.03], "output_dir": "quick/burg"},
    ),
    # Harder prompts: implicit physics language, weak structure, mixed constraints.
    Case(
        "c21",
        "Run the lid-driven vortex benchmark in the square cavity flow setting for Reynolds 80 and 120, keep raw snapshots.",
        {"equations": ["cavity"], "cavity_res": [80, 120], "save_raw": True},
    ),
    Case(
        "c22",
        "I need vorticity-streamfunction ROM checks with Re=60,90,140; reuse operators and use openai.",
        {"equations": ["cavity"], "cavity_res": [60, 90, 140], "provider": "openai", "reuse_operators": True},
    ),
    Case(
        "c23",
        "Do the 1D thermal diffusion setting (heat-like) with diffusivity nu 0.2 and 2.0. no raw dumps.",
        {"equations": ["heat"], "heat_nus": [0.2, 2.0], "save_raw": False},
    ),
    Case(
        "c24",
        "Please evaluate Burgers-like shock-steepening advection-diffusion behavior at nu=0.03 and 0.12, output to runs/shockcheck.",
        {"equations": ["burgers"], "burgers_nus": [0.03, 0.12], "output_dir": "runs/shockcheck"},
    ),
    Case(
        "c25",
        "Use Fourier-like conduction case plus the nonlinear convective transport one (Burgers-style); skip cavity.",
        {"equations": ["heat", "burgers"]},
    ),
    Case(
        "c26",
        "For lid velocity forcing study, test Re 75 95 135 with anthropic and keep operators fresh.",
        {"equations": ["cavity"], "cavity_res": [75, 95, 135], "provider": "anthropic", "reuse_operators": False},
    ),
    Case(
        "c27",
        "For the Burgers-style ROM with quadratic modal tensor term H(a,a), run nu 0.07 and 0.2.",
        {"equations": ["burgers"], "burgers_nus": [0.07, 0.2]},
    ),
    Case(
        "c28",
        "No explicit equation labels: do square-box recirculation (cavity flow) at Re=100,150 and thermal diffusion (heat-like) at nu=0.5.",
        {"equations": ["cavity", "heat"], "cavity_res": [100, 150], "heat_nus": [0.5]},
    ),
    Case(
        "c29",
        "Need boundary-control scalar input PDE only, nus: 0.1, 1.0, 3.0, provider gemini.",
        {"equations": ["heat"], "heat_nus": [0.1, 1.0, 3.0], "provider": "gemini"},
    ),
    Case(
        "c30",
        "Use the three-input boundary/source forcing Burgers-style parametric case; test nu 0.03, 0.07, 0.12 and save raw.",
        {"equations": ["burgers"], "burgers_nus": [0.03, 0.07, 0.12], "save_raw": True},
    ),
    Case(
        "c31",
        "Run all physics families but only specify: Re=80,120 and heat nu 0.2. don't save raw.",
        {"equations": ["heat", "burgers", "cavity"], "cavity_res": [80, 120], "heat_nus": [0.2], "save_raw": False},
    ),
    Case(
        "c32",
        "I want incompressible cavity benchmark and Burgers-style nonlinear transport, qwen backend, reuse on.",
        {"equations": ["burgers", "cavity"], "provider": "qwen", "reuse_operators": True},
    ),
    Case(
        "c33",
        "Please do conduction (heat-like) + cavity flow, place outputs in ablations/implicit_phys_33, raw true.",
        {"equations": ["heat", "cavity"], "output_dir": "ablations/implicit_phys_33", "save_raw": True},
    ),
    Case(
        "c34",
        "Skip the vortex case. evaluate only diffusion and nonlinear advection with deepseek chat.",
        {"equations": ["heat", "burgers"], "provider": "deepseek"},
    ),
    Case(
        "c35",
        "I care only about the square-cavity moving-lid flow benchmark; Reynolds sweep 60 80 90 110 120 140. output folder experiments/re_sweep.",
        {"equations": ["cavity"], "cavity_res": [60, 80, 90, 110, 120, 140], "output_dir": "experiments/re_sweep"},
    ),
    Case(
        "c36",
        "Nonlinear convective Burgers-style PDE with 3-component forcing at nu 0.05, and don't reuse operators.",
        {"equations": ["burgers"], "burgers_nus": [0.05], "reuse_operators": False},
    ),
    Case(
        "c37",
        "Thermal case + cavity case; openai gpt-4.1; save raw; output should be paper_runs/c37",
        {"equations": ["heat", "cavity"], "provider": "openai", "model_name": "gpt-4.1", "save_raw": True, "output_dir": "paper_runs/c37"},
    ),
    Case(
        "c38",
        "Do all equations, but for the lid-driven one only Re=100 and for heat-like only nu=1.0",
        {"equations": ["heat", "burgers", "cavity"], "cavity_res": [100], "heat_nus": [1.0]},
    ),
    Case(
        "c39",
        "Just the heat-like PDE with A,B,C linear modal operators and scalar boundary input. no raw.",
        {"equations": ["heat"], "save_raw": False},
    ),
    Case(
        "c40",
        "Run recirculating square-flow and shock-forming transport together, use anthropic, reuse operators.",
        {"equations": ["burgers", "cavity"], "provider": "anthropic", "reuse_operators": True},
    ),
    # Additional 10 cases: non-technical, application-style, single equation each.
    Case(
        "c41",
        "I want to study how temperature moves through a metal rod in a cooling line, with diffusion rates 0.1 and 0.5.",
        {"equations": ["heat"], "heat_nus": [0.1, 0.5]},
    ),
    Case(
        "c42",
        "For a room-air circulation benchmark in a square box with a moving top boundary, test Reynolds 80 and 120.",
        {"equations": ["cavity"], "cavity_res": [80, 120]},
    ),
    Case(
        "c43",
        "Run the traffic-wave style transport case where sharp fronts develop, at viscosity 0.03 and 0.07.",
        {"equations": ["burgers"], "burgers_nus": [0.03, 0.07]},
    ),
    Case(
        "c44",
        "Need a thermal equalization study for a heated bar with spread settings 0.2 and 2.0, and save raw snapshots.",
        {"equations": ["heat"], "heat_nus": [0.2, 2.0], "save_raw": True},
    ),
    Case(
        "c45",
        "Please do only the lid-driven recirculating flow benchmark at Reynolds 60, 90, and 140.",
        {"equations": ["cavity"], "cavity_res": [60, 90, 140]},
    ),
    Case(
        "c46",
        "Evaluate the nonlinear wave-steepening transport setup with three forcing channels at 0.03, 0.07, 0.12.",
        {"equations": ["burgers"], "burgers_nus": [0.03, 0.07, 0.12]},
    ),
    Case(
        "c47",
        "I need a battery-pack heat spreading scenario only, with diffusivities 0.1, 0.9, and 3.0.",
        {"equations": ["heat"], "heat_nus": [0.1, 0.9, 3.0]},
    ),
    Case(
        "c48",
        "Run a mixing-vortex benchmark in a square cavity for Reynolds 40, 100, and 160, no raw dumps.",
        {"equations": ["cavity"], "cavity_res": [40, 100, 160], "save_raw": False},
    ),
    Case(
        "c49",
        "For the single-field wave-transport case where fronts steepen, test viscosity values 0.05 and 0.2.",
        {"equations": ["burgers"], "burgers_nus": [0.05, 0.2]},
    ),
    Case(
        "c50",
        "Please run a one-dimensional thermal diffusion check for process control tuning at 0.5 and 3.0.",
        {"equations": ["heat"], "heat_nus": [0.5, 3.0]},
    ),
]


def _parse_bool_from_text(text: str, true_words: list[str], false_words: list[str]) -> bool | None:
    t = text.lower()
    for w in false_words:
        if w in t:
            return False
    for w in true_words:
        if w in t:
            return True
    return None


def _extract_number_list(text: str, key_patterns: list[str]) -> list[float] | None:
    tl = text.lower()
    for pat in key_patterns:
        m = re.search(pat, tl)
        if not m:
            continue
        seg = m.group(1)
        vals = re.findall(r"[-+]?\d*\.?\d+", seg)
        if vals:
            return [float(v) for v in vals]
    return None


def static_parse_prompt(prompt: str) -> dict[str, Any]:
    t = prompt.lower()
    parsed: dict[str, Any] = {}

    # Equations (naive keyword rules).
    eqs = set()
    if "heat" in t:
        eqs.add("heat")
    if "burgers" in t or "burger" in t:
        eqs.add("burgers")
    if "cavity" in t:
        eqs.add("cavity")
    if "all" in t or "everything" in t or "all pde" in t or "all equations" in t:
        eqs = set(EQUATION_CHOICES)
    if "no cavity" in t or "without cavity" in t:
        eqs.discard("cavity")
    if eqs:
        parsed["equations"] = sorted(eqs)

    # Provider.
    for provider in ["openai", "gemini", "deepseek", "anthropic", "qwen"]:
        if provider in t:
            parsed["provider"] = provider
            break

    # Model name (simple token extraction).
    model_match = re.search(
        r"(gpt-[\w\.\-]+|gemini-[\w\.\-]+|claude-[\w\.\-]+|deepseek-[\w\.\-]+|qwen[\w\.\-]*)",
        t,
    )
    if model_match:
        parsed["model_name"] = model_match.group(1)

    # Output dir.
    out_match = re.search(
        r"(?:output(?:\s+dir|\s+folder)?\s*(?:=|should be|is)?\s*)([a-zA-Z0-9_./\-]+)",
        t,
    )
    if out_match:
        parsed["output_dir"] = out_match.group(1)

    # save_raw / reuse_operators.
    save_raw = _parse_bool_from_text(
        t,
        true_words=["save raw", "raw on", "keep raw", "raw outputs", "save_raw"],
        false_words=["don't save raw", "do not save raw", "save raw false", "raw off", "no raw"],
    )
    if save_raw is not None:
        parsed["save_raw"] = save_raw

    reuse = _parse_bool_from_text(
        t,
        true_words=["reuse operators", "reuse on", "yes reuse", "reuse pls", "operator reuse on"],
        false_words=["don't reuse", "do not reuse", "fresh every run", "keep operators fresh", "reuse off"],
    )
    if reuse is not None:
        parsed["reuse_operators"] = reuse

    # Parameter lists.
    heat_nus = _extract_number_list(
        t,
        key_patterns=[
            r"heat[^.]*?nu(?:s)?\s*[:=]?\s*([0-9.,\s\-and]+)",
            r"heat[^.]*?at\s+nu\s*[:=]?\s*([0-9.,\s\-and]+)",
        ],
    )
    if heat_nus:
        parsed["heat_nus"] = heat_nus

    burgers_nus = _extract_number_list(
        t,
        key_patterns=[
            r"burgers?[^.]*?nu(?:s)?\s*[:=]?\s*([0-9.,\s\-and]+)",
            r"burgers?[^.]*?at\s+nu\s*[:=]?\s*([0-9.,\s\-and]+)",
        ],
    )
    if burgers_nus:
        parsed["burgers_nus"] = burgers_nus

    cavity_res = _extract_number_list(
        t,
        key_patterns=[
            r"cavity[^.]*?re(?:s)?\s*[:=]?\s*([0-9.,\s\-and]+)",
            r"re\s*[:=]?\s*([0-9.,\s\-and]+)",
        ],
    )
    if cavity_res and ("cavity" in t or "re" in t):
        parsed["cavity_res"] = cavity_res

    return parsed


def _normalize(parsed: dict[str, Any], idx: int) -> dict[str, Any]:
    return normalize_config(
        parsed,
        default_provider=DEFAULT_PROVIDER,
        default_model=DEFAULT_MODEL,
        output_dir_base=DEFAULT_OUTPUT_BASE,
        index=idx + 1,
        default_save_raw=False,
        default_reuse=False,
    )


def _numbers_match(a: list[float], b: list[float], tol: float = 1e-8) -> bool:
    if len(a) != len(b):
        return False
    a2 = sorted([float(x) for x in a])
    b2 = sorted([float(x) for x in b])
    return all(abs(x - y) <= tol for x, y in zip(a2, b2))


def score_case(pred: dict[str, Any], expected: dict[str, Any]) -> dict[str, Any]:
    field_ok: dict[str, bool] = {}
    for key, exp in expected.items():
        got = pred.get(key)
        if key in {"equations"}:
            ok = sorted(got or []) == sorted(exp or [])
        elif key in {"heat_nus", "burgers_nus", "cavity_res"}:
            ok = _numbers_match(got or [], exp or [])
        elif key in {"save_raw", "reuse_operators"}:
            ok = bool(got) == bool(exp)
        else:
            ok = got == exp
        field_ok[key] = ok

    return {
        "n_fields": len(expected),
        "n_correct_fields": sum(1 for v in field_ok.values() if v),
        "all_correct": all(field_ok.values()) if expected else True,
        "field_ok": field_ok,
    }


def run_ablation(
    mode: str,
    parser_provider: str,
    parser_model: str,
    max_cases: int | None,
) -> dict[str, Any]:
    selected = CASES[: max_cases or len(CASES)]
    results: dict[str, Any] = {"mode": mode, "n_cases": len(selected), "cases": []}

    engines = []
    if mode in {"static", "both"}:
        engines.append("static")
    if mode in {"llm", "both"}:
        engines.append("llm")

    agg = {
        engine: {"n_cases": 0, "n_exact": 0, "n_fields": 0, "n_correct_fields": 0}
        for engine in engines
    }

    for idx, case in enumerate(selected):
        case_out = {"case_id": case.case_id, "prompt": case.prompt, "expected": case.expected}

        for engine in engines:
            if engine == "static":
                parsed = static_parse_prompt(case.prompt)
            else:
                parsed = parse_prompt_with_llm(case.prompt, parser_provider, parser_model)
            norm = _normalize(parsed, idx)
            score = score_case(norm, case.expected)

            agg[engine]["n_cases"] += 1
            agg[engine]["n_exact"] += int(score["all_correct"])
            agg[engine]["n_fields"] += score["n_fields"]
            agg[engine]["n_correct_fields"] += score["n_correct_fields"]

            case_out[engine] = {
                "parsed_raw": parsed,
                "normalized": norm,
                "score": score,
            }

        results["cases"].append(case_out)

    summary = {}
    for engine in engines:
        s = agg[engine]
        summary[engine] = {
            "exact_match_rate": (s["n_exact"] / s["n_cases"]) if s["n_cases"] else None,
            "field_accuracy": (s["n_correct_fields"] / s["n_fields"]) if s["n_fields"] else None,
            "n_exact": s["n_exact"],
            "n_cases": s["n_cases"],
            "n_correct_fields": s["n_correct_fields"],
            "n_fields": s["n_fields"],
        }
    results["summary"] = summary
    return results


def print_brief(results: dict[str, Any], show_failures: bool) -> None:
    print("\n=== NL Parsing Diversity Ablation ===")
    for engine, stats in results["summary"].items():
        print(
            f"[{engine}] exact_match={stats['exact_match_rate']:.3f} "
            f"({stats['n_exact']}/{stats['n_cases']}), "
            f"field_accuracy={stats['field_accuracy']:.3f} "
            f"({stats['n_correct_fields']}/{stats['n_fields']})"
        )
    if not show_failures:
        return

    for case in results["cases"]:
        for engine in ["static", "llm"]:
            if engine not in case:
                continue
            sc = case[engine]["score"]
            if sc["all_correct"]:
                continue
            failed = [k for k, ok in sc["field_ok"].items() if not ok]
            print(f"\n- {engine} failed {case['case_id']}: fields={failed}")
            print(f"  prompt: {case['prompt']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ablation: static vs LLM NL parser robustness.")
    parser.add_argument("--mode", choices=["static", "llm", "both"], default="both")
    parser.add_argument("--parser_provider", default="openai")
    parser.add_argument("--parser_model", default="gpt-4o")
    parser.add_argument("--max_cases", type=int, default=None)
    parser.add_argument("--output", default="ablation_nl_parser_diversity_results.json")
    parser.add_argument("--show_failures", action="store_true")
    args = parser.parse_args()

    results = run_ablation(
        mode=args.mode,
        parser_provider=args.parser_provider,
        parser_model=args.parser_model,
        max_cases=args.max_cases,
    )
    Path(args.output).write_text(json.dumps(results, indent=2))
    print(f"Saved results to: {args.output}")
    print_brief(results, show_failures=args.show_failures)


if __name__ == "__main__":
    main()
