#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parse natural-language prompts into structured workflow configs.

This script uses an LLM to convert prompts into a JSON config, then (optionally)
invokes run_three_equations_workflow.py with the parsed settings.
"""

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

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

EQUATION_CHOICES = {"heat", "burgers", "cavity"}
TEST_HEAT_NUS = [0.5, 1.0, 3.0]
TEST_BURGERS_NUS = [0.03, 0.07]
TEST_CAVITY_RES = [60.0, 80.0, 90.0, 110.0, 120.0, 140.0]


def _parse_json_from_text(text: Any) -> Optional[Dict[str, Any]]:
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


def parse_prompt_with_llm(prompt: str, provider: str, model: str) -> Dict[str, Any]:
    schema = {
        "equations": ["heat", "burgers", "cavity"],
        "provider": "openai|gemini|deepseek|anthropic|qwen",
        "model_name": "string or null",
        "output_dir": "string or null",
        "save_raw": "true/false",
        "reuse_operators": "true/false",
        "heat_nus": "list of numbers or null",
        "burgers_nus": "list of numbers or null",
        "cavity_res": "list of numbers or null",
    }
    system = (
        "You are a strict parser. Return JSON only. Do not include code fences."
    )
    user = (
        "Parse the prompt into the following JSON keys. Use lowercase for equations.\n"
        f"Schema: {json.dumps(schema)}\n"
        f"Prompt: {prompt}"
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    text = call_llm_text(provider, messages, model)
    parsed = _parse_json_from_text(text) or {}
    return parsed


def normalize_config(
    parsed: Dict[str, Any],
    default_provider: str,
    default_model: Optional[str],
    output_dir_base: str,
    index: int,
    default_save_raw: bool,
    default_reuse: bool,
) -> Dict[str, Any]:
    def _coerce_test_list(values: Any, allowed: List[float]) -> List[float]:
        if values is None:
            return list(allowed)
        if isinstance(values, (int, float)):
            values = [values]
        if isinstance(values, str):
            values = [values]
        if not isinstance(values, list):
            return list(allowed)
        out: List[float] = []
        for v in values:
            try:
                fv = float(v)
            except (TypeError, ValueError):
                continue
            if any(abs(fv - a) < 1e-12 for a in allowed):
                out.append(fv)
        if not out:
            return list(allowed)
        # unique + stable sorted order
        uniq = sorted({float(v) for v in out})
        return uniq

    def _coerce_bool(value: Any) -> Optional[bool]:
        if value is None:
            return None
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"true", "yes", "y", "1"}:
                return True
            if lowered in {"false", "no", "n", "0"}:
                return False
        return None
    equations = parsed.get("equations") or ["heat", "burgers", "cavity"]
    if isinstance(equations, str):
        equations = [equations]
    equations = [e.lower().strip() for e in equations if e]
    equations = [e for e in equations if e in EQUATION_CHOICES]
    if not equations:
        equations = ["heat", "burgers", "cavity"]

    provider = (parsed.get("provider") or default_provider or "openai").lower()
    model_name = parsed.get("model_name") or default_model or DEFAULT_MODEL_BY_PROVIDER.get(provider)
    output_dir = parsed.get("output_dir") or os.path.join(output_dir_base, f"prompt_{index:02d}")
    save_raw = _coerce_bool(parsed.get("save_raw"))
    if save_raw is None:
        save_raw = default_save_raw
    reuse_operators = _coerce_bool(parsed.get("reuse_operators"))
    if reuse_operators is None:
        reuse_operators = default_reuse

    return {
        "equations": equations,
        "provider": provider,
        "model_name": model_name,
        "output_dir": output_dir,
        "save_raw": bool(save_raw),
        "reuse_operators": bool(reuse_operators),
        "heat_nus": _coerce_test_list(parsed.get("heat_nus"), TEST_HEAT_NUS),
        "burgers_nus": _coerce_test_list(parsed.get("burgers_nus"), TEST_BURGERS_NUS),
        "cavity_res": _coerce_test_list(parsed.get("cavity_res"), TEST_CAVITY_RES),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse NL prompts into workflow configs.")
    parser.add_argument("--prompt", type=str, help="Single prompt string")
    parser.add_argument("--prompts_file", type=str, help="Path to prompts file (one per line)")
    parser.add_argument("--provider", type=str, default=None,
                        choices=["openai", "gemini", "deepseek", "anthropic", "qwen"],
                        help="Unified provider for both parser and workflow.")
    parser.add_argument("--model_name", type=str, default=None,
                        help="Unified model name for both parser and workflow.")
    parser.add_argument("--parser_provider", type=str, default="openai",
                        choices=["openai", "gemini", "deepseek", "anthropic", "qwen"])
    parser.add_argument("--parser_model", type=str, default=None,
                        help="Model for parsing (defaults per provider)")
    parser.add_argument("--default_provider", type=str, default="openai",
                        choices=["openai", "gemini", "deepseek", "anthropic", "qwen"])
    parser.add_argument("--default_model", type=str, default=None,
                        help="Default workflow model (overrides provider default)")
    parser.add_argument("--output_dir_base", type=str, default="nl_runs")
    parser.add_argument("--save_raw", dest="save_raw_override", action="store_true",
                        help="Force save_raw=True in workflow runs.")
    parser.add_argument("--reuse_operators", action="store_true", help="Default reuse_operators if not parsed")
    parser.add_argument("--execute", dest="execute", action="store_true",
                        help="Execute workflow for parsed config(s). Default: enabled.")
    parser.add_argument("--no_execute", dest="execute", action="store_false",
                        help="Do not execute workflow; only print parsed config(s).")
    parser.add_argument("--merge_prompts", dest="merge_prompts", action="store_true",
                        help="Merge parsed prompts into a single workflow run. Default: enabled.")
    parser.add_argument("--no_merge_prompts", dest="merge_prompts", action="store_false",
                        help="Do not merge prompts; run one config per prompt.")
    parser.set_defaults(execute=True, merge_prompts=True, save_raw_override=None)
    args = parser.parse_args()

    parser_provider = args.provider or args.parser_provider
    default_provider = args.provider or args.default_provider
    parser_model = args.model_name or args.parser_model or DEFAULT_MODEL_BY_PROVIDER[parser_provider]
    default_model = args.model_name or args.default_model or DEFAULT_MODEL_BY_PROVIDER[default_provider]

    prompts: List[str] = []
    if args.prompt:
        prompts.append(args.prompt)
    if args.prompts_file:
        for line in Path(args.prompts_file).read_text().splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            prompts.append(stripped)
    if not prompts:
        default_file = Path("synthetic_prompts.txt")
        if default_file.exists():
            for line in default_file.read_text().splitlines():
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                prompts.append(stripped)
        else:
            raise SystemExit("Provide --prompt or --prompts_file.")

    configs = []
    for idx, prompt in enumerate(prompts, start=1):
        parsed = parse_prompt_with_llm(prompt, parser_provider, parser_model)
        config = normalize_config(
            parsed,
            default_provider=default_provider,
            default_model=default_model,
            output_dir_base=args.output_dir_base,
            index=idx,
            default_save_raw=bool(args.save_raw_override) if args.save_raw_override is not None else False,
            default_reuse=args.reuse_operators,
        )
        if args.save_raw_override is not None:
            config["save_raw"] = bool(args.save_raw_override)
        configs.append(config)

    if args.merge_prompts and configs:
        merged_heat = sorted({
            float(v) for c in configs for v in (c.get("heat_nus") or [])
            if any(abs(float(v) - a) < 1e-12 for a in TEST_HEAT_NUS)
        }) or list(TEST_HEAT_NUS)
        merged_burgers = sorted({
            float(v) for c in configs for v in (c.get("burgers_nus") or [])
            if any(abs(float(v) - a) < 1e-12 for a in TEST_BURGERS_NUS)
        }) or list(TEST_BURGERS_NUS)
        merged_cavity = sorted({
            float(v) for c in configs for v in (c.get("cavity_res") or [])
            if any(abs(float(v) - a) < 1e-12 for a in TEST_CAVITY_RES)
        }) or list(TEST_CAVITY_RES)
        merged = {
            "equations": sorted({e for c in configs for e in c.get("equations", [])}),
            "provider": configs[0]["provider"],
            "model_name": configs[0]["model_name"],
            "output_dir": args.output_dir_base,
            "save_raw": bool(args.save_raw_override) if args.save_raw_override is not None else any(c.get("save_raw") for c in configs),
            "reuse_operators": any(c.get("reuse_operators") for c in configs),
            "heat_nus": merged_heat,
            "burgers_nus": merged_burgers,
            "cavity_res": merged_cavity,
        }
        configs = [merged]

    print(json.dumps(configs, indent=2))

    if not args.execute:
        return

    for config in configs:
        cmd = [
            sys.executable,
            "src/run_three_equations_workflow.py",
            "--provider",
            config["provider"],
            "--model_name",
            config["model_name"],
            "--output_dir",
            config["output_dir"],
            "--equations",
            *config["equations"],
        ]
        if config.get("heat_nus"):
            cmd += ["--heat_nus", *[str(v) for v in config["heat_nus"]]]
        if config.get("burgers_nus"):
            cmd += ["--burgers_nus", *[str(v) for v in config["burgers_nus"]]]
        if config.get("cavity_res"):
            cmd += ["--cavity_res", *[str(v) for v in config["cavity_res"]]]
        if config["save_raw"]:
            cmd.append("--save_raw")
        if config["reuse_operators"]:
            cmd.append("--reuse_operators")
        print("\n==> Running:", " ".join(cmd))
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
