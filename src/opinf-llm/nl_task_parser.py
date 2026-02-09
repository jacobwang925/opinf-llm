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
        "heat_nus": parsed.get("heat_nus"),
        "burgers_nus": parsed.get("burgers_nus"),
        "cavity_res": parsed.get("cavity_res"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse NL prompts into workflow configs.")
    parser.add_argument("--prompt", type=str, help="Single prompt string")
    parser.add_argument("--prompts_file", type=str, help="Path to prompts file (one per line)")
    parser.add_argument("--parser_provider", type=str, default="openai",
                        choices=["openai", "gemini", "deepseek", "anthropic", "qwen"])
    parser.add_argument("--parser_model", type=str, default=None,
                        help="Model for parsing (defaults per provider)")
    parser.add_argument("--default_provider", type=str, default="openai",
                        choices=["openai", "gemini", "deepseek", "anthropic", "qwen"])
    parser.add_argument("--default_model", type=str, default=None,
                        help="Default workflow model (overrides provider default)")
    parser.add_argument("--output_dir_base", type=str, default="nl_workflow_runs")
    parser.add_argument("--save_raw", action="store_true", help="Default save_raw if not parsed")
    parser.add_argument("--reuse_operators", action="store_true", help="Default reuse_operators if not parsed")
    parser.add_argument("--execute", action="store_true", help="Execute workflow for each prompt")
    parser.add_argument("--merge_prompts", action="store_true",
                        help="Merge parsed prompts into a single workflow run")
    args = parser.parse_args()

    parser_model = args.parser_model or DEFAULT_MODEL_BY_PROVIDER[args.parser_provider]

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
        parsed = parse_prompt_with_llm(prompt, args.parser_provider, parser_model)
        config = normalize_config(
            parsed,
            default_provider=args.default_provider,
            default_model=args.default_model,
            output_dir_base=args.output_dir_base,
            index=idx,
            default_save_raw=args.save_raw,
            default_reuse=args.reuse_operators,
        )
        if not str(config["output_dir"]).endswith("_parsed"):
            config["output_dir"] = f"{config['output_dir']}_parsed"
        configs.append(config)

    if args.merge_prompts and configs:
        merged = {
            "equations": sorted({e for c in configs for e in c.get("equations", [])}),
            "provider": configs[0]["provider"],
            "model_name": configs[0]["model_name"],
            "output_dir": f"{args.output_dir_base}_parsed",
            "save_raw": any(c.get("save_raw") for c in configs),
            "reuse_operators": any(c.get("reuse_operators") for c in configs),
            "heat_nus": sorted({v for c in configs for v in (c.get("heat_nus") or [])}),
            "burgers_nus": sorted({v for c in configs for v in (c.get("burgers_nus") or [])}),
            "cavity_res": sorted({v for c in configs for v in (c.get("cavity_res") or [])}),
        }
        configs = [merged]

    print(json.dumps(configs, indent=2))

    if not args.execute:
        return

    for config in configs:
        cmd = [
            sys.executable,
            "run_three_equations_workflow.py",
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
