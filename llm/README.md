# Pure LLM Baseline

This folder contains the scripts and prompt assets for the direct pure-LLM
baseline.

## Contents

Code:
- `run_pure_llm_baseline_downsampled.py` (main runner)
- `llm_tool_calling_provider.py` (LLM API wrapper)
- `load_env.py` (API key loading)

Prompts:
- `pure_llm_baseline_prompts/`
- `pure_llm_baseline_prompts_gpt-4.1/`
- `pure_llm_baseline_prompts_gpt-4o/`
- `pure_llm_baseline_prompts_gemini-2.0-flash-exp/`

## How to run (from repo root)

```bash
LLM_REQUEST_TIMEOUT=120 python llm/run_pure_llm_baseline_downsampled.py \
  --provider openai --model_name gpt-4.1 \
  --execute --save_plots \
  --retry_until_success --max_attempts_per_case 10 \
  --retry_on_validation_failure
```

Default output location:
- `llm_runs/<model_name>/<run_id>/`

Optional overrides:
- `--output_dir <path>`
- `--output_dir_base <base_dir>`

## Notes
- This baseline prompts the LLM for full spatiotemporal fields on a coarse grid:
  - Heat/Burgers: `16 x 41`
  - Cavity: `6 x 6 x 21`
- Outputs are compared to downsampled FOM data using relative L2 error.
