# Pure LLM Baseline Backup

This folder is a **copy-only backup** of the files needed to run the
pure-LLM baseline (no operator inference / ROM). The originals remain in the
repo root; these are duplicates for convenience.

## Contents

Code:
- `run_pure_llm_baseline_downsampled.py` (main runner)
- `llm_tool_calling_provider.py` (LLM API wrapper)
- `load_env.py` (API key loading)

Datasets:
- `heat_dataset_test.pkl.gz`
- `burgers_dataset_test.pkl.gz`
- `cavity_dataset_test.pkl.gz`

Prompts (already generated):
- `pure_llm_baseline_prompts/`
- `pure_llm_baseline_prompts_gpt-4.1/`
- `pure_llm_baseline_prompts_gpt-4o/`
- `pure_llm_baseline_prompts_gemini-2.0-flash-exp/`

## How to run (from repo root)

```
LLM_REQUEST_TIMEOUT=120 python llm/run_pure_llm_baseline_downsampled.py   --provider openai --model_name gpt-4.1   --execute --save_plots   --retry_until_success --max_attempts_per_case 10 --retry_on_validation_failure
```

Results will be written to `pure_llm_baseline_results_<model>/` in the repo root.

## Notes
- This method directly prompts the LLM for full spatiotemporal fields on a
  coarse grid (Heat/Burgers: 16×41, Cavity: 6×6×21).
- Outputs are compared to downsampled FOM data using relative L2 error.
