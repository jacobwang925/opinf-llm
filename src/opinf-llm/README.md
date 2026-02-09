# opinf-llm (runtime)

This folder contains the code and artifacts needed to run the **OpInf-LLM** workflows:
- three‑equation workflow (tool‑calling)
- three‑equation workflow (codegen)
- NL task parser workflow
- ablations

## Contents
Core workflows:
- `run_three_equations_workflow.py`
- `run_three_equations_workflow_codegen.py`
- `nl_task_parser.py`

LLM tooling:
- `llm_tool_calling_provider.py`
- `llm_tool_calling_interpolation.py`
- `llm_tool_calling_cavity_parametric.py`
- `load_env.py`

Evaluation helper:
- `test_llm_operators_properly.py`

Models:
- `heat_model.pkl`
- `burgers_model.pkl`
- `cavity_model.pkl`

Ablations:
- `run_heat_ablation.py`
- `run_burgers_ablation.py`
- `run_cavity_ablation.py`

## Quick start
From the repo root:

### Three‑equation workflow (tool‑calling)
```
python run_three_equations_workflow.py --provider openai --model_name gpt-4o --output_dir 3_equations_test_results_gpt4o --save_raw
```

### Three‑equation workflow (codegen, PKL default)
```
python run_three_equations_workflow_codegen.py --provider openai --model_name gpt-4o --save_raw --save_plots
```

### NL task parser workflow
```
python nl_task_parser.py \
  --parser_provider openai --parser_model gpt-4o \
  --default_provider openai --default_model gpt-4o \
  --output_dir_base 3_equations_test_results_parsed_gpt4o \
  --merge_prompts --execute
```

## Environment
- Ensure `.env` contains API keys (OPENAI_API_KEY, etc.).
- All paths are relative to the repo root.

## Datasets
Test datasets are under `src/dataset/`. If you regenerate data, update the file paths accordingly.
