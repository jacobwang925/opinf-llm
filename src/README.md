# opinf-llm

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
- `test_llm_operators.py`

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
python src/run_three_equations_workflow.py \
  --provider openai \
  --model_name gpt-4o \
  --save_raw
```

### Three‑equation workflow (codegen)
```
python src/run_three_equations_workflow_codegen.py \
  --provider openai \
  --model_name gpt-4o \
  --save_raw \
  --save_plots \
  --reuse_code_per_equation \
  --max_attempts_per_case 8 \
  --sleep_secs 4
```

### NL task parser workflow
```
python src/nl_task_parser.py \
  --provider openai \
  --model_name gpt-4.1 \
  --prompts_file src/synthetic_prompts.txt \
  --save_raw
```

### Feature identification ablation
```
python src/run_three_equations_workflow_codegen_struct.py \
  --provider openai \
  --model_name gpt-4.1 \
  --save_raw \
  --save_plots \
  --reuse_code_per_equation \
  --max_attempts_per_case 8 \
  --sleep_secs 4
```

### NL parser ablation
```
python ablation_nl_parser_diversity.py   --mode both   --parser_provider openai   --parser_model gpt-4.1   --output ablation_nl_parser_diversity.json   --show_failures
```

## Environment
- Ensure `.env` contains API keys (OPENAI_API_KEY, etc.).

## Datasets
Test datasets are under `dataset/`. 
