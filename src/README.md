# opinf-llm

This folder contains the code and artifacts needed to run the **OpInf-LLM** workflows:
- three‑equation workflow (tool‑calling)
- three‑equation workflow (codegen)
- NL task parser workflow
- ablations

## Contents
Core workflows:
- `run_three_equations_workflow_tool_call.py`
- `run_three_equations_workflow_codegen.py`
- `run_three_equations_workflow_nl.py`

LLM tooling:
- `llm_tool_calling_provider.py`
- `llm_tool_calling_parametric_1d.py`
- `llm_tool_calling_parametric_2d.py`
- `load_env.py`

Evaluation helpers:
- `test_utility_1d.py`
- `test_utility_2d.py`

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
python src/run_three_equations_workflow_tool_call.py \
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
python src/run_three_equations_workflow_nl.py \
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
python ablation_nl_parser_diversity.py \
  --mode both \
  --parser_provider openai \
  --parser_model gpt-4.1 \
  --output ablation_nl_parser_diversity.json \
  --show_failures
```

### Equation-specific OpInf ablations
# Heat ablation
```
python src/run_heat_ablation.py \
  --provider openai \
  --model_name gpt-4o \
  --output_dir_base heat_ablation_runs \
  --save_raw
```

# Burgers ablation
```
python src/run_burgers_ablation.py \
  --provider openai \
  --model_name gpt-4o \
  --output_dir_base burgers_ablation_runs \
  --save_raw
```

# Cavity ablation
```
python src/run_cavity_ablation.py \
  --provider openai \
  --model_name gpt-4o \
  --output_dir_base cavity_ablation_runs \
  --save_raw
```


## Environment
- Ensure `.env` contains API keys (OPENAI_API_KEY, etc.).

## Datasets
Test datasets are under `dataset/`. 
