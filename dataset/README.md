# Dataset generation (OpInf-LLM)

This folder contains dataset generation/training scripts and dataset files used by ablations and workflows.

## Scripts
Heat:
- `parametric_heat_1_generate_data_separated.py`
- `parametric_heat_2_train_model.py`

Burgers:
- `parametric_burgers_1_generate_data_separated.py`
- `parametric_burgers_2_train_model.py`

Cavity:
- `cavity_2d_1_generate_data_parametric.py`
- `cavity_2d_2_train_model_parametric.py`

## Dataset files
Heat:
- `heat_dataset_train.pkl.gz`
- `heat_dataset_unified.pkl.gz`
- `heat_dataset_test.pkl.gz`

Burgers:
- `burgers_dataset_train.pkl.gz`
- `burgers_dataset_unified.pkl.gz`
- `burgers_dataset_test.pkl.gz`

Cavity:
- `cavity_dataset_train.pkl.gz`
- `cavity_dataset_test.pkl.gz`

## Used by workflows
- `src/run_three_equations_workflow_tool_call.py`
- `src/run_three_equations_workflow_codegen.py`
- `src/run_three_equations_workflow_codegen_struct.py`

## Regenerating data
Run `*_generate_data_*` scripts to create datasets, then `*_train_model_*` scripts to create model `.pkl` files.
Use each script's `--help` for exact flags and output paths.
