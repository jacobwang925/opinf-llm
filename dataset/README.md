# Dataset generation (OpInf-LLM)

This folder contains the data generation and training scripts for the datasets used by the workflows,
plus the **test datasets** that are consumed by the three‑equation workflows.

## Included scripts (used to generate the datasets used by the workflows)
Heat:
- `parametric_heat_1_generate_data_separated.py`
- `parametric_heat_2_train_model.py`

Burgers:
- `parametric_burgers_1_generate_data_separated.py`
- `parametric_burgers_2_train_model.py`

Cavity:
- `cavity_2d_1_generate_data_parametric.py`
- `cavity_2d_2_train_model_parametric.py`

## Included datasets (test sets)
- `heat_dataset_test.pkl.gz`
- `burgers_dataset_test.pkl.gz`
- `cavity_dataset_test.pkl.gz`

These are the datasets used by:
- `run_three_equations_workflow.py`
- `run_three_equations_workflow_codegen.py`
- `nl_task_parser.py`

## Regenerating data
Run the *_generate_data_* scripts to produce datasets, then *_train_model_* to produce models.
Defaults are configured to create the test datasets above. See each script's help for flags.
