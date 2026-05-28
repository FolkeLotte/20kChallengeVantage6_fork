# Architecture Overview

This project implements ADMM-based logistic regression in a federated setting.

## Main parts

- **Central orchestration** lives in [central.py](../my-fl-project/20kLogRegChallenge/logreg_challenge_20k/central.py).
- **Node-side tasks** live in [partial.py](../my-fl-project/20kLogRegChallenge/logreg_challenge_20k/partial.py).
- **Local simulation and baseline scripts** are described in [README.md](../README.md).

## High-level flow

1. The central function discovers participating organizations.
2. Each node runs `init_node` once to determine local feature dimensions and patient counts.
3. Each ADMM round performs a local X-update on every node with `admm_x_update_partial`.
4. The central node updates the consensus vector `z` and the dual variables `u`.
5. The global model is evaluated on each node with `evaluate_global_model`.
6. If the convergence criteria are satisfied, the loop stops.
7. After convergence, each node runs `collect_predictions` so the central side can compute ROC/AUC and calibration metrics.

## Data handling

The node-side preprocessing expects at least these columns:

- `patient_t_stage`
- `patient_n_stage`
- `patient_m_stage`
- `patient_overall_stage`
- `year_of_diagnosis`
- `interval_diagnosis_to_last_visit_in_days`
- `vital_status`

The preprocessing pipeline derives a binary `SurvivalStatus`, encodes the stage variables as dummies, and splits the data into train and validation sets by diagnosis year.

## Important outputs

- Training progress is tracked through objective values, SSE, and model coefficients.
- Validation progress is tracked through per-node accuracy and global RMSE.
- Final evaluation includes ROC/AUC and calibration metrics.

## Diagram

See [central_function_flow.png](../central_function_flow.png) for a visual flow of `central_function`.