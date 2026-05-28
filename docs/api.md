# API Reference (Verbatim Docstrings)

This file is generated from source docstrings without summarizing their content.

## my-fl-project/20kLogRegChallenge/logreg_challenge_20k/central.py

### Module Docstring

```text

Central (orchestrator) functions for the ADMM logistic regression algorithm.

The central function is executed on a vantage6 node just like any other method,
but it coordinates tasks across all participating organizations:

1. Each node preprocesses its local CSV into the exact feature representation
   used in your standalone script.
2. The central function runs ADMM rounds by:
   - requesting local X-updates from each node;
   - updating the global consensus vector ``z`` and the dual variables ``u``;
   - checking convergence;
   - evaluating the global model on each node's validation set.

The logic below closely mirrors the ``run_local_simulation`` routine from your
single-file simulation, but translated into the vantage6 task model.

```

### _z_objective

```text

    Exact Python implementation of the Z-update objective from your script.
    
```

### _check_convergence

```text

    Compute primal and dual residuals and their tolerances, mirroring the
    ``check_convergence`` function from your standalone code.
    
```

### _record_training_metrics

```text

    Mirror of recordAdmmVariables.m: logs objective, SSE, 'RMSE' and z-coefficients.
    
```

### _compute_roc_and_calibration

```text

    Compute global ROC/AUC and calibration metrics, analogous to the standalone
    evaluate_final_model() from the local simulation.
    
```

### central_function

[Flowchart: central_function](central_function_flow.png)

```text

    Central ADMM logistic regression.

    This function orchestrates the full ADMM procedure across all nodes and
    returns the final global coefficients plus a small convergence history.
    
```

## my-fl-project/20kLogRegChallenge/logreg_challenge_20k/partial.py

### Module Docstring

```text

Partial (node-side) functions for the ADMM logistic regression algorithm.

Each function in this module is executed on the data-owning nodes. The central
algorithm (see ``central.py``) orchestrates the ADMM rounds by repeatedly
calling these functions via vantage6 tasks.

This version is adapted to 20kChallenge-style data. On each node, the local
table/CSV is expected to contain at least the following columns:

    - patient_identifier                               (ignored)
    - patient_t_stage
    - patient_n_stage
    - patient_m_stage
    - patient_overall_stage
    - year_of_diagnosis                               (used only for time split)
    - interval_diagnosis_to_last_visit_in_days        (used only for outcome)
    - vital_status                                    (alive / dead)
    - centre                                          (ignored)
    - age_at_diagnosis                                (ignored)

We:
  - derive a binary outcome ``twoYearSurvival`` from ``vital_status`` and
    ``interval_diagnosis_to_last_visit_in_days`` following the 20kChallenge
    definition;
  - use only the four stage-like variables as features:
        {'patient_t_stage', 'patient_n_stage',
         'patient_m_stage', 'patient_overall_stage'};
  - split data into train/validation by year_of_diagnosis:
        * train: year_of_diagnosis <= 2011
        * val:   year_of_diagnosis >= 2012

```

### _compute_two_year_survival

```text

    Compute the binary two-year survival variable in line with the
    20kChallenge rules.

    Rules (2 years = 2 * 365.24 days):
      - if vital status missing                 -> NaN
      - if dead  and days <= 2y                 -> 0
      - if dead  and days  >  2y                -> 1
      - if alive and days <= 2y                 -> NaN (insufficient follow-up)
      - if alive and days  >  2y                -> 1
    
```

### _preprocess_local_dataframe

```text

    Node-local preprocessing:
      - derive twoYearSurvival from vital_status and follow-up time,
      - encode the four stage features as one-hot (K-1) dummies,
      - split into train/validation sets based on diagnosis year.
    
```

### _predict_proba

```text

    Probabilities P(y=1 | X, coef)
    
```

### _logistic_admm_objective

```text

    Objective function and gradient for the local X-update (site optimization),
    copied from your standalone script.
    
```

### init_node

```text

    Initialization call from the central function.

    It runs the full preprocessing pipeline once to determine the local number
    of features and the local patient count. The central algorithm uses this
    to set up the ADMM state and hyperparameters.
    
```

### admm_x_update_partial

```text

    Local X-update for one ADMM round.

    Parameters are provided by the central function:
    - z: current global consensus parameter vector
    - u: current local dual variable for this node
    - rho: ADMM penalty parameter
    - total_patients: global number of patients across all nodes
    
```

### evaluate_global_model

```text

    Evaluate the current global consensus vector ``z`` on the local validation
    data of this node. This mirrors the validation logic in your standalone
    simulation.
    
```

### collect_predictions

```text

    Return predictions and outcomes (val only) for the final global model.

    This is used by the central function after convergence to compute
    ROC/AUC and calibration metrics on the held-out validation set only
    (years >= 2012), consistent with compare_models.py and ADMM_Local.py.
    
```
