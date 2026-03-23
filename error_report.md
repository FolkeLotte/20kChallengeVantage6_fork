# Error Report: 20kChallengeVantage6 (local validation on Debian VM by Codex, then reproduced on MacOS by Ivan, 2026-03-13)

## Scope executed

- Mock algorithm test: `test/MockClient.py` in package `20kLogRegChallenge`
- Infra smoke checks via `v6-infrastructure-sh` (`infra.sh test`)
- Real infra-backed task execution (`central_function`) on 3 nodes with CSV dummy data

## Summary

- Core ADMM flow executes on mock and real infra.
- Real infra task reached `completed` with result payload.
- The mathematical correctness of ADMM was not checked, only the infrastructure.
- Multiple reproducibility and correctness issues were found on infra side.


## Findings

1. Invalid dependency name in `requirements.txt`
- File: `my-fl-project/20kLogRegChallenge/requirements.txt`
- Problem: contains `vantage6-tools`, which is not installable from PyPI.
- Observed error:
  - `ERROR: No matching distribution found for vantage6-tools`
- Impact: fresh environment setup fails unless manually overridden.
- Comment: did you mean `vantage6-algorithm-tools` - you already have it though.

2. Outdated client API usage in network script
- File: `my-fl-project/20kLogRegChallenge/run_on_v6_network.py`
- Problem A: uses `client.result.list(...)` but client object exposes `result.from_task(...)` / `result.get(...)`.
- Problem B: script defaults (`SERVER_PORT=7601`, `IMAGE='surfzare/20klogregchallenge'`) are not aligned to local infra setup.
- Impact: script fails or targets wrong runtime unless edited by user.
- Comment: try to make the repo agnostic and allow user to specify envvars in a config.

3. Two-year survival label logic does not match docstring/methodology
- File: `my-fl-project/20kLogRegChallenge/logreg_challenge_20k/partial.py`
- Function: `_compute_two_year_survival(...)`
- Problem: current implementation sets all `dead -> 0`, all `alive -> 1`, while comments/docstring describe censoring and 2-year threshold behavior using follow-up days.
- Impact: target definition differs from intended 2-year survival methodology.
- Comment: use the formula:
```python
  threshold = 2 * 365.24
  two_year_survival = np.where(
    (dead_status_event == 1) & (survival_time <= threshold),
    0,
    np.where(survival_time > threshold, 1, np.nan)
  )
```


## Raw notable errors captured

- Dependency install:
  - `No matching distribution found for vantage6-tools`