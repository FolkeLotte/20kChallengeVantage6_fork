# Documentation

This directory contains first-pass documentation extracted from the codebase.

## Contents

- [Architecture overview](architecture.md)
- [Orchestration with vantage6](orchestration_vantage6.md)
- [API reference (verbatim docstrings)](api.md)
- [Diagram](central_function_flow.png)

## Source of truth

The code docstrings in [my-fl-project/20kLogRegChallenge/logreg_challenge_20k/central.py](../my-fl-project/20kLogRegChallenge/logreg_challenge_20k/central.py) and [my-fl-project/20kLogRegChallenge/logreg_challenge_20k/partial.py](../my-fl-project/20kLogRegChallenge/logreg_challenge_20k/partial.py) are the primary source for these notes.

The [API reference (verbatim docstrings)](api.md) page is generated directly from those source docstrings without summarization.

## Regenerate API Docs

From the repository root:

```bash
python scripts/generate_api_docstrings.py
```

If the implementation changes, this documentation should be updated to match.