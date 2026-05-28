# Orchestration With vantage6

This note explains how task dispatch works in this repository using decorators and method names.

## Decorators In This Project

Complete inventory of decorators imported from `vantage6.algorithm.tools.decorators` in this repository:

1. `@data(1)`

- Import location: [my-fl-project/20kLogRegChallenge/logreg_challenge_20k/partial.py](../my-fl-project/20kLogRegChallenge/logreg_challenge_20k/partial.py#L39)
- Used on:
	- [init_node](../my-fl-project/20kLogRegChallenge/logreg_challenge_20k/partial.py#L312) (decorator at [partial.py#L311](../my-fl-project/20kLogRegChallenge/logreg_challenge_20k/partial.py#L311))
	- [admm_x_update_partial](../my-fl-project/20kLogRegChallenge/logreg_challenge_20k/partial.py#L332) (decorator at [partial.py#L331](../my-fl-project/20kLogRegChallenge/logreg_challenge_20k/partial.py#L331))
	- [evaluate_global_model](../my-fl-project/20kLogRegChallenge/logreg_challenge_20k/partial.py#L395) (decorator at [partial.py#L394](../my-fl-project/20kLogRegChallenge/logreg_challenge_20k/partial.py#L394))
	- [collect_predictions](../my-fl-project/20kLogRegChallenge/logreg_challenge_20k/partial.py#L427) (decorator at [partial.py#L426](../my-fl-project/20kLogRegChallenge/logreg_challenge_20k/partial.py#L426))
- Used for: marking node-side task entry points and injecting one node-local dataset as the first argument at runtime.

2. `@algorithm_client`

- Import location: [my-fl-project/20kLogRegChallenge/logreg_challenge_20k/central.py](../my-fl-project/20kLogRegChallenge/logreg_challenge_20k/central.py#L27)
- Used on:
	- [central_function](../my-fl-project/20kLogRegChallenge/logreg_challenge_20k/central.py#L213) (decorator at [central.py#L212](../my-fl-project/20kLogRegChallenge/logreg_challenge_20k/central.py#L212))
- Used for: marking the central orchestrator entrypoint and injecting a configured `AlgorithmClient`.

No other decorators from `vantage6.algorithm.tools.decorators` are used in this codebase.

Node-side task functions in [my-fl-project/20kLogRegChallenge/logreg_challenge_20k/partial.py](../my-fl-project/20kLogRegChallenge/logreg_challenge_20k/partial.py) are decorated with `@data(1)`.

- Import: `from vantage6.algorithm.tools.decorators import data`
- Meaning here: one local dataset is injected by the runtime when the task is invoked on a node.

Decorated node entry points:

- `init_node`
- `admm_x_update_partial`
- `evaluate_global_model`
- `collect_predictions`

The central entry point in [my-fl-project/20kLogRegChallenge/logreg_challenge_20k/central.py](../my-fl-project/20kLogRegChallenge/logreg_challenge_20k/central.py) is decorated with `@algorithm_client`.

- Import: `from vantage6.algorithm.tools.decorators import algorithm_client`
- Meaning here: the function receives a configured `AlgorithmClient` from the runtime.

## Registry And Dispatch Model

At runtime, tasks are dispatched by method name. The central side sends a payload with `input_["method"] = "..."`, and vantage6 resolves that name to a discoverable callable in the algorithm package.

In this repository there is no hand-written dictionary lookup table for these methods. Conceptually, the framework registry is the set of exported/discoverable functions in the algorithm package.

Package exports are defined in [my-fl-project/20kLogRegChallenge/logreg_challenge_20k/__init__.py](../my-fl-project/20kLogRegChallenge/logreg_challenge_20k/__init__.py):

- `from .central import *`
- `from .partial import *`

## Where Task Names Are Used

The central orchestrator sends these method names from [my-fl-project/20kLogRegChallenge/logreg_challenge_20k/central.py](../my-fl-project/20kLogRegChallenge/logreg_challenge_20k/central.py):

- `"method": "init_node"`
- `"method": "admm_x_update_partial"`
- `"method": "evaluate_global_model"`
- `"method": "collect_predictions"`

These correspond to the decorated functions in [my-fl-project/20kLogRegChallenge/logreg_challenge_20k/partial.py](../my-fl-project/20kLogRegChallenge/logreg_challenge_20k/partial.py) with matching names.

## Top-Level Entry Method

The job submission scripts send `"method": "central_function"`, which triggers the orchestrator function in [my-fl-project/20kLogRegChallenge/logreg_challenge_20k/central.py](../my-fl-project/20kLogRegChallenge/logreg_challenge_20k/central.py).

Call sites:

- [my-fl-project/20kLogRegChallenge/run_on_v6_network.py](../my-fl-project/20kLogRegChallenge/run_on_v6_network.py)
- [my-fl-project/20kLogRegChallenge/test/MockClient.py](../my-fl-project/20kLogRegChallenge/test/MockClient.py)

## Practical Interpretation

From a developer perspective, you write functions and decorators; you do not manually call node methods from local Python code. Instead, the framework invokes methods by name via task payloads.
