#!/usr/bin/env python3
"""Run centralized baseline and federated ADMM task, then compare outputs."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from vantage6.client import Client

FEATURE_COLUMNS = (
    "patient_t_stage",
    "patient_n_stage",
    "patient_m_stage",
    "patient_overall_stage",
)

T_MAP = {
    "T0": 0,
    "T1": 1,
    "T1a": 1,
    "T1b": 1,
    "T1c": 1,
    "T1mi": 1,
    "Tis": 1,
    "T2": 2,
    "T2a": 2,
    "T2b": 2,
    "T3": 3,
    "T4": 4,
    "Tx": 5,
}
N_MAP = {"N0": 0, "N1": 1, "N2": 2, "N3": 3, "Nx": 4}
M_MAP = {"M0": 0, "M1": 1, "M1a": 1, "M1b": 1, "M1c": 1, "Mx": 2}
S_MAP = {
    "0": 0,
    "I": 1,
    "IA": 1,
    "IA1": 1,
    "IA2": 1,
    "IA3": 1,
    "IB": 1,
    "II": 2,
    "IIA": 2,
    "IIB": 2,
    "III": 3,
    "IIIA": 3,
    "IIIB": 3,
    "IIIC": 3,
    "IV": 4,
    "IVA": 4,
    "IVB": 4,
    "Occult": 5,
    "x": 5,
}


def _map_tnm(series: pd.Series, mmap: dict) -> pd.Series:
    s = series.astype(str).str.strip()
    s = s.mask(s.str.lower() == "nan", np.nan)
    return s.map(mmap)


def _map_overall(series: pd.Series) -> pd.Series:
    num = pd.to_numeric(series, errors="coerce")
    str_stripped = series.astype(str).str.strip()
    str_stripped = str_stripped.mask(str_stripped.str.lower() == "nan", np.nan)
    is_whole = num.notna() & (num == np.floor(num))
    keys = pd.Series(index=series.index, dtype=object)
    keys.loc[series.isna()] = np.nan
    keys.loc[is_whole & series.notna()] = num.loc[is_whole & series.notna()].astype(int).astype(str)
    keys.loc[~is_whole & series.notna()] = str_stripped.loc[~is_whole & series.notna()]
    return keys.map(S_MAP)


def _prepare_stage_columns_for_dummies(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["patient_t_stage"] = _map_tnm(out["patient_t_stage"], T_MAP)
    out["patient_n_stage"] = _map_tnm(out["patient_n_stage"], N_MAP)
    out["patient_m_stage"] = _map_tnm(out["patient_m_stage"], M_MAP)
    out["patient_overall_stage"] = _map_overall(out["patient_overall_stage"])
    out["patient_t_stage"] = pd.Categorical(
        out["patient_t_stage"], categories=sorted(set(T_MAP.values()))
    )
    out["patient_n_stage"] = pd.Categorical(
        out["patient_n_stage"], categories=sorted(set(N_MAP.values()))
    )
    out["patient_m_stage"] = pd.Categorical(
        out["patient_m_stage"], categories=sorted(set(M_MAP.values()))
    )
    out["patient_overall_stage"] = pd.Categorical(
        out["patient_overall_stage"], categories=sorted(set(S_MAP.values()))
    )
    return out


def _list_data(response: Any) -> list[Any]:
    if isinstance(response, dict) and "data" in response:
        payload = response["data"]
        return payload if isinstance(payload, list) else []
    return response if isinstance(response, list) else []


def _fit_centralized(csv_path: Path) -> dict[str, Any]:
    df = pd.read_csv(csv_path)

    df = _prepare_stage_columns_for_dummies(df)

    df["SurvivalStatus"] = (
        df["vital_status"].astype(str).str.lower().str.strip() == "alive"
    ).astype(int)
    df["__diag_year__"] = pd.to_numeric(df["year_of_diagnosis"], errors="coerce")

    df = df.dropna(subset=[*FEATURE_COLUMNS, "SurvivalStatus", "__diag_year__"])

    y = df["SurvivalStatus"].to_numpy(dtype=int)
    X_df = pd.get_dummies(df[list(FEATURE_COLUMNS)], drop_first=True)
    X = X_df.to_numpy(dtype=float)
    years = df["__diag_year__"].to_numpy(dtype=int)

    train_mask = years <= 2011
    val_mask = years >= 2012

    if train_mask.sum() == 0 or val_mask.sum() == 0:
        raise RuntimeError("Train/validation split is empty; check diagnosis years")
    if len(np.unique(y[train_mask])) < 2 or len(np.unique(y[val_mask])) < 2:
        raise RuntimeError("Train or validation split has only one class")

    model = LogisticRegression(max_iter=500, solver="lbfgs")
    model.fit(X[train_mask], y[train_mask])

    train_probs = model.predict_proba(X[train_mask])[:, 1]
    val_probs = model.predict_proba(X[val_mask])[:, 1]

    train_auc = float(roc_auc_score(y[train_mask], train_probs))
    val_auc = float(roc_auc_score(y[val_mask], val_probs))

    coefficients = np.concatenate([model.intercept_, model.coef_.ravel()]).tolist()

    return {
        "train_auc": train_auc,
        "val_auc": val_auc,
        "train_count": int(train_mask.sum()),
        "val_count": int(val_mask.sum()),
        "feature_count": int(X.shape[1]),
        "feature_names": X_df.columns.tolist(),
        "coefficients": coefficients,
    }


def _extract_result_payload(raw_result: Any) -> dict[str, Any]:
    def _as_payload(value: Any) -> dict[str, Any] | None:
        if isinstance(value, dict) and value:
            return value
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return None
            try:
                decoded = json.loads(text)
            except json.JSONDecodeError:
                return None
            if isinstance(decoded, dict) and decoded:
                return decoded
        return None

    candidates = _list_data(raw_result)

    for entry in candidates:
        if isinstance(entry, dict):
            payload = _as_payload(entry.get("result"))
            if payload is not None:
                return payload

    if isinstance(raw_result, dict):
        payload = _as_payload(raw_result.get("result"))
        if payload is not None:
            return payload

    raise RuntimeError(
        "Could not extract task result payload from response: " + json.dumps(raw_result, default=str)
    )


def _wait_for_task(client: Client, task_id: int, timeout_seconds: int) -> str:
    start = time.time()
    last_status = ""

    while time.time() - start <= timeout_seconds:
        task_info = client.task.get(task_id)
        status = str(task_info.get("status", "unknown"))

        if status != last_status:
            print(f"task_id={task_id} status={status}")
            last_status = status

        if status in {"completed", "failed", "crashed", "cancelled"}:
            return status
        time.sleep(5)

    return "timeout"


def _run_federated(args: argparse.Namespace) -> dict[str, Any]:
    client = Client(args.server_url, args.server_port, args.api_path)
    client.authenticate(args.username, args.password)
    client.setup_encryption(None)

    collabs = _list_data(client.collaboration.list())
    if not collabs:
        raise RuntimeError("No collaborations found on vantage6 server")

    if args.collaboration_name:
        selected = [c for c in collabs if c.get("name") == args.collaboration_name]
        if not selected:
            raise RuntimeError(
                f"Collaboration '{args.collaboration_name}' not found. Available: "
                + ", ".join(str(c.get("name")) for c in collabs)
            )
        collaboration = selected[0]
    else:
        collaboration = collabs[0]

    collab_id = int(collaboration["id"])
    orgs = _list_data(client.organization.list(collaboration=collab_id))
    org_ids = [int(o["id"]) for o in orgs]
    if not org_ids:
        raise RuntimeError("No organizations found in selected collaboration")

    task = client.task.create(
        collaboration=collab_id,
        organizations=[org_ids[0]],
        name=f"20k synthetic ADMM ({args.num_rounds} rounds)",
        image=args.algo_image,
        description="Synthetic-data infra smoke task",
        input_={
            "method": "central_function",
            "kwargs": {
                "num_rounds": int(args.num_rounds),
                "rho": float(args.rho),
                "alpha": float(args.alpha),
                "lambda_": float(args.lambda_),
                "abs_tol": float(args.abs_tol),
                "rel_tol": float(args.rel_tol),
                "logging": bool(args.logging),
            },
        },
        databases=[{"label": args.database_label}],
    )

    task_id = int(task["id"])
    status = _wait_for_task(client, task_id, timeout_seconds=args.timeout_seconds)

    run_log = client.run.list(task=task_id)
    raw_result = client.result.from_task(task_id=task_id)

    payload: dict[str, Any] = {}
    if status == "completed":
        payload = _extract_result_payload(raw_result)

    return {
        "task_id": task_id,
        "status": status,
        "collaboration_id": collab_id,
        "org_ids": org_ids,
        "raw_runs": run_log,
        "raw_result": raw_result,
        "payload": payload,
    }


def _compare_coefficients(
    centralized: dict[str, Any], federated_payload: dict[str, Any]
) -> dict[str, Any]:
    fed = np.asarray(federated_payload.get("coefficients", []), dtype=float)
    cen = np.asarray(centralized.get("coefficients", []), dtype=float)

    if fed.size == 0 or cen.size == 0:
        return {
            "comparable": False,
            "reason": "missing coefficients",
            "centralized_len": int(cen.size),
            "federated_len": int(fed.size),
        }

    if fed.size != cen.size:
        return {
            "comparable": False,
            "reason": "coefficient length mismatch",
            "centralized_len": int(cen.size),
            "federated_len": int(fed.size),
        }

    diff = fed - cen
    l2_abs = float(np.linalg.norm(diff))
    l2_rel = float(l2_abs / (np.linalg.norm(cen) + 1e-12))

    return {
        "comparable": True,
        "centralized_len": int(cen.size),
        "federated_len": int(fed.size),
        "l2_abs_diff": l2_abs,
        "l2_rel_diff": l2_rel,
        "max_abs_diff": float(np.max(np.abs(diff))),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--full-data", required=True)
    parser.add_argument("--algo-image", required=True)
    parser.add_argument("--output-json", required=True)

    parser.add_argument("--server-url", default="http://localhost")
    parser.add_argument("--server-port", type=int, default=5070)
    parser.add_argument("--api-path", default="/api")
    parser.add_argument("--username", default="gamma-user")
    parser.add_argument("--password", default="gamma-password")
    parser.add_argument("--collaboration-name", default="")
    parser.add_argument("--database-label", default="default")

    parser.add_argument("--num-rounds", type=int, default=25)
    parser.add_argument("--rho", type=float, default=0.25)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--lambda", dest="lambda_", type=float, default=0.0)
    parser.add_argument("--abs-tol", type=float, default=1e-3)
    parser.add_argument("--rel-tol", type=float, default=1e-3)
    parser.add_argument("--logging", action="store_true")
    parser.add_argument("--timeout-seconds", type=int, default=1800)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    full_data_path = Path(args.full_data).resolve()
    output_path = Path(args.output_json).resolve()

    centralized = _fit_centralized(full_data_path)
    federated = _run_federated(args)

    comparison = _compare_coefficients(centralized, federated.get("payload", {}))

    output = {
        "inputs": {
            "full_data": str(full_data_path),
            "algo_image": args.algo_image,
            "server_url": args.server_url,
            "server_port": args.server_port,
            "api_path": args.api_path,
            "username": args.username,
            "num_rounds": args.num_rounds,
        },
        "centralized": centralized,
        "federated": {
            "task_id": federated["task_id"],
            "status": federated["status"],
            "collaboration_id": federated["collaboration_id"],
            "org_ids": federated["org_ids"],
            "payload": federated.get("payload", {}),
            "raw_runs": federated["raw_runs"],
            "raw_result": federated["raw_result"],
        },
        "comparison": comparison,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")

    print(f"saved_results={output_path}")
    print(
        "centralized_auc train={:.4f} val={:.4f}".format(
            centralized["train_auc"], centralized["val_auc"]
        )
    )

    if federated["status"] == "completed":
        fed_auc = federated.get("payload", {}).get("roc_global", {}).get("auc")
        print(f"federated_status=completed task_id={federated['task_id']} roc_global_auc={fed_auc}")
    else:
        print(f"federated_status={federated['status']} task_id={federated['task_id']}")

    print("coefficient_comparison=", json.dumps(comparison))

    if federated["status"] != "completed":
        raise SystemExit("Federated task did not complete successfully")


if __name__ == "__main__":
    main()
