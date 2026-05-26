"""
compare_models.py — Fair comparison of Centralized vs ADMM models.

Both models are evaluated on the same shared validation set:
  the years >= 2012 split from fakebeach_merged_FULL.csv.

Usage (fast):
  1. Run ADMM_Local.py first. At the end it prints:
       "Vector representation (z_arr): [...]"
  2. Paste that list as admm_coef in __main__ and uncomment it.
  3. Run: python compare_models.py

Usage (automated, slow ~hundreds of rounds):
  Leave admm_coef commented out — ADMM runs from scratch.
"""

from __future__ import annotations
import contextlib
import io
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.linear_model import LogisticRegression
from typing import List, Optional, Tuple

# ADMM_Local.py is in the same directory — import math primitives directly.
sys.path.insert(0, os.path.dirname(__file__))
from ADMM_Local import (
    load_local_patient_data,
    predict_proba,
    admm_x_update,
    admm_z_u_update,
    check_convergence,
    InstanceState,
    SiteState,
    LocalNode,
)

np.random.seed(67)


# ==============================================================================
# 1. Centralized Training
# ==============================================================================

def _central_loss(
    coef: np.ndarray,
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_patients: int,
) -> Tuple[float, np.ndarray]:
    """
    Logistic loss + gradient for BFGS.
    Matches LossLogist from LocalLogReg.py — do NOT diverge.
    """
    n_local = X_train.shape[0]
    X_design = np.hstack([np.ones((n_local, 1)), X_train])
    logits = X_design @ coef
    probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -500, 500)))
    eps = 1e-15
    probs = np.clip(probs, eps, 1.0 - eps)

    value = (-2.0 / n_patients) * np.sum(
        y_train * np.log(probs) + (1.0 - y_train) * np.log(1.0 - probs)
    )

    grad_sum = np.zeros_like(coef)
    for i in range(n_local):
        xi = X_design[i, :]
        yi = y_train[i]
        exp_term = np.exp(xi @ coef)
        grad_i = xi * (yi + (yi - 1) * exp_term) / (1 + exp_term)
        grad_sum += grad_i

    grad = (-2.0 / n_patients) * grad_sum
    return value, grad


def _train_centralized(X_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
    """Train pooled logistic regression via BFGS. Returns coef (intercept at [0])."""
    x0 = np.ones(X_train.shape[1] + 1)
    res = minimize(
        fun=_central_loss,
        x0=x0,
        args=(X_train, y_train, int(X_train.shape[0])),
        jac=True,
        method="BFGS",
        options={"disp": False},
    )
    return res.x


# ==============================================================================
# 2. Silent ADMM Runner
# ==============================================================================

def _run_admm_silent(
    site_csvs: List[str],
    num_rounds: int = 400,
    rho: float = 0.25,
    alpha: float = 1.0,
    lambda_: float = 0.0,
    abs_tol: float = 0.001,
    rel_tol: float = 0.001,
    verbose: bool = True,
) -> np.ndarray:
    """
    Run ADMM to convergence with identical math/init as ADMM_Local.run_local_simulation.
    Suppresses all per-round output. Returns z (global consensus coefficients).
    """
    local_data_list = []
    for path in site_csvs:
        ld, _ = load_local_patient_data(path)
        local_data_list.append(ld)

    patient_counts = [int(ld.X_train.shape[0]) for ld in local_data_list]
    num_features = local_data_list[0].X_train.shape[1] + 1  # +1 for intercept

    instance = InstanceState(
        rho=rho,
        alpha=alpha,
        lambda_=lambda_,
        abs_tol=abs_tol,
        rel_tol=rel_tol,
        patient_counts=patient_counts,
        x_init=np.zeros(num_features),
        u_init=np.zeros(num_features),
        z_init=np.ones(num_features),
    )

    nodes = []
    for i, data in enumerate(local_data_list):
        state = SiteState(
            x=instance.x_init.copy(),
            u=instance.u_init.copy(),
            z=instance.z_init.copy(),
            patient_count=data.X_train.shape[0],
        )
        nodes.append(LocalNode(i, data, state))

    if verbose:
        print(f"[ADMM silent] Starting ({num_rounds} rounds, rho={rho}, alpha={alpha}) ...")

    for r in range(1, num_rounds + 1):
        # Suppress per-node training accuracy prints inside admm_x_update
        with contextlib.redirect_stdout(io.StringIO()):
            for node in nodes:
                admm_x_update(node, instance)
            admm_z_u_update(nodes, instance)

        r_norm, s_norm, eps_pri, eps_dual = check_convergence(nodes, instance, logging=False)

        if verbose and r % 50 == 0:
            print(f"  Round {r:4d}: r_norm={r_norm:.4e}, s_norm={s_norm:.4e}")

        if r_norm < eps_pri and s_norm < eps_dual:
            if verbose:
                print(f"  Converged at round {r}.")
            break

    # Global consensus model is nodes[0].state.z (same on all nodes after z-update)
    return nodes[0].state.z.copy()


# ==============================================================================
# 3. Comparison Plots
# ==============================================================================

def plot_comparison_roc(
    X_val: np.ndarray,
    y_val: np.ndarray,
    central_coef: np.ndarray,
    admm_coef: np.ndarray,
) -> None:
    """Overlaid ROC curves for both models on the shared val set."""
    c_probs = predict_proba(X_val, central_coef)
    a_probs = predict_proba(X_val, admm_coef)

    c_fpr, c_tpr, _ = roc_curve(y_val, c_probs)
    a_fpr, a_tpr, _ = roc_curve(y_val, a_probs)
    c_auc = roc_auc_score(y_val, c_probs)
    a_auc = roc_auc_score(y_val, a_probs)

    plt.figure(figsize=(8, 6))
    plt.plot(c_fpr, c_tpr, color="tab:blue", lw=2,                 label=f"Centralized  (AUC={c_auc:.3f})")
    plt.plot(a_fpr, a_tpr, color="tab:red",  lw=2, linestyle="--", label=f"ADMM z       (AUC={a_auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.3)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Comparison: Centralized vs ADMM\n(shared val set, n={len(y_val)})")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_comparison_accuracy(
    X_val: np.ndarray,
    y_val: np.ndarray,
    central_coef: np.ndarray,
    admm_coef: np.ndarray,
) -> None:
    """Side-by-side accuracy and RMSE bar charts for both models."""
    c_probs = predict_proba(X_val, central_coef)
    a_probs = predict_proba(X_val, admm_coef)

    c_acc  = float(np.mean((c_probs >= 0.5).astype(int) == y_val))
    a_acc  = float(np.mean((a_probs >= 0.5).astype(int) == y_val))
    c_rmse = float(np.sqrt(np.mean((y_val - c_probs) ** 2)))
    a_rmse = float(np.sqrt(np.mean((y_val - a_probs) ** 2)))

    labels = ["Centralized", "ADMM z"]
    colors = ["tab:blue", "tab:red"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    bars1 = ax1.bar(labels, [c_acc, a_acc], color=colors)
    ax1.set_ylim([0, 1.1])
    ax1.set_ylabel("Accuracy")
    ax1.set_title("Accuracy on Shared Val Set")
    for bar, v in zip(bars1, [c_acc, a_acc]):
        ax1.text(bar.get_x() + bar.get_width() / 2, v + 0.01, f"{v:.4f}", ha="center", fontsize=11)
    ax1.grid(alpha=0.3, axis="y")

    bars2 = ax2.bar(labels, [c_rmse, a_rmse], color=colors)
    ax2.set_ylabel("RMSE")
    ax2.set_title("RMSE on Shared Val Set")
    for bar, v in zip(bars2, [c_rmse, a_rmse]):
        ax2.text(bar.get_x() + bar.get_width() / 2, v + 0.001, f"{v:.4f}", ha="center", fontsize=11)
    ax2.grid(alpha=0.3, axis="y")

    fig.suptitle(f"Model Comparison (shared val set, n={len(y_val)})", fontsize=13)
    plt.tight_layout()
    plt.show()


def plot_comparison_calibration(
    X_val: np.ndarray,
    y_val: np.ndarray,
    central_coef: np.ndarray,
    admm_coef: np.ndarray,
) -> None:
    """Overlaid calibration quantile curves for both models."""
    def _calib_curve(probs, y):
        eps = 1e-15
        probs_c = np.clip(probs, eps, 1 - eps)
        lp = np.log(probs_c / (1 - probs_c))
        lr = LogisticRegression(fit_intercept=True, solver="lbfgs")
        lr.fit(lp.reshape(-1, 1), y)
        intercept = float(lr.intercept_[0])
        slope = float(lr.coef_[0, 0])

        n_q = 10
        edges = np.quantile(probs, np.linspace(0, 1, n_q + 1))
        mean_pred, mean_obs = [], []
        for q in range(n_q):
            lo, hi = edges[q], edges[q + 1]
            mask = (probs >= lo) & (probs <= hi) if q == n_q - 1 else (probs >= lo) & (probs < hi)
            if not np.any(mask):
                continue
            mean_pred.append(float(np.mean(probs[mask])))
            mean_obs.append(float(np.mean(y[mask])))
        return mean_pred, mean_obs, intercept, slope

    c_probs = predict_proba(X_val, central_coef)
    a_probs = predict_proba(X_val, admm_coef)

    c_mp, c_mo, c_int, c_sl = _calib_curve(c_probs, y_val)
    a_mp, a_mo, a_int, a_sl = _calib_curve(a_probs, y_val)

    plt.figure(figsize=(7, 6))
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect calibration")
    plt.plot(c_mp, c_mo, "o-",  color="tab:blue", label=f"Centralized (int={c_int:.3f}, sl={c_sl:.3f})")
    plt.plot(a_mp, a_mo, "s--", color="tab:red",  label=f"ADMM z      (int={a_int:.3f}, sl={a_sl:.3f})")
    plt.xlabel("Predicted probability")
    plt.ylabel("Observed event rate")
    plt.title(f"Calibration Comparison\n(shared val set, n={len(y_val)})")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


# ==============================================================================
# 4. Orchestrator
# ==============================================================================

def compare_models(
    full_csv: str,
    site_csvs: List[str],
    admm_coef: Optional[np.ndarray] = None,
    central_coef: Optional[np.ndarray] = None,
    num_rounds: int = 10000,
    rho: float = 0.015,
    alpha: float = 1.6,
    lambda_: float = 0.0,
    abs_tol: float = 1e-12,
    rel_tol: float = 1e-12,
    include_calibration: bool = True,
) -> None:
    # 1. Load shared val set from full CSV
    print(f"Loading shared val set from: {full_csv}")
    full_data, feature_names = load_local_patient_data(full_csv)
    X_train_full = full_data.X_train
    y_train_full = full_data.y_train
    X_val        = full_data.X_val
    y_val        = full_data.y_val
    print(f"  Train: {len(y_train_full)} patients | Val: {len(y_val)} patients")

    # 2. Train centralized if not provided
    if central_coef is None:
        print("\nTraining centralized model (BFGS) ...")
        central_coef = _train_centralized(X_train_full, y_train_full)
        print("  Done.")
    else:
        print("\nUsing provided centralized coef.")

    # 3. Run ADMM if not provided
    if admm_coef is None:
        print(f"\nRunning ADMM on {len(site_csvs)} sites ...")
        admm_coef = _run_admm_silent(site_csvs, num_rounds, rho, alpha, lambda_, abs_tol, rel_tol)
        print("  Done.")
    else:
        print("\nUsing provided ADMM coef (z).")

    # 4. Coefficient comparison table
    coef_labels = ["(Intercept)"] + feature_names
    print("\n" + "=" * 90)
    print(f"{'Feature':<40} {'Centralized':>12} {'ADMM z':>12} {'|Diff|':>8} {'%Diff':>10}")
    print("-" * 90)
    abs_diffs = []
    pct_diffs = []
    for name, cv, av in zip(coef_labels, central_coef, admm_coef):
        abs_diff = abs(cv - av)
        pct_diff = (abs_diff / abs(cv) * 100) if cv != 0 else float("nan")
        abs_diffs.append(abs_diff)
        pct_diffs.append(pct_diff)
        pct_str = f"{pct_diff:>9.2f}%" if not np.isnan(pct_diff) else f"{'N/A (cv=0)':>10}"
        print(f"{name:<40} {cv:>12.6f} {av:>12.6f} {abs_diff:>8.2e} {pct_str}")
    print("-" * 90)
    valid_pct = [p for p in pct_diffs if not np.isnan(p)]
    avg_abs = float(np.mean(abs_diffs))
    avg_pct = float(np.mean(valid_pct)) if valid_pct else float("nan")
    print(f"{'Average':.<40} {'':>12} {'':>12} {avg_abs:>8.2e} {avg_pct:>9.2f}%")
    print("=" * 90)

    # 5. Plots
    print("\nGenerating comparison plots ...")
    plot_comparison_roc(X_val, y_val, central_coef, admm_coef)
    plot_comparison_accuracy(X_val, y_val, central_coef, admm_coef)
    if include_calibration:
        plot_comparison_calibration(X_val, y_val, central_coef, admm_coef)


# ==============================================================================
# ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    test_data_dir = os.path.join(
        repo_root, "my-fl-project", "20kLogRegChallenge", "test"
    )
    FULL_CSV = os.path.join(test_data_dir, "fakebeach_merged_FULL.csv")
    SITE_CSVS = [
        os.path.join(test_data_dir, "fakebeach_merged_0.csv"),
        os.path.join(test_data_dir, "fakebeach_merged_1.csv"),
        os.path.join(test_data_dir, "fakebeach_merged_2.csv"),
    ]

    # -------------------------------------------------------------------------
    # FAST WORKFLOW (recommended):
    # 1. Run ADMM_Local.py first. At the very end it prints:
    #      "Vector representation (z_arr): [x, x, x, ...]"
    # 2. Paste that list below and uncomment the admm_coef line.
    # 3. This script then runs in seconds (only centralized BFGS + plots).
    # -------------------------------------------------------------------------
    # admm_coef = np.array([...])  # <-- paste z_arr from ADMM_Local.py here

    # -------------------------------------------------------------------------
    # AUTOMATED WORKFLOW (slow — runs ADMM from scratch):
    # Leave admm_coef commented out above.
    # -------------------------------------------------------------------------

    compare_models(
        full_csv=FULL_CSV,
        site_csvs=SITE_CSVS,
        # admm_coef=admm_coef,    # uncomment after pasting z_arr above
        num_rounds=10000,
        rho=0.015,
        alpha=1.6,
        lambda_=0.0,
        abs_tol=1e-12,
        rel_tol=1e-12,
        include_calibration=True,
    )
