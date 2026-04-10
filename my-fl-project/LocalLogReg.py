import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.optimize import minimize
from typing import Any, Dict, Tuple

# ============================================================================
# 1. Configuration & Constants
# ============================================================================
np.random.seed(67)

FEATURE_COLUMNS = (
    "patient_t_stage", "patient_n_stage", "patient_m_stage", "patient_overall_stage"
)
OUTCOME_COLUMN = "SurvivalStatus"

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


def LossLogist(x,X_Train,Y_Train, patients)-> Tuple[float, np.ndarray]:
    """
    Objective function and gradient for the local X-update (site optimization),
    copied from your standalone script.
    """
    n_local = X_Train.shape[0]
    X_design = np.hstack([np.ones((n_local, 1)), X_Train])  # Add intercept
    logits = X_design @ x
    probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -500, 500)))
    eps = 1e-15
    probs = np.clip(probs, eps, 1.0 - eps)

    value = (-2.0 / patients) * np.sum( Y_Train * np.log(probs ) + (1.0 - Y_Train) * np.log(1.0 - probs) )

    # Gradient
    grad_sum = np.zeros_like(x)
    for i in range(n_local):
        xi = X_design[i, :]  # row vector
        yi = Y_Train[i]
        exp_term = np.exp(xi @ x)
        grad_i = xi * (yi + (yi - 1) * exp_term) / (1 + exp_term)
        grad_sum += grad_i

    grad = (-2.0 / patients) * grad_sum
    return value, grad
# ============================================================================
# 2. Exact Preprocessing Matches
# ============================================================================
def _compute_two_year_survival(vital_status: pd.Series, days_until_last_visit: pd.Series) -> pd.Series:
    vs = vital_status.astype(str).str.lower().str.strip()
    days = pd.to_numeric(days_until_last_visit, errors="coerce")
    out = pd.Series(np.nan, index=vital_status.index, dtype=float)
    
    dead = vs == "dead"
    alive = vs == "alive"
    out[dead] = 0
    out[alive] = 1
    return out

def load_and_preprocess_pooled_data(csv_file) -> tuple:
    df = pd.read_csv(csv_file)
    
    # 1. Compute Outcome
    df[OUTCOME_COLUMN] = _compute_two_year_survival(df["vital_status"], df["interval_diagnosis_to_last_visit_in_days"])

    # 2. Bucket synonymous stage labels, then categorical levels for K-1 dummies
    df = _prepare_stage_columns_for_dummies(df)

    # 3. Time Variable
    df["__diag_year__"] = pd.to_numeric(df["year_of_diagnosis"], errors="coerce")
    
    # 4. Drop NaNs
    df = df.dropna(subset=list(FEATURE_COLUMNS) + [OUTCOME_COLUMN, "__diag_year__"])
    print(f"Total pooled rows after cleaning: {len(df)}")

    # 5. Dummy Encoding (Drop First)
    y = df[OUTCOME_COLUMN].astype(int).to_numpy()
    df_cat = pd.get_dummies(df[list(FEATURE_COLUMNS)], drop_first=True)
    X = df_cat.to_numpy(dtype=float)
    feature_names = df_cat.columns.tolist()
    
    # 6. Time Split
    years = df["__diag_year__"].to_numpy().astype(int)
    train_mask = years <= 2011
    val_mask = years >= 2012

    return X[train_mask], y[train_mask], X[val_mask], y[val_mask], feature_names

# ============================================================================
# 3. Main Execution
# ============================================================================
if __name__ == "__main__":
    local_csvs =  "20kLogRegChallenge/test/fakebeach_merged_FULL.csv"
    # # 1. Preprocess
    X_train, y_train, X_val, y_val, feature_names = load_and_preprocess_pooled_data(local_csvs)

    num_features = X_train.shape[1]
    total_patients = X_train.shape[0]
    x_prev = np.zeros(num_features + 1)

    res = minimize(
        fun=LossLogist,
        x0=x_prev,
        args=(X_train, y_train, int(total_patients)),
        jac=True,
        method="BFGS",
        options={"disp": False},
    )
    
    print(f"\nTraining set: {X_train.shape[0]} patients, {X_train.shape[1]} features")
    print(f"Validation set: {X_val.shape[0]} patients")

    original_style_params = res.x

    print("\n" + "="*60)
    print(f"{'Index':<8} {'Variable Name':<40} {'Value':<12}")
    print("-" * 60)
    print(f"{0:<8} {'(Intercept)':<40} {original_style_params[0]:.6f}")
    for i, name in enumerate(feature_names):
        print(f"{i+1:<8} {name:<40} {original_style_params[i+1]:.6f}")
    print("="*60)

    print("\nVector representation (z_arr):")
    print(original_style_params.tolist())
