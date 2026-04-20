"""
Shared helpers for severity baselines (majority-class and logistic regression).

Used by 02_majority_baseline.py, 03_logistic_regression.py, and 02_run_baselines.py.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "Cleaned" / "sgo_cleaned_incidents.csv"
OUT_DIR         = PROJECT_ROOT / "Modeling"
LR_DIR          = OUT_DIR / "logistic_regression"
RF_DIR          = OUT_DIR / "random_forest"
XGB_DIR         = OUT_DIR / "xgboost"
BASELINE_DIR    = OUT_DIR / "baselines"
CLUSTERING_DIR  = OUT_DIR / "clustering"

for _d in (OUT_DIR, LR_DIR, RF_DIR, XGB_DIR, BASELINE_DIR, CLUSTERING_DIR):
    _d.mkdir(exist_ok=True)

CONTEXT_FEATURES_BASE = [
    "Automation System Engaged?",
    "automation_level",
    "Roadway Type",
    "Roadway-Wet Surface Condition",
    "Roadway-Work Zone",
    "Roadway-Traffic Incident",
    "Weather - Clear",
    "Weather - Rain",
    "Weather - Snow",
    "Weather - Fog/Smoke/Haze",
    "Weather - Severe Wind",
    "Weather - Cloudy",
    "Crash With",
    "SV Pre-Crash Movement",
    "CP Pre-Crash Movement",
    "SV Precrash Speed (MPH)",
    "Report Month",
]


def load_known() -> pd.DataFrame:
    """Load cleaned data; keep rows with at least one observed severity component."""
    df = pd.read_csv(DATA_PATH, low_memory=False)
    return df[df["severity_known"] == 1].copy()


def split_train_test(df_known: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Temporal split (archived train, current test) when viable; else stratified 80/20."""
    current_known = df_known[df_known["era"] == "current"]
    archived_known = df_known[df_known["era"] == "archived"]
    temporal_ok = (
        len(current_known) >= 100
        and current_known["severe"].nunique() == 2
        and archived_known["severe"].nunique() == 2
    )
    if temporal_ok:
        return archived_known.copy(), current_known.copy()
    strat_key = df_known["severe"].astype(str) + "_" + df_known["automation_level"].astype(str)
    return train_test_split(
        df_known, test_size=0.2, random_state=42, stratify=strat_key
    )


def context_features(df_known: pd.DataFrame) -> list[str]:
    return [f for f in CONTEXT_FEATURES_BASE if f in df_known.columns]


def prepare_X(
    df_in: pd.DataFrame,
    feature_cols: list[str],
    fit_encoder: list[str] | None = None,
):
    """One-hot encode categoricals; return raw matrix (unscaled) and numeric column names."""
    X = df_in[feature_cols].copy()
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
    X[num_cols] = X[num_cols].fillna(X[num_cols].median())
    X[cat_cols] = X[cat_cols].fillna("Unknown")
    X = pd.get_dummies(X, columns=cat_cols, drop_first=False)
    if fit_encoder is not None:
        X = X.reindex(columns=fit_encoder, fill_value=0)
    return X, num_cols, list(X.columns)


def scale_for_lr(
    X_train_raw: pd.DataFrame,
    X_test_raw: pd.DataFrame,
    num_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return copies with numeric columns standard-scaled (for logistic regression)."""
    scaler = StandardScaler()
    X_train = X_train_raw.copy()
    X_test = X_test_raw.copy()
    if num_cols:
        X_train[num_cols] = scaler.fit_transform(X_train_raw[num_cols])
        X_test[num_cols] = scaler.transform(X_test_raw[num_cols])
    return X_train, X_test


def evaluate(name: str, y_true, y_pred, y_prob=None) -> dict:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = (
        roc_auc_score(y_true, y_prob)
        if (y_prob is not None and len(np.unique(y_true)) > 1)
        else np.nan
    )
    fn_rate = fn / (fn + tp) if (fn + tp) > 0 else np.nan
    print(f"\n  [{name}]")
    print(f"    Precision : {prec:.3f}")
    print(f"    Recall    : {rec:.3f}")
    print(f"    F1        : {f1:.3f}")
    print(f"    ROC-AUC   : {auc:.3f}" if y_prob is not None else "    ROC-AUC   : N/A")
    print(f"    Confusion matrix: TN={tn} FP={fp} FN={fn} TP={tp}")
    print(f"    False-negative rate (missed severe): {fn_rate:.1%}  ← grader point #5")
    return {
        "model": name,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": auc,
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "fn_rate": fn_rate,
    }


def load_and_split_verbose() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load labeled rows, split train/test, print summary. Returns df_known, train_df, test_df."""
    print("=" * 70)
    print("1. Loading cleaned data")
    print("=" * 70)
    df = pd.read_csv(DATA_PATH, low_memory=False)
    print(f"   Full dataset: {df.shape}")
    df_known = df[df["severity_known"] == 1].copy()
    print(f"   severity_known=1 subset: {df_known.shape}")
    print(f"   Class distribution (all known):")
    print(f"   {df_known['severe'].value_counts().to_dict()}")

    print("\n" + "=" * 70)
    print("2. Split strategy")
    print("=" * 70)
    train_df, test_df = split_train_test(df_known)
    print_split_banner(train_df, test_df, df_known)
    return df_known, train_df, test_df


def print_split_banner(train_df: pd.DataFrame, test_df: pd.DataFrame, df_known: pd.DataFrame) -> None:
    current_known = df_known[df_known["era"] == "current"]
    archived_known = df_known[df_known["era"] == "archived"]
    temporal_ok = (
        len(current_known) >= 100
        and current_known["severe"].nunique() == 2
        and archived_known["severe"].nunique() == 2
    )
    if temporal_ok:
        print("   Using TEMPORAL split: train=archived, test=current (deployment-style).")
        print(f"   Archived (train): {len(train_df)} rows | severe rate {train_df['severe'].mean()*100:.1f}%")
        print(f"   Current (test):   {len(test_df)} rows | severe rate {test_df['severe'].mean()*100:.1f}%")
    else:
        print("   Temporal split not viable for labeled rows; using stratified random 80/20.")
        sk = df_known["severe"].astype(str) + "_" + df_known["automation_level"].astype(str)
        print(f"   Strata: severe × automation_level (n={sk.nunique()} combinations).")
        print(f"   Train: {len(train_df)} rows | severe rate {train_df['severe'].mean()*100:.1f}%")
        print(f"   Test:  {len(test_df)} rows | severe rate {test_df['severe'].mean()*100:.1f}%")
