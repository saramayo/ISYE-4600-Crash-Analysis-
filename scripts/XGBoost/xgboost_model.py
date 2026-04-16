"""
ISYE 4600 AV Crash Severity Project — XGBoost Classifier
=========================================================

Run after 01_clean_incidents.py (requires Cleaned/sgo_cleaned_incidents.csv).

What this script does
---------------------
1. Loads and splits data using the same temporal strategy as the LR baseline
   (archived → train, current → test) so results are directly comparable.
2. Trains an XGBoost binary classifier (severe vs. not severe) with
   scale_pos_weight to handle class imbalance (mirrors LR's class_weight="balanced").
3. Tunes key hyperparameters via cross-validated grid search on the training set.
4. Evaluates on the held-out test set overall and by automation level (ADS / L2).
5. Exports three importance metrics (weight, gain, cover) + SHAP values for
   deep interpretability.
6. Runs a false-negative analysis mirroring the LR output so the two are
   directly comparable.

Outputs (all written to Modeling/)
-----------------------------------
  xgb_results.csv            — precision / recall / F1 / ROC-AUC / TP/FP/FN/TN / fn_rate
                                (overall + ADS slice + L2 slice)
  xgb_feature_importance.csv — weight, gain, cover, and mean |SHAP| per feature
  xgb_false_negatives.csv    — rows misclassified as not-severe (FN)
  xgb_best_params.csv        — winning hyperparameter combination from grid search

Why no feature scaling?
-----------------------
Tree-based models split on thresholds, not distances, so StandardScaler is
unnecessary. Raw numeric values are passed directly.

Dependencies
------------
  pip install xgboost shap scikit-learn pandas numpy
  (or add to requirements.txt — see bottom of this file)
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Path bootstrap — allows running from any working directory
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent          # scripts/XGBoost/
PROJECT_ROOT = SCRIPT_DIR.parent.parent               # repo root

# Add scripts/ to sys.path so baseline_common is importable
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import baseline_common as bc                          # noqa: E402 (import after path fix)

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Lazy imports with helpful error messages
# ---------------------------------------------------------------------------
try:
    from xgboost import XGBClassifier
except ImportError:
    sys.exit(
        "\n[ERROR] xgboost is not installed.\n"
        "  Run:  pip install xgboost  (or .venv/bin/pip install xgboost)\n"
    )

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("[WARNING] shap not installed — SHAP values will be skipped.")
    print("          Run: pip install shap  to enable full interpretability output.\n")

from sklearn.model_selection import GridSearchCV, StratifiedKFold  # noqa: E402

OUT_DIR = bc.XGB_DIR   # Modeling/xgboost/


# ===========================================================================
# 1. Data loading and splitting
# ===========================================================================

def load_and_split() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load severity-known rows and apply the same temporal split used by the LR
    baseline (archived era → train, current era → test).
    """
    print("=" * 70)
    print("1. Loading and splitting data")
    print("=" * 70)

    df = pd.read_csv(bc.DATA_PATH, low_memory=False)
    print(f"   Full dataset         : {df.shape[0]:,} rows × {df.shape[1]} cols")

    df_known = df[df["severity_known"] == 1].copy()
    print(f"   severity_known = 1   : {df_known.shape[0]:,} rows")
    print(f"   Class balance        : {df_known['severe'].value_counts().to_dict()}")
    print(f"   Severe rate          : {df_known['severe'].mean()*100:.1f}%")

    train_df, test_df = bc.split_train_test(df_known)
    bc.print_split_banner(train_df, test_df, df_known)

    return df_known, train_df, test_df


# ===========================================================================
# 2. Feature matrix preparation
# ===========================================================================

def build_feature_matrices(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    df_known: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, list[str]]:
    """
    One-hot encode categoricals. XGBoost does not require scaling.
    Returns X_train, X_test (DataFrames), y_train, y_test (arrays), feature_names.
    """
    print("\n" + "=" * 70)
    print("2. Building feature matrices")
    print("=" * 70)

    feats = bc.context_features(df_known)
    print(f"   Base feature set     : {len(feats)} columns")

    # prepare_X handles NaN imputation + one-hot encoding
    X_train, num_cols, train_cols = bc.prepare_X(train_df, feats)
    X_test, _, _ = bc.prepare_X(test_df, feats, fit_encoder=train_cols)

    # No scaling for trees
    print(f"   X_train              : {X_train.shape}")
    print(f"   X_test               : {X_test.shape}")
    print(f"   One-hot encoded cols : {len(train_cols)}")

    y_train = train_df["severe"].values
    y_test  = test_df["severe"].values

    return X_train, X_test, y_train, y_test, train_cols


# ===========================================================================
# 3. Class-imbalance weight
# ===========================================================================

def compute_scale_pos_weight(y_train: np.ndarray) -> float:
    """
    scale_pos_weight = count(negative class) / count(positive class).

    This is XGBoost's analogue of sklearn's class_weight="balanced".
    It up-weights the minority (severe=1) class during training so the
    model does not simply learn to predict the majority.
    """
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    spw = n_neg / n_pos if n_pos > 0 else 1.0
    print(f"\n   Imbalance ratio (neg/pos) : {spw:.2f}  → scale_pos_weight")
    return spw


# ===========================================================================
# 4. Hyperparameter tuning via cross-validated grid search
# ===========================================================================

PARAM_GRID = {
    # Number of boosting rounds — more trees can mean better fit, but also overfitting.
    "n_estimators": [100, 300],

    # Maximum depth of each tree. Shallow trees are faster and less prone to overfitting;
    # deeper trees capture more complex interactions.
    "max_depth": [3, 5],

    # Learning rate (shrinkage). Smaller = more robust but needs more trees.
    "learning_rate": [0.05, 0.1],

    # Fraction of columns sampled per tree — adds randomness, reduces overfitting.
    "colsample_bytree": [0.7, 1.0],

    # Fraction of training rows sampled per tree.
    "subsample": [0.8, 1.0],
}


def tune_hyperparameters(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    scale_pos_weight: float,
) -> dict:
    """
    GridSearchCV over PARAM_GRID with 5-fold stratified CV.
    Optimises for ROC-AUC (appropriate for imbalanced binary classification).

    Returns the best parameter dictionary.
    """
    print("\n" + "=" * 70)
    print("3. Hyperparameter tuning (5-fold stratified CV, scoring=roc_auc)")
    print("=" * 70)

    n_combinations = 1
    for v in PARAM_GRID.values():
        n_combinations *= len(v)
    print(f"   Grid size            : {n_combinations} combinations × 5 folds = {n_combinations*5} fits")
    print("   This may take 1–3 minutes depending on your machine…\n")

    base_xgb = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        use_label_encoder=False,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(
        base_xgb,
        PARAM_GRID,
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1,
        verbose=1,
        refit=True,
    )
    grid_search.fit(X_train, y_train)

    best = grid_search.best_params_
    print(f"\n   Best params          : {best}")
    print(f"   Best CV ROC-AUC      : {grid_search.best_score_:.4f}")

    return best, grid_search.best_estimator_


# ===========================================================================
# 5. Evaluation (overall + ADS/L2 slices)
# ===========================================================================

def evaluate_model(
    model: XGBClassifier,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    test_df: pd.DataFrame,
) -> tuple[list[dict], pd.DataFrame]:
    """
    Evaluate on full test set and on ADS / L2 automation-level slices.
    Mirrors the slice analysis in logistic_regression_baseline.py exactly
    so both sets of metrics sit in comparable CSVs.

    Returns results list (for CSV) and augmented test_df with predictions.
    """
    print("\n" + "=" * 70)
    print("4. Evaluation on test set")
    print("=" * 70)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    results: list[dict] = []
    results.append(bc.evaluate("XGBoost", y_test, y_pred, y_prob))

    test_aug = test_df.copy()
    test_aug["y_pred_xgb"] = y_pred
    test_aug["y_prob_xgb"] = y_prob

    print("\n" + "=" * 70)
    print("   Automation-level slices (ADS vs L2)")
    print("=" * 70)

    for lvl in ["ADS", "L2"]:
        mask = test_aug["automation_level"] == lvl
        if mask.sum() == 0:
            print(f"   [{lvl}] — no rows in test set, skipping.")
            continue
        sub = test_aug[mask]
        y_t  = sub["severe"].values
        y_p  = sub["y_pred_xgb"].values
        y_pr = sub["y_prob_xgb"].values
        if len(np.unique(y_t)) > 1:
            results.append(bc.evaluate(f"XGB — test [{lvl}]", y_t, y_p, y_pr))
        else:
            print(f"   [{lvl}] only one class present — metrics not meaningful, skipping.")

    return results, test_aug


# ===========================================================================
# 6. Feature importance + SHAP
# ===========================================================================

def compute_importance(
    model: XGBClassifier,
    X_test: pd.DataFrame,
    feature_names: list[str],
) -> pd.DataFrame:
    """
    Collect three built-in XGBoost importance types (weight / gain / cover)
    and, if shap is available, mean absolute SHAP values on the test set.

    Conceptual definitions
    ----------------------
    weight  : how many times a feature is used to split across all trees.
              Simple count; can over-represent high-cardinality features.
    gain    : average loss reduction achieved by splits on this feature.
              The most reliable signal for "how useful is this feature?".
    cover   : average number of training samples that pass through splits
              on this feature — a measure of breadth of influence.
    SHAP    : for each test-set prediction, SHAP decomposes it into the
              additive contribution of each feature. mean |SHAP| is a
              model-agnostic, locally-grounded importance measure that is
              directly comparable across different model families.
    """
    print("\n" + "=" * 70)
    print("5. Feature importance")
    print("=" * 70)

    booster = model.get_booster()

    def get_scores(importance_type: str) -> pd.Series:
        scores = booster.get_score(importance_type=importance_type)
        return pd.Series(scores, name=importance_type)

    imp_weight = get_scores("weight")
    imp_gain   = get_scores("gain")
    imp_cover  = get_scores("cover")

    imp_df = pd.concat([imp_weight, imp_gain, imp_cover], axis=1).fillna(0)
    imp_df.index.name = "feature"
    imp_df = imp_df.reset_index()

    # SHAP values
    if SHAP_AVAILABLE:
        print("   Computing SHAP values on test set…")
        explainer  = shap.TreeExplainer(model)
        shap_vals  = explainer.shap_values(X_test)   # shape: (n_samples, n_features)
        mean_shap  = np.abs(shap_vals).mean(axis=0)
        shap_series = pd.Series(mean_shap, index=feature_names, name="mean_abs_shap")
        imp_df = imp_df.merge(
            shap_series.reset_index().rename(columns={"index": "feature"}),
            on="feature", how="left",
        )
        imp_df["mean_abs_shap"] = imp_df["mean_abs_shap"].fillna(0)
        sort_col = "mean_abs_shap"
        print("   SHAP values computed successfully.")
    else:
        sort_col = "gain"
        print("   Skipped SHAP (not installed). Sorting by gain instead.")

    imp_df = imp_df.sort_values(sort_col, ascending=False).reset_index(drop=True)

    print(f"\n   Top 15 features by {sort_col}:")
    print(imp_df.head(15).to_string(index=False))

    return imp_df


# ===========================================================================
# 7. False-negative analysis
# ===========================================================================

def false_negative_analysis(test_aug: pd.DataFrame) -> pd.DataFrame:
    """
    Identify crashes the model missed (predicted not-severe but were severe).
    Mirrors the FN analysis in logistic_regression_baseline.py.
    """
    print("\n" + "=" * 70)
    print("6. False-negative analysis — missed severe crashes")
    print("=" * 70)

    fn_mask = (test_aug["y_pred_xgb"] == 0) & (test_aug["severe"] == 1)
    fn_df   = test_aug[fn_mask].copy()

    print(f"   Total missed severe crashes (FN) : {fn_mask.sum()}")
    print(f"\n   Top Roadway Types in missed severe crashes:")
    print(fn_df["Roadway Type"].value_counts().head(6).to_string())
    print(f"\n   Top Crash Counterpart in missed severe crashes:")
    print(fn_df["Crash With"].value_counts().head(6).to_string())
    print(f"\n   Automation level breakdown of false negatives:")
    print(fn_df["automation_level"].value_counts().to_string())

    return fn_df


# ===========================================================================
# 8. Main
# ===========================================================================

def main() -> None:
    print("\n" + "=" * 70)
    print("ISYE 4600 — XGBoost Crash Severity Classifier")
    print("=" * 70 + "\n")

    # --- Load & split -------------------------------------------------------
    df_known, train_df, test_df = load_and_split()

    # --- Feature matrices ---------------------------------------------------
    X_train, X_test, y_train, y_test, feature_names = build_feature_matrices(
        train_df, test_df, df_known
    )

    # --- Class weight -------------------------------------------------------
    spw = compute_scale_pos_weight(y_train)

    # --- Tune & fit ---------------------------------------------------------
    best_params, best_model = tune_hyperparameters(X_train, y_train, spw)

    # --- Evaluate -----------------------------------------------------------
    results, test_aug = evaluate_model(best_model, X_test, y_test, test_df)

    # --- Feature importance + SHAP -----------------------------------------
    imp_df = compute_importance(best_model, X_test, feature_names)

    # --- False negatives ----------------------------------------------------
    fn_df = false_negative_analysis(test_aug)

    # --- Save outputs -------------------------------------------------------
    print("\n" + "=" * 70)
    print("7. Saving outputs to Modeling/")
    print("=" * 70)

    results_path = OUT_DIR / "xgb_results.csv"
    imp_path     = OUT_DIR / "xgb_feature_importance.csv"
    fn_path      = OUT_DIR / "xgb_false_negatives.csv"
    params_path  = OUT_DIR / "xgb_best_params.csv"

    pd.DataFrame(results).to_csv(results_path, index=False)
    imp_df.to_csv(imp_path, index=False)
    fn_df.to_csv(fn_path, index=False)
    pd.DataFrame([best_params]).to_csv(params_path, index=False)

    print(f"   {results_path}")
    print(f"   {imp_path}")
    print(f"   {fn_path}")
    print(f"   {params_path}")

    print("\n" + "=" * 70)
    print("How to compare with Logistic Regression")
    print("=" * 70)
    print("""
   DIRECTLY COMPARABLE (same metrics, same test set):
     Modeling/xgb_results.csv  ←→  Modeling/logistic_regression_results.csv
     Both report: precision, recall, F1, ROC-AUC, TP/FP/FN/TN, fn_rate
     Both break out ADS and L2 slices with the same naming convention.

   INTERPRETABILITY (conceptually parallel, different columns):
     xgb_feature_importance.csv  ←→  lr_coefficients.csv
       XGB: weight / gain / cover / mean_abs_shap (if shap installed)
       LR : coefficient / odds_ratio
     Both answer "which features matter most?" but via different lenses.
     gain ≈ closest XGB analogue to LR's |coefficient| ranking.
     mean_abs_shap is the most rigorous cross-model comparison.

   FALSE NEGATIVES (same structure):
     xgb_false_negatives.csv  ←→  false_negatives.csv
     Same breakdown: roadway type, crash counterpart, automation level.
""")

    print("Done.\n")


if __name__ == "__main__":
    main()


# ===========================================================================
# requirements (add to requirements.txt if not already present)
# ===========================================================================
# xgboost>=1.7
# shap>=0.44        # optional but strongly recommended
# scikit-learn>=1.3
# pandas>=2.0
# numpy>=1.24
