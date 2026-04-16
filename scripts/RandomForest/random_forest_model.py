"""
ISYE 4600 AV Crash Severity Project — Random Forest Classifier
===============================================================

Run after 01_clean_incidents.py (requires Cleaned/sgo_cleaned_incidents.csv).

What this script does
---------------------
1. Loads and splits data using the same temporal strategy as the LR and XGBoost
   baselines (archived → train, current → test) so all three models are directly
   comparable.
2. Trains a Random Forest binary classifier (severe vs. not severe) with
   class_weight="balanced" to handle class imbalance — the same mechanism
   as the Logistic Regression baseline.
3. Tunes key hyperparameters via cross-validated grid search on the training set.
4. Evaluates on the held-out test set overall and by automation level (ADS / L2).
5. Exports three importance measures unique to Random Forest:
     - Gini (mean decrease in impurity) — built-in to sklearn RF
     - Permutation importance — model-agnostic, measures actual test-set impact
     - Mean |SHAP| — most rigorous, locally-grounded, cross-model comparable
6. Reports the Out-of-Bag (OOB) score — an internal validation estimate unique
   to Random Forest, produced at no extra computation cost.
7. Runs a false-negative analysis mirroring the LR and XGBoost outputs.

Outputs (all written to Modeling/)
------------------------------------
  rf_results.csv              — precision / recall / F1 / ROC-AUC / TP/FP/FN/TN / fn_rate
                                (overall + ADS slice + L2 slice)
  rf_feature_importance.csv   — gini / permutation_mean / permutation_std / mean_abs_shap
  rf_false_negatives.csv      — rows misclassified as not-severe (FN)
  rf_best_params.csv          — winning hyperparameter combination from grid search
  rf_oob_score.csv            — out-of-bag accuracy estimate (RF-unique diagnostic)

How Random Forest differs from XGBoost (and why it matters here)
-----------------------------------------------------------------
  RF is a BAGGING method: many deep trees trained INDEPENDENTLY on bootstrap
  samples of the data. Each tree sees ~63% of training rows (the rest are
  "out-of-bag") and a random subset of features at each split. Predictions
  are averaged across all trees.

  XGBoost is a BOOSTING method: trees are trained SEQUENTIALLY, each one
  correcting the residual errors of all previous trees.

  Practical implication for your data:
  - RF is more robust to noisy / irrelevant features because of the random
    feature subsetting at each split (max_features).
  - XGBoost tends to be more accurate when the signal is strong, but can
    over-fit to reporting artifacts in small subgroups (like your ADS slice).
  - RF's OOB score gives a free, unbiased performance estimate on training
    data — useful for sanity-checking without touching the test set.
  - Both are non-linear and capture feature interactions; neither requires scaling.

Why no feature scaling?
-----------------------
Like XGBoost, RF splits on thresholds rather than distances.
StandardScaler is unnecessary and omitted.

Dependencies
------------
  pip install scikit-learn shap pandas numpy
  (scikit-learn is already in your requirements.txt; only shap may be new)
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
SCRIPT_DIR   = Path(__file__).resolve().parent       # scripts/RandomForest/
PROJECT_ROOT = SCRIPT_DIR.parent.parent              # repo root

# Add scripts/ to sys.path so baseline_common is importable
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import baseline_common as bc                         # noqa: E402

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Lazy import for SHAP
# ---------------------------------------------------------------------------
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("[WARNING] shap not installed — SHAP values will be skipped.")
    print("          Run: pip install shap  to enable full interpretability output.\n")

from sklearn.ensemble import RandomForestClassifier              # noqa: E402
from sklearn.inspection import permutation_importance            # noqa: E402
from sklearn.model_selection import GridSearchCV, StratifiedKFold  # noqa: E402

OUT_DIR = bc.OUT_DIR   # Modeling/


# ===========================================================================
# 1. Data loading and splitting
# ===========================================================================

def load_and_split() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load severity-known rows and apply the same temporal split used by the LR
    and XGBoost baselines (archived era → train, current era → test).
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
    One-hot encode categoricals. Random Forest does not require scaling.
    Returns X_train, X_test (DataFrames), y_train, y_test (arrays), feature_names.
    """
    print("\n" + "=" * 70)
    print("2. Building feature matrices")
    print("=" * 70)

    feats = bc.context_features(df_known)
    print(f"   Base feature set     : {len(feats)} columns")

    X_train, num_cols, train_cols = bc.prepare_X(train_df, feats)
    X_test,  _,        _          = bc.prepare_X(test_df, feats, fit_encoder=train_cols)

    print(f"   X_train              : {X_train.shape}")
    print(f"   X_test               : {X_test.shape}")
    print(f"   One-hot encoded cols : {len(train_cols)}")

    y_train = train_df["severe"].values
    y_test  = test_df["severe"].values

    return X_train, X_test, y_train, y_test, train_cols


# ===========================================================================
# 3. Hyperparameter tuning via cross-validated grid search
# ===========================================================================

PARAM_GRID = {
    # Number of trees. More trees = more stable estimates; diminishing returns
    # beyond ~300 for most datasets this size.
    "n_estimators": [100, 300],

    # Maximum depth of each tree. None = fully grown (can overfit);
    # bounded depth adds regularisation.
    "max_depth": [None, 10, 20],

    # Number of features considered at each split.
    # "sqrt" (default) = sqrt(n_features) — the classic RF recommendation.
    # 0.5 = half the features — more randomness, sometimes better generalisation.
    "max_features": ["sqrt", 0.5],

    # Minimum samples required to split an internal node.
    # Higher values prevent overfitting on small subgroups (important for ADS slice).
    "min_samples_split": [2, 10],
}


def tune_hyperparameters(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
) -> tuple[dict, RandomForestClassifier]:
    """
    GridSearchCV over PARAM_GRID with 5-fold stratified CV.
    Optimises for ROC-AUC. The fitted best estimator has oob_score=True
    so the OOB score is available immediately after fitting.
    """
    print("\n" + "=" * 70)
    print("3. Hyperparameter tuning (5-fold stratified CV, scoring=roc_auc)")
    print("=" * 70)

    n_combinations = 1
    for v in PARAM_GRID.values():
        n_combinations *= len(v)
    print(f"   Grid size            : {n_combinations} combinations × 5 folds = {n_combinations*5} fits")
    print("   This may take 1–3 minutes depending on your machine…\n")

    base_rf = RandomForestClassifier(
        class_weight="balanced",   # mirrors LR's class_weight="balanced"
        oob_score=True,            # enable out-of-bag scoring — RF-unique feature
        random_state=42,
        n_jobs=-1,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(
        base_rf,
        PARAM_GRID,
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1,
        verbose=1,
        refit=True,
    )
    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_
    best_model  = grid_search.best_estimator_

    print(f"\n   Best params          : {best_params}")
    print(f"   Best CV ROC-AUC      : {grid_search.best_score_:.4f}")

    # OOB score — available because oob_score=True was set on the base estimator.
    # This is an unbiased estimate of generalisation error computed for free during
    # training: each tree predicts only on the ~37% of rows it never saw.
    # It does NOT use the held-out test set, so it is safe to report here.
    if hasattr(best_model, "oob_score_"):
        print(f"   OOB accuracy (train) : {best_model.oob_score_:.4f}  ← RF-unique diagnostic")

    return best_params, best_model


# ===========================================================================
# 4. Evaluation (overall + ADS/L2 slices)
# ===========================================================================

def evaluate_model(
    model: RandomForestClassifier,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    test_df: pd.DataFrame,
) -> tuple[list[dict], pd.DataFrame]:
    """
    Evaluate on full test set and on ADS / L2 automation-level slices.
    Structure mirrors logistic_regression_baseline.py and xgboost_model.py
    exactly so all three results CSVs have identical columns.
    """
    print("\n" + "=" * 70)
    print("4. Evaluation on test set")
    print("=" * 70)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    results: list[dict] = []
    results.append(bc.evaluate("Random Forest", y_test, y_pred, y_prob))

    test_aug = test_df.copy()
    test_aug["y_pred_rf"] = y_pred
    test_aug["y_prob_rf"] = y_prob

    print("\n" + "=" * 70)
    print("   Automation-level slices (ADS vs L2)")
    print("=" * 70)

    for lvl in ["ADS", "L2"]:
        mask = test_aug["automation_level"] == lvl
        if mask.sum() == 0:
            print(f"   [{lvl}] — no rows in test set, skipping.")
            continue
        sub  = test_aug[mask]
        y_t  = sub["severe"].values
        y_p  = sub["y_pred_rf"].values
        y_pr = sub["y_prob_rf"].values
        if len(np.unique(y_t)) > 1:
            results.append(bc.evaluate(f"RF — test [{lvl}]", y_t, y_p, y_pr))
        else:
            print(f"   [{lvl}] only one class present — metrics not meaningful, skipping.")

    return results, test_aug


# ===========================================================================
# 5. Feature importance (Gini + Permutation + SHAP)
# ===========================================================================

def compute_importance(
    model: RandomForestClassifier,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    feature_names: list[str],
) -> pd.DataFrame:
    """
    Compute three importance measures available for Random Forest.

    Gini importance (mean decrease in impurity)
    --------------------------------------------
    Computed during training. At each split, the chosen feature reduces the
    Gini impurity by some amount; RF averages these reductions across all
    trees. Fast and free, but biased toward high-cardinality and numeric
    features. Analogous to XGBoost's 'gain'.

    Permutation importance
    ----------------------
    After training, each feature's values are randomly shuffled on the TEST
    set and the drop in ROC-AUC is measured. A large drop = the model relied
    heavily on that feature for real predictions. This is model-agnostic and
    avoids the high-cardinality bias of Gini. No equivalent in the LR output,
    but conceptually comparable to |coefficient| size.
    Returns mean and std across 5 repeated permutations.

    Mean |SHAP| (if shap installed)
    --------------------------------
    Uses TreeExplainer (same as XGBoost). For each test-set prediction, SHAP
    decomposes the output into additive feature contributions. mean |SHAP| is
    the most rigorous cross-model importance measure — directly comparable
    between RF, XGBoost, and (with KernelExplainer) LR.
    """
    print("\n" + "=" * 70)
    print("5. Feature importance")
    print("=" * 70)

    # --- Gini importance ---------------------------------------------------
    print("   Computing Gini (mean decrease in impurity)…")
    gini_imp = pd.Series(model.feature_importances_, index=feature_names, name="gini")

    # --- Permutation importance -------------------------------------------
    print("   Computing permutation importance on test set (5 repeats)…")
    perm = permutation_importance(
        model, X_test, y_test,
        n_repeats=5,
        random_state=42,
        scoring="roc_auc",
        n_jobs=-1,
    )
    perm_mean = pd.Series(perm.importances_mean, index=feature_names, name="permutation_mean")
    perm_std  = pd.Series(perm.importances_std,  index=feature_names, name="permutation_std")

    imp_df = pd.concat([gini_imp, perm_mean, perm_std], axis=1).reset_index()
    imp_df = imp_df.rename(columns={"index": "feature"})

    # --- SHAP values -------------------------------------------------------
    if SHAP_AVAILABLE:
        print("   Computing SHAP values on test set…")
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X_test)
        # Newer shap versions return a 3D array (n_samples, n_features, n_classes);
        # older versions return a list [class0_array, class1_array].
        if isinstance(shap_vals, list):
            shap_arr = shap_vals[1]          # old API: list → take class 1
        elif shap_vals.ndim == 3:
            shap_arr = shap_vals[:, :, 1]    # new API: 3D → slice class 1
        else:
            shap_arr = shap_vals             # binary single-output fallback
        mean_shap = pd.Series(
            np.abs(shap_arr).mean(axis=0),
            index=feature_names,
            name="mean_abs_shap",
        )
        imp_df = imp_df.merge(
            mean_shap.reset_index().rename(columns={"index": "feature"}),
            on="feature", how="left",
        )
        imp_df["mean_abs_shap"] = imp_df["mean_abs_shap"].fillna(0)
        sort_col = "mean_abs_shap"
        print("   SHAP values computed successfully.")
    else:
        sort_col = "permutation_mean"
        print("   Skipped SHAP (not installed). Sorting by permutation_mean instead.")

    imp_df = imp_df.sort_values(sort_col, ascending=False).reset_index(drop=True)

    print(f"\n   Top 15 features by {sort_col}:")
    print(imp_df.head(15).to_string(index=False))

    return imp_df


# ===========================================================================
# 6. Out-of-Bag score report
# ===========================================================================

def oob_report(model: RandomForestClassifier) -> dict:
    """
    The OOB score is a free internal cross-validation estimate unique to RF.

    How it works: every tree is trained on a bootstrap sample (~63% of rows).
    The remaining ~37% (the 'out-of-bag' rows) are used to evaluate that tree.
    Averaging these per-row estimates gives an unbiased accuracy estimate
    without touching the test set.

    Useful as:
    - A sanity check that the model is learning something during training.
    - An early indicator of overfitting (OOB >> test accuracy = overfit).
    - A free validation signal when the dataset is small.

    XGBoost and Logistic Regression do not produce an OOB score.
    """
    print("\n" + "=" * 70)
    print("6. Out-of-Bag (OOB) score — RF-unique diagnostic")
    print("=" * 70)

    oob = {}
    if hasattr(model, "oob_score_"):
        oob_acc = model.oob_score_
        oob["oob_accuracy"] = oob_acc
        print(f"   OOB accuracy         : {oob_acc:.4f}")
        print("""
   Interpretation
   --------------
   OOB accuracy is computed on training data rows each tree never saw.
   It is an unbiased estimate of generalisation, computed at no extra cost.

   Compare to test-set accuracy for a quick overfitting check:
     OOB ≈ test accuracy  →  model generalises well
     OOB >> test accuracy  →  possible overfitting or distribution shift
     OOB << test accuracy  →  test set may be easier than training set
                               (common with temporal splits)

   XGBoost and Logistic Regression have no equivalent metric.
""")
    else:
        print("   OOB score not available (oob_score may not have been set).")

    return oob


# ===========================================================================
# 7. False-negative analysis
# ===========================================================================

def false_negative_analysis(test_aug: pd.DataFrame) -> pd.DataFrame:
    """
    Identify crashes the model missed (predicted not-severe but were severe).
    Mirrors the FN analysis in logistic_regression_baseline.py and
    xgboost_model.py for direct comparison.
    """
    print("\n" + "=" * 70)
    print("7. False-negative analysis — missed severe crashes")
    print("=" * 70)

    fn_mask = (test_aug["y_pred_rf"] == 0) & (test_aug["severe"] == 1)
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
    print("ISYE 4600 — Random Forest Crash Severity Classifier")
    print("=" * 70 + "\n")

    # --- Load & split -------------------------------------------------------
    df_known, train_df, test_df = load_and_split()

    # --- Feature matrices ---------------------------------------------------
    X_train, X_test, y_train, y_test, feature_names = build_feature_matrices(
        train_df, test_df, df_known
    )

    # --- Tune & fit ---------------------------------------------------------
    best_params, best_model = tune_hyperparameters(X_train, y_train)

    # --- OOB score ----------------------------------------------------------
    oob = oob_report(best_model)

    # --- Evaluate -----------------------------------------------------------
    results, test_aug = evaluate_model(best_model, X_test, y_test, test_df)

    # --- Feature importance -------------------------------------------------
    imp_df = compute_importance(best_model, X_train, X_test, y_test, feature_names)

    # --- False negatives ----------------------------------------------------
    fn_df = false_negative_analysis(test_aug)

    # --- Save outputs -------------------------------------------------------
    print("\n" + "=" * 70)
    print("8. Saving outputs to Modeling/")
    print("=" * 70)

    results_path = OUT_DIR / "rf_results.csv"
    imp_path     = OUT_DIR / "rf_feature_importance.csv"
    fn_path      = OUT_DIR / "rf_false_negatives.csv"
    params_path  = OUT_DIR / "rf_best_params.csv"
    oob_path     = OUT_DIR / "rf_oob_score.csv"

    pd.DataFrame(results).to_csv(results_path, index=False)
    imp_df.to_csv(imp_path, index=False)
    fn_df.to_csv(fn_path, index=False)
    pd.DataFrame([best_params]).to_csv(params_path, index=False)
    pd.DataFrame([oob]).to_csv(oob_path, index=False)

    print(f"   {results_path}")
    print(f"   {imp_path}")
    print(f"   {fn_path}")
    print(f"   {params_path}")
    print(f"   {oob_path}")

    print("\n" + "=" * 70)
    print("How to compare with Logistic Regression and XGBoost")
    print("=" * 70)
    print("""
   DIRECTLY COMPARABLE — same metrics, same test set, same columns:
     rf_results.csv
       ←→  logistic_regression_results.csv
       ←→  xgb_results.csv
     All report: precision, recall, F1, ROC-AUC, TP/FP/FN/TN, fn_rate
     All break out ADS and L2 slices with the same naming convention.

   INTERPRETABILITY — conceptually parallel, different columns:
     rf_feature_importance.csv  (gini / permutation_mean / permutation_std / mean_abs_shap)
       ←→  lr_coefficients.csv  (coefficient / odds_ratio)
       ←→  xgb_feature_importance.csv  (weight / gain / cover / mean_abs_shap)

     Best cross-model comparison:
       mean_abs_shap (RF)  ←→  mean_abs_shap (XGB)  — identical concept, comparable values
       permutation_mean (RF)  ←→  gain (XGB)  — closest analogue, different scale
       gini (RF)  ≈  weight (XGB)  — both count-based, both biased toward frequent features

   RF-UNIQUE OUTPUTS (no equivalent in LR or XGBoost):
     rf_oob_score.csv  — out-of-bag accuracy; free internal validation estimate.
     permutation_std   — uncertainty on each feature's importance (5 repeats).

   FALSE NEGATIVES — same structure across all three models:
     rf_false_negatives.csv
       ←→  false_negatives.csv  (LR)
       ←→  xgb_false_negatives.csv  (XGB)
""")

    print("Done.\n")


if __name__ == "__main__":
    main()


# ===========================================================================
# requirements (add to requirements.txt if not already present)
# ===========================================================================
# scikit-learn>=1.3   (already present — RandomForest is part of sklearn)
# shap>=0.44          (optional but strongly recommended)
# pandas>=2.0
# numpy>=1.24
