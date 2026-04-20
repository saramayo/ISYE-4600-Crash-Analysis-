"""
ISYE 4600 AV Crash Severity Project — Step 6: Narrative scenario features

Extracts interpretable, non-leaky scenario flags from free-text narratives,
combines them with the existing tabular features, and evaluates whether they
improve the ADS-only logistic regression model (which sits at AUC ~0.50 on
tabular features alone).

Key design constraint: NO outcome words (towed, injured, airbag, etc.) are
used as features — those are the label components. Only pre-crash context
and crash dynamics from the narrative body are extracted.

Outputs (written to Modeling/logistic_regression/):
  narrative_feature_descriptions.csv  — what each flag means + prevalence
  narrative_ads_model_comparison.csv  — ADS model: tabular vs tabular+text
  narrative_pooled_model_comparison.csv — pooled model: tabular vs tabular+text
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

import narrative_utils as nu

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "Cleaned" / "sgo_cleaned_incidents.csv"
OUT_DIR = PROJECT_ROOT / "Modeling" / "logistic_regression"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TABULAR_FEATURES = [
    "Automation System Engaged?",
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


def build_X(train: pd.DataFrame, test: pd.DataFrame, features: list[str]):
    num_cols = [f for f in features if pd.api.types.is_numeric_dtype(train[f])]
    cat_cols = [f for f in features if f not in num_cols]

    Xtr = train[features].copy().astype({c: float for c in num_cols if c in train.columns})
    Xte = test[features].copy().astype({c: float for c in num_cols if c in test.columns})

    Xtr[num_cols] = Xtr[num_cols].fillna(Xtr[num_cols].median())
    Xte[num_cols] = Xte[num_cols].fillna(Xtr[num_cols].median())
    Xtr[cat_cols] = Xtr[cat_cols].fillna("Unknown")
    Xte[cat_cols] = Xte[cat_cols].fillna("Unknown")

    Xtr = pd.get_dummies(Xtr, columns=cat_cols, drop_first=False)
    Xte = pd.get_dummies(Xte, columns=cat_cols, drop_first=False)
    Xte = Xte.reindex(columns=Xtr.columns, fill_value=0)

    scaler = StandardScaler()
    if num_cols:
        num_idx = [Xtr.columns.get_loc(c) for c in num_cols if c in Xtr.columns]
        Xtr.iloc[:, num_idx] = scaler.fit_transform(Xtr.iloc[:, num_idx])
        Xte.iloc[:, num_idx] = scaler.transform(Xte.iloc[:, num_idx])

    return Xtr, Xte


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
    print(f"    ROC-AUC   : {auc:.3f}" if not np.isnan(auc) else "    ROC-AUC   : N/A")
    print(f"    TP={tp}  FP={fp}  FN={fn}  TN={tn}")
    print(f"    False-negative rate: {fn_rate:.1%}")
    return dict(model=name, precision=prec, recall=rec, f1=f1, roc_auc=auc,
                TP=tp, FP=fp, FN=fn, TN=tn, fn_rate=fn_rate)


def run_lr(X_train, X_test, y_train, y_test, name: str) -> dict:
    lr = LogisticRegression(class_weight="balanced", max_iter=2000, C=1.0, solver="lbfgs")
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    y_prob = lr.predict_proba(X_test)[:, 1]
    return evaluate(name, y_test, y_pred, y_prob)


def main() -> None:
    print("=" * 70)
    print("1. Loading cleaned data")
    print("=" * 70)

    df = pd.read_csv(DATA_PATH, low_memory=False)
    df_known = df[df["severity_known"] == 1].copy()
    print(f"   severity_known=1 rows: {len(df_known)}")

    df_known = nu.attach_narrative_flags(df_known)

    train_df = df_known[df_known["era"] == "archived"].copy()
    test_df = df_known[df_known["era"] == "current"].copy()
    print(f"   Train (archived): {len(train_df)} | severe rate {train_df['severe'].mean()*100:.1f}%")
    print(f"   Test  (current):  {len(test_df)}  | severe rate {test_df['severe'].mean()*100:.1f}%")

    # ---------------------------------------------------------------------------
    # Narrative feature descriptions
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("2. Narrative flag prevalence")
    print("=" * 70)

    flag_df = df_known[nu.NAV_FEATURES]
    prev = flag_df.mean().sort_values(ascending=False)
    for feat, rate in prev.items():
        print(f"     {feat:<30}  {rate*100:5.1f}%")

    desc_rows = []
    for feat in nu.NAV_FEATURES:
        desc_rows.append({
            "feature": feat,
            "description": nu.NAV_DESCRIPTIONS.get(feat, ""),
            "overall_pct": round(flag_df[feat].mean() * 100, 1),
            "severe_pct": round(df_known.loc[df_known["severe"] == 1, feat].mean() * 100, 1),
            "nonsevere_pct": round(df_known.loc[df_known["severe"] == 0, feat].mean() * 100, 1),
        })
    pd.DataFrame(desc_rows).to_csv(OUT_DIR / "narrative_feature_descriptions.csv", index=False)
    print("\n   Saved narrative_feature_descriptions.csv")

    # ---------------------------------------------------------------------------
    # ADS-only model comparison
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("3. ADS-only model: tabular vs tabular+narrative vs narrative-only")
    print("=" * 70)

    ads_train = train_df[train_df["automation_level"] == "ADS"].copy()
    ads_test = test_df[test_df["automation_level"] == "ADS"].copy()
    print(f"   ADS train: {len(ads_train)} (severe={ads_train['severe'].sum()})")
    print(f"   ADS test:  {len(ads_test)}  (severe={ads_test['severe'].sum()})")

    X_tr_tab, X_te_tab = build_X(ads_train, ads_test, TABULAR_FEATURES)
    r_tab = run_lr(X_tr_tab, X_te_tab, ads_train["severe"], ads_test["severe"],
                   "ADS — tabular only")

    X_tr_comb, X_te_comb = build_X(ads_train, ads_test, TABULAR_FEATURES + nu.NAV_FEATURES)
    r_comb = run_lr(X_tr_comb, X_te_comb, ads_train["severe"], ads_test["severe"],
                    "ADS — tabular + narrative")

    X_tr_nav, X_te_nav = build_X(ads_train, ads_test, nu.NAV_FEATURES)
    r_nav = run_lr(X_tr_nav, X_te_nav, ads_train["severe"], ads_test["severe"],
                   "ADS — narrative flags only")

    ads_results = pd.DataFrame([r_tab, r_comb, r_nav])
    ads_results.to_csv(OUT_DIR / "narrative_ads_model_comparison.csv", index=False)
    print("\n   Saved narrative_ads_model_comparison.csv")

    lr_final = LogisticRegression(class_weight="balanced", max_iter=2000, C=1.0, solver="lbfgs")
    lr_final.fit(X_tr_comb, ads_train["severe"])
    coef_df = pd.DataFrame({
        "feature": X_tr_comb.columns,
        "coefficient": lr_final.coef_[0],
    }).sort_values("coefficient", ascending=False)
    nav_coefs = coef_df[coef_df["feature"].str.startswith("nav_")]
    print("\n   Narrative feature coefficients (combined ADS model):")
    print(nav_coefs.to_string(index=False))

    # ---------------------------------------------------------------------------
    # Pooled model comparison
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("4. Pooled model: tabular (no automation_level) vs + narrative features")
    print("=" * 70)

    TABULAR_NO_LEVEL = [f for f in TABULAR_FEATURES if f != "automation_level"]

    X_tr_p, X_te_p = build_X(train_df, test_df, TABULAR_NO_LEVEL)
    r_pooled_tab = run_lr(X_tr_p, X_te_p, train_df["severe"], test_df["severe"],
                          "Pooled — tabular (no automation_level)")

    X_tr_pn, X_te_pn = build_X(train_df, test_df, TABULAR_NO_LEVEL + nu.NAV_FEATURES)
    r_pooled_comb = run_lr(X_tr_pn, X_te_pn, train_df["severe"], test_df["severe"],
                           "Pooled — tabular + narrative (no automation_level)")

    pooled_results = pd.DataFrame([r_pooled_tab, r_pooled_comb])
    pooled_results.to_csv(OUT_DIR / "narrative_pooled_model_comparison.csv", index=False)
    print("\n   Saved narrative_pooled_model_comparison.csv")

    # ---------------------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("5. Summary")
    print("=" * 70)
    print(f"\n  ADS model improvement (ROC-AUC):")
    print(f"    Tabular only:           {r_tab['roc_auc']:.3f}")
    print(f"    Tabular + narrative:    {r_comb['roc_auc']:.3f}  "
          f"(Δ = {r_comb['roc_auc'] - r_tab['roc_auc']:+.3f})")
    print(f"    Narrative flags only:   {r_nav['roc_auc']:.3f}")
    print(f"\n  ADS false-negative rate:")
    print(f"    Tabular only:           {r_tab['fn_rate']:.1%}")
    print(f"    Tabular + narrative:    {r_comb['fn_rate']:.1%}")
    print(f"\n  Pooled model (no automation_level) ROC-AUC:")
    print(f"    Tabular only:           {r_pooled_tab['roc_auc']:.3f}")
    print(f"    Tabular + narrative:    {r_pooled_comb['roc_auc']:.3f}")
    print("\nDone.")


if __name__ == "__main__":
    main()
