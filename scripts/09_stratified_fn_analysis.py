"""
Structured false-negative analysis for the stratified models.

Reproduces the best model per automation level (XGB for ADS, LR for L2),
identifies missed severe crashes, and profiles them by crash context.

Outputs:
  Modeling/logistic_regression/fn_xgb_ads.csv
  Modeling/logistic_regression/fn_lr_l2.csv
  Modeling/logistic_regression/fn_summary.csv
  Presentation/figures/17_fn_analysis_stratified.png
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

import baseline_common as bc
import narrative_utils as nu

_MPL = bc.PROJECT_ROOT / ".mplconfig"
_MPL.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL))
os.environ.setdefault("XDG_CACHE_HOME", str(_MPL))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def build_X(train, test, features):
    num_cols = [f for f in features if pd.api.types.is_numeric_dtype(train[f])]
    cat_cols  = [f for f in features if f not in num_cols]
    Xtr = train[features].copy(); Xte = test[features].copy()
    Xtr[num_cols] = Xtr[num_cols].fillna(Xtr[num_cols].median())
    Xte[num_cols] = Xte[num_cols].fillna(Xtr[num_cols].median())
    Xtr[cat_cols] = Xtr[cat_cols].fillna("Unknown")
    Xte[cat_cols] = Xte[cat_cols].fillna("Unknown")
    Xtr = pd.get_dummies(Xtr, columns=cat_cols, drop_first=False).astype(float)
    Xte = pd.get_dummies(Xte, columns=cat_cols, drop_first=False).astype(float)
    Xte = Xte.reindex(columns=Xtr.columns, fill_value=0.0)
    sc  = StandardScaler()
    if num_cols:
        idx = [Xtr.columns.get_loc(c) for c in num_cols if c in Xtr.columns]
        Xtr.iloc[:, idx] = sc.fit_transform(Xtr.iloc[:, idx])
        Xte.iloc[:, idx] = sc.transform(Xte.iloc[:, idx])
    return Xtr, Xte


def profile_fn(fn_df: pd.DataFrame, label: str) -> None:
    """Print top breakdown categories for false-negative crashes."""
    print(f"\n  [{label}] — {len(fn_df)} missed severe crashes")
    for col in ["Roadway Type", "Crash With", "SV Pre-Crash Movement",
                "CP Pre-Crash Movement", "automation_level"]:
        if col in fn_df.columns:
            top = fn_df[col].value_counts().head(5)
            print(f"\n    {col}:")
            for v, c in top.items():
                print(f"      {v:<40} {c:>4} ({c/len(fn_df)*100:.0f}%)")


def make_fn_figure(fn_ads: pd.DataFrame, fn_l2: pd.DataFrame) -> None:
    fig_dir = bc.PROJECT_ROOT / "Presentation" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("False-negative analysis — missed severe crashes (stratified models)",
                 fontsize=13, fontweight="semibold")

    plot_specs = [
        (axes[0][0], fn_ads, "Roadway Type",        "XGB–ADS missed: Roadway Type"),
        (axes[0][1], fn_ads, "Crash With",           "XGB–ADS missed: Crash With"),
        (axes[1][0], fn_l2,  "Roadway Type",         "LR–L2 missed: Roadway Type"),
        (axes[1][1], fn_l2,  "SV Pre-Crash Movement","LR–L2 missed: SV Pre-Crash Movement"),
    ]
    for ax, df, col, title in plot_specs:
        if col not in df.columns or df.empty:
            ax.set_visible(False)
            continue
        s = df[col].value_counts().head(7)
        ax.barh(s.index[::-1], s.values[::-1], color="#C44E52", alpha=0.85)
        ax.set_title(title, fontsize=11, fontweight="semibold")
        ax.set_xlabel("Count")
        for i, v in enumerate(s.values[::-1]):
            ax.text(v + 0.1, i, str(v), va="center", fontsize=9)

    plt.tight_layout()
    out = fig_dir / "17_fn_analysis_stratified.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n   Figure: {out}")


def main() -> None:
    print("=" * 70)
    print("False-negative analysis — stratified models")
    print("=" * 70)

    df_known = bc.load_known()
    df_known = nu.attach_narrative_flags(df_known)

    tab_feats = [f for f in bc.context_features(df_known) if f != "automation_level"]
    ads_feats = tab_feats + nu.NAV_FEATURES

    summary_rows = []

    # ------------------------------------------------------------------
    # XGB — ADS
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Reproducing XGB [ADS] to extract false negatives")
    print("=" * 70)

    ads_df   = df_known[df_known["automation_level"] == "ADS"].copy()
    train_df = ads_df[ads_df["era"] == "archived"].copy()
    test_df  = ads_df[ads_df["era"] == "current"].copy()

    sub_train, val_df = train_test_split(train_df, test_size=0.25,
                                         random_state=42, stratify=train_df["severe"])
    X_sub, X_val     = build_X(sub_train, val_df,  ads_feats)
    X_train, X_test  = build_X(train_df,  test_df, ads_feats)
    y_sub  = sub_train["severe"].values
    y_val  = val_df["severe"].values
    y_train = train_df["severe"].values
    y_test  = test_df["severe"].values

    neg = int((y_sub == 0).sum()); pos = int((y_sub == 1).sum())
    spw = (neg / max(pos, 1)) / 2
    xgb = XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=5,
                        scale_pos_weight=spw, eval_metric="logloss",
                        random_state=42, verbosity=0)
    xgb.fit(X_train, y_train)
    prob   = xgb.predict_proba(X_test)[:, 1]
    y_pred = (prob >= 0.46).astype(int)

    fn_mask   = (y_pred == 0) & (y_test == 1)
    fn_ads_df = test_df[fn_mask].copy()
    fn_ads_df["y_prob"] = prob[fn_mask]

    print(f"   Missed severe ADS crashes: {fn_mask.sum()} / {y_test.sum()}")
    profile_fn(fn_ads_df, "XGB [ADS]")
    fn_ads_df.to_csv(bc.LR_DIR / "fn_xgb_ads.csv", index=False)

    summary_rows.append({
        "model": "XGB [ADS]", "total_severe_test": int(y_test.sum()),
        "FN": int(fn_mask.sum()), "fn_rate": fn_mask.sum() / max(y_test.sum(), 1),
        "top_roadway": fn_ads_df["Roadway Type"].value_counts().index[0]
            if "Roadway Type" in fn_ads_df.columns and len(fn_ads_df) else "N/A",
        "top_crash_with": fn_ads_df["Crash With"].value_counts().index[0]
            if "Crash With" in fn_ads_df.columns and len(fn_ads_df) else "N/A",
    })

    # ------------------------------------------------------------------
    # LR — L2
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Reproducing LR [L2] to extract false negatives")
    print("=" * 70)

    l2_df    = df_known[df_known["automation_level"] == "L2"].copy()
    train_l2 = l2_df[l2_df["era"] == "archived"].copy()
    test_l2  = l2_df[l2_df["era"] == "current"].copy()

    sub_l2, val_l2   = train_test_split(train_l2, test_size=0.25,
                                        random_state=42, stratify=train_l2["severe"])
    X_sub_l2, _      = build_X(sub_l2,   val_l2,  tab_feats)
    X_train_l2, X_test_l2 = build_X(train_l2, test_l2, tab_feats)
    y_train_l2 = train_l2["severe"].values
    y_test_l2  = test_l2["severe"].values

    lr = LogisticRegression(C=0.1, class_weight="balanced", max_iter=2000,
                            random_state=42, solver="lbfgs")
    lr.fit(X_train_l2, y_train_l2)
    prob_l2   = lr.predict_proba(X_test_l2)[:, 1]
    y_pred_l2 = (prob_l2 >= 0.20).astype(int)

    fn_mask_l2 = (y_pred_l2 == 0) & (y_test_l2 == 1)
    fn_l2_df   = test_l2[fn_mask_l2].copy()
    fn_l2_df["y_prob"] = prob_l2[fn_mask_l2]

    print(f"   Missed severe L2 crashes: {fn_mask_l2.sum()} / {y_test_l2.sum()}")
    profile_fn(fn_l2_df, "LR [L2]")
    fn_l2_df.to_csv(bc.LR_DIR / "fn_lr_l2.csv", index=False)

    summary_rows.append({
        "model": "LR [L2]", "total_severe_test": int(y_test_l2.sum()),
        "FN": int(fn_mask_l2.sum()), "fn_rate": fn_mask_l2.sum() / max(y_test_l2.sum(), 1),
        "top_roadway": fn_l2_df["Roadway Type"].value_counts().index[0]
            if "Roadway Type" in fn_l2_df.columns and len(fn_l2_df) else "N/A",
        "top_crash_with": fn_l2_df["Crash With"].value_counts().index[0]
            if "Crash With" in fn_l2_df.columns and len(fn_l2_df) else "N/A",
    })

    pd.DataFrame(summary_rows).to_csv(bc.LR_DIR / "fn_summary.csv", index=False)
    print("\n" + "=" * 70)
    print("FN Summary")
    print("=" * 70)
    print(pd.DataFrame(summary_rows).to_string(index=False))

    make_fn_figure(fn_ads_df, fn_l2_df)
    print(f"\nSaved: {bc.LR_DIR / 'fn_xgb_ads.csv'}")
    print(f"Saved: {bc.LR_DIR / 'fn_lr_l2.csv'}")
    print(f"Saved: {bc.LR_DIR / 'fn_summary.csv'}")


if __name__ == "__main__":
    main()
