"""
Train separate logistic-regression models for ADS and L2.

Purpose:
- Address pooled-model failure mode where ADS severe cases are missed.
- Keep evaluation deployment-style (temporal split) within each automation level when viable.

Outputs:
- Modeling/lr_stratified_by_level_results.csv
- Modeling/lr_ads_coefficients.csv
- Modeling/lr_l2_coefficients.csv
- Presentation/figures/13_lr_pooled_vs_stratified_by_level.png
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

import baseline_common as bc

# Writable matplotlib config for sandbox/CI environments.
_MPL = bc.PROJECT_ROOT / ".mplconfig"
_MPL.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL))
os.environ.setdefault("XDG_CACHE_HOME", str(_MPL))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def split_within_level(level_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Temporal split within one automation level; fallback to stratified random."""
    curr = level_df[level_df["era"] == "current"]
    arch = level_df[level_df["era"] == "archived"]
    temporal_ok = (
        len(curr) >= 50
        and curr["severe"].nunique() == 2
        and arch["severe"].nunique() == 2
    )
    if temporal_ok:
        return arch.copy(), curr.copy(), "temporal"

    train_df, test_df = train_test_split(
        level_df,
        test_size=0.2,
        random_state=42,
        stratify=level_df["severe"],
    )
    return train_df.copy(), test_df.copy(), "stratified_random"


def run_level(df_known: pd.DataFrame, level: str) -> tuple[dict, pd.DataFrame]:
    level_df = df_known[df_known["automation_level"] == level].copy()
    train_df, test_df, split_type = split_within_level(level_df)

    feats = [f for f in bc.context_features(df_known) if f != "automation_level"]
    X_train_raw, num_cols, train_cols = bc.prepare_X(train_df, feats)
    X_test_raw, _, _ = bc.prepare_X(test_df, feats, fit_encoder=train_cols)
    X_train, X_test = bc.scale_for_lr(X_train_raw, X_test_raw, num_cols)

    y_train = train_df["severe"].values
    y_test = test_df["severe"].values

    lr = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=1000,
        random_state=42,
        solver="lbfgs",
    )
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    y_prob = lr.predict_proba(X_test)[:, 1]

    metrics = bc.evaluate(f"LR stratified [{level}]", y_test, y_pred, y_prob)
    metrics.update(
        {
            "automation_level": level,
            "split_type": split_type,
            "train_n": len(train_df),
            "test_n": len(test_df),
            "train_severe_rate": float(np.mean(y_train)),
            "test_severe_rate": float(np.mean(y_test)),
        }
    )

    coef_df = pd.DataFrame(
        {
            "feature": train_cols,
            "coefficient": lr.coef_[0],
            "odds_ratio": np.exp(lr.coef_[0]),
        }
    ).sort_values("coefficient", key=abs, ascending=False)
    return metrics, coef_df


def make_comparison_plot(out_df: pd.DataFrame, pooled: pd.DataFrame) -> Path:
    fig_dir = bc.PROJECT_ROOT / "Presentation" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    merged = []
    for level in ["ADS", "L2"]:
        pooled_row = pooled[pooled["model"] == f"LR — test [{level}]"]
        strat_row = out_df[out_df["automation_level"] == level]
        if len(pooled_row) and len(strat_row):
            merged.append(
                {
                    "automation_level": level,
                    "pooled_recall": float(pooled_row.iloc[0]["recall"]),
                    "pooled_f1": float(pooled_row.iloc[0]["f1"]),
                    "pooled_fn_rate": float(pooled_row.iloc[0]["fn_rate"]),
                    "strat_recall": float(strat_row.iloc[0]["recall"]),
                    "strat_f1": float(strat_row.iloc[0]["f1"]),
                    "strat_fn_rate": float(strat_row.iloc[0]["fn_rate"]),
                }
            )
    cmp_df = pd.DataFrame(merged)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))
    metrics = [("recall", "Recall"), ("f1", "F1"), ("fn_rate", "FN rate")]
    for ax, (m, title) in zip(axes, metrics):
        x = np.arange(len(cmp_df))
        w = 0.35
        ax.bar(x - w / 2, cmp_df[f"pooled_{m}"], width=w, label="Pooled LR")
        ax.bar(x + w / 2, cmp_df[f"strat_{m}"], width=w, label="Stratified LR")
        ax.set_xticks(x)
        ax.set_xticklabels(cmp_df["automation_level"])
        ax.set_ylim(0, 1.05)
        ax.set_title(title)
        if m == "recall":
            ax.legend(fontsize=9)
    fig.suptitle("Pooled vs separate-per-level LR models", fontsize=13, fontweight="semibold")
    fig.tight_layout()
    out = fig_dir / "13_lr_pooled_vs_stratified_by_level.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    print("=" * 70)
    print("Stratified LR by automation level")
    print("=" * 70)

    df_known = bc.load_known()
    rows = []

    ads_metrics, ads_coef = run_level(df_known, "ADS")
    l2_metrics, l2_coef = run_level(df_known, "L2")
    rows.extend([ads_metrics, l2_metrics])

    out_df = pd.DataFrame(rows)
    out_csv = bc.OUT_DIR / "lr_stratified_by_level_results.csv"
    out_df.to_csv(out_csv, index=False)
    ads_coef.to_csv(bc.OUT_DIR / "lr_ads_coefficients.csv", index=False)
    l2_coef.to_csv(bc.OUT_DIR / "lr_l2_coefficients.csv", index=False)

    pooled_path = bc.OUT_DIR / "baseline_results.csv"
    pooled_df = pd.read_csv(pooled_path) if pooled_path.exists() else pd.DataFrame()
    plot_path = None
    if not pooled_df.empty:
        plot_path = make_comparison_plot(out_df, pooled_df)

    print("\nSaved outputs:")
    print(f"  {out_csv}")
    print(f"  {bc.OUT_DIR / 'lr_ads_coefficients.csv'}")
    print(f"  {bc.OUT_DIR / 'lr_l2_coefficients.csv'}")
    if plot_path is not None:
        print(f"  {plot_path}")


if __name__ == "__main__":
    main()
