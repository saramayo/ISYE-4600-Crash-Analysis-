"""
Tune logistic-regression decision threshold for severe-crash detection.

Workflow:
1) Use the existing temporal split from baseline_common:
   - train_df: archived
   - test_df: current
2) Split archived into sub-train / validation.
3) Fit the same LR baseline on sub-train.
4) Sweep thresholds on validation and select threshold by:
   - max F1 among thresholds with precision >= min_precision
   - fallback to best F1 if no threshold meets precision floor
5) Evaluate selected threshold on current-era test.

Outputs:
- Modeling/lr_threshold_sweep_validation.csv
- Modeling/lr_threshold_selected_metrics.csv
- Presentation/figures/12_lr_threshold_tuning.png
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
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


def metrics_at_threshold(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "threshold": threshold,
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "fn_rate": 1 - recall_score(y_true, y_pred, zero_division=0),
    }


def main(min_precision: float = 0.85) -> None:
    print("=" * 70)
    print("LR threshold tuning")
    print("=" * 70)
    df_known, train_df, test_df = bc.load_and_split_verbose()

    # Validation split is within archived data only.
    strat_key = train_df["severe"].astype(str) + "_" + train_df["automation_level"].astype(str)
    sub_train, val_df = train_test_split(
        train_df,
        test_size=0.25,
        random_state=42,
        stratify=strat_key,
    )
    print(f"\nArchived split for tuning: sub-train={len(sub_train)} | validation={len(val_df)}")

    feats = bc.context_features(df_known)
    X_sub_raw, num_cols, train_cols = bc.prepare_X(sub_train, feats)
    X_val_raw, _, _ = bc.prepare_X(val_df, feats, fit_encoder=train_cols)
    X_test_raw, _, _ = bc.prepare_X(test_df, feats, fit_encoder=train_cols)
    X_sub, X_val = bc.scale_for_lr(X_sub_raw, X_val_raw, num_cols)
    _, X_test = bc.scale_for_lr(X_sub_raw, X_test_raw, num_cols)

    y_sub = sub_train["severe"].values
    y_val = val_df["severe"].values
    y_test = test_df["severe"].values

    lr = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=1000,
        random_state=42,
        solver="lbfgs",
    )
    lr.fit(X_sub, y_sub)
    val_prob = lr.predict_proba(X_val)[:, 1]
    test_prob = lr.predict_proba(X_test)[:, 1]

    thresholds = np.round(np.arange(0.10, 0.91, 0.02), 2)
    sweep = pd.DataFrame([metrics_at_threshold(y_val, val_prob, t) for t in thresholds])
    sweep.to_csv(bc.OUT_DIR / "lr_threshold_sweep_validation.csv", index=False)

    eligible = sweep[sweep["precision"] >= min_precision]
    if len(eligible):
        selected = eligible.sort_values(["f1", "recall"], ascending=False).iloc[0]
        reason = f"best validation F1 with precision >= {min_precision:.2f}"
    else:
        selected = sweep.sort_values(["f1", "recall"], ascending=False).iloc[0]
        reason = f"no threshold met precision >= {min_precision:.2f}; selected best validation F1"
    t_star = float(selected["threshold"])

    print(f"\nSelected threshold: {t_star:.2f} ({reason})")

    default_test = metrics_at_threshold(y_test, test_prob, 0.50)
    tuned_test = metrics_at_threshold(y_test, test_prob, t_star)
    summary = pd.DataFrame(
        [
            {"setting": "default_0.50", **default_test},
            {"setting": f"tuned_{t_star:.2f}", **tuned_test},
        ]
    )
    summary.to_csv(bc.OUT_DIR / "lr_threshold_selected_metrics.csv", index=False)
    print("\nCurrent-era test metrics comparison:")
    print(summary.to_string(index=False))

    # Plot for slides.
    fig_dir = bc.PROJECT_ROOT / "Presentation" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 5.5))
    plt.plot(sweep["threshold"], sweep["precision"], label="Precision")
    plt.plot(sweep["threshold"], sweep["recall"], label="Recall")
    plt.plot(sweep["threshold"], sweep["f1"], label="F1")
    plt.axvline(0.50, color="#777", linestyle="--", linewidth=1, label="Default 0.50")
    plt.axvline(t_star, color="#C44E52", linestyle="--", linewidth=2, label=f"Selected {t_star:.2f}")
    plt.title("Validation threshold sweep (archived split)")
    plt.xlabel("Decision threshold")
    plt.ylabel("Score")
    plt.ylim(0, 1.02)
    plt.legend()
    plt.tight_layout()
    out_plot = fig_dir / "12_lr_threshold_tuning.png"
    plt.savefig(out_plot, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nWrote: {bc.OUT_DIR / 'lr_threshold_sweep_validation.csv'}")
    print(f"Wrote: {bc.OUT_DIR / 'lr_threshold_selected_metrics.csv'}")
    print(f"Wrote: {out_plot}")


if __name__ == "__main__":
    main()
