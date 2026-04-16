"""
Tuned logistic regression for severe-crash prediction.

Improvements over baseline:
- Hyperparameter search on archived sub-train/validation
- Positive-class weighting search (to reduce missed severe crashes)
- Threshold search on validation probabilities

Evaluation remains deployment-style:
- Final fit on full archived train
- Test on current era
"""
from __future__ import annotations

from itertools import product

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

import baseline_common as bc


def metric_bundle(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    row = bc.evaluate("tmp", y_true, y_pred, y_prob)
    return row


def main() -> None:
    print("=" * 70)
    print("Tuned logistic regression")
    print("=" * 70)

    df_known, train_df, test_df = bc.load_and_split_verbose()
    feats = bc.context_features(df_known)

    # Validation split only inside archived train data.
    strat_key = train_df["severe"].astype(str) + "_" + train_df["automation_level"].astype(str)
    sub_train, val_df = train_test_split(
        train_df,
        test_size=0.25,
        random_state=42,
        stratify=strat_key,
    )
    print(f"\nArchived tuning split: sub-train={len(sub_train)} | validation={len(val_df)}")

    X_sub_raw, num_cols, train_cols = bc.prepare_X(sub_train, feats)
    X_val_raw, _, _ = bc.prepare_X(val_df, feats, fit_encoder=train_cols)
    X_sub, X_val = bc.scale_for_lr(X_sub_raw, X_val_raw, num_cols)
    y_sub = sub_train["severe"].values
    y_val = val_df["severe"].values

    # Search space: C and severe-class weight.
    c_grid = [0.1, 0.3, 1.0, 3.0, 10.0]
    pos_w_grid = [1.0, 1.5, 2.0, 3.0]
    thresholds = np.round(np.arange(0.20, 0.81, 0.02), 2)

    best = None
    search_rows: list[dict] = []
    for c, pos_w in product(c_grid, pos_w_grid):
        lr = LogisticRegression(
            C=c,
            class_weight={0: 1.0, 1: pos_w},
            max_iter=2000,
            random_state=42,
            solver="lbfgs",
        )
        lr.fit(X_sub, y_sub)
        val_prob = lr.predict_proba(X_val)[:, 1]

        for t in thresholds:
            y_pred = (val_prob >= t).astype(int)
            prec = float((np.sum((y_pred == 1) & (y_val == 1)) / max(np.sum(y_pred == 1), 1)))
            rec = float((np.sum((y_pred == 1) & (y_val == 1)) / max(np.sum(y_val == 1), 1)))
            f1 = 0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)
            row = {
                "C": c,
                "pos_weight": pos_w,
                "threshold": float(t),
                "precision": prec,
                "recall": rec,
                "f1": f1,
                "fn_rate": 1 - rec,
            }
            search_rows.append(row)

            # Selection rule: maximize F1, tie-break by recall, then precision.
            key = (f1, rec, prec)
            if best is None or key > best["key"]:
                best = {"key": key, **row}

    sweep_df = pd.DataFrame(search_rows).sort_values(["f1", "recall", "precision"], ascending=False)
    sweep_df.to_csv(bc.OUT_DIR / "lr_tuned_search_validation.csv", index=False)
    assert best is not None
    print(
        f"\nSelected config: C={best['C']}, pos_weight={best['pos_weight']}, "
        f"threshold={best['threshold']:.2f} | val F1={best['f1']:.3f}, recall={best['recall']:.3f}"
    )

    # Refit on full archived train and evaluate on current test.
    X_train_raw, num_cols_train, final_cols = bc.prepare_X(train_df, feats)
    X_test_raw, _, _ = bc.prepare_X(test_df, feats, fit_encoder=final_cols)
    X_train, X_test = bc.scale_for_lr(X_train_raw, X_test_raw, num_cols_train)
    y_train = train_df["severe"].values
    y_test = test_df["severe"].values

    final_lr = LogisticRegression(
        C=float(best["C"]),
        class_weight={0: 1.0, 1: float(best["pos_weight"])},
        max_iter=2000,
        random_state=42,
        solver="lbfgs",
    )
    final_lr.fit(X_train, y_train)
    test_prob = final_lr.predict_proba(X_test)[:, 1]

    t_star = float(best["threshold"])
    y_pred_tuned = (test_prob >= t_star).astype(int)
    rows: list[dict] = []
    rows.append(bc.evaluate("Logistic Regression (tuned)", y_test, y_pred_tuned, test_prob))

    # ADS/L2 slices.
    test_aug = test_df.copy()
    test_aug["y_pred_tuned"] = y_pred_tuned
    test_aug["y_prob_tuned"] = test_prob
    for lvl in ["ADS", "L2"]:
        sub = test_aug[test_aug["automation_level"] == lvl]
        if len(sub) and sub["severe"].nunique() > 1:
            rows.append(
                bc.evaluate(
                    f"LR tuned — test [{lvl}]",
                    sub["severe"].values,
                    sub["y_pred_tuned"].values,
                    sub["y_prob_tuned"].values,
                )
            )

    out_df = pd.DataFrame(rows)
    out_df.to_csv(bc.OUT_DIR / "lr_tuned_results.csv", index=False)
    test_aug[(test_aug["y_pred_tuned"] == 0) & (test_aug["severe"] == 1)].to_csv(
        bc.OUT_DIR / "lr_tuned_false_negatives.csv", index=False
    )

    coef_df = pd.DataFrame(
        {
            "feature": final_cols,
            "coefficient": final_lr.coef_[0],
            "odds_ratio": np.exp(final_lr.coef_[0]),
        }
    ).sort_values("coefficient", key=abs, ascending=False)
    coef_df.to_csv(bc.OUT_DIR / "lr_tuned_coefficients.csv", index=False)

    config_df = pd.DataFrame(
        [
            {
                "C": best["C"],
                "pos_weight": best["pos_weight"],
                "threshold": t_star,
                "validation_f1": best["f1"],
                "validation_recall": best["recall"],
                "validation_precision": best["precision"],
            }
        ]
    )
    config_df.to_csv(bc.OUT_DIR / "lr_tuned_selected_config.csv", index=False)

    print("\nSaved outputs:")
    print(f"  {bc.OUT_DIR / 'lr_tuned_results.csv'}")
    print(f"  {bc.OUT_DIR / 'lr_tuned_false_negatives.csv'}")
    print(f"  {bc.OUT_DIR / 'lr_tuned_coefficients.csv'}")
    print(f"  {bc.OUT_DIR / 'lr_tuned_search_validation.csv'}")
    print(f"  {bc.OUT_DIR / 'lr_tuned_selected_config.csv'}")

    # Side-by-side with baseline row if available.
    base_path = bc.OUT_DIR / "baseline_results.csv"
    if base_path.exists():
        base = pd.read_csv(base_path)
        base_main = base[base["model"] == "Logistic Regression"]
        if len(base_main):
            comp = pd.concat(
                [
                    base_main.assign(setting="baseline_0.50"),
                    out_df[out_df["model"] == "Logistic Regression (tuned)"].assign(
                        setting=f"tuned_{t_star:.2f}"
                    ),
                ],
                ignore_index=True,
            )[
                ["setting", "precision", "recall", "f1", "roc_auc", "fn_rate", "TP", "FP", "FN", "TN"]
            ]
            print("\nBaseline vs tuned (current-era test):")
            print(comp.to_string(index=False))


if __name__ == "__main__":
    main()
