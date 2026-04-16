"""
Logistic regression baseline for severe vs not (balanced class weights).

Run after 01_clean_incidents.py. Uses the same train/test split as majority_baseline.
Writes Modeling/lr_coefficients.csv, false_negatives.csv, and
logistic_regression_results.csv (metrics including ADS/L2 slices) when run standalone.
"""
from __future__ import annotations

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

import baseline_common as bc


def run_logistic(train_df, test_df, df_known) -> tuple[list[dict], pd.DataFrame, pd.DataFrame]:
    print("\n" + "=" * 70)
    print("Logistic regression (balanced class weights)")
    print("=" * 70)

    feats = bc.context_features(df_known)
    print(f"\n   Using {len(feats)} context features.")

    X_train_raw, num_cols_train, train_cols = bc.prepare_X(train_df, feats)
    X_test_raw, _, _ = bc.prepare_X(test_df, feats, fit_encoder=train_cols)
    X_train, X_test = bc.scale_for_lr(X_train_raw, X_test_raw, num_cols_train)
    print(f"   X_train: {X_train.shape} | X_test: {X_test.shape}")

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

    results: list[dict] = []
    results.append(bc.evaluate("Logistic Regression", y_test, y_pred, y_prob))

    coef_df = pd.DataFrame({
        "feature": train_cols,
        "coefficient": lr.coef_[0],
        "odds_ratio": np.exp(lr.coef_[0]),
    }).sort_values("coefficient", key=abs, ascending=False)

    print("\n  Top 15 features by absolute coefficient:")
    print(coef_df.head(15).to_string(index=False))

    print("\n" + "=" * 70)
    print("Automation-level performance on test set (ADS vs L2)")
    print("=" * 70)

    test_aug = test_df.copy()
    test_aug["y_pred_lr"] = y_pred
    test_aug["y_prob_lr"] = y_prob

    for lvl in ["ADS", "L2"]:
        mask = test_aug["automation_level"] == lvl
        if mask.sum() == 0:
            continue
        sub = test_aug[mask]
        y_t = sub["severe"].values
        y_p = sub["y_pred_lr"].values
        y_pr = sub["y_prob_lr"].values
        if len(np.unique(y_t)) > 1:
            results.append(bc.evaluate(f"LR — test [{lvl}]", y_t, y_p, y_pr))
        else:
            print(f"  [{lvl}] only one class present in test — skipping metrics")

    print("\n" + "=" * 70)
    print("False-negative analysis — missed severe crashes")
    print("=" * 70)

    fn_mask = (y_pred == 0) & (y_test == 1)
    fn_df = test_aug[fn_mask].copy()
    print(f"   Total missed severe crashes (FN): {fn_mask.sum()}")
    print(f"\n   Top Roadway Types in missed severe crashes:")
    print(fn_df["Roadway Type"].value_counts().head(6).to_string())
    print(f"\n   Top Crash Counterpart in missed severe crashes:")
    print(fn_df["Crash With"].value_counts().head(6).to_string())
    print(f"\n   Automation level breakdown of false negatives:")
    print(fn_df["automation_level"].value_counts().to_string())

    return results, coef_df, fn_df


def main() -> None:
    df_known, train_df, test_df = bc.load_and_split_verbose()
    results, coef_df, fn_df = run_logistic(train_df, test_df, df_known)
    pd.DataFrame(results).to_csv(bc.OUT_DIR / "logistic_regression_results.csv", index=False)
    coef_df.to_csv(bc.OUT_DIR / "lr_coefficients.csv", index=False)
    fn_df.to_csv(bc.OUT_DIR / "false_negatives.csv", index=False)
    print("\n" + "=" * 70)
    print("Saved outputs")
    print("=" * 70)
    print(f"   {bc.OUT_DIR / 'logistic_regression_results.csv'}")
    print(f"   {bc.OUT_DIR / 'lr_coefficients.csv'}")
    print(f"   {bc.OUT_DIR / 'false_negatives.csv'}")
    print("\nDone.")


if __name__ == "__main__":
    main()
