"""
Run both baselines in one go (same train/test split, one combined metrics table).

Delegates to:
  majority_baseline.py   — majority-class only
  logistic_regression_baseline.py — logistic regression + slices + FN export

For a single model, run those files directly instead.

Writes Modeling/baseline_results.csv (all metrics rows), lr_coefficients.csv, false_negatives.csv.
"""
from __future__ import annotations

import pandas as pd

import baseline_common as bc
from logistic_regression_baseline import run_logistic
from majority_baseline import run_majority


def main() -> None:
    df_known, train_df, test_df = bc.load_and_split_verbose()
    rows: list[dict] = []
    rows.extend(run_majority(train_df, test_df))
    lr_rows, coef_df, fn_df = run_logistic(train_df, test_df, df_known)
    rows.extend(lr_rows)

    pd.DataFrame(rows).to_csv(bc.OUT_DIR / "baseline_results.csv", index=False)
    coef_df.to_csv(bc.OUT_DIR / "lr_coefficients.csv", index=False)
    fn_df.to_csv(bc.OUT_DIR / "false_negatives.csv", index=False)

    print("\n" + "=" * 70)
    print("Saved (combined run)")
    print("=" * 70)
    print(f"   {bc.OUT_DIR / 'baseline_results.csv'}")
    print(f"   {bc.OUT_DIR / 'lr_coefficients.csv'}")
    print(f"   {bc.OUT_DIR / 'false_negatives.csv'}")
    print("\nDone.")


if __name__ == "__main__":
    main()
