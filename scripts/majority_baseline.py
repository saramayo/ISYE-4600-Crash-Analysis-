"""
Majority-class baseline only: always predict the training-set majority label for `severe`.

Run after 01_clean_incidents.py. Does not use crash features.
Writes Modeling/majority_baseline_results.csv when run standalone.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import baseline_common as bc


def run_majority(train_df, test_df) -> list[dict]:
    print("\n" + "=" * 70)
    print("Majority-class baseline")
    print("=" * 70)

    y_train = train_df["severe"].values
    y_test = test_df["severe"].values

    print(f"\n   Train majority class: {'severe' if y_train.mean() > 0.5 else 'non-severe'} ({y_train.mean()*100:.1f}%)")
    print(f"   Test majority class:  {'severe' if y_test.mean() > 0.5 else 'non-severe'} ({y_test.mean()*100:.1f}%)")

    train_majority = int(y_train.mean() >= 0.5)
    y_pred = np.full(len(y_test), train_majority)

    return [bc.evaluate("Majority-class baseline", y_test, y_pred)]


def main() -> None:
    _, train_df, test_df = bc.load_and_split_verbose()
    rows = run_majority(train_df, test_df)
    pd.DataFrame(rows).to_csv(bc.OUT_DIR / "majority_baseline_results.csv", index=False)
    print(f"\n   Saved {bc.OUT_DIR / 'majority_baseline_results.csv'}")
    print("\nDone.")


if __name__ == "__main__":
    main()
