"""
ADS precision improvement — compares three strategies:
  1. LR with softer class weights + threshold search
  2. Random Forest (ADS-only, narrative + tabular features)
  3. XGBoost (ADS-only, narrative + tabular features)

Selection rule: maximize F1 subject to precision >= PREC_FLOOR on validation.
Falls back to best F1 if no config meets the floor.

Outputs:
  Modeling/logistic_regression/lr_ads_precision_tuned_results.csv
  Modeling/logistic_regression/lr_ads_precision_tuned_config.csv
  Modeling/logistic_regression/ads_model_comparison.csv
  Presentation/figures/16_ads_model_comparison.png
"""
from __future__ import annotations

import os
import warnings
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import baseline_common as bc
import narrative_utils as nu

_MPL = bc.PROJECT_ROOT / ".mplconfig"
_MPL.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL))
os.environ.setdefault("XDG_CACHE_HOME", str(_MPL))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PREC_FLOOR = 0.65  # minimum acceptable precision on validation

TABULAR_FEATURES = [
    f for f in bc.CONTEXT_FEATURES_BASE if f != "automation_level"
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_X(train: pd.DataFrame, test: pd.DataFrame, features: list[str]):
    num_cols = [f for f in features if pd.api.types.is_numeric_dtype(train[f])]
    cat_cols = [f for f in features if f not in num_cols]

    Xtr = train[features].copy()
    Xte = test[features].copy()
    Xtr[num_cols] = Xtr[num_cols].fillna(Xtr[num_cols].median())
    Xte[num_cols] = Xte[num_cols].fillna(Xtr[num_cols].median())
    Xtr[cat_cols] = Xtr[cat_cols].fillna("Unknown")
    Xte[cat_cols] = Xte[cat_cols].fillna("Unknown")
    Xtr = pd.get_dummies(Xtr, columns=cat_cols, drop_first=False).astype(float)
    Xte = pd.get_dummies(Xte, columns=cat_cols, drop_first=False).astype(float)
    Xte = Xte.reindex(columns=Xtr.columns, fill_value=0.0)

    scaler = StandardScaler()
    if num_cols:
        idx = [Xtr.columns.get_loc(c) for c in num_cols if c in Xtr.columns]
        Xtr.iloc[:, idx] = scaler.fit_transform(Xtr.iloc[:, idx])
        Xte.iloc[:, idx] = scaler.transform(Xte.iloc[:, idx])
    return Xtr, Xte


def evaluate(name: str, y_true, y_pred, y_prob=None) -> dict:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    auc  = roc_auc_score(y_true, y_prob) if (y_prob is not None and len(np.unique(y_true)) > 1) else np.nan
    fn_rate = fn / (fn + tp) if (fn + tp) > 0 else np.nan
    print(f"\n  [{name}]")
    print(f"    Precision : {prec:.3f}   Recall : {rec:.3f}   F1 : {f1:.3f}   AUC : {auc:.3f}")
    print(f"    TP={tp}  FP={fp}  FN={fn}  TN={tn}   FN-rate={fn_rate:.1%}")
    return dict(model=name, precision=prec, recall=rec, f1=f1, roc_auc=auc,
                TP=int(tp), FP=int(fp), FN=int(fn), TN=int(tn), fn_rate=fn_rate)


# ---------------------------------------------------------------------------
# 1. LR with softer class weights + threshold grid
# ---------------------------------------------------------------------------

def run_lr_precision_tuned(train_df, val_df, test_df, features):
    print("\n" + "=" * 70)
    print("1. LR — class weight + threshold search (precision floor {:.0%})".format(PREC_FLOOR))
    print("=" * 70)

    X_sub, X_val = build_X(train_df, val_df, features)
    X_train, X_test = build_X(train_df, test_df, features)
    y_sub  = train_df["severe"].values
    y_val  = val_df["severe"].values
    y_train = train_df["severe"].values
    y_test  = test_df["severe"].values

    pos_weights = [1.0, 1.5, 2.0, 2.5, 3.0, "balanced"]
    c_vals      = [0.1, 0.3, 1.0, 3.0, 10.0]
    thresholds  = np.round(np.arange(0.20, 0.81, 0.02), 2)

    best = None
    search_rows = []
    for pw, c in product(pos_weights, c_vals):
        cw = "balanced" if pw == "balanced" else {0: 1.0, 1: float(pw)}
        lr = LogisticRegression(C=c, class_weight=cw, max_iter=2000,
                                random_state=42, solver="lbfgs")
        lr.fit(X_sub, y_sub)
        val_prob = lr.predict_proba(X_val)[:, 1]

        for t in thresholds:
            yp = (val_prob >= t).astype(int)
            tp = int(((yp == 1) & (y_val == 1)).sum())
            fp = int(((yp == 1) & (y_val == 0)).sum())
            fn = int(((yp == 0) & (y_val == 1)).sum())
            prec = tp / max(tp + fp, 1)
            rec  = tp / max(tp + fn, 1)
            f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            row = dict(pos_weight=str(pw), C=c, threshold=float(t),
                       precision=prec, recall=rec, f1=f1)
            search_rows.append(row)
            key = (prec >= PREC_FLOOR, f1, rec)
            if best is None or key > best["key"]:
                best = {"key": key, "pw": pw, "c": c, "t": float(t),
                        "val_prec": prec, "val_rec": rec, "val_f1": f1}

    assert best is not None
    met_floor = best["key"][0]
    print(f"   Selected: pos_weight={best['pw']}, C={best['c']}, threshold={best['t']:.2f}")
    print(f"   Val: precision={best['val_prec']:.3f}  recall={best['val_rec']:.3f}  "
          f"F1={best['val_f1']:.3f}  {'(met precision floor)' if met_floor else '(floor not met — best F1)'}")

    cw_final = "balanced" if best["pw"] == "balanced" else {0: 1.0, 1: float(best["pw"])}
    lr_final = LogisticRegression(C=best["c"], class_weight=cw_final,
                                  max_iter=2000, random_state=42, solver="lbfgs")
    lr_final.fit(X_train, y_train)
    test_prob = lr_final.predict_proba(X_test)[:, 1]
    y_pred = (test_prob >= best["t"]).astype(int)

    result = evaluate("LR (precision-tuned)", y_test, y_pred, test_prob)
    result.update({"feature_set": "narrative_flags", "threshold": best["t"],
                   "pos_weight": str(best["pw"]), "C": best["c"]})

    config_df = pd.DataFrame([{
        "pos_weight": best["pw"], "C": best["c"], "threshold": best["t"],
        "val_precision": best["val_prec"], "val_recall": best["val_rec"],
        "val_f1": best["val_f1"], "precision_floor_met": met_floor,
    }])
    config_df.to_csv(bc.LR_DIR / "lr_ads_precision_tuned_config.csv", index=False)
    return result, lr_final, X_test.columns.tolist()


# ---------------------------------------------------------------------------
# 2. Random Forest — ADS-only, narrative + tabular
# ---------------------------------------------------------------------------

def run_rf_ads(train_df, val_df, test_df, features):
    print("\n" + "=" * 70)
    print("2. Random Forest — ADS-only (narrative + tabular features)")
    print("=" * 70)

    X_sub, X_val = build_X(train_df, val_df, features)
    X_train, X_test = build_X(train_df, test_df, features)
    y_sub   = train_df["severe"].values
    y_val   = val_df["severe"].values
    y_train = train_df["severe"].values
    y_test  = test_df["severe"].values

    # Grid search on validation
    n_est_grid = [100, 200, 300]
    max_depth_grid = [None, 5, 10]
    min_leaf_grid = [1, 5, 10]
    thresholds = np.round(np.arange(0.20, 0.81, 0.02), 2)

    best = None
    for n_est, depth, leaf in product(n_est_grid, max_depth_grid, min_leaf_grid):
        rf = RandomForestClassifier(n_estimators=n_est, max_depth=depth,
                                    min_samples_leaf=leaf, class_weight="balanced",
                                    random_state=42, n_jobs=-1)
        rf.fit(X_sub, y_sub)
        val_prob = rf.predict_proba(X_val)[:, 1]

        for t in thresholds:
            yp = (val_prob >= t).astype(int)
            tp = int(((yp == 1) & (y_val == 1)).sum())
            fp = int(((yp == 1) & (y_val == 0)).sum())
            fn = int(((yp == 0) & (y_val == 1)).sum())
            prec = tp / max(tp + fp, 1)
            rec  = tp / max(tp + fn, 1)
            f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            key = (prec >= PREC_FLOOR, f1, rec)
            if best is None or key > best["key"]:
                best = {"key": key, "n_est": n_est, "depth": depth, "leaf": leaf,
                        "t": float(t), "val_prec": prec, "val_rec": rec, "val_f1": f1}

    assert best is not None
    met_floor = best["key"][0]
    print(f"   Selected: n_estimators={best['n_est']}, max_depth={best['depth']}, "
          f"min_samples_leaf={best['leaf']}, threshold={best['t']:.2f}")
    print(f"   Val: precision={best['val_prec']:.3f}  recall={best['val_rec']:.3f}  "
          f"F1={best['val_f1']:.3f}  {'(met floor)' if met_floor else '(floor not met)'}")

    rf_final = RandomForestClassifier(n_estimators=best["n_est"], max_depth=best["depth"],
                                      min_samples_leaf=best["leaf"], class_weight="balanced",
                                      random_state=42, n_jobs=-1)
    rf_final.fit(X_train, y_train)
    test_prob = rf_final.predict_proba(X_test)[:, 1]
    y_pred = (test_prob >= best["t"]).astype(int)

    result = evaluate("RF (ADS-only)", y_test, y_pred, test_prob)
    result.update({"feature_set": "narrative+tabular", "threshold": best["t"],
                   "n_estimators": best["n_est"], "max_depth": str(best["depth"]),
                   "min_samples_leaf": best["leaf"]})
    return result


# ---------------------------------------------------------------------------
# 3. XGBoost — ADS-only, narrative + tabular
# ---------------------------------------------------------------------------

def run_xgb_ads(train_df, val_df, test_df, features):
    print("\n" + "=" * 70)
    print("3. XGBoost — ADS-only (narrative + tabular features)")
    print("=" * 70)

    try:
        from xgboost import XGBClassifier
    except ImportError:
        print("   xgboost not installed — skipping.")
        return None

    X_sub, X_val = build_X(train_df, val_df, features)
    X_train, X_test = build_X(train_df, test_df, features)
    y_sub   = train_df["severe"].values
    y_val   = val_df["severe"].values
    y_train = train_df["severe"].values
    y_test  = test_df["severe"].values

    neg = int((y_sub == 0).sum()); pos = int((y_sub == 1).sum())
    scale_pw = neg / max(pos, 1)

    lr_grid        = [0.05, 0.1, 0.2]
    max_depth_grid = [3, 5, 7]
    spw_grid       = [1.0, scale_pw / 2, scale_pw]
    thresholds     = np.round(np.arange(0.20, 0.81, 0.02), 2)

    best = None
    for lr_val, depth, spw in product(lr_grid, max_depth_grid, spw_grid):
        xgb = XGBClassifier(n_estimators=200, learning_rate=lr_val, max_depth=depth,
                            scale_pos_weight=spw, eval_metric="logloss",
                            random_state=42, verbosity=0)
        xgb.fit(X_sub, y_sub)
        val_prob = xgb.predict_proba(X_val)[:, 1]

        for t in thresholds:
            yp = (val_prob >= t).astype(int)
            tp = int(((yp == 1) & (y_val == 1)).sum())
            fp = int(((yp == 1) & (y_val == 0)).sum())
            fn = int(((yp == 0) & (y_val == 1)).sum())
            prec = tp / max(tp + fp, 1)
            rec  = tp / max(tp + fn, 1)
            f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            key = (prec >= PREC_FLOOR, f1, rec)
            if best is None or key > best["key"]:
                best = {"key": key, "lr": lr_val, "depth": depth, "spw": spw,
                        "t": float(t), "val_prec": prec, "val_rec": rec, "val_f1": f1}

    assert best is not None
    met_floor = best["key"][0]
    print(f"   Selected: learning_rate={best['lr']}, max_depth={best['depth']}, "
          f"scale_pos_weight={best['spw']:.2f}, threshold={best['t']:.2f}")
    print(f"   Val: precision={best['val_prec']:.3f}  recall={best['val_rec']:.3f}  "
          f"F1={best['val_f1']:.3f}  {'(met floor)' if met_floor else '(floor not met)'}")

    xgb_final = XGBClassifier(n_estimators=200, learning_rate=best["lr"],
                               max_depth=best["depth"], scale_pos_weight=best["spw"],
                               eval_metric="logloss", random_state=42, verbosity=0)
    xgb_final.fit(X_train, y_train)
    test_prob = xgb_final.predict_proba(X_test)[:, 1]
    y_pred = (test_prob >= best["t"]).astype(int)

    result = evaluate("XGBoost (ADS-only)", y_test, y_pred, test_prob)
    result.update({"feature_set": "narrative+tabular", "threshold": best["t"],
                   "learning_rate": best["lr"], "max_depth": best["depth"],
                   "scale_pos_weight": round(best["spw"], 3)})
    return result


# ---------------------------------------------------------------------------
# Comparison figure
# ---------------------------------------------------------------------------

def make_comparison_figure(rows: list[dict]) -> None:
    df = pd.DataFrame(rows)
    fig_dir = bc.PROJECT_ROOT / "Presentation" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    metrics = ["precision", "recall", "f1"]
    labels  = ["Precision", "Recall", "F1"]
    x = np.arange(len(metrics))
    w = 0.8 / len(df)
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

    for i, (_, row) in enumerate(df.iterrows()):
        offset = (i - len(df) / 2 + 0.5) * w
        vals = [row[m] for m in metrics]
        bars = ax.bar(x + offset, vals, width=w * 0.9, label=row["model"], color=colors[i % len(colors)])
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.015,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=8, fontweight="semibold")

    ax.axhline(PREC_FLOOR, color="#C44E52", linestyle="--", linewidth=1.2,
               label=f"Precision floor ({PREC_FLOOR:.0%})")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=9, loc="upper right")
    ax.set_title("ADS model comparison — precision-tuned approaches (test set)",
                 fontsize=13, fontweight="semibold")
    fig.tight_layout()
    out = fig_dir / "16_ads_model_comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n   Figure: {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 70)
    print("ADS precision tuning — LR / RF / XGBoost")
    print("=" * 70)

    df_known = bc.load_known()
    df_known = nu.attach_narrative_flags(df_known)

    ads = df_known[df_known["automation_level"] == "ADS"]
    train_full = ads[ads["era"] == "archived"].copy()
    test_df    = ads[ads["era"] == "current"].copy()

    # 25% of train as validation for hyperparameter selection
    train_df, val_df = train_test_split(
        train_full, test_size=0.25, random_state=42, stratify=train_full["severe"]
    )

    print(f"\nADS train (sub): {len(train_df)} | val: {len(val_df)} | test: {len(test_df)}")
    print(f"Severe rate — train: {train_df['severe'].mean()*100:.1f}%  "
          f"val: {val_df['severe'].mean()*100:.1f}%  test: {test_df['severe'].mean()*100:.1f}%")

    nav_features = nu.NAV_FEATURES
    tab_features = [f for f in bc.context_features(df_known) if f != "automation_level"]
    all_features = tab_features + nav_features

    # Baseline: original LR from 05_stratified_models_ads_l2.py
    strat_path = bc.LR_DIR / "lr_stratified_by_level_results.csv"
    rows = []
    if strat_path.exists():
        strat = pd.read_csv(strat_path)
        ads_baseline = strat[strat["automation_level"] == "ADS"].iloc[0].to_dict()
        rows.append({
            "model": "LR baseline (narrative, t=0.48)",
            "precision": ads_baseline["precision"],
            "recall": ads_baseline["recall"],
            "f1": ads_baseline["f1"],
            "roc_auc": ads_baseline["roc_auc"],
            "fn_rate": ads_baseline["fn_rate"],
            "TP": int(ads_baseline["TP"]), "FP": int(ads_baseline["FP"]),
            "FN": int(ads_baseline["FN"]), "TN": int(ads_baseline["TN"]),
        })
        print(f"\n  [LR baseline (narrative, t=0.48)]")
        print(f"    Precision : {ads_baseline['precision']:.3f}   "
              f"Recall : {ads_baseline['recall']:.3f}   F1 : {ads_baseline['f1']:.3f}")

    lr_result, _, _ = run_lr_precision_tuned(train_df, val_df, test_df, nav_features)
    rows.append(lr_result)

    rf_result = run_rf_ads(train_df, val_df, test_df, all_features)
    rows.append(rf_result)

    xgb_result = run_xgb_ads(train_df, val_df, test_df, all_features)
    if xgb_result:
        rows.append(xgb_result)

    comp_df = pd.DataFrame(rows)
    out_csv = bc.LR_DIR / "ads_model_comparison.csv"
    comp_df.to_csv(out_csv, index=False)

    print("\n" + "=" * 70)
    print("COMPARISON — ADS test set")
    print("=" * 70)
    cols = ["model", "precision", "recall", "f1", "roc_auc", "fn_rate", "FP", "FN"]
    print(comp_df[cols].to_string(index=False))

    make_comparison_figure(rows)
    print(f"\nSaved: {out_csv}")


if __name__ == "__main__":
    main()
