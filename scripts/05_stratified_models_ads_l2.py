"""
All 4 model × dataset combinations:
  ADS + LR  | ADS + RF
  L2  + LR  | L2  + RF

Each model is trained/tested on its automation-level slice only (temporal split).
Features: narrative + tabular for ADS; tabular for L2.
Threshold tuned on a 25% validation split (target: precision >= PREC_FLOOR).

Outputs:
  Modeling/logistic_regression/all_stratified_results.csv
  Modeling/logistic_regression/lr_ads_coefficients.csv
  Modeling/logistic_regression/rf_ads_importances.csv
  Modeling/logistic_regression/lr_l2_coefficients.csv
  Modeling/logistic_regression/rf_l2_importances.csv
  Presentation/figures/13_2x2_model_comparison.png
"""
from __future__ import annotations

import os
from itertools import product as iproduct

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

import baseline_common as bc
import narrative_utils as nu

# Minimum precision on the validation split before we accept a threshold.
# Set at 0.65: below this, too many false alarms erode operational usefulness.
# Falls back to best-F1 if no threshold meets the floor (documented in results).
PREC_FLOOR = 0.65

_MPL = bc.PROJECT_ROOT / ".mplconfig"
_MPL.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL))
os.environ.setdefault("XDG_CACHE_HOME", str(_MPL))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Shared helpers
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


def split_within_level(level_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    # Temporal split within each level mimics deployment: train on past, test on present.
    # ADS has a large distribution shift (26% → 47% severe) across eras, so this split
    # is intentionally harder than random — it reflects the real evaluation scenario.
    curr = level_df[level_df["era"] == "current"]
    arch = level_df[level_df["era"] == "archived"]
    if len(curr) >= 50 and curr["severe"].nunique() == 2 and arch["severe"].nunique() == 2:
        return arch.copy(), curr.copy(), "temporal"
    tr, te = train_test_split(level_df, test_size=0.2, random_state=42,
                              stratify=level_df["severe"])
    return tr.copy(), te.copy(), "stratified_random"


def best_threshold(val_prob, y_val):
    """Return threshold that maximises F1 with precision >= PREC_FLOOR; fallback: best F1."""
    thresholds = np.round(np.arange(0.20, 0.81, 0.02), 2)
    best = None
    for t in thresholds:
        yp = (val_prob >= t).astype(int)
        tp = int(((yp == 1) & (y_val == 1)).sum())
        fp = int(((yp == 1) & (y_val == 0)).sum())
        fn = int(((yp == 0) & (y_val == 1)).sum())
        prec = tp / max(tp + fp, 1)
        rec  = tp / max(tp + fn, 1)
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        key  = (prec >= PREC_FLOOR, f1, rec)
        if best is None or key > best["key"]:
            best = {"key": key, "t": float(t), "prec": prec, "rec": rec, "f1": f1}
    assert best is not None
    return best


def evaluate_on_test(name, y_test, y_prob, t_star):
    yp = (y_prob >= t_star).astype(int)
    tp = int(((yp == 1) & (y_test == 1)).sum())
    fp = int(((yp == 1) & (y_test == 0)).sum())
    fn = int(((yp == 0) & (y_test == 1)).sum())
    tn = int(((yp == 0) & (y_test == 0)).sum())
    prec = tp / max(tp + fp, 1)
    rec  = tp / max(tp + fn, 1)
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(y_test, y_prob) if len(np.unique(y_test)) > 1 else np.nan
    fn_rate = fn / (fn + tp) if (fn + tp) > 0 else np.nan
    print(f"\n  [{name}]")
    print(f"    Precision={prec:.3f}  Recall={rec:.3f}  F1={f1:.3f}  AUC={auc:.3f}")
    print(f"    TP={tp}  FP={fp}  FN={fn}  TN={tn}  FN-rate={fn_rate:.1%}")
    return dict(model=name, precision=prec, recall=rec, f1=f1, roc_auc=auc,
                TP=tp, FP=fp, FN=fn, TN=tn, fn_rate=fn_rate, threshold=t_star)


# ---------------------------------------------------------------------------
# LR runner (works for any level / feature set)
# ---------------------------------------------------------------------------

def run_lr(level: str, train_df: pd.DataFrame, val_df: pd.DataFrame,
           test_df: pd.DataFrame, features: list[str]) -> tuple[dict, pd.DataFrame]:
    print(f"\n{'='*70}\nLR — {level}\n{'='*70}")

    X_sub, X_val   = build_X(train_df, val_df,  features)
    X_train, X_test = build_X(train_df, test_df, features)
    y_sub, y_val   = train_df["severe"].values, val_df["severe"].values
    y_train, y_test = train_df["severe"].values, test_df["severe"].values

    # C grid on sub-train → pick best val threshold
    c_grid = [0.1, 0.3, 1.0, 3.0, 10.0]
    best_global = None
    for c in c_grid:
        lr = LogisticRegression(C=c, class_weight="balanced", max_iter=2000,
                                random_state=42, solver="lbfgs")
        lr.fit(X_sub, y_sub)
        b = best_threshold(lr.predict_proba(X_val)[:, 1], y_val)
        b["c"] = c
        key = (b["key"][0], b["f1"], b["rec"])
        if best_global is None or key > best_global["key"]:
            best_global = {**b, "key": key}

    assert best_global is not None
    met = best_global["key"][0]
    print(f"   C={best_global['c']}, threshold={best_global['t']:.2f} | "
          f"val prec={best_global['prec']:.3f} rec={best_global['rec']:.3f} "
          f"f1={best_global['f1']:.3f} {'✓floor' if met else '(floor not met)'}")

    lr_final = LogisticRegression(C=best_global["c"], class_weight="balanced",
                                  max_iter=2000, random_state=42, solver="lbfgs")
    lr_final.fit(X_train, y_train)
    test_prob = lr_final.predict_proba(X_test)[:, 1]

    result = evaluate_on_test(f"LR [{level}]", y_test, test_prob, best_global["t"])
    result.update({"automation_level": level, "algorithm": "LR",
                   "feature_set": "narrative+tabular" if level == "ADS" else "tabular"})

    coef_df = pd.DataFrame({"feature": X_train.columns,
                             "coefficient": lr_final.coef_[0],
                             "odds_ratio": np.exp(lr_final.coef_[0])
                             }).sort_values("coefficient", key=abs, ascending=False)
    return result, coef_df


# ---------------------------------------------------------------------------
# RF runner (works for any level / feature set)
# ---------------------------------------------------------------------------

def run_rf(level: str, train_df: pd.DataFrame, val_df: pd.DataFrame,
           test_df: pd.DataFrame, features: list[str]) -> tuple[dict, pd.DataFrame]:
    print(f"\n{'='*70}\nRF — {level}\n{'='*70}")

    X_sub, X_val   = build_X(train_df, val_df,  features)
    X_train, X_test = build_X(train_df, test_df, features)
    y_sub, y_val   = train_df["severe"].values, val_df["severe"].values
    y_train, y_test = train_df["severe"].values, test_df["severe"].values

    n_est_grid     = [100, 200, 300]
    max_depth_grid = [None, 5, 10]
    min_leaf_grid  = [1, 5, 10]

    best_global = None
    best_params = {}
    for n_est, depth, leaf in iproduct(n_est_grid, max_depth_grid, min_leaf_grid):
        rf = RandomForestClassifier(n_estimators=n_est, max_depth=depth,
                                    min_samples_leaf=leaf, class_weight="balanced",
                                    random_state=42, n_jobs=-1)
        rf.fit(X_sub, y_sub)
        b = best_threshold(rf.predict_proba(X_val)[:, 1], y_val)
        key = (b["key"][0], b["f1"], b["rec"])
        if best_global is None or key > best_global["key"]:
            best_global = {**b, "key": key}
            best_params = {"n_est": n_est, "depth": depth, "leaf": leaf}

    assert best_global is not None
    met = best_global["key"][0]
    print(f"   n_est={best_params['n_est']}, depth={best_params['depth']}, "
          f"leaf={best_params['leaf']}, threshold={best_global['t']:.2f} | "
          f"val prec={best_global['prec']:.3f} rec={best_global['rec']:.3f} "
          f"f1={best_global['f1']:.3f} {'✓floor' if met else '(floor not met)'}")

    rf_final = RandomForestClassifier(n_estimators=best_params["n_est"],
                                      max_depth=best_params["depth"],
                                      min_samples_leaf=best_params["leaf"],
                                      class_weight="balanced", random_state=42, n_jobs=-1)
    rf_final.fit(X_train, y_train)
    test_prob = rf_final.predict_proba(X_test)[:, 1]

    result = evaluate_on_test(f"RF [{level}]", y_test, test_prob, best_global["t"])
    result.update({"automation_level": level, "algorithm": "RF",
                   "feature_set": "narrative+tabular" if level == "ADS" else "tabular"})

    imp_df = pd.DataFrame({"feature": X_train.columns,
                            "importance": rf_final.feature_importances_
                            }).sort_values("importance", ascending=False)
    return result, imp_df


# ---------------------------------------------------------------------------
# XGBoost runner (works for any level / feature set)
# ---------------------------------------------------------------------------

def run_xgb(level: str, train_df: pd.DataFrame, val_df: pd.DataFrame,
            test_df: pd.DataFrame, features: list[str]) -> tuple[dict, pd.DataFrame]:
    print(f"\n{'='*70}\nXGBoost — {level}\n{'='*70}")

    X_sub, X_val    = build_X(train_df, val_df,  features)
    X_train, X_test = build_X(train_df, test_df, features)
    y_sub, y_val    = train_df["severe"].values, val_df["severe"].values
    y_train, y_test = train_df["severe"].values, test_df["severe"].values

    neg = int((y_sub == 0).sum()); pos = int((y_sub == 1).sum())
    # scale_pos_weight variants: 1 (no correction), half ratio, full ratio
    spw_grid       = [1.0, (neg / max(pos, 1)) / 2, neg / max(pos, 1)]
    lr_grid        = [0.05, 0.1, 0.2]
    max_depth_grid = [3, 5, 7]

    best_global = None
    best_params = {}
    for lr_val, depth, spw in iproduct(lr_grid, max_depth_grid, spw_grid):
        xgb = XGBClassifier(n_estimators=200, learning_rate=lr_val, max_depth=depth,
                            scale_pos_weight=spw, eval_metric="logloss",
                            random_state=42, verbosity=0)
        xgb.fit(X_sub, y_sub)
        b = best_threshold(xgb.predict_proba(X_val)[:, 1], y_val)
        key = (b["key"][0], b["f1"], b["rec"])
        if best_global is None or key > best_global["key"]:
            best_global = {**b, "key": key}
            best_params = {"lr": lr_val, "depth": depth, "spw": round(spw, 3)}

    assert best_global is not None
    met = best_global["key"][0]
    print(f"   lr={best_params['lr']}, depth={best_params['depth']}, "
          f"spw={best_params['spw']}, threshold={best_global['t']:.2f} | "
          f"val prec={best_global['prec']:.3f} rec={best_global['rec']:.3f} "
          f"f1={best_global['f1']:.3f} {'✓floor' if met else '(floor not met)'}")

    xgb_final = XGBClassifier(n_estimators=200, learning_rate=best_params["lr"],
                              max_depth=best_params["depth"],
                              scale_pos_weight=best_params["spw"],
                              eval_metric="logloss", random_state=42, verbosity=0)
    xgb_final.fit(X_train, y_train)
    test_prob = xgb_final.predict_proba(X_test)[:, 1]

    result = evaluate_on_test(f"XGB [{level}]", y_test, test_prob, best_global["t"])
    result.update({"automation_level": level, "algorithm": "XGB",
                   "feature_set": "narrative+tabular" if level == "ADS" else "tabular"})

    imp_df = pd.DataFrame({"feature": X_train.columns,
                            "importance": xgb_final.feature_importances_
                            }).sort_values("importance", ascending=False)
    return result, imp_df


# ---------------------------------------------------------------------------
# 2×2 → 2×3 figure (LR / RF / XGB)
# ---------------------------------------------------------------------------

def make_comparison_figure(results: pd.DataFrame) -> None:
    fig_dir = bc.PROJECT_ROOT / "Presentation" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    levels = ["ADS", "L2"]
    algos  = [a for a in ["LR", "RF", "XGB"] if a in results["algorithm"].values]
    metrics = ["precision", "recall", "f1"]
    colors  = ["#4C72B0", "#55A868", "#C44E52"]

    n_rows, n_cols = len(algos), len(levels)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 4.5 * n_rows), sharey=True)
    if n_rows == 1:
        axes = [axes]
    fig.suptitle(f"All {n_rows * n_cols} model combinations — current-era test set",
                 fontsize=14, fontweight="semibold")

    for row_i, algo in enumerate(algos):
        for col_i, level in enumerate(levels):
            ax = axes[row_i][col_i]
            r = results[(results["algorithm"] == algo) &
                        (results["automation_level"] == level)]
            if r.empty:
                ax.set_visible(False)
                continue
            r = r.iloc[0]
            vals = [r[m] for m in metrics]
            bars = ax.bar(["Precision", "Recall", "F1"], vals, color=colors, width=0.5)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02,
                        f"{v:.2f}", ha="center", fontsize=11, fontweight="semibold")
            ax.set_ylim(0, 1.18)
            ax.axhline(PREC_FLOOR, color="#C44E52", linestyle="--", linewidth=1, alpha=0.6)
            feat = str(r.get("feature_set", "")).replace("narrative+tabular", "narr+tab")
            ax.set_title(f"{algo} — {level}\n(features: {feat}, t={r['threshold']:.2f})",
                         fontsize=11, fontweight="semibold")
            ax.text(0.98, 0.97, f"AUC={r['roc_auc']:.3f}\nFN-rate={r['fn_rate']:.1%}",
                    transform=ax.transAxes, ha="right", va="top", fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="#f5f5f5", alpha=0.8))

    fig.text(0.02, 0.5, "Score", va="center", rotation="vertical", fontsize=12)
    plt.tight_layout(rect=[0.03, 0, 1, 1])
    out = fig_dir / "13_model_comparison_all.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n   Figure: {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 70)
    print("All 4 combinations: ADS/L2 × LR/RF")
    print("=" * 70)

    df_known = bc.load_known()
    df_known = nu.attach_narrative_flags(df_known)

    tab_feats = [f for f in bc.context_features(df_known) if f != "automation_level"]
    # ADS: tabular alone gives AUC~0.50 (random). Narrative flags capture crash
    # dynamics (who struck whom, AV state, location type) that tabular fields miss.
    ads_feats = tab_feats + nu.NAV_FEATURES
    # L2: 98% severe rate — near linearly separable. Narrative adds no measurable gain.
    l2_feats  = tab_feats

    results = []
    coef_dfs = {}

    for level, feats in [("ADS", ads_feats), ("L2", l2_feats)]:
        level_df = df_known[df_known["automation_level"] == level].copy()
        train_df, test_df, split_type = split_within_level(level_df)
        train_df, val_df = train_test_split(
            train_df, test_size=0.25, random_state=42, stratify=train_df["severe"]
        )
        print(f"\n  {level}: train={len(train_df)} val={len(val_df)} test={len(test_df)} "
              f"| severe — train:{train_df['severe'].mean()*100:.0f}% "
              f"test:{test_df['severe'].mean()*100:.0f}% | split:{split_type}")

        lr_res, lr_coef = run_lr(level, train_df, val_df, test_df, feats)
        lr_res["split_type"] = split_type
        results.append(lr_res)
        coef_dfs[f"lr_{level.lower()}"] = lr_coef

        rf_res, rf_imp = run_rf(level, train_df, val_df, test_df, feats)
        rf_res["split_type"] = split_type
        results.append(rf_res)
        coef_dfs[f"rf_{level.lower()}"] = rf_imp

        xgb_res, xgb_imp = run_xgb(level, train_df, val_df, test_df, feats)
        xgb_res["split_type"] = split_type
        results.append(xgb_res)
        coef_dfs[f"xgb_{level.lower()}"] = xgb_imp

    out_df = pd.DataFrame(results)

    print("\n" + "=" * 70)
    print("SUMMARY — all 4 combinations (test set)")
    print("=" * 70)
    cols = ["model", "precision", "recall", "f1", "roc_auc", "fn_rate", "FP", "FN", "threshold"]
    print(out_df[cols].to_string(index=False))

    # Save outputs
    out_csv = bc.LR_DIR / "all_stratified_results.csv"
    out_df.to_csv(out_csv, index=False)

    # Also keep lr_stratified_by_level_results.csv pointing at best per level
    # (RF for ADS, LR for L2 — matches what presentation figures expect)
    best_ads = out_df[(out_df["automation_level"] == "ADS") &
                      (out_df["algorithm"] == "RF")].copy()
    best_l2  = out_df[(out_df["automation_level"] == "L2") &
                      (out_df["algorithm"] == "LR")].copy()
    pd.concat([best_ads, best_l2], ignore_index=True).to_csv(
        bc.LR_DIR / "lr_stratified_by_level_results.csv", index=False
    )

    coef_dfs["lr_ads"].to_csv(bc.LR_DIR / "lr_ads_coefficients.csv", index=False)
    coef_dfs["rf_ads"].to_csv(bc.LR_DIR / "rf_ads_importances.csv", index=False)
    coef_dfs["xgb_ads"].to_csv(bc.LR_DIR / "xgb_ads_importances.csv", index=False)
    coef_dfs["lr_l2"].to_csv(bc.LR_DIR / "lr_l2_coefficients.csv", index=False)
    coef_dfs["rf_l2"].to_csv(bc.LR_DIR / "rf_l2_importances.csv", index=False)
    coef_dfs["xgb_l2"].to_csv(bc.LR_DIR / "xgb_l2_importances.csv", index=False)

    print(f"\nSaved: {out_csv}")
    make_comparison_figure(out_df)


if __name__ == "__main__":
    main()
