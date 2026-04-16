"""
Generate slide-ready figures for the ISYE 4600 presentation.

Covers proposal themes + final-report recommendations:
  severity label, temporal split, reporting bias (ADS/L2, era),
  baselines (majority vs LR), confusion matrices, ADS vs L2 gap,
  false-negative context bars, interpretable odds ratios,
  clustering exploration (silhouette + PCA view).

Run from project root after Cleaned/ and Modeling/ outputs exist:
  .venv/bin/python scripts/make_presentation_figures.py

Writes PNGs under Presentation/figures/ and Presentation/SLIDE_FIGURES.txt
"""
from __future__ import annotations

from pathlib import Path
import os

# Writable matplotlib config (avoids ~/.matplotlib permission issues)
_ROOT = Path(__file__).resolve().parent.parent
_MPL = _ROOT / ".mplconfig"
_MPL.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

import baseline_common as bc

PROJECT_ROOT = _ROOT
FIG_DIR = PROJECT_ROOT / "Presentation" / "figures"
MAP_PATH = PROJECT_ROOT / "Presentation" / "SLIDE_FIGURES.txt"

# Slide-friendly aspect (16:9) and DPI
FIG_KW = dict(dpi=150, bbox_inches="tight")
WIDE = (12.0, 6.75)


def _save(name: str) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    p = FIG_DIR / f"{name}.png"
    plt.savefig(p, **FIG_KW)
    plt.close()
    return p


def style_axes(ax, title: str) -> None:
    ax.set_title(title, fontsize=14, fontweight="semibold", pad=12)
    ax.tick_params(axis="both", labelsize=11)
    sns.despine(ax=ax)


def fig_severity_and_label_breakdown(df: pd.DataFrame) -> None:
    """Severe vs non-severe + share of severe with each OR component."""
    k = df[df["severity_known"] == 1].copy()
    fig, axes = plt.subplots(1, 2, figsize=WIDE)

    vc = k["severe"].value_counts().reindex([0, 1], fill_value=0)
    axes[0].bar(["Non-severe", "Severe"], [vc[0], vc[1]], color=["#4C72B0", "#C44E52"])
    axes[0].set_ylabel("Incidents (labeled)", fontsize=12)
    style_axes(axes[0], "Outcome distribution (severity-known rows)")

    sev = k[k["severe"] == 1]
    parts = []
    for col, lab in [
        ("injury_flag", "Injury ≥ moderate"),
        ("airbag_flag", "Airbag deployed"),
        ("towed_flag", "Vehicle towed"),
    ]:
        if col in sev.columns:
            s = sev[col].dropna()
            parts.append((lab, float(s.mean()) if len(s) else 0.0))
    labs = [p[0] for p in parts]
    vals = [p[1] * 100 for p in parts]
    axes[1].barh(labs, vals, color="#55A868")
    axes[1].set_xlabel("% of severe incidents with signal = 1 (each can overlap)", fontsize=11)
    style_axes(axes[1], "Severity label OR-rule — components among severe")

    plt.tight_layout()
    _save("01_severity_outcome_and_label_components")


def fig_label_rule_schematic() -> None:
    """Simple schematic of the operational rule (talk track: justify in speech)."""
    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)
    ax.axis("off")

    boxes = [
        (0.3, 1.2, 2.2, 1.0, "Injury\n≥ Moderate"),
        (3.0, 1.2, 2.2, 1.0, "Airbag\ndeployed"),
        (5.7, 1.2, 2.2, 1.0, "Vehicle\ntowed"),
    ]
    for x, y, w, h, t in boxes:
        ax.add_patch(mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05", facecolor="#E8E8E8", edgecolor="#333"))
        ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=12, fontweight="semibold")

    ax.text(8.3, 1.7, "OR", fontsize=16, fontweight="bold")
    ax.add_patch(mpatches.FancyBboxPatch((8.9, 1.0), 0.9, 1.4, boxstyle="round,pad=0.05", facecolor="#C44E52", edgecolor="#333"))
    ax.text(9.35, 1.7, "Severe\n= 1", ha="center", va="center", fontsize=12, color="white", fontweight="bold")

    ax.text(5, 2.55, "Binary severity label (after consolidating to one row per incident)", ha="center", fontsize=13, fontweight="semibold")
    plt.tight_layout()
    _save("02_severity_label_rule_schematic")


def fig_temporal_split(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=WIDE)
    labs = ["Train\n(archived)", "Test\n(current)"]
    vals = [len(train_df), len(test_df)]
    c = ["#8172B3", "#CCB974"]
    ax.bar(labs, vals, color=c, width=0.55)
    for i, v in enumerate(vals):
        ax.text(i, v + max(vals) * 0.02, f"n = {v:,}", ha="center", fontsize=12, fontweight="semibold")
    ax.set_ylabel("Labeled incidents", fontsize=12)
    style_axes(ax, "Temporal validation — train on past reports, test on current era")
    plt.tight_layout()
    _save("03_temporal_train_test_split")


def fig_reporting_bias_strata(df: pd.DataFrame) -> None:
    k = df[df["severity_known"] == 1].copy()
    g = k.groupby(["era", "automation_level"], observed=False).agg(
        n=("severe", "size"),
        severe_rate=("severe", "mean"),
    ).reset_index()

    fig, ax = plt.subplots(figsize=WIDE)
    x = np.arange(len(g))
    ax.bar(x, g["severe_rate"] * 100, color="#4C72B0")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r['era']}\n{r['automation_level']}" for _, r in g.iterrows()], fontsize=10)
    ax.set_ylabel("Severe rate (%)", fontsize=12)
    for i, (_, r) in enumerate(g.iterrows()):
        ax.text(i, r["severe_rate"] * 100 + 1, f"n={int(r['n'])}", ha="center", fontsize=9, color="#333")
    style_axes(ax, "Reporting strata — severe rate differs by era and automation level")
    plt.tight_layout()
    _save("04_reporting_bias_severe_rate_by_stratum")


def fig_baseline_metrics(br: pd.DataFrame) -> None:
    main = br[br["model"].isin(["Majority-class baseline", "Logistic Regression"])].copy()
    fig, ax = plt.subplots(figsize=WIDE)
    metrics = ["precision", "recall", "f1"]
    x = np.arange(len(metrics))
    w = 0.35
    for i, (_, row) in enumerate(main.iterrows()):
        offset = (i - 0.5) * w
        vals = [row[m] for m in metrics]
        ax.bar(x + offset, vals, width=w, label=row["model"].replace("Logistic Regression", "Logistic regr."))

    ax.set_xticks(x)
    ax.set_xticklabels(["Precision", "Recall", "F1"], fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=10, loc="lower right")
    style_axes(ax, "Supervised baselines on current-era test set")
    plt.tight_layout()
    _save("05_baseline_metrics_majority_vs_lr")


def fig_confusion_heatmaps(br: pd.DataFrame) -> None:
    def cm_from_row(name: str):
        r = br[br["model"] == name].iloc[0]
        tn, fp, fn, tp = int(r["TN"]), int(r["FP"]), int(r["FN"]), int(r["TP"])
        return np.array([[tn, fp], [fn, tp]])

    fig, axes = plt.subplots(1, 2, figsize=WIDE)
    for ax, title, key in [
        (axes[0], "Majority-class baseline", "Majority-class baseline"),
        (axes[1], "Logistic regression", "Logistic Regression"),
    ]:
        cm = cm_from_row(key)
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            ax=ax,
            xticklabels=["Pred 0", "Pred 1"],
            yticklabels=["True 0", "True 1"],
            annot_kws={"size": 13},
        )
        ax.set_title(title, fontsize=13, fontweight="semibold")
    plt.suptitle("Confusion matrices (current-era test)", fontsize=14, fontweight="semibold", y=1.02)
    plt.tight_layout()
    _save("06_confusion_matrices")


def fig_ads_vs_l2(br: pd.DataFrame) -> None:
    sub = br[br["model"].str.startswith("LR — test", na=False)].copy()
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    labs = [m.replace("LR — test [", "").replace("]", "") for m in sub["model"]]
    x = np.arange(len(labs))
    w = 0.25
    for j, m in enumerate(["precision", "recall", "f1"]):
        ax.bar(x + (j - 1) * w, sub[m], width=w, label=m.capitalize())
    ax.set_xticks(x + w * 0)
    ax.set_xticklabels(labs)
    ax.set_ylim(0, 1.05)
    ax.legend()
    style_axes(ax, "Logistic regression — same test set, sliced by automation level")
    plt.tight_layout()
    _save("07_ads_vs_l2_metrics")


def fig_false_negatives(fn: pd.DataFrame) -> None:
    if fn.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=WIDE)
    for ax, col, title in [
        (axes[0], "Roadway Type", "False negatives — roadway type"),
        (axes[1], "Crash With", "False negatives — crash partner (Crash With)"),
    ]:
        if col not in fn.columns:
            continue
        s = fn[col].value_counts().head(8)
        s.plot(kind="barh", ax=ax, color="#C44E52")
        style_axes(ax, title)
    plt.suptitle("Where logistic regression misses severe crashes (FN analysis)", fontsize=14, fontweight="semibold", y=1.02)
    plt.tight_layout()
    _save("08_false_negative_contexts")


def fig_odds_ratios(coef_path: Path, top_n: int = 12) -> None:
    coef = pd.read_csv(coef_path)
    coef = coef.reindex(coef["coefficient"].abs().sort_values(ascending=False).index).head(top_n)
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = np.where(coef["coefficient"].values >= 0, "#C44E52", "#4C72B0")
    ax.barh(coef["feature"][::-1], coef["odds_ratio"][::-1], color=colors[::-1])
    ax.axvline(1.0, color="#333", linestyle="--", linewidth=1)
    ax.set_xlabel("Odds ratio (vs reference category / unit)", fontsize=11)
    style_axes(ax, "Interpretable drivers — top logistic regression odds ratios")
    plt.tight_layout()
    _save("09_top_odds_ratios_lr")


def fig_clustering(df_known: pd.DataFrame) -> None:
    """K-means + silhouette vs k + PCA scatter (proposal clustering deliverable)."""
    feats = bc.context_features(df_known)
    X_raw, _, _ = bc.prepare_X(df_known, feats)
    num_cols = X_raw.select_dtypes(include=[np.number]).columns.tolist()
    X = X_raw.copy()
    if num_cols:
        X[num_cols] = StandardScaler().fit_transform(X[num_cols])
    X = X.fillna(0)
    # Sample for speed (clustering is exploratory)
    if len(X) > 2500:
        X = X.sample(2500, random_state=42)

    ks = list(range(2, 9))
    sil = []
    for k in ks:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        lab = km.fit_predict(X)
        sil.append(silhouette_score(X, lab) if len(np.unique(lab)) > 1 else np.nan)

    best_k = int(ks[int(np.nanargmax(sil))])

    fig, axes = plt.subplots(1, 2, figsize=WIDE)
    axes[0].plot(ks, sil, marker="o", color="#8172B3")
    axes[0].set_xlabel("k (clusters)", fontsize=12)
    axes[0].set_ylabel("Silhouette score", fontsize=12)
    style_axes(axes[0], "Cluster quality vs k (K-means, scaled context features)")

    km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    pca = PCA(n_components=2, random_state=42)
    Z = pca.fit_transform(X.values)
    sc = axes[1].scatter(Z[:, 0], Z[:, 1], c=labels, cmap="tab10", alpha=0.5, s=12)
    axes[1].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    axes[1].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    plt.colorbar(sc, ax=axes[1], label="Cluster")
    style_axes(axes[1], f"PCA projection — K-means with k={best_k} (exploratory)")

    plt.tight_layout()
    _save("10_clustering_silhouette_and_pca")


def fig_interpretation_summary() -> None:
    """Placeholder slide: bullet themes for 'Interpretation for the system' (fill in report)."""
    fig, ax = plt.subplots(figsize=WIDE)
    ax.axis("off")
    lines = [
        "Interpretation for the system (talk track + bullets)",
        "",
        "• Prioritize validation where LR flags higher odds (see odds-ratio figure).",
        "• Address ADS vs L2 gap: model behavior differs by automation level on current-era test.",
        "• False negatives cluster on streets / intersections / passenger-car strikes — targeted scenarios.",
        "• Caveat: SGO reporting is not a census; use for prioritization signals, not population rates.",
    ]
    y = 0.92
    for i, line in enumerate(lines):
        fs = 16 if i == 0 else 13
        wt = "bold" if i == 0 else "normal"
        ax.text(0.05, y - i * 0.09, line, fontsize=fs, fontweight=wt, transform=ax.transAxes, family="sans-serif")
    plt.tight_layout()
    _save("11_interpretation_for_system_talking_slide")


def main() -> None:
    sns.set_theme(style="whitegrid", context="talk", font_scale=0.9)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(bc.DATA_PATH, low_memory=False)
    df_known = df[df["severity_known"] == 1].copy()
    train_df, test_df = bc.split_train_test(df_known)

    fig_severity_and_label_breakdown(df)
    fig_label_rule_schematic()
    fig_temporal_split(train_df, test_df)
    fig_reporting_bias_strata(df)
    br = pd.read_csv(PROJECT_ROOT / "Modeling" / "baseline_results.csv")
    fig_baseline_metrics(br)
    fig_confusion_heatmaps(br)
    fig_ads_vs_l2(br)
    fn_path = PROJECT_ROOT / "Modeling" / "false_negatives.csv"
    if fn_path.exists():
        fig_false_negatives(pd.read_csv(fn_path))
    coef_path = PROJECT_ROOT / "Modeling" / "lr_coefficients.csv"
    if coef_path.exists():
        fig_odds_ratios(coef_path)
    fig_clustering(df_known)
    fig_interpretation_summary()

    paths = sorted(FIG_DIR.glob("*.png"))
    MAP_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "Presentation figures (PNG) — map to slides",
        "Generated by scripts/make_presentation_figures.py",
        "",
    ]
    for p in paths:
        lines.append(f"- {p.name}")
    lines.extend(
        [
            "",
            "Suggested slide order:",
            "  02 — Label rule (methods)",
            "  01 — Outcome + OR components (methods / data)",
            "  03 — Temporal split (methods)",
            "  04 — Reporting bias strata (limitations)",
            "  05–06 — Baselines + confusion matrices (results)",
            "  07 — ADS vs L2 (results / limitations)",
            "  08 — False negatives (error analysis)",
            "  09 — Odds ratios (interpretation)",
            "  10 — Clustering (methods / exploratory results)",
            "  11 — Interpretation for system (discussion)",
        ]
    )
    MAP_PATH.write_text("\n".join(lines), encoding="utf-8")

    print(f"Wrote {len(paths)} figures to {FIG_DIR}")
    print(f"Slide map: {MAP_PATH}")


if __name__ == "__main__":
    main()
