"""
K-means crash scenario discovery on ADS incidents.

Uses narrative + tabular features to cluster ADS crashes into interpretable
scenario groups. Picks k via silhouette score, profiles each cluster by
dominant feature values, and assigns a human-readable scenario label.

Outputs:
  Modeling/logistic_regression/ads_cluster_assignments.csv
  Presentation/figures/18_cluster_profiles.png
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

import baseline_common as bc
import narrative_utils as nu

_MPL = bc.PROJECT_ROOT / ".mplconfig"
_MPL.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL))
os.environ.setdefault("XDG_CACHE_HOME", str(_MPL))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Readable labels derived from cluster profiles after inspection
SCENARIO_LABELS = {
    # filled in dynamically by label_cluster()
}

# Columns used for profiling (categorical only — easier to read)
PROFILE_COLS = [
    "Roadway Type",
    "Crash With",
    "SV Pre-Crash Movement",
    "CP Pre-Crash Movement",
]


def build_cluster_X(df: pd.DataFrame, features: list[str]) -> np.ndarray:
    num_cols = [f for f in features if pd.api.types.is_numeric_dtype(df[f])]
    cat_cols  = [f for f in features if f not in num_cols]
    X = df[features].copy()
    X[num_cols] = X[num_cols].fillna(X[num_cols].median())
    X[cat_cols] = X[cat_cols].fillna("Unknown")
    X = pd.get_dummies(X, columns=cat_cols, drop_first=False).astype(float)
    sc = StandardScaler()
    X.iloc[:, :] = sc.fit_transform(X)
    return X.values, list(X.columns)


def pick_k(X: np.ndarray, k_range=range(3, 9), seed: int = 42) -> tuple[int, list]:
    scores = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=seed, n_init=10)
        labels = km.fit_predict(X)
        s = silhouette_score(X, labels, sample_size=min(2000, len(X)), random_state=seed)
        scores.append((k, s))
        print(f"   k={k}  silhouette={s:.4f}")
    best_k = max(scores, key=lambda t: t[1])[0]
    print(f"\n   Best k = {best_k}")
    return best_k, scores


def profile_cluster(df_cluster: pd.DataFrame, cluster_id: int, n_total: int) -> dict:
    """Return dominant values per profile column for one cluster."""
    pct = len(df_cluster) / n_total * 100
    row = {"cluster": cluster_id, "n": len(df_cluster), "pct_of_ads": round(pct, 1),
           "severe_rate": round(df_cluster["severe"].mean() * 100, 1)}
    for col in PROFILE_COLS:
        if col in df_cluster.columns and not df_cluster[col].dropna().empty:
            top_val = df_cluster[col].value_counts().index[0]
            top_pct = df_cluster[col].value_counts().iloc[0] / len(df_cluster) * 100
            row[f"top_{col.replace(' ', '_')}"] = f"{top_val} ({top_pct:.0f}%)"
        else:
            row[f"top_{col.replace(' ', '_')}"] = "N/A"
    return row


_GLOBAL_NAV_MEANS: dict[str, float] = {}  # filled once in main()


def label_cluster(profile: dict, df_cluster: pd.DataFrame) -> str:
    """Map a cluster's dominant characteristics to a readable scenario name.

    Tabular features alone are often identical across clusters; narrative flags
    (nav_*) provide the differentiating signal for ADS crashes.
    """
    roadway    = profile.get("top_Roadway_Type", "")
    sv_move    = profile.get("top_SV_Pre-Crash_Movement", "")
    cp_move    = profile.get("top_CP_Pre-Crash_Movement", "")
    crash_with = profile.get("top_Crash_With", "")
    severe_rate = profile.get("severe_rate", 0)

    # — fixed-object / animal shortcuts (very distinctive)
    if "Animal" in crash_with:
        return f"Animal strike — {severe_rate:.0f}% severe"
    if "Fixed Object" in crash_with or "Other Fixed Object" in crash_with:
        return f"AV hit fixed object — {severe_rate:.0f}% severe"

    # — dominant narrative flag (highest mean rate among nav_ flags)
    nav_cols = [c for c in df_cluster.columns if c.startswith("nav_")]
    if nav_cols and _GLOBAL_NAV_MEANS:
        cluster_means = df_cluster[nav_cols].mean()
        # lift = cluster rate relative to global rate
        lift = cluster_means - pd.Series(_GLOBAL_NAV_MEANS).reindex(nav_cols, fill_value=0)
        top_lift_flag = lift.idxmax()
        top_lift_val  = lift.max()
        top_nav = top_lift_flag if top_lift_val > 0.05 else None
    elif nav_cols:
        nav_means = df_cluster[nav_cols].mean().sort_values(ascending=False)
        top_nav   = nav_means.index[0] if nav_means.iloc[0] > 0.05 else None
    else:
        top_nav = None

    nav_label_map = {
        "nav_av_stopped":        "AV stopped — rear-end risk",
        "nav_other_struck_av":   "Other vehicle struck AV",
        "nav_av_struck_other":   "AV struck other party",
        "nav_rear_approach":     "Rear-approach / rear-end",
        "nav_at_intersection":   "Intersection conflict",
        "nav_on_highway":        "Highway driving",
        "nav_left_turn":         "Left-turn conflict",
        "nav_lane_change":       "Lane-change conflict",
        "nav_av_changing_lanes": "AV changing lanes",
        "nav_av_turning":        "AV turning",
        "nav_av_reversing":      "AV reversing",
        "nav_in_parking_lot":    "Parking-lot scenario",
        "nav_vulnerable_user":   "Pedestrian / cyclist involved",
        "nav_emergency_vehicle": "Emergency vehicle present",
        "nav_large_vehicle":     "Large vehicle (bus / truck)",
        "nav_av_disengaged":     "AV disengaged / takeover",
        "nav_hazard_lights":     "Hazard lights / disabled AV",
        "nav_double_parked":     "Double-parked obstruction",
        "nav_speed_mentioned":   "Speed cited in narrative",
        "nav_minor_damage_lang": "Minor-damage language",
        "nav_av_moving":         "AV moving at impact",
    }

    # — roadway context
    if "Highway" in roadway or "Freeway" in roadway:
        ctx = "Highway"
    elif "Intersection" in roadway:
        ctx = "Intersection"
    elif "Parking" in roadway:
        ctx = "Parking lot"
    else:
        ctx = "Street"

    # — SV movement shorthand
    if "Stopped" in sv_move or "Parked" in sv_move:
        sv = "AV stopped"
    elif "Left Turn" in sv_move or "Right Turn" in sv_move:
        sv = "AV turning"
    elif "Backing" in sv_move:
        sv = "AV reversing"
    elif "Lane" in sv_move or "Changing" in sv_move:
        sv = "AV changing lanes"
    else:
        sv = "AV moving"

    if top_nav and top_nav in nav_label_map:
        core = nav_label_map[top_nav]
    else:
        # No nav flag elevated — this is the modal/baseline crash pattern
        core = f"Typical {ctx.lower()} crash ({sv})"

    return f"{core} — {severe_rate:.0f}% severe"


def make_cluster_figure(df_ads: pd.DataFrame, profiles: list[dict], k: int,
                        sil_scores: list) -> None:
    fig_dir = bc.CLUSTERING_DIR
    fig_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(16, 10))
    gs  = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.35)

    # — silhouette scores by k
    ax_sil = fig.add_subplot(gs[0, 0])
    ks     = [t[0] for t in sil_scores]
    sils   = [t[1] for t in sil_scores]
    best_k = ks[sils.index(max(sils))]
    ax_sil.plot(ks, sils, marker="o", color="#4C72B0")
    ax_sil.axvline(best_k, color="#C44E52", linestyle="--", alpha=0.7)
    ax_sil.set_title("Silhouette score vs k (ADS)", fontweight="semibold")
    ax_sil.set_xlabel("k"); ax_sil.set_ylabel("Silhouette score")

    # — cluster size bars
    ax_sz = fig.add_subplot(gs[0, 1])
    labels = [f"C{p['cluster']}" for p in profiles]
    sizes  = [p["n"] for p in profiles]
    ax_sz.bar(labels, sizes, color="#4C72B0", alpha=0.85)
    ax_sz.set_title("Cluster sizes (ADS)", fontweight="semibold")
    ax_sz.set_xlabel("Cluster"); ax_sz.set_ylabel("Incidents")

    # — severe rate per cluster
    ax_sv = fig.add_subplot(gs[0, 2])
    rates  = [p["severe_rate"] for p in profiles]
    colors = ["#C44E52" if r >= 50 else "#4C72B0" for r in rates]
    ax_sv.bar(labels, rates, color=colors, alpha=0.85)
    ax_sv.axhline(df_ads["severe"].mean() * 100, color="black", linestyle="--",
                  linewidth=1, label="Overall ADS avg")
    ax_sv.set_title("Severe rate per cluster", fontweight="semibold")
    ax_sv.set_xlabel("Cluster"); ax_sv.set_ylabel("% severe")
    ax_sv.legend(fontsize=8)

    # — top roadway type per cluster (stacked bar)
    ax_rd = fig.add_subplot(gs[1, :2])
    road_col = "Roadway Type"
    if road_col in df_ads.columns:
        ct = df_ads.groupby(["cluster", road_col]).size().unstack(fill_value=0)
        ct.plot(kind="bar", ax=ax_rd, stacked=True, alpha=0.85, legend=True)
        ax_rd.set_title("Roadway Type distribution per cluster", fontweight="semibold")
        ax_rd.set_xlabel("Cluster"); ax_rd.set_ylabel("Incidents")
        ax_rd.tick_params(axis="x", rotation=0)
        ax_rd.legend(fontsize=7, loc="upper right", ncol=2)

    # — scenario label table
    ax_tbl = fig.add_subplot(gs[1, 2])
    ax_tbl.axis("off")
    tbl_data = [[f"C{p['cluster']}", p.get("scenario_label", "—")[:45]] for p in profiles]
    tbl = ax_tbl.table(
        cellText=tbl_data,
        colLabels=["Cluster", "Scenario"],
        cellLoc="left", loc="center",
        colWidths=[0.15, 0.85],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    ax_tbl.set_title("Cluster scenario labels", fontweight="semibold", pad=12)

    fig.suptitle(f"ADS crash scenario clusters (k={k})", fontsize=14, fontweight="bold")
    out = fig_dir / "kmeans_cluster_profiles_figure.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n   Figure: {out}")


def main() -> None:
    print("=" * 70)
    print("ADS crash scenario clustering")
    print("=" * 70)

    df_known = bc.load_known()
    df_known = nu.attach_narrative_flags(df_known)

    ads_df = df_known[df_known["automation_level"] == "ADS"].copy()
    print(f"   ADS incidents: {len(ads_df)}  |  severe rate: {ads_df['severe'].mean()*100:.1f}%")

    tab_feats = [f for f in bc.context_features(ads_df) if f != "automation_level"]
    features  = tab_feats + nu.NAV_FEATURES
    features  = [f for f in features if f in ads_df.columns]

    print(f"\n   Building feature matrix ({len(features)} features)…")
    X, _ = build_cluster_X(ads_df, features)

    print("\n   Selecting k via silhouette score:")
    best_k, sil_scores = pick_k(X)

    km = KMeans(n_clusters=best_k, random_state=42, n_init=20)
    ads_df = ads_df.copy()
    ads_df["cluster"] = km.fit_predict(X)

    # populate global nav means so label_cluster can compute lift
    nav_cols_present = [c for c in ads_df.columns if c.startswith("nav_")]
    _GLOBAL_NAV_MEANS.update(ads_df[nav_cols_present].mean().to_dict())

    print("\n" + "=" * 70)
    print("Cluster profiles")
    print("=" * 70)

    profiles = []
    for cid in sorted(ads_df["cluster"].unique()):
        sub = ads_df[ads_df["cluster"] == cid]
        p   = profile_cluster(sub, cid, len(ads_df))
        p["scenario_label"] = label_cluster(p, sub)
        profiles.append(p)

        nav_cols_sub = [c for c in sub.columns if c.startswith("nav_")]
        if nav_cols_sub and _GLOBAL_NAV_MEANS:
            c_means = sub[nav_cols_sub].mean()
            lift_s  = c_means - pd.Series(_GLOBAL_NAV_MEANS).reindex(nav_cols_sub, fill_value=0)
            top3    = lift_s.nlargest(3)
        else:
            top3 = pd.Series(dtype=float)

        print(f"\n  Cluster {cid} — {p['n']} incidents ({p['pct_of_ads']:.1f}%)")
        print(f"    Severe rate : {p['severe_rate']:.1f}%")
        print(f"    Scenario    : {p['scenario_label']}")
        if not top3.empty:
            print(f"    Top nav lifts: " + ", ".join(f"{k}+{v:.2f}" for k, v in top3.items()))
        for col in PROFILE_COLS:
            key = f"top_{col.replace(' ', '_')}"
            print(f"    {col:<30} {p.get(key, 'N/A')}")

    bc.CLUSTERING_DIR.mkdir(parents=True, exist_ok=True)

    out_csv = bc.CLUSTERING_DIR / "ads_cluster_assignments.csv"
    save_cols = ["cluster"] + [c for c in ads_df.columns if c != "cluster"]
    ads_df[save_cols].to_csv(out_csv, index=False)
    print(f"\n   Saved cluster assignments: {out_csv}")

    summary_df = pd.DataFrame(profiles)
    summary_csv = bc.CLUSTERING_DIR / "ads_cluster_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"   Saved cluster summary    : {summary_csv}")

    make_cluster_figure(ads_df, profiles, best_k, sil_scores)


if __name__ == "__main__":
    main()
