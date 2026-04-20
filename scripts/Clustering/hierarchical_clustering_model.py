"""
ISYE 4600 AV Crash Severity Project — Hierarchical (Agglomerative) Clustering
===============================================================================

Run after 01_clean_incidents.py (requires Cleaned/sgo_cleaned_incidents.csv).

What this script does
---------------------
This script is UNSUPERVISED — the severe label is never used during training.
Hierarchical clustering discovers natural groupings by repeatedly merging the
closest pairs of clusters (agglomerative / bottom-up approach). The severe label
is only consulted AFTER clustering for post-hoc validation.

Pipeline
--------
1. Load ALL severity-known incidents (full dataset — no train/test split).
2. Scale features with StandardScaler (required for distance-based methods).
3. Reduce to 50 PCA components to manage computational cost; retain 2D for plots.
4. Compute linkage matrix on the PCA-reduced data and save dendrogram data.
5. Evaluate k = 2–10 cluster cuts using silhouette, CH, and DB scores.
6. Fit final AgglomerativeClustering with the best k.
7. Profile each cluster: size, severe rate, automation level, dominant features.
8. Post-hoc external validation: ARI and NMI vs. the severe label.

How Hierarchical Clustering differs from K-Means
-------------------------------------------------
  K-Means:
  - Requires specifying k BEFORE training; you must run it multiple times to
    find the best k.
  - Assigns each point to exactly one centroid; assumes roughly spherical clusters.
  - Objective: minimise within-cluster sum of squares (inertia).
  - Produces one fixed partitioning per run.

  Hierarchical (Agglomerative):
  - Builds a FULL TREE (dendrogram) of merges in a single pass. You choose k
    AFTER the fact by cutting the tree at a chosen height.
  - Makes no assumption about cluster shape; Ward linkage (used here) tends to
    produce compact, roughly equal-sized clusters similar to K-Means, but the
    algorithm can also use complete, average, or single linkage for
    non-spherical or chain-like clusters.
  - No concept of inertia; instead evaluated by dendrogram structure and the
    same silhouette / CH / DB metrics as K-Means (directly comparable).
  - Computationally more expensive: O(n² log n) with Ward linkage.
    For n = 5,000+, we first reduce dimensions with PCA to stay fast.

  Key insight for your project:
  - If K-Means and Hierarchical produce similar cluster structures (similar
    ARI, NMI, similar profiles), that convergence strengthens your conclusions.
  - If they disagree, it suggests the underlying data geometry is complex and
    neither algorithm fully captures it — worth noting as a limitation.

Why Ward linkage?
-----------------
  Ward minimises the total within-cluster variance at each merge step — the
  same objective as K-Means. This makes it the most natural hierarchical
  counterpart for comparison. Other linkage methods:
    complete — merge clusters by their maximum pairwise distance (conservative)
    average  — merge by mean pairwise distance (balanced)
    single   — merge by minimum distance (can produce 'chaining')

Outputs (written to Modeling/clustering/)
------------------------------------------
  hclust_k_selection.csv       — silhouette + CH + DB scores for k = 2..10
  hclust_cluster_profiles.csv  — per-cluster summary (size, severe rate, top features)
  hclust_assignments.csv       — one row per incident: cluster label + PCA coords + severe
  hclust_best_params.csv       — optimal k and its evaluation metrics
  hclust_external_validity.csv — ARI and NMI vs the severe label (post-hoc validation)
  hclust_dendrogram_data.csv   — linkage matrix excerpt (last 50 merges) for plotting

Dependencies
------------
  pip install scikit-learn scipy pandas numpy
  (scipy is likely already installed as a scikit-learn dependency)
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------
SCRIPT_DIR   = Path(__file__).resolve().parent       # scripts/Clustering/
PROJECT_ROOT = SCRIPT_DIR.parent.parent              # repo root
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import baseline_common as bc                         # noqa: E402

warnings.filterwarnings("ignore")

OUT_DIR  = bc.CLUSTERING_DIR    # Modeling/clustering/
K_RANGE  = range(2, 11)         # k values to evaluate after cutting the dendrogram
LINKAGE  = "ward"               # linkage method (see docstring above)
N_PCA_COMPONENTS = 50           # dimensions for clustering computation
N_PCA_PLOT       = 2            # dimensions for 2D visualisation


# ===========================================================================
# 1. Load data
# ===========================================================================

def load_data() -> pd.DataFrame:
    """
    Load all severity-known incidents. No train/test split — clustering is
    unsupervised and uses the full dataset.
    """
    print("=" * 70)
    print("1. Loading data  (no train/test split — unsupervised)")
    print("=" * 70)

    df = pd.read_csv(bc.DATA_PATH, low_memory=False)
    print(f"   Full dataset         : {df.shape[0]:,} rows × {df.shape[1]} cols")

    df_known = df[df["severity_known"] == 1].copy()
    print(f"   severity_known = 1   : {df_known.shape[0]:,} rows")
    print(f"   Severe rate          : {df_known['severe'].mean()*100:.1f}%")
    print(f"   Automation breakdown : {df_known['automation_level'].value_counts().to_dict()}")

    return df_known


# ===========================================================================
# 2. Feature preparation + scaling
# ===========================================================================

def build_and_scale(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """
    One-hot encode categoricals, then StandardScale everything.
    Scaling is mandatory for Ward linkage (Euclidean distance).
    """
    print("\n" + "=" * 70)
    print("2. Feature preparation + StandardScaler")
    print("=" * 70)

    feats = bc.context_features(df)
    print(f"   Base features        : {len(feats)}")

    X_raw, num_cols, feature_names = bc.prepare_X(df, feats)
    print(f"   After one-hot encode : {X_raw.shape[1]} columns")

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    print(f"   Scaled shape         : {X_scaled.shape}")

    return X_scaled, feature_names


# ===========================================================================
# 3. PCA — dual purpose: speed + 2D visualisation
# ===========================================================================

def run_pca(
    X_scaled: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, PCA, PCA]:
    """
    Fit two PCA projections:
      - n_components=N_PCA_COMPONENTS  (50D) → used for clustering computation
      - n_components=2                       → used for 2D scatter visualisation

    Why PCA before hierarchical clustering?
    ----------------------------------------
    Hierarchical clustering with Ward linkage builds an n×n distance matrix.
    For n ≈ 5,000 and 80+ features this is manageable, but applying PCA first:
      1. Removes correlated / noisy dimensions that can distort distances.
      2. Makes silhouette computation (O(n²)) faster.
      3. Produces directly interpretable 2D coordinates for presentation figures.

    The 50-component PCA typically explains 70–90% of variance for this type of
    one-hot + numeric mix, losing little information while gaining speed.
    """
    print("\n" + "=" * 70)
    print(f"3. PCA — {N_PCA_COMPONENTS}D for clustering, 2D for visualisation")
    print("=" * 70)

    pca_full = PCA(n_components=N_PCA_COMPONENTS, random_state=42)
    X_pca    = pca_full.fit_transform(X_scaled)
    var_full = pca_full.explained_variance_ratio_.sum() * 100
    print(f"   Variance explained ({N_PCA_COMPONENTS} PCs) : {var_full:.1f}%")

    pca_2d   = PCA(n_components=N_PCA_PLOT, random_state=42)
    X_2d     = pca_2d.fit_transform(X_scaled)
    var_2d   = pca_2d.explained_variance_ratio_.sum() * 100
    print(f"   Variance explained (2 PCs)  : {var_2d:.1f}%  ← used for scatter plots")

    return X_pca, X_2d, pca_full, pca_2d


# ===========================================================================
# 4. Linkage matrix + dendrogram data
# ===========================================================================

def compute_linkage(X_pca: np.ndarray) -> np.ndarray:
    """
    Compute the full agglomerative linkage matrix using scipy.

    The linkage matrix Z has shape (n-1, 4):
      Z[i, 0], Z[i, 1]  — indices of the two clusters merged at step i
      Z[i, 2]           — distance at which they were merged (height)
      Z[i, 3]           — number of original observations in the new cluster

    This is the data structure underlying a dendrogram. Cutting the tree at
    different heights gives different numbers of clusters. We save the last
    50 merges (the top of the dendrogram) which are the most interpretable.

    Unlike K-Means, this is computed ONCE and then we evaluate multiple k values
    by cutting the tree at different heights — much more efficient than
    re-running K-Means for each k.
    """
    print("\n" + "=" * 70)
    print(f"4. Computing linkage matrix (method='{LINKAGE}') — this may take ~30s…")
    print("=" * 70)

    Z = linkage(X_pca, method=LINKAGE)
    print(f"   Linkage matrix shape : {Z.shape}  (one row per merge step)")
    print(f"   Final merge distance : {Z[-1, 2]:.4f}  (height of full dendrogram)")
    print(f"   Saving last 50 merges as dendrogram excerpt…")

    return Z


# ===========================================================================
# 5. K selection — cut dendrogram at k = 2..10
# ===========================================================================

def select_k(
    Z: np.ndarray,
    X_pca: np.ndarray,
    X_scaled: np.ndarray,
) -> tuple[int, pd.DataFrame]:
    """
    Cut the dendrogram at each k in K_RANGE and evaluate using the same three
    metrics as K-Means (silhouette, CH, DB) so results are directly comparable.

    Dendrogram cutting works by finding the height in the linkage matrix that
    produces exactly k clusters — equivalent to drawing a horizontal line
    across the dendrogram and counting the vertical lines it intersects.

    K-Means must be RE-FIT for every k (each run is independent).
    Hierarchical clustering computes all k values from the SAME linkage matrix
    in a single pass — much more efficient for comparing many k values.
    """
    print("\n" + "=" * 70)
    print("5. K selection — cutting dendrogram at k = 2 … 10")
    print("=" * 70)

    rows     = []
    best_sil = -1
    best_k   = 2

    for k in K_RANGE:
        labels = fcluster(Z, k, criterion="maxclust") - 1   # 0-indexed

        sil = silhouette_score(X_pca, labels, sample_size=min(2000, len(labels)), random_state=42)
        ch  = calinski_harabasz_score(X_scaled, labels)
        db  = davies_bouldin_score(X_scaled, labels)

        rows.append({"k": k, "silhouette": sil, "calinski_harabasz": ch, "davies_bouldin": db})
        print(f"   k={k:2d}  silhouette={sil:.4f}  CH={ch:>8.1f}  DB={db:.4f}")

        if sil > best_sil:
            best_sil = sil
            best_k   = k

    k_df = pd.DataFrame(rows)
    print(f"\n   Best k by silhouette : k = {best_k}  (score = {best_sil:.4f})")

    return best_k, k_df


# ===========================================================================
# 6. Fit final model
# ===========================================================================

def fit_final(X_pca: np.ndarray, best_k: int) -> np.ndarray:
    """
    Fit AgglomerativeClustering with the selected k on the PCA-reduced data.
    sklearn's AgglomerativeClustering is preferred over scipy's fcluster for
    the final fit because it integrates cleanly with sklearn pipelines.
    """
    print("\n" + "=" * 70)
    print(f"6. Fitting final AgglomerativeClustering with k = {best_k}, linkage = '{LINKAGE}'")
    print("=" * 70)

    model  = AgglomerativeClustering(n_clusters=best_k, linkage=LINKAGE)
    labels = model.fit_predict(X_pca)
    unique, counts = np.unique(labels, return_counts=True)
    print(f"   Cluster sizes : { {int(u): int(c) for u, c in zip(unique, counts)} }")

    return labels


# ===========================================================================
# 7. Cluster profiles
# ===========================================================================

def profile_clusters(
    df: pd.DataFrame,
    labels: np.ndarray,
    X_2d: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build a human-readable profile for each cluster. Mirrors the K-Means
    profile function exactly so the two CSVs have identical columns and can be
    placed side-by-side in your presentation or analysis.
    """
    print("\n" + "=" * 70)
    print("7. Cluster profiles")
    print("=" * 70)

    aug = df.copy()
    aug["cluster"] = labels
    aug["pca_x"]   = X_2d[:, 0]
    aug["pca_y"]   = X_2d[:, 1]

    profiles = []
    for c in sorted(aug["cluster"].unique()):
        sub = aug[aug["cluster"] == c]
        p = {
            "cluster":          c,
            "n_incidents":      len(sub),
            "share_pct":        len(sub) / len(aug) * 100,
            "severe_rate_pct":  sub["severe"].mean() * 100,
            "pct_ADS":          (sub["automation_level"] == "ADS").mean() * 100,
            "pct_L2":           (sub["automation_level"] == "L2").mean() * 100,
            "top_crash_with":   sub["Crash With"].value_counts().index[0]
                                if "Crash With" in sub.columns else "",
            "top_roadway_type": sub["Roadway Type"].value_counts().index[0]
                                if "Roadway Type" in sub.columns else "",
            "top_sv_movement":  sub["SV Pre-Crash Movement"].value_counts().index[0]
                                if "SV Pre-Crash Movement" in sub.columns else "",
            "mean_speed_mph":   sub["SV Precrash Speed (MPH)"].mean()
                                if "SV Precrash Speed (MPH)" in sub.columns else np.nan,
        }
        profiles.append(p)
        print(f"\n   Cluster {c}  ({p['n_incidents']} incidents, {p['share_pct']:.1f}%)")
        print(f"     Severe rate         : {p['severe_rate_pct']:.1f}%  ← post-hoc label check")
        print(f"     ADS / L2            : {p['pct_ADS']:.1f}% / {p['pct_L2']:.1f}%")
        print(f"     Top crash type      : {p['top_crash_with']}")
        print(f"     Top roadway         : {p['top_roadway_type']}")
        print(f"     Top SV movement     : {p['top_sv_movement']}")
        print(f"     Mean speed (mph)    : {p['mean_speed_mph']:.1f}")

    profiles_df = pd.DataFrame(profiles)

    key_col = "incident_key" if "incident_key" in aug.columns else None
    keep    = (["incident_key"] if key_col else []) + \
              ["cluster", "pca_x", "pca_y", "severe", "automation_level"]
    assignments_df = aug[keep].copy()

    return profiles_df, assignments_df


# ===========================================================================
# 8. Post-hoc external validity vs. severe label
# ===========================================================================

def external_validity(labels: np.ndarray, y_true: np.ndarray) -> dict:
    """
    Compare cluster labels to the severe ground-truth label using ARI and NMI.
    Identical function structure to kmeans_model.py so both external validity
    CSVs have the same columns and are directly comparable.

    See kmeans_model.py for full interpretation guidance.
    """
    print("\n" + "=" * 70)
    print("8. External validity — cluster labels vs. severe ground truth")
    print("=" * 70)

    ari = adjusted_rand_score(y_true, labels)
    nmi = normalized_mutual_info_score(y_true, labels)
    print(f"   Adjusted Rand Index (ARI) : {ari:.4f}  (1=perfect, 0=chance)")
    print(f"   Normalized Mutual Info    : {nmi:.4f}  (1=perfect, 0=independent)")
    print("""
   Compare these values to kmeans_external_validity.csv:
     Similar ARI/NMI → both algorithms see the same underlying structure.
     Different ARI/NMI → the algorithms are sensitive to different aspects
       of the data geometry (e.g. cluster shape assumptions).
""")
    return {"ari": ari, "nmi": nmi}


# ===========================================================================
# 9. Main
# ===========================================================================

def main() -> None:
    print("\n" + "=" * 70)
    print("ISYE 4600 — Hierarchical (Agglomerative) Clustering")
    print("=" * 70 + "\n")

    df                       = load_data()
    X_scaled, feature_names  = build_and_scale(df)
    X_pca, X_2d, _, _        = run_pca(X_scaled)
    Z                        = compute_linkage(X_pca)
    best_k, k_df             = select_k(Z, X_pca, X_scaled)
    labels                   = fit_final(X_pca, best_k)
    profiles_df, assignments_df = profile_clusters(df, labels, X_2d)
    ext                      = external_validity(labels, df["severe"].values)

    best_row = k_df[k_df["k"] == best_k].copy()
    best_row["linkage"] = LINKAGE
    best_row["ari"]     = ext["ari"]
    best_row["nmi"]     = ext["nmi"]

    # Save dendrogram data — last 50 merges (top of the tree)
    dend_cols = ["cluster_1", "cluster_2", "distance", "n_members"]
    dend_df   = pd.DataFrame(Z[-50:], columns=dend_cols)

    # --- Save ---------------------------------------------------------------
    print("\n" + "=" * 70)
    print("9. Saving outputs to Modeling/clustering/")
    print("=" * 70)

    k_sel_path    = OUT_DIR / "hclust_k_selection.csv"
    profiles_path = OUT_DIR / "hclust_cluster_profiles.csv"
    assign_path   = OUT_DIR / "hclust_assignments.csv"
    params_path   = OUT_DIR / "hclust_best_params.csv"
    ext_path      = OUT_DIR / "hclust_external_validity.csv"
    dend_path     = OUT_DIR / "hclust_dendrogram_data.csv"

    k_df.to_csv(k_sel_path, index=False)
    profiles_df.to_csv(profiles_path, index=False)
    assignments_df.to_csv(assign_path, index=False)
    best_row.to_csv(params_path, index=False)
    pd.DataFrame([ext]).to_csv(ext_path, index=False)
    dend_df.to_csv(dend_path, index=False)

    print(f"   {k_sel_path}")
    print(f"   {profiles_path}")
    print(f"   {assign_path}")
    print(f"   {params_path}")
    print(f"   {ext_path}")
    print(f"   {dend_path}")

    print("\n" + "=" * 70)
    print("How to compare Hierarchical Clustering with other models")
    print("=" * 70)
    print("""
   DIRECTLY COMPARABLE with K-Means (same columns):
     hclust_k_selection.csv      ←→  kmeans_k_selection.csv
       Both: silhouette, calinski_harabasz, davies_bouldin
       NOTE: hclust has no inertia column (dendrogram height is used instead).
     hclust_external_validity.csv ←→  kmeans_external_validity.csv
       Both: ARI and NMI vs the severe label
     hclust_cluster_profiles.csv  ←→  kmeans_cluster_profiles.csv
       Identical columns — place side-by-side to compare discovered archetypes.

   UNIQUE to Hierarchical Clustering (no K-Means equivalent):
     hclust_dendrogram_data.csv
       Last 50 merges of the linkage matrix. The 'distance' column shows at
       what height each merge happened — large gaps between consecutive merges
       suggest a natural k at that level (the visual 'tall bars' in a dendrogram).

   NOT directly comparable to supervised models (LR / XGB / RF):
     See kmeans_model.py comparison section for full explanation.
""")
    print("Done.\n")


if __name__ == "__main__":
    main()


# ===========================================================================
# requirements (add to requirements.txt if not already present)
# ===========================================================================
# scikit-learn>=1.3   (already present)
# scipy>=1.11         (likely already installed as sklearn dependency)
# pandas>=2.0
# numpy>=1.24
