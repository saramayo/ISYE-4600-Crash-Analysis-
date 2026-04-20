"""
ISYE 4600 AV Crash Severity Project — K-Means Clustering
=========================================================

Run after 01_clean_incidents.py (requires Cleaned/sgo_cleaned_incidents.csv).

What this script does
---------------------
This script is UNSUPERVISED — the severe label is never used during training.
K-Means discovers natural groupings in the incident feature space on its own.
The severe label is only consulted AFTER clustering (post-hoc validation) to
measure how well the discovered clusters align with severity.

Pipeline
--------
1. Load ALL severity-known incidents (no train/test split — clustering uses
   the full dataset; there is no held-out prediction task).
2. Scale features with StandardScaler — required for any distance-based method.
3. Reduce to 2D via PCA for visualisation and silhouette speed.
4. Select optimal k via elbow (inertia) + silhouette analysis across k = 2–10.
5. Fit final K-Means with the best k.
6. Profile each cluster: size, severe rate, automation level breakdown,
   dominant crash type, dominant roadway type, mean speed.
7. Post-hoc external validation: compare cluster labels to severe (ARI, NMI).

Why K-Means is fundamentally different from LR / XGBoost / RF
--------------------------------------------------------------
  Supervised models (LR, XGB, RF) learn a mapping from features → severe label.
  Their evaluation metrics (precision, recall, F1, ROC-AUC) measure how well
  that mapping reproduces held-out ground-truth labels.

  K-Means has no label. It groups incidents by geometric similarity in feature
  space. Evaluation measures whether the resulting clusters are internally
  tight and externally separated — not whether they predict a label.

  Key implication: you cannot compare silhouette scores to F1 scores.
  They measure entirely different concepts. What you CAN compare is:
    - After labelling clusters, do high-severe-rate clusters look like the
      cases your classifiers struggled with (e.g. the ADS slice)?
    - Do the cluster profiles reveal incident archetypes not visible in the
      coefficient / importance outputs of the supervised models?

Why feature scaling is required here (unlike RF / XGBoost)
-----------------------------------------------------------
  K-Means minimises Euclidean distance. Features on large scales (e.g. speed
  in MPH) would dominate the distance calculation and drown out binary flags.
  StandardScaler (zero mean, unit variance) puts all features on equal footing.
  RF and XGBoost use thresholds, not distances, so scaling is irrelevant there.

Outputs (written to Modeling/clustering/)
------------------------------------------
  kmeans_k_selection.csv      — inertia + silhouette + CH + DB scores for k=2..10
  kmeans_cluster_profiles.csv — per-cluster summary (size, severe rate, top features)
  kmeans_assignments.csv      — one row per incident: cluster label + PCA coords + severe
  kmeans_best_params.csv      — optimal k and its evaluation metrics
  kmeans_external_validity.csv— ARI and NMI vs the severe label (post-hoc validation)

Dependencies
------------
  pip install scikit-learn pandas numpy
  (all already in your requirements.txt)
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
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

OUT_DIR = bc.CLUSTERING_DIR   # Modeling/clustering/

K_RANGE = range(2, 11)        # k values to evaluate in the selection step


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

def build_and_scale(df: pd.DataFrame) -> tuple[np.ndarray, list[str], StandardScaler]:
    """
    One-hot encode categoricals, then StandardScale everything.
    Returns the scaled array, feature names, and the fitted scaler.
    """
    print("\n" + "=" * 70)
    print("2. Feature preparation + StandardScaler")
    print("=" * 70)

    feats = bc.context_features(df)
    print(f"   Base features        : {len(feats)}")

    X_raw, num_cols, feature_names = bc.prepare_X(df, feats)
    print(f"   After one-hot encode : {X_raw.shape[1]} columns")

    scaler  = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    print(f"   Scaled shape         : {X_scaled.shape}")
    print("   Scaling: StandardScaler (zero mean, unit variance)")
    print("   NOTE: Scaling is mandatory for K-Means (Euclidean distance).")
    print("         RF and XGBoost do not require it (threshold-based splits).")

    return X_scaled, feature_names, scaler


# ===========================================================================
# 3. PCA — dimensionality reduction for speed + visualisation
# ===========================================================================

def run_pca(X_scaled: np.ndarray, n_components: int = 2) -> tuple[np.ndarray, PCA]:
    """
    Reduce to n_components principal components.

    Why PCA before K-Means?
    -----------------------
    With 80+ one-hot features, many dimensions are sparse and correlated.
    PCA rotates the feature space to find the directions of maximum variance,
    removes noise dimensions, and dramatically speeds up silhouette computation
    (which is O(n²) in dimension). The 2D projection is also used directly
    for scatter-plot visualisation in your presentation figures.

    The full scaled matrix (not PCA) is used for the final cluster fit so no
    information is discarded in the actual model.
    """
    print("\n" + "=" * 70)
    print("3. PCA dimensionality reduction (2D for visualisation & silhouette speed)")
    print("=" * 70)

    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    var_explained = pca.explained_variance_ratio_.sum() * 100
    print(f"   Variance explained by {n_components} components: {var_explained:.1f}%")
    for i, v in enumerate(pca.explained_variance_ratio_):
        print(f"   PC{i+1}: {v*100:.1f}%")

    return X_pca, pca


# ===========================================================================
# 4. K selection — elbow + silhouette
# ===========================================================================

def select_k(X_scaled: np.ndarray, X_pca: np.ndarray) -> tuple[int, pd.DataFrame]:
    """
    Evaluate K-Means for each k in K_RANGE on four metrics:

    Inertia (elbow method)
    ----------------------
    Within-cluster sum of squared distances to centroids. Always decreases as
    k increases; the 'elbow' point where the rate of decrease slows is a
    heuristic for the right k. Unique to K-Means (hierarchical uses
    dendrogram height instead).

    Silhouette score  [-1, 1]
    -------------------------
    For each point: (distance to nearest OTHER cluster - distance to own cluster)
    / max of the two. Score of 1 = perfectly separated; 0 = on boundary;
    negative = likely mis-clustered. Higher is better. Comparable across
    K-Means and Hierarchical Clustering.

    Calinski-Harabasz score  (higher = better)
    ------------------------------------------
    Ratio of between-cluster variance to within-cluster variance. Penalises
    both loose clusters and clusters that are too close together.

    Davies-Bouldin score  (lower = better)
    ---------------------------------------
    Average similarity between each cluster and its most similar neighbour.
    Lower = more distinct clusters.
    """
    print("\n" + "=" * 70)
    print("4. K selection — evaluating k = 2 … 10")
    print("=" * 70)

    rows = []
    best_sil = -1
    best_k   = 2

    for k in K_RANGE:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)

        inertia = km.inertia_
        sil     = silhouette_score(X_pca, labels, sample_size=min(2000, len(labels)), random_state=42)
        ch      = calinski_harabasz_score(X_scaled, labels)
        db      = davies_bouldin_score(X_scaled, labels)

        rows.append({"k": k, "inertia": inertia, "silhouette": sil,
                     "calinski_harabasz": ch, "davies_bouldin": db})

        print(f"   k={k:2d}  inertia={inertia:>10.1f}  silhouette={sil:.4f}"
              f"  CH={ch:>8.1f}  DB={db:.4f}")

        if sil > best_sil:
            best_sil = sil
            best_k   = k

    k_df = pd.DataFrame(rows)
    print(f"\n   Best k by silhouette : k = {best_k}  (score = {best_sil:.4f})")

    return best_k, k_df


# ===========================================================================
# 5. Fit final model
# ===========================================================================

def fit_final(X_scaled: np.ndarray, best_k: int) -> KMeans:
    """Fit K-Means with the selected k. Uses n_init=20 for stability."""
    print("\n" + "=" * 70)
    print(f"5. Fitting final K-Means with k = {best_k}")
    print("=" * 70)

    km = KMeans(n_clusters=best_k, random_state=42, n_init=20)
    km.fit(X_scaled)
    print(f"   Iterations to converge : {km.n_iter_}")
    print(f"   Final inertia          : {km.inertia_:.1f}")

    return km


# ===========================================================================
# 6. Cluster profiles
# ===========================================================================

def profile_clusters(
    df: pd.DataFrame,
    labels: np.ndarray,
    X_pca: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build a human-readable profile for each cluster showing:
      - size and share of total incidents
      - severe rate (post-hoc — the label was never used during training)
      - automation level breakdown
      - dominant crash type, roadway type, pre-crash movement
      - mean pre-crash speed

    Also returns an assignments DataFrame with one row per incident.
    """
    print("\n" + "=" * 70)
    print("6. Cluster profiles")
    print("=" * 70)

    aug = df.copy()
    aug["cluster"]  = labels
    aug["pca_x"]    = X_pca[:, 0]
    aug["pca_y"]    = X_pca[:, 1]

    profiles = []
    for c in sorted(aug["cluster"].unique()):
        sub = aug[aug["cluster"] == c]
        p = {
            "cluster":             c,
            "n_incidents":         len(sub),
            "share_pct":           len(sub) / len(aug) * 100,
            "severe_rate_pct":     sub["severe"].mean() * 100,
            "pct_ADS":             (sub["automation_level"] == "ADS").mean() * 100,
            "pct_L2":              (sub["automation_level"] == "L2").mean() * 100,
            "top_crash_with":      sub["Crash With"].value_counts().index[0]
                                   if "Crash With" in sub.columns else "",
            "top_roadway_type":    sub["Roadway Type"].value_counts().index[0]
                                   if "Roadway Type" in sub.columns else "",
            "top_sv_movement":     sub["SV Pre-Crash Movement"].value_counts().index[0]
                                   if "SV Pre-Crash Movement" in sub.columns else "",
            "mean_speed_mph":      sub["SV Precrash Speed (MPH)"].mean()
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

    profiles_df    = pd.DataFrame(profiles)
    assignments_df = aug[["incident_key" if "incident_key" in aug.columns else aug.index.name,
                           "cluster", "pca_x", "pca_y", "severe",
                           "automation_level"]].copy() if "incident_key" in aug.columns else \
                     aug[["cluster", "pca_x", "pca_y", "severe", "automation_level"]].copy()

    return profiles_df, assignments_df


# ===========================================================================
# 7. Post-hoc external validity vs. severe label
# ===========================================================================

def external_validity(labels: np.ndarray, y_true: np.ndarray) -> dict:
    """
    Compare discovered cluster labels to the severe ground-truth label using
    two information-theoretic measures that do NOT require knowing the mapping
    between cluster IDs and class labels.

    Adjusted Rand Index (ARI)  [-1, 1]
    ------------------------------------
    Measures agreement between two label assignments, corrected for chance.
    ARI = 1 means perfect agreement; ARI = 0 means agreement no better than
    random; negative = worse than random. Directly comparable between K-Means
    and Hierarchical Clustering.

    Normalized Mutual Information (NMI)  [0, 1]
    --------------------------------------------
    How much information the cluster labels share with the true labels,
    normalised so 1 = perfect correspondence and 0 = no relationship.
    Less sensitive to cluster count than ARI.

    Interpretation for this project
    --------------------------------
    High ARI/NMI → the algorithm found clusters that map well onto
    severe vs. not-severe, without ever seeing the label.
    Low ARI/NMI → the clusters capture a different structure in the
    data (e.g. automation level, crash type) rather than severity per se.
    Neither outcome is 'wrong' — both are informative.
    """
    print("\n" + "=" * 70)
    print("7. External validity — cluster labels vs. severe ground truth")
    print("=" * 70)

    ari = adjusted_rand_score(y_true, labels)
    nmi = normalized_mutual_info_score(y_true, labels)
    print(f"   Adjusted Rand Index (ARI) : {ari:.4f}  (1=perfect, 0=chance, <0=worse)")
    print(f"   Normalized Mutual Info    : {nmi:.4f}  (1=perfect, 0=independent)")
    print("""
   These metrics tell you whether the unsupervised clusters accidentally
   'rediscovered' the severe label. Compare across K-Means and Hierarchical
   Clustering to see which algorithm's groupings are more aligned with severity.
   NOTE: Low scores are not failures — they mean the clusters reveal a
   different structure (e.g. automation level or crash context) which can
   still be valuable for your analysis.
""")
    return {"ari": ari, "nmi": nmi}


# ===========================================================================
# 8. Main
# ===========================================================================

def main() -> None:
    print("\n" + "=" * 70)
    print("ISYE 4600 — K-Means Clustering")
    print("=" * 70 + "\n")

    df          = load_data()
    X_scaled, feature_names, scaler = build_and_scale(df)
    X_pca, pca  = run_pca(X_scaled)
    best_k, k_df = select_k(X_scaled, X_pca)
    km          = fit_final(X_scaled, best_k)
    labels      = km.labels_
    profiles_df, assignments_df = profile_clusters(df, labels, X_pca)
    ext         = external_validity(labels, df["severe"].values)

    best_row = k_df[k_df["k"] == best_k].copy()
    best_row["ari"] = ext["ari"]
    best_row["nmi"] = ext["nmi"]

    # --- Save ---------------------------------------------------------------
    print("\n" + "=" * 70)
    print("8. Saving outputs to Modeling/clustering/")
    print("=" * 70)

    k_sel_path    = OUT_DIR / "kmeans_k_selection.csv"
    profiles_path = OUT_DIR / "kmeans_cluster_profiles.csv"
    assign_path   = OUT_DIR / "kmeans_assignments.csv"
    params_path   = OUT_DIR / "kmeans_best_params.csv"
    ext_path      = OUT_DIR / "kmeans_external_validity.csv"

    k_df.to_csv(k_sel_path, index=False)
    profiles_df.to_csv(profiles_path, index=False)
    assignments_df.to_csv(assign_path, index=False)
    best_row.to_csv(params_path, index=False)
    pd.DataFrame([ext]).to_csv(ext_path, index=False)

    print(f"   {k_sel_path}")
    print(f"   {profiles_path}")
    print(f"   {assign_path}")
    print(f"   {params_path}")
    print(f"   {ext_path}")

    print("\n" + "=" * 70)
    print("How to compare K-Means with other models in this project")
    print("=" * 70)
    print("""
   DIRECTLY COMPARABLE across K-Means and Hierarchical Clustering:
     kmeans_k_selection.csv     ←→  hclust_k_selection.csv
       Both report: silhouette, calinski_harabasz, davies_bouldin
     kmeans_external_validity.csv ←→  hclust_external_validity.csv
       Both report: ARI and NMI vs the severe label

   NOT directly comparable to supervised models (LR / XGB / RF):
     Clustering evaluation metrics (silhouette, inertia, ARI, NMI) measure
     geometric structure, not predictive accuracy. Precision / recall / F1
     have no equivalent meaning in an unsupervised context.

   MEANINGFUL cross-model comparison (qualitative):
     kmeans_cluster_profiles.csv  ←→  lr_coefficients.csv / xgb_feature_importance.csv
       Do the features that dominate cluster separation match the features
       the supervised models found most predictive of severity?
     kmeans_assignments.csv (severe_rate per cluster) ←→ supervised model FN analysis
       Are the clusters with low severe_rate the same incidents the supervised
       models misclassified as false negatives?
""")
    print("Done.\n")


if __name__ == "__main__":
    main()


# ===========================================================================
# requirements (all already in requirements.txt)
# ===========================================================================
# scikit-learn>=1.3
# pandas>=2.0
# numpy>=1.24
