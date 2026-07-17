"""
Customer segmentation via RFM analysis using ConvexClusterer.
 
Generates all numerical results and static visualizations used in
docs/applications/customer_segmentation.md.
 
Two analysis versions:
  V1 — 3D RFM (Recency, Frequency, Monetary) without dimensionality reduction.
       Statistically correct. Centers are interpretable in business terms.
  V2 — 2D PCA projection. Enables visualization of the center fusion trajectory —
       the unique property of convex clustering not available in KMeans or DBSCAN.
 
Outputs written to results/rfm/:
  v1_metrics.json       — V1 cluster metrics and segment profiles
  v2_metrics.json       — V2 cluster metrics
  v2_fusion_path.npy    — center trajectory across iterations (for docs figure)
  rfm_summary.csv       — combined metrics table
 
Usage:
    python scripts/run_rfm_analysis.py
    python scripts/run_rfm_analysis.py --n-customers 300  # subsample size
"""
from __future__ import annotations

import argparse
import json
import time
import tracemalloc
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from convex_clustering import ConvexClusterer
from convex_clustering.utils import knn_w

_DATA_DIR = Path(__file__).parent.parent / "data"
_RESULTS_DIR = Path(__file__).parent.parent / "results" / "rfm"

def _load_rfm() -> tuple[npt.NDArray[np.float64], npt.NDArray[np.intp]]:
    x_path = _DATA_DIR / "rfm_X.npy"
    y_path = _DATA_DIR / "rfm_y.npy"
    if not x_path.exists() or not y_path.exists():
        raise FileNotFoundError(
            f"RFM data files not found in {_DATA_DIR}. "
            "Please run `scripts/generate_rfm_data.py` first."
        )
    return np.load(x_path), np.load(y_path)

def _subsample(
    X: npt.NDArray[np.float64],
    y: npt.NDArray[np.intp],
    n: int,
    seed: int = 42,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.intp], npt.NDArray[np.intp]]:
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), min(n, len(X)), replace=False)
    return X[idx], y[idx], idx

def _segment_profiles(
    X_orig: npt.NDArray[np.float64],
    labels: npt.NDArray[np.intp],
    scaler: StandardScaler,
) -> list[dict[str, object]]:
    """Return mean RFM in original (un-scaled) units per cluster."""
    profiles = []
    for seg in sorted(set(labels.tolist())):
        mask = labels == seg
        center_scaled = X_orig[mask].mean(axis=0)
        center_orig   = scaler.inverse_transform(center_scaled.reshape(1, -1))[0]
        profiles.append({
            "cluster": int(seg),
            "n_points": int(mask.sum()),
            "recency_days":   round(float(center_orig[0]), 1),
            "frequency_orders": round(float(center_orig[1]), 1),
            "monetary_gbp":   round(float(center_orig[2]), 1),
        })
    return profiles

def run_rfm_analysis(n_customers: int = 300) -> None:
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
 
    X_raw, y = _load_rfm()
    print(f"Dataset: {X_raw.shape[0]:,} customers")
 
    scaler    = StandardScaler()
    X_scaled  = scaler.fit_transform(X_raw)
 
    pca       = PCA(n_components=2, random_state=42)
    X_2d      = pca.fit_transform(X_scaled)
 
    print(f"PCA variance explained: {pca.explained_variance_ratio_.round(3)}"
          f"  (cumulative {pca.explained_variance_ratio_.sum():.3f})")
 
    X_sub, y_sub, idx = _subsample(X_scaled, y, n=n_customers)
    X_sub_2d           = X_2d[idx]
    print(f"Analysis subsample: {len(X_sub)} customers")
 
    W_3d = knn_w(X_sub,    k=5, phi=0.5)
    W_2d = knn_w(X_sub_2d, k=5, phi=0.5)
 
    records: list[dict[str, object]] = []
 
    # ── 3D RFM ─────────────────────────────────────────────────
    print("\n=== V1: 3D RFM (gamma=7, ADMM) ===")
    tracemalloc.start()
    t0 = time.perf_counter()
 
    model_v1 = ConvexClusterer(
        algorithm  = "ADMM",
        gamma      = 7.0,
        step_size  = 0.5,
        max_iter   = 10000,
        tol        = 1e-4,
        merge_tol  = 0.5,
    )
    model_v1.fit(X_sub, W_3d)
 
    elapsed_v1 = time.perf_counter() - t0
    _, peak_v1 = tracemalloc.get_traced_memory()
    tracemalloc.stop()
 
    lbl_v1  = model_v1.labels_
    k_v1    = len(set(lbl_v1.tolist()))
    sil_v1  = float(silhouette_score(X_sub, lbl_v1)) if k_v1 > 1 else -1.0
    db_v1   = float(davies_bouldin_score(X_sub, lbl_v1)) if k_v1 > 1 else -1.0
    profiles_v1 = _segment_profiles(X_sub, lbl_v1, scaler)
 
    print(f"  n_clusters={k_v1},  iter={model_v1.n_iter_}")
    print(f"  Silhouette={sil_v1:.3f},  Davies-Bouldin={db_v1:.3f}")
    print(f"  Time={elapsed_v1:.2f}s,  Peak={peak_v1/1e6:.2f}MB")
    for p in profiles_v1:
        print(f"  Cluster {p['cluster']} ({p['n_points']} pts): "
              f"R={p['recency_days']}d  F={p['frequency_orders']}  M={p['monetary_gbp']} GBP")
 
    v1_out = {
        "version": "3D_RFM",
        "algorithm": "ADMM",
        "gamma": 7.0,
        "n_customers": len(X_sub),
        "n_clusters": k_v1,
        "n_iter": model_v1.n_iter_,
        "silhouette_score": round(sil_v1, 4),
        "davies_bouldin_score": round(db_v1, 4),
        "fit_time_s": round(elapsed_v1, 3),
        "peak_memory_mb": round(peak_v1 / 1e6, 3),
        "pca_variance_explained": None,
        "segment_profiles": profiles_v1,
    }
    with open(_RESULTS_DIR / "v1_metrics.json", "w") as f:
        json.dump(v1_out, f, indent=2)
 
    records.append({
        "version": "V1 — 3D RFM",
        "n_clusters": k_v1,
        "silhouette": round(sil_v1, 3),
        "davies_bouldin": round(db_v1, 3),
        "fit_time_s": round(elapsed_v1, 3),
        "pca_variance": "—",
    })
 
    # ── PCA 2D ─────────────────────────────────────────────────
    print("\n=== V2: PCA 2D (gamma=10, ADMM) ===")
    tracemalloc.start()
    t0 = time.perf_counter()
 
    model_v2 = ConvexClusterer(
        algorithm  = "ADMM",
        gamma      = 10.0,
        step_size  = 0.5,
        max_iter   = 10000,
        tol        = 1e-4,
        merge_tol  = 0.5,
    )
    model_v2.fit(X_sub_2d, W_2d)
 
    elapsed_v2 = time.perf_counter() - t0
    _, peak_v2 = tracemalloc.get_traced_memory()
    tracemalloc.stop()
 
    lbl_v2  = model_v2.labels_
    k_v2    = len(set(lbl_v2.tolist()))
    sil_v2  = float(silhouette_score(X_sub_2d, lbl_v2)) if k_v2 > 1 else -1.0
    db_v2   = float(davies_bouldin_score(X_sub_2d, lbl_v2)) if k_v2 > 1 else -1.0
 
    print(f"  n_clusters={k_v2},  iter={model_v2.n_iter_}")
    print(f"  Silhouette={sil_v2:.3f},  Davies-Bouldin={db_v2:.3f}")
    print(f"  Time={elapsed_v2:.2f}s,  Peak={peak_v2/1e6:.2f}MB")
 
    hist_keys = sorted(model_v2.centers_hist_.keys())
    sample_keys = [k for k in hist_keys if k % 5 == 0] + [hist_keys[-1]]
    fusion_path = np.stack(
        [model_v2.centers_hist_[k] for k in sample_keys], axis=0
    )  
    np.save(_RESULTS_DIR / "v2_fusion_path.npy", fusion_path)
    print(f"  Fusion path saved: {fusion_path.shape} (frames, points, dims)")
 
    pca_var_str = (
        f"{pca.explained_variance_ratio_[0]:.3f} + "
        f"{pca.explained_variance_ratio_[1]:.3f} = "
        f"{pca.explained_variance_ratio_.sum():.3f}"
    )
 
    v2_out = {
        "version": "2D_PCA",
        "algorithm": "ADMM",
        "gamma": 10.0,
        "n_customers": len(X_sub_2d),
        "n_clusters": k_v2,
        "n_iter": model_v2.n_iter_,
        "silhouette_score": round(sil_v2, 4),
        "davies_bouldin_score": round(db_v2, 4),
        "fit_time_s": round(elapsed_v2, 3),
        "peak_memory_mb": round(peak_v2 / 1e6, 3),
        "pca_variance_explained": pca_var_str,
        "pca_components": {
            "PC1": {"R": round(float(pca.components_[0, 0]), 3),
                    "F": round(float(pca.components_[0, 1]), 3),
                    "M": round(float(pca.components_[0, 2]), 3)},
            "PC2": {"R": round(float(pca.components_[1, 0]), 3),
                    "F": round(float(pca.components_[1, 1]), 3),
                    "M": round(float(pca.components_[1, 2]), 3)},
        },
    }
    with open(_RESULTS_DIR / "v2_metrics.json", "w") as f:
        json.dump(v2_out, f, indent=2)
 
    records.append({
        "version": "V2 — PCA 2D",
        "n_clusters": k_v2,
        "silhouette": round(sil_v2, 3),
        "davies_bouldin": round(db_v2, 3),
        "fit_time_s": round(elapsed_v2, 3),
        "pca_variance": pca_var_str,
    })
 
    # ── Summary CSV ────────────────────────────────────────────────────────
    df = pd.DataFrame(records)
    csv_path = _RESULTS_DIR / "rfm_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSummary saved to {csv_path}")
    print(df.to_string(index=False))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--n-customers", type=int, default=300,
                   help="Subsample size for analysis (default: 300)")
    return p.parse_args()
 
 
if __name__ == "__main__":
    args = _parse_args()
    run_rfm_analysis(n_customers=args.n_customers)