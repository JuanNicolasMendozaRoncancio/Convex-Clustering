from __future__ import annotations

import time
import tracemalloc
from pathlib import Path

import mlflow
import numpy as np
import numpy.typing as npt
from typing import Any
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from convex_clustering import ConvexClusterer
from convex_clustering.data import load_dataset
from convex_clustering.utils import knn_w

_MLFLOW_DB = "sqlite:///mlflow.db"
_EXPERIMENT_NAME = "benchmark"
_RESULTS_DIR = Path("results/benchmark")

_DATASETS = ["blobs", "moons", "circles"]

_CONVEX_PARAMS: dict[str, dict[str, Any]] ={
    "blobs": {"algorithm": "ADMM", "gamma":100, "step_size": 0.05, "max_iter": 100000, "merge_tol": 0.5},
    "moons": {"algorithm": "DR", "gamma": 10000, "step_size": 0.5, "max_iter": 1000, "merge_tol": 0.5},
    "circles": {"algorithm": "DR", "gamma": 10000, "step_size": 0.5, "max_iter": 1000, "merge_tol": 0.5},
}

_KMEANS_PARAMS: dict[str, dict[str, Any]] = {
    "blobs": {"n_clusters": 3, "random_state": 42, "n_init": 10},
    "moons": {"n_clusters": 2, "random_state": 42, "n_init": 10},
    "circles": {"n_clusters": 2, "random_state": 42, "n_init": 10},
}

_DBSCAN_PARAMS: dict[str, dict[str, Any]] = {
    "blobs": {"eps": 0.5, "min_samples": 5},
    "moons": {"eps": 0.15, "min_samples": 5},
    "circles": {"eps": 0.15, "min_samples": 5},
}


def _run_model(model: ConvexClusterer | KMeans | DBSCAN, X: npt.NDArray[np.float64]) -> tuple[npt.NDArray[np.int_], float, float]:
    """
    Adjust the model, saves time and memory usage

    Returns
    -------
    labels: np.array
        The labels assigned by the model
    time_taken: float
        The time taken to fit the model
    peak_mb: float
        The peak memory usage in MB
    """
    tracemalloc.start()
    t0 = time.perf_counter()

    if isinstance(model, ConvexClusterer):
        W = knn_w(X, k=6, phi=0.5)
        model.fit(X, W)
        labels = model.labels_
    else:
        labels = model.fit_predict(X)

    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_mb = peak / 1024 / 1024

    return labels, elapsed, peak_mb


def _compute_metrics(
        X: npt.NDArray[np.float64],
        labels: npt.NDArray[np.int_],
        elapsed: float,
        peak_mb: float,
) -> dict[str, Any]:
    """
    Computes silhouette score, Davies-Bouldin score and compute metrics
    """
    mask = labels != -1
    n_noise = int(np.sum(~mask))
    n_clusters = len(set(labels[mask].tolist())) if mask.any() else 0

    if n_clusters >= 2 and mask.sum() >= 2:
        sil = float(silhouette_score(X[mask], labels[mask]))
        db = float(davies_bouldin_score(X[mask], labels[mask]))
    else:
        sil = -1
        db = -1

    return {
        "n_clusters_found": n_clusters,
        "n_noise_points": n_noise,
        "silhouette_score": sil,
        "davies_bouldin_score": db,
        "fit_time_s":round(elapsed, 4),
        "peak_memory_mb": round(peak_mb, 4),
    }


def run_benchmark() -> pd.DataFrame:
    mlflow.set_tracking_uri(_MLFLOW_DB)
    mlflow.set_experiment(_EXPERIMENT_NAME)

    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []

    for dataset_name in _DATASETS:
        print(f"\nProcessing dataset: {dataset_name}")

        X_raw, y_true = load_dataset(dataset_name)
        n_true_clusters = len(set(y_true.tolist()))

        scaler = StandardScaler()
        X = scaler.fit_transform(X_raw)

        models: dict[str, object] = {
            "ConvexClusterer": ConvexClusterer(**_CONVEX_PARAMS[dataset_name]),
            "KMeans": KMeans(**_KMEANS_PARAMS[dataset_name]),
            "DBSCAN": DBSCAN(**_DBSCAN_PARAMS[dataset_name]),
        }

        for model_name, model in models.items():
            print(f"\nRunning model: {model_name}")
            try:
                labels, elapsed, peak_mb = _run_model(model, X)
                metrics = _compute_metrics(X, labels, elapsed, peak_mb)
            except Exception as e:
                print(f"Failed: {e}")
                metrics = {
                    "n_clusters_found": -1,
                    "n_noise_points": -1,
                    "silhouette_score": -1,
                    "davies_bouldin_score": -1,
                    "fit_time_s": -1,
                    "peak_memory_mb": -1,
                }

            if model_name == "ConvexClusterer":
                hparams = _CONVEX_PARAMS[dataset_name]
            elif model_name == "KMeans":
                hparams = _KMEANS_PARAMS[dataset_name]
            else:
                hparams = _DBSCAN_PARAMS[dataset_name]

            with mlflow.start_run(
                run_name = f"{dataset_name}_{model_name}",
            ):
                mlflow.log_params({
                    "dataset": dataset_name,
                    "model": model_name,
                    "n_true_clusters": n_true_clusters,
                    "n_samples": X.shape[0],
                    **hparams,
                })
                mlflow.log_metrics({
                    "silhouette_score": metrics["silhouette_score"],
                    "davies_bouldin_score": metrics["davies_bouldin_score"],
                    "fit_time_s": metrics["fit_time_s"],
                    "peak_memory_mb": metrics["peak_memory_mb"],
                    "n_clusters_found": metrics["n_clusters_found"],
                    "n_noise_points": metrics["n_noise_points"],
                })
            
            record = {
                "dataset": dataset_name,
                "model": model_name,
                "true_n_clusters": n_true_clusters,
                **metrics,
            }
            records.append(record)

            print(f"    clusters found : {metrics['n_clusters_found']}")
            print(f"    silhouette     : {metrics['silhouette_score']:.3f}")
            print(f"    davies-bouldin : {metrics['davies_bouldin_score']:.3f}")
            print(f"    fit time       : {metrics['fit_time_s']:.3f}s")
            print(f"    peak memory    : {metrics['peak_memory_mb']:.2f}MB")

        df = pd.DataFrame(records)
        csv_path = _RESULTS_DIR / "benchmark_results.csv"
        df.to_csv(csv_path, index=False)
        print(f"\nBenchmark results saved to: {csv_path}")
    
    return df
    

if __name__ == "__main__":
    df_results = run_benchmark()
    print("\nBenchmarking completed.")
    print(df_results.to_string(index=False))

            