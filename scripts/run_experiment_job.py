"""
Cloud Run Job — ConvexClusterer experiment runner.

Loads a dataset from S3, runs ConvexClusterer with the given hyperparameters,
and saves results back to S3 under results/{exp_id}/.

Usage (local):
    python scripts/run_experiment_job.py --dataset blobs --algorithm ADMM --gamma 1.0

Usage (Cloud Run):
    Configured via environment variables or CLI args in the job definition.
"""
from __future__ import annotations

import mlflow
import argparse
import json
import os
import time
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import boto3
import numpy as np
from sklearn.metrics import silhouette_score
import tracemalloc

from convex_clustering import ConvexClusterer
from convex_clustering.data import load_dataset

_BUCKET = "convex-clustering-andes"
_RESULTS_PREFIX = "results"

def _make_experiment_id(
        dataset: str,
        algorithm: str,
        gamma: float,
        merge_tol: float = 0.5,
) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    return f"{dataset}_{algorithm}_gamma{gamma:.3f}_merge_tol{merge_tol:.3f}_{ts}"

def _upload_to_s3(
        local_path: Path,
        s3_key: str,
) -> None:
    s3 = boto3.client("s3",
                      region_name=os.environ.get("AWS_REGION", "eu-north-1")
    )
    s3.upload_file(str(local_path), _BUCKET, s3_key)
    print(f"Uploaded {local_path} to s3://{_BUCKET}/{s3_key}")

def run_experiment(
        dataset: str,
        algorithm: str,
        gamma: float,
        step_size: float,
        max_iter: int,
        merge_tol: float,
) -> None:
    exp_id = _make_experiment_id(dataset, algorithm, gamma, merge_tol)
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("convex_clustering")
    print(f"Starting experiment {exp_id}")

    X, y_true = load_dataset(dataset)
    print(f"Loaded dataset '{dataset}': X={X.shape}, y={y_true.shape}")

    from convex_clustering.utils import knn_w
    W = knn_w(X, k=5, phi = 0.5)

    with mlflow.start_run(run_name=exp_id):
        model = ConvexClusterer(
            algorithm=algorithm,
            gamma=gamma,
            step_size=step_size,
            max_iter=max_iter,
            merge_tol=merge_tol,
        )
        t0 = time.perf_counter()
        tracemalloc.start()
        model.fit(X, W)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        elapsed = time.perf_counter() - t0
        peak_mb = peak / 1024 / 1024
        print(f"Fit complete — {model.n_iter_} iterations, "
            f"{len(set(model.labels_.tolist()))} clusters found, "
            f"peak memory {peak_mb:.2f} MB")
        
        sil = silhouette_score(X, model.labels_) if len(set(model.labels_.tolist())) > 1 else 0.0

        mlflow.log_params({
            "dataset": dataset,
            "algorithm": algorithm,
            "gamma": gamma,
            "step_size": step_size,
            "max_iter": max_iter,
            "merge_tol": merge_tol,
        })
        mlflow.log_metrics({
            "n_clusters": float(len(set(model.labels_.tolist()))),
            "n_iter": float(model.n_iter_),
            "silhouette_score": float(sil),
            "peak_memory_mb": float(peak_mb),
        })
    
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)

            config = {
                "exp_id": exp_id,
                "dataset": dataset,
                "algorithm": algorithm,
                "gamma": gamma,
                "step_size": step_size,
                "max_iter": max_iter,
                "merge_tol": merge_tol,
                "n_samples": X.shape[0],
                "n_features": X.shape[1],
            }
            config_path = tmp_path / "config.json"
            with open(config_path, "w") as f:
                json.dump(config, f, indent=2)

            convergence_path = tmp_path / "convergence.csv"
            with open(convergence_path, "w") as f:
                f.write("iteration, center_diff\n")
                for it, val in model.history_.items():
                    f.write(f"{it}, {val}\n")
            mlflow.log_artifact(str(convergence_path), artifact_path="convergence")

            metrics_path = tmp_path / "metrics.csv"
            with open(metrics_path, "w") as f:
                f.write("Silhouette_score, fit_time_seconds, peak_memory_mb, n_clusters, n_iter\n")
                f.write(f"{sil}, {elapsed:.4f}, {peak_mb:.2f}, {len(set(model.labels_.tolist()))}, {model.n_iter_}\n")
            mlflow.log_artifact(str(metrics_path), artifact_path="metrics")

            paths_path = tmp_path / "paths.npy"
            paths_arr = np.stack(list(model.centers_hist_.values()), axis=0)
            np.save(paths_path, paths_arr)

            labels_path = tmp_path / "labels.npy"
            np.save(labels_path, model.labels_)

            for local_file, suffix in (
                (config_path, "config.json"),
                (metrics_path, "metrics.csv"),
                (convergence_path, "convergence.csv"),
                (paths_path, "paths.npy"),
                (labels_path, "labels.npy"),
            ):
                s3_key = f"{_RESULTS_PREFIX}/{exp_id}/{suffix}"
                _upload_to_s3(local_file, s3_key)

    print(f"Experiment {exp_id} completed.")

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ConvexClusterer Cloud Run Job"
    )
    parser.add_argument("--dataset",default="blobs",
                        choices=["blobs","moons", "circles"])
    parser.add_argument("--algorithm", default="DR",
                        choices=["ADMM", "AMA", "DR","RFS_L2",
                                 "Fast_RFS_L2", "RFS_L1", "Fast_RFS_L1"])
    parser.add_argument("--gamma", type=float, default=100.0)
    parser.add_argument("--step_size", type=float, default=0.01)
    parser.add_argument("--max_iter", type=int, default=10000)
    parser.add_argument("--merge_tol", type=float, default=0.5)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_experiment(
        dataset=args.dataset,
        algorithm=args.algorithm,
        gamma=args.gamma,
        step_size=args.step_size,
        max_iter=args.max_iter,
        merge_tol=args.merge_tol,
    )