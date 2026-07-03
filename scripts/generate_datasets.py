"""
Generate benchmark datasets and upload them to S3.

Run once to populate the bucket:
    python scripts/generate_datasets.py

Requires: boto3, scikit-learn
    pip install 'convex-clustering[cloud]'
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import boto3 # type: ignore[import-untyped]
import numpy as np
import numpy.typing as npt
from sklearn.datasets import make_blobs, make_circles, make_moons

_BUCKET = "convex-clustering-andes"
_S3_PREFIX = "datasets/synthetic"
_RANDOM_STATE = 42

def _make_datasets() -> dict[str, tuple[npt.NDArray[np.float64], npt.NDArray[np.int_]]]:
    X_blobs, y_blobs = make_blobs(
        n_samples=200, centers=3, cluster_std = 0.8, random_state=_RANDOM_STATE
        )
    X_moons, y_moons = make_moons(
        n_samples=200, noise=0.08, random_state=_RANDOM_STATE
        )
    X_circles, y_circles = make_circles(
        n_samples=200, noise=0.08, factor=0.5, random_state=_RANDOM_STATE
        )
    return {
        "blobs": (X_blobs, y_blobs),
        "moons": (X_moons, y_moons),
        "circles": (X_circles, y_circles),
    }

def upload_datasets() -> None:
    s3 = boto3.client(
        "s3",
        region_name = os.environ.get("AWS_REGION", "eu-north-1"),
    )

    datasets = _make_datasets()

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        for name, (X, y) in datasets.items():
            x_path = tmp_path / f"{name}_X.npy"
            y_path = tmp_path / f"{name}_y.npy"

            np.save(x_path, X)
            np.save(y_path, y)

            for local_path, suffix in ((x_path, "_X.npy"), (y_path, "_y.npy")):
                key = f"{_S3_PREFIX}/{name}{suffix}"
                s3.upload_file(str(local_path), _BUCKET, key)
                print(f"Uploaded {key} to S3 bucket {_BUCKET}.")

if __name__ == "__main__":
    upload_datasets()