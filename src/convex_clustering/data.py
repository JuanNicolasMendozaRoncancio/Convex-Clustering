from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import numpy.typing as npt

_CACHE_DIR = Path(__file__).parent.parent / ".datasets_cache"
_BUCKET = "convex-clustering-andes"
_S3_PREFIX = "datasets/synthetic"

_KNOWN_DATASETS = frozenset({"blobs", "moons", "circles"})

def load_dataset(
        name: str,
        *,
        cache_dir: Path | None = None,
        force_download: bool = False,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int_]]:
    """
    Load a benchmark dataset, downloading from S3 if not cached locally.

    Parameters
    ----------
    name : str
        Dataset name. One of: 'blobs', 'moons', 'circles'.
    cache_dir : Path, optional
        Local directory for caching. Defaults to .datasets_cache/ at repo root.
    force_download : bool, default False
        If True, re-download even if a local cache exists.

    Returns
    -------
    X : ndarray of shape (n_samples, n_features)
    y : ndarray of shape (n_samples,)
        Ground-truth cluster labels.

    Raises
    ------
    ValueError
        If name is not a known dataset.
    """
    if name not in _KNOWN_DATASETS:
        raise ValueError(f"Unknown dataset name: {name}. Must be one of {sorted(_KNOWN_DATASETS)}.")
    
    cache_dir = cache_dir or _CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)

    x_path = cache_dir / f"{name}_X.npy"
    y_path = cache_dir / f"{name}_y.npy"

    if force_download or not (x_path.exists() and y_path.exists()):
        _download_from_s3(name, x_path, y_path)

    X = np.load(x_path)
    y = np.load(y_path)
    return X, y

def _download_from_s3(
        name: str,
        x_path: Path, 
        y_path: Path
) -> None:
    try:
        import boto3 # type: ignore[import-untyped]
    except ImportError as e:
        raise ImportError(
            "boto3 is required for S3 downloads. " \
            "Install it with: pip install 'convex-clustering[cloud]'"
        ) from e
    
    s3 = boto3.client(
        "s3",
        region_name = os.environ.get("AWS_REGION", "eu-north-1"),
        ) 
    
    for suffix, local_path in (("_X.npy", x_path), ("_y.npy", y_path)):
        key = f"{_S3_PREFIX}/{name}{suffix}"
        s3.download_file(_BUCKET, key, str(local_path))