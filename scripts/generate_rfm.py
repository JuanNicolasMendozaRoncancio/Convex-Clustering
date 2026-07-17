"""
Generate an RFM (Recency, Frequency, Monetary) dataset for customer segmentation.
 
Primary source: Online Retail Dataset (UCI Machine Learning Repository).
  Download manually from:
    https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci
    or https://archive.ics.uci.edu/dataset/352/online+retail
  Place the file at: data/online_retail.xlsx
 
Fallback: if the file is not found, a realistic synthetic dataset is generated
that mirrors the statistical properties of the real dataset (~4,300 customers,
three natural business segments: Champions, At-Risk, Regulars).
 
Outputs (saved to S3 and local cache):
  - data/rfm_X.npy      — feature matrix (n_customers, 3): [Recency, Frequency, Monetary]
  - data/rfm_y.npy      — segment labels (n_customers,):   0=Champions, 1=At-Risk, 2=Regulars
  - data/rfm_meta.json  — provenance, statistics, and preprocessing parameters
 
Usage:
    python scripts/generate_rfm.py
    python scripts/generate_rfm.py --source synthetic   # force synthetic
    python scripts/generate_rfm.py --no-upload          # skip S3 upload
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import numpy.typing as npt

_BUCKET = "convex-clustering-andes"
_S3_PREFIX = "datasets/rfm"
_DATA_DIR = Path(__file__).parent.parent / "data"
_RANDOM_STATE = 42

def _load_online_retail(path: Path) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.intp], str]:
    """
    Load and compute RFM features from the Online Retail XLSX.
 
    Parameters
    ----------
    path : Path
        Path to online_retail.xlsx downloaded from UCI/Kaggle.
 
    Returns
    -------
    X : ndarray of shape (n_customers, 3)
        Raw RFM matrix [Recency (days), Frequency (orders), Monetary (GBP)].
    y : ndarray of shape (n_customers,)
        Placeholder labels (-1) — true segments unknown for real data.
    source : str
        Provenance string logged in metadata.
    """
    try:
        import pandas as pd
    except ImportError as e:
        raise ImportError(
            "pandas is required to load the Online Retail dataset. "
            "Please install it with `pip install pandas`."
        ) from e

    print(f"Loading Online Retail dataset from {path}...")
    df = pd.read_excel(path, dtype={"CustomerID": str})

    df = df.dropna(subset=["CustomerID"])
    df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]  # Remove canceled orders
    df = df[df["quantity"] > 0]
    df = df[df["unitPrice"] > 0]

    snapshot = df["InvoiceDate"].max() + pd.Timedelta(days=1)
    df["TotalPrice"] = df["quantity"] * df["unitPrice"]

    rfm = df.groupby("CustomerID").agg(
        Recency=("InvoiceDate", lambda x: (snapshot - x.max()).days),
        Frequency=("InvoiceNo", "nunique"),
        Monetary=("TotalPrice", "sum"),
    ).reset_index()

    rfm = rfm[rfm["Monetary"] > 0]

    X = rfm[["Recency", "Frequency", "Monetary"]].values.astype(np.float64)
    y = np.full(len(rfm), -1, dtype=np.intp)

    print(f"Cleaned dataset, {len(rfm)} customers")
    return X, y, "UCI Online Retail Dataset"


def _generate_synthetic() -> tuple[npt.NDArray[np.float64], npt.NDArray[np.intp], str]:
    """
    Generate a synthetic RFM dataset mirroring Online Retail statistics.
 
    Three business segments with realistic distributions:
      - Champions  (15%): low Recency, high Frequency, high Monetary
      - At-Risk    (25%): high Recency, low Frequency, low Monetary
      - Regulars   (60%): medium values across all three dimensions
 
    Returns
    -------
    X : ndarray of shape (4338, 3)
    y : ndarray of shape (4338,) — ground-truth segment labels
    source : str
    """
    rng = np.random.default_rng(_RANDOM_STATE)
    n = 4338

    n_champ = int(n * 0.15)
    n_risk = int(n * 0.25)
    n_reg = n - n_champ - n_risk

    r_c = rng.integers(1, 30, n_champ).astype(float)
    f_c = rng.integers(10, 80, n_champ).astype(float)
    m_c = rng.lognormal(7.0, 0.5, n_champ)

    r_r = rng.integers(200, 365, n_risk).astype(float)
    f_r = rng.integers(1, 5,   n_risk).astype(float)
    m_r = rng.lognormal(4.5, 0.6, n_risk) 

    r_g = rng.integers(30, 200, n_reg).astype(float)
    f_g = rng.integers(2, 15,  n_reg).astype(float)
    m_g = rng.lognormal(5.8, 0.7, n_reg)

    R = np.concatenate([r_c, r_r, r_g])
    F = np.concatenate([f_c, f_r, f_g])
    M = np.concatenate([m_c, m_r, m_g])

    X = np.column_stack([R, F, M])
    y = np.concatenate([
        np.zeros(n_champ, dtype=np.intp),
        np.ones(n_risk,   dtype=np.intp),
        np.full(n_reg, 2, dtype=np.intp),
    ])

    perm = rng.permutation(n)
    return X[perm], y[perm], "Synthetic RFM (mirrors Online Retail statistics)"


def _preprocess(
        X: npt.NDArrat[np.float64],
        winsor_pct: float = 0.99,
) -> tuple[npt.NDArray[np.float64], dict[str, list[float]]]:
    """
    Winsorize at the given percentile per feature.
 
    Why winsorize and not drop: the Online Retail dataset contains genuine
    high-value customers (wholesalers buying thousands of units). Dropping
    them discards real signal. Winsorizing caps their influence on the weight
    matrix W without removing them from the clustering.
 
    Parameters
    ----------
    X : ndarray of shape (n, 3)
        Raw RFM matrix.
    winsor_pct : float
        Percentile cap, applied independently per feature.
 
    Returns
    -------
    X_winsor : ndarray of shape (n, 3)
        Winsorized matrix.
    caps : dict
        The cap value applied to each feature (for reproducibility logging).
    """
    X_out = X.copy()
    caps: dict[str, list[float]] = {"percentile": [winsor_pct]*3, "values": []}
    for col in range(3):
        cap = float(np.percentile(X_out[:,col], winsor_pct))
        X_out[:,col] = np.clip(X_out[:,col], None, cap)
        caps["values"].append(cap)
    return X_out, caps

def _upload(local_path: Path, s3_key:str) -> None:
    try:
        import boto3
    except ImportError as e:
        raise ImportError(
            "boto3 is required to upload datasets to S3. Please install it with `pip install boto3`."
        ) from e
    
    s3 = boto3.client("s3", region_name = os.environ.get("AWS_REGION", "eu-north-1"))
    s3.upload_file(str(local_path), _BUCKET, s3_key)
    print(f"Uploaded {local_path} to s3://{_BUCKET}/{s3_key}")


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def generate_rfm(
        source: str = "auto",
        upload: bool = True,
) -> None:
    """
    Generate the RFM dataset and save locally (and optionally to S3).
 
    Parameters
    ----------
    source : str
        'auto'      — use real data if available, else synthetic.
        'real'      — require real data; raise if not found.
        'synthetic' — always use synthetic data.
    upload : bool
        Whether to upload to S3.
    """
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    retail_path = _DATA_DIR / "online_retail.xlsx"
 
    # --- Load data ---
    if source == "synthetic":
        X_raw, y, data_source = _generate_synthetic()
    elif source == "real":
        if not retail_path.exists():
            raise FileNotFoundError(
                f"Real dataset not found at {retail_path}.\n"
                "Download from: https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci\n"
                "and place it at data/online_retail.xlsx"
            )
        X_raw, y, data_source = _load_online_retail(retail_path)
    else: 
        if retail_path.exists():
            X_raw, y, data_source = _load_online_retail(retail_path)
        else:
            print("Real dataset not found — using synthetic fallback.")
            X_raw, y, data_source = _generate_synthetic()
 
    print(f"Source: {data_source}")
    print(f"Shape:  {X_raw.shape}")
 
    X_winsor, caps = _preprocess(X_raw, winsor_pct=99.0)
    print(f"Winsorization caps (p99): R={caps['values'][0]}, F={caps['values'][1]}, M={caps['values'][2]}")
 
    x_path = _DATA_DIR / "rfm_X.npy"
    y_path = _DATA_DIR / "rfm_y.npy"
    np.save(x_path, X_winsor)
    np.save(y_path, y)
    print(f"Saved {x_path.name} and {y_path.name}")
 
    meta = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": data_source,
        "n_customers": int(X_winsor.shape[0]),
        "features": ["Recency_days", "Frequency_orders", "Monetary_GBP"],
        "random_state": _RANDOM_STATE,
        "winsorization": caps,
        "s3_bucket": _BUCKET,
        "s3_prefix": _S3_PREFIX,
        "files": {
            "rfm_X.npy": f"md5:{_md5(x_path)}",
            "rfm_y.npy": f"md5:{_md5(y_path)}",
        },
    }
    meta_path = _DATA_DIR / "rfm_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved {meta_path.name}")
 
    if upload:
        print("\nUploading to S3...")
        for local, suffix in [(x_path, "rfm_X.npy"), (y_path, "rfm_y.npy"), (meta_path, "rfm_meta.json")]:
            _upload(local, f"{_S3_PREFIX}/{suffix}")
 
    print("\nDone.")
 
 
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate RFM dataset for customer segmentation.")
    parser.add_argument("--source", default="auto", choices=["auto", "real", "synthetic"])
    parser.add_argument("--no-upload", dest="upload", action="store_false")
    return parser.parse_args()
 
if __name__ == "__main__":
    args = _parse_args()
    generate_rfm(source=args.source, upload=args.upload)