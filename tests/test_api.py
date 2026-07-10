"""
Tests for the FastAPI endpoints: /cluster, /algorithms, /compare.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sklearn.datasets import make_blobs

from app.main import app
from convex_clustering.utils import knn_w

client = TestClient(app)

@pytest.fixture(scope = "module")
def blob_payload() -> dict:
    X, _ = make_blobs(n_samples=20, centers=2, cluster_std=0.5, random_state=7)
    W = knn_w(X, k=3, phi=0.5)
    return {
        "X": X.tolist(),
        "W": W.tolist(),
        "algorithm": "ADMM",
        "gamma": 100.0,
        "step_size": 0.01,
        "max_iter": 1000,
        "tol": 1e-4,
        "merge_tol": 0.5,
    }

# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

def test_root_health_check() -> None:
    """GET / returns 200 and status: ok."""
    response = client.get("/")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "docs" in body

# ---------------------------------------------------------------------------
# GET /algorithms/
# ---------------------------------------------------------------------------
 
def test_algorithms_returns_all_seven() -> None:
    """GET /algorithms/ lists all seven implemented algorithms."""
    response = client.get("/algorithms/")
    assert response.status_code == 200
    body = response.json()
    names = {a["name"] for a in body["algorithms"]}
    expected = {"ADMM", "AMA", "DR", "RFS_L2", "Fast_RFS_L2", "RFS_L1", "Fast_RFS_L1"}
    assert names == expected
 
 
def test_algorithms_each_has_description_and_parameters() -> None:
    """Every algorithm entry has a non-empty description and parameters dict."""
    response = client.get("/algorithms/")
    assert response.status_code == 200
    for algo in response.json()["algorithms"]:
        assert algo["description"], f"{algo['name']} has empty description"
        assert algo["parameters"], f"{algo['name']} has empty parameters"

# ---------------------------------------------------------------------------
# POST /cluster/  — happy paths
# ---------------------------------------------------------------------------
 
def test_cluster_admm_returns_valid_response(blob_payload: dict) -> None:
    """POST /cluster/ with ADMM returns labels, centers, n_clusters, convergence."""
    response = client.post("/cluster/", json=blob_payload)
    assert response.status_code == 200, response.text
    body = response.json()
 
    n_samples = len(blob_payload["X"])
    assert len(body["labels"]) == n_samples
    assert len(body["cluster_centers"]) == n_samples
    assert body["n_clusters"] >= 1
    assert body["n_iter"] >= 1
    assert isinstance(body["convergence"], list)
 
 
def test_cluster_labels_are_integers(blob_payload: dict) -> None:
    """All labels in the response are Python ints (not floats)."""
    response = client.post("/cluster/", json=blob_payload)
    assert response.status_code == 200
    for label in response.json()["labels"]:
        assert isinstance(label, int), f"Expected int label, got {type(label)}"
 
 
@pytest.mark.parametrize("algorithm", ["ADMM", "AMA", "DR"])
def test_cluster_splitting_algorithms(blob_payload: dict, algorithm: str) -> None:
    """ADMM, AMA, and DR all return a valid response on the same payload."""
    payload = {**blob_payload, "algorithm": algorithm}
    response = client.post("/cluster/", json=payload)
    assert response.status_code == 200, f"{algorithm} failed: {response.text}"
    body = response.json()
    assert body["n_clusters"] >= 1
    assert len(body["labels"]) == len(blob_payload["X"])
 
 
def test_cluster_convergence_curve_is_non_empty(blob_payload: dict) -> None:
    """Convergence list has at least one entry and each entry has iteration and center_diff."""
    response = client.post("/cluster/", json=blob_payload)
    assert response.status_code == 200
    convergence = response.json()["convergence"]
    assert len(convergence) > 0
    for entry in convergence:
        assert "iteration" in entry
        assert "center_diff" in entry
        assert isinstance(entry["center_diff"], float)

# ---------------------------------------------------------------------------
# POST /cluster/  — error paths
# ---------------------------------------------------------------------------
 
def test_cluster_invalid_algorithm_returns_422(blob_payload: dict) -> None:
    """Requesting an unknown algorithm returns 422 Unprocessable Entity."""
    payload = {**blob_payload, "algorithm": "KMeans"}
    response = client.post("/cluster/", json=payload)
    assert response.status_code == 422
 
 
 
def test_cluster_mismatched_W_shape_returns_422(blob_payload: dict) -> None:
    """W with wrong number of rows returns 422."""
    n = len(blob_payload["X"])
    # W has n-1 rows instead of n
    W_bad = [[0.0] * n for _ in range(n - 1)]
    payload = {**blob_payload, "W": W_bad}
    response = client.post("/cluster/", json=payload)
    assert response.status_code == 422
 
 
def test_cluster_empty_X_returns_422(blob_payload: dict) -> None:
    """An empty X matrix returns 422."""
    payload = {**blob_payload, "X": [], "W": []}
    response = client.post("/cluster/", json=payload)
    assert response.status_code == 422
 
 
def test_cluster_gamma_zero_returns_422(blob_payload: dict) -> None:
    """gamma=0 violates gt=0 constraint and returns 422."""
    payload = {**blob_payload, "gamma": 0.0}
    response = client.post("/cluster/", json=payload)
    assert response.status_code == 422

# ---------------------------------------------------------------------------
# POST /compare/  — happy paths
# ---------------------------------------------------------------------------
 
def test_compare_returns_results_for_each_algorithm(blob_payload: dict) -> None:
    """POST /compare/ with three algorithms returns three result entries."""
    payload = {
        "X": blob_payload["X"],
        "W": blob_payload["W"],
        "algorithms": ["ADMM", "AMA", "DR"],
        "gamma": blob_payload["gamma"],
        "step_size": blob_payload["step_size"],
        "max_iter": blob_payload["max_iter"],
        "tol": blob_payload["tol"],
        "merge_tol": blob_payload["merge_tol"],
    }
    response = client.post("/compare/", json=payload)
    assert response.status_code == 200, response.text
    results = response.json()["results"]
    assert len(results) == 3
    returned_names = {r["algorithm"] for r in results}
    assert returned_names == {"ADMM", "AMA", "DR"}
 
 
def test_compare_silhouette_in_valid_range(blob_payload: dict) -> None:
    """Silhouette scores are in [-1.0, 1.0]."""
    payload = {
        "X": blob_payload["X"],
        "W": blob_payload["W"],
        "algorithms": ["ADMM", "DR"],
        "gamma": blob_payload["gamma"],
        "step_size": blob_payload["step_size"],
        "max_iter": blob_payload["max_iter"],
        "tol": blob_payload["tol"],
        "merge_tol": blob_payload["merge_tol"],
    }
    response = client.post("/compare/", json=payload)
    assert response.status_code == 200
    for result in response.json()["results"]:
        sil = result["silhouette_score"]
        assert -1.0 <= sil <= 1.0, f"Silhouette out of range: {sil}"

# ---------------------------------------------------------------------------
# POST /compare/  — error paths
# ---------------------------------------------------------------------------
 
def test_compare_invalid_algorithm_returns_422(blob_payload: dict) -> None:
    """An unknown algorithm in the list returns 422."""
    payload = {
        "X": blob_payload["X"],
        "W": blob_payload["W"],
        "algorithms": ["ADMM", "NotAnAlgorithm"],
        "gamma": 10.0,
        "step_size": 0.5,
        "max_iter": 100,
        "tol": 1e-4,
        "merge_tol": 0.5,
    }
    response = client.post("/compare/", json=payload)
    assert response.status_code == 422
 
 
def test_compare_empty_algorithms_list_returns_422(blob_payload: dict) -> None:
    """An empty algorithms list returns 422."""
    payload = {
        "X": blob_payload["X"],
        "W": blob_payload["W"],
        "algorithms": [],
        "gamma": 10.0,
        "step_size": 0.5,
        "max_iter": 100,
        "tol": 1e-4,
        "merge_tol": 0.5,
    }
    response = client.post("/compare/", json=payload)
    assert response.status_code == 422
 