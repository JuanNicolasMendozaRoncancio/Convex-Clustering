"""
Pydantic schemas — the API contract for all endpoints.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Shared input types
# ---------------------------------------------------------------------------

Matrix = list[list[float]]


class ClusterRequest(BaseModel):
    """
    Input for POST /cluster.

    X : (n_samples, n_features) data matrix.
    W : (n_samples, n_samples) symmetric weight matrix.
        W[i][j] > 0 means there is an edge between points i and j.
    algorithm : one of the seven supported algorithms.
    gamma : regularization strength. Higher → fewer, larger clusters.
    step_size : maps to nu (ADMM/AMA), rho (DR), epsilon (RFS variants).
    max_iter : maximum number of iterations.
    tol : convergence tolerance (not used by RFS_L2 / Fast_RFS_L2).
    merge_tol : distance threshold below which two centers are fused into
                the same cluster label.
    """

    X: Matrix = Field(
        description="Data matrix: list of n_samples rows, each with n_features floats.",
        examples=[[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [5.0, 5.0], [6.0, 5.0], [5.0, 6.0]]],
    )
    W: Matrix = Field(
        description="Weight matrix: square symmetric matrix of shape (n_samples, n_samples).",
    )
    algorithm: str = Field(
        default="ADMM",
        description="Algorithm to use for clustering.",
    )
    gamma: float = Field(
        default=10.0,
        gt=0,
        description="Regularization parameter. Higher values produce fewer clusters.",
    )
    step_size: float = Field(
        default=0.5,
        gt=0,
        description="Step size for the iterative update (nu / rho / epsilon depending on algorithm).",
    )
    max_iter: int = Field(
        default=1000,
        gt=0,
        description="Maximum number of iterations.",
    )
    tol: float = Field(
        default=1e-4,
        gt=0,
        description="Convergence tolerance.",
    )
    merge_tol: float = Field(
        default=0.5,
        gt=0,
        description="Distance threshold below which two final centers are merged into the same cluster.",
    )

    @field_validator("algorithm")
    @classmethod
    def algorithm_must_be_valid(cls, v: str) -> str:
        valid = {
            "ADMM", "AMA", "DR",
            "RFS_L2", "Fast_RFS_L2",
            "RFS_L1", "Fast_RFS_L1",
        }
        if v not in valid:
            raise ValueError(f"algorithm must be one of {sorted(valid)}, got '{v}'")
        return v


class CompareRequest(BaseModel):
    """
    Input for POST /compare.

    Same X and W as ClusterRequest, plus a list of algorithms to compare.
    Each algorithm runs with the same shared hyperparameters.
    """

    X: Matrix = Field(
        description="Data matrix: list of n_samples rows, each with n_features floats.",
        examples=[[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [5.0, 5.0], [6.0, 5.0], [5.0, 6.0]]],
    )
    W: Matrix = Field(
        description="Weight matrix: square matrix of shape (n_samples, n_samples).",
    )
    algorithms: list[str] = Field(
        default=["ADMM", "AMA", "DR"],
        description="List of algorithms to compare.",
    )
    gamma: float = Field(default=10.0, gt=0)
    step_size: float = Field(default=0.5, gt=0)
    max_iter: int = Field(default=1000, gt=0)
    tol: float = Field(default=1e-4, gt=0)
    merge_tol: float = Field(default=0.5, gt=0)

    @field_validator("algorithms")
    @classmethod
    def algorithms_must_be_valid(cls, v: list[str]) -> list[str]:
        valid = {
            "ADMM", "AMA", "DR",
            "RFS_L2", "Fast_RFS_L2",
            "RFS_L1", "Fast_RFS_L1",
        }
        invalid = [a for a in v if a not in valid]
        if invalid:
            raise ValueError(
                f"Unknown algorithms: {invalid}. Valid choices: {sorted(valid)}"
            )
        if len(v) == 0:
            raise ValueError("algorithms list must not be empty.")
        return v


# ---------------------------------------------------------------------------
# Response types
# ---------------------------------------------------------------------------

class ClusterResponse(BaseModel):
    """
    Output of POST /cluster.

    labels : cluster label for each sample (0-indexed).
    cluster_centers : final center coordinates for each sample, shape (n_samples, n_features).
    n_clusters : number of distinct clusters found.
    n_iter : number of iterations actually run.
    convergence : list of {iteration, center_diff} dicts — the convergence curve.
                  Sparse: only iterations stored in history_ are included.
    """

    labels: list[int]
    cluster_centers: list[list[float]]
    n_clusters: int
    n_iter: int
    convergence: list[dict[str, Any]]


class AlgorithmInfo(BaseModel):
    """Description of a single algorithm."""

    name: str
    description: str
    parameters: dict[str, str]


class AlgorithmsResponse(BaseModel):
    """Output of GET /algorithms."""

    algorithms: list[AlgorithmInfo]


class AlgorithmMetrics(BaseModel):
    """
    Metrics for one algorithm in POST /compare.

    silhouette_score : between -1 and 1. Higher is better.
                       -1.0 is returned when n_clusters == 1 (undefined).
    n_clusters : number of clusters found.
    n_iter : iterations run.
    """

    algorithm: str
    labels: list[int]
    n_clusters: int
    n_iter: int
    silhouette_score: float


class CompareResponse(BaseModel):
    """Output of POST /compare."""

    results: list[AlgorithmMetrics]