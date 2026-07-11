"""
Router: POST /cluster

Runs ConvexClusterer on the provided data and returns labels,
cluster centers, and the convergence curve.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from fastapi import APIRouter, HTTPException

from app.schemas import ClusterRequest, ClusterResponse
from convex_clustering import ConvexClusterer

router = APIRouter()


def _to_convergence_list(history: dict[int, float]) -> list[dict[str, Any]]:
    """Convert the model's history_ dict to a JSON-serializable list."""
    return [{"iteration": it, "center_diff": val} for it, val in history.items()]


@router.post(
    "/",
    response_model=ClusterResponse,
    summary="Run convex clustering",
    description=(
        "Fits ConvexClusterer on the provided data matrix X with weight "
        "matrix W. Returns cluster labels, final cluster centers, and the "
        "convergence curve (iteration vs. center difference)."
    ),
)
def run_cluster(request: ClusterRequest) -> ClusterResponse:
    """
    POST /cluster

    Converts the validated Pydantic request to numpy arrays, runs the
    selected algorithm, and serializes the result back to JSON-compatible
    Python types.
    """
    X = np.asarray(request.X, dtype=np.float64)
    W = np.asarray(request.W, dtype=np.float64)

    if len(request.X) == 0:
        raise HTTPException(status_code=422, detail="X must not be empty.")
    n = len(request.X)
    n_cols = len(request.X[0]) if request.X else 0
    if n_cols == 0:
        raise HTTPException(status_code=422, detail="X rows must not be empty.")
    if any(len(row) != n_cols for row in request.X):
        raise HTTPException(status_code=422, detail="All rows of X must have the same length.")
    if len(request.W) != n or any(len(row) != n for row in request.W):
        raise HTTPException(
            status_code=422,
            detail=f"W must be square ({n}x{n}) and match X's n_samples.",
        )

    try:
        model = ConvexClusterer(
            algorithm=request.algorithm,
            gamma=request.gamma,
            step_size=request.step_size,
            max_iter=request.max_iter,
            tol=request.tol,
            merge_tol=request.merge_tol,
        )
        model.fit(X, W)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Algorithm failed with an internal error: {exc}",
        ) from exc

    return ClusterResponse(
        labels=model.labels_.tolist(),
        cluster_centers=model.cluster_centers_.tolist(),
        n_clusters=int(len(set(model.labels_.tolist()))),
        n_iter=model.n_iter_,
        convergence=_to_convergence_list(model.history_),
    )