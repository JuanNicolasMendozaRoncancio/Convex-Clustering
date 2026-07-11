"""
Router: POST /compare

Runs multiple algorithms on the same dataset and returns a side-by-side
comparison of clustering metrics.

"""
from __future__ import annotations

import numpy as np
from fastapi import APIRouter, HTTPException
from sklearn.metrics import silhouette_score

from app.schemas import AlgorithmMetrics, CompareRequest, CompareResponse
from convex_clustering import ConvexClusterer

router = APIRouter()


@router.post(
    "/",
    response_model=CompareResponse,
    summary="Compare multiple algorithms",
    description=(
        "Runs each requested algorithm on the same (X, W) input and returns "
        "a side-by-side comparison of n_clusters, n_iter, and silhouette score. "
        "All algorithms share the same gamma, step_size, max_iter, tol, and "
        "merge_tol hyperparameters."
    ),
)
def compare_algorithms(request: CompareRequest) -> CompareResponse:
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


    results: list[AlgorithmMetrics] = []

    for algo_name in request.algorithms:
        try:
            model = ConvexClusterer(
                algorithm=algo_name,
                gamma=request.gamma,
                step_size=request.step_size,
                max_iter=request.max_iter,
                tol=request.tol,
                merge_tol=request.merge_tol,
            )
            model.fit(X, W)
        except Exception as exc:
            results.append(
                AlgorithmMetrics(
                    algorithm=algo_name,
                    labels=[],
                    n_clusters=0,
                    n_iter=0,
                    silhouette_score=-1.0,
                )
            )
            print(f"[compare] {algo_name} failed: {exc}")
            continue

        labels = model.labels_.tolist()
        n_clusters = len(set(labels))

        if n_clusters > 1:
            sil = float(silhouette_score(X, model.labels_))
        else:
            sil = -1.0

        results.append(
            AlgorithmMetrics(
                algorithm=algo_name,
                labels=labels,
                n_clusters=n_clusters,
                n_iter=model.n_iter_,
                silhouette_score=sil,
            )
        )

    return CompareResponse(results=results)