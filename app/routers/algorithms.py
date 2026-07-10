"""
Router: GET /algorithms
 
Returns the list of available algorithms with their descriptions and
configurable parameters.
"""
from __future__ import annotations

from fastapi import APIRouter

from app.schemas import AlgorithmInfo, AlgorithmsResponse

router = APIRouter()

_ALGORITHM_CATALOG: list[AlgorithmInfo] = [
    AlgorithmInfo(
        name="ADMM",
        description=(
            "Alternating Direction Method of Multipliers. "
            "Splits the convex clustering objective into a quadratic primal "
            "update (closed-form linear system solve) and a group-soft-threshold "
            "dual update. Robust convergence across a wide range of gamma values. "
            "Good default choice."
        ),
        parameters={
            "gamma": "Regularization strength (float, > 0). Higher → fewer clusters.",
            "step_size": "Dual step size nu (float, > 0). Values in [0.1, 2.0] work well.",
            "max_iter": "Maximum iterations (int). 1000 is sufficient for most datasets.",
            "tol": "Convergence tolerance on center difference (float). Default 1e-4.",
            "merge_tol": "Distance below which two centers are merged (float). Default 0.5.",
        },
    ),
    AlgorithmInfo(
        name="AMA",
        description=(
            "Alternating Minimization Algorithm. "
            "A splitting method that alternates between an unconstrained "
            "gradient step on the primal and a projection onto the dual "
            "feasible set. Cheaper per iteration than ADMM (no linear "
            "system solve) but may require more iterations to converge."
        ),
        parameters={
            "gamma": "Regularization strength (float, > 0).",
            "step_size": "Primal step size nu (float, > 0). Smaller values → more stable.",
            "max_iter": "Maximum iterations (int).",
            "tol": "Convergence tolerance on center difference (float).",
            "merge_tol": "Distance below which two centers are merged (float).",
        },
    ),
    AlgorithmInfo(
        name="DR",
        description=(
            "Douglas-Rachford splitting (primal form). "
            "Reformulates clustering as finding a fixed point of the "
            "DR operator, which alternates between a proximal step for "
            "the data fidelity term and a resolve via the graph Laplacian. "
            "Typically converges in fewer iterations than AMA on "
            "well-conditioned graphs."
        ),
        parameters={
            "gamma": "Regularization strength (float, > 0).",
            "step_size": "Splitting parameter rho (float, > 0).",
            "max_iter": "Maximum iterations (int).",
            "tol": "Convergence tolerance (float).",
            "merge_tol": "Distance below which two centers are merged (float).",
        },
    ),
    AlgorithmInfo(
        name="RFS_L2",
        description=(
            "Regularized Forward Stagewise with L2 penalty. "
            "Adapts the RF-S regression algorithm (a forward stagewise "
            "path to the Lasso) to the clustering setting via an L2 "
            "group penalty on center differences. Runs for exactly "
            "max_iter steps (no early stopping)."
        ),
        parameters={
            "gamma": "Regularization strength (float, > 0).",
            "step_size": "Stagewise step size epsilon (float, > 0). Smaller → finer path.",
            "max_iter": "Number of stagewise steps (int). No early stopping.",
            "merge_tol": "Distance below which two centers are merged (float).",
        },
    ),
    AlgorithmInfo(
        name="Fast_RFS_L2",
        description=(
            "Fast Regularized Forward Stagewise with L2 penalty. "
            "Accelerated version of RFS_L2 that precomputes correlation "
            "updates in closed form, reducing per-iteration cost. "
            "gamma can be a single float or a list for a regularization path."
        ),
        parameters={
            "gamma": (
                "Regularization strength (float or list[float], > 0). "
                "Pass a list to trace the full regularization path."
            ),
            "step_size": "Stagewise step size epsilon (float, > 0).",
            "max_iter": "Number of stagewise steps (int).",
            "merge_tol": "Distance below which two centers are merged (float).",
        },
    ),
    AlgorithmInfo(
        name="RFS_L1",
        description=(
            "Regularized Forward Stagewise with L1 penalty. "
            "Uses an L1 (entry-wise absolute value) penalty on center "
            "differences instead of L2. Produces sparser fusions — "
            "coordinates fuse independently rather than jointly."
        ),
        parameters={
            "gamma": "Regularization strength (float, > 0).",
            "step_size": "Stagewise step size epsilon (float, > 0).",
            "max_iter": "Maximum iterations M (int).",
            "tol": "Convergence threshold cauchy (float).",
            "merge_tol": "Distance below which two centers are merged (float).",
        },
    ),
    AlgorithmInfo(
        name="Fast_RFS_L1",
        description=(
            "Fast Regularized Forward Stagewise with L1 penalty. "
            "Accelerated version of RFS_L1 using precomputed Kronecker "
            "products of the incidence matrix. Same statistical model "
            "as RFS_L1, lower computational cost per iteration."
        ),
        parameters={
            "gamma": "Regularization strength (float, > 0).",
            "step_size": "Stagewise step size epsilon (float, > 0).",
            "max_iter": "Maximum iterations M (int).",
            "tol": "Convergence threshold cauchy (float).",
            "merge_tol": "Distance below which two centers are merged (float).",
        },
    ),
]


@router.get(
    "/",
    response_model=AlgorithmsResponse,
    summary="List available convex clustering algorithms.",
    description=(
        "Returns a list of supported convex clustering algorithms, "
        "each with a description and the configurable parameters."
    ),
)
def list_algorithms() -> AlgorithmsResponse:
    return AlgorithmsResponse(algorithms=_ALGORITHM_CATALOG)