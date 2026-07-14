# Algorithms
 
Convex clustering solves the following optimization problem:
 
$$
\min_{U} \frac{1}{2} \|U - X\|_F^2 + \gamma \sum_{(i,j) \in E} w_{ij} \|u_i - u_j\|
$$
 
where $X$ is the data matrix, $U$ contains the cluster centers (one per point),
$E$ is the edge set of the k-NN graph, and $w_{ij}$ are the graph weights.
As γ increases, pairs of centers are progressively fused until all points
share a single center.
 
All seven algorithms solve this problem but differ in how they decompose the
objective, what operations dominate each iteration, and which norm they use
for the penalty.
 
---
 
## ADMM
 
**Alternating Direction Method of Multipliers.** Splits the objective into a
quadratic primal subproblem (solved exactly via a linear system) and a
group-soft-threshold dual update. The linear system is assembled once and
reused across iterations, making ADMM efficient when the graph structure
is fixed.
 
**When to use:** Default choice. Robust across a wide range of γ values and
graph densities. Convergence is well-understood theoretically.
 
**Key parameters:** `step_size` maps to the dual step size `nu`. Values in
`[0.1, 2.0]` work well for most datasets.
 
---
 
## AMA
 
**Alternating Minimization Algorithm.** Replaces the linear system solve of
ADMM with a cheaper gradient step on the primal, followed by a projection
onto the dual feasible set. Each iteration is faster than ADMM but the
method typically requires more iterations to reach the same tolerance.
 
**When to use:** When n is large and the linear system solve in ADMM becomes
the bottleneck.
 
---
 
## DR (Douglas-Rachford)
 
**Douglas-Rachford splitting in primal form.** Finds a fixed point of the
DR operator by alternating between a proximal step for the data fidelity
term and a resolve via the graph Laplacian (factorized once with `scipy.sparse.linalg.factorized`).
 
**When to use:** Well-conditioned graphs (moderate density, balanced weights).
Typically converges in fewer iterations than AMA.
 
---
 
## RFS\_L2
 
**Regularized Forward Stagewise with L2 group penalty.** Adapts the
forward stagewise regression path (Efron et al., 2004) to the clustering
setting. At each step, the algorithm selects the edge-feature pair with
the largest absolute correlation and takes a small step of size `epsilon`
in that direction. Runs for exactly `max_iter` steps — no early stopping.
 
**When to use:** When you want to trace the full regularization path with
fine granularity. Set `step_size` (epsilon) small and `max_iter` large.
 
---
 
## Fast\_RFS\_L2
 
**Accelerated RFS\_L2.** Precomputes the correlation update in closed form,
reducing the per-iteration cost. Accepts `gamma` as a single float or a
list of floats to trace the regularization path across multiple γ values
in a single call.
 
```python
model = ConvexClusterer(
    algorithm="Fast_RFS_L2",
    gamma=[1.0, 10.0, 100.0],  # regularization path
)
model.fit(X, W)
```
 
!!! note
    Because Fast_RFS_L2 iterates over a gamma list rather than solver steps,
    `centers_hist_` contains one entry per gamma value (not per iteration).
    The dashboard guards against this automatically.
 
---
 
## RFS\_L1
 
**Regularized Forward Stagewise with L1 entry-wise penalty.** Uses an
absolute-value penalty on individual center coordinate differences instead
of the L2 group norm. Coordinates fuse independently rather than jointly,
producing sparser fusions where only some dimensions of a pair of centers
are pulled together.
 
**When to use:** When you expect the cluster structure to be axis-aligned
or when you want different fusion behavior per feature dimension.
 
---
 
## Fast\_RFS\_L1
 
**Accelerated RFS\_L1.** Uses precomputed Kronecker products of the
incidence matrix to reduce the per-iteration cost of RFS\_L1. Same
statistical model, lower computational cost.
 
---
 
## Choosing γ
 
γ controls the number of clusters:
 
- **γ too small:** every point is its own cluster (no fusion).
- **γ too large:** all points fuse into a single cluster.
- **Practical approach:** run a sweep over `[1, 10, 100, 1000]` and
  inspect the silhouette score or the convergence curve in the dashboard.
The `merge_tol` parameter controls how close two final centers need to be
to be assigned the same label. For well-converged runs, `merge_tol=0.5`
works well. Increase it if you see more clusters than expected.