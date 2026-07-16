# Benchmark

Comparison of `ConvexClusterer` against KMeans and DBSCAN on three standard
synthetic datasets. The goal is to honestly document when it wins, when it loses, and why.

---

## Experimental Setup

**Datasets**, three scikit-learn synthetic datasets with 200 points each:

- `blobs` — three well-separated Gaussian clusters (cluster_std=0.8)
- `moons` — two interleaved half-moons (noise=0.08)
- `circles` — two concentric circles (noise=0.08, factor=0.5)

**Preprocessing** — `StandardScaler` applied to all three models before any
computation. KMeans and DBSCAN are sensitive to scale by construction and ConvexClusterer also uses distances through the
weight matrix W, so standardizing all three produces the fairest comparison.

**Hyperparameters** — each model receives its best known parameters for each
dataset, like this the benchmark compares models at their best.

**Metrics**:

- **Silhouette** ∈ [-1, 1] — measures intra-cluster cohesion vs inter-cluster
  separation, the ideal value is 1. Undefined (reported as —) when only one cluster
  is found.
- **Davies-Bouldin** ∈ [0, ∞) — measures the average similarity between each
  cluster and its most similar neighbor, the ideal value is 0.
- **Fit time** — wall-clock seconds measured with `tracemalloc`.
- **Noise points** — only relevant for DBSCAN (label -1).

---

## Results

### blobs — well-separated Gaussian clusters

| Model             | Clusters found | Silhouette | Davies-Bouldin | Time (s) | Memory (MB) |
|-------------------|:--------------:|:----------:|:--------------:|:--------:|:-----------:|
| ConvexClusterer   | **3**          | **0.878**  | **0.173**      | 17.823   | 2.22        |
| KMeans            | **3**          | **0.878**  | **0.173**      | 7.749    | 0.24        |
| DBSCAN            | **3**          | **0.878**  | **0.173**      | 0.013    | 0.07        |

All three models produce identical quality results. Well-separated Gaussian
blobs is exactly the case KMeans was designed for, and all three converge to
the same optimal solution.

The difference is purely computational: DBSCAN at 0.013s, KMeans at 7.7s,
ConvexClusterer at 17.8s. On simple, well-conditioned structure, convex
clustering offers no quality advantage over the baselines.

**Convex clustering advantage here:** none in quality. The only structural
difference is that ConvexClusterer discovered 3 clusters without being told —
KMeans received `n_clusters=3` as input, an artificial advantage that would
not exist in a real use case where k is unknown.

---

### moons — non-convex geometry

| Model             | Clusters found | Noise | Silhouette | Davies-Bouldin | Time (s) |
|-------------------|:--------------:|:-----:|:----------:|:--------------:|:--------:|
| ConvexClusterer   | **2**          | 0     | 0.379      | 1.028          | 0.036    |
| KMeans            | **2**          | 0     | 0.489      | 0.813          | 0.082    |
| DBSCAN            | 14             | 99    | 0.620      | 0.431          | 0.008    |

ConvexClusterer correctly finds 2 clusters but with low silhouette (0.379).
The reason is geometric: the kNN graph with k=6 connects nearby points in
Euclidean space, but the two moons overlap locally — points on one moon have
Euclidean neighbors on the other. The fusion pressure from the regularizer
acts on those cross-moon edges, pulling centers toward each other and
producing final centers that do not reflect the true moon structure.

KMeans achieves higher silhouette (0.489) with 2 correct clusters, but for
the wrong reason: it divides the space with a vertical hyperplane, producing
two left/right halves that have good internal cohesion by construction but do
not correspond to the actual moons.

DBSCAN fragments into 14 clusters with 99 noise points — `eps=0.15` is too
strict for the density of this standardized dataset. Its silhouette of 0.620
is high but misleading as it measures the quality of 14 micro-clusters, not of
the 2 structural clusters.

**Conclusion for moons:** no model solves the problem cleanly with default
parameters. Convex clustering has the right property (discovering k without
input) but the Euclidean kNN weight function does not capture the moon
structure. With a weight function that reflects local connectivity (e.g.,
geodesic distance), convex clustering would improve substantially.

---

### circles — nested geometry

| Model             | Clusters found | Noise | Silhouette | Davies-Bouldin | Time (s) |
|-------------------|:--------------:|:-----:|:----------:|:--------------:|:--------:|
| ConvexClusterer   | 1              | 0     | —          | —              | 0.036    |
| KMeans            | **2**          | 0     | 0.352      | 1.170          | 0.095    |
| DBSCAN            | 7              | 153   | 0.781      | 0.266          | 0.011    |

The most demanding case. ConvexClusterer collapses all points into a single
cluster as the knn graph is not a good way to diferenciate both circles, the nearest Euclidean neighbors of a
point on the outer ring include both other outer-ring points and inner-ring
points that are close in Euclidean distance. Fusion pressure distributes
uniformly in all directions and the centers collapse to the global centroid.

KMeans also fails to recover the true structure: it divides the circles into
left/right hemispheres with low silhouette (0.352). DBSCAN with `eps=0.15`
fragments into 7 micro-clusters with 153 noise points (76% of the dataset
unassigned).

**Conclusion for circles:** no Euclidean distance-based method with reasonable
parameters solves this problem correctly. Circles is a pathological case for
Euclidean clustering in general — it requires kernel methods or feature space
transformations regardless of the clustering algorithm.

---

## Summary

| Dataset   | Quality winner       | Speed winner | ConvexClusterer discovers k |
|-----------|:--------------------:|:------------:|:---------------------------:|
| `blobs`   | Tie (all equal)      | DBSCAN       | ✓ Yes                       |
| `moons`   | None clearly         | DBSCAN       | ✓ Yes                       |
| `circles` | None clearly         | DBSCAN       | No (total collapse)         |

**When to use convex clustering:**

Convex clustering is the right choice when the number of clusters is unknown
and an interpretable solution with convergence guarantees is required. The
convex nature of the problem guarantees that the algorithm finds the global
minimum. The cost is
compute time: between 2× and 1000× slower than the baselines on these
200-point datasets.

**When not to use convex clustering:**

When cluster geometry is non-convex and the Euclidean kNN graph does not
capture the structural connectivity (the `circles` cases). In
these cases, no Euclidean distance-based method works well without a feature
space transformation. The limitation belongs to the Euclidean kNN weight
function, not to convex clustering as a method.

**The structural difference from the baselines:**

KMeans requires `k` as input. DBSCAN requires `eps` and `min_samples`.
ConvexClusterer requires `gamma`. All three require tuning — the difference
is semantic: `gamma` controls regularization strength along a continuous
solution path (from n clusters down to 1), while KMeans's `k` is a discrete
decision with no path structure. This regularization path property is unique
to convex clustering and enables systematic exploration of the solution space.

---

## Reproducibility

```bash
# Generate results and log to MLflow
python scripts/benchmark.py

# Inspect runs in the MLflow UI
mlflow ui --backend-store-uri sqlite:///mlflow.db
# Navigate to http://localhost:5000 → experiment "benchmark"
```

Results are saved to `results/benchmark/benchmark_results.csv`.
MLflow runs are logged to the `benchmark` experiment inside `mlflow.db`.