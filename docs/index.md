# Convex Clustering
 
[![CI](https://github.com/JuanNicolasMendozaRoncancio/Convex-Clustering/actions/workflows/ci.yml/badge.svg)](https://github.com/JuanNicolasMendozaRoncancio/Convex-Clustering/actions/workflows/ci.yml)
 
Convex clustering reformulates clustering as a convex optimization problem,
producing a **continuous regularization path** from each point being its own
cluster (γ = 0) to all points fused into a single cluster (γ → ∞). Unlike
k-means or DBSCAN, the number of clusters is not an input — it emerges from
the regularization strength γ.
 
This library implements **seven algorithms** behind a unified,
scikit-learn-compatible `ConvexClusterer` estimator.
 
---
 
## Algorithms
 
| Algorithm | Penalty | Notes |
|-----------|---------|-------|
| `ADMM` | L2 group | Robust default. Closed-form primal update. |
| `AMA` | L2 group | Cheaper per iteration than ADMM. |
| `DR` | L2 group | Douglas-Rachford. Fast on well-conditioned graphs. |
| `RFS_L2` | L2 group | Forward stagewise. Runs exactly `max_iter` steps. |
| `Fast_RFS_L2` | L2 group | Accelerated RFS_L2. Accepts a γ list for the path. |
| `RFS_L1` | L1 entry-wise | Coordinates fuse independently. Sparser fusions. |
| `Fast_RFS_L1` | L1 entry-wise | Accelerated RFS_L1 via reformulation of `RFS_L1`. |
 
---
 
## Quickstart
 
```python
import numpy as np
from convex_clustering import ConvexClusterer, knn_w
 
X = np.random.default_rng(0).normal(size=(50, 2))
W = knn_w(X, k=3, phi=0.5)
 
model = ConvexClusterer(algorithm="ADMM", gamma=10.0)
model.fit(X, W)
 
print(model.labels_)           # cluster label per point
print(model.cluster_centers_)  # final fused center per point
```
 
Scikit-learn Pipeline compatible:
 
```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
 
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clusterer", ConvexClusterer(algorithm="DR", gamma=5.0)),
])
pipe.fit(X, W)
```
 
---
 
## Installation
 
```bash
# Development (includes tests, linting, type checking)
pip install -e ".[dev]"
 
# API + Dashboard
pip install -e ".[api,dashboard]"
```
 
---
 
## Live Demo
 
An interactive dashboard is deployed on Streamlit Cloud where you can run
any algorithm and animate the center fusion trajectory in real time.
 
---
 
## Project Structure
 
```
src/convex_clustering/
    algorithms.py   ← ConvexClusterer + seven algorithm implementations
    regression.py   ← Boosting, rfs_sparse, fastrfs_sparse
    utils.py        ← knn_w, built_edges, compute_b_penal
    viz.py          ← animation_save, plot_graph_weights
app/
    main.py         ← FastAPI application
    routers/        ← /cluster, /algorithms, /compare
    schemas.py      ← Pydantic request/response models
```