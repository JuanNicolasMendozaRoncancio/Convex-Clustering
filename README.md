# Convex Clustering

[![CI](https://github.com/JuanNicolasMendozaRoncancio/Convex-Clustering/actions/workflows/ci.yml/badge.svg)](https://github.com/JuanNicolasMendozaRoncancio/Convex-Clustering/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://juannicolasmendozaroncancio.github.io/Convex-Clustering/)
[![Dashboard](https://img.shields.io/badge/dashboard-Streamlit-ff4b4b)](https://convex-clustering-jnmr.streamlit.app)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)]()
[![License: MIT](https://img.shields.io/badge/license-MIT-green)]()

Convex clustering reformulates clustering as a convex optimization problem,
producing a **continuous regularization path** from each point being its own
cluster (γ = 0) to all points fused into a cluster for each connected component (γ → ∞). Unlike
k-means or DBSCAN, the number of clusters is not an input, it emerges from
the regularization strength γ, and the full fusion trajectory is available for
inspection at every iteration.

This library implements **seven algorithms** behind a unified,
scikit-learn-compatible `ConvexClusterer` estimator, exposed through a REST
API, an interactive dashboard, and a suite of benchmarks against classical
baselines.

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
| `Fast_RFS_L1` | L1 entry-wise | Accelerated RFS_L1. |

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
print(model.centers_hist_)     # full fusion trajectory
```

scikit-learn Pipeline compatible:

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
# Core library
pip install -e "."

# Development (tests, linting, type checking, benchmarks)
pip install -e ".[dev]"

# API + Dashboard
pip install -e ".[api,dashboard]"
```

---

## Project Structure

```
src/convex_clustering/
    algorithms.py     ← ConvexClusterer and seven algorithm implementations
    regression.py     ← Boosting (RFS / FastRFS), rfs_sparse, fastrfs_sparse
    utils.py          ← knn_w, built_edges, construct_weighted_laplacian
    data.py           ← load_dataset (S3-backed, local cache)
    viz.py            ← animation_save, plot_graph_weights

app/
    main.py           ← FastAPI application entry point
    routers/          ← /cluster, /algorithms, /compare
    schemas.py        ← Pydantic request/response models
    dashboard.py      ← Streamlit dashboard

scripts/
    generate_datasets.py      ← Generate synthetic datasets → S3
    run_experiment_job.py     ← Cloud Run job: single experiment
    run_experiments.sh        ← Full experiment matrix (3×3×4 = 36 runs)
    benchmark.py              ← ConvexClusterer vs KMeans vs DBSCAN
    generate_rfm.py           ← RFM dataset from Online Retail or synthetic
    run_rfm_analysis.py       ← Customer segmentation
    generate_rfm_figures.py   ← Figures for documentation

docs/
    algorithms.md             ← Algorithm descriptions and parameter guide
    benchmark.md              ← Benchmark results vs baselines algorithms
    regression.md             ← Boosting / RF-S documentation
    applications/
        customer_segmentation.md  ← RFM segmentation case study

tests/
    test_algos.py      ← Convergence and center accuracy tests
    test_regression.py ← Boosting R² and coefficient equivalence tests
    test_api.py        ← FastAPI endpoint tests (TestClient)
```

---

## Live Artifacts

| Artifact | URL |
|----------|-----|
| Interactive dashboard | [convex-clustering.streamlit.app](https://convex-clustering-jnmr.streamlit.app) |
| Project documentation | [juannicolasmendozaroncancio.github.io/Convex-Clustering](https://juannicolasmendozaroncancio.github.io/Convex-Clustering/) |

The dashboard exposes three tabs: interactive clustering with animated fusion
trajectory, Lasso vs Boosting comparison, and weight graph visualization.

---

## REST API

```bash
# Start locally
uvicorn app.main:app --reload

# Or with Docker Compose (API + Dashboard together)
docker compose up --build
```

**Endpoints:**

- `GET /` — health check
- `GET /algorithms/` — list all seven algorithms with descriptions
- `POST /cluster/` — run ConvexClusterer, returns labels + convergence curve
- `POST /compare/` — run multiple algorithms side-by-side, returns silhouette scores

Example request:

```bash
curl -X POST http://localhost:8000/cluster/ \
  -H "Content-Type: application/json" \
  -d '{"X": [[0,0],[1,0],[5,5],[6,5]], "W": [[0,0.8,0,0],[0.8,0,0,0],[0,0,0,0.8],[0,0,0.8,0]], "algorithm": "ADMM", "gamma": 10.0}'
```

---

## Benchmark

Comparison of `ConvexClusterer` against KMeans and DBSCAN on three standard
synthetic datasets (200 points, `StandardScaler` applied to all models):

| Dataset | Model | Clusters found | Silhouette | Davies-Bouldin | Time (s) |
|---------|-------|:--------------:|:----------:|:--------------:|:--------:|
| blobs | ConvexClusterer | 3 | 0.878 | 0.173 | 17.8 |
| blobs | KMeans | 3  | 0.878 | 0.173 | 7.7 |
| blobs | DBSCAN | 3  | 0.878 | 0.173 | 0.01 |
| moons | ConvexClusterer | 2  | 0.379 | 1.028 | 0.04 |
| moons | KMeans | 2  | 0.489 | 0.813 | 0.08 |
| moons | DBSCAN | 14  | 0.620 | 0.431 | 0.01 |
| circles | ConvexClusterer | 1  | — | — | 0.04 |
| circles | KMeans | 2  | 0.352 | 1.170 | 0.09 |
| circles | DBSCAN | 7  | 0.781 | 0.266 | 0.01 |

**Key takeaway:** ConvexClusterer matches KMeans and DBSCAN on
well-separated Gaussian data without being told the number of clusters k.
On non-convex geometry (moons, circles), no Euclidean distance-based method
works without a feature transformation. The limitation belongs to the
Euclidean kNN weight function, not to convex clustering as a method.

The structural advantage of convex clustering that has no equivalent in
KMeans or DBSCAN is the **regularization path**. γ controls a continuous
spectrum from n clusters to #number of connected componets clusters, and `centers_hist_` exposes the full fusion
trajectory at every iteration.

Full analysis in [`docs/benchmark.md`](docs/benchmark.md).

---

## Application: Customer Segmentation

Convex clustering applied to RFM (Recency, Frequency, Monetary) customer
segmentation on the [Online Retail Dataset (UCI)](https://archive.ics.uci.edu/dataset/352/online+retail).
Four versions were implemented comparing different dimensionality reduction
strategies:

| Version | Space | Silhouette | Notes |
|---------|:-----:|:----------:|-------|
| V1 — 3D RFM | 3D scaled | 0.491 | Centers interpretable in business units |
| V2 — PCA 2D | 2D linear | 0.540 | Fusion trajectory visible |
| V3 — 3D → UMAP viz | 3D (viz in 2D) | 0.491 | Statistically correct + visually clear |
| V4 — UMAP 2D | 2D non-linear | 0.524 | Cleaner boundaries than PCA |

Full case study in [`docs/applications/customer_segmentation.md`](docs/applications/customer_segmentation.md).

---

## Experiment Tracking

All experiments are logged to MLflow with SQLite backend:

```bash
# Run the full experiment matrix (3 datasets × 3 algorithms × 4 γ values)
bash scripts/run_experiments.sh

# Run the benchmark
python scripts/benchmark.py

# Inspect results
mlflow ui --backend-store-uri sqlite:///mlflow.db
# → http://localhost:5000
```

Results are stored in S3 under `results/{exp_id}/` with standardized structure:
`config.json`, `metrics.csv`, `convergence.csv`, `paths.npy`, `labels.npy`.

---

## Running on Google Cloud

Build and push the image:

```bash
docker build -t gcr.io/{PROJECT_ID}/convex-clustering-job .
docker push gcr.io/{PROJECT_ID}/convex-clustering-job
```

Create the job:

```bash
gcloud run jobs create convex-clustering-job \
  --image gcr.io/{PROJECT_ID}/convex-clustering-job \
  --region europe-north1 \
  --set-env-vars AWS_REGION=eu-north-1 \
  --set-secrets AWS_ACCESS_KEY_ID=aws-key-id:latest \
  --set-secrets AWS_SECRET_ACCESS_KEY=aws-secret-key:latest
```

Execute a single experiment:

```bash
gcloud run jobs execute convex-clustering-job \
  --args="--dataset,blobs,--algorithm,ADMM,--gamma,1.0"
```

---

## CI / CD

Three parallel jobs on every push to `main`:

| Job | Tool | What it checks |
|-----|------|----------------|
| `pytest` | pytest and TestClient | Convergence, center accuracy, API endpoints |
| `ruff` | ruff 0.4 | Style, imports, naming |
| `mypy` | mypy strict | Full type correctness across `src/`, `scripts/`, `app/` |

Documentation deploys automatically to GitHub Pages via `.github/workflows/docs.yml`.

---

## Tech Stack

| Layer | Technologies |
|-------|-------------|
| Core library | Python 3.10–3.12, NumPy, SciPy, scikit-learn |
| Packaging | pyproject.toml, hatchling, optional dependency groups |
| Type checking | mypy strict, numpy.typing |
| Testing | pytest, pytest-cov, FastAPI TestClient |
| Linting | ruff (pycodestyle + pyflakes + isort + pyupgrade + bugbear) |
| CI/CD | GitHub Actions (3 parallel jobs), GitHub Pages |
| Experiment tracking | MLflow 3.x, SQLite backend, S3 artifacts |
| Cloud storage | AWS S3 (boto3), datasets versioned via manifest.json + MD5 |
| Cloud compute | Google Cloud Run Jobs, Docker |
| API | FastAPI, Pydantic v2, Uvicorn, httpx2 |
| Dashboard | Streamlit, Plotly, networkx, PyVis |
| Documentation | MkDocs Material, mkdocstrings, NumPy docstrings |

---

## Author

**Juan Nicolás Mendoza Roncancio**
Mathematics — Universidad Nacional de Colombia - Bogotá, Colombia

M.Sc. in Applied Mathematics, AI & Engineering (Diplôme d'Ingénieur) — Mines Paris PSL - Paris, France

M.Sc. in Mathematics – Statistics, Machine Learning and Algorithms - Sorbonne Université - Paris, France

[GitHub](https://github.com/JuanNicolasMendozaRoncancio) · [Documentation](https://juannicolasmendozaroncancio.github.io/Convex-Clustering/) · [Dashboard](https://convex-clustering-jnmr.streamlit.app)