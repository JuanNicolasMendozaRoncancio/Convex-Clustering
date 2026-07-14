# REST API
 
The library is also exposed as a REST API built with FastAPI. The API is
documented interactively at `/docs` (Swagger UI) when the server is running.
 
---
 
## Running locally
 
```bash
# With Docker Compose (API + Dashboard together)
docker compose up --build
 
# API only
uvicorn app.main:app --reload
```
 
The API runs on `http://localhost:8000`. The interactive docs are at
`http://localhost:8000/docs`.
 
---
 
## Running on Cloud Run
 
Build and push the image:
 
```bash
docker build -t gcr.io/{PROJECT_ID}/convex-clustering-api .
docker push gcr.io/{PROJECT_ID}/convex-clustering-api
```
 
Deploy:
 
```bash
gcloud run deploy convex-clustering-api \
  --image gcr.io/{PROJECT_ID}/convex-clustering-api \
  --region europe-north1 \
  --allow-unauthenticated
```
 
---
 
## Endpoints
 
### `GET /`
 
Health check. Returns `{"status": "ok"}`.
 
---
 
### `GET /algorithms`
 
Returns the list of available algorithms with descriptions and configurable
parameters.
 
**Response:**
 
```json
{
  "algorithms": [
    {
      "name": "ADMM",
      "description": "...",
      "parameters": { "gamma": "...", "step_size": "..." }
    }
  ]
}
```
 
---
 
### `POST /cluster`
 
Runs `ConvexClusterer` on the provided data and returns labels, centers,
and the convergence curve.
 
**Request body:**
 
```json
{
  "X": [[0.0, 0.0], [1.0, 0.0], [5.0, 5.0]],
  "W": [[0, 0.6, 0], [0.6, 0, 0], [0, 0, 0]],
  "algorithm": "ADMM",
  "gamma": 10.0,
  "step_size": 0.5,
  "max_iter": 1000,
  "tol": 1e-4,
  "merge_tol": 0.5
}
```
 
**Response:**
 
```json
{
  "labels": [0, 0, 1],
  "cluster_centers": [[0.5, 0.0], [0.5, 0.0], [5.0, 5.0]],
  "n_clusters": 2,
  "n_iter": 47,
  "convergence": [
    {"iteration": 3, "center_diff": 0.021},
    {"iteration": 4, "center_diff": 0.009}
  ]
}
```
 
---
 
### `POST /compare`
 
Runs multiple algorithms on the same `(X, W)` input and returns a
side-by-side comparison of silhouette score, number of clusters, and
number of iterations.
 
**Request body:**
 
```json
{
  "X": [[0.0, 0.0], [1.0, 0.0], [5.0, 5.0]],
  "W": [[0, 0.6, 0], [0.6, 0, 0], [0, 0, 0]],
  "algorithms": ["ADMM", "AMA", "DR"],
  "gamma": 10.0,
  "step_size": 0.5,
  "max_iter": 1000,
  "tol": 1e-4,
  "merge_tol": 0.5
}
```
 
**Response:**
 
```json
{
  "results": [
    {
      "algorithm": "ADMM",
      "labels": [0, 0, 1],
      "n_clusters": 2,
      "n_iter": 47,
      "silhouette_score": 0.82
    }
  ]
}
```