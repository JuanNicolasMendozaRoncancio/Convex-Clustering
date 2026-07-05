# Convex Clustering
[![CI](https://github.com/JuanNicolasMendozaRoncancio/Convex-Clustering/actions/workflows/ci.yml/badge.svg)](https://github.com/JuanNicolasMendozaRoncancio/Convex-Clustering/actions/workflows/ci.yml)

Convex interpretable methods, ideal for interpretable outputs.

Implements seven algorithms for convex clustering: ADMM, AMA,
Douglas-Rachford (DR), RFS_L2, Fast_RFS_L2, RFS_L1 and Fast_RFS_L1,
exposed through a unified, scikit-learn-compatible `ConvexClusterer`
estimator.

## Installation (development)

```bash
pip install -e ".[dev]"
```

## Status

This project is under active development as part of an industrialization
portfolio. Full documentation, examples and benchmarks are in progress.


## Runing experiments on Google Cloud

Bluid and push the image:

    docker build -t gcr.io/{PROJECT_ID}/convex-clustering-job .
    docker push gcr.io/{PROJECT_ID}/convex-clustering-job

Create the job:

    gcloud run jobs create convex-clustering-job \
      --image gcr.io/{PROJECT_ID}/convex-clustering-job \
      --region europe-north1 \
      --set-env-vars AWS_REGION=eu-north-1 \
      --set-secrets AWS_ACCESS_KEY_ID=aws-key-id:latest \
      --set-secrets AWS_SECRET_ACCESS_KEY=aws-secret-key:latest

Execute an experiment:

    gcloud run jobs execute convex-clustering-job \
      --args="--dataset,blobs,--algorithm,ADMM,--gamma,1.0"