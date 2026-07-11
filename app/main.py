"""
FastAPI application — Convex Clustering API.
 
Entry point: uvicorn app.main:app --reload

This module only creates the app and registers routers
----------------------------------------------------------
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from collections.abc import AsyncGenerator

from fastapi import FastAPI

from app.routers import algorithms, cluster, compare


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    yield

app = FastAPI(
    title = "Convex Clustering API",
    description = (
        "REST API for convex clustering via ADMM, AMA, Douglas-Rachford, "
        "and RF-S variants. Implements the ConvexClusterer estimator with "
        "a scikit-learn-compatible interface.\n\n"
        "**Portfolio project** — Juan Nicolás Mendoza Roncancio  \n"
        "Source: https://github.com/JuanNicolasMendozaRoncancio/Convex-Clustering"
    ),
    version = "0.1.0",
    lifespan=lifespan,
)

# Register routers.
app.include_router(cluster.router, prefix = "/cluster", tags = ["clustering"])
app.include_router(algorithms.router, prefix = "/algorithms", tags = ["algorithms"])
app.include_router(compare.router, prefix = "/compare", tags = ["comparison"])


@app.get("/", tags = ["Health"])
def root() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok",
            "docs": "/docs",
            "version": "0.1.0",
            }