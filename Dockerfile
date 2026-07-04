# syntax=docker/dockerfile:1

# ---- base image ----
# python:3.11-slim is Debian, no development utilities.
# "slim" reduces image size from ~1GB (python:3.11) to ~130MB.
# 3.11 coincide with the version used by CI — environment consistency.
FROM python:3.11-slim

# Avoids Python to write .pyc files to disk and buffers stdout/stderr.
# Without PYTHONUNBUFFERED=1, the print() statements from the job won't appear in the
# Cloud Run logs until the process completes — making debugging impossible.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Copies only pyproject.toml first.
# Why: Docker caches layers. If you copy the entire codebase and then install dependencies,
# any change in a .py file will invalidate the dependencies layer and pip will reinstall everything from scratch.
# By copying pyproject.toml first, the dependencies layer is only rebuilt when the dependencies change,
# not when the code changes. This can save 2-3 minutes per build in images with boto3 + sklearn.
COPY pyproject.toml .

# Instals dependencies from pyproject.toml + extra cloud (boto3).
# --no-cache-dir: doesn't save the pip cache in the image — reduce size.
# The package is installed in non-editable mode because in production there is no
# git repository — only the copied files.
RUN pip install --no-cache-dir ".[cloud]"

# Copies the source code after installing dependencies.
COPY src/ ./src/
COPY scripts/ ./scripts/

# Reinstalls the package now that the source code is available.
# The first installation resolved and downloaded dependencies (cached layer).
# This second installation registers the package with the actual code.
RUN pip install --no-cache-dir -e .

# The entrypoint is the job script. The arguments come from the Cloud Run job definition.
# ENTRYPOINT in exec form (list) instead of shell form (string):
# with exec form the Python process is PID 1 and receives SIGTERM signals directly.
# with shell form, bash is PID 1 and may swallow the signals
# — the container does not shut down cleanly when Cloud Run cancels it.
ENTRYPOINT ["python", "scripts/run_experiment_job.py"]
CMD ["--dataset", "blobs", "--algorithm", "DR", "--gamma", "1.0"]