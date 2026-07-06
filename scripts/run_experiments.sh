#!/usr/bin/env bash
# =============================================================================
# run_experiments.sh — runs the complete matrix of clustering experiments
#
# Usage:
#   bash scripts/run_experiments.sh
#
# Prerequisites:
#   - Environment with convex-clustering installed (pip install -e ".[dev]")
#   - AWS credentials in environment variables:
#       export AWS_ACCESS_KEY_ID=...
#       export AWS_SECRET_ACCESS_KEY=...
#       export AWS_REGION=eu-north-1
#
# The matrix covers:
#   3 datasets × 3 algorithms × 4 gamma values = 36 experiments
#   Each run saves results in S3 under results/{exp_id}/
#   and logs metrics in MLflow (SQLite backend: mlflow.db)
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# BASH_SOURCE[0] is the path of the script being executed.
# cd + pwd converts any relative path to an absolute one.
# This ensures the script works regardless of from where it is called.
PROJECT_ROOT="$(cd "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"
# We move to the root of the project so that run_experiment_job.py
# can find mlflow.db and the relative paths correctly.

# -----------------------------------------------------------------------------
# Verification of credentials before launching 36 experiments
# -----------------------------------------------------------------------------
if [[ -z "${AWS_ACCESS_KEY_ID:-}" || -z "${AWS_SECRET_ACCESS_KEY:-}" ]]; then
    echo "Error: AWS credentials are not set in environment variables."
    echo "Please set AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, and AWS_REGION."
    exit 1
fi

export AWS_REGION="${AWS_REGION:-eu-north-1}"
# If AWS_REGION is not defined, use eu-north-1 as default.
# :- is conditional expansion in bash: uses the value if it exists, otherwise uses the default.

# -----------------------------------------------------------------------------
# Experiment matrix definition
# -----------------------------------------------------------------------------
DATASETS=("blobs" "moons" "circles")
ALGORITHMS=("ADMM" "AMA" "DR")
GAMMAS=("1" "10" "100" "1000")

STEP_SIZE="0.05"
MAX_ITER="1000"
MERGE_TOL="0.5"

TOTAL=0
PASSED=0
FAILED=0
FAILED_RUNS=()


# Principal loop to iterate over the experiment matrix
echo "============================================================"
echo "  Convex Clustering — Experiment Matrix"
echo "  Datasets:   ${DATASETS[*]}"
echo "  Algorithms: ${ALGORITHMS[*]}"
echo "  Gammas:     ${GAMMAS[*]}"
echo "  Total runs: $((${#DATASETS[@]} * ${#ALGORITHMS[@]} * ${#GAMMAS[@]}))"
echo "============================================================"
echo ""

for dataset in "${DATASETS[@]}"; do
    for algorithm in "${ALGORITHMS[@]}"; do
        for gamma in "${GAMMAS[@]}"; do
            TOTAL=$((TOTAL + 1))
            RUN_ID="${dataset}/${algorithm}/gamma=${gamma}"

            echo "──────────────────────────────────────────"
            echo "  Run ${TOTAL}: ${RUN_ID}"
            echo "──────────────────────────────────────────"

            if python scripts/run_experiment_job.py \
                --dataset   "$dataset"   \
                --algorithm "$algorithm" \
                --gamma     "$gamma"     \
                --step_size "$STEP_SIZE" \
                --max_iter  "$MAX_ITER"  \
                --merge_tol "$MERGE_TOL"; then
                PASSED=$((PASSED + 1))
                echo "  ✓ OK"
            else
                FAILED=$((FAILED + 1))
                FAILED_RUNS+=("$RUN_ID")
                echo "  ✗ FAILED — continuando con el siguiente run"
            fi
            echo ""
        done
    done
done

#Final results summary
echo "============================================================"
echo "  Summary of runs:"
echo "  Total:  ${TOTAL}"
echo "  OK:     ${PASSED}"
echo "  Failed: ${FAILED}"
if [[ ${#FAILED_RUNS[@]} -gt 0 ]]; then
    echo ""
    echo "  Failed runs:"
    for run in "${FAILED_RUNS[@]}"; do
        echo "    - ${run}"
    done
fi
echo "============================================================"

# Exit code no-cero if there were failures
if [[ $FAILED -gt 0 ]]; then
    exit 1
fi