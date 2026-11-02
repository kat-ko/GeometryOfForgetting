#!/bin/bash
# One v3 arm on a compute node, then its NumPy build.
# Per-arm cost only: 2 CPUs, one BLAS thread. Re-time if the grid's CPU count changes.
#SBATCH --job-name=gof-v3-time
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/v3-time-one-%j.log
set -euo pipefail

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
uv sync --locked
uv run --locked python scripts/v3_one_arm.py
