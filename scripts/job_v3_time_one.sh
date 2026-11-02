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
export PYTHONUNBUFFERED=1

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
# Lockfile selects NumPy by Python version: 2.2.6 below 3.11, 2.4.6 on 3.11.
# The laptop grid ran on 3.11.9 / NumPy 2.4.6. The login image is 3.10.
uv python install 3.11
uv sync --python 3.11 --locked
uv run --python 3.11 --locked python scripts/v3_one_arm.py
