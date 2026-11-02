#!/bin/bash
#SBATCH --job-name=gof-v3
#SBATCH --partition=main
#SBATCH --nodelist=grime
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=/home/kathrin/logs/%x-%j.out
#SBATCH --error=/home/kathrin/logs/%x-%j.err

set -euo pipefail

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export WORKERS=$SLURM_CPUS_PER_TASK

cd /home/kathrin/GeometryOfForgetting

uv sync

echo "node: $(hostname)  cpus: $SLURM_CPUS_PER_TASK  started: $(date -Is)"
uv run python -m ENTRYPOINT_PLACEHOLDER
echo "finished: $(date -Is)"