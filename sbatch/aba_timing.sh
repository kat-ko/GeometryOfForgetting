#!/bin/bash
# A–B–A pre-FULL job: (1) bottleneck-n calibration on the FULL init representation,
# (2) one FULL arm per γ0 at the slowest cell (H8, s_f 0.1), sequential, wall + peak RSS,
# (3) the QUICK notebook end to end at 3 γ0 (tests the analysis code before FULL; output
#     goes to ~/logs, never into the repo, so the tree stays clean).
#SBATCH --job-name=aba-timing
#SBATCH --partition=main
#SBATCH --nodelist=grime
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/home/kathrin/logs/%x-%j.out
#SBATCH --error=/home/kathrin/logs/%x-%j.err

set -euo pipefail

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

cd /home/kathrin/GeometryOfForgetting
uv sync --frozen --extra figures --extra notebook

echo "node: $(hostname)  cpus: $SLURM_CPUS_PER_TASK  commit: $(git rev-parse --short HEAD)  started: $(date -Is)"
git status --porcelain | head

echo "== calibrate"
uv run --frozen python notebooks/core/aba_cluster.py calibrate
echo "== timing (FULL, sequential)"
uv run --frozen python notebooks/core/aba_cluster.py timing

echo "== QUICK notebook at 3 γ0 ($SLURM_CPUS_PER_TASK workers)"
OUT=/home/kathrin/logs/aba-quick-$SLURM_JOB_ID
mkdir -p "$OUT"
cd notebooks/core
ABA_MODE=quick ABA_RUN_GRID=1 ABA_WORKERS=$SLURM_CPUS_PER_TASK \
  uv run --frozen jupyter nbconvert --to notebook --execute aba_controlled.ipynb \
  --ExecutePreprocessor.kernel_name=python3 --ExecutePreprocessor.timeout=7200 --output-dir "$OUT"
cd ../..
git status --porcelain | head
echo "finished: $(date -Is)"
