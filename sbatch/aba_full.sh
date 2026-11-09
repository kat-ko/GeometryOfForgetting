#!/bin/bash
# A–B–A FULL grid (n = 8, 219 arms) + notebook in FULL mode. Submit ONLY after docs/28 is signed.
# Sizing (grime job 4316, commit 7945c7d, one process per arm, sequential):
#   slowest arms (H8, s_f 0.1): γ0=0.1 173 s, γ0=1 68 s, γ0=10 61 s; peak RSS 128 MB.
#   Upper bound ≈ 73 × (173 + 68 + 61) s ≈ 6.1 CPU-h → ≈ 12 min on 32 workers + a 3-min tail.
#   Notebook (§3–§4 at FULL incl. floors at K = 8 for 3 γ0, then readings) adds well under 1 h.
#SBATCH --job-name=aba-full
#SBATCH --partition=main
#SBATCH --nodelist=grime
#SBATCH --cpus-per-task=32
#SBATCH --mem=16000M   # 32 × ~0.13 GB + kernel; grime keeps 32 GB unschedulable (MemSpecLimit)
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
if [ -n "$(git status --porcelain)" ]; then echo "tree is dirty; refusing to run"; git status --porcelain; exit 1; fi

echo "== FULL grid"
uv run --frozen python notebooks/core/aba_cluster.py grid --workers "$SLURM_CPUS_PER_TASK"

echo "== FULL notebook"
OUT=/home/kathrin/GeometryOfForgetting/results/aba/notebook-full-$SLURM_JOB_ID
mkdir -p "$OUT"
cd notebooks/core
ABA_MODE=full ABA_RUN_GRID=0 ABA_WORKERS=$SLURM_CPUS_PER_TASK \
  uv run --frozen jupyter nbconvert --to notebook --execute aba_controlled.ipynb \
  --ExecutePreprocessor.kernel_name=python3 --ExecutePreprocessor.timeout=7200 --output-dir "$OUT"
cd ../..
git status --porcelain | head
echo "finished: $(date -Is)"
