#!/bin/bash
# A–B–A FULL held-out replication: seeds 8–15 (n = 8, 219 arms) + notebook in FULL mode.
# Same signed rules as docs/28; submit ONLY after the replication amendment is committed and the
# bit-identity check (sbatch/aba_check.sh) has passed. Reported beside job 4349, never pooled.
# Sizing, measured on the original FULL run (grime job 4349, 32 workers): grid wall 1975 s,
# arms 144–412 s each in parallel (3–4× their sequential times), notebook ~5 min, 38 min end to end.
# Earlier sequential timing (grime job 4316, commit 7945c7d, one process per arm):
#   slowest arms (H8, s_f 0.1): γ0=0.1 173 s, γ0=1 68 s, γ0=10 61 s; peak RSS 128 MB.
#   (the 12-min projection from these was 3× too low; see job 4349 above).
#SBATCH --job-name=aba-full-rep
#SBATCH --partition=main
#SBATCH --nodelist=grime
#SBATCH --cpus-per-task=32
#SBATCH --mem=16000M   # 32 × ~0.13 GB + kernel; grime keeps 32 GB unschedulable (MemSpecLimit)
#SBATCH --time=02:00:00
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

echo "== FULL grid, replication seeds 8–15"
uv run --frozen python notebooks/core/aba_cluster.py grid --seed-offset 8 --workers "$SLURM_CPUS_PER_TASK"

echo "== FULL notebook, replication seeds 8–15"
OUT=/home/kathrin/GeometryOfForgetting/results/aba/notebook-full-rep-$SLURM_JOB_ID
mkdir -p "$OUT"
cd notebooks/core
ABA_MODE=full ABA_SEED_OFFSET=8 ABA_RUN_GRID=0 ABA_WORKERS=$SLURM_CPUS_PER_TASK \
  uv run --frozen jupyter nbconvert --to notebook --execute aba_controlled.ipynb \
  --ExecutePreprocessor.kernel_name=python3 --ExecutePreprocessor.timeout=7200 --output-dir "$OUT"
cd ../..
git status --porcelain | head
echo "finished: $(date -Is)"
