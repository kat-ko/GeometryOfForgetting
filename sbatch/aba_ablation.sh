#!/bin/bash
# docs/31 stopping-rule ablation: the replication grid (seeds 8–15, 219 arms) under a FIXED budget of
# A1 2300, B 2400, A2 2000 updates (fresh B = B budget), identical for every γ0; then the fixed-budget
# floors (seed 8, K = 8). Submit ONLY after docs/31 is committed (pre-commit).
# No notebook: its identity/validation checks assume criterion stopping; readings are computed locally by
# notebooks/core/aba_summary.py --ablation (docs/31 §6).
# Sizing (grime job 4453, lib 0ee97f82b6bee621): the heaviest arm (seed 8, γ0 0.1, H8, frozen) took 184 s
# sequential (train 112, GLUE 66), peak RSS 124 MB. Every fixed arm trains the same ~9100 updates, so
# ≈ 184 s × 4 (measured parallel slowdown, job 4349) ≈ 740 s per arm; 219 arms / 32 workers ≈ 7 rounds
# ≈ 85 min, plus floors. --time 4 h for margin.
#SBATCH --job-name=aba-ablation
#SBATCH --partition=main
#SBATCH --nodelist=grime
#SBATCH --cpus-per-task=32
#SBATCH --mem=16000M   # 32 × ~0.13 GB + overhead; check AllocMem (grime keeps 32 GB unschedulable)
#SBATCH --time=04:00:00
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

FLAGS="--seed-offset 8 --stopping fixed --budget 2300,2400,2000"
echo "== docs/31 ablation grid, seeds 8–15, fixed budget"
uv run --frozen python notebooks/core/aba_cluster.py grid $FLAGS --workers "$SLURM_CPUS_PER_TASK"
echo "== docs/31 fixed-budget floors (seed 8, K = 8)"
uv run --frozen python notebooks/core/aba_cluster.py floors $FLAGS
git status --porcelain | head
echo "finished: $(date -Is)"
