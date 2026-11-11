#!/bin/bash
# docs/36 Phase 1b (extreme cases, run 1): fixed budget on phase B and fresh B per γ0 (3200 / 600 / 100),
#   A1 and A2 at criterion; 11 cells × 3 γ0 × seeds 16–23 = 264 arms, paired with job 4651.
# Checkpoint: results/aba/FULL_<lib>_fixed.jsonl (one config per γ0, separated by the config dict).
# Integrity checks (docs/36 §2) run at the end (check1b); readings are computed locally by
# notebooks/core/aba_fixed_budget_b.ipynb (load only). EXPLORATORY.
# Sizing: SIZING_PLACEHOLDER
#SBATCH --job-name=aba-map-p1b
#SBATCH --partition=main
#SBATCH --nodelist=grime
#SBATCH --cpus-per-task=64
#SBATCH --mem=32000M   # job 4651: MaxRSS 9.1 GB on 64 workers
#SBATCH --time=06:00:00
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

echo "== docs/36 Phase 1b"
uv run --frozen python notebooks/core/aba_cluster.py map1b --workers "$SLURM_CPUS_PER_TASK"
echo "== docs/36 integrity checks vs job 4651"
uv run --frozen python notebooks/core/aba_cluster.py check1b
git status --porcelain | head
echo "finished: $(date -Is)"
