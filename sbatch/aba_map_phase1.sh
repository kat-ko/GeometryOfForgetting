#!/bin/bash
# docs/34 Phase 1 (amended 2026-11-10): the A–B–A similarity map, two slices, seeds 16–23.
#   H sweep {0,2,4,6,8,12,16} at s_f {1, 0.9}; s_f sweep {1,0.95,0.9,0.8,0.6,0.4,0.2,0.1} at H {0,4,8}.
#   32 configurations × 3 γ0 × 8 seeds = 768 arms + 3 fresh-noise identity arms = 771.
# Checkpoint: results/aba/FULL_<lib>.jsonl (map config dict carries probe_hammings). Readings are
# computed locally by notebooks/core/aba_similarity_map.ipynb (load only). EXPLORATORY.
# Sizing (grime job 4648, lib e3011fa49aca1994, commit 3650407): docs/28 arm reproduced bit for bit;
# the worst map arm (seed 16, γ0 0.1, H16, s_f 1) took 170 s sequential (train 107, GLUE 61), peak RSS
# 125 MB; γ0 10 arms take ~60 s. × 3–4 parallel slowdown, 771 arms / 64 workers ≈ 12 rounds ≈ 1.5–2.5 h.
#SBATCH --job-name=aba-map-p1
#SBATCH --partition=main
#SBATCH --nodelist=grime
#SBATCH --cpus-per-task=64
#SBATCH --mem=32000M   # 64 × ~0.13 GB + overhead; grime had ~270 GB free at submission
#SBATCH --time=08:00:00
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

echo "== docs/34 Phase 1"
uv run --frozen python notebooks/core/aba_cluster.py map1 --workers "$SLURM_CPUS_PER_TASK"
git status --porcelain | head
echo "finished: $(date -Is)"
