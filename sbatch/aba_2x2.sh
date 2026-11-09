#!/bin/bash
# docs/32 step 2: the A–B–A stopping 2x2, 4 cells × 32 arms (h 4 frozen/drift × γ0 0.1/10 × seeds 8–15),
# then the §2 integrity check on every (Cr,Cr) and (Fx,Fx) row. Submit ONLY after
# sbatch/aba_2x2_timing.sh has passed and Kati has given the go.
# Readings are computed locally: aba_summary.py --twobytwo FULL_<hash>.jsonl FULL_<hash>_fixed.jsonl.
# Sizing (to be replaced by the timing job): ≤ 184 s per arm sequential × 4 (parallel slowdown) ≈ 12 min per
# cell of 32 arms on 32 workers; 4 cells ≈ 50 min. --time 3 h for margin.
#SBATCH --job-name=aba-2x2
#SBATCH --partition=main
#SBATCH --nodelist=grime
#SBATCH --cpus-per-task=32
#SBATCH --mem=16000M   # 32 × ~0.13 GB + overhead; check AllocMem (grime keeps 32 GB unschedulable)
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

echo "== docs/32 cells"
uv run --frozen python notebooks/core/aba_cluster.py cells --workers "$SLURM_CPUS_PER_TASK"
echo "== docs/32 §2 integrity check, every (Cr,Cr) and (Fx,Fx) row"
uv run --frozen python notebooks/core/aba_cluster.py match
git status --porcelain | head
echo "finished: $(date -Is)"
