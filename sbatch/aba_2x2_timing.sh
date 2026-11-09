#!/bin/bash
# docs/32 step 1 (after the docs/32 pre-commit f03573f): the integrity check on one arm per existing
# cell, then ONE new-cell arm, sequential and timed, to size sbatch/aba_2x2.sh.
#   match (Cr,Cr) seed 8, γ0 10, h 4 drift   vs job 4396 (FULL_4e110835b8fc2863.jsonl)
#   match (Fx,Fx) seed 8, γ0 0.1, h 4 frozen vs job 4458 (FULL_0ee97f82b6bee621_fixed.jsonl)
#   arm   (Fx,Cr) seed 8, γ0 0.1, h 4 frozen: A1 fixed 2300, B to criterion, A2 fixed 2000
# Sizing: a criterion arm took 61 s and a fixed arm 184 s sequential (jobs 4316, 4453); peak RSS ~130 MB.
#SBATCH --job-name=aba-2x2-timing
#SBATCH --partition=main
#SBATCH --nodelist=grime
#SBATCH --cpus-per-task=2
#SBATCH --mem=4000M
#SBATCH --time=01:00:00
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

echo "== docs/32 §2 integrity check, (Cr,Cr) vs job 4396"
uv run --frozen python notebooks/core/aba_cluster.py match --spec 8,10,4,0.9
echo "== docs/32 §2 integrity check, (Fx,Fx) vs job 4458"
uv run --frozen python notebooks/core/aba_cluster.py match --spec 8,0.1,4,1 --stopping fixed --budget 2300,2400,2000
echo "== docs/32 timing: one (Fx,Cr) arm, sequential"
uv run --frozen python notebooks/core/aba_cluster.py arm --spec 8,0.1,4,1 --stopping fixed --budget 2300,-,2000
echo "finished: $(date -Is)"
