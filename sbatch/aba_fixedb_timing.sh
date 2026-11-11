#!/bin/bash
# docs/36 step 1 (Phase 1b, fixed budget on phase B per γ0): reproductions under the new lib hash, then
# ONE fixed-B arm, sequential and timed. Runs on grime, where jobs 4349 and 4651 ran (bit-identity).
#   1. reproduce        : docs/28 arm (job 4349: seed 0, γ0 10, H8, s_f 0.1), bit for bit
#   2. reproduce --map  : job 4651 arm (seed 16, γ0 0.1, H16, s_f 1), bit for bit
#   3. arm              : seed 19, γ0 0.1, H16, s_f 1 with B and fresh B fixed at 3200 (job 4651's largest
#                         B count, 3132): first_hit B must be 3132.
#SBATCH --job-name=aba-1b-timing
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

echo "== docs/36 reproduction 1: docs/28 arm vs job 4349"
uv run --frozen python notebooks/core/aba_cluster.py reproduce
echo "== docs/36 reproduction 2: map arm vs job 4651"
uv run --frozen python notebooks/core/aba_cluster.py reproduce --map
echo "== docs/36 timing: one fixed-B arm, sequential"
uv run --frozen python notebooks/core/aba_cluster.py arm --map --spec 19,0.1,16,1 --stopping fixed --budget -,3200,-
echo "finished: $(date -Is)"
