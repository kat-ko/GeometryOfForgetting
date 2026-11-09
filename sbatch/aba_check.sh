#!/bin/bash
# Bit-identity check for the seed_offset change (docs/28 replication amendment).
# Reruns ONE FULL arm (seed 0, γ0 = 10, H8, s_f 0.1) under the new aba_lib hash and compares it
# field by field with job 4349's row in results/aba/FULL_2528909f2feaf3f3.jsonl
# (all fields except lib_hash, stamp, timing). Exits non-zero on any difference.
# Sizing: this arm took 61 s sequential in job 4316; peak RSS 128 MB.
#SBATCH --job-name=aba-check
#SBATCH --partition=main
#SBATCH --nodelist=grime
#SBATCH --cpus-per-task=2
#SBATCH --mem=4000M
#SBATCH --time=00:30:00
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

uv run --frozen python notebooks/core/aba_cluster.py reproduce
echo "finished: $(date -Is)"
