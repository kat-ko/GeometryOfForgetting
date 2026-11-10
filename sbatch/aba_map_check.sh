#!/bin/bash
# docs/34 §7: (1) reproduce ONE docs/28 FULL arm bit for bit under the new aba_lib hash (job 4349's
# seed 0, γ0 10, H8, s_f 0.1 row; every reference field compared; the added relations.taskvec_* fields
# are listed); (2) ONE docs/34 map arm, sequential and timed, to size Phase 1 (771 arms).
# Timing arm: seed 16, γ0 0.1, H16, s_f 1 (new level, lazy learner).
# Sizing: criterion arms took 57–149 s sequential in jobs 4453 and 4533; peak RSS ~125 MB.
#SBATCH --job-name=aba-map-check
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

echo "== docs/34 §7 reproduction: one docs/28 FULL arm vs job 4349"
uv run --frozen python notebooks/core/aba_cluster.py reproduce
echo "== docs/34 timing: one map arm, sequential"
uv run --frozen python notebooks/core/aba_cluster.py arm --map --spec 16,0.1,16,1
echo "finished: $(date -Is)"
