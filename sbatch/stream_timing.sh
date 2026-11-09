#!/bin/bash
# docs/29 §8 step 2 — SUBMIT ONLY AFTER Kati has signed docs/26 and docs/29.
# 1. Parity (Kati's condition 3): the FULL A–B–A arm seed 8, γ0 10, H8, jump run through aba_lib and,
#    as a T = 2 stream, through stream_lib; every shared quantity must be bit-identical. Stops on failure.
# 2. ONE design-scale stream arm with every measurement (docs/29 §2b): draw 8, γ0 0.1 (most updates),
#    h 8, jump; T = 16, M 150, n_t 200; sequential and timed (train, fresh, GLUE, NTK, floor).
#    Row to results/stream/timing_<stream_hash>.jsonl, never a grid checkpoint.
# Sizing (an estimate, to be replaced by this job's measurement): ~290 GLUE calls at ~4 s (A–B–A FULL:
# 66 s for 16 calls at γ0 0.1) ≈ 20 min, plus 16 stream tasks and 15 fresh learners (~2000 updates each
# at γ0 0.1; A–B–A: 112 s per ~9100 updates) ≈ 6 min, plus 17 NTK snapshots ⇒ ~30–40 min sequential;
# parity ≈ 3 min (two FULL A–B–A-sized runs). 2 CPUs, sequential; peak RSS of an A–B–A arm was 124 MB.
# The grid (288 arms) is sized from step 2 × 4 (the measured parallel slowdown, jobs 4349/4458).
#SBATCH --job-name=stream-timing
#SBATCH --partition=main
#SBATCH --nodelist=grime
#SBATCH --cpus-per-task=2
#SBATCH --mem=4000M
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

echo "== parity (FULL): A–B–A arm seed 8, γ0 10, H8, jump as a T = 2 stream"
uv run --frozen python notebooks/core/stream_cluster.py parity --spec 8,10,8,0.1
echo "== one design-scale stream arm: draw 8, γ0 0.1, h 8, jump, T = 16"
uv run --frozen python notebooks/core/stream_cluster.py arm --spec 8,0.1,8,0.1 --T 16
git status --porcelain | head
echo "finished: $(date -Is)"
