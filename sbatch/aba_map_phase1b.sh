#!/bin/bash
# docs/36 Phase 1b (extreme cases, run 1): fixed budget on phase B and fresh B per γ0 (3200 / 600 / 100),
#   A1 and A2 at criterion; 11 cells × 3 γ0 × seeds 16–23 = 264 arms, paired with job 4651.
# Checkpoint: results/aba/FULL_<lib>_fixed.jsonl (one config per γ0, separated by the config dict).
# Integrity checks (docs/36 §2) run at the end (check1b); readings are computed locally by
# notebooks/core/aba_fixed_budget_b.ipynb (load only). EXPLORATORY.
# Sizing (estimate before measurement; submitted with --dependency=afterok on the timing job 4787):
#   a fixed-B γ0 0.1 arm ≈ 250–300 s sequential (job 4651 worst criterion arm 170 s, + ~1900 extra
#   B / fresh-B updates); × ~5 parallel slowdown (job 4651) → ≈ 48 min of 64-worker rounds + tail ≈ 1–1.5 h.
#   The measured timing (job 4787) goes into docs/36 §7.
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

echo "== docs/36 guard: the timed fixed-B arm (job 4787) must have first_hit B = 3132 (job 4651 steps B)"
uv run --frozen python - <<'PY'
import json, sys
sys.path.insert(0, "notebooks/core")
import aba_lib as L
p = L.REPO / "results" / "aba" / f"timing_{L.lib_hash()}.jsonl"
rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
r = [x for x in rows if x["key"] == "seed=19,gamma=0.1,H=16,s_f=1" and x["config"].get("fixed_steps") == [None, 3200, None]]
hit = r[-1]["behaviour"]["first_hit"]["B"] if r else None
print(f"timed arm first_hit B = {hit}")
sys.exit(0 if hit == 3132 else 1)
PY
echo "== docs/36 Phase 1b"
uv run --frozen python notebooks/core/aba_cluster.py map1b --workers "$SLURM_CPUS_PER_TASK"
echo "== docs/36 integrity checks vs job 4651"
uv run --frozen python notebooks/core/aba_cluster.py check1b
git status --porcelain | head
echo "finished: $(date -Is)"
