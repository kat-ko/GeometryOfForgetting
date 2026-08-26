#!/bin/bash
# Unique-n campaign: (a) γ=5 5×8, (b) width γ=10 5×8, (c) K=3 unconfound.
# Sequential so the 192-worker cap is not shared. Log: results/logs/unique_n_campaign.log
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p results/logs results/gamma_5_n40 results/width_g10_n40 results/unconfound_k3
LOG=results/logs/unique_n_campaign.log
exec > >(tee -a "$LOG") 2>&1

echo "=== $(date -Is) (a) γ=5 5×8 unique → results/gamma_5_n40 ==="
.venv/bin/python scripts/run_gamma_ext.py --gamma 5 --streams 5 --seeds 8 --out results/gamma_5_n40

echo "=== $(date -Is) (b) width γ=10 5×8 unique → results/width_g10_n40 ==="
.venv/bin/python scripts/run_width.py --gammas 10 --streams 5 --seeds 8 --out results/width_g10_n40

echo "=== $(date -Is) (c) K=3 unconfound 3 arr × 40 inits × 4 corners × 2 γ ==="
.venv/bin/python scripts/run_unconfound.py

echo "=== $(date -Is) campaign done ==="
