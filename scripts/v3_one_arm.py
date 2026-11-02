"""Time one reduced-scale v3 arm and print the BLAS build it ran under.

The arm is the costly calibration cell: gamma=0.03, s_f=0.1, Hamming 8, replicate 0.
One process, one BLAS thread. This is a per-arm cost, not a contention measurement.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

for _blas_threads in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_blas_threads] = "1"

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "notebooks")]

import numpy as np

import v3_lib as v


def blas_line() -> str:
    cfg = np.show_config(mode="dicts")
    blas = cfg.get("Build Dependencies", {}).get("blas", {})
    return (
        f"numpy {np.__version__} blas={blas.get('name')} "
        f"version={blas.get('version')} config={blas.get('openblas configuration')}"
    )


if __name__ == "__main__":
    print("digest", v.CODE_DIGEST)
    print("cpu_count", os.cpu_count())
    print("threads", {k: os.environ[k] for k in (
        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")})
    print(blas_line())
    spec = v.ArmSpec(gamma=0.03, s_f=0.1, hamming=8, replicate=0)
    arm = v.run_arm(v.V3Config(), spec)
    steps = sum(r["steps"] for r in arm["rows"][1:])
    print(
        f"arm gamma={spec.gamma} s_f={spec.s_f} hamming={spec.hamming} "
        f"replicate={spec.replicate} usable={arm['usable']} steps={steps} "
        f"wall_s={arm['wall_seconds']:.1f}"
    )
