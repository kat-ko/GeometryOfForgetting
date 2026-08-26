"""Process-pool helper for the measurement scripts.

BLAS threads are pinned to 1 in the workers: the per-sample linear algebra is
small (P×P Grams, one NNLS), so multi-threaded BLAS only adds contention once we
are already running one process per core.

**Worker count is capped at 192, measured on the real workload**
(`results/scaling_colgen.json`): 2.83 eval/s at 128, **3.39 at 192**, 3.47 at 254. 254
buys 2% over 192 for 48% more latency per evaluation, so 192 is the operating point.

This cap replaces an earlier one of 128, and the correction is instructive. The 128 cap
came from `results/scaling.json`, which showed throughput *falling* past 128 — the
workload was then memory-bandwidth-bound, since the dense NNLS Gram matrix at 46 MB
exceeded the 32 MB L3 and every worker streamed it from DRAM. Two things changed. The
column-generation solver shrank the working set enough that the workload is no longer
bandwidth-bound, which moved the peak to the right; and `scaling.json` had been measured
at `n_t = 20` while the grid runs at `n_t = 200`, so it characterised an evaluation ten
times cheaper than the real one. The peak's location is a property of the workload, so a
scaling curve has to be measured at the settings it will be used to size.

`nproc` reports 256 on this 2×64-core EPYC 7763 only because of SMT, so 192 is already
oversubscribed against 128 physical cores — which now helps, the workload having become
latency-bound rather than bandwidth-bound.
"""

from __future__ import annotations

import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor

MAX_WORKERS = 192

# `spawn`, not the Linux default `fork`. A forked worker inherits the parent's
# already-imported modules, so a pool launched after an edit runs the *old* code while
# the repository holds the new — which is how an 8-arm pilot came to time a superseded
# solver. `spawn` re-imports from disk in every worker. It costs a second of startup
# per worker against arms that run for tens of minutes.
START_METHOD = "spawn"

_PIN = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}


def pin_threads() -> None:
    """Call before importing numpy in a worker entry point."""
    os.environ.update(_PIN)


def n_workers(cap: int | None = None) -> int:
    n = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
    n = min(max(1, (n or 1) - 2), MAX_WORKERS)
    return min(n, cap) if cap else n


def pmap(fn, jobs, *, cap: int | None = None, chatty: bool = True):
    """Run `fn` over `jobs` in a process pool, preserving order."""
    pin_threads()
    workers = min(n_workers(cap), max(1, len(jobs)))
    if chatty:
        print(f"  [{workers} workers x {len(jobs)} jobs]", flush=True)
    if workers == 1:
        return [fn(j) for j in jobs]
    ctx = mp.get_context(START_METHOD)
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
        return list(ex.map(fn, jobs))
