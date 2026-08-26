"""Simulation capacity `α_sim` — ground truth for `02` §2a.

Wraps `correlated_capacity`'s **`check_data_separability_general`** (a public
function; nothing in `third_party/` is modified) and owns the bisection itself.

Two reasons the vendored entry point `manifold_simcap_analysis` is not called
directly:

1. **It is generic-only as shipped.** `compute_sep_Nc_general` draws its own
   random balanced labels internally (`manifold_simcap_analysis.py:180`), so it
   cannot produce retained or tilted capacity — the very thing we need α_sim for
   once `replicaMFT` turned out to be label-invariant. The label ensemble must be
   injectable, so the bisection has to live on our side.
2. **It calls global `np.random.seed`** (`:173`), which violates `AGENTS.md` §4.

Everything else follows the vendored procedure: κ = 0, random normalized
projection `R^N → R^{N_cur}`, bisection on `N_cur` for the 0.5-separability point,
linear interpolation between the bracketing values, `α_sim = P / N_c`.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..core import Ensemble

_VENDOR = Path(__file__).resolve().parents[3] / "third_party" / "correlated_capacity"
if str(_VENDOR) not in sys.path:
    sys.path.insert(0, str(_VENDOR))

from capacity.manifold_simcap_analysis import (  # noqa: E402
    check_data_separability_general,
)

ESTIMATOR = "correlated_capacity@de8dac79760e8eb552c036604503efcb87e1e6b6"


@dataclass
class SimcapResult:
    alpha_sim: float
    N_c: float
    P: int
    N: int
    n_rep: int
    ensemble: str
    N_vec: list[int]
    p_vec: list[float]
    estimator: str = ESTIMATOR

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def _p_separable(
    manifolds, N_cur: int, n_rep: int, rng: np.random.Generator, draw_y
) -> float:
    """Fraction of `n_rep` random projections to `N_cur` dims that stay separable."""
    N = manifolds[0].shape[0]
    sep = []
    for _ in range(n_rep):
        W = rng.standard_normal((N, N_cur))
        W /= np.sqrt(np.sum(W**2, axis=0, keepdims=True))
        Xsub = [W.T @ X for X in manifolds]
        try:
            ok, *_ = check_data_separability_general(Xsub, np.asarray(draw_y(), float))
        except ValueError:
            ok = False
        sep.append(bool(ok))
    return float(np.mean(sep))


def simcap(
    manifolds,
    rng: np.random.Generator,
    *,
    ensemble: Ensemble | None = None,
    n_rep: int = 10,
    p_tol: float = 0.05,
    max_steps: int = 20,
) -> SimcapResult:
    """`α_sim = P / N_c` for a chosen dichotomy ensemble.

    ``manifolds[μ]``: ``(N, M_μ)``. Works at any β — this is the ground truth the
    GLUE core is validated against on the α channel (`02` §2a).
    """
    P, N = len(manifolds), manifolds[0].shape[0]
    ens = ensemble or Ensemble()
    draw_y = ens.sampler(P, rng)
    mans = [np.ascontiguousarray(X, dtype=np.float64) for X in manifolds]

    lo, hi = 2, N
    seen: dict[int, float] = {}

    def p_at(n: int) -> float:
        if n not in seen:
            seen[n] = _p_separable(mans, n, n_rep, rng, draw_y)
        return seen[n]

    if p_at(hi) < 0.5:
        raise RuntimeError(f"not separable even at N={hi}: p={seen[hi]:.2f}")

    for _ in range(max_steps):
        if hi - lo <= 1:
            break
        mid = (lo + hi) // 2
        p = p_at(mid)
        if abs(p - 0.5) < p_tol:
            lo, hi = mid, mid + 1
            p_at(hi)
            break
        if p < 0.5:
            lo = mid
        else:
            hi = mid

    N_vec = sorted(seen)
    p_vec = [seen[n] for n in N_vec]
    below = [n for n in N_vec if seen[n] < 0.5]
    above = [n for n in N_vec if seen[n] >= 0.5]
    if not below:
        N_c = float(min(above))
    else:
        b, a = max(below), min(above)
        N_c = float(np.interp(0.5, [seen[b], seen[a]], [b, a])) if a != b else float(a)

    return SimcapResult(
        alpha_sim=float(P / N_c),
        N_c=N_c,
        P=P,
        N=N,
        n_rep=n_rep,
        ensemble=ens.label,
        N_vec=[int(n) for n in N_vec],
        p_vec=p_vec,
    )
