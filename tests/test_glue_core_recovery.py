"""Fast §B.5 ground-truth recovery checks for `src/glue/core.py` (`02` §1).

Reduced parameters so the suite stays quick; the full B.5 sweep
(P=2, M=200, N=1000, n_t=200, 3 seeds) is `scripts/run_glue_core_recovery.py`
and its output is `results/glue_core_recovery.json`.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.glue import core
from src.manifolds import generator

P, N, M, N_T = 2, 300, 80, 80
D_FIXED, R_FIXED = 4, 1.0


def _measure(*, D=D_FIXED, R=R_FIXED, rho_C=0.0, seed=0):
    rng = np.random.default_rng(seed)
    arr = generator.make_arrangement(P, N, D, R, M, rng, rho_C=rho_C)
    return core.glue_measures(
        core.from_arrangement(arr.points), np.random.default_rng(seed + 100), n_t=N_T
    )


def _is_increasing(xs, slack=0.0):
    return all(b >= a - slack for a, b in zip(xs, xs[1:]))


def test_D_eff_tracks_ground_truth_dimension():
    Ds = [2, 4, 8]
    res = [_measure(D=D) for D in Ds]
    assert _is_increasing([r.D_eff for r in res])
    assert all(abs(r.D_eff - D) / D < 0.35 for r, D in zip(res, Ds))
    assert not _is_increasing([r.alpha for r in res])  # capacity falls with D


def test_R_eff_tracks_ground_truth_radius_without_moving_D_eff():
    Rs = [0.8, 1.4, 2.0]
    res = [_measure(R=R) for R in Rs]
    assert _is_increasing([r.R_eff for r in res])
    assert all(abs(r.R_eff - R) / R < 0.35 for r, R in zip(res, Rs))
    D_effs = [r.D_eff for r in res]
    assert max(D_effs) - min(D_effs) < 0.35 * np.mean(D_effs)  # axes stay separable


def test_rho_c_glue_tracks_generated_center_correlation():
    rhos = [0.0, 0.4, 0.8]
    res = [_measure(rho_C=r) for r in rhos]
    assert _is_increasing([r.rho_c_glue for r in res])
    assert res[-1].rho_c_glue == pytest.approx(0.8, abs=0.2)


def test_psi_eff_in_unit_interval_for_the_generic_ensemble():
    """`Ψ_eff ∈ (0,1]` — but only under the label average. See the next test."""
    for r in (_measure(), _measure(D=8), _measure(R=2.0)):
        assert 0.0 < r.Psi_eff <= 1.0
        assert r.identity_residual < 1e-8


def test_fixed_y_psi_eff_may_exceed_one_when_centers_align_with_y():
    """The `(0,1]` range is a property of the label average, not of the estimator.

    `a` uses `(S_y S_yᵀ)†` while `c` uses `(S_{y,0}S_{y,0}ᵀ + S_{y,1}S_{y,1}ᵀ)†`; the two
    differ by center–axis cross-terms which vanish under `E_y` but survive at fixed `y`.
    Surviving is not sufficient, though — see the `sep=0` assertion below, where fixing
    `y` alone leaves `Ψ_eff` *under* the bound. What lifts it past 1 is the centers being
    organized *for* that dichotomy, so the quantity measures task-specific geometric
    organization rather than merely escaping a normalization.

    The grid relies on this: 21.6% of retained measurements exceed 1 (max 1.78), and
    reading that as a bug would have meant discarding the whole utility channel.
    Regression-guards the scoping in `00` §6.2.
    """
    rng = np.random.default_rng(20260812)
    P_, N_, M_ = 8, 120, 20
    y = np.sign(rng.standard_normal(P_))
    y[y == 0] = 1.0
    u = rng.standard_normal(N_)
    u /= np.linalg.norm(u)

    def measure(sep):
        # Centers at ±sep·u by the sign of y: the y dichotomy is one direction, a random
        # one has to cut two overlapping blobs.
        mans = [(sep * y[mu] * u)[:, None] + rng.standard_normal((N_, M_)) / np.sqrt(N_)
                for mu in range(P_)]
        kw = dict(n_t=80)
        return (core.glue_measures(mans, np.random.default_rng(1), **kw),
                core.glue_measures(mans, np.random.default_rng(1),
                                   ensemble=core.Ensemble(kind="retained", y_ref=y), **kw))

    generic_flat, retained_flat = measure(0.0)
    generic, retained = measure(2.0)

    # Fixing y is necessary but not sufficient: with centers unaligned it stays bounded.
    assert retained_flat.Psi_eff <= 1.0
    assert retained.Psi_eff > 1.0
    # The generic ensemble respects the bound either way — the tilt is what differs.
    assert generic_flat.Psi_eff <= 1.0
    assert generic.Psi_eff <= 1.0
    # The decomposition still closes, which is why α stays trustworthy above the bound.
    for r in (generic_flat, retained_flat, generic, retained):
        assert r.identity_residual < 1e-8
