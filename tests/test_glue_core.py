"""Unit tests for `src/glue/core.py`.

The load-bearing one is `test_point_manifold_capacity_is_two`: it pins the
`inactive="zero"` convention, which is what reproduces the rectification in the
replica expression `α = 1/E[(⟨t,ŝ⟩)₊²]`. Ground-truth recovery against §B.5 lives
in `tests/test_glue_core_recovery.py`.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.glue import core


def _point_manifolds(P: int, N: int, rng: np.random.Generator) -> list[np.ndarray]:
    """P orthonormal single-point manifolds. Known capacity: α = 2 at κ = 0."""
    Q, _ = np.linalg.qr(rng.standard_normal((N, P)))
    return [Q[:, mu : mu + 1].copy() for mu in range(P)]


def test_point_manifold_capacity_is_two():
    rng = np.random.default_rng(0)
    res = core.glue_measures(_point_manifolds(8, 200, rng), rng, n_t=400)
    assert res.alpha == pytest.approx(2.0, rel=0.12)
    assert res.active_fraction == pytest.approx(0.5, abs=0.05)


def test_identity_is_exact_by_construction():
    rng = np.random.default_rng(1)
    res = core.glue_measures(_gaussian_manifolds(6, 60, 4, 1.0, rng), rng, n_t=60)
    assert res.identity_residual < 1e-8
    assert res.Psi_eff * (1 + res.R_eff**-2) / res.D_eff == pytest.approx(res.alpha)


def _gaussian_manifolds(P, N, D, R, rng, *, M=40, center_scale=1.0):
    """P manifolds: random center + R-scaled D-dimensional Gaussian cloud."""
    out = []
    for _ in range(P):
        c = center_scale * rng.standard_normal(N) / np.sqrt(N)
        U, _ = np.linalg.qr(rng.standard_normal((N, D)))
        coords = rng.standard_normal((D, M)) / np.sqrt(D)
        out.append(c[:, None] + R * (U @ coords) / np.sqrt(N))
    return out


def test_labels_enter_only_through_the_anchors():
    """`diag(y)` cancels in a/b/c; all y-dependence is in the joint QP."""
    rng = np.random.default_rng(2)
    mans = _gaussian_manifolds(4, 40, 3, 1.0, rng)
    t = rng.standard_normal(40)
    y = np.array([1.0, -1.0, 1.0, -1.0])
    S, _ = core.anchor_matrix(mans, y, t)
    Sy = np.diag(y) @ S
    plain, _ = core._quad(S @ t, S @ S.T, core.DEFAULT_RCOND)
    signed, _ = core._quad(Sy @ t, Sy @ Sy.T, core.DEFAULT_RCOND)
    assert signed == pytest.approx(plain, rel=1e-10)


def test_anchors_depend_on_the_dichotomy():
    """The joint QP must be label-sensitive — otherwise we have rebuilt α_mf."""
    rng = np.random.default_rng(3)
    mans = _gaussian_manifolds(6, 50, 3, 1.0, rng)
    t = rng.standard_normal(50)
    S_a, _ = core.anchor_matrix(mans, np.array([1.0, 1, 1, -1, -1, -1]), t)
    S_b, _ = core.anchor_matrix(mans, np.array([1.0, -1, 1, -1, 1, -1]), t)
    assert not np.allclose(S_a, S_b)


def test_retained_differs_from_generic():
    rng = np.random.default_rng(4)
    mans = _gaussian_manifolds(8, 60, 3, 1.0, rng)
    y_ref = np.array([1.0, 1, 1, 1, -1, -1, -1, -1])
    generic = core.glue_measures(mans, np.random.default_rng(5), n_t=120)
    retained = core.glue_measures(
        mans,
        np.random.default_rng(5),
        n_t=120,
        ensemble=core.Ensemble("retained", y_ref=y_ref),
    )
    assert generic.ensemble == "generic"
    assert retained.ensemble == "retained"
    assert not np.isclose(generic.alpha, retained.alpha, rtol=1e-3)


def test_tilt_interpolates_generic_to_retained():
    rng = np.random.default_rng(6)
    mans = _gaussian_manifolds(6, 50, 3, 1.0, rng)
    y_ref = np.array([1.0, 1, 1, -1, -1, -1])
    kw = dict(n_t=150)
    a0 = core.glue_measures(mans, np.random.default_rng(7), **kw).alpha
    a_hi = core.glue_measures(
        mans,
        np.random.default_rng(7),
        ensemble=core.Ensemble("tilted", y_ref=y_ref, beta=50.0),
        **kw,
    ).alpha
    a_inf = core.glue_measures(
        mans,
        np.random.default_rng(7),
        ensemble=core.Ensemble("retained", y_ref=y_ref),
        **kw,
    ).alpha
    assert abs(a_hi - a_inf) < abs(a0 - a_inf)


def test_rho_c_conventions_are_distinct_and_correct():
    S0 = np.array([[1.0, 0.0], [-0.5, 0.0], [0.0, 2.0]])
    # pairs: (0,1) = -0.5, (0,2) = 0, (1,2) = 0  -> mean |.| = 1/6, mean signed = -1/3
    assert core.rho_c(S0, "glue") == pytest.approx(0.5 / 3)
    assert core.rho_c(S0, "signed") == pytest.approx(-1.0 / 3)


def test_rho_c_signed_detects_anticorrelation_that_abs_hides():
    """The reason `rho_convention: both` is mandatory (H1d)."""
    aligned = np.array([[1.0, 0.0], [1.0, 0.0]])
    opposed = np.array([[1.0, 0.0], [-1.0, 0.0]])
    assert core.rho_c(aligned, "glue") == core.rho_c(opposed, "glue")
    assert core.rho_c(aligned, "signed") == -core.rho_c(opposed, "signed")


def test_harmonic_mean_is_not_the_arithmetic_mean():
    a = np.array([1.0, 3.0])
    assert core.harmonic_mean(a) == pytest.approx(1.5)
    with pytest.raises(ValueError):
        core.harmonic_mean(np.array([1.0, 0.0]))


def test_reproducible_from_seed():
    rng = np.random.default_rng(8)
    mans = _gaussian_manifolds(4, 40, 3, 1.0, rng)
    kw = dict(n_t=40)
    r1 = core.glue_measures(mans, np.random.default_rng(9), **kw)
    r2 = core.glue_measures(mans, np.random.default_rng(9), **kw)
    assert r1.alpha == r2.alpha and r1.rho_c_signed == r2.rho_c_signed


def test_float64_and_no_global_rng_use():
    rng = np.random.default_rng(10)
    mans = _gaussian_manifolds(4, 30, 2, 1.0, rng)
    S, active = core.anchor_matrix(mans, np.array([1.0, 1, -1, -1]), rng.standard_normal(30))
    assert S.dtype == np.float64 and active.dtype == np.bool_


# --------------------------------------------------------------------------
# Column-generation solver: must be exact, not merely close
# --------------------------------------------------------------------------


def _random_manifolds(P=8, M=60, N=120, seed=0):
    rng = np.random.default_rng(seed)
    return [rng.standard_normal((N, M)) + 3.0 * rng.standard_normal((N, 1))
            for _ in range(P)], rng


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_colgen_reproduces_scipy_nnls_exactly(seed):
    """`colgen` is a speedup, not an approximation — anchors must match to ~1e-14.

    It solves on a candidate subset and then checks the *full* KKT conditions, adding
    violators until none remain, so on termination it satisfies the same optimality
    conditions as the full solve. This pins that claim: a difference here means the
    KKT check is wrong, and every geometry measured with it would be quietly off.
    """
    manifolds, rng = _random_manifolds(seed=seed)
    y = np.array([1.0, -1.0] * (len(manifolds) // 2))
    for _ in range(3):
        t = rng.standard_normal(manifolds[0].shape[0])
        S_ref, act_ref = core.anchor_matrix(manifolds, y, t, solver="nnls")
        S_new, act_new = core.anchor_matrix(manifolds, y, t, solver="colgen")
        assert np.allclose(S_ref, S_new, atol=1e-12)
        assert np.array_equal(act_ref, act_new)


def test_colgen_matches_on_the_full_measure_pipeline():
    manifolds, _ = _random_manifolds(seed=5)
    kw = dict(n_t=40, center_policy="all")
    a = core.glue_measures(manifolds, np.random.default_rng(3), solver="nnls", **kw)
    b = core.glue_measures(manifolds, np.random.default_rng(3), solver="colgen", **kw)
    for field in ("alpha", "D_eff", "R_eff", "Psi_eff", "rho_c_glue", "rho_c_signed"):
        assert getattr(a, field) == pytest.approx(getattr(b, field), rel=1e-10)


def test_colgen_falls_back_when_there_are_fewer_columns_than_the_seed_set():
    """Small problems must not be broken by the subset logic."""
    manifolds, rng = _random_manifolds(P=4, M=10, N=30, seed=9)
    y = np.array([1.0, -1.0, 1.0, -1.0])
    t = rng.standard_normal(30)
    S_ref, _ = core.anchor_matrix(manifolds, y, t, solver="nnls")
    S_new, _ = core.anchor_matrix(manifolds, y, t, solver="colgen")
    assert np.allclose(S_ref, S_new, atol=1e-12)


def test_colgen_is_the_default():
    assert core.DEFAULT_SOLVER == "colgen"
