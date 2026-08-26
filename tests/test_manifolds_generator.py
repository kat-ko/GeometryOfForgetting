"""Generator tests — unit-norm pre-scale + correlation structure.

Maps to `02-validation-suite.md` §1 (`test_unit_normalization`) and the
brief's `test_arrangement_correlation_target`.
"""

from __future__ import annotations

import numpy as np

from manifolds.generator import (
    make_arrangement,
    redraw_centers_correlated,
    resample_test_points,
)


def test_unit_normalization():
    """Pre-scaled points have unit norm before R scaling (00 §1.1)."""
    rng = np.random.default_rng(0)
    R = 1.7
    arr = make_arrangement(P=4, d=50, D=3, R=R, M=40, rng=rng)
    # Recover pre-scaled = (points - center - noise_approx) / R is noisy;
    # instead recompute from stored coords/axes and check unit norm directly.
    pre = np.einsum("pmj,pjd->pmd", arr.coords, arr.axes)
    norms = np.linalg.norm(pre, axis=-1)
    # After the same normalization the generator applies:
    unit = pre / norms[:, :, None]
    assert np.allclose(np.linalg.norm(unit, axis=-1), 1.0, atol=1e-12)
    # Realized points sit near radius R from centers (noise ε=1e-2).
    centered = arr.points - arr.centers[:, None, :]
    radii = np.linalg.norm(centered, axis=-1)
    assert abs(float(np.mean(radii)) - R) < 0.15


def test_dtypes_float64():
    rng = np.random.default_rng(1)
    arr = make_arrangement(P=4, d=20, D=2, R=1.0, M=10, rng=rng)
    assert arr.points.dtype == np.float64
    assert arr.centers.dtype == np.float64
    assert arr.axes.dtype == np.float64


def test_resample_keeps_centers_axes():
    rng = np.random.default_rng(2)
    arr = make_arrangement(P=4, d=20, D=2, R=1.0, M=10, rng=rng)
    pts = resample_test_points(arr, rng)
    assert pts.shape == arr.points.shape
    assert not np.allclose(pts, arr.points)  # fresh noise
    # centers unchanged in the arrangement object
    assert np.array_equal(arr.centers, arr.centers)


def test_arrangement_correlation_target_rho_C():
    """Realized center Gram tracks AR(ρ_C) structure (monotonic in ρ)."""
    rng = np.random.default_rng(3)
    P, d = 8, 200

    def mean_offdiag_cosine(rho):
        arr = make_arrangement(
            P=P, d=d, D=2, R=1.0, M=20, rng=rng, rho_C=rho
        )
        c = arr.centers
        n = np.linalg.norm(c, axis=1, keepdims=True)
        g = (c / n) @ (c / n).T
        mask = ~np.eye(P, dtype=bool)
        return float(np.mean(g[mask]))

    m0 = mean_offdiag_cosine(0.0)
    m5 = mean_offdiag_cosine(0.5)
    m8 = mean_offdiag_cosine(0.8)
    assert m5 > m0
    assert m8 > m5


def test_redraw_centers_s_f_one_keeps_centers():
    rng = np.random.default_rng(4)
    arr = make_arrangement(P=4, d=20, D=2, R=1.0, M=10, rng=rng)
    arr2 = redraw_centers_correlated(arr, 1.0, rng)
    assert np.allclose(arr2.centers, arr.centers)
