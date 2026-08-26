"""`02` §4 — the alignment knob's *mechanics*.

Gate 2 itself (`test_alignment_moves_initial_capacity`) is an empirical question
and lives in Phase 0, not here. These tests check that the geodesic is a geodesic
and that the reassembly preserves what it claims to preserve — if those fail, a
Gate-2 failure would be uninterpretable.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.manifolds import generator
from src.models.alignment import (
    aligned_init,
    center_subspace,
    grassmann_geodesic,
    principal_angles,
)

D_AMB, N, P, D_INT = 40, 60, 8, 3
A_GRID = np.linspace(0.0, 1.0, 11)


def _setup(seed=0):
    rng = np.random.default_rng(seed)
    arr = generator.make_arrangement(P, D_AMB, D_INT, 1.0, 30, rng)
    U_C, k = center_subspace(arr.centers, D=D_INT)
    return rng.standard_normal((N, D_AMB)), U_C, k


def test_alignment_preserves_norm_and_rank():
    W, U_C, _ = _setup()
    f0, r0 = np.linalg.norm(W), np.linalg.matrix_rank(W)
    for a in A_GRID:
        Wa = aligned_init(W, U_C, a)
        assert np.linalg.norm(Wa) == pytest.approx(f0, rel=1e-10)
        assert np.linalg.matrix_rank(Wa) == r0


def _subspace_distance(A: np.ndarray, B: np.ndarray) -> float:
    """‖P_A − P_B‖_F between orthonormal bases.

    Used instead of the largest principal angle for the endpoint checks: `arccos`
    is ill-conditioned near 1, so identical subspaces come out at ~1.5e-8 (√eps)
    however exact the geodesic is. The projector distance is well-conditioned and
    reaches 1e-14 here, so it actually tests the implementation.
    """
    return float(np.linalg.norm(A @ A.T - B @ B.T))


def test_grassmann_endpoints():
    W, U_C, k = _setup()
    V = np.linalg.svd(W, full_matrices=False)[2].T
    Y = V[:, :k]

    # a = 0 spans the original subspace, a = 1 spans the center subspace
    assert _subspace_distance(grassmann_geodesic(Y, U_C, 0.0), Y) < 1e-10
    assert _subspace_distance(grassmann_geodesic(Y, U_C, 1.0), U_C) < 1e-10


def test_principal_angles_monotone_in_a():
    W, U_C, k = _setup()
    V = np.linalg.svd(W, full_matrices=False)[2].T
    Y = V[:, :k]
    to_target = [
        float(np.max(principal_angles(grassmann_geodesic(Y, U_C, a), U_C))) for a in A_GRID
    ]
    assert all(b <= a + 1e-9 for a, b in zip(to_target, to_target[1:]))
    assert to_target[0] > to_target[-1] + 1e-3


def test_geodesic_is_not_linear_interpolation():
    """The distinction the Grassmann rule in AGENTS.md exists to enforce."""
    W, U_C, k = _setup()
    V = np.linalg.svd(W, full_matrices=False)[2].T
    Y = V[:, :k]
    geo = grassmann_geodesic(Y, U_C, 0.5)
    lin, _ = np.linalg.qr(0.5 * Y + 0.5 * U_C)
    assert float(np.max(principal_angles(geo, lin))) > 1e-3


def test_alignment_output_is_float64_and_deterministic():
    W, U_C, _ = _setup()
    assert aligned_init(W, U_C, 0.4).dtype == np.float64
    assert np.array_equal(aligned_init(W, U_C, 0.4), aligned_init(W, U_C, 0.4))


def test_center_subspace_k_rule():
    _, U_C, k = _setup()
    assert k == min(P - 1, 2 * D_INT)
    np.testing.assert_allclose(U_C.T @ U_C, np.eye(k), atol=1e-10)
