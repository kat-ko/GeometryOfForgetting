"""I6 for the nested readout-path rank projection (`results/rank_precommit.json`)."""

from __future__ import annotations

import numpy as np

from src.models import MODULES, ScalingConfig, TwoModuleNet, paired_init, projection_rng

D, N = 40, 128


def _cfgs(gamma, N=N):
    return {m: ScalingConfig(N=N, d=D, gamma_0=gamma) for m in MODULES}


def _model(gamma, rank, seed=0, N=N):
    return TwoModuleNet.init(
        _cfgs(gamma, N=N), paired_init(seed)["shape"],
        readout_rank=rank,
        projection_rng=projection_rng(seed),
    )


def test_W0_identical_across_gamma_at_fixed_rank():
    ref = _model(0.03, 4)
    other = _model(10.0, 4)
    for m in MODULES:
        assert np.array_equal(ref.W[m], other.W[m])


def test_W0_identical_across_rank_at_fixed_gamma():
    full = _model(1.0, N)
    tight = _model(1.0, 4)
    for m in MODULES:
        assert np.array_equal(full.W[m], tight.W[m])


def test_Q_identical_across_gamma_at_fixed_rank():
    a, b = _model(0.03, 4), _model(10.0, 4)
    for m in MODULES:
        assert np.array_equal(a.Q[m], b.Q[m])
        assert a.Q[m].shape == (4, N)


def test_Q_is_nested_across_rank():
    """r=4 is a row subset of r=16 of r=N. Rank does not redraw."""
    q4 = _model(1.0, 4)
    q16 = _model(1.0, 16)
    qN = _model(1.0, N)
    for m in MODULES:
        np.testing.assert_array_equal(q4.Q[m], q16.Q[m][:4])
        np.testing.assert_array_equal(q16.Q[m], qN.Q[m][:16])
        assert qN.Q[m].shape == (N, N)
        np.testing.assert_allclose(qN.Q[m] @ qN.Q[m].T, np.eye(N), atol=1e-12)


def test_Q_at_full_rank_is_not_identity():
    mdl = _model(1.0, N)
    for m in MODULES:
        assert not np.array_equal(mdl.Q[m], np.eye(N))


def test_registered_path_is_identity():
    mdl = TwoModuleNet.init(_cfgs(1.0), paired_init(0)["shape"])
    for m in MODULES:
        assert np.array_equal(mdl.Q[m], np.eye(N))


def test_Q_does_not_come_from_the_shape_stream():
    """A fifth spawn would have shifted W. Rank must leave the shape stream alone."""
    w_only = TwoModuleNet.init(_cfgs(1.0), paired_init(0)["shape"])
    ranked = _model(1.0, 4)
    for m in MODULES:
        assert np.array_equal(w_only.W[m], ranked.W[m])


def test_output_zero_at_init_with_rank():
    X = np.random.default_rng(1).standard_normal((16, D))
    assert np.all(_model(10.0, 4).forward(X) == 0.0)


def test_full_rank_matches_unprojected_forward_and_hidden_step():
    """Nested r=N is an orthogonal change of readout coordinates: W trajectory
    and f match the registered path; u lives in the Q-basis."""
    X = np.random.default_rng(2).standard_normal((8, D))
    y = np.sign(np.random.default_rng(3).standard_normal(8))
    plain = TwoModuleNet.init(_cfgs(1.0), paired_init(7)["shape"])
    full = _model(1.0, N, seed=7)
    np.testing.assert_array_equal(plain.forward(X), full.forward(X))
    plain.sgd_step(X, y)
    full.sgd_step(X, y)
    np.testing.assert_allclose(plain.forward(X), full.forward(X), atol=1e-12)
    for m in MODULES:
        np.testing.assert_allclose(plain.W[m], full.W[m], atol=1e-12)
        np.testing.assert_allclose(plain.u[m], full.Q[m].T @ full.u[m], atol=1e-12)


def test_projection_rows_are_orthonormal():
    Q = _model(1.0, 4).Q["A"]
    gram = Q @ Q.T
    np.testing.assert_allclose(gram, np.eye(4), atol=1e-12)
