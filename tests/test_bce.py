"""BCE is an opt-in loss; MSE path and I5/I6 are unchanged."""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.models import MODULES, ScalingConfig, TwoModuleNet, paired_init
from src.models.network import bce_with_logits
from src.train.loop import TrainConfig, train_xy

D, N = 40, 128


def _model(gamma=1.0, seed=0, lr0=0.01):
    cfg = {
        m: ScalingConfig(N=N, d=D, gamma_0=gamma, parameterization="mup",
                         lr_scaling="quadratic", lr0=lr0)
        for m in MODULES
    }
    return TwoModuleNet.init(cfg, paired_init(seed)["shape"])


def test_bce_at_init_is_log_two():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((16, D))
    y = np.sign(rng.standard_normal(16))
    y[y == 0] = 1.0
    mdl = _model()
    assert np.all(mdl.forward(X) == 0.0)
    assert bce_with_logits(mdl.forward(X), y) == pytest.approx(math.log(2.0))
    _, _, loss, _ = mdl.grads(X, y, loss="bce")
    assert loss == pytest.approx(math.log(2.0))


def test_bce_gradients_match_finite_differences():
    rng = np.random.default_rng(6)
    X = rng.standard_normal((8, D))
    y = np.sign(rng.standard_normal(8))
    y[y == 0] = 1.0
    mdl = _model(seed=7)
    mdl.u["A"] = rng.standard_normal(N) * 0.1
    mdl.u["B"] = rng.standard_normal(N) * 0.1

    gW, gu, _, _ = mdl.grads(X, y, loss="bce")
    eps = 1e-6
    for m in MODULES:
        for _ in range(3):
            i, j = rng.integers(N), rng.integers(D)
            probe = mdl.copy()
            probe.W[m][i, j] += eps
            hi = bce_with_logits(probe.forward(X), y)
            probe.W[m][i, j] -= 2 * eps
            numeric = (hi - bce_with_logits(probe.forward(X), y)) / (2 * eps)
            assert gW[m][i, j] == pytest.approx(numeric, rel=1e-5, abs=1e-8)

            i = rng.integers(N)
            probe = mdl.copy()
            probe.u[m][i] += eps
            hi = bce_with_logits(probe.forward(X), y)
            probe.u[m][i] -= 2 * eps
            numeric = (hi - bce_with_logits(probe.forward(X), y)) / (2 * eps)
            assert gu[m][i] == pytest.approx(numeric, rel=1e-5, abs=1e-8)


def test_bce_first_step_moves_only_the_readout():
    rng = np.random.default_rng(2)
    X = rng.standard_normal((16, D))
    y = np.sign(rng.standard_normal(16))
    y[y == 0] = 1.0
    mdl = _model()
    W0 = {m: mdl.W[m].copy() for m in MODULES}
    mdl.sgd_step(X, y, loss="bce")
    for m in MODULES:
        assert np.array_equal(mdl.W[m], W0[m])
        assert not np.allclose(mdl.u[m], 0.0)


def test_bce_refuses_worst_of_k():
    rng = np.random.default_rng(3)
    mdl = _model()
    X = rng.standard_normal((8, D))
    y = np.ones(8)
    cfg = TrainConfig(stopping="worst_of_k", loss="bce", steps_per_task=2)
    with pytest.raises(ValueError, match="worst_of_k"):
        train_xy(mdl, X, y, cfg, rng)


def test_mse_grads_default_is_unchanged():
    rng = np.random.default_rng(6)
    X = rng.standard_normal((8, D))
    y = np.sign(rng.standard_normal(8))
    mdl = _model(seed=7)
    mdl.u["A"] = rng.standard_normal(N) * 0.1
    mdl.u["B"] = rng.standard_normal(N) * 0.1
    a = mdl.grads(X, y)
    b = mdl.grads(X, y, loss="mse")
    for m in MODULES:
        assert np.allclose(a[0][m], b[0][m])
        assert np.allclose(a[1][m], b[1][m])
    assert a[2] == pytest.approx(b[2])
