"""Adam is an opt-in update; SGD path and I3 on core configs are unchanged."""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from src.models import MODULES, ScalingConfig, TwoModuleNet, paired_init
from src.pipeline import Phase1Spec
from src.train.loop import TrainConfig, train_xy

D, N = 40, 128


def _model(gamma=1.0, seed=0, lr0=0.01):
    cfg = {
        m: ScalingConfig(N=N, d=D, gamma_0=gamma, parameterization="mup",
                         lr_scaling="quadratic", lr0=lr0)
        for m in MODULES
    }
    return TwoModuleNet.init(cfg, paired_init(seed)["shape"])


def _batch(seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((16, D))
    y = np.sign(rng.standard_normal(16))
    y[y == 0] = 1.0
    return X, y


def test_core_train_config_is_sgd():
    cfg = Phase1Spec().train_config
    assert cfg.optimizer == "sgd"
    assert cfg.loss == "mse"
    assert TrainConfig().optimizer == "sgd"


def test_adam_step_has_no_weight_decay():
    src = inspect.getsource(TwoModuleNet.adam_step) + inspect.getsource(TwoModuleNet._adam_apply)
    assert "weight_decay" not in src
    assert "wd" not in src


def test_first_adam_step_is_sign_normalized_sgd():
    """Bias-corrected first step: Δ = lr · g / (|g| + ε), not lr · g."""
    X, y = _batch(1)
    mdl = _model(seed=2)
    gW, gu, _, _ = mdl.grads(X, y)
    probe = mdl.copy()
    loss = probe.adam_step(X, y, eps=1e-8)
    assert np.isfinite(loss)
    for m in MODULES:
        assert np.array_equal(probe.W[m], mdl.W[m])  # I5: gW = 0 at u=0
        g = gu[m]
        expected = mdl.u[m] - mdl.cfg[m].lr * g / (np.abs(g) + 1e-8)
        assert np.allclose(probe.u[m], expected)


def test_adam_first_step_moves_only_the_readout():
    X, y = _batch(2)
    mdl = _model()
    W0 = {m: mdl.W[m].copy() for m in MODULES}
    mdl.adam_step(X, y)
    for m in MODULES:
        assert np.array_equal(mdl.W[m], W0[m])
        assert not np.allclose(mdl.u[m], 0.0)


def test_adam_differs_from_sgd_after_two_steps():
    X, y = _batch(3)
    sgd = _model(seed=4)
    adam = _model(seed=4)
    sgd.sgd_step(X, y)
    sgd.sgd_step(X, y)
    adam.adam_step(X, y)
    adam.adam_step(X, y)
    assert not np.allclose(sgd.u["A"], adam.u["A"])
    assert not np.allclose(sgd.W["A"], adam.W["A"])


def test_sgd_default_does_not_allocate_adam_state():
    X, y = _batch(4)
    mdl = _model()
    mdl.sgd_step(X, y)
    assert mdl._adam_t == 0
    assert mdl._adam_m == {}


def test_train_xy_adam_reaches_on_a_toy_batch():
    rng = np.random.default_rng(5)
    mdl = _model(lr0=0.1)
    X, y = _batch(5)
    cfg = TrainConfig(steps_per_task=200, target_loss=0.05, optimizer="adam",
                      record_every=50)
    rec = train_xy(mdl, X, y, cfg, rng)
    assert rec.converged
    assert rec.final_loss <= 0.05


def test_unknown_optimizer_is_refused():
    rng = np.random.default_rng(6)
    mdl = _model()
    X, y = _batch(6)
    cfg = TrainConfig(steps_per_task=2, optimizer="rmsprop")
    with pytest.raises(ValueError, match="optimizer"):
        train_xy(mdl, X, y, cfg, rng)


def test_step_dispatch_default_matches_sgd_step():
    X, y = _batch(7)
    a = _model(seed=8)
    b = _model(seed=8)
    a.sgd_step(X, y)
    b.step(X, y)
    for m in MODULES:
        assert np.array_equal(a.W[m], b.W[m])
        assert np.array_equal(a.u[m], b.u[m])
