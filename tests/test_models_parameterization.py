"""`02` §3 — parameterization orthogonality, the gate for the (a, γ) design."""

from __future__ import annotations

import numpy as np
import pytest

from src.models import MODULES, ScalingConfig, TwoModuleNet, paired_init
from src.models.parameterization import N_BASE

D, N = 40, 128


def _cfgs(gamma, N=N, parameterization="mup", lr_scaling="quadratic", lr0=0.01):
    return {
        m: ScalingConfig(N=N, d=D, gamma_0=gamma, parameterization=parameterization,
                         lr_scaling=lr_scaling, lr0=lr0)
        for m in MODULES
    }


def _model(gamma, seed=0, **kw):
    return TwoModuleNet.init(_cfgs(gamma, **kw), paired_init(seed)["shape"])


def test_hidden_init_identical_across_gamma():
    """I6 — γ must not touch the hidden weights, or γ and `a` are confounded."""
    ref = _model(0.01)
    for gamma in (0.1, 1.0, 10.0):
        other = _model(gamma)
        for m in MODULES:
            assert np.array_equal(ref.W[m], other.W[m])


def test_output_zero_at_init():
    """I5 — zero-init readout, so `f(x; θ₀) = 0` exactly for every γ."""
    X = np.random.default_rng(1).standard_normal((16, D))
    for gamma in (0.01, 1.0, 10.0):
        assert np.all(_model(gamma).forward(X) == 0.0)


def test_base_width_equivalence():
    """`02` §3 — resolves the base-width constant of `00` §4.2.

    Checked on the forward pass **and** the first gradient step, as specified.
    """
    X = np.random.default_rng(2).standard_normal((32, D))
    y = np.sign(np.random.default_rng(3).standard_normal(32))

    ntp = _model(1.0, N=N_BASE, parameterization="ntp")
    mup = _model(1.0, N=N_BASE, parameterization="mup")

    assert mup.cfg["A"].gamma_eff == pytest.approx(1.0, abs=0.0)
    assert mup.cfg["A"].output_scale == pytest.approx(ntp.cfg["A"].output_scale, rel=0, abs=0)
    assert mup.cfg["A"].lr == pytest.approx(ntp.cfg["A"].lr, rel=0, abs=0)

    np.testing.assert_array_equal(ntp.forward(X), mup.forward(X))
    ntp.sgd_step(X, y)
    mup.sgd_step(X, y)
    for m in MODULES:
        np.testing.assert_allclose(ntp.W[m], mup.W[m], rtol=0, atol=0)
        np.testing.assert_allclose(ntp.u[m], mup.u[m], rtol=0, atol=0)


def test_base_width_equivalence_fails_away_from_base_width():
    """The equivalence is specific to N = N_base; otherwise the two differ."""
    other_N = 4 * N_BASE
    ntp = ScalingConfig(N=other_N, d=D, gamma_0=1.0, parameterization="ntp")
    mup = ScalingConfig(N=other_N, d=D, gamma_0=1.0, parameterization="mup")
    assert mup.output_scale != ntp.output_scale
    assert mup.lr == pytest.approx(ntp.lr * other_N / N_BASE)


def test_lr_scales_with_gamma():
    for gamma in (0.1, 1.0, 4.0):
        c = ScalingConfig(N=N, d=D, gamma_0=gamma, lr0=0.01)
        assert c.lr == pytest.approx(0.01 * gamma**2 * N / N_BASE)


def test_corrected_lr_uses_gamma_two_over_L_above_one():
    """`00` §4.3 — Atanasov: `η* ∝ γ^(2/L)` for γ ≫ 1, `γ²` below."""
    below = ScalingConfig(N=N, d=D, gamma_0=0.5, lr_scaling="corrected", lr0=0.01)
    above = ScalingConfig(N=N, d=D, gamma_0=9.0, lr_scaling="corrected", lr0=0.01, depth=2)
    assert below.lr == pytest.approx(0.01 * 0.5**2 * N / N_BASE)
    assert above.lr == pytest.approx(0.01 * 9.0 ** (2 / 2) * N / N_BASE)
    assert above.lr != pytest.approx(0.01 * 9.0**2 * N / N_BASE)


def test_lr_applies_to_both_W_and_u():
    """Changing one without the other changes contribution magnitude, not richness."""
    X = np.random.default_rng(4).standard_normal((16, D))
    y = np.sign(np.random.default_rng(5).standard_normal(16))
    slow, fast = _model(1.0, lr0=0.01), _model(1.0, lr0=0.02)
    for mdl in (slow, fast):
        mdl.sgd_step(X, y)          # only u moves (dL/dW ∝ u = 0)
        mdl.sgd_step(X, y)          # now both move
    for m in MODULES:
        assert not np.allclose(slow.u[m], fast.u[m])
        assert not np.allclose(slow.W[m], fast.W[m])


def test_gradients_match_finite_differences():
    """The whole numpy-with-analytic-gradients choice rests on this."""
    rng = np.random.default_rng(6)
    X = rng.standard_normal((8, D))
    y = np.sign(rng.standard_normal(8))
    mdl = _model(1.0, seed=7)
    mdl.u["A"] = rng.standard_normal(N) * 0.1   # move off the u = 0 point
    mdl.u["B"] = rng.standard_normal(N) * 0.1

    def loss_of(model):
        return 0.5 * np.mean((model.forward(X) - y) ** 2)

    gW, gu, _ = mdl.grads(X, y)
    eps = 1e-6
    for m in MODULES:
        for _ in range(3):
            i, j = rng.integers(N), rng.integers(D)
            probe = mdl.copy()
            probe.W[m][i, j] += eps
            hi = loss_of(probe)
            probe.W[m][i, j] -= 2 * eps
            numeric = (hi - loss_of(probe)) / (2 * eps)
            assert gW[m][i, j] == pytest.approx(numeric, rel=1e-5, abs=1e-9)

            i = rng.integers(N)
            probe = mdl.copy()
            probe.u[m][i] += eps
            hi = loss_of(probe)
            probe.u[m][i] -= 2 * eps
            numeric = (hi - loss_of(probe)) / (2 * eps)
            assert gu[m][i] == pytest.approx(numeric, rel=1e-5, abs=1e-9)


def test_float64_and_paired_streams():
    mdl = _model(1.0)
    assert mdl.W["A"].dtype == np.float64 and mdl.u["A"].dtype == np.float64
    streams = paired_init(0)
    assert set(streams) == {"shape", "data", "stream", "probe"}
    # the shape stream must be reproducible and independent of the others
    assert np.array_equal(paired_init(0)["shape"].standard_normal(5),
                          streams["shape"].standard_normal(5))
