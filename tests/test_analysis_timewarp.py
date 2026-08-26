"""The time-warp machinery must find a warp when one exists and not otherwise.

`00` §12's consequence is severe — it would restrict every heterogeneity contrast
in the paper — so the test needs to be trustworthy in both directions before its
verdict is believed.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.analysis import timewarp
from src.analysis.trajectories import CHANNELS, GeometryTrajectory


def _traj(gamma, steps, rate=1.0, offset=0.0, noise=0.0, seed=0):
    """A synthetic trajectory: channels are smooth functions of elapsed `rate·step`.

    Phase is `log(1 + rate·(s−1))` so that every rate starts at the same geometry
    at `s = 1`, which is what the paired-init design guarantees in the real runs.
    """
    rng = np.random.default_rng(seed)
    s = np.asarray(steps, dtype=float)
    phase = np.log1p(rate * (s - 1.0))
    chans = {}
    for k, c in enumerate(CHANNELS):
        base = 1.0 + 0.3 * (k + 1) + 0.25 * np.tanh(phase - 3.0) + offset
        chans[c] = base * np.exp(noise * rng.standard_normal(len(s)))
    return GeometryTrajectory(gamma=gamma, seed=seed, steps=s,
                              loss=np.exp(-0.5 * phase), channels=chans)


def _traj_shifted(gamma, steps, rate=1.0, noise=0.0, seed=0):
    """An exact rate rescaling: an identical shape shifted by `log(rate)` in log-step.

    Used where the point is whether the scale is recovered *exactly*. `_traj` uses
    the `log1p` form, which shares a start (realistic) but is not a pure shift, so
    the best single rate is a compromise rather than the generating value.
    """
    rng = np.random.default_rng(seed)
    s = np.asarray(steps, dtype=float)
    phase = np.log(s) + np.log(rate)
    chans = {}
    for k, c in enumerate(CHANNELS):
        base = 1.0 + 0.3 * (k + 1) + 0.25 * np.tanh(phase - 3.0)
        chans[c] = base * np.exp(noise * rng.standard_normal(len(s)))
    return GeometryTrajectory(gamma=gamma, seed=seed, steps=s,
                              loss=np.exp(-0.5 * phase), channels=chans)


STEPS = np.geomspace(1, 4000, 12)


def test_identical_trajectories_have_zero_residual():
    a = _traj(1.0, STEPS)
    r, per = timewarp.residual(a, a, 1.0)
    assert r == pytest.approx(0.0, abs=1e-9)
    assert all(v == pytest.approx(0.0, abs=1e-9) for v in per.values())


def test_pure_rate_rescaling_is_detected_and_undone():
    """The positive control: two runs that ARE one trajectory at two speeds."""
    a = _traj_shifted(1.0, STEPS, rate=1.0)
    b = _traj_shifted(10.0, STEPS, rate=8.0)
    # convention: `residual` compares a(s) against b(s / scale), so b being 8x
    # faster is recovered as scale ~ 8
    scale, res, _ = timewarp.best_rate_warp(a, b)
    assert scale == pytest.approx(8.0, rel=0.1)
    assert res < 1.0
    out = timewarp.time_reparameterization_test(a, b)
    assert not out["h3_testable"]
    assert "COINCIDE" in out["verdict"]


def test_genuinely_different_shapes_are_not_warped_away():
    """The negative control: no monotone warp should rescue a shape difference."""
    a = _traj(1.0, STEPS)
    b = _traj(10.0, STEPS)
    # invert one channel's trend, which no reparameterization of time can fix
    b.channels["D_eff"] = b.channels["D_eff"][::-1] * 1.6
    out = timewarp.time_reparameterization_test(a, b)
    assert out["h3_testable"]
    assert out["warps"][2]["residual_in_floors"] > 1.0


def test_residual_is_measured_in_noise_floors():
    """A displacement of exactly one floor on every channel must read as 1.0."""
    a = _traj(1.0, STEPS)
    b = _traj(1.0, STEPS)
    for c in CHANNELS:
        b.channels[c] = a.channels[c] * (1.0 + timewarp.NOISE_FLOOR_CV[c])
    r, per = timewarp.residual(a, b, 1.0)
    assert r == pytest.approx(1.0, rel=1e-6)
    assert all(v == pytest.approx(1.0, rel=1e-6) for v in per.values())


def test_dtw_is_no_worse_than_the_best_rate_warp():
    """DTW is the most generous monotone warp, so it must not lose to a subset."""
    a = _traj(1.0, STEPS, rate=1.0)
    b = _traj(10.0, STEPS, rate=5.0, noise=0.01, seed=3)
    _, rate_res, _ = timewarp.best_rate_warp(a, b)
    dtw_res, _ = timewarp.dtw_residual(a, b)
    assert dtw_res <= rate_res + 1e-9


def test_measured_warp_uses_loss_not_a_fit():
    a = _traj(1.0, STEPS, rate=1.0)
    b = _traj(10.0, STEPS, rate=8.0)
    c = timewarp.measured_warp_scale(a, b, target_loss=0.2)
    assert c > 0 and np.isfinite(c)
    # b is the faster run, so it needs fewer steps and the scale exceeds 1
    assert c > 1.0


def test_non_overlapping_supports_are_infinite_not_silently_zero():
    a = _traj(1.0, np.geomspace(1, 10, 5))
    b = _traj(1.0, np.geomspace(1e6, 1e7, 5))
    r, _ = timewarp.residual(a, b, 1.0)
    assert not np.isfinite(r)
