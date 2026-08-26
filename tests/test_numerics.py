"""A clip must not be able to absorb a real violation as if it were rounding.

The ρ_c bug's mechanism: `np.clip(rho, 0, 0.995)` written for float safety, fed values
up to 1.49, returning plausible numbers from invalid input.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.numerics import clip_to_noise


def test_float_error_is_absorbed_silently():
    """The legitimate case, which must stay ergonomic or it will be worked around."""
    s = np.array([1.0 + 2e-16, 0.5, -1.0 - 1e-16])
    out = clip_to_noise(s, -1.0, 1.0, what="cosine")
    assert out.max() <= 1.0 and out.min() >= -1.0
    assert np.allclose(out, [1.0, 0.5, -1.0])


def test_a_real_violation_raises_instead_of_being_clipped():
    with pytest.raises(ValueError, match="beyond the"):
        clip_to_noise(np.array([1.487]), 0.0, 0.995, what="rho_c")


def test_the_rho_c_bug_would_now_be_caught_at_the_clip():
    """Had `radius_from_rho` used this, the 15-38 shares could not have been produced."""
    measured = np.array([0.196, 1.235, 1.487])  # from the pilot representations
    with pytest.raises(ValueError, match="rho_c"):
        clip_to_noise(measured, 0.0, 0.995, what="rho_c")


def test_one_sided_bounds():
    assert clip_to_noise(-1e-17, 0.0, None, what="dual") == 0.0
    with pytest.raises(ValueError):
        clip_to_noise(-0.5, 0.0, None, what="dual")
    assert clip_to_noise(1e6, None, None, what="unbounded") == 1e6


def test_tolerance_is_the_stated_premise():
    clip_to_noise(1.05, 0.0, 1.0, atol=0.1, what="loose")
    with pytest.raises(ValueError):
        clip_to_noise(1.05, 0.0, 1.0, atol=0.01, what="tight")


def test_scalars_stay_scalars():
    out = clip_to_noise(0.5, 0.0, 1.0, what="x")
    assert isinstance(out, float) and out == 0.5


def test_principal_angles_rejects_non_orthonormal_input():
    """The guard reaching the call site it protects."""
    from src.models.alignment import principal_angles

    rng = np.random.default_rng(0)
    Y = np.linalg.qr(rng.standard_normal((20, 4)))[0]
    assert np.all(np.isfinite(principal_angles(Y, Y)))
    with pytest.raises(ValueError, match="principal angle"):
        principal_angles(Y * 3.0, Y)  # not orthonormal: cosines exceed 1
