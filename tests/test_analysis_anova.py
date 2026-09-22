"""Pin the balanced two-way ANOVA against a constructed interaction."""

from __future__ import annotations

import numpy as np
import pytest

from src.analysis.anova import fill_balanced, two_way_balanced
from src.analysis.grid import REGISTERED_GAMMAS


def test_no_interaction_is_not_significant():
    rng = np.random.default_rng(0)
    data = np.zeros((3, 4, 8))
    for i in range(3):
        for j in range(4):
            data[i, j] = i + 0.5 * j + 0.05 * rng.standard_normal(8)
    r = two_way_balanced(data)
    assert r["p_interaction"] > 0.05
    assert r["p_row"] < 1e-6
    assert r["p_col"] < 1e-6


def test_constructed_interaction_is_detected():
    rng = np.random.default_rng(1)
    data = np.zeros((2, 2, 8))
    data[0, 0] = 0.0 + 0.02 * rng.standard_normal(8)
    data[0, 1] = 0.0 + 0.02 * rng.standard_normal(8)
    data[1, 0] = 0.0 + 0.02 * rng.standard_normal(8)
    data[1, 1] = 1.0 + 0.02 * rng.standard_normal(8)
    r = two_way_balanced(data)
    assert r["p_interaction"] < 1e-6


def test_fill_balanced_rejects_unequal_n():
    with pytest.raises(ValueError, match="unbalanced"):
        fill_balanced([1.0, 2.0, 3.0], [0, 0, 1], ["a", "b", "a"], [0, 1], ["a", "b"])


def test_registered_gamma_condition_interaction_df():
    """§7.1 is 6 γ × 4 conditions × n=8, not 4×4. F(15, 168) is that two-way."""
    n_g, n_c, n = len(REGISTERED_GAMMAS), 4, 8
    assert n_g == 6
    data = np.zeros((n_g, n_c, n))
    data[:] = np.arange(n_g * n_c * n, dtype=np.float64).reshape(n_g, n_c, n)
    r = two_way_balanced(data)
    assert r["df_row"] == 5
    assert r["df_col"] == 3
    assert r["df_interaction"] == 15
    assert r["df_error"] == 168
