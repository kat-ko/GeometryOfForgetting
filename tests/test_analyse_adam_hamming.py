"""Adam Hamming-only readings are named before numbers and mutually exclusive."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from analyse_adam_hamming import apply_readings  # noqa: E402


def _cell(mean: float, floors: float, lo: float, hi: float) -> dict:
    return {
        "n": 8,
        "mean": mean,
        "mean_floors": floors,
        "ci95": [lo, hi],
    }


def _gd_like_grid() -> dict:
    """γ=10 Hamming GD table, enough for the three Hamming-axis readings."""
    g = 10.0
    return {
        (g, "frozen", 1.0): _cell(+0.1321, +7.1, +0.1220, +0.1422),
        (g, "frozen", 0.75): _cell(-1.2642, -68.2, -1.4074, -1.1209),
        (g, "frozen", 0.5): _cell(-1.2805, -69.1, -1.3790, -1.1820),
        (g, "frozen", 0.25): _cell(-1.2772, -68.9, -1.4048, -1.1496),
        (g, "drift", 1.0): _cell(+0.1248, +6.7, +0.0950, +0.1546),
        (g, "drift", 0.75): _cell(-1.1356, -61.3, -1.2166, -1.0546),
        (g, "drift", 0.5): _cell(-1.3026, -70.3, -1.4046, -1.2007),
        (g, "drift", 0.25): _cell(-1.3816, -74.6, -1.4905, -1.2727),
        (g, "jump", 1.0): _cell(-1.0203, -55.1, -1.1056, -0.9350),
        (g, "jump", 0.75): _cell(-1.4250, -76.9, -1.4904, -1.3596),
        (g, "jump", 0.5): _cell(-1.4447, -78.0, -1.5056, -1.3838),
        (g, "jump", 0.25): _cell(-1.4386, -77.6, -1.5557, -1.3215),
    }


def test_gd_like_table_reads_hamming_reproduces():
    applied = apply_readings(_gd_like_grid())
    assert applied["finding3_recovered"] is True
    assert applied["primary_reading"] == "hamming_reproduces"
    assert applied["held"] == ["cliff", "drift_gradation", "crossover"]
    assert applied["failed"] == []
    assert "γ" in applied["cannot_speak_to"]


def test_split_reads_partial():
    grid = _gd_like_grid()
    # Destroy crossover only: drift worse than frozen at s_r=0.75, still graded.
    grid[(10.0, "drift", 0.75)] = _cell(-1.297, -70.0, -1.30, -1.28)
    grid[(10.0, "drift", 0.25)] = _cell(-1.382, -74.6, -1.50, -1.35)
    applied = apply_readings(grid)
    assert applied["primary_reading"] == "partial"
    assert applied["held"] == ["cliff", "drift_gradation"]
    assert applied["failed"] == ["crossover"]


def test_none_hold_reads_hamming_fails():
    grid = _gd_like_grid()
    for level in ("frozen", "jump"):
        grid[(10.0, level, 0.75)] = _cell(+0.10, +5.4, +0.09, +0.11)
        grid[(10.0, level, 0.25)] = _cell(-0.20, -10.8, -0.25, -0.15)
    grid[(10.0, "drift", 0.75)] = _cell(-1.20, -64.8, -1.25, -1.15)
    grid[(10.0, "drift", 0.25)] = _cell(-1.21, -65.3, -1.26, -1.16)
    applied = apply_readings(grid)
    assert applied["primary_reading"] == "hamming_fails"
    assert applied["held"] == []
    assert set(applied["failed"]) == {"cliff", "drift_gradation", "crossover"}


def test_recovery_failure_stops():
    grid = _gd_like_grid()
    grid[(10.0, "drift", 1.0)] = _cell(-0.10, -5.4, -0.12, -0.08)
    applied = apply_readings(grid)
    assert applied["primary_reading"] == "finding3_fails_to_recover"
    assert applied["finding3_recovered"] is False
