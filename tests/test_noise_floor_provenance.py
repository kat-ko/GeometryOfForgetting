"""The registered noise floors must still match the artifact they were read from.

`NOISE_FLOOR_CV` in `src/analysis/timewarp.py` is a hardcoded copy of the per-channel dispersion
measured by `run_cost_model.py` at the registered operating point, n_t=200. Every floor-denominated
number in the paper divides by one of these four constants, and nothing connected them to
`results/cost_model.json` -- regenerating the cost model would not update the constants and no check
compared them. That is the whole of the risk: not that they are wrong today, but that they could
become wrong without anything saying so.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.analysis.timewarp import NOISE_FLOOR_CV

ROOT = Path(__file__).resolve().parents[1]
COST_MODEL = ROOT / "results" / "cost_model.json"
REGISTERED_N_T = 200


def _measured_row() -> dict:
    rows = json.loads(COST_MODEL.read_text())["noise_vs_n_t"]
    matching = [r for r in rows if r["n_t"] == REGISTERED_N_T]
    assert matching, f"cost_model.json has no n_t={REGISTERED_N_T} row to denominate against"
    return matching[0]


@pytest.mark.parametrize("channel", sorted(NOISE_FLOOR_CV))
def test_floor_matches_cost_model(channel: str) -> None:
    row = _measured_row()
    assert channel in row, (
        f"{channel!r} is used as a noise floor but cost_model.json does not measure it")
    measured = row[channel]["cv_pct"] / 100.0
    hardcoded = NOISE_FLOOR_CV[channel]
    # The constants are the measured values rounded to four decimals, so the tolerance is the
    # rounding and nothing more.
    assert hardcoded == pytest.approx(measured, abs=5e-5), (
        f"{channel}: constant {hardcoded} no longer matches the measured "
        f"{measured:.6f} at n_t={REGISTERED_N_T}. Every floor-denominated number in the paper "
        f"divides by this; update the constant and re-run the inventory.")


def test_no_floor_is_silently_missing() -> None:
    """A channel dropped from the artifact should fail loudly rather than fall back to a default."""
    row = _measured_row()
    missing = sorted(set(NOISE_FLOOR_CV) - set(row))
    assert not missing, f"floors with no measurement behind them: {missing}"
