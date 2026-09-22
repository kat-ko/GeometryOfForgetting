"""Margin recovery is the current-task y f, and the one-arm cell is the Hamming one-arm."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from measure_mse_stop_margin import (  # noqa: E402
    _spec_at, _spec_at_gamma, four_stream_ids, readout_margins,
)
from run_hamming import one_arm  # noqa: E402


def test_margin_is_mean_yf():
    y = np.array([1.0, 1.0, -1.0, -1.0])
    f = np.array([0.8, 0.4, -0.6, 0.0])
    got = readout_margins(f, y)
    assert got["mean"] == np.mean(y * f)
    assert got["p05"] == np.percentile(y * f, 5)
    assert got["p25"] == np.percentile(y * f, 25)
    assert got["sign_accuracy"] == 0.75
    assert abs(got["post_step_mse"] - 0.5 * np.mean((f - y) ** 2)) < 1e-12


def test_one_arm_cells_are_the_hamming_one_arm_at_both_gamma():
    base = one_arm()
    g10 = _spec_at_gamma(10.0)
    g1 = _spec_at_gamma(1.0)
    assert g10.gamma_0 == 10.0
    assert g1.gamma_0 == 1.0
    for s in (g10, g1):
        assert s.stream_id == base.stream_id
        assert s.input_level == "frozen"
        assert s.readout_similarity == 0.5
        assert s.lr0 == 5.0
        assert s.target_loss == 0.05
        assert s.seed == 0
        assert s.train_config.stopping == "matched_loss"
        assert s.train_config.loss == "mse"


def test_four_streams_are_first_reserved_including_one_arm():
    ids = four_stream_ids()
    assert len(ids) == 4
    assert one_arm().stream_id == ids[0]
    g = _spec_at(10.0, ids[3])
    assert g.stream_id == ids[3]
    assert g.input_level == "frozen"
    assert g.readout_similarity == 0.5
    assert g.train_config.loss == "mse"
