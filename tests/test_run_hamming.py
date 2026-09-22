"""Hamming slice is 192 reserved arms, exact Hamming, stride 2, frozen one-arm."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from run_hamming import (  # noqa: E402
    INPUT_LEVELS, INPUT_S_F, READOUT, TRACKED_STRIDE, HammingSpec, arms, one_arm,
)
from src import pipeline as pl  # noqa: E402
from src.manifolds.dichotomies import (  # noqa: E402
    hamming_distance, hamming_for_readout_similarity,
)
from src.reservations import reserved_stream_ids  # noqa: E402


def test_slice_is_192_reserved_quadratic_stride2():
    specs = arms()
    assert len(specs) == 192
    assert len({s.key for s in specs}) == 192
    reserved = reserved_stream_ids()
    assert {s.stream_id for s in specs} == set(reserved)
    assert {s.seed for s in specs} == {0}
    assert {s.gamma_0 for s in specs} == {1.0, 10.0}
    assert {s.input_level for s in specs} == set(INPUT_LEVELS)
    assert {s.readout_similarity for s in specs} == set(READOUT)
    assert all(s.arrangement_source == "stream_rng" for s in specs)
    assert all(s.lr_scaling == "quadratic" for s in specs)
    assert all(s.tracked_stride == TRACKED_STRIDE for s in specs)
    assert all(s.P == 16 and s.lr0 == 5.0 and s.target_loss == 0.05 for s in specs)
    assert all(s.feature_similarity == INPUT_S_F[s.input_level] for s in specs)
    assert all(s.condition == s.input_level for s in specs)


def test_requested_hamming_is_exact_at_p16():
    for s_r, h in zip(READOUT, (0, 2, 4, 6)):
        got = hamming_for_readout_similarity(16, s_r)
        assert got == h
        assert 1.0 - 2 * got / 16 == s_r


def test_one_arm_is_frozen_half_hamming_first_reserved():
    spec = one_arm()
    pre = json.loads((ROOT / "results" / "hamming_precommit.json").read_text())
    assert spec.gamma_0 == 10.0
    assert spec.input_level == "frozen"
    assert spec.readout_similarity == 0.5
    assert spec.feature_similarity == 1.0
    assert spec.stream_id == min(reserved_stream_ids())
    assert spec.tracked_stride == 2
    assert spec.lr_scaling == "quadratic"
    assert spec.train_config.optimizer == "sgd"
    assert pre["written_before_running"] is True
    assert pre["written_before_running_applies_to"] == "192-arm Hamming first slice"
    assert pre["not_in_phase1"] is True
    assert pre["design"]["first_slice"]["n_arms"] == 192
    assert pre["design"]["held_fixed"]["tracked_stride"] == 2
    assert 0.03 not in pre["design"]["first_slice"]["gammas"]
    readings = pre["readings_committed_before_seeing_numbers"]
    assert "frozen_intermediate" in readings
    assert "frozen_outside" in readings
    will_not = " ".join(pre["will_not_do"]).lower()
    assert "expand before the one-arm" in will_not
    assert "p=32" in will_not
    assert "tracked_stride=4" in will_not


def test_stride2_gives_shared_lag4_and_lag12_on_two_tasks():
    spec = HammingSpec(tracked_stride=2)
    sched = pl.schedule(spec.T, spec.tracked_stride)
    assert pl.n_evals(sched, spec) == 106
    lag = {k: set() for k in (4, 12)}
    for b, js in sched.items():
        for j in js:
            d = b - j
            if d in lag:
                lag[d].add(j)
    assert lag[12] == {0, 2}
    assert lag[4] == {0, 2, 4, 6, 8, 10}


def test_same_stream_id_and_s_r_pair_A0_across_input_levels():
    sid = min(reserved_stream_ids())
    streams = []
    for level in INPUT_LEVELS:
        spec = HammingSpec(
            gamma_0=10.0, a=0.0, condition=level, seed=0, stream_id=sid,
            input_level=level, feature_similarity=INPUT_S_F[level],
            readout_similarity=0.5, tracked_stride=2,
            arrangement_source="stream_rng",
        )
        streams.append(pl.build_stream(spec))
    a = streams[0]
    assert hamming_distance(a.dichotomies[0], a.dichotomies[1]) == 4
    for s in streams[1:]:
        assert np.array_equal(a.dichotomies, s.dichotomies)
        assert np.array_equal(a.arrangements[0].centers, s.arrangements[0].centers)
    # Input level moves later centres; frozen copies, drift/jump do not stay put.
    frozen, drift, jump = streams
    assert np.allclose(frozen.arrangements[1].centers, frozen.arrangements[0].centers)
    assert not np.allclose(drift.arrangements[1].centers, drift.arrangements[0].centers)
    assert not np.allclose(jump.arrangements[1].centers, jump.arrangements[0].centers)
    assert not np.allclose(drift.arrangements[1].centers, jump.arrangements[1].centers)


def test_precommit_readings_are_named_before_the_slice_json():
    """The named outcomes exist in the pre-commit regardless of whether analysis has run."""
    pre = json.loads((ROOT / "results" / "hamming_precommit.json").read_text())
    names = set(pre["readings_committed_before_seeing_numbers"])
    assert names >= {
        "monotone_dose", "binary_only", "nonmonotone",
        "finding3_fails_to_recover", "frozen_intermediate", "frozen_outside", "mixed",
    }
    from analyse_hamming import axis_reading, dose_reading, frozen_reading
    assert dose_reading({1.0: 0.1, 0.75: 0.0, 0.5: -0.2, 0.25: -0.5}) == "monotone_dose"
    assert dose_reading({1.0: 0.2, 0.75: -0.4, 0.5: -0.1, 0.25: -0.3}) == "binary_only"
    assert dose_reading({1.0: 0.0, 0.75: 0.2, 0.5: -0.4, 0.25: 0.1}) == "nonmonotone"
    inside = {
        "frozen": {0.75: -0.2, 0.5: -0.3, 0.25: -0.4},
        "drift": {0.75: -0.1, 0.5: -0.2, 0.25: -0.3},
        "jump": {0.75: -0.5, 0.5: -0.6, 0.25: -0.7},
    }
    assert frozen_reading(inside) == "frozen_intermediate"
    outside = dict(inside)
    outside["frozen"] = {0.75: 0.9, 0.5: 0.9, 0.25: 0.9}
    assert frozen_reading(outside) == "frozen_outside"
    mono = {s: 0.1 * i for i, s in enumerate((1.0, 0.75, 0.5, 0.25))}
    not_mono = {1.0: 0.2, 0.75: -0.1, 0.5: 0.0, 0.25: -0.3}
    assert axis_reading({"frozen": mono, "drift": mono, "jump": mono}) == "monotone_dose"
    assert axis_reading({"frozen": not_mono, "drift": mono, "jump": mono}) == "nonmonotone"


def test_cliff_range_is_endpoints_not_max_of_three():
    """Post-cliff range is |0.75 − 0.25|, so a dip at 0.50 does not inflate it."""
    from analyse_hamming import cliff_capacity

    def cell(floors, lo, hi):
        return {"mean_floors": floors, "ci95": [lo, hi]}

    grid = {
        (10.0, "jump", 1.0): cell(-55.1, -1.2, -0.9),
        (10.0, "jump", 0.75): cell(-76.9, -1.50, -1.35),
        (10.0, "jump", 0.5): cell(-78.0, -1.51, -1.38),
        (10.0, "jump", 0.25): cell(-77.6, -1.56, -1.32),
    }
    st = cliff_capacity(grid, 10.0, "jump")
    assert st["cliff"] == pytest.approx(21.8)
    assert st["post_range"] == pytest.approx(0.7)
    assert st["post_ci_overlap"] is True
