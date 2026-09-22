"""Order first-slice pre-commit is on file before any order code or arm."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_order_precommit_is_written_before_running():
    pre = json.loads((ROOT / "results" / "order_precommit.json").read_text())
    assert pre["written_before_running"] is True
    assert pre["written_before_running_applies_to"] == "64-arm order first slice"
    assert pre["second_learner_line"] == "closed"
    assert pre["design"]["optimizer"] == "sgd"
    assert pre["design"]["loss"] == "mse"
    assert pre["design"]["held_fixed"]["lr0"] == 5.0
    assert pre["design"]["held_fixed"]["tracked_stride"] == 2
    sl = pre["design"]["first_slice"]
    assert sl["n_arms_new"] == 64
    assert sl["gammas"] == [10.0]
    assert sl["schedules_trained"] == ["reverse", "shuffle"]
    assert sl["schedules_paired_from_hamming"] == ["forward"]
    assert sl["input_levels"] == ["frozen", "drift"]
    assert sl["s_r"] == [0.75, 0.25]
    assert pre["design"]["primary_cell"]["presentation_index"] == 8
    assert pre["design"]["primary_cell"]["lag"] == 4
    readings = pre["readings_committed_before_seeing_numbers"]
    assert set(readings) == {
        "weaker_prediction_holds",
        "weaker_prediction_fails",
        "partial",
        "forward_identity_fails",
    }
    assert pre["operationalization"]["written_before_64_arm_numbers"] is True
    assert pre["one_arm_first"]["governs"] is False
    assert "reverse" in pre["one_arm_first"]["spec"]
    assert "drift" in pre["one_arm_first"]["spec"]
    will_not = " ".join(pre["will_not_do"]).lower()
    assert "fourth learner" in will_not
    assert "adam" in will_not
    assert "accumulation hypothesis" in will_not
    assert "results/phase1/" in will_not
    assert pre["not_in_phase1"] is True
    assert pre["pairing"]["forward_source"] == "results/hamming/"
