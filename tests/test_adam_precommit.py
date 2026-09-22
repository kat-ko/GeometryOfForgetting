"""Adam Hamming pre-commit is on file before any Adam code or arm."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_adam_precommit_is_written_before_running():
    pre = json.loads((ROOT / "results" / "adam_precommit.json").read_text())
    assert pre["written_before_running"] is True
    assert pre["written_before_running_applies_to"] == "192-arm Adam Hamming first slice"
    assert pre["not_in_phase1"] is True
    assert pre["results_dir"] == "results/adam/"
    assert pre["i3_exception"]["licensed"] is True
    assert pre["i3_exception"]["never"] == "results/phase1/"
    assert pre["design"]["loss"] == "mse"
    assert pre["design"]["optimizer"]["name"] == "adam"
    assert pre["design"]["held_fixed"]["target_loss"] == 0.05
    assert pre["design"]["held_fixed"]["lr0"] == 0.0002
    assert pre["misses"]["inherited_missed"] is True
    assert pre["misses"]["lr0_inherited"] == 5.0
    assert pre["design"]["first_slice"]["n_arms"] == 192
    assert 0.03 not in pre["design"]["first_slice"]["gammas"]
    assert pre["richness_gate"]["bar"].startswith("mean(γ=10) / mean(γ=1) ≥ 10")
    readings = pre["readings_committed_before_seeing_numbers"]
    assert set(readings) == {
        "stream_property",
        "learner_property",
        "finding3_fails_to_recover",
        "mixed",
        "richness_manipulation_collapses",
    }
    will_not = " ".join(pre["will_not_do"]).lower()
    assert "another classification loss" in will_not
    assert "reopen the ce arm" in will_not
    assert "results/phase1/" in will_not
    assert "192 before the richness gate" in will_not
    assert pre["one_arm_first"]["governs"] is False
    assert pre["one_arm_first"]["reached_target"] is True
    one = json.loads((ROOT / "results" / "adam_one_arm.json").read_text())
    assert one["reached_target"] is True
    assert one["governs"] is False
    assert one["n_tasks_missed"] == 0
    assert one["spec"]["lr0"] == 0.0002


def test_adam_hamming_precommit_is_written_before_the_96():
    pre = json.loads((ROOT / "results" / "adam_hamming_precommit.json").read_text())
    assert pre["written_before_running"] is True
    assert pre["written_before_running_applies_to"] == "96-arm Adam Hamming-only at γ=10"
    assert pre["not_the_192"] is True
    assert pre["gate_reading"] == "partial_manipulation"
    assert pre["design"]["first_slice"]["n_arms"] == 96
    assert pre["design"]["first_slice"]["gammas"] == [10.0]
    assert pre["design"]["held_fixed"]["lr0"] == 0.0002
    assert "γ" in pre["cannot_speak_to"]
    assert "finding 1" in pre["cannot_speak_to"]
    assert "channel reorganisation" in pre["cannot_speak_to"]
    readings = pre["readings_committed_before_seeing_numbers"]
    assert set(readings) == {
        "hamming_reproduces",
        "hamming_fails",
        "partial",
        "finding3_fails_to_recover",
    }
    assert pre["one_arm_first"]["governs"] is False
    assert "drift" in pre["one_arm_first"]["spec"]
    assert "not frozen" in pre["one_arm_first"]["spec"]
    will_not = " ".join(pre["will_not_do"]).lower()
    assert "launch the 192" in will_not
    assert "finding 1" in will_not
    assert pre["operationalization"]["written_before_96_arm_numbers"] is True
