"""Adam Hamming runner is 192 reserved arms, Adam, MSE pin, one-arm first."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from run_adam import (  # noqa: E402
    arms, gate_arms, gate_reading, hamming_only_arms, one_arm, one_arm_hamming,
)
from run_hamming import one_arm as hamming_one_arm  # noqa: E402
from src.reservations import reserved_stream_ids  # noqa: E402


def test_adam_slice_is_192_reserved_adam_mse():
    specs = arms()
    assert len(specs) == 192
    assert len({s.key for s in specs}) == 192
    assert {s.stream_id for s in specs} == set(reserved_stream_ids())
    assert all(s.optimizer == "adam" for s in specs)
    assert all(s.train_config.optimizer == "adam" for s in specs)
    assert all(s.train_config.loss == "mse" for s in specs)
    assert all(s.lr0 == 0.0002 and s.target_loss == 0.05 for s in specs)
    assert all(s.lr_scaling == "quadratic" for s in specs)
    assert all("opt=adam" in s.key for s in specs)


def test_adam_one_arm_matches_hamming_cell():
    a = one_arm()
    h = hamming_one_arm()
    assert a.gamma_0 == h.gamma_0 == 10.0
    assert a.input_level == h.input_level == "frozen"
    assert a.readout_similarity == h.readout_similarity == 0.5
    assert a.stream_id == h.stream_id == min(reserved_stream_ids())
    assert a.tracked_stride == h.tracked_stride == 2
    assert a.optimizer == "adam"
    assert a.lr0 == 0.0002
    assert h.lr0 == 5.0
    assert h.train_config.optimizer == "sgd"
    assert a.train_config.optimizer == "adam"


def test_hamming_spec_asdict_does_not_grow_an_optimizer_field():
    h = asdict(hamming_one_arm())
    assert "optimizer" not in h
    a = asdict(one_arm())
    assert a["optimizer"] == "adam"


def test_full_slice_refuses_before_one_arm_report(tmp_path, monkeypatch):
    import run_adam

    monkeypatch.setattr(run_adam, "ONE_ARM_REPORT", tmp_path / "missing.json")
    monkeypatch.setattr(sys, "argv", ["run_adam.py"])
    with pytest.raises(SystemExit, match="one-arm first"):
        run_adam.main()


def test_full_slice_refuses_before_richness_gate(tmp_path, monkeypatch):
    import run_adam

    monkeypatch.setattr(run_adam, "GATE_REPORT", tmp_path / "missing.json")
    monkeypatch.setattr(sys, "argv", ["run_adam.py"])
    with pytest.raises(SystemExit, match="richness gate first"):
        run_adam.main()


def test_gate_is_16_frozen_half_both_gamma():
    specs = gate_arms()
    assert len(specs) == 16
    assert {s.gamma_0 for s in specs} == {1.0, 10.0}
    assert {s.input_level for s in specs} == {"frozen"}
    assert {s.readout_similarity for s in specs} == {0.5}
    assert {s.stream_id for s in specs} == set(reserved_stream_ids())
    assert all(s.optimizer == "adam" and s.lr0 == 0.0002 for s in specs)


def test_gate_reading_bands_named_before_numbers():
    assert gate_reading(10.0) == "clears_decade"
    assert gate_reading(9.99) == "partial_manipulation"
    assert gate_reading(3.0) == "partial_manipulation"
    assert gate_reading(2.99) == "richness_manipulation_collapses"
    assert gate_reading(3.2) == "partial_manipulation"


def test_hamming_one_arm_is_drift_not_the_gate_cell():
    a = one_arm_hamming()
    g = one_arm()
    assert a.gamma_0 == 10.0
    assert a.input_level == "drift"
    assert a.feature_similarity == 0.9
    assert a.readout_similarity == 0.5
    assert a.stream_id == min(reserved_stream_ids())
    assert a.optimizer == "adam"
    assert a.lr0 == 0.0002
    assert g.input_level == "frozen"
    assert a.key != g.key


def test_hamming_only_is_96_gamma10():
    specs = hamming_only_arms()
    assert len(specs) == 96
    assert len({s.key for s in specs}) == 96
    assert {s.gamma_0 for s in specs} == {10.0}
    assert {s.input_level for s in specs} == {"frozen", "drift", "jump"}
    assert {s.readout_similarity for s in specs} == {1.0, 0.75, 0.5, 0.25}
    assert {s.stream_id for s in specs} == set(reserved_stream_ids())
    assert all(s.optimizer == "adam" and s.lr0 == 0.0002 for s in specs)


def test_full_slice_refuses_unless_clears_decade(monkeypatch):
    import run_adam

    monkeypatch.setattr(sys, "argv", ["run_adam.py"])
    with pytest.raises(SystemExit, match="do not launch the 192"):
        run_adam.main()


def test_hamming_only_refuses_before_its_one_arm(tmp_path, monkeypatch):
    import run_adam

    monkeypatch.setattr(run_adam, "HAMMING_ONE_ARM_REPORT", tmp_path / "missing.json")
    monkeypatch.setattr(sys, "argv", ["run_adam.py", "--hamming-only"])
    with pytest.raises(SystemExit, match="hamming one-arm first"):
        run_adam.main()
