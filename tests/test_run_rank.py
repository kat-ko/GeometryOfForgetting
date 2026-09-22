"""Rank first slice is 288 reserved arms under nested Q, not the un-nested one-arm."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from run_rank import RankSpec, arms, one_arm  # noqa: E402
from src.reservations import reserved_stream_ids  # noqa: E402


def test_slice_is_288_reserved_nested_quadratic():
    specs = arms()
    assert len(specs) == 288
    assert len({s.key for s in specs}) == 288
    reserved = reserved_stream_ids()
    assert {s.stream_id for s in specs} == set(reserved)
    assert {s.seed for s in specs} == {0}
    assert {s.readout_rank for s in specs} == {4, 16, 300}
    assert {s.gamma_0 for s in specs} == {0.03, 1.0, 10.0}
    assert {s.condition for s in specs} == {"S-HH", "S-HL", "S-LH", "S-LL"}
    assert all(s.arrangement_source == "stream_rng" for s in specs)
    assert all(s.lr_scaling == "quadratic" for s in specs)
    assert all(s.q_nested for s in specs)
    assert all(s.lr0 == 5.0 and s.target_loss == 0.05 for s in specs)


def test_one_arm_is_the_precommit_cell_under_nested_q():
    spec = one_arm()
    pre = json.loads((ROOT / "results" / "rank_precommit.json").read_text())
    assert spec.gamma_0 == 10.0
    assert spec.readout_rank == 4
    assert spec.condition == "S-HL"
    assert spec.stream_id == min(reserved_stream_ids())
    assert spec.q_nested is True
    assert spec.lr_scaling == "quadratic"
    assert pre["one_arm"]["governs"] is False
