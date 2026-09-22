"""Scope first slice is 192 reserved arms, worst-of-K, K in {1,2,4}."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from run_scope import ScopeSpec, arms, one_arm  # noqa: E402
from src.reservations import reserved_stream_ids  # noqa: E402


def test_slice_is_192_reserved_worst_of_k():
    specs = arms()
    assert len(specs) == 192
    assert len({s.key for s in specs}) == 192
    reserved = reserved_stream_ids()
    assert {s.stream_id for s in specs} == set(reserved)
    assert {s.seed for s in specs} == {0}
    assert {s.gamma_0 for s in specs} == {1.0, 10.0}
    assert {s.scope_K for s in specs} == {1, 2, 4}
    assert all(s.n_outputs == s.scope_K for s in specs)
    assert all(s.lr_scaling == "quadratic" for s in specs)
    assert all(s.train_config.stopping == "worst_of_k" for s in specs)
    assert all(s.lr0 == 5.0 and s.target_loss == 0.05 for s in specs)
    assert 0.03 not in {s.gamma_0 for s in specs}


def test_one_arm_is_the_precommit_cell():
    spec = one_arm()
    pre = json.loads((ROOT / "results" / "scope_precommit.json").read_text())
    assert spec.gamma_0 == 10.0
    assert spec.scope_K == 4
    assert spec.n_outputs == 4
    assert spec.condition == "S-HL"
    assert spec.stream_id == min(reserved_stream_ids())
    assert spec.train_config.stopping == "worst_of_k"
    assert pre["written_before_running"] is True
    will_not = " ".join(pre["will_not_do"]).lower()
    assert "breadth |s|" in will_not or "fall back to breadth" in will_not
