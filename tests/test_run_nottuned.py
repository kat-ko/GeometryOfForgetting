"""Not-tuned duplicate is 192 reserved arms, quadratic, spherical, Q = I."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from run_nottuned import NottunedSpec, arms, one_arm  # noqa: E402
from src.analysis.grid import REGISTERED_GAMMAS  # noqa: E402
from src.reservations import reserved_stream_ids  # noqa: E402


def test_slice_is_192_reserved_quadratic_spherical_unconstrained():
    specs = arms()
    assert len(specs) == 192
    assert len({s.key for s in specs}) == 192
    reserved = reserved_stream_ids()
    assert {s.stream_id for s in specs} == set(reserved)
    assert {s.seed for s in specs} == {0}
    assert {s.gamma_0 for s in specs} == set(REGISTERED_GAMMAS)
    assert {s.condition for s in specs} == {"S-HH", "S-HL", "S-LH", "S-LL"}
    assert all(s.arrangement_source == "stream_rng" for s in specs)
    assert all(s.lr_scaling == "quadratic" for s in specs)
    assert all(s.manifold_kind == "spherical" for s in specs)
    assert all(s.lr0 == 5.0 and s.target_loss == 0.05 for s in specs)
    assert all(not hasattr(s, "readout_rank") for s in specs)


def test_one_arm_is_the_precommit_cell():
    spec = one_arm()
    pre = json.loads((ROOT / "results" / "nottuned_precommit.json").read_text())
    assert spec.gamma_0 == 10.0
    assert spec.condition == "S-HL"
    assert spec.stream_id == min(reserved_stream_ids())
    assert spec.manifold_kind == "spherical"
    assert spec.lr_scaling == "quadratic"
    assert not hasattr(spec, "readout_rank")
    assert pre["written_before_running"] is True
    assert pre["written_before_running_applies_to"] == "192-arm not-tuned reserved duplicate"
    held = pre["design"]["held_fixed"]
    assert held["lr0"] == 5.0
    assert held["target_loss"] == 0.05
    assert "not informed by the reserved set" in held["lr0_provenance"].lower()
    assert pre["design"]["grid"]["n_arms"] == 192
    assert 0.03 in pre["design"]["grid"]["gammas"]
