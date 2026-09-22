"""Reserved set never informs lr0, the rank grid, or the stopping criterion.

`docs/16` §Amendments A3. The property is provenance of those three decisions,
not whether a reserved id string appears in a file. New-axis pre-commits may
name reserved ids as experimental subjects; that is the intended use.
Artifacts that *chose* lr0 / the grid / stopping must not contain reserved
ids as design inputs, and the rank pre-commit must record the registered pin.
"""

from __future__ import annotations

import json

from src.pipeline import Phase1Spec
from src.reservations import (
    RESERVATION_PATH,
    ROOT,
    design_decision_paths,
    load,
    old_init_seeds,
    reserved_ids_in_path,
    reserved_stream_ids,
)

REGISTERED_LR0 = 5.0
REGISTERED_TARGET_LOSS = 0.05
A5_RANKS = [4, 16, 300]


def test_reservation_file_names_two_populations():
    rec = load()
    assert rec["rule"].startswith("The reserved set never informs")
    old = rec["populations"]["old"]
    reserved = rec["populations"]["reserved"]
    assert old["keying"] == "legacy_seed"
    assert reserved["keying"] == "stream_rng"
    assert old_init_seeds() == tuple(range(8))
    ids = reserved_stream_ids()
    assert ids == frozenset(range(10000, 10008))
    assert rec["stream_rng"]["salt"] == 20260817


def test_reserved_ids_are_outside_the_used_label_block():
    """0–7 were stored as labels and used as genuine-n keys after the RNG fix."""
    assert min(reserved_stream_ids()) >= 10000


def test_the_reservation_file_itself_is_not_a_design_decision_artifact():
    rec = load()
    listed = [p.resolve() for p in design_decision_paths(rec)]
    assert RESERVATION_PATH.resolve() not in listed


def test_artifacts_that_chose_lr0_do_not_contain_reserved_ids():
    """phase0 / phase1 / cost_model informed the registered pin. Reserved ids
    there would mean the reservation leaked into the design input."""
    reserved = reserved_stream_ids()
    hits = []
    for path in design_decision_paths():
        found = reserved_ids_in_path(path, reserved)
        if found:
            hits.append(f"{path.relative_to(ROOT)}: {sorted(found)}")
    assert hits == [], (
        "reserved stream id in an artifact that informed lr0, the rank grid, "
        "or stopping:\n  " + "\n  ".join(hits)
    )


def test_lr0_and_stopping_are_the_registered_pin():
    spec = Phase1Spec()
    assert spec.lr0 == REGISTERED_LR0
    assert spec.target_loss == REGISTERED_TARGET_LOSS
    pre = json.loads((ROOT / "results" / "rank_precommit.json").read_text())
    held = pre["design"]["held_fixed"]
    assert held["lr0"] == REGISTERED_LR0
    assert held["target_loss"] == REGISTERED_TARGET_LOSS
    assert "registered" in held["lr0_provenance"].lower()
    assert "not informed by the reserved set" in held["lr0_provenance"].lower()
    assert "not informed by the reserved set" in held["target_loss_provenance"].lower()
    assert "registered" in pre["misses"]["lr0_pinned_on"].lower()


def test_rank_grid_provenance_is_the_brief_not_reserved_pilots():
    pre = json.loads((ROOT / "results" / "rank_precommit.json").read_text())
    assert pre["design"]["first_slice"]["ranks"] == A5_RANKS
    assert "docs/16 A5" in pre["design"]["held_fixed"]["rank_grid_provenance"]
    assert "not informed by the reserved set" in pre["design"]["held_fixed"]["rank_grid_provenance"].lower()
    will_not = " ".join(pre["will_not_do"]).lower()
    assert "inform lr0, the rank grid, or stopping from reserved-set" in will_not


def test_isotropic_precommit_uses_registered_pin():
    pre = json.loads((ROOT / "results" / "isotropic_precommit.json").read_text())
    held = pre["design"]["held_fixed"]
    assert held["lr0"] == REGISTERED_LR0
    assert held["target_loss"] == REGISTERED_TARGET_LOSS
    assert "not informed by the reserved set" in held["lr0_provenance"].lower()
    assert "not informed by the reserved set" in held["target_loss_provenance"].lower()
    assert 0.03 not in pre["design"]["first_slice"]["gammas"]
    will_not = " ".join(pre["will_not_do"]).lower()
    assert "inform lr0" in will_not
    assert "unit-normalize" in will_not
    assert "cross rank" in will_not


def test_nottuned_precommit_uses_registered_pin():
    pre = json.loads((ROOT / "results" / "nottuned_precommit.json").read_text())
    held = pre["design"]["held_fixed"]
    assert held["lr0"] == REGISTERED_LR0
    assert held["target_loss"] == REGISTERED_TARGET_LOSS
    assert "not informed by the reserved set" in held["lr0_provenance"].lower()
    assert "not informed by the reserved set" in held["target_loss_provenance"].lower()
    assert pre["design"]["grid"]["gammas"] == [0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
    will_not = " ".join(pre["will_not_do"]).lower()
    assert "inform lr0" in will_not
    assert "reuse rank" in will_not
    assert "reuse isotropic" in will_not


def test_scope_precommit_uses_registered_pin():
    pre = json.loads((ROOT / "results" / "scope_precommit.json").read_text())
    held = pre["design"]["held_fixed"]
    assert held["lr0"] == REGISTERED_LR0
    assert held["target_loss"] == REGISTERED_TARGET_LOSS
    assert "not informed by the reserved set" in held["lr0_provenance"].lower()
    assert "not informed by the reserved set" in held["target_loss_provenance"].lower()
    assert 0.03 not in pre["design"]["first_slice"]["gammas"]
    will_not = " ".join(pre["will_not_do"]).lower()
    assert "inform lr0" in will_not
    assert "|s|" in will_not
    assert "mean-reduced" in will_not


def test_hamming_precommit_uses_registered_pin():
    pre = json.loads((ROOT / "results" / "hamming_precommit.json").read_text())
    held = pre["design"]["held_fixed"]
    assert held["lr0"] == REGISTERED_LR0
    assert held["target_loss"] == REGISTERED_TARGET_LOSS
    assert "not informed by the reserved set" in held["lr0_provenance"].lower()
    assert "not informed by the reserved set" in held["target_loss_provenance"].lower()
    assert 0.03 not in pre["design"]["first_slice"]["gammas"]
    assert pre["design"]["first_slice"]["n_arms"] == 192
    assert pre["design"]["held_fixed"]["tracked_stride"] == 2
    assert pre["design"]["held_fixed"]["P"] == 16
    assert pre["not_in_phase1"] is True
    assert pre["one_arm_first"]["governs"] is False
    readings = pre["readings_committed_before_seeing_numbers"]
    assert "frozen_intermediate" in readings and "frozen_outside" in readings
    will_not = " ".join(pre["will_not_do"]).lower()
    assert "inform lr0" in will_not
    assert "p=32" in will_not
    assert "tracked_stride=4" in will_not
    assert "expand before the one-arm" in will_not


def test_recurrence_precommit_uses_registered_pin():
    pre = json.loads((ROOT / "results" / "recurrence_precommit.json").read_text())
    held = pre["design"]["held_fixed"]
    assert held["lr0"] == REGISTERED_LR0
    assert held["target_loss"] == REGISTERED_TARGET_LOSS
    assert "not informed by the reserved set" in held["lr0_provenance"].lower()
    assert "not informed by the reserved set" in held["target_loss_provenance"].lower()
    assert 0.03 not in pre["design"]["first_slice"]["gammas"]
    assert pre["design"]["k"] == 12
    assert pre["design"]["first_slice"]["n_arms"] == 64
    assert pre["design"]["first_slice"]["conditions"] == ["S-HL", "S-LL"]
    assert pre["design"]["corners_refused"]["conditions"] == ["S-HH", "S-LH"]
    will_not = " ".join(pre["will_not_do"]).lower()
    assert "inform lr0" in will_not
    assert "worst-of-k" in will_not or "grouped outputs" in will_not
    assert "steps(k) versus steps(0)" in will_not


def test_rank_precommit_does_not_claim_to_govern_the_one_arm():
    pre = json.loads((ROOT / "results" / "rank_precommit.json").read_text())
    assert pre["one_arm_preceded_this_commit"] is True
    assert pre["written_before_running_applies_to"] == "288-arm first slice"
    assert pre["one_arm"]["governs"] is False
    assert pre["one_arm"]["ran_before_this_commit"] is True
