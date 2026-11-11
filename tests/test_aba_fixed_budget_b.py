"""docs/36 (Phase 1b): fixed budget on phase B per γ0, A1 and A2 at criterion; `first_hit` logging.
Toy scale. Bit-identity with job 4651 at design scale is checked on grime (`aba_cluster.py check1b`)."""
from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

import pytest

CORE = Path(__file__).resolve().parents[1] / "notebooks" / "core"
sys.path.insert(0, str(CORE))
import aba_lib as L  # noqa: E402

TOY = L.ABAConfig(quick=True, P=8, d=20, D=2, M=10, N=32, n_t=10, ntk_per_manifold=5,
                  refit_steps=50, bottleneck_n=8)
SPEC = L.ArmSpec(0, 1.0, 4, 0.9)


def _run(cfg):
    return json.loads(json.dumps(L.run_arm(cfg, SPEC, stamp={"test": True})))


@pytest.fixture(scope="module")
def cr():
    return _run(TOY)


@pytest.fixture(scope="module")
def fx(cr):
    b = max(cr["behaviour"]["steps"]["B"], cr["behaviour"]["steps"]["freshB"]) + 25
    return b, _run(dataclasses.replace(TOY, stopping="fixed", fixed_steps=(None, b, None)))


def test_first_hit_equals_steps_under_criterion(cr):
    beh = cr["behaviour"]
    assert beh["first_hit"] == beh["steps"]
    assert "FT_beh_hit" not in beh


def test_fixed_b_keeps_a1_and_first_hit_matches_criterion(cr, fx):
    b, row = fx
    beh, ref = row["behaviour"], cr["behaviour"]
    # A1 is the same computation (docs/36 integrity check 1)
    assert beh["steps"]["A1"] == ref["steps"]["A1"]
    assert row["geometry"]["A@A1"] == cr["geometry"]["A@A1"]
    assert row["backward"]["init_to_A1"] == cr["backward"]["init_to_A1"]
    assert row["forward"] == cr["forward"]                       # forward unchanged by design (docs/36 §3)
    # B and fresh B fixed; their first step at criterion is the criterion step count (check 2)
    assert row["status"]["B"] == row["status"]["freshB"] == "fixed"
    assert beh["steps"]["B"] == beh["steps"]["freshB"] == b
    assert beh["first_hit"]["B"] == ref["steps"]["B"]
    assert beh["first_hit"]["freshB"] == ref["steps"]["freshB"]
    assert beh["FT_beh_hit"] == ref["FT_beh"]
    # A2 stays at criterion
    assert row["status"]["A2"] == "converged" and beh["first_hit"]["A2"] == beh["steps"]["A2"]
    assert row["usable"]


def test_first_hit_none_when_budget_too_short(cr):
    b = max(1, cr["behaviour"]["steps"]["B"] // 4)
    row = _run(dataclasses.replace(TOY, stopping="fixed", fixed_steps=(None, b, None)))
    if cr["behaviour"]["first_loss"]["B"] > TOY.target_loss:
        assert row["behaviour"]["first_hit"]["B"] is None
        assert row["behaviour"]["FT_beh_hit"] is None


def test_phase1b_configs_and_specs():
    base = L.ABAConfig.make(False)
    assert L.PHASE1B_BUDGET == {0.1: 3200, 1.0: 600, 10.0: 100}
    assert len(L.PHASE1B_CELLS) == len(set(L.PHASE1B_CELLS)) == 11
    keys = set()
    for g, b in L.PHASE1B_BUDGET.items():
        cfg = L.phase1b_cfg(base, g)
        assert cfg.gammas == (g,) and list(cfg.seeds) == list(range(16, 24))
        kf = json.loads(json.dumps(cfg.key_fields()))
        assert kf["stopping"] == "fixed" and kf["fixed_steps"] == [None, b, None]
        assert kf["probe_hammings"] == [0, 2, 4, 6, 8, 12, 16]          # same seed draws as job 4651
        specs = L.phase1b_specs(cfg)
        assert len(specs) == 88 and all(s.gamma == g for s in specs)
        keys |= {s.key for s in specs}
    assert len(keys) == 264
    # every docs/36 cell is a Phase 1 cell, so each arm has its job 4651 partner
    p1 = {(s.hamming, s.s_f) for s in L.phase1_specs(L.map_config(base))}
    assert set(L.PHASE1B_CELLS) <= p1


def test_analysis_contrasts_and_cluster_modes():
    import aba_fixedb_analysis as FB
    import aba_cluster as C
    assert len(FB.CONTRASTS) == 12
    assert all(a in FB.CELLS and b in FB.CELLS for a, b in FB.CONTRASTS)
    assert C.P1B_TIMING_SPEC.key == "seed=19,gamma=0.1,H=16,s_f=1"
    assert C.MAP_REF_JSONL.name == "FULL_e3011fa49aca1994.jsonl"
