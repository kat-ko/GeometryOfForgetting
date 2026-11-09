"""aba_lib per-phase stopping (docs/32 §6): a None entry in fixed_steps means criterion stopping
for that phase only. Toy scale. Bit-identity of the all-criterion and all-fixed cells with jobs
4396 / 4458 at design scale is checked on the cluster (`aba_cluster.py match`)."""
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
BUDGET = (30, 40, 20)


def _cell(a_fixed: bool, b_fixed: bool) -> L.ABAConfig:
    if not a_fixed and not b_fixed:
        return TOY
    steps = (BUDGET[0] if a_fixed else None, BUDGET[1] if b_fixed else None, BUDGET[2] if a_fixed else None)
    return dataclasses.replace(TOY, stopping="fixed", fixed_steps=steps)


@pytest.fixture(scope="module")
def rows():
    return {(a, b): json.loads(json.dumps(L.run_arm(_cell(a, b), SPEC, stamp={"test": True})))
            for a in (False, True) for b in (False, True)}


def test_status_per_phase(rows):
    for (a, b), r in rows.items():
        want = {"A1": "fixed" if a else "converged", "A2": "fixed" if a else "converged",
                "B": "fixed" if b else "converged", "freshB": "fixed" if b else "converged"}
        assert r["status"] == want, (a, b)
        assert r["usable"]
        if a:
            assert r["behaviour"]["steps"]["A1"] == BUDGET[0]
            assert r["behaviour"]["steps"]["A2"] == BUDGET[2]
        if b:
            assert r["behaviour"]["steps"]["B"] == BUDGET[1]
            assert r["behaviour"]["steps"]["freshB"] == BUDGET[1]


def test_a_phase_depends_only_on_the_a_rule(rows):
    for a in (False, True):
        r0, r1 = rows[a, False], rows[a, True]
        assert r0["geometry"]["A@A1"] == r1["geometry"]["A@A1"]
        assert r0["backward"]["init_to_A1"] == r1["backward"]["init_to_A1"]
        assert r0["trajectory"]["A1"] == r1["trajectory"]["A1"]


def test_fresh_b_depends_only_on_the_b_rule(rows):
    for b in (False, True):
        assert rows[False, b]["behaviour"]["steps"]["freshB"] == rows[True, b]["behaviour"]["steps"]["freshB"]


def test_config_dict_round_trips_with_none():
    cfg = _cell(True, False)
    d = json.loads(json.dumps(cfg.key_fields()))
    assert d["fixed_steps"] == [BUDGET[0], None, BUDGET[2]]
    assert d["stopping"] == "fixed"
    assert "stopping" not in TOY.key_fields()          # criterion rows keep job 4349's dict


def test_cluster_cells_and_budget_parsing():
    sys.path.insert(0, str(CORE))
    import aba_cluster as C
    assert len(C.SPECS_2X2) == 32 and len({s.key for s in C.SPECS_2X2}) == 32
    base = L.ABAConfig.make(False)
    for cell in C.CELLS_2X2:
        assert C.cell_of(C.cell_cfg(base, cell)) == cell
    assert C.cell_cfg(base, "Fx,Fx").key_fields()["fixed_steps"] == [2300, 2400, 2000]
    assert C.cell_cfg(base, "Cr,Cr").key_fields() == base.key_fields()
