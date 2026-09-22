"""Recurrence first slice is 64 reserved arms, re-present vs s_r-matched control."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from manifolds.dichotomies import readout_similarity  # noqa: E402
from manifolds.streams import (  # noqa: E402
    StreamConfig,
    apply_recurrence,
    make_stream,
    recurrence_control_rng,
)
from run_recurrence import RecurrenceSpec, arms, one_arm  # noqa: E402
from src.pipeline import build_stream  # noqa: E402
from src.reservations import reserved_stream_ids  # noqa: E402

K = 12


def test_slice_is_64_reserved_matched_loss():
    specs = arms()
    assert len(specs) == 64
    assert len({s.key for s in specs}) == 64
    reserved = reserved_stream_ids()
    assert {s.stream_id for s in specs} == set(reserved)
    assert {s.seed for s in specs} == {0}
    assert {s.gamma_0 for s in specs} == {1.0, 10.0}
    assert {s.recurrence_mode for s in specs} == {"represent", "control"}
    assert {s.condition for s in specs} == {"S-HL", "S-LL"}
    assert {s.recurrence_k for s in specs} == {K}
    assert all(s.lr_scaling == "quadratic" for s in specs)
    assert all(s.train_config.stopping == "matched_loss" for s in specs)
    assert all(s.lr0 == 5.0 and s.target_loss == 0.05 for s in specs)
    assert 0.03 not in {s.gamma_0 for s in specs}


def test_one_arm_is_the_precommit_cell():
    spec = one_arm()
    pre = json.loads((ROOT / "results" / "recurrence_precommit.json").read_text())
    assert spec.gamma_0 == 10.0
    assert spec.recurrence_mode == "represent"
    assert spec.recurrence_k == pre["design"]["k"] == K
    assert spec.condition == "S-HL"
    assert spec.stream_id == min(reserved_stream_ids())
    assert spec.train_config.stopping == "matched_loss"
    assert pre["written_before_running"] is True
    will_not = " ".join(pre["will_not_do"]).lower()
    assert "steps(k) versus steps(0)" in will_not
    assert "worst-of-k" in will_not or "grouped outputs" in will_not


def test_represent_copies_task0_and_keeps_arrangements():
    cfg = StreamConfig(stream_id=10000, condition="S-HL", T=16, d=40, M=20, D=2)
    base = make_stream(cfg, np.random.default_rng(0))
    rec = apply_recurrence(
        base, k=K, mode="represent", rng=recurrence_control_rng(10000, K)
    )
    assert np.array_equal(rec.dichotomies[K], rec.dichotomies[0])
    assert np.array_equal(rec.dichotomies[:K], base.dichotomies[:K])
    assert readout_similarity(rec.dichotomies[K], rec.dichotomies[0]) == pytest.approx(1.0)
    for t in range(cfg.T):
        assert np.allclose(rec.arrangements[t].centers, base.arrangements[t].centers)


def test_control_matches_sr_and_is_novel():
    cfg = StreamConfig(stream_id=10000, condition="S-HL", T=16, d=40, M=20, D=2)
    base = make_stream(cfg, np.random.default_rng(0))
    ctrl = apply_recurrence(
        base, k=K, mode="control", rng=recurrence_control_rng(10000, K)
    )
    target = readout_similarity(base.dichotomies[0], base.dichotomies[K - 1])
    got = readout_similarity(ctrl.dichotomies[K], ctrl.dichotomies[K - 1])
    assert got == pytest.approx(target)
    assert not np.array_equal(ctrl.dichotomies[K], base.dichotomies[0])
    assert not np.array_equal(ctrl.dichotomies[K], -base.dichotomies[0])
    assert np.array_equal(ctrl.dichotomies[:K], base.dichotomies[:K])
    for t in range(K):
        assert not np.array_equal(ctrl.dichotomies[K], base.dichotomies[t])
        assert not np.array_equal(ctrl.dichotomies[K], -base.dichotomies[t])


def test_pair_shares_prefix_on_reserved_streams():
    """A8: same arrangement, same prefix, pair differs only at k."""
    for cond in ("S-HL", "S-LL"):
        for sid in sorted(reserved_stream_ids()):
            common = dict(
                stream_id=sid, condition=cond, recurrence_k=K,
                d=40, M=20, D=2, P=16, T=16,
            )
            rep = build_stream(RecurrenceSpec(recurrence_mode="represent", **common))
            ctrl = build_stream(RecurrenceSpec(recurrence_mode="control", **common))
            assert np.array_equal(rep.dichotomies[:K], ctrl.dichotomies[:K])
            assert np.allclose(rep.S_r[:K, :K], ctrl.S_r[:K, :K])
            assert np.array_equal(rep.dichotomies[K], rep.dichotomies[0])
            assert readout_similarity(
                ctrl.dichotomies[K], ctrl.dichotomies[K - 1]
            ) == pytest.approx(
                readout_similarity(rep.dichotomies[0], rep.dichotomies[K - 1])
            )
            for t in range(16):
                assert np.allclose(
                    rep.arrangements[t].centers, ctrl.arrangements[t].centers
                )


def test_high_readout_corners_are_hamming_zero_at_p16():
    """Registered s_r=0.9 rounds to h=0. A8 has no novel control."""
    from src.pipeline import stream_rng
    from src.manifolds.dichotomies import hamming_distance
    for cond in ("S-HH", "S-LH"):
        cfg = StreamConfig(stream_id=10000, condition=cond, T=16, d=40, M=20, D=2)
        base = make_stream(cfg, stream_rng(10000))
        for t in range(1, 16):
            assert hamming_distance(base.dichotomies[0], base.dichotomies[t]) in (0, 16)
        with pytest.raises(ValueError, match="degenerate"):
            apply_recurrence(
                base, k=K, mode="represent", rng=recurrence_control_rng(10000, K)
            )
    cfg = StreamConfig(stream_id=0, condition="S-HL", T=4, d=20, M=10, D=2)
    base = make_stream(cfg, np.random.default_rng(1))
    with pytest.raises(ValueError, match="degenerate"):
        apply_recurrence(base, k=1, mode="represent", rng=np.random.default_rng(0))


def test_classify_includes_zero_is_no_difference():
    from analyse_recurrence import _classify
    assert _classify({"n": 8, "excludes_0": False, "d_steps_mean": 0.94}) == "no_difference"
    assert _classify({"n": 8, "excludes_0": True, "d_steps_mean": -1.0}) == "savings"
    assert _classify({"n": 8, "excludes_0": True, "d_steps_mean": 1.0}) == "reverse"


def test_geometry_and_position_tables_are_marked_not_a_reading():
    from analyse_recurrence import (
        GEOM_BOUNDARIES, K, MODULE, TASK, _geometry_cell, _index, _steps_k_vs_0,
    )

    def fake(gamma, cond, sid, mode, steps0, stepsk, alpha12):
        geom = []
        for b in (0, 4) + GEOM_BOUNDARIES:
            geom.append({
                "module": MODULE, "boundary": b, "task": TASK,
                "ensemble": "retained", "alpha": alpha12 if b == 12 else 1.0,
            })
        tasks = [{"steps_taken": steps0 if i != K else stepsk} for i in range(16)]
        return {
            "usable": True,
            "spec": {
                "gamma_0": gamma, "condition": cond, "stream_id": sid,
                "recurrence_mode": mode, "recurrence_k": K,
            },
            "tasks": tasks,
            "geometry": geom,
            "stream": {"S_r": np.eye(16).tolist()},
        }

    recs = []
    for sid in reserved_stream_ids():
        for cond in ("S-HL", "S-LL"):
            recs.append(fake(10.0, cond, sid, "represent", 40, 17, 0.6))
            recs.append(fake(10.0, cond, sid, "control", 40, 20, 0.5))
    idx = _index(recs)
    pos = _steps_k_vs_0(idx, 10.0)
    assert pos["not_a_reading"] is True
    assert pos["steps_0"]["n"] == 8
    assert pos["steps_0"]["mean"] == pytest.approx(40.0)
    assert pos["steps_k"]["mean"] == pytest.approx(17.0)
    geom = _geometry_cell(idx, 10.0, "represent")
    assert geom["12"]["n"] == 8
    assert geom["12"]["mean"] == pytest.approx(0.6)


def test_slice_reading_is_the_precommit_key():
    path = ROOT / "results" / "recurrence_slice.json"
    if not path.exists():
        pytest.skip("slice not written yet")
    rec = json.loads(path.read_text())
    pre = json.loads((ROOT / "results" / "recurrence_precommit.json").read_text())
    assert rec["n_usable"] == 64
    assert rec["n_arms_missed"] == 0
    assert rec["reading"]["key"] in pre["readings_committed_before_seeing_numbers"]
    assert rec["cell"]["unique_n"] == 8
    assert rec["steps_k_versus_0"]["10"]["not_a_reading"] is True
    assert rec["geometry"]["note"] == pre["comparator"]["geometry"]
    assert rec["geometry"]["boundaries"] == [8, 12, 15]
    assert rec["control_check"]["held"] is True
    assert rec["control_check"]["max_sr_match_error"] == 0.0
    assert rec["structural_finding"]["key"] == "high_readout_refused"
    assert rec["structural_finding"]["corners"] == ["S-HH", "S-LH"]
    mde10 = rec["mde"]["10"]
    assert mde10["baseline"].startswith("steps(control, k)")
    assert mde10["control_steps_k"]["n"] == 8
    assert mde10["savings_ruled_out_at_95_frac_of_control"] < 0.05
    assert mde10["ci_halfwidth_frac_of_control"] < 0.15


def test_mde_uses_control_baseline_not_steps0():
    from analyse_recurrence import _mde
    delta = {
        "d_steps_sem": 0.6907391227621943,
        "ci95": [-0.41634868061390073, 2.2913486806139005],
    }
    control = {"n": 8, "mean": 12.75, "sem": 0.4, "per_stream": {}}
    m = _mde(delta, control)
    assert m["savings_ruled_out_at_95_steps"] == pytest.approx(0.41634868061390073)
    assert m["savings_ruled_out_at_95_frac_of_control"] == pytest.approx(
        0.41634868061390073 / 12.75
    )
    assert m["ci_halfwidth_steps"] == pytest.approx(1.96 * 0.6907391227621943)
    # Must not be ~13% of an 11-step guess, nor of steps(0).
    assert m["ci_halfwidth_frac_of_control"] == pytest.approx(
        1.96 * 0.6907391227621943 / 12.75
    )
