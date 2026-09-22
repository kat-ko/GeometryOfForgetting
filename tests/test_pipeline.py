"""Tests for the Phase 1 pipeline wiring.

The expensive part (a full arm) is smoke-tested once at reduced size; the rest are
structural properties that a schedule or budget change could silently break.
"""

from __future__ import annotations

import numpy as np
import pytest

from src import pipeline as pl
from src.models import paired_init

SMOKE = dict(T=4, P=8, M=40, N=80, d=60, n_t=20, tracked_stride=2, steps_per_task=3000)


def test_schedule_includes_every_tracked_task_own_baseline():
    """Attribution differences against boundary `j`; without it a task has no baseline."""
    for T, stride in [(16, 4), (16, 8), (8, 2), (5, 3)]:
        sched = pl.schedule(T, stride)
        tracked = {j for js in sched.values() for j in js}
        for j in tracked:
            assert j in sched, f"task {j} tracked but boundary {j} not measured"
            assert j in sched[j], f"task {j} not measured at its own baseline"


def test_schedule_measures_the_final_boundary_and_widens_the_lag():
    sched = pl.schedule(16, 4)
    assert 15 in sched
    assert sched[15] == [0, 4, 8, 12]
    lags = sorted({b - j for b, js in sched.items() for j in js if b != j})
    assert len(lags) > 1, "retention followed at a single lag cannot show a rate"


def test_design_point_fits_the_measured_eval_budget():
    spec = pl.Phase1Spec()
    n = pl.n_evals(pl.schedule(spec.T, spec.tracked_stride), spec)
    assert n <= pl.EVAL_BUDGET, f"{n} evals exceeds the {pl.EVAL_BUDGET}-eval cost model"
    assert n > pl.EVAL_BUDGET // 2, "budget left unused — measure more"


def test_paired_init_makes_hidden_weights_independent_of_gamma_and_a():
    """I6: the γ and `a` axes are orthogonal at init only if `shape` is untouched."""
    specs = [pl.Phase1Spec(gamma_0=g, **SMOKE) for g in (0.03, 10.0)]
    stream = pl.build_stream(specs[0], paired_init(0)["stream"])
    models = [pl.build_model(s, stream) for s in specs]
    for m in specs[0].module_list:
        assert np.array_equal(models[0].W[m], models[1].W[m])
        assert np.all(models[0].u[m] == 0.0)


def test_alignment_changes_the_init_but_gamma_does_not():
    stream = pl.build_stream(pl.Phase1Spec(**SMOKE), paired_init(0)["stream"])
    base = pl.build_model(pl.Phase1Spec(a=0.0, **SMOKE), stream)
    aligned = pl.build_model(pl.Phase1Spec(a=1.0, **SMOKE), stream)
    assert not np.allclose(base.W["A"], aligned.W["A"])


def test_forgetting_metrics_match_the_reference_sheet_worked_example():
    """`cl-metrics.md`: A 100→80 gives CF 20 / CFr 20%; B 40→30 gives CF 10 / CFr 25%.

    The two metrics rank the tasks oppositely, which is exactly why both are reported.
    """
    acc = np.array([[1.00, np.nan], [0.80, 0.40]])
    a = pl.forgetting_metrics(acc)["per_task"][0]
    assert a["CF"] == pytest.approx(0.20) and a["CFr"] == pytest.approx(0.20)

    acc_b = np.array([[0.40, np.nan], [0.30, 0.90]])
    b = pl.forgetting_metrics(acc_b)["per_task"][0]
    assert b["CF"] == pytest.approx(0.10) and b["CFr"] == pytest.approx(0.25)
    assert a["CF"] > b["CF"] and a["CFr"] < b["CFr"]


def test_forgetting_metrics_ignore_unmeasured_entries():
    acc = np.full((3, 3), np.nan)
    acc[0, 0], acc[2, 0] = 0.9, 0.5
    f = pl.forgetting_metrics(acc)
    assert list(f["per_task"]) == [0]
    assert f["CF"] == pytest.approx(0.4)


def test_probe_measures_include_a_graded_alternative_to_saturated_accuracy():
    """Accuracy saturates at sub-critical load; margin and held-out must be present."""
    spec = pl.Phase1Spec(**SMOKE)
    stream = pl.build_stream(spec, paired_init(0)["stream"])
    model = pl.build_model(spec, stream)
    d = pl.probe_decodability(model, stream.arrangements[0].points, stream.probe, ("A",))["A"]
    assert set(d) >= {"accuracy", "margin", "margin_p05", "heldout_manifold_accuracy"}
    assert 0.0 <= d["heldout_manifold_accuracy"] <= 1.0
    assert d["margin"] != pytest.approx(d["accuracy"])


def test_probe_margin_responds_to_a_representation_that_separates_better():
    """A representation containing the probe direction must beat a random one."""
    spec = pl.Phase1Spec(**SMOKE)
    stream = pl.build_stream(spec, paired_init(0)["stream"])
    model = pl.build_model(spec, stream)
    pts, y = stream.arrangements[0].points, stream.probe

    rand = pl.probe_decodability(model, pts, y, ("A",))["A"]
    centers = np.asarray([pts[mu].mean(axis=0) for mu in range(pts.shape[0])])
    direction = (centers[y > 0].mean(axis=0) - centers[y < 0].mean(axis=0))
    model.W["A"][0] = direction / np.linalg.norm(direction) * np.linalg.norm(model.W["A"][0])
    planted = pl.probe_decodability(model, pts, y, ("A",))["A"]
    assert planted["margin"] > rand["margin"]


def test_stream_id_keys_the_arrangement_not_the_init_seed():
    """stream_id is the arrangement draw; seed is init. They must not be confounded."""
    kw = dict(T=8, P=8, M=20, d=40, D=2)
    a = pl.build_stream(pl.Phase1Spec(stream_id=0, seed=0, **kw))
    b = pl.build_stream(pl.Phase1Spec(stream_id=1, seed=0, **kw))
    c = pl.build_stream(pl.Phase1Spec(stream_id=0, seed=7, **kw))
    d = pl.build_stream(pl.Phase1Spec(stream_id=0, seed=0, gamma_0=10.0, **kw))
    assert not np.array_equal(a.S_r, b.S_r)
    assert not np.array_equal(a.S_f, b.S_f)
    assert np.array_equal(a.S_r, c.S_r) and np.array_equal(a.S_f, c.S_f)
    assert np.array_equal(a.S_r, d.S_r) and np.array_equal(a.S_f, d.S_f)


def test_run_arm_is_json_ready_and_the_identity_closes(tmp_path):
    import json

    rec = pl.run_arm(pl.Phase1Spec(**SMOKE))
    json.dumps(rec)
    assert rec["n_evals"] == pl.n_evals(pl.schedule(4, 2), pl.Phase1Spec(**SMOKE))
    assert max(abs(g["identity_residual"]) for g in rec["geometry"]) < 1e-10
    assert all(abs(a["residual"]) < 1e-10 for a in rec["attribution"])
    assert {g["ensemble"] for g in rec["geometry"]} == {"generic", "retained"}
    assert all(a["lag"] > 0 for a in rec["attribution"])


def test_measurement_rng_is_shared_across_boundaries():
    """Two boundaries must differ because the representation moved, not because the
    estimator resampled `(y, t)`. Same task, same module, unchanged weights => equal."""
    spec = pl.Phase1Spec(**SMOKE)
    stream = pl.build_stream(spec, paired_init(0)["stream"])
    model = pl.build_model(spec, stream)
    args = (model, stream.arrangements[0].points, stream.dichotomies[0], spec, "A")
    a = pl._measure(*args, 0, 0, 12345)
    b = pl._measure(*args, 3, 0, 12345)
    assert a.alpha == pytest.approx(b.alpha, rel=1e-12)
    assert a.D_eff == pytest.approx(b.D_eff, rel=1e-12)
    c = pl._measure(*args, 3, 0, 999)
    assert c.alpha != pytest.approx(a.alpha, rel=1e-12)


def test_run_arm_refuses_l3():
    """A21: diagnostic SGD does not lift sequential arms."""
    spec = pl.Phase1Spec(n_hidden_layers=2, **SMOKE)
    with pytest.raises(NotImplementedError, match="L=3 training"):
        pl.run_arm(spec)
