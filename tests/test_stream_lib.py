"""notebooks/core/stream_lib.py (docs/29 §8, Kati's conditions 2–4), at toy scale.

(j, t) indexing, realised ρ_in, the fresh-learner baseline, cache-key coverage, neutral
probes, and parity with aba_lib (an A–B–A arm's A and B as a T = 2 stream, bit for bit).
"""
from __future__ import annotations

import math
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

CORE = Path(__file__).resolve().parents[1] / "notebooks" / "core"
sys.path.insert(0, str(CORE))
import aba_lib as L  # noqa: E402
import stream_lib as S  # noqa: E402

TOY = L.ABAConfig(quick=True, P=8, d=20, D=2, M=10, N=32, n_t=10, ntk_per_manifold=5,
                  refit_steps=50, bottleneck_n=8)
T = 3
GAMMA = 1.0


@pytest.fixture(scope="module")
def drift():
    inp = S.inputs_from_stream(TOY, draw=3, hamming=2, s_f=0.9, T=T)
    return inp, S.run_stream(TOY, inp, GAMMA, S.StreamSpec(3, GAMMA, 2, 0.9), stamp={"test": True})


def test_backward_keys_are_every_j_le_t_plus_init(drift):
    _, row = drift
    want = {f"{j},init" for j in range(T)} | {f"{j},{t}" for t in range(T) for j in range(t + 1)}
    assert set(row["geometry"]["backward"]) == want
    want_ch = set()
    for j in range(T):
        want_ch.add(f"{j},init>{j}")
        for t in range(j + 1, T):
            want_ch |= {f"{j},{t - 1}>{t}", f"{j},{j}>{t}"}
    assert set(row["backward"]) == want_ch
    assert set(row["forward"]) == {str(t) for t in range(T - 1)}


def test_j_t_is_task_j_measured_after_training_task_t(drift):
    inp, row = drift
    net = L.OneModuleNet(inp.W0, GAMMA, TOY)
    for t in range(2):
        L.train_to_criterion(net, *L.flatten(inp.arrs[t].points, inp.ys[t]), TOY)
    direct = L.rep_capacity(net, inp.arrs[0].points, inp.ys[0], TOY, inp.glue_key)
    assert direct == row["geometry"]["backward"]["0,1"]
    nxt = L.rep_capacity(net, inp.arrs[2].points, inp.ys[2], TOY, inp.glue_key)
    assert nxt == row["geometry"]["forward_after"]["1"]["next"]


def test_forward_total_is_against_init_on_next_tasks_manifolds(drift):
    _, row = drift
    G = row["geometry"]
    for t in range(T - 1):
        tot = L.channels(G["backward"][f"{t + 1},init"], G["forward_after"][str(t)]["next"])
        assert tot == row["forward"][str(t)]["total"]
        neu = [math.log(a["alpha"] / b["alpha"])
               for a, b in zip(G["forward_after"][str(t)]["neutral"], G["forward_neutral_init"][str(t)])]
        assert row["forward"][str(t)]["specific"] == tot["total"] - float(np.mean(neu))


def test_neutral_probes_are_orthogonal_to_y_t_and_novel(drift):
    inp, row = drift
    for t in range(T - 1):
        assert row["relations"]["neutral_s_r_to_y_t"][t] == [0.0] * TOY.n_probe_neutral
        banned = [*inp.ys, *[-y for y in inp.ys]]
        for k, yk in enumerate(inp.neutral[t]):
            assert not any(np.array_equal(yk, b) for b in banned)
            assert not any(np.array_equal(yk, o) or np.array_equal(yk, -o) for o in inp.neutral[t][:k])


def test_rho_in_is_realised_centre_cosine_not_nominal(drift):
    inp, row = drift
    rho = np.asarray(row["relations"]["rho_in"])
    for i in range(T):
        for j in range(T):
            c = L.centre_cosines(inp.arrs[i].centers, inp.arrs[j].centers)
            assert rho[i, j] == float(np.mean(np.diag(c)))
    assert np.allclose(np.diag(rho), 1.0)
    assert np.allclose(rho, rho.T)
    lag1 = [rho[t, t + 1] for t in range(T - 1)]
    assert all(abs(v - 0.9) < 0.3 for v in lag1)
    assert any(v != 0.9 for v in lag1)          # realised, not the nominal s_f


def test_rho_in_is_one_under_frozen_input():
    inp = S.inputs_from_stream(TOY, draw=4, hamming=2, s_f=1.0, T=T)
    for i in range(T):
        for j in range(T):
            assert np.isclose(np.mean(np.diag(L.centre_cosines(inp.arrs[i].centers, inp.arrs[j].centers))), 1.0)


def test_realised_s_r_matches_hamming(drift):
    _, row = drift
    for t in range(T - 1):
        assert row["relations"]["S_r"][t][t + 1] == 1 - 2 * 2 / TOY.P


def test_fresh_learner_baseline(drift):
    inp, row = drift
    for t in range(T - 1):
        fnet = L.OneModuleNet(inp.W0, GAMMA, TOY)
        ph = L.train_to_criterion(fnet, *L.flatten(inp.arrs[t + 1].points, inp.ys[t + 1]), TOY)
        assert row["behaviour"]["fresh_steps"][t] == ph["steps"]
        assert row["forward"][str(t)]["FT_beh"] == math.log(
            (1 + ph["steps"]) / (1 + row["behaviour"]["steps"][t + 1]))


def test_stream_hash_covers_every_dependency(tmp_path, monkeypatch):
    names = {p.name for p in S.HASHED_FILES}
    assert {"aba_lib.py", "stream_lib.py", "streams.py", "generator.py", "dichotomies.py"} <= names
    copies = []
    for p in S.HASHED_FILES:
        q = tmp_path / p.name
        shutil.copyfile(p, q)
        copies.append(q)
    monkeypatch.setattr(S, "HASHED_FILES", copies)
    h0 = S.stream_hash()
    assert h0 == S.stream_hash(copies)
    for q in copies:
        orig = q.read_bytes()
        q.write_bytes(orig + b"\n# edit\n")
        assert S.stream_hash() != h0, q.name
        q.write_bytes(orig)
    assert S.stream_hash() == h0


def test_parity_with_aba_lib_at_toy_scale():
    import stream_cluster as SC
    spec = L.ArmSpec(0, 1.0, 4, 0.9)
    aba = L.run_arm(TOY, spec, stamp={"test": True})
    st = S.run_stream(TOY, S.inputs_from_aba(TOY, spec), spec.gamma, stamp={"test": True}, floor=False)
    n, diffs = SC.parity_diffs(aba, st)
    assert n > 30
    assert diffs == []


def test_stream_refuses_fixed_budget():
    import dataclasses
    cfg = dataclasses.replace(TOY, stopping="fixed", fixed_steps=(5, 5, 5))
    inp = S.inputs_from_stream(TOY, draw=3, hamming=2, s_f=0.9, T=2)
    with pytest.raises(ValueError):
        S.run_stream(cfg, inp, GAMMA)
