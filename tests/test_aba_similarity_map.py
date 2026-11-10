"""docs/34 similarity map (Phase 1, amended 2026-11-10): levels, specs, key fields and the
task-vector similarity of §4. Toy scale; no map arm runs here."""
from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

import numpy as np

CORE = Path(__file__).resolve().parents[1] / "notebooks" / "core"
sys.path.insert(0, str(CORE))
import aba_lib as L  # noqa: E402

TOY = L.ABAConfig(quick=True, P=16, d=20, D=2, M=10, N=32, n_t=10, ntk_per_manifold=5,
                  refit_steps=50, bottleneck_n=8)


def test_map_levels_and_phase1_specs():
    cfg = L.map_config(L.ABAConfig.make(False))
    assert cfg.hammings == (0, 2, 4, 6, 8, 12, 16)
    assert cfg.s_fs == (1.0, 0.95, 0.9, 0.8, 0.6, 0.4, 0.2, 0.1)
    assert list(cfg.seeds) == list(range(16, 24))
    specs = L.phase1_specs(cfg)
    assert len(specs) == len({s.key for s in specs}) == 768 + 3
    design = [s for s in specs if s.noise == "copy"]
    cells = {(s.hamming, s.s_f) for s in design}
    assert len(cells) == 7 * 2 + 8 * 3 - 6 == 32
    assert {(h, sf) for h in cfg.hammings for sf in (1.0, 0.9)} <= cells
    assert {(h, sf) for h in (0, 4, 8) for sf in cfg.s_fs} <= cells
    assert sum(s.noise == "fresh" for s in specs) == 3


def test_every_b_label_is_balanced_at_its_hamming_and_h16_flips():
    cfg = dataclasses.replace(L.map_config(TOY))
    for seed in (16, 17):
        base = L.seed_base(cfg, seed)
        for h, y in base.y_Bs.items():
            assert int(y.sum()) == 0 and int(np.sum(y != base.y_A)) == h
        assert np.array_equal(base.y_Bs[16], -base.y_A)
        assert L.s_r(base.y_A, base.y_Bs[16]) == -1.0


def test_key_fields_carry_probe_hammings_only_off_default():
    assert "probe_hammings" not in L.ABAConfig.make(False).key_fields()
    assert "probe_hammings" not in L.ABAConfig.make(True).key_fields()
    kf = L.map_config(L.ABAConfig.make(False)).key_fields()
    assert json.loads(json.dumps(kf))["probe_hammings"] == [0, 2, 4, 6, 8, 12, 16]


def test_default_seed_base_draws_unchanged_by_the_map_levels():
    """H4/H8 B labels use their own streams; y* and the neutral probes depend on the level set."""
    d, m = L.seed_base(TOY, 3), L.seed_base(L.map_config(TOY), 3)
    for h in (0, 4, 8):
        assert np.array_equal(d.y_Bs[h], m.y_Bs[h])
    assert np.array_equal(d.y_A, m.y_A) and np.array_equal(d.W0, m.W0)


def test_task_vector_is_min_norm_least_squares():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((40, 60))          # under-determined: min-norm matters
    y = np.sign(rng.standard_normal(40))
    w = L.task_vector(X, y)
    assert np.allclose(w, np.linalg.pinv(X) @ y)
    X2 = rng.standard_normal((80, 10))         # over-determined: the least-squares solution
    assert np.allclose(L.task_vector(X2, y.repeat(2)), np.linalg.lstsq(X2, y.repeat(2), rcond=None)[0])


def test_task_vector_cos_at_identity_and_flip():
    cfg = L.map_config(TOY)
    base = L.seed_base(cfg, 16)
    XA, yA = L.flatten(base.A.points, base.y_A)
    assert np.isclose(L.task_vector_cos(XA, yA, XA, yA), 1.0)
    _, yF = L.flatten(base.A.points, base.y_Bs[16])
    assert np.isclose(L.task_vector_cos(XA, yA, XA, yF), -1.0)


def test_run_arm_records_taskvec_and_h0_is_gamma_independent():
    cfg = L.map_config(TOY)
    rows = [L.run_arm(cfg, L.ArmSpec(16, g, 6, 0.6), stamp={"test": True}) for g in (0.1, 10.0)]
    for r in rows:
        bad = [b for b in L.check_schema(r) if "taskvec" in b or "rho_in" in b]   # toy refit < checkpoints
        assert not bad, bad
        assert -1.0 <= r["relations"]["taskvec_cos_input"] <= 1.0
    assert rows[0]["relations"]["taskvec_cos_h0"] == rows[1]["relations"]["taskvec_cos_h0"]
    assert rows[0]["relations"]["taskvec_cos_input"] == rows[1]["relations"]["taskvec_cos_input"]
    assert rows[0]["config"]["probe_hammings"] == [0, 2, 4, 6, 8, 12, 16]
