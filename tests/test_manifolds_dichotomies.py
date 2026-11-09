"""Dichotomy / stream tests — `02-validation-suite.md` §5."""

from __future__ import annotations

import numpy as np
import pytest

from manifolds.dichotomies import (
    assert_balanced,
    hamming_distance,
    hamming_for_readout_similarity,
    readout_similarity,
    sample_at_hamming,
    sample_balanced,
)
from manifolds.labeling import make_labeling
from manifolds.streams import StreamConfig, make_stream


def test_dichotomies_balanced():
    rng = np.random.default_rng(0)
    for _ in range(50):
        y = sample_balanced(16, rng)
        assert_balanced(y)
        assert y.sum() == 0


def test_hamming_distances_even():
    rng = np.random.default_rng(1)
    for _ in range(30):
        y1 = sample_balanced(16, rng)
        y2 = sample_balanced(16, rng)
        assert hamming_distance(y1, y2) % 2 == 0


def test_sign_symmetry():
    rng = np.random.default_rng(2)
    y = sample_balanced(16, rng)
    assert readout_similarity(y, y) == pytest.approx(1.0)
    assert readout_similarity(y, -y) == pytest.approx(1.0)


def test_sample_at_hamming_exact():
    rng = np.random.default_rng(3)
    y = sample_balanced(16, rng)
    for h in (0, 2, 4, 8, 16):
        y2 = sample_at_hamming(y, h, rng)
        assert hamming_distance(y, y2) == h
        assert_balanced(y2)


def test_readout_similarity_target_via_hamming():
    """s_r from h matches the inversion within one Hamming step."""
    P = 16
    for s_r in (0.1, 0.5, 0.9):
        h = hamming_for_readout_similarity(P, s_r)
        # s_r = 1 - 2h/P for h ≤ P/2
        s_r_real = abs(1.0 - 2.0 * h / P)
        assert abs(s_r_real - s_r) <= 2.0 / P + 1e-12


def test_labeling_families_counts():
    lab = make_labeling(16)
    assert len(lab.all_single_factor()) == 4
    assert len(lab.all_xor()) == 6
    for y in lab.all_single_factor() + lab.all_xor():
        assert_balanced(y)


def test_stream_reproducible():
    cfg = StreamConfig(stream_id=7, condition="S-HL", T=8, d=40, M=20, D=2)
    s1 = make_stream(cfg, np.random.default_rng(100))
    s2 = make_stream(cfg, np.random.default_rng(100))
    assert np.array_equal(s1.dichotomies, s2.dichotomies)
    assert np.allclose(s1.arrangements[0].centers, s2.arrangements[0].centers)
    assert np.array_equal(s1.probe, s2.probe)


def test_legacy_shared_rng_draws_dichotomies_from_rng_first():
    """Pre-E16 behaviour, kept only to reproduce old streams: dichotomies come
    from `rng` itself, before the arrangements."""
    cfg = StreamConfig(stream_id=7, condition="S-HL", T=8, d=40, M=20, D=2)
    s_legacy = make_stream(cfg, np.random.default_rng(100), legacy_shared_rng=True)
    rng = np.random.default_rng(100)
    y0 = sample_balanced(cfg.P, rng)
    assert np.array_equal(s_legacy.dichotomies[0], y0)
    s_new = make_stream(cfg, np.random.default_rng(100))
    assert not np.array_equal(s_legacy.arrangements[0].points, s_new.arrangements[0].points)
    with pytest.raises(ValueError):
        make_stream(cfg, np.random.default_rng(100), legacy_shared_rng=True,
                    rng_dichotomies=np.random.default_rng(1))


def _cell(h, s_f=0.9, d=40, M=20, D=2, P=16, T=6):
    return StreamConfig(stream_id=0, condition="x", T=T, P=P, d=d, M=M, D=D,
                        feature_similarity=s_f, readout_similarity=1 - 2 * h / P)


def test_default_pairs_task0_across_readout_similarity():
    """E16: with no rng_dichotomies passed, task 0 (arrangement and labels) is
    identical across every Hamming level at a fixed seed."""
    task0 = []
    for h in (0, 2, 4, 6, 8):
        s = make_stream(_cell(h), np.random.default_rng(5))
        for t in range(1, 6):
            assert int(np.sum(s.dichotomies[t] != s.dichotomies[t - 1])) == h
        task0.append((s.arrangements[0].points, s.dichotomies[0]))
    for pts, y0 in task0[1:]:
        assert np.array_equal(pts, task0[0][0])
        assert np.array_equal(y0, task0[0][1])


def test_default_dichotomies_do_not_depend_on_geometry():
    """E16: the label sequence depends on the seed and h only, not on d, M, D or s_f."""
    ref = make_stream(_cell(4), np.random.default_rng(21)).dichotomies
    for kw in ({"d": 60}, {"M": 35}, {"D": 3}, {"s_f": 0.1}, {"s_f": 1.0}):
        assert np.array_equal(make_stream(_cell(4, **kw), np.random.default_rng(21)).dichotomies, ref), kw


def test_default_stream_deterministic_per_seed():
    a = make_stream(_cell(4), np.random.default_rng(33))
    b = make_stream(_cell(4), np.random.default_rng(33))
    c = make_stream(_cell(4), np.random.default_rng(34))
    assert np.array_equal(a.dichotomies, b.dichotomies)
    assert np.array_equal(a.arrangements[2].points, b.arrangements[2].points)
    assert not np.array_equal(a.arrangements[0].points, c.arrangements[0].points)


def test_separate_dichotomy_rng_pairs_task0_across_readout_similarity():
    P = 16
    task0 = []
    for h in (0, 2, 4, 6, 8):
        cfg = StreamConfig(stream_id=0, condition="x", T=6, P=P, d=40, M=20, D=2,
                           feature_similarity=0.9, readout_similarity=1 - 2 * h / P)
        s = make_stream(cfg, np.random.default_rng(5),
                        rng_dichotomies=np.random.default_rng(6))
        for t in range(1, cfg.T):
            assert int(np.sum(s.dichotomies[t] != s.dichotomies[t - 1])) == h
        task0.append((s.arrangements[0].points, s.dichotomies[0]))
    for pts, y0 in task0[1:]:
        assert np.array_equal(pts, task0[0][0])
        assert np.array_equal(y0, task0[0][1])


def test_readout_similarity_target_on_stream():
    cfg = StreamConfig(stream_id=1, condition="S-HL", T=8, d=40, M=20, D=2)
    stream = make_stream(cfg, np.random.default_rng(11))
    # S-HL: s_r = 0.1 between consecutive tasks
    for t in range(1, cfg.T):
        s_r = readout_similarity(stream.dichotomies[t - 1], stream.dichotomies[t])
        # within one Hamming step of target 0.1
        assert abs(s_r - 0.1) <= 2.0 / cfg.P + 1e-9


def test_probe_never_trained():
    cfg = StreamConfig(stream_id=2, condition="S-HH", T=8, d=40, M=20, D=2)
    stream = make_stream(cfg, np.random.default_rng(12))
    for y in stream.dichotomies:
        assert not np.array_equal(stream.probe, y)
        assert not np.array_equal(stream.probe, -y)


def test_similarity_matrices_recorded():
    cfg = StreamConfig(stream_id=3, condition="S-LL", T=6, d=30, M=15, D=2)
    stream = make_stream(cfg, np.random.default_rng(13))
    assert stream.S_f.shape == (6, 6)
    assert stream.S_r.shape == (6, 6)
    assert np.allclose(np.diag(stream.S_f), 1.0)
    assert np.allclose(np.diag(stream.S_r), 1.0)
