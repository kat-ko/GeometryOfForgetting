"""Task streams with controlled (t, t−1) feature/readout similarity.

Spec: `docs/00-math-spec.md` §3; config in `docs/01-experiments.md` §3.
Streams are generated once per ``stream_id`` from ``rng_stream`` and shared
across configs (blocking factor). ``make_stream`` itself never reads
``cfg.stream_id`` — the caller must key ``rng`` by ``stream_id``
(``pipeline.stream_rng``). Passing ``paired_init(seed)["stream"]`` makes
``stream_id`` a stored no-op.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .dichotomies import (
    assert_balanced,
    hamming_for_readout_similarity,
    readout_similarity,
    sample_at_hamming,
    sample_balanced,
)
from .generator import Arrangement, make_arrangement, redraw_centers_correlated
from .labeling import FactorLabeling, make_labeling


# Hiratani 2×2 corners (01 §3.1)
HIRATANI = {
    "S-HH": (0.9, 0.9),
    "S-HL": (0.9, 0.1),  # catastrophic corner
    "S-LH": (0.1, 0.9),
    "S-LL": (0.1, 0.1),
}


@dataclass(frozen=True)
class StreamConfig:
    stream_id: int
    condition: str  # e.g. "S-HL" or "S-fixed-r"
    T: int = 16
    P: int = 16
    d: int = 150
    M: int = 150
    D: int = 4
    R: float = 1.0
    feature_similarity: float | None = None  # overrides condition if set
    readout_similarity: float | None = None
    probe_target_s_r: float = 0.5  # target s_r(y*, y_t) typical distance
    rho_A: float = 0.0
    psi_gen: float = 0.0
    manifold_kind: str = "spherical"


@dataclass(frozen=True)
class Stream:
    """One realized task stream.

    Attributes
    ----------
    dichotomies : (T, P) int8
    arrangements : list of Arrangement, length T
    S_f, S_r : (T, T) float64 full similarity matrices
    probe : (P,) int8 held-out dichotomy y* (never in training tasks)
    probe_s_r : (T,) float64 s_r(y*, y_t) for each task
    """

    config: StreamConfig
    labeling: FactorLabeling
    dichotomies: np.ndarray
    arrangements: tuple[Arrangement, ...]
    S_f: np.ndarray
    S_r: np.ndarray
    probe: np.ndarray
    probe_s_r: np.ndarray


def _resolve_similarities(cfg: StreamConfig) -> tuple[float, float]:
    if cfg.feature_similarity is not None and cfg.readout_similarity is not None:
        return float(cfg.feature_similarity), float(cfg.readout_similarity)
    if cfg.condition in HIRATANI:
        return HIRATANI[cfg.condition]
    if cfg.condition == "S-fixed-r":
        if cfg.readout_similarity is None:
            raise ValueError("S-fixed-r requires readout_similarity")
        return 1.0, float(cfg.readout_similarity)
    raise ValueError(f"unknown stream condition {cfg.condition!r}")


def make_stream(cfg: StreamConfig, rng: np.random.Generator) -> Stream:
    """Build a stream with controlled s_f(t,t−1) and s_r(t,t−1).

    Only consecutive similarities are controlled; the full T×T matrices are
    recorded as covariates. The probe y* is sampled once and verified not to
    appear (up to sign) among training dichotomies.
    """
    s_f, s_r = _resolve_similarities(cfg)
    labeling = make_labeling(cfg.P)
    h = hamming_for_readout_similarity(cfg.P, s_r)

    # --- dichotomies ---
    ys = np.zeros((cfg.T, cfg.P), dtype=np.int8)
    ys[0] = sample_balanced(cfg.P, rng)
    for t in range(1, cfg.T):
        ys[t] = sample_at_hamming(ys[t - 1], h, rng)

    # --- arrangements ---
    arrs: list[Arrangement] = []
    arrs.append(
        make_arrangement(
            cfg.P,
            cfg.d,
            cfg.D,
            cfg.R,
            cfg.M,
            rng,
            rho_C=0.0,
            rho_A=cfg.rho_A,
            psi_gen=cfg.psi_gen,
            kind=cfg.manifold_kind,
        )
    )
    for t in range(1, cfg.T):
        arrs.append(redraw_centers_correlated(arrs[t - 1], s_f, rng))

    # --- full similarity matrices ---
    S_r = np.zeros((cfg.T, cfg.T), dtype=np.float64)
    for i in range(cfg.T):
        for j in range(cfg.T):
            S_r[i, j] = readout_similarity(ys[i], ys[j])

    S_f = np.zeros((cfg.T, cfg.T), dtype=np.float64)
    # Controlled values on the sub-/super-diagonal; elsewhere: realized
    # center-correlation proxy (mean pairwise cosine of centers).
    for i in range(cfg.T):
        for j in range(cfg.T):
            if i == j:
                S_f[i, j] = 1.0
            elif abs(i - j) == 1:
                S_f[i, j] = s_f
            else:
                S_f[i, j] = _center_correlation(arrs[i].centers, arrs[j].centers)

    # --- held-out probe ---
    probe = _sample_probe(ys, cfg.P, cfg.probe_target_s_r, rng)
    probe_s_r = np.array(
        [readout_similarity(probe, ys[t]) for t in range(cfg.T)],
        dtype=np.float64,
    )

    return Stream(
        config=cfg,
        labeling=labeling,
        dichotomies=ys,
        arrangements=tuple(arrs),
        S_f=S_f,
        S_r=S_r,
        probe=probe,
        probe_s_r=probe_s_r,
    )


def _center_correlation(c1: np.ndarray, c2: np.ndarray) -> float:
    """Mean pairwise cosine between corresponding centers (proxy for long-range s_f)."""
    n1 = np.linalg.norm(c1, axis=1)
    n2 = np.linalg.norm(c2, axis=1)
    safe = (n1 > 0) & (n2 > 0)
    if not np.any(safe):
        return 0.0
    cos = np.sum(c1[safe] * c2[safe], axis=1) / (n1[safe] * n2[safe])
    return float(np.mean(cos))


def _sample_probe(
    ys: np.ndarray,
    P: int,
    target_s_r: float,
    rng: np.random.Generator,
    *,
    max_tries: int = 10_000,
) -> np.ndarray:
    """Sample y* not equal (up to sign) to any training dichotomy."""
    training = list(ys) + [(-y).astype(np.int8) for y in ys]
    h = hamming_for_readout_similarity(P, target_s_r)
    # Anchor off the first task at the target Hamming distance, then reject
    # collisions with the stream.
    for _ in range(max_tries):
        cand = sample_at_hamming(ys[0], h, rng)
        if any(np.array_equal(cand, t) for t in training):
            continue
        return cand
    # Fallback: keep drawing uniform balanced until unique.
    for _ in range(max_tries):
        cand = sample_balanced(P, rng)
        if any(np.array_equal(cand, t) for t in training):
            continue
        return cand
    raise RuntimeError("failed to sample a probe dichotomy distinct from the stream")


def recurrence_control_rng(stream_id: int, k: int) -> np.random.Generator:
    """Independent of `stream_rng`. Control sampling must not consume arrangement draws."""
    return np.random.default_rng([20260817, int(stream_id), int(k), 0xA8])


def replace_task_dichotomy(stream: Stream, t: int, y: np.ndarray) -> Stream:
    """Rewrite dichotomy t. Arrangements stay put (A8: same arrangement)."""
    y = np.asarray(y, dtype=np.int8)
    assert_balanced(y)
    ys = stream.dichotomies.copy()
    if t < 0 or t >= len(ys):
        raise ValueError(f"task index {t} out of range for T={len(ys)}")
    if y.shape != (ys.shape[1],):
        raise ValueError(f"dichotomy shape {y.shape} != {(ys.shape[1],)}")
    ys[t] = y
    T = ys.shape[0]
    S_r = np.empty((T, T), dtype=np.float64)
    for i in range(T):
        for j in range(T):
            S_r[i, j] = readout_similarity(ys[i], ys[j])
    probe_s_r = np.array(
        [readout_similarity(stream.probe, ys[i]) for i in range(T)],
        dtype=np.float64,
    )
    return Stream(
        config=stream.config,
        labeling=stream.labeling,
        dichotomies=ys,
        arrangements=stream.arrangements,
        S_f=stream.S_f,
        S_r=S_r,
        probe=stream.probe,
        probe_s_r=probe_s_r,
    )


def apply_recurrence(
    stream: Stream,
    *,
    k: int,
    mode: str,
    rng: np.random.Generator,
) -> Stream:
    """Re-present y_0 at index k, or a novel y_k matched on s_r to y_{k−1}.

    Prefix dichotomies 0..k−1 and every arrangement are unchanged.
    """
    T = stream.config.T
    if k < 1 or k >= T:
        raise ValueError(f"recurrence index k={k} must be in [1, {T})")
    y0 = stream.dichotomies[0]
    y_prev = stream.dichotomies[k - 1]
    target = readout_similarity(y0, y_prev)
    if target >= 1.0 - 1e-12:
        raise ValueError(
            f"s_r(y_0, y_{k - 1})=1; re-presentation at k={k} is degenerate"
        )
    if mode == "represent":
        return replace_task_dichotomy(stream, k, y0)
    if mode != "control":
        raise ValueError(f"unknown recurrence mode {mode!r}")
    h = hamming_for_readout_similarity(stream.config.P, target)
    blocked = list(stream.dichotomies[:k]) + [
        (-y).astype(np.int8) for y in stream.dichotomies[:k]
    ]
    for _ in range(10_000):
        cand = sample_at_hamming(y_prev, h, rng)
        if any(np.array_equal(cand, b) for b in blocked):
            continue
        if abs(readout_similarity(cand, y_prev) - target) > 1e-12:
            continue
        return replace_task_dichotomy(stream, k, cand)
    raise RuntimeError("failed to sample a novel control dichotomy matched on s_r")

