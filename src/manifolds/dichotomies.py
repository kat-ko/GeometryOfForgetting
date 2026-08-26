"""Balanced dichotomies and readout similarity.

Spec: `docs/00-math-spec.md` §2.2–§2.3. Invariant I9: dichotomies must be balanced.
"""

from __future__ import annotations

import numpy as np


def assert_balanced(y: np.ndarray) -> None:
    y = np.asarray(y)
    if y.ndim != 1:
        raise ValueError(f"dichotomy must be 1-d; got shape {y.shape}")
    if y.size % 2 != 0:
        raise ValueError(f"P={y.size} must be even for a balanced dichotomy")
    if not np.all(np.isin(y, [-1, 1])):
        raise ValueError(f"dichotomy entries must be ±1; got {np.unique(y)}")
    if int(y.sum()) != 0:
        raise ValueError(f"dichotomy not balanced: sum={y.sum()}")


def sample_balanced(P: int, rng: np.random.Generator) -> np.ndarray:
    """Uniform over balanced dichotomies: choose P/2 indices for +1."""
    if P % 2 != 0:
        raise ValueError(f"P must be even; got {P}")
    y = -np.ones(P, dtype=np.int8)
    plus = rng.choice(P, size=P // 2, replace=False)
    y[plus] = 1
    assert_balanced(y)
    return y


def hamming_distance(y1: np.ndarray, y2: np.ndarray) -> int:
    """Number of positions that differ. Always even between balanced dichotomies."""
    y1 = np.asarray(y1)
    y2 = np.asarray(y2)
    if y1.shape != y2.shape:
        raise ValueError("shape mismatch")
    return int(np.sum(y1 != y2))


def readout_similarity(y1: np.ndarray, y2: np.ndarray) -> float:
    """s_r = |2 · overlap / P − 1|, overlap = #{i : y_i = y'_i}.

    Sign-symmetric: s_r(y, y) = s_r(y, −y) = 1.
    """
    y1 = np.asarray(y1)
    y2 = np.asarray(y2)
    P = y1.size
    overlap = int(np.sum(y1 == y2))
    return float(abs(2.0 * overlap / P - 1.0))


def sample_at_hamming(
    y: np.ndarray,
    h: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return a balanced dichotomy at Hamming distance ``h`` from ``y``.

    Achieved by swapping ``h/2`` of the +1 indices with ``h/2`` of the −1
    indices. ``h`` must be even and in ``{0, 2, …, P}``.
    """
    y0 = np.asarray(y, dtype=np.int8)
    assert_balanced(y0)
    out = y0.copy()
    P = out.size
    if h < 0 or h > P or h % 2 != 0:
        raise ValueError(f"h must be even in [0, P]; got h={h}, P={P}")
    if h == 0:
        return out
    n_swap = h // 2
    plus = np.flatnonzero(out == 1)
    minus = np.flatnonzero(out == -1)
    take_plus = rng.choice(plus, size=n_swap, replace=False)
    take_minus = rng.choice(minus, size=n_swap, replace=False)
    out[take_plus] = -1
    out[take_minus] = 1
    assert_balanced(out)
    d = hamming_distance(y0, out)
    if d != h:
        raise RuntimeError(f"internal error: requested h={h}, got {d}")
    return out


def hamming_for_readout_similarity(P: int, s_r: float) -> int:
    """Invert s_r = |2·overlap/P − 1| to an even Hamming distance.

    For h ∈ [0, P/2], s_r = 1 − 2h/P ⇒ h = (P/2)·(1 − s_r). Round to nearest even.
    """
    if not (0.0 <= s_r <= 1.0):
        raise ValueError(f"s_r must be in [0, 1]; got {s_r}")
    h_float = 0.5 * P * (1.0 - s_r)
    h = int(round(h_float))
    if h % 2 == 1:
        h_lo, h_hi = h - 1, h + 1
        h = h_lo if abs(h_lo - h_float) <= abs(h_hi - h_float) else h_hi
    h = max(0, min(P, h))
    if h % 2 == 1:
        h -= 1
    return h
