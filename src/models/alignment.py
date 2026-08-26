"""Alignment initialization — the wealth knob (`00` §5).

Varies the overlap between `W_m(0)`'s row space and the manifold-center subspace
at **fixed Frobenius norm and fixed rank**, so that `a` is orthogonal to `γ`.

Interpolation is along the **Grassmann geodesic**, never linear-then-reorthonormalize:
the latter makes `a` a nonlinear, seed-dependent parameterization and destroys
comparability across seeds (`AGENTS.md`, Grassmann rule).

Scope note: the `a` axis is **cut from the Phase 1 factorial** (D4 pre-authorized).
This module exists because Gate 2 (`02` §4 —
`test_alignment_moves_initial_capacity`) is still to be run and recorded for the
full paper, and it needs a correct geodesic to be a meaningful test.
"""

from __future__ import annotations

import numpy as np

from src.numerics import clip_to_noise
RCOND = 1e-10


def center_subspace(centers: np.ndarray, k: int | None = None, *, D: int | None = None):
    """Top-`k` right singular vectors of the `P × d` center matrix (`00` §5.1).

    Default `k = min(P − 1, 2D)`; `k` is returned so it can be recorded in the
    config, as the spec requires.
    """
    C = np.asarray(centers, dtype=np.float64)
    P, d = C.shape
    if k is None:
        if D is None:
            raise ValueError("pass k explicitly, or D so that k = min(P-1, 2D)")
        k = min(P - 1, 2 * D)
    k = max(1, min(k, min(P, d)))
    _, _, Vt = np.linalg.svd(C, full_matrices=False)
    return np.ascontiguousarray(Vt[:k].T), k


def grassmann_geodesic(Y: np.ndarray, U: np.ndarray, a: float) -> np.ndarray:
    """Geodesic from `span(Y)` to `span(U)` on the Grassmannian (`00` §5.2).

        A = (I − Y Yᵀ) U (Yᵀ U)†      thin SVD A = Q_A Σ_A Z_Aᵀ
        Θ = arctan(Σ_A)               Y(a) = Y Z_A cos(aΘ) + Q_A sin(aΘ)

    `Y(0) = Y Z_A` spans `span(Y)`; `Y(1)` spans `span(U)`. `(Yᵀ U)` is inverted
    with an explicit-rcond pseudo-inverse because it is near-singular when the
    subspaces are nearly orthogonal.
    """
    Y = np.asarray(Y, dtype=np.float64)
    U = np.asarray(U, dtype=np.float64)
    if Y.shape != U.shape:
        raise ValueError(f"Y and U must have the same shape; got {Y.shape}, {U.shape}")
    if not (0.0 <= a <= 1.0):
        raise ValueError(f"a must be in [0, 1]; got {a}")

    YtU = Y.T @ U
    A = (U - Y @ YtU) @ np.linalg.pinv(YtU, rcond=RCOND)
    Q_A, s_A, Zt_A = np.linalg.svd(A, full_matrices=False)
    theta = np.arctan(s_A)
    if not np.all(np.isfinite(theta)):
        raise RuntimeError("non-finite principal angles; (YᵀU) is degenerate")

    Yg = Y @ Zt_A.T @ np.diag(np.cos(a * theta)) + Q_A @ np.diag(np.sin(a * theta))
    # Re-orthonormalize against float error only; the geodesic is orthonormal
    # analytically, so this must be a no-op up to rounding.
    Q, _ = np.linalg.qr(Yg)
    return np.ascontiguousarray(Q)


def principal_angles(Y: np.ndarray, U: np.ndarray) -> np.ndarray:
    """Principal angles between two subspaces, ascending.

    The singular values are cosines and so are analytically in [0, 1]; the clip absorbs
    float error only, and says so, because a value materially above 1 means the inputs
    are not orthonormal and the angles would be meaningless rather than slightly off.
    """
    s = np.linalg.svd(np.asarray(Y).T @ np.asarray(U), compute_uv=False)
    s = clip_to_noise(s, -1.0, 1.0, atol=1e-8,
                      what="singular value of YᵀU (cosine of a principal angle)")
    return np.arccos(s)[::-1].copy()


def aligned_init(
    W_shape: np.ndarray,
    U_C: np.ndarray,
    a: float,
) -> np.ndarray:
    """Rotate `W`'s row space toward the center subspace at alignment `a` (`00` §5.1).

    Steps 3–6: thin SVD `W̃ = Q Σ Vᵀ`; move the first `k` columns of `V` along the
    geodesic toward `U_C`; re-orthonormalize the *remaining* columns against them
    in order; reassemble with `Σ` untouched, so Frobenius norm and rank are
    preserved exactly.
    """
    W = np.asarray(W_shape, dtype=np.float64)
    U_C = np.asarray(U_C, dtype=np.float64)
    k = U_C.shape[1]
    Q, S, Vt = np.linalg.svd(W, full_matrices=False)
    V = Vt.T
    if k > V.shape[1]:
        raise ValueError(f"k={k} exceeds the rank budget {V.shape[1]} of W")

    V_new = V.copy()
    V_new[:, :k] = grassmann_geodesic(V[:, :k], U_C, a)

    # Gram-Schmidt the trailing columns against the rotated block, preserving order.
    for j in range(k, V_new.shape[1]):
        v = V_new[:, j]
        v = v - V_new[:, :j] @ (V_new[:, :j].T @ v)
        v = v - V_new[:, :j] @ (V_new[:, :j].T @ v)  # one reorthogonalization pass
        nrm = np.linalg.norm(v)
        if nrm < 1e-12:
            raise RuntimeError(f"trailing column {j} collapsed during Gram-Schmidt")
        V_new[:, j] = v / nrm

    return Q @ np.diag(S) @ V_new.T
