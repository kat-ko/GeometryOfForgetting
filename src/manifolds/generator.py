"""Synthetic manifold generator — Chou et al. ICML 2025 App. D.1.1.

Spec: `docs/00-math-spec.md` §1. Source sheet: `docs/reference/manifold-generator.md`
(agent-verified; blank `Verified by (human)` — do not treat as code-citeable until signed).

All draws go through an explicit ``numpy.random.Generator`` (AGENTS.md §4).
Geometry arrays are float64.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

NOISE_EPS = 1e-2


@dataclass(frozen=True)
class Arrangement:
    """One fixed arrangement of P manifolds in R^d.

    Attributes
    ----------
    centers : (P, d) float64
    axes : (P, D, d) float64
        ``axes[i, j, :]`` is the j-th axis of manifold i.
    coords : (P, M, D) float64
        Intrinsic coordinates per point (drawn once; reused when only noise
        is resampled).
    points : (P, M, d) float64
        Realized point clouds: center + R * unit-normalized pre-scaled + ε noise.
    """

    centers: np.ndarray
    axes: np.ndarray
    coords: np.ndarray
    points: np.ndarray
    P: int
    d: int
    D: int
    M: int
    R: float
    rho_C: float
    rho_A: float
    psi_gen: float
    kind: str = "spherical"


def ar_covariance(P: int, rho: float) -> np.ndarray:
    """Autoregressive covariance C(ρ)_{ij} = ρ^{|i-j|}."""
    if not (0.0 <= rho < 1.0):
        raise ValueError(f"rho must be in [0, 1); got {rho}")
    idx = np.arange(P)
    return np.power(rho, np.abs(idx[:, None] - idx[None, :])).astype(np.float64)


def _gaussian_rows(rng: np.random.Generator, n_rows: int, d: int) -> np.ndarray:
    """Rows ~ N(0, I_d / d)."""
    return rng.standard_normal((n_rows, d)).astype(np.float64) / np.sqrt(d)


def _apply_cholesky_left(matrix: np.ndarray, chol: np.ndarray) -> np.ndarray:
    """Left-multiply a (P, …) array by Cholesky factor of C along the P axis."""
    P = matrix.shape[0]
    rest = matrix.shape[1:]
    flat = matrix.reshape(P, -1)
    return (chol @ flat).reshape((P, *rest))


def make_arrangement(
    P: int,
    d: int,
    D: int,
    R: float,
    M: int,
    rng: np.random.Generator,
    *,
    rho_C: float = 0.0,
    rho_A: float = 0.0,
    psi_gen: float = 0.0,
    eps: float = NOISE_EPS,
    kind: str = "spherical",
) -> Arrangement:
    """Build P manifolds in R^d.

    ``kind="spherical"`` is D.1.1 primary: intrinsic dim D, unit-normalized
    pre-scale, plus ε noise. ``kind="isotropic_gaussian"`` is the D.1.1
    variant ``M_i = {u₀ + R·v_k}`` — no intrinsic dimension, no unit-norm
    (unit-norm would put points on a sphere of dim d−1).
    """
    if kind not in ("spherical", "isotropic_gaussian"):
        raise ValueError(f"unknown manifold kind {kind!r}")
    if P < 2 or d < 1 or M < 1:
        raise ValueError(f"invalid sizes P={P}, d={d}, D={D}, M={M}")
    if R <= 0:
        raise ValueError(f"R must be positive; got {R}")
    if kind == "spherical" and D < 1:
        raise ValueError(f"spherical manifolds need D ≥ 1; got {D}")
    if kind == "isotropic_gaussian" and rho_A != 0.0:
        raise ValueError("isotropic Gaussian has no axes; rho_A must be 0")

    centers = _gaussian_rows(rng, P, d)  # (P, d)

    if kind == "isotropic_gaussian":
        if rho_C > 0.0:
            chol_C = np.linalg.cholesky(ar_covariance(P, rho_C))
            centers = _apply_cholesky_left(centers, chol_C)
        if psi_gen != 0.0:
            q = rng.standard_normal(P).astype(np.float64)
            centers = centers * (1.0 + psi_gen * q)[:, None]
        points = _realize_isotropic(centers, R, M, rng)
        axes = np.zeros((P, 0, d), dtype=np.float64)
        coords = np.zeros((P, M, 0), dtype=np.float64)
        return Arrangement(
            centers=centers, axes=axes, coords=coords, points=points,
            P=P, d=d, D=0, M=M, R=float(R),
            rho_C=float(rho_C), rho_A=0.0, psi_gen=float(psi_gen),
            kind="isotropic_gaussian",
        )

    axes = np.stack(
        [_gaussian_rows(rng, P, d) for _ in range(D)],
        axis=1,
    )  # (P, D, d)

    if rho_C > 0.0:
        chol_C = np.linalg.cholesky(ar_covariance(P, rho_C))
        centers = _apply_cholesky_left(centers, chol_C)
    if rho_A > 0.0:
        chol_A = np.linalg.cholesky(ar_covariance(P, rho_A))
        for j in range(D):
            axes[:, j, :] = _apply_cholesky_left(axes[:, j, :], chol_A)

    if psi_gen != 0.0:
        q = rng.standard_normal(P).astype(np.float64)
        centers = centers * (1.0 + psi_gen * q)[:, None]

    coords = rng.standard_normal((P, M, D)).astype(np.float64)
    points = _realize_points(centers, axes, coords, R, eps, rng)

    return Arrangement(
        centers=centers,
        axes=axes,
        coords=coords,
        points=points,
        P=P,
        d=d,
        D=D,
        M=M,
        R=float(R),
        rho_C=float(rho_C),
        rho_A=float(rho_A),
        psi_gen=float(psi_gen),
        kind="spherical",
    )


def _realize_points(
    centers: np.ndarray,
    axes: np.ndarray,
    coords: np.ndarray,
    R: float,
    eps: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """center + R * unit(Σ_j s_j u_j) + ε v."""
    P, M, D = coords.shape
    d = centers.shape[1]
    assert axes.shape == (P, D, d)
    # coords (P, M, D) × axes (P, D, d) → (P, M, d)
    pre_scaled = np.einsum("pmj,pjd->pmd", coords, axes)
    norms = np.linalg.norm(pre_scaled, axis=-1, keepdims=True)
    safe = np.where(norms > 0.0, norms, 1.0)
    unit = pre_scaled / safe
    unit = np.where(norms > 0.0, unit, 0.0)
    noise = rng.standard_normal((P, M, d)).astype(np.float64) / np.sqrt(d)
    return centers[:, None, :] + R * unit + eps * noise


def _realize_isotropic(
    centers: np.ndarray,
    R: float,
    M: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """D.1.1 isotropic Gaussian: ``u₀ + R·v_k``, v_k ~ N(0, I_d/d). No unit-norm."""
    P, d = centers.shape
    v = rng.standard_normal((P, M, d)).astype(np.float64) / np.sqrt(d)
    return centers[:, None, :] + R * v


def resample_test_points(
    arrangement: Arrangement,
    rng: np.random.Generator,
    *,
    eps: float = NOISE_EPS,
) -> np.ndarray:
    """Same centers/axes/coords; fresh noise. Returns (P, M, d) points."""
    if arrangement.kind == "isotropic_gaussian":
        return _realize_isotropic(arrangement.centers, arrangement.R, arrangement.M, rng)
    return _realize_points(
        arrangement.centers,
        arrangement.axes,
        arrangement.coords,
        arrangement.R,
        eps,
        rng,
    )


def redraw_centers_correlated(
    arrangement: Arrangement,
    rho_C: float,
    rng: np.random.Generator,
    *,
    eps: float = NOISE_EPS,
) -> Arrangement:
    """Redraw centers correlated at ``rho_C`` against ``arrangement.centers``.

    Implements feature similarity s_f (`00` §2.4): s_f = 1 means the
    arrangement is unchanged (centers copied); otherwise
    ``c_new = ρ · c_prev + √(1−ρ²) · z`` with z ~ N(0, I_d/d) scaled to the
    previous centers' RMS, then points re-realized with fresh noise.
    Axes and intrinsic coords are kept (readout/feature axes are orthogonal
    by design — only the arrangement of centers moves under s_f).
    """
    if rho_C < 0.0 or rho_C > 1.0:
        raise ValueError(f"rho_C for redraw must be in [0, 1]; got {rho_C}")

    P, d = arrangement.centers.shape
    if rho_C >= 1.0 - 1e-15:
        new_centers = arrangement.centers.copy()
        effective_rho = 1.0
    elif rho_C <= 0.0:
        new_centers = _gaussian_rows(rng, P, d)
        effective_rho = 0.0
    else:
        z = _gaussian_rows(rng, P, d)
        prev = arrangement.centers
        prev_rms = float(np.sqrt(np.mean(prev**2)))
        z_rms = float(np.sqrt(np.mean(z**2)))
        z_scaled = z * (prev_rms / z_rms if z_rms > 0 else 1.0)
        new_centers = rho_C * prev + np.sqrt(1.0 - rho_C**2) * z_scaled
        effective_rho = float(rho_C)

    points = (
        _realize_isotropic(new_centers, arrangement.R, arrangement.M, rng)
        if arrangement.kind == "isotropic_gaussian"
        else _realize_points(
            new_centers,
            arrangement.axes,
            arrangement.coords,
            arrangement.R,
            eps,
            rng,
        )
    )
    return Arrangement(
        centers=new_centers,
        axes=arrangement.axes.copy(),
        coords=arrangement.coords.copy(),
        points=points,
        P=arrangement.P,
        d=arrangement.d,
        D=arrangement.D,
        M=arrangement.M,
        R=arrangement.R,
        rho_C=effective_rho,
        rho_A=arrangement.rho_A,
        psi_gen=arrangement.psi_gen,
        kind=arrangement.kind,
    )
