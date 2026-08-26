"""GLUE core: exact three-factor capacity decomposition from anchor points.

Estimator string ``glue_core@<sha>``. Implements `docs/00-math-spec.md` §6.1–§6.2
(= `docs/reference/glue-decomposition.md`, ICLR 2026 §B.3):

    a(y,t) = (S_y     t)ᵀ (S_y     S_yᵀ)†                       (S_y     t)
    b(y,t) = (S_{y,1} t)ᵀ (S_{y,1} S_{y,1}ᵀ)†                   (S_{y,1} t)
    c(y,t) = (S_{y,1} t)ᵀ (S_{y,0} S_{y,0}ᵀ + S_{y,1} S_{y,1}ᵀ)† (S_{y,1} t)

    α = P/E[a]   D_eff = E[b]/P   R_eff = √(E[c]/E[b−c])   Ψ_eff = E[c]/E[a]

Why this module exists: `replicaMFT` is label-invariant by construction (the
replica derivation integrates the dichotomy average out analytically), so it can
never give retained or tilted capacity — see `third_party/VENDORED.md`. Nothing
in `third_party/` is modified here.

**The anchor QP is joint over all P manifolds, not per-manifold.** In §B.3
`S_y = diag(y)·S`, and `diag(y)` cancels algebraically in all three quadratic
forms (`diag(y)² = I` and `pinv(D A D) = D A† D` for orthogonal diagonal `D`), so
`a`, `b`, `c` are invariant to `y` *given* `S`. All `y`-dependence must therefore
enter through the anchor points, which happens only if they come from the joint QP
where a single separating direction couples the manifolds. A per-manifold
composition would silently reproduce replicaMFT's label-invariance.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Iterable, Literal, Sequence

import numpy as np
from scipy.optimize import lsq_linear, nnls

from src.numerics import clip_to_noise
from src import provenance

# Hash of this file as it was when imported. A forked worker inherits the parent's
# value, which is how `provenance.assert_current` detects that it is running an older
# estimator than the working tree holds (`src/provenance.py`).
_SOURCE = provenance.register(__file__)

DEFAULT_RCOND = 1e-10
DEFAULT_N_T = 200

# `"colgen"` is exact — it returns the same optimum as `"nnls"` to machine precision
# (pinned by `tests/test_glue_core.py`) and is 2.7× faster single-threaded, because the
# anchor QP's solution uses ~64 of its 2400 columns. It also shrinks the working set
# from 5.8 MB to 288 KB, which matters far more than the serial speedup: the estimator
# is memory-bandwidth-bound under parallelism (`results/scaling.json`).
DEFAULT_SOLVER = "colgen"
DUAL_MASS_TOL = 1e-9

InactivePolicy = Literal["zero", "maxproj"]
CenterPolicy = Literal["all", "active"]


def harmonic_mean(alphas: np.ndarray) -> float:
    """Reduce a per-manifold capacity vector. `AGENTS.md` §4 — pinned.

    `α = P/N_crit` and critical dimensions add across manifolds, so the
    arithmetic mean of `α_i` is a bug, not a convention choice.
    """
    a = np.asarray(alphas, dtype=np.float64).ravel()
    if a.size == 0:
        raise ValueError("empty capacity vector")
    if np.any(a <= 0):
        raise ValueError(f"capacities must be positive; got min={a.min()}")
    return float(1.0 / np.mean(1.0 / a))


# --------------------------------------------------------------------------
# Dichotomy ensembles  (`00` §7)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Ensemble:
    """The analyst's choice of dichotomy collection `Y` (`00` §7).

    ``kind="generic"``  β = 0, uniform over balanced dichotomies.
    ``kind="retained"`` β → ∞, `y` fixed at ``y_ref``.
    ``kind="tilted"``   `P(y) ∝ exp(β⟨y, y_ref⟩)` over balanced dichotomies,
                        by exact enumeration (C(16,8) = 12870 — feasible).
    """

    kind: Literal["generic", "retained", "tilted"] = "generic"
    y_ref: np.ndarray | None = None
    beta: float = 0.0

    def __post_init__(self) -> None:
        if self.kind in ("retained", "tilted") and self.y_ref is None:
            raise ValueError(f"kind={self.kind!r} requires y_ref")
        if self.kind == "tilted" and not np.isfinite(self.beta):
            raise ValueError("infinite beta is kind='retained', not 'tilted'")

    @property
    def label(self) -> str:
        if self.kind == "tilted":
            return f"tilted(beta={self.beta:g})"
        return self.kind

    def sampler(self, P: int, rng: np.random.Generator):
        if self.kind == "retained":
            y = np.asarray(self.y_ref, dtype=np.float64)
            return lambda: y
        if self.kind == "generic":
            return lambda: _sample_balanced(P, rng)
        table = _balanced_table(P)
        logits = self.beta * (table @ np.asarray(self.y_ref, dtype=np.float64))
        w = np.exp(logits - logits.max())
        w /= w.sum()
        return lambda: table[rng.choice(table.shape[0], p=w)]


def _sample_balanced(P: int, rng: np.random.Generator) -> np.ndarray:
    y = -np.ones(P, dtype=np.float64)
    y[rng.choice(P, size=P // 2, replace=False)] = 1.0
    return y


_TABLE_CACHE: dict[int, np.ndarray] = {}


def _balanced_table(P: int) -> np.ndarray:
    """All balanced dichotomies as rows. Exact enumeration; P ≤ 20."""
    if P > 20:
        raise ValueError(f"exact tilt enumeration is for P <= 20; got P={P}")
    if P not in _TABLE_CACHE:
        rows = []
        for plus in itertools.combinations(range(P), P // 2):
            y = -np.ones(P, dtype=np.float64)
            y[list(plus)] = 1.0
            rows.append(y)
        _TABLE_CACHE[P] = np.asarray(rows, dtype=np.float64)
    return _TABLE_CACHE[P]


# --------------------------------------------------------------------------
# Anchor points via the joint capacity QP
# --------------------------------------------------------------------------


def _solve_duals(Gt: np.ndarray, t: np.ndarray, solver: str) -> np.ndarray:
    """Non-negative duals of the anchor QP.

    Primal (κ = 0, `00` §6.1):  min ½‖v − t‖²  s.t.  G v ≤ 0, rows of `G` = `y_μ z^μ_i`.
    KKT gives `v = t − Gᵀλ`, `λ ≥ 0`, so the dual is the NNLS problem
    `λ = argmin_{λ≥0} ‖Gᵀλ − t‖²`. `Gᵀ` is passed in as `Gt` (N × ΣM).
    """
    if solver == "nnls":
        lam, _ = nnls(Gt, t)
        return lam
    if solver == "colgen":
        return _nnls_colgen(Gt, t)
    if solver == "lsq":
        res = lsq_linear(Gt, t, bounds=(0.0, np.inf), method="trf", tol=1e-10)
        # `lsq_linear` enforces the bound, so any negativity here is float error; a
        # materially negative dual would mean the solve failed rather than drifted.
        return clip_to_noise(res.x, 0.0, None, atol=1e-8, what="anchor QP dual λ")
    raise ValueError(f"unknown solver {solver!r}")


def _nnls_colgen(
    Gt: np.ndarray, t: np.ndarray, *, n_init: int = 120, n_add: int = 60,
    tol: float = 1e-9, max_rounds: int = 50,
) -> np.ndarray:
    """NNLS by column generation — the same optimum as `scipy.nnls`, several × faster.

    The anchor QP has `P·M` columns (2400 at the design point) but its solution is
    extremely sparse: measured, **64 nonzeros**. At most `N` columns can be active, and
    only points near the margin can be anchors at all, so solving over all 2400 wastes
    almost all of the work.

    Solve on a candidate subset, then check the **full** KKT conditions. For
    `min_{λ≥0} ‖Aλ − t‖²` the optimum is characterised by `g = Aᵀ(Aλ − t) ≥ 0`
    everywhere, with `g_i = 0` wherever `λ_i > 0`. The sub-solve gives the second
    condition on the subset by construction; the first is checked on every column at
    the cost of one mat-vec, and any violators are added and the subset re-solved.
    On termination the full problem's KKT conditions hold, and NNLS is convex, so
    **this is the exact optimum, not an approximation.**

    Seeded with the columns of largest `Aᵀt`, since the gradient at `λ = 0` is `−Aᵀt`
    and those are the only columns that can enter first.
    """
    n_col = Gt.shape[1]
    if n_col <= n_init:
        lam, _ = nnls(Gt, t)
        return lam

    scores = Gt.T @ t
    J = np.argpartition(scores, -n_init)[-n_init:]
    J.sort()
    lam = np.zeros(n_col)

    for _ in range(max_rounds):
        sub, _ = nnls(Gt[:, J], t)
        lam[:] = 0.0
        lam[J] = sub
        grad = Gt.T @ (Gt[:, J] @ sub - t)
        grad[J] = np.inf                       # already stationary on the subset
        violators = np.flatnonzero(grad < -tol)
        if violators.size == 0:
            return lam
        if violators.size > n_add:
            violators = violators[np.argsort(grad[violators])[:n_add]]
        J = np.union1d(J, violators)

    raise RuntimeError(
        f"column generation did not converge in {max_rounds} rounds "
        f"({J.size} of {n_col} columns active) — fall back to solver='nnls'"
    )


def anchor_matrix(
    manifolds: Sequence[np.ndarray],
    y: np.ndarray,
    t: np.ndarray,
    *,
    inactive: InactivePolicy = "zero",
    solver: str = DEFAULT_SOLVER,
    dual_tol: float = DUAL_MASS_TOL,
) -> tuple[np.ndarray, np.ndarray]:
    """Anchor points `S ∈ R^{P×N}` (rows) for one `(y, t)` sample.

    ``manifolds[μ]`` has shape ``(N, M_μ)`` — features × points, the lab
    convention shared with `replicaMFT` and `manifold_simcap_analysis`.

    Rows are the **unsigned** dual-weighted averages `Σ_i λ^μ_i z^μ_i / Σ_i λ^μ_i`
    exactly as in `00` §6.1; `diag(y)` is applied (and cancels) downstream.

    A manifold with zero dual mass is inactive: its constraint is slack, so it
    contributes nothing to the separating direction. ``inactive="zero"`` gives it
    a zero row, which is what reproduces the rectification in the replica
    expression — see `test_point_manifold_capacity_is_two`. ``"maxproj"`` mirrors
    `replicaMFT`'s fallback to the maximum-projection point and is kept for
    diagnosis only.

    Returns ``(S, active)``.
    """
    P = len(manifolds)
    N = manifolds[0].shape[0]
    y = np.asarray(y, dtype=np.float64)

    blocks = [np.ascontiguousarray(Z, dtype=np.float64) for Z in manifolds]
    Gt = np.concatenate([y[mu] * blocks[mu] for mu in range(P)], axis=1)  # (N, ΣM)
    lam = _solve_duals(Gt, np.asarray(t, dtype=np.float64), solver)

    S = np.zeros((P, N), dtype=np.float64)
    active = np.zeros(P, dtype=bool)
    offset = 0
    for mu, Z in enumerate(blocks):
        m = Z.shape[1]
        w = lam[offset : offset + m]
        offset += m
        mass = float(w.sum())
        if mass > dual_tol:
            S[mu] = (Z @ w) / mass
            active[mu] = True
        elif inactive == "maxproj":
            S[mu] = Z[:, int(np.argmax(y[mu] * (t @ Z)))]
    return S, active


# --------------------------------------------------------------------------
# Quadratic forms and measures
# --------------------------------------------------------------------------


def _quad(u: np.ndarray, gram: np.ndarray, rcond: float) -> tuple[float, int]:
    """`uᵀ gram† u` with an explicit rcond; also returns the effective rank."""
    w, V = np.linalg.eigh(np.asarray(gram, dtype=np.float64))
    w_max = float(w.max()) if w.size else 0.0
    if w_max <= 0.0:
        return 0.0, 0
    keep = w > rcond * w_max
    if not np.any(keep):
        return 0.0, 0
    proj = V[:, keep].T @ u
    return float(np.sum(proj**2 / w[keep])), int(keep.sum())


@dataclass
class GlueResult:
    """Output of `glue_measures`. `identity_residual` must be ~0 by construction."""

    alpha: float
    D_eff: float
    R_eff: float
    Psi_eff: float
    rho_c_glue: float
    rho_c_signed: float
    E_a: float
    E_b: float
    E_c: float
    identity_residual: float
    active_fraction: float
    rank_a: float
    rank_b: float
    rank_c: float
    P: int
    N: int
    n_t: int
    ensemble: str
    inactive_policy: str
    center_policy: str
    rcond: float
    estimator: str = "glue_core"
    anchor_centers: np.ndarray | None = field(default=None, repr=False)

    def to_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if k != "anchor_centers"}
        return d


def glue_measures(
    manifolds: Sequence[np.ndarray],
    rng: np.random.Generator,
    *,
    n_t: int = DEFAULT_N_T,
    ensemble: Ensemble | None = None,
    rcond: float = DEFAULT_RCOND,
    inactive: InactivePolicy = "zero",
    center_policy: CenterPolicy = "all",
    solver: str = DEFAULT_SOLVER,
) -> GlueResult:
    """Three-factor decomposition of manifold capacity for a chosen ensemble `Y`.

    ``manifolds[μ]``: ``(N, M_μ)`` float array. Works at any `β` — generic,
    retained, and tilted all route through here (`00` §7 estimator-routing table).

    ``center_policy`` selects how the anchor center `s⁰_μ = E[s^μ(y,t)]` averages
    over samples where manifold `μ` is inactive: ``"all"`` is the literal reading
    of §B.3; ``"active"`` conditions on activity. FLAGGED — the §B.5 R/D recovery
    sweep decides which is right, and the choice is recorded in the result key.
    """
    P = len(manifolds)
    if P == 0:
        raise ValueError("no manifolds")
    N = manifolds[0].shape[0]
    for mu, Z in enumerate(manifolds):
        if Z.ndim != 2 or Z.shape[0] != N:
            raise ValueError(f"manifold {mu} must be (N={N}, M); got {Z.shape}")
    if P % 2 != 0:
        raise ValueError(f"P must be even for balanced dichotomies; got {P}")

    # Independent streams so the `t` sequence is identical across ensembles at the
    # same seed. Comparisons across β are then paired; sharing one stream makes the
    # ensembles differ by Monte-Carlo noise on `t` as well as by `Y`, and at
    # n_t = 150 that noise is ~10% of α — larger than the effects being compared.
    rng_t, rng_y = rng.spawn(2)
    ens = ensemble or Ensemble()
    draw_y = ens.sampler(P, rng_y)

    S_all = np.zeros((n_t, P, N), dtype=np.float64)
    T_all = np.zeros((n_t, N), dtype=np.float64)
    active_all = np.zeros((n_t, P), dtype=bool)
    for k in range(n_t):
        t = rng_t.standard_normal(N)
        S, active = anchor_matrix(
            manifolds, draw_y(), t, inactive=inactive, solver=solver
        )
        S_all[k], T_all[k], active_all[k] = S, t, active

    if center_policy == "all":
        S0 = S_all.mean(axis=0)
    else:
        counts = active_all.sum(axis=0)[:, None]
        S0 = np.where(counts > 0, S_all.sum(axis=0) / np.maximum(counts, 1), 0.0)

    G0 = S0 @ S0.T
    a = np.empty(n_t)
    b = np.empty(n_t)
    c = np.empty(n_t)
    ranks = np.empty((n_t, 3))
    for k in range(n_t):
        S, t = S_all[k], T_all[k]
        S1 = S - S0
        a[k], ranks[k, 0] = _quad(S @ t, S @ S.T, rcond)
        G1 = S1 @ S1.T
        u1 = S1 @ t
        b[k], ranks[k, 1] = _quad(u1, G1, rcond)
        c[k], ranks[k, 2] = _quad(u1, G0 + G1, rcond)

    E_a, E_b, E_c = float(a.mean()), float(b.mean()), float(c.mean())
    if E_a <= 0:
        raise RuntimeError("E[a] <= 0: no manifold was ever active — check inputs")
    alpha = P / E_a
    D_eff = E_b / P
    gap = E_b - E_c
    R_eff = float(np.sqrt(E_c / gap)) if gap > 0 else float("inf")
    Psi_eff = E_c / E_a

    # Exact by construction; a nonzero residual is a bug in this file (`02` §2c).
    residual = (
        abs(Psi_eff * (1.0 + R_eff**-2) / D_eff - alpha) / alpha
        if D_eff > 0 and np.isfinite(R_eff)
        else float("nan")
    )

    return GlueResult(
        alpha=float(alpha),
        D_eff=float(D_eff),
        R_eff=R_eff,
        Psi_eff=float(Psi_eff),
        rho_c_glue=rho_c(S0, convention="glue"),
        rho_c_signed=rho_c(S0, convention="signed"),
        E_a=E_a,
        E_b=E_b,
        E_c=E_c,
        identity_residual=float(residual),
        active_fraction=float(active_all.mean()),
        rank_a=float(ranks[:, 0].mean()),
        rank_b=float(ranks[:, 1].mean()),
        rank_c=float(ranks[:, 2].mean()),
        P=P,
        N=N,
        n_t=n_t,
        ensemble=ens.label,
        inactive_policy=inactive,
        center_policy=center_policy,
        rcond=rcond,
        anchor_centers=S0,
    )


def pairwise_measures(
    manifolds: Sequence[np.ndarray],
    rng: np.random.Generator,
    *,
    max_pairs: int | None = None,
    **kwargs,
) -> dict:
    """Lab-standard `pairwise` estimation: run at `P = 2` over pairs, then pool.

    The lab estimates GLUE two manifolds at a time; `glue_measures` solves one
    joint QP over all `P` (`full_P`). The two are **not** the same estimator — the
    joint QP lets every manifold compete for the same `t`, so anchors, and hence
    `D_eff` and `ρ_c`, differ. `01` Phase 0 compares them and reports both:
    `pairwise` is comparable to published values, `full_P` is primary.

    `α` pools by the harmonic mean (`α = P/N_crit`; critical dimensions add).
    Geometry pools by the arithmetic mean over pairs, and `ρ_c` likewise — for
    `ρ_c` that *is* the cross-pair average of a pairwise quantity, so it is the
    closest analogue of the `full_P` definition available at `P = 2`.
    """
    P = len(manifolds)
    pairs = [(i, j) for i in range(P) for j in range(i + 1, P)]
    if max_pairs is not None and max_pairs < len(pairs):
        idx = rng.choice(len(pairs), size=max_pairs, replace=False)
        pairs = [pairs[i] for i in sorted(idx)]

    streams = rng.spawn(len(pairs))
    out = [
        glue_measures([manifolds[i], manifolds[j]], streams[k], **kwargs)
        for k, (i, j) in enumerate(pairs)
    ]
    keys = ("D_eff", "R_eff", "Psi_eff", "rho_c_glue", "rho_c_signed")
    pooled = {k: float(np.mean([getattr(r, k) for r in out])) for k in keys}
    pooled["alpha"] = harmonic_mean(np.array([r.alpha for r in out]))
    pooled["n_pairs"] = len(pairs)
    pooled["estimation_mode"] = "pairwise"
    pooled["estimator"] = "glue_core"
    return pooled


def rho_c(S0: np.ndarray, convention: Literal["glue", "signed"]) -> float:
    """Anchor-center correlation, mean over pairs `μ ≠ ν` (`00` §6.1 C3).

    ``"glue"``   `|⟨s⁰_μ, s⁰_ν⟩|`               — abs, **unnormalized**; reporting
                                                  and comparability with published
                                                  GLUE values.
    ``"signed"`` `⟨s⁰_μ,s⁰_ν⟩/(‖s⁰_μ‖‖s⁰_ν‖)`   — signed, normalized; the **required**
                                                  H1d instrument (under `|·|`,
                                                  decorrelation and anticorrelation
                                                  are indistinguishable).

    Both are always computed (`rho_convention: both` is mandatory) and the two
    numbers are not comparable to each other.
    """
    S0 = np.asarray(S0, dtype=np.float64)
    P = S0.shape[0]
    if P < 2:
        return float("nan")
    G = S0 @ S0.T
    off = ~np.eye(P, dtype=bool)
    if convention == "glue":
        return float(np.abs(G[off]).mean())
    if convention == "signed":
        norms = np.sqrt(np.diag(G))
        denom = np.outer(norms, norms)
        with np.errstate(invalid="ignore", divide="ignore"):
            C = np.where(denom > 0, G / denom, 0.0)
        return float(C[off].mean())
    raise ValueError(f"unknown convention {convention!r}")


def from_arrangement(points: np.ndarray) -> list[np.ndarray]:
    """Convert `src.manifolds` output `(P, M, N)` to the `(N, M)`-per-manifold list."""
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 3:
        raise ValueError(f"expected (P, M, N); got {pts.shape}")
    return [np.ascontiguousarray(pts[mu].T) for mu in range(pts.shape[0])]


def stack_capacity(alphas: Iterable[float]) -> float:
    """Alias for `harmonic_mean`, for call sites reducing per-manifold α."""
    return harmonic_mean(np.asarray(list(alphas), dtype=np.float64))
