"""Controlled two-task paradigm, run as A → B → A (docs/26 Part B; Kati, 2026-11-06).

Self-contained on purpose: generator, one-module (1M) network, training, NTK and a
vendored copy of the GLUE estimator live here, so a reader sees the whole mechanism
in one file. `aba_controlled.ipynb` asserts that the vendored GLUE matches
`src.glue.core.glue_measures` whenever `src` can be imported (the docs/25 §7
build-script rule is waived for this notebook).

Design (all choices stated; see the notebook's config table):
- Data: P balanced-labelled spherical manifolds in R^d (Chou et al. 2025 generator,
  copied from `src/manifolds/generator.py`). Dichotomies are random balanced draws;
  B's is a Hamming step from A's.
- Input change: B's centres are an AR(1) redraw of A's at ρ = s_f, axes and shapes
  held (centre displacement). **s_f = 1.0 means B's points ARE A's points**, with no
  noise redraw, so Hamming 0 at s_f 1.0 is an exact identity control.
- Learner: one hidden ReLU layer, μP, readout initialised at 0, full-batch GD on
  0.5·mean(err²), lr = lr0·γ0²·(N/64). Stop when the training loss, checked BEFORE an
  update, is ≤ target (so a task already at criterion takes 0 steps).
- Pairing: A, y_A, W(0), test noise, the NTK probe set and the GLUE stream depend on
  the seed only; y_B on (seed, Hamming); B's centres on (seed, s_f). Every cell and
  both γ0 share them.
"""

from __future__ import annotations

import os

# Before numpy: spawned workers re-import this module, and BLAS reads these at load.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import functools
import hashlib
import itertools
import json
import math
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from multiprocessing import get_context
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
from scipy.optimize import lsq_linear, nnls

LIB_PATH = Path(__file__).resolve()
REPO = LIB_PATH.parents[2]
SALT = 20261106
NOISE_EPS = 1e-2


# ==========================================================================
# Provenance
# ==========================================================================


def lib_hash() -> str:
    """sha256 of this file's bytes, 16 hex chars. Part of every cache key."""
    return hashlib.sha256(LIB_PATH.read_bytes()).hexdigest()[:16]


def git_stamp() -> dict:
    """Commit and dirty flag of the repo this file lives in."""
    try:
        def run(*args: str) -> str:
            return subprocess.run(["git", *args], cwd=REPO, capture_output=True,
                                  text=True, timeout=30).stdout.strip()
        commit = run("rev-parse", "--short=12", "HEAD") or "unknown"
        dirty = bool(run("status", "--porcelain"))
        return {"commit": commit, "dirty": dirty}
    except Exception:  # noqa: BLE001 — a stamp must never stop a run
        return {"commit": "unknown", "dirty": None}


# ==========================================================================
# Configuration
# ==========================================================================


@dataclass(frozen=True)
class ABAConfig:
    quick: bool = True
    # data
    P: int = 16
    d: int = 150
    D: int = 4
    R: float = 1.0
    M: int = 50
    # network and training
    N: int = 300
    lr0: float = 5.0
    n_base: int = 64
    target_loss: float = 0.05
    max_steps: int = 20_000
    record_every: int = 25
    # measurement
    n_t: int = 50
    center_policy: str = "all"
    ntk_per_manifold: int = 20
    probe_hamming: int = 4          # y* anchored at s_r = 0.5 from y_A, as streams._sample_probe
    n_probe_neutral: int = 4        # neutral y*_k at s_r(y*_k, y_A) = 0 (forward-transfer baseline)
    refit_steps: int = 2000         # fixed GD budget for the refit-readout probe
    # design
    gammas: tuple = (0.1, 10.0)
    hammings: tuple = (0, 4, 8)
    s_fs: tuple = (1.0, 0.9, 0.1)
    n_seeds: int = 3

    @classmethod
    def make(cls, quick: bool) -> "ABAConfig":
        if quick:
            return cls(quick=True)
        return cls(quick=False, M=150, n_t=200, n_seeds=8)

    def key_fields(self) -> dict:
        """Everything that changes an arm's result (the design lists do not)."""
        d = asdict(self)
        for k in ("quick", "gammas", "hammings", "s_fs", "n_seeds"):
            d.pop(k)
        return d


# ==========================================================================
# Generator  (copied from src/manifolds/generator.py, spherical kind, rho_C=rho_A=psi=0)
# ==========================================================================


@dataclass(frozen=True)
class Arr:
    centers: np.ndarray   # (P, d)
    axes: np.ndarray      # (P, D, d)
    coords: np.ndarray    # (P, M, D)
    points: np.ndarray    # (P, M, d) training points
    test: np.ndarray      # (P, M, d) same manifolds, fresh noise
    R: float


def _gaussian_rows(rng: np.random.Generator, n_rows: int, d: int) -> np.ndarray:
    """Rows ~ N(0, I_d / d)."""
    return rng.standard_normal((n_rows, d)).astype(np.float64) / np.sqrt(d)


def _realize_points(centers, axes, coords, R, eps, rng) -> np.ndarray:
    """center + R * unit(Σ_j s_j u_j) + ε v."""
    P, M, D = coords.shape
    d = centers.shape[1]
    pre_scaled = np.einsum("pmj,pjd->pmd", coords, axes)
    norms = np.linalg.norm(pre_scaled, axis=-1, keepdims=True)
    safe = np.where(norms > 0.0, norms, 1.0)
    unit = np.where(norms > 0.0, pre_scaled / safe, 0.0)
    noise = rng.standard_normal((P, M, d)).astype(np.float64) / np.sqrt(d)
    return centers[:, None, :] + R * unit + eps * noise


def make_arrangement(P: int, d: int, D: int, R: float, M: int,
                     rng: np.random.Generator, rng_test: np.random.Generator) -> Arr:
    """Same draw order as `src.manifolds.generator.make_arrangement` (spherical)."""
    centers = _gaussian_rows(rng, P, d)
    axes = np.stack([_gaussian_rows(rng, P, d) for _ in range(D)], axis=1)
    coords = rng.standard_normal((P, M, D)).astype(np.float64)
    points = _realize_points(centers, axes, coords, R, NOISE_EPS, rng)
    test = _realize_points(centers, axes, coords, R, NOISE_EPS, rng_test)
    return Arr(centers, axes, coords, points, test, float(R))


def redraw_centers(arr: Arr, rho: float, rng: np.random.Generator) -> Arr:
    """AR(1) centre redraw at `rho`, axes and coords held (`redraw_centers_correlated`).

    `rho = 1` returns `arr` itself — the exact-copy convention for s_f = 1.0.
    """
    if rho >= 1.0:
        return arr
    if rho < 0.0:
        raise ValueError(f"rho must be in [0, 1]; got {rho}")
    P, d = arr.centers.shape
    if rho == 0.0:
        new = _gaussian_rows(rng, P, d)
    else:
        z = _gaussian_rows(rng, P, d)
        prev_rms = float(np.sqrt(np.mean(arr.centers ** 2)))
        z_rms = float(np.sqrt(np.mean(z ** 2)))
        z = z * (prev_rms / z_rms if z_rms > 0 else 1.0)
        new = rho * arr.centers + np.sqrt(1.0 - rho ** 2) * z
    points = _realize_points(new, arr.axes, arr.coords, arr.R, NOISE_EPS, rng)
    test = _realize_points(new, arr.axes, arr.coords, arr.R, NOISE_EPS, rng)
    return Arr(new, arr.axes, arr.coords, points, test, arr.R)


# ==========================================================================
# Dichotomies  (copied from src/manifolds/dichotomies.py and streams._sample_probe)
# ==========================================================================


def sample_balanced(P: int, rng: np.random.Generator) -> np.ndarray:
    y = -np.ones(P, dtype=np.int8)
    y[rng.choice(P, size=P // 2, replace=False)] = 1
    return y


def sample_at_hamming(y: np.ndarray, h: int, rng: np.random.Generator) -> np.ndarray:
    """Balanced dichotomy at Hamming distance `h` from `y`: swap h/2 (+1) with h/2 (−1)."""
    out = np.asarray(y, dtype=np.int8).copy()
    P = out.size
    if h < 0 or h > P or h % 2:
        raise ValueError(f"h must be even in [0, P]; got {h}")
    if h == 0:
        return out
    plus, minus = np.flatnonzero(out == 1), np.flatnonzero(out == -1)
    tp = rng.choice(plus, size=h // 2, replace=False)
    tm = rng.choice(minus, size=h // 2, replace=False)
    out[tp], out[tm] = -1, 1
    assert int(np.sum(out != y)) == h and int(out.sum()) == 0
    return out


def s_r(y1: np.ndarray, y2: np.ndarray) -> float:
    """Signed label relation `1 − 2·Hamming/P` (docs/26 A3). −1 for y2 = −y1."""
    y1, y2 = np.asarray(y1), np.asarray(y2)
    return float(1.0 - 2.0 * np.sum(y1 != y2) / y1.size)


def sample_probe(y_A: np.ndarray, others: Sequence[np.ndarray], h: int,
                 rng: np.random.Generator, max_tries: int = 10_000) -> np.ndarray:
    """Held-out y*: Hamming `h` from y_A, equal (up to sign) to no training dichotomy."""
    banned = [np.asarray(y, np.int8) for y in [y_A, *others]]
    banned += [-y for y in banned]
    for _ in range(max_tries):
        cand = sample_at_hamming(y_A, h, rng)
        if not any(np.array_equal(cand, b) for b in banned):
            return cand
    raise RuntimeError("no probe dichotomy distinct from A and every B")


# ==========================================================================
# Paired design
# ==========================================================================


@dataclass(frozen=True)
class SeedBase:
    """Everything that depends on the seed only (shared by every cell and both γ0)."""
    seed: int
    A: Arr
    y_A: np.ndarray
    y_Bs: dict
    y_star: np.ndarray
    W0: np.ndarray
    ntk_idx: np.ndarray       # (P, k) point indices for the NTK probe set
    y_star_neutral: tuple = ()   # K balanced dichotomies at s_r = 0 from y_A


@functools.lru_cache(maxsize=16)
def seed_base(cfg: ABAConfig, seed: int) -> SeedBase:
    ss = np.random.SeedSequence([SALT, seed])
    r_arr, r_test, r_lab, r_init, r_ntk, r_probe = (np.random.default_rng(s)
                                                    for s in ss.spawn(6))
    A = make_arrangement(cfg.P, cfg.d, cfg.D, cfg.R, cfg.M, r_arr, r_test)
    y_A = sample_balanced(cfg.P, r_lab)
    y_Bs = {h: sample_at_hamming(y_A, h, np.random.default_rng([SALT, seed, 1, h]))
            for h in cfg.hammings}
    y_star = sample_probe(y_A, list(y_Bs.values()), cfg.probe_hamming, r_probe)
    W0 = r_init.standard_normal((cfg.N, cfg.d))
    k = min(cfg.ntk_per_manifold, cfg.M)
    ntk_idx = np.stack([r_ntk.choice(cfg.M, size=k, replace=False) for _ in range(cfg.P)])
    # Neutral probes: Hamming P/2 from y_A (s_r = 0), balanced, distinct (up to sign)
    # from y_A, every y_B, the anchored y* and each other. Own stream: earlier draws unchanged.
    r_neu = np.random.default_rng([SALT, seed, 4])
    others = [*y_Bs.values(), y_star]
    neutral = []
    for _ in range(cfg.n_probe_neutral):
        yk = sample_probe(y_A, others + neutral, cfg.P // 2, r_neu)
        assert s_r(yk, y_A) == 0.0
        neutral.append(yk)
    return SeedBase(seed, A, y_A, y_Bs, y_star, W0, ntk_idx, tuple(neutral))


def b_arrangement(cfg: ABAConfig, base: SeedBase, s_f: float, noise: str = "copy") -> Arr:
    """B's manifolds. At s_f = 1.0, `noise="copy"` returns A itself; `"fresh"` keeps
    A's centres, axes and coordinates and redraws only the ε noise (choice check)."""
    if noise not in ("copy", "fresh"):
        raise ValueError(f"noise must be 'copy' or 'fresh'; got {noise!r}")
    if s_f >= 1.0 and noise == "fresh":
        A = base.A
        rng = np.random.default_rng([SALT, base.seed, 3])
        points = _realize_points(A.centers, A.axes, A.coords, A.R, NOISE_EPS, rng)
        test = _realize_points(A.centers, A.axes, A.coords, A.R, NOISE_EPS, rng)
        return Arr(A.centers, A.axes, A.coords, points, test, A.R)
    return redraw_centers(base.A, s_f,
                          np.random.default_rng([SALT, base.seed, 2, int(round(s_f * 1000))]))


def flatten(points: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    P, M, d = points.shape
    return points.reshape(P * M, d), np.repeat(np.asarray(y, np.float64), M)


def centre_cosines(c1: np.ndarray, c2: np.ndarray) -> np.ndarray:
    """(P, P) cosine between manifold centres of two arrangements."""
    n1 = c1 / np.linalg.norm(c1, axis=1, keepdims=True)
    n2 = c2 / np.linalg.norm(c2, axis=1, keepdims=True)
    return n1 @ n2.T


# ==========================================================================
# 1M network
# ==========================================================================


class OneModuleNet:
    """f(x) = s · uᵀ relu(β0 W x), β0 = d^-1/2, s = N^-1/2 / (γ0 √(N/n_base)), u(0) = 0.

    Identical to one module of `src.models.network.TwoModuleNet` at L = 2 with
    `ScalingConfig(parameterization="mup", lr_scaling="quadratic")`.
    """

    def __init__(self, W0: np.ndarray, gamma: float, cfg: ABAConfig):
        self.N, self.d = W0.shape
        self.gamma = float(gamma)
        self.W = np.array(W0, dtype=np.float64, copy=True)
        self.u = np.zeros(self.N)
        self.W_init = self.W.copy()
        self.beta0 = self.d ** -0.5
        self.s = self.N ** -0.5 / (gamma * math.sqrt(self.N / cfg.n_base))
        self.lr = cfg.lr0 * gamma ** 2 * (self.N / cfg.n_base)

    def hidden(self, X: np.ndarray) -> np.ndarray:
        return np.maximum(X @ self.W.T * self.beta0, 0.0)

    def forward(self, X: np.ndarray) -> np.ndarray:
        return self.s * (self.hidden(X) @ self.u)

    def loss(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(0.5 * np.mean((self.forward(X) - y) ** 2))

    def loss_grads(self, X: np.ndarray, y: np.ndarray):
        H = self.hidden(X)
        err = self.s * (H @ self.u) - y
        B = y.shape[0]
        gu = self.s * (H.T @ err) / B
        delta = (H > 0) * (self.s * err)[:, None] * self.u[None, :]
        gW = self.beta0 * (delta.T @ X) / B
        return float(0.5 * np.mean(err ** 2)), gW, gu

    def weight_change(self) -> float:
        return float(np.linalg.norm(self.W - self.W_init) / np.linalg.norm(self.W_init))

    # -- NTK -----------------------------------------------------------------

    def ntk_blocks(self, Xp: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Readout block K_u = s²HHᵀ and hidden block K_W = s²β0²(XXᵀ)⊙(D diag(u²) Dᵀ)."""
        H = self.hidden(Xp)
        Dm = (H > 0).astype(np.float64)
        Ku = self.s ** 2 * (H @ H.T)
        KW = self.s ** 2 * self.beta0 ** 2 * (Xp @ Xp.T) * ((Dm * self.u ** 2) @ Dm.T)
        return Ku, KW

    def jacobian(self, Xp: np.ndarray) -> np.ndarray:
        """Explicit ∂f/∂(W, u) per point, (n, N·d + N). For the analytic check only."""
        H = self.hidden(Xp)
        Dm = (H > 0).astype(np.float64)
        rows = []
        for x, h, dm in zip(Xp, H, Dm):
            dW = np.outer(self.s * self.u * dm * self.beta0, x)
            rows.append(np.concatenate([dW.ravel(), self.s * h]))
        return np.asarray(rows)

    def _f_at(self, theta: np.ndarray, Xp: np.ndarray) -> np.ndarray:
        W = theta[: self.N * self.d].reshape(self.N, self.d)
        u = theta[self.N * self.d:]
        return self.s * (np.maximum(Xp @ W.T * self.beta0, 0.0) @ u)


def cka(K: np.ndarray, L: np.ndarray) -> float:
    """Centred kernel alignment ⟨K̃, L̃⟩ / (‖K̃‖ ‖L̃‖)."""
    def centre(A):
        return A - A.mean(0, keepdims=True) - A.mean(1, keepdims=True) + A.mean()
    Kc, Lc = centre(K), centre(L)
    den = np.sqrt(np.sum(Kc * Kc) * np.sum(Lc * Lc))
    return float(np.sum(Kc * Lc) / den) if den > 0 else float("nan")


def ntk_fd_check(net: OneModuleNet, Xp: np.ndarray, rng: np.random.Generator,
                 n_dirs: int = 4, eps: float = 1e-6) -> dict:
    """(i) J Jᵀ equals K_u + K_W; (ii) J δ equals a central finite difference."""
    J = net.jacobian(Xp)
    Ku, KW = net.ntk_blocks(Xp)
    K = Ku + KW
    k_err = float(np.max(np.abs(J @ J.T - K)) / np.max(np.abs(K)))
    theta = np.concatenate([net.W.ravel(), net.u])
    fd_err = 0.0
    for _ in range(n_dirs):
        delta = rng.standard_normal(theta.size)
        delta /= np.linalg.norm(delta)
        fd = (net._f_at(theta + eps * delta, Xp) - net._f_at(theta - eps * delta, Xp)) / (2 * eps)
        an = J @ delta
        fd_err = max(fd_err, float(np.max(np.abs(fd - an)) / max(np.max(np.abs(an)), 1e-300)))
    return {"K_vs_JJt_rel": k_err, "Jdelta_vs_fd_rel": fd_err}


# ==========================================================================
# Training
# ==========================================================================


def train_to_criterion(net: OneModuleNet, X: np.ndarray, y: np.ndarray, cfg: ABAConfig,
                       monitors: dict | None = None) -> dict:
    """Full-batch GD until loss ≤ target, the loss checked BEFORE each update.

    Records train loss and each monitor's (loss, sign accuracy) every `record_every`
    updates and at the final state. `first_loss` is the zero-shot loss on this task.
    """
    monitors = monitors or {}
    W_start = net.W.copy()
    recs: list[dict] = []

    def record(step: int, train_loss: float) -> None:
        row = {"step": step, "train_loss": train_loss}
        for name, (Xm, ym) in monitors.items():
            f = net.forward(Xm)
            row[f"{name}_loss"] = float(0.5 * np.mean((f - ym) ** 2))
            row[f"{name}_acc"] = float(np.mean(np.sign(f) == ym))
        recs.append(row)

    steps, first_loss, status = 0, None, "cap"
    while True:
        loss, gW, gu = net.loss_grads(X, y)
        if first_loss is None:
            first_loss = loss
        if not np.isfinite(loss):
            status = "diverged"
            record(steps, loss)
            break
        if loss <= cfg.target_loss:
            status = "converged"
            record(steps, loss)
            break
        if steps >= cfg.max_steps:
            record(steps, loss)
            break
        if steps % cfg.record_every == 0:
            record(steps, loss)
        net.W -= net.lr * gW
        net.u -= net.lr * gu
        steps += 1
    return {"steps": steps, "status": status, "converged": status == "converged",
            "first_loss": float(first_loss), "final_loss": float(loss),
            "r_t": float(np.linalg.norm(net.W - W_start) / np.linalg.norm(W_start)),
            "records": recs}


REFIT_CHECKPOINTS = (100, 500)


def refit_probe(net: OneModuleNet, X: np.ndarray, y: np.ndarray, X_test: np.ndarray,
                y_test: np.ndarray, cfg: ABAConfig) -> dict:
    """Behavioural counterpart of capacity: fit a FRESH readout (u = 0) on the frozen
    hidden features of `net`, full-batch GD with `net`'s own s and lr, for exactly
    `cfg.refit_steps` steps (no early stop), and score it on the test points.

    In μP, lr·s² = lr0/N, so the effective step on the readout is the same at every γ0
    given H. `net` is not modified.
    """
    H, Ht = net.hidden(X), net.hidden(X_test)
    s, lr, B = net.s, net.lr, y.shape[0]
    u = np.zeros(H.shape[1])
    out = {}

    def score(tag):
        f = s * (Ht @ u)
        out[f"loss@{tag}"] = float(0.5 * np.mean((f - y_test) ** 2))
        out[f"acc@{tag}"] = float(np.mean(np.sign(f) == y_test))

    for t in range(1, cfg.refit_steps + 1):
        err = s * (H @ u) - y
        u -= lr * s * (H.T @ err) / B
        if t in REFIT_CHECKPOINTS:
            score(t)
    out["train_loss"] = float(0.5 * np.mean((s * (H @ u) - y) ** 2))
    score("end")
    out["loss"], out["acc"] = out["loss@end"], out["acc@end"]
    return out


# ==========================================================================
# GLUE  — vendored from src/glue/core.py (and src/numerics.clip_to_noise).
# Unchanged except: provenance registration removed; pairwise/harmonic helpers omitted.
# ==========================================================================

DEFAULT_RCOND = 1e-10
DEFAULT_N_T = 200
DEFAULT_SOLVER = "colgen"
DUAL_MASS_TOL = 1e-9

InactivePolicy = Literal["zero", "maxproj"]
CenterPolicy = Literal["all", "active"]


def clip_to_noise(x, lo, hi, *, atol: float = 1e-9, what: str = "value"):
    a = np.asarray(x, dtype=np.float64)
    over = 0.0 if hi is None else float(np.max(a - hi, initial=0.0))
    under = 0.0 if lo is None else float(np.max(lo - a, initial=0.0))
    worst = max(over, under)
    if worst > atol:
        raise ValueError(
            f"{what} is {worst:.3e} outside [{lo}, {hi}], far beyond the {atol:.1e} "
            f"tolerance for float error.")
    out = np.clip(a, -np.inf if lo is None else lo, np.inf if hi is None else hi)
    return float(out) if np.isscalar(x) or out.ndim == 0 else out


@dataclass(frozen=True)
class Ensemble:
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


def _solve_duals(Gt: np.ndarray, t: np.ndarray, solver: str) -> np.ndarray:
    if solver == "nnls":
        lam, _ = nnls(Gt, t)
        return lam
    if solver == "colgen":
        return _nnls_colgen(Gt, t)
    if solver == "lsq":
        res = lsq_linear(Gt, t, bounds=(0.0, np.inf), method="trf", tol=1e-10)
        return clip_to_noise(res.x, 0.0, None, atol=1e-8, what="anchor QP dual λ")
    raise ValueError(f"unknown solver {solver!r}")


def _nnls_colgen(Gt: np.ndarray, t: np.ndarray, *, n_init: int = 120, n_add: int = 60,
                 tol: float = 1e-9, max_rounds: int = 50) -> np.ndarray:
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
        grad[J] = np.inf
        violators = np.flatnonzero(grad < -tol)
        if violators.size == 0:
            return lam
        if violators.size > n_add:
            violators = violators[np.argsort(grad[violators])[:n_add]]
        J = np.union1d(J, violators)
    raise RuntimeError(
        f"column generation did not converge in {max_rounds} rounds "
        f"({J.size} of {n_col} columns active) — fall back to solver='nnls'")


def anchor_matrix(manifolds: Sequence[np.ndarray], y: np.ndarray, t: np.ndarray, *,
                  inactive: InactivePolicy = "zero", solver: str = DEFAULT_SOLVER,
                  dual_tol: float = DUAL_MASS_TOL) -> tuple[np.ndarray, np.ndarray]:
    P = len(manifolds)
    N = manifolds[0].shape[0]
    y = np.asarray(y, dtype=np.float64)
    blocks = [np.ascontiguousarray(Z, dtype=np.float64) for Z in manifolds]
    Gt = np.concatenate([y[mu] * blocks[mu] for mu in range(P)], axis=1)
    lam = _solve_duals(Gt, np.asarray(t, dtype=np.float64), solver)
    S = np.zeros((P, N), dtype=np.float64)
    active = np.zeros(P, dtype=bool)
    offset = 0
    for mu, Z in enumerate(blocks):
        m = Z.shape[1]
        w = lam[offset: offset + m]
        offset += m
        mass = float(w.sum())
        if mass > dual_tol:
            S[mu] = (Z @ w) / mass
            active[mu] = True
        elif inactive == "maxproj":
            S[mu] = Z[:, int(np.argmax(y[mu] * (t @ Z)))]
    return S, active


def _quad(u: np.ndarray, gram: np.ndarray, rcond: float) -> tuple[float, int]:
    w, V = np.linalg.eigh(np.asarray(gram, dtype=np.float64))
    w_max = float(w.max()) if w.size else 0.0
    if w_max <= 0.0:
        return 0.0, 0
    keep = w > rcond * w_max
    if not np.any(keep):
        return 0.0, 0
    proj = V[:, keep].T @ u
    return float(np.sum(proj ** 2 / w[keep])), int(keep.sum())


@dataclass
class GlueResult:
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
        return {k: v for k, v in self.__dict__.items() if k != "anchor_centers"}


def glue_measures(manifolds: Sequence[np.ndarray], rng: np.random.Generator, *,
                  n_t: int = DEFAULT_N_T, ensemble: Ensemble | None = None,
                  rcond: float = DEFAULT_RCOND, inactive: InactivePolicy = "zero",
                  center_policy: CenterPolicy = "all",
                  solver: str = DEFAULT_SOLVER) -> GlueResult:
    P = len(manifolds)
    if P == 0:
        raise ValueError("no manifolds")
    N = manifolds[0].shape[0]
    for mu, Z in enumerate(manifolds):
        if Z.ndim != 2 or Z.shape[0] != N:
            raise ValueError(f"manifold {mu} must be (N={N}, M); got {Z.shape}")
    if P % 2 != 0:
        raise ValueError(f"P must be even for balanced dichotomies; got {P}")

    rng_t, rng_y = rng.spawn(2)
    ens = ensemble or Ensemble()
    draw_y = ens.sampler(P, rng_y)

    S_all = np.zeros((n_t, P, N), dtype=np.float64)
    T_all = np.zeros((n_t, N), dtype=np.float64)
    active_all = np.zeros((n_t, P), dtype=bool)
    for k in range(n_t):
        t = rng_t.standard_normal(N)
        S, active = anchor_matrix(manifolds, draw_y(), t, inactive=inactive, solver=solver)
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
    residual = (abs(Psi_eff * (1.0 + R_eff ** -2) / D_eff - alpha) / alpha
                if D_eff > 0 and np.isfinite(R_eff) else float("nan"))

    return GlueResult(
        alpha=float(alpha), D_eff=float(D_eff), R_eff=R_eff, Psi_eff=float(Psi_eff),
        rho_c_glue=rho_c(S0, convention="glue"), rho_c_signed=rho_c(S0, convention="signed"),
        E_a=E_a, E_b=E_b, E_c=E_c, identity_residual=float(residual),
        active_fraction=float(active_all.mean()),
        rank_a=float(ranks[:, 0].mean()), rank_b=float(ranks[:, 1].mean()),
        rank_c=float(ranks[:, 2].mean()), P=P, N=N, n_t=n_t, ensemble=ens.label,
        inactive_policy=inactive, center_policy=center_policy, rcond=rcond,
        anchor_centers=S0)


def rho_c(S0: np.ndarray, convention: Literal["glue", "signed"]) -> float:
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


# ==========================================================================
# Measurement
# ==========================================================================


def as_manifolds(points: np.ndarray) -> list[np.ndarray]:
    """(P, M, n) → list of (n, M), the GLUE convention."""
    return [np.ascontiguousarray(points[mu].T) for mu in range(points.shape[0])]


def capacity(manifolds: list[np.ndarray], y: np.ndarray, cfg: ABAConfig, rng_key) -> dict:
    """Retained (fixed-y) GLUE. `rng_key` is fixed per seed so boundaries are paired."""
    r = glue_measures(manifolds, np.random.default_rng(rng_key), n_t=cfg.n_t,
                      ensemble=Ensemble(kind="retained", y_ref=np.asarray(y, np.float64)),
                      center_policy=cfg.center_policy)
    return {k: float(getattr(r, k)) for k in
            ("alpha", "Psi_eff", "R_eff", "D_eff", "rho_c_glue", "rho_c_signed",
             "identity_residual", "active_fraction")}


def rep_capacity(net: OneModuleNet, points: np.ndarray, y, cfg, rng_key) -> dict:
    P, M, d = points.shape
    H = net.hidden(points.reshape(P * M, d)).reshape(P, M, -1)
    return capacity(as_manifolds(H), y, cfg, rng_key)


def channels(g0: dict, g1: dict) -> dict:
    """Δlog α = Δlog Ψ + Δlog(1 + R⁻²) − Δlog D, exactly, between two measurements."""
    total = math.log(g1["alpha"] / g0["alpha"])
    util = math.log(g1["Psi_eff"] / g0["Psi_eff"])
    rad = math.log((1 + g1["R_eff"] ** -2) / (1 + g0["R_eff"] ** -2))
    dim = -math.log(g1["D_eff"] / g0["D_eff"])
    return {"total": total, "utility": util, "radius": rad, "dimension": dim,
            "residual": total - (util + rad + dim)}


def forgetting_running_max(recs: list[dict], key: str) -> dict:
    """Signed forgetting over one phase against the running max (final point excluded).

    `recs[0]` is the state entering the phase. Positive = forgot. A 0-step phase has
    one record, so the window is that record and forgetting is exactly 0.
    """
    vals = np.array([r[key] for r in recs], dtype=np.float64)
    window = vals[:-1] if vals.size > 1 else vals
    acc_like = key.endswith("_acc")
    peak = window.max() if acc_like else window.min()
    final = vals[-1]
    return {"peak": float(peak), "final": float(final),
            "forgetting": float(peak - final) if acc_like else float(final - peak)}


def richness_snapshot(net: OneModuleNet, K0: np.ndarray, Ku0: np.ndarray,
                      probe_A: np.ndarray, yA_p: np.ndarray,
                      probe_B: np.ndarray, yB_p: np.ndarray,
                      prev: tuple | None = None) -> tuple[dict, tuple]:
    """Kernel changes, all normalised by ‖K(0)‖_F so the three are on one scale.

    K_W(0) = 0 (readout starts at 0) and K(0) = K_u(0), so `dK_W` is the hidden-
    weight block's growth in units of the initial kernel. `dK_full` mixes readout-
    block change (`dK_u`) with that growth.

    `prev` = (K, K_u, K_W) at the previous phase boundary: adds the phase-to-phase
    changes `d*_step` (each ‖now − prev‖ / ‖K_prev‖_F) and `align_step` = CKA(K_prev, K).
    Returns (measures, (K, K_u, K_W)) so the caller can chain phases.
    """
    Ku, KW = net.ntk_blocks(probe_A)
    K = Ku + KW
    KuB, KWB = net.ntk_blocks(probe_B)
    n0 = np.linalg.norm(K0)
    step = {}
    if prev is not None:
        Kp, Kup, KWp = prev
        npv = np.linalg.norm(Kp)
        step = {"dK_full_step": float(np.linalg.norm(K - Kp) / npv),
                "dK_u_step": float(np.linalg.norm(Ku - Kup) / npv),
                "dK_W_step": float(np.linalg.norm(KW - KWp) / npv),
                "align_step": cka(Kp, K)}
    return {**step,
        "weight_change": net.weight_change(),
        "dK_full": float(np.linalg.norm(K - K0) / n0),
        "dK_u": float(np.linalg.norm(Ku - Ku0) / np.linalg.norm(Ku0)),
        "dK_W": float(np.linalg.norm(KW - (K0 - Ku0)) / n0),
        "KW_share": float(np.linalg.norm(KW) / np.linalg.norm(K)),
        "cka_K_yA": cka(K, np.outer(yA_p, yA_p)),
        "cka_K_yB": cka(KuB + KWB, np.outer(yB_p, yB_p)),
    }, (K, Ku, KW)


# ==========================================================================
# One arm
# ==========================================================================


@dataclass(frozen=True)
class ArmSpec:
    seed: int
    gamma: float
    hamming: int
    s_f: float
    noise: str = "copy"     # only meaningful at s_f = 1.0

    @property
    def key(self) -> str:
        k = f"seed={self.seed},gamma={self.gamma:g},H={self.hamming},s_f={self.s_f:g}"
        return k if self.noise == "copy" else k + f",noise={self.noise}"


def _logratio_steps(a: int, b: int) -> float:
    """log((1+a)/(1+b)). The +1 keeps 0-step phases (identity control) finite."""
    return math.log((1 + a) / (1 + b))


def run_arm(cfg: ABAConfig, spec: ArmSpec, stamp: dict | None = None) -> dict:
    t_start = time.perf_counter()
    timing = {"train_s": 0.0, "glue_s": 0.0, "ntk_s": 0.0, "refit_s": 0.0}
    base = seed_base(cfg, spec.seed)
    A, y_A, y_star = base.A, base.y_A, base.y_star
    y_B = base.y_Bs[spec.hamming]
    B = b_arrangement(cfg, base, spec.s_f, spec.noise)
    glue_key = [SALT, spec.seed, 9]

    XA, yA = flatten(A.points, y_A)
    XB, yB = flatten(B.points, y_B)
    monitors = {"A_test": flatten(A.test, y_A), "B_test": flatten(B.test, y_B)}
    idx = base.ntk_idx
    probe_A = np.concatenate([A.points[mu, idx[mu]] for mu in range(cfg.P)])
    probe_B = np.concatenate([B.points[mu, idx[mu]] for mu in range(cfg.P)])
    yA_p = np.repeat(y_A.astype(np.float64), idx.shape[1])
    yB_p = np.repeat(y_B.astype(np.float64), idx.shape[1])

    net = OneModuleNet(base.W0, spec.gamma, cfg)
    init_loss = net.loss(XA, yA)

    def geo(tag, points, y):
        t0 = time.perf_counter()
        out = rep_capacity(net, points, y, cfg, glue_key)
        timing["glue_s"] += time.perf_counter() - t0
        return out

    kprev = [None]

    def rich():
        t0 = time.perf_counter()
        out, kprev[0] = richness_snapshot(net, K0, Ku0, probe_A, yA_p, probe_B, yB_p,
                                          prev=kprev[0])
        timing["ntk_s"] += time.perf_counter() - t0
        return out

    XA_test, yA_test = monitors["A_test"]

    def refit():
        t0 = time.perf_counter()
        out = refit_probe(net, XA, yA, XA_test, yA_test, cfg)
        timing["refit_s"] += time.perf_counter() - t0
        return out

    def train(model, X, y, mons):
        t0 = time.perf_counter()
        out = train_to_criterion(model, X, y, cfg, mons)
        timing["train_s"] += time.perf_counter() - t0
        return out

    Ku0, KW0 = net.ntk_blocks(probe_A)
    K0 = Ku0 + KW0
    neutral = base.y_star_neutral
    g, r, ph, rf = {}, {}, {}, {}
    r["init"] = rich()
    rf["init"] = refit()
    g["A@init"] = geo("A", A.points, y_A)
    g["B_yB@init"] = geo("B", B.points, y_B)
    g["B_ystar@init"] = geo("B*", B.points, y_star)
    for k, yk in enumerate(neutral):
        g[f"B_ystarN{k}@init"] = geo("B*N", B.points, yk)

    ph["A1"] = train(net, XA, yA, monitors)
    r["A1"] = rich()
    rf["A1"] = refit()
    g["A@A1"] = geo("A", A.points, y_A)
    g["B_yB@A1"] = geo("B", B.points, y_B)
    g["B_ystar@A1"] = geo("B*", B.points, y_star)
    for k, yk in enumerate(neutral):
        g[f"B_ystarN{k}@A1"] = geo("B*N", B.points, yk)

    ph["B"] = train(net, XB, yB, monitors)
    r["B"] = rich()
    rf["B"] = refit()
    g["A@B"] = geo("A", A.points, y_A)

    ph["A2"] = train(net, XA, yA, monitors)
    r["A2"] = rich()
    rf["A2"] = refit()
    g["A@A2"] = geo("A", A.points, y_A)

    fresh = OneModuleNet(base.W0, spec.gamma, cfg)
    ph["freshB"] = train(fresh, XB, yB, None)

    ft_total = channels(g["B_yB@init"], g["B_yB@A1"])
    ft_anch = math.log(g["B_ystar@A1"]["alpha"] / g["B_ystar@init"]["alpha"])
    ft_neu = {str(k): math.log(g[f"B_ystarN{k}@A1"]["alpha"] / g[f"B_ystarN{k}@init"]["alpha"])
              for k in range(len(neutral))}
    ft_neu_mean = float(np.mean(list(ft_neu.values())))
    steps = {k: v["steps"] for k, v in ph.items()}
    out = {
        "key": spec.key, "spec": asdict(spec), "config": cfg.key_fields(),
        "lib_hash": lib_hash(), "stamp": stamp or git_stamp(),
        "usable": all(v["converged"] for v in ph.values()),
        "status": {k: v["status"] for k, v in ph.items()},
        "init_loss": init_loss, "lr": net.lr, "output_scale": net.s,
        "relations": {
            "s_r_AB": s_r(y_A, y_B), "s_r_ystar_A": s_r(y_star, y_A),
            "s_r_ystar_B": s_r(y_star, y_B),
            "s_r_ystar_neutral_A": {str(k): s_r(yk, y_A) for k, yk in enumerate(neutral)},
            "s_r_ystar_neutral_B": {str(k): s_r(yk, y_B) for k, yk in enumerate(neutral)},
            "rho_in_AB": float(np.mean(np.diag(centre_cosines(A.centers, B.centers)))),
        },
        "geometry": g,
        "backward": {"init_to_A1": channels(g["A@init"], g["A@A1"]),
                     "A1_to_B": channels(g["A@A1"], g["A@B"]),
                     "B_to_A2": channels(g["A@B"], g["A@A2"]),
                     "A1_to_A2": channels(g["A@A1"], g["A@A2"])},
        "forward": {"total": ft_total,
                    "ystar_neutral": {**ft_neu, "mean": ft_neu_mean},
                    "specific": ft_total["total"] - ft_neu_mean,
                    "ystar_anchored": ft_anch,
                    "specific_anchored": ft_total["total"] - ft_anch},
        "refit": rf,
        "behaviour": {
            "steps": steps,
            "first_loss": {k: v["first_loss"] for k, v in ph.items()},
            "FT_beh": _logratio_steps(steps["freshB"], steps["B"]),
            "savings": _logratio_steps(steps["A1"], steps["A2"]),
            "forget_B_acc": forgetting_running_max(ph["B"]["records"], "A_test_acc"),
            "forget_B_loss": forgetting_running_max(ph["B"]["records"], "A_test_loss"),
            "A_test_loss_end": {k: ph[k]["records"][-1]["A_test_loss"] for k in ("A1", "B", "A2")},
        },
        "richness": {**{k: v for k, v in r.items()},
                     "r_t": {k: ph[k]["r_t"] for k in ("A1", "B", "A2")}},
        "trajectory": {k: ph[k]["records"] for k in ("A1", "B", "A2")},
    }
    timing["wall_s"] = time.perf_counter() - t_start
    out["timing"] = timing
    return out


# ==========================================================================
# Grid (spawn workers; JSONL checkpoint keyed by lib hash + config + spec)
# ==========================================================================


def all_specs(cfg: ABAConfig) -> list[ArmSpec]:
    """The design grid plus the fresh-noise identity check (Hamming 0, s_f 1.0, seed 0)."""
    grid = [ArmSpec(s, float(g), int(h), float(f))
            for s in range(cfg.n_seeds) for g in cfg.gammas
            for h in cfg.hammings for f in cfg.s_fs]
    fresh = [ArmSpec(0, float(g), 0, 1.0, "fresh") for g in cfg.gammas]
    return grid + fresh


def trained_floor(cfg: ABAConfig, gamma: float, seed: int = 0, K: int = 6) -> dict:
    """Noise floor on the representation after training on A: SD(log α_A) over K
    estimator seeds, at this config's n_t. The resolution unit for Δlog α at this γ0."""
    base = seed_base(cfg, seed)
    net = OneModuleNet(base.W0, gamma, cfg)
    X, y = flatten(base.A.points, base.y_A)
    train_to_criterion(net, X, y, cfg)
    logs = [math.log(rep_capacity(net, base.A.points, base.y_A, cfg, [SALT, seed, 2000 + k])["alpha"])
            for k in range(K)]
    return {"gamma": gamma, "seed": seed, "K": K, "n_t": cfg.n_t,
            "floor": float(np.std(logs, ddof=1)), "mean_log_alpha": float(np.mean(logs))}


# Every per-arm output the notebook reads, as dotted paths into the result dict.
_CH = ("total", "utility", "radius", "dimension", "residual")
SCHEMA: dict[str, str] = {
    **{f"backward.{p}.{c}": f"Δlog α_A {p}, channel {c}" for p in
       ("init_to_A1", "A1_to_B", "B_to_A2", "A1_to_A2") for c in _CH},
    **{f"behaviour.forget_B_{m}.{k}": f"forgetting on A during B vs running {'max' if m == 'acc' else 'min'} ({k})"
       for m in ("acc", "loss") for k in ("peak", "final", "forgetting")},
    **{f"forward.total.{c}": f"Δlog α_{{y_B}} on B's manifolds, init → after A, {c}" for c in _CH},
    **{f"forward.ystar_neutral.{k}": f"Δlog α_{{y*_{k}}} (neutral, s_r = 0 to y_A) on B's manifolds, init → after A"
       for k in ("0", "1", "2", "3", "mean")},
    "forward.specific": "FT_geo specific = Δlog α_{y_B} − mean_k Δlog α_{y*_k} (neutral baseline)",
    "forward.ystar_anchored": "Δlog α_{y*} for the anchored probe (s_r = +0.5 to y_A); comparison only",
    "forward.specific_anchored": "Δlog α_{y_B} − Δlog α_{y* anchored}; comparison only",
    "relations.s_r_AB": "s_r(y_A, y_B), signed",
    "relations.s_r_ystar_A": "s_r(y* anchored, y_A), signed",
    "relations.s_r_ystar_B": "s_r(y* anchored, y_B), signed",
    **{f"relations.s_r_ystar_neutral_{t}.{k}": f"s_r(y*_{k} neutral, y_{t})"
       for t in ("A", "B") for k in ("0", "1", "2", "3")},
    **{f"refit.{ph}.{m}": f"fresh readout on frozen features after {ph}, {m}"
       for ph in ("init", "A1", "B", "A2")
       for m in ("loss", "acc", "train_loss", "loss@100", "acc@100", "loss@500", "acc@500")},
    **{f"richness.{ph}.{k}": f"{k}: kernel change from previous boundary to {ph}"
       for ph in ("A1", "B", "A2") for k in ("dK_full_step", "dK_u_step", "dK_W_step", "align_step")},
    "relations.rho_in_AB": "mean centre cosine A ↔ B (input relation)",
    "behaviour.FT_beh": "log((1+steps fresh B)/(1+steps B))",
    "behaviour.savings": "log((1+steps A1)/(1+steps A2))",
    **{f"behaviour.steps.{p}": f"raw steps, {p}" for p in ("A1", "B", "A2", "freshB")},
    **{f"behaviour.first_loss.{p}": f"zero-shot loss, {p}" for p in ("A1", "B", "A2", "freshB")},
    **{f"richness.{ph}.{k}": f"{k} after {ph}" for ph in ("init", "A1", "B", "A2")
       for k in ("weight_change", "dK_full", "dK_u", "dK_W", "KW_share", "cka_K_yA", "cka_K_yB")},
    **{f"richness.r_t.{p}": f"r_t within {p}" for p in ("A1", "B", "A2")},
    **{f"geometry.{b}.{k}": f"{k} at {b}" for b in ("A@init", "A@A1", "A@B", "A@A2")
       for k in ("rho_c_signed", "rho_c_glue", "alpha", "identity_residual")},
}


def check_schema(row: dict) -> list[str]:
    """Paths in SCHEMA that are missing or non-finite in `row`."""
    bad = []
    for path in SCHEMA:
        v = row
        try:
            for part in path.split("."):
                v = v[part]
            if not np.isfinite(float(v)):
                bad.append(f"{path}={v}")
        except (KeyError, TypeError, ValueError):
            bad.append(f"{path} missing")
    return bad


def cache_path(cfg: ABAConfig) -> Path:
    return REPO / "results" / "aba" / f"{'QUICK' if cfg.quick else 'FULL'}_{lib_hash()}.jsonl"


def load_results(cfg: ABAConfig) -> list[dict]:
    p = cache_path(cfg)
    if not p.exists():
        return []
    want = json.dumps(cfg.key_fields(), sort_keys=True)
    rows = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if json.dumps(row["config"], sort_keys=True) == want:
                rows.append(row)
    return rows


def _arm_job(args) -> dict:
    cfg, spec, stamp = args
    return run_arm(cfg, spec, stamp)


def run_grid(cfg: ABAConfig, specs: list[ArmSpec], *, workers: int | None = None,
             progress=print) -> list[dict]:
    """Run missing arms in spawned workers; append each result to the checkpoint."""
    path = cache_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    done = {r["key"]: r for r in load_results(cfg)}
    todo = [s for s in specs if s.key not in done]
    stamp = git_stamp()
    if todo:
        workers = workers or min(6, max(1, (os.cpu_count() or 2) - 1))   # laptop memory cap
        with ProcessPoolExecutor(max_workers=min(workers, len(todo)),
                                 mp_context=get_context("spawn")) as ex:
            futs = {ex.submit(_arm_job, (cfg, s, stamp)): s for s in todo}
            for i, fut in enumerate(as_completed(futs), 1):
                res = fut.result()
                with path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(res) + "\n")
                done[res["key"]] = res
                progress(f"[{i}/{len(todo)}] {res['key']}  wall {res['timing']['wall_s']:.0f}s"
                         f"  usable={res['usable']}")
    return [done[s.key] for s in specs if s.key in done]
