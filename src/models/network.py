"""Two-module ReLU network with analytic gradients (`00` §4.1).

    h_m(x) = ReLU(β₀ · W_m x),          m ∈ {A, B},  W_m ∈ R^{N×d}
    f(x)   = Σ_m (β_L / γ_m) · u_mᵀ h_m(x)

(`00` §4.1 writes `u_mᵀ ReLU(h_m(x))`; ReLU is idempotent so the two readings
coincide.)

Implemented in numpy with hand-derived gradients rather than in a DL framework:
the model is two layers with an MSE loss, so the gradients are four lines, and
this keeps float64 throughout and makes every run bitwise reproducible from
`(seed, config)` — both required by `AGENTS.md` §4. Phase 1 is thousands of tiny
independent runs on CPU, which suits many single-threaded processes rather than
one accelerator.

Invariants enforced here:

- **I5** `f(x; θ₀) = 0` exactly, because `u_m` is initialized to zero.
- **I6** `σ² = 1` in both parameterizations and hidden weights are drawn from a
  γ-independent RNG stream, so `W_m(0)` is **bitwise identical across γ₀**. This
  is what makes `γ` and `a` orthogonal.
- **I8** one shared readout; no task-conditioned head.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .parameterization import ScalingConfig

MODULES = ("A", "B")


@dataclass
class TwoModuleNet:
    """Modules share the input and sum into a single scalar readout."""

    cfg: dict[str, ScalingConfig]
    W: dict[str, np.ndarray]
    u: dict[str, np.ndarray]
    _init_W: dict[str, np.ndarray] = field(default_factory=dict, repr=False)

    # -- construction --------------------------------------------------------

    @classmethod
    def init(
        cls,
        cfg: dict[str, ScalingConfig],
        rng: np.random.Generator,
        *,
        W_init: dict[str, np.ndarray] | None = None,
    ) -> "TwoModuleNet":
        """Draw `W_m ~ N(0, 1)` (σ² = 1, both parameterizations) and set `u_m = 0`.

        `rng` must be the **shape stream**, independent of γ — see `paired_init`.
        `W_init` overrides the draw, which is how the alignment knob injects a
        rotated initialization at fixed Frobenius norm and rank.
        """
        W, u = {}, {}
        for m in MODULES:
            c = cfg[m]
            W[m] = (
                np.ascontiguousarray(W_init[m], dtype=np.float64)
                if W_init is not None
                else rng.standard_normal((c.N, c.d))
            )
            if W[m].shape != (c.N, c.d):
                raise ValueError(f"W[{m}] must be (N={c.N}, d={c.d}); got {W[m].shape}")
            u[m] = np.zeros(c.N, dtype=np.float64)
        return cls(cfg=cfg, W=W, u=u, _init_W={m: W[m].copy() for m in MODULES})

    # -- forward -------------------------------------------------------------

    def hidden(self, X: np.ndarray, module: str) -> np.ndarray:
        """`h_m(X)` for `X` of shape `(B, d)`. Returns `(B, N)`."""
        c = self.cfg[module]
        return np.maximum(np.asarray(X, dtype=np.float64) @ self.W[module].T * c.beta_0, 0.0)

    def forward(self, X: np.ndarray) -> np.ndarray:
        """`f(X)` for `X` of shape `(B, d)`. Returns `(B,)`."""
        out = np.zeros(np.asarray(X).shape[0], dtype=np.float64)
        for m in MODULES:
            out += self.cfg[m].output_scale * (self.hidden(X, m) @ self.u[m])
        return out

    def representation(self, X: np.ndarray, module: str | None = None) -> np.ndarray:
        """Hidden representation for geometry measurement.

        `module=None` concatenates the modules. Cross-module comparability needs
        the `00` §6.3 preprocessing (`raw` / `gaussianized`) applied downstream —
        widths and scales differ, so concatenated raw activations are not directly
        comparable between modules.
        """
        if module is not None:
            return self.hidden(X, module)
        return np.concatenate([self.hidden(X, m) for m in MODULES], axis=1)

    def manifold_representation(
        self, points: np.ndarray, module: str | None = None
    ) -> list[np.ndarray]:
        """`(P, M, d)` input manifolds → list of `(N, M)` arrays, the GLUE layout."""
        pts = np.asarray(points, dtype=np.float64)
        return [self.representation(pts[mu], module).T.copy() for mu in range(pts.shape[0])]

    # -- gradients -----------------------------------------------------------

    def grads(self, X: np.ndarray, y: np.ndarray) -> tuple[dict, dict, float]:
        """Analytic gradients of `L = mean_b ½(f(x_b) − y_b)²`.

            dL/du_m = (1/B) Σ_b e_b · s_m · h_m(x_b)
            dL/dW_m = (1/B) Σ_b e_b · s_m · β₀ · (u_m ⊙ 1[z_m > 0]) x_bᵀ

        Note `dL/dW_m ∝ u_m`, so with `u_m(0) = 0` the first step moves only the
        readout. That is a property of the zero-init, not a bug.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        B = X.shape[0]
        H = {m: self.hidden(X, m) for m in MODULES}
        err = sum(self.cfg[m].output_scale * (H[m] @ self.u[m]) for m in MODULES) - y

        gW, gu = {}, {}
        for m in MODULES:
            c = self.cfg[m]
            gu[m] = c.output_scale * (H[m].T @ err) / B
            # d h / d z is the ReLU mask; H[m] > 0 is exactly 1[z > 0].
            delta = (H[m] > 0) * (c.output_scale * err)[:, None] * self.u[m][None, :]
            gW[m] = c.beta_0 * (delta.T @ X) / B
        return gW, gu, float(0.5 * np.mean(err**2))

    def sgd_step(self, X: np.ndarray, y: np.ndarray) -> float:
        """One SGD step at the per-module learning rate. Returns the pre-step loss."""
        gW, gu, loss = self.grads(X, y)
        for m in MODULES:
            lr = self.cfg[m].lr
            self.W[m] -= lr * gW[m]
            self.u[m] -= lr * gu[m]
        return loss

    # -- diagnostics ---------------------------------------------------------

    def weight_change(self, module: str) -> float:
        """`‖W_m − W_m(0)‖_F / ‖W_m(0)‖_F` — the standard richness readout."""
        W0 = self._init_W[module]
        return float(np.linalg.norm(self.W[module] - W0) / np.linalg.norm(W0))

    def copy(self) -> "TwoModuleNet":
        return TwoModuleNet(
            cfg=dict(self.cfg),
            W={m: self.W[m].copy() for m in MODULES},
            u={m: self.u[m].copy() for m in MODULES},
            _init_W={m: v.copy() for m, v in self._init_W.items()},
        )


def paired_init(seed: int) -> dict[str, np.random.Generator]:
    """Named RNG streams so conditions are *paired*, not merely seeded alike.

    `shape` draws `W_m(0)` and must never be consumed by anything γ-dependent —
    that is what makes hidden weights bitwise identical across γ (I6). `data` and
    `stream` are separate so changing the training schedule cannot perturb the
    initialization, and vice versa.
    """
    shape, data, stream, probe = np.random.default_rng(seed).spawn(4)
    return {"shape": shape, "data": data, "stream": stream, "probe": probe}
