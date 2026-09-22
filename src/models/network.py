"""Two-module ReLU network with analytic gradients (`00` §4.1).

    h¹_m(x) = ReLU(β₀ · W_m x),                W_m ∈ R^{N×d}
    h²_m(x) = ReLU(β_hid · W²_m h¹_m(x)),     W²_m ∈ R^{N×N}   (L = 3 only)
    f(x)    = Σ_m (β_L / γ_m) · u_mᵀ h^{last}_m(x)

L = 2 is the registered model (`n_hidden_layers=1`). L = 3 adds the hidden-type
tensor. Sequential training at L = 3 stays refused until Verification 2
(`docs/16` A6 / A21). Diagnostic SGD on a toy batch is a separate entrypoint.

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


def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    out = np.empty_like(z, dtype=np.float64)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def bce_with_logits(f: np.ndarray, y_pm1: np.ndarray) -> float:
    """Mean BCE of logits `f` against labels in `{±1}` (remapped to `{0,1}`)."""
    f = np.asarray(f, dtype=np.float64).ravel()
    y = np.asarray(y_pm1, dtype=np.float64).ravel()
    y01 = 0.5 * (y + 1.0)
    sp = np.where(f >= 0.0, f + np.log1p(np.exp(-f)), np.log1p(np.exp(f)))
    return float(np.mean(sp - y01 * f))


@dataclass
class TwoModuleNet:
    """Modules share the input and sum into a single scalar readout."""

    cfg: dict[str, ScalingConfig]
    W: dict[str, np.ndarray]
    u: dict[str, np.ndarray]
    W2: dict[str, np.ndarray] = field(default_factory=dict)
    Q: dict[str, np.ndarray] = field(default_factory=dict)
    readout_rank: int | None = None
    _init_W: dict[str, np.ndarray] = field(default_factory=dict, repr=False)
    _init_W2: dict[str, np.ndarray] = field(default_factory=dict, repr=False)
    n_hidden_layers: int = 1
    n_outputs: int = 1
    _adam_t: int = field(default=0, repr=False, compare=False)
    _adam_m: dict = field(default_factory=dict, repr=False, compare=False)
    _adam_v: dict = field(default_factory=dict, repr=False, compare=False)

    # -- construction --------------------------------------------------------

    @classmethod
    def init(
        cls,
        cfg: dict[str, ScalingConfig],
        rng: np.random.Generator,
        *,
        W_init: dict[str, np.ndarray] | None = None,
        n_hidden_layers: int = 1,
        n_outputs: int = 1,
        readout_rank: int | None = None,
        projection_rng: np.random.Generator | None = None,
    ) -> "TwoModuleNet":
        """Draw `W_m ~ N(0, 1)` (σ² = 1, both parameterizations) and set `u_m = 0`.

        `rng` must be the **shape stream**, independent of γ — see `paired_init`.
        `W_init` overrides the draw, which is how the alignment knob injects a
        rotated initialization at fixed Frobenius norm and rank.
        `n_hidden_layers=2` adds a second hidden layer (`L = 3`). Stream
        training at that depth is refused until Verification 2 is signed
        (`docs/16` A6 / A21). The forward pass and diagnostic SGD do not
        wait on that signature.
        `readout_rank` r places a frozen Q_m ∈ ℝ^{r×N} on the readout path.
        Q is nested: one N×N orthonormal basis from `projection_rng`, first r
        rows. `readout_rank=None` is the registered path (Q = I_N, no draw).
        """
        if n_hidden_layers not in (1, 2):
            raise ValueError(f"n_hidden_layers must be 1 or 2; got {n_hidden_layers}")
        if n_outputs < 1:
            raise ValueError(f"n_outputs must be ≥ 1; got {n_outputs}")
        W, u, W2, Q = {}, {}, {}, {}
        for m in MODULES:
            c = cfg[m]
            r = c.N if readout_rank is None else int(readout_rank)
            if r < 1 or r > c.N:
                raise ValueError(f"readout_rank must be in 1..N={c.N}; got {r}")
            if n_outputs != 1 and readout_rank is not None:
                raise ValueError("rank projection is scalar-readout only")
            W[m] = (
                np.ascontiguousarray(W_init[m], dtype=np.float64)
                if W_init is not None
                else rng.standard_normal((c.N, c.d))
            )
            if W[m].shape != (c.N, c.d):
                raise ValueError(f"W[{m}] must be (N={c.N}, d={c.d}); got {W[m].shape}")
            if n_hidden_layers == 2:
                W2[m] = rng.standard_normal((c.N, c.N))
            if readout_rank is None:
                Q[m] = np.eye(c.N, dtype=np.float64)
            else:
                Q[m] = draw_readout_projection(c.N, r, projection_rng)
            u[m] = np.zeros(r if n_outputs == 1 else (n_outputs, c.N), dtype=np.float64)
        return cls(cfg=cfg, W=W, u=u, W2=W2, Q=Q, readout_rank=readout_rank,
                   _init_W={m: W[m].copy() for m in MODULES},
                   _init_W2={m: W2[m].copy() for m in W2},
                   n_hidden_layers=n_hidden_layers, n_outputs=n_outputs)

    # -- forward -------------------------------------------------------------

    def hidden(self, X: np.ndarray, module: str) -> np.ndarray:
        """Last hidden layer for `X` of shape `(B, d)`. Returns `(B, N)`."""
        return self._hidden_chain(X, module)[-1]

    def _hidden_chain(self, X: np.ndarray, module: str) -> tuple[np.ndarray, ...]:
        """Hidden activations from first to last layer. Each `(B, N)`."""
        c = self.cfg[module]
        h1 = np.maximum(np.asarray(X, dtype=np.float64) @ self.W[module].T * c.beta_0, 0.0)
        if self.n_hidden_layers == 1:
            return (h1,)
        h2 = np.maximum(h1 @ self.W2[module].T * c.beta_hid, 0.0)
        return (h1, h2)

    def preactivations(self, X: np.ndarray, module: str) -> tuple[np.ndarray, ...]:
        """Pre-ReLU activations `(z1,)` or `(z1, z2)`, each `(B, N)`."""
        c = self.cfg[module]
        z1 = np.asarray(X, dtype=np.float64) @ self.W[module].T * c.beta_0
        if self.n_hidden_layers == 1:
            return (z1,)
        h1 = np.maximum(z1, 0.0)
        z2 = h1 @ self.W2[module].T * c.beta_hid
        return (z1, z2)

    def forward(self, X: np.ndarray) -> np.ndarray:
        """`f(X)` for `X` of shape `(B, d)`. Returns `(B,)` or `(B, n_outputs)`."""
        B = np.asarray(X).shape[0]
        if self.n_outputs == 1:
            out = np.zeros(B, dtype=np.float64)
            for m in MODULES:
                out += self.cfg[m].output_scale * (self._projected(X, m) @ self.u[m])
            return out
        out = np.zeros((B, self.n_outputs), dtype=np.float64)
        for m in MODULES:
            out += self.cfg[m].output_scale * (self.hidden(X, m) @ self.u[m].T)
        return out

    def _projected(self, X: np.ndarray, module: str) -> np.ndarray:
        """Readout-path features: `h Qᵀ`, shape `(B, r)`. Q is frozen."""
        return self.hidden(X, module) @ self.Q[module].T

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

    def grads(
        self, X: np.ndarray, y: np.ndarray, *, diagnostic: bool = False,
        loss: str = "mse",
    ) -> tuple[dict, dict, float, dict]:
        """Analytic gradients of MSE `L = mean_b ½(f − y)²` or of BCE on logits.

        BCE remaps `y ∈ {±1}` to `{0,1}` inside the loss (Chou D.1.1). `dL/df`
        is `σ(f) − y_{01}`; the rest of the backprop is the MSE chain with that
        error. Scalar readout only.

            dL/du_m = (1/B) Σ_b e_b · s_m · h_m(x_b)
            dL/dW_m = (1/B) Σ_b e_b · s_m · β₀ · (u_m ⊙ 1[z_m > 0]) x_bᵀ

        At L = 3 the last-layer delta backprops through `W²` with scale `β_hid`.
        The same `cfg.lr` will be applied to `W`, `W²`, and `u`.

        Note `dL/dW_m ∝ u_m` at L = 2, and `dL/dW²_m ∝ u_m` at L = 3, so with
        `u_m(0) = 0` the first step moves only the readout. That is a property
        of the zero-init, not a bug.

        L = 3 requires `diagnostic=True`. Stream training stays blocked
        (`docs/16` A6 / A21).
        """
        if self.n_hidden_layers != 1 and not diagnostic:
            raise NotImplementedError(
                "L=3 training is blocked on the human μP derivation "
                "(docs/16 §Amendments A6). Forward pass and init-α are allowed.")
        if loss not in ("mse", "bce"):
            raise ValueError(f"unknown loss {loss!r}")
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        B = X.shape[0]
        chain = {m: self._hidden_chain(X, m) for m in MODULES}
        H = {m: chain[m][-1] for m in MODULES}
        pred = self.forward(X)
        if loss == "mse":
            err = pred - y
            loss_val = float(0.5 * np.mean(err ** 2))
        else:
            if self.n_outputs != 1:
                raise ValueError("bce is scalar-readout only")
            y_r = y.ravel()
            if not np.all(np.isin(y_r, (-1.0, 1.0))):
                raise ValueError("bce expects ±1 labels; remapping is inside the loss")
            pred_r = np.asarray(pred, dtype=np.float64).ravel()
            y01 = 0.5 * (y_r + 1.0)
            err = sigmoid(pred_r) - y01
            loss_val = bce_with_logits(pred_r, y_r)
        gW, gu, gW2 = {}, {}, {}
        for m in MODULES:
            c = self.cfg[m]
            Q = self.Q[m]
            if self.n_outputs == 1:
                Hp = H[m] @ Q.T
                gu[m] = c.output_scale * (Hp.T @ err) / B
                v = Q.T @ self.u[m]
                delta = (H[m] > 0) * (c.output_scale * err)[:, None] * v[None, :]
            else:
                gu[m] = c.output_scale * (err.T @ H[m]) / B
                delta = (H[m] > 0) * (c.output_scale * (err @ self.u[m]))
            if self.n_hidden_layers == 1:
                gW[m] = c.beta_0 * (delta.T @ X) / B
            else:
                h1 = chain[m][0]
                gW2[m] = c.beta_hid * (delta.T @ h1) / B
                delta1 = (h1 > 0) * (c.beta_hid * (delta @ self.W2[m]))
                gW[m] = c.beta_0 * (delta1.T @ X) / B
        return gW, gu, loss_val, gW2

    def sgd_step(self, X: np.ndarray, y: np.ndarray, *, diagnostic: bool = False,
                 loss: str = "mse") -> float:
        """One SGD step at the per-module learning rate. Returns the pre-step loss.

        The same `lr` is applied to `W_m`, `u_m`, and `W²_m` when that tensor
        exists. L = 3 requires `diagnostic=True`.
        """
        gW, gu, loss_val, gW2 = self.grads(X, y, diagnostic=diagnostic, loss=loss)
        for m in MODULES:
            lr = self.cfg[m].lr
            self.W[m] -= lr * gW[m]
            self.u[m] -= lr * gu[m]
            if m in gW2:
                self.W2[m] -= lr * gW2[m]
        return loss_val

    def adam_step(self, X: np.ndarray, y: np.ndarray, *, diagnostic: bool = False,
                  loss: str = "mse", beta1: float = 0.9, beta2: float = 0.999,
                  eps: float = 1e-8) -> float:
        """One Adam step at the per-module learning rate. No weight decay (I1).

        Moments persist on this object across calls, including across tasks
        of a stream. Bias-corrected, library defaults (`docs/22`). Returns
        the pre-step loss. Off-design: core runs stay on `sgd_step`.
        """
        if eps <= 0:
            raise ValueError("Adam eps must be positive")
        gW, gu, loss_val, gW2 = self.grads(X, y, diagnostic=diagnostic, loss=loss)
        self._adam_t += 1
        t = self._adam_t
        for m in MODULES:
            lr = self.cfg[m].lr
            self._adam_apply(self.W[m], gW[m], ("W", m), lr, beta1, beta2, eps, t)
            self._adam_apply(self.u[m], gu[m], ("u", m), lr, beta1, beta2, eps, t)
            if m in gW2:
                self._adam_apply(self.W2[m], gW2[m], ("W2", m), lr, beta1, beta2, eps, t)
        return loss_val

    def _adam_apply(self, param: np.ndarray, grad: np.ndarray, key: tuple,
                    lr: float, beta1: float, beta2: float, eps: float, t: int) -> None:
        m = self._adam_m.get(key)
        if m is None:
            m = np.zeros_like(param)
            v = np.zeros_like(param)
            self._adam_m[key] = m
            self._adam_v[key] = v
        else:
            v = self._adam_v[key]
        m *= beta1
        m += (1.0 - beta1) * grad
        v *= beta2
        v += (1.0 - beta2) * (grad * grad)
        mhat = m / (1.0 - beta1 ** t)
        vhat = v / (1.0 - beta2 ** t)
        param -= lr * mhat / (np.sqrt(vhat) + eps)

    def step(self, X: np.ndarray, y: np.ndarray, *, diagnostic: bool = False,
             loss: str = "mse", optimizer: str = "sgd",
             adam_beta1: float = 0.9, adam_beta2: float = 0.999,
             adam_eps: float = 1e-8) -> float:
        """Dispatch one update. Default is SGD; Adam is the `docs/22` opt-in."""
        if optimizer == "sgd":
            return self.sgd_step(X, y, diagnostic=diagnostic, loss=loss)
        if optimizer == "adam":
            return self.adam_step(X, y, diagnostic=diagnostic, loss=loss,
                                  beta1=adam_beta1, beta2=adam_beta2, eps=adam_eps)
        raise ValueError(f"unknown optimizer {optimizer!r}")

    # -- diagnostics ---------------------------------------------------------

    def weight_change(self, module: str) -> float:
        """`‖W_m − W_m(0)‖_F / ‖W_m(0)‖_F` — the standard richness readout."""
        W0 = self._init_W[module]
        return float(np.linalg.norm(self.W[module] - W0) / np.linalg.norm(W0))

    def weight_change_W2(self, module: str) -> float:
        """`‖W²_m − W²_m(0)‖_F / ‖W²_m(0)‖_F`. L = 3 only."""
        W0 = self._init_W2[module]
        return float(np.linalg.norm(self.W2[module] - W0) / np.linalg.norm(W0))

    def copy(self) -> "TwoModuleNet":
        return TwoModuleNet(
            cfg=dict(self.cfg),
            W={m: self.W[m].copy() for m in MODULES},
            u={m: self.u[m].copy() for m in MODULES},
            W2={m: v.copy() for m, v in self.W2.items()},
            Q={m: v.copy() for m, v in self.Q.items()},
            readout_rank=self.readout_rank,
            _init_W={m: v.copy() for m, v in self._init_W.items()},
            _init_W2={m: v.copy() for m, v in self._init_W2.items()},
            n_hidden_layers=self.n_hidden_layers,
            n_outputs=self.n_outputs,
            _adam_t=self._adam_t,
            _adam_m={k: v.copy() for k, v in self._adam_m.items()},
            _adam_v={k: v.copy() for k, v in self._adam_v.items()},
        )


def paired_init(seed: int) -> dict[str, np.random.Generator]:
    """Named RNG streams so conditions are *paired*, not merely seeded alike.

    `shape` draws `W_m(0)` and must never be consumed by anything γ-dependent —
    that is what makes hidden weights bitwise identical across γ (I6). `data` and
    `stream` are separate so changing the training schedule cannot perturb the
    initialization, and vice versa.

    Do not extend `spawn(4)`. A fifth stream would shift every existing paired
    draw. The readout projection has its own generator (`projection_rng`).
    """
    shape, data, stream, probe = np.random.default_rng(seed).spawn(4)
    return {"shape": shape, "data": data, "stream": stream, "probe": probe}


PROJECTION_SALT = 20260901


def projection_rng(seed: int) -> np.random.Generator:
    """Draws the nested Q basis. Independent of γ, of r, and of the shape stream."""
    return np.random.default_rng([PROJECTION_SALT, int(seed)])


def draw_readout_projection(
    N: int, r: int, rng: np.random.Generator | None
) -> np.ndarray:
    """Frozen Q ∈ ℝ^{r×N}: first r rows of one N×N orthonormal draw.

    Nested: Q_r = Q_N[:r, :]. Rank does not redraw. `rng` is required.
    """
    if rng is None:
        raise ValueError("projection_rng is required when readout_rank is set")
    if r < 1 or r > N:
        raise ValueError(f"readout_rank must be in 1..N={N}; got {r}")
    A = rng.standard_normal((N, N))
    Qfull, _ = np.linalg.qr(A)
    return np.ascontiguousarray(Qfull[:, :r].T, dtype=np.float64)
