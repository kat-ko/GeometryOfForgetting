"""Sequential training over a task stream (`00` §3, `01` Phase 1).

One task = one balanced dichotomy `y_t` over the `P` manifolds of that task's
arrangement. Targets are `±1` per manifold, shared by every point of the manifold,
so the network must map a whole manifold to one side — which is what makes
manifold capacity the right readout.

Boundaries are where everything is measured. The loop records performance on the
current task, on every past task (retained), and on a held-out probe dichotomy
never trained on; geometry is measured separately by `src/analysis/` from the
representations this loop exposes, so that the expensive estimator can run at a
Tier-2 subset of boundaries without re-training.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..models import MODULES, TwoModuleNet


@dataclass(frozen=True)
class TrainConfig:
    """`01` §1: `stopping ∈ {"matched_loss", "fixed_steps"}`.

    **Stopping must be on loss, not accuracy.** From `u_m(0) = 0`, a single step
    gives `u_m ∝ Σ_b y_b h(x_b)` — the kernel/Hebbian readout — and the *sign* of
    `f` is then independent of the learning rate and of `γ`. So train accuracy
    jumps to ~0.99 at step 1 for every `γ`, and an accuracy criterion halts
    training before any feature learning occurs, which is the thing under study.
    Loss keeps falling long after accuracy saturates, so `target_loss` is what
    actually matches training progress across `γ` (measured: `01` Phase 0).
    `loss="bce"` is the second-learner arm (`docs/21`): same architecture and
    streams, binary cross-entropy with labels remapped `{±1}→{0,1}` inside
    the step. Default remains MSE. `worst_of_k` is MSE-only.
    `optimizer="adam"` is the second-learner arm (`docs/22`): same MSE pin,
    different update. Default remains SGD (I3). No weight decay.
    """

    steps_per_task: int = 2000
    batch_size: int | None = None      # None = full batch
    stopping: str = "matched_loss"     # "matched_loss" | "fixed_steps" | "worst_of_k"
    target_loss: float = 0.05
    record_every: int = 50
    loss: str = "mse"  # "mse" | "bce"; BCE remaps ±1 → {0,1} inside the step
    optimizer: str = "sgd"  # "sgd" | "adam"; Adam is docs/22, never a core default
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8


@dataclass
class TaskRecord:
    task: int
    final_loss: float
    steps_taken: int
    train_accuracy: float
    loss_curve: list[float]
    weight_change: dict[str, float]
    converged: bool = True

    @property
    def usable_for_forgetting(self) -> bool:
        """A task the network never learned cannot meaningfully be forgotten.

        Retained-capacity and forgetting numbers for a non-converged task confound
        forgetting with under-training, so any run containing one is invalid for
        H1/H2 and must be reported, not silently averaged in. Under
        `matched_loss`, `converged` additionally certifies that training progress
        is *matched across `γ`*, without which the γ contrast is confounded with
        how far each arm got.
        """
        return self.converged


@dataclass
class BoundaryRecord:
    """Measured after finishing task `t`. Indices are absolute task indices."""

    after_task: int
    current_accuracy: float
    retained_accuracy: dict[int, float]
    probe_accuracy: float
    weight_change: dict[str, float]
    mean_retained_accuracy: float = field(init=False)

    def __post_init__(self) -> None:
        vals = list(self.retained_accuracy.values())
        self.mean_retained_accuracy = float(np.mean(vals)) if vals else float("nan")


def flatten_task(points: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """`(P, M, d)` manifolds + `(P,)` or `(P, K)` labels → `(P·M, d)` and `(P·M,)` or `(P·M, K)`."""
    pts = np.asarray(points, dtype=np.float64)
    P, M, d = pts.shape
    y = np.asarray(y, dtype=np.float64)
    X = pts.reshape(P * M, d)
    if y.ndim == 1:
        if y.shape != (P,):
            raise ValueError(f"dichotomy must be ({P},); got {y.shape}")
        return X, np.repeat(y, M)
    if y.ndim == 2:
        if y.shape[0] != P:
            raise ValueError(f"labels must be ({P}, K); got {y.shape}")
        return X, np.repeat(y, M, axis=0)
    raise ValueError(f"labels must be ({P},) or ({P}, K); got {y.shape}")


def flatten_group(
    point_list: list[np.ndarray],
    ys: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """K arrangements and K dichotomies → one multi-output batch.

    ``ys`` is ``(K, P)``. Each arrangement is labeled with all K dichotomies
    (manifold identities persist; the cut is which dichotomies are one task).
    """
    ys = np.asarray(ys, dtype=np.float64)
    if ys.ndim != 2:
        raise ValueError(f"group dichotomies must be (K, P); got {ys.shape}")
    K, P = ys.shape
    if len(point_list) != K:
        raise ValueError(f"{len(point_list)} arrangements vs K={K} dichotomies")
    Xs, Ys = [], []
    y_pk = ys.T  # (P, K)
    for pts in point_list:
        if pts.shape[0] != P:
            raise ValueError(f"arrangement P={pts.shape[0]} vs dichotomy P={P}")
        X, Y = flatten_task(pts, y_pk)
        Xs.append(X)
        Ys.append(Y)
    return np.concatenate(Xs, axis=0), np.concatenate(Ys, axis=0)


def per_output_mse(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    """½ mean_b (f_k − y_k)² per output. Scalar target → length-1 array."""
    err2 = (np.asarray(pred, dtype=np.float64) - np.asarray(target, dtype=np.float64)) ** 2
    if err2.ndim == 1:
        return np.array([0.5 * float(np.mean(err2))], dtype=np.float64)
    return 0.5 * np.mean(err2, axis=0)


def accuracy(
    model: TwoModuleNet,
    X: np.ndarray,
    y: np.ndarray,
    *,
    output_index: int | None = None,
) -> float:
    """Sign agreement. `f = 0` counts as wrong, so accuracy at init is 0, not 0.5."""
    pred = model.forward(X)
    target = np.asarray(y)
    if pred.ndim == 2:
        k = 0 if output_index is None else output_index
        pred = pred[:, k]
        if target.ndim == 2:
            target = target[:, k]
    return float(np.mean(np.sign(pred) == np.sign(target)))


def manifold_accuracy(
    model: TwoModuleNet,
    points: np.ndarray,
    y: np.ndarray,
    *,
    output_index: int | None = None,
) -> float:
    X, target = flatten_task(points, y)
    return accuracy(model, X, target, output_index=output_index)


def train_task(
    model: TwoModuleNet,
    points: np.ndarray,
    y: np.ndarray,
    cfg: TrainConfig,
    rng: np.random.Generator,
    *,
    task_index: int = 0,
) -> TaskRecord:
    """SGD on one task, in place. Returns the record; the model is mutated."""
    X, target = flatten_task(points, y)
    return train_xy(model, X, target, cfg, rng, task_index=task_index)


def train_xy(
    model: TwoModuleNet,
    X: np.ndarray,
    target: np.ndarray,
    cfg: TrainConfig,
    rng: np.random.Generator,
    *,
    task_index: int = 0,
) -> TaskRecord:
    """SGD on a (possibly multi-output) batch. ``worst_of_k`` halts when every
    output's ½ mean_b (f_k − y_k)² is at target. For one output that is
    identical to ``matched_loss`` (both use the pre-step prediction).
    """
    if getattr(model, "n_hidden_layers", 1) != 1:
        raise NotImplementedError(
            "L=3 training is blocked on the human μP derivation "
            "(docs/16 §Amendments A6). Forward pass and init-α are allowed.")
    if cfg.stopping not in ("matched_loss", "fixed_steps", "worst_of_k"):
        raise ValueError(f"unknown stopping {cfg.stopping!r}")
    if cfg.loss not in ("mse", "bce"):
        raise ValueError(f"unknown loss {cfg.loss!r}")
    if cfg.optimizer not in ("sgd", "adam"):
        raise ValueError(f"unknown optimizer {cfg.optimizer!r}")
    if cfg.loss == "bce" and cfg.stopping == "worst_of_k":
        raise ValueError("worst_of_k is MSE-only")
    n = X.shape[0]
    curve: list[float] = []
    loss = float("nan")
    step = 0
    per = np.array([np.nan])
    for step in range(1, cfg.steps_per_task + 1):
        if cfg.batch_size is None or cfg.batch_size >= n:
            xb, yb = X, target
        else:
            idx = rng.choice(n, size=cfg.batch_size, replace=False)
            xb, yb = X[idx], target[idx]
        per = per_output_mse(model.forward(xb), yb) if cfg.stopping == "worst_of_k" else None
        loss = model.step(
            xb, yb, loss=cfg.loss, optimizer=cfg.optimizer,
            adam_beta1=cfg.adam_beta1, adam_beta2=cfg.adam_beta2,
            adam_eps=cfg.adam_eps,
        )
        if step % cfg.record_every == 0 or step == 1:
            curve.append(loss)
        hit = (
            (cfg.stopping == "matched_loss" and loss <= cfg.target_loss)
            or (cfg.stopping == "worst_of_k" and float(np.max(per)) <= cfg.target_loss)
        )
        if hit:
            break
    converged = (
        (loss <= cfg.target_loss) if cfg.stopping == "matched_loss"
        else (float(np.max(per)) <= cfg.target_loss) if cfg.stopping == "worst_of_k"
        else True
    )
    return TaskRecord(
        task=task_index,
        final_loss=loss,
        steps_taken=step,
        train_accuracy=accuracy(model, X, target),
        loss_curve=curve,
        weight_change={m: model.weight_change(m) for m in MODULES},
        converged=converged,
    )


def run_stream(
    model: TwoModuleNet,
    task_points: list[np.ndarray],
    task_dichotomies: np.ndarray,
    cfg: TrainConfig,
    rng: np.random.Generator,
    *,
    probe_y: np.ndarray | None = None,
    probe_points: np.ndarray | None = None,
) -> tuple[list[TaskRecord], list[BoundaryRecord]]:
    """Train tasks in order, measuring at every boundary.

    `task_points[t]` is `(P, M, d)` for task `t` — a separate array per task
    because feature similarity `s_f` is imposed by redrawing the arrangement.

    Retained accuracy is evaluated on each past task's **own** arrangement and
    dichotomy, with the network as it stands. No readout is refit, so this is
    forgetting of the trained solution, not of the representation; the probe is
    what isolates the representation, and it gets a freshly trained readout in
    `src/analysis/` rather than here.

    Check `all(t.converged for t in tasks)` before using a run for H1/H2. Under a
    fixed step budget, later tasks are *harder* than the first — they start from a
    solution to a different dichotomy and must overwrite it — so a budget tuned on
    task 0 silently leaves later tasks unlearned, and their apparent "forgetting"
    is under-training.
    """
    ys = np.asarray(task_dichotomies)
    if len(task_points) != ys.shape[0]:
        raise ValueError(f"{len(task_points)} arrangements vs {ys.shape[0]} dichotomies")

    tasks: list[TaskRecord] = []
    boundaries: list[BoundaryRecord] = []
    for t, pts in enumerate(task_points):
        tasks.append(train_task(model, pts, ys[t], cfg, rng, task_index=t))
        boundaries.append(
            BoundaryRecord(
                after_task=t,
                current_accuracy=manifold_accuracy(model, pts, ys[t]),
                retained_accuracy={
                    j: manifold_accuracy(model, task_points[j], ys[j]) for j in range(t)
                },
                probe_accuracy=(
                    manifold_accuracy(
                        model,
                        pts if probe_points is None else probe_points,
                        probe_y,
                    )
                    if probe_y is not None
                    else float("nan")
                ),
                weight_change={m: model.weight_change(m) for m in MODULES},
            )
        )
    return tasks, boundaries
