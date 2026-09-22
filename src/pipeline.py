"""Phase 1 pipeline: one arm = one (γ, `a`, stream condition, seed) sequential run.

Trains a stream task by task and measures Tier-2 geometry at scheduled boundaries,
producing everything Figures 2 and 3 need from a single pass: retained capacity and
its three factors per (past task × boundary × module), generic capacity alongside,
the accuracy matrix for `CF`/`CFr`, and the manipulation checks `00` §11 requires
logged unconditionally.

Three things here are decisions rather than plumbing, and all three came out of
Phase 0:

1. **The measurement RNG is fixed across boundaries.** Every geometry evaluation in
   a run draws the same `(y, t)` samples, so a change between two boundaries is the
   representation moving and not the estimator resampling. Without this, the
   attribution differences a run of ~40 evaluations, each carrying ~1–2% Monte-Carlo
   noise, and the noise does not cancel.
2. **Retained and generic share that stream too**, so the generic/retained crossing
   of Figure 3 is a paired comparison at every boundary.
3. **The schedule must contain each tracked task's own baseline boundary.** Task `j`'s
   forgetting is measured against the geometry at boundary `j`, immediately after it
   was learned. A schedule that measures task 0 only at the end has no denominator,
   so `schedule()` builds the baselines in rather than trusting the caller.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass

import numpy as np

from src.analysis.attribution import GeometryPoint, attribute
from src.glue.core import Ensemble, glue_measures, pairwise_measures
from src.manifolds.streams import (
    Stream, StreamConfig, apply_recurrence, make_stream, recurrence_control_rng,
)
from src.models import MODULES, ScalingConfig, TwoModuleNet, paired_init
from src.models.network import projection_rng
from src.models.alignment import aligned_init, center_subspace
from src.train.loop import (
    TrainConfig, flatten_group, flatten_task, manifold_accuracy, train_task, train_xy,
)

from src import provenance

_SOURCE = provenance.register(__file__)

EVAL_BUDGET = 40  # `01` §4 cost model: 153,600 evals over 3,840 runs


@dataclass(frozen=True)
class Phase1Spec:
    """One arm. Defaults are the `01` design point; only the first four vary in Phase 1."""

    gamma_0: float = 1.0
    a: float = 0.0
    condition: str = "S-HL"
    seed: int = 0
    stream_id: int = 0

    T: int = 16
    P: int = 16
    d: int = 150
    M: int = 150
    D: int = 4
    R: float = 1.0
    N: int = 300

    n_t: int = 200
    center_policy: str = "all"
    estimation_mode: str = "full_P"      # `pairwise` also computed if `also_pairwise`
    also_pairwise: bool = False
    measure_generic: bool = True

    lr0: float = 5.0
    target_loss: float = 0.05
    steps_per_task: int = 20_000
    record_every: int = 200

    tracked_stride: int = 4
    module_list: tuple[str, ...] = MODULES
    n_hidden_layers: int = 1

    @property
    def key(self) -> str:
        return (f"gamma={self.gamma_0:g},a={self.a:g},cond={self.condition},"
                f"stream={self.stream_id},seed={self.seed}")

    @property
    def train_config(self) -> TrainConfig:
        return TrainConfig(steps_per_task=self.steps_per_task, stopping="matched_loss",
                           target_loss=self.target_loss, record_every=self.record_every,
                           loss=getattr(self, "loss", "mse"),
                           optimizer=getattr(self, "optimizer", "sgd"))


def schedule(T: int, stride: int = 4) -> dict[int, list[int]]:
    """`boundary -> tracked past tasks to measure there`, baselines included.

    Tracked tasks are `0, stride, 2·stride, …`; each is measured at its own boundary
    (the baseline) and at every later measurement boundary, which are the tracked
    boundaries plus the last. Retention is followed over a *widening* interval rather
    than at one fixed lag, so the same run supports both "how much is lost by the
    end" and "how fast is it lost".
    """
    tracked = list(range(0, T, max(1, stride)))
    boundaries = sorted(set(tracked) | {T - 1})
    return {b: [j for j in tracked if j <= b] for b in boundaries}


def n_evals(sched: dict[int, list[int]], spec: Phase1Spec) -> int:
    per_boundary = [len(js) + (1 if spec.measure_generic else 0) for js in sched.values()]
    return sum(per_boundary) * len(spec.module_list)


def stream_rng(stream_id: int) -> np.random.Generator:
    """Arrangement + dichotomies keyed by `stream_id`, independent of init seed.

    `make_stream` never reads `cfg.stream_id`. The spec (`01` §1, `streams.py`) says
    streams are generated once per `stream_id` and shared across configs. Passing
    `paired_init(seed)["stream"]` made `stream_id` a stored no-op: 5 stream ids × 8
    seeds wrote 8 unique (arrangement, dichotomy, init) triples, each five times.
    This generator is the blocking factor the filenames already claimed.
    """
    return np.random.default_rng([20260817, int(stream_id)])


def build_stream(spec: Phase1Spec, rng: np.random.Generator | None = None) -> Stream:
    cfg = StreamConfig(
        stream_id=spec.stream_id, condition=spec.condition, T=spec.T,
        P=spec.P, d=spec.d, M=spec.M, D=spec.D, R=spec.R,
        manifold_kind=getattr(spec, "manifold_kind", "spherical"),
        feature_similarity=getattr(spec, "feature_similarity", None),
        readout_similarity=getattr(spec, "readout_similarity", None),
    )
    stream = make_stream(cfg, stream_rng(spec.stream_id) if rng is None else rng)
    mode = getattr(spec, "recurrence_mode", None)
    if not mode:
        return stream
    k = int(getattr(spec, "recurrence_k"))
    crng = recurrence_control_rng(spec.stream_id, k)
    return apply_recurrence(stream, k=k, mode=mode, rng=crng)


def build_model(spec: Phase1Spec, stream: Stream) -> TwoModuleNet:
    """Paired init: the shape matrix depends on the seed only, never on γ or `a`.

    This is what makes capacity-at-init exactly flat in γ (Phase 0 measured it as
    bitwise identical) and hence the γ and `a` axes orthogonal at initialization.
    """
    streams = paired_init(spec.seed)
    lr_scaling = getattr(spec, "lr_scaling", "quadratic")
    readout_rank = getattr(spec, "readout_rank", None)
    cfg = {m: ScalingConfig(N=spec.N, d=spec.d, gamma_0=spec.gamma_0, lr0=spec.lr0,
                            lr_scaling=lr_scaling)
           for m in spec.module_list}
    W_init = None
    if spec.a > 0:
        base = streams["shape"].standard_normal((spec.N, spec.d))
        U_C, _ = center_subspace(stream.arrangements[0].centers, D=spec.D)
        W_init = {m: aligned_init(base, U_C, spec.a) for m in spec.module_list}
    proj = None if readout_rank is None else projection_rng(spec.seed)
    return TwoModuleNet.init(cfg, streams["shape"], W_init=W_init,
                             readout_rank=readout_rank, projection_rng=proj,
                             n_outputs=int(getattr(spec, "n_outputs", 1)),
                             n_hidden_layers=int(getattr(spec, "n_hidden_layers", 1)))


@dataclass
class GeometryRecord:
    """One geometry evaluation: module × boundary × (past task | generic)."""

    module: str
    boundary: int
    task: int | None            # None for the generic ensemble
    ensemble: str
    alpha: float
    D_eff: float
    R_eff: float
    Psi_eff: float
    rho_c_glue: float
    rho_c_signed: float
    identity_residual: float
    pairwise: dict | None = None

    def point(self) -> GeometryPoint:
        return GeometryPoint.from_result(
            self, label=f"{self.module}@t{self.boundary}"
            + (f"/task{self.task}" if self.task is not None else "/generic"))


def _measure(model, points, y, spec, module, boundary, task, rng_seed) -> GeometryRecord:
    manifolds = model.manifold_representation(points, module)
    ens = Ensemble() if y is None else Ensemble(kind="retained", y_ref=np.asarray(y, float))
    res = glue_measures(
        manifolds, np.random.default_rng(rng_seed), n_t=spec.n_t,
        ensemble=ens, center_policy=spec.center_policy,
    )
    pw = None
    if spec.also_pairwise:
        pw = pairwise_measures(
            manifolds, np.random.default_rng(rng_seed), n_t=spec.n_t,
            ensemble=ens, center_policy=spec.center_policy, max_pairs=8,
        )
    return GeometryRecord(
        module=module, boundary=boundary, task=task, ensemble=ens.label,
        alpha=res.alpha, D_eff=res.D_eff, R_eff=res.R_eff, Psi_eff=res.Psi_eff,
        rho_c_glue=res.rho_c_glue, rho_c_signed=res.rho_c_signed,
        identity_residual=res.identity_residual, pairwise=pw,
    )


def run_arm(spec: Phase1Spec, *, verbose: bool = False) -> dict:
    """Train one stream sequentially, measuring geometry on the schedule.

    Returns a JSON-ready record. `usable` is false if any task failed to reach
    `target_loss`: retained capacity for a task the network never learned confounds
    forgetting with under-training, and such runs must be reported, not averaged in.
    """
    if int(getattr(spec, "n_hidden_layers", 1)) != 1:
        raise NotImplementedError(
            "L=3 training is blocked on the human μP derivation "
            "(docs/16 §Amendments A6). Forward pass and init-α are allowed.")
    t_start = time.time()
    streams = paired_init(spec.seed)
    rng_train = streams["data"]      # minibatch order only; unused at full batch
    if getattr(spec, "arrangement_source", "stream_id") == "legacy_seed":
        # Registered-grid population: arrangement keyed by init seed, not stream_id.
        stream = build_stream(spec, streams["stream"])
    else:
        stream = build_stream(spec)      # arrangement/dichotomies keyed by stream_id
    model = build_model(spec, stream)
    K = int(getattr(spec, "scope_K", 1))
    if K < 1 or spec.T % K != 0:
        raise ValueError(f"T={spec.T} is not divisible by scope_K={K}")
    if K == 1:
        sched = schedule(spec.T, spec.tracked_stride)
    else:
        tracked = list(range(0, spec.T, max(1, spec.tracked_stride)))
        group_ends = list(range(K - 1, spec.T, K))
        sched = {b: [j for j in tracked if j <= b] for b in group_ends}

    # One measurement seed per (module, task-or-generic), reused at every boundary.
    meas_base = int(np.random.default_rng([spec.seed, 20260811]).integers(2**32))

    def seed_for(module: str, task: int | None) -> int:
        return meas_base + 1013 * spec.module_list.index(module) + (
            0 if task is None else 1 + task)

    tasks, geometry = [], []
    acc = np.full((spec.T, spec.T), np.nan)
    checks = []
    n_groups = spec.T // K
    for g in range(n_groups):
        start, stop = g * K, (g + 1) * K
        last = stop - 1
        arrs = [stream.arrangements[i].points for i in range(start, stop)]
        ys = stream.dichotomies[start:stop]
        if K == 1:
            tasks.append(train_task(
                model, arrs[0], ys[0], spec.train_config, rng_train, task_index=last))
            pts, y_chk = arrs[0], ys[0]
        else:
            Xg, tg = flatten_group(arrs, ys)
            tasks.append(train_xy(
                model, Xg, tg, spec.train_config, rng_train, task_index=g))
            pts, y_chk = arrs[-1], ys[-1]
        for j in range(last + 1):
            acc[last, j] = manifold_accuracy(
                model, stream.arrangements[j].points, stream.dichotomies[j],
                output_index=j % K)
        X, _ = flatten_task(pts, y_chk)
        checks.append({
            "boundary": last,
            "weight_change": {m: model.weight_change(m) for m in spec.module_list},
            "output_variance_share": _output_variance_share(model, X, spec.module_list),
            "probe_decodability": probe_decodability(
                model, stream.arrangements[last].points, stream.probe, spec.module_list),
        })

        if last in sched:
            for m in spec.module_list:
                for j in sched[last]:
                    geometry.append(_measure(
                        model, stream.arrangements[j].points, stream.dichotomies[j],
                        spec, m, last, j, seed_for(m, j)))
                if spec.measure_generic:
                    geometry.append(_measure(
                        model, stream.arrangements[last].points, None,
                        spec, m, last, None, seed_for(m, None)))
            if verbose:
                print(f"  [{spec.key}] boundary {last}: {len(geometry)} evals so far",
                      flush=True)

    usable = all(tk.converged for tk in tasks)
    return {
        "spec": asdict(spec),
        "key": spec.key,
        "code": provenance.code_stamp(),
        "usable": usable,
        "n_evals": len(geometry),
        "eval_budget": EVAL_BUDGET,
        "schedule": {str(k): v for k, v in sched.items()},
        "tasks": [asdict(tk) for tk in tasks],
        "accuracy_matrix": acc.tolist(),
        "forgetting": forgetting_metrics(acc),
        "manipulation_checks": checks,
        "geometry": [asdict(g) for g in geometry],
        "attribution": attribute_run(geometry, spec),
        "stream": {
            "S_f": stream.S_f.tolist(), "S_r": stream.S_r.tolist(),
            "probe_s_r": stream.probe_s_r.tolist(),
        },
        "wall_seconds": time.time() - t_start,
    }


def probe_decodability(
    model, points, probe_y, modules, *,
    ridge: float = 1e-6, n_holdout: int = 4, n_splits: int = 4, seed: int = 0,
) -> dict:
    """Probe-dichotomy measures for the held-out dichotomy `y*` — Figure 3's overlay.

    A *refit* readout is required: the trained `u` was fitted to the current task's
    dichotomy, so reading `y*` off it measures the readout rather than the
    representation. Generic capacity is a claim about what the representation *could*
    support, so the probe gets its own readout.

    **Only `margin` and `margin_p05` may be plotted** — measured, `results/probe_check.json`
    (`scripts/run_probe_check.py`, γ ∈ {0.03, 1, 10} × 2 seeds at the design point):

    | measure | γ signal | noise | SNR | verdict |
    |---|---|---|---|---|
    | `accuracy` | 0 | 0 | — | **saturated at 1.0** |
    | `margin` | 0.0132 | 0.0033 | 4.0 | usable |
    | `margin_p05` | 0.0102 | 0.0032 | 3.2 | usable |
    | `heldout_manifold_accuracy` | 0.0494 | 0.1206 | 0.4 | **noise** |

    - **`accuracy`** is exactly 1.0 at initialization and after training, in both
      modules, at every γ. The load is `P/N = 16/300 = 0.053` against a critical
      capacity near 0.3, so every balanced dichotomy is separable with room to spare
      and separability at fixed sub-critical load cannot track capacity. Plotting it
      would have produced a flat line at 1.0, read as "generic capacity is preserved".
    - **`margin`** — `mean(y·f) / (‖w‖ · mean‖x‖)`, the graded quantity underneath the
      thresholded one, and the one capacity theory is actually about: capacity is the
      load at which the margin reaches zero. It separates γ cleanly and its change
      over training is sign-consistent across seeds within every arm — rich training
      *reduces* the probe margin, γ=0.03 leaves it untouched to four decimals.
      `margin_p05` is the 5th-percentile point margin, weighting the hard points that
      set the capacity limit rather than the bulk.
    - **`heldout_manifold_accuracy`** — fit on `P − n_holdout` manifolds, test on
      manifolds never seen — is retained as a **control, not an overlay**. It sits at
      chance (0.41–0.46) with a within-run split sd of 0.06–0.16, which is correct
      rather than broken: the probe is a *random* balanced dichotomy, so there is no
      shared structure for a readout fitted on 12 manifolds to extend to 4 unseen
      ones. Its apparent γ=0.03 dip is one seed's initialization (0.329 vs 0.495 at
      init, and γ=0.03 barely moves), not an effect of γ. Its value is as a check that
      the probe is genuinely unstructured — if it ever rose above chance, the probe
      would be leaking factor structure.
    """
    pts = np.asarray(points, dtype=np.float64)
    P, M = pts.shape[0], pts.shape[1]
    half = M // 2
    y = np.asarray(probe_y, dtype=np.float64)
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y > 0), np.flatnonzero(y < 0)
    k = max(1, n_holdout // 2)

    def fit(X, target):
        X = np.hstack([X, np.ones((X.shape[0], 1))])
        G = X.T @ X
        G += ridge * np.trace(G) / X.shape[1] * np.eye(X.shape[1])
        return np.linalg.solve(G, X.T @ target)

    def apply(w, X):
        return np.hstack([X, np.ones((X.shape[0], 1))]) @ w

    out: dict[str, dict] = {}
    for m in modules:
        H = np.stack([model.representation(pts[mu], m) for mu in range(P)])  # (P, M, N)
        Xtr = H[:, :half].reshape(P * half, -1)
        Xte = H[:, half:].reshape(P * (M - half), -1)
        ytr, yte = np.repeat(y, half), np.repeat(y, M - half)

        w = fit(Xtr, ytr)
        f = apply(w, Xte)
        scale = float(np.linalg.norm(w[:-1]) * np.mean(np.linalg.norm(Xte, axis=1)))
        signed = yte * f / scale

        gen = []
        for _ in range(n_splits):
            held = np.concatenate([rng.choice(pos, k, replace=False),
                                   rng.choice(neg, k, replace=False)])
            keep = np.setdiff1d(np.arange(P), held)
            w_s = fit(H[keep].reshape(len(keep) * M, -1), np.repeat(y[keep], M))
            f_s = apply(w_s, H[held].reshape(len(held) * M, -1))
            gen.append(float(np.mean(np.sign(f_s) == np.sign(np.repeat(y[held], M)))))

        out[m] = {
            "accuracy": float(np.mean(np.sign(f) == np.sign(yte))),
            "margin": float(np.mean(signed)),
            "margin_p05": float(np.percentile(signed, 5)),
            "heldout_manifold_accuracy": float(np.mean(gen)),
            "heldout_manifold_sd": float(np.std(gen, ddof=1)) if n_splits > 1 else 0.0,
        }
    return out


def _output_variance_share(model, X, modules) -> dict[str, float]:
    """`00` §11 gradient-starvation detector: which module drives the output."""
    contrib = {}
    for m in modules:
        u = model.u[m]
        h = model._projected(X, m) if getattr(model, "n_outputs", 1) == 1 else model.hidden(X, m)
        v = h @ u if u.ndim == 1 else h @ u.T
        contrib[m] = float(np.var(v))
    total = sum(contrib.values())
    return {m: (v / total if total > 0 else float("nan")) for m, v in contrib.items()}


def forgetting_metrics(acc: np.ndarray) -> dict:
    """`CF` and `CFr` (`docs/reference/cl-metrics.md`) from the accuracy matrix.

    `acc[i, j]` is accuracy on task `j` after training task `i`. Peak is
    `acc[j, j]`, not Graldi's `max_{t ∈ {j,...,T−1}} a_{t,j}` (Def. A.3).
    Equivalent when accuracy on `j` never recovers after learning. Recorded in
    `protocol-deviations.md` D.2. `CFr` normalizes by that peak, which is what
    makes it comparable across tasks of different attained accuracy — and the
    two metrics can rank arms differently, so both are reported.
    """
    T = acc.shape[0]
    per_task = {}
    for j in range(T - 1):
        peak, final = acc[j, j], acc[T - 1, j]
        if not np.isfinite(peak) or not np.isfinite(final):
            continue
        per_task[j] = {"peak": float(peak), "final": float(final),
                       "CF": float(peak - final),
                       "CFr": float((peak - final) / peak) if peak > 0 else float("nan")}
    if not per_task:
        return {"per_task": {}, "CF": float("nan"), "CFr": float("nan")}
    return {
        "per_task": per_task,
        "CF": float(np.mean([v["CF"] for v in per_task.values()])),
        "CFr": float(np.mean([v["CFr"] for v in per_task.values()])),
        "final_mean_accuracy": float(np.nanmean(acc[T - 1, :])),
    }


def attribute_run(geometry: list[GeometryRecord], spec: Phase1Spec) -> list[dict]:
    """Attribute each tracked task's retained-capacity change to the three factors.

    Baseline is the task's first measured boundary (its own learning boundary).
    On the registered K=1 path that boundary equals the task index. `lag` is
    boundaries elapsed from that baseline.
    """
    by_key = {(g.module, g.task, g.boundary): g for g in geometry if g.task is not None}
    earliest: dict[tuple[str, int], GeometryRecord] = {}
    for g in geometry:
        if g.task is None:
            continue
        key = (g.module, g.task)
        if key not in earliest or g.boundary < earliest[key].boundary:
            earliest[key] = g
    out = []
    for (module, task, boundary), rec in sorted(by_key.items()):
        base = earliest.get((module, task))
        if base is None or rec.boundary == base.boundary:
            continue
        a = attribute(base.point(), rec.point(), strict=False, gamma=spec.gamma_0)
        out.append({"module": module, "task": task, "boundary": rec.boundary,
                    "lag": rec.boundary - base.boundary, **a.to_dict()})
    return out
