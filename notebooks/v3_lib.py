"""Live computation for `project-overview-v3.ipynb` (reduced-scale robustness config).

Everything the v3 notebook shows is produced by these functions at run time; nothing
is read from `results/`. The training and geometry primitives are the registered ones
(`src/train/loop.py`, `src/glue/core.py`); what differs from `src/pipeline.run_arm` is
the measurement schedule: task 0 is measured at *every* boundary (plus at init), so
trajectories cover lags 0..T−1 instead of the stride-4 subset.

Reduced scale is a robustness configuration, not the paper configuration. `M` and
`n_t` are smaller than the design point, so `lr0` and `target_loss` are re-pinned by
`calibrate` (Phase 0 decade ladder, `docs/01`, `results/adam_lr0_repin.md`) rather
than inherited.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from pathlib import Path

# BLAS reads these at library load. Setting them after `import numpy` does not
# reconfigure an already-initialized OpenBLAS, and a forked worker inherits that
# state. Windows spawn re-imports and would see the assignment in parallel_map;
# this one covers the parent and any fork.
for _blas_threads in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_blas_threads] = "1"

import numpy as np
from scipy import stats

from src.analysis.grid import REGISTERED_GAMMAS
from src.glue.core import Ensemble, glue_measures
from src.manifolds.generator import resample_test_points
from src.manifolds.streams import Stream, StreamConfig, make_stream
from src.models import MODULES, ScalingConfig, TwoModuleNet, paired_init
from src.train.loop import TrainConfig, flatten_task, train_task
from src import provenance

def code_digest() -> str:
    """Hash of every `src/**/*.py` and this file. Part of every cache key."""
    root = Path(__file__).resolve().parents[1]
    h = hashlib.sha256()
    for p in sorted((root / "src").rglob("*.py")) + [Path(__file__).resolve()]:
        h.update(p.relative_to(root).as_posix().encode())
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


CODE_DIGEST = code_digest()

GAMMAS = tuple(float(g) for g in REGISTERED_GAMMAS)
S_F_LEVELS = (1.0, 0.9, 0.1)
HAMMING_LEVELS = (0, 2, 4, 6, 8)

STREAM_SALT = 20261028
CALIBRATION_STREAM_ID = 30000          # used only by `calibrate`, never evaluated
EVAL_STREAM_BASE = 30100               # replicate r uses stream_id = base + r, seed = r


@dataclass(frozen=True)
class V3Config:
    T: int = 16
    P: int = 16
    d: int = 150
    M: int = 50
    D: int = 4
    R: float = 1.0
    N: int = 300
    n_t: int = 50
    lr0: float = 5.0
    target_loss: float = 0.05
    steps_per_task: int = 20_000
    record_every: int = 200
    geometry_module: str = "A"

    def train_config(self) -> TrainConfig:
        return TrainConfig(steps_per_task=self.steps_per_task, stopping="matched_loss",
                           target_loss=self.target_loss, record_every=self.record_every)


def s_r_for_hamming(h: int, P: int) -> float:
    return 1.0 - 2.0 * h / P


def cell_label(s_f: float, h: int) -> str:
    return f"s_f={s_f:g}, Hamming={h}"


def build_stream(cfg: V3Config, s_f: float, h: int, stream_id: int) -> Stream:
    """Arrangements and dichotomies from independent generators keyed by `stream_id`.

    Task 0's arrangement and dichotomy are therefore identical across all
    `(s_f, Hamming)` cells of one replicate.
    """
    scfg = StreamConfig(stream_id=stream_id, condition=cell_label(s_f, h), T=cfg.T,
                        P=cfg.P, d=cfg.d, M=cfg.M, D=cfg.D, R=cfg.R,
                        feature_similarity=float(s_f),
                        readout_similarity=s_r_for_hamming(h, cfg.P))
    stream = make_stream(scfg, np.random.default_rng([STREAM_SALT, stream_id, 0]),
                         rng_dichotomies=np.random.default_rng([STREAM_SALT, stream_id, 1]))
    for t in range(1, cfg.T):
        got = int(np.sum(stream.dichotomies[t] != stream.dichotomies[t - 1]))
        if got != h:
            raise AssertionError(f"task {t}: Hamming {got} != {h}")
    return stream


def build_model(cfg: V3Config, gamma: float, seed: int, lr0: float | None = None) -> TwoModuleNet:
    """Paired init (I6): `W(0)` depends on `seed` only, never on γ or the cell."""
    sc = {m: ScalingConfig(N=cfg.N, d=cfg.d, gamma_0=gamma,
                           lr0=cfg.lr0 if lr0 is None else lr0) for m in MODULES}
    return TwoModuleNet.init(sc, paired_init(seed)["shape"])


def readout_floor(cfg: V3Config, stream: Stream, seed: int) -> float:
    """Exact least-squares readout loss at `W(0)` for task 0 — the attainable floor.

    γ only rescales the readout features, so the floor is the same at every γ.
    """
    model = build_model(cfg, 1.0, seed)
    X, y = flatten_task(stream.arrangements[0].points, stream.dichotomies[0])
    F = np.hstack([model.cfg[m].output_scale * model.hidden(X, m) for m in MODULES])
    u, *_ = np.linalg.lstsq(F, y, rcond=1e-12)
    return float(0.5 * np.mean((F @ u - y) ** 2))


def _geometry(model, points, y, cfg: V3Config, seed: int) -> dict:
    manifolds = model.manifold_representation(points, cfg.geometry_module)
    ens = Ensemble() if y is None else Ensemble(kind="retained", y_ref=np.asarray(y, float))
    r = glue_measures(manifolds, np.random.default_rng(seed), n_t=cfg.n_t,
                      ensemble=ens, center_policy="all")
    return {"alpha": r.alpha, "Psi_eff": r.Psi_eff, "R_eff": r.R_eff, "D_eff": r.D_eff,
            "rho_c_glue": r.rho_c_glue, "rho_c_signed": r.rho_c_signed,
            "identity_residual": r.identity_residual}


def _behaviour(model, X0: np.ndarray, y0: np.ndarray) -> dict:
    f = model.forward(X0)
    m = y0 * f
    return {"sign_accuracy": float(np.mean(np.sign(f) == np.sign(y0))),
            "margin_mean": float(np.mean(m)),
            "margin_p05": float(np.percentile(m, 5))}


@dataclass(frozen=True)
class ArmSpec:
    gamma: float
    s_f: float
    hamming: int
    replicate: int

    @property
    def stream_id(self) -> int:
        return EVAL_STREAM_BASE + self.replicate

    @property
    def seed(self) -> int:
        return self.replicate


def run_arm(cfg: V3Config, spec: ArmSpec, *, geometry: bool = True) -> dict:
    """Train all T tasks; measure task 0 at init and after every boundary.

    Behaviour is measured on fresh points re-sampled once from task 0's manifolds
    (same points at every boundary). Geometry uses the training points of task 0,
    retained (fixed `y_0`) and generic, each with one measurement seed reused at every
    boundary so changes between boundaries are the representation, not the estimator.
    """
    t0 = time.time()
    stream = build_stream(cfg, spec.s_f, spec.hamming, spec.stream_id)
    model = build_model(cfg, spec.gamma, spec.seed)
    tcfg = cfg.train_config()
    rng_train = paired_init(spec.seed)["data"]
    arr0, y0 = stream.arrangements[0], stream.dichotomies[0]
    X0, t_0 = flatten_task(resample_test_points(arr0, np.random.default_rng(
        [STREAM_SALT, spec.stream_id, 2])), y0)
    meas = int(np.random.default_rng([spec.seed, 20260811]).integers(2**32))

    rows = []

    def record(boundary: int, task_rec=None) -> None:
        row = {"boundary": boundary, **_behaviour(model, X0, t_0),
               "weight_change": {m: model.weight_change(m) for m in MODULES}}
        if task_rec is not None:
            row.update(steps=task_rec.steps_taken, final_loss=task_rec.final_loss,
                       converged=bool(task_rec.converged))
        if geometry:
            row["retained"] = _geometry(model, arr0.points, y0, cfg, meas + 1)
            row["generic"] = _geometry(model, arr0.points, None, cfg, meas)
        rows.append(row)

    record(-1)
    for t in range(cfg.T):
        rec = train_task(model, stream.arrangements[t].points, stream.dichotomies[t],
                         tcfg, rng_train, task_index=t)
        record(t, rec)
    return {"spec": asdict(spec), "config": asdict(cfg), "rows": rows,
            "usable": all(r["converged"] for r in rows[1:]),
            "S_f": stream.S_f.tolist(), "S_r": stream.S_r.tolist(),
            "code": provenance.code_stamp(), "wall_seconds": time.time() - t0}


def _calib_job(args) -> dict:
    cfg, lr0, gamma, n_tasks, seed = args
    stream = build_stream(cfg, 0.1, 8, CALIBRATION_STREAM_ID)
    model = build_model(cfg, gamma, seed, lr0=lr0)
    tcfg = cfg.train_config()
    rng = paired_init(seed)["data"]
    out = []
    t0 = time.time()
    with np.errstate(over="ignore", invalid="ignore"):
        for t in range(n_tasks):
            rec = train_task(model, stream.arrangements[t].points, stream.dichotomies[t],
                             tcfg, rng, task_index=t)
            out.append({"task": t, "steps": rec.steps_taken, "final_loss": rec.final_loss,
                        "converged": bool(rec.converged and np.isfinite(rec.final_loss))})
            if not out[-1]["converged"]:
                break
    return {"lr0": lr0, "gamma": gamma, "tasks": out, "seconds": time.time() - t0,
            "all_converged": len(out) == n_tasks and all(o["converged"] for o in out)}


def calibrate(cfg: V3Config, *, inherited: float = 5.0, n_tasks: int = 4,
              cap_margin: float = 4.0, seed: int = 0, workers: int | None = None,
              checkpoint: str | None = None) -> dict:
    """Phase 0 check of the inherited `lr0` on a reserved calibration stream.

    Rule (chosen 2026-10-29, before the lower ladder rungs were seen; `results/LOG.md`):
    keep `inherited` if, on the first `n_tasks` tasks of stream `CALIBRATION_STREAM_ID`
    (`s_f=0.1`, Hamming 8), every γ reaches `target_loss` on every task using at most
    `steps_per_task / cap_margin` steps, and `10 × inherited` also reaches at every γ
    (stability a decade up). Otherwise `pin` is None and the caller must stop: walking
    the ladder is a human decision, not an automatic fallback.

    `checkpoint` is a JSON-lines file of finished jobs; jobs already in it (same code
    and config) are not re-run, so an interrupted calibration resumes.
    """
    lr0s = (inherited, 10.0 * inherited)
    jobs = [(cfg, lr0, g, n_tasks, seed) for g in GAMMAS for lr0 in lr0s]
    results = parallel_map_checkpointed(
        _calib_job, jobs, checkpoint, key=lambda j: [CODE_DIGEST, asdict(j[0]), *j[1:]],
        workers=workers)
    max_steps = cfg.steps_per_task / cap_margin
    at = [r for r in results if r["lr0"] == inherited]
    up = [r for r in results if r["lr0"] == 10.0 * inherited]
    reaches = all(r["all_converged"] for r in at)
    margin_ok = all(t["steps"] <= max_steps for r in at for t in r["tasks"])
    stable_up = all(r["all_converged"] for r in up)
    pin = inherited if (reaches and margin_ok and stable_up) else None
    return {"inherited": inherited, "reaches": reaches, "margin_ok": margin_ok,
            "max_steps_allowed": max_steps, "stable_decade_up": stable_up,
            "pin": pin, "results": results}


def _arm_job(args) -> dict:
    cfg, spec, geometry = args
    return run_arm(cfg, spec, geometry=geometry)


def parallel_map(fn, jobs, workers: int | None = None) -> list:
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[k] = "1"
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(fn, jobs))


def parallel_map_checkpointed(fn, jobs, checkpoint: str | None, *, key,
                              workers: int | None = None) -> list:
    """`parallel_map` that appends each finished result to a JSON-lines file.

    Only results whose stored key equals the job's key are reused; a changed
    config never picks up a stale result.
    """
    if checkpoint is None:
        return parallel_map(fn, jobs, workers)
    done = {}
    if os.path.exists(checkpoint):
        with open(checkpoint, encoding="utf-8") as fh:
            for line in fh:
                rec = json.loads(line)
                done[json.dumps(rec["key"])] = rec["result"]
    keys = [json.dumps(key(j)) for j in jobs]
    todo = [(k, j) for k, j in zip(keys, jobs) if k not in done]
    if todo:
        workers = workers or max(1, (os.cpu_count() or 2) - 1)
        for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            os.environ[k] = "1"
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(fn, j): k for k, j in todo}
            for fut in as_completed(futs):
                res = fut.result()
                done[futs[fut]] = res
                with open(checkpoint, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps({"key": json.loads(futs[fut]), "result": res},
                                        default=float) + "\n")
    return [done[k] for k in keys]


def run_grid(cfg: V3Config, specs: list[ArmSpec], *, workers: int | None = None,
             checkpoint: str | None = None) -> list[dict]:
    jobs = [(cfg, s, True) for s in specs]
    return parallel_map_checkpointed(
        _arm_job, jobs, checkpoint, key=lambda j: [CODE_DIGEST, asdict(j[0]), asdict(j[1])],
        workers=workers)


def all_cells():
    return [(s_f, h) for s_f in S_F_LEVELS for h in HAMMING_LEVELS]


# ---------------------------------------------------------------- inspection helpers

def centre_cosine_matrix(stream: Stream) -> np.ndarray:
    """Measured `T×T` mean cosine between corresponding centres of tasks i and j.

    `stream.S_f` stores the *nominal* `s_f` on the first off-diagonal; this is the
    realized value everywhere.
    """
    C = np.stack([a.centers for a in stream.arrangements])          # (T, P, d)
    U = C / np.linalg.norm(C, axis=2, keepdims=True)
    return np.einsum("ipd,jpd->ij", U, U) / C.shape[1]


def pca_basis(X: np.ndarray, k: int = 3) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean, top-`k` principal axes `(k, d)`, and their explained-variance ratios."""
    X = np.asarray(X, dtype=np.float64)
    mu = X.mean(axis=0)
    _, s, Vt = np.linalg.svd(X - mu, full_matrices=False)
    var = s ** 2
    return mu, Vt[:k], var[:k] / var.sum()


def manifold_geometry(manifolds, y=None, *, n_t: int, seed: int) -> dict:
    """GLUE on `(N, M)` manifolds; retained if `y` is given, generic otherwise."""
    ens = Ensemble() if y is None else Ensemble(kind="retained", y_ref=np.asarray(y, float))
    r = glue_measures(list(manifolds), np.random.default_rng(seed), n_t=n_t,
                      ensemble=ens, center_policy="all")
    return {"alpha": r.alpha, "Psi_eff": r.Psi_eff, "R_eff": r.R_eff, "D_eff": r.D_eff,
            "rho_c_glue": r.rho_c_glue, "rho_c_signed": r.rho_c_signed,
            "identity_residual": r.identity_residual}


def input_manifolds(points: np.ndarray) -> list[np.ndarray]:
    """`(P, M, d)` points → GLUE layout, with the input itself as the representation."""
    return [np.ascontiguousarray(points[i].T) for i in range(points.shape[0])]


def train_first_task(cfg: V3Config, gamma: float, stream: Stream, seed: int) -> TwoModuleNet:
    model = build_model(cfg, gamma, seed)
    train_task(model, stream.arrangements[0].points, stream.dichotomies[0],
               cfg.train_config(), paired_init(seed)["data"], task_index=0)
    return model


def three_terms(g: dict) -> dict:
    """Log factors whose signed sum is `log α` (exact identity, ICLR 2026 §B.3)."""
    return {"utility": np.log(g["Psi_eff"]),
            "radius": np.log1p(g["R_eff"] ** -2),
            "dimension": -np.log(g["D_eff"])}


def _floor_job(args):
    cfg, manifolds, y, seed = args
    g = manifold_geometry(manifolds, y, n_t=cfg.n_t, seed=seed)
    return {"log_alpha": float(np.log(g["alpha"])), **{k: float(v) for k, v in three_terms(g).items()},
            "rho_c_glue": g["rho_c_glue"], "rho_c_signed": g["rho_c_signed"]}


def noise_floor(cfg: V3Config, manifolds, y, *, K: int = 12, base_seed: int = 777,
                workers: int | None = None) -> dict:
    """Estimator floor: sd over `K` measurement seeds on one fixed representation."""
    rows = parallel_map(_floor_job, [(cfg, manifolds, y, base_seed + k) for k in range(K)],
                        workers)
    return {k: float(np.std([r[k] for r in rows], ddof=1)) for k in rows[0]}


# ---------------------------------------------------------------- aggregation

def ci(values: np.ndarray, axis: int = 0) -> tuple[np.ndarray, np.ndarray, int]:
    """Mean and 95% half-width `t_{0.975, n−1} · SEM` over replicates.

    The t quantile, not 1.96: at the few replicates a live notebook affords, 1.96
    understates the interval (n=5: 2.78 vs 1.96).
    """
    v = np.asarray(values, dtype=np.float64)
    n = v.shape[axis]
    if n == 0:
        return np.full(v.shape[1 - axis] if v.ndim > 1 else (), np.nan), np.nan, 0
    mean = np.mean(v, axis=axis)
    if n < 2:
        return mean, np.full_like(mean, np.nan), n
    half = stats.t.ppf(0.975, n - 1) * np.std(v, axis=axis, ddof=1) / np.sqrt(n)
    return mean, half, n


def trajectory(arms: list[dict], s_f: float, h: int, gamma: float, getter) -> np.ndarray:
    """`(n_usable_replicates, T+1)` array; column 0 is init, column t+1 is boundary t."""
    out = []
    for a in arms:
        s = a["spec"]
        if (s["s_f"], s["hamming"], s["gamma"]) != (s_f, h, gamma) or not a["usable"]:
            continue
        out.append([getter(r) for r in a["rows"]])
    return np.array(out, dtype=np.float64)


def unusable(arms: list[dict]) -> list[dict]:
    return [a["spec"] for a in arms if not a["usable"]]


def with_pin(cfg: V3Config, lr0: float) -> V3Config:
    return replace(cfg, lr0=float(lr0))
