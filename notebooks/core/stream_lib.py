"""1M stream driver for the docs/29 grid, built on `aba_lib` (Kati, 2026-11-09).

Conditions (docs/29 §8):
  1. Imports from `aba_lib` only: no function is copied or modified here.
  2. `stream_hash()` covers aba_lib.py, this file and the src stream generator
     (streams.py, plus generator.py and dichotomies.py, which make_stream calls).
  3. Parity: an A–B–A arm's exact A and B run as a T = 2 stream (`inputs_from_aba`)
     reproduces `aba_lib.run_arm` bit for bit (`stream_cluster.py parity`).
  4. Tests: tests/test_stream_lib.py.

Measurement definitions are docs/29 §2b. Indexing: task j read after task t is the key
"j,t" (t = "init" before any training); lag = t − j. Forward at boundary t is from task t
to task t+1, t = 0..T−2.
"""
from __future__ import annotations

import hashlib
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import aba_lib as L  # noqa: E402
from aba_lib import (  # noqa: E402
    SALT, NOISE_EPS, ABAConfig, Arr, OneModuleNet, train_to_criterion, rep_capacity, channels,
    richness_snapshot, sample_probe, s_r, centre_cosines, flatten, _realize_points,
    _logratio_steps, git_stamp,
)

REPO = L.REPO
sys.path.insert(0, str(REPO / "src"))
from manifolds.streams import StreamConfig, make_stream  # noqa: E402

HASHED_FILES = [
    L.LIB_PATH,
    Path(__file__).resolve(),
    REPO / "src" / "manifolds" / "streams.py",
    REPO / "src" / "manifolds" / "generator.py",
    REPO / "src" / "manifolds" / "dichotomies.py",
]
K_FLOOR = 8


def stream_hash(files=None) -> str:
    """sha256 over the bytes of every file the stream results depend on, 16 hex chars."""
    h = hashlib.sha256()
    for p in (HASHED_FILES if files is None else files):
        h.update(Path(p).name.encode("utf-8"))
        h.update(Path(p).read_bytes())
    return h.hexdigest()[:16]


@dataclass(frozen=True)
class StreamSpec:
    draw: int
    gamma: float
    hamming: int
    s_f: float

    @property
    def key(self) -> str:
        return f"draw={self.draw},gamma={self.gamma:g},h={self.hamming},s_f={self.s_f:g}"


@dataclass
class StreamInputs:
    """Everything a stream arm consumes; γ0 is applied in `run_stream`."""
    arrs: list            # T × Arr (training points + test realisation)
    ys: list              # T × (P,) labels
    neutral: list         # (T−1) × tuple of K neutral probes for forward t → t+1
    W0: np.ndarray
    ntk_idx: np.ndarray   # (P, k) indices into every task's points
    glue_key: list
    draw: int


def inputs_from_stream(cfg: ABAConfig, draw: int, hamming: int, s_f: float, T: int) -> StreamInputs:
    """The docs/29 stream: src `make_stream` (post-E16 default label generator)."""
    scfg = StreamConfig(stream_id=draw, condition="x", T=T, P=cfg.P, d=cfg.d, M=cfg.M, D=cfg.D,
                        R=cfg.R, feature_similarity=float(s_f),
                        readout_similarity=1.0 - 2.0 * hamming / cfg.P)
    st = make_stream(scfg, np.random.default_rng([SALT, draw, 0]))
    arrs = []
    for t, a in enumerate(st.arrangements):
        test = _realize_points(a.centers, a.axes, a.coords, a.R, NOISE_EPS,
                               np.random.default_rng([SALT, draw, 7, t]))
        arrs.append(Arr(a.centers, a.axes, a.coords, a.points, test, float(a.R)))
    ys = [np.asarray(y, np.int8) for y in st.dichotomies]
    neutral = []
    for t in range(T - 1):
        rng = np.random.default_rng([SALT, draw, 4, t])
        picks: list = []
        for _ in range(cfg.n_probe_neutral):
            yk = sample_probe(ys[t], [*ys, *picks], cfg.P // 2, rng)
            assert s_r(yk, ys[t]) == 0.0
            picks.append(yk)
        neutral.append(tuple(picks))
    W0 = np.random.default_rng([SALT, draw, 8]).standard_normal((cfg.N, cfg.d))
    k = min(cfg.ntk_per_manifold, cfg.M)
    r_ntk = np.random.default_rng([SALT, draw, 6])
    ntk_idx = np.stack([r_ntk.choice(cfg.M, size=k, replace=False) for _ in range(cfg.P)])
    return StreamInputs(arrs, ys, neutral, W0, ntk_idx, [SALT, draw, 9], draw)


def inputs_from_aba(cfg: ABAConfig, spec: L.ArmSpec) -> StreamInputs:
    """Parity only: A then B of an A–B–A arm, exactly as `aba_lib.run_arm` builds them."""
    base = L.seed_base(cfg, spec.seed)
    B = L.b_arrangement(cfg, base, spec.s_f, spec.noise)
    return StreamInputs([base.A, B], [base.y_A, base.y_Bs[spec.hamming]],
                        [tuple(base.y_star_neutral)], base.W0, base.ntk_idx,
                        [SALT, spec.seed, 9], spec.seed)


def _jt(j: int, t) -> str:
    return f"{j},{t}"


def run_stream(cfg: ABAConfig, inp: StreamInputs, gamma: float, spec: StreamSpec | None = None,
               stamp: dict | None = None, floor: bool = True) -> dict:
    """One stream arm: train tasks 0..T−1 in order, measure every (j, t). docs/29 §2b."""
    if cfg.stopping != "criterion":
        raise ValueError("stream_lib runs matched-loss stopping only (docs/29 K9)")
    t_start = time.perf_counter()
    timing = {"train_s": 0.0, "fresh_s": 0.0, "glue_s": 0.0, "ntk_s": 0.0, "floor_s": 0.0}
    T = len(inp.arrs)
    P = cfg.P
    idx = inp.ntk_idx
    data = [flatten(a.points, y) for a, y in zip(inp.arrs, inp.ys)]
    monitors = {f"T{j}_test": flatten(a.test, y) for j, (a, y) in enumerate(zip(inp.arrs, inp.ys))}
    probes = [np.concatenate([a.points[mu, idx[mu]] for mu in range(P)]) for a in inp.arrs]
    yprobe = [np.repeat(np.asarray(y, np.float64), idx.shape[1]) for y in inp.ys]

    net = OneModuleNet(inp.W0, gamma, cfg)
    init_loss = net.loss(*data[0])

    def geo(points, y):
        t0 = time.perf_counter()
        out = rep_capacity(net, points, y, cfg, inp.glue_key)
        timing["glue_s"] += time.perf_counter() - t0
        return out

    Ku0, KW0 = net.ntk_blocks(probes[0])
    K0 = Ku0 + KW0
    kprev = [None]

    def rich(t_next: int):
        t0 = time.perf_counter()
        out, kprev[0] = richness_snapshot(net, K0, Ku0, probes[0], yprobe[0],
                                          probes[t_next], yprobe[t_next], prev=kprev[0])
        timing["ntk_s"] += time.perf_counter() - t0
        return out

    g: dict = {}          # backward capacity, "j,t"
    fwd_g: dict = {}      # forward capacity: "t" -> {"next": {...}, "neutral": [...]} after task t
    fwd_init: dict = {}   # neutral probes at init, per t
    richness = {"init": rich(min(1, T - 1))}
    for j in range(T):
        g[_jt(j, "init")] = geo(inp.arrs[j].points, inp.ys[j])
    for t in range(T - 1):
        fwd_init[str(t)] = [geo(inp.arrs[t + 1].points, yk) for yk in inp.neutral[t]]

    tasks = []
    floor_logs = None
    for t in range(T):
        t0 = time.perf_counter()
        ph = train_to_criterion(net, *data[t], cfg, monitors)
        timing["train_s"] += time.perf_counter() - t0
        tasks.append(ph)
        richness[str(t)] = rich(min(t + 1, T - 1))
        for j in range(t + 1):
            g[_jt(j, t)] = geo(inp.arrs[j].points, inp.ys[j])
        if t < T - 1:
            fwd_g[str(t)] = {"next": geo(inp.arrs[t + 1].points, inp.ys[t + 1]),
                             "neutral": [geo(inp.arrs[t + 1].points, yk) for yk in inp.neutral[t]]}
        if t == 0 and floor:
            t0 = time.perf_counter()
            floor_logs = [math.log(rep_capacity(net, inp.arrs[0].points, inp.ys[0], cfg,
                                                [SALT, inp.draw, 2000 + k])["alpha"])
                          for k in range(K_FLOOR)]
            timing["floor_s"] += time.perf_counter() - t0

    fresh = []
    for t in range(T - 1):
        t0 = time.perf_counter()
        fnet = OneModuleNet(inp.W0, gamma, cfg)
        fresh.append(train_to_criterion(fnet, *data[t + 1], cfg, None))
        timing["fresh_s"] += time.perf_counter() - t0

    steps = [ph["steps"] for ph in tasks]
    backward = {}
    for j in range(T):
        backward[_jt(j, f"init>{j}")] = channels(g[_jt(j, "init")], g[_jt(j, j)])
        for t in range(j + 1, T):
            backward[_jt(j, f"{t - 1}>{t}")] = channels(g[_jt(j, t - 1)], g[_jt(j, t)])
            backward[_jt(j, f"{j}>{t}")] = channels(g[_jt(j, j)], g[_jt(j, t)])
    forward = {}
    for t in range(T - 1):
        tot = channels(g[_jt(t + 1, "init")], fwd_g[str(t)]["next"])
        neu = [math.log(a["alpha"] / b["alpha"]) for a, b in zip(fwd_g[str(t)]["neutral"], fwd_init[str(t)])]
        nm = float(np.mean(neu))
        forward[str(t)] = {"total": tot, "neutral": neu, "neutral_mean": nm,
                           "specific": tot["total"] - nm,
                           "FT_beh": _logratio_steps(fresh[t]["steps"], steps[t + 1])}
    S_r = [[s_r(inp.ys[i], inp.ys[j]) for j in range(T)] for i in range(T)]
    rho_in = [[float(np.mean(np.diag(centre_cosines(inp.arrs[i].centers, inp.arrs[j].centers))))
               for j in range(T)] for i in range(T)]
    out = {
        "key": spec.key if spec else None, "spec": asdict(spec) if spec else None,
        "config": {**cfg.key_fields(), "T": T}, "stream_hash": stream_hash(),
        "stamp": stamp or git_stamp(),
        "usable": all(ph["converged"] for ph in tasks) and all(f["converged"] for f in fresh),
        "init_loss": init_loss, "lr": net.lr, "output_scale": net.s,
        "relations": {"S_r": S_r, "rho_in": rho_in,
                      "neutral_s_r_to_y_t": [[s_r(yk, inp.ys[t]) for yk in inp.neutral[t]]
                                             for t in range(T - 1)]},
        "geometry": {"backward": g, "forward_after": fwd_g, "forward_neutral_init": fwd_init},
        "backward": backward,
        "forward": forward,
        "behaviour": {
            "steps": steps,
            "status": [ph["status"] for ph in tasks],
            "first_loss": [ph["first_loss"] for ph in tasks],
            "final_loss": [ph["final_loss"] for ph in tasks],
            "fresh_steps": [f["steps"] for f in fresh],
            "fresh_first_loss": [f["first_loss"] for f in fresh],
            "fresh_status": [f["status"] for f in fresh],
        },
        "richness": {**richness, "r_t": [ph["r_t"] for ph in tasks]},
        "floor": None if floor_logs is None else {
            "K": K_FLOOR, "logs": floor_logs, "sd": float(np.std(floor_logs, ddof=1))},
        "trajectory": [ph["records"] for ph in tasks],
    }
    timing["wall_s"] = time.perf_counter() - t_start
    out["timing"] = timing
    return out


def run_spec(cfg: ABAConfig, spec: StreamSpec, T: int = 16, stamp: dict | None = None) -> dict:
    inp = inputs_from_stream(cfg, spec.draw, spec.hamming, spec.s_f, T)
    return run_stream(cfg, inp, spec.gamma, spec, stamp)


def cache_path(cfg: ABAConfig) -> Path:
    return REPO / "results" / "stream" / f"{'QUICK' if cfg.quick else 'FULL'}_{stream_hash()}.jsonl"
