"""Smoke: N=32 (capacity-limited / plottable) and N=100 (lowest subcritical candidate).

Pre-commit: `results/width_smoke_precommit.json` — written before any arm in this
directory. Decision rules live there; this script only produces the records.

Writes to `results/width_smoke/`, never `results/phase1/` (`Phase1Spec.key` omits
`N`; an off-design arm in the registered directory would be silently adopted).

    python scripts/run_width_smoke.py --init-only     # seconds: readout floor vs N
    python scripts/run_width_smoke.py --time-one      # one N=32 γ=10 arm, then stop
    python scripts/run_width_smoke.py                 # 4 full arms + cloud dumps

`--init-only` does not train. Full arms are 2 widths × {γ=0.03, 10} × S-HL ×
stream 0 × seed 0.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from _par import pin_threads  # noqa: E402

pin_threads()

from _par import pmap  # noqa: E402
from src import pipeline as pl  # noqa: E402
from src import provenance  # noqa: E402
from src.models import MODULES, paired_init  # noqa: E402
from src.train.loop import flatten_task, train_task  # noqa: E402

PRE = ROOT / "results" / "width_smoke_precommit.json"
OUT = ROOT / "results" / "width_smoke"

N_TRAIN = (32, 100)
GAMMAS = (0.03, 10.0)
N_CURVE = (32, 50, 64, 80, 100, 150, 300)
N_DICHOTOMIES = 32
CLOUD_BOUNDARIES = (0, 15)


def _require_precommit() -> dict:
    if not PRE.is_file():
        raise SystemExit(
            f"missing {PRE.relative_to(ROOT)} — write the pre-commit before any arm."
        )
    return json.loads(PRE.read_text())


def _spec(N: int, gamma: float, *, stream_id: int = 0, seed: int = 0) -> pl.Phase1Spec:
    return pl.Phase1Spec(
        gamma_0=gamma, a=0.0, condition="S-HL", stream_id=stream_id, seed=seed, N=N,
    )


def _path(spec: pl.Phase1Spec, out: Path) -> Path:
    return out / f"N{spec.N}__{spec.key.replace(',', '__').replace('=', '-')}.json"


def balanced_dichotomy(P: int, rng: np.random.Generator) -> np.ndarray:
    y = np.ones(P, dtype=np.float64)
    y[rng.choice(P, P // 2, replace=False)] = -1.0
    return y


def lstsq_readout(H: np.ndarray, y: np.ndarray) -> dict:
    """Exact linear least squares of y (±1, length B) on features H (B, F)."""
    u, *_ = np.linalg.lstsq(H, y, rcond=None)
    pred = H @ u
    err = pred - y
    acc = float(np.mean(np.sign(pred) == np.sign(y)))
    return {
        "mse": float(0.5 * np.mean(err ** 2)),
        "sign_accuracy": acc,
        "rank": int(np.linalg.matrix_rank(H)),
        "n_features": int(H.shape[1]),
    }


def features(model, X: np.ndarray, which: str) -> np.ndarray:
    if which == "A":
        return model.hidden(X, "A")
    if which == "AB":
        return np.concatenate([model.hidden(X, m) for m in MODULES], axis=1)
    raise ValueError(which)


def pack_clouds(model, points: np.ndarray) -> np.ndarray:
    """Module A hidden states, shape (P, M, N)."""
    pts = np.asarray(points, dtype=np.float64)
    return np.stack([model.hidden(pts[mu], "A") for mu in range(pts.shape[0])])


def init_readout_at_N(N: int, *, n_random: int = N_DICHOTOMIES) -> dict:
    spec = _spec(N, 1.0)
    stream = pl.build_stream(spec)
    model = pl.build_model(spec, stream)
    pts, y0 = stream.arrangements[0].points, stream.dichotomies[0]
    X, Y0 = flatten_task(pts, y0)
    rng = np.random.default_rng([20260819, N, 32])
    dichotomies = [y0, *[balanced_dichotomy(spec.P, rng) for _ in range(n_random)]]
    labels = ["y0", *[f"rand_{i}" for i in range(n_random)]]
    rows = []
    for lab, y in zip(labels, dichotomies):
        Y = np.repeat(y, spec.M)
        rows.append({
            "label": lab,
            "A": lstsq_readout(features(model, X, "A"), Y),
            "AB": lstsq_readout(features(model, X, "AB"), Y),
        })
    def summarise(which: str) -> dict:
        mses = np.array([r[which]["mse"] for r in rows])
        accs = np.array([r[which]["sign_accuracy"] for r in rows])
        return {
            "median_mse": float(np.median(mses)),
            "p90_mse": float(np.quantile(mses, 0.9)),
            "frac_mse_le_target": float(np.mean(mses <= spec.target_loss)),
            "frac_perfect_sign": float(np.mean(accs >= 1.0 - 1e-12)),
            "y0_mse": float(rows[0][which]["mse"]),
            "y0_sign_accuracy": float(rows[0][which]["sign_accuracy"]),
            "rank": rows[0][which]["rank"],
        }
    return {
        "N": N,
        "P_over_N": spec.P / N,
        "packing_P_Dp1_over_N": spec.P * (spec.D + 1) / N,
        "n_dichotomies": len(dichotomies),
        "A": summarise("A"),
        "AB": summarise("AB"),
        "per_dichotomy": rows,
        "eta": float(model.cfg["A"].lr),
        "gamma_eff": float(model.cfg["A"].gamma_eff),
    }


def run_init_curve(ns: tuple[int, ...] = N_CURVE) -> dict:
    t0 = time.time()
    by_N = [init_readout_at_N(n) for n in ns]
    return {
        "generated_by": "scripts/run_width_smoke.py --init-only",
        "precommit": str(PRE.relative_to(ROOT)),
        "wall_seconds": time.time() - t0,
        "target_loss": 0.05,
        "by_N": by_N,
    }


def _run_arm_with_clouds(spec: pl.Phase1Spec) -> tuple[dict, dict]:
    """Same measurement as `pl.run_arm`, plus module-A clouds at init / t=0 / t=15."""
    t_start = time.time()
    streams = paired_init(spec.seed)
    rng_train = streams["data"]
    stream = pl.build_stream(spec)
    model = pl.build_model(spec, stream)
    sched = pl.schedule(spec.T, spec.tracked_stride)
    meas_base = int(np.random.default_rng([spec.seed, 20260811]).integers(2**32))

    def seed_for(module: str, task: int | None) -> int:
        return meas_base + 1013 * spec.module_list.index(module) + (
            0 if task is None else 1 + task)

    clouds = {
        "N": spec.N,
        "gamma_0": spec.gamma_0,
        "init": pack_clouds(model, stream.arrangements[0].points),
        "y0": np.asarray(stream.dichotomies[0], dtype=np.float64),
    }
    tasks, geometry, checks = [], [], []
    acc = np.full((spec.T, spec.T), np.nan)
    for t in range(spec.T):
        pts, y = stream.arrangements[t].points, stream.dichotomies[t]
        tasks.append(train_task(model, pts, y, spec.train_config, rng_train, task_index=t))
        for j in range(t + 1):
            acc[t, j] = pl.manifold_accuracy(
                model, stream.arrangements[j].points, stream.dichotomies[j])
        X, _ = flatten_task(pts, y)
        checks.append({
            "boundary": t,
            "weight_change": {m: model.weight_change(m) for m in spec.module_list},
            "output_variance_share": pl._output_variance_share(model, X, spec.module_list),
            "probe_decodability": pl.probe_decodability(
                model, stream.arrangements[t].points, stream.probe, spec.module_list),
        })
        if t in CLOUD_BOUNDARIES:
            clouds[f"t{t}"] = pack_clouds(model, stream.arrangements[0].points)
        if t in sched:
            for m in spec.module_list:
                for j in sched[t]:
                    geometry.append(pl._measure(
                        model, stream.arrangements[j].points, stream.dichotomies[j],
                        spec, m, t, j, seed_for(m, j)))
                if spec.measure_generic:
                    geometry.append(pl._measure(
                        model, stream.arrangements[t].points, None,
                        spec, m, t, None, seed_for(m, None)))
    usable = all(tk.converged for tk in tasks)
    rec = {
        "spec": asdict(spec),
        "key": spec.key,
        "code": provenance.code_stamp(),
        "usable": usable,
        "n_evals": len(geometry),
        "eval_budget": pl.EVAL_BUDGET,
        "schedule": {str(k): v for k, v in sched.items()},
        "tasks": [asdict(tk) for tk in tasks],
        "accuracy_matrix": acc.tolist(),
        "forgetting": pl.forgetting_metrics(acc),
        "manipulation_checks": checks,
        "geometry": [asdict(g) for g in geometry],
        "attribution": pl.attribute_run(geometry, spec),
        "stream": {
            "S_f": stream.S_f.tolist(), "S_r": stream.S_r.tolist(),
            "probe_s_r": stream.probe_s_r.tolist(),
        },
        "wall_seconds": time.time() - t_start,
        "smoke": True,
    }
    return rec, clouds


def _job(payload: tuple[dict, str]) -> dict:
    provenance.assert_current()
    spec = pl.Phase1Spec(**payload[0])
    out = Path(payload[1])
    path = _path(spec, out)
    if path.exists():
        stored = json.loads(path.read_text())
        if stored.get("spec") == json.loads(json.dumps(asdict(spec))):
            return {"key": spec.key, "N": spec.N, "skipped": True,
                    "usable": stored.get("usable")}
    rec, clouds = _run_arm_with_clouds(spec)
    path.write_text(json.dumps(rec))
    npz = path.with_suffix(".clouds.npz")
    save = {k: v for k, v in clouds.items() if k not in ("N", "gamma_0")}
    np.savez_compressed(npz, **{k: np.asarray(v) for k, v in save.items()},
                        N=clouds["N"], gamma_0=clouds["gamma_0"])
    return {"key": spec.key, "N": spec.N, "gamma_0": spec.gamma_0,
            "skipped": False, "usable": rec["usable"],
            "wall_seconds": rec["wall_seconds"],
            "n_converged": sum(1 for t in rec["tasks"] if t["converged"]),
            "n_tasks": len(rec["tasks"])}


def arms() -> list[pl.Phase1Spec]:
    return [_spec(n, g) for n in N_TRAIN for g in GAMMAS]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--init-only", action="store_true")
    p.add_argument("--time-one", action="store_true")
    p.add_argument("--out", type=str, default="results/width_smoke")
    p.add_argument("--cap", type=int, default=4)
    args = p.parse_args()
    pre = _require_precommit()
    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    if out.resolve() == (ROOT / "results" / "phase1").resolve():
        raise SystemExit("refusing to write smoke arms into results/phase1/")
    out.mkdir(parents=True, exist_ok=True)

    init_path = out / "init_readout.json"
    if args.init_only or not init_path.exists():
        print("init-readout curve…", flush=True)
        curve = run_init_curve()
        init_path.write_text(json.dumps(curve, indent=2))
        for row in curve["by_N"]:
            a = row["A"]
            print(
                f"  N={row['N']:<4d} P/N={row['P_over_N']:.3f}  "
                f"A median MSE={a['median_mse']:.4f}  "
                f"frac MSE≤0.05={a['frac_mse_le_target']:.2f}  "
                f"frac sign=1={a['frac_perfect_sign']:.2f}  "
                f"y0 MSE={a['y0_mse']:.4f}",
                flush=True,
            )
        print(f"  wrote {init_path.relative_to(ROOT)} in {curve['wall_seconds']:.1f}s",
              flush=True)
        if args.init_only:
            return

    if args.time_one:
        spec = _spec(32, 10.0)
        t0 = time.time()
        rec = _job((asdict(spec), str(out)))
        dt = time.time() - t0
        print(f"  N=32 γ=10 one arm: {dt:.1f}s usable={rec.get('usable')}", flush=True)
        return

    specs = arms()
    print(
        f"width smoke: {len(specs)} arms N={N_TRAIN} γ={GAMMAS} "
        f"precommit={pre['written']} -> {out.relative_to(ROOT)}",
        flush=True,
    )
    t0 = time.time()
    res = pmap(_job, [(asdict(s), str(out)) for s in specs], cap=args.cap)
    done = [r for r in res if not r.get("skipped")]
    print(f"\n  {len(done)} run, {len(res) - len(done)} skipped, "
          f"{time.time() - t0:.0f}s wall", flush=True)
    for r in res:
        extra = "skipped" if r.get("skipped") else f"{r.get('wall_seconds', 0):.0f}s"
        print(
            f"  N={r['N']} γ={r.get('gamma_0', '?')} usable={r.get('usable')} "
            f"converged_tasks={r.get('n_converged', '?')}/{r.get('n_tasks', 16)} {extra}",
            flush=True,
        )


if __name__ == "__main__":
    main()
