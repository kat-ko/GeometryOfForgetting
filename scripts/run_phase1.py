"""Phase 1 grid: γ × homogeneous `a` × Hiratani 2×2 streams × seeds (`01` §4).

Writes one JSON per arm under `results/phase1/` and a pooled summary at
`results/phase1_summary.json`. Per-arm files rather than one blob because the grid is
thousands of runs and a partial grid must remain usable — `--resume` skips arms whose
file already exists.

Analysis protocol reminder (`AGENTS.md` §8.2): the summary reports the diagnostics
that would reveal an artifact alongside every effect — non-converged arms, identity
residuals, factor cancellation, and the center-collapse share of the radius channel.
Interpret those first.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from _par import pin_threads  # noqa: E402

pin_threads()

import numpy as np  # noqa: E402

from _par import pmap  # noqa: E402
from src import pipeline as pl  # noqa: E402
from src import provenance  # noqa: E402
from src.analysis import attribution as AT  # noqa: E402
from src.analysis.attribution import (  # noqa: E402
    Attribution, GeometryPoint, attribute, attribution_table)

GAMMAS = (0.03, 0.1, 0.3, 1.0, 3.0, 10.0)   # as Phase 0, so the axes are comparable
A_GRID = (0.0, 0.5, 1.0)
CONDITIONS = ("S-HH", "S-HL", "S-LH", "S-LL")
SEEDS = tuple(range(8))
STREAMS_PER_CONDITION = 5

OUT = ROOT / "results" / "phase1"


def arms(*, gammas=GAMMAS, a_grid=A_GRID, conditions=CONDITIONS, seeds=SEEDS,
         streams=STREAMS_PER_CONDITION, **overrides) -> list[pl.Phase1Spec]:
    """The grid. `a > 0` only at the reference γ — the `a` sweep is homogeneous and
    is there to strengthen Figure 1's orthogonality panel, not to be crossed with γ."""
    out = []
    for g, cond, s, seed in itertools.product(gammas, conditions, range(streams), seeds):
        out.append(pl.Phase1Spec(gamma_0=g, a=0.0, condition=cond, stream_id=s,
                                 seed=seed, **overrides))
    for a, cond, s, seed in itertools.product(
            [x for x in a_grid if x > 0], conditions, range(streams), seeds):
        out.append(pl.Phase1Spec(gamma_0=1.0, a=a, condition=cond, stream_id=s,
                                 seed=seed, **overrides))
    # Order so that an interrupted grid still contains Figure 2's core contrast.
    # Priority 0 is the extreme-γ homogeneous cells — the lazy-vs-rich comparison the
    # attribution rests on. Then the mid-γ fill, then the `a` sweep, which strengthens
    # Figure 1 but carries no hypothesis. Within a tier the order is a fixed shuffle:
    # the first wave then spans conditions and seeds, so a wiring problem shows up in
    # the earliest completions, and since steps-to-target spans 39× across γ, mixing
    # keeps the slow arms from bunching and leaving workers idle at the end.
    def priority(s: pl.Phase1Spec) -> int:
        if s.a > 0:
            return 2
        return 0 if s.gamma_0 in (min(gammas), max(gammas)) else 1

    shuffled = [out[i] for i in np.random.default_rng(20260811).permutation(len(out))]
    return sorted(shuffled, key=priority)


def _path(spec: pl.Phase1Spec) -> Path:
    # `spec.key` omits T/P/M/N/n_t, so a reduced-size arm can collide with a real one.
    # Smoke output therefore goes to its own directory, and `--resume` additionally
    # verifies the stored spec before trusting a file.
    sub = "smoke" if spec.n_t != pl.Phase1Spec().n_t else "."
    return OUT / sub / (spec.key.replace(",", "__").replace("=", "-") + ".json")


def _job(spec: pl.Phase1Spec) -> dict:
    # Refuse to measure with code older than the working tree. Under `spawn` this
    # cannot trigger; it is the backstop for anyone who switches back to `fork`.
    provenance.assert_current()
    path = _path(spec)
    if path.exists():
        try:
            stored = json.loads(path.read_text())["spec"]
        except (json.JSONDecodeError, KeyError):
            stored = None
        # Compare through the same serialization: `module_list` is a tuple in Python
        # and a list once round-tripped, so a direct `==` never matches.
        if stored == json.loads(json.dumps(asdict(spec))):
            return {"key": spec.key, "path": str(path), "skipped": True}
        print(f"  re-running {spec.key}: stored spec differs", flush=True)
    rec = pl.run_arm(spec)
    path.write_text(json.dumps(rec))
    return {"key": spec.key, "path": str(path), "skipped": False,
            "usable": rec["usable"], "wall_seconds": rec["wall_seconds"]}


def _reattribute(rec: dict) -> list[tuple[str, int, int, Attribution]]:
    """Recompute attribution from the arm's stored geometry.

    Deliberately *not* reusing `rec["attribution"]`, which was computed inside the
    worker against whatever calibration was current then. Geometry is the measurement;
    attribution is analysis over it. Re-deriving here means a calibration correction is
    a re-summarize rather than a 26-hour re-run — which is what the `rho_c` convention
    fix needed, the per-arm files having been written with the superseded constants.
    """
    pts = {(g["module"], g["task"], g["boundary"]): g
           for g in rec["geometry"] if g["task"] is not None}
    out = []
    for (mod, task, b), g in sorted(pts.items()):
        origin = pts.get((mod, task, task))
        if origin is None or b <= task:
            continue
        out.append((mod, task, b - task, attribute(
            GeometryPoint.from_result(origin, label=f"{mod}@{task}"),
            GeometryPoint.from_result(g, label=f"{mod}@{b}"),
            strict=False)))
    return out


def rho_coverage(recs: list[dict]) -> dict:
    """Per-condition ρ_c range, and how much of it the calibration actually covers.

    The ρ_c → R_eff calibration is fitted on synthetic manifolds over
    `attribution.RHO_FIT_RANGE` and is refused outside it, so the bottom row of Figure 2
    is populated only where representations land inside. The pilot landed at 0.33–0.48,
    but it spanned two γ values out of six and one `a`, so the grid will visit conditions
    it did not. Reporting coverage per condition here means the answer comes from a log
    after 4 hours rather than from counting gaps in a figure — and if a large fraction
    falls outside, that is a finding about representations, not a plotting problem.

    Both conventions are reported. `rho_c_signed` is the calibration's input; the range
    of `rho_c_glue` is recorded too, since it is what §8 requires alongside R and its
    excursion past 1 is exactly what made the unnormalized convention unusable here.
    """
    lo, hi = AT.RHO_FIT_RANGE
    groups: dict[tuple, list[dict]] = {}
    for r in recs:
        if not r["usable"]:
            continue
        sp = r["spec"]
        for g in r["geometry"]:
            if g["task"] is not None:
                groups.setdefault((sp["gamma_0"], sp["a"], sp["condition"]), []).append(g)

    out = {}
    for k, gs in sorted(groups.items()):
        signed = np.array([g["rho_c_signed"] for g in gs])
        glue = np.array([g["rho_c_glue"] for g in gs])
        inside = (signed >= lo - AT.RHO_DOMAIN_TOL) & (signed <= hi + AT.RHO_DOMAIN_TOL)
        out["|".join(map(str, k))] = {
            "n": int(signed.size),
            "rho_c_signed": {"min": float(signed.min()), "max": float(signed.max()),
                             "median": float(np.median(signed))},
            "rho_c_glue": {"min": float(glue.min()), "max": float(glue.max()),
                           "median": float(np.median(glue)),
                           "n_above_one": int((glue > 1.0).sum())},
            "in_calibration_range": float(inside.mean()),
        }
    return out


def summarize(paths: list[Path]) -> dict:
    """Pool attribution by (γ, `a`, condition, module, lag) and surface the checks."""
    recs = [json.loads(p.read_text()) for p in paths]
    usable = [r for r in recs if r["usable"]]

    groups: dict[tuple, list[Attribution]] = {}
    forgetting: dict[tuple, list[float]] = {}
    for r in usable:
        sp = r["spec"]
        for mod, _task, lag, att in _reattribute(r):
            key = (sp["gamma_0"], sp["a"], sp["condition"], mod, lag)
            groups.setdefault(key, []).append(att)
        fk = (sp["gamma_0"], sp["a"], sp["condition"])
        forgetting.setdefault(fk, []).append(r["forgetting"]["CFr"])

    pooled = {"|".join(map(str, k)): attribution_table(v) for k, v in sorted(groups.items())}
    resid = [abs(g["identity_residual"]) for r in usable for g in r["geometry"]]
    return {
        "n_arms": len(recs),
        "n_usable": len(usable),
        "non_converged": [r["key"] for r in recs if not r["usable"]],
        "max_identity_residual": float(max(resid)) if resid else None,
        "attribution": pooled,
        "rho_coverage": rho_coverage(recs),
        "forgetting_CFr": {"|".join(map(str, k)): {
            "mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
            "n": len(v)} for k, v in sorted(forgetting.items())},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="tiny grid at reduced size, for wiring checks only")
    ap.add_argument("--pilot", action="store_true",
                    help="8 arms at FULL settings: extreme γ × 2 conditions × 2 seeds. "
                         "The point is to see a real Figure 2 before committing the grid")
    # Resume is the default, and discarding is explicit. It was the other way round,
    # which cost the pilot: re-invoking `--pilot` to *test* resume deleted all eight
    # completed arms before running, because deletion was the default and `--resume` the
    # opt-in. On a 4-hour grid, with `results/phase1/` in `.gitignore` and so no git
    # safety net, an accidental re-invocation would have destroyed the run. A default
    # should not be the destructive branch.
    ap.add_argument("--resume", action="store_true",
                    help="accepted and ignored; resuming is the default")
    ap.add_argument("--fresh", action="store_true",
                    help="discard existing arms and recompute. Files are moved to "
                         "results/phase1/.trash-<timestamp>/, not deleted")
    ap.add_argument("--cap", type=int, default=None, help="max worker processes")
    ap.add_argument("--summarize-only", action="store_true")
    args = ap.parse_args()

    (OUT / "smoke").mkdir(parents=True, exist_ok=True)

    if args.smoke:
        specs = arms(gammas=(0.3, 10.0), a_grid=(), conditions=("S-HL",), seeds=(0,),
                     streams=1, T=4, P=8, M=40, N=80, d=60, n_t=25,
                     tracked_stride=2, steps_per_task=3000)
    elif args.pilot:
        specs = arms(gammas=(0.03, 10.0), a_grid=(), conditions=("S-HL", "S-LL"),
                     seeds=(0, 1), streams=1)
    else:
        specs = arms()

    if not args.summarize_only:
        if args.fresh:
            existing = [p for p in (_path(s) for s in specs) if p.exists()]
            if existing:
                trash = OUT / f".trash-{time.strftime('%Y%m%d-%H%M%S')}"
                trash.mkdir(parents=True, exist_ok=True)
                for p in existing:
                    p.rename(trash / p.name)
                print(f"--fresh: moved {len(existing)} existing arms to {trash.name}/")
        budget = pl.n_evals(pl.schedule(specs[0].T, specs[0].tracked_stride), specs[0])
        n_done = sum(1 for s in specs if _path(s).exists())
        print(f"{len(specs)} arms x {budget} evals = {len(specs) * budget:,} evaluations"
              + (f"  ({n_done} already on disk, resuming)" if n_done else ""))
        done = pmap(_job, specs, cap=args.cap)
        ran = [d for d in done if not d.get("skipped")]
        if ran:
            wall = sum(d["wall_seconds"] for d in ran)
            print(f"  {len(ran)} arms run, {sum(1 for d in ran if not d['usable'])} "
                  f"non-converged, {wall / 3600:.2f} core-hours")

    summary = summarize([_path(s) for s in specs if _path(s).exists()])
    name = ("phase1_summary_smoke.json" if args.smoke
            else "phase1_summary_pilot.json" if args.pilot else "phase1_summary.json")
    (ROOT / "results" / name).write_text(json.dumps(summary, indent=2))

    print(f"\n{summary['n_usable']}/{summary['n_arms']} usable; "
          f"max identity residual {summary['max_identity_residual']:.2e}")
    print("\npooled attribution (Δlog α and factor terms, per group):")
    for key, t in list(summary["attribution"].items())[:12]:
        terms = "  ".join(f"{k[:3]} {t['terms'][k]:+.4f}" for k in ("utility", "radius",
                                                                   "dimension"))
        cen = t["center_attributable_median"]
        print(f"  {key:34s} dlogA {t['dlog_alpha']:+.4f}  {terms}  "
              f"dom {t['dominant']:<9s} center {cen if cen is None else round(cen, 2)}")
    lo, hi = AT.RHO_FIT_RANGE
    cov = summary["rho_coverage"]
    print(f"\nρ_c coverage — calibration fitted on [{lo:.2f}, {hi:.2f}], "
          f"refused outside (±{AT.RHO_DOMAIN_TOL:.2f}):")
    for key, c in sorted(cov.items(), key=lambda kv: kv[1]["in_calibration_range"]):
        s, g = c["rho_c_signed"], c["rho_c_glue"]
        print(f"  {key:22s} signed [{s['min']:.3f}, {s['max']:.3f}]  "
              f"glue [{g['min']:.3f}, {g['max']:.3f}] ({g['n_above_one']:4d}/{c['n']:4d} > 1)"
              f"   in range {100 * c['in_calibration_range']:5.1f}%")
    if cov:
        overall = np.mean([c["in_calibration_range"] for c in cov.values()])
        print(f"  {'OVERALL':22s} {100 * overall:5.1f}% of measurements usable for the "
              f"center-collapse panel")
        worst = min(cov.items(), key=lambda kv: kv[1]["in_calibration_range"])
        if worst[1]["in_calibration_range"] < 0.5:
            print(f"  NOTE: {worst[0]} is only "
                  f"{100 * worst[1]['in_calibration_range']:.0f}% in range — the "
                  f"calibration needs extending before that condition's radius channel "
                  f"can be attributed.")

    print(f"\nwrote results/{name}")


if __name__ == "__main__":
    main()
