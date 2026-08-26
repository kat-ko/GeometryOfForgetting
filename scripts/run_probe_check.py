"""Is the probe-dichotomy measure saturated? (artifact check, `AGENTS.md` §8.2)

Figure 3 overlays a probe measure on generic capacity. The obvious choice —
accuracy of a refit linear readout on the held-out dichotomy `y*` — turned out to be
exactly 1.0 at the design point, both at initialization and after training. The load
is `P/N = 16/300 = 0.053` against a critical capacity near 0.3, so every balanced
dichotomy is separable with room to spare. A flat line at 1.0 would have been read as
"generic capacity is preserved" rather than as a dead instrument.

This records the saturation and checks that the replacements move: the readout
*margin*, and generalization to *held-out manifolds*. A probe measure has to satisfy
two things to be worth plotting — it must move with training, and it must move
differently across γ. Otherwise it cannot carry Figure 3.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import pipeline as pl  # noqa: E402
from src.models import paired_init  # noqa: E402
from src.train.loop import train_task  # noqa: E402

MEASURES = ("accuracy", "margin", "margin_p05", "heldout_manifold_accuracy")


def one(gamma: float, seed: int, steps: int) -> dict:
    spec = pl.Phase1Spec(T=2, gamma_0=gamma, seed=seed, steps_per_task=steps)
    streams = paired_init(seed)
    stream = pl.build_stream(spec, streams["stream"])
    model = pl.build_model(spec, stream)
    pts, y = stream.arrangements[0].points, stream.dichotomies[0]

    before = pl.probe_decodability(model, pts, stream.probe, ("A",))["A"]
    rec = train_task(model, pts, y, spec.train_config, streams["data"])
    after = pl.probe_decodability(model, pts, stream.probe, ("A",))["A"]

    return {"gamma": gamma, "seed": seed, "steps_taken": rec.steps_taken,
            "final_loss": rec.final_loss, "converged": rec.converged,
            "before": before, "after": after,
            "delta": {k: after[k] - before[k] for k in MEASURES}}


def verdict(runs: list[dict], gammas: list[float]) -> dict:
    """A probe measure earns Figure 3 only if its γ signal beats its own noise.

    Three tests, and a measure must pass all three. **Saturation**: pinned at the
    ceiling, so it cannot move. **Signal vs noise**: the spread of γ means must exceed
    twice the noise, where noise is the larger of the seed-to-seed spread and the
    within-run split sd — a γ difference smaller than the seed difference is a seed
    difference. **Sign consistency**: the change over training must have the same sign
    at every seed within an arm, since a measure whose direction flips between seeds
    cannot support a directional claim however large its mean.
    """
    out = {}
    for k in MEASURES:
        by_g = {g: [r for r in runs if r["gamma"] == g] for g in gammas}
        means = [float(np.mean([r["after"][k] for r in by_g[g]])) for g in gammas]
        seed_noise = float(np.mean([np.std([r["after"][k] for r in by_g[g]], ddof=1)
                                    for g in gammas if len(by_g[g]) > 1]))
        split_noise = float(np.mean([r["after"].get(k.replace(
            "heldout_manifold_accuracy", "heldout_manifold_sd"), 0.0)
            if k == "heldout_manifold_accuracy" else 0.0 for r in runs]))
        noise = max(seed_noise, split_noise)
        signal = float(max(means) - min(means))
        saturated = bool(all(abs(r["delta"][k]) < 1e-9 for r in runs)
                         and all(abs(r["after"][k] - 1.0) < 1e-9 for r in runs))
        consistent = all(
            len({np.sign(round(r["delta"][k], 6)) for r in by_g[g]}) == 1 for g in gammas)
        out[k] = {
            "per_gamma_mean": dict(zip(map(str, gammas), means)),
            "signal_across_gamma": signal, "seed_noise": seed_noise,
            "split_noise": split_noise, "snr": signal / noise if noise > 0 else float("inf"),
            "saturated": saturated, "sign_consistent_over_training": bool(consistent),
            "mean_abs_change": float(np.mean([abs(r["delta"][k]) for r in runs])),
            "usable": bool(not saturated and signal > 2 * noise and consistent),
        }
    return out


def main() -> None:
    reanalyze = "--reanalyze" in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    path = ROOT / "results" / "probe_check.json"

    if reanalyze:
        runs = json.loads(path.read_text())["runs"]
        gammas = sorted({r["gamma"] for r in runs}, reverse=True)
    else:
        gammas = [float(g) for g in (args or [10.0, 1.0, 0.03])]
        runs = [one(g, seed, 20_000) for g in gammas for seed in (0, 1)]

    v = verdict(runs, gammas)
    usable = [k for k in MEASURES if v[k]["usable"]]
    path.write_text(json.dumps({"runs": runs, "verdict": v,
                                "usable_for_figure_3": usable}, indent=2))

    hdr = "  ".join(f"g={g:<7g}" for g in gammas)
    print(f"{'measure':28s} {hdr}  {'signal':>7} {'noise':>7} {'snr':>5}  sign  usable")
    for k in MEASURES:
        d = v[k]
        cells = "  ".join(f"{d['per_gamma_mean'][str(g)]:+9.4f}" for g in gammas)
        noise = max(d["seed_noise"], d["split_noise"])
        print(f"{k:28s} {cells}  {d['signal_across_gamma']:7.4f} {noise:7.4f} "
              f"{d['snr']:5.1f}  {str(d['sign_consistent_over_training']):5s} "
              f"{'YES' if d['usable'] else 'no'}"
              + ("   (SATURATED)" if d["saturated"] else ""))
    print("\nusable for Figure 3:", usable)


if __name__ == "__main__":
    main()
