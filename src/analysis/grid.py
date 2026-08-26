"""Tidy views over the Phase 1 grid, shared by every figure script.

Loading and pooling live here rather than in each figure so that all figures see the same
data under the same conventions — in particular the same two, which are easy to get
wrong independently in four places:

**Attribution is re-derived from stored geometry**, never read from the `attribution`
field a worker wrote. Geometry is the measurement; attribution is analysis over it. This
is what let the ρ_c calibration fix be a re-summarize rather than a 26-hour re-run.

**Headline figures use `a = 0` only.** On `a > 0` arms the two control modules are
bitwise identical (`docs/06-scope.md` §4), so pooling A and B there halves the apparent
standard error without adding information. `a > 0` arms are for the alignment panel,
module A only.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from src import provenance
from src.analysis import timewarp
from src.analysis.attribution import (FACTORS, Attribution, GeometryPoint, attribute,
                                      attribution_table)

_SOURCE = provenance.register(__file__)

ROOT = Path(__file__).resolve().parents[2]
ARMS = ROOT / "results" / "phase1"

# The registered Phase 1 design (`docs/01-experiments.md`). Anything else in `ARMS` is an
# off-design arm that no headline figure agreed to include.
REGISTERED_GAMMAS = (0.03, 0.1, 0.3, 1.0, 3.0, 10.0)
REGISTERED_N = 300


def load(a_zero_only: bool = True, arms_dir: Path | None = None) -> list[dict]:
    """Every usable arm on disk, by default the homogeneous γ grid only.

    **Refuses off-design arms in the registered directory.** `Phase1Spec.key` carries γ but
    not `N`, so an exploratory arm dropped into `results/phase1/` either collides with a grid
    arm's filename or, worse, does not — and is then silently *adopted* by every figure, the
    scope audit included, with nothing failing and no diff to notice. That is the same failure
    class as the stale fork: no error, wrong artifact. Off-design runs live in their own
    directory and are loaded by passing `arms_dir` explicitly, which is a visible act.
    """
    registered = arms_dir is None
    recs = []
    for p in sorted((arms_dir or ARMS).glob("*.json")):
        r = json.loads(p.read_text())
        if registered:
            g, n = r["spec"]["gamma_0"], r["spec"].get("N", REGISTERED_N)
            if g not in REGISTERED_GAMMAS or n != REGISTERED_N:
                raise ValueError(
                    f"{p.name} is an off-design arm in the registered grid directory "
                    f"(γ={g:g}, N={n}; registered: γ∈{REGISTERED_GAMMAS}, N={REGISTERED_N}). "
                    f"Left there it would be adopted by every figure without any failure. "
                    f"Move it to its own directory under results/ and load it with "
                    f"`load(arms_dir=...)`."
                )
        if not r.get("usable"):
            continue
        if a_zero_only and r["spec"]["a"] != 0:
            continue
        recs.append(r)
    return recs


def gammas(recs: list[dict]) -> list[float]:
    return sorted({r["spec"]["gamma_0"] for r in recs})


def conditions(recs: list[dict]) -> list[str]:
    return sorted({r["spec"]["condition"] for r in recs})


# --- attribution --------------------------------------------------------------

def attributions(recs: list[dict], *, module: str | None = None
                 ) -> list[tuple[dict, str, int, int, Attribution]]:
    """`(spec, module, task, lag, Attribution)` for every retained-capacity comparison.

    The `before` point is the task's own boundary — the geometry as the task was left,
    which is the only baseline against which "forgetting" means anything.

    The arm's γ is passed through so the center-collapse share is gated by the `R_eff` noise
    floor measured *at that γ* (W4), rather than by one constant that was too permissive at
    high richness. Nothing else in the attribution depends on it.
    """
    out = []
    for r in recs:
        pts = {(g["module"], g["task"], g["boundary"]): g
               for g in r["geometry"] if g["task"] is not None}
        for (mod, task, b), g in sorted(pts.items()):
            if module is not None and mod != module:
                continue
            origin = pts.get((mod, task, task))
            if origin is None or b <= task:
                continue
            out.append((r["spec"], mod, task, b - task, attribute(
                GeometryPoint.from_result(origin), GeometryPoint.from_result(g),
                strict=False, gamma=r["spec"]["gamma_0"])))
    return out


def shares_by(recs: list[dict], *keys: str, module: str | None = None) -> dict:
    """Pooled attribution grouped by any spec fields, e.g. `shares_by(recs, 'gamma_0')`.

    Returns `key -> attribution_table(...)`, whose `terms` are means in log space (where
    the identity is additive) and whose `shares` are recomputed from the pooled terms
    rather than averaged, a mean of ratios not being the ratio of means.
    """
    groups: dict[tuple, list[Attribution]] = defaultdict(list)
    for spec, _mod, _task, _lag, att in attributions(recs, module=module):
        groups[tuple(spec[k] for k in keys)].append(att)
    return {(k if len(k) > 1 else k[0]): attribution_table(v)
            for k, v in sorted(groups.items())}


# --- capacity: generic vs retained -------------------------------------------

def capacity_curves(recs: list[dict], *keys: str) -> dict:
    """Generic and retained capacity, pooled by spec fields.

    Generic capacity is label-agnostic (`ensemble='generic'`); retained is
    `α(·; y_j)` for a specific past task. H2 is about whether these trade off, so they
    are kept separate here and never averaged together.
    """
    gen: dict[tuple, list[float]] = defaultdict(list)
    ret: dict[tuple, list[float]] = defaultdict(list)
    for r in recs:
        k = tuple(r["spec"][x] for x in keys)
        for g in r["geometry"]:
            (gen if g["ensemble"] == "generic" else ret)[k].append(g["alpha"])
    out = {}
    for k in sorted(set(gen) | set(ret)):
        key = k if len(k) > 1 else k[0]
        out[key] = {}
        for name, src in (("generic", gen), ("retained", ret)):
            v = np.array(src.get(k, []), dtype=float)
            out[key][name] = {
                "mean": float(v.mean()) if v.size else None,
                "sem": float(v.std(ddof=1) / np.sqrt(v.size)) if v.size > 1 else None,
                "n": int(v.size)}
    return out


# --- probe decodability -------------------------------------------------------

def probe_curves(recs: list[dict], *keys: str, measure: str = "margin") -> dict:
    """Probe decodability pooled by spec fields.

    `accuracy` is deliberately not the default: it saturated at 1.000 in the Phase 0
    check and carries no signal. `margin` is the primary measure and
    `heldout_manifold_accuracy` the control.
    """
    vals: dict[tuple, list[float]] = defaultdict(list)
    for r in recs:
        k = tuple(r["spec"][x] for x in keys)
        for mc in r["manipulation_checks"]:
            for _m, d in mc["probe_decodability"].items():
                if measure in d:
                    vals[k].append(d[measure])
    return {(k if len(k) > 1 else k[0]): {
        "mean": float(np.mean(v)), "sem": float(np.std(v, ddof=1) / np.sqrt(len(v))),
        "n": len(v)} for k, v in sorted(vals.items()) if v}


# --- forgetting and γ* --------------------------------------------------------

def forgetting_by(recs: list[dict], *keys: str, metric: str = "CFr") -> dict:
    vals: dict[tuple, list[float]] = defaultdict(list)
    for r in recs:
        vals[tuple(r["spec"][x] for x in keys)].append(r["forgetting"][metric])
    return {(k if len(k) > 1 else k[0]): {
        "mean": float(np.mean(v)), "sem": float(np.std(v, ddof=1) / np.sqrt(len(v))),
        "n": len(v)} for k, v in sorted(vals.items())}


def gamma_star(recs: list[dict], *, condition: str | None = None) -> dict:
    """γ minimising final-average error — the operating point H2's crossing is read against.

    Interpolated in log γ by a parabola through the argmin and its neighbours, since the
    grid is coarse (six points, half-decade spacing) and the discrete argmin would be
    quantised to a grid value.
    """
    sel = [r for r in recs if condition is None or r["spec"]["condition"] == condition]
    err: dict[float, list[float]] = defaultdict(list)
    for r in sel:
        err[r["spec"]["gamma_0"]].append(1.0 - r["forgetting"]["final_mean_accuracy"])
    gs = sorted(err)
    if not gs:
        return {}
    means = np.array([np.mean(err[g]) for g in gs])
    i = int(means.argmin())
    out = {"gamma_grid": gs, "mean_error": means.tolist(),
           "argmin_on_grid": gs[i], "min_error": float(means[i])}
    if 0 < i < len(gs) - 1:
        x = np.log10([gs[i - 1], gs[i], gs[i + 1]])
        c = np.polyfit(x, means[i - 1:i + 2], 2)
        if c[0] > 0:
            out["gamma_star_interpolated"] = float(10 ** (-c[1] / (2 * c[0])))
    return out


# --- ρ_c trajectories ---------------------------------------------------------

def rho_c_trajectories(recs: list[dict], *, convention: str = "rho_c_signed") -> dict:
    """`(γ, condition) -> boundary -> mean ρ_c`, for the decorrelation-rate test (H1d).

    Defaults to the signed convention: H1d is a claim about *direction* of center
    movement, and the absolute convention cannot represent direction at all.
    """
    vals: dict[tuple, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in recs:
        k = (r["spec"]["gamma_0"], r["spec"]["condition"])
        for g in r["geometry"]:
            if g["ensemble"] == "generic":
                vals[k][g["boundary"]].append(g[convention])
    return {k: {b: {"mean": float(np.mean(v)),
                    "sem": float(np.std(v, ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0,
                    "n": len(v)}
                for b, v in sorted(bs.items())}
            for k, bs in sorted(vals.items())}


# --- noise-floor units --------------------------------------------------------

def floors(delta_log: float, channel: str, *, gamma: float | None = None) -> float:
    """A log-space change expressed in Monte-Carlo noise floors of its own channel.

    The unit that makes "did anything move" answerable: 0.2 floors is nothing happening,
    65 floors is a real excursion. Channels without a measured floor raise rather than
    silently borrowing another channel's.

    Channels whose floor was measured to depend on γ (currently `R_eff`) use the floor at
    `gamma` when it is given, and the largest measured floor when it is not — a ruler that is
    too short flatters the result, so the default errs the other way. `alpha` and `D_eff` were
    measured γ-flat, so their registered floors are used at every γ.
    """
    per_gamma = timewarp.PER_GAMMA_FLOOR_CV.get(channel)
    if per_gamma is not None:
        return float(abs(delta_log) / np.log1p(
            per_gamma.get(gamma, max(per_gamma.values())) if gamma is not None
            else max(per_gamma.values())))
    if channel not in timewarp.NOISE_FLOOR_CV:
        raise KeyError(
            f"no measured noise floor for {channel!r}; available: "
            f"{sorted(timewarp.NOISE_FLOOR_CV) + sorted(timewarp.PER_GAMMA_FLOOR_CV)}. "
            f"Borrowing another channel's floor would make the number meaningless.")
    return float(abs(delta_log) / np.log1p(timewarp.NOISE_FLOOR_CV[channel]))


def excursion_by_gamma(recs: list[dict], channel: str = "alpha") -> dict:
    """Total geometric excursion per γ in noise floors, over the whole training window.

    Extends the Phase 0 progression (0.2 / 9.8 / 27.2 / 65.2 floors) to the full grid.
    """
    traj: dict[float, list[float]] = defaultdict(list)
    for r in recs:
        seq = [g for g in r["geometry"] if g["ensemble"] == "generic"]
        by_b = {}
        for g in seq:
            by_b.setdefault(g["boundary"], []).append(g[channel])
        bs = sorted(by_b)
        if len(bs) < 2:
            continue
        first, last = np.mean(by_b[bs[0]]), np.mean(by_b[bs[-1]])
        if first > 0 and last > 0:
            traj[r["spec"]["gamma_0"]].append(floors(np.log(last / first), channel))
    return {g: {"mean": float(np.mean(v)), "sem": float(np.std(v, ddof=1) / np.sqrt(len(v))),
                "n": len(v)} for g, v in sorted(traj.items()) if v}


def coverage_note(recs: list[dict]) -> str:
    """One line stating how much of the center-collapse panel is populated."""
    tot = ok = 0
    for _spec, _m, _t, _l, att in attributions(recs):
        tot += 1
        ok += att.center.get("attributable_fraction") is not None
    return (f"center-collapse share reported for {ok:,}/{tot:,} comparisons "
            f"({100 * ok / max(tot, 1):.0f}%)")


FACTOR_LABELS = {"utility": r"$\Psi_{\mathrm{eff}}$", "radius": r"$1+R_{\mathrm{eff}}^{-2}$",
                 "dimension": r"$-D_{\mathrm{eff}}$"}
FACTOR_COLORS = {"utility": "#4C72B0", "radius": "#DD8452", "dimension": "#55A868"}
__all__ = ["load", "gammas", "conditions", "attributions", "shares_by", "capacity_curves",
           "probe_curves", "forgetting_by", "gamma_star", "rho_c_trajectories", "floors",
           "excursion_by_gamma", "coverage_note", "FACTORS", "FACTOR_LABELS",
           "FACTOR_COLORS"]
