"""Tier 2: characterise the gain in the benign corner. Reads stored geometry, trains nothing.

Three questions, kept separate because they have different exposures.

**2.1 — how the gain decomposes.** The qualitative shape is already known from the Tier 1 audit:
utility and dimension positive against a negative radius term from γ = 1 onward. This quantifies it
against the loss corners at matched richness. Low risk: it is measurement of a shape already seen.

**2.2 — whether the gain is general.** Lag and task position are confounded by construction, so
this reports two matched series and never pools them: position at matched lag 4 (tasks 0, 4, 8) and
lag within task 0 (lags 4, 8, 12, 15). The framing for each possible outcome was fixed in
`results/LOG.md` before this script was run.

**2.3 — where the gain peaks.** Reported as a bootstrap CI over the on-grid argmax, because the
grid resolution is one step and the question is whether the location is identified at all. The text
rule is unconditional and does not depend on what comes back: no sentence in the paper contains both
this peak and the behavioural optimum, since γ* is itself unidentified and pairing two unresolved
locations is not evidence of agreement.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.analysis import grid as G  # noqa: E402
from src.analysis.attribution import FACTORS, attribution_table  # noqa: E402
from src.analysis.ledger import BENIGN, THREE, _obs, pick  # noqa: E402

MIN_FLOORS = 2.0
N_BOOT = 4000
SEED = 20260813
OUT = ROOT / "results" / "tier2_backward_transfer.json"

MATCHED_LAG = 4
MATCHED_POSITIONS = (0, 4, 8)
LAG_SERIES_TASK = 0
LAG_SERIES = (4, 8, 12, 15)


def floors(dlog: float) -> float:
    return float(np.sign(dlog) * G.floors(dlog, "alpha"))


def cell(rows: list[dict]) -> dict:
    """Pooled attribution for a cell, with a bootstrap CI over its arms."""
    t = attribution_table([r["att"] for r in rows])
    per_arm = np.array([r["att"].dlog_alpha for r in rows])
    rng = np.random.default_rng(SEED)
    draws = np.array([per_arm[rng.integers(0, len(per_arm), len(per_arm))].mean()
                      for _ in range(N_BOOT)])
    lo, hi = float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))
    fl = floors(t["dlog_alpha"])
    return {
        "n": len(rows), "dlog_alpha": t["dlog_alpha"], "floors": fl,
        "ci": [lo, hi], "ci_floors": [floors(lo), floors(hi)],
        "excludes_zero": bool(lo > 0 or hi < 0),
        "resolved_gain": bool(fl >= MIN_FLOORS and lo > 0),
        "terms": t["terms"], "shares": t["shares"],
    }


# --- 2.1 ----------------------------------------------------------------------

def signed_shares(terms: dict) -> dict:
    """Each term as a fraction of the **net** change, keeping its sign.

    `attribution` reports `|term| / Σ|term|` deliberately, because opposing factors make shares of
    a net difference blow up. In the three loss corners every term has the same sign and the two
    agree. In the benign corner the radius term opposes the other two, and the magnitude share
    then reads as a contribution when it is a subtraction — the §A.3 fault, arriving through a
    normalisation rather than an absolute value. Reported here because this is the one place in
    the paper where the terms disagree in sign.
    """
    net = sum(terms.values())
    if net == 0:
        return {f: None for f in FACTORS}
    return {f: float(terms[f] / net) for f in FACTORS}


def per_corner_terms(obs: list[dict]) -> dict:
    """All four corners separately, which is where 2.1's actual answer turned out to be.

    Pooling the three loss corners hides that the radius term is negative in *every* corner,
    including the one that gains. The gain is not a corner doing the opposite of forgetting in all
    three channels; it is a corner doing the opposite in two of them while the third does what it
    does everywhere.
    """
    out = {}
    for g in sorted({r["gamma"] for r in obs}):
        row = {}
        for c in (BENIGN, *THREE):
            p = cell(pick(obs, conditions=(c,), gamma=g, lag=12))
            row[c] = {"floors": p["floors"], "dlog_alpha": p["dlog_alpha"],
                      "terms": p["terms"], "resolved": abs(p["floors"]) >= MIN_FLOORS}
        resolved = [c for c, v in row.items() if v["resolved"]]
        out[f"{g:g}"] = {
            "corners": row,
            "radius_negative_everywhere": bool(
                resolved and all(row[c]["terms"]["radius"] < 0 for c in resolved)),
            "utility_and_dimension_split_by_sign": bool(
                resolved and len({np.sign(row[c]["terms"]["utility"]) for c in resolved}) > 1),
        }
    return out


def channel_composition(obs: list[dict]) -> dict:
    """The three-term decomposition of the gain, against the losses at matched γ."""
    out = {}
    for g in sorted({r["gamma"] for r in obs}):
        gain = cell(pick(obs, conditions=(BENIGN,), gamma=g, lag=12))
        loss = cell(pick(obs, conditions=THREE, gamma=g, lag=12))
        both_resolved = abs(gain["floors"]) >= MIN_FLOORS and abs(loss["floors"]) >= MIN_FLOORS
        out[f"{g:g}"] = {
            "gain": gain, "loss": loss, "both_resolved": both_resolved,
            "gain_signed_shares": signed_shares(gain["terms"]),
            "loss_signed_shares": signed_shares(loss["terms"]),
            "gain_term_signs": {f: int(np.sign(gain["terms"][f])) for f in FACTORS},
            "loss_term_signs": {f: int(np.sign(loss["terms"][f])) for f in FACTORS},
            "radius_opposes_in_gain": bool(
                np.sign(gain["terms"]["radius"]) != np.sign(gain["terms"]["utility"])),
            "all_negative_in_loss": bool(
                all(loss["terms"][f] < 0 for f in FACTORS)),
        }
    return out


# --- 2.2 ----------------------------------------------------------------------

def generality(obs: list[dict], gamma: float) -> dict:
    """Two matched series. Never pooled: lag and position are confounded by construction."""
    pos = {str(t): cell(pick(obs, conditions=(BENIGN,), gamma=gamma, task=t, lag=MATCHED_LAG))
           for t in MATCHED_POSITIONS}
    lag = {str(lg): cell(pick(obs, conditions=(BENIGN,), gamma=gamma,
                              task=LAG_SERIES_TASK, lag=lg))
           for lg in LAG_SERIES}
    every = {}
    for t in sorted({r["task"] for r in obs}):
        for lg in sorted({r["lag"] for r in obs if r["task"] == t}):
            rows = pick(obs, conditions=(BENIGN,), gamma=gamma, task=t, lag=lg)
            if rows:
                every[f"task{t}|lag{lg}"] = cell(rows)

    n_pos = sum(1 for v in pos.values() if v["resolved_gain"])
    n_lag = sum(1 for v in lag.values() if v["resolved_gain"])
    if n_pos == len(pos) and n_lag == len(lag):
        verdict = "general"
    elif n_pos == 1 and pos["0"]["resolved_gain"]:
        verdict = "primacy"
    else:
        verdict = "gradient"

    # Position and lag effects, each measured only within its own matched series.
    pf = [pos[str(t)]["floors"] for t in MATCHED_POSITIONS]
    lf = [lag[str(lg)]["floors"] for lg in LAG_SERIES]
    return {
        "gamma": gamma, "position_at_matched_lag_4": pos, "lag_within_task_0": lag,
        "every_cell": every, "n_positions_resolved": n_pos, "n_positions": len(pos),
        "n_lags_resolved": n_lag, "n_lags": len(lag), "verdict": verdict,
        "position_effect_task0_over_task8": float(pf[0] / pf[-1]) if pf[-1] else None,
        "lag_effect_15_over_4": float(lf[-1] / lf[0]) if lf[0] else None,
        "position_monotone": bool(all(a >= b for a, b in zip(pf, pf[1:]))),
        "n_cells_resolved": sum(1 for v in every.values() if v["resolved_gain"]),
        "n_cells": len(every),
    }


# --- 2.3 ----------------------------------------------------------------------

def _vertex(x: np.ndarray, y: np.ndarray) -> float:
    """Vertex of the least-squares parabola in log γ. Exact for unequal spacing; the grid is
    0.03/0.1/0.3/1/3/10, which is not quite uniform in log."""
    a, b, _ = np.polyfit(np.asarray(x, float), np.asarray(y, float), 2)
    return float(x[len(x) // 2]) if a >= 0 else float(-b / (2 * a))


def peak_location(obs: list[dict]) -> dict:
    """Two different questions, kept apart.

    The bootstrap over the **on-grid argmax** asks which of six sampled richnesses is highest.
    That is not the location of the peak: the grid steps by a factor of ~3.2, so an argmax
    identified with certainty still leaves the continuous maximum bracketed only by its two
    neighbours. The second statistic interpolates a parabola in log γ through the argmax and its
    neighbours and bootstraps *that*, which is a CI over a location — and it is the one to read.
    """
    gammas = sorted({r["gamma"] for r in obs})
    by_g = {g: np.array([r["att"].dlog_alpha
                         for r in pick(obs, conditions=(BENIGN,), gamma=g, lag=12)])
            for g in gammas}
    means = np.array([by_g[g].mean() for g in gammas])
    lg = np.log10(gammas)
    rng = np.random.default_rng(SEED)
    argmax, vertices = [], []
    for _ in range(N_BOOT):
        m = np.array([by_g[g][rng.integers(0, len(by_g[g]), len(by_g[g]))].mean()
                      for g in gammas])
        k = int(np.argmax(m))
        argmax.append(gammas[k])
        if 0 < k < len(gammas) - 1:
            vertices.append(_vertex(lg[k - 1:k + 2], m[k - 1:k + 2]))
    argmax = np.array(argmax)
    counts = {f"{g:g}": int((argmax == g).sum()) for g in gammas}
    lo, hi = float(np.percentile(argmax, 2.5)), float(np.percentile(argmax, 97.5))
    span = sum(1 for g in gammas if lo <= g <= hi)
    k0 = int(np.argmax(means))
    vlo, vhi = (float(10 ** np.percentile(vertices, 2.5)),
                float(10 ** np.percentile(vertices, 97.5))) if vertices else (None, None)
    bracket = [gammas[max(0, k0 - 1)], gammas[min(len(gammas) - 1, k0 + 1)]]
    return {
        "gammas": gammas, "means": means.tolist(),
        "on_grid_argmax": float(gammas[k0]),
        "bootstrap_ci": [lo, hi], "argmax_distribution": counts,
        "modal_share": max(counts.values()) / N_BOOT,
        "grid_steps_spanned": span,
        "on_grid_argmax_identified": bool(span <= 1),
        "grid_step_factor": float(gammas[1] / gammas[0]),
        "neighbour_bracket": bracket,
        "interpolated_peak": float(10 ** _vertex(lg[k0 - 1:k0 + 2], means[k0 - 1:k0 + 2])),
        "interpolated_peak_ci": [vlo, vhi],
        "interpolated_peak_ci_width_in_steps": (
            float(np.log10(vhi / vlo) / np.log10(gammas[1] / gammas[0]))
            if vlo and vhi else None),
        "location_identified": False,  # set below, on the interpolated CI
        "text_rule": ("Unconditional, fixed before this ran: no sentence in the paper contains "
                      "both this peak location and the behavioural optimum. γ* is itself "
                      "unidentified (CI [0.339, 4.426], on-grid argmin split 34.6/34.4 between "
                      "γ = 1 and γ = 3), so pairing them would pair two unresolved locations. "
                      "If the CI below spans more than one grid step, §5.1 says the gain is "
                      "largest at low-to-moderate richness and decays, and makes no location "
                      "claim."),
    }


def _argmax_of_fit(x: np.ndarray, y: np.ndarray, *, degree: int) -> float:
    """Where a polynomial fit peaks, searched on the sampled range only."""
    grid = np.linspace(x[0], x[-1], 2001)
    return float(grid[int(np.argmax(np.polyval(np.polyfit(x, y, degree), grid)))])


def peak_model_sensitivity(obs: list[dict]) -> dict:
    """How much of the interpolated CI's tightness is the assumption rather than the data?

    A CI from resampling arms propagates sampling noise in the means and nothing else. It is
    conditional on the curve being a parabola in log γ, which three points spaced half a decade
    apart cannot test. Refitting under other defensible choices separates the two: if the vertex
    moves further across functional forms than the CI is wide, the CI is reporting the precision
    of an assumption.
    """
    gammas = sorted({r["gamma"] for r in obs})
    lg = np.log10(gammas)
    means = np.array([np.mean([r["att"].dlog_alpha for r in
                               pick(obs, conditions=(BENIGN,), gamma=g, lag=12)])
                      for g in gammas])
    k = int(np.argmax(means))
    fits = {
        "parabola through the argmax and its two neighbours":
            _vertex(lg[k - 1:k + 2], means[k - 1:k + 2]),
        "parabola through all six richnesses":
            _vertex(lg, means),
        "parabola through the four central richnesses":
            _vertex(lg[1:5], means[1:5]),
        "parabola through the argmax and its two next-nearest neighbours":
            _vertex(lg[[k - 2, k, k + 1]], means[[k - 2, k, k + 1]]),
        "cubic through all six, maximised on the sampled range":
            _argmax_of_fit(lg, means, degree=3),
    }
    vals = {k_: float(10 ** v) for k_, v in fits.items()}
    step = np.log10(gammas[1] / gammas[0])
    spread_steps = float((max(fits.values()) - min(fits.values())) / step)
    return {"vertex_by_model": vals, "spread_in_grid_steps": spread_steps,
            "range": [float(min(vals.values())), float(max(vals.values()))]}


def peak_paired_and_per_lag(obs: list[dict]) -> dict:
    """Two checks on the located peak.

    **Paired.** Arms at different richnesses share seeds and stream instantiations, so resampling
    each richness independently breaks a pairing that exists in the design. Resampling the
    (seed, stream) unit and recomputing every richness from the same units is the matched
    estimator.

    **Per lag.** The peak above is computed at lag 12. 2.2 shows the gain's richness dependence is
    itself lag-dependent, so a peak that moves with lag would be a property of lag 12 rather than
    of the corner.
    """
    gammas = sorted({r["gamma"] for r in obs})
    lg = np.log10(gammas)
    rng = np.random.default_rng(SEED)

    rows = pick(obs, conditions=(BENIGN,), lag=12)
    units = sorted({(r["seed"], r["stream_id"]) for r in rows})
    by_unit: dict = {u: {} for u in units}
    for r in rows:
        by_unit[(r["seed"], r["stream_id"])][r["gamma"]] = r["att"].dlog_alpha
    complete = [u for u in units if len(by_unit[u]) == len(gammas)]
    vertices = []
    for _ in range(N_BOOT):
        draw = [by_unit[complete[i]] for i in rng.integers(0, len(complete), len(complete))]
        m = np.array([np.mean([d[g] for d in draw]) for g in gammas])
        k = int(np.argmax(m))
        if 0 < k < len(gammas) - 1:
            vertices.append(_vertex(lg[k - 1:k + 2], m[k - 1:k + 2]))
    paired = {
        "n_units": len(complete),
        "ci": [float(10 ** np.percentile(vertices, 2.5)),
               float(10 ** np.percentile(vertices, 97.5))] if vertices else None,
    }
    if paired["ci"]:
        paired["ci_width_in_steps"] = float(
            np.log10(paired["ci"][1] / paired["ci"][0]) / np.log10(gammas[1] / gammas[0]))

    per_lag = {}
    for lag in LAG_SERIES:
        means = np.array([np.mean([r["att"].dlog_alpha for r in
                                   pick(obs, conditions=(BENIGN,), gamma=g,
                                        task=LAG_SERIES_TASK, lag=lag)])
                          for g in gammas])
        k = int(np.argmax(means))
        per_lag[str(lag)] = {
            "on_grid_argmax": float(gammas[k]),
            "interpolated_peak": float(10 ** _vertex(lg[max(0, k - 1):k + 2],
                                                     means[max(0, k - 1):k + 2]))
            if 0 < k < len(gammas) - 1 else None,
            "means": means.tolist(),
        }
    peaks = [v["interpolated_peak"] for v in per_lag.values() if v["interpolated_peak"]]
    span = (float(np.log10(max(peaks) / min(peaks)) / np.log10(gammas[1] / gammas[0]))
            if peaks else None)
    # A range test alone would call a monotone slide "stable" whenever it happens to fit inside
    # one grid step. Systematic movement is the thing being asked about, so test the trend.
    monotone = bool(peaks and (all(a > b for a, b in zip(peaks, peaks[1:]))
                               or all(a < b for a, b in zip(peaks, peaks[1:]))))
    return {
        "paired_bootstrap": paired, "per_lag": per_lag,
        "peak_range_across_lags": [float(min(peaks)), float(max(peaks))] if peaks else None,
        "peak_span_in_grid_steps": span,
        "peak_moves_monotonically_with_lag": monotone,
        "peak_is_a_property_of_the_corner": bool(not monotone and span is not None and span <= 0.5),
    }


def finalise_peak(p: dict, sens: dict, checks: dict) -> dict:
    p["model_sensitivity"] = sens
    p["checks"] = checks
    ci_w = p["interpolated_peak_ci_width_in_steps"]
    p["model_error_dominates"] = bool(sens["spread_in_grid_steps"] > ci_w)
    # A location counts as identified only if it survives both sources of uncertainty inside one
    # grid step. Sampling noise alone is the wrong test when the functional form is unverifiable.
    # Three sources of uncertainty, and the third is the one that decides it: sampling noise,
    # the choice of functional form, and which lag the series is read at. A peak located to a
    # fifth of a grid step at one lag is not a located peak if it slides a full step across lags.
    p["location_identified_at_lag_12"] = bool(
        ci_w is not None and max(ci_w, sens["spread_in_grid_steps"]) <= 1.0)
    p["location_identified"] = bool(
        p["location_identified_at_lag_12"]
        and checks["peak_is_a_property_of_the_corner"])
    return p


def main() -> None:
    obs = _obs()
    report = {
        "generated_by": "scripts/tier2_backward_transfer.py",
        "reads": ["results/phase1/*.json"],
        "trained_or_measured_anything": False,
        "gate": {"min_floors": MIN_FLOORS, "bootstrap_n": N_BOOT, "seed": SEED},
        "2.1_channel_composition": channel_composition(obs),
        "2.1_per_corner_terms": per_corner_terms(obs),
        "2.2_generality": {f"{g:g}": generality(obs, g) for g in (1.0, 3.0, 10.0)},
        "2.3_peak_location": finalise_peak(
            peak_location(obs), peak_model_sensitivity(obs),
            peak_paired_and_per_lag(obs)),
    }

    c = report["2.1_channel_composition"]
    print("=== 2.1 channel composition (lag 12, task 0) ===")
    print(f"{'γ':>6} {'gain':>7} {'loss':>7} | {'gain signed shares u/r/d':>26} | "
          f"{'loss signed shares u/r/d':>26}")
    for g, d in c.items():
        gs = " ".join(f"{d['gain_signed_shares'][f]:+.3f}" for f in FACTORS)
        ls = " ".join(f"{d['loss_signed_shares'][f]:+.3f}" for f in FACTORS)
        mark = "  <- radius opposes" if d["radius_opposes_in_gain"] else ""
        print(f"{g:>6} {d['gain']['floors']:>+7.1f} {d['loss']['floors']:>+7.1f} | {gs:>26} | "
              f"{ls:>26}{mark}")

    pc = report["2.1_per_corner_terms"]
    print("\n=== 2.1 per corner: which channels separate the gain from the losses ===")
    for g, d in pc.items():
        if not any(v["resolved"] for v in d["corners"].values()):
            continue
        flags = ("radius negative in every resolvable corner"
                 if d["radius_negative_everywhere"] else "radius signs differ")
        print(f"  γ = {g}: {flags}; utility splits by sign: "
              f"{d['utility_and_dimension_split_by_sign']}")
        for c, v in d["corners"].items():
            if v["resolved"]:
                t = " ".join(f"{f[:3]} {v['terms'][f]:+.4f}" for f in FACTORS)
                print(f"      {c}: {v['floors']:+6.1f} fl  {t}")

    print("\n=== 2.2 generality ===")
    for g, d in report["2.2_generality"].items():
        print(f"\nγ = {g}: verdict **{d['verdict']}** "
              f"({d['n_positions_resolved']}/{d['n_positions']} positions at lag 4, "
              f"{d['n_lags_resolved']}/{d['n_lags']} lags within task 0, "
              f"{d['n_cells_resolved']}/{d['n_cells']} cells overall)")
        for t in MATCHED_POSITIONS:
            v = d["position_at_matched_lag_4"][str(t)]
            print(f"    task {t:>2} lag 4: {v['floors']:+7.1f} floors  "
                  f"CI [{v['ci_floors'][0]:+.1f}, {v['ci_floors'][1]:+.1f}]  "
                  f"{'resolved gain' if v['resolved_gain'] else 'unresolved'}")
        for lg in LAG_SERIES:
            v = d["lag_within_task_0"][str(lg)]
            print(f"    task  0 lag {lg:>2}: {v['floors']:+7.1f} floors  "
                  f"CI [{v['ci_floors'][0]:+.1f}, {v['ci_floors'][1]:+.1f}]  "
                  f"{'resolved gain' if v['resolved_gain'] else 'unresolved'}")

    p = report["2.3_peak_location"]
    print("\n=== 2.3 peak location ===")
    print(f"  which sampled γ is highest: {p['on_grid_argmax']:g}, "
          f"{p['modal_share']:.1%} of bootstrap draws, spanning {p['grid_steps_spanned']} step(s)")
    print(f"  argmax distribution: {p['argmax_distribution']}")
    print(f"  BUT the grid steps by ×{p['grid_step_factor']:.2f}, so the continuous peak is "
          f"bracketed only by {p['neighbour_bracket']}")
    print(f"  interpolated peak (parabola in log γ): {p['interpolated_peak']:.3f}, "
          f"95% CI [{p['interpolated_peak_ci'][0]:.3f}, {p['interpolated_peak_ci'][1]:.3f}] "
          f"= {p['interpolated_peak_ci_width_in_steps']:.2f} grid steps wide")
    s = p["model_sensitivity"]
    print("  but that CI is conditional on the functional form. Refitting:")
    for k, v in s["vertex_by_model"].items():
        print(f"    γ = {v:6.3f}   {k}")
    print(f"  spread across functional forms: {s['spread_in_grid_steps']:.2f} grid steps "
          f"({'DOMINATES' if p['model_error_dominates'] else 'below'} the sampling CI)")
    ck = p["checks"]
    pb = ck["paired_bootstrap"]
    print(f"  paired bootstrap over {pb['n_units']} (seed, stream) units: "
          f"CI [{pb['ci'][0]:.3f}, {pb['ci'][1]:.3f}] "
          f"= {pb['ci_width_in_steps']:.2f} grid steps")
    print("  peak by lag (task 0): " + ", ".join(
        f"lag {k} → {v['interpolated_peak']:.2f}" if v["interpolated_peak"]
        else f"lag {k} → argmax {v['on_grid_argmax']:g} (at an endpoint)"
        for k, v in ck["per_lag"].items()))
    print(f"  peak moves monotonically with lag: {ck['peak_moves_monotonically_with_lag']}; "
          f"span {ck['peak_span_in_grid_steps']:.2f} grid steps over "
          f"[{ck['peak_range_across_lags'][0]:.2f}, {ck['peak_range_across_lags'][1]:.2f}]")
    print(f"  peak is a property of the corner (rather than of the lag it is read at): "
          f"{ck['peak_is_a_property_of_the_corner']}")
    print(f"  location identified at lag 12: {p['location_identified_at_lag_12']}; "
          f"as a property of the corner: {p['location_identified']}")

    OUT.write_text(json.dumps(report, indent=1, default=float))
    print(f"\nwrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
