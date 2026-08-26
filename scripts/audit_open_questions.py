"""Answer the resolution, gate and sign-mixture questions in `docs/11-role-and-open-questions.md`.

Five questions that need computation rather than reading, all from stored geometry:

    Q4  every reported number ranked by how far it sits above its own resolution floor
    Q5  every gate in the analysis, its threshold, and the fraction it excludes
    Q6  the sign-mixture audit, run on the three terms separately and on rho_c and the probe
        margin -- the four-corner pooling hid a gain in `Delta log alpha`, and the same check was
        never run on anything else we pool
    Q8  the between-richness / within-richness split, for every pair of measures rather than the
        one pair it was noticed on
    Q9  whether the saturation that breaks additivity at gamma=30 is visible in anything but rho_c

Writes `results/open_questions.json`. Measures nothing new: every number is re-derived from the
arms already on disk.

    python scripts/audit_open_questions.py
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.analysis import attribution as A  # noqa: E402
from src.analysis import grid as G  # noqa: E402

BENIGN = "S-HH"
FORGETTING = ("S-HL", "S-LH", "S-LL")
RESOLVED = 2.0  # floors; below this a value is not distinguishable from measurement noise

# Attribution term -> the measured channel whose floor denominates it. `G.floors` refuses an
# unknown name rather than borrowing a neighbour's floor, which is why this map is explicit.
TERM_FLOOR = {"alpha": "alpha", "utility": "Psi_eff", "radius": "R_eff", "dimension": "D_eff"}


def probe_margin(r: dict) -> float | None:
    """Mean probe margin for one arm.

    Lives under `manipulation_checks[*]["probe_decodability"][module]["margin"]`, not under a
    top-level `probe` key -- that key exists but is null on every arm in the registered grid, so
    reading it silently yields nothing rather than failing.
    """
    vals = []
    for chk in r.get("manipulation_checks") or []:
        for _mod, d in (chk.get("probe_decodability") or {}).items():
            if isinstance(d, dict) and d.get("margin") is not None:
                vals.append(float(d["margin"]))
    return float(np.mean(vals)) if vals else None


def delta_rho_c(r: dict, ensemble: str) -> list[float]:
    """Per-(module, task) change in rho_c across the stream, for one arm.

    First to last boundary, which is the quantity Figure 4 reports by corner.
    """
    per = defaultdict(dict)
    for b in r.get("geometry", []):
        if b.get("ensemble") != ensemble or b.get("rho_c_signed") is None:
            continue
        per[(b.get("module"), b.get("task"))][b.get("boundary")] = float(b["rho_c_signed"])
    out = []
    for series in per.values():
        bs = sorted(k for k in series if k is not None)
        if len(bs) >= 2:
            out.append(series[bs[-1]] - series[bs[0]])
    return out


def arm_scalars(r: dict) -> dict:
    """The per-arm summary quantities that Q6 and Q8 both need."""
    blocks = r.get("geometry", [])

    def mean(ensemble, field):
        v = [float(b[field]) for b in blocks
             if b.get("ensemble") == ensemble and b.get(field) is not None]
        return float(np.mean(v)) if v else np.nan

    out = {"generic_alpha": mean("generic", "alpha"),
           "retained_alpha": mean("retained", "alpha"),
           "generic_D_eff": mean("generic", "D_eff"),
           "generic_R_eff": mean("generic", "R_eff"),
           "generic_Psi_eff": mean("generic", "Psi_eff"),
           "generic_rho_c": mean("generic", "rho_c_signed")}
    pm = probe_margin(r)
    if pm is not None:
        out["probe_margin"] = pm
    f = r.get("forgetting")
    if isinstance(f, dict) and f.get("CFr") is not None:
        out["forgetting_CFr"] = float(np.mean(np.atleast_1d(f["CFr"])))
    return out


def _spearman(x: list[float], y: list[float]) -> float:
    if len(x) < 3:
        return float("nan")
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    rx, ry = rx - rx.mean(), ry - ry.mean()
    d = float(np.sqrt((rx ** 2).sum() * (ry ** 2).sum()))
    return float((rx * ry).sum() / d) if d else float("nan")


# ---------------------------------------------------------------------------------------------
# Q4 -- margin over the floor
# ---------------------------------------------------------------------------------------------

def margins(recs: list[dict]) -> dict:
    """Every pooled per-(condition, gamma) quantity, in floors, ranked by |margin|.

    The question is which reported numbers a reviewer can push on. A quantity at 2.1 floors is
    reported as resolvable and is one analysis choice away from not being; one at 69.7 is not.
    """
    per = defaultdict(list)
    for spec, _m, _t, lag, att in G.attributions(recs, module="A"):
        if lag == 12:
            per[(spec["condition"], spec["gamma_0"])].append(att)

    rows = []
    for (cond, g), atts in sorted(per.items()):
        tab = A.attribution_table(atts)
        for channel, key in (("alpha", "dlog_alpha"), ("utility", "utility"),
                             ("radius", "radius"), ("dimension", "dimension")):
            v = tab["dlog_alpha"] if key == "dlog_alpha" else tab["terms"][key]
            fl = G.floors(v, TERM_FLOOR[channel], gamma=g)
            rows.append({"quantity": f"{channel} @ {cond}, gamma={g:g}", "condition": cond,
                         "gamma": g, "channel": channel, "value": v, "floors": fl,
                         "abs_floors": abs(fl), "resolved": abs(fl) >= RESOLVED,
                         "n": len(atts)})

    # The pooled three-corner series, which is what the paper actually quotes.
    pooled = defaultdict(list)
    for spec, _m, _t, lag, att in G.attributions(recs, module="A"):
        if lag == 12 and spec["condition"] in FORGETTING:
            pooled[spec["gamma_0"]].append(att)
    for g, atts in sorted(pooled.items()):
        tab = A.attribution_table(atts)
        fl = G.floors(tab["dlog_alpha"], "alpha", gamma=g)
        rows.append({"quantity": f"alpha @ three forgetting corners, gamma={g:g}",
                     "condition": "forgetting", "gamma": g, "channel": "alpha",
                     "value": tab["dlog_alpha"], "floors": fl, "abs_floors": abs(fl),
                     "resolved": abs(fl) >= RESOLVED, "n": len(atts)})

    rows.sort(key=lambda r: r["abs_floors"])
    near = [r for r in rows if r["resolved"] and r["abs_floors"] < 3 * RESOLVED]
    return {"all": rows, "n_total": len(rows),
            "n_resolved": sum(r["resolved"] for r in rows),
            "n_within_3x_of_gate": len(near),
            "closest_reported": [r["quantity"] for r in near[:12]]}


# ---------------------------------------------------------------------------------------------
# Q5 -- every gate and what it excludes
# ---------------------------------------------------------------------------------------------

def gates(recs: list[dict], width: list[dict], ext: list[dict]) -> dict:
    """Each gate, its threshold, and the fraction of candidate cells it removes."""
    out = []

    # 1. The 2-floor resolution gate on the magnitude panel.
    per = defaultdict(list)
    for spec, _m, _t, lag, att in G.attributions(recs, module="A"):
        if lag == 12:
            per[(spec["condition"], spec["gamma_0"])].append(att)
    tot = kept = 0
    for (_c, g), atts in per.items():
        tab = A.attribution_table(atts)
        tot += 1
        kept += abs(G.floors(tab["dlog_alpha"], "alpha", gamma=g)) >= RESOLVED
    out.append({"gate": "resolution gate on |Delta log alpha|", "threshold": f"{RESOLVED} floors",
                "applies_to": "every (condition, gamma) cell at lag 12", "candidates": tot,
                "kept": kept, "excluded": tot - kept,
                "excluded_frac": round((tot - kept) / tot, 4) if tot else None,
                "what_is_excluded": "the lazy end, gamma <= 0.1, where nothing moves either way"})

    # 2. The R_eff floor gate behind panel (d), which is gamma-dependent by design.
    for g in sorted({s["gamma_0"] for s, *_ in G.attributions(recs, module="A")}):
        cand = adm = 0
        for spec, _m, _t, lag, att in G.attributions(recs, module="A"):
            if lag != 12 or spec["gamma_0"] != g:
                continue
            cand += 1
            adm += abs(att.terms["radius"]) >= A.min_dlog_R_for(g)
        out.append({"gate": "R_eff floor gate (panel d)",
                    "threshold": f"{A.min_dlog_R_for(g):.5f} in |Delta log R| "
                                 f"({A.FLOOR_GATE_K}x the measured floor at this richness)",
                    "applies_to": f"per-arm radius terms at gamma={g:g}", "candidates": cand,
                    "kept": adm, "excluded": cand - adm,
                    "excluded_frac": round((cand - adm) / cand, 4) if cand else None,
                    "what_is_excluded": "arms whose radius barely moved, where the ratio would be "
                                        "noise over noise"})

    # 3. The rho_c calibration domain.
    inside = outside = 0
    for r in recs:
        for blk in r.get("geometry", []):
            v = blk.get("rho_c_signed")
            if v is None:
                continue
            inside += A.rho_in_domain(float(v))
            outside += not A.rho_in_domain(float(v))
    tot = inside + outside
    out.append({"gate": "rho_c calibration domain", "threshold": f"tol {A.RHO_DOMAIN_TOL}",
                "applies_to": "every recorded rho_c used for center-collapse conversion",
                "candidates": tot, "kept": inside, "excluded": outside,
                "excluded_frac": round(outside / tot, 6) if tot else None,
                "what_is_excluded": "values outside the range the synthetic calibration was "
                                    "fitted on; these are refused, not clipped"})

    # 4. Arm usability, across all three result sets.
    for label, rs in (("registered grid", recs), ("width arms", width), ("gamma=30 probe", ext)):
        if not rs:
            continue
        usable = sum(bool(r.get("usable", True)) for r in rs)
        out.append({"gate": "arm usability", "threshold": "convergence + artifact checks",
                    "applies_to": label, "candidates": len(rs), "kept": usable,
                    "excluded": len(rs) - usable,
                    "excluded_frac": round((len(rs) - usable) / len(rs), 4),
                    "what_is_excluded": "arms failing a manipulation or convergence check"})
    return {"gates": out}


# ---------------------------------------------------------------------------------------------
# Q6 -- sign mixture in everything we pool, not just Delta log alpha
# ---------------------------------------------------------------------------------------------

def sign_mixture(recs: list[dict], width: list[dict], ext: list[dict]) -> dict:
    """Does any pooled quantity average values of opposite sign?

    The four-corner pooling averaged a gain against three losses and the mean concealed it. That
    check was run on `Delta log alpha` over 520 cells. It was never run on the three terms
    separately, on rho_c, or on the probe margin -- all of which we also pool.
    """
    findings = []

    def scan(name: str, cells: dict, pooled_over: str):
        """cells: label -> list of per-arm values that get averaged into one reported number."""
        for label, vals in sorted(cells.items()):
            v = np.asarray([x for x in vals if np.isfinite(x)], float)
            if v.size < 4:
                continue
            pos, neg = int((v > 0).sum()), int((v < 0).sum())
            if pos and neg:
                minority = min(pos, neg) / v.size
                # A mixture only matters if the minority is big enough to move the mean, or if
                # the mean is small relative to the spread it is averaging.
                mean, sd = float(v.mean()), float(v.std(ddof=1))
                findings.append({
                    "quantity": name, "cell": label, "pooled_over": pooled_over,
                    "n": int(v.size), "n_positive": pos, "n_negative": neg,
                    "minority_frac": round(minority, 4), "mean": mean,
                    "sd": sd, "mean_over_sd": round(mean / sd, 4) if sd else None,
                    "material": bool(minority >= 0.10 and abs(mean) < 2 * sd)})

    # (a) the three terms, pooled the way the paper pools them
    for term in A.FACTORS:
        cells = defaultdict(list)
        for spec, _m, _t, lag, att in G.attributions(recs, module="A"):
            if lag == 12 and spec["condition"] in FORGETTING:
                cells[f"three corners, gamma={spec['gamma_0']:g}"].append(att.terms[term])
        scan(f"Delta log {term}", cells, "three forgetting corners x 40 arms")

    # (b) the same terms inside the benign corner
    for term in A.FACTORS:
        cells = defaultdict(list)
        for spec, _m, _t, lag, att in G.attributions(recs, module="A"):
            if lag == 12 and spec["condition"] == BENIGN:
                cells[f"{BENIGN}, gamma={spec['gamma_0']:g}"].append(att.terms[term])
        scan(f"Delta log {term} ({BENIGN})", cells, "benign corner, 40 arms")

    # (c) rho_c endpoint change, per corner -- Figure 4's quantity
    for rs, tag in ((recs, "registered grid"), (ext, "gamma=30 probe"),
                    (width, "width arms")):
        if not rs:
            continue
        for ensemble in ("generic", "retained"):
            cells = defaultdict(list)
            for r in rs:
                s = r["spec"]
                for d in delta_rho_c(r, ensemble):
                    cells[f"{s['condition']}, gamma={s['gamma_0']:g} ({tag})"].append(d)
            scan(f"Delta rho_c [{ensemble}]", cells, f"per corner x richness, {tag}")

    # (d) the probe margin, pooled across richness -- the H2d quantity
    cells = defaultdict(list)
    for r in recs:
        pm = probe_margin(r)
        if pm is not None:
            cells[f"margin, gamma={r['spec']['gamma_0']:g}"].append(pm)
    scan("probe margin", cells, "all arms at a richness")

    return {"checked": ["Delta log utility", "Delta log radius", "Delta log dimension",
                        "Delta rho_c (generic and retained)", "probe margin"],
            "n_mixtures_found": len(findings),
            "n_material": sum(f["material"] for f in findings),
            "findings": sorted(findings, key=lambda f: -f["minority_frac"])}


# ---------------------------------------------------------------------------------------------
# Q8 -- between-richness vs within-richness, for every pair of measures
# ---------------------------------------------------------------------------------------------

def between_within(recs: list[dict]) -> dict:
    """Does any other pair of measures reverse between richness levels the way H2d's does?

    The reported instance is generic capacity against probe margin: positively correlated inside a
    richness level, oppositely signed across them. That is Simpson's paradox, and one instance is
    an anecdote. This runs the same split on every available pair.
    """
    per_arm: dict[tuple, dict] = {}
    for r in recs:
        s = r["spec"]
        per_arm[(s["gamma_0"], s["condition"], s["stream_id"], s["seed"])] = arm_scalars(r)

    names = sorted({k for v in per_arm.values() for k in v})
    rows = []
    for a, b in combinations(names, 2):
        xs = [(k[0], v[a], v[b]) for k, v in per_arm.items()
              if a in v and b in v and np.isfinite(v[a]) and np.isfinite(v[b])]
        if len(xs) < 20:
            continue
        # between: correlate the per-richness means
        by_g = defaultdict(list)
        for g, x, y in xs:
            by_g[g].append((x, y))
        gs = sorted(by_g)
        between = _spearman([float(np.mean([p[0] for p in by_g[g]])) for g in gs],
                            [float(np.mean([p[1] for p in by_g[g]])) for g in gs])
        withins = [_spearman([p[0] for p in by_g[g]], [p[1] for p in by_g[g]]) for g in gs]
        wmean = float(np.nanmean(withins))
        rows.append({"pair": f"{a} vs {b}", "n_arms": len(xs), "n_richness_levels": len(gs),
                     "between_richness_spearman": round(float(between), 4),
                     "within_richness_spearman_mean": round(wmean, 4),
                     "within_by_gamma": {f"{g:g}": round(float(w), 4)
                                         for g, w in zip(gs, withins)},
                     "reverses": bool(np.isfinite(between) and np.isfinite(wmean)
                                      and between * wmean < 0 and abs(between) > 0.5
                                      and abs(wmean) > 0.1)})
    rows.sort(key=lambda r: -abs(r["between_richness_spearman"]
                                 - r["within_richness_spearman_mean"]))
    return {"pairs": rows, "n_pairs": len(rows),
            "n_reversing": sum(r["reverses"] for r in rows),
            "reversing": [r["pair"] for r in rows if r["reverses"]]}


# ---------------------------------------------------------------------------------------------
# Q9 -- is the gamma=30 saturation specific to rho_c?
# ---------------------------------------------------------------------------------------------

def saturation(recs: list[dict], ext: list[dict]) -> dict:
    """Compare the 3->10 step with the 10->30 step for every quantity available at all three.

    Additivity holds over the registered range and fails at gamma=30 because the main effects
    saturate while the interaction does not. If other quantities flatten over the same step, the
    saturation is a property of the richness range; if only rho_c does, it is specific.
    """
    def pooled(rs, g, getter):
        vals = []
        for r in rs:
            if r["spec"]["gamma_0"] != g:
                continue
            v = getter(r)
            if v is not None and np.isfinite(v):
                vals.append(float(v))
        return float(np.mean(vals)) if vals else None

    def ens(ensemble, field):
        def f(r):
            b = [x[field] for x in r.get("geometry", [])
                 if x.get("ensemble") == ensemble and x.get(field) is not None]
            return float(np.mean(b)) if b else None
        return f

    quantities = {
        "generic alpha": ens("generic", "alpha"),
        "retained alpha": ens("retained", "alpha"),
        "generic D_eff": ens("generic", "D_eff"),
        "generic R_eff": ens("generic", "R_eff"),
        "generic rho_c": ens("generic", "rho_c_signed"),
    }
    rows = []
    for name, getter in quantities.items():
        v3, v10 = pooled(recs, 3.0, getter), pooled(recs, 10.0, getter)
        v30 = pooled(ext, 30.0, getter)
        if None in (v3, v10, v30):
            rows.append({"quantity": name, "note": "not available at all three richnesses"})
            continue
        s1, s2 = v10 - v3, v30 - v10
        rows.append({"quantity": name, "at_gamma_3": v3, "at_gamma_10": v10, "at_gamma_30": v30,
                     "step_3_to_10": s1, "step_10_to_30": s2,
                     "ratio_second_to_first": round(s2 / s1, 4) if s1 else None,
                     "saturating": bool(s1 and abs(s2) < 0.5 * abs(s1))})
    return {"quantities": rows,
            "n_saturating": sum(bool(r.get("saturating")) for r in rows),
            "saturating": [r["quantity"] for r in rows if r.get("saturating")]}


# ---------------------------------------------------------------------------------------------
# Q10 -- what richness values exist anywhere in the record
# ---------------------------------------------------------------------------------------------

def richness_coverage(recs: list[dict], width: list[dict], ext: list[dict]) -> dict:
    """Is there anything between gamma=3 and gamma=10 in any result set?"""
    seen = {}
    for label, rs in (("registered grid", recs), ("width arms", width),
                      ("gamma=30 probe", ext)):
        seen[label] = sorted({r["spec"]["gamma_0"] for r in rs})
    everything = sorted({g for v in seen.values() for g in v})
    return {"by_result_set": seen, "all_richness_values": everything,
            "between_3_and_10": [g for g in everything if 3 < g < 10],
            "answer": "nothing in the record lies strictly between gamma=3 and gamma=10"
                      if not [g for g in everything if 3 < g < 10] else "see between_3_and_10"}


def main() -> None:
    recs = G.load()
    width = G.load(arms_dir=ROOT / "results" / "width")
    ext = G.load(arms_dir=ROOT / "results" / "gamma_ext")
    print(f"loaded {len(recs)} grid arms, {len(width)} width arms, {len(ext)} gamma=30 arms")

    out = {
        "generated_by": "scripts/audit_open_questions.py",
        "q4_margins": margins(recs),
        "q5_gates": gates(recs, width, ext),
        "q6_sign_mixture": sign_mixture(recs, width, ext),
        "q8_between_within": between_within(recs),
        "q9_saturation": saturation(recs, ext),
        "q10_richness_coverage": richness_coverage(recs, width, ext),
    }
    p = ROOT / "results" / "open_questions.json"
    p.write_text(json.dumps(out, indent=2, default=float) + "\n")

    m = out["q4_margins"]
    print(f"\nQ4  {m['n_resolved']}/{m['n_total']} pooled quantities resolved; "
          f"{m['n_within_3x_of_gate']} sit within 3x of the gate")
    print(f"Q5  {len(out['q5_gates']['gates'])} gates catalogued")
    s = out["q6_sign_mixture"]
    print(f"Q6  {s['n_mixtures_found']} sign mixtures, {s['n_material']} material")
    b = out["q8_between_within"]
    print(f"Q8  {b['n_pairs']} measure pairs, {b['n_reversing']} reverse between/within: "
          f"{b['reversing']}")
    print(f"Q9  saturating over 10->30: {out['q9_saturation']['saturating']}")
    print(f"Q10 {out['q10_richness_coverage']['answer']}")
    print(f"\nwrote {p.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
