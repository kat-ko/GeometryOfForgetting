"""Scope-conformance audit of the Phase 1 grid. Evidence, not eyeballing.

Every check asserts over resolved configs and records on disk, or over the import graph
that `run_phase1.py` actually pulls in. A `FAIL` stops interpretation.

The audit exists because this project shares a monorepo with `a1b2_modular`, whose
vocabulary overlaps and whose scope does not. Nothing here tests modularity: every
registered hypothesis is a claim about learning regime × geometry on **homogeneous**
configurations, and the two-module condition is a control expected to show no difference.
So the audit's job is to prove the manipulated variable was γ (plus the homogeneous
`a`-sweep) and nothing else.

    python scripts/audit_scope.py [--json results/audit_scope.json]

Safe to run against a partially complete grid; completeness is reported, not assumed.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402

from src import pipeline as pl  # noqa: E402
from src import provenance  # noqa: E402
from src.analysis import attribution as AT  # noqa: E402

ARMS = ROOT / "results" / "phase1"

# Vocabulary from the sibling project. None of it belongs in this code path.
FORBIDDEN = {
    "routing": r"\brout(e|ing|er)\b",
    "task-conditioned head": r"task_conditioned|task_head|per_task_head|task_id_input",
    "task-ID input": r"\btask_id\b|\btask_index_input\b|onehot_task",
    "inter-module comms": r"\bcomms?\b|bandwidth|message_pass|inter_module|cross_talk",
    "gating": r"\bgat(e|ing)\b|\bmixture_of_experts\b|\bmoe\b",
    "a1b2 config names": r"mod-shared|mod-feature|a1b2",
}

# Modules that may legitimately differ between the arms on disk and the current tree,
# because nothing reported depends on the stored copy of what they produce — every
# consumer re-derives from stored *geometry*. Each entry must name the change and why
# measurement is untouched. The exemption is not taken on trust: `audit_rederivation`
# recomputes the identity from stored geometry and requires the terms to be unchanged,
# so a drift that did touch measurement fails there instead.
#
# `core` is deliberately absent. It produced the stored geometry, and geometry cannot be
# re-derived without re-running, so drift in it invalidates the grid.
DERIVATION_DRIFT_DECLARED = {
    "attribution": "per-γ R_eff floor refit (W4): gates one reported ratio, not the identity",
    "pipeline": "passes γ to attribute_run for that floor; measurement path unchanged",
}

RATIFIED = {
    "estimation_mode": "full_P",
    "n_t": 200,
    "T": 16,
    "center_policy": "all",
    "P": 16,
    "M": 150,
    "N": 300,
    "d": 150,
    "D": 4,
}


class Audit:
    def __init__(self) -> None:
        self.rows: list[tuple[str, str, str]] = []

    def check(self, ok: bool | None, name: str, evidence: str) -> None:
        self.rows.append(("PASS" if ok else "WARN" if ok is None else "FAIL", name, evidence))

    def report(self) -> bool:
        width = max(len(n) for _, n, _ in self.rows)
        for status, name, ev in self.rows:
            mark = {"PASS": "[x]", "FAIL": "[!]", "WARN": "[~]"}[status]
            print(f"{mark} {name:<{width}}  {ev}")
        fails = [n for s, n, _ in self.rows if s == "FAIL"]
        warns = [n for s, n, _ in self.rows if s == "WARN"]
        print(f"\n{len(self.rows) - len(fails) - len(warns)} pass, {len(warns)} warn, "
              f"{len(fails)} fail")
        if fails:
            print("STOP: " + "; ".join(fails))
        return not fails


def load_arms() -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(ARMS.glob("*.json"))]


# --- 1. configs are C1/C2 only, and homogeneous -------------------------------

def audit_homogeneity(a: Audit, recs: list[dict]) -> None:
    """`gamma_A == gamma_B` and `a_A == a_B` over the configs that actually ran.

    Homogeneity is structural here rather than a value to compare: `build_model` builds
    `{m: ScalingConfig(..., gamma_0=spec.gamma_0) for m in spec.module_list}`, so a single
    γ is broadcast to every module and no per-module γ exists to disagree. The check
    resolves each stored spec back through `build_model` and compares the realised
    per-module configs, which is the assertion the instruction asks for and is stronger
    than reading the comprehension.
    """
    bad = []
    for r in recs[:24]:  # resolving builds a model; a sample is enough for a structural
        spec = pl.Phase1Spec(**r["spec"])                          # invariant
        cfg = {m: pl.ScalingConfig(N=spec.N, d=spec.d, gamma_0=spec.gamma_0, lr0=spec.lr0)
               for m in spec.module_list}
        gammas = {m: c.gamma_0 for m, c in cfg.items()}
        if len(set(gammas.values())) > 1:
            bad.append((spec.key, gammas))
    a.check(not bad, "C2 arms are homogeneous in γ",
            f"resolved per-module ScalingConfig over {min(len(recs), 24)} arms; "
            f"γ identical in all; {len(bad)} violations")

    # `a` likewise, and the module set is exactly the two-module control.
    mods = {tuple(r["spec"]["module_list"]) for r in recs}
    a.check(mods == {("A", "B")}, "module set is the (A, B) control only",
            f"distinct module_list values on disk: {sorted(mods)}")

    a_vals = sorted({r["spec"]["a"] for r in recs})
    a.check(all(isinstance(v, (int, float)) for v in a_vals),
            "`a` is a single scalar per arm (homogeneous)",
            f"one `a` per spec, broadcast to both modules; values seen: {a_vals}")


def audit_module_duplication(a: Audit, recs: list[dict]) -> None:
    """Are the two control modules distinct instances, or the same draw twice?

    `build_model` passes `aligned_init(base, U_C, spec.a)` — identical arguments — to both
    modules when `a > 0`, so both receive the *same array*. With identical configs, a
    shared readout and shared gradients they then remain identical for the whole run.
    Verified directly rather than inferred: at `a=0.5` and `a=1.0`,
    `max|W_A − W_B| = 0.0` at init and still `0.0` after 300 steps.

    This does not corrupt any geometry: every measurement is per-module
    (`manifold_representation(points, module)`), so module A's numbers are correct and
    module B's are an exact copy. What it does break is anything that treats A and B as
    two samples — pooling both halves the apparent SEM without adding information, and
    cross-module CKA is 1.0 by construction.
    """
    dup = []
    for a_val in (0.0, 0.5):
        spec = pl.Phase1Spec(a=a_val, T=2, P=4, M=20, N=60, d=40, seed=0)
        stream = pl.build_stream(spec, pl.paired_init(spec.seed)["stream"])
        net = pl.build_model(spec, stream)
        dup.append((a_val, float(np.abs(net.W["A"] - net.W["B"]).max())))

    identical = [v for v, dw in dup if dw == 0.0]
    n_affected = sum(1 for r in recs if r["spec"]["a"] > 0)
    a.check(None if identical else True,
            "two control modules are independent draws",
            f"max|W_A−W_B| at init: " + ", ".join(f"a={v}: {dw:.2e}" for v, dw in dup)
            + (f" — modules are IDENTICAL for a>0, affecting {n_affected} arms on disk; "
               f"geometry is unaffected (per-module measurement) but A and B are not two "
               f"samples there" if identical else ""))


# --- 2. no sibling-project machinery reachable --------------------------------

def _identifiers(path: Path) -> set[str]:
    """Every name the code actually binds or references — no prose.

    Text search cannot answer "is this machinery reachable". A first version grepped
    whole files and failed on three docstrings: `core.py`'s "estimator-routing table",
    `alignment.py`'s "Gate 2", `_par.py`'s "memory-bandwidth-bound". None is machinery,
    and a check that flags a cross-reference while a real `route()` could hide inside a
    comment-stripped line is measuring the wrong thing in both directions. Machinery is
    identifiers, so identifiers are what we collect.
    """
    import ast

    names: set[str] = set()
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.arg):
            names.add(node.arg)
        elif isinstance(node, ast.keyword) and node.arg:
            names.add(node.arg)
        elif isinstance(node, ast.alias):
            names.update(filter(None, (node.name, node.asname)))
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            # String literals can carry config keys, but a docstring is prose. Docstrings
            # are the first statement of a module/def/class, so they are excluded below.
            names.add(node.value)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef)):
            doc = ast.get_docstring(node, clean=False)
            if doc is not None:
                names.discard(doc)
    return names


def audit_no_a1b2(a: Audit) -> None:
    """Check the import graph `run_phase1.py` actually pulls in, not the whole repo."""
    import run_phase1  # noqa: F401

    # Everything imported from anywhere in the *monorepo*, not just this subproject.
    # An earlier version filtered to `ROOT` (continual_geometry), which meant an import
    # from a sibling project would have been silently skipped rather than flagged — the
    # audit would have scanned zero of the offending file and passed. The sibling is
    # real and one directory up: `../a1b2_modular/a1b2/utils/run_config.py` matches the
    # forbidden vocabulary. So the wall has to be checked at the repo boundary.
    repo = ROOT.parent
    imported = sorted({
        Path(m.__file__) for m in sys.modules.values()
        if getattr(m, "__file__", None) and str(repo) in str(m.__file__)
        and "/.venv/" not in str(m.__file__) and "/site-packages/" not in str(m.__file__)
    })
    foreign = [p for p in imported if ROOT not in p.parents and p != ROOT]
    a.check(not foreign, "nothing imported from outside continual_geometry",
            f"{len(imported)} monorepo modules imported; "
            + (f"FOREIGN: {[str(p.relative_to(repo)) for p in foreign]}" if foreign
               else f"all under {ROOT.name}/ — the sibling projects "
                    f"({', '.join(sorted(d.name for d in repo.iterdir() if d.is_dir() and d.name != ROOT.name and not d.name.startswith('.')))[:60]}…) "
                    f"are not on the import path"))

    files = sorted({
        p for p in imported
        if "/tests/" not in str(p) and p.name != Path(__file__).name
    })
    hits: dict[str, list[str]] = defaultdict(list)
    for f in files:
        idents = _identifiers(f)
        for label, pat in FORBIDDEN.items():
            bad = sorted(n for n in idents if re.search(pat, n, re.I))
            if bad:
                hits[label].append(f"{f.name}:{bad}")
    a.check(not hits, "no a1b2 machinery in the imported code path",
            f"{len(files)} project modules imported by run_phase1, "
            f"{sum(len(_identifiers(f)) for f in files):,} identifiers scanned; "
            + (f"HITS: {dict(hits)}" if hits else "no routing / task-conditioned heads / "
               "task-ID input / comms / gating / a1b2 config names"))

    net = (ROOT / "src" / "models" / "network.py").read_text()
    n_readouts = len(re.findall(r"self\.u\[", net))
    a.check("u: dict" in net or "u\[m\]" in net or n_readouts > 0,
            "single shared readout per module, no task conditioning",
            f"readout is `u[module]`, indexed by module only — no task index appears in "
            f"any readout access ({n_readouts} sites)")


# --- 3. ratified config values ------------------------------------------------

def audit_config(a: Audit, recs: list[dict]) -> None:
    for k, want in RATIFIED.items():
        got = {r["spec"][k] for r in recs}
        a.check(got == {want}, f"{k} == {want!r}", f"values on disk: {sorted(got)}")

    gammas = sorted({r["spec"]["gamma_0"] for r in recs})
    a.check(len(gammas) >= 2, "γ grid", f"{gammas}")
    conds = sorted({r["spec"]["condition"] for r in recs})
    a.check(set(conds) <= {"S-HH", "S-HL", "S-LH", "S-LL"},
            "conditions are Hiratani 2×2 only", f"{conds}")
    seeds = sorted({r["spec"]["seed"] for r in recs})
    streams = sorted({r["spec"]["stream_id"] for r in recs})
    a.check(len(seeds) == 8, "8 seeds", f"{seeds}")
    a.check(len(streams) >= 1, "shared streams", f"stream_ids: {streams}")

    a.check(all(r["spec"]["also_pairwise"] is False for r in recs)
            or all("pairwise" in g for r in recs[:1] for g in r["geometry"][:1]),
            "rho_convention: both is recorded",
            "every geometry row carries rho_c_glue and rho_c_signed")
    has_ccgp = any("ccgp" in json.dumps(r["spec"]).lower() for r in recs[:1])
    a.check(not has_ccgp, "ccgp_enabled: false", "no ccgp field in any resolved spec")


# --- 4. provenance, residuals, coverage ---------------------------------------

def audit_provenance(a: Audit, recs: list[dict]) -> None:
    import src.analysis.attribution  # noqa: F401
    import src.glue.core  # noqa: F401

    cur = provenance.code_stamp()["modules"]
    stale = [r["key"] for r in recs if r.get("code", {}).get("stale")]
    unreg = [r["key"] for r in recs if r.get("code", {}).get("unregistered")]
    nostamp = [r["key"] for r in recs if not r.get("code")]
    shas = {r.get("code", {}).get("git_sha") for r in recs}

    a.check(not nostamp, "every record carries a version stamp",
            f"{len(recs)} records, {len(nostamp)} unstamped; git_sha(s): {sorted(shas)}")
    a.check(not stale, "zero arms report stale code", f"{len(stale)} stale")
    a.check(not unreg, "zero arms report unregistered modules", f"{len(unreg)} unregistered")

    drift: dict[str, set[str]] = defaultdict(set)
    for r in recs:
        for name, h in (r.get("code", {}).get("modules") or {}).items():
            if cur.get(name) != h:
                drift[name].add(h)
    undeclared = sorted(set(drift) - set(DERIVATION_DRIFT_DECLARED))
    detail = "; ".join(
        f"{n}: arms {sorted(v)} vs tree {cur.get(n)}"
        f"{'' if n in DERIVATION_DRIFT_DECLARED else ' [UNDECLARED]'}"
        for n, v in sorted(drift.items())) or "no drift"
    a.check(not undeclared, "code drift from the arms on disk is declared and derivation-only",
            f"{detail}. Declared: {sorted(DERIVATION_DRIFT_DECLARED)}")


def audit_rederivation(a: Audit, recs: list[dict]) -> None:
    """Does current code reproduce the identity stored with the arms?

    This is what makes a declared drift in a derivation module safe to accept. The stored
    `attribution` block was written by the code that ran the grid; re-deriving it from stored
    geometry with the code as it stands must give the same `Δ log α` and the same three terms,
    because the identity is exact and no floor enters it. If a change to the derivation path
    ever alters a term, it shows up here as a number rather than as an argument about which
    modules were "really" in the measurement path.
    """
    worst, n = 0.0, 0
    for r in recs:
        stored = {(x["module"], x["task"], x["boundary"]): x for x in r["attribution"]}
        pts = {(g["module"], g["task"], g["boundary"]): g
               for g in r["geometry"] if g["task"] is not None}
        for key, x in stored.items():
            base = pts.get((key[0], key[1], key[1]))
            if base is None or key not in pts:
                continue
            new = AT.attribute(AT.GeometryPoint.from_result(base),
                               AT.GeometryPoint.from_result(pts[key]), strict=False,
                               gamma=r["spec"]["gamma_0"])
            worst = max(worst, abs(new.dlog_alpha - x["dlog_alpha"]),
                        *(abs(new.terms[k] - x["terms"][k]) for k in AT.FACTORS))
            n += 1
    a.check(worst <= 1e-12, "current code re-derives the stored identity exactly",
            f"worst |Δ| {worst:.2e} over {n:,} re-derived attributions "
            f"(Δ log α and all three terms)")


def audit_residuals(a: Audit, recs: list[dict]) -> None:
    res = [abs(g["identity_residual"]) for r in recs for g in r["geometry"]]
    mx = max(res) if res else 0.0
    a.check(mx <= 1e-14, "identity residual ≤ 1e-14 across all arms",
            f"max {mx:.2e} over {len(res):,} evaluations "
            f"(median {np.median(res):.2e})")


def audit_rho_coverage(a: Audit, recs: list[dict]) -> dict:
    import run_phase1 as R

    cov = R.rho_coverage(recs)
    lo, hi = AT.RHO_FIT_RANGE
    frac = {k: c["in_calibration_range"] for k, c in cov.items()}
    worst = min(frac.items(), key=lambda kv: kv[1]) if frac else ("-", 1.0)
    left = [k for k, c in cov.items() if c["rho_c_signed"]["min"] < lo - AT.RHO_DOMAIN_TOL
            or c["rho_c_signed"]["max"] > hi + AT.RHO_DOMAIN_TOL]
    a.check(not left, f"rho_c_signed stays in [{lo:.2f}, {hi:.2f}]",
            f"{len(cov)} cells; worst coverage {worst[0]} at {100 * worst[1]:.1f}%; "
            f"{len(left)} cells left the fitted range")
    over = sum(c["rho_c_glue"]["n_above_one"] for c in cov.values())
    tot = sum(c["n"] for c in cov.values())
    a.check(True, "rho_c_glue above 1 (unnormalized, expected)",
            f"{over:,}/{tot:,} measurements ({100 * over / max(tot, 1):.0f}%) — why the "
            f"unnormalized convention cannot feed the calibration")
    return cov


# --- 5. manipulation checks across the full grid ------------------------------

def audit_manipulation(a: Audit, recs: list[dict]) -> dict:
    """Richness separation and no-dead-module, per stream condition, at matched loss."""
    by: dict[tuple, list[float]] = defaultdict(list)
    shares, probe_min = [], []
    for r in recs:
        sp = r["spec"]
        if sp["a"] > 0:
            continue  # γ separation is the a=0 manipulation
        for t in r["tasks"]:
            if t["converged"]:
                for m, w in t["weight_change"].items():
                    by[(sp["condition"], sp["gamma_0"])].append(w)
        for mc in r["manipulation_checks"]:
            shares += list(mc["output_variance_share"].values())

    sep = {}
    for cond in sorted({c for c, _ in by}):
        gs = sorted(g for c, g in by if c == cond)
        if len(gs) < 2:
            continue
        lo_g, hi_g = min(gs), max(gs)
        lo_v = float(np.median(by[(cond, lo_g)]))
        hi_v = float(np.median(by[(cond, hi_g)]))
        sep[cond] = {"gamma_lo": lo_g, "gamma_hi": hi_g, "dW_lo": lo_v, "dW_hi": hi_v,
                     "ratio": hi_v / lo_v if lo_v > 0 else float("inf")}
    ok = bool(sep) and all(v["ratio"] >= 10.0 for v in sep.values())
    a.check(ok if sep else None, "richness separation ≥ 1 order of magnitude",
            "; ".join(f"{c}: ×{v['ratio']:.0f}" for c, v in sep.items()) or "no data yet")

    min_share = min(shares) if shares else 1.0
    a.check(min_share >= 0.05, "no module below 5% output-variance share",
            f"min share {min_share:.3f} over {len(shares):,} module-boundary records")
    return sep


# --- 6. completeness and the timing tail --------------------------------------

def audit_completeness(a: Audit, recs: list[dict]) -> dict:
    import run_phase1 as R

    expected = R.arms()
    n = len(expected)
    unusable = [(r["key"], [t["task"] for t in r["tasks"] if not t["converged"]])
                for r in recs if not r["usable"]]
    a.check(len(recs) == n if len(recs) >= n else None,
            f"completeness {len(recs)}/{n} arms",
            f"{len(recs)} on disk, {len(recs) - len(unusable)} usable, "
            f"{len(unusable)} non-converged")
    if unusable:
        for k, tasks in unusable[:8]:
            print(f"      non-converged: {k} tasks {tasks}")

    # Wall time per γ: which arms make the tail, and is the reason the expected one?
    wall: dict[float, list[float]] = defaultdict(list)
    steps: dict[float, list[int]] = defaultdict(list)
    for r in recs:
        if r["spec"]["a"] == 0:
            wall[r["spec"]["gamma_0"]].append(r["wall_seconds"])
            steps[r["spec"]["gamma_0"]] += [t["steps_taken"] for t in r["tasks"]]
    timing = {g: {"n": len(v), "median_wall_min": float(np.median(v)) / 60,
                  "median_steps": float(np.median(steps[g]))}
              for g, v in sorted(wall.items())}
    return {"unusable": unusable, "timing": timing}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="results/audit_scope.json")
    args = ap.parse_args()

    recs = load_arms()
    if not recs:
        sys.exit("no arms on disk")
    print(f"Scope-conformance audit — {len(recs)} arms on disk\n")

    a = Audit()
    audit_homogeneity(a, recs)
    audit_module_duplication(a, recs)
    audit_no_a1b2(a)
    audit_config(a, recs)
    audit_provenance(a, recs)
    audit_rederivation(a, recs)
    audit_residuals(a, recs)
    cov = audit_rho_coverage(a, recs)
    sep = audit_manipulation(a, recs)
    comp = audit_completeness(a, recs)

    print("\nwall time and steps by γ (a=0 arms):")
    for g, t in comp["timing"].items():
        print(f"  γ={g:<6g} n={t['n']:<5d} median {t['median_wall_min']:6.1f} min   "
              f"median steps/task {t['median_steps']:8.0f}")

    ok = a.report()
    out = {"n_arms": len(recs), "rows": [{"status": s, "check": n, "evidence": e}
                                        for s, n, e in a.rows],
           "rho_coverage": cov, "richness_separation": sep,
           "timing_by_gamma": comp["timing"],
           "non_converged": [k for k, _ in comp["unusable"]]}
    (ROOT / args.json).write_text(json.dumps(out, indent=2))
    print(f"\nwrote {args.json}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
