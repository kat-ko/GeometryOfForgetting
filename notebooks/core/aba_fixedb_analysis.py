"""docs/36: Phase 1b analysis, fixed budget on phase B per γ0 (extreme cases, run 1), against job 4651.

Load only (no training, not hashed). Used by aba_cluster.py check1b and by
notebooks/core/aba_fixed_budget_b.ipynb (built by _build_aba_fixed_budget_b.py). EXPLORATORY (docs/34):
95% paired t-intervals over the 8 seeds, no family. Cr = the job 4651 row of the same seed, γ0 and cell;
Fx = the docs/36 row.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats

import aba_lib as L
import aba_map_analysis as A

GAMMAS = A.GAMMAS
CELLS = L.PHASE1B_CELLS
H_SWEEP = [(h, 1.0) for h in (0, 2, 4, 8, 12, 16)]
# docs/36 §5(ii): the 12 adjacent contrasts, each (from cell, to cell).
CONTRASTS = ([((a, 1.0), (b, 1.0)) for a, b in ((0, 2), (2, 4), (4, 8), (8, 12), (12, 16))]
             + [((0, a), (0, b)) for a, b in ((1.0, 0.9), (0.9, 0.4), (0.4, 0.1))]
             + [((h, 1.0), (h, 0.1)) for h in (4, 8)]
             + [((a, 0.1), (b, 0.1)) for a, b in ((0, 4), (4, 8))])
MAP_JSONL = L.REPO / "results" / "aba" / "FULL_e3011fa49aca1994.jsonl"     # job 4651 (Cr)

# Per-arm quantities: the docs/35 ones (aba_map_analysis.QUANTS, without the cross-cell FT_beh specific)
# plus measured exposure during B (docs/36 §5b).
_Q = {q: A.QUANTS[q] for q in ("f", "dA", "rho", "u", "r", "d", "bott", "accA", "drhoc", "savings", "stepsB")}
QUANTS = {**_Q,
          "rB": ("weight change during B, r_B", lambda r, D: r["richness"]["r_t"]["B"]),
          "dKu": ("ΔK_u(A1 → B)", lambda r, D: r["richness"]["B"]["dK_u_step"]),
          "hitB": ("first step at criterion on B", lambda r, D: r["behaviour"]["first_hit"]["B"]
                   if "first_hit" in r["behaviour"] else r["behaviour"]["steps"]["B"])}


# ------------------------------------------------------------------ loading

def fixed_cfgs(base: L.ABAConfig | None = None) -> dict:
    base = base or L.ABAConfig.make(False)
    return {g: L.phase1b_cfg(base, g) for g in GAMMAS}


def load_fixed(base: L.ABAConfig | None = None, path: Path | None = None) -> tuple[dict, list[dict]]:
    """({(seed, γ0, H, s_f): row}, all docs/36 rows) from the shared fixed checkpoint."""
    cfgs = fixed_cfgs(base)
    path = path or L.cache_path(cfgs[0.1])
    if not Path(path).exists():
        return {}, []
    want = {json.dumps(c.key_fields(), sort_keys=True): {s.key for s in L.phase1b_specs(c)} for c in cfgs.values()}
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        keys = want.get(json.dumps(r["config"], sort_keys=True))
        if keys is not None and r["key"] in keys:
            rows.append(r)
    D = {(r["spec"]["seed"], r["spec"]["gamma"], r["spec"]["hamming"], r["spec"]["s_f"]): r for r in rows}
    return D, rows


def load_criterion(path: Path = MAP_JSONL, base: L.ABAConfig | None = None) -> dict:
    """The job 4651 rows (Cr), {(seed, γ0, H, s_f): row}."""
    base = base or L.ABAConfig.make(False)
    D, _, _ = A.load_map(path, L.map_config(base))
    return D


def usable_1b(r: dict) -> bool:
    """docs/36 §2: A1, A2 converged; B, fresh B ran the budget, finite, and reached the criterion."""
    hit = r["behaviour"].get("first_hit", {})
    return bool(r["usable"]) and hit.get("B") is not None and hit.get("freshB") is not None


# ------------------------------------------------------------------ integrity (docs/36 §2)

def integrity(FX: tuple[dict, list[dict]], CR: dict, base: L.ABAConfig | None = None) -> list[tuple[str, bool, str]]:
    D, rows = FX
    cfgs = fixed_cfgs(base)
    n_exp = sum(len(L.phase1b_specs(c)) for c in cfgs.values())
    chk = [("all docs/36 arms present", len(rows) == n_exp and len(D) == n_exp, f"{len(rows)} rows, {len(D)} keys / {n_exp}")]
    missing = [k for k in D if k not in CR]
    chk.append(("every arm has its job 4651 row", not missing, f"{len(missing)} missing"))
    a1 = [k for k in D if k in CR and not (
        D[k]["behaviour"]["steps"]["A1"] == CR[k]["behaviour"]["steps"]["A1"]
        and D[k]["geometry"]["A@A1"] == CR[k]["geometry"]["A@A1"]
        and D[k]["backward"]["init_to_A1"] == CR[k]["backward"]["init_to_A1"])]
    chk.append(("1. A1 bit-identical to job 4651 (steps, A@A1, init→A1)", not a1, f"{len(a1)} differ"))
    hits = [k for k in D if k in CR and not (
        D[k]["behaviour"]["first_hit"]["B"] == CR[k]["behaviour"]["steps"]["B"]
        and D[k]["behaviour"]["first_hit"]["freshB"] == CR[k]["behaviour"]["steps"]["freshB"])]
    chk.append(("2. first_hit B / fresh B = job 4651 steps B / fresh B", not hits,
                f"{len(hits)} differ" + (f", e.g. {sorted(hits)[:3]}" if hits else "")))
    bad_budget = [k for k in D if D[k]["behaviour"]["steps"]["B"] != L.PHASE1B_BUDGET[k[1]]
                  or D[k]["behaviour"]["steps"]["freshB"] != L.PHASE1B_BUDGET[k[1]]]
    chk.append(("B and fresh B ran exactly the γ0 budget", not bad_budget, f"{len(bad_budget)} differ"))
    us = sum(usable_1b(r) for r in rows)
    stamps = {json.dumps(r["stamp"], sort_keys=True) for r in rows}
    clean = all(not r["stamp"].get("dirty", True) for r in rows)
    hashes = {r["lib_hash"] for r in rows}
    chk.append(("3. all usable (A1, A2 converged; B, fresh B full budget, finite, reached criterion)",
                us == n_exp, f"{us}/{n_exp}"))
    chk.append(("3. one commit, clean tree", len(stamps) == 1 and clean, "; ".join(sorted(stamps))[:120]))
    chk.append(("3. one lib hash", len(hashes) == 1, ", ".join(sorted(hashes))))
    return chk


# ------------------------------------------------------------------ quantities

def vals(D, q, g, cell, seeds) -> np.ndarray:
    return np.array([QUANTS[q][1](D[s, g, *cell], D) for s in seeds], float)


def side_by_side(FXD, CR, seeds) -> dict:
    """{(q, γ0, cell): (Cr ci, Fx ci, (Fx − Cr) ci)} for every quantity and cell."""
    out = {}
    for g in GAMMAS:
        for cell in CELLS:
            for q in QUANTS:
                c, x = vals(CR, q, g, cell, seeds), vals(FXD, q, g, cell, seeds)
                out[q, g, cell] = (A.ci(c), A.ci(x), A.ci(x - c))
    return out


# ------------------------------------------------------------------ predictions (rules fixed in docs/36 §5)

READING_RULES_1B = """\
Reading rules (docs/36 §4–§5, committed 2026-11-11 before any Phase 1b arm). 95% paired t-intervals over
the 8 seeds (df 7), EXPLORATORY, no family. Cr = job 4651 row of the same seed, γ0 and cell; Fx = docs/36
row. Held overall if held at every γ0; not held overall if not held at any γ0; otherwise unresolved.
- (i) Gap: g = f(H12, s_f 1) − f(H4, s_f 1) per seed (independent draws within a seed: a distributional
  comparison paired by seed); Δg = g_Fx − g_Cr. Held if Δg's interval is wholly < 0; not held if wholly > 0;
  unresolved if it contains 0. Null: Δg = 0.
- (ii) Shape: 12 adjacent contrasts in f (H at s_f 1: 0→2, 2→4, 4→8, 8→12, 12→16; s_f at H0: 1→0.9,
  0.9→0.4, 0.4→0.1; s_f 1→0.1 at H4 and H8; H at s_f 0.1: 0→4, 4→8). Resolved under Cr if its Cr interval
  excludes 0 (sign σ); under Fx reversed if wholly on the −σ side, kept if wholly on the σ side. Unresolved
  Cr contrasts are not read. Spearman ρ_s of the 11 cell means of f, Cr vs Fx. Held at a γ0 if every
  resolved contrast is kept and ρ_s ≥ 0.9; not held if any resolved contrast reverses or ρ_s < 0.5;
  otherwise unresolved. Null: a resolved adjacent contrast reverses.
- (iii) The H0-drift gain is not reversed by extra exposure: Δlog α_A(A1 → B) at H0 × s_f 0.9 under Fx.
  Held if its interval is wholly ≥ 0; not held if wholly < 0; otherwise unresolved. Null: < 0.
"""


def _combine(res: list[str]) -> str:
    if all(r == "held" for r in res):
        return "held"
    if any(r == "not held" for r in res):
        return "not held"
    return "unresolved"


def _signs(x) -> str:
    x = np.asarray(x)
    return f"{int((x < 0).sum())}−/{int((x > 0).sum())}+ of {len(x)}"


def read_predictions(FXD, CR, seeds) -> dict:
    out = {}
    # (i)
    res, det = [], []
    for g in GAMMAS:
        gc = vals(CR, "f", g, (12, 1.0), seeds) - vals(CR, "f", g, (4, 1.0), seeds)
        gx = vals(FXD, "f", g, (12, 1.0), seeds) - vals(FXD, "f", g, (4, 1.0), seeds)
        c = A.ci(gx - gc)
        res.append("held" if c[2] < 0 else ("not held" if c[1] > 0 else "unresolved"))
        closed = 1 - gx.mean() / gc.mean() if gc.mean() != 0 else float("nan")
        det.append(f"γ0 {g:g}: Δg {A._f(c)} ({_signs(gx - gc)}); g_Cr {A._f(A.ci(gc))}, g_Fx {A._f(A.ci(gx))}; "
                   f"share of the gap closed {closed:+.2f}")
    out["(i) H4–H12 gap shrinks"] = (_combine(res), det)
    # (ii)
    res, det = [], []
    for g in GAMMAS:
        kept = rev = nread = 0
        lines = []
        for a, b in CONTRASTS:
            dc = vals(CR, "f", g, b, seeds) - vals(CR, "f", g, a, seeds)
            dx = vals(FXD, "f", g, b, seeds) - vals(FXD, "f", g, a, seeds)
            cc, cx = A.ci(dc), A.ci(dx)
            if cc[1] > 0 or cc[2] < 0:
                sig = 1 if cc[1] > 0 else -1
                if (sig > 0 and cx[2] < 0) or (sig < 0 and cx[1] > 0):
                    rev += 1
                    tag = "REVERSED"
                elif (sig > 0 and cx[1] > 0) or (sig < 0 and cx[2] < 0):
                    kept += 1
                    tag = "kept"
                else:
                    tag = "not kept (Fx interval contains 0)"
            else:
                nread += 1
                tag = "not read (Cr unresolved)"
            lines.append(f"    H{a[0]} s_f {a[1]:g} → H{b[0]} s_f {b[1]:g}: Cr {A._f(cc)}, Fx {A._f(cx)}: {tag}")
        mc = [vals(CR, "f", g, c, seeds).mean() for c in CELLS]
        mx = [vals(FXD, "f", g, c, seeds).mean() for c in CELLS]
        rho = stats.spearmanr(mc, mx).statistic
        same = list(np.argsort(mc)) == list(np.argsort(mx))
        n_res = len(CONTRASTS) - nread
        ok = rev == 0 and kept == n_res and rho >= 0.9
        bad = rev > 0 or rho < 0.5
        res.append("held" if ok else ("not held" if bad else "unresolved"))
        det.append(f"γ0 {g:g}: {kept}/{n_res} resolved contrasts kept, {rev} reversed, {nread} not read; "
                   f"Spearman ρ_s {rho:+.3f}; rank order identical: {same}")
        det += lines
    out["(ii) shape survives"] = (_combine(res), det)
    # (iii)
    res, det = [], []
    for g in GAMMAS:
        x = vals(FXD, "dA", g, (0, 0.9), seeds)
        c = A.ci(x)
        res.append("held" if c[1] >= 0 else ("not held" if c[2] < 0 else "unresolved"))
        det.append(f"γ0 {g:g}: Δlog α_A(A1 → B) Fx {A._f(c)} ({_signs(x)}); "
                   f"Cr {A._f(A.ci(vals(CR, 'dA', g, (0, 0.9), seeds)))}")
    out["(iii) H0-drift gain not reversed"] = (_combine(res), det)
    return out
