"""Adam Hamming-only slice at γ=10 against `results/adam_hamming_precommit.json`.

Does not train. Cannot speak to γ. Readings named before numbers.

    python scripts/analyse_adam_hamming.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from analyse_hamming import (  # noqa: E402
    INPUTS, LAG, MIN_FLOORS, MODULE, S_R, TASK,
    _fmt_cell, _fmt_rho, _grid, _lag4_rows, _rho_rows, _rows,
    cliff_capacity, cliff_rho, pack, r1,
)
from src.reservations import reserved_stream_ids  # noqa: E402

PRECOMMIT = ROOT / "results" / "adam_hamming_precommit.json"
ARMS = ROOT / "results" / "adam"
OUT = ROOT / "results" / "adam_hamming_slice.json"
MD = ROOT / "results" / "adam_hamming_slice.md"
GAMMA = 10.0
AXIS = ("cliff", "drift_gradation", "crossover")


def cliff_holds(grid, gamma: float = GAMMA) -> bool:
    """Frozen and jump are cliffs: large 1.0→0.75, unresolved post-cliff."""
    for level in ("frozen", "jump"):
        st = cliff_capacity(grid, gamma, level)
        if st["cliff"] < MIN_FLOORS:
            return False
        if st["post_range"] >= MIN_FLOORS and not st["post_ci_overlap"]:
            return False
    return True


def drift_gradation_holds(grid, gamma: float = GAMMA) -> bool:
    st = cliff_capacity(grid, gamma, "drift")
    return st["post_range"] >= MIN_FLOORS and not st["post_ci_overlap"]


def crossover_holds(grid, gamma: float = GAMMA) -> bool:
    """At 0.75 drift better than frozen; at 0.25 frozen better than drift."""
    f75 = r1(grid[(gamma, "frozen", 0.75)]["mean_floors"])
    d75 = r1(grid[(gamma, "drift", 0.75)]["mean_floors"])
    f25 = r1(grid[(gamma, "frozen", 0.25)]["mean_floors"])
    d25 = r1(grid[(gamma, "drift", 0.25)]["mean_floors"])
    return d75 > f75 and f25 > d25


def finding3_recovered(grid, gamma: float = GAMMA) -> bool:
    return grid[(gamma, "drift", 1.0)]["mean"] > 0 and grid[(gamma, "jump", 1.0)]["mean"] < 0


def apply_readings(grid) -> dict:
    pre = json.loads(PRECOMMIT.read_text())
    recovered = finding3_recovered(grid)
    held_map = {
        "cliff": cliff_holds(grid),
        "drift_gradation": drift_gradation_holds(grid),
        "crossover": crossover_holds(grid),
    }
    held = [k for k in AXIS if held_map[k]]
    failed = [k for k in AXIS if not held_map[k]]
    if not recovered:
        primary = "finding3_fails_to_recover"
    elif not failed:
        primary = "hamming_reproduces"
    elif not held:
        primary = "hamming_fails"
    else:
        primary = "partial"
    return {
        "finding3_recovered": recovered,
        "cliff_holds": held_map["cliff"],
        "drift_gradation_holds": held_map["drift_gradation"],
        "crossover_holds": held_map["crossover"],
        "held": held,
        "failed": failed,
        "primary_reading": primary,
        "cannot_speak_to": pre["cannot_speak_to"],
        "operationalization": pre["operationalization"],
    }


def main() -> None:
    pre = json.loads(PRECOMMIT.read_text())
    if not pre.get("written_before_running"):
        sys.exit("pre-commit missing")
    if pre.get("written_before_running_applies_to") != "96-arm Adam Hamming-only at γ=10":
        sys.exit("this analyser governs the 96; check adam_hamming_precommit.json")
    reserved = reserved_stream_ids()
    paths = sorted(ARMS.glob("*.json"))
    raw_all = [json.loads(p.read_text()) for p in paths]
    raw = []
    for r in raw_all:
        sp = r.get("spec") or {}
        if float(sp.get("gamma_0", 0)) != GAMMA:
            continue
        if sp.get("optimizer") != "adam":
            sys.exit(f"non-Adam record in results/adam/: {sp}")
        raw.append(r)
    if len(raw) != 96:
        sys.exit(f"expected 96 γ=10 Adam arms, found {len(raw)}")
    sids = {r["spec"]["stream_id"] for r in raw}
    if sids != set(reserved):
        sys.exit(f"stream ids {sorted(sids)} are not the reserved set")
    for r in raw:
        sp = r["spec"]
        if sp.get("arrangement_source") != "stream_rng":
            sys.exit("non-reserved arrangement_source")
        if sp.get("lr_scaling") != "quadratic":
            sys.exit("non-quadratic record")
        if int(sp.get("tracked_stride", 0)) != 2:
            sys.exit("tracked_stride is not 2")
        if int(sp.get("P", 0)) != 16:
            sys.exit("P is not 16")
        if sp.get("input_level") not in INPUTS:
            sys.exit(f"bad input_level {sp.get('input_level')}")
        if float(sp.get("lr0", 0)) != 0.0002:
            sys.exit("lr0 is not the signed pin")

    n_usable = sum(1 for r in raw if r.get("usable"))
    n_miss = len(raw) - n_usable
    usable = [r for r in raw if r.get("usable")]
    if n_miss:
        print(f"WARNING: {n_miss} unusable arm(s); primary uses the usable set", flush=True)

    rows = _rows(usable)
    grid = _grid(rows)
    for level in INPUTS:
        for s in S_R:
            cell = grid[(GAMMA, level, s)]
            if cell["n"] != 8:
                sys.exit(f"cell γ={GAMMA:g} {level} s_r={s} has n={cell['n']}, want 8")

    applied = apply_readings(grid)
    rho = _grid(_rho_rows(usable), field="d_rho_c")
    lag4 = _lag4_rows(usable)
    lag4_pack: dict = {}
    for x in lag4:
        if x["gamma"] != GAMMA:
            continue
        lag4_pack.setdefault((x["input_level"], x["s_r"], x["task"]), []).append(x["dlog_alpha"])
    lag4_out = {
        f"gamma={GAMMA:g}|{k[0]}|s_r={k[1]:g}|task={k[2]}": pack(v)
        for k, v in sorted(lag4_pack.items())
    }
    cells_out = {
        f"gamma={GAMMA:g}|{level}|s_r={s:g}": grid[(GAMMA, level, s)]
        for level in INPUTS for s in S_R
    }
    rho_out = {
        f"gamma={GAMMA:g}|{level}|s_r={s:g}": rho[(GAMMA, level, s)]
        for level in INPUTS for s in S_R
    }
    structure = {
        "capacity": {
            f"gamma={GAMMA:g}|{level}": cliff_capacity(grid, GAMMA, level)
            for level in INPUTS
        },
        "rho_c": {
            f"gamma={GAMMA:g}|{level}": cliff_rho(rho, GAMMA, level)
            for level in INPUTS
        },
        "crossover": {
            "at_0.75": "drift better than frozen" if applied["crossover_holds"] else "did not hold",
            "at_0.25": "frozen better than drift" if applied["crossover_holds"] else "did not hold",
            "frozen_0.75_floors": r1(grid[(GAMMA, "frozen", 0.75)]["mean_floors"]),
            "drift_0.75_floors": r1(grid[(GAMMA, "drift", 0.75)]["mean_floors"]),
            "frozen_0.25_floors": r1(grid[(GAMMA, "frozen", 0.25)]["mean_floors"]),
            "drift_0.25_floors": r1(grid[(GAMMA, "drift", 0.25)]["mean_floors"]),
        },
        "cannot_speak_to": pre["cannot_speak_to"],
    }
    rec = {
        "generated_by": "scripts/analyse_adam_hamming.py",
        "precommit": "results/adam_hamming_precommit.json",
        "n_arms": len(raw),
        "n_usable": n_usable,
        "n_missed": n_miss,
        "unique_n": 8,
        "gamma": GAMMA,
        "cell": {"task": TASK, "lag": LAG, "module": MODULE},
        "applied": applied,
        "structure": structure,
        "cells": cells_out,
        "delta_rho_c_retained_task0": rho_out,
        "lag4_position": lag4_out,
        "will_not": pre["will_not_do"],
        "cannot_speak_to": pre["cannot_speak_to"],
    }
    OUT.write_text(json.dumps(rec, indent=2) + "\n")

    def cap_row(level):
        st = structure["capacity"][f"gamma={GAMMA:g}|{level}"]
        return (f"| {level} | {st['cliff']:+.1f} | {st['post_range']:.1f} | "
                f"{'overlap' if st['post_ci_overlap'] else '**resolved**'} |")

    reading = applied["primary_reading"]
    lines = [
        "# Adam Hamming-only slice (γ=10)",
        "",
        "Unique n=8, module A, task 0, lag 12. 95% CI = mean ± 1.96 SEM. "
        "Adam, MSE, `lr0=0.0002`. **This arm cannot speak to γ.**",
        "",
        f"**n_arms = {len(raw)}, usable = {n_usable}, missed = {n_miss}.**",
        "",
        f"**Finding 3 recovery** (s_r=1, γ=10): "
        f"{'yes' if applied['finding3_recovered'] else 'NO — stop'}. "
        f"drift {grid[(GAMMA, 'drift', 1.0)]['mean_floors']:+.1f} fl, "
        f"jump {grid[(GAMMA, 'jump', 1.0)]['mean_floors']:+.1f} fl.",
        "",
    ]
    if applied["finding3_recovered"]:
        lines += [
            f"**Primary reading: `{reading}`.** "
            f"Held: {', '.join(applied['held']) or 'none'}. "
            f"Failed: {', '.join(applied['failed']) or 'none'}.",
            "",
        ]
    else:
        lines += [
            f"**Primary reading: `{reading}`.** Stop. The Adam streams are "
            "not the same object. Hamming-axis flags were computed and are "
            "not a reading.",
            "",
        ]
    lines += [
        "## Cliff, post-cliff range",
        "",
        "Cliff = floors at s_r=1.0 minus floors at 0.75 (1-decimal). "
        "Post-cliff range = |floors(0.75) − floors(0.25)|.",
        "",
        "| input | cliff (1.0 → 0.75) | range 0.75 → 0.25 | 0.75 vs 0.25 CI |",
        "|---|---|---|---|",
        cap_row("frozen"),
        cap_row("drift"),
        cap_row("jump"),
        "",
        "## Crossover",
        "",
        f"s_r=0.75: drift {r1(grid[(GAMMA, 'drift', 0.75)]['mean_floors']):+.1f}, "
        f"frozen {r1(grid[(GAMMA, 'frozen', 0.75)]['mean_floors']):+.1f}. "
        f"s_r=0.25: drift {r1(grid[(GAMMA, 'drift', 0.25)]['mean_floors']):+.1f}, "
        f"frozen {r1(grid[(GAMMA, 'frozen', 0.25)]['mean_floors']):+.1f}.",
        "",
        "## Δ log α, task 0 lag 12, γ=10",
        "",
        "| input \\ s_r | 1.00 | 0.75 | 0.50 | 0.25 |",
        "|---|---|---|---|---|",
    ]
    for level in INPUTS:
        cells = [_fmt_cell(grid[(GAMMA, level, s)]) for s in S_R]
        lines.append(f"| {level} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "## Δρ_c, retained task 0, lag 12 (reported, not a reading)",
        "",
        "| input \\ s_r | 1.00 | 0.75 | 0.50 | 0.25 |",
        "|---|---|---|---|---|",
    ]
    for level in INPUTS:
        cells = [_fmt_rho(rho[(GAMMA, level, s)]) for s in S_R]
        lines.append(f"| {level} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "Cannot speak to: " + "; ".join(pre["cannot_speak_to"]) + ".",
        "",
        "Source: `results/adam_hamming_slice.json`. "
        "Pre-commit: `results/adam_hamming_precommit.json`.",
        "",
    ]
    MD.write_text("\n".join(lines))
    print(json.dumps({
        "n_arms": len(raw),
        "n_usable": n_usable,
        "primary_reading": reading,
        "held": applied["held"],
        "failed": applied["failed"],
        "finding3_recovered": applied["finding3_recovered"],
    }, indent=2))


if __name__ == "__main__":
    main()
