"""Recurrence slice against `results/recurrence_precommit.json`. Does not train.

Primary: paired Δ = steps(represent, k=12) − steps(control, k=12),
S-HL and S-LL, unique n=8, γ=10.

    python scripts/analyse_recurrence.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.reservations import reserved_stream_ids  # noqa: E402

PRECOMMIT = ROOT / "results" / "recurrence_precommit.json"
ARMS = ROOT / "results" / "recurrence"
OUT = ROOT / "results" / "recurrence_slice.json"
K = 12
LICENSED = (1.0, 10.0)
FORGETTING = ("S-HL", "S-LL")
MODES = ("represent", "control")
GEOM_BOUNDARIES = (8, 12, 15)
MODULE = "A"
TASK = 0


def _load() -> list[dict]:
    paths = sorted(ARMS.glob("*.json"))
    return [json.loads(p.read_text()) for p in paths]


def _spec(rec: dict) -> dict:
    return rec["spec"]


def _steps(rec: dict) -> list[int]:
    return [int(t["steps_taken"]) for t in rec["tasks"]]


def _index(recs: list[dict]) -> dict[tuple, dict]:
    out = {}
    for rec in recs:
        s = _spec(rec)
        key = (float(s["gamma_0"]), s["condition"], int(s["stream_id"]),
               s["recurrence_mode"])
        out[key] = rec
    return out


def _prefix_ok(a: dict, b: dict, k: int) -> bool:
    sa, sb = _steps(a), _steps(b)
    if sa[:k] != sb[:k]:
        return False
    ra = np.asarray(a["stream"]["S_r"], float)
    rb = np.asarray(b["stream"]["S_r"], float)
    return bool(np.allclose(ra[:k, :k], rb[:k, :k]))


def _paired_delta(idx: dict, g: float, conditions: tuple[str, ...]) -> dict:
    reserved = sorted(reserved_stream_ids())
    per_stream = {}
    n_prefix_fail = 0
    for sid in reserved:
        ds = []
        for c in conditions:
            rep = idx.get((g, c, sid, "represent"))
            ctrl = idx.get((g, c, sid, "control"))
            if rep is None or ctrl is None:
                continue
            if not rep.get("usable") or not ctrl.get("usable"):
                continue
            if not _prefix_ok(rep, ctrl, K):
                n_prefix_fail += 1
                continue
            ds.append(_steps(rep)[K] - _steps(ctrl)[K])
        if ds:
            per_stream[str(sid)] = float(np.mean(ds))
    n = len(per_stream)
    if n == 0:
        return {"n": 0, "n_prefix_fail": n_prefix_fail}
    vals = list(per_stream.values())
    mean = float(np.mean(vals))
    sem = float(np.std(vals, ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    lo, hi = mean - 1.96 * sem, mean + 1.96 * sem
    return {
        "n": n,
        "d_steps_mean": mean,
        "d_steps_sem": sem,
        "ci95": [lo, hi],
        "excludes_0": bool(lo > 0 or hi < 0),
        "n_prefix_fail": n_prefix_fail,
        "per_stream": per_stream,
    }


def _classify(d: dict) -> str:
    if d.get("n", 0) == 0:
        return "no_difference"
    if not d.get("excludes_0"):
        return "no_difference"
    return "savings" if d["d_steps_mean"] < 0 else "reverse"


def _summarise(per_stream: dict[str, float]) -> dict:
    n = len(per_stream)
    if n == 0:
        return {"n": 0, "per_stream": per_stream}
    vals = list(per_stream.values())
    mean = float(np.mean(vals))
    sem = float(np.std(vals, ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    return {
        "n": n,
        "mean": mean,
        "sem": sem,
        "per_stream": per_stream,
    }


def _stream_mean(idx: dict, g: float, mode: str, value) -> dict[str, float]:
    """Mean of S-HL and S-LL for each reserved stream. Skips missing pairs."""
    reserved = sorted(reserved_stream_ids())
    per_stream = {}
    for sid in reserved:
        xs = []
        for c in FORGETTING:
            rec = idx.get((g, c, sid, mode))
            if rec is None or not rec.get("usable"):
                continue
            xs.append(value(rec))
        if xs:
            per_stream[str(sid)] = float(np.mean(xs))
    return per_stream


def _retained_alpha(rec: dict, boundary: int) -> float:
    hits = [
        g for g in rec["geometry"]
        if g["module"] == MODULE
        and g["boundary"] == boundary
        and g["task"] == TASK
        and g["ensemble"] == "retained"
    ]
    if len(hits) != 1:
        raise ValueError(
            f"expected one retained geometry point "
            f"(module={MODULE}, task={TASK}, boundary={boundary}), got {len(hits)}"
        )
    return float(hits[0]["alpha"])


def _steps_k_vs_0(idx: dict, g: float) -> dict:
    """Position-confounded. Reported. Not a reading."""
    s0 = _summarise(_stream_mean(idx, g, "represent", lambda r: _steps(r)[0]))
    sk = _summarise(_stream_mean(idx, g, "represent", lambda r: _steps(r)[K]))
    return {
        "not_a_reading": True,
        "steps_0": s0,
        "steps_k": sk,
    }


def _geometry_cell(idx: dict, g: float, mode: str) -> dict:
    """Retained α on task 0. Reported. Not a reading."""
    out = {}
    for b in GEOM_BOUNDARIES:
        out[str(b)] = _summarise(
            _stream_mean(idx, g, mode, lambda r, bb=b: _retained_alpha(r, bb))
        )
    return out


Z_975 = 1.96
Z_80 = 0.841621233572914  # Φ^{-1}(0.80)


def _mde(delta: dict, control_k: dict) -> dict:
    """Bounds relative to steps(control, k), the identification baseline.

    Savings is negative Δ. The 95% CI excludes Δ below its lower end, so
    savings larger than max(0, −lo) steps are ruled out. The two-sided 80%
    MDE is (z_0.975 + z_0.80) · SEM. Neither uses steps(0).
    """
    sem = float(delta["d_steps_sem"])
    lo, hi = delta["ci95"]
    hw = Z_975 * sem
    mde80 = (Z_975 + Z_80) * sem
    baseline = float(control_k["mean"])
    ruled = float(max(0.0, -lo))
    return {
        "baseline": "steps(control, k), unique n=8, mean of S-HL and S-LL",
        "control_steps_k": control_k,
        "ci_halfwidth_steps": hw,
        "mde_80_twosided_steps": mde80,
        "savings_ruled_out_at_95_steps": ruled,
        "ci_halfwidth_frac_of_control": hw / baseline,
        "mde_80_twosided_frac_of_control": mde80 / baseline,
        "savings_ruled_out_at_95_frac_of_control": ruled / baseline,
        "ci95": [lo, hi],
    }


def _control_check(idx: dict) -> dict:
    """Realised A8 match on the trained arms, not only on the generator tests."""
    reserved = sorted(reserved_stream_ids())
    max_err = 0.0
    n_pairs = 0
    n_rep_not_y0 = 0
    n_ctrl_is_y0 = 0
    ctrl_to_y0 = []
    for g in LICENSED:
        for sid in reserved:
            for c in FORGETTING:
                rep = idx.get((g, c, sid, "represent"))
                ctrl = idx.get((g, c, sid, "control"))
                if rep is None or ctrl is None:
                    continue
                if not rep.get("usable") or not ctrl.get("usable"):
                    continue
                sr = np.asarray(rep["stream"]["S_r"], float)
                sc = np.asarray(ctrl["stream"]["S_r"], float)
                if abs(sr[0, K] - 1.0) > 1e-12:
                    n_rep_not_y0 += 1
                err = abs(sc[K, K - 1] - sr[0, K - 1])
                max_err = max(max_err, float(err))
                if abs(sc[0, K] - 1.0) <= 1e-12:
                    n_ctrl_is_y0 += 1
                ctrl_to_y0.append(float(sc[0, K]))
                n_pairs += 1
    held = (
        n_pairs > 0
        and max_err == 0.0
        and n_rep_not_y0 == 0
        and n_ctrl_is_y0 == 0
    )
    return {
        "held": held,
        "n_pairs": n_pairs,
        "max_sr_match_error": max_err,
        "n_represent_not_y0": n_rep_not_y0,
        "n_control_is_y0": n_ctrl_is_y0,
        "control_sr_to_y0": {
            "min": float(min(ctrl_to_y0)) if ctrl_to_y0 else None,
            "max": float(max(ctrl_to_y0)) if ctrl_to_y0 else None,
        },
    }


def main() -> None:
    pre = json.loads(PRECOMMIT.read_text())
    reserved = reserved_stream_ids()
    raw = _load()
    if not raw:
        sys.exit("no arms in results/recurrence/")
    sids = {_spec(r)["stream_id"] for r in raw}
    if sids != set(reserved):
        sys.exit(f"stream ids {sorted(sids)} are not the reserved set")
    if any(_spec(r).get("recurrence_k") != K for r in raw):
        sys.exit("unexpected recurrence_k")
    if any(_spec(r).get("recurrence_mode") not in MODES for r in raw):
        sys.exit("unexpected recurrence_mode")

    n_miss = sum(1 for r in raw if not r.get("usable"))
    idx = _index(raw)
    check = _control_check(idx)
    if not check["held"]:
        sys.exit(f"A8 control construction did not hold: {check}")
    d10 = _paired_delta(idx, 10.0, FORGETTING)
    d1 = _paired_delta(idx, 1.0, FORGETTING)
    readings = pre["readings_committed_before_seeing_numbers"]
    if d10.get("n_prefix_fail") or d1.get("n_prefix_fail"):
        key = "prefix_mismatch"
    else:
        a, b = _classify(d10), _classify(d1)
        if a != b and not (a == "savings" and b == "no_difference"):
            key = "mixed"
        else:
            key = a

    ctrl_k = {
        "1": _summarise(_stream_mean(idx, 1.0, "control", lambda r: _steps(r)[K])),
        "10": _summarise(_stream_mean(idx, 10.0, "control", lambda r: _steps(r)[K])),
    }
    out = {
        "generated_by": "scripts/analyse_recurrence.py",
        "precommit": "results/recurrence_precommit.json",
        "n_files": len(raw),
        "n_usable": sum(1 for r in raw if r.get("usable")),
        "n_arms_missed": n_miss,
        "cell": {"k": K, "quantity": "steps_taken", "unique_n": 8,
                 "unit": "reserved stream_id, seed=0"},
        "low_readout": FORGETTING,
        "structural_finding": {
            "key": "high_readout_refused",
            "corners": pre["design"]["corners_refused"]["conditions"],
            "reason": pre["design"]["corners_refused"]["reason"],
            "not": "a scoping compromise",
            "text": (
                "At P=16, registered consecutive s_r=0.9 rounds to Hamming 0. "
                "Every dichotomy in S-HH and S-LH is y_0 up to sign, so a novel "
                "A8 control does not exist. Recurrence is only testable in the "
                "low-readout corners. Exposure and similarity are not independently "
                "settable at this scope."
            ),
        },
        "control_check": check,
        "paired_represent_minus_control": {
            "1": d1,
            "10": d10,
        },
        "mde": {
            "1": _mde(d1, ctrl_k["1"]),
            "10": _mde(d10, ctrl_k["10"]),
        },
        "steps_k_versus_0": {
            "note": pre["comparator"]["not_the_primary"],
            "1": _steps_k_vs_0(idx, 1.0),
            "10": _steps_k_vs_0(idx, 10.0),
        },
        "geometry": {
            "note": pre["comparator"]["geometry"],
            "module": MODULE,
            "task": TASK,
            "ensemble": "retained",
            "boundaries": list(GEOM_BOUNDARIES),
            "1": {m: _geometry_cell(idx, 1.0, m) for m in MODES},
            "10": {m: _geometry_cell(idx, 10.0, m) for m in MODES},
        },
        "reading": {
            "key": key,
            "text": readings[key],
            "operationalization": pre["operationalization"],
        },
    }
    OUT.write_text(json.dumps(out, indent=2))
    print(json.dumps({
        "n_files": out["n_files"],
        "n_usable": out["n_usable"],
        "n_arms_missed": n_miss,
        "paired_represent_minus_control": out["paired_represent_minus_control"],
        "mde": {
            g: {
                "control_steps_k": out["mde"][g]["control_steps_k"]["mean"],
                "savings_ruled_out_at_95_steps": out["mde"][g][
                    "savings_ruled_out_at_95_steps"],
                "savings_ruled_out_at_95_frac_of_control": out["mde"][g][
                    "savings_ruled_out_at_95_frac_of_control"],
                "ci_halfwidth_frac_of_control": out["mde"][g][
                    "ci_halfwidth_frac_of_control"],
            }
            for g in ("1", "10")
        },
        "control_check": out["control_check"]["held"],
        "reading": out["reading"]["key"],
    }, indent=2))


if __name__ == "__main__":
    main()
