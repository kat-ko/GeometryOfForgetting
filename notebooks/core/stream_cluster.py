"""Cluster driver for the 1M stream (docs/29 §8). sbatch only for FULL; never on the login node.

    python notebooks/core/stream_cluster.py parity --spec 8,10,8,0.1   # A–B–A arm as a T = 2 stream, bit for bit
    python notebooks/core/stream_cluster.py arm --spec 8,0.1,8,0.1     # one T = 16 stream arm, sequential, timed
    add --quick for the QUICK scale (M 50, n_t 50).

Timing arms go to results/stream/timing_<stream_hash>.jsonl, never into a grid checkpoint.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

sys.path.insert(0, str(Path(__file__).resolve().parent))
import aba_lib as L  # noqa: E402
import aba_cluster as C  # noqa: E402   (exact JSON diff, peak RSS)
import stream_lib as S  # noqa: E402



def parity_pairs(aba: dict, st: dict) -> list[tuple[str, object, object]]:
    """(name, aba value, stream value) for every quantity both produce, after JSON."""
    a = json.loads(json.dumps(aba))
    s = json.loads(json.dumps(st))
    G, F = s["geometry"]["backward"], s["geometry"]
    ren = {"T0_test": "A_test", "T1_test": "B_test"}

    def traj(recs):
        out = []
        for r in recs:
            q = {}
            for k, v in r.items():
                for old, new in ren.items():
                    k = k.replace(old, new)
                q[k] = v
            out.append(q)
        return out

    K = len(s["forward"]["0"]["neutral"])
    pairs = [
        ("init_loss", a["init_loss"], s["init_loss"]), ("lr", a["lr"], s["lr"]),
        ("output_scale", a["output_scale"], s["output_scale"]),
        ("geometry A@init", a["geometry"]["A@init"], G["0,init"]),
        ("geometry A@A1", a["geometry"]["A@A1"], G["0,0"]),
        ("geometry A@B", a["geometry"]["A@B"], G["0,1"]),
        ("geometry B_yB@init", a["geometry"]["B_yB@init"], G["1,init"]),
        ("geometry B_yB@A1", a["geometry"]["B_yB@A1"], F["forward_after"]["0"]["next"]),
        *[(f"geometry B_ystarN{k}@init", a["geometry"][f"B_ystarN{k}@init"], F["forward_neutral_init"]["0"][k])
          for k in range(K)],
        *[(f"geometry B_ystarN{k}@A1", a["geometry"][f"B_ystarN{k}@A1"], F["forward_after"]["0"]["neutral"][k])
          for k in range(K)],
        ("backward init_to_A1", a["backward"]["init_to_A1"], s["backward"]["0,init>0"]),
        ("backward A1_to_B", a["backward"]["A1_to_B"], s["backward"]["0,0>1"]),
        ("forward total", a["forward"]["total"], s["forward"]["0"]["total"]),
        ("forward neutral", [a["forward"]["ystar_neutral"][str(k)] for k in range(K)], s["forward"]["0"]["neutral"]),
        ("forward neutral mean", a["forward"]["ystar_neutral"]["mean"], s["forward"]["0"]["neutral_mean"]),
        ("forward specific", a["forward"]["specific"], s["forward"]["0"]["specific"]),
        ("FT_beh", a["behaviour"]["FT_beh"], s["forward"]["0"]["FT_beh"]),
        ("steps A1, B", [a["behaviour"]["steps"]["A1"], a["behaviour"]["steps"]["B"]], s["behaviour"]["steps"]),
        ("steps fresh B", a["behaviour"]["steps"]["freshB"], s["behaviour"]["fresh_steps"][0]),
        ("first_loss A1, B", [a["behaviour"]["first_loss"]["A1"], a["behaviour"]["first_loss"]["B"]],
         s["behaviour"]["first_loss"]),
        ("first_loss fresh B", a["behaviour"]["first_loss"]["freshB"], s["behaviour"]["fresh_first_loss"][0]),
        ("richness init", a["richness"]["init"], s["richness"]["init"]),
        ("richness A1", a["richness"]["A1"], s["richness"]["0"]),
        ("richness B", a["richness"]["B"], s["richness"]["1"]),
        ("r_t A1, B", [a["richness"]["r_t"]["A1"], a["richness"]["r_t"]["B"]], s["richness"]["r_t"]),
        ("s_r_AB", a["relations"]["s_r_AB"], s["relations"]["S_r"][0][1]),
        ("rho_in_AB", a["relations"]["rho_in_AB"], s["relations"]["rho_in"][0][1]),
        ("trajectory A1", a["trajectory"]["A1"], traj(s["trajectory"][0])),
        ("trajectory B", a["trajectory"]["B"], traj(s["trajectory"][1])),
    ]
    return pairs


def parity_diffs(aba: dict, st: dict) -> tuple[int, list[str]]:
    pairs = parity_pairs(aba, st)
    diffs = [f"{name}{d}" for name, x, y in pairs for d in C._diff(x, y)]
    return len(pairs), diffs


def parity(cfg: L.ABAConfig, spec: L.ArmSpec) -> int:
    t0 = time.perf_counter()
    aba = L.run_arm(cfg, spec)
    t1 = time.perf_counter()
    st = S.run_stream(cfg, S.inputs_from_aba(cfg, spec), spec.gamma, floor=False)
    t2 = time.perf_counter()
    n, diffs = parity_diffs(aba, st)
    print(f"parity {spec.key} ({'QUICK' if cfg.quick else 'FULL'}): aba_lib {t1 - t0:.0f}s, stream_lib {t2 - t1:.0f}s; "
          f"{n} quantities compared; lib {L.lib_hash()}, stream_hash {S.stream_hash()}")
    if diffs:
        print(f"NOT bit-identical: {len(diffs)} differences")
        for d in diffs[:50]:
            print("  ", d)
        return 1
    print("bit-identical")
    return 0


def arm(cfg: L.ABAConfig, spec: S.StreamSpec, T: int) -> None:
    out = L.REPO / "results" / "stream" / f"timing_{S.stream_hash()}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    res = S.run_spec(cfg, spec, T)
    wall = time.perf_counter() - t0
    tm = res["timing"]
    b = res["behaviour"]
    print(f"{spec.key} T={T} ({'QUICK' if cfg.quick else 'FULL'}): wall {wall:.0f}s (train {tm['train_s']:.0f}, "
          f"fresh {tm['fresh_s']:.0f}, GLUE {tm['glue_s']:.0f}, NTK {tm['ntk_s']:.0f}, floor {tm['floor_s']:.0f}); "
          f"GLUE calls {len(res['geometry']['backward']) + sum(1 + len(v['neutral']) for v in res['geometry']['forward_after'].values()) + sum(len(v) for v in res['geometry']['forward_neutral_init'].values())}; "
          f"steps {b['steps']}; fresh {b['fresh_steps']}; usable {res['usable']}; peak RSS {C.peak_rss_mb():.0f} MB",
          flush=True)
    with out.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(res) + "\n")
    print(f"row written to {out} ({out.stat().st_size / 1e6:.1f} MB file)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("parity", "arm"))
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--spec", required=True, help="seed/draw,gamma,hamming,s_f")
    ap.add_argument("--T", type=int, default=16)
    a = ap.parse_args()
    cfg = L.ABAConfig.make(a.quick)
    sd, g, h, f = a.spec.split(",")
    print(f"mode {a.mode}  {'QUICK' if a.quick else 'FULL'}  lib {L.lib_hash()}  stream_hash {S.stream_hash()}  "
          f"stamp {L.git_stamp()}", flush=True)
    if a.mode == "parity":
        sys.exit(parity(cfg, L.ArmSpec(int(sd), float(g), int(h), float(f))))
    arm(cfg, S.StreamSpec(int(sd), float(g), int(h), float(f)), a.T)
