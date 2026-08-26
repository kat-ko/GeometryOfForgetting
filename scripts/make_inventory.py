"""Every number that appears in the paper, with the file it comes from — and a check that it does.

This is not a transcription. Each entry below carries the *expected* value alongside a getter
that reads it back out of the artifact it came from, and the script fails if any of them has
drifted. So the inventory cannot quietly become a stale copy of numbers that have since changed:
if the analysis moves, this fails, and the paper's number is wrong until someone updates both.

That is the specific failure this project has already had four times — a figure corrected in the
log while a superseded value survived somewhere it had propagated to. A list of numbers that is
merely *written down* would have caught none of them.

**Two links, not one.** Verifying value against source leaves the link that actually failed
unchecked: `S-LH`'s Δρ_c was read from a 4-arm cell, and the drafted sentence "shares agree to
within 0.033" survived a re-read giving 0.032, because nothing compared the prose to the value. A
row may therefore carry `quote=`, a literal fragment that must appear verbatim in the draft. The
chain checked is then source → value → sentence, and a number cannot be corrected in one place
alone.

    python scripts/make_inventory.py [--check]

Writes `docs/08-inventory.md`. `--check` verifies without writing.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

FIG = ROOT / "figures"
RES = ROOT / "results"
OUT = ROOT / "docs" / "08-inventory.md"


def agrees(expected: float, got: float) -> bool:
    """Does the source still round to the number the paper prints?

    Comparison is at the precision the paper quotes, not at an arbitrary tolerance: a paper
    saying "0.2 floors" is making a claim about the first decimal, and 0.176 satisfies it while
    0.24 does not. Very small values (residuals) are compared relatively instead, since their
    decimal places are not the claim.
    """
    if abs(expected) < 1e-6:
        return abs(got - expected) <= 0.01 * abs(expected)
    s = repr(float(expected))
    decimals = len(s.split(".")[1]) if "." in s and "e" not in s else 6
    return round(got, decimals) == round(float(expected), decimals)


def sidecar(manifest: dict, fig_id: str) -> str:
    """The JSON a figure writes beside its PDF, which is what the numbers are read from."""
    name = next(n for n in manifest["figures"][fig_id]["outputs"]
                if n.endswith(".json"))
    return f"figures/{name}"


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def figjson(fig_id: str) -> dict:
    """The JSON sidecar of a figure in the manifest, so the inventory reads the *final* figure."""
    man = load(FIG / "MANIFEST.json")
    name = next(n for n in man["figures"][fig_id]["outputs"] if n.endswith(".json"))
    return load(FIG / name)


def rows() -> list[dict]:
    """One dict per number: where it is printed, what it is, and how to re-read it."""
    f2 = figjson("fig2")
    f2b = figjson("fig2_benign")
    f2l4 = figjson("fig2_lag4")
    f4 = figjson("fig4")
    fw = figjson("fig_width")
    mn = load(RES / "measurement_null.json")
    au = load(RES / "audit_scope.json")
    ap = load(RES / "audit_propagation.json")
    man = load(FIG / "MANIFEST.json")
    u8 = load(RES / "unique_n8.json")

    d2, d2b, d2l4 = f2["data"], f2b["data"], f2l4["data"]

    R = []

    def add(section, name, expected, units, source, getter, quote=None, n=None):
        R.append({"section": section, "name": name, "expected": expected, "units": units,
                  "source": source, "getter": getter, "quote": quote, "n": n})

    # --- §5.1, from Figure 2 -------------------------------------------------
    src2 = f"figures/{[n for n in man['figures']['fig2']['outputs'] if n.endswith('.json')][0]}"
    add("§5.1", "forgetting at γ₀=0.03 (nothing happens)", 0.2, "α noise floors", src2,
        lambda: abs(d2["0.03"]["floors"]))
    add("§5.1", "forgetting at γ₀=0.1 (first resolvable)", 2.1, "α noise floors", src2,
        lambda: abs(d2["0.1"]["floors"]))
    add("§5.1", "forgetting at γ₀=10", 69.7, "α noise floors", src2,
        lambda: abs(d2["10.0"]["floors"]), quote="falls by 69.7 floors at γ₀ = 10", n=120)
    add("§5.1", "Δ log α at γ₀=10", -1.2917, "log units", src2,
        lambda: d2["10.0"]["dlog_alpha"])
    # The composition range starts where the magnitude first clears the gate. The regroup moved
    # that from γ₀=0.3 to 0.1, which moved the range's left endpoint from 0.19 to 0.08.
    add("§5.1", "utility share at γ₀=0.1 (first resolvable)", 0.083, "fraction of total motion",
        src2, lambda: d2["0.1"]["shares"]["utility"], quote="grows from 0.08 to 0.45")
    add("§5.1", "utility share at γ₀=10", 0.449, "fraction of total motion", src2,
        lambda: d2["10.0"]["shares"]["utility"])
    add("§5.1", "radius share at γ₀=0.1", 0.387, "fraction of total motion", src2,
        lambda: d2["0.1"]["shares"]["radius"], quote="radius term falls from 0.39 to 0.11")
    add("§5.1", "radius share at γ₀=10", 0.114, "fraction of total motion", src2,
        lambda: d2["10.0"]["shares"]["radius"])
    add("§5.1", "dimension share at γ₀=0.1", 0.530, "fraction of total motion", src2,
        lambda: d2["0.1"]["shares"]["dimension"],
        quote="dimension term falls from 0.53 to 0.44")
    add("§5.1", "dimension share at γ₀=10", 0.437, "fraction of total motion", src2,
        lambda: d2["10.0"]["shares"]["dimension"])
    add("§5.1", "arms with a positive utility term, γ₀≥1", 0.0, "fraction", src2,
        lambda: max(d2[k]["sign_positive_fraction"]["utility"] for k in ("1.0", "3.0", "10.0")))
    add("§5.1", "center-collapse share at γ₀=0.1", 0.053, "fraction of Δ log R_eff", src2,
        lambda: d2["0.1"]["center_share_median"])
    add("§5.1", "center-collapse share at γ₀=3", 0.442, "fraction of Δ log R_eff", src2,
        lambda: d2["3.0"]["center_share_median"])
    add("§5.1", "center-collapse share at γ₀=10", 0.450, "fraction of Δ log R_eff", src2,
        lambda: d2["10.0"]["center_share_median"])
    add("§5.1", "panel (d) coverage at γ₀≥1", 1.0, "fraction of cells", src2,
        lambda: min(d2[k]["center_coverage"] for k in ("1.0", "3.0", "10.0")))
    add("§5.1", "arms behind each richness level", 120, "arm-comparisons", src2,
        lambda: d2["10.0"]["n"])

    # --- §5.1, the benign corner --------------------------------------------
    srcb = sidecar(man, "fig2_benign")
    add("§5.1", "S-HH capacity change at γ₀=1 (largest on the grid)", 11.8, "α noise floors", srcb,
        lambda: d2b["1.0"]["floors"], quote="γ₀ = 1 with 11.8 floors", n=40)
    add("§5.1", "S-HH capacity change at γ₀=10", 6.2, "α noise floors", srcb,
        lambda: d2b["10.0"]["floors"], quote="falling to 6.2 by γ₀ = 10", n=40)
    add("§5.1", "S-HH Δ log α at γ₀=1", 0.2194, "log units", srcb,
        lambda: d2b["1.0"]["dlog_alpha"])
    add("§5.1", "S-HH Δ log α at γ₀=10", 0.1141, "log units", srcb,
        lambda: d2b["10.0"]["dlog_alpha"])

    # --- §5.1, Tier 2: generality, channel signature, and the peak we decline to locate ------
    srct = "results/tier2_backward_transfer.json"
    t2 = json.loads((ROOT / srct).read_text())
    g10 = t2["2.2_generality"]["10"]
    add("§5.1", "S-HH gain at matched lag 4, task 0, γ₀=10", 8.7, "α noise floors", srct,
        lambda: g10["position_at_matched_lag_4"]["0"]["floors"],
        quote="+8.7, +12.6 and +12.8 floors for tasks", n=40)
    add("§5.1", "S-HH gain at matched lag 4, task 8, γ₀=10", 12.8, "α noise floors", srct,
        lambda: g10["position_at_matched_lag_4"]["8"]["floors"], n=40)
    add("§5.1", "S-HH gain within task 0 at lag 15, γ₀=10", 3.1, "α noise floors", srct,
        lambda: g10["lag_within_task_0"]["15"]["floors"],
        quote="+8.7, +9.0, +6.2 and +3.1 floors at lags", n=40)
    add("§5.1", "S-HH (task, lag) cells resolving as gains at γ₀=10", 10, "of 10 cells", srct,
        lambda: g10["n_cells_resolved"], quote="All ten (task, lag)", n=400)
    add("§5.1", "S-HH signed utility share of the net gain at γ₀=10", 0.484, "share", srct,
        lambda: t2["2.1_channel_composition"]["10"]["gain_signed_shares"]["utility"],
        quote="+0.48 utility and +0.62 dimension against −0.11 radius", n=40)
    add("§5.1", "S-HH signed radius share of the net gain at γ₀=10", -0.108, "share", srct,
        lambda: t2["2.1_channel_composition"]["10"]["gain_signed_shares"]["radius"], n=40)
    # Quoted only as a number we decline to promote to a location claim: it moves with lag.
# Quoted only as a number we decline to promote to a location claim: it moves with lag.
    add("§5.1", "interpolated peak of the S-HH gain at lag 12", 1.23, "γ₀", srct,
        lambda: t2["2.3_peak_location"]["interpolated_peak"],
        quote="puts it at 1.23 with a paired-bootstrap interval", n=8)
    add("§5.1", "S-HH peak CI lower at unique n=8", 1.12, "γ₀",
        "results/unique_n8.json",
        lambda: u8["peak_lag12"]["ci_unique"][0],
        quote="[1.12, 1.35]", n=8)
    add("§5.1", "peak of the S-HH gain at lag 4", 2.86, "γ₀", srct,
        lambda: t2["2.3_peak_location"]["checks"]["per_lag"]["4"]["interpolated_peak"],
        quote="2.86, 1.75, 1.23 and 0.96 at lags", n=40)
    add("§5.1", "peak of the S-HH gain at lag 15", 0.96, "γ₀", srct,
        lambda: t2["2.3_peak_location"]["checks"]["per_lag"]["15"]["interpolated_peak"], n=40)

    # --- §5.3, lag robustness ------------------------------------------------
    srcl = sidecar(man, "fig2_lag4")
    add("§5.3", "largest share difference, lag 12 vs lag 4 (task 0)", 0.032, "share", srcl,
        lambda: max(abs(d2[k]["shares"][f] - d2l4[k]["shares"][f])
                    for k in ("0.3", "1.0", "3.0", "10.0")
                    for f in ("utility", "radius", "dimension")),
        quote="shares agree to within 0.032")
    add("§5.3", "magnitude ratio, lag 12 : lag 4 within task 0, at γ₀=10", 1.27, "ratio", srcl,
        lambda: d2["10.0"]["dlog_alpha"] / d2l4["10.0"]["dlog_alpha"],
        quote="the magnitude changes by 1.27×")

    # --- §5.3, the lag/task confound. Three-corner, from the propagation audit; the drafted
    # 1.34× / 1.7× pair was four-corner and is superseded.
    srcp = "results/audit_propagation.json"
    add("§5.3", "lag effect within task 0 at γ₀=10 (lag 4 → 15)", 1.29, "ratio", srcp,
        lambda: ap["w5"]["lag_series"]["10|3-corner|ratio"],
        quote="forgetting grows 1.29× from lag 4 to 15", n=360)
    add("§5.3", "task-position effect at fixed lag 4, γ₀=10 (task 8 → 0)", 1.56, "ratio", srcp,
        lambda: ap["w5"]["task_series"]["10|3-corner|ratio"],
        quote="it grows 1.56× from task 8 to task 0", n=360)

    # --- §B, the variance decomposition. Three-corner throughout: the four-corner version was
    # largely encoding the sign difference, so §B was rewritten rather than renumbered.
    v3 = ap["variance"]["3-corner"]["variance"]
    add("§B", "variance in forgetting explained by stream instantiation alone", 0.000, "R²", srcp,
        lambda: v3["stream alone"], quote="R^2 = 0.000 is tautological", n=1200)
    add("§B", "variance explained by initialization seed alone", 0.007, "R²", srcp,
        lambda: v3["seed alone"], quote="R^2 = 0.007", n=1200)
    add("§B", "variance explained by distance above the asymptote + lag", 0.589, "R²", srcp,
        lambda: v3["distance + lag"], quote="accounts for 0.589", n=1200)
    add("§B", "variance explained by stream condition alone", 0.523, "R²", srcp,
        lambda: v3["condition alone"], quote="the stream occupies for 0.523", n=1200)
    add("§B", "variance explained by condition + distance + lag", 0.882, "R²", srcp,
        lambda: v3["condition + distance + lag"], quote="with 0.882 between them", n=1200)
    add("§B", "residual sd after all five predictors", 0.143, "log units", srcp,
        lambda: v3["residual_sd"], quote="0.143 in log units", n=1200)
    add("§B", "total sd of per-arm forgetting at γ₀=10", 0.436, "log units", srcp,
        lambda: v3["total_sd"], quote="against a total of 0.436", n=1200)

    # --- §5.3, width. Pools all four conditions on both sides: a like-for-like contrast across
    # widths, so it certifies invariance rather than the absolute share levels of §5.1.
    srcw = sidecar(man, "fig_width")
    wc = lambda n: fw["cells"][f"N{n}_g10"]  # noqa: E731
    add("§5.3", "forgetting at γ₀=10, N=150", 47.9, "α noise floors", srcw,
        lambda: wc(150)["floors"])
    add("§5.3", "forgetting at γ₀=10, N=600", 47.5, "α noise floors", srcw,
        lambda: wc(600)["floors"])
    add("§5.3", "utility share at γ₀=10, N=150", 0.472, "share", srcw,
        lambda: wc(150)["shares"]["utility"])
    add("§5.3", "utility share at γ₀=10, N=600", 0.438, "share", srcw,
        lambda: wc(600)["shares"]["utility"])
    add("§5.3", "radius share at γ₀=10, N=150", 0.139, "share", srcw,
        lambda: wc(150)["shares"]["radius"])
    add("§5.3", "radius share at γ₀=10, N=600", 0.112, "share", srcw,
        lambda: wc(600)["shares"]["radius"])
    add("§5.3", "dimension share at γ₀=10, N=150", 0.389, "share", srcw,
        lambda: wc(150)["shares"]["dimension"])
    add("§5.3", "dimension share at γ₀=10, N=600", 0.450, "share", srcw,
        lambda: wc(600)["shares"]["dimension"])

    # --- §5.3, width at the three-corner grouping, which is what §5.1's levels are quoted at.
    # `n` is the point of these rows: the middle line is 16 arms, not 160.
    w3 = lambda n: abs(ap["width"][f"10|{n}"]["three"]["floors_signed"])  # noqa: E731
    add("§5.3", "forgetting at γ₀=10, N=150 (three-corner)", 65.7, "α noise floors", srcp,
        lambda: w3("150"), n=12)
    add("§5.3", "forgetting at γ₀=10, N=300 matched subset (three-corner)", 66.6,
        "α noise floors", srcp, lambda: w3("300 (matched)"), n=12)
    add("§5.3", "forgetting at γ₀=10, N=600 (three-corner)", 65.6, "α noise floors", srcp,
        lambda: w3("600"), n=12)
    # Computed at the three-corner grouping, matching the spread it is compared against. At four
    # corners the same gap is 5.1%, and quoting the two at different groupings would make the
    # "three times the spread" comparison meaningless.
    add("§5.3", "matched-subset vs full-grid gap at N=300, γ₀=10 (duplicate-stream; superseded)",
        4.5, "% of the full-grid value",
        srcp, lambda: 100 * abs(
            ap["width"]["10|300 (matched)"]["three"]["floors_signed"]
            / ap["width"]["10|300 (full grid)"]["three"]["floors_signed"] - 1),
        n="12 vs 120")
    add("§5.3", "utility-share spread across widths at γ₀=10 (three-corner, duplicate-stream)", 0.024, "share",
        srcp, lambda: max(ap["width"][f"10|{n}"]["three"]["shares"]["utility"]
                          for n in ("150", "300 (matched)", "600"))
        - min(ap["width"][f"10|{n}"]["three"]["shares"]["utility"]
              for n in ("150", "300 (matched)", "600")))
    wn = load(RES / "width_g10_n40.json")
    add("§5.3", "forgetting spread N=150 vs 600 at γ₀=10 (genuine n=40)", 0.62, "%",
        "results/width_g10_n40.json",
        lambda: wn["three_corner_spread_pct"],
        quote="by 0.62%", n=120)
    add("§5.3", "alignment-share spread at γ₀=10 (genuine n=40)", 0.020, "share",
        "results/width_g10_n40.json",
        lambda: wn["utility_share_delta"],
        quote="alignment share by 0.020", n=120)
    add("§5.3", "radius share at γ₀=10, N=150 (genuine n=40, three-corner)", 0.146, "share",
        "results/width_g10_n40.json",
        lambda: wn["radius_share_150"],
        quote="0.146", n=120)
    add("§5.3", "radius share at γ₀=10, N=600 (genuine n=40, three-corner)", 0.098, "share",
        "results/width_g10_n40.json",
        lambda: wn["radius_share_600"],
        quote="0.098", n=120)
    add("§5.3", "dimension share at γ₀=10, N=150 (genuine n=40, three-corner)", 0.394, "share",
        "results/width_g10_n40.json",
        lambda: wn["dimension_share_150"],
        quote="0.394", n=120)
    add("§5.3", "dimension share at γ₀=10, N=600 (genuine n=40, three-corner)", 0.463, "share",
        "results/width_g10_n40.json",
        lambda: wn["dimension_share_600"],
        quote="0.463", n=120)
    add("§5.3", "radius+dimension share at N=150 (genuine n=40)", 0.540, "share",
        "results/width_g10_n40.json",
        lambda: wn["radius_plus_dimension_150"],
        quote="0.540", n=120)
    add("§5.3", "radius+dimension share at N=600 (genuine n=40)", 0.560, "share",
        "results/width_g10_n40.json",
        lambda: wn["radius_plus_dimension_600"],
        quote="0.560", n=120)
    k3v = load(RES / "unconfound_k3.json")
    add("§B", "stream R² on the crossed arrangement×init run", 0.001, "R²",
        "results/unconfound_k3.json",
        lambda: k3v["stream_R2_gamma10"],
        quote="measures stream R^2 = 0.001", n=3600)

    # --- the sign audit. One number, and it is the one §5.1's backward-transfer claim rests on.
    add("§5.1", "cells examined for a capacity gain", 520, "cells", srcp,
        lambda: ap["sign_audit"]["n_cells"])
    add("§5.1", "resolvable capacity gains outside S-HH", 0, "cells", srcp,
        lambda: len(ap["sign_audit"]["resolved_gains_outside_benign"]))

    # --- §5.4, from Figure 4 -------------------------------------------------
    src4 = f"figures/{[n for n in man['figures']['fig4']['outputs'] if n.endswith('.json')][0]}"
    eff = f4["effects_by_gamma"]["10.0"]
    add("§5.4", "S-HL Δρ_c at γ₀=10 (the decorrelating corner)", -0.0550, "ρ_c", src4,
        lambda: f4["rich"]["S-HL"]["delta_mean"])
    add("§5.4", "S-LH Δρ_c at γ₀=10 (largest movement)", 0.1098, "ρ_c", src4,
        lambda: f4["rich"]["S-LH"]["delta_mean"])
    add("§5.4", "S-LL Δρ_c at γ₀=10 (unresolved)", 0.0114, "ρ_c", src4,
        lambda: f4["rich"]["S-LL"]["delta_mean"])
    add("§5.4", "S-LL unique sign-test p at γ₀=10", 0.73, "p",
        "results/unique_n8.json",
        lambda: u8["fig4"]["10"]["cells"]["S-LL"]["p_unique"],
        quote="unique sign test p = 0.73", n=8)
    add("§5.4", "S-LL file sign-test p at γ₀=10 (copy-inflated)", 0.15, "p", src4,
        lambda: f4["rich"]["S-LL"]["sign_test_p"],
        quote="copy-inflated sign test p = 0.15")
    add("§5.4", "lazy-arm drift band, lower", 0.0051, "ρ_c", src4,
        lambda: f4["baseline"]["band"][0])
    add("§5.4", "lazy-arm drift band, upper", 0.0101, "ρ_c", src4,
        lambda: f4["baseline"]["band"][1])
    add("§5.4", "S-HL arms declining at γ₀=10", 1.0, "fraction", src4,
        lambda: f4["rich"]["S-HL"]["fraction_declining"])
    add("§5.4", "S-HL median Spearman(ρ_c, block) at γ₀=10", -0.79, "ρ", src4,
        lambda: f4["rich"]["S-HL"]["spearman_median"])
    add("§5.4", "readout-similarity main effect at γ₀=10", 0.1043, "ρ_c", src4,
        lambda: eff["readout"])
    add("§5.4", "feature-similarity main effect at γ₀=10", -0.0606, "ρ_c", src4,
        lambda: eff["feature"])
    add("§5.4", "interaction at γ₀=10", 0.0059, "ρ_c", src4,
        lambda: eff["interaction"])
    add("§5.4", "additive-model error on the held-out corner", 0.0117, "ρ_c", src4,
        lambda: abs(eff["additive_prediction_SHH"] - eff["measured_SHH"]))
    add("§5.4", "all four corners start together (spread)", 0.0014, "ρ_c", src4,
        lambda: max(f4["rich"][c]["rho_c_first"] for c in f4["rich"])
        - min(f4["rich"][c]["rho_c_first"] for c in f4["rich"]))

    # --- §5.4, the γ₀=5 onset probe. The arm-level statistics below are real but are *not* the
    # onset evidence: 5×8 crossed arms carry five arrangement draws, and the onset is a claim about
    # arrangements. They are pinned to the long draft, which says so; the paper quotes the
    # arrangement-level reading. See docs/14-claim-security.md §6, results/seed_security.json.
    g5 = load(RES / "gamma5_onset.json")
    g5n40 = load(RES / "gamma5_n40_onset.json")
    ss = load(RES / "seed_security.json")
    shl5 = ss["gamma5_onset"]["corners"]["S-HL"]
    add("§5.4", "S-HL Δρ_c at γ₀=5 (arm-level; not the onset evidence)", -0.0102, "ρ_c",
        "results/gamma5_n40_onset.json",
        lambda: g5n40["by_gamma"]["5"]["S-HL"]["delta_mean"],
        quote="decisive: −0.0102 ± 0.0032", n=40)
    add("§5.4", "S-HL sign-test p at γ₀=5 (arm-level; not the onset evidence)", 0.00068, "p",
        "results/gamma5_n40_onset.json",
        lambda: g5n40["by_gamma"]["5"]["S-HL"]["sign_test_p"],
        quote="sign-test p = 6.8×10⁻⁴", n=40)
    add("§5.4", "S-HL arms declining at γ₀=5 (arm-level; not the onset evidence)", 31, "of 40",
        "results/gamma5_n40_onset.json",
        lambda: g5n40["by_gamma"]["5"]["S-HL"]["fraction_declining"]
                * g5n40["by_gamma"]["5"]["S-HL"]["n_arms"],
        quote="declining on 31 of 40", n=40)
    add("§5.4", "S-HL arrangements declining at γ₀=5 (5×8 set)", 3, "of 5 arrangements",
        "results/seed_security.json",
        lambda: shl5["arrangement_unit"]["sign_test"]["n_declining"],
        quote="three of five", n=5)
    add("§5.4", "S-HL arrangements declining at γ₀=5 (8-arrangement precommit)", 4,
        "of 8 arrangements", "results/gamma5_k8.json",
        lambda: load(RES / "gamma5_k8.json")["S-HL_n_arrangements_negative"],
        quote="four of eight", n=8)
    add("§5.4", "S-HL γ₀=5 interval over 8 arrangements, lower", -0.023, "ρ_c",
        "results/gamma5_k8.json",
        lambda: load(RES / "gamma5_k8.json")["S-HL_t_ci_95"][0],
        quote="over arrangements $[-0.023, +0.010]$", n=8)
    add("§5.4", "S-HL γ₀=5 interval over 8 arrangements, upper", 0.010, "ρ_c",
        "results/gamma5_k8.json",
        lambda: load(RES / "gamma5_k8.json")["S-HL_t_ci_95"][1], n=8)
    add("§5.4", "S-HL γ₀=5 interval over 5 arrangements, lower (superseded as onset evidence)",
        -0.037, "ρ_c",
        "results/seed_security.json",
        lambda: shl5["arrangement_unit"]["spread"]["t_ci_95"][0], n=5)
    add("§5.4", "S-HL γ₀=5 interval over 5 arrangements, upper (superseded as onset evidence)",
        0.016, "ρ_c",
        "results/seed_security.json",
        lambda: shl5["arrangement_unit"]["spread"]["t_ci_95"][1], n=5)
    add("§5.4", "S-HL γ₀=5 SEM over arrangements vs over arms", 3.0, "ratio",
        "results/seed_security.json",
        lambda: shl5["arrangement_unit"]["spread"]["sem"]
                / shl5["arm_unit"]["sem_assuming_independent_arms"], n=5)
    add("§5.4", "S-HL unique seeds declining at γ₀=5 (duplicate-stream set)", 3,
        "of 4 unique seeds", "results/unique_n8.json",
        lambda: u8["gamma5"]["S-HL"]["n_declining_unique"],
        quote="3 of 4 unique", n=4)
    add("§5.4", "registered-grid onset bracket (γ₀ = 3 → 10)", 3.3, "ratio",
        "results/gamma5_onset.json",
        lambda: g5["onset"]["previous_bracket_factor"],
        quote="a factor of 3.3")
    add("§5.4", "S-HL unique seeds declining at γ₀=3", 4, "of 8 unique seeds",
        "results/unique_n8.json",
        lambda: u8["fig4"]["3"]["cells"]["S-HL"]["n_declining_unique"],
        quote="4 of 8 unique", n=8)
    add("§5.4", "S-HL unique seeds declining at γ₀=10", 8, "of 8 unique seeds",
        "results/unique_n8.json",
        lambda: u8["fig4"]["10"]["cells"]["S-HL"]["n_declining_unique"],
        quote="8 of 8 unique seeds", n=8)
    add("§5.4", "S-HL unique sign-test p at γ₀=10", 0.0078, "p",
        "results/unique_n8.json",
        lambda: u8["fig4"]["10"]["cells"]["S-HL"]["p_unique"],
        quote="p=0.0078, the floor of a two-sided sign test", n=8)

    # --- §5.4, what K=3 says about the magnitude. Sign robust on both units; size is not.
    k3s = ss["k3_arrangement_component"]
    add("§5.4", "S-HL inits declining in every K=3 arrangement", 40, "of 40",
        "results/seed_security.json",
        lambda: k3s["S-HL_inits_declining_in_all_arrangements"],
        quote="all 40 initialisations decline in each of its three arrangements", n=120)
    add("§5.4", "S-HL Δρ_c smallest across K=3 arrangements", -0.026, "ρ_c",
        "results/seed_security.json",
        lambda: k3s["corners"]["S-HL"]["spread"]["max"],
        quote="magnitude runs from $-0.026$", n=3)
    add("§5.4", "S-HL Δρ_c largest across K=3 arrangements", -0.051, "ρ_c",
        "results/seed_security.json",
        lambda: k3s["corners"]["S-HL"]["spread"]["min"],
        quote="$-0.026$ to $-0.051$", n=3)
    add("§5.1", "unique seeds behind each three-corner richness level", 24,
        "unique seeds", "results/unique_n8.json",
        lambda: 3 * u8["unique_seeds_per_cell"],
        quote="24 unique seeds", n=24)
    add("§5.1", "S-HH radius mean in SEM8 at γ₀=1", 3.9, "SEM8",
        "results/unique_n8.json",
        lambda: u8["shh_radius"]["1"]["mean_over_sem"],
        quote="3.9 SEM from zero", n=8)
    add("§5.1", "S-HH radius mean in SEM8 at γ₀=10", 2.5, "SEM8",
        "results/unique_n8.json",
        lambda: u8["shh_radius"]["10"]["mean_over_sem"],
        quote="and 2.5 at", n=8)
    g30u = load(RES / "gamma30_unique.json")
    add("§5.4", "γ=30 interaction unique SEM ratio", 1.74, "SEM",
        "results/gamma30_unique.json",
        lambda: g30u["interaction_over_sem_unique"],
        quote="1.74 SEM", n=4)
    add("§5.4", "γ=30 interaction file SEM ratio (copy-inflated)", 3.9, "SEM",
        "results/gamma30_unique.json",
        lambda: g30u["interaction_over_sem_files"],
        quote="3.9 SEM figure was", n=16)

    # --- §5.1, S-HH radius bound (the unresolved series, not a shared-channel sign)
    add("§5.1", "S-HH radius term at γ₀=1 in R_eff floors", 1.9, "R_eff noise floors", srcb,
        lambda: abs(d2b["1.0"]["terms"]["radius"]) / __import__(
            "numpy").log1p(__import__("src.analysis.attribution",
                                      fromlist=["x"]).R_EFF_FLOOR_CV[1.0]),
        quote="1.9, 0.2 and 1.1 floors", n=40)
    add("§5.1", "S-HH radius term at γ₀=3 in R_eff floors", 0.2, "R_eff noise floors", srcb,
        lambda: abs(d2b["3.0"]["terms"]["radius"]) / __import__(
            "numpy").log1p(__import__("src.analysis.attribution",
                                      fromlist=["x"]).R_EFF_FLOOR_CV[3.0]), n=40)
    add("§5.1", "S-HH radius term at γ₀=10 in R_eff floors", 1.1, "R_eff noise floors", srcb,
        lambda: abs(d2b["10.0"]["terms"]["radius"]) / __import__(
            "numpy").log1p(__import__("src.analysis.attribution",
                                      fromlist=["x"]).R_EFF_FLOOR_CV[10.0]), n=40)

    # --- measurement floors --------------------------------------------------
    pg = mn["per_gamma"]
    cv = lambda gam, ch: float(np.sqrt(  # noqa: E731
        (pg[gam]["cells"]["A|retained"][ch]["cv"] ** 2
         + pg[gam]["cells"]["B|retained"][ch]["cv"] ** 2) / 2))
    add("floors", "registered α floor (CV)", 0.0187, "CV", "src/analysis/timewarp.py",
        lambda: __import__("src.analysis.timewarp", fromlist=["x"]).NOISE_FLOOR_CV["alpha"])
    add("floors", "measured R_eff floor at γ₀=0.03", 0.0028, "CV", "results/measurement_null.json",
        lambda: cv("0.03", "R_eff"))
    add("floors", "measured R_eff floor at γ₀=10", 0.0112, "CV", "results/measurement_null.json",
        lambda: cv("10", "R_eff"))
    add("floors", "panel (d) gate, in floors", 3.0, "multiple of the floor",
        "src/analysis/attribution.py",
        lambda: __import__("src.analysis.attribution", fromlist=["x"]).FLOOR_GATE_K)

    # --- provenance. Read out of the audit's own evidence strings, so these cannot claim a
    # cleaner grid than the audit last certified.
    def audit_says(fragment: str) -> str:
        for row in au["rows"] if "rows" in au else au.get("checks", []):
            text = " ".join(str(x) for x in (row.values() if isinstance(row, dict) else row))
            if fragment in text:
                return text
        raise KeyError(f"audit has no check mentioning {fragment!r}")

    add("audit", "arms in the registered grid", 1280, "arms", "results/audit_scope.json",
        lambda: float(audit_says("completeness").split("completeness")[1].split("/")[0]))
    add("audit", "arms usable", 1280, "arms", "results/audit_scope.json",
        lambda: float(audit_says("on disk,").split("on disk,")[1].split("usable")[0]))
    add("audit", "worst identity residual", 4.44e-16, "log units", "results/audit_scope.json",
        lambda: float(audit_says("identity residual").split("max ")[1].split(" ")[0]))
    return R


# The submitted files, in the order a quote is looked for. `paper/` is the paper; `07-writeup.md`
# is the long draft it was cut from and is no longer what gets sent.
SUBMITTED = ("paper/body.tex", "paper/figures.tex", "paper/figures-appendix.tex")
LONG_DRAFT = "docs/07-writeup.md"

# Enough of a unicode->ascii map to let a fragment written in the markdown draft be recognised in
# LaTeX source. Only characters that actually occur in the quotes are here; an unmapped character
# survives normalisation and simply fails to match, which is the safe direction.
_SYMBOLS = {
    "γ": "gamma", "ρ": "rho", "α": "alpha", "Ψ": "psi", "ψ": "psi", "Δ": "delta",
    "σ": "sigma", "β": "beta", "η": "eta", "²": "2", "⁻": "-", "₀": "0", "×": "x",
    "−": "-", "–": "-", "—": "-", "≥": ">=", "≤": "<=", "≈": "~", "\u00a0": " ",
    "“": '"', "”": '"', "’": "'",
}


def normalise(text: str) -> str:
    r"""Reduce prose to a form in which markdown and LaTeX say the same thing.

    The prose-pinning quotes were written against the markdown draft; the submitted file is LaTeX.
    `falls by 69.7 floors at γ₀ = 10` and `falls by 69.7 floors\nat $\gamma_0 = 10$` are the same
    sentence and must compare equal, so: expand the symbols, drop LaTeX markup and math delimiters,
    lowercase, and collapse every run of punctuation or whitespace to one space. Line breaks go
    with it, which also fixes the old check's inability to see a quote spanning a wrapped line.
    """
    for k, v in _SYMBOLS.items():
        text = text.replace(k, v)
    text = re.sub(r"\\(?:textbf|textit|emph|texttt|mathrm|text)\s*\{", "{", text)
    text = re.sub(r"\\[a-zA-Z]+", lambda m: m.group(0)[1:], text)   # \gamma -> gamma
    text = re.sub(r"[^0-9a-zA-Z.\-]+", " ", text.lower())
    return " " + re.sub(r"\s+", " ", text).strip() + " "


def line_of(raw: str, needle_norm: str) -> int | None:
    """Line number of the first line whose running normalised window contains the quote."""
    lines = raw.splitlines()
    for i in range(len(lines)):
        window = normalise(" ".join(lines[i:i + 6]))
        if needle_norm.strip() in window:
            return i + 1
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    sources = {}
    for rel in SUBMITTED + (LONG_DRAFT,):
        p = ROOT / rel
        if p.exists():
            raw = p.read_text()
            sources[rel] = (raw, normalise(raw))

    bad, out = [], []
    for row in rows():
        section, name = row["section"], row["name"]
        expected, getter = row["expected"], row["getter"]
        got, where = None, None
        if getter is not None:
            try:
                got = getter()
            except Exception as e:  # noqa: BLE001
                bad.append(f"{section} {name}: getter failed ({e})")
            if got is not None and isinstance(expected, (int, float)) \
                    and not agrees(expected, got):
                bad.append(f"{section} {name}: paper says {expected}, "
                           f"{row['source']} now gives {got:.6g}")
        # The second link: is the value actually what the draft says? A corrected artifact and a
        # stale sentence agree with each other about nothing, and this is where two of our four
        # errors lived.
        in_submission = None
        if row["quote"]:
            needle = normalise(row["quote"])
            for rel in SUBMITTED:
                if rel in sources and needle.strip() in sources[rel][1]:
                    ln = line_of(sources[rel][0], needle)
                    where = f"{rel}:{ln}" if ln else rel
                    in_submission = True
                    break
            else:
                raw, norm = sources.get(LONG_DRAFT, ("", ""))
                if needle.strip() in norm:
                    ln = line_of(raw, needle)
                    where = f"{LONG_DRAFT}:{ln}" if ln else LONG_DRAFT
                    in_submission = False
                else:
                    bad.append(f"{section} {name}: no sentence in the paper or the long draft "
                               f"contains {row['quote']!r} — prose and artifact have diverged")
        out.append({**row, "got": got, "quoted_at": where, "in_submission": in_submission})

    if bad:
        print("DRIFT — the paper and its sources disagree:")
        print("\n".join(f"  {b}" for b in bad))
    else:
        quoted = sum(1 for r in out if r["quoted_at"])
        in_sub = sum(1 for r in out if r["in_submission"])
        only_draft = sum(1 for r in out if r["in_submission"] is False)
        print(f"all {len(out)} numbers match their sources; "
              f"{quoted} also verified against the sentence that quotes them "
              f"({in_sub} in the submitted paper, {only_draft} in the long draft only)")
    if args.check:
        sys.exit(1 if bad else 0)

    lines = [
        "# Number inventory",
        "",
        "Every number that appears in the paper, with the artifact it is read from. Generated by",
        "`scripts/make_inventory.py`, which **re-reads each value from its source and fails if it",
        "has drifted** — so this is a check, not a transcription. Run `--check` before submitting.",
        "",
        "Two links are checked, not one. *source → value* catches an artifact that has moved;",
        "*value → sentence* catches a draft that has not. The `quoted at` column gives the file",
        "and line verified to contain the value; a blank means the number is inventoried but its",
        "prose is not yet pinned. The `n` column is stated wherever arm count is part of",
        "reading the number correctly.",
        "",
        "**Quotes are looked for in the submitted paper first** (`paper/body.tex`,",
        "`paper/figures.tex`, `paper/figures-appendix.tex`) and only then in `docs/07-writeup.md`,",
        "which is the long draft the paper was cut from and is not what gets sent. A row pinned to",
        "`07-writeup.md` is therefore a number that exists in the draft but **not in the paper** —",
        "useful when deciding what a length cut costs. Matching is done on normalised text, so a",
        "fragment written in markdown is recognised in LaTeX and a quote may span wrapped lines.",
        "",
        f"Verified against sources: **{len(out) - len(bad)}/{len(out)}**. "
        f"Prose pinned: **{sum(1 for r in out if r['quoted_at'])}** "
        f"(**{sum(1 for r in out if r['in_submission'])}** in the paper, "
        f"**{sum(1 for r in out if r['in_submission'] is False)}** in the long draft only).",
        "",
    ]
    for sec in dict.fromkeys(r["section"] for r in out):
        lines += [f"## {sec}", "",
                  "| quantity | value | units | n | source | quoted at |",
                  "|---|---|---|---|---|---|"]
        for r in out:
            if r["section"] != sec:
                continue
            val = r["expected"] if isinstance(r["expected"], str) else f"{r['expected']:g}"
            lines.append(f"| {r['name']} | {val} | {r['units']} | {r['n'] or ''} | "
                         f"`{r['source']}` | {r['quoted_at'] or ''} |")
        lines.append("")
    lines += [
        "## Numbers whose source is the log rather than an artifact",
        "",
        "These are recorded in `results/LOG.md` at the entry named, and are not machine-checked",
        "here because they come from one-off analyses rather than a figure's sidecar. Each is",
        "reproducible from the script named in its log entry.",
        "",
        "| quantity | value | log entry |",
        "|---|---|---|",
        "| generic capacity, γ₀=0.03 → 10 | 0.3043 → 0.3750 (11.3 floors) | §2b H2a–H2c |",
        "| retained capacity, γ₀=0.03 → 10 | 0.3129 → 1.3710 (4.4×) | §2b H2a–H2c |",
        "| retained − generic gap, γ₀=0.03 / 10 | +0.0086 / +0.9960 | §2b H2a–H2c |",
        "| generic capacity per corner at γ₀=10 | S-LH 0.2898 (only fall), S-HL 0.4507 "
        "| §2b H2a–H2c |",
        "| probe margin, γ₀=1 → 10 | 0.0673 → 0.0531 | §2b H2a–H2c |",
        "| within-γ capacity/margin correlation at γ₀=10 | +0.322 | §2b H2d |",
        "| error-minimizing richness | γ₀=1 on-grid, 1.61 interpolated | γ* is not identified |",
        "| γ₀=30 generic / retained capacity | 0.4159 / 1.4175 (ratio 3.408) | Two cheap checks |",
        "| γ₀=30 S-HL Δρ_c | −0.096 (4 unique/corner; sign-test p = 0.125) | γ=30 probe |",
        "| γ₀=30 interaction | +0.016, 1.74 unique SEM (does not clear; was 3.9 SEM on 16 files) "
        "| results/gamma30_unique.json |",
        "| behavioral forgetting per corner | S-HH −0.0005…+0.0001, S-HL +0.41 "
        "| Two cheap checks |",
        "| panel (d) γ₀=3 → 10 step | −0.010, CI [−0.038, +0.070] (unresolved) | Panel (d) refit |",
        "",
        "## Held pending a decision, not merely unchecked",
        "",
        "Nothing is currently held. The last entry was §B's variance decomposition, computed on",
        "the four-corner pool where one corner carried the opposite sign; the three-corner values",
        "reordered the table rather than shifting it, so §B was rewritten rather than renumbered",
        "and its numbers are now inventoried above. This section stays in the document because a",
        "number whose status is undecided should be visibly undecided rather than left standing at",
        "its old value.",
        "",
    ]
    OUT.write_text("\n".join(lines))
    print(f"wrote {OUT.relative_to(ROOT)}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
