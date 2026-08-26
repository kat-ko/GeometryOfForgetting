# 13 — Experiment registry

Every run that happened, in one place. `results/LOG.md` is the narrative record and is chronological
and interleaved with reasoning; this is the index. Compiled 2026-08-13 from a full read of the LOG
(3414 lines), the 17 `scripts/run_*.py` entry points, and the contents of `results/`.

**Answering Q1 of `11-role-and-open-questions.md`: no such document existed before this one.**

Column meanings. **arms** is completed measured arms, not evaluations. **paper?** is whether any
number from the run reaches `paper/body.tex` or `docs/07-writeup.md` — `—` means the run informed a
decision without contributing a quoted number, which is true of most of the validation tier.

---

## Validation tier — estimator and manipulation checks

Run before the grid, to establish that the measurement and the manipulation both do what they claim.
None of these produced a number quoted in the paper; all of them gate one that is.

| run | purpose | key parameters | arms / evals | date | results | paper? |
|---|---|---|---|---|---|---|
| **Gate 1** `run_glue_core_recovery.py` | ground-truth recovery for `src/glue/core.py` (§B.5) | P=2, M=200, N=1000, n_t=200, 3 seeds; 162 s | 180 estimations | 08-11 | `results/glue_core_recovery.json` | — (read by `tests/test_analysis_attribution.py`) |
| **Gate 2** α_core vs α_sim | cross-check the two capacity channels | P=8, d=200, D=4, R=1, M=60 | — | 08-11 | same file | — |
| **Gate 2a** `run_gate_2a.py` | capacity validity over the thermodynamic-limit knobs | P∈{8,16,32} × N∈{300,600,1200}, β=0; 411 s | 81 estimations | 08-11 | `results/gate_2a.json` | — (fixed P=16, N=300) |
| **§2c diagnostic** `run_psi_eff_diagnostic.py` | replicaMFT identity vs GLUE core | n_t=200; P∈{8,16} | — | 08-11 | `results/psi_eff_diagnostic.json` | — reclassified from gate to diagnostic |
| **P2a** center policy | `"all"` vs `"active"` anchor policy | §B.5 protocol; n_t∈{200,1000} | 180 estimations | 08-11 | merged into `glue_core_recovery.json` | — |
| **P2b** `run_scale_compression.py` | is under-recovery of large D, R bias or sampling? | P∈{2,8,16}; M=50…800; D sweep | — | 08-11 | `results/scale_compression.json` | caveat in methods |
| **Estimator follow-ups** `run_estimator_followups.py` | N-padding artifact; M cost; MDE at 5 vs 8 seeds | P=16; M∈{150,400,800} | — | 08-11 | `results/estimator_followups.json` | — (chose 8 seeds) |
| **Mode constancy** `run_mode_constancy.py` | is the `pairwise`/`full_P` offset constant? | 9 geometries × 3 seeds × 2 modes | 54 | 08-11 | `results/mode_constancy.json` | — (made `full_P` mandatory) |
| **P1** `run_cost_model.py` | Phase 1 feasibility and per-channel noise floors | P=16, M=150, N=300, n_t∈{50…800} | — | 08-11 | `results/cost_model.json` | registered floors, quoted in methods |
| **Probe saturation** `run_probe_check.py` | is the probe-dichotomy measure saturated? | γ∈{0.03,1,10}; margin vs accuracy vs held-out | — | 08-11 | `results/probe_check.json` | — (selected `margin` for H2d) |
| **Ψ_eff bound** `run_psi_bound_check.py` | is retained Ψ_eff > 1 a defect or a property? | grid arms + synthetic | — | 08-12 | `results/psi_bound_check.json` | scoped in methods |

The first pass of **mode constancy** was run at d=150 and redone at N=300; the first pass of Phase 0
**check 3** was run under accuracy stopping, failed, and was redone under matched-loss stopping. Both
supersessions are in the LOG.

## Phase 0 — manipulation validation

Five registered checks. `results/phase0.json` holds four; the fifth is `results/timewarp.json`.
Common parameters: P=16, N=300, M=150, d=150, D=4, R=1, n_t=200, `full_P`, 3 seeds.

| # | check | pass criterion | result | date | key | paper? |
|---|---|---|---|---|---|---|
| 1 | capacity-at-init tracks `a` | monotone, range ≥ 2× floor | α 0.3091 → 0.3940; rank corr +1.00; range 23.97% vs 3.74% threshold | 08-11 | `gate2_alignment` | — |
| 2 | capacity-at-init flat in γ | variation within floor | representations bitwise identical; α = 0.310595 at both γ=0.03 and γ=10 | 08-11 | `gamma_flat` | — |
| 3 | richness separation ‖ΔW‖/‖W‖ | ≥ 1 order of magnitude | 2.53 decades at matched loss; 39.5× steps | 08-11 | `richness_separation` | — |
| 4 | `pairwise` vs `full_P` | agree within floor on α, R_eff, D_eff, ρ_c | α agrees to ~1%; D_eff and Ψ_eff diverge ~26% | 08-11 | `estimation_mode` | — |
| 5 | time reparameterisation `run_timewarp.py` | residual after warping **exceeds** floor | γ=1 vs 10 DTW residual 11.54 floors; lazy-arm excursion 0.2 floors — **H3 survives** | 08-11 | `results/timewarp.json` | §5.5 (H3 gate) |

**None of the five numeric outcomes appears in `paper/body.tex`.** Check 5 is referenced in
`07-writeup.md` §5.5 as "a Phase 0 gate that would have killed H3"; the rest are cited only as having
passed. See Q2 in `docs/12-record-report.md`.

## Capacity and throughput

| run | purpose | parameters | date | results | paper? |
|---|---|---|---|---|---|
| **Scaling (nnls)** `run_scaling.py` | throughput vs worker count | n_t=20; workers 1…254 | 08-11 | `results/scaling.json` | — superseded |
| **Scaling (colgen)** | real-workload throughput | n_t=200; nnls vs colgen at 128/192/254 workers | 08-11 | `results/scaling_colgen.json` | — **not reproducible from a named entry point**; see Q3 |

## Phase 1 — the registered grid

| run | purpose | parameters | arms | date | results | paper? |
|---|---|---|---|---|---|---|
| **Smoke grid** | wiring check | T=4, P=8, M=40, N=80, n_t=25, 1 seed | 2 | 08-11 | `results/phase1/smoke/` | — |
| **Full grid, first launch** | the registered experiment | 1280 arms, 48,640 evals, 254 workers | **0** | 08-11 | — | **killed** after 79 min with zero arms complete |
| **Pilot** | de-risk the pipeline before a 4 h launch | γ∈{0.03,10} × 4 conditions × 1 seed | 8 | 08-11 | `results/phase1_summary_pilot.json` | — arms **destroyed** by a `--resume` test; summary survived |
| **Full grid** | the registered experiment | 6 γ × 4 conditions × 5 streams × 8 seeds × (a=0 and a>0); P=16, N=300, M=150, T=16 | **1280** (960 at a=0) | 08-11/12 | `results/phase1/*.json`, `results/phase1_summary.json` | **yes — every figure** |
| **Scope audit** `audit_scope.py` | conformance of every arm to the registration | 1280 arms | — | 08-12 | `results/audit_scope.json` | notebook §1 |

Registered richness grid: γ₀ ∈ {0.03, 0.1, 0.3, 1, 3, 10}. Conditions are the Hiratani 2×2:
`S-HH`, `S-HL`, `S-LH`, `S-LL`.

## Robustness and extension arms

| run | purpose | parameters | arms | wall / cost | date | results | paper? |
|---|---|---|---|---|---|---|---|
| **W1** `run_width.py` | does the reorganisation survive changing N? | N∈{150,600} vs 300; γ∈{0.03,1,10}; 4 conditions × 2 streams × 2 seeds | 96 | 58 min; 34.5 core-h (1294 s/arm) | 08-12 | `results/width/*.json` | §5.3 |
| **W4** `run_measurement_null.py` | per-γ estimator noise floor on a fixed representation | 1 arm per γ × 4 measurement seeds | — | 1188 s | 08-12 | `results/measurement_null.json` | §5.3, all floor denominators |
| **γ=30 probe** `run_gamma_ext.py` | does the `S-HL` decorrelation extend past the sweep? | γ=30; 4 conditions × 4 streams × 4 seeds | 64 | 741 s; 10.2 core-h (571 s/arm) | 08-12 | `results/gamma_ext/*.json` | appendix §E |
| **γ=5 probe** `run_gamma_ext.py --gamma 5` | locate the `S-HL` onset inside the 3→10 step | γ=5; same 4×4×4 as γ=30 | 64 | 9.51 core-h (535 s/arm); all usable | 08-17 | `results/gamma_5/*.json`, `results/gamma5_onset.json` | §5.4, fig:corners-gamma3 |

**W2** (lag dependence), **W5/W5a/W5b** (lag decontamination) and the **module-divergence**
exploration are re-poolings of stored geometry with no new compute; they are recorded in
`results/audit_propagation.json`.

## Post-hoc analysis, all zero-compute

| analysis | purpose | date | results | paper? |
|---|---|---|---|---|
| Figure 2, four-corner | γ-sweep attribution | 08-12 | superseded sidecars | — superseded |
| Figure 2, three-corner | same, with the gaining corner removed | 08-12 | `figures/fig2_gamma_sweep__forgetting__*` | **yes**, primary |
| Figure 4 | ρ_c trajectories by corner | 08-12 | `figures/fig4_corners_rho_c__*` | §5.4 |
| H2a–H2d | capacity crossing and probe margin | 08-12 | `src/analysis/ledger.py` §9 | §5.2 |
| γ\* identification | behavioural optimum location | 08-12 | ledger | **yes — reported as not identified** |
| Tier 1 propagation audit `audit_propagation.py` | four-corner vs three-corner regroup | 08-13 | `results/audit_propagation.json` | §A.1, §B |
| Tier 2 `tier2_backward_transfer.py` | channel composition, generality, peak location | 08-13 | `results/tier2_backward_transfer.json` | §5.1 |
| Open questions `audit_open_questions.py` | margins, gates, sign mixtures, between/within, saturation | 08-13 | `results/open_questions.json` | see `docs/12-record-report.md` |

## Planned and not run

| item | status |
|---|---|
| **W6** — rotation measure for H6 | the quantity was never stored; needs a re-run. LOG 2179–2183 |
| **M=800 robustness arm** | planned in the estimator follow-ups; no evidence on disk that it ran |
| **Reinstating `a` as a heterogeneity axis** | Gate 2 passed; scope question raised and not taken |
| **Split-CIFAR100 pilot** | spec only, `docs/10-cifar-pilot-spec.md` |
| **C3/C4 conditions, heterogeneous γ pairs, depth variation, T=40** | deferred to a sequel |
