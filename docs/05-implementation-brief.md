# 05 — Implementation Brief (NeurReps sprint)

```
Status:   RATIFIED. This document is the plan of record through 2026-08-24.
Deadline: NeurReps Extended Abstract, Aug 24 AoE.
          Corrected 2026-08-13 from Aug 22, which this brief carried and
          `09-post-generation-brief.md` contradicted. Aug 24 is authoritative;
          Kati confirmed. The two dates disagreed for three days.
Scope:    Phase 1 (cut), homogeneous configs only. See §7 for what is NOT built.
Overrides: where this brief conflicts with 00/01/02, this brief wins until the
          §1 reconciliation pass makes them agree. After that pass, the specs win.
```

The standing rules in `AGENTS.md` remain in force, including §8.1
(verified/inferred, downgrade-to-question). Flag-and-stop items are listed in §8
of this brief.

---

## 0. Decisions ratified (no further confirmation needed)

| # | Decision | Consequence |
|---|---|---|
| D1 | **Algorithm relabel confirmed.** In arXiv:2503.18114v2: Algorithm 1 = simulated capacity (α_sim); **Algorithm 2 = the geometric estimator** (α, R, D, ρ_c, ρ_a, ψ). Your reading stands; the spec label was wrong. | Apply the relabel everywhere (§1.1) |
| D2 | **β-tilt adopted, staged.** Fixed-`y` retained capacity is implemented first (legitimate: the dichotomy collection Y is analyst-chosen). The tilted ensemble P(y) ∝ exp(β⟨y, y_j⟩) is implemented second, gated on the β-validity sweep. | §1.2, §4.3 |
| D3 | **Ψ_eff identity test is a new validation gate.** Ψ_eff = α · D_eff / (1 + R_eff⁻²) is algebraically derivable from `manifold_analysis_corr` outputs. If validated, the full three-factor attribution runs on public code and GLUE leaves the critical path. | §3.3 |
| D4 | **Gate-2 contingency pre-authorized:** if the alignment knob fails Phase 0, the a-axis is **dropped** for the abstract. Do not debug the geodesic; do not build the input-dim fallback. H1/H2/H6 need only γ. | §5, Day 14 |
| D5 | **Deadline scope:** configs C1/C2 only; Hiratani 2×2 streams; T=16; 5–6 seeds × 10 shared streams; Tier-2 = {α, R_eff, D_eff, ρ_c (both conventions), Ψ_eff}; CCGP/shattering behind a config flag, default off. | §7 |
| D6 | C4 (Version fields), C5 (glue-refinements status), C6-adjacent inventory fixes: proceed as you proposed. | §1.4 |

Terminology, pinned now (update `AGENTS.md` §3 in the reconciliation pass):

- `generic_capacity` — random-dichotomy ensemble (β = 0)
- `retained_capacity` — fixed dichotomy y_j of a past task (the β → ∞ limit)
- `tilted_capacity(beta)` — the interpolation; D2 stage two
- `psi_eff` / "effective utility" — the scalar factor in α; **never** abbreviated `psi`
- `center_axis_alignment` — the pairwise measure ψ_{μν}; **never** written `psi`
- The term "aligned capacity/margin" is retired; grep and replace in docs

---

## 1. Reconciliation pass (docs before code — half a day, do first)

The specs are stale relative to the ratified corrections. Bring them to agreement
in one commit so no code is written against a superseded formula.

### 1.1 Algorithm relabel (D1)

"Algorithm 1" → "Algorithm 2" wherever it refers to the geometric estimator:

- `03-references.md`: §A Chou row; §E.4 table column header and decision line;
  §E.5 heading and body; §F `glue-algorithm.md` line; §G "Resolved" line.
- `03-references.md` lines 20–24: the v1↔v2 renumbering rationale is
  **unverified** (both your reads show the same numbering). Keep the `Version:`
  rule, replace the rationale with: "the estimator is Algorithm 2 in v2;
  v1 numbering unverified."
- `00-math-spec.md` §6 heading/intro ("Algorithm 1 and Definition B.6") and the
  §7 reference ("as in Algorithm 1 Step 1").
- `PROJECT.md` §5.1 if it names the algorithm.
- Do NOT touch `docs/reference/manifold-generator.md`'s naming-flag note; it
  documented the discrepancy correctly and is now historical record.

### 1.2 Apply the 04-corrections that are not yet in the specs

Work through `docs/04-spec-corrections-iclr2026.md`'s apply-order and verify
each is actually applied (several are not):

1. `00` §6.2 → the **exact three-factor identity**
   `α = Ψ_eff · (1 + R_eff⁻²) / D_eff`; remove the UNCERTAIN marker and the
   `(1+R⁻²)/D` two-factor form as "the approximation"; add the a/b/c
   definitions and the substitution proof (in `04` §C1).
2. `00` §6.1 → **stacked-matrix form**: S ∈ R^{P×N}, S_y = diag(y)S,
   pseudo-inverses are **P×P**. Remove the ambiguous per-vector Step-3 notation.
3. `02` §2b → **delete** (there is no approximation to test; the identity is
   exact). §2a (α_sim vs α_mf, mean-field validity) becomes the sole gate.
   Renumber references.
4. `00` §8 → attribution in log space:
   `Δlog α = Δlog Ψ_eff + Δlog(1 + R_eff⁻²) − Δlog D_eff`, exact, zero
   residual. Keep the Wakhloo-duality caveat (decomposition, not mechanism;
   report ρ_a with R).
5. `00` §6.3 → `raw` / `gaussianized` (Wakhloo 2023 Gaussianization), replacing
   `raw` / `standardized`.
6. `00` §7 → replace fixed-`y` "aligned capacity" section with:
   `retained_capacity` (fixed y_j; legitimate analyst choice of Y) and
   `tilted_capacity(beta)`; note stage-two gating.
7. `00` §13 → re-derive the cost model: the QP over anchor points dominates;
   the linear algebra is P×P (16×16), not N×N. Update the cut-order note.
8. `01` §1 config schema → add `estimation_mode: pairwise | full_P`,
   `rho_convention: both` (mandatory; both always computed — see §E.4 two-role rule),
   `beta: float | null`, `ccgp_enabled: bool = false`.
9. `01` Phase 0 → add the pairwise-vs-full-P comparison; keep the
   P×N sweep as restructured (N-first decision order).
10. `02` §2a → add the **β axis** to the sweep (P × N × β), with the margin
    and bisection conventions already pinned there unchanged.
11. `AGENTS.md` §3 → the terminology block from §0 above.
12. `PROJECT.md` §2 P1 → restate: P ≥ 16 is required by **task structure**
    (dichotomy-as-task, factorial labeling), not by estimator validity; drop
    the "P=2 kills GLUE" phrasing (the lab's own pipeline is pairwise).

### 1.3 Vendor one more repo

`chung-neuroai-lab/ood-generalization-geometry` @ pinned SHA →
`third_party/ood-generalization-geometry/`, `.git` stripped, entry in
`VENDORED.md`. Purpose: the **Gaussianization preprocessing implementation**,
GLUE calling conventions, and `environment.yml` (QP solver versions). Tag:
reference, NOT an estimator. Same firewall rules.

### 1.4 Housekeeping (C4/C5)

- Add `Version:` to all five existing reference-sheet headers and the
  `reference/README.md` template (alongside `Derived from:`).
- Fix `glue-refinements.md` status in `03` §F/§G and `reference/README.md`
  (drafted, S1 pending).

**Commit checkpoint:** one commit, message "reconciliation: relabel + 04-corrections".
Everything after this builds against consistent specs.

---

## 2. Build order and module specifications

Language/env per `AGENTS.md` §4 (float64 in geometry, explicit rcond, four
isolated RNGs, config-dataclass-driven, append-only results keyed by
`(config_hash, seed, stream_id, boundary, module)` + estimator version string).

### 2.1 `src/manifolds/` (Day 10–11)

**`generator.py`** — Chou App. D.1.1 exactly (`00` §1 post-reconciliation):

- `make_arrangement(P, d, D, R, rho_C, rho_A, psi_gen, rng) -> Arrangement`
  with axes/centers/noise ~ N(0, I_d/d), coords ~ N(0,1), ε = 1e-2, and the
  **unit-norm normalization of pre-scaled points before R-scaling** (verified
  verbatim: "well-normalized to unit norm"). Correlations via Cholesky of
  C(ρ) = (ρ^|i−j|), applied to centers (ρ_C) and per-axis-index (ρ_A).
- `resample_test_points(arrangement, rng)` — same centers/axes, fresh noise.
- Project parameters: d=150, P=16, M=150; D, R stream variables.

**`labeling.py`** — bijective f: [16] → {0,1}⁴, **labels only, centers random**.
Dichotomy families: `single_factor` (4), `xor` (6), `random_balanced`.

**`dichotomies.py`** —
- `sample_balanced(rng)`, `sample_at_hamming(y, h, rng)` (swap h/2 of the +1s
  with h/2 of the −1s; h even), `readout_similarity(y1, y2)` =
  |2·overlap/P − 1|.

**`streams.py`** —
- Hiratani 2×2: `s_f, s_r ∈ {0.1, 0.9}` → conditions S-HH/S-HL/S-LH/S-LL.
  Secondary: `S-fixed-r` (s_f = 1.0, s_r ∈ {0.1, 0.5, 0.9}).
- Feature similarity: redraw centers at correlation ρ_C = s_f against the
  previous arrangement.
- Record full T×T matrices `S_f`, `S_r`. Control only (t, t−1); long-range is
  covariate, not controlled.
- **Held-out probe y\***: sampled once per stream at recorded s_r distances,
  never trained. Mandatory.
- Streams generated once per `stream_id` from `rng_stream`, shared across all
  configs (blocking factor).
- T = 16.

Tests: `02` §5 verbatim (balanced, even Hamming, similarity targets, sign
symmetry, probe-never-trained, matrices recorded, reproducibility) **plus**
`test_arrangement_correlation_target` (realized center correlation matches ρ_C).

### 2.2 `src/glue/adapters/` (Day 11)

Vendored code stays byte-identical; all corrections live here.

**`mft_adapter.py`** — wraps `replicaMFT.manifold_analysis_corr`:
- Pins κ = 0. Casts to float64. Injects seeding around the vendored global-RNG
  usage (seed set/restore per call from an explicit Generator). Logs n_t and
  the estimator version string `replicaMFT@d56eda1d…`.
- Returns `CapacityResult(alpha, R_eff, D_eff, rho_c_glue, n_t, version)`.
- `n_t = 200` default; `1000` fallback flag for noise-floor use.
- **Derived field:** `psi_eff = alpha * D_eff / (1 + R_eff**(-2))` — marked
  `derived_via_identity=True` until §3.3 validates it.

**`simcap_adapter.py`** — wraps `correlated_capacity.manifold_simcap_analysis`:
- General point-cloud α_sim, κ = 0 (hardcoded upstream — assert it), bisection
  over projection dimension at fixed (P, M). Version string
  `correlated_capacity@de8dac79…`. Same RNG injection.

**`signed_rho.py`** — **VERIFY-FIRST / flag-and-stop.** Signed-normalized ρ_c
needs the anchor centers `s⁰_μ`. Check whether `manifold_analysis_corr` exposes
them (or can with a ≤5-line read of internals). **If centers are not exposable
without modifying the QP path: STOP AND FLAG** — that is **not** a graceful
degradation to abs-only. H1d is untestable under `|·|` alone; report whether the
fix is adapter-level extraction vs. a `third_party/patches/` change (Kati
sign-off) or dropping H1d from the abstract, and wait. If centers *are*
exposable: always compute both
`rho_c_glue = |⟨s⁰_μ, s⁰_ν⟩|` (reporting / comparability) and
`rho_c_signed = ⟨s⁰_μ, s⁰_ν⟩/(‖s⁰_μ‖‖s⁰_ν‖)` (required H1d instrument).
`rho_convention: both` is **mandatory** — no code path may reduce it to one
(`03` §E.4).

**`preprocessing.py`** — Gaussianization ported from the vendored
`ood-generalization-geometry` implementation (cite file path in docstring);
plus identity (`raw`). Every Tier-2 call runs both; divergence is logged as a
finding, never resolved silently.

**`ensembles.py`** — dichotomy ensemble control:
- `generic()` — uniform random balanced y per sample (estimator default).
- `retained(y_j)` — fixed y across all n_t samples (t still varies).
- `tilted(y_j, beta)` — Metropolis or exact enumeration over balanced
  dichotomies at P=16 (C(16,8)=12,870 — **exact enumeration is feasible**;
  prefer it, weights ∝ exp(β·⟨y, y_j⟩)). Stage two; build the interface now,
  wire after the β-validity sweep.

`estimation_mode`: `full_P` (all 16 manifolds jointly) and `pairwise`
(2-manifold subsamples, 100 repetitions, lab protocol). Both implemented;
Phase 0 compares.

### 2.3 `src/measures/` (Day 11–12)

- `pr.py` — participation ratio, effective rank.
- `cka.py` — **linear CKA only** (assert no RBF path).
- `decode.py` — linear decodability of any dichotomy from h_m; fresh-readout
  probe evaluation (train linear head on y\* at each boundary; Johnston–Fusi
  classifier-generalization protocol placeholder: standard logistic decoder,
  exact protocol swap-in when Kati verifies the sheet).
- `drift.py` — rotation (mean sin²θ over principal angles of top-k subspaces,
  k fixed by config, never variance-thresholded), expansion (ΔPR), reuse,
  overwrite per `00` §9.
- `checks.py` — manipulation checks, logged unconditionally per boundary:
  ‖ΔW_m‖_F/‖W_m‖_F, per-module NTK change (finite-sample on the probe set),
  output-variance share, gradient-norm share, capacity-at-init.

### 2.4 `src/models/` (Day 12)

- `parameterization.py` — per-module (γ_m): output scale β_L/γ_m with
  β_L = N^{-1/2}, η_m = η₀·γ_m²·N applied to W_m **and** its readout slice;
  σ² = 1 hidden init regardless of γ (unit test: bitwise identical W across γ);
  zero-init readout (unit test: f(x;θ₀) = 0 exactly);
  `lr_scaling: quadratic | corrected` (corrected: γ^(2/L) for γ > 1, L = 2).
  `test_base_width_equivalence` at N₀ = 64: **derive** the constant, write
  `docs/reference/parameterization-derivation.md`, do not guess.
- `alignment_init.py` — Grassmann geodesic per `00` §5 (principal-angle
  parameterization, NOT lerp+orthonormalize; rcond-logged pseudo-inverse in
  (YᵀU)⁻¹; norm/rank preservation unit tests). Paired-init discipline: one W̃
  per seed from `rng_init_shape`, reused across every config at that seed.
- Architecture: two ReLU modules width N = 300 (C2) or one module width 600
  (C1); single shared readout; no task conditioning. Guardrail tests `02` §7.

### 2.5 `src/train/` (Day 12)

- Plain SGD, no momentum/weight-decay/clipping/normalization (guardrail tests).
- Stopping: `matched_loss` with target from the Day-14 pilot; log full loss
  trajectories regardless.
- Per-boundary: Tier-1 measures + checks; Tier-2 every `tier2_interval = 4`.
- Checkpoint per boundary; idempotent resume; every artifact records resolved
  config, git SHA, estimator versions, RNG states.

### 2.6 `src/analysis/` (Day 13–14, alongside)

- `attribution.py` — log-space three-factor decomposition per past task j:
  Δlog α over (t_j → t), split into Δlog Ψ_eff + Δlog(1+R_eff⁻²) − Δlog D_eff.
  Output: the attribution table (task × boundary × module × component).
- `noise_floors.py` — identifiability floor (independent seeds, same
  condition/timepoint) and trajectory floor (same seed, data order varies) for
  every measure; **8 seeds** (deadline-reduced from 15); MDE table at planned
  seed count → `results/noise_floors.json` **and a human-readable MDE table**
  Kati fills thresholds from.
- `sweeps.py` — §2a harness: α_sim vs α_mf over P ∈ {8,16,32} × N ∈ {300,600}
  (drop N=1200 unless the answer is ambiguous) × β ∈ {0, 0.5, 1, 2, 4, ∞}
  where ∞ = retained. Relative-error surfaces → `results/mft_validity.json`.

---

## 3. Validation sequence (gates, in order — Day 12–13)

Run and pass in this order. Each writes its artifact before the next starts.

### 3.1 §1 ground-truth recovery
B.5 settings exactly: N=1000, P=2, M=200; D 2→10 and R 0.8→2.0 (correlations
zero); then D=4, R=1 fixed while ρ_c, ρ_a, ψ each sweep 0→0.8. At P=2, ρ_a/ψ
recovery is skipped (not exposed by replicaMFT) — record as N/A-pending-GLUE,
not as pass. Sign checks: capacity ↓ in D, R, ρ_c; ↑ in ρ_a (via generator-side
ρ_A at fixed labels — document exactly what is being checked given the
estimator gap).

### 3.2 α_sim smoke test
`simcap_adapter` on the §3.1 spherical cases where α is known/recoverable;
confirms convergence on our manifold class before it is trusted as the §2a
ground truth.

### 3.3 Ψ_eff identity test (`test_psi_eff_identity`) — D3
On one synthetic configuration with controlled geometry:
(a) compute α, R_eff, D_eff via `mft_adapter`; derive Ψ_eff by the identity;
(b) compute E[a], E[b], E[c] directly (own implementation of the three
quadratic forms from the anchor points — this is ~40 lines given the QP
solution, and is P×P algebra);
(c) assert Ψ_eff(derived) ≈ E[c]/E[a] and α ≈ P/E[a] within estimator noise
(same t-samples for both paths, so agreement should be tight).
**Pass →** record in `VENDORED.md`: three-factor attribution is
replicaMFT-complete; GLUE off the critical path.
**Fail →** flag-and-stop; attribution falls back to (α, R, D) two-factor with
Ψ_eff reported as unvalidated-derived; Figure 2 caption changes.

### 3.4 §2a gate: P × N × β sweep
Per §2.6. Decision order pre-registered: raise N first; accept-and-report
second; P→32 last (Kati's call, with the evidence). The **β-validity range**
is the headline output: it determines whether retained/tilted capacity runs on
public code (wide) or Figure 3 waits on GLUE (narrow).

### 3.5 Phase 0 (Day 14)
1. γ-separation: ‖ΔW_m‖ across γ grid separates ≥ 1 order of magnitude —
   else **stop** (parameterization bug).
2. Capacity-at-init flat in γ (identity-init assertion backs this).
3. Gate 2: capacity-at-init monotone in `a` with range ≥ 2× identifiability
   floor. **Fail → D4: drop the a-axis.** No debugging, no fallback build.
4. Time-reparameterization test (`00` §12) on two large-γ homogeneous runs —
   recorded for the paper's H3 scoping note even though H3 is cut.
5. `estimation_mode` comparison: pairwise vs full_P at P=16 on one condition.
6. Matched-loss target selection from pilot loss curves.
7. Noise floors (8 seeds) + MDE table → hand to Kati for thresholds.

---

## 4. Phase 1 (cut) — Days 15–17

### 4.1 Grid

- Configs: **C1** (dense 2N=600, γ swept) and **C2** (two modules, γ_A = γ_B,
  a_A = a_B) only.
- γ ∈ {0.01, 0.03, 0.1, 0.3, 1.0, 3.0} (log-spaced through γ₀\* ≈ 0.1 and the
  transition; extend to 10 only if the corrected-LR arm needs it).
- a ∈ {0.0, 0.5, 1.0} — **omit entirely if Gate 2 failed** (D4).
- Streams: S-HH, S-HL, S-LH, S-LL + S-fixed-r(0.5); T=16.
- 5 seeds × 10 shared streams (raise to 6 seeds only if Day-13 cost model
  allows; never cut streams below 10).
- One arm at `lr_scaling: corrected` on the diagonal γ points (the Graldi
  confound check).

### 4.2 Measurement per run
Tier-1 every boundary; Tier-2 every 4th: α, R_eff, D_eff, Ψ_eff, ρ_c (both
conventions) — generic ensemble AND retained ensemble for every past task j.
Tilted ensemble only if §3.4 returned a usable β range (D2 stage two).
CCGP off (`ccgp_enabled: false`).

### 4.3 Outputs (mapped to figures)
- **Fig 2 (money):** retained-capacity trajectory on task 1 with exact
  three-factor log-attribution, lazy vs rich, with the ICLR-2026
  overcompression signature (low D_eff, low Ψ_eff) marked for the H1c
  comparison.
- **Fig 3:** generic capacity vs γ falling, retained rising, crossing vs
  argmin(average error); probe-y\* metric overlaid as the label-aware
  validation (H2d). If β range was wide: tilt curves as inset.
- **Fig 4:** ρ_c trajectories (signed) across the 2×2 — the Menghi/H1d panel —
  with S-HL vs others for H6. (Heterogeneity Fig 4a is cut with Phase 2.)
- Fig 1 assets: capacity-at-init vs a and vs γ (or the γ manipulation checks
  if a-axis dropped).

Kill criteria live from `01` §5: attribution fails to separate regimes →
rotation/expansion fallback becomes Fig 2; generic capacity fails probe
validation → reported as the negative finding, not suppressed.

---

## 5. Contingency table (pre-decided; execute without asking)

| Trigger | Action |
|---|---|
| Gate 2 fails | Drop a-axis (D4). Fig 1 becomes manipulation-check panel |
| Ψ_eff identity fails | Two-factor attribution; Ψ_eff flagged unvalidated |
| β range narrow | Fixed-y retained only; tilt to full paper; note the quantified GLUE case |
| §2a bad at P=16, good at N=600 | N=600 project-wide; re-time cost model |
| §2a bad at P=16 at all N | **Stop; Kati decides** (P=32 is a design change) |
| GLUE attribution doesn't separate regimes | rotation/expansion is Fig 2; reframe |
| Compute overrun | Cut n_t → tier2_interval → retained-tasks evaluated; never seeds/streams |

---

## 6. Escalation — flag-and-stop, no exceptions

1. Signed-ρ_c / H1d: anchor centers `s⁰_μ` not exposable without modifying
   vendored QP internals (§2.2). **Not a degradation to abs-only** — flag and
   wait (adapter extraction vs. `third_party/patches/` vs. drop H1d).
2. Ψ_eff identity test fails (§3.3).
3. §2a fails at P=16 across all N (§5).
4. Any formula in `00` post-reconciliation appears inconsistent with a
   reference sheet or vendored implementation — stop, do not "fix".
5. Any pre-registration threshold decision — Kati only.
6. Anything requiring a paper locator you cannot verify against a source in
   `papers/` or a fetched page.

## 7. Do-NOT-build list (deadline scope)

Heterogeneous configs C3–C6 and every H3/H5 analysis; Phases 2–4; T=40 (H4);
Split-CIFAR100 confirmation; CCGP/shattering execution (generator-side
factorial structure IS built — only the measures stay off); the input-dim
wealth fallback; recurrence; the per-unit γ-spectrum; `optimal-coding-statistics`
wiring; any GLUE-dependent measure (ρ_a, ψ pairwise) beyond recording
N/A-pending.

## 8. Daily checkpoints

Each day ends with: a commit, the day's `results/*.json` artifacts, and a
≤10-line status note in `results/LOG.md` (done / blocked / flags raised).
Day 13's note must contain the P/N/β recommendation with the error surfaces.
Day 14's must contain the MDE table for Kati's thresholds.
