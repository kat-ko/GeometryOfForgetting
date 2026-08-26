# 01 — Experiment Registry, Phases, and Pre-Registration

**Project rule: decisions are pre-registered before compute is committed.**
Adding an experiment means adding it here first, with its outcome thresholds,
*then* running it. An agent asked to run something not registered here should ask
whether it should be registered.

---

## 1. Configuration schema

Every run is fully determined by one config. No hidden defaults in function
bodies. Fields:

```yaml
# --- identity ---
name: str
phase: int
registered: bool              # must be true to run

# --- manifolds ---
d: 150
P: 16
M: 150
D: int                        # intrinsic dimension
R: float                      # radius
rho_A: 0.0
psi_generator: 0.0
factor_structure: "labeling_only"     # do not change

# --- stream ---
T: 16
feature_similarity: float     # s_f(t, t-1)
readout_similarity: float     # s_r(t, t-1)
dichotomy_family: "random_balanced" | "single_factor" | "xor" | "mixed"
probe_dichotomy: true
stream_id: int                # shared across configs — blocking factor

# --- architecture ---
n_modules: 1 | 2
N: 300
gamma: [float, ...]           # per module
alignment: [float, ...]       # per module
wealth_knob: "alignment" | "input_dim"
frozen_modules: [bool, ...]
lr_scaling: "quadratic" | "corrected"
readout_init: "zero"          # do not change

# --- training ---
optimizer: "sgd"              # do not change
lr0: float
epochs_per_task: int
stopping: "matched_loss" | "fixed_steps"
target_loss: float            # if matched_loss
joint_training: false         # true = i.i.d. control

# --- evaluation ---
tier2_interval: 4
n_t: 200
kappa: 0.0                    # do not change
estimation_mode: "pairwise" | "full_P"   # pairwise = lab standard (2 manifolds/QP); full_P = all P jointly. Compared in Phase 0
capacity_modes: ["raw", "gaussianized"]  # gaussianized = Wakhloo 2023 preprocessing (00 §6.3)
rho_convention: "both"       # MANDATORY — both always computed; no path may select glue_abs or signed_normalized alone. rho_c_glue = reporting/comparability; rho_c_signed = H1d instrument (03 §E.4)
beta: 0.0                    # tilt for tilted_capacity: 0 = generic, ∞ = retained (00 §7)
ccgp_enabled: true           # per-family CCGP (00 §10)

# --- reproducibility ---
seed: int
```

**Seeding.** Four independent generators derived from `seed`, never shared:
`rng_arrangement`, `rng_init_shape`, `rng_stream`, `rng_data_order`.
The paired-initialization design requires holding some fixed while varying others.

**Paired initialization.** One shape matrix `W̃` drawn per `seed` from
`rng_init_shape` and reused across **all** configs at that seed, scaled and
rotated per condition. Conditions differ only by the scalars under test.

**Shared streams.** Stream realizations generated once from `rng_stream` keyed by
`stream_id`, reused across configs. Stream identity is a blocking factor in the
statistical model, not residual noise.

---

## 2. Architecture configurations

All at matched total width `2N` and matched parameter count.

| ID | Modules | γ | a | Purpose / rules out |
|---|---|---|---|---|
| `C1` | 1 (width 2N) | swept | swept | dense baseline; recovers γ₀\* |
| `C2` | 2 | `[γ, γ]` | `[a, a]` | **primary control** — isolates modularity per se |
| `C3` | 2 | `[γ_lo, γ_hi]` | `[a_hi, a_lo]` | **treatment** — wealthy-lazy + poor-rich |
| `C4` | 2 | `[γ_hi, γ_lo]` | `[a_hi, a_lo]` | anti-diagonal — does the *specific* pairing matter? |
| `C5` | 2 | `[γ, —]` | `[a, a]` | module B frozen — lazy dynamics vs. static random features |
| `C6` | any of above | — | — | **`joint_training: true`** — the sequentiality control |

`C6` is the headline control. Required pattern: heterogeneity helps sequentially
and **not** jointly, on identical data. If it helps jointly, the effect is
ensembling.

`C4` is easy to forget and important: if the anti-diagonal works as well as the
diagonal, the story about *which* geometry pairs with *which* update rule is
wrong, and the finding is generic diversity.

---

## 3. Stream conditions

### 3.1 Primary — Hiratani 2×2

| ID | `feature_similarity` | `readout_similarity` | Prediction |
|---|---|---|---|
| `S-HH` | high (0.9) | high (0.9) | benign |
| `S-HL` | high (0.9) | low (0.1) | **catastrophic corner** — largest rotation/overwrite, largest ρ_c increase |
| `S-LH` | low (0.1) | high (0.9) | relatively benign |
| `S-LL` | low (0.1) | low (0.1) | intermediate |

### 3.2 Secondary

| ID | Description |
|---|---|
| `S-fixed-r` | `s_f = 1.0` (arrangement frozen), `s_r ∈ {0.1, 0.5, 0.9}` — isolated readout axis |
| `S-blocked` / `S-shuffled` | matched marginals, run-ordered vs shuffled — **deferred to full paper** |

### 3.3 Naturalistic confirmation

Split-CIFAR100, 10 tasks × 10 classes (`P = 10` per task).
**Not** Split-CIFAR10 or Split-MNIST: at `P = 2` there is only one balanced
dichotomy (up to sign), so the factorial/dichotomy design collapses and the
generic/retained distinction has nothing to range over. (This is a
*task-structure* argument, not an estimability one — the lab estimates GLUE
pairwise at `P = 2` routinely; see `PROJECT.md` §2 P1 and 04 §C6.)

---

## 4. Phases

### Phase 0 — Manipulation validation (~2 days)

| Check | Pass criterion | Failure action | Result (2026-08-11) |
|---|---|---|---|
| Capacity-at-init tracks `a` | monotone, range ≥ 2× noise floor | fall back to `wealth_knob: input_dim` | **PASS** — monotone, rank corr +1.00, range 24.0% vs 3.74% needed |
| Capacity-at-init flat in `γ` | variation within noise floor | parameterization bug — **stop** | **PASS, exactly** — representations bitwise identical across γ |
| `‖ΔW_m‖/‖W_m‖` separates across γ | ≥ 1 order of magnitude | parameterization bug — **stop** | **PASS** — 2.53 decades at matched loss (0.0015 → 0.503), all arms converged |
| **Estimation-mode comparison** (spec §13) | `pairwise` vs `full_P` agree within noise floor on {α, R_eff, D_eff, ρ_c} | if they diverge, report both — `pairwise` is comparable-to-published (lab standard), `full_P` is primary | **DIVERGE on geometry, agree on α** — see below |
| **Time-reparameterization test** (spec §12) | residual after warping **exceeds** noise floor | if trajectories coincide, **H3 is dead** — restrict contrasts to γ≪1 vs γ~1, proceed with H1/H2 only | **PASS — H3 survives.** γ=1 vs 10 and γ=3 vs 10 differ by 11.5 and 7.1 noise floors under the most generous monotone warp |

**Gate 2 passed. RESOLVED 2026-08-11: `a` is not reinstated as a heterogeneity
axis.** `05` D4 pre-authorized dropping `a` on a Gate-2 failure; it did not fail
(`α` at init rises strictly monotonically 0.309 → 0.394, 6.4× the required range,
in the expected direction). The knob works. But the binding constraint is now
analysis time rather than compute, and H3 is still at risk pending the
time-reparameterization test, so **`C3` and `C4` stay cut** — heterogeneity is the
full paper's spine, not a rushed fourth result in this sprint.

**What is included: a homogeneous `a` sweep.** `C2` at `a ∈ {0, 0.5, 1}`, fixed
`γ`, no heterogeneous pairs. Cheap, strengthens Figure 1's orthogonality panel
(γ and `a` are exactly orthogonal at init — see the flat-in-γ row above), and
banks the trajectory data H3 needs.

**RESOLVED 2026-08-11 — `full_P` is MANDATORY for all Tier-2 measurement.**
`results/mode_constancy.json` (`scripts/run_mode_constancy.py`, 9 geometries × 3
seeds) tested whether the mode offset is constant. It is not: the
`pairwise/full_P` ratio on `D_eff` ranges 1.038 → 1.253 (CV 7.1%) and on `Ψ_eff`
1.050 → 1.283. Decisively, in `Δlog D_eff` — the quantity attribution uses — the
two modes disagree by **9.7× the noise floor** on the comparison Phase 1 actually
makes (`rep_init → rep_rich`: −0.227 vs −0.348), while agreeing to 0.1× the floor
on two near-identical representations. The disagreement grows with the size of the
geometry change, which is precisely the pathology that would contaminate Figure 2
undetectably. `pairwise` is therefore computed and reported **only** where
comparability with published values is the point.

Appendix material either way: published GLUE *capacities* are comparable across
estimation modes (`α` agrees to ~1% in every geometry tested, because the `D_eff`
and `Ψ_eff` distortions are co-directional and cancel in
`α = Ψ_eff(1+R_eff⁻²)/D_eff`) while published *geometries* are not. A 20–25%
`D_eff` offset between modes is an estimator convention, not a finding.

**Guard when building arrangements for estimation: keep `P(D+1) ≪ ambient`.** A
first pass at this check placed synthetic manifolds in the input dimension
`d = 150`, where `P(D+1) = 144` of 150 at `D = 8` — nearly degenerate, `D_eff`
collapses for reasons unrelated to what is being measured. This is why `02` §1
runs B.5 at `N = 1000`.

**Train accuracy is not a usable progress measure in this model, and stopping must
be on loss.** From `u_m(0) = 0`, one gradient step gives `u_m ∝ Σ_b y_b h(x_b)`,
the kernel readout, whose sign is independent of the learning rate and of γ. Train
accuracy therefore jumps to ~0.99 at step 1 identically in every arm, and an
accuracy-based stopping rule halts before any feature learning. `stopping:
"matched_loss"` with `target_loss` is mandatory, not one of two equal options.

**`lr0 = 5.0`, `target_loss = 0.05` — pinned by measurement, not taste.** At
`lr0 = 0.2` the arms at γ ≤ 1 plateau near loss 0.34 and never reach the target,
which would silently confound γ with training progress. This is *not* an
expressivity floor — solving the readout exactly at fixed `W` gives loss 0.0002,
since the targets are constant within each manifold — it is simply too small a
step. At `lr0 = 5.0` every γ reaches 0.05 at every seed, and the system remains
stable at `lr0 = 50`. **Any change to `M`, `P`, or `N` invalidates this value**;
re-run `scripts/run_phase0_richness.py`.

**Steps-to-target spans 39.5× across γ** (2883 at γ=0.03 → 73 at γ=10) at matched
loss. This was the natural first warp for the time-reparameterization test (`00`
§12), and it fails to align the trajectories — see below.

**RESOLVED 2026-08-11 — H3 is testable; `00` §12's fallback does not fire.**
`results/timewarp.json`. Under the most generous monotone warp available (DTW),
γ=1 vs 10 and γ=3 vs 10 differ by **11.5** and **7.1** noise floors. Trajectories
at different large γ are not one trajectory at two speeds, so **heterogeneity
contrasts are unrestricted**. Three qualifications, all load-bearing:

1. **DESIGN CONSTRAINT (binds the full paper's heterogeneity contrasts): every
   large-γ heterogeneity contrast must use well-separated γ with γ = 10 as one
   endpoint.** γ=1 vs 3 is marginal at 1.05 ± 0.29 — inside the decision boundary,
   and those are the two arms that move least (9.8 and 27.2 floors), so a 1-vs-3
   contrast is not demonstrably a shape difference rather than a rate difference.
   A heterogeneous pair drawn from `{1, 3}` cannot support an H3 claim. This
   carries forward to `C3`/`C4` when heterogeneity is built for the full paper.
2. **At γ = 0.03 the geometry does not move** — total excursion 0.2 noise floors
   over the window while the loss falls 0.499 → 0.010, i.e. all learning is in the
   readout. Comparisons against it are *vacuous* for this test rather than
   coincident: a static trajectory is trivially the slowed opening of any other.
   Excursions in floors: γ=0.03 → **0.2**, γ=1 → 9.8, γ=3 → 27.2, γ=10 → 65.2.
3. **Matching loss does not match geometry.** The rate that best aligns geometry is
   10–20× the rate that aligns loss (161.6 vs 8.0 for γ=1 vs 10), and even that
   leaves 6.9 floors. `matched_loss` equalizes training progress, not
   representational change.

### FOR THE METHODS SECTION — the training protocol's two non-obvious facts

Both are measured, both are easy for a reader to assume away, and both change how
the γ axis should be read. State them together where the stopping rule is described.

1. **Stopping must be on loss, and this is load-bearing rather than a preference.**
   From `u(0) = 0`, one gradient step gives the kernel readout
   `u ∝ Σ_b y_b h(x_b)`, whose *sign* is independent of the learning rate and of γ.
   Train accuracy is therefore ~0.99 after a single step in every arm. An
   accuracy criterion would halt training before any feature learning — the γ axis
   would be silently dead and the null would look clean.
2. **Matched loss equalizes training progress, not representational change.** The
   time rescaling that best aligns the *geometry* trajectories of γ=1 and γ=10 is
   **161.6×**, against **8.0×** for the rescaling that aligns their *loss* — and
   even at the geometry-optimal rate a residual of **6.9 noise floors** remains.
   Loss and representational geometry run on different clocks, so a matched-loss
   protocol equalizes the first and not the second. Any claim of the form "compared
   at equal performance" must not be read as "compared at equal representational
   change".

Also from the trajectories: **ρ_c is the most dynamic channel by a wide margin**
(122 floors of movement at γ=10, against 22 for `R_eff`, 19 for `α`, 18 for
`D_eff`), which is favourable for H1d.

#### Phase 0 precondition — mean-field validity at project `P` (decide **before** building)

Capacity theory is a thermodynamic-limit result in **both** `P` and `N`. `P = 16`
was chosen for GLUE estimability (vs Split-CIFAR10's `P = 2`) and four binary
factors — "better than 2" is **not** "large enough". The `02-validation-suite.md`
§1 recovery tests run at `P = 2` and say nothing about capacity accuracy at
`P = 16`. `α_sim` (now vendored, general point-cloud) makes this checkable cheaply.
Sweep the two limit knobs jointly:

```
test_mean_field_validity_at_project_P:
    sweep P in {8, 16, 32} x N in {300, 600, 1200}   (M fixed)
    compare alpha_sim vs alpha_mf   at each (P, N)   # generic (beta=0) only
    compare alpha_sim vs alpha_core at each (P, N)   # generic + retained
    record relative error surfaces -> results/mft_validity.json
```

**Fix the first lever correctly.** `N` (module width) is nearly free — no design
commitments attached; `P` is expensive — it is tied to the factorial design
(4 factors → 16, 5 → 32), and changing it changes the dichotomy families, CCGP
structure, single-factor/XOR counts, and every similarity calculation.

**RESOLVED 2026-08-11 — `P = 16, N = 300` stands; escalation not triggered.**
`results/gate_2a.json` (`scripts/run_gate_2a.py`, `P × N`, 3 seeds, M=60, n_t=200):
at the design point `α_core` is within **3.4%** of `α_sim` and `α_mf` within
**1.6%** — the best mean-field agreement anywhere in the sweep, and inside
`α_sim`'s own seed scatter (±0.017). `α_core` varies < 2% across `N` at fixed
geometry while `α_sim` itself drifts upward with ambient `N`, so the residual is
at least partly the simulation estimator's own bias. **The five-factor `P = 32`
redesign is off the table**; the ladder below is retained as the record of what
would have happened. See `docs/reference/glue-core-validation.md` §3.

The pre-specified order, had §2a failed at `(P=16, N=300)`:

1. **Raise `N`** (300 → 600 → 1200). Costs compute and nothing else. Prefer this.
2. If capacity accuracy is still marginal but `N` cannot go higher, **accept and
   report capacity as approximate** (feeds the two-factor attribution fallback,
   `00` §8 — a caption caveat, since the decomposition itself is exact).
3. **Raise `P` (→ 5 factors, `P = 32`) only as a last resort.** Design cost:
   dichotomy families become 5 single-factor + 10 XOR, `f: [P] → {0,1}^5`, and every
   `P`-dependent number in `00`/`01` shifts.

This is a scientific call for the human; the sweep produces the evidence. Note it
runs on `replicaMFT` + `correlated_capacity` + our own `src/glue/core.py` alone —
**no GLUE dependency**.

**`α_mf` is generic-only (verified 2026-08-11, `00` §6.1).** `replicaMFT`'s
`manifold_analysis_corr` takes no `y`; the replica derivation integrates the
dichotomy average out analytically, so it is **label-invariant by construction**
and can never yield retained or tilted capacity. It is a cross-check on generic
`α`, nothing more. Retained/tilted capacity and the whole three-factor
decomposition route through `α_sim` and `α_core` (`src/glue/core.py`,
estimator string `glue_core@<sha>`, subject to the same §9 pooling firewall).

**α aggregation is pinned to the harmonic mean** `1/mean(1/α_i)` wherever a
per-manifold `α` vector is reduced (`α = P/N_crit`; critical dimensions add).
Arithmetic means of `α_i` are a bug, not a convention choice.

### Phase 1 — H1, H2, H6 (homogeneous only)

Grid: `γ` sweep × **homogeneous** `a ∈ {0, 0.5, 1}` × Hiratani 2×2 streams,
`C1`/`C2`, **8 seeds**, 20 streams. `C3`/`C4` cut (see Phase 0). `estimation_mode:
full_P` throughout; `pairwise` only for the comparability appendix.

**Seeds fixed at 8 (2026-08-11).** The MDE figures below are lower bounds computed
from estimator Monte-Carlo variance at fixed manifolds, and the variance sources
they exclude — network initialization and stream realization — are exactly what
seeds average over. +1.7 h wall is worth it.

**Cost (measured 2026-08-11, `results/cost_model.json`) — compute is not the
binding constraint.** 52.4 s/eval at `n_t = 200`, one BLAS thread per worker,
248 workers:

| grid | runs | evals | wall |
|---|---|---|---|
| 10 streams × 5 seeds (`05` brief figure) | 1,200 | 48,000 | **2.8 h** |
| **20 streams × 8 seeds (this doc)** | 3,840 | 153,600 | **9.0 h** |

The two figures disagreed; both fit, so **the larger one stands** and the `05`
brief's number should be read as a lower bound.

**SUPERSEDED 2026-08-11 — the table above is wrong by ~9×; compute IS a binding
constraint.** `results/scaling.json`. The cost model measured 52.4 s/eval **in
isolation** and projected wall time by dividing core-seconds by 248 workers, i.e. it
assumed perfect scaling and never measured it. Measured against a live 254-worker
grid, one eval takes **775.8 s**. Throughput peaks at **128 workers** (5.17 eval/s)
and *falls* beyond it — 254 workers is slower than 32. Causes: `nproc` reports 256 on
this 2×64-core EPYC only because of SMT, so 128 is the physical count; and the anchor
QP has `P·M = 2400` variables, whose 46 MB Gram exceeds the 32 MB L3, making the
estimator memory-bandwidth-bound rather than compute-bound.

| grid | evals | wall at 128 workers |
|---|---|---|
| 1,280 arms × 38 evals (γ×6, `a`×2, 4 conditions, 5 streams, 8 seeds) | 48,640 | **26.1 h** + ~2 h training |

`_par.n_workers` now caps at 128. **The `n_t` → Tier-2 interval →
retained-tasks-evaluated cut order is INVOKED** — how far, pending decision. Note that
`n_t` noise largely cancels in the paired `Δlog` that attribution uses, because the
measurement RNG is shared across boundaries, so the noise-floor table below overstates
the cost of cutting `n_t` for Figure 2's quantities.

**Seed count, on evidence.** From the measured noise floors, MDE (two-sided,
power 0.8) at 5 vs 8 seeds per group: α 3.78% → 2.82%, `D_eff` 2.55% → 1.90%,
`R_eff` 1.01% → 0.75%, `Ψ_eff` 2.81% → 2.09%, `ρ_c` 1.96% → 1.46% — uniformly
**25.4% tighter for +1.7 h wall**. 8 seeds is the right call. Caveat: these floors
are estimator Monte-Carlo variance at fixed manifolds, so they are **lower bounds**
on the true MDE, which also carries network-init and stream variability. Recompute
from the Phase 0 pilot spread before the Day-14 MDE table is finalised.

Outputs: forgetting-attribution table (exact three-factor, log-space);
generic/retained crossing; probe-dichotomy
validation of generic capacity; KTA measured alongside; **Wakhloo/Slatton four
statistics** (dimensionality, total correlation, signal–signal factorization,
signal–noise factorization) measured alongside as the sharpened Gate-3 competitor
to generic capacity. Measure them regardless of the RQ2-reframe decision; their
exact definitions are pending human transcription to
`docs/reference/optimal-coding-statistics.md` and must not be wired into the Gate-3
comparison until verified.

**This phase alone is a complete NeurReps abstract.** No heterogeneity claims
required. Do not gate submission on Phase 2.

Kill criteria:
- GLUE attributions do not separate by regime → fall back to the rotation/expansion decomposition (spec §9), which is far more robust
- Generic capacity does not track the probe metric → report as a negative finding about label-agnostic geometry measures

#### Phase 1 measurement plan (built 2026-08-11; `src/pipeline.py`, `scripts/run_phase1.py`)

One arm = one `(γ, a, condition, stream, seed)`. `run_arm` trains the stream and
returns everything Figures 2 and 3 need from a single pass.

**Schedule — 38 evaluations per arm against the cost model's 40.** Tracked tasks are
every 4th; each is measured at its own boundary and at every later measurement
boundary: `{0:[0], 4:[0,4], 8:[0,4,8], 12:[0,4,8,12], 15:[0,4,8,12]}`. Two
requirements this satisfies, both asserted by tests:

- **Each tracked task's own boundary is measured.** Task `j`'s forgetting is
  attributed against the geometry immediately after `j` was learned; a schedule
  without that boundary has no denominator.
- **The lag widens.** Retention at a single fixed lag can show how much is lost but
  not how fast.

**The measurement RNG is fixed across boundaries and shared between the generic and
retained ensembles.** A difference between two boundaries is then the representation
moving rather than the estimator resampling `(y, t)`, and the generic/retained
crossing is paired at every boundary. With ~40 evaluations per arm each carrying
1–2% Monte-Carlo noise, an unshared stream would put noise in the differences that
does not cancel.

**Probe measure: `margin`, not accuracy** (`results/probe_check.json`). Probe
*accuracy* is exactly 1.0 at initialization and after training in every arm — load
`P/N = 0.053` against a critical capacity near 0.3, so every balanced dichotomy is
separable and separability at sub-critical load cannot track capacity. It would have
plotted flat at 1.0 and read as "generic capacity preserved". The readout **margin**
separates γ at SNR 4.0 with sign-consistent movement over training, and is the
principled choice besides: capacity is the load at which the margin reaches zero.
Held-out-*manifold* accuracy is retained as a **control only** — it sits at chance,
correctly, because `y*` is a random dichotomy with no structure to generalize, and its
apparent γ effect is one seed's initialization. If it ever rises above chance the
probe is leaking factor structure.

**`R_eff` is reported jointly with `ρ_c`, quantitatively.** Fitted on the B.5
center-correlation sweep, `log R_eff = 0.0126 + 0.3548·(−log(1−ρ_c))` (R² = 0.99986,
`D_eff` flat to 0.38% across the sweep), so attribution reports what fraction of each
observed `Δ log R_eff` the observed `Δ ρ_c` accounts for. Denominators below the
`R_eff` noise floor are suppressed rather than divided by. Fractions ≫ 1 are expected
on representations and are meaningful — the calibration is fitted where `ρ_C` alone
varies, so it over-predicts where radius and centers move together; it is a bound on
the center-collapse share, never a correction to `R_eff`.

### Phase 2 — H3, H5 (heterogeneous)

All of `C1`–`C6`, sequential and joint, paired inits, shared streams.

Kill criteria:
- Pair mean-γ predicts performance (H5) → the effect is a reparameterization
- Joint training shows the same effect → the effect is ensembling

### Phase 3 — H4 (regime half-life)

`T = 40` at three similarity levels. Track per-module regime over the stream via
the §11 manipulation checks. Measure the half-life of the initial regime
difference.

Confound to address: loss of plasticity may be an artifact of *abrupt*
environment change and largely mitigated under gradual change. Include a
gradual-change arm (`s_f`, `s_r` drifting smoothly) as a control.

### Phase 4 — Stream variance and predictability

Deferred to the full paper. Registered here so it is not silently re-scoped in.

---

## 5. Pre-registration table

Fill thresholds **after** the noise-floor measurement (`02-validation-suite.md` §6),
not before. A threshold below a measure's minimum detectable effect is not a
prediction.

| ID | Hypothesis | Measure | Direction | Threshold | Disconfirming observation |
|---|---|---|---|---|---|
| H1a | Rich forgetting is radius/utility-accounted | share of `Δ log α` from `Δ log(1+R_eff⁻²)` and `Δ log Ψ_eff` | rich > lazy | _TBD_ | shares equal within noise, or lazy > rich |
| H1b | Lazy forgetting is ρ_c-accounted | `Δ rho_c_glue` (covariate; `rho_c_signed` for direction) | lazy > rich | _TBD_ | reversed or null |
| H1c | Forgetting ≡ ultra-rich OOD signature | sign pattern over (R, ψ, D) | matches Chou Fig 7c | qualitative match on all 3 | any sign mismatch |
| H1d | Progressive center-decorrelation (Menghi) | **`rho_c_signed` trajectory** (NOT `rho_c_glue` — the absolute value erases the sign and makes H1d untestable; see `03` §E.4) | decreasing over stream, faster in rich | _TBD_ | flat or increasing |
| H2a | Generic capacity falls with γ | `α_generic` vs `γ` | monotone decreasing | _TBD_ | non-monotone or flat |
| H2b | Retained capacity rises with γ | `α_retained` vs `γ` | monotone increasing | _TBD_ | non-monotone or flat |
| H2c | Crossing predicts γ\* | argmax(`α_generic × α_retained`) | = argmin(average error) | within 1 grid step | separated by >1 grid step |
| H2d | **Generic capacity is label-valid** | corr(`α_generic`, probe metric) | positive | _TBD_ | null or negative → reportable negative result |
| H3a | Division of labour | `α_generic`(wealthy-lazy) > `α_generic`(poor-rich) | as stated | _TBD_ | equal or reversed |
| H3b | Reliance shifts with similarity | output-variance share vs `s_r(t,t−1)` | rich share ↑ as `s_r` ↓ | _TBD_ | flat |
| H4 | Regime erodes | `‖ΔW_m‖` separation vs stream position | decreasing; half-life shorter at high `s_f` | _TBD_ | separation persists to T=40 |
| H5 | γ_eff insufficiency | residual after regressing outcome on pair-mean γ | non-zero | _TBD_ | pair-mean γ explains everything |
| H6 | Catastrophic corner is geometrically worst | rotation, `Δρ_c` across S-HH/HL/LH/LL | max at `S-HL` | _TBD_ | max elsewhere |

---

## 6. Statistical model

```
outcome ~ alignment * gamma * feature_similarity * readout_similarity
          + (1 | seed) + (1 | stream_id)
```

- Paired differences per seed (enabled by shared `W̃`), not difference of means
- Effect sizes with intervals; **no p-values at small seed counts**
- Long-range similarity `s_f(t,j)`, `s_r(t,j)` enter drift regressions as covariates

**Budget rule:** for stream-level hypotheses, more streams beats more seeds.
~10 seeds × 20 streams over 40 seeds × 5 streams.

---

## 7. Figure targets

The abstract is 4 pages ≈ 4 figures. Every registered experiment should map to one.

| Fig | Content | Depends on |
|---|---|---|
| 1 | Setup + `a`/`γ` orthogonality demonstration | Phase 0 |
| 2 | **Forgetting attribution**, lazy vs rich, with Chou Fig 7c overlay | Phase 1 |
| 3 | Generic vs retained capacity vs γ, with probe metric overlaid | Phase 1 |
| 4a | Heterogeneous vs homogeneous on the front + γ_eff control | Phase 2 |
| 4b | *(fallback if Phase 2 fails)* regime half-life | Phase 3 |

Both versions of Figure 4 are coherent papers. Do not treat 4b as a failure state.
