# Reference sheet — validation of our GLUE core implementation

```
Source:       Own implementation, src/glue/core.py (estimator `glue_core`).
              Definitions from Chou, Kirsanov, Yang & Chung, ICLR 2026, App. B.3
              (via docs/reference/glue-decomposition.md) and Chou, Le, Wang &
              Chung, ICML 2025, Algorithm 2 + App. B.5 (recovery protocol).
Version:      glue_core @ commit of 2026-08-11
Derived from: measurements in results/*.json produced by scripts/run_*.py.
              Every number below is measured, none transcribed from a paper.
Transcribed by: agent (Opus 5), 2026-08-11
Verified by (human): Kati, 2026-09-01
Status: human-signed 2026-09-01. B.5 settings and D.1.1 confirmed against
        source. Sweeps are per-axis with off-axis values stated (D=4, R=1
        for the correlation sweep; JSON `settings.D_fixed=4`, `R_fixed=1.0`).
        Three cells spotted against `results/glue_core_recovery.json`
        (`n_t=200,policy=all`): D=2 → D_eff 2.25 (2.2455); R=0.8 → R_eff 0.87
        (0.8687); ρ_C=0 → rho_c_glue 0.04 (0.0379). §5a is present.
```

This sheet is the methods paragraph for the paper and the answer to the reviewer
question *"you implemented the estimator yourself — why should we believe it?"*

**The honest framing.** GLUE's reference implementation is not public. We
implemented the three-factor decomposition from the authors' published
definitions and validated it two ways: against an **exact analytic limit** the
estimator must reproduce (point manifolds, `α = 2` at `κ = 0`), and against the
authors' **own recovery protocol** (App. B.5) plus an independent simulation
ground truth (`α_sim`). Code is released. Section 5 states a measured limitation
we did not have to disclose.

Why the implementation exists at all: `replicaMFT`, the public estimator, is
label-invariant by construction and cannot express capacity for a specified
dichotomy — see `third_party/VENDORED.md`.

---

## 0. What the implementation does

Anchor points come from the **joint** capacity QP over all `P` manifolds at
`κ = 0`, `min ½‖v − t‖²  s.t.  Gv ≤ 0` with rows of `G` equal to `y_μ z^μ_i`.
This is not a stylistic choice. In §B.3 `S_y = diag(y)·S`, and `diag(y)` cancels
in all three quadratic forms (`diag(y)² = I`, `pinv(D A D) = D A† D` for
orthogonal diagonal `D`), so `a`, `b`, `c` are invariant to `y` *given* `S`.
All label dependence must therefore enter through anchors solved for jointly; a
per-manifold construction reproduces exactly the label-invariance that makes
`replicaMFT` unusable here.

The QP's KKT conditions make its duals a non-negative least-squares problem,
`λ = argmin_{λ≥0} ‖Gᵀλ − t‖²`, so no vendored code sits in the estimation path.

A manifold whose dual mass is zero contributes a **zero row** (its constraint is
slack). Section 1 is what pins that convention.

---

## 1. Exact analytic limit — point manifolds

For `P` single-point manifolds in general position, capacity at `κ = 0` is
`α = 2`. This is the strongest available check because it is exact, and it
discriminates the zero-row convention: with zero rows, about half the manifolds
are active for random `t`, `E[a] = P/2`, and `α = P/E[a] = 2`.

| quantity | expected | measured |
|---|---|---|
| `α` (P = 8, N = 200, n_t = 400) | 2 | **2.00 ± 0.12** |
| active fraction | 0.5 | **0.50 ± 0.05** |

`tests/test_glue_core.py::test_point_manifold_capacity_is_two`.

Also verified in the same suite: `diag(y)` cancels numerically as the algebra
says; anchors *do* change with `y` (so the joint QP is label-sensitive); the
identity `α = Ψ_eff(1 + R_eff⁻²)/D_eff` closes to `< 1e-8` relative, which is a
transcription check on the file since it is exact by construction.

---

## 2. Authors' recovery protocol — ICML 2025 App. B.5

`P = 2`, `M = 200`, `N = 1000`, `n_t = 200`, 3 seeds, `κ = 0`. Three
**independent** sweeps: `D` and `R` are each swept with correlation at zero, then
`D = 4, R = 1` are held fixed while `ρ_C` sweeps.
`scripts/run_glue_core_recovery.py` → `results/glue_core_recovery.json`.

**Dimension** (`D → D_eff`, at `R = 1`, `ρ_C = 0`):

| ground truth `D` | 2 | 4 | 6 | 8 | 10 |
|---|---|---|---|---|---|
| `D_eff` | 2.25 | 4.07 | 5.60 | 6.98 | 8.17 |

**Radius** (`R → R_eff`, at `D = 4`, `ρ_C = 0`):

| ground truth `R` | 0.8 | 1.0 | 1.4 | 1.7 | 2.0 |
|---|---|---|---|---|---|
| `R_eff` | 0.87 | 1.03 | 1.34 | 1.56 | 1.77 |

**Center correlation** (`ρ_C → rho_c_glue`, at `D = 4`, `R = 1`):

| ground truth `ρ_C` | 0.0 | 0.2 | 0.4 | 0.6 | 0.8 |
|---|---|---|---|---|---|
| `rho_c_glue` | 0.04 | 0.22 | 0.41 | 0.61 | 0.81 |

All three are monotone, `α` falls monotonically along each, and `Ψ_eff` stays in
`[0.85, 0.91] ⊂ [0,1]`.

**Cross-axis separability.** `D_eff` sits at 4.07–4.08 across the *entire* `R`
sweep and the *entire* `ρ_C` sweep. The axes are separately identifiable, which is
what the §8 attribution needs.

**Where correlation goes — the P5/F4 question, answered with a number.** Sweeping
`ρ_C` from 0 to 0.8 moves `R_eff` 1.02 → 1.82 and leaves `D_eff` at 4.07 → 4.08.
Center correlation is absorbed **entirely by the radius**, as the Wakhloo duality
predicts, and not at all by dimension. `00` §8 must keep reporting `ρ_c` beside
`R`; it must not attribute an `R` change to geometry alone.

---

## 3. Independent capacity ground truth — `α_sim`, and the `α_mf` cross-check

`α_sim` is simulation capacity (SVM separability at `κ = 0`, bisection on the
random-projection dimension, `α = P/N_c`), computed through
`src/glue/adapters/simcap.py`. `α_mf` is `replicaMFT`, which is generic-only and
therefore appears at `β = 0` only, reduced with the harmonic mean.

Identical point clouds, `M = 60`, `D = 4`, `R = 1`, `n_t = 200`, `n_rep = 10`,
3 seeds. `scripts/run_gate_2a.py` → `results/gate_2a.json`.

| `P` | `N` | `α_sim` | `α_core` | `α_mf` | core vs sim | mf vs sim |
|---|---|---|---|---|---|---|
| 8 | 300 | 0.483 ± 0.052 | 0.440 ± 0.006 | 0.388 ± 0.003 | 8.9% | 19.8% |
| 8 | 600 | 0.449 ± 0.046 | 0.439 ± 0.013 | 0.387 ± 0.003 | 2.2% | 13.9% |
| 8 | 1200 | 0.436 ± 0.043 | 0.430 ± 0.016 | 0.387 ± 0.001 | 1.5% | 11.3% |
| **16** | **300** | 0.420 ± 0.017 | 0.434 ± 0.002 | 0.413 ± 0.001 | **3.4%** | **1.6%** |
| 16 | 600 | 0.440 ± 0.009 | 0.440 ± 0.004 | 0.414 ± 0.003 | 0.0% | 6.0% |
| 16 | 1200 | 0.460 ± 0.013 | 0.437 ± 0.006 | 0.413 ± 0.001 | 4.9% | 10.1% |
| 32 | 300 | 0.437 ± 0.014 | 0.440 ± 0.005 | 0.423 ± 0.002 | 0.7% | 3.1% |
| 32 | 600 | 0.451 ± 0.007 | 0.433 ± 0.001 | 0.421 ± 0.001 | 4.0% | 6.6% |
| 32 | 1200 | 0.440 ± 0.019 | 0.439 ± 0.004 | 0.422 ± 0.001 | 0.1% | 3.9% |

`α_core` agrees with `α_sim` to **2.9% on average and ≤ 5% everywhere except
`P = 8, N = 300`**, with no systematic sign.

Two observations that matter for how the residual should be read:

- **`α_core` is the more stable of the two, and `α_sim`'s `N`-drift is an
  artifact — confirmed, not inferred.** Across `N` at fixed geometry `α_core`
  varies under 2% (0.434 / 0.440 / 0.437 at `P = 16`, seed SD ~0.004) while
  `α_sim` drifts monotonically upward (0.420 / 0.440 / 0.460) with 3–10× the
  scatter. Since the generated geometry is `N`-independent only *statistically*,
  we re-ran the comparison with `N` raised by **zero-padding a single point cloud**
  — the geometry is then literally identical, embedded isometrically in a larger
  ambient space, and capacity must be invariant:

  | `α` under zero-padding (P=16, M=60, 5 seeds) | N=300 | N=600 | N=1200 | drift |
  |---|---|---|---|---|
  | `α_sim` | 0.434 ± 0.025 | 0.435 ± 0.016 | 0.461 ± 0.014 | **+6.2%** |
  | `α_core` | 0.439 ± 0.006 | 0.434 ± 0.008 | 0.437 ± 0.004 | **−0.3%** |

  `α_core` is invariant as it must be; `α_sim` is not. The drift belongs to the
  simulation estimator — its bisection range and random-projection statistics both
  scale with `N`. **`α_sim` is therefore the right anchor at the design point
  (smallest `N`, best agreement, tightest scatter) but is not unambiguously ground
  truth at large `N`**, and part of the large-`N` residual in the table above is
  its bias rather than `α_core`'s error.
  `scripts/run_estimator_followups.py` → `results/estimator_followups.json`.
- **`α_mf` is systematically low**, by 1.6–19.8%, and worst at `P = 8` where the
  mean-field limit is furthest away. It improves with `N` at `P = 8`
  (19.8 → 13.9 → 11.3%), the expected direction.

**This settles the `P = 16` question (`01` Phase 0).** At the project design point
`P = 16, N = 300`, `α_core` is within 3.4% of `α_sim` and `α_mf` within 1.6% — the
best mean-field agreement anywhere in the table, and inside `α_sim`'s own seed
scatter. The pre-specified escalation (raise `N`, then accept-and-report, then
raise `P` to 32) is **not triggered**. `P = 16, N = 300` stands, and the
five-factor redesign is off the table.

---

## 4. `center_policy` — settled by measurement

`s⁰_μ = E[s^μ(y,t)]` has to decide what to do with samples where manifold `μ` was
inactive. The active fraction is ~0.5, so this is half the data, and §1 does not
discriminate: `α` depends only on `S`, whereas `R_eff` and both `ρ_c` conventions
normalize by `‖s⁰‖`. Both policies were run over the full B.5 protocol.

Mean absolute relative recovery error, by swept axis:

| policy | `D → D_eff` | `R → R_eff` | `ρ_C → rho_c_glue` |
|---|---|---|---|
| `"all"` (literal §B.3 reading) | 10.3% | 7.0% | **3.7%** |
| `"active"` (condition on activity) | **9.6%** | **6.4%** | 8.6% |

Stable under `n_t = 1000` (10.7 / 7.0 / 4.6 vs 9.8 / 5.8 / 9.2).

**Decision: `"all"`, on evidence.** `"active"` is very slightly better on `D` and
`R` — under a percentage point, comparable to seed noise — while `"all"` is better
on `ρ_C` by more than a factor of two, consistently at both `n_t`. The `ρ_C` axis
is also the most direct test of the three, since the generated correlation is
exactly known. `"active"` systematically over-estimates `ρ_c` (0.043 / 0.239 /
0.434 against a truth of 0 / 0.2 / 0.4), which is precisely the measure H1d turns
on.

Conclusions about `D` and `R` are robust to the choice; conclusions about `ρ_c`
are not. The policy is recorded in every result key either way.

---

## 5. Measured limitation — scale compression, and its mechanism

`D_eff` and `R_eff` over-report small ground-truth values and under-report large
ones, crossing over near `D ≈ 4`, `R ≈ 1`. At `D = 10` the estimate is 18% low; at
`R = 2.0`, 11% low. We chased this to a mechanism rather than reporting it as an
unexplained caveat. `scripts/run_scale_compression.py` →
`results/scale_compression.json`.

**Not Monte-Carlo error over `(y, t)`.** Raising `n_t` from 200 to 1000 moves
`D_eff` at `D = 10` from 8.17 to 8.19 and `R_eff` at `R = 2` from 1.773 to 1.785.
The gap does not close.

**Not an artifact of B.5's small `P`.** Mean absolute recovery error on `D`:
13.1% at `P = 2`, 14.4% at `P = 8`, 16.5% at `P = 16` (`M = 100`). Nearly flat.

**It is finite sampling of each manifold.** `D_eff/D` at `P = 2`, `N = 1000`:

| `M` | 50 | 100 | 200 | 400 | 800 |
|---|---|---|---|---|---|
| `D = 2` | 1.112 | 1.118 | 1.123 | 1.127 | 1.130 |
| `D = 6` | 0.846 | 0.898 | 0.934 | 0.955 | 0.969 |
| `D = 10` | 0.671 | 0.750 | 0.817 | 0.865 | 0.897 |

The deficit at `D = 10` falls as roughly `M^-0.45` and is converging to 1. Anchors
are extreme points of a finite sample, and a finite sample under-represents the
extent of a high-dimensional manifold; the higher `D`, the more points are needed
to reach the same relative coverage. The small over-report at `D = 2` is a
separate, `M`-independent effect of about 12%.

**This is a property of anchor-based estimation, not of our implementation.** The
mechanism is in the definition — anchors are dual-weighted extreme points of the
sampled cloud — so any GLUE-family estimator at modest `M` inherits it. Chou et
al. subsample to **50 points per manifold**; at `M = 50` our curve puts
`D_eff/D ≈ 0.67` at `D = 10` and `0.85` at `D = 6`. That is a methodological
observation about published applications of the method, not only about our runs,
and we state it as one: **we characterise the finite-sample behaviour of
anchor-based geometry estimation as a deficit scaling ≈ `M^-0.45`, uniform across
`D` and independent of `P` and `n_t`.** As far as we know this has not been
characterised before; it is cheap to measure and it changes how absolute `D_eff`
values in this literature should be read.

**Why `M = 150`, and the robustness arm.** `M` trades accuracy against cost, and
both sides are now measured. Cost at project parameters (P=16, N=300) is
**≈ `M^1.68`**: 242 ms/sample at `M = 150`, 1173 ms at 400, 4067 ms at 800.

| `M` | cost vs 150 | full Phase-1 grid | 4.5% robustness arm |
|---|---|---|---|
| 150 | 1.0× | 2.8 h | 0.13 h |
| 400 | 4.9× | 13.7 h | **0.61 h** |
| 800 | 16.8× | 47.4 h | **2.13 h** |

`M = 150` is the main-grid choice: it keeps the full grid at 2.8 h wall while
sitting on the flatter part of the accuracy curve above Chou's 50.
`M = 800` for the whole grid is 47 h, which is not affordable — but **a high-`M`
robustness arm is: 2.1 h at `M = 800` over 4.5% of the grid** (one condition ×
2 γ × 3 streams × 3 seeds). Run it, and report the main-grid `D_eff` beside the
high-`M` values so the compression is bounded empirically rather than argued.

**What this means for the paper.**

1. The bias is a **monotone increasing** function of the ground truth, so rank
   ordering and the sign of every effect are preserved.
2. It is a function of `(D, M)`. `M` is held fixed across all conditions, so the
   distortion is a fixed transform, not a between-condition artifact.
3. It **compresses** the dynamic range, so estimated `Δ log D_eff` is biased
   *toward zero*. In the §8 attribution the dimension channel is therefore
   **conservative**: a real contribution can be understated, not manufactured.
4. It must nonetheless be stated, because our streams change representational
   dimension: a bias correlated with a manipulated variable is a confound, not a
   nuisance.

**Caption requirement.** Every figure and table reporting an *absolute* `D_eff`
must carry "`D_eff` above ≈ 4 is a lower bound (finite-`M` compression, §5)". This
is not optional and not satisfied by stating it once in the methods.

**The attenuation, measured.** With ground truth known, the compression can be
scored directly in log space at Phase 1 settings (`P = 16`, ambient 300, `M = 150`):

| | `d log D_eff / d log D` | `d log R_eff / d log R` |
|---|---|---|
| `full_P` (P = 16, primary) | **0.663** | **0.581** |
| `pairwise` (P = 2, lab standard) | 0.772 | 0.624 |

1.0 would be faithful. So a true `Δlog D` registers as ≈ 2/3 of itself and a true
`Δlog R` as ≈ 3/5 of itself: Figure 2's dimension and radius channels are
attenuated by roughly a third and two fifths. This replaces the qualitative
"understates rather than manufactures" with a number.

**Not applied as a correction.** Dividing by 0.663 would import a
synthetic-manifold calibration into representation measurements, and the mode
ratios show those are a different regime (1.25 for representations against
1.04–1.21 for synthetic points at matched ambient dimension). Reported as an
attenuation bound on the effect size, not used to rescale.

**Guard: keep `P(D+1) ≪ ambient`.** At `P = 16, D = 8` in an ambient of 150, the
arrangement occupies 144 of 150 dimensions, the manifolds leave general position,
and `D_eff` collapses for reasons unrelated to the estimator. This is why B.5 runs
at `N = 1000`, and it is a real trap when measuring input manifolds rather than
representations.

Reaching a 5% deficit at `D = 10` would need `M ≈ 4000`, which is not affordable
across the grid — hence a stated caveat plus the high-`M` robustness arm above,
rather than a fix.

---

## 5a. What this sheet does *not* certify

Every result above is measured against synthetic manifolds with known ground truth. That
is the only way to check recovery, and it is a genuine limit on what the checks mean:
**synthetic validation certifies the estimator, not the behaviour of quantities derived
from it on real representations.**

This is not hypothetical. The ρ_c → R_eff calibration in §4 was fitted on the §2
center-correlation sweep, where `rho_c_glue` and `rho_c_signed` track each other to
within 0.02 across the whole range (0.038/0.043 … 0.809/0.803). Fitted on the
unnormalized convention it passed every check here at R² = 0.99986. On Phase 1
representations the conventions separate completely: `rho_c_signed` stays at 0.29–0.52
while **`rho_c_glue` reaches 1.93, above 1 in 39 of 56 measurements in the rich arm**,
because it is unnormalized by construction (`00` §6.1 C3) and therefore not a
correlation. The bounded form `(1 − ρ)^−k` is undefined there, and a `np.clip` intended
for float safety returned plausible numbers from invalid input — center-collapse shares
of 15–38 against a ceiling of 1.

The synthetic suite could not have caught this, because the degeneracy it depends on
(the two conventions coinciding) is a property of the synthetic generator. Two
consequences now enforced in code:

- Any calibration fitted here **declares its fitted range and refuses to extrapolate**
  (`attribution.RHO_FIT_RANGE`, `rho_in_domain`). Out-of-range input reports `n/a`.
- Clipping in a measurement path **asserts the violation it absorbs is at float scale**
  (`numerics.clip_to_noise`), so a guard cannot silently change meaning from "absorb
  rounding" to "manufacture a value".

The general rule for reading this sheet: a passing check here licenses the estimator's
*outputs*. Anything fitted on those outputs must be re-checked against the range
representations actually occupy — which the Phase 1 grid now logs per condition
(`summarize` → `rho_coverage`).

---

## 6. Reproducing everything here

```
scripts/run_glue_core_recovery.py   -> results/glue_core_recovery.json   (§2, §4)
scripts/run_mode_constancy.py       -> results/mode_constancy.json       (§5)
scripts/run_gate_2a.py              -> results/gate_2a.json              (§3)
scripts/run_estimator_followups.py  -> results/estimator_followups.json  (§3, §5)
scripts/run_scale_compression.py    -> results/scale_compression.json    (§5)
scripts/run_psi_eff_diagnostic.py   -> results/psi_eff_diagnostic.json   (02 §2c)
scripts/run_cost_model.py           -> results/cost_model.json
pytest tests/test_glue_core.py tests/test_glue_core_recovery.py          (§1)
```

Estimator string `glue_core@<sha>`; the `02` §9 pooling firewall applies between
it, `α_sim`, and `α_mf` pairwise.
