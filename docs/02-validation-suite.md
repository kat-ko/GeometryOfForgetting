# 02 — Validation Suite

**A component is not done when it runs. It is done when its test here passes.**

Geometry estimators fail silently: they return well-formatted, plausible numbers
that are wrong. Every test below exists because a specific silent failure is
possible.

Tests are ordered by dependency. **§1 and §2 are hard gates** — do not build on an
unvalidated capacity estimator.

---

## 1. Manifold generator — ground-truth recovery

**Gate. Must pass before any GLUE work.**

Source: Chou et al. (ICML 2025) Appendix B.5, Figures 9–12. Their stated settings:

```
N = 1000 (ambient), P = 2, M = 200
Sweep 1 (dimension & radius; correlations zero): D_ground 2→10, R_ground 0.8→2.0
Sweep 2 (alignment): D_ground = 4, R_ground = 1 fixed; ρ_c, ρ_a, ψ each 0→0.8
```

| Test | Assertion |
|---|---|
| `test_dimension_recovery` | `D_mf` increases monotonically with `D_ground` over 2→10 |
| `test_radius_recovery` | `R_mf` increases monotonically with `R_ground` over 0.8→2.0 |
| `test_center_alignment_recovery` | `ρ_c` (estimated) increases monotonically with `ρ_C` (generator) over 0→0.8 |
| `test_axis_alignment_recovery` | `ρ_a` (estimated) increases monotonically with `ρ_A` (generator) over 0→0.8 |
| `test_capacity_direction_D` | capacity **decreases** as `D_ground` increases |
| `test_capacity_direction_R` | capacity **decreases** as `R_ground` increases |
| `test_capacity_direction_rho_c` | capacity **decreases** as center correlation increases |
| `test_capacity_direction_rho_a` | capacity **increases** as axis correlation increases |
| `test_unit_normalization` | pre-scaled manifold points have unit norm before `R` scaling |

The last four are sign checks and are the cheapest way to catch a transcription
error in Step 4 of the algorithm. Note `ρ_a` and `ρ_c` move capacity in
**opposite** directions.

If monotonicity fails, the estimator is wrong. Do not proceed, do not "adjust
tolerances."

---

## 2. Capacity fidelity — one gate (mean-field validity)

The three-factor decomposition `α = Ψ_eff·(1 + R_eff⁻²)/D_eff` is an **exact
identity** (`00` §6.2, ICLR 2026 §B.3), so there is no approximation-fidelity gap
to test — the former **§2b is deleted**. Two capacity quantities remain, with one
real error source between them:

| Quantity | Source | Error it exposes |
|---|---|---|
| `α_sim` | simulation, bisection on N; **works at any `y`**. Reference standard **at small `N`**, not unconditional ground truth — see the drift note below | its own ambient-`N` artifact |
| `α_mf` | replica mean-field (`replicaMFT`) — **label-invariant, β = 0 only** | does mean-field theory hold at our P, M, N? |
| `α_core` | our GLUE core (`src/glue/core.py`), ambient anchors under shared `y` | does our three-factor estimator recover ground truth, at any β? |

**α_sim source (verified):** `third_party/correlated_capacity`
`capacity/manifold_simcap_analysis.py::manifold_simcap_analysis(XtotT, n_rep, seed)`
computes **general point-cloud** simulation capacity (SVM separability at κ=0,
bisection on feature dimension) — confirmed to accept arbitrary `(N, P_i)` point
clouds, so it is valid at project parameters. `α_sim`, `α_mf`, and any GLUE number
are **different estimators**: the §9 pooling firewall applies pairwise
(`{repo}@{sha}` in the result key). Patch `manifold_simcap_analysis` for RNG
isolation first — it calls global `np.random.seed` (see `third_party/VENDORED.md`).

### §2a — Mean-field validity (the sole capacity gate)

**Gate.**

**Corrected 2026-08-11 — the β axis is `α_sim`-only on the mean-field side.**
`α_mf` is **label-invariant by construction** (`00` §6.1): the replica derivation
integrates the dichotomy average out analytically, so `replicaMFT` cannot produce
capacity at `β > 0`. The β sweep therefore compares `α_sim` against **`α_core`**
(our GLUE core), with `α_mf` participating **only at β = 0**.

```
test_mean_field_validity:
    at project parameters (d=150, P=16, M=150, N=300, D and R in range),
    beta = 0      : compare alpha_sim  vs  alpha_mf   (replicaMFT)   # MFT validity
                    compare alpha_sim  vs  alpha_core (glue core)    # core validity
    beta in {0.5, 1, 2, 4, inf}  (inf == retained, y fixed at y_j):
                    compare alpha_sim  vs  alpha_core                # core only
    record mean and max relative error per beta -> results/mft_validity.json
```

**`α_sim` is the reference standard, not unconditional truth (amended 2026-08-11).**
Re-running `α_sim` on a **single point cloud zero-padded** to larger `N` — identical
geometry, isometrically embedded, so capacity is invariant by construction — gives
0.434 → 0.435 → **0.461** across `N = 300/600/1200` (**+6.2%**), while `α_core`
gives 0.439 → 0.434 → 0.437 (**−0.3%**). `α_sim`'s ambient-`N` dependence is an
**estimator artifact** (bisection range and projection statistics both scale with
`N`). Consequence for this gate: `α_sim` anchors the comparison **at the design
point**, where `N` is smallest, agreement is best and its scatter is tightest; at
larger `N`, disagreement must **not** be attributed to `α_core` by default. Report
both. `results/estimator_followups.json`.

**Ran 2026-08-11 at β = 0 over `P ∈ {8,16,32} × N ∈ {300,600,1200}`**
(`scripts/run_gate_2a.py` → `results/gate_2a.json`). At the design point
`(P=16, N=300)`: `α_core` within **3.4%** of `α_sim`, `α_mf` within **1.6%**.
Mean `α_core` error across the surface 2.9%, no systematic sign; `α_mf` is
systematically low (1.6–19.8%, worst at `P = 8`, improving with `N` as expected).
**`P = 16, N = 300` confirmed** — see `01` Phase 0 and
`docs/reference/glue-core-validation.md` §3. The β > 0 half of this gate is still
to run.

`β = 0` answers whether mean-field holds at our parameters at all.
**`β = ∞` validates retained capacity, which is what the §8 money figure runs on**
— and it now tests *our* estimator, not a vendored one, which is why `§2a` and the
B.5 recovery sweeps (§1) together are the acceptance criteria for `src/glue/core.py`.

`docs/reference/glue-refinements.md`'s A2 concern is **resolved structurally, not
empirically**: A2 is not a question of mean-field *accuracy* under a degenerate
ensemble, it is the reason `α_mf` has no `y` argument at all. 04 §C4 (fixed-`y` is
a legitimate analyst choice) still stands — it just cannot be served by replicaMFT.

If `α_sim` and `α_core` disagree substantially at a given β, our estimator is wrong
at that β and **no downstream capacity claim there is safe** — a stop, not a caption
change. See also the P×N sweep in `01-experiments.md` Phase 0
(`test_mean_field_validity_at_project_P`), which decides whether `(P, N)` are large
enough *before* anything is built.

**Conventions the comparison must fix (verified against vendored code):**

- **Margin.** Both sides at **κ = 0**. `α_mf` (replicaMFT) is called with `kappa=0`;
  `α_sim` (`manifold_simcap_analysis` → `check_data_separability_general`) hard-codes
  `kappa = 0`. If either side were run at κ ≠ 0 the gate compares different
  quantities and reports a mean-field failure that isn't one — assert κ=0 on both.
- **What is held fixed.** Both estimators receive the **identical** per-manifold
  point clouds (same `P`, `M`, same ambient `N`). `α_sim` finds critical capacity by
  **bisecting the random-projection dimension at fixed `(P, M)`** (`α_sim = P / N_c`);
  it does **not** vary `P`. `α_mf` is the replica estimate on the same clouds.
  Compare the two capacity ratios directly.
- **RNG.** `manifold_simcap_analysis` calls global `np.random.seed` — must be run
  through the `src/glue/adapters/` RNG-injection wrapper (see `third_party/VENDORED.md`),
  not called directly, or the noise floor is not reproducible.

*(§2b — approximation fidelity — deleted 2026-08-10: the decomposition is an exact
identity, so there is no `(1+R⁻²)/D` approximation to test. See `00` §6.2 and 04
§C1. `results/approximation_fidelity.json` is no longer produced.)*

### §2c — Ψ_eff identity: **diagnostic**, not a gate (reclassified 2026-08-11)

Originally D3's gate. Reading the source showed `replicaMFT`'s `R_M`/`D_M` are the
**PRX-2018 eq-28/29** functionals, *not* `R_eff = √(E[c]/E[b−c])` and
`D_eff = E[b]/P`. So `Ψ_eff = α·D_eff/(1+R_eff⁻²)` computed from replicaMFT outputs
is **expected to disagree** with `E[c]/E[a]`. Run it anyway — cheap, and it converts
a reasoned expectation into a recorded measurement.

```
test_psi_eff_identity_diagnostic:
    psi_from_replicaMFT = alpha_mf * D_M / (1 + R_M**-2)      # harmonic-mean alpha
    psi_from_core       = E[c] / E[a]                          # src/glue/core.py
    record both + relative gap -> results/psi_eff_diagnostic.json
    EXPECTED: mismatch. A pass would be the surprise and must be explained.
```

Within `src/glue/core.py` the identity is exact **by construction** (both sides come
from the same `a/b/c`), so the meaningful internal check is
`assert P/E[a] == Psi_eff*(1+R_eff**-2)/D_eff` to float64 tolerance — a
transcription check on the implementation, included in the core's unit tests.

---

## 3. Parameterization orthogonality

**Gate for the whole (a, γ) factorial design.**

```
test_hidden_init_identical_across_gamma:
    for gamma in [0.01, 0.1, 1.0, 10.0]:
        build model with same seed
        assert W_hidden is bitwise identical across all gamma
```

```
test_base_width_equivalence:
    assert  muP(gamma_0=1, N=64)  ≡  NTP(N=64)
    numerically, to float64 tolerance, on forward pass and first gradient step
```

The second test pins the base-width constant (spec §4.2). Verification 1 is
signed: A.3 does not state `N/N₀` (dangling caption); independent derivation
from Table 1 confirms `(N/N_base)` (`parameterization-derivation.md`).

```
test_lr_scales_with_gamma:
    assert per-module lr == lr0 * gamma_m**2 * N        (quadratic mode)
    assert corrected mode uses gamma**(2/L) for gamma > 1
    assert the same lr applies to BOTH W_m and u_m
```

```
test_output_zero_at_init:
    assert f(x; theta_0) == 0 exactly, for all gamma
```

---

## 4. Alignment knob (Gate 2 — open empirical question)

```
test_alignment_preserves_norm_and_rank:
    for a in linspace(0, 1, 11):
        assert ||W(a)||_F == ||W(0)||_F        (float64 tolerance)
        assert rank(W(a)) == rank(W(0))
```

```
test_grassmann_endpoints:
    assert subspace(Y(0)) == subspace(V_k)
    assert subspace(Y(1)) == subspace(U_C)
    assert principal angles are monotone in a
```

```
test_alignment_moves_initial_capacity:      # GATE 2 — may fail
    sweep a in linspace(0, 1, 7)
    assert alpha_generic at init is monotone in a
    assert range(alpha_generic) >= 2 * identifiability_noise_floor
```

**Gate 2 is expected to be the most likely failure in the project.** If
`test_alignment_moves_initial_capacity` fails, do not attempt to fix the geodesic.
Switch `wealth_knob` to `"input_dim"` and record the loss of orthogonality with γ
as a known limitation.

---

## 5. Stream generator

```
test_dichotomies_balanced:          all sampled dichotomies sum to zero
test_hamming_distances_even:        all pairwise distances between balanced dichotomies are even
test_readout_similarity_target:     s_r(t, t-1) matches config within one Hamming step
test_sign_symmetry:                 s_r(y, y) == s_r(y, -y) == 1.0
test_feature_similarity_target:     realized center correlation matches rho_C target
test_probe_never_trained:           y* does not appear in any training task, any stream
test_similarity_matrices_recorded:  full T×T S_f and S_r written for every stream
test_stream_reproducible:           same stream_id + seed -> identical stream
```

---

## 6. Noise floors

**Required before any effect is interpreted. Not optional, not deferrable.**

For every measure in {`α_generic`, `α_retained`, `Ψ_eff`, `R_eff`, `D_eff`,
`rho_c_glue`, `rho_c_signed`, `ρ_a`, `ψ`, `PR`, `CKA`, `CCGP`, `shattering`,
rotation, `ΔPR`, reuse}:

```
identifiability_floor:
    15 seeds, SAME condition, SAME timepoint
    -> spread of the measure across seeds

trajectory_floor:
    same seed, same config, ONLY data order varies
    -> spread of the measure
```

Then:

```
minimum_detectable_effect(measure) = f(identifiability_floor, n_seeds_planned)
```

Write all of it to `results/noise_floors.json`.

**Rules that follow:**
- Any reported effect must exceed **both** floors.
- Any measure whose MDE exceeds its expected effect is **demoted from
  confirmatory to exploratory** in `01-experiments.md` §5 *before* thresholds are
  written.
- `R_eff` and `ψ` are expected to be the noisiest. Plan for them failing.

---

## 7. Training loop

```
test_no_weight_decay:               optimizer has weight_decay == 0
test_no_normalization_layers:       model contains no BatchNorm/LayerNorm/RMSNorm
test_optimizer_is_sgd:              no Adam/RMSProp/momentum in core configs
test_activation_is_relu:            no tanh/sigmoid/GELU
test_no_gradient_clipping:          no clip_grad_* call in the core path
test_single_readout:                exactly one readout; no task-ID input reaches the model
test_matched_loss_stopping:         under stopping="matched_loss", all conditions terminate within tolerance of target_loss
```

These are guardrail tests. They exist because these are exactly the changes an
agent is most likely to make as an "improvement."

---

## 8. Manipulation checks (runtime assertions, not unit tests)

Logged every boundary; violations raise a **warning in the results record**, not
an exception — a run that loses its manipulation mid-stream is still data.

| Check | Threshold |
|---|---|
| `‖ΔW_m‖/‖W_m‖` separation across γ | ≥ 1 order of magnitude; warn if it collapses below 3× |
| per-module gradient-norm share | warn if any module < 5% for 3 consecutive boundaries (gradient starvation) |
| per-module output-variance share | warn if any module < 5% (module effectively dead) |
| capacity-at-init vs `a` | warn if non-monotone |
| pseudo-inverse effective rank | log always; warn on rank deficiency > 20% |

---

## 9. Reproducibility

```
test_rng_isolation:
    changing rng_data_order must NOT change the initialization
    changing rng_init_shape must NOT change the manifold arrangement
    changing rng_stream must NOT change either
```

```
test_paired_init:
    for fixed seed, W_tilde is identical across all configs in that seed
    conditions differ only by the scalars under test
```

```
test_shared_streams:
    for fixed stream_id, the dichotomy sequence is identical across configs
```

```
test_artifact_completeness:
    every result file records: resolved config, git SHA, estimator version,
    full RNG seed state, noise-floor version
```

**Pooling rule:** geometry numbers computed under different estimator versions
must **never** be pooled. The estimator version is part of the result key.

---

## 10. Definition of done (per component)

1. Its test section above passes
2. Its noise floor is **measured**, not assumed
3. Its cost at target parameters is recorded in `results/cost_model.json`

"It runs without erroring" is not done.
