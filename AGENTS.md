# AGENTS.md

Rules for any LLM agent working in this repository.

Read `docs/00-math-spec.md` before writing any numerical code. Read
`docs/02-validation-suite.md` before claiming any component works.

---

## 1. What this project is

A research codebase measuring **representational geometry under sequential
learning**. We train small feedforward networks on a stream of tasks and measure
manifold capacity and its geometric decomposition (GLUE) at every task boundary.

The scientific output is a *measurement*, not a method. We are not trying to
improve continual-learning benchmark numbers. Do not add regularizers, replay
buffers, or CL algorithms unless a spec file asks for them.

## 2. Hard invariants — never violate without an explicit instruction

These encode experimental design decisions. Breaking one silently invalidates
results without producing an error.

| # | Invariant | Why |
|---|---|---|
| I1 | **No weight decay** anywhere | It shrinks large-init modules over a stream, silently converting lazy → rich |
| I2 | **No normalization layers** (BatchNorm, LayerNorm, RMSNorm) | They normalize away the initialization scale that *is* the manipulation |
| I3 | **Plain SGD only** in core runs | Adam's per-parameter normalization undoes the per-module learning-rate scaling; γ becomes decorative |
| I4 | **ReLU only** | Saturating activations make large-init units *dead*, not *lazy* — a different mechanism |
| I5 | **Readout initialized to zero** | Guarantees f(x; θ₀) = 0; otherwise lazy modules dominate the output at t=0 for reasons unrelated to richness |
| I6 | **Hidden-layer weights at init must be identical across γ** | This is the orthogonality proof for the (a, γ) factorial design. There is a unit test. Do not "fix" it |
| I7 | **No gradient clipping** | Clips rich units preferentially, compressing the manipulation |
| I8 | **Single shared readout; no task-conditioned head** | A task head absorbs the credit-allocation effect we are measuring |
| I9 | **Dichotomies must be balanced** | Unbalanced dichotomies are degenerate and contaminate similarity geometry. Stated deviation from Chou D.1.1 (uniform `{±1}`); `docs/reference/protocol-deviations.md` |
| I10 | **P, M, and N must match across modules** in any comparison | Capacity is a ratio to ambient dimension; unequal widths give incomparable numbers |

If a task seems to require breaking one of these, stop and ask.

## 3. Terminology discipline

Ambiguity here has already caused problems in prior work. Use these terms and
only these terms.

| Term | Means | Never use for |
|---|---|---|
| **rich / lazy** | update magnitude, controlled by γ | anything about initialization quality |
| **wealthy / poor** | capacity at initialization, controlled by alignment `a` | anything about update magnitude |
| **generic capacity** | capacity averaged over *random* balanced dichotomies (β = 0) | current-task performance |
| **retained capacity** | capacity for a *fixed* past dichotomy `y_j` (β → ∞ limit) | generic capacity |
| **tilted capacity `(β)`** | capacity under `P(y) ∝ exp(β⟨y, y_j⟩)`; β=0 → generic, β→∞ → retained | retained (which is only its β→∞ endpoint) |
| **effective utility `psi_eff` (Ψ_eff)** | the `E[c]/E[a]` capacity factor (excess-compression term) in `α = Ψ_eff·(1+R_eff⁻²)/D_eff` | center–axis alignment ψ |
| **center–axis alignment `ψ`** | the pairwise geometric measure `E[\|⟨s⁰_μ, s¹_ν⟩\|]` | effective utility `psi_eff` |
| **rho_c_glue / rho_c_signed** | abs-unnormalized center corr (reporting/comparability) / signed-normalized (required H1d instrument); `rho_convention: both` always | each other; never drop either; numbers not comparable |
| **readout similarity** | Hamming-based similarity between dichotomies | input/feature similarity |
| **feature similarity** | correlation between manifold *arrangements* across tasks | readout similarity |
| **drift** | any change in representation across boundaries | forgetting (which is a performance quantity) |
| **rotation / expansion / reuse / overwrite** | the four integration modes, each with a defined estimator | loose description of drift |

Variable naming in code must follow this. `gamma` is update magnitude.
`alignment` (or `a`) is initial geometry. Do not introduce `richness` as a
variable name — it is ambiguous between the two.

**Retired (2026-08-10):** "aligned capacity" / "aligned margin" — use
`retained_capacity` (fixed `y_j`) or `tilted_capacity(β)`. Also keep **`psi_eff`
(effective utility — a capacity *factor*)** strictly distinct from **`ψ` /
`center_axis_alignment` (a pairwise *geometric measure*)**; they are different
quantities that historically shared the letter ψ. Never conflate them in prose or
in variable names.

## 4. Numerical conventions

- Float64 throughout the geometry pipeline. Capacity estimation involves
  pseudo-inverses of near-singular Gram matrices; float32 is not sufficient.
- All pseudo-inverses use an explicit rcond; do not rely on library defaults.
  Log the effective rank whenever a pseudo-inverse is taken.
- Random state: every stochastic component takes an explicit
  `numpy.random.Generator`. No global seeding, no bare `np.random.*`.
- Any function that samples must be reproducible from (seed, config) alone.
- **α aggregation is the harmonic mean.** Wherever a per-manifold capacity vector
  `α_i` is reduced to one number, use `1 / mean(1/α_i)`. `α = P/N_crit` and
  critical dimensions add across manifolds, so an arithmetic mean of `α_i` is a
  bug, not a convention choice. `replicaMFT` returns such a vector.
- **`α_mf` is generic-only.** `replicaMFT` is label-invariant by construction
  (`00` §6.1, `third_party/VENDORED.md`). Never pass, fake, or imply a `y` for it;
  never use it for retained or tilted capacity. Those route through `α_sim`
  (`correlated_capacity`) and `α_core` (`src/glue/core.py`).
- **`res_coeff0` is not ρ_c.** Record it as `center_cos_abs` if recorded at all.

## 5. Experiment hygiene

- **Paired initialization.** One shape matrix `W_tilde` is drawn per seed and
  reused across *all* conditions in that seed, scaled per condition. Conditions
  must differ only by the scalars under test. This is not an optimization; it is
  the statistical design.
- **Shared streams.** Stream realizations are generated once and reused across
  configurations. Stream identity is a blocking factor, not noise.
- **Manipulation checks are unconditional.** Every run logs
  `||ΔW_m||/||W_m||`, per-module NTK change, per-module output-variance share,
  and per-module gradient-norm share, at every boundary. These are how we detect
  that an experiment silently stopped testing what we think.
- Every result artifact records the full config, the git SHA, and the estimator
  version. Geometry numbers computed under different estimator versions must
  never be pooled.
- **Any run projected over ~30 minutes gets one measured arm first.** Time a
  single arm end-to-end, then scale by arms/workers. This is a rule, not a
  lesson: cost has been mis-projected three times in this project — the Phase 1
  grid by 9× (3 h projected, 26 h actual), the scaling curve by measuring at
  `n_t=20` while the grid ran at `n_t=200`, and the γ=30 extension by 3.8× in the
  safe direction (45–55 min projected, 741 s actual, from applying a 96-worker
  contention factor to a 64-worker wave). Every miss came from projecting;
  measuring first has been right every time it was applied. Contention, not
  per-arm work, is what the projection gets wrong, and it does not extrapolate
  across worker counts — so the measured arm must be re-measured if the worker
  count changes materially.

## 6. Things that are known to be uncertain

Do not paper over these. If code depends on one, mark it `# UNCERTAIN:` with a
pointer to the relevant section of `docs/00-math-spec.md`.

- **Resolved (2026-09-01, Verification 1 signed):** the base-width constant
  relating μP to NTP (§4.2). A.3 does not state it (dangling caption).
  Independent derivation from Table 1 confirms `(N/N_base)`. Sequential L=3
  training still waits on a separate μP-at-L=3 derivation.
- **Resolved (2026-08-10):** the capacity decomposition is the **exact** identity
  `α = Ψ_eff·(1 + R_eff⁻²)/D_eff` (ICLR 2026 §B.3), not an approximation. The old
  two-factor `(1+R⁻²)/D` was the ICML approximation; `Ψ_eff` is the missing third
  factor. The `φ(R√D)` (MMCR) form is still **wrong** here — any code path
  assuming it is a bug.
- The legacy estimator takes a margin parameter κ; our reference formulation is
  κ = 0. Estimators are **not** interchangeable at κ ≠ 0.
- Whether the alignment knob `a` actually moves capacity-at-init monotonically is
  an open empirical question (Gate 2). Code must not assume it does.

## 7. Definition of done

A component is done when:

1. Its validation test in `docs/02-validation-suite.md` passes.
2. It has a noise floor measured, not assumed.
3. Its cost at target parameters is measured and recorded in `results/cost_model.json`.

"It runs without erroring" is not done. Geometry estimators fail silently by
returning plausible-looking numbers.

## 8. What to do when blocked

Prefer asking over guessing. This codebase's failure mode is not crashes; it is
producing well-formatted numbers that mean nothing. If a spec is ambiguous, the
correct action is to stop and flag it, not to pick an interpretation.

## 8.1 Verified vs inferred (applies to specs too, not only agent output)

Every factual claim — in a reference sheet, in a spec file, or in chat — is either:

- **verified**: the primary source was fetched and read (PDF/paper/code/rendered
  table), OR
- **inferred**: derived from a summary, a search snippet, another repo doc, or
  reasoning.

Rules:

1. Mark which. In sheets use the header `Status:` and `Derived from:` fields
   (`docs/reference/README.md`). In chat, say "verified (read X)" or "inferred".
2. **Any recommendation resting on an inferred claim is downgraded to a question
   until the claim is verified.** Do not act on it. (This is the half that matters:
   tagging alone does not stop a bad action; the downgrade does.)
3. This applies to the numbers already written into `00-math-spec.md` and the other
   specs. A cell transcribed from a search snippet is `inferred` until checked
   against the source, regardless of who wrote it.
4. **Claims about our own record cite where the record lives, or they are marked
   inference. This binds every participant, human included** (added 2026-08-12, at
   Kati's instruction, after she presented reconstructed grid results as record —
   see `results/LOG.md` "Finding 5", closing note). A result stated as measured names
   the arm count, the file under `results/`, or the script that produced it. A number
   recalled from memory is `inferred` until it is looked up.
5. **Structural impossibility is the cheapest check available — run it first.** Two of
   the fabricated claims were not merely unsupported but impossible: a cross-module CKA
   value when no CKA code exists in the repo, and a null C1-vs-C2 comparison when every
   arm on disk is `('A','B')`. Before checking whether a reported number is *right*,
   check whether the code and data to produce it exist. That is one `rg` and one
   `Counter` over the specs, and it refutes an entire class of claim outright.
6. **Accidental correctness does not launder a fabricated claim.** The fabricated
   Ψ_eff result ("rises in absolute terms during training") supported an interpretation
   that survived checking, but the real mechanism was different: a retained-versus-generic
   gap, not an absolute rise. When a claim turns out unsupported, re-derive the
   conclusion from the real measurement rather than keeping the conclusion and swapping
   in the new evidence.

## 8.2 Artifact check — mandatory before interpreting any measurement

**Before stating what a result means, state what would have to be true for it to be
an artifact, and check that.** Write both the candidate artifact and its check into
`results/LOG.md` alongside the result. A result reported without this step is not
finished, whatever its p-value.

This is not a general exhortation to be careful. It is here because every silent
bug in this project so far has been caught by a diagnostic rather than by a test,
and each would have produced a clean, publishable-looking number:

| Result as it first appeared | Would have concluded | Actual artifact |
|---|---|---|
| Non-convergence at low γ | γ needs a bigger step budget | Accuracy saturates at step 1 — the kernel readout is sign-correct immediately, so stopping on accuracy halted before any feature learning |
| Loss plateau at 0.34 below γ≈3 | Lazy arm is expressivity-limited | `lr0` too small; the exact readout solution reaches 0.0002 |
| `D_eff` ratio 1.52 at `D`=8 | Mode offset grows steeply with dimension | `P(D+1)` = 144 of 150 ambient — arrangement nearly degenerate |
| DTW residual 1.18 for γ=1 vs 10 | **H3 is dead** | Coverage required in *either* series, so a static trajectory matched the other's flat opening |

The last one is the case to remember: a false negative that would have killed a
hypothesis and looked like a finding.

Checks that have earned their place as defaults:

- **Compare against an exactly solvable limit.** The readout solved by least
  squares; point manifolds where `α = 2`; a zero-padded embedding where the answer
  must be unchanged.
- **Vary the thing that should not matter.** Ambient dimension at fixed geometry,
  `n_t`, seed, `P`. If the number moves, it is the estimator's, not the system's.
- **Check the null direction too.** For any alignment or fitting procedure, confirm
  it *fails* on inputs it should fail on. A method that always finds agreement has
  not tested anything.
- **Ask whether the quantity moved at all** before interpreting agreement about how
  it moved. A static trajectory or a saturated metric agrees with everything.
