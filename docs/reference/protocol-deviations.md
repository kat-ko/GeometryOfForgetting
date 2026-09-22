# Protocol deviations from source papers

Stated departures from a source protocol, as opposed to implementation of it.
A deviation that is the right call still has to be recorded, because capacity
and similarity geometry are known to depend on the thing we changed.

---

## D.1. Balanced dichotomies (Chou D.1.1)

**Source.** Appendix D.1.1 samples the `P` labels randomly from a uniform
distribution on `{±1}`, not balanced.

**What we do.** Every task dichotomy is balanced: `y ∈ {±1}^P` with
`Σ_i y_i = 0` (I9; `00` §2.2, §6). Generic capacity averages over balanced
dichotomies only.

**Why.** Unbalanced splits are degenerate and contaminate the similarity axis
that the Hiratani 2×2 is built on. Hamming-controlled `s_r` is not well-defined
in the same way for unbalanced labels.

**Why it is a deviation, not a silent improvement.** Capacity depends on label
sparsity. Chung, Lee, Sompolinsky, *Classification and Geometry of General
Perceptual Manifolds*, Phys. Rev. X 8, 031003 (2018), derive a sparsity–radius
scaling relation. Restricting to balanced dichotomies therefore changes the
ensemble the estimator sees relative to the generator's uniform `{±1}` protocol.

**Where recorded.** This file; `00` §2.2 and §6 (estimator); I9 in `AGENTS.md`.

---

## D.2. CF/CFr peak is `a_{i,i}`, not Graldi's running max (A.3/A.4)

**Source.** Graldi Def. A.3/A.4 (App. A.1): peak on task `i` is
`max_{t ∈ {i,...,T−1}} a_{t,i}`, then drop to `a_{T,i}`. Average over `T−1`
tasks.

**What we do.** `pipeline.forgetting_metrics` uses `acc[j, j]` as peak.

**Why it is recorded.** Equivalent when accuracy on `j` is highest at the end
of training `j` and never recovers. Not equivalent if a later task restores
accuracy on `j` before the end. **S-HH is the geometric case where a past
task improves during subsequent training.** If that appears in accuracy as
well as in retained capacity, the two formulas diverge in the condition the
paper is about. Current behavioural CF there is ~0; if accuracy is at
ceiling they still coincide. The 2026-07-31 sheet's toy-example check passed
because in that example the two references coincide (`07-writeup.md` §A.7).
Do not switch the code to the running max without a pre-commit: existing
arms would re-rank.

**Where recorded.** This file; `docs/reference/cl-metrics.md`;
`07-writeup.md` §A.7.

---

## D.3. Training schedule vs Graldi's infinite-width arm (A.4)

**Match.** 2-layer ReLU MLP, full-batch GD on MSE, last layer 0, SGD without
momentum or weight decay. Same model class as their theoretical arm.

**Deviations.** (1) Cosine LR, no warmup, restarted per task vs constant LR
with matched-loss stopping. (2) Their MLP arm is 30 MNIST samples, 2 tasks,
`ρ = 0`; `γ₀* ≈ 0.1` is from the ResNet experiments. Any comparison of `γ*`
is across architectures.

**Where recorded.** `00` §4.1; `parameterization.md`;
`parameterization-derivation.md`.

---

## Not deviations (signed as matching source)

- Chou B.5 recovery settings (`validation-settings.md`, signed 2026-09-01).
- Chou D.1.1 generator, including pre-`R` unit-norm (`manifold-generator.md`,
  signed 2026-09-01).
- Graldi Table 1 four cells (`parameterization-derivation.md`, signed
  2026-09-01). `(N/N_base)` is an independent derivation from Table 1, not a
  deviation from A.3 (A.3 does not state the constant; the caption's forward
  reference is dangling). Verification 1 signed.
