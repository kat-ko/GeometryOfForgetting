# 33 — A–B–A results: one-page draft

**Status: DRAFT for Kati, 2026-11-10.**
- Sources: docs/28 §9.3, docs/30, docs/31 and docs/32 only. Each number below is copied from a table
  generated there.
- Figures: F1–F4 (`docs/figures/aba_F*.png`, docs/30 §3).
- Numbers are given as **seeds 8–15 (confirmatory) | seeds 0–7 (original)**, as means with 99.17%
  t-intervals over n = 8. The two runs are never pooled.

## Setup

A one-hidden-layer ReLU network (N = 300) is trained by full-batch gradient descent. Its richness is set by
the μP scale γ0 ∈ {0.1, 1, 10}. It learns a task A, then a task B, then A again. B changes the labels by a
Hamming distance h out of P = 16 and the input centres by s_f, where 1 is frozen, 0.9 is drift and 0.1 is
jump.

Every phase is trained to matched loss (≤ 0.05), so **every γ0 claim holds at matched current-task fit**.
Capacity is the retained fixed-label GLUE capacity, log α_A, of A's labels on A's manifolds.

The replication (seeds 8–15) is the confirmatory run, because seeds 0–2 of the original share draws with the
exploratory runs.

## Results

**Richness raises what A builds (H2, F1a).**
- Every step up in γ0 increases the init→A gain:
  - 0.1 → 1: +0.861 [+0.828, +0.894] | +0.859 [+0.837, +0.882];
  - 1 → 10: +0.944 [+0.878, +1.009] | +0.977 [+0.912, +1.043].
- The gain itself is about 0.25 at γ0 0.1, 1.1 at γ0 1 and 2.06 at γ0 10.

**…and the fraction of it that B removes (H1, F1b).** At h 4, B removes a larger share of A's gain at γ0 10
than at γ0 0.1:

| h 4 configuration | f(10) − f(0.1), seeds 8–15 | seeds 0–7 |
|---|---|---|
| frozen | +0.332 [+0.229, +0.436] | +0.277 [+0.154, +0.400] |
| drift | +0.333 [+0.233, +0.434] | +0.313 [+0.225, +0.401] |

At h 8 (frozen and drift; not read for H1), B removes 0.74–0.91 of the gain at every γ0 (seeds 8–15).

**The sign of H1 survives the stopping protocol; its size does not (F4).** Seeds 8–15 only.
- **Sign.** Under every combination of criterion or fixed training on A and on B, the h 4 drift contrast stays
  positive, with every interval above 0: +0.18 to +0.33.
- **Size.** The size depends on the protocol, by two routes:
  - **γ0 10.** Over-training A lowers f(10): −0.08 at drift, −0.14 at frozen.
  - **γ0 0.1.** Extra B updates raise f(0.1): +0.06 at drift, +0.07 at frozen.

  Both routes are "nonzero, possibly trivial" at SESOI 0.10.
- **h 4 frozen.** The contrast is unresolved when both phases are fixed.
- **Interpretation (untested):** updates past the criterion barely move a rich learner.

**The rich learner loses capacity through utility and dimension (H5a, F2).**
- At γ0 10, utility + dimension carry 0.920–0.933 | 0.935–0.952 of the A → B loss in all four h 4/h 8 frozen
  and drift configurations.
- At γ0 0.1 the radius share is 0.38–0.41 | 0.36–0.38.
- **H5b is not replicated.** "No channel carries more than 0.5" held at h 8 and was unresolved at h 4 in
  seeds 8–15.

**Relearning A restores the gain (H6, F1c).**
- ρ ≥ 0.9 at each of the 12 (γ0, h 4/8, frozen/drift) strata, in both runs.
- The lazy learner overshoots: ρ = 1.08–1.17 | 1.10–1.20 at γ0 0.1.

**Accuracy follows capacity at frozen input (F3).** The bottleneck-probe accuracy on A rises with log α_A
across phases and γ0. This is descriptive.

**Same labels under drift add capacity (H3).**
- B with h 0 at drift raises A's capacity: +0.048 [+0.033, +0.062] | +0.035 [+0.015, +0.055] at γ0 0.1, and
  +0.070 [+0.049, +0.090] | +0.066 [+0.041, +0.091] at γ0 10.
- It is **positive in both runs, with the CI excluding 0, but below the pre-set resolution in the
  replication**.

**Prior learning slows a related task specifically (H4).** At γ0 10, the specific behavioural forward
transfer, FT_beh(h 8 frozen) − FT_beh(h 8 jump), is −0.459 [−0.591, −0.328] | −0.417 [−0.606, −0.229].

## Limits

- **Exposure.** Under matched loss, the learners' exposure differs 40–60-fold between γ0 0.1 and 10. docs/31–32
  bound how much this matters for H1. It is not addressed for the other hypotheses.
- **Jump.** H6 was not read at jump. At h 4 jump and h 8 jump, γ0 1 and 10 sit below 0.9 in seeds 8–15:
  for example, ρ = 0.862 at h 8 jump, γ0 10 (docs/30 §1).
- **Provenance.** One stray seed-0 arm in the replication checkpoint is excluded by key (E18).
- **One learner.** Richness here is γ0 within one architecture. The stream grid (docs/29) and the
  multi-learner extension are pending.
