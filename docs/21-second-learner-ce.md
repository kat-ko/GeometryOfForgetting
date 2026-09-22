# 21 — Second learner: binary cross-entropy on the existing Hamming streams

```
Status:  limitation written. Reading: no_shared_operating_point.
         No pin. No ce_precommit.json. No slice. Closed.
Signed:  match on margin; four-stream MSE spread is the tolerance; if no
         single BCE target hits both γ inside that band, stop — do not pin
         per-γ. That reading fired. The two-fact reading of the table is
         the deliverable (docs/15 limitation 2; docs/07 §A.9).
Next:    stream axes. Order (`docs/23`), then spacing, then P=32.
         Second-learner line closed.
Not this arm: another loss, per-γ pins, reopening the CE slice, Adam on
              this document, L=3, P=32, order, spacing, CIFAR, figure edits.
```

---

## Why this is next, and why it is not a stream arm

A geometry paper that happened to use a CL stream is complete without order
and spacing. A CL-theory paper that happened to use geometry is not complete
with one learner: the joint-outcome argument is empty until a second learner
sits on the same streams.

Order and spacing first would produce setting results that a second learner
might immediately relativise. A second learner on streams that do not exist
yet tests nothing. The Hamming 2×3 already has the cliff, the drift
gradation, the crossover, and the capacity/ρ_c dissociation — so a second
learner has something to disagree with.

Not EWC. Not replay. Those change the objective. The clean contrast holds
the geometry measurement and the streams fixed, and varies one property of
the learner.

Cross-entropy first, not Adam. Smaller change. D.1.1 anticipates binary
cross-entropy with labels reassigned to `{0,1}` (`manifold-generator.md`).
`docs/15` already lists “MSE to ±1 is not classification risk” as a
limitation. This arm converts that sentence into a measurement.

The paper sentence that needs this arm: the field's domain-IL and
task-IL are corners of the Hamming factorial *and the cells between*,
axes independent, GLUE at every cell. Standard benchmarks cannot sit
between those scenarios. Until a second learner agrees or disagrees,
that sentence is a claim about this MLP.

---

## Design (held)

- **Streams.** The existing Hamming 2×3: same reserved `stream_id`s
  10000–10007, same `s_r` × input levels, same T=16, stride=2, P=16.
  Dichotomies and arrangements are regenerated from `stream_rng` as now;
  they pair with the MSE arms at fixed (`stream_id`, `s_r`, input level).
- **Architecture.** Unchanged: L=2, N=300, ReLU, shared scalar readout,
  readout at 0 (I5), paired init.
- **Optimiser.** Full-batch GD. Not Adam. Start at quadratic `lr0=5`
  (the MSE pin). Re-pin rule below.
- **Loss.** Binary cross-entropy. Labels remapped `{±1} → {0,1}` as D.1.1
  states. Not a departure from the generator.
- **First slice.** γ ∈ {1, 10} × 3 input × 4 Hamming × 8 ids = 192 unique
  arms, off-design directory, never `results/phase1/`.
- **Primary cell.** Unique n=8, module A, task 0, lag 12, Δ log α.
  Cliff, post-cliff range, crossover, and Δρ_c reported as on the MSE
  Hamming slice. Margin comparison (how close the two arms landed) is
  reported alongside, not assumed.

---

## Stopping — signed: match on margin, pin BCE loss by measurement

Matched-loss stopping exists so that a γ contrast is a contrast of
*regime*, not of training progress. MSE `target_loss=0.05` was pinned
empirically at `lr0=5` — Phase 0 found 0.2 left γ ≤ 1 stranded at 0.34
(`docs/01`, `docs/15`).

BCE is on a different scale. Initial loss is ~log 2 ≈ 0.693, the floor
is 0, and the loss-to-accuracy relationship is not MSE-to-±1. So 0.05
means something different, and any number picked by analogy is arbitrary.

The quantity both losses produce is the **readout margin**: distance from
the decision boundary `f = 0`,

    m = mean_b [ y_b f(x_b) ]

on the current-task batch, `y ∈ {±1}`, `f` the network output (regression
target for MSE; logit for BCE). Capacity is the load at which a margin
reaches zero; this is the operating-point analogue the two objectives
share. Also record the 5th percentile of the pointwise `y f` (hard
points). Loss value is not that quantity.

### Rejected (for the record)

- **BCE = 0.05.** Arbitrary. No reason the two losses' values correspond.
  0.05 BCE is roughly 98% mean confidence, plausibly far deeper than MSE
  0.05.
- **Matched fractional reduction** (MSE 1.0 → 0.05 is 95%, so BCE
  0.693 → 0.035). Assumes comparable reduction curves. BCE's tail is
  logarithmic and MSE's is quadratic; the same fraction sits at different
  margin levels.
- **Matched steps.** Destroys the thing matched-loss exists for. The γ
  contrast collapses back into a compute contrast.

Do not pick 0.05 by analogy and leave it unexamined.

**Measured counterfactual (one-arm replay).** Homogeneous conversion of MSE
0.05 is `m ≈ 0.684`. Measured mean margin is ~0.85 (γ=10: 0.853, γ=1:
0.832). The analogy number would have stopped BCE about 20% shallower in
margin than MSE actually stops. That is why the pin is measured.

The 2.5% spread between those two γ is small enough that a single BCE
target looks answerable. n=2 is thin for a freeze: the tolerance is the
range of mean margins on **four reserved streams × both γ**, not those
two points.

**Distribution, not only the mean.** At MSE stop, mean ~0.85 and p05
~0.33: the network is comfortably correct on most points and marginal on
a tail. BCE's gradient is largest where the margin is smallest, so it
will preferentially pull that tail in. Two arms can match on mean margin
with different distributions. Capacity is computed from anchor points —
the extreme points — so the tail is what the measurement is most
sensitive to. Report **mean, p05, and p25** for both losses at the
pinned targets, in the pre-commit and again in the slice. A substantial
distribution difference at matched mean is a stated caveat on the
comparison, not a reason to re-pin. The pin remains the mean.

### How to pin

1. **Measure where MSE 0.05 sits in margin terms** on the reserved Hamming
   streams. Checkpoints are not on disk. Training-only replay, identity
   `steps_taken` vs stored Hamming JSON. Mean `y f` at the returned state
   is the pin quantity; p05 and p25 are reported.
2. **Four reserved streams × γ ∈ {1, 10}** (frozen × s_r=0.5) give the
   tolerance: the range of those cell means. Not a number we choose.
3. **Bisect on BCE loss** at the original two cells (γ=1 and 10 on the
   Hamming one-arm stream) until both mean margins land inside that
   range.
4. **Pin that BCE value and freeze it**, exactly as 0.05 was pinned.

### If no single BCE value hits both γ within tolerance

That is the result. The two objectives have no shared operating point, so
"same learner, different loss" is not a comparison that can be made at
matched progress. Report it and **stop**. Do not pin per-γ — that
reintroduces the confound matched-loss exists to remove. Do not run the
slice.

The CE arm was proposed to test whether the cliff, the drift gradation,
and the crossover are properties of the stream or of this learner. That
test requires a shared operating point. Without one the arm cannot answer
it, but the failure is a *measured* limitation with a number attached —
stronger than the current §9 sentence "MSE to ±1 is not classification
risk." Either outcome of the pilot is reportable.

The old fallback (fractional reduction 0.693 → 0.035) applied only if
margin was not cheaply recoverable. Margin was recovered. That fallback
is retired for this arm.

### Pilot readings, named before numbers

- `shared_operating_point`: one BCE target places both γ inside the MSE
  four-stream band. Freeze that target. Report mean/p05/p25 for both
  losses.
- `no_shared_operating_point`: no such target. Stop. Do not pin per-γ.
  Do not write `ce_precommit.json`. The writeup records a measured
  limitation.
- `replay_mismatch`: four-stream `steps_taken` disagrees with stored
  Hamming records. Stop. No tolerance.
- `lr0_miss`: BCE at `lr0=5` fails to reach a candidate target. Stop the
  search. Apply the signed re-pin rule, record it, then resume — not a
  silent change inside the bisect.

### One-arm replay (not a freeze)

`scripts/measure_mse_stop_margin.py --one-arm`. Frozen × s_r=0.5 ×
stream 10000, γ ∈ {1, 10}. Both `steps_taken` lists match the stored
Hamming JSON. Wall 153 s. `results/mse_stop_margin_one_arm.json`.

| γ | mean m over 16 tasks | task 0 | mean p05 |
|---|---|---|---|
| 10 | 0.8527 | 0.779 | 0.323 |
| 1 | 0.8316 | 0.745 | 0.342 |

Spread 2.5%. Homogeneous conversion 0.684 is not this measurement.
Hard-point p05 is not the pin. Four-stream replay supplies the tolerance
before any freeze.

### Four-stream MSE replay (tolerance, not a freeze)

`scripts/measure_mse_stop_margin.py --four-streams`. Frozen × s_r=0.5,
ids 10000–10003, γ ∈ {1, 10}. All eight `steps_taken` match. Wall 252 s.
`results/mse_stop_margin_four_streams.json`.

Band **[0.8308, 0.8582]**, spread 0.0274, grand mean 0.844. γ=10 cells
0.853–0.858; γ=1 cells 0.831–0.837. p05 stays ~0.33; p25 ~0.71 (γ=10)
and ~0.66 (γ=1).

### Pilot reading (applied)

`scripts/pilot_bce_margin.py`. One-arm then eight-round bisect.
`results/bce_margin_bisect.json`.

**`no_shared_operating_point`.** When γ=10's mean sits in the band
(L=0.454, m=0.850), γ=1 is at 0.777. The BCE gap at matched L is ~0.07
against a 0.027 band. p05 is negative at every L (−0.47 to −0.60) against
MSE p05 ~+0.33. `lr0=5` reached every candidate; this is not an lr0 miss.

No pin. No JSON. No slice. Do not pin per-γ.

### What the two facts say

The between-γ gap is the pin failing. It is also a statement about BCE:
under MSE the two richness regimes sit at comparable margins at matched
loss; under BCE they do not. The loss function determines whether
"matched progress" is even definable across γ.

The negative p05 is the larger finding. At every L tried, a tail of
points remains on the wrong side of `f=0` (p05 −0.47 to −0.60 against
MSE +0.33; at the L that matches γ=10's mean, p05 is −0.49). p25 is
~0.16 against MSE ~0.71. That is not a stopping-rule problem. MSE to ±1
penalises confident-correct points as much as it rewards fixing wrong
ones, so it pulls the whole distribution together. BCE's gradient
vanishes on confidently-correct points and concentrates on the boundary,
so it produces a long negative tail while the mean looks fine.

Capacity is computed from anchor points — extreme points, near the
boundary. The two objectives differ precisely in the region the
measurement is most sensitive to. Even with a shared mean, "same learner,
different loss" would have compared representations whose anchor
geometry was shaped by different pressures. The pin was never going to
be sufficient. The pilot found the reason.

### What CE bought (descending order)

1. **A measured §9 sentence.** On this architecture, these streams, and
   this `lr0`, MSE and BCE have no shared operating point — at matched
   mean margin the BCE tail sits on the wrong side of the boundary (p05
   −0.49 against +0.33). Written: `docs/15` limitation 2; `docs/07` §A.9.
2. **A methodological point that generalises.** Comparing learners at
   "matched training progress" presumes a shared operating point. Two
   standard objectives on one architecture do not have one. Papers that
   compare losses at matched epochs or matched loss are assuming
   something checkable and rarely checked.
3. **A constraint on the geometry literature.** Manifold capacity is
   computed from anchors near the boundary, and the two objectives
   differ most there. Capacity values obtained under different training
   objectives are not straightforwardly comparable. Measured, not
   argued.

### Not another loss. Yes to Adam.

Hinge, logistic with different scaling, and focal have the same
structure as BCE: gradient concentrated near the boundary, vanishing
away from it. They will all produce the negative tail and most will have
the same between-γ mismatch. Do not spend pilots rediscovering the
obstruction.

Adam is a different variable: same MSE, same `target_loss=0.05`, same
streams, different update. The matched-loss pin transfers. That arm is
`docs/22`. I3 is licensed for it as an off-design exception, never
`results/phase1/`, with a richness-separation gate before any γ claim
is interpreted.

---

## `lr0` — signed re-pin, before any slice

`lr0=5` was pinned for MSE. BCE's gradient with respect to the logit is
bounded where MSE's is not, so the same rate may behave very differently,
especially at γ=10 where η ≈ 2.3×10³ (`docs/15`).

**If BCE at `lr0=5` fails to reach the pinned target** on the pilot
cells, `lr0` is re-pinned **once, uniformly across all cells, by the same
procedure used for MSE** (`docs/01`: 0.2 stranded γ≤1 at loss 0.34; 5.0
reaches every γ; 50 remains stable). Record the new value as a design
change **before any slice runs**. Do not re-pin after seeing which
Hamming cells fail. Do not re-pin per cell. Do not inform `lr0` from
geometry on this axis or from the reserved set.

If `lr0=5` reaches the pin, it stays 5.

---

## Sequence (this arm is closed)

1. One-arm MSE replay (done). Identity held. 0.684 vs 0.85 recorded.
2. Four-stream MSE replay (done). Band [0.831, 0.858].
3. BCE `--one-arm` (done). L=0.35 → mean 1.31, p05 −0.33. `lr0=5` reached.
4. BCE `--bisect` (done). Reading: `no_shared_operating_point`.
5. Write the measured limitation (done). `docs/15` limitation 2;
   `docs/07` §A.9; ledger `operating_point`. Do not freeze a pin. Do not
   run the slice. Do not pin per-γ.
6. Adam ran (`docs/22`). Gate `partial_manipulation`. Hamming-only
   `finding3_fails_to_recover`. Second-learner line closed. Next is
   order (`docs/23`), not a fourth learner.

Will not: write `ce_precommit.json`; pick BCE=0.05; match steps; pin
per-γ; try another classification loss; treat a tail mismatch as a
re-pin; pool populations; treat CIFAR as this finding; edit figures.

---

## Readings, named before numbers (γ=10, unique n=8)

These were the slice outcomes. They were never applied: the pin failed,
so there is no slice. They remain here as the question the arm could
not ask.

If the cliff (h=0 vs h≥2 under frozen and jump), the drift post-cliff
gradation (0.75 vs 0.25 CIs non-overlapping), and the frozen–drift
crossover reproduce under BCE, they are properties of the stream.

If they do not, they are properties of this learner under MSE, and the
paper’s scope narrows to exactly that. Still publishable. More honest than
one learner with a general-sounding claim.

Named outcomes:

- `stream_property`: cliff, drift gradation, and crossover all reproduce
  (same signs and the same “resolved vs overlapping CI” pattern as MSE).
- `learner_property`: at least one of the three fails. Report which.
  Do not salvage a stream claim from the survivors after seeing numbers.
- `finding3_fails_to_recover`: at s_r=1, drift vs jump does not keep
  gain vs loss on Δ log α. Stop. The CE streams are not the same object.
- `mixed`: γ=10 and γ=1 disagree on stream_property vs learner_property.

Cliff definition as on the MSE slice: floors(s_r=1.0) − floors(s_r=0.75);
post-cliff range = |floors(0.75) − floors(0.25)|; a monotonicity claim
requires that range to clear two floors.

The margin comparison (mean, p05, p25 for both losses) is reported with
the reading. A distribution difference at matched mean is a caveat, not
a named slice outcome.
