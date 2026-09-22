# 20 — Restatement of the existing grid, and the next arm

```
Status:  Finding 4 restated. Second-learner line closed. Next: order.
Signed:  Hamming 2×3; cliff not dose; three operating-point conditions
         in docs/07 §A.9. Framing: geometry with a well-specified stream
         (this file §9), written before the Hamming-only numbers.
         Order pre-commit: docs/23, results/order_precommit.json.
Not next: a fourth learner, Adam, CE, replay, EWC, architecture, L=3,
          testing the Adam accumulation hypothesis, figures,
          CIFAR-as-finding, treating order as a substitute second learner.
```

---

## 1. Finding 4's recast — accept or rewrite before it is drafted

The existing notebook sentence:

> The two similarity axes act separately. Readout similarity drives Δρ_c
> positive (centres converge). Feature similarity drives it negative
> (centres decorrelate).

On this grid those axes are not similarity levels. Consecutive `s_r = 0.9`
is Hamming 0: the dichotomy never changes. Consecutive `s_r = 0.1` is
Hamming 8: it changes to orthogonal. Consecutive `s_f` is slow drift versus
a jump. The 2×2 is **task repetition × input change**.

**Claim (signed, after the Hamming arm):**

> Input change drives centre convergence and saturates almost immediately.
> Task repetition does not, as a general statement. On the registered 2×2,
> unique n=8, drift vs jump at a repeated task differs by +0.055 in Δρ_c
> (drift +0.055, jump +0.110). On the reserved Hamming set, the same
> isolation is unresolved under drift (+0.002, CI includes 0) and +0.104
> under jump. Those are two populations; they are not pooled. Under frozen
> input, task change drives divergence and that Δρ_c is graded (0.75 vs
> 0.25 CIs do not overlap) while capacity is a cliff. Capacity and centre
> geometry dissociate. Additivity on the registered 2×2 remains an
> empirical fact about how repetition and input change combined there; it
> is not evidence that two similarity axes act separately, and it is not
> a finding that repeating a task drives convergence.

What this is not: a statement that “readout similarity drives convergence”
with the word “repetition” swapped in. Repetition is a binary on the 2×2.
On the reserved set it is not even a resolved Δρ_c effect under drift.

γ=10 unique n=8, already measured: S-HH +0.055 (same task, drift), S-LH
+0.110 (same task, jump), S-HL −0.055 (different tasks, drift), S-LL +0.011
(different tasks, jump, unresolved). Same-task mean +0.082; different-task
mean −0.022. Jump minus drift at fixed same-task +0.055. Interaction +0.006
(0.82 unique SEM), unresolved.

---

## 2. Restatement of what was already measured

For pasting. Not applied. Setup table first.

### Setup table (notebook §2) — this is the first line to change

| property | in these runs |
|---|---|
| **similarity (consecutive)** | Hamming of the split, and centre correlation of the arrangement, each set on the step from `t−1` to `t`. At P=16 the requested `s_r=0.9` is Hamming 0 (identical split); `s_r=0.1` is Hamming 8 (orthogonal) — which Hamming distances exist is set by P, see scope. The 2×2 is therefore **same-task vs different-task**, crossed with **slow drift vs jump**. Long-range similarity is a realized property of the walk, not a design parameter. At lag 12, high-`s_f` streams still have `s_f(0,12)≈0.28`; low-`s_f` streams are at 0. |
| **order** | fixed, one pass, no designed recurrence |
| **scope** | one split over sixteen manifolds, uniform across tasks. P=16 sets which Hamming distances exist: `{0, 2, 4, 6, 8}`, hence `s_r ∈ {1.0, 0.75, 0.5, 0.25, 0}`. The design used only the identical cell and the far end. |
| **change** | consecutive arrangement displacement is `1−s_f`: **0.1** in the cells labelled high feature similarity, **0.9** in the cells labelled low. Change co-varies with the feature axis by construction. Training is still blocked (one task, then the next). “Abrupt at every boundary” describes the training cut, not the geometry trajectory. |
| **access** | no task identity, no boundary signal, no reset |
| **agent** | one hidden layer, ReLU, 300 units, shared scalar readout, full-batch GD on MSE |

Eight unique arrangement draws sit behind each reported cell.

### Notebook §3–4 (after the table)

The sixteen clouds are **not** held fixed in the reported 2×2. `A_0` is
drawn fresh; `A_t` redraws centres at correlation `s_f`. Points are
re-realized at every task. `s_f=1` (centres copied) is `S-fixed-r`, which
was registered and not run. Even there, centres are copied and points are
re-sampled; bitwise-identical clouds are not what the code does.

A task is a balanced split of the sixteen manifolds. In S-HH and S-LH every
task is the same split (`y_t = y_0`). In S-HL and S-LL the split changes by
eight labels each step.

Interference in the reported 2×2 is therefore not “rule-only.” Rule-only is
the frozen-arrangement condition, which has no results.

### Finding 1 (γ composition)

Unchanged as a γ result. One flag, not a rewrite of the numbers: the
three-corner pool (S-HL, S-LH, S-LL) mixes different-task streams with
S-LH, which repeats the same split on a jumping arrangement. The
three-corner pool is the set of cells that lose retained capacity on task
0, whatever mechanism produced the loss. That is the estimand. It is not a
result about a homogeneous class of forgetting streams.

### Finding 2 / surplus (generic vs retained)

Unchanged. Generic and retained are still two ensembles on the same
representation.

### Finding 3 — a finding, not a correction

When the dichotomy is held at `y_0` for every task, retained capacity of
`(A_0, y_0)` at lag 12 **rises** if the arrangement drifts and **falls** if
it jumps. At γ=10, unique n=8, module A: S-HH **+6.2** floors, S-LH
**−55.8** floors. Welch p = 8×10⁻⁷; all eight seeds in the same direction.
Every GLUE term is positive-net in S-HH (utility and dimension raising α)
and negative in S-LH. Behavioural CFr on task 0 stays near zero in both
(0.001 vs 0.032). The cells differ only in whether the input distribution
drifts or jumps.

This is domain shift with the task held constant. It is not backward
transfer from other tasks, and it is not a restatement of a failed
similarity contrast. The existing literature rarely isolates that.

At γ=1 the same split is +11.8 vs −14.0 floors. The gain under drift peaks
in the middle of the γ grid; the loss under a jump grows with richness.

### Finding 4

The empirical registered 2×2 and its unresolved interaction stay, as facts
about that population. The sentence “repeating a task drives centre
convergence” does not. Draft from the signed §1 claim after the Hamming arm.

Under frozen, Δρ_c is graded in Hamming while capacity is a cliff. Under
drift, capacity is graded while Δρ_c is a cliff then unresolved. Jump Δρ_c
is already +0.104 at Hamming 0. The two measures come apart.

### Closing paragraph

Finding 1 is the loss and its composition with γ. Finding 2 is generic
versus retained on the same representation. Finding 3 is input drift at a
fixed task: the same split accumulates retained capacity under a drifting
input distribution and is destroyed under a jumping one. Finding 4 is that
input change drives centre convergence and saturates almost immediately;
task change drives divergence under a frozen input; capacity and Δρ_c
dissociate. The registered +0.055 and the reserved +0.002 are both true
and must not be pooled.

---

## 3. Frozen as a third input level — signed 2×3

Ask: four Hamming levels × {frozen, drift, jump} × γ ∈ {1, 10} × 8 reserved
stream_ids, seed=0, a=0, N=300, quadratic, matched-loss 0.05, P=16.

Hamming is exact at these values (no rounding):

| requested `s_r` | h | realized consecutive `s_r` | what it is |
|---|---|---|---|
| 1.00 | 0 | 1.00 | same task |
| 0.75 | 2 | 0.75 | two labels flip |
| 0.50 | 4 | 0.50 | four flip |
| 0.25 | 6 | 0.25 | six flip |

`s_r=1.0` stays as a designed same-task cell.

Input levels: frozen `s_f=1` (existing `S-fixed-r`: centres copied, points
re-realized); drift `s_f=0.9`; jump `s_f=0.1`. Frozen is the rule-only cell
the setup has been describing and which has no experiment.

| factorial | unique arms | vs registered-like 2×2 |
|---|---|---|
| 4 × {drift, jump} × 2 γ × 8 | **128** | — |
| 4 × {frozen, drift, jump} × 2 γ × 8 | **192** | **+64** (one third more) |

Arm count: the extra 64 is one recurrence first-slice. That comparison is
about factorial size, not wall. Each arm has `tracked_stride=2` (106 evals).
Measured one-arm (γ=10, frozen, s_r=0.5, stream 10000): **1243 s**, all 16
tasks reached target. Serial 192 × 1243 s ≈ 66.3 h. Do not divide by 192
workers as a forecast.

Frozen at `s_r=1` is continued training on the same centres and the same
split — the no-change null. Frozen at `s_r<1` is the rule-only cell the
setup has described since the first draft and which has never been
measured. Drift vs jump at `s_r=1` recovers finding 3 on the reserved set.
Those three cells are why the third input level is not decorative.

**Signed:** run the 2×3. Off-design directory, not `results/phase1/`.
One-arm first.

---

## 4. Schedule — signed B: T=16, stride 2

The registered schedule is T=16, `tracked_stride=4`. Lag 12 exists only for
task 0. Lag 4 exists for tasks 0, 4, 8, 12. W5b: at matched lag 4, task
position (×1.7) is larger than the whole within-task-0 lag effect (×1.34).
Figure 2 was immune only because it restricted to task 0.

A (keep stride 4) is cheaper and matches finding 3's estimand. The
continuity argument is weaker than it looks: §3's recovery check already
re-establishes finding 3 on the reserved set at `s_r=1`. This arm is the
template anchored streams will inherit. If they inherit T=16 with stride 4,
the confound propagates into the axis designed to fix long-range
similarity.

**Signed B, specifically:** keep T=16; change the measurement stride so
several tasks share a lag. Extending T is not needed yet.

At T=16, which tasks have lag 12 / lag 4:

| stride | n_evals (2 modules) | lag-12 tasks | lag-4 tasks |
|---|---|---|---|
| 1 | 304 | 0, 1, 2, 3 | 0–11 |
| **2 (lock)** | **106** | **0, 2** | **0, 2, 4, 6, 8, 10** |
| 3 | 54 | 0, 3 | none |
| 4 (registered) | 38 | 0 only | 0, 4, 8 |

Stride 2: tasks 0, 2, 4, 6, 8, 10 share lag 4, so position is a reported
covariate rather than a confound. Lag 12 is available to two tasks (0 at
boundary 12, 2 at boundary 14) without extending T. Training is still 16
tasks; geometry evals ~2.8×. `EVAL_BUDGET=40` is a default-spec test only;
`run_arm` does not refuse.

Primary cell remains unique n=8, module A, **task 0, lag 12** (still
present at stride 2). The lag-4 series across task position is reported,
not a reading.

---

## 5. Pre-commit — signed

`results/hamming_precommit.json`. Written before any Hamming arm.

- **Question.** At P=16, is retained capacity of `(A_0, y_0)` graded in
  consecutive Hamming at each input level, and does frozen (rule-only)
  sit between drift and jump at fixed Hamming?
- **Population.** Reserved arrangements, `stream_rng`, ids
  10000–10007, seed=0, unique n=8. Not old / `legacy_seed`.
- **Held fixed.** P=16, N=300, a=0, L=2, quadratic, `lr0=5`,
  `target_loss=0.05`. Pins not informed by the reserved set.
- **Readout.** `s_r ∈ {1.0, 0.75, 0.5, 0.25}`, requested as those
  values so Hamming does not round.
- **Input.** {frozen, drift, jump} with `s_f ∈ {1.0, 0.9, 0.1}`.
- **T / schedule.** T=16, `tracked_stride=2` (§4 lock).
- **First slice.** γ ∈ {1, 10}. Primary at γ=10. γ=0.03 is not a control.
  4 × 3 × 2 × 8 = **192** unique arms.
- **Primary object.** Unique n=8, module A, task 0, lag 12 of Δ log α
  against Hamming, per input level. 95% CI = mean ± 1.96 SEM. Lag-4 series
  across task position is reported, not a reading.
- **Recovery.** At `s_r=1`, drift vs jump must recover the sign of
  finding 3 (gain vs loss) or the slice is not measuring the same object.
- **Rule-only.** Frozen × `s_r<1` is interference with centres held.
  Frozen × `s_r=1` is continued training with no task change and no centre
  motion.
- **Δρ_c.** Reported. Graded Hamming on frozen is the first test of
  whether repetition-vs-change was hiding a dose. Not the primary.
- **Misses.** An arm that misses `target_loss` on any task is unusable
  and counted. `lr0` is not re-pinned.
- **One-arm first.** γ=10, `s_r=0.5`, frozen, first reserved id.
  Governs nothing. Then the slice.
- **Will not.** P=32; old arrangements; inform `lr0` from this axis;
  pool Hamming 0 with 0.75; treat consecutive `s_r` as lag-12 `s_r`;
  expand A8; edit figures from this arm; inherit stride 4.

**Readings, named before numbers (γ=10, unique n=8):**

- `monotone_dose`: at frozen and at drift, mean Δ log α of task 0 is
  monotone in Hamming. Dose-response is established.
- `binary_only`: `s_r=1` separates from the three changing-task levels,
  which do not order. The axis is still same-vs-different, now with extra
  points that do not help.
- `nonmonotone`: report the order. Do not fit a story afterwards.
- `finding3_fails_to_recover`: at `s_r=1`, drift vs jump does not keep
  gain vs loss. Stop. The new streams are not the same object.
- `frozen_intermediate`: at fixed Hamming, frozen sits between drift and
  jump on Δ log α. Input change is graded, and the rule-only condition is
  the endpoint.
- `frozen_outside`: frozen falls outside the drift–jump interval. Copying
  centres is not the zero of the input axis, and the input manipulation is
  not a single dimension. Genuinely possible: points are re-realized at
  frozen centres; bitwise-identical clouds are not what the code does.
- `mixed`: γ=10 and γ=1 disagree on which of the first three applies.
  Report the split.

---

## 6. Order of work

Hamming slice done. Finding 4 restated (this file, notebook). Second-learner
line closed (`docs/07` §A.9). Next is a stream arm, single-learner.

1. **Second learner — binary cross-entropy — on the existing Hamming streams.**
   Done as a pilot. Reading `no_shared_operating_point`. Closed.
2. **Adam** on the existing Hamming streams. Gate `partial_manipulation`.
   Hamming-only `finding3_fails_to_recover`. Closed. Not a comparison.
3. **Order**, audit's weaker prediction, matched composition, schedule B
   (`docs/23`). Pre-commit on file. Not yet run. Not a substitute second
   learner.
4. **Spacing**, intervening Hamming and input level held.
5. **P=32** with its own §2a gate.

Not: a fourth learner, another classification loss, per-γ pins, reopening
CE, L=3 (A6), replay, EWC, testing the Adam accumulation hypothesis,
figure edits, naturalistic arm except as estimator validation.

## 7. What the table says (supersedes the named readings)

The arm worked. Finding 3 recovered. The raw table is a cliff, not a dose.

| γ=10 | cliff 1.0→0.75 | range 0.75→0.25 |
|---|---|---|
| frozen | 75.3 floors | 0.7 |
| drift | 68.0 | **13.3, CIs do not overlap** |
| jump | 21.8 | 0.7 |

Same shape at γ=1: cliffs 35.0 / 36.4 / 12.3, post-cliff 0.4 / **10.0** / 2.5.

Task-similarity gradation is visible only when the input also drifts. Frozen is
insensitive to how much the task changed; jump is dominated by the input change.
Frozen and drift therefore cross: at s_r=0.75 drift is better (−61.3 vs −68.2);
at s_r=0.25 frozen is better (−68.9 vs −74.6). That is why `frozen_outside` fired.

`monotone_dose` and `mixed` failed to discriminate this (artifact family A.8).
Going forward: a monotonicity reading requires the post-threshold range to
clear the floor.

**Δρ_c comes apart from capacity.** Frozen capacity is a cliff; frozen Δρ_c is
graded (0.75 vs 0.25 CIs do not overlap). Drift capacity is graded; drift Δρ_c
is a cliff then unresolved. Jump Δρ_c is already large at s_r=1. Repeating a
task does not, on this population, drive a resolved convergence under drift
(+0.002, CI includes 0).

**Lag 4, schedule B.** Position ratio |task 0 / task 8| is cell-dependent
(~1.2–1.4× on most forgetting cells; ×3.45 on jump at s_r=1; inverted on
drift at s_r=1). W5 is measured here, not argued.

**P=32** is now a cliff-resolution arm: h=2 is 6.25% of labels against 12.5%
at P=16. Still needs its own §2a gate. Still a scope change. Not next.

Full tables: `results/hamming_slice.md`.

---

## 8. The sentence that makes this CL theory

The field's two named scenarios are domain-incremental and task-incremental
learning. The Hamming 2×3 contains both as corners **and the cells between
them**, with the two axes set independently and a geometric decomposition at
every cell. Nothing in the standard benchmarks lets you sit between those
scenarios, because the partition is inherited from the dataset.

That sentence is already true of the experiment. It is not a license to cite
CIFAR. Exporting GLUE to a stream whose similarity structure cannot be stated
would be the opposite of Line 2. A naturalistic arm, if it ever runs, is
estimator validation — does GLUE work on real representations — not
generalisation of the finding.

What the sentence needs to survive a reviewer is the second learner.
Until a second learner agrees or disagrees, it is a claim about this
MLP. Three attempts could not set that comparison up. CE failed the
shared operating point (`docs/21`). The Adam gate failed the
manipulation (`docs/22`). The Adam slice failed recovery of finding 3.
None of these is a second learner disagreeing about the finding. All
three are the comparison not being set up. There is no comparison.

That is not a setback. **The joint-outcome claim has to be made more
carefully than "a second learner agrees."** What the paper can say:
every finding is a property of this learner on these streams, the
streams are specified more completely than anything in the literature,
and three attempts to vary the learner ran into operating-point
obstructions that are themselves measured. That is stronger than a
CL-theory claim resting on one comparison that did not work.

## 9. Framing, before the Hamming-only arm reports

Worth deciding now, before the 96 report, whether that is enough to
carry the CL-theory framing or whether the paper is better positioned
as geometry with an unusually well-specified stream. Both are
defensible.

- **CL theory.** The Hamming 2×3 sits between domain-incremental and
  task-incremental, with both axes set independently. The second-learner
  item then says: two attempts to vary the learner hit the
  operating-point problem, which constrains how CL comparisons can be
  made; the Hamming-axis findings survive an optimiser change, if they
  do. The joint outcome is not "a second learner agrees."
- **Geometry with a well-specified stream.** The measurements are about
  this MLP, this stream, this estimator. The CL scenarios name the
  design; they do not yet license a theory claim about continual
  learning. The second-learner sequence returned two obstructions and
  no comparison.

The second is what the evidence currently supports. Do not resolve this
by more compute. Do not wait for the 96 to pick. The 96 can at most add
"the Hamming-axis findings are not artefacts of full-batch GD," which
is a geometry statement with a well-specified stream, not a second
learner agreeing.

The 96 reported `finding3_fails_to_recover` (`results/adam_hamming_slice.md`).
That addition did not arrive. Hamming-axis flags are not a reading.
The reason the second-learner attempts failed is what they failed *at*:
three conditions of a comparison that could not be set up, not a
disagreement about the finding. Order is next as a single-learner stream
axis (`docs/23`), not as a substitute second learner.
