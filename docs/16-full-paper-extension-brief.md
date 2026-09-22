# 16 — Full-Paper Extension: Experimental Brief

```
Target:   Full paper (ICLR or ICML — venue not yet fixed; do not optimise for
          either deadline). Multiple people working; correctness is the binding
          constraint, not schedule.
Standing: Every rule in AGENTS.md remains in force. Pre-committed readings before
          arms exist. Measured cost before any run over ~30 min. Off-design
          results never enter results/phase1/.
Status:   Specification. Nothing here is authorised to run until §0 is closed
          and each axis has a written pre-commit.
Amended:  2026-09-01 — eight decisions in §Amendments; original wording below
          is unchanged so the disagreement stays visible. Adopted order is
          in the amendment, not in §8.
```

**Amendments (2026-09-01).** The eight gaps named in the reading of this brief
are closed in §Amendments at the end of this file. Do not treat the original
§0–§8 wording as current for those eight items.

---

## 0. Two prerequisites, before any new axis

### 0.1 The learning-rate law must be settled first

Every existing result uses `η = lr0 · γ₀² · (N/N_base)`. `00` §4.3 already records
that Atanasov et al. give `η* ∝ γ^(2/L)` for `γ ≫ 1`, and that the quadratic law
is therefore misspecified in the rich regime. `lr_scaling="corrected"` is
implemented and has never been run.

This blocks the depth axis specifically: **`L` appears in the exponent, so adding
a hidden layer changes the LR law.** Depth and LR-law would be confounded from
the first arm.

**Run first:** the corrected-LR arm at `L = 2`, over the full registered γ grid,
all four conditions, at the design point. This produces the like-for-like
reference that every subsequent axis is compared against.

**Pre-commit before running.** Three outcomes, all informative:
- Channel composition unchanged under corrected LR → the quadratic law was
  harmless here, existing results stand, and this becomes an appendix robustness
  arm.
- Composition changes at `γ ≥ 1` only → consistent with Atanasov's stated
  regime; the paper reports both laws and states which is primary.
- Composition changes throughout → the γ sweep is partly an LR sweep. That is a
  finding about the parameterisation and it must be reported as one.

### 0.2 A held-out arrangement set, reserved now

`lr0 = 5.0`, `target_loss = 0.05`, the γ grid and the measurement budget were all
chosen against arrangements that then produced the reported results. That is
ordinary practice and it is also a weakness nobody has to accept.

**Reserve a set of arrangement seeds that are touched by no design decision, no
pilot, no hyperparameter choice, and no exploratory analysis.** Every headline
result in the full paper is additionally reported on that set. Record the
reserved seeds in a committed file before any new arm runs, so the reservation is
verifiable rather than asserted.

This is cheap and very few papers do it. It is the strongest available answer to
"were these numbers tuned into existence."

---

## 1. Direct dimensionality control — highest priority

**Why.** Reviewer mobw on the prior paper: the causal claim is more accurately
read as "the regime, which co-varies with dimensionality, gates the effect." That
objection now lands harder, because one of the three attribution channels *is*
dimension, and γ moves dimensionality and optimisation dynamics together. This is
the single item between the current work and that reviewer's stated bar for a
main-conference paper.

**Design.** Low-rank linear projection between hidden layer and readout,
rank `r ∈ {2, 4, 8, 16, N}`, crossed with the γ grid. Full 2×2 in the sense that
matters: rank at fixed γ, and γ at fixed rank.

**Implementation constraints.**
- The projection must not change the parameterisation. Verify that hidden-weight
  initialisation remains bitwise identical across γ *and* across rank, as `I6`
  requires. Add a unit test.
- Rank is applied to the *readout path*, so the measured representation `h` is
  unchanged in dimension — but its task-relevant dimension is constrained. State
  explicitly in the spec which of the two is intended; they are different
  experiments.
- `lr0` and `target_loss` were pinned at `r = N`. Confirm every rank reaches the
  loss target; if not, `lr0` needs re-pinning per rank and that must be recorded
  as a design change, not absorbed silently.

**Pre-commit.**
- Channel reorganisation reproduces under rank control at fixed γ → dimensionality
  is doing the work, and the attribution is licensed as a claim about
  representational dimension.
- Reorganisation tracks γ regardless of rank → the driver is optimisation
  dynamics; the paper says so and the dimension channel is described as
  co-varying rather than causal.
- Mixed → report the interaction; do not pick a reading afterwards.

---

## 2. Depth

**Why.** The dominant generalizability doubt, and a specific mechanistic
hypothesis. Sixteen random Gaussian centres in `d = 150` under one ReLU are near
separable at initialisation (α ≈ 0.31 before training). The network is
re-arranging an adequate geometry rather than building one — which is plausibly
*why* the utility channel dominates. With depth, features must be constructed,
and the dimension channel should carry more.

**Design.** One additional hidden layer, `L = 3`. Not a sweep. One extra depth is
sufficient to test whether the result is a property of a single ReLU.

**Implementation constraints.**
- The μP parameterisation must be extended to `L = 3` and the base-width
  equivalence re-derived. `docs/reference/parameterization-derivation.md` is
  human-owned; the extension needs the same treatment.
- The LR law exponent changes with `L`. Use the outcome of §0.1.
- Report **capacity at initialisation** at both depths. If it is already lower at
  `L = 3`, that is the scarcity hypothesis being confirmed at init, before any
  training, and it is worth its own line.
- Width must be chosen so that parameter count is comparable, or the comparison
  is depth-plus-capacity. State which.

**Pre-commit.**
- Channel composition invariant to depth → the result is not an artefact of a
  single nonlinearity, and the claim strengthens.
- Dimension channel carries more at `L = 3` → the scarcity hypothesis is
  confirmed, and the paper states that channel composition depends on whether the
  representation must build features or only re-arrange them. This is the more
  interesting outcome and it must not be reported as a failure of the original
  result.
- Composition changes unpredictably → report; do not interpret post hoc.

---

## 3. Scope as a manipulated axis

**Why.** The Line 2 audit finds that scope — the breadth of experience grouped
under one task label — is specified by essentially no benchmark, and its Figure 1
("one stream, several partitions") is a thought experiment because on real
benchmarks the partition *is* the dataset. In this design it need not be.

**Primary design — granularity of the cut.** Fix a sequence of dichotomies, then
partition it three ways: one dichotomy per task (fine), two jointly (medium), four
jointly (coarse). Same underlying experience, three observer-imposed cuts.

Requires a multi-output readout for the coarse conditions. Keep the shared
readout; add outputs, do not add task conditioning.

**Fallback design — breadth per task.** A task is a dichotomy over a subset
`S ⊆ {1..16}`, `|S| ∈ {4, 8, 16}`. Keeps the scalar readout entirely. Use this if
the multi-output version introduces confounds that cannot be controlled (the loss
scale changes with output count; check this before choosing).

**Constraint from Line 2 itself.** That work argues reporting is insufficient — a
property varied without being shown to matter has not been probed. So the scope
axis must be accompanied by a demonstration that scope changes something for this
learner, not merely that it was varied.

**Pre-commit.**
- Forgetting differs by partition at matched underlying stream → the audit's
  central claim is demonstrated rather than argued, and it is a headline result.
- No difference → scope is not consequential for this learner in this regime,
  which is a substantive negative and must be reported as one. It would also
  bound the audit's own claim, which is worth saying plainly.

---

## 4. Recurrence in the stream

**Why.** The sharpest available test of the paper's central interpretive claim.
If retention is a surplus that relaxes toward a baseline, re-encountering an
earlier dichotomy should be **faster to re-learn than to learn first** — a savings
effect. If forgetting is overwriting, re-learning costs what first learning cost.

**Design.** Re-present task 0's dichotomy at a later position in the stream, after
several intervening tasks. Measure steps-to-target-loss on the re-encounter
against steps-to-target on the original, and the capacity trajectory across the
re-encounter.

**Implementation constraints.**
- Matched-loss stopping makes steps-to-target the natural savings measure. Record
  it explicitly; it is currently logged but not analysed.
- Control for position in the stream: a task at position 12 may be faster for
  reasons unrelated to having been seen before. **The control is a novel
  dichotomy at the same position**, matched for similarity to its predecessor.
- Line 2 lists recurrence as an exposure property, and notes it belongs to
  exposure when the schedule returns to a region and to change when the process
  repeats. This is the former. State it.

**Pre-commit.**
- Re-learning is faster than first learning, with the position control → savings;
  the surplus reading is supported by a behavioural measure independent of the
  geometry.
- Equal → the surplus reading is not supported behaviourally, and the paper says
  the geometric and behavioural pictures diverge. That is reportable and it is
  the outcome that would most change the framing.

---

## 5. Continual-learning baselines — appendix, but treated properly

**Why, revised.** Not a benchmark comparison — the contribution is a measurement,
not a method, so "does modularity beat EWC" is a question this paper does not
ask. But EWC and replay *change the geometry by construction*: EWC constrains
directions via a Fisher penalty, replay re-presents old input distributions. So
the decomposition can say **which channel each method protects.**

That is an application of the instrument rather than a defence against a
criticism, and it is a genuine result. It also pre-empts the objection at low
cost.

**Design.** One γ (rich, where forgetting is largest and the channels are most
distinguishable), all four similarity conditions, three arms: none / EWC /
replay. Report the attribution table for each.

**Implementation constraints.**
- EWC: diagonal Fisher, penalty strength swept over at least three values, since
  a single λ chosen badly is not a baseline.
- Replay: buffer size stated as a fraction of the task; the Line 2 replay case
  turned on objective and number of passes, so record both.
- Both interact with the invariants: EWC adds a penalty term to the loss, replay
  changes what "one task" means for the matched-loss stopping rule. Neither may
  silently alter the stopping criterion. State how each is handled.
- These are off-design. Separate results directory, not pooled, off-design guard
  applies.

**Pre-commit.** Which channel each method protects is the measurement; there is
no pass/fail. But state in advance that a null — both methods reduce forgetting
without differentiating the channels — is a reportable outcome and not a reason
to expand the sweep.

---

## 6. Naturalistic arm

**Why.** The one criticism the Line 2 framing reframes but does not remove.
Lowest priority and the most likely to consume disproportionate effort.

**Design.** Per `docs/10-cifar-pilot-spec.md`. `P = 10` changes the estimation
regime, so the first job is establishing the estimator behaves there at all —
noise floors, whether the `M^-0.45` compression law applies, pairwise vs `full_P`.
One measured arm before any projection.

**Scope honestly.** Raw pixels into the same MLP is a *lower bound* on
generalisation: if the decomposition works there it likely works on learned
features; if it fails, learned features are not ruled out. Say this in the spec
and in the paper.

**Constraint.** Feature similarity cannot be set independently of class identity
on real images, so the 2×2 does not transfer. This arm tests the readout axis
only, and must be scoped that way — building four corners a different way and
comparing them under the same labels would be the worst instance of the
mismatched-population fault in the appendix.

---

## 7. Two additions not requested

### 7.1 A formal interaction test

Reviewer mobw asked for one on the prior paper's headline and it was not
provided. The headline here is γ × condition on channel composition. With the
sampling-unit machinery already built, this is straightforward, and it closes the
last outstanding statistical criticism.

Specify the model and the unit before running it: the unit is the unique
(arrangement, init) draw, not the file.

### 7.2 Theory, at the level the reviews asked for

Reviewer mobw #7: *"even a toy linear-network analysis deriving the regime in
which task subspaces must compete for limited directions would strengthen the
contribution considerably."*

Two tractable targets, neither requiring new experiments:

- **Why the utility channel dominates when representational capacity exceeds task
  demand.** The scarcity hypothesis in §2 is currently a verbal argument. If it
  can be stated for a linear or single-ReLU model, it becomes a prediction that
  the depth arm tests rather than an interpretation of the depth arm's result.
- **Why the two similarity axes act separately on centre geometry.** Hiratani
  derives their interaction behaviourally in a linear teacher-student model.
  Whether their approximate additivity on `ρ_c` follows from that setup, or is
  independent of it, is a question with a derivable answer.

This is not an experimental item and it does not block anything. It is the
difference between a descriptive paper and one the reviews would call
main-conference.

---

## 8. Order

1. §0.1 corrected-LR reference, and §0.2 reserved arrangements — both before anything else
2. §1 rank control
3. §2 depth
4. §3 scope, §4 recurrence (independent; can proceed in parallel)
5. §5 CL baselines, §7.1 interaction test
6. §6 naturalistic arm
7. §7.2 theory throughout, not at the end

Each axis: written pre-commit before arms exist, measured cost before launch,
separate results directory, and a report before the next axis starts.

## 9. What does not change

The sampling-unit discipline (unique n, arrangement vs initialisation, population
table). The resolution gates and the floors. The artifact-family appendix. The
inventory and prose-pinning. The scope audit. Nothing in `results/phase1/` is
regenerated or added to.

Every new axis inherits the same question the last two weeks established as
primary: **at what sampling unit is this finding true?**

---

## Amendments (2026-09-01)

Authoritative decisions closing the eight gaps in the reading of this brief.
The sections above are the original text. Where an original sentence would now
be false, the replacement is here; the original is not silently edited.

### A1. The γ cut is γ > 1, not γ ≥ 1

Original §0.1: "Composition changes at `γ ≥ 1` only".

**Amended.** The code is authoritative: `lr_scaling="corrected"` fires only when
`gamma_0 > 1` (`src/models/parameterization.py`). At γ = 1 the two laws are
identical. **γ = 1 is a negative control**: any composition change there is not
the LR law and must be diagnosed, not absorbed.

### A2. Unique-n throughout, no copy inflation

Original §0.1: "over the full registered γ grid, all four conditions".

**Amended.** That meant coverage, not a second Phase 1 of 960 files.
§0.1 is **6 γ × 4 conditions × 8 unique arrangements = 192 arms**, with
`stream_id` drawing the arrangement under the fixed RNG for every *new* axis.
The same unique-n=8 rule applies to every new axis. No stream_id copies.

### A3. Old vs reserved arrangements, two named populations

Original §0.2 reserved "arrangement seeds" without naming the comparator
population. Running §0.1 on the reserved set would make the LR comparison a
mismatched-population comparison.

**Amended.** Two populations, recorded in `results/reserved_arrangements.json`:

- **Old arrangements.** The unique n=8 of the registered grid. On that grid,
  arrangement was keyed by `paired_init(seed)["stream"]`, not by `stream_id`.
  Identifiers are init seeds 0–7. **This is the LR comparator**: corrected-LR
  versus quadratic, to choose the primary law. Reproducing these arrangements
  requires `arrangement_source="legacy_seed"`, not `stream_rng`.
- **Reserved arrangements.** Eight `stream_id`s under `pipeline.stream_rng`
  (salt 20260817), in a block that has never been used as a stream id.
  **Every new axis** uses this set, plus a duplicate of the headline
  measurement under the primary law as the not-tuned report.

**Rule, in the file itself:** the reserved set never informs `lr0`, the rank
grid, or the stopping criterion. Tests assert that provenance directly
(registered pin, A5 grid, stopping rule) and that artifacts which *chose*
those pins do not contain reserved ids. New-axis pre-commits may name
reserved ids as experimental subjects; that is the intended use.

### A4. Missed loss under corrected LR is the manipulation, not a failure

At γ = 10, L = 2, corrected gives η ∝ γ¹ against γ² — a 100× smaller
rich-regime learning rate. Many rich arms will need far more steps and some
may not reach 0.05 within 20k.

**Pre-commit:** `lr0` is not re-pinned after seeing a miss. Record misses as
an outcome; if a large fraction miss, that is a finding about the corrected
law's practicality at this loss target, reported as such. If re-pinning
becomes necessary it is a design change with its own entry, decided before
the arms and not after. `lr0` and `target_loss` stay at the registered
5.0 / 0.05.

### A5. Rank is on the readout path; first slice; same miss rule

Original §1 listed `r ∈ {2, 4, 8, 16, N}` and allowed silent re-pinning.

**Amended.** Rank is a low-rank projection on the **readout path**, so
`h ∈ ℝ^N` is unchanged and GLUE's ambient dimension is untouched.
Hidden-width reduction is a different experiment and is not this one.

First slice: **γ ∈ {0.03, 1, 10} × r ∈ {4, 16, N}**, one measured arm, then
expand on the result. Rank waits on the primary LR law from §0.1.

If a rank cannot reach target at the `lr0` pinned on r = N, that rank is
reported as not reaching target rather than re-pinned, unless re-pinning is
decided in advance for all ranks uniformly.

### A6. Depth sequential arms stay blocked; init-α moves to the front

Original §2 required the μP extension before any depth work, and left width
choice unstated.

**Amended.** μP at L = 3 is human-derived; `parameterization-derivation.md` is
human-owned and unwritten; `parameterization.md` already flags μP+1/√L as out
of scope and not human-verified. An agent-derived L = 3 parameterisation is
what §8.1 forbids acting on. **Sequential depth arms wait on that derivation.**

**Init-α is not blocked.** Capacity at initialisation at L = 2 versus L = 3,
no training, no stream. A forward pass does not require the μP derivation.
If α at init is already lower at L = 3, the scarcity hypothesis is confirmed
before any sequential arm exists. Do it as soon as a second hidden layer
exists in the model code. Training at L = 3 must refuse until the derivation
exists.

**Width: report both.** Matching N confounds depth with capacity. Matching
parameter count confounds depth with width. Init-α reports L=3 at N=150
(param-matched) and L=3 at N=300 (width-matched). Forward-pass α licenses
no claim about the learning regime.

### A7. Scope: decide the readout before the pre-commit

Original §3 named granularity primary and the loss-scale confound as a
check-before-choosing, but did not require the decision before the pre-commit.

**Amended.** Granularity remains primary. The readout decision comes **before**
the scope pre-commit. Check the loss-scale confound first — matched-loss
stopping is calibrated at one output, and the coarse conditions change output
count. Mean-reduced stopping leaves a graded residual (0.05 mean vs 0.062
worst-of-4). **License granularity with worst-of-K stopping** (halt when
every output is at target). Do not fall back to breadth `|S|`: that is
class-count-as-scope, which Line 2 records as unspecified.

### A8. Recurrence position control, specified

Original §4: "a novel dichotomy at the same position, matched for similarity
to its predecessor."

**Amended.** The control is a **novel dichotomy at the same stream index,
matched on `s_r` to its own predecessor**, in the same condition corner, on
the same arrangement. Comparison: re-presented task-0 dichotomy at index k,
versus a fresh dichotomy at index k with the same readout-similarity to the
task at k−1. Same arrangement seed for both, drawn from the reserved set.
That isolates "seen before" from "position in the stream" and from
"similarity to what precedes it."

### A9. Two reorderings, adopted in full

Original §8 queued the §7.1 interaction test behind CL baselines and left
§7.2 theory "throughout" without an owner or a start.

**Amended order, start now:**

1. Commit the reserved-arrangement file (zero compute).
2. Run the interaction test on existing unique-n=8 data. Model and unit
   specified before running. Unit is the unique (arrangement, init) draw, not
   the file. Report before starting §0.1.
3. Write the §0.1 pre-commit, then measure one arm, then launch. 192 arms,
   unique n=8, **old** arrangements, γ > 1 as the cut with γ = 1 as negative
   control. Report the miss rate alongside the composition result.
4. In parallel: second hidden layer in the model code and the init-α check
   (L = 2 vs L = 3, forward pass only, parameter-count matched).
5. In parallel: scope readout confound — does matched-loss stopping survive
   a multi-output readout, and how does loss scale with output count? This
   decides granularity versus breadth, and it must land before the scope
   pre-commit is written.

§7.1 runs on existing unique-n=8 data, needs no new arms, and is the one
statistical object the reviewer explicitly asked for.

§7.2 theory runs in parallel from step 1, owned by a person rather than
queued. It is not blocked by any experiment. One of its two targets — why
utility dominates when capacity exceeds demand — would turn the depth arm
from an interpretation into a test of a prediction.

**Not yet:** rank (waits on the primary LR law), depth *arms* (wait on the
human μP derivation), scope and recurrence arms (wait on their pre-commits),
CL baselines, CIFAR.

### A10. Source checks, 2026-09-01

Graldi Table 1 and Chou App. B.5 / D.1.1 verified against source.

- Table 1's four cells match `00` §4.2, including σ²=1. Graldi's `D` is our
  `d`. App. A.3 is in hand and **does not state the constant**; the Table 1
  caption's forward reference is dangling. Independent derivation from Table 1
  (both rows give `γ₀ = N^{-1/2}`; `N → N/N₀` makes `γ₀ = 1` the NTP point at
  `N₀ = 64`) confirms the agent's. **Verification 1 signed off.** At `N = 300`,
  NTP-equivalent `γ₀ ≈ 0.462` (between 0.3 and 1; computed, not interpreted).
- B.5 settings and the D.1.1 unit-norm sentence confirmed.
  `validation-settings.md`, `manifold-generator.md`, and
  `glue-core-validation.md` are signed.
- **Stated deviation:** D.1.1 samples labels uniformly from `{±1}`; we restrict
  to balanced dichotomies. Recorded in `protocol-deviations.md` and in `00` §6.
- D.1.1's isotropic-Gaussian variant (`M_i = {u₀ + R·v_k}`, no intrinsic
  dimension) is a clean control for the rank axis — dimension cannot carry an
  attribution in manifolds that have none by construction. Rank is licensed
  under quadratic after A11 / A15; the isotropic-Gaussian control is A16.
- D.1.1 anticipates binary cross-entropy (`{0,1}` labels). A non-MSE arm would
  not be departing from the framework. Not a reason to change now.

**Human-owned, still outstanding:** the μP-at-L=3 derivation (the gate on
the depth axis). A.3 does not license it. Verification 1 and 2 are signed.

### A11. §0.1 corrected-LR reading, 2026-09-01

192/192 arms usable, 0 loss-target misses. Report:
`results/corrected_lr.json`, against `results/corrected_lr_precommit.json`.

- **γ ≤ 1:** corrected ≡ quadratic. Paired Δ = 0 on the unique-n=8 old
  arrangements (negative control; the laws are identical and the run is
  deterministic).
- **γ = 3, 10:** utility share unchanged (three-corner, task 0, lag 12):
  0.414 vs 0.414 at γ=3; 0.449 vs 0.450 at γ=10. Share CIs include 0. The
  signed utility *term* is detectably smaller under the slower LR (paired
  p ≪ 0.05) but the move is 0.004 / 0.020 against terms of 0.35 / 0.58 —
  not a composition change. Magnitude at γ=10 is 69.7 → 67.1 floors.
- **Steps** scale with the LR ratio: ×2.9 at γ=3 (3× smaller η), ×8.8 at
  γ=10 (10× smaller η). A4's "100×" is a slip: at L=2 the ratio is γ, not γ².
- **Applied reading:** `composition_unchanged_at_gamma_gt_1`. The quadratic
  law was harmless here. Existing results stand. Corrected-LR is an appendix
  robustness arm. **Primary law for subsequent axes remains quadratic.**

Rank proceeds under quadratic, on the reserved set, against
`results/rank_precommit.json`.

### A12. Rank pre-commit locked, 2026-09-01

Three decisions before rank code (now in the JSON, not only the brief), as
amended by A14:

- **I6 is four checks:** W(0) identical across γ at fixed r; W(0) identical
  across r at fixed γ (rank does not consume the shape stream); Q identical
  across γ at fixed r; Q nested across r (`Q_r = Q_N[:r, :]`, RNG
  `np.random.default_rng([20260901, seed])`, independent of r). Q is frozen.
  At r = N, Q is that full-rank orthogonal map, not I_N.
- **γ = 0.03 is excluded, not a control.** A null there cannot fail. Do not
  later report that the effect vanishes at low γ. Composition claims licensed
  at γ ∈ {1, 10}.
- **Misses:** lr0 stays 5.0, pinned on r = N. A miss at r < N is not
  separable from an LR that is wrong at that rank. Report as not reaching
  target at that pin; not a rank-capacity claim.

The one-arm ran before this file was committed and is not governed by it.
**This commit governs the 288-arm first slice.**

### A13. Rank one-arm, 2026-09-01

γ=10, r=4, S-HL, first reserved arrangement, seed 0, quadratic, lr0=5.0.
Usable; 0/16 tasks missed; mean 24.4 steps (max 70); wall 495 s.
Report: `results/rank_one_arm.json`. Used the un-nested, r-dependent
projection (I_N at r = N). Not pooled with the slice. 495 s × 288 = 39.6 h
is a lower bound; budget 60–100 h serial.

### A14. Pre-commit commit and nested Q, 2026-09-02

`results/rank_precommit.json` is committed after the one-arm and before the
288-arm slice. Nested Q replaces independent draws at each r. Reservation
tests assert lr0 / grid / stopping provenance, not string presence of
reserved ids in new-axis pre-commits.

### A15. Rank first slice, 2026-09-02

288 arms, nested Q, reserved set, 192 workers, wall 4.61 h.
`results/rank_slice.json` against `results/rank_precommit.json`.

- **Misses:** 8/288, all at γ=0.03 r=4. Licensed cells γ ∈ {1, 10} × all r:
  0 misses. The eight misses are reported as not reaching target at the lr0
  pinned on r=N, not as a rank-capacity claim, and they sit in the excluded
  γ=0.03 band.
- **Three-corner utility share** (task 0, lag 12, module A, unique n=8):
  at γ=1, 0.339 / 0.344 / 0.343 for r=4 / 16 / N; at γ=10, 0.449 / 0.445 /
  0.448. Rank effect at γ=10 is 0.0007 against a γ=1→10 step of 0.106 at
  r=N. r=N recovers the reserved-set headline.
- **Applied reading:** `reorganisation_tracks_gamma_regardless_of_rank`.
  The driver is optimisation dynamics. The dimension channel is co-varying
  rather than causal. Do not read γ=0.03 as the effect vanishing at low γ.

### A16. Isotropic-Gaussian control, 2026-09-02

D.1.1 variant `M_i = {u₀ + R·v_k}` on the reserved set, quadratic, unconstrained
readout (`Q = I`). 2 γ × 4 conditions × 8 reserved ids = 64 arms. γ ∈ {1, 10}
only; γ=0.03 is excluded, not a control. Rank is not recrossed. `v_k` is not
unit-normalized. Pre-commit `results/isotropic_precommit.json` is written
before any arm. One-arm first: γ=10, S-HL, first reserved arrangement.
Readings locked: if the γ=1→10 utility-share step reproduces (~0.106 at
spherical r=N), data-dimension is not necessary; if it vanishes, input
dimension was required even though readout rank was not.

64/64 usable, 0 misses. `results/isotropic_slice.json` against
`results/isotropic_precommit.json`. Parent runner was interrupted after the
files were written; wall is the arm-file mtime span, 0.67 h (serial sum 25.1 h).

- **Three-corner utility share** (task 0, lag 12, module A, unique n=8):
  0.040 at γ=1, 0.190 at γ=10, step 0.149. Spherical reserved r=N step is
  0.106. Levels sit below the spherical shares (0.343 / 0.448); the locked
  object is the step, not the level.
- **Applied reading:** `gamma_reorganisation_reproduces`. Data-dimension is
  not necessary for the reorganisation. Rank reading stands: the driver is
  optimisation dynamics.

### A17. Not-tuned reserved duplicate, 2026-09-02

A3's headline duplicate under quadratic on the reserved set. 6 registered γ ×
4 corners × 8 reserved ids = 192 arms. Spherical manifolds, unconstrained
readout (`Q = I`), not nested rank-N and not isotropic. Pre-commit
`results/nottuned_precommit.json` before any arm. One-arm first: γ=10, S-HL,
first reserved arrangement. Locked objects: γ=1→10 three-corner utility-share
step versus the old-arrangement quadratic step (0.117), and the signed-utility
γ × condition interaction. γ=0.03 is on the grid because it is registered;
it is not a control.

192/192 usable, 0 misses. `results/nottuned_slice.json` against
`results/nottuned_precommit.json`. Wall is the arm-file mtime span, 1.61 h
(serial sum 156 h).

- **Three-corner utility share** (task 0, lag 12, module A, unique n=8):
  0.343 at γ=1, 0.448 at γ=10, step **0.106**. Old-arrangement quadratic:
  0.332 / 0.449, step 0.117. γ ∈ {0.3, 1, 3, 10} gated 8/8. γ=0.03 is
  ungated (0/8, 0.27 floors) and is not a composition claim. γ=0.1 is 5/8.
- **Signed-utility interaction:** F(15, 168) = 215.7, p ≈ 2.9×10⁻¹⁰¹.
  Permutation: no shuffle of 999 exceeded observed (p = 0.001 floor).
- **Applied reading:** `headline_reproduces_on_reserved`. The headline was
  not tuned into existence on the old set.

### A18. Scope granularity pre-commit, 2026-09-03

A7 licensed multi-output granularity with worst-of-K stopping. Pre-commit
`results/scope_precommit.json` before any arm. First slice: γ ∈ {1, 10} ×
K ∈ {1, 2, 4} × 4 corners × 8 reserved = 192 arms. `|S|` is refused.
K=1 is a recovery control against the not-tuned reserved share at γ=10.
One-arm first: γ=10, K=4, S-HL, first reserved arrangement. Primary object:
paired (K=4 − K=1) three-corner utility share at γ=10, unique n=8, lag 12
from the group's own baseline.

### A19. Recurrence pre-commit, 2026-09-03

Written after the scope reading `forgetting_differs_by_partition`, before any
recurrence arm. Pre-commit `results/recurrence_precommit.json`. A8 control:
at index k=12, re-present task 0 versus a novel dichotomy matched on `s_r` to
the predecessor, same arrangement, reserved set. First slice: γ ∈ {1, 10} ×
{represent, control} × {S-HL, S-LL} × 8 reserved = 64 arms. Sequential
matched-loss. S-HH and S-LH are refused: at P=16, consecutive s_r=0.9 rounds
to Hamming 0, so a novel A8 control does not exist. Primary object: paired
(represent − control) steps at k=12, S-HL and S-LL, unique n=8, γ=10.
steps(k) versus steps(0) is not the identification.

### A20. Recurrence first slice, 2026-09-04

64 arms, reserved set, quadratic, matched-loss, k=12, S-HL and S-LL.
`results/recurrence_slice.json` against `results/recurrence_precommit.json`.
64/64 usable, 0 misses, 0 prefix failures.

- **Primary:** paired Δ = steps(represent, k=12) − steps(control, k=12),
  unique n=8, γ=10. Mean +0.94, SEM 0.69, 95% CI [−0.42, +2.29], includes 0.
- **γ=1 (not primary):** mean +4.63, SEM 3.65, CI [−2.52, +11.77], includes 0.
- **Applied reading:** `no_difference`. The surplus reading is not supported
  behaviourally. The geometric and behavioural pictures diverge. This is a
  results clause, not a limitation.
- **MDE, baseline = steps(control, k), unique n=8.** Do not use steps(0).
  At γ=10 the control takes 12.75 steps. The 95% CI excludes savings larger
  than 0.42 steps (3.3% of that control). Two-sided 80% MDE is 1.94 steps
  (15% of control). CI half-width is 1.35 steps (11% of control). At γ=1
  the control takes 101 steps. The 95% CI excludes savings larger than 2.52
  steps (2.5% of control). Half-width 7.15 steps (7.1% of control). The
  γ=1 bound is not loose once it is taken against the identification
  baseline rather than against steps(0).
- **Control construction held.** 32/32 pairs. max |s_r(y_k, y_{k−1}) −
  s_r(y_0, y_{k−1})| = 0. Represent S_r(y_0, y_12) = 1. No control is y_0
  (control S_r(y_0, y_12) in [0, 0.25]).
- **Structural finding, not a scoping note.** S-HH and S-LH are refused
  because at P=16 registered consecutive s_r=0.9 rounds to Hamming 0. Every
  dichotomy in those streams is y_0 up to sign, so a novel A8 control does
  not exist. Recurrence is only testable in the low-readout corners.
  Exposure and similarity are not independently settable at this scope.

Do not expand the 64-arm slice. Do not pool S-HH or S-LH. Do not treat
steps(k) versus steps(0) as the identification. Do not read the retained-α
table as a finding. The surplus (retained above generic) is the original
grid result. This slice tests whether that surplus predicts savings. It
does not.

### A21. μP-at-L=3 diagnostic protocol, 2026-09-04

A6 still holds: sequential depth arms wait on a human-signed Verification 2.
This amendment licenses **only** the diagnostic that can falsify the unsigned
hypothesis, written in `results/mup_l3_hypothesis.json` before any Gate 2
number exists.

**Hypothesis (unsigned).** No `1/√L`. One `η` on `W`, `W2`, and `u`. Hidden-type
row `β_hid = N^{-1/2}`. `γ` and `N/64` do not move with `L`. σ²=1, I6, I5
unchanged. A11 does not carry. Not μP+1/√L, not TP5 per-layer LRs.

**Gate 0.** Analytic `W2` gradients exist so the tests can run.
`sgd_step(..., diagnostic=True)` is the diagnostic entrypoint. `train_xy`,
`run_stream`, and `run_arm` refuse `L=3` regardless. This is not lifting A6.

**Gate 1.** At `N=64`, `γ₀=1`, `L=3`: μP ≡ NTP on the forward pass and the
first gradient step, float64, tensors `W`, `W2`, `u`. Fail → stop.

**Gate 2.** Toy-batch SGD at two widths, no stream. Pre-committed pass/fail
in the hypothesis file. Mixed is fail. Do not retune the test after seeing
numbers.

**Gate 3.** Kati signs Verification 2. The agent must not sign
`parameterization-derivation.md`. Stream `NotImplementedError` stays until
that signature.

**Measured, 2026-09-04.** Hypothesis
`results/mup_l3_hypothesis.json`, diagnostics
`results/mup_l3_diagnostics.json`. Gate 1 passed (μP ≡ NTP at N=64, γ₀=1,
L=3, W / W2 / u). Gate 2 mixed fail: A (preactivation ratio 0.845 ∈ [1/3, 3])
and C (‖ΔW‖/‖W‖ and ‖ΔW2‖/‖W2‖ larger at γ=10 than at γ=1, both widths)
passed the pre-committed criteria; B failed (μP argmin distance 0, NTP
argmin distance 0 — both pinned at the grid edge lr0=1.0; loss still
decreasing in lr0 after K=16). Mixed is fail. Do not sign Verification 2.
Do not lift stream training. Do not retune the test.

### A22. L=2 Gate 2 control, 2026-09-04

The L=3 Gate 2 mixed fail does not say whether Table 1 is wrong at L=3 or
whether this `(K, lr0 grid, toy batch)` cannot see μP vs NTP on any network.
Control: clone Gate 2 at L=2, where Verification 1 is already signed.
Pre-commit `results/mup_l2_gate2_control_precommit.json`, written before the
control numbers. Same `d`, batch, seeds, K, `lr0` grid, widths, pass
intervals. Only change: `n_hidden_layers=1`, last preactivation is `z1`, no
`W2`, ordinary `sgd_step`.

Primary object: check B. Two readings, committed before numbers:

- `protocol_cannot_certify` — B fails the same way as L=3. Next search for an
  operating point is L=2 only.
- `protocol_is_strong_enough` — B passes at L=2. Combined with the L=3 fail,
  that is an L=3 problem. Stop.

Do not retune the L=3 contract. Do not lift A6. Do not sign Verification 2.

**Measured, 2026-09-04.** `results/mup_l2_gate2_control.json` against that
pre-commit. Applied reading: `protocol_cannot_certify`.

Gate 1 passed. Check B failed the same way as L=3: μP distance 0, NTP
distance 0, all four argmins at the grid edge `lr0=1.0`. Loss still
decreasing in `lr0` after K=16 (L=2 μP at N=64: 0.266 at `lr0=1.0`). A and
C passed the written inequalities; `rms(Δz1)` ~ 2×10⁻⁷ and `‖ΔW‖/‖W‖` ~
10⁻⁹ at γ=1, same order as L=3. Those are empirical floors, not a reading.

The L=3 mixed fail is not about depth. Next measurement is an operating
point where L=2 μP vs NTP actually separates. That search is L=2 only.

### A23. L=2 operating-point search, then one L=3 apply, 2026-09-07

A22 applied `protocol_cannot_certify`. Search for the smallest `(K, lr0-grid
tail)` at L=2 where Gate 2 check B passes, freeze it, apply once at L=3.
Pre-commit `results/mup_l2_operating_point_precommit.json`, written before
the search. G0 is the original grid. H = {3, 10, 30, 100, 300, 1000}.
K ∈ {16, 32, 64, 128, 256, 512}. Order: K outer, tail length inner; skip
(K=16, n=0). First pass freezes. If none pass, do not run L=3.

L=3 identification is check B at the frozen point. Not Verification 2.
Do not lift A6. Do not retune the original L=3 Gate 2 file.

**Measured, 2026-09-07.** Search
`results/mup_l2_operating_point.json`. Frozen L=3 contract
`results/mup_l3_gate2_frozen_precommit.json` written before L=3 numbers.
L=3 result `results/mup_l3_gate2_frozen.json`. Applied reading:
`l3_b_fails`.

L=2 freeze: first pass at K=16, n_tail=2, grid through `lr0=10`. n_tail=1
(through 3) still had all four argmins at the new edge. At the freeze, μP
distance 1 (N=64 at 10, N=300 at 3) and NTP distance 1 (same two points).
μP and NTP chose the same argmins; the inequalities passed because adjacent
counts as μP-stable and a one-step move counts as NTP-slide.

L=3 on that frozen grid: all four argmins at `lr0=10`. μP distance 0, NTP
distance 0. Check B fails. A and C at original lr0 are recorded, not a
licence.

Stop. Do not invent `1/√L` or per-layer LRs. Do not lift A6. Do not sign
Verification 2.

### A24. Loss-vs-width contrast on the frozen grid, 2026-09-07

A23's argmin test did not isolate parameterization. New identification:
loss after K=16 at named `lr0=10` on the frozen grid, L=2 and L=3, no
search. Contrast present: μP loss(N=300) ≤ loss(N=64) and NTP
loss(N=300) > loss(N=64). Pre-commit
`results/mup_loss_width_precommit.json`, written before these numbers.

Three readings: `instrument_blind` (L=2 absent), `table_supported` (both
present), `l3_problem` (L=2 present, L=3 absent). Not Verification 2.
Do not lift A6.

**Measured, 2026-09-07.** `results/mup_loss_width.json` against that
pre-commit. Applied reading: `table_supported`.

At named `lr0=10`, K=16, N=64 loss is identical across parameterization
(L=2: 0.114; L=3: 0.031). Width ratios loss(N=300)/loss(N=64):

- L=2 μP 0.796 (does not increase), NTP 1.612 (increases)
- L=3 μP 0.466 (does not increase), NTP 1.898 (increases)

Empirical support for Table 1 at L=3 on this contrast. Not Verification 2.
Do not lift A6. Do not start sequential depth arms. Do not sign
`parameterization-derivation.md`.

### A25. ‖ΔW‖ floor, then loss-vs-width, 2026-09-07

A24 may be readout-only. Freeze the first K in
{16, 32, 64, 128, 256, 512, 1024, 2048, 4096} where L=2 μP N=64 module-A
`‖ΔW‖/‖W‖ ≥ 0.001` at named `lr0=10`. Then apply the A24 contrast at that K
on L=2 and L=3. Pre-commit `results/mup_dw_floor_precommit.json`, written
before these numbers. Floor search is L=2 only. Not Verification 2.
Do not lift A6.

**Measured, 2026-09-07.** `results/mup_dw_floor.json`. Frozen K written to
`results/mup_dw_floor_frozen.json` before L=3. Applied reading:
`table_supported_at_floor`.

First K in the sequence already clears the floor: K=16, L=2 μP N=64
`‖ΔW‖/‖W‖ = 0.56` (rms Δz = 0.80). The ~10⁻⁹ figures were A/C at
`lr0=0.001`, not A24's named `lr0=10`. A24 was already in a
feature-moving regime.

Loss ratios at that K are the A24 numbers (same protocol): L=2 μP 0.796 /
NTP 1.612; L=3 μP 0.466 / NTP 1.898. Contrast present at both depths.

Recorded, not a reading: at L=2, `‖ΔW‖/‖W‖` is ~0.50–0.56 for μP and NTP
at both widths — the loss contrast is not “μP moves W more.” At L=3, μP
N=64 `‖ΔW‖/‖W‖ = 0.044` (also above the floor); NTP N=300 is 0.018 on W
and 0.011 on W2.

Not Verification 2. Do not lift A6. Do not start sequential depth arms.

### A26. Unique n=8 on the frozen contrast, 2026-09-07

A24/A25 used seed=0. Replicate the named-`lr0=10`, K=16 loss-vs-width
contrast on unique n=8, L=2 and L=3. `data_seed = init_seed + 1` so seed 0
keeps the A24 batch. Seed 0 must match A24. Pre-commit
`results/mup_loss_width_seeds_precommit.json`, written before these
numbers. Readings: `all_supported` (8/8 both depths), `l3_problem` (8/8
L=2, not L=3), `instrument_blind` (not 8/8 at L=2). Not Verification 2.
Do not lift A6.

**Measured, 2026-09-07.** `results/mup_loss_width_seeds.json`. Applied
reading: `instrument_blind`. Seed 0 matched A24. Contrast present on **6/8**
seeds at L=2 and 4/8 at L=3. L=3 is not a reading.

L=2 misses: seed 1, μP ratio 1.106 (loss increased with width); seed 6,
NTP ratio 0.902 (NTP did not increase). The named-`lr0=10` contrast does
not replicate at unique n=8 on this toy protocol.

Not Verification 2. Do not lift A6. Do not start sequential depth arms.

### A27. n=32 rate, 2026-09-07

A26 `instrument_blind` stands. Expand to seeds 0–31 for a rate and
failure-mode counts on the same named-`lr0=10`, K=16 contrast. Not a
32/32 licence. Seeds 0–7 must match A26. Pre-commit
`results/mup_loss_width_n32_precommit.json`, written before these numbers.
Do not lift A6.

**Measured, 2026-09-07.** `results/mup_loss_width_n32.json`. Applied reading:
`recorded_rate`. Seeds 0–7 matched A26. Status:
`results/mup_l3_validation_status.json`.

- L=2: **16/32** (0.50), 95% CI [0.32, 0.68]. Failures: 10 μP increased,
  6 NTP did not increase.
- L=3: 21/32 (0.66), 95% CI [0.47, 0.81]. Recorded. Not a licence.
  Failures: 5 μP increased, 5 NTP did not, 1 both.

A26's 6/8 sat in the upper tail of this rate. Does not retune A26.
Not Verification 2. Do not lift A6.

### A28. Manifold instrument, n=32, 2026-09-07

The Gaussian `lr0=10`, K=16 contrast is a coin flip at L=2. Different
instrument: one spherical arrangement, one balanced dichotomy,
`flatten_task`, registered `lr0=5`, K=128, P=16, M=40, d=60, seeds 0–31.
Same loss-vs-width identification. Pass: k ≥ 24 at a depth. Pre-commit
`results/mup_manifold_n32_precommit.json`, written before these numbers.
Not a stream. Does not retune A26/A27. Not Verification 2. Do not lift A6.

**Measured, 2026-09-07.** `results/mup_manifold_n32.json`. Applied reading:
`l3_problem`.

- L=2: **25/32** ≥ 24 (0.78), 95% CI [0.60, 0.91]. All 7 misses are
  μP loss rising with width. NTP rose with width on all 32 seeds.
- L=3: **19/32** < 24 (0.59), 95% CI [0.41, 0.76]. All 13 misses are
  μP loss rising with width. NTP rose with width on all 32 seeds.
- Neither depth reached loss ≤ 0.05 at both widths after K=128 (0/32).
  Recorded, not a reading.

The CIs overlap. The committed split is the k≥24 rule, not a CI test.
Stop. Do not invent `1/√L`. Do not lift A6. Not Verification 2.

### A29. μP width-failure trajectories, 2026-09-07

A28 misses are all μP loss rising with width. Same manifold task, μP only,
snapshot K ∈ {16, 32, 64, 128, 256, 512}, n=32. Ask whether L=3 K=128
fails are worse at every earlier snapshot or overshoot, and whether K=512
rescues. Pre-commit `results/mup_l3_mup_traj_precommit.json`, written
before these numbers. K=128 μP losses must match A28. Does not retune A28.
Not a stream. Do not lift A6.

**Measured, 2026-09-07.** `results/mup_l3_mup_traj.json`. Prefix matched
A28. Cohort n=13 (same seeds). Applied reading: `always_worse_majority`.
K=512 recorded: `rescues_majority`.

- Shape: **13/13** always_worse through K=128. Not overshoot at the A28 halt.
- K=512: **13/13** rescued (ratio ≤ 1). L=2 K=128 fails were also 7/7
  always_worse.
- A28's k≥24 at K=128 is not retuned. L=3 μP was behind at every snapshot
  up to 128 and ahead by 512 on every miss.

Do not invent `1/√L`. Do not lift A6. Not Verification 2.

### A30. Manifold instrument at K=512, n=32, 2026-09-07

A29 rescued all 13 L=3 μP-increase seeds at K=512 on μP ratio alone. Full
contrast (μP and NTP), same k≥24 rule as A28, same manifolds, K=512.
Pre-commit `results/mup_manifold_k512_precommit.json`, written before these
numbers. μP ratios at 512 must match A29. Does not retune A28 at K=128.
Not a stream. Do not lift A6.

**Measured, 2026-09-07.** `results/mup_manifold_k512.json`. Prefix matched
A29. Applied reading: `table_supported`.

- L=2: **32/32** ≥ 24 (1.00), 95% CI [0.89, 1.00]. NTP rose with width on
  all 32 seeds.
- L=3: **32/32** ≥ 24 (1.00), 95% CI [0.89, 1.00]. NTP rose with width on
  all 32 seeds.
- A28's 13 L=3 μP-increase seeds: **13/13** present at K=512 (recorded, not
  a reading).
- Loss ≤ 0.05 at both μP widths: **32/32** at both depths (recorded, not a
  reading). At A28's K=128 this was 0/32.

A28's k≥24 at K=128 still stands as `l3_problem`. Do not invent `1/√L`.
Do not lift A6. Not Verification 2.

### A31. Third width N=150 interpolation at K=512, 2026-09-07

A30 is 64 vs 300. Same instrument, add N=150. Per seed: μP loss non-increasing
along 64 → 150 → 300, NTP increasing along that chain. Same k≥24 rule.
Pre-commit `results/mup_manifold_n150_precommit.json`, written before these
numbers. Endpoint ratios must match A30. Does not retune A30. Not a stream.
Do not lift A6.

**Measured, 2026-09-07.** `results/mup_manifold_n150.json`. Prefix matched
A30. Applied reading: `table_supported`.

- L=2: **30/32** ≥ 24 (0.94), 95% CI [0.79, 0.99]. Both misses are μP not
  monotone at N=150. NTP increasing on all 32. Endpoints 32/32 (recorded).
- L=3: **31/32** ≥ 24 (0.97), 95% CI [0.84, 1.00]. One miss is μP not
  monotone at N=150. NTP increasing on all 32. Endpoints 32/32 (recorded).

Does not retune A30. Do not invent `1/√L`. Do not lift A6. Not Verification 2.

### A32. Same contrast at γ₀=10, K=512, n=32, 2026-09-07

Table 1's other distinctive row is the γ scaling. Same A30 manifolds, lr0=5,
K=512, widths 64 and 300, γ₀=10. Same k≥24 rule. Pre-commit
`results/mup_manifold_gamma10_precommit.json`, written before these numbers.
NTP ignores γ₀, so NTP ratios must match A30. Do not retune lr0. Does not
retune A30 at γ₀=1. Not a stream. Do not lift A6.

**Measured, 2026-09-07.** `results/mup_manifold_gamma10.json`. NTP prefix
matched A30. Applied reading: `table_supported`.

- L=2: **32/32** ≥ 24 (1.00), 95% CI [0.89, 1.00].
- L=3: **32/32** ≥ 24 (1.00), 95% CI [0.89, 1.00].

Does not retune A30 at γ₀=1. Do not invent `1/√L`. Do not lift A6. Not
Verification 2. The agent does not sign `parameterization-derivation.md`.

### A33. μP 1/√L stress-test at the A28 halt, 2026-09-07

A28 is halt-dependent. Stress-test whether a missing `1/√L` factor, not
slower dynamics, explains L=3 k<24 at K=128. Same A28 manifolds, lr0=5,
K=128. μP η only: Table 1 times `√(2/L)` (`L_ref=2` so L=2 is the A28
control; absolute `1/√L` would move L=2). NTP unchanged. Diagnostic script
only — not in `ScalingConfig`. Pre-commit
`results/mup_manifold_sqrtL_precommit.json`, written before these numbers.
L=2 μP losses and NTP ratios must match A28. Does not retune A28. Does not
adopt `1/√L`. Not Verification 2. Do not lift A6.

**Measured, 2026-09-07.** `results/mup_manifold_sqrtL.json`. Prefix matched
A28. Applied reading: `does_not_rescue`.

- L=2: **25/32** (identical to A28; control).
- L=3: **17/32** < 24 (0.53), 95% CI [0.35, 0.71]. A28 was 19/32. All 15
  misses are μP loss rising with width. NTP rose with width on all 32.
- A28's 13 L=3 μP-increase seeds: **0/13** present (recorded).

Smaller η at L=3 made K=128 slightly worse, not better. Supports
slower-not-wrong-law. Do not adopt `1/√L`. Do not retune A28. Do not sign
Verification 2. Do not lift A6.

