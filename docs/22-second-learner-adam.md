# 22 — Second learner: Adam on the existing Hamming streams

```
Status:  Gate is a finding (partial_manipulation). Hamming-only 96 run.
         Reading: finding3_fails_to_recover. Stop.
Signed:  16-arm gate, unique n=8. 96-arm γ=10, usable 96, missed 0.
         Drift at s_r=1: −0.3 fl (mean −0.0055, CI includes 0, 3+/5−).
         Jump at s_r=1: −88.4 fl. Gain-under-drift did not hold.
         Hamming-axis flags are not a reading.
Next:    stream axes. Order (`docs/23`). Second-learner line closed.
         Hypothesis in the log, not a reading: matched-loss × Adam.
Not this arm: launching 192, another loss, per-γ pins, reopening CE,
              EWC/replay, L=3, figures, CIFAR-as-finding, results/phase1/,
              speaking to γ / finding 1 / channel reorganisation,
              treating the 96 as a second learner agreeing,
              testing the accumulation hypothesis.
```

---

## Why this is the second-learner arm that can still run

The CE arm (`docs/21`) asked whether the cliff, the drift gradation, and
the crossover are properties of the stream or of this learner. It could
not ask that, because MSE and BCE have no shared operating point. That
failure is written (`docs/15` limitation 2, `docs/07` §A.9). It is not a
reason to try another classification loss.

Any other classification loss (hinge, logistic with different scaling,
focal) has the same structure as BCE: gradient concentrated near the
boundary, vanishing away from it. They will produce the negative tail
and most will have the same between-γ mismatch. Do not spend pilots
rediscovering the obstruction.

Adam is a different variable. Same loss, same ±1 targets, same
`target_loss=0.05`, same stopping rule, different update. The
matched-loss pin transfers because the objective is unchanged, so the
obstruction that killed CE does not apply.

It is also the better test for the purpose. The second-learner arm
exists to ask whether the cliff, the drift gradation, and the crossover
are properties of the stream or of this learner. The most common
reviewer objection is not "you used MSE" — it is "you used plain
full-batch gradient descent and nobody does that." Adam answers the
objection that actually gets raised, and it answers it cleanly.

This is an explicit exception to I3 (`docs/AGENTS.md`): plain SGD only
in **core** runs, because Adam's per-parameter normalisation undoes the
μP learning-rate scaling. The exception is licensed here, for this
off-design directory, never `results/phase1/`. The richness
manipulation must be checked, not assumed.

---

## Design (held)

- **Streams.** The existing Hamming 2×3: reserved `stream_id`s
  10000–10007, `s_r ∈ {1.0, 0.75, 0.5, 0.25}`, input frozen / drift /
  jump, T=16, stride=2, P=16. Dichotomies and arrangements regenerated
  from `stream_rng` as now; they pair with the GD-MSE arms at fixed
  (`stream_id`, `s_r`, input level).
- **Architecture.** Unchanged: L=2, N=300, ReLU, shared scalar readout,
  readout at 0 (I5), paired init.
- **Loss.** MSE to ±1. `target_loss=0.05`. The registered pin. Not
  informed by this axis or by the reserved set.
- **Optimiser.** Adam, full batch. Library defaults: β₁=0.9, β₂=0.999,
  ε=10⁻⁸. No weight decay (I1). No gradient clipping (I7). Not a search.
- **Learning-rate law.** Quadratic μP, inherited. Adam's effective step
  is scale-free, so the law may be decorative — that is what the
  richness gate exists to detect. Do not "fix" the exponent inside this
  arm.
- **`lr0`.** Inherit 5, the GD pin. Almost certainly wrong for Adam.
  Re-pin rule below.
- **First slice.** γ ∈ {1, 10} × 3 input × 4 Hamming × 8 ids = 192
  unique arms, off-design directory `results/adam/`, never
  `results/phase1/`.
- **Primary cell.** Unique n=8, module A, task 0, lag 12, Δ log α.
  Cliff, post-cliff range, crossover, and Δρ_c reported as on the
  GD-MSE Hamming slice.

---

## I3 exception and the richness gate

Adam's per-parameter normalisation partially undoes the μP
learning-rate scaling. That was the reason for excluding it from the
core grid. This arm therefore does not get to assume that γ=1 and γ=10
are still two richness regimes.

**Gate, named before numbers.** Unique n=8, frozen × `s_r=0.5`, after
task 0, both γ. Training-only — the gate is a property of the update,
not of GLUE. Quantity: `‖ΔW‖_F / ‖W(0)‖_F` module A. Mean at γ=10
over mean at γ=1. Three bands, named before the 16-arm numbers:

- `clears_decade`: ratio ≥ 10. Launch the 192.
- `partial_manipulation`: 3 ≤ ratio < 10. Stop for a decision. Do not
  launch 192.
- `richness_manipulation_collapses`: ratio < 3. Fail clearly. A
  Hamming-only Adam arm at one γ is a later decision, not this
  launch.

The one-arm smoke check is ~3.2× (1.83 / 0.58). n=1, not the gate.
Under GD the same separation was 2.53 decades. Adam's first step is
`lr · g/(|g|+ε)`, so gradient magnitude divides out and the
normalisation removes most of what μP's γ was doing to the step. That
is why Adam was excluded from the core grid.

The 192-arm runner refuses to start unless this gate reads
`clears_decade`. That refusal is correct. `partial_manipulation` is
the right label, and it is a finding, not a failed check.

File it with the `lr0=5` explosion as the two halves of one statement:
μP and adaptive optimisers do not compose. Under Adam, γ buys speed
rather than feature movement, because `lr · g / (|g|+ε)` divides out
gradient magnitude and μP's γ² lives in `lr`. Every result in the
lazy/rich literature is obtained under SGD. This is a checkable
caution nobody states. One cell, one architecture — measured.
Written: `docs/07` §A.9, `docs/15` limitation 4.

---

## `lr0` — signed re-pin, applied

`lr0=5` was pinned for full-batch GD. Adam's first step is scale-free
(`Δ ≈ lr · g / (|g| + ε)`), so 5 exploded: at γ=10, loss **0.5 → 3×10¹⁰
in three steps** (`lr = 2343.75`). μP puts γ² into the learning rate;
Adam treats that number as a scale-free step size. The two do not
compose. That is a measured caution, not an assertion.

The Phase 0 procedure, applied once, uniformly, on the one-arm cell
(task 0, both γ), decade ladder:

- `2e-5` strands γ=1 at loss 0.36 (too-small analog of GD's 0.2 at 0.34)
- `0.0002` reaches both γ
- `0.002` (decade up) still reaches

**Pin is `0.0002`.** Recorded in `results/adam_precommit.json` and
`results/adam_lr0_repin.md` before any slice. Do not re-pin a second
time. Do not re-pin per cell. Do not inform `lr0` from the Hamming
axis or from the richness gate.

An unrepaired miss at `0.0002` is `lr0_miss`: stop. Do not run the slice.


---

## One-arm first

Same cell as the Hamming one-arm: γ=10, frozen, `s_r=0.5`, stream
10000, a=0, N=300, quadratic, `matched_loss`, `target_loss=0.05`,
`lr0=0.0002` (re-pinned), tracked_stride=2. Governs nothing except:
did Adam at the pinned `lr0` reach 0.05, and what is `‖ΔW‖/‖W‖` as a
smoke check.

Then the 16-arm richness gate (this file, `--richness-gate`). The 192
launches only if that gate reads `clears_decade`.

---

## Readings, named before numbers (γ=10, unique n=8)

The CE slice outcomes transfer, plus a fifth for the I3 risk.

- `stream_property`: cliff (h=0 vs h≥2 under frozen and jump), drift
  post-cliff gradation (0.75 vs 0.25 CIs non-overlapping), and
  frozen–drift crossover all reproduce (same signs and the same
  "resolved vs overlapping CI" pattern as GD-MSE).
- `learner_property`: at least one of the three fails. Report which.
  Do not salvage a stream claim from the survivors after seeing
  numbers.
- `finding3_fails_to_recover`: at `s_r=1`, drift vs jump does not keep
  gain vs loss on Δ log α. Stop. The Adam streams are not the same
  object.
- `mixed`: γ=10 and γ=1 disagree on `stream_property` vs
  `learner_property`.
- `richness_manipulation_collapses`: the richness gate fails (γ=10 /
  γ=1 of mean `‖ΔW‖/‖W‖` after task 0, frozen × `s_r=0.5`, unique n=8,
  does not clear one decade). The arm speaks only to the Hamming axis
  within each γ. Do not interpret γ-dependent claims. Do not pool γ.

Cliff definition as on the MSE slice: floors(`s_r=1.0`) −
floors(`s_r=0.75`); post-cliff range = |floors(0.75) − floors(0.25)|;
a monotonicity claim requires that range to clear two floors.

If `richness_manipulation_collapses` fires, `mixed` is not available as
a γ-contrast reading; within-γ Hamming descriptions remain.

---

## Sequence (no slice until JSON, then one-arm)

1. Write this file and `results/adam_precommit.json` (this step).
2. Implement Adam as an opt-in on the training loop. Default remains
   full-batch GD. I3 tests on core configs stay green.
3. One-arm at inherited `lr0=5`. If miss, re-pin once by Phase 0,
   record, one-arm again. If still miss: `lr0_miss`, stop.
4. 192-arm slice into `results/adam/`. Never `results/phase1/`.
5. Richness gate on unique n=8, frozen × `s_r=0.5`, after task 0.
6. Apply one of the five readings. Then order.

Will not: another classification loss; per-γ pins; reopen CE; EWC or
replay; inform `lr0` from this axis; include γ=0.03 as a control;
P=32 on this slice; pool Hamming 0 with 0.75; treat consecutive `s_r`
as lag-12 `s_r`; inherit stride 4; expand A8; edit figures from this
arm; expand before the one-arm is reported; write Adam results into
`results/phase1/`; launch the 192 unless `clears_decade`; interpret γ,
finding 1, or the channel reorganisation from the Hamming-only arm.

---

## Hamming-only at γ=10 (the arm that can still run)

The 192 is refused. The Hamming-axis claims were found at both γ, so a
single γ is enough to ask whether they reproduce under a different
optimiser. γ=10, for two reasons: every effect is largest there, so a
failure to reproduce is unambiguous; and Adam's `lr` at the pin is
0.094 rather than 0.00094 with 502 steps at γ=1.

**This arm cannot speak to γ.** Finding 1 and the channel reorganisation
are out of scope. Those need a working richness manipulation, and
Adam does not supply one. Do not interpret a γ-dependent claim from
these 96. Do not pool with the GD γ=1 cells. Do not treat a
reproduction as "the second learner agrees."

Pre-commit: `results/adam_hamming_precommit.json`, written before the
new one-arm and before the 96. `written_before_running` applies to
the 96-arm Hamming-only slice, not to the historical 192.

**One-arm first**, on a Hamming cell that is not frozen × s_r=0.5:
γ=10, **drift**, s_r=0.5, stream 10000, `lr0=0.0002`. Governs nothing
except: did this cell reach 0.05 at the pinned `lr0`. Then the 96.

**Readings, named before numbers**, narrower than the original five.
Unique n=8, module A, task 0, lag 12, Δ log α, γ=10 only.

- `hamming_reproduces` — cliff, drift gradation, and frozen–drift
  crossover all hold. The Hamming-axis findings are not artefacts of
  full-batch GD.
- `hamming_fails` — none of the three hold. Name all three. The
  findings are scoped to the optimiser.
- `partial` — a split: report which held and which did not. No
  post-hoc story.

Recovery carries over, as a stop, not as a fourth Hamming reading: at
s_r=1, drift versus jump must keep gain versus loss on Δ log α, or
the arm is not measuring the same object (`finding3_fails_to_recover`).

Operationalization, named before numbers:

- **Cliff holds** if frozen and jump each have floors(1.0) −
  floors(0.75) ≥ 2 (1-decimal) and the post-cliff range |0.75 − 0.25|
  does not clear two floors (range < 2, or 0.75 vs 0.25 CIs overlap).
- **Drift gradation holds** if under drift the 0.75 vs 0.25 CIs do not
  overlap and that range is ≥ 2 floors.
- **Crossover holds** if at s_r=0.75 drift is better than frozen
  (higher 1-decimal floors) and at s_r=0.25 frozen is better than
  drift.

The three readings are mutually exclusive given recovery: all three
hold / none hold / a split. Do not salvage a stream claim from the
survivors after seeing numbers.

**Applied: `finding3_fails_to_recover`.** Unique n=8, γ=10, s_r=1:
drift −0.3 fl (mean −0.0055, CI includes 0, 3+/5−); jump −88.4 fl.
Stop. Hamming-axis flags are not a reading. Table:
`results/adam_hamming_slice.md`.

---

## Hypothesis, not a reading — do not test

Finding 3 is accumulation: the same task repeated sixteen times under
a slowly drifting input builds retained capacity. Accumulation
requires the weights to keep moving in a consistent direction across
boundaries. The gate measured that Adam moves weights 3.19× between γ
where SGD moves them by a decade, and reaches target in 24 steps.

Twenty-four steps per task is not enough trajectory for accumulation
to be visible. Under Adam the network arrives at the loss target fast
and stops. The drift condition then has no accumulated movement to
show, while the jump — a discrete disruption rather than an
accumulated one — still registers, and registers larger because there
is less compensating movement.

If that is right, matched-loss stopping and Adam interact: "matched
progress" and "comparable trajectory" come apart when one optimiser
arrives in 24 steps and the other in 500. That is the third
operating-point condition (`docs/07` §A.9). Log only. Do not test.

