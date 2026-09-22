# 23 — Order: matched composition, three schedules, schedule B

```
Status:  pre-commit on file. Second-learner line closed. Not yet run.
Signed:  audit's weaker prediction; matched composition; schedule B so
         position stays a covariate. Forward is the existing Hamming
         slice, not rerun.
Next:    one-arm, then the 64. After this arm: spacing, then P=32.
Not this arm: a fourth learner, Adam, CE, replay, EWC, architecture,
              width, jump, γ=1, L=3, figures, CIFAR-as-finding,
              results/phase1/, testing the Adam accumulation hypothesis.
```

---

## Why this arm, and why now

The second-learner line is closed (`docs/07` §A.9). Three attempts, three
obstructions, each a different condition of "same setting, different
learner." Order and spacing were always going to be single-learner
experiments. `docs/20` §9 said so before the Hamming-only numbers.
This arm characterises the instrument on a well-specified stream. It
is not a substitute second learner.

Line 2's exposure layer is the unset knob that remains (A8 ran;
recurrence is not order). The original prediction — "if task change is
a threshold, order should not matter" — was too strong: order moves
position, and the lag-4 series already shows position ranging 1.2–3.45×
and inverting under drift at s_r=1. The audit's replacement is the
testable version.

**Weaker prediction.** After matching the measured task's position,
consecutive-Hamming dose beyond the cliff should not further separate
schedules.

---

## Design (held)

- **Composition.** The bag of 16 `(arrangement, dichotomy)` pairs is
  the Hamming 2×3 stream at that `(stream_id, input_level, s_r)`.
  Schedules permute presentation order only. Pairs stay glued.
- **Three schedules.**
  - `forward` — generated order. **Already measured** in
    `results/hamming/`. Not rerun. Pair at
    `(stream_id, input_level, s_r, gamma_0)`.
  - `reverse` — presentation indices `15, 14, …, 0`. Consecutive
    Hamming is preserved on a constant-Hamming walk. This is the
    position manipulation.
  - `shuffle` — one permutation per `stream_id`, from
    `order_rng(stream_id)`, rejected if it is identity or reverse.
    Consecutive Hamming is realized, not designed. This is composition
    holding the bag, not the walk.
- **Learner.** Unchanged: L=2, N=300, ReLU, shared scalar readout,
  readout at 0, paired init, **full-batch GD**, MSE, `target_loss=0.05`,
  `lr0=5`, quadratic μP. I3 holds. Not Adam.
- **Schedule B.** T=16, `tracked_stride=2`. Position is a covariate:
  lag 4 exists for presentation indices `{0, 2, 4, 6, 8, 10}`.
- **Population.** Reserved ids 10000–10007, `stream_rng`, seed=0.
- **First slice.** γ=10 only. Input frozen and drift (cliff vs
  graded). Hamming `s_r ∈ {0.75, 0.25}` (beyond the cliff). Schedules
  reverse and shuffle (forward from the Hamming slice). Unique n=8.
  **64** new arms. Jump is out: it has no post-cliff dose. γ=1 is out:
  this is a single-learner stream axis, primary at γ=10.
- **Primary cell.** Unique n=8, module A, **lag 4, presentation index
  8**, Δ log α. Position is matched. Identity of the dichotomy at that
  index is *not* matched — that is composition. Lag 12 of original
  task 0 is not the primary: reverse puts that dichotomy last.
- **Reported, not a reading.** Lag-4 series across presentation
  index. Δρ_c. Forward identity against the Hamming slice (a stop if
  it fails, not a Hamming-axis reading).

---

## Readings, named before numbers (γ=10, unique n=8)

Schedule range at a cell = max − min of 1-decimal floors across
`{forward, reverse, shuffle}` at lag 4, presentation index 8.

- `weaker_prediction_holds` — on frozen and on drift,
  `range(s_r=0.25) − range(s_r=0.75)` does not clear two floors.
  Consecutive-Hamming dose beyond the cliff does not further separate
  schedules.
- `weaker_prediction_fails` — that difference clears two floors on
  frozen and on drift. Name the magnitudes. Order effects grow with
  Hamming past the cliff.
- `partial` — one input holds, the other fails. Report the split. No
  post-hoc story.
- `forward_identity_fails` — a forward cell used as the third
  schedule does not match the Hamming slice (same
  `(stream_id, input, s_r)` at γ=10). Stop. The pairing is broken.

Recovery of finding 3 is inherited from the Hamming slice (forward,
s_r=1, drift vs jump). Do not rerun it. Do not make it a reading of
this arm.

A monotonicity claim on schedule range still requires the difference
to clear two floors, not just the ordering to hold (A.8).

---

## Sequence (no slice until JSON, then one-arm)

1. Write this file and `results/order_precommit.json` (this step).
2. Implement order as a permutation of an existing Hamming stream.
   Default presentation remains `forward`. I3 stays green.
3. One-arm: γ=10, reverse, drift, `s_r=0.75`, stream 10000. Governs
   nothing except: did this presentation reach 0.05 at `lr0=5`.
4. 64-arm slice into `results/order/`. Never `results/phase1/`.
   Forward cells are read from `results/hamming/`, not retrained.
5. Apply one of the four readings.

Will not: a fourth learner; Adam; CE; replay; EWC; architecture or
width variant; jump on this slice; γ=1 on this slice; include s_r=1
as a dose cell; re-pin `lr0`; treat consecutive s_r as lag-12 s_r;
inherit stride 4; P=32; L=3; edit figures from this arm; write into
`results/phase1/`; test the Adam accumulation hypothesis; salvage a
stream claim from a partial split after seeing numbers.

---

## What this is allowed to mean

A result here is a property of this learner on these streams, with
position matched. It is not learner-generality. It is not a Line 2
claim that order has been probed for "the" continual learner. It is
the exposure axis, measured.
