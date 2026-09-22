# Adam `lr0` re-pin — Phase 0 procedure, once, uniformly

Inherited `lr0=5` (the GD pin) missed on the Adam one-arm cell. Re-pinned
**once**, before any slice, by the Phase 0 procedure (`docs/01`, `docs/22`).
Not per cell. Not after seeing Hamming cells. Not informed by the richness
gate.

**New pin: `lr0 = 0.0002`.**

## Inherited miss

Adam's first bias-corrected step is `Δ ≈ lr · g / (|g| + ε)`, so the μP
quadratic `lr = lr0 · γ² · (N/N_base)` is used as a scale-free step. At
γ=10, `lr0=5` is `lr = 2343.75`. A 20-step probe on frozen × s_r=0.5 ×
stream 10000, task 0:

| γ | lr | step-1 loss | step-3 loss | ΔW/W at step 20 |
|---|---|---|---|---|
| 10 | 2343.75 | 648 | 3.1×10¹⁰ | 7785 |
| 1 | 23.44 | 3.96 | 9.5×10⁴ | 82 |

A full one-arm at `lr0=5` was started and killed after 11 min of 99% CPU
with no boundary print: `steps_per_task=20_000` of an exploding loss.
No file was written under `results/adam/`.

## Decade ladder (task 0, same cell, both γ, max 2000 steps)

Stated before looking: decade down from the inherited miss until a
too-small analog appears, then pin the first value that reaches 0.05
at both γ, and confirm a decade up still reaches (Phase 0: 0.2 stranded
γ≤1 at 0.34; 5.0 reaches; 50 stable).

| lr0 | γ | lr | steps | reached | final loss | ΔW/W |
|---|---|---|---|---|---|---|
| 0.2 | 10 | 93.75 | 2000 | no | 0.069 | 483 |
| 0.2 | 1 | 0.938 | 17 | yes | 0.045 | 3.22 |
| 0.02 | 10 | 9.375 | 48 | yes | 0.046 | 50.1 |
| 0.02 | 1 | 0.094 | 9 | yes | 0.043 | 0.60 |
| 0.002 | 10 | 0.938 | 7 | yes | 0.033 | 2.28 |
| 0.002 | 1 | 0.0094 | 59 | yes | 0.049 | 0.56 |
| **0.0002** | **10** | **0.094** | **24** | **yes** | **0.047** | **1.83** |
| **0.0002** | **1** | **0.00094** | **504** | **yes** | **0.050** | **0.58** |
| 2e-5 | 10 | 0.0094 | 168 | yes | 0.050 | 1.72 |
| 2e-5 | 1 | 9.4e-5 | 2000 | **no** | **0.356** | 0.27 |

`2e-5` at γ=1 is the too-small analog (stranded at 0.36, against GD's
0.2 stranded at 0.34). Going up: `0.0002` is the first that reaches
both; `0.002` (decade up) still reaches. That is the pin.

ΔW ratios at the pin are not a reason to choose a different `lr0`. The
richness gate is applied after the slice, on unique n=8, and may fail.

## Will not

Re-pin a second time. Re-pin per γ. Inform `lr0` from the Hamming axis
or from whether the richness gate would pass.
