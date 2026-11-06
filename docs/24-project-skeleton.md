# 24 — Project skeleton: Representational Geometry of Forgetting and Transfer

```
Status:  proposal. Supersedes nothing until signed; then it is the top of
         the hierarchy and every arm must name its slot in §4–§6.
Purpose: one question, one measured object, three claims, a fixed list of
         ablations, three extension axes. Everything else is archive.
```

---

## 0. Why the project feels unordered

23 docs (~9,000 lines), ~120 result files, a 10 MB v3 notebook. Each arm was
justified locally, but validations, robustness checks and findings were filed
as peers. Of the original hypotheses, H2 (crossing) failed, H1d/H6 were
killed, H3–H5 were never run. What survived is a *different and cleaner*
story than `PROJECT.md` tells. This document states that story and nothing
else.

---

## 1. Question

> When a network learns a new task, how does the geometry that supported
> earlier tasks change — and which part of that change is set by the
> **stream**, which by the **learner**?

The second clause is the Line 2 factorisation, `outcome = f(stream; learner)`,
made measurable.

---

## 2. The one measured object

For a past task `j` after training through task `t`, with `α` the GLUE
capacity of the hidden representation for task `j`'s fixed dichotomy:

```
Δlog α_j(t) = Δlog Ψ_eff  +  Δlog(1 + R_eff⁻²)  −  Δlog D_eff        (exact)
              utility         radius                dimension
```

- **Sign is the outcome.** `< 0` forgetting, `> 0` backward transfer. Forgetting
  and transfer are one quantity, not two metrics.
- **Relation between tasks:** `Δρ_c` (centre correlation between task `j` and
  the current task).
- **Unit:** α floors (measurement-noise floor). Sampling unit: unique
  (arrangement, init) draw.

**Gap in the title.** "Transfer" currently rests on backward transfer in one
corner. Forward transfer — `α_{t+1}` measured on the representation *before*
task `t+1` is trained — is not in the core measurement. Either add it (cheap if
activations at boundaries are retained) or title the paper
"…Forgetting and Backward Transfer". Decide before figures.

---

## 3. Claims

| | claim | status | evidence |
|---|---|---|---|
| M | Forgetting/transfer decomposes **exactly** into three geometric channels; the estimator recovers ground truth | done | identity residual at machine precision; `glue_core_recovery.json`, Gate 2a |
| C1 | **Learner sets magnitude and channel.** Richness increases forgetting and shifts it from radius to alignment/utility | Tier A | `14` §4 A1–A2; rank arm: driver is optimisation dynamics, not dimension |
| C2 | **Stream sets sign.** Transfer appears only when the task repeats under slowly drifting input. Task change acts as a cliff, graded only under drift | Tier A (cliff, S-HH gain); learner-specific? see §6.1 | `14` A3; `hamming_slice.md` |
| C3 | **Centre geometry dissociates from capacity.** Input change drives centre convergence; task change drives graded divergence while capacity steps | Tier A (sign, ordering); magnitudes soft across arrangements | `14` A4–A5; `20` §1 |

C1 is an interaction, not a main effect (`interaction_gamma_condition.json`):
the richness gradient of channel composition differs by stream condition.
State it that way.

---

## 4. Core experiment (what every figure rests on)

| fixed | value |
|---|---|
| data | P=16 synthetic manifolds, d=150, factorial labelling |
| task | one balanced dichotomy, scalar ±1 readout |
| learner | 1 hidden layer ReLU, N=300, μP, full-batch GD, MSE, stop at matched loss |
| measurement | GLUE `full_P` at every boundary, past tasks' fixed dichotomies |

| varied | levels |
|---|---|
| richness γ₀ | 0.1, 1, 10 (grid has 0.03–30) |
| task change | Hamming 0 / 4 / 8 (repeat / partial / orthogonal) |
| input change | frozen / drift / jump |
| replication | n=8 unique arrangements |

Already run. No new compute needed for the core.

**Decided:** the existing results use two identical modules summed into one
readout (config C2), a leftover of the modularity origin. The core is rerun as
one module (C1), with forward transfer and measured richness added to the
same runs (`docs/25` §1).

---

## 5. Ablations — one alternative explanation per row (appendix table)

| alternative explanation | arm | result | status |
|---|---|---|---|
| it is the generator's structure | isotropic Gaussian manifolds | reorganisation reproduces | done |
| design was tuned on the arrangements reported | reserved arrangements | headline reproduces | done |
| γ sweep is secretly an LR sweep | corrected LR law | composition unchanged at γ>1 | done |
| it is dimensionality, not dynamics | rank bottleneck | tracks γ regardless of rank | done |
| it is capacity load | width N=150…600 | invariant within a bound | done (Tier B) |
| "similarity" is a dose | Hamming grading | cliff, not dose | done |
| task boundaries are given, not chosen | scope / partition | forgetting differs by partition | done |
| geometric surplus implies savings | recurrence | no behavioural difference | done (null) |
| it is task position, not composition | order | — | pre-committed (`23`) |
| it is exposure spacing | spacing | — | not started |

Anything not answering a named alternative is not an ablation and does not run.

---

## 6. Extensions for the full paper (ICML / CCN 2027)

Each axis asks one question: **do C1–C3 keep their sign?** Magnitudes are
reported per block, not compared across blocks.

### 6.1 Learners — fix the comparison rule first

The repo closed the second-learner line because MSE/BCE and GD/Adam have no
shared operating point (`07` §A.9). That obstruction comes from requiring γ to
mean the same thing across learners. Two changes remove it:

1. **Learner as a block.** Each learner gets its own operating point, set by the
   same behavioural rule (train to criterion on the current task). Only
   within-learner contrasts are compared across learners, and only by sign.
2. **Measured richness on the x-axis.** Replace the γ label with a measured
   quantity on task 0 (relative weight change ‖ΔW‖/‖W₀‖, or kernel change).
   The rank arm already says the driver is dynamics, so the axis should
   measure dynamics. Every learner then becomes points on one axis, and
   Adam's `partial_manipulation` stops being a blocker.

Under this rule the existing Adam result is a **reading, not an obstruction**:
the S-HH gain did not appear (drift −0.3 fl, CI includes 0). Either C2 is
GD-specific or Adam sits at a different measured richness. That is
answerable from data already on disk.

Then, in order: width (larger range), depth L=3 (blocked on the μP derivation),
Adam, CE.

### 6.2 CL methods — as instruments, not competitors

| method | constrains | prediction (pre-commit) |
|---|---|---|
| EWC | parameter directions | protects alignment/utility channel |
| replay | input distribution | protects dimension/radius; restores S-HH-like transfer |

Traps already known: stopping rule must be stated per method (EWC changes the
objective; replay changes what a task is); λ at ≥3 values.

### 6.3 Data — real manifolds, designed dichotomies

Keep the stream design; replace the synthetic generator.

- **CIFAR-100:** 16 classes as manifolds (P=16, not CIFAR-10's P=10), inputs as
  pixels or frozen pretrained features. Dichotomies drawn exactly as now, so
  the task-change axis transfers intact. Input change must be redefined
  (within-class subset or augmentation drift).
- **Neural data:** the same construction on recorded population responses to
  object categories as fixed input manifolds. Links C3 to the ρ_c direction
  measured in human MEG (Menghi et al.).

Venue decides which branch is primary: ICML → CIFAR; CCN → neural.

---

## 7. Minimal code — the proof of concept

Four scripts, each ≤ 60 lines, laptop-minutes, one figure each. Import only
the estimator from `src/glue`; write everything else inline so a reader sees
the whole mechanism.

| script | does | shows |
|---|---|---|
| `poc/01_stream.py` | arrangement, 4-task stream at chosen Hamming × input change | stream glyph + similarity matrix |
| `poc/02_train.py` | one-module net, γ ∈ {0.1, 10}, hidden states at boundaries | loss with stream strip above |
| `poc/03_decompose.py` | Δlog α of task 0 into three channels; identity residual | the identity holds; channel bars |
| `poc/04_contrast.py` | repeat vs change × drift vs jump at one γ, n=8 | C2 and C3 in one table |

`project-overview-v3.ipynb` moves to archive. The POC is Figure 1's source
and the entry point for collaborators.

---

## 8. Figures

| fig | claim | content |
|---|---|---|
| 1 | M | instrument: identity, stream glyph, retained vs generic |
| 2 | C1 | channel composition vs measured richness, stratified by condition |
| 3 | C2 | Hamming × input 2×3: cliff, drift gradation, the S-HH gain |
| 4 | C3 | Δρ_c vs Δα across conditions — the dissociation |
| 5 | §6 | sign table of C1–C3 across learners, methods, datasets |
| A | §5 | ablation table |

---

## 9. Rules that keep it small

1. A new arm names its slot (§4 core, §5 row, §6 axis) **before** its
   pre-commit. No slot, no run.
2. One canonical result file per slot. Everything else moves to
   `results/archive/`.
3. Active docs: this file, `00-math-spec.md`, the ablation table, `LOG.md`.
   Others archived, not deleted.
4. Validation and robustness are never reported as findings.

---

## 10. Decisions (Kati, 2026-11-03)

1. **Measured richness:** yes as the cross-learner axis; every claim is also
   reported against fixed γ₀ and the two compared (`docs/25` §3).
2. **Forward transfer:** added to the core measurement, minimally — one
   geometric and one behavioural measure, conditional on task structure
   (`docs/25` §4). Title kept.
3. **One module:** the core is rerun as C1; no architecture choice without a
   stated reason (`docs/25` §1–2).
4. **Venue:** deferred until the method is drafted; ICML and CCN deadlines are
   close enough that the choice can follow the method.
