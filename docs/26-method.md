# 26 — Method: decisions and draft

```
Status:  proposal. Part A resolves the four open points; Part B is the
         method section they produce. Signed Part A = pre-commit input for
         the core rerun (docs/25 §8 step 5). Spacing pre-commit moves to
         docs/27 (one-line change in docs/25 §6.2).
Principle: every manipulated variable has a measured counterpart, and
         claims are stated on the measured one. This is what makes the
         method general (any learner, any data) and specific (exact
         definitions) at once.
Amended: 2026-11-06, Kati. Amendments are marked inline and the original text
         stays visible (struck through where replaced). The docs/25 §6.2
         pointer to docs/27 is done.
```

---

# Part A — Recommendations on the four open points

## A1. Operating point and richness (learner)

**Stopping rule: relative loss.** Stop task `t` when the current-task loss
falls to `ε · L_t(0)`, where `L_t(0)` is the loss of a zero-output network on
task `t`. Default ~~`ε = 0.05`~~ **`ε = 0.1`**.

- ~~It **reduces exactly to the existing rule**: with readout at zero,
  `f(θ₀) = 0`, so MSE on ±1 targets gives `L(0) = 1` and the core's matched
  loss 0.05 is already `ε = 0.05`. No existing result changes meaning.~~
  **[Amended 2026-11-06, Kati]** The loss is `0.5·mean(err²)`
  (`src/models/network.py:241`), so `L(0) = 0.5`, not 1. **ε = 0.1**, which
  keeps the target at 0.05. ε = 0.05 would have stopped at 0.025.
- It is defined the same way for any loss: BCE/CE `L(0) = log 2`. For EWC,
  apply it to the task term only, never the penalty. For replay, to the
  current-task data only. This fixes the "stopping rule per method" trap by
  construction.
- It does **not** make learners match on margin, and it should not:
  learners are compared within themselves (blocks), not at a common point.
  That is what the CE arm could not get, and does not need.

**Standard-practice check, as an ablation row:** rerun the confirmatory cells
under a fixed step budget per task, the usual CL protocol. If C1–C3 keep
their sign, the stopping rule is not doing the work.

**Measured richness: per task, read at the operating point.**
- `r_t = ‖W_end(t) − W_start(t)‖_F / ‖W_start(t)‖_F` for every task. ~~Already
  logged (`TaskRecord.weight_change`).~~ **[Amended 2026-11-06]**
  `TaskRecord.weight_change` is cumulative from initialisation
  (`network.py:129, 343`), so it equals `r_t` only at t = 0. `r_t` needs new
  code: store `W` at each task start.
- **[Amended 2026-11-06, Kati] Primary richness measure: the analytic NTK** on a
  fixed probe set. Report the relative change of the full NTK, the relative
  change of the readout block `K_u = s²HHᵀ`, and `CKA(K, yyᵀ)`. Weight change
  from init and `r_t` are secondary. Full definitions are in `docs/25` §3.
- **Learner-level summary:** `r_0`. Task 0 starts from initialisation for
  every learner, so it is the only comparable point across learners.
- ~~**Secondary:** kernel change on a fixed probe set, `1 − CKA(K_start, K_end)`
  of the hidden Gram matrix. Functional rather than parametric. Pre-commit
  that the two agree in ordering.~~ **[Amended 2026-11-06, Kati]** Dropped.
  Hidden-Gram CKA measures the representation, which is the outcome, so it is
  not a richness axis.
- The trajectory `r_0, r_1, …` is reported, not summarised. It tests
  endogenous regime drift (a network becoming lazy after task 0, the
  Graldi "pretraining effect"), which belongs to the big picture and costs
  nothing.

## A2. Forward transfer (measurement)

Make forward and backward transfer **the same measurement in two
directions**, each geometric and behavioural. Backward is already defined.

**Geometric: split specific from generic.** On task `t+1`'s manifolds, from
`h_0` (init) to `h_t` (after task `t`):

```
total     Δlog α_fwd(t)   = log α_{y_{t+1}}(h_t) − log α_{y_{t+1}}(h_0)
generic   Δlog α_gen(t)   = same, averaged over random balanced dichotomies
specific  FT_geo(t)       = Δlog α_fwd(t) − Δlog α_gen(t)
```

- `generic` is the original "option value" idea (RQ2), now with a defined
  job: it separates "the representation became more separable in general"
  from "the past prepared *this* task". Only `specific` is transfer.
- `h_0` is identical across γ₀ (invariant I6), so the baseline is shared.
- Decompose `FT_geo` into the same three channels.

**Behavioural: paired fresh learner.**
`FT_beh(t) = log( steps_fresh(t+1) / steps_stream(t+1) )`, with the fresh
learner on the same init `W(0)`, trained on task `t+1` alone. Positive means
the stream made it faster.
- Cost: training only, no GLUE measurement. Measurement dominates arm cost,
  so this is cheap. Run it for every task, not a subsample.
- **Report also, not read:** zero-shot loss on task `t+1` before training on
  it (`loss_curve[0]`), the standard GEM-style forward transfer. Free; keeps
  comparability with the CL literature.

**Primary question, unchanged:** which channel of `FT_geo` predicts
`FT_beh`, conditional on the label and input relation between `t` and
`t+1`. ~~Same-dichotomy pairs are excluded.~~

**[Amended 2026-11-06, Kati] — A2 as decided:**
- **Specific:** `FT_geo = Δlog α_{y_{t+1}} − Δlog α_{y*}`, using the existing
  held-out probe dichotomy y*. It is a fixed-y measure of the same estimator kind
  as `y_{t+1}`. Record `s_r(y*, y_t)`.
- **Channels:** only `Δlog α_{y_{t+1}}` is decomposed.
- The generic-ensemble subtraction above (`Δlog α_gen`) is **dropped**.
- **Behavioural:** steps to criterion against a fresh learner from the same
  init, as above.
- **Zero-shot:** store the first loss of each task, not the whole loss curve
  (v3 did not store `loss_curve`).
- **Exclusion:** only Hamming 0 with frozen input. Under drift or jump, a
  Hamming-0 pair is not the same task.

## A3. Stream (setting)

**Add a two-task paradigm as the controlled experiment.** `A → B`, one pair,
with both relations set exactly. This is the standard design in CL theory
(two-task teacher–student) and removes the problem that a random walk only
controls consecutive steps.

| paradigm | controls | answers |
|---|---|---|
| two-task `A → B` | label relation and input relation between A and B, exactly | mechanism: C1–C3, forward transfer |
| stream `T = 16` | consecutive relations; the rest is realised | accumulation, position, order, spacing |

**[Amended 2026-11-06, Kati]** Order:
1. The controlled two-task paradigm comes **first**. It runs as **A → B → A**,
   which adds relearning: savings, and A's geometry after re-exposure.
2. The stream paradigm comes second.

Boundary 1 of each stream (A → B) is the bridge between the two. Spec and
implementation: `notebooks/core/aba_controlled.ipynb` and `aba_lib.py`.

**In the stream, analyse on realised relations.** Retained capacity of task
`j` at time `t` is read against `S_r(j, t)` and `ρ_in(j, t)`, the realised
label and input relation between the measured and the current task, plus the
mean relation over intervening tasks. The cliff becomes an exact statement
(`S_r(j, t) = 1`), and the random walk stays, since it is the natural stream.

**Name the operators, then measure them.**
- *Label relation:* `S_r(y, y') = 1 − 2·Hamming(y, y')/P`. Manipulated by
  Hamming steps.
- *Input relation:* the correlation of manifold centres between tasks.
  Manipulated by the AR(1) redraw `c' = ρ c + √(1−ρ²) z`. Axes and shapes are
  held, so this is **centre displacement**, not general input change. Say so.
  At lag `k` the expected correlation is `ρ^k` (drift 0.9 is about 0.28 at
  lag 12).
- For real data, the operator is rebuilt (for example, a class-preserving
  transformation of graded strength), and `ρ_in` is **measured on the
  inputs**. The claim is stated on `ρ_in`, so the axis means the same thing
  across datasets even when the manipulation differs.

## A4. Measurement and inference

- **Fixed-dichotomy capacity.** One sentence in the method: `Ψ_eff ∈ [0, 1]`
  holds for the generic ensemble only. Under a fixed dichotomy it can exceed 1;
  simulated capacity confirms `α` there (LOG, Finding 5). Detail in an
  appendix.
- **Estimator convention.** `full_P` throughout. State that `D_eff` and
  `Ψ_eff` differ by 20–25% from the pairwise convention used in prior GLUE
  work, so numbers are not directly comparable.
- **Units.** Report effects in `log α`, with 95% intervals over the sampling
  unit. Floors become the **resolution criterion** (resolved when the
  interval excludes 0 and `|effect| ≥ 2` floors), not the reporting unit.
  Reviewers read log-ratios. Floors are internal.
- **Inference.** Sampling unit: unique (arrangement, init) draw. Paired
  design across γ₀ and conditions. A **confirmatory set** of six tests:
  C1, C2, C3, the forward-transfer primary, and the two richness-axis
  agreements. Holm correction across the set. Everything else is
  exploratory and labelled in the caption. `n = 8` by default; ~~`n = 16`
  only where a magnitude of `Δρ_c` is claimed (between-arrangement variance
  is 5–10× within, `docs/14` §5).~~ **[Amended 2026-11-06, Kati]** No n = 16.
  `Δρ_c` magnitudes are exploratory at n = 8.
- **[Amended 2026-11-06, Kati] Confirmatory-test specification (to draft, then
  Kati signs).** For each of the six tests:
  - the primary cell;
  - the paired contrast per unit;
  - the statistic: a sign count plus a bootstrap CI over unique draws;
  - the criterion;
  - Holm correction across the six.

  C1 is an interaction contrast. The richness-axis agreements need a **positive
  agreement criterion**: same sign, with the CI excluding 0 on both axes, and
  the same ordering of conditions. Failure to reject a difference does not
  count. Kati sets the thresholds.

---

# Part B — Method section (draft)

## 1. Setting

A **task** is a pair `τ = (X, y)`: `P` input manifolds `X = {X_μ}` and a
balanced dichotomy `y ∈ {±1}^P` labelling each manifold. A **stream** is a
sequence `τ_1, …, τ_T` presented in order; a **learner** is trained on each
task to a common criterion before the next (§2). Two relations between any
pair of tasks define the setting:

- **label relation** `S_r(τ, τ')`, agreement of the dichotomies;
- **input relation** `ρ_in(τ, τ')`, correlation of the manifold centres.

Both are manipulated and both are measured; results are stated on the
measured relation. A third, **schedule** (order and spacing of
presentations), varies with composition held fixed.

We use two paradigms. In the **two-task** paradigm both relations between
`A` and `B` are set exactly; it isolates mechanism. In the **stream**
paradigm (`T = 16`) only consecutive relations are set and the rest are
realised and recorded; it measures how effects accumulate with lag and
position.

**Instantiation.** Synthetic manifolds after Chou et al. (2025): `P = 16`
four-dimensional spherical manifolds in `d = 150`, `M = 150` points each~~,
labelled by four binary factors (labelling only; centres drawn
independently)~~. **[Amended 2026-11-06]** Each manifold is a unit 3-sphere of
radius R in a random 4-dimensional span, plus small isotropic noise, with
independently drawn centres. Dichotomies are random balanced draws or Hamming
steps from the previous one. Label relation is set by Hamming steps; input relation by an
AR(1) redraw of the centres with axes and shapes held (centre displacement).
Real-data instantiations (§4) keep the task definition and measure `ρ_in`.

## 2. Learner and operating point

**Core learner.** One-hidden-layer ReLU network, width `N = 300`, scalar
readout initialised at zero, full-batch gradient descent on MSE, in the μP
parameterisation of Graldi et al. (2025), whose scalar `γ₀` sets feature
learning (richness) while initial weights are held identical across `γ₀`.
Every non-default choice and its reason is listed in the choices register
(Appendix).

**Operating point.** Each task is trained until current-task loss reaches
~~`ε = 0.05`~~ **`ε = 0.1`** of the zero-output loss (**[Amended 2026-11-06]**:
the loss is `½·mean(err²)`, so the zero-output loss is 0.5 and the target is 0.05). The rule is defined identically for any
loss or algorithm, so different learners are each run at their own operating
point and compared only through within-learner contrasts.

**Measured richness.** ~~Relative weight change per task, `r_t`, with `r_0`
as the learner-level summary and kernel change as a secondary measure.~~
**[Amended 2026-11-06]** The analytic NTK on a fixed probe set is primary:
relative change of the full kernel, relative change of its readout block, and
kernel–label alignment. Relative weight change from initialisation and per
task (`r_t`) are secondary. Within
a learner, `γ₀` is manipulated and `r` checks it; across learners, `r` is
the axis.

## 3. Measurement

**Representation geometry.** For a representation `h` and a dichotomy `y`,
manifold capacity `α_y(h)` (GLUE; Chou et al. 2025) is the critical load at
which a linear readout realises `y` on the manifolds `{h(X_μ)}`. It factors
exactly (Chou et al. 2026):

```
log α = log Ψ_eff + log(1 + R_eff⁻²) − log D_eff
        utility       radius            dimension
```

so any change in capacity splits into three geometric channels with no
residual. We use the `full_P` estimator throughout. The **centre correlation**
`ρ_c` between two tasks' manifold centres in `h` describes how the
representation relates tasks rather than how well it supports one.

**Transfer in two directions.** All measured at task boundaries.

| | geometric | behavioural |
|---|---|---|
| **backward** (task `j`, after `t > j`) | `Δlog α_{y_j}` from end of `j` to `t`, three channels | loss on `j`, end of `j` → `t` |
| **forward** (task `t+1`, before training it) | specific `FT_geo`: change in `α_{y_{t+1}}` minus change in ~~generic capacity~~ **capacity for the held-out probe y\*** [Amended 2026-11-06], from `h_0` to `h_t` | `FT_beh`: steps to criterion vs a paired fresh learner |

Negative backward change is forgetting; positive is backward transfer. The
~~generic term, capacity averaged over random dichotomies,~~ **probe term,
capacity for a held-out dichotomy never trained on** [Amended 2026-11-06],
separates a representation that became more separable in general from one
prepared for a particular task.

## 4. Design and inference

**Core design.** Richness `γ₀ ∈ {0.03, …, 10}` × label relation × input
relation (frozen, drift, jump), in both paradigms. Initialisations and
streams are shared across conditions (paired design).
**[Amended 2026-11-06]** The two-task paradigm (A → B → A) runs first, and the
stream second.

**Ablations.** Each tests one named alternative explanation (generator
structure, tuning leakage, learning-rate law, dimensionality versus
dynamics, width, stopping rule, schedule) and is reported in one table.
**[Amended 2026-11-06]** Width is N ∈ {150, 600} against the core's N = 300.

**Extensions.** The same measurement on further learners (~~width,~~ depth,
optimiser, loss), on continual-learning methods used as instruments (EWC
constrains parameter directions, replay re-presents inputs; we ask which
channel each protects), and on real manifolds with designed dichotomies
(image classes; neural population responses). The question each time is
whether the within-learner contrasts keep their sign.

**Inference.** The sampling unit is a unique (arrangement, initialisation)
draw, `n = 8`. **[Amended 2026-11-06]** C2 (stream sets sign) is scoped to
full-batch gradient descent. Effects are reported in `log α` with 95% intervals and called
resolved when the interval excludes zero and the effect exceeds twice the
measurement floor. Six confirmatory tests, Holm-corrected; all else is
exploratory and labelled as such.
