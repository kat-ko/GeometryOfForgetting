# S-HH versus S-LH at unique n=8

Same dichotomy throughout both corners (`S_r ≡ 1`). The contrast is whether
the arrangement **drifts** (S-HH, AR(1) at 0.9) or **jumps** (S-LH, s_f=0.1).
Unit: unique (arrangement, init) draw, n=8 per cell, module A. Zero new
training. Source: `scripts/analyse_shh_vs_slh.py` → `results/shh_vs_slh.json`.

No figure or writeup edits.

## What the contrast is

| corner | dichotomy | arrangement |
|---|---|---|
| S-HH | `y_0` repeated 16× | slow drift, `s_f(0,12)≈0.28` |
| S-LH | `y_0` repeated 16× | jump, `s_f(0,12)≈0` |

Retained capacity of task 0 is always measured on `(A_0, y_0)`. Later training
is the same labels on either a nearby arrangement or a new one.

## Task 0, lag 12 — the finding-3 quantity

| γ | S-HH floors | S-LH floors | HH−LH | Welch p | HH 8/8 sign | LH 8/8 sign |
|---|---|---|---|---|---|---|
| 0.03 | +0.8 | +0.2 | +0.6 | 1×10⁻⁷ | + (unresolved) | + (unresolved) |
| 0.1 | **+4.5** | +0.1 | +4.4 | 5×10⁻¹¹ | gain | unresolved |
| 0.3 | **+9.3** | **−3.8** | +13.2 | 1×10⁻¹¹ | gain | loss |
| 1 | **+11.8** | **−14.0** | +25.8 | 2×10⁻⁸ | gain | loss |
| 3 | **+10.9** | **−29.8** | +40.6 | 3×10⁻⁷ | gain | loss |
| 10 | **+6.2** | **−55.8** | +62.0 | 8×10⁻⁷ | gain | loss |

Wilcoxon paired-by-seed p = 0.0078 at every γ (all 8 seeds, HH > LH). Bootstrap
CIs on the means exclude zero from γ=0.1 (HH) and from γ=0.3 (LH).

Repeating the dichotomy is **not** sufficient for a gain. The gain requires a
related input distribution. Jumping the arrangement while keeping the labels
produces a richness-dependent **loss** on the original clouds, larger than the
HH gain and growing with γ.

The restatement “continued training raises retained capacity, and the rise
survives a drifting input distribution” is true of S-HH and false of S-LH. The
isolated claim is: **at a fixed dichotomy, drift raises retained capacity of
the original arrangement; a jump destroys it.**

## Channels, task 0 lag 12

Signed log-space terms (positive = pushed α up). Unique n=8.

| γ | term | S-HH | S-LH | HH−LH | p |
|---|---|---|---|---|---|
| 1 | utility | +0.132 | −0.077 | +0.209 | 3×10⁻⁹ |
| 1 | radius | −0.015 | −0.073 | +0.058 | 3×10⁻⁸ |
| 1 | dimension | +0.103 | −0.109 | +0.212 | 6×10⁻⁸ |
| 10 | utility | +0.055 | −0.411 | +0.466 | 1×10⁻⁶ |
| 10 | radius | −0.012 | −0.157 | +0.145 | 7×10⁻⁶ |
| 10 | dimension | +0.071 | −0.466 | +0.537 | 4×10⁻⁷ |

In S-HH, utility and dimension raise α against a small unresolved radius
subtraction. In S-LH every term is negative. The jump does not spare a channel.

## Lag and position

Task 0, floors, unique n=8.

| γ | lag | S-HH | S-LH |
|---|---|---|---|
| 1 | 4 / 8 / 12 / 15 | +9.1 / +11.9 / +11.8 / +11.2 | −6.5 / −11.7 / −14.0 / −14.3 |
| 10 | 4 / 8 / 12 / 15 | +8.7 / +9.0 / +6.2 / +3.1 | −36.2 / −51.4 / −55.8 / −56.0 |

The HH lag-decay at γ=10 is as previously reported. The LH loss grows then
plateaus. The contrast is present at every lag.

Matched lag 4, by task position:

| γ | task | S-HH | S-LH |
|---|---|---|---|
| 1 | 0 / 4 / 8 | +9.1 / +10.4 / +10.2 | −6.5 / −1.5 / −2.0 |
| 10 | 0 / 4 / 8 | +8.7 / +12.6 / +12.8 | −36.2 / −16.1 / −12.0 |

HH gain increases with position (more remaining same-task drift after the
task). LH loss is worst on task 0 (the longest remaining jump-stream) and
milder later. That is “how much of the jump-stream is still to come,” not a
privilege of the first task.

## Finding 4 reread — Δρ_c (generic, signed)

Unique n=8. Labels are the corrected 2×2.

**γ=10**

| corner | what it is | Δρ_c | SEM | Spearman | sign-test p |
|---|---|---|---|---|---|
| S-HH | same task, drift | **+0.055** | 0.008 | +0.89 | 0.0078 |
| S-LH | same task, jump | **+0.110** | 0.007 | +0.89 | 0.0078 |
| S-HL | different tasks, drift | **−0.055** | 0.008 | −0.79 | 0.0078 |
| S-LL | different tasks, jump | +0.011 | 0.006 | +0.06 | 0.73 |

Same-task mean +0.082; different-task mean −0.022. At fixed same-task,
jump − drift = **+0.055** (LH converges *more* than HH).

The published readout main effect (+0.104) is mostly same-versus-different
task. “Readout similarity drives centre convergence” rereads as **repeating
the dichotomy drives centre convergence**. Both same-task cells converge; the
one different-task drifting cell is the only decorrelator.

The feature main effect (−0.061) is drift versus jump, and it does not have
the sign a “similar inputs keep centres together” story would want: at a
fixed repeated dichotomy, jumping collapses centres more.

Additivity is still a good description of the four cells (interaction +0.006).
What it adds are now **repetition × input change**, not two similarity levels.

γ=3 is the same pattern at smaller amplitude (same-task +0.075, different-task
+0.021, S-HL unresolved rather than decorrelating).

## Generic capacity, behaviour, optimisation

Generic Δ log α (first to last boundary), floors:

| γ | S-HH | S-LH |
|---|---|---|
| 1 | −3.2 | −8.2 |
| 10 | −12.1 | −32.9 |

Generic capacity falls in both; more under jumps. Retained-of-task-0 and
generic move together in sign in S-LH and oppositely in S-HH (retained rises,
generic falls). That is the surplus distinction, now at a fixed dichotomy.

Behavioural CFr is small in both at γ≥1 (HH ~0, LH 0.006–0.013). Task-0 CFr at
γ=10 is 0.001 vs 0.032, not resolved as a contrast (p=0.074). Geometry and
accuracy remain different objects: S-LH can lose 56 capacity floors on `(A_0,
y_0)` while still classifying those clouds.

Mean steps per task (matched-loss 0.05): γ=1, 48 vs 92; γ=10, 6 vs 12. The
jump corner takes about twice as long to reach the same loss. Probe margin at
stream end is similar (γ=1: 0.067 vs 0.065; γ=10: 0.054 vs 0.044).

## What this does to the four findings

1. **Channel reorganisation (γ).** Untouched as a γ result. Flag only: S-LH
   sits in the three-corner forgetting pool, and under the reread that pool
   mixes different-task forgetting with same-task domain shift. Whether to
   keep the pool is a later decision; not re-analysed here.
2. **Surplus (generic vs retained).** Untouched as a distinction. S-HH is the
   cell where they move oppositely.
3. **The gain.** Not backward transfer. Continued training on `y_0` raises
   retained capacity of `(A_0, y_0)` **when the arrangement drifts**, and
   destroys it **when the arrangement jumps**. The S-HH/S-LH split is the
   evidence; it was on disk and not reported.
4. **The two axes on ρ_c.** Recast, not discarded. Repeating a dichotomy
   drives centre convergence; changing the task while drifting is the
   decorrelating cell. Jump versus drift at a fixed repeated task moves ρ_c
   in the opposite direction from a “feature similarity” label.

## Not done

No figure edits, no writeup edits, no new arms. Readout-axis repair (P=16
levels `{1.0, 0.75, 0.5, 0.25}` vs P=32) and anchored/order streams wait on
this being read.
