# Realized stream structure (unique n=8 grid)

Zero training. Source: `scripts/analyse_realized_stream_structure.py` →
`results/realized_stream_structure.json`. Population: registered Phase 1,
`a=0`, `N=300`, one file per `(γ, condition, seed)` (192 unique arms). Stream
matrices depend only on `(condition, seed)` (8 streams per corner). Outcomes:
task 0, lag 12, module A. No figure edits.

## 0. Do arrangements change?

**Yes, in every 2×2 stream that was run.** `make_stream` draws `A_0` then
`A_t = redraw_centers_correlated(A_{t-1}, s_f)`. `s_f ∈ {0.1, 0.9}` are both
strictly less than 1, so centers move and points are re-realized. `s_f = 1`
(copy the previous arrangement) is only `S-fixed-r`, which was registered and
not run.

Notebook §4 and the “clouds never change / only the answer changes” setup
sentence are therefore **false for the 2×2** and true only of an unrun
condition. Interference is not rule-only.

Recording caveat: `S_f[t, t±1]` is written as the **generating** `s_f`, not
the realized cosine. Lags `|i-j|≥2` are realized mean pairwise cosine of
centers. Consecutive realized cosine is not in the file.

## Hamming rounding at `P=16` (load-bearing)

`h = round((P/2)(1−s_r))` to nearest even.

| nominal consecutive `s_r` | Hamming | realized consecutive `s_r` |
|---|---|---|
| 0.9 (S-HH, S-LH) | **0** | **1.0** |
| 0.1 (S-HL, S-LL) | **8** | **0.0** |

High-readout corners are not a small-step walk. Every dichotomy is `y_0`
(8/8 streams, `S_r` identically 1). A8 already recorded this as the reason
S-HH/S-LH have no novel control.

Low-readout consecutive steps are **orthogonal**, not `s_r=0.1`.

## 1. Realized similarity at the lag Figure 2 uses

Task 0 vs task `t`, unique n=8 streams per corner. Lag 12:

| corner | nom. `s_f` | `s_f(0,12)` mean (sd) [min, max] | AR(1) `s_f^12` | nom. `s_r` | `s_r(0,12)` mean (sd) [min, max] |
|---|---|---|---|---|---|
| S-HH | 0.9 | **0.283** (0.013) [0.264, 0.303] | 0.282 | 0.9 | **1.00** (0) |
| S-HL | 0.9 | **0.284** (0.013) [0.264, 0.305] | 0.282 | 0.1 | **0.25** (0.23) [0.00, 0.50] |
| S-LH | 0.1 | **0.003** (0.013) [−0.010, 0.025] | ~0 | 0.9 | **1.00** (0) |
| S-LL | 0.1 | **0.004** (0.012) [−0.007, 0.025] | ~0 | 0.1 | **0.25** (0.23) [0.00, 0.50] |

Effective separation at lag 12 vs nominal consecutive gap 0.8:

- Feature: high−low = **0.28** (corners still separate; labels 0.9 vs 0.1 overstate it).
- Readout: high−low = **0.75** (1.00 vs 0.25). They do **not** collapse. The
  high corner is identity, not a saturated random walk.

Mean `s_f(0,t)` along the stream (high `s_f` pooled from S-HH; AR(1) `0.9^t`):

| t | 1 | 2 | 4 | 8 | 12 |
|---|---|---|---|---|---|
| realized (S-HH) | 0.900* | 0.808 | 0.651 | 0.430 | 0.283 |
| `0.9^t` | 0.900 | 0.810 | 0.656 | 0.430 | 0.282 |

\*t=1 is the stored generating value.

Low-readout `s_r(0,t)` (S-HL): t=1: 0.00; t=4: 0.13; t=8: 0.25; t=12: 0.25.

## 2. Decay law

High `s_f` is AR(1) to sampling noise: MAE of mean `s_f(0,t)` vs `0.9^t` over
t=2…15 is **0.002**. Lag 12 is 0.283 vs 0.282.

Low `s_f` is already ~0 from t=2 (mean ~0.00–0.01, sign-flipping). `0.1^12` is
10⁻¹²; realized is a small residual cosine, not a distinct lag-12 correlation.

Random balanced pair baseline (`P=16`, 20 000 pairs): mean `s_r=0.192`,
P(`s_r=0`)=0.38, P(`s_r=1`)=0.0002.

Low-readout `s_r(0,12)=0.25` (range 0–0.5) sits near that baseline, slightly
above it. The high-readout walk has **not** saturated: it never started
(`s_r=1` at every lag).

## 3. Unintended recurrence and finding 3

`min_{t=1…12} s_r(0,t)`:

| corner | min `s_r` | unique dichotomies (of 16) |
|---|---|---|
| S-HH, S-LH | **1.0** (8/8) | **1** |
| S-HL, S-LL | **0.0** (8/8) | 15.9 |

S-HH does not wander back toward `y_0`. It **never left**. Every task is
task 0’s labels on a drifting arrangement. That is the generating process,
not a subset of streams.

`max_t s_f(0,t)` over t=1…12 is identically the consecutive nominal (0.9 or
0.1) because t=1 is stored as that value. It has no within-corner variance.

Load-bearing check — S-HH gain (Δ log α, task 0 lag 12) vs closest
*arrangement* approach `s_f(0,12)`, unique n=8:

| γ | mean floors | Spearman vs `s_f(0,12)` | p | R² |
|---|---|---|---|---|
| 1 | +11.8 | −0.10 | 0.82 | 0.009 |
| 10 | +6.2 | −0.29 | 0.49 | 0.073 |

`min_t s_r` is constant in S-HH, so that correlation is undefined.

The “some S-HH streams re-train a nearby `y_0`” story is **ruled out as a
within-corner dose**. The remaining statement is structural: S-HH *is*
re-training `y_0` on slowly moving clouds, in every stream. A8 never tested
savings in this corner (control does not exist at `P=16`). The recurrence
null in S-HL/S-LL does not speak to S-HH.

## 4. Which predictor wins

Unique n=32 at one γ (4 corners × 8 seeds). Dependent: Δ log α at task 0 lag 12.

| predictor | R² at γ=1 | R² at γ=10 |
|---|---|---|
| **corner label** | **0.973** | **0.951** |
| `min_t s_r(0,t)` | 0.698 | 0.544 |
| realized `s_r(0,12)` | 0.624 | 0.473 |
| realized `s_f(0,12)` | 0.042 | 0.119 |
| net displacement `1−s_f(0,12)` | 0.042 | 0.119 |
| path `Σ_{t=2}^{12}(1−s_f(0,t))` | 0.044 | 0.122 |

The corner label wins. Realized readout scalars are mostly the Hamming
rounding (1 vs ~0), a two-level coarsening of the four-corner factor. Realized
lag-12 feature similarity does **not** beat the label. This is not a case
where a hidden realized covariate outpredicts the generating parameter.

## 5. Change and similarity are entangled

By construction, consecutive change is `1−s_f`: **0.1** in S-HH/S-HL and
**0.9** in S-LH/S-LL. High feature similarity *is* slow drift; low *is* a
jump. Line 2’s change property co-varies with feature similarity and is not
held constant across corners. “Abrupt at every boundary” describes the
training schedule, not the geometry trajectory.

## What this does to the brief’s hypotheses

1. Notebook §4 / rule-only interference: **needs correcting** (2×2 rearranges
   centers). No figure change in this pass.
2. Corners collapse at lag 12: **feature yes, to 0.28 vs 0; readout no**
   (1 vs 0.25). High readout is identity, not a saturated walk.
3. Finding 3 as accidental re-presentation in some S-HH streams: **no**. All
   S-HH streams are `y_0` throughout; within-corner `s_f(0,12)` does not track
   gain. Restate as: S-HH is the same dichotomy on an AR(1) arrangement at
   `ρ=0.9`, which is still `ρ≈0.28` at lag 12.
4. Audit moral (realized > label): **not supported** on this outcome at this
   unit. The label is the better predictor; the interesting fact is that the
   label does not mean what the 0.9/0.1 tokens suggest.
