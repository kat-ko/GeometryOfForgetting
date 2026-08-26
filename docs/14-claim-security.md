# 14 — Project check: what the current experiments support

Compiled 2026-08-19 from the artifacts, not from the drafts. Every number below is re-read from a
named file by a named script; where a number contradicts something in `paper/body.tex` or
`docs/07-writeup.md`, the artifact wins and the disagreement is called out in §6.

Sources: `results/seed_security.json` (`scripts/audit_seed_security.py`, new for this check),
`results/unique_n8.json`, `results/unconfound_k3.json`, `results/width_g10_n40.json`,
`results/gamma5_n40_onset.json`, `results/gamma30_unique.json`, `results/audit_propagation.json`,
`results/tier2_backward_transfer.json`, `results/measurement_null.json`, the figure sidecars, and
`src/analysis/ledger.py` for the pooling rules. `scripts/make_inventory.py --check` passes: all 102
inventoried numbers match their sources, 49 are additionally pinned to the sentence that quotes them
(31 in the submitted paper, 18 in the long draft only).

The question this document exists to answer is not "what did we find" but **"at what sampling unit
is each finding true"**. That turns out to separate the results into three sharply different tiers,
and to put one sentence of the submitted paper at risk.

---

## 1. What exists

2,944 measured arms on disk, in seven populations that must not be pooled with each other:

| population | arms | design of one cell | role |
|---|---|---|---|
| `results/phase1/` (registered grid) | 1,280 (960 at `a=0`) | 8 unique × 5 unread copies | Figures 2 and 4, §5.1–5.3, §B |
| `results/gamma_5/` | 64 | 4 unique × 4 copies | superseded "before" at γ₀=5 |
| `results/gamma_5_n40/` | 160 | 5 arrangements × 8 inits | §5.4 onset intermediate point |
| `results/width/` | 96 | 2 per cell | superseded; still what fig:width plots |
| `results/width_g10_n40/` | 320 | 5 arrangements × 8 inits | §5.3 width bound |
| `results/gamma_ext/` (γ₀=30) | 64 | 4 unique × 4 copies | appendix robustness, fenced |
| `results/unconfound_k3/` | 960 | 3 arrangements × 40 inits | appendix robustness (crossed) |

Provenance on the grid is clean and was checked rather than assumed: worst attribution-identity
residual 4.4×10⁻¹⁶ over 48,640 evaluations, 0 of 33 scope-audit checks failing, 0 of 1,280 arms
non-converged, two git SHAs across the set (a mid-grid solver change, both re-derived identically),
and 9,600 retained-capacity comparisons available in module A.

---

## 2. Three sampling units, and why they are not interchangeable

`run_arm` draws an arm from two independent generators plus a measurement seed:

* **initialisation** — `paired_init(seed)` draws `W(0)` (and, separately, the measurement seed
  `default_rng([seed, 20260811])`). Shared across conditions, which is what makes a corner contrast
  a paired contrast.
* **arrangement** — `stream_rng(stream_id) = default_rng([20260817, stream_id])` draws the manifold
  arrangement *and* the task dichotomies.
* **the arm** — the crossed (arrangement, initialisation) cell one file corresponds to.

A SEM or sign test over arms is an inference about **new arms**. It is only an inference about new
arrangements if the arms sample arrangements independently. The three populations differ exactly
here, and it decides what each one can support:

| population | arrangements sampled | inits sampled | what its arm-level interval means |
|---|---|---|---|
| registered grid | 8, **confounded** with init | 8, confounded with arrangement | a genuine arrangement-scale interval (§2.1), but no attribution to either factor |
| `gamma_5_n40`, `width_g10_n40` | **5** | 8 | too narrow: 40 crossed cells are not 40 draws (§2.2) |
| `unconfound_k3` | **3** | 40 | excellent on the init axis, weak on the arrangement axis, exactly as its pre-commit said |

The duplication on the grid is exact, not approximate: across the five `stream_id` copies of each
seed at γ₀=10, the largest spread in Δρ_c is **0.00×10⁰**. The copies carry no information at all.

### 2.1 The grid's unique-n=8 interval is the right size — a genuinely reassuring result

Because `seed` keyed the arrangement as well as the initialisation, the grid's 8 unique arms per
cell are 8 independent (arrangement, init) draws. So `SEM8` should already be an arrangement-scale
interval. `unconfound_k3` lets us check that against a real arrangement axis:

| corner at γ₀=10 | grid `SEM8` | K=3 between-arrangement SEM | ratio |
|---|---|---|---|
| S-HH | 0.0078 | 0.0044 | 1.77 |
| S-HL | 0.0078 | 0.0081 | 0.97 |
| S-LH | 0.0067 | 0.0072 | 0.94 |
| S-LL | 0.0060 | 0.0109 | 0.55 |

Same order of magnitude on all four, within a factor of two either way. **The unique-n=8 relabelling
did not just shrink the n; it produced intervals of the correct scale for generalising to a new
arrangement.** That is the strongest single argument that the grid's headline inference is sound.

### 2.2 The RNG-fixed 5×8 arms are the weak spot

`gamma_5_n40` and `width_g10_n40` have 40 genuinely distinct arms per cell, which is what they were
run for, but only **5 arrangements**. Their arm-level SEM is therefore about 3× too small for an
arrangement-level claim (γ₀=5 S-HL: 0.0032 over arms against 0.0096 over the 5 arrangement means),
and — a design fact independent of any result — a sign test on 5 units has a floor of p=0.0625, so
**no arrangement-level claim from these sets can reach α=0.05 however unanimous it is.**

---

## 3. What a sign test can do at each n we have

| n | p if unanimous | p if one unit flips | can reach α=0.05 | where this n occurs |
|---|---|---|---|---|
| 4 | 0.125 | 0.625 | **no** | γ₀=30 probe; superseded γ₀=5 set |
| 5 | 0.0625 | 0.375 | **no** | arrangement-level tests in the 5×8 arms |
| 8 | 0.0078 | **0.0703** | yes | registered grid, unique n |
| 16 | 3.1×10⁻⁵ | 5.2×10⁻⁴ | yes | file-level γ₀=30 (copy-inflated) |
| 40 | 1.8×10⁻¹² | 7.5×10⁻¹¹ | yes | K=3 within-arrangement init axis |

Two consequences worth stating in the paper's own voice:

* Every 8/8 result on the grid is **one seed away from p=0.070**, which does not clear 0.05. The
  γ₀=10 S-HL result is not fragile in fact — K=3 backs it independently at 40/40 inits in each of
  3 arrangements — but it is fragile *as supported by the grid alone*.
* The γ₀=30 probe and the old γ₀=5 set cannot reject at α=0.05 under any outcome. Treating them as
  "did not reach significance" would misdescribe a design limit as a measurement.

---

## 4. Claims by security tier

### Tier A — survives both sampling units, and is safe to state plainly

1. **Rich training forgets far more, and the amount is resolvable.** Three-corner forgetting at
   γ₀=10 is −69.7 α floors (Δ log α = −1.2917, n=120, lag 12, task 0), against 0.2 floors at
   γ₀=0.03. First resolvable at γ₀=0.1 (2.1 floors) — but see Tier B on that endpoint.
2. **The composition of forgetting reorganises with richness.** Alignment share 0.08 → 0.45, radius
   0.39 → 0.11, dimension 0.53 → 0.44 across γ₀ = 0.1 → 10. K=3 reproduces the direction in **3/3**
   arrangements (alignment up, radius down from γ₀=1 to 10).
3. **`S-HH` gains retained capacity — a gain, not a smaller loss.** Grid: +11.8 floors at γ₀=1,
   +6.2 at γ₀=10. K=3 per arrangement: +11.56 / +11.68 / +12.27 at γ₀=1 and +5.74 / +5.17 / +4.58 at
   γ₀=10, sign-stable 3/3. General across the stream: 10 of 10 (task, lag) cells resolve as gains at
   each of γ₀ = 1, 3, 10. And it is specific to that corner: of 520 cells examined, 172 have a
   positive mean, 98 clear ±2 floors, and **0 resolvable gains occur outside `S-HH`** (largest
   elsewhere 0.74 floors, a third of the gate).
4. **The corner ordering at γ₀=10.** `S-LH` moves most (+0.110) and `S-HL` is the corner that
   decorrelates (−0.055). Grid: 8/8 unique seeds, p=0.0078. K=3: **40/40 inits declining in every
   one of the 3 arrangements.** This is the best-supported claim in the paper.
5. **The 2×2 separates into two main effects.** Readout similarity drives convergence (+0.104),
   feature similarity drives decorrelation (−0.061), and the additive model predicts the held-out
   corner to 0.012. Across K=3 arrangements the readout effect is tight (+0.1095 / +0.0949 /
   +0.1030) and the feature effect is sign-stable but varies 2.8× (−0.0646 / −0.0643 / −0.0228).
   State the readout effect as a magnitude; state the feature effect as a direction.
6. **Task position is a larger effect than lag, and the composition does not care.** 1.56× from
   task 8 to 0 at fixed lag 4, against 1.29× from lag 4 to 15 within task 0; shares agree to within
   0.032 between lag 12 and lag 4.
7. **Initialisation and arrangement explain almost none of the per-arm variation.** Init R²=0.007
   (grid, n=1,200); arrangement R²=0.001 measured on K=3 where arrangements actually vary (n=3,600).
   What does explain it: distance above the common asymptote plus lag (0.589) and which corner
   (0.523), reaching 0.882 together, with residual sd 0.143 against a total of 0.436 (7.7 floors).
   *This coexists with §2.1 without contradiction: arrangement barely moves an individual arm's
   forgetting, yet it shifts the corner-level Δρ_c means — a small variance component on a big
   quantity and a large one on a small quantity.*
8. **The H2d correlation reverses with pooling.** Pooled Spearman −0.128 [−0.150, −0.102]; within
   γ₀ it is positive at γ₀ ≥ 1 (+0.108, +0.190, +0.322) and null below; only γ₀=10 survives full
   stratification by condition and boundary (+0.135). The mechanism generalises; the pair does not.

### Tier B — real, but only as a bound. Do not upgrade to a measurement

| claim | number | why it stays a bound |
|---|---|---|
| width invariance | 0.62% between N=150 and 600 at γ₀=10 (−71.6 vs −72.0 floors, n=120 each) | marginally, arrangement moves the magnitude **17×** further than width does (8.8–10.6% spread across the 5 arrangements at fixed width). Paired within arrangement the direction is consistent — N=600 the slightly larger loss in all five (+0.20, +0.13, +1.57, +0.09, +0.26 floors, mean +0.45) — but the 95% interval is [−0.34, +1.23] floors and 5 units cannot reach α=0.05. Two widths, and N=300 is a different population |
| the radius channel in `S-HH` | −0.0154 / −0.0015 / −0.0124 log units at γ₀ = 1 / 3 / 10 | below the 3σ `R_eff` gate at every γ₀ ≥ 1 (1.9, 0.2, 1.1 `R_eff` floors). Distinguishable as a population mean at γ₀=1 (3.9 SEM8) but not at γ₀=3 (0.3) or γ₀=10 (2.5). A bound under 2 floors, not a sign |
| where the gain peaks | interpolated 1.23, unique CI [1.12, 1.35] | the peak slides with the lag it is read at (2.86 → 0.96 across lags 4 → 15) and the spread across defensible functional forms (0.28) exceeds the sampling CI. No location claim |
| onset of forgetting | 2.1 floors at γ₀=0.1, margin +0.10 floors (4.9%) | the margin sits inside the floor's own uncertainty: 4 measurement seeds give a CV a 41% relative SE, so a floor is known to ×/÷1.50 (a factor of ~2.3 between the ±1 SE ends). Keep "between γ₀ = 0.1 and 0.3" |
| the ruler itself | α floor flat in γ (registered 1.87% stands); `R_eff` CV 0.28% → 1.13% across γ, **not** flat | 4 seeds per γ. Every "clears its gate" statement inherits the ×/÷1.50 |

### Tier C — not licensed. State as unresolved, or do not state

| claim | grid | K=3 across arrangements | verdict |
|---|---|---|---|
| `S-LL` converges at γ₀=10 | +0.0114, 3/8 unique, p=0.73 | +0.0017 / +0.0182 / **−0.0195** — sign flips | nothing, not a small effect. At the top edge of the lazy-arm drift band [+0.005, +0.010] |
| interaction at γ₀=10 | +0.006, 0.82 SEM8 | **−0.0130 / +0.0050 / −0.0160** — sign flips | genuinely unresolved; the additive model stands. The sign flip is new evidence *for* the paper's existing caution |
| additivity fails at γ₀=30 | +0.016 at 1.74 unique SEM (3.89 on 16 copy-inflated files) | — | not licensed. Paired-seed 1.53 SEM, bootstrap CI includes 0, 1 of 4 seeds opposite sign |
| γ₀ = 3 → 10 centre-collapse step | −0.010, CI [−0.038, +0.070] | — | unresolved; both endpoints |
| `S-HL` magnitude as a number | −0.055 (grid, unique n=8) | −0.0500 / −0.0511 / **−0.0262** | the *sign* is Tier A; the *magnitude* spans 2× across arrangements and the grid's −0.055 falls **outside** the three-arrangement range. Quote the sign and the ordering, not −0.055 as a population value |

---

## 5. How secure are the findings with respect to seeds and arms — summary

**Secure against new initialisations.** Every leading result. Initialisation explains 0.007 of
per-arm forgetting; `S-HL` at γ₀=10 declines for 40 of 40 inits in each K=3 arrangement; the γ₀=5
effect holds for 8 of 8 inits (p=0.0078) once averaged over arrangements. Nothing in this project is
at risk from the choice of `W(0)`.

**Secure against new arrangements — for the grid results.** The grid's 8 unique arms are 8
independent arrangement draws and its `SEM8` matches the K=3 between-arrangement SEM to within a
factor of two (§2.1). The three qualitative claims that K=3 was pre-committed to test all reproduce
3/3.

**Not secure against new arrangements — for two things.** (a) *Magnitudes* of Δρ_c: the
between-arrangement SEM is 5.5–10× the within-arrangement SEM for all four corners, so any Δρ_c
magnitude quoted with a within-arm SEM understates the uncertainty about the population of
arrangements by nearly an order of magnitude. (b) The γ₀=5 onset point — see §6.

**Under-powered by construction.** n=4 sets (γ₀=30) and arrangement-level tests on 5×8 sets can
never reach α=0.05. Three arrangements estimate a standard deviation only to about ±50%
(1/√(2(n−1)) at n=3), so all the between-arrangement numbers in §4 are themselves coarse — they
bound that variance component rather than measure it.

**One clean piece of luck.** The measurement seed is derived from `seed`, not from `stream_id`, so
each arrangement group contains the same set of measurement seeds. The between-arrangement
differences reported here are therefore not inflated by measurement noise.

---

## 6. The one sentence in the submitted paper this audit puts at risk

`paper/body.tex:216–218` and `paper/figures-appendix.tex:44–45` currently say the γ₀=5 probe is
"already negative … (−0.0102 ± 0.0032, declining on 31 of 40, p = 6.8×10⁻⁴)", and body.tex adds
"The onset sits between 3 and 5".

Those numbers are correct as arm-level statistics. They are not an arrangement-level result, and the
onset is an arrangement-level claim. Re-pooled by arrangement:

| arrangement | mean Δρ_c at γ₀=5 | arms declining |
|---|---|---|
| 0 | −0.0114 | 8/8 |
| 1 | −0.0169 | 8/8 |
| 2 | **+0.0084** | 5/8 |
| 3 | −0.0419 | 8/8 |
| 4 | **+0.0109** | 2/8 |

Three of five arrangements decorrelate; two move the other way, and both of those land **on the
lazy-arm drift band [+0.005, +0.010]** — +0.0084 inside it, +0.0109 just above it — i.e. in those two
arrangements γ₀=5 is indistinguishable from an arm that has barely moved.
Arrangement-level sign test 3/5, p=1.0 (ceiling 0.0625); SEM over arrangements 0.0096 against
0.0032 over arms; 95% t interval over arrangements **[−0.0368, +0.0165], which includes zero**.

Meanwhile the two bracket endpoints are arrangement-level results from the grid's 8 independent
draws: γ₀=3 is 4/8, p=1.0 (absent) and γ₀=10 is 8/8, p=0.0078 (present).

**Honest form.** The onset lies between γ₀=3 and γ₀=10 (bracket factor 3.33). At γ₀=5 the
arm-level mean is negative, but that is not the unit the onset needs.

This is a downgrade of a claim, not an error in a number — the arithmetic on the 5×8 set is
right and the population label was what was missing. It is the same fault family as §A.4
(mismatched populations), one level deeper.

---

## 7. The 8-arrangement pre-commit — scored

`results/gamma5_k8_precommit.json` was written before any of the 256 arms existed. Rule: ≥7/8
arrangement means negative *and* the 95% t interval excludes zero ⇒ locate the onset at 5;
fewer than 7/8 ⇒ keep the 3→10 bracket and call γ₀=5 arrangement-dependent.

**Outcome** (`results/gamma5_k8.json`; 256/256 usable; 1,716 s wall; arrangements 0–4 identical
to `gamma_5_n40`, 160/160). Arrangement means:

| arr | Δρ_c | arms declining (Spearman) |
|---|---|---|
| 0 | −0.0114 | 8/8 |
| 1 | −0.0169 | 8/8 |
| 2 | +0.0084 | 5/8 |
| 3 | −0.0419 | 8/8 |
| 4 | +0.0109 | 2/8 |
| 5 | +0.0015 | 8/8 |
| 6 | +0.0166 | 6/8 |
| 7 | −0.0221 | 8/8 |

**4/8 negative.** 95% t interval **[−0.0234, +0.0097], includes zero.** Sign-test p = 1.0
(ceiling 0.0078). Verdict: **arrangement_dependent**. The three new arrangements added two more
non-negative means. Power was enough to clear α=0.05 had the effect been arrangement-stable; it
is not.

§5.4 now quotes four of eight and the 3→10 bracket. The arm-level p = 6.8×10⁻⁴ is not cited as
onset evidence.

**Would not settle anything further.** K=5 on the unconfound (already ruled out as fishing); a
genuine-n N=300 width arm; a D-sweep; γ₀=30 at genuine n. Another γ₀=5 arrangement sweep would
be estimating a variance the paper does not claim.

**Unaffected.** Tier A items 1–8, both prereg refutations (H1d and H6), the six-family error
appendix, and the width/radius/peak bounds in Tier B.

---

## 8. Standing summary

Of nine tested hypotheses, four failed and three of those four have replacements the drafts argue
are better than the originals; three more (H3a/H3b, H4, H5) were never run because Phases 2 and 3
were cut, and are reported as untested. The measurement chain is in good order: 102 inventoried
numbers all re-derive, the identity residual is at machine precision, no arm failed to converge, and
the duplication fault that dominated the last week is now fully characterised — including the
discovery, in this check, that it left the grid's intervals correctly scaled.

The honest position is: **one very well-supported qualitative structure** (richness reorganises
which geometric channel carries forgetting; the 2×2 separates into a convergence effect and a
decorrelation effect; one corner gains capacity and only that one), **a set of magnitudes that are
firm within an arrangement and soft across arrangements**, and **a short list of things we have
decided not to claim**. The γ₀=5 onset location is in that last list: 4 of 8 arrangements, scored
against a rule written before the arms existed.
