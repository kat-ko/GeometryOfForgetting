# 35 — A–B–A similarity map, Phase 1: summary (EXPLORATORY)

**Status: DRAFT for Kati, 2026-11-11.** Phase 2 does not start until Kati has read this (docs/34).
- Source: `notebooks/core/aba_similarity_map.ipynb`, executed 2026-11-11 against
  `results/aba/FULL_e3011fa49aca1994.jsonl`. Analysis code is in `notebooks/core/aba_map_analysis.py`. Every
  number below is copied from the notebook output or computed with the same functions.
- **EXPLORATORY throughout.** Means with 95% t-intervals over the 8 seeds (16–23) are descriptive only. There
  is no family, nothing is pooled across configurations, and no confirmatory claim is made.
- **Exposure.** Under matched-loss stopping a similar B needs fewer updates, so every similarity effect
  includes its effect on exposure. The notebook plots updates on B beside every map, and the setting cards
  (notebook §5) list them per configuration.

## 1. Run and integrity

- **Job.** Job 4651 ran on grime with 64 CPUs and 32 GB, on commit `07b3047` (clean tree) with lib hash
  `e3011fa49aca1994`. It took 1 h 48 min: the grid was 6485 s and MaxRSS 9.1 GB. All 771 arms finished:
  32 configurations × 3 γ₀ × 8 seeds, plus the 3 fresh-noise identity arms.
- **Reproduction before submission (job 4648).** The docs/28 arm from job 4349 (seed 0, γ₀ 10, H8, s_f 0.1)
  reran bit-identical under the new hash. Only the two task-vector fields are new; they were listed as added,
  not compared.
- **Transfer.** The SHA-256 of the checkpoint is identical on the cluster and the laptop.
- **Validation (notebook §4): 10/10 PASS.**
  - all 771 arms present, and all usable (every phase converged);
  - schema complete and finite;
  - one commit, one lib hash;
  - realised s_r = 1 − H/8 exactly;
  - ρ_in = 1 at s_f 1;
  - task-vector cos on h₀ identical across γ₀;
  - identity cell: 0 updates on B and Δlog α_A = 0 (24 arms);
  - fresh-noise identity arm: Δlog α_A = +0.0000 at every γ₀.

## 2. Expectation readings (rules fixed and committed before the run; notebook §6)

| | Reading | Key numbers |
|---|---|---|
| E1, Hiratani corners | **held** | At every γ₀, both f and bottleneck forgetting are largest at H8 × s_f 1. f is smallest at H4 × s_f 0.2. |
| E2, shape in s_f | **held: monotone, no peak** | Spearman of f with s_f is +0.86 (γ₀ 0.1, H4) and +0.98 to +1.00 elsewhere. No interior s_f has an interval above both endpoints, so the peak of Lee et al. and Hiratani is not seen. |
| E3, shape in H | **held** (see caveat a) | f(H8) − f(H0), paired, ranges +0.77 to +0.91 with every interval above 0, and has 0 inversions over H 0–8. Δlog α_A at H16 is above H8: +0.08 to +1.79. |
| E4, H16 at γ₀ 0.1 | **held by the equivalence rule** (see caveat b) | Δlog α_A(A → B) at H16 × s_f 1 is +0.025 [+0.020, +0.029]. Recorded: γ₀ 10 +0.025 [−0.055, +0.104]; γ₀ 1 +0.005 [−0.009, +0.018]. |
| E5, Menghi tension | **held** | FT_beh at H0 × s_f 0.4 / 0.6 is positive at every γ₀: γ₀ 0.1 +0.41 / +0.59, γ₀ 1 +0.90 / +1.14, γ₀ 10 +1.21 / +1.48. All intervals are above 0, so the partly-similar-input interference reported by Menghi is not seen. |
| E6, richness | **held** | Spearman of the cell means of f over 32 cells: 0.1 vs 1 +0.974; 0.1 vs 10 +0.939; 1 vs 10 +0.984. |
| E7, restoration | **unresolved** (held in 1 of 4) | At s_f ≥ 0.9, mean ρ is 0.94–1.01 (all ≥ 0.9). At s_f 0.1, mean ρ is 0.918 / 0.882 / 0.884 / 0.841 for γ₀ 1 H4, γ₀ 1 H8, γ₀ 10 H4 and γ₀ 10 H8. Only γ₀ 10 H8 has its interval wholly below 0.9 ([+0.812, +0.870]). |

**Caveats on two readings.**
- **(a) E3 part (ii) holds by construction.** The network's own accuracy on A at the end of B equals
  1 − H/16 at s_f 1 (0.250 at H12 and 0.000 at H16), as already seen for F5. The behavioural part of E3 is
  therefore not informative. Part (i), the dose up to H8, and part (iii), the recovery of geometry at H16,
  carry the reading.
- **(b) E4: the two clauses of the rule overlap.** The interval [+0.020, +0.029] lies inside ±0.05 (the
  "held" clause) and also excludes 0 (the "not held" clause). The code tests "held" first. Read as equivalence
  within the SESOI, A's capacity is unchanged. Read strictly, it rises slightly: f = −0.10. **Kati to decide
  which clause governs.** No reading is changed here.

## 3. Maps against the nominal axes (notebook §5, profiles)

The f profiles below are copied from the table behind the profile figures. Updates on B are in the setting
cards.

**H sweep: f against H** (columns H 0, 2, 4, 6, 8, 12, 16)

| s_f | γ₀ | H0 | H2 | H4 | H6 | H8 | H12 | H16 |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.1 | +0.00 | +0.05 | +0.35 | +0.65 | +0.77 | +0.70 | −0.10 |
| 1 | 1 | +0.00 | +0.30 | +0.60 | +0.80 | +0.89 | +0.82 | −0.00 |
| 1 | 10 | +0.00 | +0.53 | +0.70 | +0.83 | +0.84 | +0.79 | −0.01 |
| 0.9 | 0.1 | −0.16 | −0.01 | +0.23 | +0.52 | +0.69 | +0.80 | +0.37 |
| 0.9 | 1 | −0.06 | +0.23 | +0.49 | +0.73 | +0.84 | +0.93 | +0.43 |
| 0.9 | 10 | −0.03 | +0.43 | +0.63 | +0.77 | +0.80 | +0.84 | +0.33 |

- **Dose, then a plateau, then a collapse at a pure flip.** f rises from H0 to H8, stays high at H12, and at
  H16 × s_f 1 (B = −A) drops back to about 0 at every γ₀, the rich learner included. At s_f 0.9, H16 still
  removes 0.33–0.43 of A's gain.
- **Richness steepens the start of the dose.** At H2, f is 0.05 at γ₀ 0.1, 0.30 at γ₀ 1 and 0.53 at γ₀ 10.
- **H12 is not the mirror image of H4.** In label terms, B at H12 is B at H4 with the sign flipped. Yet f is
  0.70–0.82 at H12 against 0.35–0.70 at H4, at s_f 1. H12 also needs more updates on B: γ₀ 0.1 2382 vs 1422,
  γ₀ 10 44 vs 22 (medians). So this difference includes exposure.
- **Bottleneck forgetting has the same shape at a smaller scale.** At s_f 1 it peaks at H8 (γ₀ 0.1 +0.05,
  γ₀ 1 +0.23, γ₀ 10 +0.21) and is about 0 at H16.

**s_f sweep: f against s_f** (columns s_f 1, 0.95, 0.9, 0.8, 0.6, 0.4, 0.2, 0.1)

| H | γ₀ | 1 | 0.95 | 0.9 | 0.8 | 0.6 | 0.4 | 0.2 | 0.1 |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.1 | +0.00 | −0.07 | −0.16 | −0.26 | −0.30 | −0.23 | −0.11 | −0.06 |
| 0 | 1 | +0.00 | −0.03 | −0.06 | −0.10 | −0.11 | −0.07 | +0.01 | +0.04 |
| 0 | 10 | +0.00 | −0.02 | −0.03 | −0.04 | −0.03 | +0.01 | +0.13 | +0.14 |
| 4 | 0.1 | +0.35 | +0.30 | +0.23 | +0.16 | +0.04 | +0.02 | −0.01 | +0.04 |
| 4 | 1 | +0.60 | +0.55 | +0.49 | +0.41 | +0.22 | +0.13 | +0.07 | +0.11 |
| 4 | 10 | +0.70 | +0.66 | +0.63 | +0.56 | +0.37 | +0.25 | +0.16 | +0.21 |
| 8 | 0.1 | +0.77 | +0.75 | +0.69 | +0.62 | +0.40 | +0.30 | +0.22 | +0.17 |
| 8 | 1 | +0.89 | +0.88 | +0.84 | +0.77 | +0.52 | +0.36 | +0.25 | +0.19 |
| 8 | 10 | +0.84 | +0.82 | +0.80 | +0.76 | +0.62 | +0.46 | +0.35 | +0.29 |

- **At H4 and H8, f falls monotonically as the input centres move apart.** Exposure does not drive this:
  updates on B at H8 change little across s_f (γ₀ 0.1 from 1949 at s_f 1 to 1572 at s_f 0.1). At H4, the
  slight upturn at s_f 0.1 is inside the intervals.
- **At H0, the sign of f depends on γ₀.** With the same labels on moved inputs, B **adds** capacity to A at
  γ₀ 0.1, down to f = −0.30 at s_f 0.6. At γ₀ 10, B costs capacity once the inputs move far: Δlog α_A is
  −0.27 at s_f 0.2 and −0.29 at s_f 0.1. Updates on B at H0 fall steeply as s_f approaches 1 (γ₀ 0.1: 1466
  at s_f 0.1, 0 at s_f 1), so this axis is exposure-confounded.

**Channel shares of Δlog α_A(A → B)** (utility / radius / dimension)
- **The split is set mostly by γ₀, not by the cell.** At H4–H12, wherever Δ_total is clearly non-zero:
  γ₀ 0.1 about 0.2 / 0.4 / 0.4; γ₀ 1 about 0.43 / 0.18 / 0.40; γ₀ 10 about 0.52 / 0.06 / 0.42. This matches
  E6's "only size and channel change".
- **Where Δ_total is near 0, the shares are not readable.** Examples: H16 × s_f 1 at γ₀ 1 (+2.39 / −2.32 /
  +0.93) and H4 × s_f 0.1 at γ₀ 0.1 (−0.63 / +0.53 / +1.10).

**Restoration ρ.** At γ₀ 0.1, ρ is above 1 in every cell (1.03–1.30), so relearning A overshoots A's
original capacity. At γ₀ ≥ 1, see E7.

## 4. Maps against measured similarity (notebook §5, second figure)

- **Neither measured similarity orders f on its own.**
  - At H8 the task-vector cosine is about 0 at every s_f (medians −0.05 to +0.03), yet f runs from 0.17 to
    0.89.
  - Against the task-vector cosine, f is highest near 0 and falls towards both −1 (H16) and +1 (H0).
  - Against realised ρ_in, the cells separate by H.
- **Consequence for docs/29.** The rewrite as map-predicted vs observed (docs/34 §8) needs the map indexed by
  both coordinates, S_r and ρ_in, not by a single similarity.
- **FT_beh rises with the task-vector cosine at every γ₀**, up to +7.5 / +5.8 / +3.9 at cos = 1 (H0 × s_f 1,
  where B is already learned).
- **Updates on B fall steeply as the task-vector cosine approaches 1 at H0.** This is the exposure confound
  in measured units.

## 5. Third sample of the docs/28 configurations (amendment b; unpooled, not a docs/28 reading)

The configurations are H 4 and 8 × s_f 1, 0.9 and 0.1 × 3 γ₀, with the quantities f, Δlog α_A, ρ and
FT_beh. That is 72 cells, each compared with seeds 8–15 and with seeds 0–7 (144 comparisons).
- **143 of 144 95% intervals overlap.** At 95%, about 7 non-overlaps would be unsurprising.
- **The one exception:** FT_beh at H4 × s_f 0.1 × γ₀ 0.1 is +0.244 [+0.202, +0.286] here against +0.149
  [+0.106, +0.191] for seeds 8–15. It overlaps seeds 0–7: +0.192 [+0.134, +0.250].
- **Full table:** notebook §5.

## 6. Limits

- EXPLORATORY: 95% intervals for description, no family.
- One learner (1M, μP, full-batch GD) at matched current-task fit, so every similarity effect includes its
  effect on exposure.
- Two slices only. Interactions between H and s_f away from the slices need Phase 2, which waits for Kati.
- s_f moves the input centres; it is not the literature's input-feature similarity. H changes which
  directions separate the classes as well as the readout.
- Optional Phase 1b (a fixed budget on the H sweep at s_f 1) runs only if Kati asks. It would separate the
  H12-vs-H4 difference from exposure.

## 7. Open for Kati

1. E4 rule overlap: equivalence clause or strict clause (§2 caveat b).
2. E3 part (ii) holds by construction: keep it, or drop it from future readings.
3. Phase 1b (fixed budget) yes or no; Phase 2 go or no go.
4. docs/29 rewrite: index the map prediction by (S_r, ρ_in) jointly (§4).
