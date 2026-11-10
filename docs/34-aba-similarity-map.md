# 33 — A–B–A similarity map: before the stream

```
Status:  RATIFIED by Kati, 2026-11-10. Overrides docs/29 §8 and docs/25 §8
         ordering until this study is reported.
Slot:    docs/24 §4 core, controlled paradigm (A–B–A), exploratory extension.
Effect:  the stream is PARKED. docs/29 stays unsigned; sbatch/stream_timing.sh
         is not submitted. No stream arm of any kind runs until Kati says so.
Kind:    EXPLORATORY. No confirmatory claims. Expectations are written before
         the run (§6) and read as held / not held / unresolved, but nothing
         here enters a Bonferroni family.
```

## 1. Why the stream waits

The A–B–A results say *which corners* of similarity are harmful or benign. They
do not say what the *shape* of the dependence is, because each axis has only three
levels (H ∈ {0, 4, 8}, s_f ∈ {1, 0.9, 0.1}).

1. **The literature's open questions are two-task questions.** The questions are:
   - whether forgetting is monotonic or peaked in similarity;
   - how feature and readout similarity interact;
   - whether forgetting is in the readout or in the representation.

   They are answered in controlled pairs, not in streams:
   - Hiratani 2024: catastrophic at high feature × low readout similarity, and
     non-monotonic in feature similarity.
   - Lee, Goldt & Saxe 2021: "intermediate task similarity leads to greatest
     forgetting".
   - Menghi et al. 2025: label structure × input similarity interact, and
     representations go from aligned to anticorrelated.
2. **The stream cannot be read without a map.** In a 16-task stream, realised
   similarity, lag, position and accumulation are mixed. A dense A–B–A map gives
   a **prediction for the stream**: if forgetting were pairwise, the lag-1 change at
   position 8 would equal the A–B–A value at the same realised similarity. The
   stream then tests deviations from that prediction (accumulation, history).
   That is what the stream uniquely measures, and it is how docs/29 will be
   rewritten (§8).
3. **It is cheap.** A–B–A arms take minutes.

## 2. What is already established (A–B–A, docs/28, 30, 31, 32)

At matched current-task fit, per configuration:
- **The worst corner is frozen input × H8** (same inputs, conflicting labels).
  It has the most geometric and behavioural forgetting and negative forward
  transfer (H4).
- **The benign corner is H0 × drift** (same labels, drifting inputs). A gains
  slightly (H3: positive, below resolution in the replication), and forward
  transfer is strong.
- **Labels are graded:** H0 < H4 < H8 at frozen and drift, every γ₀.
  Behavioural forgetting at frozen input equals H/P, so it is set by the readout.
- **Inputs:** frozen ≈ drift. Jump removes behavioural forgetting and shrinks
  geometric removal.
- **Richness** changes size and channel, not the ordering. Its size is
  protocol-dependent (docs/31, docs/32).

## 3. Design

| item | value |
|---|---|
| learner, data, training, measurement | exactly as docs/28 §2 (1M, N = 300, μP, matched-loss stopping C2/C5, M = 150, n_t = 200, bottleneck probe n = 16, neutral probes K = 4) |
| γ₀ | 0.1, 1, 10 |
| label relation H | **0, 2, 4, 6, 8, 12, 16**. Only even values keep B balanced (8 vs 8). s_r = 1 − H/8 goes from +1 to −1. |
| input relation s_f | **1, 0.95, 0.9, 0.8, 0.6, 0.4, 0.2, 0** |
| seeds | **16–23** (fresh; 0–15 are spent on the confirmatory runs) |
| controls | identity cell (H0, s_f 1, exact copy); one fresh-noise identity arm per γ₀ |

**Phase 1, two slices (run first):**
- an H sweep at s_f ∈ {1, 0.9};
- an s_f sweep at H ∈ {4, 8}.

That is (7·2 + 8·2 − 4 overlap) × 3 × 8 = **624 arms**. Phase 1 already answers
the shape questions on both axes.

**Phase 2, the full grid**, only if Phase 1 shows structure that the slices do
not resolve (for example, an interaction): 7 × 8 × 3 × 8 = **1344 arms** (Phase 1
arms reused).

**H = 16, the readout/representation diagnostic.** B has every label of A flipped.
That is the same task up to the readout's sign, and fixed-dichotomy capacity is
identical for y and −y.
- If the network only flips its readout, behaviour collapses (accuracy on A →
  0), while A's capacity stays.
- A loss of capacity at H16 means that the network rebuilt features it did not
  need to. That bears directly on rich vs lazy.

## 4. Measured similarity, in the literature's units

Our axes do not map one-to-one onto Hiratani's or Lee's:
- Their input-feature similarity is about which input directions the task uses.
  Our s_f is about how far the inputs move.
- Our H is not pure readout similarity, because changing labels also changes
  which directions separate the classes.

So both axes are also **measured** per seed and reported next to the nominal
values:
- **Input relation:** realised ρ_in(A, B), the mean cosine of matched centres
  (docs/29 §2b definition).
- **Task-vector similarity:** the cosine between the minimum-norm least-squares
  linear readouts of A and B, fitted on the input points (w = X⁺y, per task).
  Compute it also on the init representation h₀, and report both.
- **Label relation:** s_r(y_A, y_B) (exact).

Every map is shown against nominal axes and against measured similarity.

## 5. Outputs

One notebook, `notebooks/core/aba_similarity_map.ipynb`. It answers one question,
"how do forgetting and transfer depend on label and input similarity?", in the
docs/25 §7 format. Per γ₀:
- maps of: fraction removed f; Δlog α_A (A1 → B); channel shares; FT_geo
  (neutral) and specific FT_beh; bottleneck-probe forgetting; Δρ_c (signed,
  normalised); savings; exposure (steps);
- the same quantities against the measured similarities of §4;
- a setting card per configuration (docs/29 §2a);
- signed values only, no pooling across configurations, and 95% intervals over
  seeds for description (no family).

## 6. Expectations (write into the notebook before the run; read verbatim after)

These are directions, not thresholds:
- **E1 (Hiratani corners).** Forgetting (f and bottleneck) is largest at high
  input similarity × H8, and smallest at low input similarity, at every γ₀.
- **E2 (shape in s_f).** At H4 and H8, f is monotonic in s_f, *or* peaked at
  intermediate s_f. Record which. A peak would match Lee et al. and Hiratani.
- **E3 (shape in H).** At s_f ∈ {1, 0.9}, f rises from H0 to H8 (a dose). Behaviour
  at H12 and H16 falls below chance on A while geometry recovers towards H0.
- **E4 (H16).** At γ₀ 0.1, A's capacity is unchanged at H16 (readout flip only).
  At γ₀ 10, record whether capacity is lost (feature rebuilding).
- **E5 (Menghi tension).** At intermediate s_f (0.4–0.6) with H0, forward transfer
  stays positive. If it turns negative, that matches Menghi's partly-similar-
  input interference.
- **E6 (richness).** The ordering across similarity cells is the same at every γ₀;
  only size and channel change.

## 7. Rules

- `aba_lib` changes (new H and s_f levels, the task-vector similarity) change the
  hash. First reproduce ONE docs/28 FULL arm bit for bit under the new hash.
- One commit, clean tree. Time ONE arm, then size the job with the measured 3–4×
  parallel slowdown. Cluster only: no laptop runs, no laptop watchers.
- Expectations (§6) are committed before submission.
- Results are reported per configuration with setting cards. "Exploratory" goes
  in every caption.

## 8. After the map

1. Write `docs/34-similarity-map-summary.md` (per configuration, both nominal
   and measured axes).
2. Rewrite docs/29 as **map-predicted vs observed**. For each stream reading, the
   prediction is the A–B–A map value at the stream's realised S_r and ρ_in. The
   confirmatory tests target the deviation, which measures accumulation and
   history.
3. Only then is the stream unparked.

## References

- Hiratani 2024, NeurIPS: https://arxiv.org/abs/2405.20236
- Lee, Goldt & Saxe 2021, ICML: https://proceedings.mlr.press/v139/lee21e.html
- Menghi et al. 2025, Nat. Commun.: https://pmc.ncbi.nlm.nih.gov/articles/PMC12669730
