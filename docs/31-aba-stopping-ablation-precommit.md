# 31 — A–B–A stopping-rule ablation: pre-commit

**Status: committed 2026-11-09, before the ablation job was submitted.**
- The design and budget were decided by Kati on 2026-11-09 (item 10 of her review, budget rule "max over
  arms").
- The ablation must be read before docs/29 is signed.

## 1. Why

docs/28 trains every phase to matched loss: training stops at loss ≤ 0.05. The learner therefore sets
its own exposure, meaning its number of updates. The setting cards in docs/30 §4 show how different this is.
On seeds 8–15, the median updates are:

| phase | γ0 0.1 | γ0 10 |
|---|---|---|
| A1 | 1938 | 51 |
| B at h 8 frozen | 2137 | 36 |

So every γ0 contrast in docs/28 is also an exposure contrast. This ablation fixes exposure, identical for every
γ0, and asks whether the γ0 effects keep their sign. Fixed training per task is also the standard
continual-learning protocol.

## 2. Design (fixed)

Identical to the docs/28 replication (seeds 8–15, job 4396) except for stopping.

| item | value |
|---|---|
| stopping | `ABAConfig.stopping = "fixed"`: exactly the budget of full-batch GD updates per phase, whatever the loss. The loss is checked before each update, as in C5. |
| budget | **A1 2300, B 2400, A2 2000**; fresh B takes the B budget. The same for every γ0. |
| budget rule | The largest criterion step count of any replication arm, rounded up. Measured maxima: A1 2266, B 2373, A2 1973, fresh B 2365. So every learner reaches loss ≤ 0.05 in every phase, and the ratios stay readable. |
| design, seeds | The same 219 specs: γ0 {0.1, 1, 10} × h {0, 4, 8} × s_f {1, 0.9, 0.1} × seeds 8–15, plus 3 fresh-noise arms at seed 8. |
| everything else | As docs/28 §2: data, learner, lr, capacity measurement, probes, n_t = 200. |
| floors | Recomputed under the fixed budget, on seed 8, K = 8 estimator seeds, after A (`aba_cluster.py floors`). The representation after A is not the criterion one. |
| code | `aba_lib` with the `stopping` option (commit `2245806`, lib `0ee97f82b6bee621`). Criterion mode is unchanged: the job-4349 bit-identity check is rerun (§6). |
| usable arm | Every phase ran its full budget with a finite loss. |

**Accepted trade-off.** Exposure is matched; fit is not. Rich learners train roughly 50–100× past criterion,
so their final loss is far below 0.05. The final loss per γ0 and phase is reported (§5) so that this is
visible.

## 3. Readings

- H1, H2, H5a and H6 are read per stratum, by the docs/28 §3–§4 rules: t = 3.636, the 2-floor resolution, and
  the not-read rule for ratios.
- The fixed-budget floors are used.
- The same code reads both runs: `aba_summary.readings`. It is checked to reproduce every docs/28 §9.3 reading
  of these four hypotheses in both criterion runs.
- **Not read:**
  - H4 and anything step-based (FT_beh, savings): under equal budgets these are ratios of equal numbers.
  - H3 and H5b: not part of the question.

## 4. Pre-committed reading

Each stratum is compared with the same stratum of the criterion replication (seeds 8–15, docs/28 §9.3).

- **H1 and H2**, which are contrasts:
  - **same sign**: the fixed-budget contrast is resolved with the criterion sign;
  - **sign flip**: it is resolved with the opposite sign;
  - **unresolved**: otherwise.
- **H5a and H6**, which are thresholds:
  - **same reading**: the same side of the threshold (held / held, or not held / not held);
  - **flip**: held ↔ not held;
  - **unresolved**: the fixed-budget reading is unresolved.
- A stratum not read in either run is reported as **not read**.
- **Interpretation, per stratum only:**
  - Same sign or same reading: the γ0 effect in that stratum is **not driven by exposure**.
  - A sign flip or flip in **any** stratum is reported as a finding: the effect in that stratum depends on
    exposure.
  - Unresolved is neither agreement nor a flip.
- Strata are not pooled, and no cross-stratum summary is formed.

**Expected direction (stated before running).**
- Under matched-loss stopping, the lazy learner (γ0 0.1) got 60–70× more updates on B than the rich one (h 4 frozen 1493 vs 21; h 8 frozen 2137 vs 36)
  (γ0 10). Yet B removed a smaller share of A's gain from the lazy learner: h 4 frozen 0.39 vs 0.72, h 4 drift
  0.26 vs 0.60.
- So exposure plausibly worked *against* the γ0 effect. With equal budgets the rich learner gets far more
  updates on B.
- The H1 difference d = f(10) − f(0.1) at h 4 frozen and h 4 drift is therefore expected to **keep its sign or
  grow**.
- This makes the ablation a robustness check rather than a likely result-changer.

## 5. Reported alongside

The final training loss per configuration, γ0 and phase (A1, B, A2, fresh B), as median [min, max] over
seeds, beside the criterion target of 0.05. Also the fixed-budget floors next to the criterion floors.

## 6. Execution

1. **Check job** (`sbatch/aba_check.sh`, grime job 4453):
   - the criterion seed-0 arm under lib `0ee97f82b6bee621` must be bit-identical to job 4349's row;
   - then one fixed-budget arm (seed 8, γ0 0.1, h 8, frozen) is timed sequentially to size the job.
   - If the arm is not bit-identical, nothing runs.
   - **Result (2026-11-09):** **bit-identical** (17 fields, ignoring lib_hash, stamp and timing).
     The fixed arm took 184 s sequential (train 112 s, GLUE 66 s), peak RSS 124 MB. Its final losses
     were A1 0.036, B 0.047, A2 0.040 and fresh B 0.047, so even the slowest learner ends below 0.05.
     The ablation job is sized from this: about 740 s per arm in parallel (×4), about 85 min for the
     grid, `--time 4 h`.
2. **Ablation job** (`sbatch/aba_ablation.sh`):
   - the grid with `--seed-offset 8 --stopping fixed --budget 2300,2400,2000`, into
     `results/aba/FULL_0ee97f82b6bee621_fixed.jsonl`;
   - then `floors` with the same flags.
   - The notebook is not executed: its identity and validation checks assume criterion stopping.
3. Copy back the checkpoint and floors. `python notebooks/core/aba_summary.py --ablation <jsonl> <floors>`
   writes §7. The readings are reported as computed.

## 7. Results

<!-- BEGIN GENERATED: ablation results -->
(not yet run)
<!-- END GENERATED: ablation results -->
