# Reference sheet — continual-learning metrics (Graldi definitions A.1–A.10)

```
Source: Graldi, Breccia, Lanzillotta, Hofmann & Noci, "The Importance of Being
        Lazy: Scaling Limits of Continual Learning", ICML 2025 —
        arXiv 2506.16884v2, Appendix A.1 (Definitions A.1–A.10).
Version: arXiv:2506.16884v2
Derived from: Appendix A.1 pasted from the PDF (human, 2026-09-01). NOT copied
              from 00-math-spec.md.
Transcribed by: agent, 2026-09-01, from the human paste (supersedes 2026-07-31
                draft that used a_{i,i} as peak).
Verified by (human): Kati, 2026-09-01 — A.3/A.4 are the running max, 1/(T−1),
                     sum to T−1. Toy-example check does not distinguish the two
                     references.
Status: source formulas corrected 2026-09-01. Code still uses a_{i,i}
        (protocol-deviations.md D.2). Do not treat this sheet as a license to
        change pipeline.forgetting_metrics. Artifact family: 07-writeup.md §A.7.
```

---

## Notation

`a_{j,i}` = test accuracy on task `T_i` after training on `T_j`,
`i, j ∈ {1..T}`. `T` = number of tasks. `L_{j,i}` = training-loss analogue.

## Accuracy metrics (A.1–A.6)

| Def | Metric | Source formula |
|---|---|---|
| A.1 | Learning Accuracy (LA) | `LA = (1/T) Σ_{i=1}^T a_{i,i}` |
| A.2 | Learning Error (LE) | `LE = 1 − LA` |
| A.3 | Catastrophic Forgetting (CF) | `CF = 1/(T−1) Σ_{i=1}^{T−1} [ max_{t∈{i,...,T−1}} (a_{t,i}) − a_{T,i} ]` |
| A.4 | Catastrophic Forgetting Rate (CFr) | `CFr = 1/(T−1) Σ_{i=1}^{T−1} [max_t (a_{t,i}) − a_{T,i}] / max_t (a_{t,i})` with the same `max_{t∈{i,...,T−1}}` |
| A.5 | Average Accuracy (AA) | mean of the last row after all tasks: `⟨ a_{T,i} ⟩_i` (source writes 0-indexed `a_{T−1,i}`) |
| A.6 | Average Error (AE) | `AE = 1 − AA` |

**Peak is a running max, not `a_{i,i}`.** Three discrepancies vs the 2026-07-31
draft: normalisation is `1/(T−1)` not `1/T`, the sum runs to `T−1` not `T`, and
the reference is the maximum accuracy achieved on task `i` after it has been
learned, not the accuracy immediately after training it.

Those two references coincide only if accuracy on a task is highest right after
learning it. **S-HH is a case where a past task's retained capacity improves
during subsequent training.** If that appears behaviourally as well as
geometrically, `max_t ≠ a_{i,i}` and the two formulas diverge exactly in the
condition the paper is about. Current behavioural CF in that corner is ~0
(§5.1: −0.0005 to +0.0001); if accuracy is already at ceiling they still
coincide. Do not switch the code. Recorded in `protocol-deviations.md` D.2
and `07-writeup.md` §A.7.

**Toy example (source A.1):** two-task, peak = `a_{1,1}`. Model A 100→80 ⇒ CF
20%, CFr 20%. Model B 40→30 ⇒ CF 10%, CFr 25%. CFr favors A. **The check
passed because in that example the two references coincide.** A formula check
on a case where the discrepancy is invisible cannot catch it.

## Loss-based analogues (A.7–A.10)

Source copies the accuracy algebra onto `L_{j,i}`:

| Def | Metric |
|---|---|
| A.7 | Learning Loss (LL) `= (1/T) Σ_i L_{i,i}` |
| A.8 | CF (loss) — same sum as A.3 with `L` in place of `a` |
| A.9 | CFr (loss) — same as A.4 with `L` |
| A.10 | Average Loss (AL) — mean of the last row of `L` |

If loss *rises* after other tasks, A.8's terms are negative. That is the source
formula. A "drop in loss" reading is not what the algebra does.

## What the code does

`pipeline.forgetting_metrics` uses `acc[j, j]` as peak, not
`max_{t ∈ {j,...,T−2}} acc[t, j]`. Equivalent when accuracy on task `j` is
highest at the end of training `j` and never recovers. Recorded as a
deviation in `protocol-deviations.md`. Do not silently switch to the running
max; that would re-rank existing arms.

---

## Key scalar results (context, not implementation)

- Optimal degree of feature learning `γ₀* ≈ 0.1` for the primary CL setting;
  transfers across widths (and, with μP+1/√L, across depths).
- `γ₀*` shifts **toward 1 as task similarity rises**.
- Lazy→rich transition `γ₀^LRT`: below it CFr is low and width-scaling helps;
  above it CFr grows and width-scaling stops helping.
- **Pretraining effect:** at high task similarity, features learned on task 1
  transfer; later tasks show no further feature evolution beyond a γ₀ threshold —
  the network becomes effectively lazy after task 1 (basis for H4).

## Task-similarity proxies used by Graldi (this project's P2 critique)

- **Permuted-MNIST:** similarity = fraction of pixels **not** permuted between
  tasks (inner square permuted first). 1.0 = original MNIST; 0.0 = all pixels
  permuted.
- **Split-TinyImageNet / CIFAR:** classes-per-task as a structural proxy for
  non-stationarity (e.g. "5/2" = 5 tasks of 2 classes).

Both are **input-side or structural proxies, never a measured representational
similarity** — exactly the gap `PROJECT.md` §2 (P2) says this project closes by
recording the full `s_f`/`s_r` similarity matrices.
