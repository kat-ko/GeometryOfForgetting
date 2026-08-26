# Reference sheet — continual-learning metrics (Graldi definitions A.1–A.10)

```
Source: Graldi, Breccia, Lanzillotta, Hofmann & Noci, "The Importance of Being
        Lazy: Scaling Limits of Continual Learning", ICML 2025 —
        arXiv 2506.16884, Appendix A.1 (Definitions A.1–A.10), §3.
Version: arXiv:2506.16884v2
Transcribed by: agent (Opus 4.8), 2026-07-31
Verified by (human): ____________          # blank until checked
Status: agent-verified against arXiv text (CFr/CF formulas cross-checked
        against the paper's own toy example). NOT human-verified.
        Code may not cite this sheet until "Verified by (human)" is filled.
```

---

## Notation

`a_{j,i}` = test accuracy on task `i` after training through task `j`,
`i, j ∈ {1..T}`. `T` = number of tasks. (`L_{j,i}` = training loss analogue.)

## Accuracy metrics

| Def | Metric | Formula |
|---|---|---|
| A.1 | Learning Accuracy (LA) — plasticity | `LA = ⟨ a_{i,i} ⟩_i` |
| A.2 | Learning Error (LE) | `LE = 1 − LA` |
| A.3 | Catastrophic Forgetting (CF) | `CF = ⟨ a_{i,i} − a_{T,i} ⟩_i`  (avg **absolute** drop) |
| A.4 | **Catastrophic Forgetting rate (CFr)** | `CFr = ⟨ (a_{i,i} − a_{T,i}) / a_{i,i} ⟩_i`  (avg **relative** drop) |
| A.5 | Average Accuracy (AA) — stability/plasticity tradeoff | `AA = ⟨ a_{T,i} ⟩_i` |
| A.6 | Average Error (AE) | `AE = 1 − AA` |

**CFr is the metric this project references** (`PROJECT.md` §5.1). Verified against
Graldi's toy example: model A 100→80 ⇒ CF 20, CFr 20%; model B 40→30 ⇒ CF 10,
CFr 25%. CFr correctly favors A. (Confirms the `/a_{i,i}` normalization and index
convention above.)

## Loss-based analogues (for infinite-width / MLP runs)

Substitute `L` for `a`; drops become **increases** in loss.

| Def | Metric |
|---|---|
| A.7 | Learning Loss (LL) `= ⟨ L_{i,i} ⟩_i` |
| A.8 | Catastrophic Forgetting (CF, loss) — avg loss increase |
| A.9 | Catastrophic Forgetting rate (CFr, loss) — avg relative loss increase |
| A.10 | Average Loss (AL) `= ⟨ L_{T,i} ⟩_i` |

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
