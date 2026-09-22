# Reference sheet — γ₀ parameterization (NTP / μP)

```
Source:       Graldi, Breccia, Lanzillotta, Hofmann & Noci, "The Importance of
              Being Lazy: Scaling Limits of Continual Learning", ICML 2025 —
              arXiv 2506.16884, Table 1 + Appendix A.3 (parameterization) +
              A.4 (training details).
Version:      arXiv:2506.16884v2
Derived from: Table 1 cells (human, 2026-09-01). Appendix A.2–A.4 pasted from
              the PDF (human, 2026-09-01). NOT copied from docs/00-math-spec.md.
Transcribed by: agent (Opus 4.8), 2026-07-31
Verified by (human): Kati, 2026-09-01 — Table 1 four cells, A.3 text, Verification 1
Status: Table 1 signed. A.3 transcribed: it identifies the table and does not
        write N/N₀. The caption's "Further details are in App. A.3" is dangling.
        (N/N_base) is an independent derivation from Table 1, confirmed against
        the agent's. See parameterization-derivation.md.
```

Implemented in `docs/00-math-spec.md` §4.

---

## Table 1 — scaling (verified against rendered arXiv HTML v2, rows 66–71)

| | NTP | μP (Mean Field) |
|---|---|---|
| Branch scale `β_ℓ` | `N^{-1/2}` (ℓ>0), `D^{-1/2}` (ℓ=0) | same |
| Output scale `γ` | `1` | `γ₀ · N^{1/2}` |
| LR schedule `η(t)` | `η₀(t)` | `η₀(t) · γ₀² · N` |
| Weight variance `σ_ℓ²` | `1` | `1` |

**⚠ Notation:** in Graldi, `D` is the **input dimension** (`x ∈ R^D`), i.e. **our
`d`**. `docs/00-math-spec.md` §4.2 correctly renders the `ℓ=0` branch scale as
`d^{-1/2}`. Do not confuse Graldi's `D` with our intrinsic manifold dimension `D`.

**Provenance (A.3, verified):** Table 1 uses the notation of Bordelon et al.
(2023, Tab. 1) with the NTP from Yang & Hu (2020, Tab. 1). PyTorch standard
parameterization (SP) is **equivalent to the NTP used here**.

---

## Facts from A.2 / A.3 / A.4 (A.3 in hand, 2026-09-01)

- **A.3** (verbatim substance): many equivalent parameterizations exist; Tab. 1
  uses Bordelon et al. (2023, Tab. 1) notation with the NTP of Yang & Hu (2020,
  Tab. 1); PyTorch SP ≡ this NTP. A.3 does not write the N/N₀ algebra. The
  Table 1 caption's forward reference to A.3 is dangling.
- **Base width `N = 64`, base depth `L = 6`** (A.2).
- **Optimizer:** SGD, **no momentum, no weight decay** (A.4). Matches I1, I3.
  Infinite-width arm: 2-layer ReLU MLP, full-batch GD on MSE, last layer 0 —
  the same model class as ours.
- **LR schedule (Graldi):** cosine, no warmup, **restarted at each task**;
  batch size 128. **Deviation:** we use constant LR with matched-loss stopping.
- **Readout zero-init:** last layer 0 so output = 0 at `t = 0` (A.4, Yang et al.
  2022b App. D.2). Matches I5.
- **`γ₀* ≈ 0.1` is from their ResNet experiments**, not the MLP arm (30 MNIST
  samples, 2 tasks, `ρ = 0`). Any comparison of our `γ*` to theirs is across
  architectures.
- **Split-CIFAR10:** 5 tasks × 2 classes, **separate head per task**,
  `η₀(0) = 30.0`. Separate-head + P=2 is what I8 / P1 reject.

---

## Base-width normalization — independent derivation (Verification 1 signed)

A.3 does not state the constant. Set μP = NTP in Table 1: both rows give
`γ₀ = N^{-1/2}`. Making `γ₀ = 1` the NTP point at `N₀ = 64` is `N → N/N₀`:
`γ_μP = γ₀ (N/N_base)^{1/2}`, `η_μP = η₀ γ₀² (N/N_base)`. Confirmed against
the agent's. At `N = 300`, NTP-equivalent `γ₀ = (64/300)^{1/2} ≈ 0.462`
(between 0.3 and 1). No coincidence claim.
