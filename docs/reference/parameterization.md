# Reference sheet — γ₀ parameterization (NTP / μP)

```
Source:       Graldi, Breccia, Lanzillotta, Hofmann & Noci, "The Importance of
              Being Lazy: Scaling Limits of Continual Learning", ICML 2025 —
              arXiv 2506.16884, Table 1 + Appendix A.3 (parameterization) +
              A.4 (training details).
Version:      arXiv:2506.16884v2
Derived from: Table 1 cells read from the rendered arXiv HTML (v2), Table 1
              rows (Branch/Output/LR/Weight-variance). Prose from the PDF text
              (A.3/A.4). NOT copied from docs/00-math-spec.md.
Transcribed by: agent (Opus 4.8), 2026-07-31
Verified by (human): ____________          # blank until checked
Status: agent-verified against the rendered arXiv HTML table AND cross-checked
        equal to docs/00-math-spec.md §4.2 (so the spec's Table 1 is also
        confirmed correct — the earlier circularity is resolved). Still awaiting
        human sign-off; code may not cite until "Verified by (human)" is filled.
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

## Facts verified from A.3 / A.4 prose

- **Base width `N₀ = 64`.** At base width, μP and NTP are equivalent ("at
  base-width [64] the μP and NTP are equivalent", Figs 8/8-caption). This is the
  width at which the μP↔NTP normalization is anchored.
- **Optimizer:** SGD, **no momentum, no weight decay** (A.4). Matches invariants
  I1, I3.
- **LR schedule (Graldi's own runs):** cosine schedule, no warmup, **restarted at
  the start of each task**; batch size 128. *(Our project uses plain SGD at a small
  fixed epoch count — see `00` §4/§6; we do not inherit the cosine schedule.)*
- **Readout zero-init:** last layer initialized to 0 so output = 0 at `t = 0`
  ("advised in Yang et al. 2022b, App. D.2"). Matches invariant I5.
- **Depth (out of scope for us):** μP+1/√L (Bordelon et al. 2023) decouples feature
  learning from depth; branch scale `β_ℓ` then depends on `N` and `L`. Recorded
  only because the corrected-LR arm (`00` §4.3) touches depth `L`.
- **Split-CIFAR10 (their setting):** 5 tasks × 2 classes, **separate head per
  task**, 5 epochs/task, `η₀(0) = 30.0`. NB: separate-head + P=2 is exactly what
  our design rejects (invariant I8; problem P1).

---

## UNCERTAIN — base-width normalization constant (blocks nothing until §4.2 test)

`docs/00-math-spec.md` §4.2 requires `μP(γ₀=1, N=64) ≡ NTP(N=64)` to float64.
**Confirmed by extraction:** Graldi states equivalence at base width 64 but does
**not** write out the normalization constant. It must be **derived and
unit-tested**, not guessed (see `docs/02-validation-suite.md` §3
`test_base_width_equivalence`), and the derivation recorded in
`parameterization-derivation.md`.
