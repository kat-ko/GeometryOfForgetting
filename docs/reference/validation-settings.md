# Reference sheet — GLUE estimator validation settings

```
Source: Chou, Le, Wang & Chung, "Feature Learning beyond the Lazy-Rich
        Dichotomy", ICML 2025 — arXiv 2503.18114, Appendix B.5 (Figs 9–12)
        and Appendix D.1.1.
Version: arXiv:2503.18114v2
Transcribed by: agent (Opus 4.8), 2026-07-31
Verified by (human): Kati, 2026-09-01
Status: human-signed 2026-09-01 against Chou App. B.5. Settings match:
        N=1000, P=2, M=200; D 2→10 and R 0.8→2 with correlations zero; then
        D=4, R=1 while ρ_c, ρ_a, ψ sweep 0→0.8. Secondary effect (large ρ_c
        → effective radius increases) confirmed in Chou's text.
```

Used by `docs/02-validation-suite.md` §1 (ground-truth recovery).

---

## B.5 ground-truth recovery — parameter settings

Ambient / manifold sizes (both sweeps):

| Symbol | Value |
|---|---|
| `N` (ambient) | 1000 |
| `P` (manifolds) | 2 |
| `M` (points/manifold) | 200 |

**Sweep 1 — dimension & radius recovery** (all correlations set to 0):
- `D_ground`: 2 → 10
- `R_ground`: 0.8 → 2.0

**Sweep 2 — alignment recovery:**
- **`D_ground` fixed at 4, `R_ground` fixed at 1** *(explicit in B.5; now also
  in `02-validation-suite.md` §1)*
- `ρ_c_ground`, `ρ_a_ground`, `ψ_ground`: each swept 0 → 0.8

---

## Direction of effect (B.5 figure captions; sign checks for §1)

| Ground-truth ↑ | Capacity | Secondary effect noted in B.5 |
|---|---|---|
| `D_ground` (Fig 9) | **decreases** | — |
| `R_ground` (Fig 10) | **decreases** | — |
| `ρ_c_ground` (Fig 11) | **decreases** | large ρ_c → effective **radius increases** |
| `ρ_a_ground` (Fig 12) | **increases** | large ρ_a → effective **dimension decreases** |

`ρ_c` and `ρ_a` move capacity in **opposite** directions (matches `02` §1).

> The authoritative, per-measure sign table with the App. B.4 derivation lives in
> `glue-sign-conventions.md` (human-owned, high transcription risk). This sheet
> only records the B.5 empirical directions used as validation targets.
