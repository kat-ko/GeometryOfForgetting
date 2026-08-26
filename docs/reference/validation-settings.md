# Reference sheet — GLUE estimator validation settings

```
Source: Chou, Le, Wang & Chung, "Feature Learning beyond the Lazy-Rich
        Dichotomy", ICML 2025 — arXiv 2503.18114, Appendix B.5 (Figs 9–12)
        and Appendix D.1.1.
Version: arXiv:2503.18114v2
Transcribed by: agent (Opus 4.8), 2026-07-31
Verified by (human): ____________          # blank until checked
Status: agent-verified against arXiv text; NOT human-verified.
        Code may not cite this sheet until "Verified by (human)" is filled.
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
- **`D_ground` fixed at 4, `R_ground` fixed at 1** *(← explicit in B.5; the current
  `02-validation-suite.md` §1 omits these fixed values — add them)*
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
