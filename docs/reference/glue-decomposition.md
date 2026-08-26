# Reference sheet — exact three-factor capacity decomposition

```
Source:       Chou, Kirsanov, Yang & Chung, "Diagnosing Generalization Failures
              from Representational Geometry Markers", ICLR 2026, Appendix B.3
              ("Effective geometric measures and the capacity decomposition").
Version:      arXiv:2603.01879v1
Derived from: arXiv:2603.01879v1 §B.3 full text (PDF in papers/, verified
              complete 2026-08-10). Cross-checked against the geometric estimator
              (Algorithm 2) of arXiv:2503.18114v2. NOT copied from any repo doc.
Transcribed by: agent (Opus 4.8), 2026-08-10
Verified by (human): ____________          # blank until checked
Status: agent-verified against the ICLR 2026 PDF §B.3; NOT human-verified.
        Code may not cite this sheet until "Verified by (human)" is filled.
```

Implemented in `docs/00-math-spec.md` §6.1–§6.2 and §8. This sheet is the
source-of-record extraction; the spec is the implementation target.

---

## The identity

```
α  =  Ψ_eff · (1 + R_eff⁻²) / D_eff
```

Equivalently, with `N_crit` the critical ambient dimension,
`N_crit = P · D_eff / (Ψ_eff · (1 + R_eff⁻²))` and `α = P / N_crit`.

This is an **exact algebraic identity**, not an approximation. The ICML 2025
two-factor form `α ≈ (1 + R⁻²)/D` (Eq. B.7) is its `Ψ_eff → 1` special case; the
third factor `Ψ_eff` is exactly the former unexplained residual.

---

## Anchor points and the stacked matrices

For a Gaussian direction `t ~ N(0, I_N)` and a dichotomy `y ∈ {±1}^P`, solve the
capacity QP; the non-negative dual variables `λ^μ_i` give the anchor point of
class `μ`:

```
s^μ(y,t) = ( Σ_i λ^μ_i z^μ_i ) / ( Σ_i λ^μ_i ),     z^μ_i = i-th point of manifold μ
```

Stack anchors as rows of `S(y,t) ∈ R^{P×N}`; `S_y = diag(y) · S`. Split into
center and axis parts:

```
s^μ_0      = E_{y,t}[ s^μ(y,t) ]              (row-stack → S_0,   S_{y,0} = diag(y) S_0)
s^μ_1(y,t) = s^μ(y,t) − s^μ_0                 (row-stack → S_1,   S_{y,1} = diag(y) S_1)
```

---

## The three scalars (P×P pseudo-inverses)

`†` = Moore–Penrose pseudo-inverse. All Gram matrices below are **P×P**, not N×N.

```
a(y,t) = (S_y     t)ᵀ (S_y     S_yᵀ)†                       (S_y     t)
b(y,t) = (S_{y,1} t)ᵀ (S_{y,1} S_{y,1}ᵀ)†                   (S_{y,1} t)
c(y,t) = (S_{y,1} t)ᵀ (S_{y,0} S_{y,0}ᵀ + S_{y,1} S_{y,1}ᵀ)† (S_{y,1} t)
```

## The measures (expectations over the sampled (y,t))

```
α      = P / E[a]
D_eff  = (1/P) · E[b]
R_eff  = sqrt( E[c] / E[b − c] )
Ψ_eff  = E[c] / E[a]                # "effective utility"; ∈ [0,1] generic only
```

**The `[0,1]` range holds under the label average only** (`00` §6.2, measured
2026-08-12). At fixed `y` — retained and tilted ensembles — `Ψ_eff` can exceed 1,
because `a`'s `(S_y S_yᵀ)†` and `c`'s `(S_{y,0}S_{y,0}ᵀ + S_{y,1}S_{y,1}ᵀ)†` differ by
center–axis cross-terms that vanish under `E_y` and survive at fixed `y`. Fixing `y`
is necessary but not sufficient: the excess appears only as centers **align with the
dichotomy**, and at zero alignment fixed-`y` `Ψ_eff` is 0.856, below the bound. Read
`Ψ_eff > 1` as a measurement of task-specific geometric organization relative to the
random-dichotomy normalization. The identity below closes either way, so `α` and the
attribution are unaffected.

**Check the identity closes:** `(1 + R_eff⁻²) = 1 + E[b−c]/E[c] = E[b]/E[c]`, so
`Ψ_eff · (1 + R_eff⁻²) / D_eff = (E[c]/E[a]) · (E[b]/E[c]) · (P/E[b]) = P/E[a] = α`. ✓

**Attribution (00 §8).** Because the identity is exact, forgetting decomposes
additively in logs with **zero residual**:
`Δ log α = Δ log Ψ_eff + Δ log(1 + R_eff⁻²) − Δ log D_eff`.

---

## Pairwise alignment measures (Def B.6 / §B.3 — convention C3)

**Absolute value, unnormalized, cross-index** (the lab convention of record =
`rho_c_glue`, default):

```
ρ_c(μ,ν) = | ⟨ s^μ_0, s^ν_0 ⟩ |
ρ_a(μ,ν) = E_{y,t}[ | ⟨ s^μ_1(y,t), s^ν_1(y,t) ⟩ | ]
ψ(μ,ν)   = E_{y,t}[ | ⟨ s^μ_0, s^ν_1(y,t) ⟩ | ]
```

Report the mean over pairs `μ ≠ ν`. The signed, normalized `rho_c_signed`
(`⟨s^μ_0,s^ν_0⟩ / (‖s^μ_0‖‖s^ν_0‖)`) is **always** also computed — required for
H1d (Menghi anticorrelation lives in the negative range; under `|·|` H1d is
untestable). Two roles (`03` §E.4): `rho_c_glue` = reporting/comparability;
`rho_c_signed` = H1d instrument. `rho_convention: both` is mandatory. Numbers
under the two conventions are not comparable.

⚠️ **`ψ` (center–axis alignment, a pairwise geometric measure) is a different
quantity from `Ψ_eff` (effective utility, a scalar capacity factor)** despite the
shared Greek letter — see `AGENTS.md` §3.
