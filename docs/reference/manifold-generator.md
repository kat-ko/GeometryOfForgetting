# Reference sheet — synthetic manifold generator

```
Source: Chou, Le, Wang & Chung, "Feature Learning beyond the Lazy-Rich
        Dichotomy", ICML 2025 — arXiv 2503.18114, Appendix D.1.1
        ("Synthetic data generation").
Version: arXiv:2503.18114v2
Transcribed by: agent (Opus 4.8), 2026-07-31
Verified by (human): ____________          # blank until checked
Status: agent-verified against arXiv text; NOT human-verified.
        Code may not cite this sheet until "Verified by (human)" is filled.
```

Implemented in `docs/00-math-spec.md` §1. This sheet is the source-of-record
extraction; the spec is the implementation target.

---

## Isotropic spherical manifolds (D.1.1)

`d` = data dimension; `P` manifolds; size `M`; radius `R`; intrinsic dim `D`.

```
M_i = { u_0^i + R · Σ_{j=1}^{D} s_j^k u_j^i  +  ε v_k }_{k ∈ [M]}

  u_j^i ~ N(0, I_d / d)      axes        (j = 1..D)
  u_0^i ~ N(0, I_d / d)      center
  s_j^k ~ N(0, 1)            coordinates
  v_k   ~ N(0, I_d / d)      noise
  ε     = 1e-2               (paper: ϵ = 10^-2)
```

**Normalization (verbatim, verified):** D.1.1 reads "The pre-scaled points in the
manifolds `{ Σ_{j=1}^{D} s_j^k u_j }_{k∈[M]}` are well-normalized to unit norm."
So each pre-scaled point (the `Σ_j s_j^k u_j^i` term, **before** multiplying by `R`
and adding the center `u_0^i`) is scaled to unit norm. *(Read from arXiv 2503.18114
full text, D.1.1 — earlier truncation resolved.)*

**Test manifolds:** same centers and axes, **resample the noise `v_k`**.

**Isotropic Gaussian manifolds (variant):** some experiments drop the intrinsic
dimension `D` and define manifolds directly (no `Σ_j s_j^k u_j^i` term). Not our
primary generator; recorded for completeness.

---

## Correlated manifolds (D.1.1)

Autoregressive covariance over the manifold index:

```
C = ( ρ^{|i-j|} )_{ij} ∈ R^{P×P},   ρ ∈ [0, 1)   (ρ = ρ_C or ρ_A)
```

- **Centers (ρ_C):** multiply the `P × d` center matrix `M_C` by the Cholesky
  factor of `C_C`.
- **Axes (ρ_A):** for each axis `i = 1..D` independently, multiply the `P × d`
  axis-column matrix `M_A^i` by the Cholesky factor of `C_A^i`.
- **Center–axis (ψ):** scale each center `u_0` by `(1 + ψ · q)`, `q ~ N(0,1)`.

---

## Naming flag — RESOLVED 2026-08-10

In `arXiv:2503.18114v2` the capacity/geometry estimator is **Algorithm 2**
("Estimate manifold capacity and effective geometric measures"); **Algorithm 1**
is the simulated-capacity (`α_sim`) bisection. Verified against the complete PDF
(`results/LOG.md` Step 0). `docs/00-math-spec.md` §6, `docs/03-references.md`, and
`docs/04-spec-corrections-iclr2026.md` are now corrected to Algorithm 2.
Definition B.6 (effective geometric measures) defines ρ_c, ρ_a, ψ.
