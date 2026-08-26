# 00 — Mathematical and Implementation Specification

Every formula here must be implemented exactly as written. Where a formula is
uncertain or must be derived, it is marked **UNCERTAIN** and carries a required
unit test. Do not resolve an uncertainty by guessing.

Notation: `d` input dimension, `N` module width, `P` number of manifolds,
`M` points per manifold, `D` intrinsic manifold dimension, `T` stream length.

---

## 1. Manifold generator

Source: Chou, Le, Wang & Chung (ICML 2025), Appendix D.1.1.

### 1.1 Isotropic spherical manifolds

For each manifold `i ∈ [P]`:

```
M_i = { u_0^i + R · Σ_{j=1}^{D} s_j^k u_j^i  +  ε v_k }_{k ∈ [M]}

  u_j^i ~ N(0, I_d / d)        axes, j = 1..D
  u_0^i ~ N(0, I_d / d)        center
  s_j^k ~ N(0, 1)              coordinates
  v_k   ~ N(0, I_d / d)        noise
  ε     = 1e-2
```

**Required normalization:** the pre-scaled points `{ Σ_{j=1}^{D} s_j^k u_j^i }_{k∈[M]}`
are normalized to unit norm *before* scaling by `R` and adding the center.
Omitting this changes the meaning of `R` and breaks validation §1 of
`02-validation-suite.md`.

Test manifolds use the same centers and axes but resample `v_k`.

### 1.2 Correlated manifolds

Autoregressive covariance over manifold index:

```
C(ρ) = ( ρ^{|i-j|} )_{ij} ∈ R^{P×P},   ρ ∈ [0, 1)
```

- **Center correlation ρ_C:** form the `P × d` matrix of centers, left-multiply by
  the Cholesky factor of `C(ρ_C)`.
- **Axis correlation ρ_A:** for each axis index `j ∈ [D]` independently, form the
  `P × d` matrix of `j`-th axes and left-multiply by the Cholesky factor of `C(ρ_A)`.
- **Center–axis correlation ψ:** scale each center `u_0^i` by `(1 + ψ · q_i)`,
  `q_i ~ N(0,1)`.

### 1.3 Project parameters

| Symbol | Value | Note |
|---|---|---|
| `d` | 150 | input dimension |
| `P` | 16 | = 2^4, four binary factors |
| `M` | 150 | points per manifold |
| `D` | stream variable | intrinsic dimension |
| `R` | stream variable | radius |

---

## 2. Factorial structure and tasks

### 2.1 Labeling

Index the `P = 16` manifolds by four binary factors `(f₁, f₂, f₃, f₄)`,
`f: [P] → {0,1}^4`, bijective.

**Design decision (do not change):** factor structure lives in the *labeling
only*. Manifold centers are drawn i.i.d. and are **not** arranged so factors
correspond to orthogonal directions. This forces abstraction to be learned rather
than inherited, and prevents confounding with the alignment manipulation.

### 2.2 Dichotomy families

A task is a **balanced** dichotomy `y ∈ {±1}^P` with `Σ_i y_i = 0`.

| Family | Definition | Count |
|---|---|---|
| `single_factor` | `y_i = 2·f_j(i) − 1` for some `j` | 4 |
| `xor` | `y_i = 2·(f_j(i) ⊕ f_k(i)) − 1`, `j < k` | 6 |
| `random_balanced` | uniform over balanced dichotomies | C(16,8) = 12870 |

### 2.3 Readout similarity

```
s_r(y, y') = | 2 · overlap(y, y') / P  −  1 |,   overlap = #{i : y_i = y'_i}
```

Sign-symmetric (a dichotomy and its negation are the same task).
Range `[0,1]`; `s_r = 1` means identical or exactly complementary.

**Sampling at target similarity:** to obtain `s_r` corresponding to Hamming
distance `h`, swap `h/2` of the `+1` indices with `h/2` of the `−1` indices.
Hamming distances between balanced dichotomies are always even; `h ∈ {0,2,…,P}`.

### 2.4 Feature similarity

```
s_f(t, t') = ρ_C used when redrawing the arrangement for task t relative to t'
```

Implemented by redrawing centers with the §1.2 procedure at correlation `ρ_C`
against the previous arrangement. `s_f = 1` means the arrangement is unchanged.

---

## 3. Stream construction

### 3.1 Primary design — Hiratani 2×2

Four corners, `{s_f ∈ {low, high}} × {s_r ∈ {low, high}}`.
Predicted catastrophic corner: **high `s_f`, low `s_r`**.

### 3.2 Bookkeeping requirements

- Control `s_f(t, t−1)` and `s_r(t, t−1)` only.
- **Record the full `T × T` similarity matrices `S_f` and `S_r`.** Long-range
  similarity is an uncontrolled random walk; it is used as a covariate in the
  drift regression, not controlled.
- **Held-out probe dichotomy `y*`**: sampled once per stream, never trained on,
  evaluated at every boundary with a freshly trained readout. Must be at
  controlled `s_r` distance from the stream's dichotomies — record it.

### 3.3 Lengths

`T = 16` primary. `T = 40` for the regime half-life measurement (Phase 3).

---

## 4. Network and parameterization

### 4.1 Architecture

```
h_m(x) = ReLU( β₀ · W_m x ),          m ∈ {A, B},  W_m ∈ R^{N×d}
f(x)   = Σ_m (β_L / γ_m) · u_mᵀ ReLU(h_m(x))
```

`β₀ = d^{-1/2}`, `β_L = N^{-1/2}`.

### 4.2 Scaling table

Source: Graldi et al. (ICML 2025), Table 1.

| | NTP | μP (Mean Field) |
|---|---|---|
| Branch scale `β_ℓ` | `N^{-1/2}` (ℓ>0), `d^{-1/2}` (ℓ=0) | same |
| Output scale `γ` | `1` | `γ₀ · N^{1/2}` |
| LR schedule `η(t)` | `η₀(t)` | `η₀(t) · γ₀² · N` |
| Weight variance `σ_ℓ²` | `1` | `1` |

**Critical invariant (I6):** `σ² = 1` in *both* parameterizations. Hidden-layer
weights at initialization are therefore **identical across γ₀**. This is what
makes `γ` and `a` orthogonal. Unit-tested in `02-validation-suite.md` §3.

**Per-module application:** `γ_m` sets both the output scale and the learning
rate `η_m = η₀ · γ_m² · N`, applied to `W_m` **and** `u_m`. Changing one without
the other changes contribution magnitude, not richness.

**UNCERTAIN — must derive and unit-test.** Graldi et al. normalize μP to be
equivalent to NTP at base width `N = 64`, but do not write the constant out.
Derive it, assert `μP(γ₀=1, N=64) ≡ NTP(N=64)` numerically to float64 tolerance,
and record the derivation in `docs/reference/parameterization-derivation.md`.

### 4.3 Corrected-LR arm

Atanasov et al. (ICLR 2025) find `η* ∝ γ²` for `γ ≪ 1` and `η* ∝ γ^(2/L)` for
`γ ≫ 1`. Graldi's `γ²` scaling is therefore misspecified in the rich regime, and
they flag it as a possible cause of their sharp transition.

Implement `lr_scaling ∈ {"quadratic", "corrected"}` as a config field.
`"corrected"` uses `η_m = η₀ · γ_m^(2/L) · N` for `γ_m > 1`, `γ_m²` otherwise,
with `L` the depth (here `L = 2`). Required for the robustness arm; the effect
must survive it.

### 4.4 Readout

`u_m` initialized to **zero**, guaranteeing `f(x; θ₀) = 0` (I5).
Single shared readout across modules. No task-conditioned head (I8).

---

## 5. Alignment initialization (the wealth knob)

Goal: vary the overlap between `W_m(0)`'s row space and the manifold-center
subspace, at **fixed Frobenius norm and fixed rank**, so that `a` is orthogonal
to `γ`.

### 5.1 Procedure

1. Compute the center subspace: SVD of the `P × d` center matrix; take the top-`k`
   right singular vectors `U_C ∈ R^{d×k}` (orthonormal). Use `k = min(P−1, D·2)`;
   record `k` in the config.
2. Draw the shape matrix `W̃ ~ N(0,1)^{N×d}` (one per seed; see paired-init rule).
3. Thin SVD: `W̃ = Q Σ Vᵀ`, with `V ∈ R^{d×min(N,d)}`. Let `V_k` = first `k`
   columns of `V`.
4. Interpolate `V_k → U_C` along the **Grassmann geodesic** at parameter `a ∈ [0,1]`.
5. Reassemble `V(a)` by replacing the first `k` columns of `V` with the
   interpolated basis, then re-orthonormalizing the *remaining* columns against
   them (Gram–Schmidt, preserving order).
6. `W_m(0) = Q Σ V(a)ᵀ`. Singular values `Σ` unchanged ⇒ Frobenius norm and rank
   preserved.

### 5.2 Grassmann geodesic

Given orthonormal `Y, U ∈ R^{d×k}` (from `V_k` and `U_C`):

```
Compute  A = (I − Y Yᵀ) U (Yᵀ U)^{-1}
Thin SVD: A = Q_A Σ_A Z_Aᵀ
Principal angles: Θ = arctan(Σ_A)        (diagonal)
Geodesic:  Y(a) = Y Z_A cos(aΘ) + Q_A sin(aΘ)
```

`Y(0) = Y Z_A` (same subspace as `Y`), `Y(1)` spans `U`.

**Do not** use linear interpolation followed by re-orthonormalization (I:
Grassmann rule in AGENTS.md). It makes `a` a nonlinear, seed-dependent
parameterization and destroys comparability.

Guard: `(Yᵀ U)` may be near-singular if the subspaces are nearly orthogonal.
Use an explicit pseudo-inverse with logged rcond, and assert principal angles are
finite.

### 5.3 Fallback

**Gate 2 is open**: it is not established that `a` moves capacity-at-init
monotonically with adequate dynamic range. If validation §4 fails, fall back to
Chung's manipulation — varying input manifold intrinsic dimension `D` to control
initial wealth — accepting loss of orthogonality with `γ`. Implement behind
`wealth_knob ∈ {"alignment", "input_dim"}`.

---

## 6. GLUE estimator

Source: Chou, Le, Wang & Chung (ICML 2025), **Algorithm 2** ("Estimate manifold
capacity and effective geometric measures") and Definition B.6. The exact
three-factor decomposition and the `a/b/c` scalar forms below are from Chou,
Kirsanov, Yang & Chung, "Diagnosing Generalization Failures from Representational
Geometry Markers" (ICLR 2026, `arXiv:2603.01879v1`), Appendix B.3 — see
`docs/reference/glue-decomposition.md` (verified 2026-08-10).

Reference implementation for the subset {capacity, R, D, ρ_c}:
`schung039/neural_manifolds_replicaMFT`, function `manifold_analysis_corr(X, kappa, n_t)`,
data shaped `(num_features, num_samples)` per manifold, default `n_t = 200`.
(NB: **Algorithm 1** in the ICML paper is the *simulated* capacity `α_sim`
bisection — a different estimator, used in `02` §2a. Do not conflate.)

**We operate at margin κ = 0.** The legacy estimator's κ parameter must be set to
0; estimators are not interchangeable at κ ≠ 0.

### 6.1 Estimator in stacked-matrix (P-space) form

Input: `{M_μ}_{μ∈[P]}`, each `M_μ` a set of `M` points in `R^N`; `n_t` samples.
`†` = pseudo-inverse with explicit rcond, logged effective rank.

**Anchor points.** For a Gaussian direction `t ~ N(0, I_N)` and dichotomy `y`,
solve the capacity QP and read the non-negative dual variables `λ^μ_i`. The
anchor point of class `μ` is
```
s^μ(y,t) = ( Σ_i λ^μ_i(y,t) · z^μ_i ) / ( Σ_i λ^μ_i(y,t) ),   z^μ_i = i-th point of M_μ
```
Stack the anchors as **rows** of `S(y,t) ∈ R^{P×N}` and set `S_y = diag(y) · S`.

**Center / axis split.**
```
s^μ_0        = E_{y,t}[ s^μ(y,t) ]          (anchor center of class μ)
s^μ_1(y,t)   = s^μ(y,t) − s^μ_0             (axis component)
```
with row-stacked `S_0`, `S_1(y,t)` and `S_{y,0} = diag(y) S_0`,
`S_{y,1}(y,t) = diag(y) S_1(y,t)`.

**Three scalars — all pseudo-inverses are `P×P` (here 16×16), not `N×N`.**
```
a(y,t) = (S_y      t)ᵀ (S_y      S_yᵀ)†      (S_y      t)
b(y,t) = (S_{y,1}  t)ᵀ (S_{y,1}  S_{y,1}ᵀ)†  (S_{y,1}  t)
c(y,t) = (S_{y,1}  t)ᵀ (S_{y,0} S_{y,0}ᵀ + S_{y,1} S_{y,1}ᵀ)† (S_{y,1} t)
```
The dominant cost is the QP for the anchor points, **not** this linear algebra
(see §13).

**Measures** (expectations over the `n_t` samples of `(y, t)`):
```
α       = P / E[a]
D_eff   = (1/P) · E[b]
R_eff   = sqrt( E[c] / E[b − c] )
Ψ_eff   = E[c] / E[a]                 # "effective utility"; ∈ [0,1] generic only
```
**The `Ψ_eff ∈ [0,1]` range is a property of the label average, not of the
estimator — measured 2026-08-12.** It holds for the generic ensemble, where the
expectation runs over random dichotomies, and that is the form the source papers
state. Under a **fixed-`y` (retained or tilted) ensemble it can and does exceed 1**:
the grid puts 21.6% of retained measurements above 1, to a maximum of 1.78.

**The mechanism is center alignment with the dichotomy, and fixing `y` is necessary
but not sufficient.** The two pseudo-inverses differ by center–axis cross-terms —
`a` uses `(S_y S_yᵀ)†`, `c` uses `(S_{y,0}S_{y,0}ᵀ + S_{y,1}S_{y,1}ᵀ)†` — and those
terms vanish under `E_y` while surviving at fixed `y`. But surviving is not the same
as being large: what makes them large is the arrangement actually being *organized
for* that labeling. You cannot align with a dichotomy you are averaging over, so
fixing `y` is the necessary condition; the sufficient condition is geometric.

Demonstrated on ground truth in `scripts/run_psi_bound_check.py --synthetic`, which
holds the arrangement fixed and varies only how far centers are separated *along* `y`:

| centers sep / radius | generic `Ψ_eff` | fixed-`y` `Ψ_eff` |
|---|---|---|
| 0.0 | 0.4709 | **0.8559** |
| 0.5 | 0.4637 | 1.1638 |
| 1.0 | 0.4908 | 1.7312 |
| 2.0 | 0.4939 | 2.3876 |
| 4.0 | 0.5030 | **3.6555** |

Generic `Ψ_eff` never leaves the bound; fixed-`y` `Ψ_eff` crosses 1 only once centers
align, and at zero alignment **sits below 1**. Identity closes at 1e-16 in every row.
So `Ψ_eff > 1` is not an artifact of the tilt being permitted — it is a *measurement
of task-specific geometric organization*, with a monotone dose–response on ground
truth.

Read fixed-`y` `Ψ_eff > 1` as **task-specific utility relative to the
random-dichotomy normalization**, not as an error. `α` is unaffected — `α_sim` at
fixed `y` agrees with the GLUE `α` to 1.5% on a real γ=10 arm with `Ψ_eff = 1.49`
(`results/psi_bound_check.json`), and the three-factor identity closes regardless.
This is a **known divergence from the range stated in the source papers**, on our
empirical basis; the papers do not discuss the fixed-`y` case.
**Aggregation rule (pinned).** Wherever an estimator returns a *per-manifold*
capacity vector `α_i` (replicaMFT does), reduce with the **harmonic mean**
`α = 1 / mean(1/α_i)` — **never** the arithmetic mean. Rationale: `α = P/N_crit`
and critical dimensions add across manifolds.

**Implementation conventions, settled by measurement 2026-08-11**
(`docs/reference/glue-core-validation.md`; all numbers reproducible from
`scripts/run_*.py`):

- **Joint QP, not per-manifold.** Anchors come from one QP over all `P·M`
  constraints. `diag(y)` cancels in `a/b/c`, so a per-manifold construction would
  be label-invariant — i.e. would silently rebuild `α_mf`'s limitation.
- **Inactive manifolds contribute a zero row.** Pinned by the exact point-manifold
  limit: this reproduces `α = 2` at `κ = 0` (measured 2.00 ± 0.12).
- **`center_policy = "all"`** for `s⁰_μ = E[s^μ(y,t)]` — the literal §B.3 reading.
  Chosen on evidence: `"active"` is marginally better on `D`/`R` recovery (< 1 pp)
  but more than 2× worse on `ρ_C` recovery (8.6% vs 3.7%), and it systematically
  over-estimates `ρ_c`, which is H1d's instrument. Recorded in every result key.
- **Independent `t` and `y` RNG streams**, so comparisons across `β` are paired.
- **Known bias — scale compression.** `D_eff` and `R_eff` over-report below
  `D ≈ 4` / `R ≈ 1` and under-report above (18% low at `D = 10`, 11% at `R = 2`).
  Mechanism measured: finite sampling of each manifold — the deficit shrinks as
  `M^-0.45` and is independent of `n_t` and of `P`. It is monotone, so signs and
  rank orders are preserved, and it biases `Δ log D_eff` **toward zero**, making
  the §8 dimension channel conservative. Absolute `D_eff` above 4 is a lower
  bound and must be reported as such.

#### What `replicaMFT` actually provides — read before using it

Verified 2026-08-11 by reading `mftma/manifold_analysis_correlation.py`. These
limits are **structural** (properties of the replica theory GLUE refines), not
implementation gaps:

- **`α_mf` is label-invariant by construction — generic capacity only.**
  `manifold_analysis_corr(XtotT, kappa, n_t)` takes **no `y`** and analyses each
  manifold **independently**: the dichotomy average is integrated out
  *analytically* in the replica derivation, so there is no `y` to pass.
  **Retained and tilted capacity are not extractable from it at any effort.**
  This is exactly GLUE's relaxed assumption A2 (`docs/reference/glue-refinements.md`).
- **`res_coeff0` is NOT the anchor-based ρ_c.** It is the mean over `μ≠ν` of the
  **absolute cosine** between *global-mean-subtracted **manifold** centers* (raw
  class means) in a `P−1` basis — abs **and** normalized, on manifold centers,
  not anchor centers `s⁰_μ`. It matches **neither** `rho_c_glue` (abs,
  unnormalized) **nor** `rho_c_signed`. Record it as `center_cos_abs`
  (replicaMFT-specific); **do not report it as ρ_c**.
- **`R_M`/`D_M` are the PRX-2018 eq-28/29 functionals**, not the `R_eff`/`D_eff`
  above. The derived `Ψ_eff = α·D_eff/(1+R_eff⁻²)` is therefore **expected to
  fail** against replicaMFT outputs; `02` §2c is reclassified gate → **diagnostic**.
- **Anchor centers are formed then discarded**, in a *per-manifold* frame
  (per-manifold centering, norm division, and a per-manifold QR re-basis when
  `N > M`), so cross-manifold `⟨s⁰_μ, s⁰_ν⟩` is not recoverable from its returns.

**Consequence.** The three-factor decomposition is computed by **our own GLUE
core** (`src/glue/core.py`, estimator string `glue_core@<sha>`), which builds
anchor points in the **ambient** frame under a shared `y` per §6.1. `replicaMFT`
is retained as a **cross-check on generic α only**, under the §9 pooling firewall.

**Pairwise alignment measures (Def B.6 / ICLR 2026 §B.3 — absolute value,
unnormalized, cross-index; convention C3):**
```
ρ_c(μ,ν) = | ⟨ s^μ_0, s^ν_0 ⟩ |
ρ_a(μ,ν) = E_{y,t}[ | ⟨ s^μ_1(y,t), s^ν_1(y,t) ⟩ | ]
ψ(μ,ν)   = E_{y,t}[ | ⟨ s^μ_0, s^ν_1(y,t) ⟩ | ]
```
Report the mean over pairs `μ ≠ ν`. These need the anchor centers/axes, which
`replicaMFT` does not expose (`ρ_a`, `ψ` are GLUE-only for now; exposing them
touches vendored QP internals — VERIFY-FIRST, `05` §2.2, Kati sign-off).

**H1d instrument — signed ρ_c.** Alongside the lab-convention `rho_c_glue` above,
also compute
```
rho_c_signed(μ,ν) = ⟨ s^μ_0, s^ν_0 ⟩ / (‖s^μ_0‖ ‖s^ν_0‖)
```
Menghi's anticorrelation result lives in the **negative** range, which `|·|`
erases — so the signed form is the **required H1d instrument** (under abs alone,
H1d is untestable). **"Primary" is two roles, not one verdict** (`03` §E.4):
`rho_c_glue` = primary for reporting / comparability with published GLUE;
`rho_c_signed` = primary for the H1d hypothesis test. `rho_convention: both` is
**mandatory** — both are always computed; **no code path may reduce it to one.**
Numbers under the two conventions are **not** comparable; state this in the paper.

**Flag-and-stop:** if anchor centers `s⁰_μ` are not exposable from
`manifold_analysis_corr` without a vendored QP patch, stop and report — do **not**
degrade to abs-only (`05` §2.2 / `03` §E.4).

### 6.2 Capacity decomposition — exact, three factors

```
α = Ψ_eff · (1 + R_eff⁻²) / D_eff        # EXACT identity, not an approximation
```

Substituting §6.1: `(1 + R_eff⁻²) = E[b]/E[c]`, so
`Ψ_eff·(1+R_eff⁻²)/D_eff = (E[c]/E[a])·(E[b]/E[c])·(P/E[b]) = P/E[a] = α`.
Source: ICLR 2026 §B.3 (verified 2026-08-10). The earlier two-factor
`(1 + R⁻²)/D` was the ICML *approximation* (Eq. B.7); the third factor `Ψ_eff` is
exactly what used to be the unexplained "residual". **There is no approximation
term to test** — `02` §2b is deleted; the sole remaining capacity gate is `02`
§2a (`α_sim` vs `α_mf`, mean-field validity).

### 6.3 Cross-module comparability

Capacity is not invariant to anisotropic rescaling, and modules differ in scale
by construction. Every capacity/GLUE call must be run twice and both recorded:

- `raw`: representations as-is
- `gaussianized`: **Gaussianization preprocessing (Wakhloo et al., 2023)** applied
  per manifold before estimation, following the lab's own pipeline (subsample to
  50 points per manifold, pairwise GLUE, Gaussianize to ensure initial linear
  separability). Extract the procedure into `docs/reference/correlation-duality.md`.

Report `raw` as primary; `gaussianized` must agree qualitatively. Divergence is a
finding, not an error, and must be surfaced rather than silently resolved.

---

## 7. Generic, retained, and tilted capacity

The ensemble is the analyst's choice of dichotomy collection `Y`; capacity is the
expectation over `(y ~ Y, t ~ N(0, I_N))` (ICLR 2026 §B.3). Fixing `|Y| = 1`
retains the `t`-expectation and is a **legitimate** analyst choice — the earlier
"fixed-`y` exits the definition" claim was too strong (04 §C4).

**Generic capacity** `α_generic(m, t)`: `y` drawn uniformly over balanced
dichotomies (the `β = 0` ensemble; as in Algorithm 2 Step 1). Default,
**label-agnostic**.

**Retained capacity** `α_retained(m, t; y_j)`: `y` fixed at past task `j`'s
dichotomy for all `n_t` samples (`t` still varies) — the `β → ∞` limit.
Implement as an explicit `fixed_dichotomy: Optional[np.ndarray]` argument, not by
hacking the sampler. This is the primary instrument for §8 attribution.
**Stage one (D2): build now.**

**Tilted capacity** `tilted_capacity(m, t; y_j, β)`: `y` sampled from the tilted
distribution `P(y) ∝ exp(β⟨y, y_j⟩)` over balanced dichotomies. At `P = 16`,
`C(16,8) = 12870` — exact enumeration with weights `∝ exp(β⟨y, y_j⟩)` is feasible;
prefer it. `β = 0 →` generic, `β → ∞ →` retained. **Stage two (D2): build the
interface now, wire after the β-validity sweep** (`02` §2a, `05` §3.4) confirms
the usable β range. Precedent: Montanari et al. (2019), invoked by Chou et al.
App. A.1.

(Terminology: the term "aligned capacity/margin" is **retired**; use
`retained_capacity` for the fixed-`y_j` quantity — see `AGENTS.md` §3.)

**Estimator routing (pinned 2026-08-11).** `y` is not an optional argument that
some estimators happen to lack — it is what separates the two theories:

| Quantity | `α_sim` | `α_core` (`src/glue/core.py`) | `α_mf` (`replicaMFT`) |
|---|---|---|---|
| generic (β = 0) | yes | yes | yes (cross-check only) |
| retained (β = ∞) | yes | yes | **impossible** |
| tilted (0 < β < ∞) | yes | yes | **impossible** |
| three-factor `Ψ_eff, R_eff, D_eff` | no | yes | **no** (different functionals) |

`α_mf` is label-invariant *by construction* (§6.1): the replica derivation
integrates the dichotomy average out analytically, so no amount of adapter work
recovers a `y`-dependent number from it. Every retained and tilted quantity in
this project routes through `α_sim` (ground truth) and `α_core` (decomposition).

**Mandatory label-aware validation.** Generic capacity is a label-agnostic
structural metric, and the plasticity literature has theoretical counterexamples
showing such metrics can look favourable where gradient descent cannot progress.
Generic capacity must therefore be validated against the **held-out probe
metric**: train a fresh linear readout on `y*` from module `m`'s representation at
boundary `t`, measure generalization following Johnston & Fusi's
classifier-generalization protocol. If `α_generic` does not track this, that is a
reportable negative result — do not suppress it.

---

## 8. Attribution of forgetting — exact log-space decomposition

For each past task `j`, module `m`, boundary `t`, using **retained capacity**
`α_retained(m, ·; y_j)` and its three factors (§6). Let `Δ` denote the change
from boundary `t_j` (immediately after training task `j`) to boundary `t`.

Because `α = Ψ_eff · (1 + R_eff⁻²) / D_eff` is **exact** (§6.2), attribution is
additive in log-space with **zero residual**:

```
Δ log α  =  Δ log Ψ_eff  +  Δ log(1 + R_eff⁻²)  −  Δ log D_eff
```

Report the three component contributions per (task `j` × boundary `t` × module
`m`). No residual term and no finite-difference cross-terms — the identity closes
exactly. (If `05` §3.3 fails to validate the derived `Ψ_eff`, fall back to a
two-factor `(R_eff, D_eff)` decomposition with `Ψ_eff` reported as
unvalidated-derived — a caption change, per `05` contingency table.)

**Interpretation constraint (Wakhloo duality).** Wakhloo, Sussman & Chung (PRL
2023) establish that centroid correlations are effectively equivalent to reduced
center separation, and axis correlations to shrunk radii. The components are
therefore **not independent**. Attribution is a *decomposition*, never a causal
mechanism, and all outputs and figure captions must say so. Always report `ρ_a`
and `R` jointly.

---

## 9. Drift decomposition

Fixed probe set `X_probe`, evaluated per module at every boundary.
`S_m(t)` = top-`k` principal subspace of `h_m(X_probe; t)`, `k` fixed by config
(not by a variance threshold — a threshold makes `k` time-varying and the
principal angles incomparable).

| Mode | Estimator |
|---|---|
| **Rotation** | mean `sin²θ_i` over principal angles between `S_m(t)` and `S_m(t−1)` |
| **Expansion** | `ΔPR = PR(t) − PR(t−1)`, participation ratio |
| **Reuse** | for incoming task `t` with discriminative direction `Δ_t` (difference of class-conditional means of manifold centers, computed in the representation at `t−1`): `‖Proj_{S_m(t−1)} Δ_t‖² / ‖Δ_t‖²` |
| **Overwrite** | rotation of the subspace spanned by past task `j`'s manifold centers, paired with the drop in linear decodability of `y_j` |

Participation ratio: `PR = (Σλ_i)² / Σλ_i²` over the covariance eigenvalues of
`h_m(X_probe)`.

---

## 10. Other measures

- **Linear CKA only.** Never RBF-CKA — the kernel width interacts with `γ`.
- **CCGP**: train a linear decoder on a dichotomy using a subset of conditions,
  test on held-out conditions, following Bernardi et al. (2020). Report per
  dichotomy family (`single_factor`, `xor`, `random_balanced`).
- **Shattering dimensionality**: fraction of balanced dichotomies linearly
  separable at fixed decoder capacity. Subsample dichotomies; record how many.
- **Kernel–target alignment**: `CKA(K_t^NTK, y yᵀ)`. Required alongside capacity
  for the novelty-positioning check (see `03-references.md` §Gate 3).

---

## 11. Manipulation checks (logged unconditionally, every boundary)

| Quantity | Purpose |
|---|---|
| `‖ΔW_m(t)‖_F / ‖W_m(t−1)‖_F` | did the γ manipulation hold? Must separate by ≳1 order of magnitude across γ |
| per-module NTK change | independent richness check |
| per-module output-variance share | gradient starvation detector |
| per-module gradient-norm share | gradient starvation detector |
| `α_generic` at init | wealth manipulation check; must track `a`, must be flat in `γ` |

---

## 12. Time-reparameterization test (Phase 0 gate)

Atanasov et al. find that networks of different **large** γ optimize along
similar trajectories up to a reparameterization of time. If true here, two
modules with different large γ are the same trajectory at different speeds, and
the division-of-labour hypothesis is void.

Test: for two homogeneous runs at `γ₁ ≠ γ₂` (both `≳ 1`), fit a monotone time
warp `τ` minimizing the distance between geometry trajectories
`(α, R, D, ρ_c)(γ₁, t)` and `(α, R, D, ρ_c)(γ₂, τ(t))`. If residual after warping
is within the trajectory noise floor, the trajectories coincide.

**Consequence if they coincide:** restrict all heterogeneity contrasts to
`γ ≪ 1` vs `γ ~ 1`, and record that H3 is not testable in the large-γ range.

**RAN 2026-08-11 — they do not coincide. H3 is testable; the restriction does not
apply.** `results/timewarp.json`, `src/analysis/timewarp.py`. Implementation notes
that matter for interpreting the number:

- Residuals are expressed in **units of the per-channel noise floor**, so 1.0 is
  the decision boundary and the value is directly readable.
- Three warps of increasing generosity: the **measured** steps-to-target-loss ratio
  (no free parameters), the **best single rate** (one parameter), and **DTW** (the
  most generous monotone warp; fixed start — justified because paired init gives
  every γ identical hidden weights and a zero readout, so all trajectories begin at
  the same geometry — and free end, since a faster run has travelled further by the
  last checkpoint). Coincidence requires only one warp to succeed; H3 survives only
  if none do.
- **Vacuity guard.** If a trajectory's total excursion is under ~3 noise floors it
  is reported vacuous, not coincident. A static trajectory is trivially the slowed
  opening of any other, so alignment carries no information about shape. This
  applies to γ = 0.03, whose geometry moves 0.2 floors.

Result: γ=1 vs 10 and γ=3 vs 10 differ by **11.5** and **7.1** floors; γ=1 vs 3 is
marginal at 1.05. Use well-separated γ with γ = 10 as one endpoint.

---

## 13. Cost model

GLUE requires one QP per `(y, t)` sample. At `P=16, M=150, N=300`, the QP
constraint matrix `G` is `2400 × 300` per sample; `n_t = 200` default. **The QP
dominates.** The post-QP linear algebra (§6.1 `a/b/c`) is over `P×P` (16×16)
pseudo-inverses, not `N×N` — negligible by comparison. The earlier N×N framing
overstated it; the Tier-2 budget is likely less constraining than assumed, which
may free the seed count (re-check at the Day-13 cost measurement).

The lab's own pipeline is **pairwise** (subsample 2 manifolds × 50 points, GLUE
per pair, ~100 repetitions; ICLR 2026 §B.3 / 04 §C6). Phase 0 compares
`estimation_mode ∈ {pairwise, full_P}`; pairwise QPs are far cheaper (`G` is
`100 × N`) and give comparability with published numbers.

Total Tier-2 evaluations ≈ `n_configs × n_seeds × n_streams × ⌈T/tier2_interval⌉ × n_modules × (1 + n_past_tasks)`.

**Required before Phase 1:** time one full GLUE evaluation at target parameters
(both `estimation_mode`s), multiply out the planned grid, write
`results/cost_model.json`, and fix `n_t` and `tier2_interval` from it. Do not tune
these mid-experiment.

**Cut order under budget pressure:** `n_t` → `tier2_interval` → number of past
tasks evaluated → **never seeds, never streams**.
