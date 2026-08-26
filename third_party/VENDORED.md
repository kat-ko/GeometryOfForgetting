# Vendored estimators

Pinned third-party capacity/geometry code. Vendored (not pip-installed) so we can
read it, verify it against `docs/00-math-spec.md`, and patch it for float64 /
explicit rcond / RNG isolation. See `docs/03-references.md` §D.

`.git` histories were stripped after cloning; the pinned SHA below is the
provenance of record. Re-clone from the URL at that SHA to reproduce.

**Fetched:** 2026-07-31 (both repos), default branch, `git clone` then SHA pinned.

## Corrections policy — do NOT edit `third_party/` in place

`third_party/` is kept **byte-identical to upstream** so the SHA↔directory
correspondence stays meaningful. All corrections (RNG-Generator injection to remove
global `np.random.seed`, float64 enforcement, explicit pseudo-inverse `rcond` with
logged rank) live in a thin **`src/glue/adapters/`** layer that wraps these
functions — see `src/glue/adapters/README.md`. If an in-place edit is ever
unavoidable, record it as a `.patch` under **`third_party/patches/`** and reference
it here; never silently mutate vendored source.

---

## Estimator version strings (for the results key)

Every geometry result records which estimator produced it as `{repo}@{sha}`.
The `docs/02-validation-suite.md` §9 pooling firewall applies **pairwise across
all three** — numbers from different estimators must never be pooled.

| Version string | Role |
|---|---|
| `correlated_capacity@de8dac79760e8eb552c036604503efcb87e1e6b6` | Ground-truth `α_sim` at **any** `y`. **Not GLUE.** |
| `glue_core@<our-sha>` | **Ours** (`src/glue/core.py`, not vendored): α, D_eff, R_eff, Ψ_eff, anchor ρ_c both conventions |
| `replicaMFT@d56eda1d86a71f4f4601ef1acc3254e20259a80c` | **Downgraded 2026-08-11** to cross-check on **generic α only** — see capability limit below |
| `GLUE@<pending>` | Full GLUE incl. ρ_a, ψ — pending early-access (form in §GLUE below) |

`glue_core` is ours but is versioned and firewalled like a third-party estimator:
it is a distinct estimator and its numbers must never be pooled with `α_mf` or
`α_sim`.

---

## replicaMFT — `schung039/neural_manifolds_replicaMFT`

- **URL:** https://github.com/schung039/neural_manifolds_replicaMFT
- **Fetched:** 2026-07-31
- **Pinned SHA:** `d56eda1d86a71f4f4601ef1acc3254e20259a80c`
- **Entry point:** `mftma/manifold_analysis_correlation.py::manifold_analysis_corr(XtotT, kappa, n_t, t_vecs=None, n_reps=10)`
- **Returns (verified by reading source):** `(a_Mfull_vec, R_M_vec, D_M_vec, res_coeff0, KK)`.

### CAPABILITY LIMIT — verified finding, 2026-08-11

Established by reading `mftma/manifold_analysis_correlation.py` end to end. These
are **structural properties of the replica mean-field theory** (Chung/Cohen, PRX
2018 / Nat Comms 2020) that GLUE was written to refine — **not** implementation
gaps, and **not** fixable by an adapter, a patch, or any amount of effort.

1. **Label-invariant by construction — generic capacity only.**
   `manifold_analysis_corr(XtotT, kappa, n_t)` has **no `y` argument** and analyses
   each manifold independently. The dichotomy average is integrated out
   *analytically* when deriving the self-consistent equations, so `y` was
   eliminated before the code existed. GLUE instead samples `(y, t)` and averages
   *numerically*, which is why GLUE supports an analyst-chosen ensemble `Y` and
   this does not. **Retained and tilted capacity are not extractable at any
   effort.** This is GLUE's relaxed assumption A2 seen in code.
2. **`res_coeff0` is not ρ_c.** It is the mean over `μ≠ν` of the **absolute
   cosine** between *global-mean-subtracted manifold centers* (raw class means) in
   a `P−1` basis. Abs **and** normalized, on manifold centers not anchor centers —
   it matches **neither** `rho_c_glue` (abs, unnormalized) nor `rho_c_signed`.
   Record as `center_cos_abs`; **never report it as ρ_c**. (`00` §6.1 corrected.)
3. **`R_M`/`D_M` are the PRX-2018 eq-28/29 functionals**, not
   `R_eff = √(E[c]/E[b−c])` and `D_eff = E[b]/P`. The `Ψ_eff` identity is expected
   to **fail** against these outputs — `02` §2c is now a diagnostic, not a gate.
4. **Anchor centers are computed then discarded**, in a per-manifold frame
   (centering, norm division, per-manifold QR re-basis when `N > M`), so
   cross-manifold `⟨s⁰_μ, s⁰_ν⟩` is unrecoverable from the return values.
5. **`α` is returned per manifold.** Reduce with the **harmonic mean**
   `1/mean(1/α_i)` — never arithmetic.

**Consequence:** downgraded from primary estimator to *cross-check on generic α
only*. Everything label-dependent routes through `α_sim` and our `src/glue/core.py`.
Useful residue: `maxproj` and `minimize_vt_sq` are public module-level functions
and remain usable as building blocks — calling them is not a modification.

- **Provides:** generic per-manifold α (+ PRX R_M/D_M, `center_cos_abs`).
  **Missing:** any `y`-dependence, anchor-based ρ_c, ρ_a, ψ.
- **Use at κ = 0** (`docs/00-math-spec.md` §6; estimators are not interchangeable at κ ≠ 0).
- **Patch-needs before use:** no explicit `rcond` on pseudo-inverses and no float64
  enforcement observed — apply per `AGENTS.md` §4 and log effective rank.

## correlated_capacity — `awakhloo/correlated_capacity`

- **URL:** https://github.com/awakhloo/correlated_capacity  (= Zenodo v1.0.0, doi 10.5281/zenodo.7844169)
- **Fetched:** 2026-07-31
- **Pinned SHA:** `de8dac79760e8eb552c036604503efcb87e1e6b6`
- **Reproduces:** Wakhloo, Sussman & Chung, PRL 131, 027301 (2023).
- **Provides:**
  - `capacity/manifold_simcap_analysis.py::manifold_simcap_analysis(XtotT, n_rep, seed)`
    — **general point-cloud simulation capacity α_sim** (SVM separability via cvxopt QP
    at κ=0, bisection on feature dim). Input is a sequence of `(N, P_i)` arrays.
  - `capacity/sphere_sim_capacity.py::sphere_simcap(sphere_axes, ...)` — sphere-parameterized
    α_sim (analytic axes + internal surface sampling; iterative constraint augmentation).
  - `capacity/replica_correlations.py` — correlated replica mean-field capacity (the duality).
  - `capacity/mean_field_cap.py`, `low_rank_capacity.py` — base MFT and low-rank approximation.
- **Amendment 3 finding (verified by reading source):** `manifold_simcap_analysis` handles
  **general point clouds**, not only spheres. Therefore α_sim **can** serve as the
  `docs/02-validation-suite.md` §2a gate (α_sim vs α_mf) at project parameters.
  `sphere_sim_capacity` is the spherical special case, sufficient for the §1
  ground-truth-recovery tests but not required for §2a.
- **NOT a GLUE source:** it does not emit ρ_a or ψ. Distinct estimator; keep the §9
  pooling firewall.
- **Patch-needs before use:** `manifold_simcap_analysis.py:173` and
  `sphere_sim_capacity.py:194` call global `np.random.seed(...)` — **violates
  `AGENTS.md` §4** (explicit `numpy.random.Generator`, no global seeding). Wrap for
  RNG isolation before use. Enforce float64 / explicit rcond as above.

---

## GLUE — full estimator (ρ_a, ψ), pending

- **Code is NOT public.** `chung-neuroai-lab/feature-learning-geometry` (the ICML 2025
  figure-repro repo) *calls* GLUE; it does not contain Algorithm 1, ρ_a, or ψ (verified
  by reading its README + file tree).
- **Early-access request form:**
  https://docs.google.com/forms/d/e/1FAIpQLSc_IHUkc2zlJv0DIhSL_tiyD7Ty4nCeFdW0U7s-hCVWchefBg/viewform
- Until granted: ρ_a and ψ remain the open dependency; do not fabricate them.
