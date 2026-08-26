# 03 — Reference Material: What to Add to the Repo and Why

---

## Principle: extract, don't dump

**Do not put PDFs in the repo context.** A 40-page paper eats context, and the
agent will confabulate around the parts it half-read. The failure mode is
specific and predictable: it will reproduce a formula from a *related* paper with
different conventions and the code will run.

Instead, for each paper: extract the specific equations, tables, and parameter
values the code needs into a short verified sheet in `docs/reference/`, with a
citation and a page/equation number. Then the agent reads a one-page sheet
instead of a paper, and you can check the transcription once.

**Rule:** if a number appears in `src/`, it appears in `docs/reference/` with a
source. No magic constants.

**Rule (added 2026-07-31): every sheet pins the exact version it was extracted
from.** The provenance header carries a `Version:` field — e.g.
`arXiv:2503.18114v2`. This is not pedantry: a sheet keyed to the wrong version
drifts silently.

**Algorithm numbering — resolved 2026-08-10 (verified against the complete
`arXiv:2503.18114v2` PDF; see `results/LOG.md` Step 0).** In v2, **Algorithm 1 =
"Estimate simulated manifold capacity"** (the `α_sim` bisection) and **Algorithm 2
= "Estimate manifold capacity and effective geometric measures"** (the geometric
estimator: α, R, D, ρ_c, ρ_a, ψ). Earlier docs mis-cited the geometric estimator
as "Algorithm 1"; that is corrected throughout this file and `00`/`04`.

PDFs themselves go in `papers/` (gitignored, or git-lfs) for *human* reading. The
agent should not be pointed at them.

---

## A. Full text genuinely needed

These **six** you should have and read closely, because the extraction is
non-trivial and you'll need to consult them repeatedly.

| Paper | What you need | Extract to |
|---|---|---|
| **Chou, Le, Wang & Chung** — *Feature Learning beyond the Lazy-Rich Dichotomy: Insights from Representational Geometry*, ICML 2025 (spotlight). **Pin `arXiv:2503.18114v2`** (11 Jul 2025, post-acceptance). | **Algorithm 2** (the geometric estimator — verified vs v2 PDF 2026-08-10; Algorithm 1 is the α_sim bisection), Definition B.6, Definition 2.3, Appendix B.4/Fig 3c (sign conventions), Appendix B.5 (validation settings), Appendix D.1.1 (generator), Figure 7c (ultra-rich OOD signature), Eq. B.4/B.7. ⚠️ **The paper is internally inconsistent** — see §E.4. | `glue-algorithm.md`, `glue-sign-conventions.md`, `manifold-generator.md`, `validation-settings.md` |
| **Chou, Kirsanov, Yang & Chung** — *Diagnosing Generalization Failures from Representational Geometry Markers*, ICLR 2026. **Pin `arXiv:2603.01879v1`**. | **Appendix B.3**: the *exact* three-factor identity `α = Ψ_eff·(1+R_eff⁻²)/D_eff`, the `a/b/c` scalar forms, `D_eff`/`R_eff`/`Ψ_eff` definitions, and the ρ_c/ρ_a/ψ conventions (abs, unnormalized, cross-index). Verified 2026-08-10. Closest competitor to RQ2 (§C7 of `04`). | `glue-decomposition.md` |
| **Chou, Kim, Arend, Yang, Mensh, Slatton, Wakhloo, Shim, Perich & Chung** — *Geometry linked to untangling efficiency reveals structure and computation in neural populations*, **bioRxiv 2025**, doi 10.1101/2024.02.26.582157. Open access. | GLUE is the **parent** method; the ICML paper *applies* it. GLUE refined assumptions relative to **prior replica mean-field capacity theory — i.e. exactly what `replicaMFT` implements**. You need *which* assumptions changed (reported: (A1) Gaussian correlations between manifolds, (A2) random task labels). **Supplementary S1** holds the closed-form capacity formula and the relaxed-assumption ρ_a/ψ definitions — not yet extracted. | `glue-refinements.md` |
| **Graldi, Breccia, Lanzillotta, Hofmann & Noci** — *The Importance of Being Lazy: Scaling Limits of Continual Learning*, ICML 2025. `arXiv:2506.16884`. | Table 1 (the parameterization — **transcribe exactly**), App. A.3 (base-width normalization, N₀=64), metric definitions A.1–A.10, App. A.4 (LR values, epochs, optimizer) | `parameterization.md`, `cl-metrics.md` |
| **Wakhloo, Sussman & Chung** — *Linear Classification of Neural Manifolds with Correlated Variability*, Phys. Rev. Lett. 2023 **+ Supplementary** | Claim 1 (capacity under arbitrary correlations); the centroid↔distance / axis↔radius **duality**. Constrains what §8 attribution can claim (problem P5) and answers cross-module anisotropy. | `correlation-duality.md` |
| **Wakhloo, Slatton & Chung** — *Neural population geometry and optimal coding of tasks with shared latent structure*, **Nature Neuroscience 2026**, doi 10.1038/s41593-025-02183-y (open access); `arXiv:2402.16770`. | The analytic optimum for tasks sharing a latent structure. Four statistics summarizing dimensionality, factorization and correlation structure govern linear-readout generalization. Sharpened Gate-3 competitor to generic capacity (RQ2). Also: optimal codes are lower-dimensional / higher-correlation **early** and the reverse **late** — converges with Menghi's anticorrelation trajectory (H1d). ⚠️ The specific four statistic names circulating in these docs are **inferred from the abstract**, not transcribed from the body. | `optimal-coding-statistics.md` |

The Wakhloo PRL supplement is the one people skip. Don't — it constrains §8.

---

## B. Extract a formula sheet, skip the full text

| Paper | Extract | Why not full text |
|---|---|---|
| **Cohen, Chung, Lee & Sompolinsky, Nat Comms 2020** | Simulated capacity `α_sim`. **MANDATORY**, not optional: it is the `02-validation-suite.md` §2a mean-field-validity gate. **Verified convention:** `α_sim := P / Σ_{n∈[N]} p_n`, estimated by bisection over the **projection dimension n**, at fixed P and N — *not* over P, *not* over N. Margin implicitly 0 (Def 2.1: `y_i⟨θ,s⟩ ≥ 0`), consistent with κ=0. Also verified: `\|α_sim − α_mf\| = O(1/N)` — convergence is in **N**, which is why the P×N sweep raises N first. | Definition + convention; implementation already vendored as `correlated_capacity` |
| **Montanari, Ruan, Sohn & Yan, 2019** — *The generalization error of max-margin linear classifiers* | **NEW, load-bearing.** Capacity under a label distribution **biased toward a task direction**. This is the theory behind **tilted capacity** — the `β` enhancement over retained (fixed-`y`) capacity — see §E.6. Chou et al. invoke it in App. A.1 and build on it in Theorem C.4 / Prop. C.7. | You need the biased-label capacity result, not the full high-dimensional asymptotics |
| **Bernardi, Benna, Rigotti, Munuera, Fusi & Salzman, Cell 2020** | CCGP protocol; shattering-dimensionality definition; dichotomy terminology | You need the protocol, not the monkey physiology |
| **Johnston & Fusi, Nat Comms 2023** | Classifier-generalization and regression-generalization metric definitions; latent-variable → representation setup. Supplies the label-aware probe metric H2 requires. | You need the metric; the RL half is out of scope |
| **Atanasov, Meterez, Simon & Pehlevan, ICLR 2025** | `η* ∝ γ²` (γ≪1) / `γ^(2/L)` (γ≫1); the large-γ time-reparameterization claim | Two facts, both already in `00-math-spec.md` §4.3 and §12 |
| **Chung, Lee & Sompolinsky, PRX 2018** | Capacity definition; thermodynamic/proportional limit conditions | Foundational context; the estimator you use is downstream |
| **Liu, Baratin, Cornford, Mihalas, Shea-Brown & Lajoie, ICLR 2024** | Rank-controlled initialization protocol | Only needed if you activate the recurrence extension |

---

## C. Read, don't put in the repo at all

Scientific context. The agent doesn't need these; you do. Keep them in `papers/`.

- **Hiratani, NeurIPS 2024** — feature × readout similarity 2×2 (encoded in `01-experiments.md` §3.1)
- **Menghi, Johnston, Viganò, Hinrichs, Maess, Fusi & Doeller, Nat Comms 2025** — the ρ_c ↔ MEG connection (motivation, not implementation)
- **Nature Human Behaviour 2025** — humans + networks, intermediate similarity
- **Canatar, Feather, Wakhloo & Chung, NeurIPS 2023 (SNAP)** — novelty positioning
- **Canatar, Bordelon & Pehlevan, Nat Comms 2021** — spectral bias / task-model alignment
- **Plasticity-loss literature** (Dohare 2024; Lyle 2023/2025; Lewandowski 2024; the 2026 label-agnostic-diagnostics counterexample paper) — constrains H2
- **Your own arXiv 2604.27656** — established prior context

---

## D. Code to vendor or reference

| Source | Status | Use |
|---|---|---|
| `schung039/neural_manifolds_replicaMFT` | **vendored** `@d56eda1d…` | **Primary near-term estimator.** `manifold_analysis_corr(XtotT, kappa, n_t, …)` → `(α, R, D, ρ_c, K)` (verified). Data as `(num_features, num_samples)` per manifold. `n_t=200`. Set `kappa=0`. Missing: `ρ_a`, `ψ` |
| `awakhloo/correlated_capacity` | **vendored** `@de8dac79…` | **Ground-truth α_sim** (`manifold_simcap_analysis`, general point-cloud, κ=0) for the §2a gate, plus correlated-capacity / duality (`replica_correlations.py`). **Not GLUE** (no ρ_a, ψ). |
| GLUE (full estimator: ρ_a, ψ) | **request-only** | Code is **NOT public** — verified. `chung-neuroai-lab/feature-learning-geometry` (ICML figure-repro) only *calls* GLUE. Early-access form: see §H. |
| `chung-neuroai-lab/feature-learning-geometry` | reference, not estimator | 97% notebooks. Useful only for **calling conventions** and Fig 7c parameters (`DNNs/config.json`, `run_feature_analysis.py`, cached capacity results). Do **not** treat as an estimator source. |
| `chung-neuroai-lab/ood-generalization-geometry` | **to vendor** — reference, not estimator (04 §C8) | MIT, figure-repro built on GLUE. Source for the **Gaussianization preprocessing** (`00` §6.3), the **pairwise subsampling protocol**, and `environment.yml` (QP solver versions). |
| `chung-neuroai-lab/SNAP` | reference | Spectral measures (KTA, eigenspectra) for the Gate-3 positioning check |

**Estimator versioning — THREE estimators.** Every result records its producer as
`{repo}@{sha}`. The `02-validation-suite.md` §9 pooling firewall applies
**pairwise across all three**: `replicaMFT`, `correlated_capacity`, GLUE-pending
must never be pooled. NB: GLUE was written to *refine* the mean-field assumptions
`replicaMFT` makes — they may differ **systematically**, so `glue-refinements.md`
is a prerequisite for trusting replicaMFT Phase-1 numbers.

**Vendoring policy — keep `third_party/` byte-identical to upstream.** Do **not**
patch vendored code in place: that breaks the SHA↔directory correspondence that
pinning exists to guarantee. All corrections (RNG injection, float64, explicit
rcond) live in a thin `src/glue/adapters/` layer. If an in-place edit is truly
unavoidable, record it as a `.patch` in `third_party/patches/` referenced from
`VENDORED.md`. See `third_party/VENDORED.md` for pinned SHAs, upstream URLs,
fetch date, capability inventory, and per-repo patch-needs.

---

## E. Non-paper information to add

Things that aren't in any paper and that the agent cannot infer.

### E.1 `docs/reference/parameterization-derivation.md`
The base-width normalization constant, derived and unit-tested. Graldi states
μP↔NTP equivalence at base width N₀=64 but does **not** write the constant out.
Currently **UNCERTAIN** in `00` §4.2. Derive once, test
(`02` §3 `test_base_width_equivalence`), never re-derive.

### E.2 `results/cost_model.json`
Measured wall-clock for one GLUE evaluation at target parameters, and the
multiplied-out grid. Must exist before Phase 1.

### E.3 `results/noise_floors.json`
Per-measure identifiability and trajectory floors, and the minimum detectable
effect at planned seed count. Everything downstream depends on this.

### E.4 `docs/reference/glue-sign-conventions.md` — now a **discrepancy log**, not a formula sheet

⚠️ **The source paper gives three mutually inconsistent versions of the same
measures** (verified against `arXiv:2503.18114v2`). This is the single most
error-prone thing in the project.

| Measure | Definition B.6 | Algorithm 2 (pseudocode) |
|---|---|---|
| ρ_c | `(1/P(P−1)) Σ_{i≠j} \|⟨s_i⁰,s_j⁰⟩\|` — **absolute value, unnormalized** | `(1/P(P−1)) Σ_{i≠j} (s_i⁰)ᵀs_j⁰ /(‖s_i⁰‖‖s_j⁰‖)` — **signed, normalized** |
| ρ_a | `E[\|⟨s_i¹,s_j¹⟩\|]` — abs, unnormalized | signed, normalized by both norms |
| ψ | `E[\|⟨s_i⁰, s_i¹⟩\|]` — **index `i` twice, inside Σ_{i≠j}** | `(s_i⁰)ᵀ s_j¹[k]/(…)` — **cross index `j`** |
| D_mf | `E[‖proj_cone({s¹})t‖²]` | `(1/n_tP) Σ_k t¹[k]ᵀ G¹[k]† t¹[k]` |

Definition 2.3 (main text, "simplified") gives yet a third form for `D_mf` and `R_mf`.

**Decision — REVISED 2026-08-10 (convention C3, per 04 §C3 + ICLR 2026 §B.3).
"Primary" does two separate jobs; do not collapse them into one verdict:**

- **Primary for reporting / comparability** = `rho_c_glue` — the
  Definition B.6 / ICLR 2026 §B.3 convention: `ρ_c`, `ρ_a`, `ψ` **absolute value,
  unnormalized, cross-index**. This is the number reported in the paper alongside
  published GLUE values so a reviewer can compare. (This reverses the earlier
  "use the signed form as primary" decision.)
- **Primary for the H1d hypothesis test** = `rho_c_signed` — signed, normalized
  `⟨s⁰_μ,s⁰_ν⟩/(‖s⁰_μ‖‖s⁰_ν‖)`. Under an absolute value, **decorrelation and
  anticorrelation are indistinguishable**, and Menghi's decorrelation prediction
  is not merely weakened but **untestable**. H1d's measure is `rho_c_signed`,
  explicitly (see `01` pre-registration table).

**`rho_convention: both` is MANDATORY, not a selectable default.** Both are always
computed; **no code path may reduce it to one.** The two are not comparable — say
so in the paper. `00-math-spec.md` §6.1 specifies both.

**Flag-and-stop (effective now, before §2.2).** `rho_c_signed` needs the anchor
centers `s⁰_μ`. If `manifold_analysis_corr` does **not** expose them (`signed_rho.py`
VERIFY-FIRST, `05` §2.2), that is a **flag-and-stop**, **not** a graceful
degradation to abs-only: it means H1d needs either an adapter-level extraction or
a `third_party/patches/` change to vendored QP internals (Kati's sign-off), or H1d
drops from the abstract. Report what would be required and wait — do not silently
proceed with `rho_c_glue` alone.

**Also record (verified empirical directions, App. B.5 / Fig 3c):** `ρ_c` and
`ρ_a` move capacity in **opposite** directions; large ρ_c → effective **radius
increases**; large ρ_a → effective **dimension decreases**; center-axis alignment
ψ has a **non-monotone** relationship with capacity.

**Open question for Chou/Le:** which convention is authoritative, and is the B.6
ψ index a typo?

### E.5 Dimensional ambiguity in Algorithm 2, Step 3 — **resolved by the stacked-matrix form**

```
t¹[k] ← Σ_i s_i¹[k] t_k          # reads as a sum of scalars ⟨s_i¹[k], t_k⟩
G¹[k] ← Σ_i s_i¹[k] (s_i¹[k])ᵀ   # N×N (sum of outer products)
```
The per-vector pseudocode is ambiguous (does `t¹[k]` typecheck as scalar or
`N`-vector?). **Resolved 2026-08-10:** ICLR 2026 §B.3 gives the estimator in
**stacked-matrix form** — anchors as rows of `S ∈ R^{P×N}`, `S_y = diag(y)S`, and
the `a/b/c` scalars use **P×P** Grams `S_y S_yᵀ` etc. `00-math-spec.md` §6.1 now
transcribes that form, which removes the free-index ambiguity. Confirm against
`replicaMFT`'s implementation and mark `# INFERRED-FROM-REFERENCE-IMPL:` where the
adapter maps replicaMFT internals onto `a/b/c`; the Chou/Le question is no longer
blocking.

### E.6 Retained capacity (fixed-`y`) is legitimate — tilt is an **enhancement**, not a fix

**DOWNGRADED 2026-08-10 (per 04 §C4).** The earlier claim here — that fixing `y`
"exits the definition of capacity" — was **too strong**. The capacity ensemble is
the analyst's choice of dichotomy collection `Y`; the capacity is the expectation
over `(y ~ Y, t ~ N(0,I_N))`, and fixing `|Y| = 1` keeps the `t`-expectation. It
is a **legitimate analyst choice**, now named **retained capacity** (`00` §7). The
exact three-factor identity holds for any ensemble, so §8 attribution is fine at
fixed `y`. (GLUE's relaxed assumption A2 concerns mean-field *accuracy* under such
a degenerate ensemble — a separate, empirical question, settled by the `β = ∞`
point of the `02` §2a sweep. C4 and A2 are compatible, not contradictory.)

**Tilt is an enhancement.** Sampling from `P(y) ∝ exp(β⟨y, y_j⟩)` gives:

- `β = 0` → generic capacity
- `β → ∞` → retained capacity (fixed `y_j`)
- intermediate `β` → an ensemble tilted toward task `j`

Precedent: Chou et al. App. A.1 note the biased-`y` setting explicitly and cite
Montanari et al. (2019); Theorem C.4 / Prop. C.7 build on it.

Better than the binary split: it yields a curve rather than two points, and is a
better operationalization of the hedging claim ("option value for tasks *near*
`j`" vs "for arbitrary tasks"). Staged (D2): build retained now, wire tilt after
the β-validity sweep confirms the usable range.

**The validity limit is checkable without GLUE.** `α_sim` imposes no label
assumption. Sweep β, compare `α_sim` vs `α_mf`, find where they diverge — that is
the largest tilt `replicaMFT` supports. Add β as a third axis to the `02` §2a
gate alongside P and N. Wide usable range ⇒ Phase 1's money figure runs on public
code. Narrow ⇒ a precise, quantified case for GLUE access.

### E.7 Hardware/environment notes
GPU memory limits, QP solver choice and version (cvxpy backend affects
reproducibility), thread pinning if the QP is CPU-bound.

### E.8 `NEGATIVE_RESULTS.md`
Running log of what was tried and failed, with config hash. In a project with
this many kill criteria the record of what was ruled out is a deliverable, and
the NeurReps Extended Abstract track accepts negative findings.

---

## F. Directory

```
docs/reference/
├── README.md                          # [done] index + provenance rule (add Version: field)
├── manifold-generator.md              # [done] App D.1.1 — verified vs v2
├── validation-settings.md             # [done] App B.5 — verified vs v2
├── parameterization.md                # [done] Graldi Table 1 (verified vs rendered HTML)
├── cl-metrics.md                      # [done] LA/LE/AA/AE/CF/CFr
├── glue-algorithm.md                  # [human] Algorithm 2 (v2) + Def B.6, verbatim, both forms
├── glue-decomposition.md              # [done] ICLR 2026 §B.3 — exact 3-factor identity + a/b/c
├── glue-sign-conventions.md           # [human] DISCREPANCY LOG — see §E.4
├── glue-refinements.md                # [drafted→verify] GLUE (bioRxiv) vs prior MFT assumptions; A2↔C4 noted; needs S1
├── correlation-duality.md             # [draft→verify] Wakhloo PRL Claim 1 + duality
├── parameterization-derivation.md     # [human] base-width constant — derived + tested
├── biased-capacity.md                 # [human] NEW — Montanari tilt; replaces fixed-y (§E.6)
├── ccgp-protocol.md                   # [draft→verify] Bernardi
├── generalization-metrics.md          # [human] Johnston & Fusi probe metric
└── optimal-coding-statistics.md       # [human] Wakhloo/Slatton four statistics

papers/                                # PDFs for humans; gitignored
results/                               # cost_model.json, noise_floors.json, mft_validity.json
src/glue/adapters/                     # corrections layer (RNG, float64, rcond)
third_party/
├── VENDORED.md                        # pinned SHAs, URLs, fetch date, patch-needs
├── patches/                           # .patch files IFF in-place edits unavoidable
├── replicaMFT/                        # vendored, pinned
├── correlated_capacity/               # vendored, pinned
└── ood-generalization-geometry/       # vendored, pinned — REFERENCE ONLY (Gaussianization, pairwise protocol, environment.yml); NOT an estimator (04 §C8)
```

---

## G. What to add first

**Done:** `manifold-generator.md`, `validation-settings.md`, `parameterization.md`,
`cl-metrics.md`; vendored `replicaMFT` + `correlated_capacity`.
**Resolved:** Algorithm numbering — **v2: geometric estimator = Algorithm 2,
α_sim = Algorithm 1** (verified vs complete PDF 2026-08-10; the earlier "v2 =
Algorithm 1" note was wrong). GLUE author list (10 authors, incl. Slatton &
Wakhloo). D.1.1 normalization wording. B.5 settings. α_sim bisection convention.
Exact three-factor identity + `a/b/c` (ICLR 2026 §B.3).

In order, from here:

1. **Re-verify the four completed sheets against `arXiv:2503.18114v2`** and add
   the `Version:` field. Cheap; prevents silent v1/v2 drift.
2. **Run the measurement-validity gates now — no GLUE needed.** `02` §1 (D/R/ρ_c
   recovery) and the whole §2a gate — including the P×N **and β** sweep — run on
   `replicaMFT` + `correlated_capacity` alone. Only ρ_a/ψ recovery needs GLUE.
   This unblocks the P=16-vs-32 decision with public code only.
3. **`glue-refinements.md` from the bioRxiv text + Supplementary S1.** Tells you
   whether replicaMFT Phase-1 numbers are usable at all.
4. `glue-sign-conventions.md` as the §E.4 discrepancy log, then `glue-algorithm.md`.
5. `biased-capacity.md` (§E.6) — before `00` §7 is implemented.
6. `correlation-duality.md` — before any §8 attribution claim is written.
7. `optimal-coding-statistics.md` — before wiring the Gate-3 comparison (RQ2).

---

## H. Where to get them

**Legend:** ✅ link verified 2026-07-31 · ⚠️ citation verified, DOI/URL to resolve

### Blocking set

| Paper | Link |
|---|---|
| ✅ Chou, Le, Wang & Chung, ICML 2025 | `https://arxiv.org/abs/2503.18114` — **use v2**: `https://arxiv.org/html/2503.18114v2` · PDF `https://arxiv.org/pdf/2503.18114` |
| ✅ Chou, Kirsanov, Yang & Chung, ICLR 2026 | `arXiv:2603.01879v1` — three-factor identity + a/b/c (App. B.3); PDF in `papers/` (verified complete 2026-08-10) |
| ✅ Graldi et al., ICML 2025 | `https://arxiv.org/abs/2506.16884` · HTML `https://arxiv.org/html/2506.16884v1` |
| ⚠️ Chou et al., GLUE, bioRxiv 2025 | doi `10.1101/2024.02.26.582157` → `https://www.biorxiv.org/content/10.1101/2024.02.26.582157v2.full-text` (agent-reported; **fetch Supplementary S1 separately**) |
| ✅ Wakhloo, Slatton & Chung, Nat Neuro 2026 | `https://www.nature.com/articles/s41593-025-02183-y` (open access) · preprint `https://arxiv.org/abs/2402.16770` |
| ⚠️ Wakhloo, Sussman & Chung, PRL 2023 | *Linear Classification of Neural Manifolds with Correlated Variability*, Phys. Rev. Lett. — resolve DOI; **get the Supplementary** |

### Formula-sheet set

| Paper | Note |
|---|---|
| ⚠️ Montanari, Ruan, Sohn & Yan (2019) | *The generalization error of max-margin linear classifiers: High-dimensional asymptotics in the overparametrized regime* — arXiv, Nov 2019 (`arXiv:1911.*`); resolve exact ID |
| ⚠️ Cohen, Chung, Lee & Sompolinsky (2020) | *Separability and geometry of object manifolds in deep neural networks*, Nat Commun 11:746 |
| ⚠️ Chung, Lee & Sompolinsky (2018) | *Classification and geometry of general perceptual manifolds*, Phys Rev X 8:031003 |
| ⚠️ Bernardi et al. (2020) | *The geometry of abstraction in the hippocampus and prefrontal cortex*, Cell 183(4):954–967 |
| ⚠️ Johnston & Fusi (2023) | *Abstract representations emerge naturally in neural networks trained to perform multiple tasks*, Nat Commun |
| ⚠️ Atanasov, Meterez, Simon & Pehlevan (2025) | *The Optimization Landscape of SGD Across the Feature Learning Strength*, ICLR 2025 |
| ⚠️ Liu, Baratin, Cornford, Mihalas, Shea-Brown & Lajoie (2024) | *How connectivity structure shapes rich and lazy learning in neural circuits*, ICLR 2024 |

### Context set

| Paper | Note |
|---|---|
| ⚠️ Hiratani (2024) | *Disentangling and mitigating the impact of task similarity for continual learning*, NeurIPS 2024 |
| ⚠️ Menghi, Johnston, Viganò, Hinrichs, Maess, Fusi & Doeller (2025) | *The effects of task similarity during representation learning in brains and neural networks*, Nat Commun |
| ⚠️ Canatar, Feather, Wakhloo & Chung (2023) | *A Spectral Theory of Neural Prediction and Alignment*, NeurIPS 2023 (spotlight) |
| ⚠️ Canatar, Bordelon & Pehlevan (2021) | *Spectral bias and task-model alignment…*, Nat Commun 12:2914 |

### Code

| Resource | Link |
|---|---|
| ✅ replicaMFT | `https://github.com/schung039/neural_manifolds_replicaMFT` |
| ⚠️ correlated_capacity | `https://github.com/awakhloo/correlated_capacity` · Zenodo doi `10.5281/zenodo.7844169` (agent-reported) |
| ✅ ICML figure-repro (**not** an estimator) | `https://github.com/chung-neuroai-lab/feature-learning-geometry` |
| ✅ SNAP | `https://github.com/chung-neuroai-lab/SNAP` |
| ✅ **GLUE early-access form** | `https://docs.google.com/forms/d/e/1FAIpQLSc_IHUkc2zlJv0DIhSL_tiyD7Ty4nCeFdW0U7s-hCVWchefBg/viewform` |

**Code availability, verified in v2:** "Codes will be available for public usage
in the final version. Requests on accessing to the current version of the code
should be made to the first and corresponding author." Contacts:
`{cchou, hle, schung}@flatironinstitute.org`. The form is the official channel.