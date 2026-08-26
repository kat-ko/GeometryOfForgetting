# 04 — Spec Corrections from Chou et al. ICLR 2026 + author correspondence

> **ERRATUM (2026-08-10).** This file predates the Algorithm-numbering finding.
> Throughout §C2/§C3 below, every reference to **"Algorithm 1"** as the *geometric
> estimator* should read **"Algorithm 2"** per arXiv:2503.18114**v2** (verified
> 2026-08-10; see `05-implementation-brief.md` §0 D1). In that paper, **Algorithm 1
> is the α_sim bisection** (simulated capacity) and **Algorithm 2 is the geometric
> estimator** (α, R, D, ρ_c, ρ_a, ψ). The body below is left unchanged as a
> correction-log record; apply its apply-order with that substitution.

```
Trigger:  (a) email reply from Chi-Ning Chou, 2026-07-31
          (b) Chou, Kirsanov, Yang & Chung, "Diagnosing Generalization Failures
              from Representational Geometry Markers", ICLR 2026,
              arXiv:2603.01879v1 — Appendix B.3 in particular
Status:   corrections C1–C5 are VERIFIED against the paper text.
          Apply before any further estimator work.
```

Appendix B.3 of the ICLR 2026 paper is the most complete public statement of the
GLUE decomposition to date. It supersedes several things in `00-math-spec.md`,
including three claims I asserted in earlier analysis. Corrections C1 and C2 are
the load-bearing ones.

---

## C1 — The capacity decomposition is **three-way and exact**, not two-way and approximate

**Spec currently says (`00` §6.2):** `α_mf ≈ (1 + R_mf⁻²) / D_mf`, flagged UNCERTAIN,
with `02` §2b testing its fidelity at a 15% threshold.

**The actual relation (ICLR 2026 §B.3, Eq. 2 and §B.3.1):**

```
α  =  Ψ_eff · (1 + R_eff⁻²) / D_eff

equivalently  N_crit = P · D_eff / (Ψ_eff · (1 + R_eff⁻²)),   α = P / N_crit
```

There is a **third measure, effective utility Ψ_eff**, that the spec omits entirely.
The papers give its range as `[0,1]`; we have since measured that this holds for the
generic (random-dichotomy) ensemble only and that fixed-`y` values legitimately exceed
1 — see `00` §6.2 for the mechanism and the ground-truth demonstration.

**And the relation is exact, not an approximation.** With the paper's definitions

```
a(y,t) = (S_y t)ᵀ (S_y S_yᵀ)† (S_y t)
b(y,t) = (S_{y,1} t)ᵀ (S_{y,1} S_{y,1}ᵀ)† (S_{y,1} t)
c(y,t) = (S_{y,1} t)ᵀ (S_{y,0} S_{y,0}ᵀ + S_{y,1} S_{y,1}ᵀ)† (S_{y,1} t)

α      = P / E[a]
D_eff  = (1/P) · E[b]
R_eff  = sqrt( E[c] / E[b − c] )
Ψ_eff  = E[c] / E[a]
```

substitution gives

```
Ψ_eff (1+R_eff⁻²)/D_eff = (E[c]/E[a]) · (E[b]/E[c]) · (P/E[b]) = P/E[a] = α   ∎
```

### Consequences

| Item | Was | Now |
|---|---|---|
| `00` §6.2 | approximation, UNCERTAIN | **exact identity, three factors** — remove the UNCERTAIN marker |
| `02` §2b | test α_mf vs approximation at 15% | **delete.** There is nothing to test; they are algebraically identical |
| `02` §2a | α_sim vs α_mf | **unchanged and now the only approximation gate.** Still real: mean-field vs simulation |
| `00` §8 attribution | finite differences in (R, D) + unexplained residual | **exact three-way multiplicative decomposition.** Take logs: `log α = log Ψ_eff + log(1+R_eff⁻²) − log D_eff`. Attribution becomes additive in log-space with **zero residual** |
| Figure 2 | attribution with a residual bar | attribution with three exact components |

This is a strict improvement. The "residual" in the old §8 scheme was Ψ_eff all
along — a defined quantity with an interpretation, not estimator noise.

### What Ψ_eff means, and why it matters for H1/H2

Per Table 1/4: `Ψ_eff` quantifies **excessive compression** — manifolds become
more separable when `Ψ_eff` is *large*; collapsing manifolds to points sends
`Ψ_eff → 0`. In feature-learning terms, low `Ψ_eff` indicates inefficient
compression of within-class variability.

H1 must be restated over three components, and the rich-regime prediction now has
a natural home: over-compression (low `D_eff`, low `Ψ_eff`) is the paper's
signature of *overspecialization*. That is exactly what we predict rich-regime
forgetting looks like.

---

## C2 — The estimator operates in **P-space**, not N-space (major compute correction)

I previously flagged a dimensional ambiguity in Algorithm 1 Step 3 — that
`t¹[k] = Σ_i s_i¹[k] t_k` reads as a scalar while `G¹[k]` looked like an N×N
matrix, and proposed inferring the resolution from `replicaMFT`.

**Resolved, and my inference was wrong.** In the stacked-matrix notation:

- `S_y := diag(y) S`, where `S ∈ R^{P×N}` holds the anchor points as rows
- `S_{y,1} t` is a **P-vector** (P×N times N-vector)
- `S_{y,1} S_{y,1}ᵀ` is **P×P**, not N×N

So `a`, `b`, `c` are scalars formed by **P×P** pseudo-inverses.

**Implication for `00` §13 (cost model): pseudo-inverses are 16×16, not 300×300.**
The dominant cost is the QP for anchor points, not the linear algebra. The cost
model must be re-derived; the Tier-2 budget is likely far less constraining than
assumed, which may free the seed count.

`00` §6.1 should be rewritten in the stacked-matrix form. It is unambiguous, and
it removes the need to reverse-engineer intent from vendored code.

---

## C3 — Convention on ρ_c, ρ_a, ψ: the lab uses **absolute value, unnormalized, cross-index**

I recommended implementing Algorithm 1's signed-normalized forms. **Two
independent papers now agree against that.** ICLR 2026 §B.3 gives:

```
ρ^c_{μ,ν} := |⟨s^μ_0, s^ν_0⟩|
ρ^a_{μ,ν} := E_{y,t}[ |⟨s^μ_1(y,t), s^ν_1(y,t)⟩| ]
ψ_{μ,ν}  := E_{y,t}[ |⟨s^μ_0, s^ν_1(y,t)⟩| ]
```

Absolute value, no norm denominators — matching ICML Definition B.6, not
Algorithm 1. **The `ψ` cross-index question is also settled**: `s^μ_0` against
`s^ν_1`, different indices. The B.6 same-index form was a typo.

**Revised recommendation — compute both, label them distinctly:**

| Name | Form | Purpose |
|---|---|---|
| `rho_c_glue` (default) | `\|⟨s⁰_μ, s⁰_ν⟩\|`, unnormalized | lab convention; comparable to published numbers |
| `rho_c_signed` (project extension) | `⟨s⁰_μ, s⁰_ν⟩ / (‖s⁰_μ‖‖s⁰_ν‖)` | **required for H1d** |

The scientific argument for the signed version is unaffected by the convention
question and still stands: **Menghi's anticorrelation result lives entirely in the
negative range.** Under `|⟨·,·⟩|` you cannot distinguish decorrelation from
anticorrelation, and H1d is untestable. Report the lab convention as primary for
comparability, the signed version as the H1d instrument, and state the difference
explicitly in the paper.

Note also that the unnormalized form is **not scale-invariant** — which is exactly
the cross-module anisotropy problem (see C5).

### Notation collision — fix in the glossary now

`Ψ_eff` (effective utility, a scalar factor in α; in [0,1] for the generic ensemble,
unbounded above at fixed `y` — `00` §6.2) and `ψ_{μ,ν}`
(center–axis alignment, pairwise) are **different quantities with near-identical
symbols.** Mandate in `AGENTS.md` §3:

- `psi_eff` / "effective utility" — never abbreviated to `psi`
- `center_axis_alignment` — never written `psi`

---

## C4 — My "fixed-`y` exits the definition" claim was **too strong**

I argued in the §E.6 note that fixing `y` for aligned capacity destroys the
ensemble and voids `α ≈ (1+R⁻²)/D`.

The paper contradicts this directly: <cite>a dichotomy vector y and a collection
Y ⊂ {−1,1}^P are chosen by the analyst</cite>, with 1-vs-rest or all dichotomies
given as *common* choices — not the only admissible ones. The expectation is over
`(y, t)` jointly; with `|Y| = 1` the expectation over `t` remains, and `a`, `b`,
`c` are still well-defined. **Aligned capacity at fixed `y` is legitimate.**

**But the tilt is still worth adopting**, now as an enhancement rather than a repair:

- it yields a *curve* over β rather than two points
- it operationalizes "option value for tasks *near* `y_j`" versus "for arbitrary tasks", which is a better statement of the hedging claim
- it has precedent (Montanari) and is explicitly anticipated by Chou et al.

Downgrade `03-references.md` §E.6 from "the fixed-`y` version isn't salvageable"
to "the tilt generalizes the analyst's choice of `Y` and is preferred." The
Montanari sheet stays Priority 1 — the biased-label result is what licenses the
intermediate β — but it no longer blocks §7.

---

## C5 — Preprocessing: **Gaussianization**, not standardization

Chi-Ning's reply treats the anisotropy question as separate from GLUE, noting the
same issue affects all decoding, dimensionality-reduction and RSA methods, and
that they preprocess raw representations following what the experimentalist would
have done, then run GLUE and compare the resulting values.

The paper names the concrete step: for each manifold they subsample to 50 points,
run GLUE on each manifold pair, and apply **Gaussianization preprocessing
(Wakhloo et al., 2023)** to ensure initial linear separability.

**Replace `00` §6.3's `raw` / `standardized` pair with `raw` / `gaussianized`.**
Simple per-module standardization was my invention; Gaussianization is the lab's
method, comes from the same PRL paper that supplies the correlation duality, and
is the defensible choice under review. Extract it into
`docs/reference/correlation-duality.md` when that sheet is written.

The dual-reporting discipline is unchanged: report both, require qualitative
agreement, treat divergence as a finding.

---

## C6 — Practice notes from the paper (not corrections, but they change defaults)

| Item | Their practice | Our decision |
|---|---|---|
| **Manifold subsampling** | 50 points per manifold | our `M = 150` is more generous; keep, but record that published numbers use 50 |
| **Pairwise analysis** | subsample 2 classes, 50 points each, GLUE per pair, **100 repetitions** | **open decision — see below** |
| `n_t` | 200 typical; 1000 for noisier data (email) | keep 200; add 1000 as the noise-floor fallback |
| Margin | κ = 0 throughout | unchanged |

### The pairwise-vs-full-P decision, and what it does to the P question

Their pipeline runs GLUE on **manifold pairs**, averaging over many random
subsamples, rather than on all P manifolds jointly. This has two consequences.

**First, it weakens my earlier "Split-CIFAR10 has P=2, therefore GLUE is
meaningless" argument.** If pairwise estimation with repeated subsampling is the
lab's standard practice, then P=2 per estimate is the operating regime, not a
defect. That objection should be dropped from `PROJECT.md` §2 (P1) and
`03-references.md`, and **not used in the paper**, because a reviewer from this
group will know their own pipeline.

**The valid form of the argument survives, but it is about task structure, not
estimation:** our design needs P large because *the dichotomy is the task*. With
P = 2 there is exactly one balanced dichotomy up to sign — no similarity axis, no
factorial structure, no CCGP. P = 16 is required by §2.2, not by the estimator.
Restate P1 accordingly.

**Second, the P×N sweep in Phase 0 changes shape.** Estimation validity is now
partly decoupled from P. Add `estimation_mode ∈ {pairwise, full_P}` to the config
and measure both in Phase 0. If pairwise-averaged estimates match full-P at
P = 16, use pairwise for comparability with published numbers and full-P as the
project measure. If they diverge, that is itself worth a sentence.

The P = 16 vs 32 question is now driven by **dichotomy-space richness**, not
mean-field validity. That is a cheaper question to settle.

---

## C7 — Novelty: the ICLR 2026 paper is the closest competitor yet to RQ2

It is *not* a scoop, but it is nearer than SNAP, and it must be cited and
distinguished.

**What they show:** ID-only geometric markers forecast OOD generalization;
specifically, reductions in effective dimension and utility predict weaker OOD
performance across architectures, optimizers and datasets, and predict transfer
better than ID accuracy (73.0% vs 37.2% on their pretrained-model benchmark).

**Why this is close to our RQ2:** measuring geometry on data the model has been
trained on, to predict performance on tasks it has not seen, is the same move as
generic capacity. `D_eff` and `Ψ_eff` are, functionally, a validated
"option-value" marker.

**Three defensible distinctions:**

1. **Setting.** Theirs is stationary — train once, measure once, predict transfer. Ours is sequential: geometry is measured repeatedly along a stream, and the question is how it *changes* under continued training. Their future-work list covers language, RL, multimodal, and neuroscience parallels — **it does not mention continual or sequential learning.** Cite that gap directly.
2. **Direction of the claim.** They predict *forward* transfer to unseen classes. We ask about *retention* of past tasks alongside acquisition of new ones. Forgetting has no analogue in their setup.
3. **Mechanism vs prognosis.** They are explicit that the framework is diagnostic and that strengthening the theoretical basis is future work. Our attribution is mechanistic, over the exact three-way decomposition.

**A gift, though: their result licenses our H2 direction.** They find high `D_eff`
and high `Ψ_eff` → better transfer, in the regime after meaningful feature
learning, where excessive compression signals overspecialization. That is a
published, validated version of "generic capacity is option value." H2 no longer
has to establish that from scratch — it can *assume* it and ask what sequential
training does to it.

**One scope warning to respect.** They find the geometry–OOD correlation is
specific to class-level shifts and does **not** extend to corruption shifts where
the label space is unchanged. Our streams vary the *dichotomy* — a label-space
change — so we are on the side where the correlation holds. Say so explicitly;
do not let a reviewer raise it first.

---

## C8 — Code status: unchanged, and verified again

`chung-neuroai-lab/ood-generalization-geometry` (MIT, 7 commits, `analysis/`,
`scripts/`, `models/`, 3 notebooks) — **README states the analysis was built on
top of GLUE and links the same early-access form.** Figure-reproduction code, not
an estimator. Identical pattern to `feature-learning-geometry`.

**Still vendor it**, for: calling conventions, the Gaussianization preprocessing
implementation, the pairwise subsampling protocol, and `environment.yml` (solver
versions matter for QP reproducibility). Tag as reference, not estimator.

The early-access form remains the channel for GLUE proper. Chi-Ning's reply
directs there explicitly, so submitting it is the expected next step.

---

## Apply-order

1. **`00` §6.2 → exact three-way identity**; add `Ψ_eff` throughout; remove UNCERTAIN
2. **`00` §6.1 → stacked-matrix form**; P×P pseudo-inverses
3. **`02` §2b → delete**; renumber; `02` §2a is now the sole approximation gate
4. **`00` §8 → log-space additive attribution**, three components, zero residual
5. **`00` §6.3 → `raw` / `gaussianized`**
6. **`00` §13 → re-derive the cost model** (P×P, not N×N)
7. `AGENTS.md` §3 → `psi_eff` vs `center_axis_alignment` naming rule
8. `01` §1 → add `estimation_mode`, `rho_convention` config fields
9. `01` Phase 0 → add pairwise-vs-full-P comparison; re-scope the P sweep
10. `PROJECT.md` §2 (P1) → restate as task-structure, not estimation
11. `PROJECT.md` §5 → add ICLR 2026 as load-bearing; §15 → add the C7 distinctions
12. `03-references.md` §E.6 → downgrade per C4

Items 1–4 change what Figure 2 shows. Do them before any attribution code.
