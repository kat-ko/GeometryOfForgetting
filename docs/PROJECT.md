# Geometric Signatures of Sequential Learning

**Project specification — Line 1 extension**
Target: NeurReps 2026, Extended Abstract track (4 pages, non-archival)
Status: pre-implementation
Last updated: 30 July 2026

---

## 0. One-paragraph summary

Manifold capacity theory characterizes how task-relevant representations untangle during learning, and the GLUE decomposition attributes changes in capacity to interpretable geometric components (radius, dimension, center alignment, axis alignment, center-axis alignment). Every application to date has been to **stationary** training. Sequential learning imposes a constraint with no stationary analogue: a representation must support the task currently being optimized *and* tasks it is no longer shown, without knowing what comes next. This project asks what catastrophic forgetting looks like **geometrically** — whether the capacity drop on past tasks decomposes into distinguishable modes, whether those modes differ by learning regime, and whether rich learning purchases current-task margin by spending the representation's capacity to support arbitrary future tasks. The setting is continual learning; the object is representational geometry.

---

## 1. Motivation

### 1.1 The scientific gap

Three literatures converge on the same phenomenon without sharing vocabulary:

- **Scaling-limit CL theory** (Graldi et al., ICML 2025) shows the degree of feature learning γ₀ controls forgetting, with an optimum at γ₀\* ≈ 0.1 that shifts toward 1 as task similarity rises. Measured via 1−CKA and accuracy drop. Tells us *how much* changed, not *what* changed.
- **Representational geometry** (Chou et al., GLUE, bioRxiv 2025; Chou et al., ICML 2025) provides a decomposition of *what* changed, validated across seven neuroscience datasets — but only for stationary training.
- **Human/animal work on task similarity** (Menghi et al., Nat Comms 2025; Hiratani, NeurIPS 2024) shows that similarity structure determines interference, and that the operative quantity is the **correlation between task representations** — which is precisely GLUE's center alignment ρ_c.

Nobody has decomposed catastrophic forgetting into manifold geometry. The instrument exists, the empirical anchors exist, and the connection has not been made.

### 1.2 Why sequentiality is the driver, not the setting

In joint/i.i.d. training there is a single optimal representation for the task distribution; capacity spent maintaining anything else is waste. This is why a *single* optimal richness γ\* exists.

In sequential training the learner optimizes for task *t* but is graded on *t+1…T*, drawn from an unknown similarity distribution. The optimal **policy** is not the optimal **solution to the current task**. Manifold capacity already contains the distinction this requires, unused:

> **Generic capacity** — separability under an *arbitrary* future dichotomy. Option value: what the representation can still support.
>
> **Retained capacity** — how well the representation supports a *fixed* dichotomy `y_j` (β→∞ tilt).

Capacity is defined as an average over random dichotomies **y** ~ {±1}^P. The randomization object of the theory becomes the experimental variable. The hypothesis is that rich learning maximizes the second by spending the first — the stability–plasticity trade-off restated as a measurable geometric quantity.

### 1.3 Why NeurReps

The venue rewards geometry as the object, not CL as a benchmark. Themes explicitly include representational geometry in neural data and statistical learning theory in the context of geometry. The Extended Abstract track is non-archival and accepts early-stage results and negative findings — which matters, because several phases below have real kill criteria, and because it does not encumber the ICLR/ICML 2027 full paper the way the archival Proceedings track (30% new-material rule) would.

### 1.4 Relation to existing work in this line

The prior Line 1 result — dimensionality gates whether modularity shapes representational geometry (arXiv 2604.27656, not yet accepted) — established *that* effective dimensionality conditions modular benefit. This project asks *by what geometric mechanism*, using a validated instrument rather than CKA. Dimensionality-gating is prior work here, not the headline.

---

## 2. Problems this project must solve

| # | Problem | Why it blocks |
|---|---|---|
| P1 | Standard CL benchmarks lack the **task structure** a dichotomy-geometry design needs | Split-CIFAR10 gives P=2 manifolds/task; with P=2 there is exactly one balanced dichotomy up to sign — **no similarity axis, no factorial structure, no CCGP.** P=16 is required by the *design* (four binary factors, §6.2), **not** by the estimator: the lab estimates GLUE **pairwise (P=2) routinely** (04 §C6). *(Earlier "GLUE estimation is meaningless at P=2" was wrong — dropped.)* |
| P2 | Task similarity is proxied, never measured, in the CL literature | Graldi et al. use unpermuted-pixel fraction (input only, single head) and classes-per-task (a structural proxy, never measured, under a different head structure). The two claims are not comparable |
| P3 | γ conflates available geometry with the update operator | A γ sweep alone cannot attribute drift differences to "dimensionality" |
| P4 | Observed geometry is seed-dependent and estimator-noisy | Measures differ in invariance class; GLUE alignment measures are expected to be noisiest |
| P5 | GLUE components are not independent (Wakhloo duality) | Axis correlation is observationally equivalent to radius shrinkage; attribution is a decomposition, not a mechanism |

---

## 3. Research questions

**RQ1.** Does catastrophic forgetting have a distinguishable geometric signature — can the capacity drop on past tasks be attributed to specific GLUE components, and does the attribution differ by learning regime?

**RQ2.** Does rich learning trade generic capacity for retained capacity, and is that trade the mechanism of the stability–plasticity front?

**RQ3.** Can a population heterogeneous in initial geometry and update magnitude occupy a point on the front that no homogeneous population reaches — and is the mechanism a geometric division of labour or an averaging effect?

---

## 4. Hypotheses

### H1 — Forgetting has distinguishable geometric modes
Capacity drop on task *j* after training on later tasks is accounted for differently by regime:
- **Rich:** radius expansion + center-axis alignment increase, dimension roughly preserved
- **Lazy:** center alignment (ρ_c) increase, radius roughly preserved

*Anchor 1:* Chou et al. found radius expansion and increased center-axis alignment explain the capacity drop underlying ultra-rich OOD failure. **H1 predicts forgetting in the rich regime carries the same geometric signature as ultra-rich OOD failure** — i.e. they are one phenomenon.

*Anchor 2:* Menghi et al. found correlated task representations in the similar-structure group and anticorrelated in the dissimilar group, with practice shifting toward anticorrelation. **H1b: drift trajectories show progressive center-decorrelation, faster in rich modules.**

*Constraint (P5):* stated as decomposition, not causal mechanism. Report ρ_a and R jointly and check consistency with the duality.

### H2 — The generic/retained trade
Generic capacity decreases monotonically with γ₀ across the stream while retained capacity increases; their crossing predicts γ₀\*. Specifically, the γ₀ minimizing average error coincides with the maximum of (generic capacity × retained capacity) within seed noise.

*Mandatory validation:* generic capacity is **label-agnostic**, and the plasticity literature has theoretical counterexamples showing label-agnostic structural metrics (representation rank, eNTK rank) can look favourable where gradient descent cannot progress. Generic capacity must therefore be validated against a **label-aware** held-out-task metric — Johnston & Fusi's novel-task generalization measure — not asserted.

### H3 — Geometric division of labour *(weakened)*
In a heterogeneous pair, the wealthy-lazy module retains higher generic capacity and lower drift; the poor-rich module attains higher retained capacity and higher drift. Readout reliance shifts toward the rich module as similarity to the recent past drops.

*Constraint:* Atanasov et al. found networks of different **large** γ optimize along similar trajectories up to a time reparameterization. Restrict the heterogeneity contrast to γ ≪ 1 vs. γ ~ 1. Add an explicit time-rescaling test in Phase 0: if the two modules' trajectories coincide under rescaling, H3 is dead.

### H4 — Endogenous regime drift
Initial regime does not persist. Graldi et al. document the mechanism (*pretraining effect*): at high similarity, features from task 1 transfer, later tasks show no further evolution beyond a γ₀ threshold, and the network becomes effectively lazy after task 1. Prediction: an initially poor-rich module becomes wealthy-lazy once generic capacity crosses a threshold; heterogeneity has a similarity-dependent half-life.

*Positioning:* this is plasticity-loss-adjacent. Cite rank-collapse mechanisms rather than rediscovering them. Competing explanation to rule out: loss of plasticity may be an artifact of *abrupt* environment change and largely mitigated under gradual change.

### H5 — γ_eff sufficiency (killer control)
If pair mean-γ predicts performance, H3 is a reparameterization and there is no result. Matched-mean heterogeneous configurations must differ from homogeneous ones.

### H6 — Hiratani's 2×2, geometrically *(new)*
The catastrophic corner (high feature similarity × low readout similarity) shows the largest overwrite/rotation signature and the largest ρ_c increase.

*Value:* validates the setup against an analytic result in the linear limit, and measures something Hiratani's linear teacher-student model cannot — it has no manifold structure.

---

## 5. Literature

### 5.1 Load-bearing (must read before writing)

| Work | Contributes | Note |
|---|---|---|
| **Chou, Kim, Arend, Yang, Mensh, Shim, Perich & Chung**, bioRxiv 2025 (GLUE) | Estimator definitions, sign conventions, how each measure moves capacity | **Preprint** (doi 10.1101/2024.02.26.582157), full text open — **not** Nature Neuroscience. Code **request-only** (form in `third_party/VENDORED.md`). GLUE relaxed two assumptions of prior mean-field theory (Gaussian correlations; random labels) — see `docs/reference/glue-refinements.md`. Distinct from the Nat Neuro 2026 paper below. *Verify exact author list per bioRxiv version.* |
| **Chou, Le, Wang & Chung**, ICML 2025 | Manifold capacity as richness measure; the ultra-rich OOD signature (Fig 7c); synthetic manifold generator (App. D.1.1); **Algorithm 2** (the geometric estimator; Algorithm 1 is the α_sim bisection) | The direct methodological parent. **Pin `arXiv:2503.18114v2`** |
| **Chou, Kirsanov, Yang & Chung**, ICLR 2026 | The **exact** three-factor identity `α = Ψ_eff·(1+R_eff⁻²)/D_eff` (App. B.3); ID-only geometry markers (`D_eff`, `Ψ_eff`) forecast OOD generalization | `arXiv:2603.01879v1`. **Closest competitor to RQ2** (stationary, forward-transfer only — distinguished in §15); source of the attribution decomposition (`00` §8) and `docs/reference/glue-decomposition.md` |
| **Graldi, Breccia, Lanzillotta, Hofmann & Noci**, ICML 2025 | γ₀ parameterization; γ₀\* ≈ 0.1; similarity shifts the lazy-rich transition; the pretraining effect; CFr metric | The baseline this project extends and critiques |
| **Menghi, Johnston, Viganò, Hinrichs, Maess, Fusi & Doeller**, Nat Comms 2025 | Human MEG: similar-structure training → correlated representations → worse performance; practice → anticorrelation | **ρ_c is the shared order parameter.** Primary Fusi-visit entry point |
| **Hiratani**, NeurIPS 2024 | Feature × readout similarity 2×2; high-feature/low-readout is catastrophic; nonmonotonicity on the *feature* axis | Defines the stream design (H6). Note: nonmonotonicity is NOT on the readout axis |
| **Atanasov, Meterez, Simon & Pehlevan**, ICLR 2025 | η\* ∝ γ² for γ≪1, γ^(2/L) for γ≫1; large-γ trajectories equal up to time reparameterization | Threat to H3; source of the LR correction |
| **Wakhloo, Sussman & Chung**, PRL 2023 | Duality: centroid correlations ≈ reduced center separation; axis correlations ≈ shrunk radii. Capacity under arbitrary correlations | Constrains H1's claim form; answers cross-module anisotropy |
| **Wakhloo, Slatton & Chung**, Nat Neuro 2026 | Analytic optimum for tasks with **shared latent structure**; four statistics (dimensionality, total correlation, signal–signal & signal–noise factorization) govern linear-readout generalization; optimal codes go low-dim/high-corr early → high-dim/low-corr late | doi 10.1038/s41593-025-02183-y; arXiv 2402.16770, open access. **Sharpened Gate-3 competitor to generic capacity.** *Whether it reframes RQ2 = pending scientific decision (flagged, not yet taken).* |
| **Johnston & Fusi**, Nat Comms 2023 | Multi-task learning → abstract representations; D latent factors → high-dim representation; classifier/regression generalization metrics | **Our design is theirs, sequentialized.** Provides the held-out-task metric for H2 |
| **Bernardi, Benna, Rigotti, Munuera, Fusi & Salzman**, Cell 2020 | CCGP and shattering dimensionality conventions | Match terminology exactly for legibility |

### 5.2 Positioning / must-cite

- **Canatar, Feather, Wakhloo & Chung**, NeurIPS 2023 (SNAP) — spectral↔geometric bridge for regression. Reviewers will expect awareness. Code: `chung-neuroai-lab/SNAP`
- **Canatar, Bordelon & Pehlevan**, Nat Comms 2021 — spectral bias and task-model alignment. The Gate-3 competitor framing
- **Liu, Baratin, Cornford, Mihalas, Shea-Brown & Lajoie**, ICLR 2024 — connectivity rank shapes rich/lazy; the second regime knob
- **Nature Human Behaviour 2025** (humans + networks, A–B–A rule tasks) — intermediate similarity promotes shared representations → higher transfer at cost of greater interference. Behavioural anchor
- **Plasticity-loss literature** (Dohare et al. 2024; Lyle et al. 2023/2025; Lewandowski et al. 2024) — rank/eNTK/Hessian collapse as mechanisms; the label-agnostic-diagnostic counterexamples that constrain H2
- **Chung, Lee & Sompolinsky**, PRX 2018; **Cohen, Chung, Lee & Sompolinsky**, Nat Comms 2020 — capacity theory foundations
- **Prior Line 1 work** (arXiv 2604.27656) — dimensionality gates modularity; established context, not headline

### 5.3 Background

Chizat, Oyallon & Bach 2019; Yang & Hu 2021; Bordelon & Pehlevan 2022; Kumar et al. 2024 (grokking as lazy→rich); Flesch et al. 2022; Rigotti et al. 2013; RanDumb; Béna & Goodman 2025.

---

## 6. Task setup

### 6.1 Generator

Sequentialize the Chou et al. synthetic manifold generator (App. D.1.1):

```
M_i = { u_0^i + R · Σ_{j=1..D} s_j^k u_j^i + ε v_k }_{k=1..M}
  u_j ~ N(0, I_d/d),  s_j^k ~ N(0,1),  v_k ~ N(0, I_d/d),  ε = 1e-2
  pre-scaled manifold points normalized to unit norm
```

Correlated variants via autoregressive covariance C = (ρ^|i−j|) mixed in by Cholesky decomposition, separately for centers (ρ_C) and axes (ρ_A).

**Parameters:** d ≈ 150, P = 16 (4 binary factors), M ≈ 150, D and R as stream variables.

### 6.2 Factorial structure

Index the P = 16 manifolds by 4 binary latent factors (f₁…f₄). Dichotomy families:

| Family | Count | Property |
|---|---|---|
| Single-factor (y = f_i) | 4 | maximally abstract; high CCGP |
| XOR/parity (y = f_i ⊕ f_j) | 6 | requires nonlinear mixing; low CCGP; high shattering demand |
| Random balanced | many | generic |

**Design decision:** factor structure is reflected in the **labeling only**, not in the geometry of the centers. Centers are random. This forces abstraction to be *learned* rather than inherited, and prevents confounding with the wealth manipulation.

*Rationale:* makes CCGP and shattering dimensionality well-defined; gives a similarity axis beyond Hamming distance (shared abstract structure vs. shared labels); yields the sharper H3 prediction that wealthy-lazy supports single-factor dichotomies and poor-rich supports XOR.

### 6.3 Task definition and similarity

Task *t* = a **balanced** dichotomy **y**_t ∈ {±1}^P over the arrangement.

- **Readout similarity** s_r(t,t′) = |2·overlap(**y**_t, **y**_t′)/P − 1| ∈ [0,1]. Sampling at exact Hamming distance: swap d/2 of the +1s with d/2 of the −1s. Distances between balanced dichotomies are even; sign symmetry applies.
- **Feature similarity** s_f(t,t′) = correlation between manifold *arrangements*, controlled via ρ_C when re-drawing centers between tasks.

Unbalanced dichotomies are excluded — degenerate, and they contaminate similarity geometry.

### 6.4 Stream conditions

**Primary (H6): Hiratani 2×2.** {s_f low, high} × {s_r low, high}, four corners. High-feature/low-readout is the predicted catastrophic corner.

**Secondary:** `fixed-s` at three levels of s_r with s_f = 1 (the isolated-readout condition); `blocked` vs `shuffled` at matched marginals (deferred to full paper).

**Similarity bookkeeping.** Control s(t, t−1); **record the full similarity matrix S** and use s(t, j) as covariates in the drift regression. Do not attempt to control long-range similarity — the random walk on the hypercube makes it uncontrollable, and recording is sufficient.

**Held-out probe dichotomy y\***, never trained, evaluated at every boundary via a freshly trained readout. This is the label-aware ground truth for generic capacity and is **mandatory** (see H2).

**Length:** T = 16 primary; T = 40 for the H4 half-life measurement.

### 6.5 Training

Single-pass or fixed small epoch count per task. Plain SGD. **No weight decay** (it silently converts lazy modules to rich ones over a long stream). No normalization layers. ReLU (positive homogeneity keeps the γ analysis clean; tanh saturates and produces dead, not lazy, units). Zero-init readout so f(x; θ₀) = 0. No gradient clipping. Report at matched current-task loss, not matched steps.

### 6.6 Confirmation run

Split-CIFAR100 at 10 tasks × 10 classes (P = 10 per task) — the minimum P that gives naturalistic **dichotomy structure**. **Not** Split-CIFAR10 or Split-MNIST (P = 2 ⇒ one dichotomy up to sign; see P1 — a task-structure limit, not an estimability one).

---

## 7. Architecture

```
h_m = ReLU(W_m x),    m ∈ {A, B},  width N each (N ≈ 300)
f(x) = Σ_m α_m · u_mᵀ h_m
α_m = 1/(γ_m √N),     η_m = η₀ · γ_m² · N   (with corrected-scaling arm: η ∝ γ^(2/L))
```

Single shared readout, **no task-conditioned head** — a task head absorbs exactly the credit-allocation effect being measured.

### 7.1 Two orthogonal initialization knobs

**Update magnitude (lazy ↔ rich): γ_m.**

**Initial geometry (poor ↔ wealthy): subspace alignment a_m.** Compute the center subspace U_C from the P centers; take W̃'s top-k right singular subspace V_k; interpolate along the **Grassmann geodesic** from V_k toward U_C at parameter a; reassemble W(0) with the original singular values.

> Use a proper geodesic (via principal angles and their SVD parameterization), not linear interpolation with re-orthonormalization — the latter makes *a* nonlinear and non-comparable across seeds.

**Orthogonality proof:** in the Graldi parameterization, weight variance σ² = 1 in both NTP and μP; only output scale and LR schedule differ. Hidden-layer weights at initialization are therefore **identical across γ₀**, so capacity-at-init is a pure function of *a*. **Assert this in a unit test.** It belongs in Figure 1.

*Fallback if the alignment knob fails Gate 2:* Chung's input-dimension manipulation, at the cost of orthogonality with γ.

### 7.2 Configurations (matched total width 2N and parameters)

| # | A | B | Rules out |
|---|---|---|---|
| 1 | dense 2N, γ swept | — | recovers γ₀\* baseline |
| 2 | (a, γ) | (a, γ) | **modularity per se — primary control** |
| 3 | (a_hi, γ_lo) | (a_lo, γ_hi) | **treatment: wealthy-lazy + poor-rich** |
| 4 | (a_hi, γ_hi) | (a_lo, γ_lo) | anti-diagonal — tests whether the *specific* pairing matters |
| 5 | (a, γ) | frozen | lazy dynamics vs. static random features |
| 6 | any of the above, **joint i.i.d. training** | | **sequentiality — headline control** |

Required pattern from #6: heterogeneity helps sequentially and not jointly, on identical data. If it helps jointly, this is ensembling.

---

## 8. Measurement protocol

### Tier 1 — every boundary (cheap)
- Participation ratio, effective rank per module
- Linear CKA between h_m(t) and h_m(t′), all boundary pairs → per-module drift matrix
- Linear decodability of every past dichotomy per module
- **Held-out probe y\***: fresh readout, Johnston & Fusi generalization metric
- Per-module output-variance share; per-module gradient-norm share
- **Manipulation checks (unconditional):** ‖ΔW_m‖/‖W_m‖, per-module NTK change

### Tier 2 — every 4th boundary (expensive)
- Generic capacity α_m (averaged over random dichotomies)
- Retained capacity α_m^j (fixed `y_j`) for each past task *j*
- Full GLUE: R_mf, D_mf, ρ_c, ρ_a, ψ, for current and past dichotomies
- CCGP and shattering dimensionality over the factorial structure

### Primary novel measurement — geometric attribution of forgetting
For each past task *j* and boundary *t*: compute Δα_m^j and attribute it across GLUE components. **Output is an attribution table: how much of the forgetting on task *j* is radius, dimension, center alignment.** This is the paper.

### Secondary — rotation/expansion decomposition
Principal angles between top-k subspaces at *t* and *t′* (rotation = overwriting) reported separately from ΔPR (expansion = accretion). Prediction: rich rotates, lazy at most expands. Robust and cheap; the fallback if GLUE proves too noisy.

### Three-way integration mode (per module, per boundary)
| Mode | Measurement | Interpretation |
|---|---|---|
| Reuse | new task's between-manifold variance captured by the *pre-existing* top-k subspace | transfer |
| Expand | PR increase; top-k principal angles ≈ 0 | accretion, no cost |
| Overwrite | pre-existing top-k rotates; old-task decodability drops | interference |

### Estimation constraints
1. **Match P, M, N across modules.** Capacity is a ratio to ambient dimension.
2. **Anisotropy.** Capacity is not invariant to anisotropic rescaling, and lazy/rich modules differ in scale by construction. Report raw *and* per-module-gaussianized (Wakhloo 2023 preprocessing, `00` §6.3); confirm qualitative agreement. Wakhloo et al. supplement is the reference.
3. **Decomposition form.** The capacity decomposition is the **exact** three-factor identity `α = Ψ_eff·(1 + R_eff⁻²)/D_eff` (ICLR 2026 §B.3) — not an approximation, so attribution is quantitative by construction (`00` §8). The MMCR `α ≈ φ(R√D)` form is a different, wrong parameterization here. The remaining empirical question is only whether *mean-field* capacity holds (`α_sim` vs `α_mf`, `02` §2a); if it fails at some tilt β, capacity at that β is reported as approximate.
4. **Margin.** Legacy code takes κ; Chou et al. work at κ = 0. Estimators are not interchangeable at κ ≠ 0.

### Statistical design
- **Paired initializations:** one shape matrix W̃ per seed, scaled per condition. Every weight *direction* identical across conditions; only scale differs. Converts to paired differences per seed.
- **Shared stream realizations:** generate K streams once; run every configuration on all K.
- Model: `outcome ~ a × γ × s_f × s_r + (1|seed) + (1|stream)`. Effect sizes with intervals; no p-values at small seed counts.
- **Budget rule:** more streams beats more seeds for stream-level hypotheses. ~10 seeds × 20 streams over 40 seeds × 5 streams.

### Noise floors (mandatory, before interpreting anything)
For every measure compute:
1. **Identifiability floor** — measure between two independent seeds of the *same* condition at the *same* timepoint
2. **Trajectory floor** — same seed, same config, different data order only

Any effect must clear both. Derive the minimum detectable effect per measure at the planned seed count. Expect R_mf and ψ to be noisiest; if their MDE exceeds the expected effect, demote them from confirmatory to exploratory *before* writing thresholds.

---

## 9. Implementation steps

| Step | Task | Est. | Notes |
|---|---|---|---|
| 1 | Manifold generator + factorial labeling | 1–2 d | From App. D.1.1; centers random, factors in labeling only |
| 2 | GLUE estimator | 1–3 d | **Start with `schung039/neural_manifolds_replicaMFT`** — gives capacity, R, D, **ρ_c** today. ρ_a and ψ pending Nat Neuro code or own implementation of Algorithm 1 |
| 2b | Estimator validation | 1 d | Reproduce App. B.5 ground-truth recovery: N=1000, P=2, M=200, D 2→10, R 0.8→2. **Non-optional** |
| 3 | Parameterization + architecture | 1–2 d | Per-module (α_m, η_m); Grassmann-geodesic alignment init; unit test asserting identical hidden init across γ |
| 4 | Stream generator | 1 d | Balanced dichotomies; exact-Hamming sampling; ρ_C-controlled arrangement redraw; probe y\*; record full S |
| 5 | Evaluation harness | 2 d | Tier 1/Tier 2 split; unconditional manipulation-check logging |
| 6 | **Cost model** | 0.5 d | Time one full GLUE eval at target (P, M, N, n_t); multiply out the grid; fix Tier-2 interval and n_t **before** Phase 1 |
| 7 | Noise floors | 1 d | 15 seeds, one config, full pipeline |
| 8 | Statistical model + pre-registration table | 0.5 d | Write thresholds *after* Step 7 |
| 9 | Phase 1 | — | The submittable unit |

**Compute note.** GLUE requires a QP per sampled (y, t) pair. Recommended default n_t = 200. At P=16, M=150, N=300 the constraint matrix is 2400×300 per sample. Multiply by modules × boundaries × configs × seeds. Cut n_t and Tier-2 frequency before cutting seeds.

---

## 10. Phases and kill criteria

**Phase 0 — manipulation validation (~2 d)**
Verify: (i) capacity-at-init tracks *a*, flat in γ; (ii) ‖ΔW_m‖ separates by ≳1 order of magnitude across γ; (iii) **time-reparameterization test** — do the two modules' trajectories coincide under rescaling?
*Kill:* (i) fails → fall back to input-dimension wealth manipulation. (ii) fails → parameterization is wrong, stop. (iii) coincide → **H3 is dead**, proceed with H1/H2 only.

**Phase 1 — H1, H2, H6 (homogeneous only)**
γ × *a* sweep, Hiratani 2×2 streams, 8 seeds. Produces the forgetting-attribution table, the generic/retained crossing, and the probe-dichotomy validation. **This alone is a complete NeurReps abstract.** No heterogeneity claims required.
*Kill:* GLUE attributions don't separate by regime → fall back to rotation/expansion decomposition. Generic capacity fails to track the probe metric → report as a negative finding about label-agnostic geometry measures (publishable in this track).

**Phase 2 — H3, H5 (heterogeneous)**
All six configurations, sequential and joint, paired inits, shared streams.
*Kill:* pair mean-γ predicts everything (H5), or joint shows the same effect.

**Phase 3 — H4**
T = 40 at three similarity levels. Half-life of the initial regime difference.
*Most likely to surprise.* A null is publishable.

**Phase 4 — stream variance/predictability.** Deferred to full paper.

---

## 11. Figures (4-page budget = 4 figures, ~1500 words)

1. **Setup + orthogonality.** Fixed manifold arrangement, factorial labeling, dichotomy-as-task, the (a, γ) 2×2, and capacity-at-init confirming *a*/γ orthogonality.
2. **The forgetting attribution.** *(money figure)* Capacity on task 1 across the stream with GLUE decomposition, lazy vs. rich. Side panel overlaying the ultra-rich OOD signature from Chou et al. Fig 7c to show the signatures match.
3. **Generic vs. retained.** Both as functions of γ, with γ₀\* marked and the probe-dichotomy metric overlaid as validation.
4. **Either** heterogeneous vs. homogeneous on the front with per-module division of labour and the γ_eff control, **or** — if Phase 2 fails — the H4 half-life plot, reframing from "heterogeneity helps" to "heterogeneity does not persist; what maintains it?"

Both versions of Fig. 4 are coherent papers.

---

## 12. Risk register

| Risk | Prob. | Impact | Mitigation |
|---|---|---|---|
| Large-γ time-reparameterization kills H3 | Medium | Severe for H3 | Phase 0 test; restrict contrast to γ≪1 vs γ~1; H1/H2 stand alone |
| GLUE alignment measures too noisy at P=16 | Medium | Moderate | Step 7 catches it; fall back to rotation/expansion + ρ_c only |
| Generic capacity fails the label-aware validation | Medium | Reframe | Publish as a negative finding about label-agnostic diagnostics |
| Novelty overlap with SNAP / task-model alignment | Medium | Framing loss | Measure KTA alongside; if inseparable, lead on GLUE attribution alone |
| Alignment knob doesn't produce wealth | Low-Med | Moderate | Input-dimension fallback |
| Wakhloo duality blocks clean attribution | High | Bounded | State as decomposition, not mechanism; report ρ_a and R jointly |
| Compute overrun | Medium | Moderate | Step 6 before Phase 1; cut n_t, never seeds |
| α ≈ (1+R⁻²)/D poor at these parameters | Medium | Moderate | Verify in Step 2b; if poor, attribution is qualitative — declare it |
| Deadline collision with CL4FMAgents (29 Aug) | High | Scope | Plan for Phase 1 only |

---

## 13. Open questions

**Resolvable in pilots**
- Does alignment move capacity-at-init monotonically with adequate range?
- Do generic capacity and KTA dissociate anywhere in the γ × a grid?
- Do `pairwise` and `full_P` GLUE estimates agree at P=16? (estimability is not in doubt — the lab runs pairwise at P=2; this is a comparability check — 04 §C6, `01` Phase 0)
- Over what tilt range β does mean-field capacity hold (`α_sim` vs `α_mf`)? *(The decomposition itself is now an exact identity — no approximation to test; `02` §2a/§2b.)*

**Genuinely open**
- Is heterogeneity of initial geometry sufficient to break symmetry, or is asymmetric readout structure also required? Shared gradients may pull both modules to redundant solutions. Cross-module CKA answers it empirically; no theory predicts which.
- Does the generic/retained trade have a **fixed exchange rate** or a regime-dependent one? Fixed ⇒ a conservation law, and heterogeneity only relocates on the front. Regime-dependent ⇒ heterogeneity may genuinely dominate. This is the deepest question in the plan and nobody has an answer.
- Ecological validity: a factorial dichotomy stream buys identifiability at the cost of representativeness. The Split-CIFAR100 confirmation is a thin hedge and should not be oversold.

**Deferred**
Stream variance/predictability; recurrence via rank-controlled init (now defensible given the Chung/Liu protocol, but the scale-based γ knob confounds richness with spectral radius — use rank); depth; per-unit γ-spectrum; corrected LR scaling as a factor rather than a robustness arm.

---

## 14. Contribution claims, ranked

1. **Geometric attribution of catastrophic forgetting via GLUE** — no precedent
2. **Generic vs. retained capacity** as the geometric form of stability–plasticity — no precedent; created by sequentiality
3. **Forgetting ≡ ultra-rich OOD failure** as one geometric phenomenon — testable unification
4. **ρ_c as the bridge to human MEG data** (Menghi et al.) — computational counterpart to an existing empirical result
5. **Alignment-based wealth manipulation orthogonal to γ** — small methods contribution
6. **Dichotomy-Hamming similarity over a fixed arrangement** — clean readout-similarity isolation, useful independently

Items 1–3 suffice. Heterogeneity (H3) is upside.

---

## 15. Anticipated objections

| Objection | Response |
|---|---|
| "Graldi et al. with different plots" | They measure 1−CKA and accuracy drop; we measure *what changed geometrically*. We also test their own flagged LR confound |
| "Chung et al. applied to CL" | All capacity work is stationary. The generic/retained distinction does not exist in the stationary setting |
| "Synthetic manifolds" | Standard benchmarks lack the **task structure** the design needs (P=2 ⇒ one dichotomy up to sign; no similarity axis/factorial/CCGP). Plus one Split-CIFAR100 confirmation. *(Not an estimability claim — GLUE runs pairwise at P=2.)* |
| "Heterogeneity beating a point estimate is a priori true" | H5 and the joint-training control are load-bearing; H1/H2 stand without any heterogeneity claim |
| "The 2×2 isn't orthogonal" | It is — identical hidden-weight variance across γ. Shown in Fig 1 |
| "This is task-model alignment" | SNAP cited; KTA measured alongside; Chung's own comparison shows alignment measures mis-order initial feature wealth |
| "This is the ICLR 2026 geometry-markers paper" | Theirs is **stationary** (train once, predict *forward* transfer to unseen classes); ours is **sequential** — geometry measured repeatedly along a stream, asking about *retention* of past tasks. Their future-work omits continual/sequential learning; forgetting has no analogue in their setup. Attribution here is mechanistic over the exact three-way decomposition. Our dichotomy streams are label-space shifts, where their geometry–OOD correlation holds (not corruption shifts, where it does not) |

---

## 16. Immediate next actions

- [x] Email Chou / Le / Chung re: GLUE code — **sent 30 Jul 2026**
- [x] Confirm arXiv 2604.27656 provenance — own work, unaccepted, no conflict
- [ ] Install `neural_manifolds_replicaMFT`; reproduce App. B.5 ground-truth checks
- [ ] Read in order: Menghi (Nat Comms 2025) → Hiratani (NeurIPS 2024) → Atanasov (ICLR 2025) → Wakhloo (PRL 2023 + supplement) → Johnston & Fusi (Nat Comms 2023) → SNAP
- [ ] Implement Phase 0 including the time-reparameterization test
- [ ] Cost model before committing to Phase 1 grid
