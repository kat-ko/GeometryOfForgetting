# Reference sheet — what GLUE refined vs prior mean-field capacity theory

```
Source:       Chou, Kim, Arend, Yang, Mensh, Shim, Perich & Chung, "Geometry
              Linked to Untangling Efficiency (GLUE)", bioRxiv 2025
              (doi 10.1101/2024.02.26.582157), main text §Intro + §Results.
Version:      bioRxiv 10.1101/2024.02.26.582157 v2
Derived from: bioRxiv v2 rendered full text (main text only). The new capacity
              formula and its derivation are in "Supplementary Information S1",
              which is NOT yet extracted (see gap flag below).
Transcribed by: agent (Opus 4.8), 2026-07-31
Verified by (human): ____________          # blank until checked
Status: main-text assumption facts agent-verified against bioRxiv text; the
        project-implication section is AGENT INFERENCE for human confirmation;
        the S1 formula is NOT captured. Code may not cite until verified.
```

Why this sheet exists (F3): our primary estimator `replicaMFT` implements the
**prior** mean-field theory that GLUE was written to improve on. If GLUE relaxed
assumptions `replicaMFT` makes, replicaMFT numbers may be **systematically** biased,
not just "a different version." This sheet records which assumptions changed.

---

## What GLUE refined (verified from main text)

GLUE builds on perceptron capacity and its manifold-capacity extension (refs [20,21]
in GLUE = the Chung/Cohen mean-field lineage that `replicaMFT` implements). Quoting
the main text:

> "previous theoretical work [20, 21] relies on **two critical simplifying
> assumptions — Gaussian correlations between manifolds and random task labels —
> which often fail in biological datasets by ignoring higher-order correlations and
> task-specific structures."

GLUE's stated advance:

> "We **eliminated all the mathematical assumptions** inherent in previous
> theoretical approaches … we derived a **new manifold capacity formula** which …
> can now be expressed as a closed-form formula **directly dependent on the data and
> a specific choice of task label (yµ)**, whereas the formulas from previous work
> rely on assumptions of random directions [20], pre-processing [28], or
> transformations [21] of the data and task labels."

So the two relaxed assumptions are:

| # | Prior assumption (in `replicaMFT`) | GLUE change |
|---|---|---|
| A1 | **Gaussian correlations** between manifolds | dropped — handles higher-order correlation structure |
| A2 | **Random task labels** | dropped — capacity is now a closed form in the *specific* label `yµ` |

Also recorded (main text §definitions): two capacity families —
`α_sim` (simulation; intuitive but "mathematically ad hoc") and
`α_mf` (mean-field; analytically tractable, but its accuracy depends on exactly
A1/A2). This is the same `α_sim` vs `α_mf` split used by `02` §2a.

---

## Implication for THIS project — AGENT INFERENCE, confirm before relying

Mapping A1/A2 onto our design (not stated in GLUE; my reasoning):

- **A2 (random labels).** Our **generic capacity** (§7) averages over *random*
  dichotomies — so A2 is *satisfied* and replicaMFT should be sound there. But our
  **retained capacity** and the **forgetting attribution** (§7–§8) fix the label at a
  *specific* task `y_j`. That is exactly the regime GLUE's label-dependent formula
  was built for and where the old random-label formula is expected to be weakest.
  → **Prediction: replicaMFT retained/attribution numbers may carry a systematic bias
  that GLUE would remove; generic-capacity numbers should be comparatively safe.**
  → **Not in conflict with 04 §C4 (fixed-`y` is a legitimate analyst choice).** C4
  concerns the estimator's *definition*; A2 concerns mean-field *accuracy* under a
  degenerate (fixed-label) ensemble — an empirical question, measured exactly by the
  `β = ∞` point of the `02` §2a sweep (`05` §3.4). The two documents describe
  different levels and are compatible.
- **A1 (Gaussian correlations).** Our synthetic generator builds correlations via
  Gaussian AR + Cholesky (`00` §1.2) — Gaussian by construction, so A1 is ~satisfied
  for the synthetic stream. For the **Split-CIFAR100** confirmation (naturalistic),
  higher-order correlations could bite and A1 may matter.

If correct, this says: the §2a mean-field gate is necessary but **not sufficient** —
it validates `α_mf` on *random-label* generic capacity, but the retained/attribution
path (the money figure) is where GLUE-vs-replicaMFT divergence would live. Worth an
explicit `α_sim` vs `α_mf` check at a **fixed** dichotomy, not only random ones.

---

## Gap — required before implementing full GLUE

The actual new closed-form formula, and the definitions of ρ_a and ψ under the
relaxed assumptions, are in **Supplementary Information S1**, not extracted here.
`glue-algorithm.md` / `glue-sign-conventions.md` (human-owned) must be built from S1
+ the ICML Algorithm, not from this sheet. This sheet answers *which assumptions
changed*, not *what the new estimator is*.
