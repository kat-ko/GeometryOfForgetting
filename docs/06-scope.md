# Scope — what this project tests, and what it does not

```
Status: ratified. Read before writing any paper-facing text.
Audit:  scripts/audit_scope.py -> results/audit_scope.json (§1 checklist, evidence-backed)
```

This document exists because the monorepo contains a sibling project,
`a1b2_modular`, whose vocabulary overlaps this one's and whose scope does not. The
overlap is a live source of drift, so the boundary is written down rather than
remembered.

## 1. This project does not test modularity effects

Every registered hypothesis is a claim about **learning regime × geometry**, tested on
**homogeneous** configurations:

| | claim | manipulated |
|---|---|---|
| H1 | regime-specific attribution of forgetting to geometric channels | γ |
| H2 | generic/retained capacity trade | γ |
| H6 | similarity-corner effect on geometric disruption | stream 2×2 |
| H1d | signed-ρ_c decorrelation rate | γ |

The manipulated variable is γ, plus the homogeneous `a`-sweep. Modularity claims —
heterogeneous pairs, division of labour, H3, H5 — belong to the **full paper** and are
deliberately deferred. The reason is sequencing, not disinterest: the prior paper
(arXiv:2606.17889) showed modularity's benefit is *conditional on the geometry*, so the
geometry is characterised first.

## 2. Why C2 (two identical modules) is in the grid anyway

Three reasons, none of which is "testing modularity".

**It is a control, not a hypothesis.** C1 vs C2 isolates whether block structure per se,
at matched width and parameter count, changes any geometric result. The expected answer
is **no difference**, and that expectation is what licenses attributing H1/H2/H6 to γ
rather than to architecture. A C1/C2 divergence would be a confound to report, not a
finding to claim.

**It banks the homogeneous baseline for the sequel.** Per-module measurement on C2
(cross-module CKA, per-module gradient and output-variance shares) is what the full
paper's H3 analysis will need as its null, and it costs approximately nothing to collect
now.

**The symmetry question rides along observationally.** Do identical modules under shared
gradients stay redundant, or does anything break symmetry? Unregistered, exploratory
only, and reported as such.

## 3. Consequences for outputs

- **"Module A/B" in any figure means the two halves of the homogeneous control**,
  measured separately per spec. It never means two functionally distinct modules.
- In paper-facing text the abstract barely says "module". C2 appears once, as *"results
  are unchanged under a matched two-module partition (App. X)"*.
- **If draft text starts making claims *about* modules, that is scope drift — flag it,
  do not write it.**
- Nothing from the `a1b2_modular` vocabulary — routing, task-conditioned heads,
  task-ID input, comms, bandwidth, `mod-shared`/`mod-feature` — belongs in this
  project's code path or outputs. `audit_scope.py` enforces this over the import graph
  that `run_phase1.py` actually pulls in, by scanning **identifiers** rather than text:
  an early text-based version flagged three docstrings (`core.py`'s "estimator-routing
  table", `alignment.py`'s "Gate 2", `_par.py`'s "memory-bandwidth-bound") while a real
  `route()` could have hidden in a comment. Machinery is identifiers, so identifiers are
  what the audit reads.

## 4. One caveat the audit surfaced, recorded here because it changes what a label means

For `a > 0`, the two control modules are **not independent draws** — `build_model` passes
`aligned_init(base, U_C, spec.a)` with identical arguments to both, so both receive the
same array, and with identical configs, a shared readout and shared gradients they stay
bitwise identical for the whole run (verified: `max|W_A − W_B| = 0.0` at init and after
300 steps; at `a = 0` the same measurement gives 4.9). This affects the 320 `a`-sweep
arms and not the 960 γ arms, so **no headline result is touched** — H1, H2, H6 and H1d
are all measured at `a = 0`.

It does not corrupt any geometry, because every measurement is per-module
(`manifold_representation(points, module)`): module A's numbers are right and module B's
are an exact copy. What it does break is treating A and B as two samples on those arms —
pooling both halves the apparent standard error without adding information, and
cross-module CKA is 1.0 by construction, so the symmetry question of §2 is unanswerable
on the `a`-sweep. Accordingly, on `a > 0` arms: report module A only, and exclude them
from the C1/C2 control and from any CKA analysis.
