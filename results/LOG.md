# Build / decision log

Append-only. Newest entries at the bottom of each day.

---

## 2026-08-10

### Step 0 — Chou v2 PDF integrity check (per 05 brief, pre-reconciliation)

`papers/Feature Learning beyond the Lazy-Rich Dichotomy - Insights from
Representational Geometry.pdf` (re-downloaded arXiv:2503.18114v2):

- size = 10,494,734 bytes — **not** the truncated 524,288 ✓
- trailer = `%%EOF` present ✓
- extracts via `gs -sDEVICE=txtwrite` (233,239 chars) ✓
- algorithm captions confirmed from the complete PDF:
  - Algorithm 1 = "Estimate simulated manifold capacity" (α_sim bisection)
  - Algorithm 2 = "Estimate manifold capacity and effective geometric measures"
    (the geometric estimator; α, R, D, ρ_c, ρ_a, ψ)

**PASS.** D1 relabel (Algorithm 1 → Algorithm 2 for the geometric estimator) is
now confirmed against the primary PDF, not only the arXiv HTML. Reconciliation
may proceed.

### Reconciliation pass (05 §1) — files changed

Single logical change-set (the "one commit" of 05 §1.2). No code, docs only.

- **04-spec-corrections-iclr2026.md** — dated erratum header (Algorithm 1→2 for
  the geometric estimator); body left unchanged as a correction-log record (§1.1).
- **00-math-spec.md**
  - §6 heading → Algorithm 2; NB that Algorithm 1 = α_sim.
  - §6.1 → rewritten in stacked-matrix (P-space) form: anchors as rows of
    `S∈R^{P×N}`, `S_y=diag(y)S`, exact `a/b/c` scalars with **P×P** pseudo-inverses;
    measures α, D_eff, R_eff, Ψ_eff; ρ_c/ρ_a/ψ (abs, unnormalized, cross-index, C3);
    added `rho_c_signed` H1d instrument and `rho_convention`.
  - §6.2 → **exact** three-factor identity `α=Ψ_eff·(1+R_eff⁻²)/D_eff`; UNCERTAIN
    marker removed; algebraic closure shown.
  - §6.3 → `raw`/`standardized` → `raw`/`gaussianized` (Wakhloo 2023).
  - §7 → generic / **retained** / **tilted(β)**; "aligned" retired; D2 staging.
  - §8 → **log-space additive** attribution, zero residual (3 factors).
  - §13 → cost model re-derived for P×P linear algebra; pairwise vs full_P note.
- **02-validation-suite.md** — §2 collapsed to the single §2a gate (mean-field
  validity) with the **β axis** added (P×N×β; β=∞ ≡ retained); §2b deleted
  (exact identity); §2c Ψ_eff identity check (GLUE-dependent) added; noise-floor
  list updated (α_retained, Ψ_eff, R_eff/D_eff, rho_c_glue/rho_c_signed).
- **01-experiments.md** — config: `estimation_mode`, `rho_convention`, `beta`,
  `ccgp_enabled`, `capacity_modes: [raw, gaussianized]`; Phase 0 pairwise-vs-full_P
  row; deleted-§2b reference fixed; pre-reg H1a (log-space shares), H1b, H2b/H2c,
  Fig 3 → retained; §3.3 P=2 restated as task-structure.
- **AGENTS.md** — §3 terminology table: retained/tilted, psi_eff vs
  center_axis_alignment, rho_c_glue/rho_c_signed; "aligned" retired; §6 approximation
  bullet → exact-identity resolved.
- **PROJECT.md** — P1 restated (task-structure, not estimability); ICML entry →
  Algorithm 2; ICLR-2026 added as load-bearing (§5.1) and as a §15 objection (C7
  distinctions); §6.6/§13/§8 estimability + approximation wording updated;
  "aligned"→retained, "standardized"→gaussianized throughout.
- **03-references.md** — version-drift note corrected; ICML entry → Algorithm 2;
  ICLR-2026 row added to §A/§H; §E.4 pseudocode col → Algorithm 2 and convention
  **reversed to C3** (rho_c_glue primary); §E.5 marked resolved by stacked-matrix
  form; §E.6 downgraded (retained is legitimate; tilt is enhancement); §D/§F add
  ood-generalization-geometry (reference-only) and glue-decomposition.md; §G
  resolved-numbering corrected; §B Montanari phrasing.
- **reference/README.md** — `Version:` field added to the provenance template;
  status table updated (glue-decomposition added, glue-algorithm→Alg2,
  glue-refinements status).
- **reference/{manifold-generator, validation-settings, parameterization,
  cl-metrics, glue-refinements}.md** — `Version:` field added; manifold-generator
  naming flag marked RESOLVED; glue-refinements A2↔C4 non-conflict cross-ref line.
- **reference/glue-decomposition.md** — NEW agent-drafted sheet (ICLR 2026 §B.3,
  exact 3-factor identity + a/b/c), awaiting human verify.

Verified vs inferred: all scientific substitutions trace to sources verified
2026-08-10 (Chou v2 PDF Step-0 check; ICLR 2026 §B.3 for the identity/a-b-c). New
reference sheets carry blank `Verified by (human)` and may not be cited by code
until signed off.

### Git — initial import + reconciliation (two commits, into the parent monorepo)

Repo root is the **parent monorepo**
`/home/kat/workspace/Structure-Function-Analysis-of-Network-Topologies` (holds
unrelated projects). `continual_geometry/` was **entirely untracked**; verified
no nested `.git`, no leftover `.git` in `third_party/`, subproject not ignored.
Committed as two objects for legible history (per Kati's steer):

- `32b8b65e` — **import vendored deps + gitignore**: `third_party/` source
  (replicaMFT, correlated_capacity, VENDORED.md, patches/), `papers/.gitignore`,
  and a new `continual_geometry/.gitignore`.
- `2099b0af` — **spec reconciliation (05 §1)**: docs-only + `src/glue/adapters/README.md`
  + this log.

**Gitignore conflict resolved:** the parent `.gitignore` has a blanket `results`
rule that would drop `continual_geometry/results/`. Added an **anchored**
`!/results/` in `continual_geometry/.gitignore` so this project's audit trail
(LOG.md, future *.json evidence) is tracked while vendored
`third_party/**/results/` artifacts (~78 .npy/CSV) stay ignored. `papers/` PDFs
(48 MB) remain ignored.

**Deferred (not part of §1):** vendoring `ood-generalization-geometry` (Gaussianization)
moved to §2.2 (its own commit, when `preprocessing.py` is the active task) — it is
figure-repro code needed only at preprocessing, and keeps the reconciliation diff clean.

*(This entry post-dates the two commits above; fold it into the next daily
checkpoint commit — §8.)*

---

## 2026-08-11

### Amendment before §2 — ρ_c two-role rule + H1d flag-and-stop

Per Kati ratification: **"primary" is two jobs**. `rho_c_glue` = reporting /
comparability with published GLUE; `rho_c_signed` = **required H1d instrument**
(under `|·|`, H1d is untestable). `rho_convention: both` is **mandatory** — no
path may reduce to one. Applied in `03` §E.4 (already had the split), `00` §6.1,
`01` config + H1d pre-reg (already named `rho_c_signed`), `AGENTS` §3, `05` §2.2
and §6 item 1, `glue-decomposition.md`.

**Escalation live:** if `manifold_analysis_corr` does not expose anchor centers
`s⁰_μ`, that is flag-and-stop (adapter extraction vs `third_party/patches/` vs
drop H1d) — **not** degradation to abs-only.

### Spot-check (load-bearing reconciliation)

1. `00` §6.2 — exact identity present; UNCERTAIN gone from capacity approx;
   two-factor only mentioned historically.
2. `02` — §2b deleted; §2a has β axis; no live cross-ref treating §2b as a gate.
3. "aligned capacity/margin" retired in AGENTS/00/01 pre-reg; remaining hits in
   `04` (correction log) and `05` (historical apply-order) only; PROJECT leftovers
   cleaned.
4. `00` §8 — log-space additive, zero residual; Wakhloo caveat (report ρ_a with R)
   intact.
5. `glue-decomposition.md` — `Verified by (human)` still blank.
6. `04` erratum header present; body unchanged.

### Gitignore confirm

`git check-ignore -v continual_geometry/results/LOG.md` → exit 1 (not ignored).
Probe JSON `results/_gitignore_probe.json` appeared in `git status` (then removed).

### §2 start — `src/manifolds/`

Scaffolded: `generator.py`, `labeling.py`, `dichotomies.py`, `streams.py`,
tests under `tests/`, `pyproject.toml` + `requirements.txt`. Local `.venv` via
`uv` (gitignored). **`pytest`: 15 passed** (`02` §5 stream/dichotomy suite +
unit-norm + ρ_C monotonicity).

### replicaMFT capability audit — three findings, one structural

Triggered by the §2.2 verify-first anchor-center check. Read
`mftma/manifold_analysis_correlation.py` end to end. Framing per Kati: these are
**properties of the replica mean-field theory that GLUE refines**, not bugs.

- **F2 (structural, decisive).** `manifold_analysis_corr` takes no `y` because the
  dichotomy average is integrated out *analytically* in the replica derivation —
  `y` was eliminated before the code was written. GLUE samples `(y, t)` and
  averages *numerically*, which is precisely why it supports an analyst-chosen
  ensemble `Y`. ⇒ `α_mf` gives **generic capacity only, permanently**. Retained,
  tilted, and per-task attribution are not extractable at any effort. This is
  assumption A2 of `glue-refinements.md`, seen in code.
- **F1 (spec error we were carrying).** `res_coeff0` = mean over `μ≠ν` of the
  **absolute cosine** between global-mean-subtracted **manifold** centers (raw
  class means) in a `P−1` basis. Abs *and* normalized, on manifold centers not
  anchor centers ⇒ matches **neither** `rho_c_glue` nor `rho_c_signed`.
  Renamed `center_cos_abs`; `00` §6.1 corrected.
- **F4 (follows from F2).** `R_M`/`D_M` are the PRX-2018 eq-28/29 functionals, not
  `R_eff = √(E[c]/E[b−c])` / `D_eff = E[b]/P`. The D3 `Ψ_eff` identity is therefore
  **expected to fail**; reclassified gate → diagnostic (`02` §2c).
- Also: anchor centers are formed (`s_all.mean(axis=1)`) then discarded, in a
  per-manifold frame (centering, norm division, per-manifold QR re-basis when
  `N > M`) ⇒ cross-manifold `⟨s⁰_μ, s⁰_ν⟩` unrecoverable from returns.

**Decision (Kati): build the GLUE core ourselves.** New `src/glue/core.py`,
estimator string `glue_core@<sha>`, firewalled like any other estimator. Scope:
α, D_eff, R_eff, Ψ_eff, anchor ρ_c in **both** conventions. Not ρ_a / ψ_{μν}.
Nothing in `third_party/` modified — clears the firewall. Hard deadline: validated
against B.5 recovery by **end of Day 14**, else Figure 2 falls back to the
rotation/expansion decomposition (principal angles vs ΔPR).

### Spec corrections 1–5 applied (pre-code commit)

1. `00` §6.1 — replicaMFT capability block: label-invariance, `res_coeff0` ≠ ρ_c,
   `R_M`/`D_M` ≠ `R_eff`/`D_eff`, anchor centers discarded.
2. `00` §7 estimator-routing table + `01` Phase 0 — `α_mf` generic-only; retained
   and tilted route through `α_sim` and `α_core`.
3. `02` §2a — β axis is `α_sim` vs `α_core`; `α_mf` participates at β = 0 only.
   §2c reclassified to diagnostic.
4. Harmonic-mean α aggregation pinned in `00` §6.1, `01`, `AGENTS` §4.
5. `third_party/VENDORED.md` — capability limit recorded as a verified finding;
   replicaMFT downgraded to generic-α cross-check; `glue_core` added to the
   version-string table.

### FLAG — technical correction to the build route (needs Kati's eye, not a stop)

The instruction says anchor points "via `maxproj` composed **per-manifold** in the
ambient frame". Composing per-manifold reproduces exactly the structure that makes
`α_mf` label-invariant, so it would silently re-inherit the limitation we are
escaping. Reason: in ICLR B.3, `S_y = diag(y)·S`, and `diag(y)` **cancels
algebraically** in all three quadratic forms —

```
a = (S_y t)ᵀ (S_y S_yᵀ)† (S_y t) = (S t)ᵀ (S Sᵀ)† (S t)     since diag(y)² = I
```

so `a`, `b`, `c` are invariant to `y` *given* `S`. **All** `y`-dependence must
therefore enter through the anchor points themselves, which it does only if they
come from the **joint** QP over all `P·M` constraints with rows `y_μ z^μ_j`
(Algorithm 2 / `00` §6.1) — where the shared separating direction couples the
manifolds. Implementing joint; `maxproj`/`minimize_vt_sq` still usable for the
`P = 2` cross-check. No vendored code touched either way.

### `src/glue/core.py` built and **validated** — both targets pass

Joint anchor QP at κ=0: `min ½‖v−t‖²  s.t. Gv ≤ 0`, rows of `G` = `y_μ z^μ_i`.
KKT gives `v = t − Gᵀλ`, so the duals are the NNLS problem
`λ = argmin_{λ≥0} ‖Gᵀλ − t‖²` — solved with `scipy.optimize.nnls`, no cvxopt and
no vendored code in the path. Anchors are the unsigned dual-weighted averages
`Σλ z / Σλ` per `00` §6.1.

**Convention resolved by a hard check, not by choice.** A manifold with zero dual
mass gets a **zero row**. That reproduces the rectification in the replica
expression `α = 1/E[(⟨t,ŝ⟩)₊²]`: for `P` orthonormal point manifolds ~half are
active, `E[a] = P/2`, so `α = 2` — the known κ=0 point-manifold capacity.
`test_point_manifold_capacity_is_two` measures **α = 2.00 ± 0.12**, active
fraction 0.50. This is the load-bearing test for the whole formulation.

**Also found and fixed:** ensembles must use an RNG stream for `y` independent of
the one for `t` (`rng.spawn(2)`), or comparisons across β differ by Monte-Carlo
noise on `t` as well as by `Y`. At `n_t = 150` that noise is ~10% of α — larger
than the effects being compared. Tilt-β=50 and retained now agree to <1%.

**Gate 1 — §B.5 ground-truth recovery** (P=2, M=200, N=1000, n_t=200, 3 seeds;
`results/glue_core_recovery.json`, 162 s). Recovery is close to identity on all
three axes, with clean cross-axis separation:

| ground truth | 2 / 0.8 / 0.0 | 4 / 1.0 / 0.2 | 6 / 1.4 / 0.4 | 8 / 1.7 / 0.6 | 10 / 2.0 / 0.8 |
|---|---|---|---|---|---|
| `D_eff` vs D | 2.25 | 4.07 | 5.60 | 6.98 | 8.17 |
| `R_eff` vs R | 0.87 | 1.02 | 1.34 | 1.56 | 1.77 |
| `rho_c_glue` vs ρ_C | 0.04 | 0.22 | 0.41 | 0.61 | 0.81 |

`D_eff` is pinned at 4.07–4.08 across **both** the R and ρ_C sweeps — the axes are
separable. `α` falls monotonically on all three. `Ψ_eff ∈ [0.85, 0.91] ⊂ [0,1]` ✓.
`identity_residual < 1e-8` everywhere (exact by construction, so this is a
transcription check on the file).

**F4 resolved with a number.** The cross-sheet ambiguity — does axis/center
correlation land in `R` or in `D`? — is now measured at our parameters: sweeping
ρ_C from 0 → 0.8 moves `R_eff` 1.02 → 1.82 and leaves `D_eff` at 4.07 → 4.08.
**Correlation is absorbed entirely by the radius**, as Wakhloo's duality predicts,
not by dimension. §8 attribution must keep reporting ρ_c alongside R.

**Gate 2 — α channel vs `α_sim`.** P=8, d=200, D=4, R=1, M=60:
`α_sim = 0.471` vs `α_core = 0.433` (8%, and `α_sim` is coarse here — `N_c = 17`).

**Deadline met on Day 3, not Day 14.** The rotation/expansion fallback for
Figure 2 is not needed.

### `src/glue/adapters/simcap.py` — and a correction to the α_sim assumption

**`manifold_simcap_analysis` is generic-only as shipped.** `compute_sep_Nc_general`
(`:180`) draws its *own* random balanced labels internally, so the vendored entry
point cannot produce retained or tilted capacity either — the "α_sim runs a real
SVM with real labels" claim holds for `check_data_separability_general` (which
takes explicit labels and *is* public), not for the entry point above it. The
adapter therefore owns the bisection: injected `Ensemble`, injected
`numpy.random.Generator` (which also removes the global `np.random.seed` violation
of `AGENTS.md` §4), vendored SVM call unchanged. Nothing in `third_party/` touched.

Figure 3 is still safe — but via this adapter, not via the vendored function.

### `02` §2c Ψ_eff diagnostic — ran, mismatched as predicted

`results/psi_eff_diagnostic.json`, n_t = 200:

| | P=8 | P=16 |
|---|---|---|
| `Ψ_eff` via replicaMFT identity | 0.694 | 0.665 |
| `Ψ_eff` direct `E[c]/E[a]` (core) | 0.772 | 0.768 |
| relative gap | **10.1%** | **13.4%** |
| `D_M` vs `D_eff` | 3.26 / 3.70 | 3.32 / 3.58 |
| `R_M` vs `R_eff` | 1.069 / 1.001 | 0.967 / 0.992 |
| `α_mf` vs `α_core` | 0.399 / 0.417 | 0.415 / 0.433 |
| `center_cos_abs` vs `rho_c_glue` / `rho_c_signed` | 0.143 / 0.044 / 0.025 | 0.074 / 0.040 / 0.000 |

Confirms F4 (the PRX functionals are not the GLUE ones) and F1 (`res_coeff0` is a
third quantity, ~3× `rho_c_glue` here). Note `α_mf` and `α_core` agree to ~4% on
the **generic** channel — exactly the cross-check role replicaMFT now has.

### Environment

Added `scipy` (core dependency). Vendored estimators need `cvxopt`, `autograd`,
and **`pymanopt==0.2.5`** — replicaMFT imports `pymanopt.solvers`, removed
upstream in pymanopt 1.0, so that pin must not be raised. Recorded in
`requirements.txt` and as the `vendored` extra in `pyproject.toml`.
**`pytest`: 30 passed.**

### Still open

- `preprocessing.py` and ood-geometry vendoring remain deferred, per instruction.
- `center_policy` (`"all"` vs `"active"` when averaging anchors over samples where
  a manifold is inactive) is a flagged convention; `"all"` is the literal §B.3
  reading and passes recovery, so it stays default. Recorded in the result key.
- ρ_a and ψ_{μν} deliberately out of scope.

---

### P1 — cost model: Phase 1 fits in **2.8 h wall**, no cuts needed

`scripts/run_cost_model.py` → `results/cost_model.json`. Machine: 256 logical
cores (2 × AMD EPYC 7763), 503 GB. Workers pinned to 1 BLAS thread each.

**The 18 s/eval figure was a multithreaded-BLAS number.** Pinned to one thread the
base point (P=16, M=150, N=300, n_t=200) costs **52.4 s/eval**. That is the right
number to plan with, because throughput is set by total core-seconds, not by
per-eval latency, and one process per core beats few processes with threaded BLAS.

Grid as specified — T=16, Tier-2 every 4th boundary → 40 evals/run;
6 γ × 4 conditions × 10 streams × 5 seeds = 1,200 runs = **48,000 evaluations**:

| `n_t` | s/eval | serial | wall @248 workers | α CV |
|---|---|---|---|---|
| 50 | 13.1 | 175 h | 0.70 h | 3.23% |
| 100 | 26.2 | 349 h | 1.41 h | 2.82% |
| **200** | **52.4** | **699 h** | **2.82 h** | **1.87%** |
| 400 | 104.8 | 1397 h | 5.63 h | 1.03% |
| 800 | 209.6 | 2795 h | 11.27 h | 0.72% |

**Decision: keep `n_t = 200`. The cut order is not invoked** — no reduction to
Tier-2 interval, retained-tasks-evaluated, seeds, or streams. Even `n_t = 400`
(CV 1.0%) is affordable at 5.6 h if a measure turns out to need it.

Noise floors at the base point (12 replicates, `n_t = 200`), for `02` §6:
α 1.87%, D_eff 1.26%, R_eff 0.50%, Ψ_eff 1.39%, ρ_c 0.97%.

Scaling, for future sizing: cost is superlinear in `P` (≈ P^2.1: 70 / 288 /
1285 ms per sample at P = 8/16/32, M=150, N=300) and ≈ M^1.6, and grows with `N`.
`P = 32` would have cost ~4.5× — worth noting given that §2a came out against
needing it. `α_sim` is cheap by comparison (1.4–35 s per full bisection).

### P2a — `center_policy`: `"all"` wins, on evidence

Full B.5 protocol under both policies, at `n_t ∈ {200, 1000}`
(`results/glue_core_recovery.json`, 180 estimations, 69 s). Mean absolute relative
recovery error:

| policy | `D → D_eff` | `R → R_eff` | `ρ_C → rho_c_glue` |
|---|---|---|---|
| `"all"` | 10.3% | 7.0% | **3.7%** |
| `"active"` | **9.6%** | **6.4%** | 8.6% |

`"active"` wins by under a percentage point on `D` and `R` — comparable to seed
noise — while `"all"` wins by more than 2× on `ρ_C`, consistently at both `n_t`.
`ρ_C` is also the most direct of the three tests since the generated correlation
is exactly known, and `"active"` systematically over-estimates `ρ_c`
(0.043/0.239/0.434 vs truth 0/0.2/0.4) — the measure H1d turns on.
**Default stays `"all"`, now with evidence rather than by default.** Conclusions
about D and R are robust to the choice; conclusions about ρ_c are not. Kept in the
result key either way.

### P2b — scale compression is finite-`M`, and we found the mechanism

Three candidate causes, all tested (`results/scale_compression.json`):

1. **Not `(y,t)` Monte-Carlo.** `n_t` 200 → 1000 moves `D_eff` at `D=10` from 8.17
   to 8.19, `R_eff` at `R=2` from 1.773 to 1.785. Gap does not close.
2. **Not an artifact of B.5's `P = 2`.** Mean |rel err| on `D`: 13.1% (P=2),
   14.4% (P=8), 16.5% (P=16). Nearly flat.
3. **It is finite sampling of each manifold.** `D_eff/D` at P=2, N=1000:

| `M` | 50 | 100 | 200 | 400 | 800 |
|---|---|---|---|---|---|
| D=2 | 1.112 | 1.118 | 1.123 | 1.127 | 1.130 |
| D=6 | 0.846 | 0.898 | 0.934 | 0.955 | 0.969 |
| D=10 | 0.671 | 0.750 | 0.817 | 0.865 | 0.897 |

The deficit falls as ≈ `M^-0.45` and converges to 1. Anchors are extreme points of
a finite sample, and a finite sample under-represents a high-dimensional
manifold's extent; higher `D` needs more points for the same coverage. The ~12%
over-report at `D = 2` is a separate, `M`-independent effect.

**Consequence — a caveat, not a fix.** 5% deficit at `D = 10` would need
`M ≈ 4000`, unaffordable at Phase-1 scale. But the bias is monotone (signs and
rank orders preserved), a fixed transform at fixed `M` (which we hold constant),
and it **compresses** dynamic range — so estimated `Δ log D_eff` is biased *toward
zero* and the §8 dimension channel is **conservative**: a real contribution can be
understated, never manufactured. Stated in `00` §6.1 and in the paper.

### P2c — B.5 presentation corrected

The earlier table read as a confounded diagonal because three independent sweeps
were compressed into one row-per-measure. The script and the sheet now report each
sweep separately with the off-axis knobs stated (`D` and `R` each swept at
`ρ_C = 0`; `D = 4, R = 1` held while `ρ_C` sweeps). The underlying runs were always
independent — this was a reporting defect, not a design one.

### P3 — §2a at β = 0: **`P = 16, N = 300` confirmed, escalation not triggered**

`scripts/run_gate_2a.py` → `results/gate_2a.json`. 81 estimations, 411 s.

| P | N | α_sim | α_core | α_mf | core–sim | mf–sim |
|---|---|---|---|---|---|---|
| 8 | 300 | 0.483 ± 0.052 | 0.440 ± 0.006 | 0.388 | 8.9% | 19.8% |
| 8 | 600 | 0.449 ± 0.046 | 0.439 ± 0.013 | 0.387 | 2.2% | 13.9% |
| 8 | 1200 | 0.436 ± 0.043 | 0.430 ± 0.016 | 0.387 | 1.5% | 11.3% |
| **16** | **300** | 0.420 ± 0.017 | 0.434 ± 0.002 | 0.413 | **3.4%** | **1.6%** |
| 16 | 600 | 0.440 ± 0.009 | 0.440 ± 0.004 | 0.414 | 0.0% | 6.0% |
| 16 | 1200 | 0.460 ± 0.013 | 0.437 ± 0.006 | 0.413 | 4.9% | 10.1% |
| 32 | 300 | 0.437 ± 0.014 | 0.440 ± 0.005 | 0.423 | 0.7% | 3.1% |
| 32 | 600 | 0.451 ± 0.007 | 0.433 ± 0.001 | 0.421 | 4.0% | 6.6% |
| 32 | 1200 | 0.440 ± 0.019 | 0.439 ± 0.004 | 0.422 | 0.1% | 3.9% |

`α_core` vs `α_sim`: 2.9% mean, ≤5% everywhere except (P=8, N=300), no systematic
sign. At the design point both `α_core` (3.4%) and `α_mf` (1.6%) sit inside
`α_sim`'s own seed scatter. **The `N`→600→1200 and `P`→32 escalation ladder is not
invoked; the five-factor redesign is off the table.**

**Caveat recorded rather than buried:** the clean `O(1/N)` trend appears only at
`P = 8` for `α_mf` (19.8 → 13.9 → 11.3%). Elsewhere there is no trend, because
`α_core` is flat in `N` (< 2% variation at fixed geometry, seed SD ~0.004) while
**`α_sim` itself drifts upward with ambient `N`** (0.420 → 0.440 → 0.460 at P=16)
with 3–10× larger seed SD. The generated geometry is `N`-independent, so that
drift is a property of the simulation estimator — its bisection range and
projection statistics both scale with `N`. Part of the residual at large `N` is
therefore `α_sim`'s bias, not `α_core`'s error. Neither is asserted correct; both
are reported. This does not affect the design-point conclusion, which is at the
smallest `N`.

### Documentation

`docs/reference/glue-core-validation.md` created — the paper's methods paragraph
and the reviewer answer to "you implemented the estimator yourself". Records, with
numbers: the α=2 exact limit, B.5 recovery per axis, the α_sim/α_mf cross-check,
the center_policy decision, and the compression limitation with its mechanism.
`Verified by (human)` blank.

### Follow-ups closing the sheet (`results/estimator_followups.json`)

**1. `α_sim`'s N-drift is an artifact — confirmed, not inferred.** Re-ran with `N`
raised by **zero-padding one point cloud**: identical geometry, isometric
embedding, capacity invariant by construction.

| P=16, M=60, 5 seeds | N=300 | N=600 | N=1200 | drift |
|---|---|---|---|---|
| `α_sim` | 0.434 ± 0.025 | 0.435 ± 0.016 | 0.461 ± 0.014 | **+6.2%** |
| `α_core` | 0.439 ± 0.006 | 0.434 ± 0.008 | 0.437 ± 0.004 | **−0.3%** |

`α_core` is invariant as it must be; `α_sim` is not. `02` §2a amended: `α_sim` is
the reference standard **at the design point** (smallest N, best agreement,
tightest scatter), not unconditional ground truth — large-`N` disagreement must
not be charged to `α_core` by default.

**2. Cost of `M` measured at project scale: ≈ `M^1.68`** (242 / 1173 / 4067
ms/sample at M = 150 / 400 / 800, P=16, N=300). Full grid: 2.8 h / 13.7 h / 47.4 h.
`M = 150` justified as the main-grid point; **a high-`M` robustness arm at `M = 800`
costs 2.1 h over 4.5% of the grid** and is now planned, so the compression is
bounded empirically rather than argued. (An earlier log line estimated `M^1.6` by
extrapolating from the 50–150 range; the direct measurement at project scale
confirms it.)

**3. Compression reframed from caveat to methods finding.** The mechanism is in
the *definition* — anchors are dual-weighted extreme points of the sampled cloud —
so every GLUE-family estimator at modest `M` inherits it. Chou et al. subsample to
**M = 50**, where our curve puts `D_eff/D ≈ 0.67` at `D = 10`. The sheet now states
the finite-sample behaviour of anchor-based geometry estimation as a deficit
scaling ≈ `M^-0.45`, uniform in `D`, independent of `P` and `n_t`. Also added: a
**caption requirement** that every figure reporting absolute `D_eff` carries the
lower-bound note, not just the methods section.

**4. MDE at 5 vs 8 seeds** — uniformly **25.4% tighter for +1.7 h wall**
(α 3.78 → 2.82%, `D_eff` 2.55 → 1.90%, `R_eff` 1.01 → 0.75%, `Ψ_eff` 2.81 → 2.09%,
`ρ_c` 1.96 → 1.46%). 8 seeds adopted. **Stated as a lower bound**: these floors are
estimator Monte-Carlo variance at fixed manifolds and exclude network-init and
stream variability; recompute from the Phase 0 pilot before the Day-14 table.

**5. Grid discrepancy found and resolved.** The `05` brief's figure (10 streams ×
5 seeds = 1,200 runs, 48,000 evals, 2.8 h) disagrees with `01` Phase 1 (20 streams
× 8 seeds = 3,840 runs, 153,600 evals, **9.0 h**). Both fit; the larger stands.

**Compute is no longer the binding constraint — engineering time is.** Next:
`src/models/` → training loop → Phase 0, no further validation detours. Phase 0's
own runs are cheap, so Gate-2 alignment and the time-reparameterization test get
run properly rather than minimally even though both feed cut hypotheses.

### `src/models/` and `src/train/` built — 55 tests pass

**Framework decision: numpy with hand-derived gradients, no torch/jax.** The model
is two layers with an MSE loss, so the gradients are four lines, and this buys
float64 throughout (`AGENTS.md` §4 — torch defaults to float32 and would need
dtype management at every op), bitwise reproducibility from `(seed, config)`, and
no dependency. Phase 1 is thousands of tiny independent CPU runs, which suits many
single-threaded processes rather than one accelerator. Gradients are verified
against central finite differences to `rtol=1e-5`
(`test_gradients_match_finite_differences`) — the whole choice rests on that test.

**Base-width constant derived (`00` §4.2 UNCERTAIN, `02` §3).** Requiring
`μP(γ₀=1, N=N_base) ≡ NTP(N=N_base)` fixes it uniquely: the width factors are
*relative* to the base width, `N^{1/2} → (N/64)^{1/2}` in the output scale and
`N → N/64` in the learning rate. At `γ₀=1, N=64` both give `γ_eff = 1`, `η = η₀`.
`test_base_width_equivalence` asserts bitwise equality on the forward pass **and**
the first gradient step, plus a companion test that the equivalence is specific to
`N = N_base`. **AGENT-DERIVED, not human-verified** —
`docs/reference/parameterization-derivation.md` is human-owned and unwritten; the
test pins the behaviour but the justification should be checked against Graldi §3.

Invariants under test: I5 `f(x;θ₀) = 0` exactly for every γ (zero-init readout);
I6 hidden weights **bitwise identical across γ** (dedicated γ-independent `shape`
RNG stream) — this is what makes γ and `a` orthogonal; I8 single shared readout.
`paired_init(seed)` exposes four named streams (`shape`/`data`/`stream`/`probe`)
so conditions are paired rather than merely seeded alike.

Alignment (`00` §5) is built despite the a-axis being cut from the Phase 1
factorial, because **Gate 2 is still to be run and recorded** and needs a correct
geodesic to be interpretable. Grassmann geodesic, norm- and rank-preserving
reassembly, endpoints and monotone principal angles all tested — including an
explicit test that the geodesic differs from linear-interpolate-then-reorthonormalize,
which is the failure the AGENTS Grassmann rule exists to prevent. Endpoint checks
use the projector distance `‖P_A − P_B‖_F` rather than the largest principal
angle: `arccos` is ill-conditioned near 1, so identical subspaces read ~1.5e-8
(√eps) no matter how exact the geodesic is, whereas the projector distance reaches
1e-14 and actually tests the code.

**Design hazard found while testing — under-training masquerades as forgetting.**
A step budget tuned on task 0 leaves later tasks unlearned, because later tasks
start from a solution to a *different* dichotomy and must overwrite it. Measured
at test scale (P=8, M=20, N=64), per-task final train accuracy:

| lr₀ | steps | task 0 | 1 | 2 | 3 |
|---|---|---|---|---|---|
| 0.05 | 400 | 0.93 | 0.78 | 0.62 | 0.69 |
| 0.05 | 2000 | 0.95 | 0.88 | 0.75 | 0.76 |
| 0.2 | 2000 | 0.99 | 1.00 | 0.98 | 0.99 |
| 0.5 | 2000 | 1.00 | 1.00 | 1.00 | 1.00 |

A task the network never learned cannot be forgotten, so its "forgetting" is
under-training and would have entered H1/H2 silently. `TrainConfig` now carries
a convergence target with early stop, and `TaskRecord.converged` /
`.usable_for_forgetting` record the outcome per task. **Phase 0 must verify
`all(t.converged)` across the γ grid before Phase 1 commits compute.**

---

## 2026-08-11 — Phase 0 (`scripts/run_phase0.py` → `results/phase0.json`)

Project config: `P=16, N=300, M=150, d=150, D=4, R=1.0, n_t=200`, γ ∈ {0.03, 0.1,
0.3, 1, 3, 10}, 3 seeds.

### GATE 2 PASSES — and this reopens a closed decision

The check `05` called "the most likely single failure in the project", and for
which D4 pre-authorized dropping the `a` axis, **passed decisively**.
Capacity-at-init against alignment `a`:

| `a` | 0.00 | 0.17 | 0.33 | 0.50 | 0.67 | 0.83 | 1.00 |
|---|---|---|---|---|---|---|---|
| `α` | 0.3091 | 0.3212 | 0.3383 | 0.3557 | 0.3736 | 0.3874 | 0.3940 |

Strictly monotone, rank correlation **+1.00**, range **23.97%** against a required
2× noise floor of 3.74% — **6.4× the threshold**. Direction is the expected one:
aligning `W(0)`'s row space with the manifold-center subspace raises capacity at
init, which is what "wealth" is supposed to mean.

**Decision returning to the human.** The `a` axis was cut by pre-authorization
anticipating this failure. It did not fail, the knob works, and compute is no
longer binding (`01` Phase 1: 9.0 h wall at 20 streams × 8 seeds). Reinstating `a`
is now a *scope* question rather than a feasibility one — it is what makes `C3`
(wealthy-lazy + poor-rich) and `C4` (anti-diagonal) meaningful, and those are the
architecture conditions the modularity story rests on. **Not reinstating it
unilaterally**; flagging that the grounds for the cut are gone.

### Capacity-at-init flat in γ — exact, by construction

Not a statistical pass. `h_m = ReLU(β₀ W_m x)` contains no γ, and the readout is
zero-init, so the representation at init is **bitwise identical across γ** and
`α` is equal to the last bit (0.310595 at γ = 0.03 and γ = 10). Verified both ways.
This is I6 holding, and it is what makes γ and `a` orthogonal.

### `pairwise` vs `full_P`: capacity agrees, geometry does not

| measure | `full_P` | `pairwise` | rel. diff | floor | |
|---|---|---|---|---|---|
| `α` | 0.3078 | 0.3109 | **1.02%** | 1.87% | agree |
| `R_eff` | 0.8718 | 0.8664 | **0.62%** | 0.50% | ~agree |
| `D_eff` | 4.548 | 5.734 | **26.1%** | 1.26% | **diverge** |
| `Ψ_eff` | 0.6041 | 0.7657 | **26.8%** | 1.39% | **diverge** |
| `ρ_c_glue` | 0.5980 | 0.6272 | 4.88% | 0.97% | diverge |
| `ρ_c_signed` | 0.5219 | 0.5348 | 2.49% | 0.97% | diverge |

The structure is the interesting part. `D_eff` and `Ψ_eff` inflate by **the same
factor** under pairwise — 1.2609 and 1.2676, a ratio of **1.0053** — so they
cancel in `α = Ψ_eff(1 + R_eff⁻²)/D_eff` to within 0.5%, and `R_eff` barely moves.
The estimation mode is close to a **pure common rescaling of the `D`/`Ψ` pair**.
Mechanistically this is what a joint QP should do: at `P = 2` only two manifolds
compete for each `t`, anchors are less constrained, and the apparent dimension
rises.

Consequences:

1. **Published GLUE capacities are comparable to ours; published GLUE
   *geometries* are not.** A 26% offset in `D_eff` is a mode difference, not a
   finding. Worth stating — the literature reports pairwise geometry.
2. **The attribution is in the affected channels.** `α` is safe either way, but
   Figure 2 decomposes into `D_eff` and `Ψ_eff`. If the ~1.26 factor is *constant
   across conditions* it cancels in `Δlog D_eff` and the attribution is untouched;
   if it drifts with the representation, it does not. **This must be checked at
   two or more conditions before Phase 1 commits** — one condition cannot
   distinguish a constant offset from a varying one. Added to the gate list.
3. `full_P` remains primary, `pairwise` reported alongside, per `01` §4.

### Stopping criterion was wrong, found by its own diagnostic

The convergence check flagged non-convergence at low γ, and the detail disproved
the obvious reading. Per-task train accuracy was **identical across all six γ**
(0.991 / 0.994 / 0.982 / 0.972) with stopping at step 1 on the first task. Reason:
from `u = 0`, one gradient step gives `u ∝ Σ_b y_b h(x_b)` — the kernel/Hebbian
readout — and

    f(x) ∝ Σ_b y_b h(x_b)ᵀ h(x)

whose **sign does not depend on the learning rate or on γ**. So train accuracy
jumps to ~0.99 at step 1 in every arm, and an accuracy-based stopping rule halts
training before any feature learning — precisely the thing under study. `01` §1
specifies `stopping: "matched_loss"`; the implementation used accuracy. Fixed:
`TrainConfig` now takes `stopping ∈ {"matched_loss", "fixed_steps"}` with
`target_loss`, and `test_accuracy_saturates_at_step_one_so_stopping_must_use_loss`
pins the reason so it cannot be undone. Check 3 rerun under matched loss
(`scripts/run_phase0_richness.py`).

Note this also means **train accuracy is near-useless as a progress measure in
this model** — it saturates immediately at every γ. Loss, `‖ΔW‖/‖W‖`, and the
geometry are the informative axes.

### Richness separation PASSES at matched loss — after fixing `lr0`

The first matched-loss rerun failed at γ ≤ 1: loss plateaued near 0.34 and never
reached the 0.05 target within 5,000 steps/task. Two candidate explanations, and
the obvious one was wrong.

**Ruled out — an expressivity floor.** `2N = 600` readout parameters against
`P·M = 2400` samples looks underparameterized, which would put a hard floor under
the lazy arm. Solving the readout exactly by least squares at fixed `W` gives loss
**0.0002**: no floor. The targets are constant within each manifold, so there are
only `P = 16` distinct values to fit and 600 features fit them trivially.

**Actual cause — `lr0 = 0.2` was simply too small.** μP's `η ∝ γ₀²` makes the
*function-space* rate γ-independent (`Δf ∝ η·s²` with `s = β_L/γ_eff`; the product
is 6.66e-4 at both γ = 0.03 and γ = 10, identical as designed). What differs is
that feature learning speeds the fit at large γ, so a global rate too small to
converge leaves only the rich arms finishing. Raising `lr0` fixes every arm at
once, and the system is stable well beyond the value chosen:

| γ | `lr0`=0.2 | `lr0`=5.0 | `lr0`=50 |
|---|---|---|---|
| 0.03 | 0.426 (no conv.) | **0.050** @1196 steps | 0.050 @120 |
| 1.0 | 0.415 (no conv.) | **0.049** @264 | 0.045 @28 |
| 10 | 0.050 @1006 | **0.049** @42 | 0.032 @6 |

**`lr0 = 5.0` adopted** for Phase 1. Re-run at 3 seeds, `T = 2`, matched loss 0.05:

| γ | 0.03 | 0.1 | 0.3 | 1.0 | 3.0 | 10 |
|---|---|---|---|---|---|---|
| `‖ΔW‖/‖W‖` | 0.0015 | 0.0125 | 0.0486 | 0.1274 | 0.2525 | 0.5025 |
| steps to target | 2883 | 2330 | 1305 | 515 | 203 | 73 |

**PASS: 2.53 decades** of `ΔW` separation at matched loss, against the 1-decade
requirement, and **every arm reaches the target at every seed**. The separation is
therefore richness, not differential training progress.

**Carry forward for H3.** Reaching the *same loss* takes **39.5× more steps** at
γ = 0.03 than at γ = 10. That spread is the raw material for the
time-reparameterization test (`00` §12): if warping step count by this factor
collapses the geometry trajectories onto each other, H3 is dead. The factor is now
measured, so the test has a principled warp to try first rather than a fitted one.

---

## 2026-08-11 — mode constancy: the `pairwise` offset is NOT constant

`scripts/run_mode_constancy.py` → `results/mode_constancy.json`. Nine geometries ×
3 seeds × both modes: six synthetic points moving one axis at a time from the
Phase 1 base (`D` ∈ {2,4,8}, `R` ∈ {0.5,1,1.5}, `ρ_C` ∈ {0,0.4}) and three network
representations (init, trained lazy γ=0.03, trained rich γ=10), since Phase 1
measures representations rather than generated manifolds.

**A first pass was wrong and is worth recording.** It placed the synthetic
manifolds in the *input* dimension `d = 150`, where `P(D+1) = 144` of 150
dimensions at `D = 8`: the arrangement is nearly degenerate, the manifolds are no
longer in general position, and `D_eff` collapses for reasons unrelated to
estimation mode. That inflated the effect (`D_eff` ratio reached 1.52). Redone in
the *measurement* ambient dimension `N = 300`. **General rule: keep
`P(D+1) ≪ ambient`** — this is why B.5 uses `N = 1000` — and it is now a guard
worth having in any arrangement built for estimation.

**Verdict: not constant.** The `pairwise/full_P` ratio on `D_eff` ranges
1.038 → 1.253 (CV 7.05%) and on `Ψ_eff` 1.050 → 1.283 (CV 7.74%).

The decisive statistic is not the ratio's spread but whether `Δlog D_eff` — the
quantity the attribution actually uses — survives the mode change:

| comparison | `Δlog D_eff` full_P | pairwise | error | vs floor |
|---|---|---|---|---|
| `rep_init → rep_rich` | −0.2273 | −0.3483 | 0.1210 | **9.7×** |
| `rep_lazy → rep_rich` | −0.2247 | −0.3441 | 0.1194 | **9.5×** |
| `rep_init → rep_lazy` | −0.0026 | −0.0042 | 0.0015 | 0.1× |
| worst overall | +0.7199 | +0.9075 | 0.1876 | 15× |

The third row is the internal control: two representations that are geometrically
almost identical, where the two modes agree to a tenth of the noise floor. So the
disagreement is not noise — **it grows with the size of the geometry change**,
which is exactly the pathology that would contaminate Figure 2. The first two rows
are the comparison Phase 1 actually makes (same network, different training
states), and there the error is ~10× the floor.

**Consequence, per the ratified rule: `full_P` is mandatory for all Tier-2
measurement.** `pairwise` is computed and reported only where comparability with
published values is the point. The appendix gets the observation that published
GLUE *geometries* are not comparable across estimation modes while *capacities*
are — `α` agrees to ~1% in every geometry tested, because the `D_eff` and `Ψ_eff`
distortions are co-directional and largely cancel in `α = Ψ_eff(1+R_eff⁻²)/D_eff`.

### Bonus, and it sharpens the compression caveat

With ground truth known at these points, both modes can be scored directly. In
log space, against the generated values:

| | `d log D_eff / d log D` | `d log R_eff / d log R` |
|---|---|---|
| `full_P` (P=16) | **0.663** | **0.581** |
| `pairwise` (P=2) | 0.772 | 0.624 |

(1.0 would be faithful.) This upgrades the earlier qualitative statement — "the
bias is monotone and compresses dynamic range, so the dimension channel
understates rather than manufactures" — into a **measured attenuation factor**:
at Phase 1 settings, a true `Δlog D` registers as ≈ 0.66 of itself and a true
`Δlog R` as ≈ 0.58 of itself. Figure 2's dimension and radius channels are
attenuated by roughly a third and two fifths respectively. Two consequences:

1. The conservatism claim now has a number attached instead of a direction.
2. It is *not* being used as a correction. Dividing by 0.663 would import the
   synthetic-manifold calibration into representation measurements, which the
   `rep_*` points show are a different regime (their mode ratios sit at 1.25 while
   the synthetic points sit at 1.04–1.21). Reported as an attenuation bound, not
   applied as a factor.

---

## 2026-08-11 — `src/analysis/` and the time-reparameterization test: **H3 survives**

`src/analysis/{trajectories,timewarp}.py`, `scripts/run_timewarp.py` →
`results/timewarp.json`. Phase 0's fifth and last check. Geometry
`(α, D_eff, R_eff, ρ_c)` recorded at 9 logarithmically spaced checkpoints to 4000
steps, γ ∈ {0.03, 1, 3, 10}, 3 seeds, `full_P`, `n_t = 200`. The measurement RNG is
held fixed across checkpoints so successive points differ because the
representation moved, not because the estimator resampled `(y,t)`.

Three warps of increasing generosity, residuals in units of the noise floor so
1.0 is the decision boundary: the **measured** warp (the steps-to-target-loss
ratio, no free parameters), the **best rate** (one free parameter), and **DTW**
(the most generous monotone warp there is; fixed start, which the paired-init
design justifies, and free end).

### First: the lazy arm's geometry does not move at all

Total geometric excursion over the whole training window:

| γ | 0.03 | 1.0 | 3.0 | 10.0 |
|---|---|---|---|---|
| excursion (noise floors) | **0.2** | 9.8 | 27.2 | 65.2 |

At γ = 0.03 the loss falls 0.499 → 0.010 while the representation geometry moves
**two tenths of one noise floor** — `D_eff` goes 4.436 → 4.431. All of the learning
is in the readout. This is the lazy limit confirmed by direct geometric
measurement rather than by a weight-norm proxy, and it makes the γ ≪ 1 vs γ ~ 1
contrast a contrast between *no* geometric change and geometric change, which is
what H1/H2 need.

It also makes every comparison involving γ = 0.03 **vacuous** for this test: a
trajectory that does not move is trivially the slowed-down opening of any other
trajectory, so "they coincide" says nothing about shape. Those pairs are now
labelled vacuous rather than counted as coincidence.

### Verdict: H3 is testable

| pair | measured | best rate | best monotone (DTW) | verdict |
|---|---|---|---|---|
| 0.03 vs 1 | 4.10 | 0.04 | 0.09 | vacuous |
| 0.03 vs 3 | 11.81 | 0.05 | 1.24 | vacuous |
| 0.03 vs 10 | 30.67 | 0.10 | 12.86 | vacuous |
| **1 vs 3** | 10.27 | 5.54 | **1.05** | coincide (marginal, ±0.29) |
| **1 vs 10** | 35.06 | 6.92 | **11.54** | **DIFFER** |
| **3 vs 10** | 26.24 | 21.32 | **7.06** | **DIFFER** |

Two of the three informative large-γ pairs differ by 7–12 noise floors under the
most generous monotone warp available. **Trajectories at different large γ are not
one trajectory at two speeds**, the division-of-labour hypothesis survives, and
heterogeneity contrasts are unrestricted — `00` §12's fallback does not fire.

Caveat worth carrying: `1 vs 3` is marginal at 1.05 ± 0.29, and those are the two
arms that move least (9.8 and 27.2 floors). So the honest rule is **use
well-separated γ with γ = 10 as one endpoint**; a 1-vs-3 contrast is not
demonstrably a shape difference.

### Two substantive by-products

**Matching loss does not match geometry.** The parameter-free measured warp fails
everywhere (4–35 floors). And the rate that best aligns *geometry* is 10–20× the
rate that aligns *loss*: for 1 vs 10, best-rate scale 161.6 against a loss-based
8.0. Geometry and loss evolve on different clocks, and even the best geometric rate
leaves 6.92 floors. This is worth a sentence in the paper: matched-loss stopping
equalizes training progress, not representational change.

**ρ_c is by far the most dynamic channel.** Movement per channel at γ = 10, in
floors: `ρ_c` 122, `R_eff` 22, `α` 19, `D_eff` 18. Good news for H1d, which turns
on ρ_c: the measure with the tightest noise floor is also the one that moves most.

### Two bugs, one of which nearly produced the wrong headline

1. `measured_warp_scale` returned the inverse of the convention `residual` uses.
   Caught by a unit test asserting a known rate is recovered.
2. DTW required coverage in *either* series rather than **both**. That let a nearly
   static trajectory be matched against the other's flat opening segment and score
   a near-zero residual — a degenerate warp reported as coincidence. Under the
   buggy version `1 vs 10` scored 1.18 and would have been called marginal
   coincidence; corrected it is 11.54. Both directions are now pinned by controls:
   a synthetic pure-rate-rescaling must be recovered and called coincident, and a
   synthetic shape inversion must survive every warp.

---

## 2026-08-11 — Phase 1 pipeline, and the artifact check becomes a rule

Phase 0 is closed. `AGENTS.md` §8.2 now makes the pattern that closed it mandatory:
**before interpreting any result, state what would have to be true for it to be an
artifact, and check that.** Both of the last two bugs were caught by diagnostics
rather than by tests, and the DTW one was a false negative that would have killed H3
and looked like a finding. The section carries the four cases so far as a table, plus
the four checks that have earned default status: compare against an exactly solvable
limit; vary what should not matter; confirm the method *fails* where it should;
and ask whether the quantity moved at all before interpreting agreement about how it
moved.

It earned its keep within the hour — see the probe finding below.

### Built

- `src/analysis/attribution.py` — the exact three-factor log-space decomposition, plus
  the `ρ_c → R_eff` conversion.
- `src/pipeline.py` — one arm end to end: stream → sequential training → Tier-2
  geometry on a schedule → attribution, accuracy matrix, `CF`/`CFr`, and the `00` §11
  manipulation checks.
- `scripts/run_phase1.py` — the grid, resumable, one JSON per arm.
- `scripts/run_probe_check.py` → `results/probe_check.json`.
- `scripts/fig_gamma_excursion.py` → `results/fig_gamma_excursion.csv` and
  `results/figures/gamma_excursion.{pdf,png}`.
- 26 new tests; 90 pass.

**Schedule and budget.** Tracked tasks every 4th, measured at their own boundary
(the attribution baseline) and at every later measurement boundary, so retention is
followed over a widening lag rather than at one fixed lag: `{0:[0], 4:[0,4],
8:[0,4,8], 12:[0,4,8,12], 15:[0,4,8,12]}`. That is **38 evaluations per arm against
the cost model's 40**, asserted by a test. The measurement RNG is fixed across
boundaries and shared between the generic and retained ensembles, so a change between
two boundaries is the representation moving, and the generic/retained crossing is
paired at every boundary.

### The probe measure was dead, and its first replacement was noise

Figure 3 overlays a probe metric on generic capacity. The obvious choice — accuracy
of a refit linear readout on the held-out dichotomy `y*` — is **exactly 1.0**, at
initialization and after training, in both modules, at every γ. The load is
`P/N = 16/300 = 0.053` against a critical capacity near 0.3, so every balanced
dichotomy is separable with room to spare, and separability at fixed sub-critical
load cannot track capacity. It would have plotted as a flat line at 1.0 and read as
"generic capacity is preserved".

Three candidates, scored on whether the γ signal beats the measure's own noise:

| measure | γ=10 | γ=1 | γ=0.03 | signal | noise | SNR | verdict |
|---|---|---|---|---|---|---|---|
| `accuracy` | 1.0000 | 1.0000 | 1.0000 | 0 | 0 | — | **saturated** |
| `margin` | 0.0500 | 0.0632 | 0.0598 | 0.0132 | 0.0033 | **4.0** | use |
| `margin_p05` | 0.0462 | 0.0565 | 0.0521 | 0.0102 | 0.0032 | 3.2 | use |
| `heldout_manifold_accuracy` | 0.4608 | 0.4635 | 0.4142 | 0.0494 | 0.1206 | 0.4 | **noise** |

**`margin` is the probe measure**, and it is the principled one rather than a
fallback: capacity *is* the load at which the margin reaches zero, so the margin is
the graded quantity underneath the thresholded one. Its change over training is
sign-consistent across seeds in every arm — rich training reduces the probe margin,
γ=0.03 leaves it unchanged to four decimals, which is the geometrically static lazy
arm showing up again on an independent measure.

**Held-out-manifold accuracy is a control, not an overlay.** It sits at chance with a
within-run split sd of 0.06–0.16, which is *correct* rather than broken: `y*` is a
random balanced dichotomy, so there is no shared structure for a readout fitted on 12
manifolds to extend to 4 unseen ones. Its apparent γ=0.03 dip is one seed's
initialization (0.329 vs 0.495 at init, and γ=0.03 barely moves), not an effect of γ.
Kept because if it ever rises above chance, the probe is leaking factor structure.

The artifact rule is what produced this: the first question asked of a 1.0 was "did
this quantity move at all", and the second, asked of the 0.414, was "is this
difference bigger than the seed difference". It was not.

### `ρ_c → R_eff`: the conversion is measured, and it does not transfer unchanged

Fitted on the B.5 center-correlation sweep, where ground-truth `ρ_C` 0 → 0.8 drives
`R_eff` 1.02 → 1.82 while `D_eff` stays flat to **0.38%** — the radius absorbs center
correlation essentially alone, as Wakhloo's duality says:

    log R_eff = 0.0126 + 0.3548 · (−log(1 − ρ_c))      R² = 0.99986

`R_eff ∝ (1 − ρ_c)^(−0.355)`, five points, near-exact. Attribution now reports what
fraction of each observed `Δ log R_eff` the observed `Δ ρ_c` accounts for, which makes
`00` §8's joint-reporting requirement quantitative.

Two guards, both from the first smoke grid. The denominator is **floored at the
`R_eff` noise floor** (CV 0.50%): fractions of −134 were coming from radius changes of
order 1e-3, and a ratio whose denominator is unresolvable is not a measurement.
And fractions **≫ 1 survive legitimately** — one group reports 42 — which is
informative rather than broken: it means `ρ_c` collapsed far more than the radius
followed, so the synthetic calibration (fitted where `ρ_C` is the *only* thing
varying) over-predicts on representations, where radius and centers move together.
**Open, to settle at the design point:** whether that over-prediction is systematic
enough to quote a transfer factor, or whether the conversion should be reported only
as a bound. Not answerable at smoke settings (`n_t = 25`).

### Smoke grid runs; the shape is right and is not yet evidence

Reduced settings (`T=4, P=8, M=40, N=80, n_t=25`) — a wiring check, not a
measurement. Retained capacity falls at every lag, far more at γ=10 (Δlog α ≈ −0.95
to −1.28 at lag 1) than at γ=0.3 (−0.11), and the dominant factor differs between
them: utility at γ=10, radius at γ=0.3. That is the H1 shape. It is also 25 `(y,t)`
samples and one seed, so it goes in this log and nowhere near a figure.

Identity residual across every evaluation in the suite: **≤ 3.5e-16**.

---

## 2026-08-11 — **the Phase 1 cost model was wrong by 9×**, and why

Launched the full grid (1280 arms, 48,640 evals) on 254 workers. After 79 minutes,
**zero arms had completed.** Load was a steady 267 and every worker had accumulated
79 minutes of CPU, so nothing was blocked — it was simply far slower than projected.
Timed one evaluation against the live grid: **775.8 s, against the 52.4 s the cost
model assumed.** Killed it (no arm was near completion, so nothing was lost) and
measured the throughput curve properly — `results/scaling.json`,
`scripts/run_scaling.py`, `n_t = 20` to keep it short, which is legitimate because
cost is linear in `n_t` while the memory footprint driving contention is set by `P·M`.

| workers | eval latency | throughput | effective cores | efficiency |
|---|---|---|---|---|
| 1 | 5.71 s | 0.166 eval/s | 0.9 | 0.94 |
| 32 | 6.77 s | 4.078 eval/s | 27.6 | 0.86 |
| 64 | 11.27 s | 4.633 eval/s | 52.2 | 0.82 |
| **128** | 20.44 s | **5.169 eval/s** | 105.7 | 0.83 |
| 192 | 40.26 s | 4.150 eval/s | 167.1 | 0.87 |
| 254 | 60.10 s | 3.749 eval/s | 225.3 | 0.89 |

**Throughput peaks at 128 workers and then falls. Running 254 was worse than running
32.** Two structural causes:

1. **128 physical cores, not 256.** `nproc` reports 256 on this 2×64-core EPYC 7763
   because of SMT, so 254 workers was already 2× oversubscribed before anything else.
2. **The anchor QP is not small.** It has `P·M = 2400` variables, so its Gram is
   2400² × 8 B ≈ **46 MB against a 32 MB L3**. Every worker streams that from DRAM on
   every one of `n_t` samples, so the workload is memory-bandwidth-bound and cores past
   the bandwidth limit contribute nothing.

**The error in `results/cost_model.json` was methodological, not arithmetic.** It
measured 52.4 s/eval in isolation and projected wall time by dividing total
core-seconds by 248, i.e. it assumed perfect scaling and never measured it. The
`01` §4 line "throughput is set by total core-seconds, so one thread per worker is the
correct configuration for an embarrassingly parallel grid" was the load-bearing wrong
sentence: the grid is embarrassingly parallel in its *control flow* and
bandwidth-coupled in its *memory access*, and only the first was checked.

`_par.n_workers` now caps at 128 (was `nproc − 2` = 254), which alone is a 1.4×
throughput gain over what was running.

**Revised cost: the full grid is 26.1 h at 128 workers, against the 3.0 h projected.**
Training adds ~2 h. This invokes the `01` §4 cut order (`n_t` → Tier-2 interval →
retained-tasks-evaluated) for the first time — pre-authorized there, and the decision
of how far to cut is pending.

Worth noting for the paper's reproducibility section, since it generalizes: **the cost
of anchor-based GLUE estimation is bandwidth-bound in `P·M`, not compute-bound.** At
`M = 150` and `P = 16` the QP working set exceeds a 32 MB L3, so the estimator does not
scale with core count on a shared-memory machine past the bandwidth limit. Anyone
sizing a GLUE run from a single-process timing will over-project by roughly an order of
magnitude. Chou et al.'s `M = 50` puts the working set at ~5 MB, inside L3 — their
subsampling has a performance rationale as well as a statistical one.

---

## Where the 26 h actually came from, and a calibration bug the pilot caught

### The 9.3× decomposed

The projection was 2.82 h: 48,000 evals × 52.4 s/eval ÷ **248 workers**. The measured
figure was 26.1 h. Multiplying out:

| term | factor | note |
|---|---|---|
| grid size 48,000 → 48,640 | **1.01×** | essentially nothing |
| workers 248 → 128 | **1.94×** | the cap imposed after the scaling measurement |
| per-eval cost 52.4 s → ~247 s | **4.72×** | contention at 128 workers |

1.01 × 1.94 × 4.72 = 9.24, and 2.82 h × 9.24 = 26.1 h. ✓

**The grid did not grow — it is 1.3% larger than planned.** The entire overrun is
memory-bandwidth contention, and the worker cap is not a separate cause but a symptom
of the same one: throughput peaked at 128 because beyond that the memory system
saturated, so the cap was the best available response to contention, not an independent
choice. The original model's error was to measure per-eval cost single-threaded in
isolation and assume linear scaling to 248 workers. For a memory-bandwidth-bound
estimator that assumption is worth a factor of ~9.

The practical consequence inverts the obvious reading: **cutting the grid is a weak
lever and the solver is a strong one.** Halving the grid halves the cost, once. But
per-eval cost is inflated 4.7× by bandwidth, so shrinking the estimator's working set
attacks the dominant term — and should pay more in parallel than it does alone.

### Solver change: speed only, exact to machine precision

`_nnls_colgen` replaces the dense active-set solve with column generation over the
anchor matrix. On real Phase 1 data, single-threaded, clean process, current code:

| solver | s/eval | α | D_eff | R_eff |
|---|---|---|---|---|
| `nnls` (previous) | 54.4 | 0.316013 | 5.055807 | 0.960511 |
| `colgen` (current) | **21.3** | 0.316013 | 5.055807 | 0.960511 |

Identical to every digit reported, and to 3.7e-15 under test. **2.55× faster.** The
54.4 s also confirms the 52.4 s baseline held — per-eval cost never regressed. So the
pilot's geometry stands (it ran the reference solver); only its *timings* describe
superseded code. Contended throughput at 128 workers is the number that resizes the
grid and is still to be measured.

### The stale fork, made structural

The pilot's workers ran the pre-change solver: the parent imported the module, the edit
landed, and `fork` handed every worker the parent's already-imported copy. Two changes,
neither relying on vigilance:

1. **`spawn`, not `fork`** (`scripts/_par.py`). Workers re-import from disk. Costs about
   a second of startup against arms that run tens of minutes.
2. **Version stamps** (`src/provenance.py`). Every result record carries the git SHA,
   the dirty flag, and a per-module source hash; `assert_current()` at worker start
   refuses to run stale.

The part that is easy to get wrong: **the hash must be taken at import time.** Hashing
the file at worker start cannot detect this failure, because the forked child re-reads
the *new* bytes from disk while executing the *old* bytes in memory, and would report
agreement. Only a value frozen at import travels with the fork.
`tests/test_provenance.py` asserts this by actually forking.

### The pilot's real find: the ρ_c calibration was fitted on the wrong convention

The first draft figure reported center-collapse shares of **15–38**, where 1.0 means the
radius change is entirely center collapse. That is not a finding, it is an artifact, and
the artifact check (`AGENTS.md` §8.2) found it:

- `radius_from_rho` uses R_eff ∝ (1 − ρ_c)^−0.355, which needs ρ ∈ [0, 1).
- It was fitted on **`rho_c_glue`**, which is **unnormalized** (`00` §6.1 C3) and so is
  not a correlation. On Phase 1 representations it runs to **1.487**, median 1.235, with
  **71% of measurements above the fitted maximum of 0.809**.
- A `np.clip(rho, 0, 0.995)` then turned every out-of-domain input into a large finite
  number. The domain violation produced no error, just plausible-looking results.

Why it survived the B.5 validation: on synthetic manifolds the two conventions almost
coincide (0.038/0.043, 0.220/0.236, 0.409/0.426, 0.609/0.614, 0.809/0.803). They
separate only on representations. **A calibration validated on synthetic data was
carrying an assumption the synthetic data could not test.**

Fixed by refitting on `rho_c_signed`, the normalized convention, which on the same
representations sits at **0.331–0.481 — inside the fitted range**:

    log R_eff = 0.002911 + 0.36483 · (−log(1 − ρ_c_signed)),  R² = 0.9997

and by replacing the clip with a hard refusal outside `RHO_FIT_RANGE`. Corrected
center-collapse shares are **0.28–0.76**: between a third and three quarters of the
radius change is accounted for by center collapse, the rest being anisotropy. That is
an interpretable result where 15–38 was not.

Also structural: `summarize` now **re-derives attribution from stored geometry** rather
than reusing what the worker computed. Geometry is the measurement, attribution is
analysis over it, so a calibration correction is now a re-summarize rather than a 26 h
re-run — which is exactly what this fix needed.

### Draft Figure 2 (γ=10, S-HL/S-LL, 4 of 8 pilot arms)

`results/figures/attribution_pilot.png`. Plumbing confirmed end to end: Tier-2 output
reaches the attribution code in the right shape, the identity closes (max residual
0.0), and the figure script runs.

At γ₀=10, retained capacity falls by Δlog α ≈ −1.10 to −1.52 (a 3.0–4.6× drop), and the
split is stable across lag: **utility ≈ 48%, dimension ≈ 44%, radius ≈ 8%**. Most of the
loss is present by lag 3 and deepens slowly. The lazy arm is still running.

On the encoding problem: signed contributions cannot use a naive stacked area, since
`−Δ log D_eff` is negative by construction. The figure stacks positives up from zero and
negatives down from zero and overlays `Δ log α` as a line at the signed total, so the
line reads correctly whether or not the terms share a sign. In this arm all three happen
to be negative; mixed signs are handled and will appear in the lazy arm.

### Pilot sizing

8 arms and 304 evals was not the single-cell pilot intended — one γ, one condition, one
seed, ~38 evals. The de-risking value was reached at the first completed arm. Size
pilots to the smallest unit that exercises the whole path.

### Grid resized on measured throughput: 26.1 h → 4.0 h, no cuts needed

Measured on the **real** workload at `n_t = 200`, spawned pool, clean processes
(`results/scaling_colgen.json`):

| solver | workers | latency/eval | throughput | full grid (48,640) |
|---|---|---|---|---|
| `nnls` | 128 | 218.3 s | 0.547 eval/s | **24.7 h** |
| `colgen` | 128 | 33.4 s | 2.829 eval/s | 4.8 h |
| `colgen` | 192 | 45.3 s | 3.390 eval/s | **4.0 h** |
| `colgen` | 254 | 66.9 s | 3.470 eval/s | 3.9 h |

The `nnls@128` row predicts 24.7 h against the 26.1 h actually observed — 5% — which is
the check that this measurement describes the same thing the grid did.

**colgen is 5.17× faster in parallel against 2.55× single-threaded.** The gap between
those two numbers is the point: the win is not arithmetic, it is the working set. A
solver that fits in cache stops competing for memory bandwidth, so the benefit compounds
with the worker count rather than being independent of it. This is why the earlier
reading — that grid size drove the overrun and the solver was not the lever — had it
backwards: the grid was 1.3% over plan, and the solver was worth 6.5×.

Two consequences. **No cuts to the grid are required**; the scope conversation is moot at
4 h. And the worker cap moves 128 → 192, because the peak is a property of the workload
and the workload changed: colgen is no longer bandwidth-bound, so SMT past 128 physical
cores now helps where it previously hurt. 254 buys 2% over 192 for 48% more latency.

A related defect in the old curve: `run_scaling.py` measured at `n_t = 20` while the grid
runs at `n_t = 200`, so `scaling.json` characterised an evaluation ten times cheaper than
the real one and located the peak on the wrong workload. Now pinned to the grid's value.

### Pilot complete: 8/8 arms, max identity residual 3.4e-16

The γ contrast is the headline and it is large. At **γ₀ = 10**, Δlog α = −1.10 to −1.52
(a 3.0–4.6× loss of retained capacity). At **γ₀ = 0.03**, Δlog α = −0.006 to −0.010.
**Roughly 150× less forgetting in the lazy arm** — H1's predicted direction, at a
magnitude far beyond noise.

The more interesting result is that the *mechanism* differs, not only the amount:

| | utility | dimension | radius |
|---|---|---|---|
| γ₀ = 10 | **48%** | 44% | 8% |
| γ₀ = 0.03 | ~5% | 47% | 48% |

In the rich arm forgetting is dominated by `Ψ_eff`, the alignment between the readout
and the manifold axes. In the lazy arm utility barely moves and what little loss occurs
splits evenly between radius and dimension. That is a qualitative difference in kind and
is what the shared-axis figure hides, hence the native-scale inset.

Center-collapse share is `n/a` throughout the lazy arm — Δlog R_eff ≈ 0.004 sits below
the 0.005 noise floor, so the guard declines to report a ratio. Working as intended.

`results/figures/attribution_pilot.png`.

---

## Pre-launch hardening, and a destructive default that cost the pilot

### The `np.clip` audit

Four hits in the measurement path. Two are not guards: `network.py:79` is the ReLU, and
`core.py:366` is a divide-by-zero guard already made dead by an enclosing `np.where`. The
other two were real instances of the ρ_c pattern — legitimate absorbers of float error
that never checked the violation *was* float error:

- `alignment.py:74`, `arccos(clip(s, −1, 1))` on cosines of principal angles.
- `core.py:157`, `maximum(res.x, 0)` on duals from a bound-constrained solve.

Both are correct in intent: `arccos(1 + 2e-16)` is a NaN and propagating it is worse than
clipping. What was missing is that neither could distinguish 2e-16 from 0.5. Both now go
through `numerics.clip_to_noise`, which clips **and asserts nothing moved by more than a
stated tolerance**. A clip can no longer change meaning from "absorb rounding" to
"manufacture a value" without failing. `tests/test_numerics.py` includes the pilot's own
`rho_c` values as a case: fed to the old bounds they now raise instead of returning 15–38.

### ρ_c coverage is now logged per condition

`summarize` → `rho_coverage` reports, per (γ, `a`, condition): the range of both
conventions, how many `rho_c_glue` values exceed 1, and the fraction of measurements
inside `RHO_FIT_RANGE`. Out-of-range points report `n/a` and are never extrapolated. On
the pilot, coverage is **100%** in all four cells, with `rho_c_signed` at 0.29–0.52 —
mid-range, so the grid has room either side. The unnormalized convention meanwhile
reaches **1.928, above 1 in 39 of 56 measurements** in the rich arm, which is the clearest
statement yet of why it cannot feed a bounded-domain formula.

### The default was the destructive branch, and it deleted the pilot

Verifying checkpoint-resume before the 4 h launch was the right instinct and found a real
defect the hard way. `--resume` was **opt-in**, and the default branch ran
`_path(s).unlink()` over every arm in the grid before starting. So invoking `--pilot` *to
test resume* deleted all eight completed arms and began recomputing them. With
`results/phase1/` in `.gitignore` there was no way back. The pooled summary and figure
survived, and the full grid subsumes those cells, so nothing was lost that the grid will
not regenerate — but on the 4 h grid the same keystroke would have destroyed the run.

The fault is not that resume was broken; the skip logic was correct, including the
tuple/list round-trip subtlety. It is that **the default was the destructive branch.**
Fixed by inverting it: resuming is now the default, `--resume` is accepted and ignored,
and `--fresh` discards by **moving arms to `results/phase1/.trash-<timestamp>/`** rather
than deleting. `tests/test_phase1_resume.py` drives `main()` through the branch that did
the damage, which the first version of the test did not — it used `--summarize-only` and
so would have passed against the bug.

Verified end to end on the smoke grid: the second invocation reports `2 already on disk,
resuming`, recomputes nothing, finishes in 0.83 s, identical residual.

This is a fourth member of the silent-failure family, with a twist worth naming. The
first three produced plausible output measuring the wrong thing. This one produced *no*
output and destroyed input, and it was triggered **by the act of verification** —
checking whether resume worked was what broke it. Verification steps need the same
scrutiny as measurement steps.

---

## Scope conformance audit (§1), run against 742 arms mid-grid

`scripts/audit_scope.py` → `results/audit_scope.json`. Re-runs unchanged at completion.
**29 pass, 2 warn, 0 fail.** Scope statement now written down as `docs/06-scope.md`.

The checks that matter, with their evidence:

| check | evidence |
|---|---|
| C2 arms homogeneous in γ | per-module `ScalingConfig` resolved from stored specs; γ identical, 0 violations |
| module set is the (A,B) control only | one distinct `module_list` on disk: `('A','B')` |
| no a1b2 machinery reachable | 23 imported modules, **1,995 identifiers** scanned; no routing / task-conditioned heads / task-ID input / comms / gating |
| single shared readout | readout is `u[module]`, indexed by module only; no task index at any of 5 sites |
| ratified config | `full_P`, `n_t=200`, `T=16`, `center_policy=all`, P=16, M=150, N=300, d=150, D=4 — single value each |
| γ grid / conditions / seeds | 0.03–10 (6 values), Hiratani 2×2 only, 8 seeds, 5 shared streams |
| both ρ_c conventions recorded | every geometry row carries `rho_c_glue` and `rho_c_signed` |
| version stamps | 742/742 stamped, **0 stale, 0 unregistered**, 0 module-hash mismatches |
| identity residual | **max 4.44e-16** over 28,158 evaluations (median 1.15e-16) |
| ρ_c coverage | 24 cells, **100% in range in every cell**, 0 cells left [0.04, 0.80] |
| richness separation | S-HH ×246, S-HL ×157, S-LH ×117, S-LL ×112 — all ≥ 1 order of magnitude |
| no dead module | min output-variance share **0.247** over 23,712 module-boundary records |

Two warnings, both understood: completeness (742/1280, grid running) and the `a > 0`
module-duplication caveat recorded in `06-scope.md` §4.

**The audit's own first version failed, and the failure is instructive.** It grepped whole
files for forbidden vocabulary and reported a FAIL on three docstrings — `core.py`'s
"estimator-routing table", `alignment.py`'s "Gate 2", `_par.py`'s "memory-bandwidth-bound".
All prose, no machinery. The deeper problem was symmetric: a text search that trips on a
cross-reference would equally miss a real `route()` sitting in a stripped comment line, so
it was wrong in both directions. Rewritten to walk the AST and match **identifiers only** —
names bound or referenced, docstrings excluded — which is what "no machinery reachable"
actually means. Fifth instance of a check measuring the wrong thing; caught this time
because a FAIL demanded explanation rather than a pass being accepted.

### The completion tail is the *lazy* arms, not the rich ones

Wall time and steps-to-target per γ, over the 741 `a = 0` arms complete so far:

| γ₀ | n | median wall | median steps/task |
|---|---|---|---|
| 0.03 | 160 | **115.9 min** | 1,469 |
| 0.1 | 73 | 65.7 min | 744 |
| 0.3 | 102 | 44.4 min | 285 |
| 1 | 119 | 32.7 min | 92 |
| 3 | 127 | 26.3 min | 31 |
| 10 | 160 | **26.8 min** | 11 |

Monotone, and in the opposite direction to the natural guess: **the lazy arms are 4.3×
slower, driven by 134× more SGD steps to reach the target loss.** The reason is
matched-loss stopping — at small γ feature learning is suppressed, so the function moves
slowly and many more steps are needed to drive loss to 0.05, whereas γ=10 arrives in ~11.
Rich arms move further *in weight space* (which is the richness manipulation, ×157) but
reach the loss target far sooner, and it is the loss target that ends a task. Nothing
pathological; the tail is priced in.

Grid ordering interacts with this as designed. Tier 0 (γ ∈ {0.03, 10}) is **complete at
320/320**, so Figure 2's core lazy-vs-rich contrast is already fully in hand and does not
depend on the remaining runtime. Tier 1 (mid γ) stands at 421/640 and contains the
remaining γ=0.1 arms, which is what stretches the end. Tier 2 (the `a`-sweep, all at
γ=1) has not started and will be fast.

---

## §2a draft — Figure 2 across the γ sweep (776 arms; re-run at completion)

`scripts/fig2_gamma_sweep.py` → `figures/fig2_gamma_sweep__pooled__lag12__*.{pdf,png,json}`,
filename carrying the arm count and a SHA over the contributing keys and module hashes.
Reads through `src/analysis/grid.py`, which re-derives attribution from stored geometry
and excludes `a > 0`.

### Artifact check, before the result

**Could the composition trend be a lag artifact?** Pooling every lag mixes lag-1
comparisons with lag-15, so a γ difference in composition could be a difference in how far
through the stream the average comparison sits. Ruled out by matching lag: the figure is
at lag 12 throughout, and the trend is present at fixed lag.

**Could the shares be attributing a change that did not happen?** Yes, and at low γ they
were. `shares` are `|term| / Σ|term|`, defined whether or not anything moved. At γ = 0.03
and 0.1 the total change is **0.1 and 0.5 noise floors** — below resolution — yet the naive
plot showed confident shares (utility 0.52 and 0.60). Shares are now drawn only where
|Δ log α| ≥ 2 floors, and low-γ cells read "not resolvable".

**Could the utility share of 0.02 at γ = 0.3 mean utility is inactive?** No, and this was
the subtler trap. Cancellation is ~0 at every γ (median 0.000, so factors are not
fighting), so the near-zero pooled utility term is not opposing contributions within an
arm — it is the term **crossing zero across arms**. Reported as a sign structure in its own
panel rather than hidden inside an absolute-value ratio.

**Could the low-γ positive utility term be a real positive effect?** No — the sign flips on
53% and 55% of arms at γ = 0.03 and 0.1, i.e. it has no sign. An earlier draft of the panel
called this "Ψ_eff changes sign", which overclaims the unresolved side. Points whose
positive fraction lies in (0.35, 0.65) are now drawn hollow, and the claim is the weaker,
defensible one: **utility is unresolved below γ ≈ 0.3 and resolvably negative from γ = 1.**

### The result

At matched lag 12, pooled over the 2×2, module A:

| γ₀ | Δlog α (floors) | Δlog α | Ψ_eff | −D_eff | 1+R⁻² | frac. Ψ>0 | ρ_c share | n |
|---|---|---|---|---|---|---|---|---|
| 0.03 | 0.1 | +0.0012 | +0.0005 | +0.0001 | +0.0006 | 0.53 | n/a | 160 |
| 0.1 | 0.5 | +0.0083 | +0.0054 | −0.0005 | +0.0035 | 0.55 | 0.04 | 86 |
| 0.3 | 3.7 | −0.0686 | −0.0012 | −0.0348 | −0.0326 | 0.39 | 0.13 | 106 |
| 1 | 16.0 | −0.2970 | −0.0840 | −0.1227 | −0.0903 | 0.26 | 0.23 | 130 |
| 3 | 29.5 | −0.5466 | −0.2219 | −0.2340 | −0.0907 | 0.28 | 0.42 | 134 |
| 10 | **50.7** | −0.9402 | −0.4213 | −0.4053 | −0.1136 | 0.25 | 0.43 | 160 |

**Magnitude is smooth, not a transition.** Forgetting rises monotonically from 0.1 floors
at γ = 0.03 to **50.7 floors at γ = 10**, accelerating in log γ with no discontinuity. The
lazy arms do not forget at all in the resolvable sense — 0.1 floors is nothing happening,
consistent with the Phase 0 finding that the lazy arm is geometrically static.

**Composition shifts smoothly too, with one qualitative event.** Over γ = 1 → 10 the
utility share climbs 0.28 → 0.45 while radius falls 0.31 → 0.12 and dimension holds ~0.42.
The one non-smooth feature is that **Ψ_eff becomes resolvable and negative between γ = 0.3
and γ = 1**, going from no detectable sign to co-dominant with dimension. So the answer to
"smooth or transition near γ₀* ≈ 0.1" is: the magnitude is smooth; the composition has a
threshold, and it sits at γ ∈ (0.3, 1), **not at 0.1**. Measured γ* (argmin of final
average error) is **1.0 on the grid, 0.615 interpolated in log γ**, so the composition
threshold brackets γ* rather than the pre-registered 0.1. Reporting the location as an
interval, since six half-decade points cannot place it tighter.

> **Superseded in two ways by the per-γ floor refit below** (see "Panel (d) refit"). The floor
> was refitted, so the numbers are now 0.12 → 0.24 → 0.43 → 0.42 at 88–100% coverage; and the
> final step was never resolved in either fit, so "rises through γ = 10" should read "rises
> steeply through γ = 3, then plateaus". The paragraph is kept as written for the record.

**The radius channel is increasingly center collapse.** The ρ_c-attributable share of
Δ log R_eff rises 0.13 → 0.23 → 0.42 → 0.43 with γ, at 90–100% panel coverage for
γ ≥ 0.3. So in rich arms nearly half the radius change is centers moving rather than
manifolds expanding — which is why `00` §8 requires ρ_c reported alongside R, and it is
the panel that stops the radius channel being misread. At γ = 0.03 coverage is 19% and the
share is suppressed: the radius does not measurably move, so there is nothing to attribute.

Two things to flag against the pre-registration, deferred to §2d: **the dimension channel
is much larger than the radius-heavy forecast for H1a** (−0.41 vs −0.11 at γ = 10, i.e.
dimension is 3.6× radius), and the utility result — the more novel half — is confirmed and
large. Overall coverage of the center-collapse panel across all comparisons is 73%.

---

## Finding 5: the `Ψ_eff ∈ [0,1]` bound was never ours to assume for retained capacity

Raised by a question about whether Ψ_eff rises in absolute terms during rich training. It
does not — but checking it surfaced retained `Ψ_eff > 1`, flagged-and-stopped because the
utility channel *is* the H1 result, and then **resolved by measurement: the bound is a
property of the label average, not of the estimator.** Ruling and check design: Kati.

This is the inverse of the `rho_c` clip bug. There, a guard manufactured in-range values
and hid a domain violation. Here, refusing to guard surfaced a real property of the
estimator that the source papers do not discuss. Same protocol, opposite polarity — which
is the argument for the protocol rather than for either outcome.

### What was actually measured

**Generic (label-agnostic) Ψ_eff falls monotonically with boundary, it does not rise:**

| γ₀ | boundary 0 | 4 | 8 | 12 | 15 | init → end |
|---|---|---|---|---|---|---|
| 0.03 | 0.6065 | 0.6073 | 0.6054 | 0.6100 | 0.6018 | 0.992× |
| 1 | 0.6142 | 0.6111 | 0.6073 | 0.6087 | 0.6003 | 0.977× |
| 10 | 0.5981 | 0.5763 | 0.5615 | 0.5558 | 0.5459 | **0.913×** |

**But there is a real overshoot, on the retained ensemble**, which is a different quantity
and the reason the two readings can both be true. Retained Ψ_eff for a task, measured at
that task's own boundary versus later ones:

| γ₀ | at own boundary | at later boundaries | ratio |
|---|---|---|---|
| 1 | 0.8911 | 0.8720 | 0.979 |
| 10 | **1.2941** | 1.0429 | **0.806** |

So learning a task lifts *that task's* retained utility from the ≈0.60 generic baseline to
≈1.29, and later tasks relax it back toward ≈1.04. The "same currency, different reference
point" reading is supported — what forgetting removes is the task-specific utility gain
that learning produced — but the mechanism is a **retained-vs-generic** gap, not a rise in
absolute Ψ_eff over training. Generic utility drifts *down* by 8.7% at γ=10.

### The bound question

**Retained Ψ_eff exceeds 1 on 21.6% of measurements, to a maximum of 1.782.** Generic
Ψ_eff never does (max 0.6550 over 8,680 measurements).

| ensemble | n | min | median | max | fraction > 1 |
|---|---|---|---|---|---|
| generic | 8,680 | 0.4486 | 0.6051 | 0.6550 | 0.000 |
| retained | 24,304 | 0.6764 | 0.7963 | **1.7823** | **0.216** |

What is ruled out: **it is not an arithmetic fault.** The three-factor identity closes to
`max|residual| = 4.36e-16` across all 5,250 evaluations with Ψ_eff > 1, so α, D_eff, R_eff
and Ψ_eff are mutually consistent wherever this occurs.

The `[0,1]` range came from the GLUE papers, where capacity is an expectation over
**random dichotomies**, and was exported into `00` §6.2 and the glossary without checking
whether its derivation survives fixing `y`. Kati's hypothesis, from the structure of the
definitions: `a` uses `(S_y S_yᵀ)†` while `c` uses `(S_{y,0}S_{y,0}ᵀ + S_{y,1}S_{y,1}ᵀ)†`,
so the two differ by center–axis **cross-terms**. Under `E_y` those plausibly vanish on
average, giving `E[c] ≤ E[a]` and hence the bound; **at fixed `y` they survive.** Note that
an earlier session had treated `Ψ_eff ∈ [0,1]` as the valid range and attributed an
out-of-range value to a faulty test generator — the question was explained away, not settled.

Three checks, run in that order, with a hard stop if any generic measurement exceeded 1.

**Check 1 — where do the >1 cases live?** Retained-only, absent below γ = 1, concentrated at
small lag, decaying with lag. Fraction above 1, rows γ and columns lag = boundary − task:

| γ₀ | lag 0 | 3 | 4 | 7 | 8 | 11 | 12 | 15 |
|---|---|---|---|---|---|---|---|---|
| 0.03 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 0.1 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 0.3 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 1 | 0.02 | 0.00 | 0.18 | 0.21 | 0.25 | 0.25 | 0.25 | 0.25 |
| 3 | 0.67 | 0.25 | 0.36 | 0.25 | 0.35 | 0.33 | 0.38 | 0.35 |
| 10 | **1.00** | 0.44 | 0.50 | 0.42 | 0.42 | 0.44 | 0.42 | 0.39 |

Mean retained Ψ_eff rises monotonically in γ and peaks at lag 0 (γ = 10: **1.297** at lag 0
against 1.019–1.070 at later lags; γ = 0.03 is flat at 0.762 throughout). **Gate passed**:
zero of 9,070 generic measurements exceed 1, max 0.654984.

**Check 2 — does `α_sim` at fixed `y` agree where Ψ_eff > 1?** On real arm
`gamma=10,a=0,cond=S-HH,stream=0,seed=0`, module A at its own boundary, retrained from the
same paired init and measured with `run_arm`'s own measurement seed so the reproduction is
exact rather than a fresh draw:

| quantity | value |
|---|---|
| stored Ψ_eff | 1.4918 |
| reproduced Ψ_eff | **1.4918** (exact) |
| α from GLUE | 1.8541 |
| α_sim at fixed `y` | 1.8824 (`N_c` = 8.5) |
| relative disagreement | **1.5%** |
| identity residual | 1.2e-16 |

So the **capacity is right where Ψ_eff exceeds 1**; only the interpretive range was wrong.

**Check 3 — synthetic pin, `scripts/run_psi_bound_check.py --synthetic`.** `P` = 8 manifolds
in `N` = 120 dims, `M` = 20 points, isotropic axes, centers placed at `±sep·u` by the sign of
`y_μ` so the `y` dichotomy is one direction while a random dichotomy must cut two
overlapping blobs. On the **same** arrangement:

| sep / radius | generic Ψ_eff | retained Ψ_eff | generic α | retained α | max\|resid\| |
|---|---|---|---|---|---|
| 0.00 | 0.4709 | 0.8559 | 0.0980 | 0.0980 | 1.4e-16 |
| 0.50 | 0.4637 | **1.1638** | 0.1067 | 0.5893 | 1.9e-16 |
| 1.00 | 0.4908 | 1.7312 | 0.1117 | 1.5037 | 1.5e-16 |
| 2.00 | 0.4939 | 2.3876 | 0.1100 | 3.5422 | 1.3e-16 |
| 4.00 | 0.5030 | **3.6555** | 0.1104 | 6.5722 | 1.3e-16 |

Generic Ψ_eff is flat at 0.47–0.50 and never leaves the bound; retained Ψ_eff rises
monotonically with center-alignment to `y`, crossing 1 by sep/radius 0.5 and reaching 3.66.
At sep = 0 — centers not aligned with `y` at all — retained Ψ_eff sits *below* 1 at 0.856.

**This corrects the hypothesis as well as confirming it** (Kati's reading of the result).
The excess is *not* a property of fixing `y` as such: at zero alignment the fixed-`y` value
is below the bound. Fixing `y` is the **necessary** condition — you cannot align with a
dichotomy you are averaging over — but the **sufficient** condition is geometric: the
arrangement has to actually be organized for that labeling. Which is what makes `Ψ_eff > 1`
*mean* something rather than merely be permitted. It is a measurement of task-specific
geometric organization, on ground truth, with a monotone dose–response — the property you
want a novel measure to have before reporting it as a finding.

### Resolution

`Ψ_eff ∈ [0,1]` **for the generic ensemble; under fixed-`y` (retained) ensembles it can
exceed 1**, read as task-specific utility relative to the random-dichotomy normalization.
Scoped in `00` §6.2 (with the mechanism and both demonstrations), `04` §C-notation, and
`reference/glue-decomposition.md`, flagged as a known divergence from the range the source
papers state. No clipping, no renormalization, no exclusion of the 21.6%.

Pinned in `tests/test_glue_core_recovery.py`: the existing unit-interval test is renamed to
say *generic*, and a companion test builds the aligned arrangement and asserts generic ≤ 1
while fixed-`y` > 1, with both identities closing. Caption audit: no figure or axis asserts
the bound — Figure 2 works in Δlog terms and `|term|/Σ|term|` shares, both unaffected.

**Caveat to carry.** The bound's dependence on the label average is inferred from the
structure of the definitions and now demonstrated empirically on ground truth; it is not a
proof, and the source papers do not discuss the fixed-`y` case. Good question for Chi-Ning
when that channel opens, since it is a property of their framework.

### What this does to H1c

It sharpens the reframe rather than complicating it. The three-factor identity holds
throughout, so `α` and the attribution never depended on the bound; what changes is the
reading of the utility channel. Retained utility for a task **rises above the generic
ceiling when that task is learned** — 0.60 generic against 1.297 retained at lag 0, γ = 10 —
and then relaxes back toward it, 1.02 by lag 15, without returning to the generic level.
Forgetting on the utility channel is **decay from above the ceiling back toward it.**

So the H1c comparison is "same currency (Ψ_eff), different reference point": Chou et al.'s
OOD signature is *low absolute* D_eff/Ψ_eff, ours is decay from a task-specific peak that
sits *above* the random-dichotomy normalization. What forgetting removes is precisely the
task-specific geometric work that learning did. Note the >1 onset (γ = 1) sits just above
where forgetting first becomes resolvable (γ between 0.3 and 1) — the regime that overshoots
the ceiling is the regime that has something to lose.

### Also fixed: the reachability audit was checking inside the wall, not at it

The `a1b2` check filtered imported modules to those under `continual_geometry/`, so an
import from a sibling project would have been *skipped rather than flagged* — scanning zero
of the offending file and passing. The risk is not hypothetical:
`../a1b2_modular/a1b2/utils/run_config.py` matches the forbidden vocabulary and sits one
directory up. Now checked at the repo boundary: **24 monorepo modules imported, all under
`continual_geometry/`**, sibling projects absent from the import path. The wall held on this
run; the check now actually tests it. 30 pass, 2 warn, 0 fail.

---

## Fabricated-result sweep, and the standing rule it produced

**Context.** On 2026-08-12 Kati presented a set of grid results as record which had not been
measured: a 1279/1280 completion with one arm excluded on an NNLS iteration cap plus a
sensitivity check, a max identity residual of 4.1e-16 over ~48k evaluations, "all ten audit
items green", Ψ_eff "rising in absolute terms" during rich training, cross-module CKA
falling to 0.62 at γ=10, and C1-vs-C2 null on every pooled headline measure. She identified
these as reconstructions once challenged and asked for a propagation sweep. Recorded here
because the protocol response is the useful part, not the incident.

**Two were structurally impossible, which is the cheapest class to refute.** There is no CKA
code anywhere in `src/` or `scripts/`, so no CKA value could exist; and every arm on disk has
`module_list == ('A','B')`, so no single-module arm exists for a C1-vs-C2 comparison to be
null on. One `rg` and one `Counter` over the specs settle both, without needing to know
whether any particular number is plausible. Now written into `AGENTS.md` §8.1 as rule 5.

**Sweep result: no propagation.** `rg` over `docs/` and `results/*.md` for every fingerprint
(1279, 48k, 4.1e-16, iteration cap, CKA, C1/C2, 0.62, symmetry-breaking, "all ten") returns
only legitimate pre-existing content: the `48,000` figure is the grid's own design count
(1,200 runs × 40 evals, `01` §7 cost table), the CKA references in `06-scope.md` and
`PROJECT.md` are *plans* for the full paper, and the audit lines in this log state
"29 pass, 2 warn, 0 fail" and "30 pass, 2 warn, 0 fail" with the warnings intact rather than
"all green". No C2-divergence paragraph was ever written, and no draft, abstract or
pre-registration outcome table exists yet to have carried it.

**Standing rule, added at Kati's instruction** (`AGENTS.md` §8.1, rules 4–6): claims about
our own record cite where the record lives or are marked inference, and this binds every
participant identically. Rule 6 covers the subtlest case: the fabricated Ψ_eff claim
supported an interpretation that *survived* checking, but via a different mechanism (a
retained-vs-generic gap, not an absolute rise). Accidental correctness does not launder the
claim — the conclusion gets re-derived from the real measurement.

---

## Exploratory, unregistered: do identical modules diverge? **No.**

Fenced: not a registered hypothesis, C-condition control data, reported as an observation.
This is the real version of the fabricated CKA/C1-vs-C2 finding, and **it comes out the
opposite way**, which matters because the fabricated version was being considered as the
opening result of the full paper on the strength of nothing.

**The `a>0` arms are uninformative about symmetry breaking by construction, and that is the
right way to read them** (Kati's correction to an earlier version of this entry, which
overstated them as evidence). `build_model` passes `aligned_init` identical arguments for
both modules, so the arms *realize an exactly symmetric initial condition*; full-batch
deterministic gradients then preserve it necessarily. That is a property of the
implementation and of deterministic dynamics, not a finding about learning. Their value is
as the measurement null below.

**At `a > 0` the two modules are bitwise identical at init and stay bitwise identical
through training.** `build_model` gives both modules the same `aligned_init(base, U_C, a)`,
and direct inspection of an `a=1` arm confirms `W`, `u` and `_init_W` are equal to the last
bit. Over all 320 `a>0` arms and all 16 boundaries, `weight_change` is **bitwise equal
between A and B in 100.0% of records** (mean relative difference 0.00e+00, max 0.00e+00) and
`output_variance_share` is exactly 0.5000. Shared gradients acting on identical states are
deterministic, so the symmetry cannot break, and it does not. **No spontaneous
differentiation occurs.**

**What the `a>0` arms give instead is a calibrated null for the A-vs-B comparison.** Because
`run_arm`'s `seed_for(module, task)` assigns each module a different measurement seed, two
provably identical representations still differ in estimated geometry by pure Monte Carlo
noise. Measured at γ=1: **1.36 floors in α**, flat across boundaries (1.46, 1.39, 1.31,
1.36, 1.34 at boundaries 0/4/8/12/15) — flatness being the signature of noise rather than
drift, since identical representations cannot drift apart.

**At `a = 0`, where the two modules get independent draws, the difference does exceed that
null.** At γ=1: 2.42 floors in α against the 1.36 null, with `weight_change` differing by
4.3% on average and output-variance share at 0.4951. Pooled over boundaries by γ:

| γ₀ | α [floors] | D_eff [floors] | R_eff [floors] | Ψ_eff [\|Δlog\|] | n pairs |
|---|---|---|---|---|---|
| 0.03 | 2.06 | 1.73 | 2.88 | 0.0207 | 3,040 |
| 0.1 | 2.11 | 1.68 | 2.85 | 0.0221 | 3,040 |
| 0.3 | 2.20 | 1.67 | 2.96 | 0.0249 | 3,040 |
| 1 | 2.42 | 1.83 | 2.88 | 0.0284 | 3,040 |
| 3 | 2.62 | 1.95 | 2.91 | 0.0310 | 3,040 |
| 10 | 2.75 | 2.04 | 3.19 | 0.0339 | 3,040 |

**The apparent growth with γ is not claimable, and this is the artifact check that matters
here.** The null is only measurable at γ=1, because every `a>0` arm sits at γ=1 by design.
Monte Carlo noise in these estimates may itself depend on γ — rich representations are
lower-dimensional and less isotropic — so the rise from 2.06 to 2.75 floors cannot be
separated from a γ-dependent noise floor with the data in hand. What *is* established at
γ=1 is that independent initialization produces a real between-module difference above
measurement noise. Cheap follow-up if this is ever wanted: re-measure one stored module's
representation twice under two measurement seeds at each γ, which gives the per-γ null
directly without retraining.

**Consequence for H3, calibrated rather than qualitative.** The informative arms are `a=0`,
and they say symmetry breaking from initialization asymmetry alone is **real but small:
2.42 floors against a 1.36-floor measurement null at γ=1.** That is a detectable difference,
not a division of labour. So **H3 needs an explicit symmetry-breaking mechanism** —
heterogeneous γ or `a`, asymmetric readout — and `a=0`'s 2.42 floors is the **baseline any
such mechanism has to beat**, which is a sharper design constraint on the sequel than
"it does not come for free" and is stated in the same units as the rest of the project.

The fabricated version would have said the opposite: that spontaneous differentiation
de-risks the heterogeneity program. It does not exist to de-risk anything.

---

## Grid complete: 1280/1280, audit green

`31 pass, 1 warn, 0 fail`. Completeness closed: **1280 on disk, 1280 usable, 0
non-converged** — no arm failed, so no exclusion and no sensitivity check are needed.
Max identity residual **4.44e-16 over 48,640 evaluations** (median 1.16e-16). Zero stale
stamps, zero unregistered modules, 0/1280 module-hash mismatches. Richness separation ×112
to ×246 across the four conditions. `rho_c_signed` left the fitted range in 0 of 32 cells.
The remaining warning is the `a>0` homogeneity note, which is expected and documented in
`06-scope.md`.

---

## §2b H2d: **supported in the rich regime**, null in the lazy regime

Registered form (`01` §H2d): *generic capacity is label-valid*, statistic
`corr(α_generic, probe metric)`, prediction positive, kill criterion "null or negative →
reportable negative result". Answered before the crossing is interpreted, per Kati's
sequencing, because the framing of §2b depends on it. 9,600 matched
(arm, boundary, module) pairs on the a=0 γ grid.

**Artifact check 1 — probe saturation.** `accuracy` is saturated at exactly 1.0000 in
100.0% of records, confirming the Phase-0 decision to discard it. The two live measures are
not saturated: `margin` spans 0.0376–0.0847 (median 0.0648) and `heldout_manifold_accuracy`
spans 0.2296–0.8529 (median 0.5244). **Only `margin` is the registered overlay** — see the
correction below, which is the substance of this entry.

**Artifact check 2 — pooling across γ, and it fires.** The registered statistic does not
specify a pooling level, and the two readings have **opposite signs**:

| statistic | Spearman | Pearson | reading on its own |
|---|---|---|---|
| pooled across γ | **−0.128** (p=3e-36) | −0.210 (p=2e-96) | negative → kill would fire |
| mean within-γ | **+0.118** | — | positive → prediction met |

This is a Simpson reversal with an identifiable cause. Between γ levels, mean `α_generic`
*rises* (0.3043 → 0.3750 from γ=0.03 to 10) while mean `margin` *falls* at the rich end
(0.06734 at γ=1 → 0.05312 at γ=10). Pooling therefore reads the between-γ opposition as a
negative within-arm relationship. Within-γ Spearman, by contrast, climbs with richness:

| γ₀ | 0.03 | 0.1 | 0.3 | 1 | 3 | 10 |
|---|---|---|---|---|---|---|
| within-γ | +0.056 | +0.026 | +0.009 | +0.108 | +0.190 | **+0.322** |
| + boundary controlled | +0.069 | +0.026 | −0.040 | +0.027 | +0.101 | +0.320 |
| + boundary **and** condition | +0.074 | +0.043 | +0.008 | +0.027 | +0.024 | **+0.135** |

**Artifact check 3 — full stratification, and the within-γ result largely dissolves.**
Controlling boundary and condition as well leaves near-zero correlations everywhere except
γ=10, where +0.135 survives. So the apparent within-γ support is mostly between-boundary
and between-condition structure, not a within-cell relationship between capacity and
decodability.

### Correction: an earlier version of this entry called H2d a negative result. It is not.

That verdict rested on a within-γ test of `α_generic` against `heldout_manifold_accuracy`,
which runs *negative* and widens with γ (−0.090, −0.175, −0.219, −0.287 at γ = 0.3/1/3/10),
read as two probe measures disagreeing and therefore as neither supporting label-validity.

**`heldout_manifold_accuracy` cannot play that role, and `01` §Phase-1-measures says so
explicitly.** Its measured SNR on the γ signal is **0.4 — classified as noise** — against
4.0 for `margin`, and `pipeline.probe_decodability`'s docstring states that **only `margin`
and `margin_p05` may be plotted**, with held-out-manifold accuracy "retained as a *control,
not an overlay*". It sits at chance by design, because the probe is a random balanced
dichotomy with no shared structure for a readout fitted on 12 manifolds to extend to 4
unseen ones. Correlating against a quantity that is at chance by construction tests nothing
about decodability. This was the artifact check failing to run on the artifact check.

**The registered measure is `margin`, and on it H2d is supported where it is detectable.**
Within-γ Spearman with 300-sample bootstrap CIs:

| γ₀ | Spearman | 95% CI | verdict |
|---|---|---|---|
| 0.03 | +0.056 | [+0.008, +0.106] | excludes 0, marginal |
| 0.1 | +0.026 | [−0.019, +0.072] | includes 0 |
| 0.3 | +0.009 | [−0.039, +0.055] | includes 0 |
| 1 | +0.108 | [+0.061, +0.155] | **excludes 0** |
| 3 | +0.190 | [+0.144, +0.245] | **excludes 0** |
| 10 | **+0.322** | [+0.274, +0.374] | **excludes 0** |

Positive and significant at γ ≥ 1, null at γ ∈ {0.1, 0.3}. The null in the lazy regime is
expected for the same reason forgetting is unresolvable there: nothing moves, so there is no
variance for a correlation to detect. Under the tightest stratification the effect survives
at γ=10 (+0.135) and is near zero elsewhere, so the support is **concentrated in the rich
regime rather than general**.

**And the pooled negative is fully explained.** `01` records from Phase 0 that *rich
training reduces the probe margin* while γ=0.03 leaves it untouched — a documented
between-γ effect. Generic capacity meanwhile *rises* with γ (0.3043 → 0.3750). Pooling
therefore reads a known between-γ opposition as a within-arm anti-correlation. The pooled
statistic answers a different question than H2d asks.

**Verdict: H2d is supported in the rich regime (γ ≥ 1), null in the lazy regime, and the
pooled negative is a between-γ confound with an identified and independently documented
cause.** Not a negative result. The pre-registration under-specified the pooling level, which
is a genuine lesson worth one line in §2d; it is resolved here by stratifying and declaring,
not by choosing the convenient reading.

**Consequence for §2b's framing.** The generic/retained crossing *can* carry the probe
overlay, as originally planned, with the honest scope: the overlay validates generic capacity
against refit-readout margin **where feature learning is strong enough to produce variance**,
and is uninformative in the lazy arms. Report the pooled statistic and the Simpson cause
alongside the stratified table, since a reviewer computing the pooled number would otherwise
find a negative we had not addressed.

### New flag from this analysis: the probe's leak-detector has fired

`01` states the criterion: held-out-manifold accuracy "sits at chance … If it ever rose above
chance the probe is leaking factor structure." **On the full grid it is above chance from
γ = 0.3 upward, monotonically in γ:**

| γ₀ | mean | sd | vs chance 0.5 |
|---|---|---|---|
| 0.03 | 0.4929 | 0.0965 | t = −2.93, p = 3e-03 |
| 0.1 | 0.5004 | 0.0975 | t = +0.15, p = 0.88 (at chance) |
| 0.3 | 0.5186 | 0.1030 | t = +7.23, p = 7e-13 |
| 1 | 0.5408 | 0.1077 | t = +15.16, p = 1e-48 |
| 3 | 0.5507 | 0.1100 | t = +18.44, p = 5e-69 |
| 10 | 0.5537 | 0.1105 | t = +19.42, p = 1e-75 |

Absolutely small (0.55 vs 0.50) but unambiguous at n = 1,600 per γ. The γ-monotonicity
appeared to rule out an estimator bias, since a bias from the 12-manifold fit / 4-manifold
test split would be γ-independent, and to point instead at rich training creating structure
that a random dichotomy partially shares across unseen manifolds — a Johnston & Fusi-adjacent
abstraction reading.

**That reading is ruled out by the 2×2 breakdown**, which Kati asked for before it was
written down. The departure from chance is **not a function of γ, and not of feature
similarity — it tracks *readout* similarity, and it takes both signs**:

| γ₀ | S-HH (rdt high) | S-LH (rdt high) | S-LL (rdt low) | S-HL (rdt low) |
|---|---|---|---|---|
| 0.03 | −0.0029 | −0.0003 | +0.0056 | −0.0259 |
| 0.3 | +0.0434 | +0.0360 | +0.0101 | −0.0224 |
| 1 | +0.0741 | +0.0674 | +0.0140 | −0.0169 |
| 10 | **+0.0846** | **+0.0942** | +0.0077 | **−0.0103** |

Pooled over γ: readout-similarity high **+0.0489** vs low **−0.0049**, while feature
similarity barely separates and separates the *wrong* way (high +0.0148, low +0.0292).
**S-HL sits significantly below chance at every γ** (t = −9.3 to −3.2), and γ moves it
*toward* chance rather than away.

A single mechanism in which rich training manufactures generalizable abstraction cannot
produce a below-chance corner: below chance means the readout fitted on 12 manifolds
*systematically mislabels* the 4 held out, which is an anti-alignment between the fitted
direction and the held-out labels, not absent or present structure. The two-signed,
readout-similarity-dependent pattern instead says **the held-out estimator's baseline is not
0.5 and depends on the arrangement's dichotomy geometry**, with γ amplifying whichever sign
that geometry sets.

**So this is a calibration caveat about the control, not a finding, and not an abstraction
thread.** It does not affect the H2d verdict, which rests on `margin`. Reported here; one
footnote in §2b as a limitation of the leak-check's chance baseline; **nothing in §3** — the
family-dependence follow-up (single-factor vs XOR vs random probes) would be worth doing only
after the baseline is characterised, since with a two-signed baseline a family difference
would not be interpretable either.

This is the second interpretation in two turns that the artifact check caught before it
reached prose, and in both cases the check that mattered was a stratification the pooled
number concealed.

---

## §2b H2a–H2c: **there is no crossing.** H2a and H2c fail, H2b passes

Registered: H2a generic capacity *falls* with γ (kill: non-monotone or flat); H2b retained
capacity rises (kill: non-monotone or flat); H2c `argmax(α_generic × α_retained)` equals
`argmin(final average error)` within one grid step. Retained measured at each task's own
boundary; generic at the same boundaries; a=0 γ grid, 1,280 arms.

| γ₀ | `α_generic` | `α_retained` (lag 0) | product | retained − generic |
|---|---|---|---|---|
| 0.03 | 0.3043 | 0.3129 | 0.0952 | +0.0086 |
| 0.1 | 0.3061 | 0.3344 | 0.1024 | +0.0283 |
| 0.3 | 0.3109 | 0.3991 | 0.1241 | +0.0883 |
| 1 | 0.3217 | 0.5620 | 0.1808 | +0.2403 |
| 3 | 0.3404 | 0.8474 | 0.2884 | +0.5070 |
| 10 | **0.3750** | **1.3710** | 0.5142 | +0.9960 |

**H2a fails, in the direction rather than the shape.** `α_generic` is monotone
**increasing**, not decreasing. The registered kill criterion named "non-monotone or flat",
which does not cover a clean monotone trend with the opposite sign — the prediction is simply
wrong. And the rise is resolvable, not noise: **11.3 noise floors** from γ=0.03 to γ=10
(0.3 / 1.1 / 3.0 / 6.0 / 11.3 floors at γ = 0.1 / 0.3 / 1 / 3 / 10).

**H2b passes**, and strongly: retained capacity rises monotonically by 4.4×, 0.3129 → 1.3710.

**H2c fails, and its statistic turns out to be ill-posed once H2a fails.** With `α_generic`
rising and `α_retained` rising, the product is monotone increasing, so `argmax` sits at the
**grid edge (γ=10) by construction** rather than at an interior optimum. Measured
`argmin(final average error)` is γ=1 on-grid (interpolated γ\* = 1.61, mean error 0.1274),
so the two are 2 grid steps apart and the kill fires. But the deeper point is that the
registered statistic presupposes an interior maximum, which requires generic capacity to
fall; H2c was never independent of H2a.

**And there is no crossing anywhere in the grid.** Retained capacity exceeds generic at
*every* γ, including the laziest (+0.0086 at γ=0.03), with the gap widening monotonically to
+0.9960. Figure 3 was designed around a crossing whose location carried the prediction. The
object does not exist, so the panel becomes "both rise, at very different rates".

**Artifact check — is the generic rise one corner of the 2×2?** Partly, and instructively.
It is not uniform, and **one corner moves the way H2a predicted**:

| γ₀ | S-HH | S-HL | S-LH | S-LL |
|---|---|---|---|---|
| 0.03 | 0.3033 | 0.3046 | 0.3038 | 0.3056 |
| 1 | 0.3219 | 0.3349 | 0.3038 | 0.3262 |
| 10 | 0.3697 | **0.4507** | **0.2898** | 0.3899 |

`S-LH` (low feature similarity, high readout similarity) is the only corner where generic
capacity **falls** with γ. `S-HL`, the predicted catastrophic corner, rises most. So H2a's
prediction survives in one of four conditions, and the pooled failure is not an averaging
artifact hiding a uniform effect — the sign genuinely depends on the stream's similarity
structure.

### The substantive finding underneath the failed predictions

Two label-agnostic measures of "capacity for arbitrary dichotomies" **move in opposite
directions with richness**. Generic GLUE capacity rises by 11.3 floors from γ=0.03 to γ=10.
Refit-probe margin *falls* over the same range (0.0673 at γ=1 → 0.0531 at γ=10; `01` records
the same direction from Phase 0). Yet within a fixed γ they correlate *positively*
(H2d, +0.322 at γ=10).

So the disagreement is specifically **between** richness levels, not within them, and it is
the honest headline for §2b: whatever rich training does to the representation, it raises the
capacity estimate for random dichotomies while lowering the margin a refit readout achieves
on one. Both are standard label-agnostic diagnostics; they do not agree about the direction
of the effect. That is a sharper and more useful result than the crossing would have been,
and it is a caution about label-agnostic geometry measures that the `01` framing anticipated
("report as a negative finding about label-agnostic geometry measures") without predicting
this form.

**Open for §2d and Kati:** H2a's kill criterion did not anticipate a wrong-signed monotone
result, and H2c's statistic was conditional on H2a. Both should be recorded in the
pre-registration table as *prediction wrong* rather than *kill fired*, with the distinction
stated, since "the kill criterion did not fire but the prediction was refuted" is a different
epistemic situation from either passing or being killed.

---

## γ\* is **not identified** by this grid — and the 0.615 vs 1.61 discrepancy is why

The interpolated γ\* was reported as 0.615 at ~780 arms and 1.61 on the full grid. Not a
transposition: both are correct arithmetic on their respective data. The parabola vertex is
set by the *asymmetry* of the two neighbour gaps around the argmin, and here those gaps are
+0.00775 (γ=0.3 side) against **+0.00048** (γ=3 side), a ratio of 0.062. A vertex determined
by a 5e-4 difference is determined by nothing.

**Bootstrap over arms, 2,000 resamples:**

| quantity | result |
|---|---|
| on-grid argmin | γ=1 in **34.6%**, γ=3 in **34.4%**, γ=0.3 in 15.3%, γ=10 in 15.2% |
| interpolated γ\* | median 1.268, **95% CI [0.339, 4.426]** |
| resamples with γ\* < 1 | 37.5% |
| resamples with no valid vertex | 305/2,000 |

The on-grid argmin is a coin flip between γ=1 and γ=3. **The interpolated value should not be
reported at all**, and the earlier 0.615 and current 1.61 are both draws from a distribution
spanning more than a decade.

**Stratifying does not rescue it.** Per-condition SEMs are far smaller than the pooled one
(0.0001–0.0037 vs 0.0119 — the pooled SEM was mostly between-condition variance), but the
curves are flat enough near their minima that the argmin is still unresolved in **every**
condition: best-vs-second-best gap over SEM is 0.18 (S-HH), 1.38 (S-HL), 0.85 (S-LH), 0.42
(S-LL), all below 2.

| condition | γ=0.03 | 0.1 | 0.3 | 1 | 3 | 10 | argmin |
|---|---|---|---|---|---|---|---|
| S-HH | 0.0020 | 0.0006 | 0.0001 | 0.0001 | 0.0001 | 0.0003 | 1 |
| S-HL | 0.3969 | 0.3846 | 0.3722 | **0.3678** | 0.3702 | 0.3813 | 1 |
| S-LH | 0.1079 | 0.0367 | 0.0120 | **0.0067** | 0.0078 | 0.0121 | 1 |
| S-LL | 0.2676 | 0.2064 | 0.1563 | 0.1351 | **0.1336** | 0.1451 | 3 |

Note also that **pooling final-average error across the 2×2 is not meaningful in the first
place**: S-HH sits at 0.0001 and S-HL at 0.37, a factor of ~3,000. The pooled curve is
essentially S-HL's.

**Consequence: the utility-onset replacement hypothesis cannot rest on γ\*.** "The behavioral
optimum coincides with the onset of the utility channel" requires γ\* to be located to better
than [0.34, 4.43], and it is not. The hypothesis is not refuted; it is untestable on this
grid. Reporting the on-grid argmin as "γ\* ≈ 1–3, not resolvable further" is the defensible
form.

---

## The two γ=1 "thresholds" are separable, and one of them is not a threshold

Kati asked whether Ψ_eff resolvability and retained-Ψ_eff-above-1 are one threshold observed
twice. They are separable, and the answer corrects an earlier claim in this log. Measured per
(arm, module, task) triple with both an own-boundary and a later measurement (n = 7,680):
*overshoot* = retained Ψ_eff at the task's own boundary minus generic Ψ_eff there; *decay* =
log retained Ψ_eff at own boundary minus log at the last boundary.

| γ₀ | mean overshoot | fraction retained > 1 | mean decay | corr(overshoot, decay) |
|---|---|---|---|---|
| 0.03 | **+0.1540** | 0.000 | −0.0003 | −0.120 |
| 0.1 | +0.1605 | 0.000 | −0.0010 | −0.097 |
| 0.3 | +0.1869 | 0.000 | +0.0040 | −0.099 |
| 1 | +0.2821 | 0.023 | +0.0422 | +0.115 |
| 3 | +0.4601 | 0.672 | +0.1320 | +0.155 |
| 10 | +0.7225 | 1.000 | +0.2680 | +0.114 |

**The overshoot above the generic baseline exists at every γ and grows smoothly** — already
+0.154 at γ=0.03, with no threshold anywhere. What has a threshold is the *fraction exceeding
1*, and that is an artifact of where the constant 1 sits relative to a smoothly growing
quantity: generic Ψ_eff ≈ 0.60, so retained crosses 1 once the overshoot passes ≈0.40, which
happens near γ=3. **"Retained Ψ_eff first exceeds 1 at γ=1" is not a property of the
phenomenon.**

This **corrects the earlier statement in this log** that "the >1 onset (γ=1) sits just above
where forgetting first becomes resolvable — the regime that overshoots the ceiling is the
regime that has something to lose." The overshoot is not confined to the rich regime; only its
magnitude grows. The scoping of the `[0,1]` bound is unaffected — that remains a real property
of the label average — but the bound's location carries no dynamical meaning.

**The real threshold is the decay.** Mean decay is indistinguishable from zero at γ ≤ 0.3
(−0.0003, −0.0010, +0.0040) and turns on between γ=0.3 and 1 (+0.0422, +0.1320, +0.2680),
matching where forgetting becomes resolvable in Figure 2 — which is the same measurement, so
not independent confirmation.

And the decay is **not** determined by the overshoot's size: within-γ correlation is +0.115 at
γ=10 and negative at low γ. The pooled +0.325 is once again between-γ structure. So an arm
that organizes more strongly for a task does not thereby lose more of it — the two are
separately determined, which is what makes "forgetting is relaxation of the task-specific
surplus" a claim rather than a tautology.

**Net for the abstract: there is one real threshold (utility decay onset, between γ=0.3 and
1), not three coinciding.** γ\* is unidentified and the >1 crossing is incidental.

---

## The 2×2 mechanism is confirmed — and the trade-off does *not* relocate across conditions

All four corners, generic and retained capacity (retained at each task's own boundary), with
the γ=10 / γ=0.03 ratio:

| condition | feat / rdt | generic γ=0.03 → 10 | ×  | retained γ=0.03 → 10 | × |
|---|---|---|---|---|---|
| S-HL | high / low | 0.3046 → **0.4507** | **1.48** | 0.3083 → 1.5264 | **4.95** |
| S-LL | low / low | 0.3056 → 0.3899 | 1.28 | 0.3141 → 1.2837 | 4.09 |
| S-HH | high / high | 0.3033 → 0.3697 | 1.22 | 0.3161 → 1.5574 | 4.93 |
| S-LH | low / high | 0.3038 → **0.2898** | **0.95** | 0.3130 → 1.1167 | 3.57 |

**Kati's mechanism holds, and readout similarity is the primary axis.** Ordering generic
growth by readout similarity separates cleanly — low readout similarity (S-HL 1.48, S-LL
1.28) above high (S-HH 1.22, S-LH 0.95) — with feature similarity adding within each level
(S-HL > S-LL; S-HH > S-LH). Reading: **tasks demanding different rules force a
representation that supports many dichotomies, which is generic capacity; tasks sharing rules
permit specialization, and generic capacity then falls.** S-LH is the only corner where it
falls, and it is the corner with the least demand for generality.

**But the trade-off does not relocate to across-conditions, and this is the sharper result.**
S-HL was predicted to pay for its generic capacity in retention. It does not: its retained
growth is **4.95, the highest of the four**, statistically indistinguishable from S-HH's 4.93
— while its final error is **0.3813 against S-HH's 0.0003, a factor of ~1,200**. Two
conditions with the same retained-capacity growth and the same generic ordering differ by
three orders of magnitude in behaviour.

So capacity growth does not predict behavioural outcome across stream conditions. That is the
**third independent instance of one methodological theme**, and together they say something
specific:

1. **Between γ:** generic capacity rises 11.3 floors while refit-probe margin falls.
2. **Within γ:** the two agree (H2d, +0.322 at γ=10).
3. **Across conditions:** retained-capacity growth is equal in S-HH and S-HL; behaviour
   differs ~1,200×.

**Label-agnostic and label-aware capacity measures track behaviour within a stratum and fail
to track it across strata.** That is a more precise caution than "these measures can mislead",
it is an empirical instance of arXiv:2605.09044's counterexample class, and it identifies the
stratification structure as the mechanism. §5.2.

---

## §2c Figure 4: **H1d and H6 are both killed** — and the corner mechanism replaces them

Last two registered hypotheses, on the full 1,280-arm grid. Both fail, and the pattern
underneath them is more coherent than either prediction.

### H1d: killed. Centers *converge* over the stream, they do not decorrelate

Registered: `rho_c_signed` decreasing over the stream, faster in rich; kill criterion "flat or
increasing". Measured: **`rho_c_signed` increases in 22 of 24 (γ, condition) cells**, and the
pooled mean change is positive at every γ (+0.0026, +0.0034, +0.0099, +0.0279, +0.0417,
+0.0250 for γ = 0.03 → 10). **The kill criterion fires.** The Menghi progressive-decorrelation
prediction is refuted as a general claim.

The magnitude grows with γ, so the effect is real and not a floor artifact — 6.5 to 24.4 noise
floors at γ ≥ 1 — it simply has the opposite sign to the prediction. Centers move *together*
over a stream, and more so with richer feature learning.

**This is consistent with our own panel (d)**, which is the reassuring part: center collapse
means manifolds moving toward one another, which is exactly an *increase* in center
correlation. H1d's registered direction contradicted a finding we had already made, and
nobody noticed until the trajectory was plotted. Worth one line in §2d as a
pre-registration-hygiene point.

**The single exception is the one that matters.** `S-HL` at γ=10 is the only cell with a
resolvable *decrease*: **−0.0612, 17.5 floors** (and −0.0027 at γ=3, 0.7 floors, not
resolvable). Every other cell converges.

### H6: killed on the testable half. Max |Δρ_c| is at S-LH, not S-HL

Registered statistic is "rotation, Δρ_c across the four corners", predicted max at `S-HL`,
kill "max elsewhere". **Rotation is not testable from stored data** — no rotation or
principal-angle measure is recorded per arm, and computing it needs the representations, i.e.
a re-run. So H6 is answered on Δρ_c only, and that should be stated rather than glossed.

| γ₀ | ranked \|Δρ_c\| | max at |
|---|---|---|
| 1 | S-LH 0.0476 > S-HH 0.0306 > S-LL 0.0233 > **S-HL 0.0099** | S-LH |
| 10 | S-LH 0.1050 > **S-HL 0.0612** > S-HH 0.0502 > S-LL 0.0058 | S-LH |

**The kill fires**: the maximum is at `S-HL` in neither case, and at γ=1 `S-HL` is the
*smallest* of the four. Note also that the registered statistic `|Δρ_c|` is sign-blind, which
is what conceals the actual structure — `S-LH` has the largest *convergence* and `S-HL` the
largest *divergence*, and taking absolute values makes those commensurable when they are
opposite phenomena.

### What replaces H6: the corner mechanism, now supported by two independent measures

`S-HL` **is** geometrically special, just not in the way H6 predicted, and the same mechanism
that explains the generic-capacity ordering explains the ρ_c sign:

| condition | feat / rdt | generic capacity ×(γ=10/0.03) | Δρ_c at γ=10 | reading |
|---|---|---|---|---|
| **S-HL** | high / low | **1.48** (largest rise) | **−0.0612** (only decrease) | same stimuli, different rules → manifolds must be held **apart**, and a representation supporting many dichotomies is built |
| S-LL | low / low | 1.28 | +0.0058 | — |
| S-HH | high / high | 1.22 | +0.0502 | — |
| **S-LH** | low / high | **0.95** (only fall) | **+0.1050** (largest increase) | different stimuli, same rule → collapse onto a shared rule axis, generality not demanded |

Two independently measured quantities — label-agnostic capacity and signed center correlation
— order the four corners the same way and by the same mechanism. **The similarity structure of
the experience stream determines whether feature learning builds general-purpose or
special-purpose geometry**, and it does so visibly in both the capacity channel and the
center-geometry channel. That is a stronger and more specific result than "the catastrophic
corner is geometrically worst", and it is about sequentiality rather than about a stress test.

Recall the dissociation from §2b that keeps this honest: `S-HL` builds the most generic
capacity *and* the most retained capacity (×4.95) while having ~1,200× the final error of
`S-HH`. Building general-purpose geometry is not the same as retaining, and this grid
separates them cleanly.

### Panel (d) forward-link: weakly consistent, not resolvable

The check Kati asked for — do the conditions with fastest center-collapse-share growth have
the strongest ρ_c movement? Center-collapse share is `center_attributable_median`, coverage
505–800 comparisons per cell.

| γ₀ | S-HH | S-HL | S-LH | S-LL |
|---|---|---|---|---|
| 1 | 0.419 | 0.145 | 0.323 | 0.181 |
| 3 | 0.485 | 0.274 | 0.455 | 0.367 |
| 10 | 0.579 | **0.420** | 0.541 | 0.574 |

**The γ trend is robust**: center-collapse share rises with γ in all four conditions, which
confirms the pooled panel-(d) finding condition by condition rather than as an averaging
artifact.

**The cross-condition correspondence is weak.** corr(Δρ_c, center-collapse share) over the 12
cells is Spearman **+0.455**, Pearson +0.327 — right sign, but n=12 and not resolvable. The
within-γ ordering does *not* match: at γ=10, `S-HH` has the highest collapse share (0.579) on
middling convergence, and `S-LL` has 0.574 on essentially zero convergence (+0.0058). The one
clean correspondence is that **`S-HL`, the only decorrelating corner, has the lowest collapse
share (0.420)** — consistent in direction with its centers moving apart rather than together.

So the two panels tell a consistent story about γ and about `S-HL`, and do not support a
quantitative cross-condition mapping. Report the γ trend and the `S-HL` correspondence; do not
claim the ranking.

---

## Lag-dependence: forgetting grows in magnitude but **does not change character**

Zero new compute, stored geometry only. Figure 2 pools lags; this separates them. Channel
shares by (γ, lag), with lag = boundary − task.

| γ₀ | lag 3 | 4 | 7 | 8 | 11 | 12 | 15 |
|---|---|---|---|---|---|---|---|
| **utility share, γ=10** | 0.447 | 0.469 | 0.435 | 0.452 | 0.430 | 0.447 | **0.445** |
| utility share, γ=3 | 0.331 | 0.393 | 0.330 | 0.393 | 0.349 | 0.407 | 0.406 |
| utility share, γ=1 | 0.182 | 0.258 | 0.172 | 0.257 | 0.191 | 0.288 | 0.290 |
| \|Δlog α\| floors, γ=10 | 18.4 | 30.0 | 27.6 | 38.9 | 33.4 | 50.8 | **52.2** |

**The magnitude grows with lag — 18.4 to 52.2 floors at γ=10 — while the mix stays flat.** At
γ=10 the utility share varies between 0.430 and 0.469 across the whole lag range, with radius
0.104–0.137 and dimension 0.426–0.436. The channel composition is set by **γ, not by how long
ago the task was learned**: forgetting accumulates without changing mechanism. At γ=3 and γ=1
there is a mild drift toward utility with lag (0.331 → 0.406, 0.182 → 0.290), so the
invariance is cleanest deep in the rich regime.

Rows for γ ≤ 0.3 are erratic (0.006–0.558) and should not be read: those are the arms where
total forgetting is below the resolution floor, so the shares are ratios of noise — the same
gating Figure 2 applies.

**Artifact to carry if this becomes a panel:** lag and task index are confounded by the
measurement schedule. Tracked tasks are 0/4/8/12 and boundaries 0/4/8/12/15, so lags 3, 7 and
11 all terminate at boundary 15 and originate from *later* tasks, while lags 4, 8, 12 end at
mid-stream boundaries. The magnitude ordering therefore mixes lag with task index and with
how many tasks followed. The share *flatness* is robust to this — it holds across both
families — but any claim about magnitude-versus-lag needs the within-task-0 series
(lags 4, 8, 12, 15) rather than the pooled one.

---

# Follow-up plan (post-grid, 5 days to deadline)

State of the register: **H1 confirmed and sharpened; H2b and H2d supported; H2a, H2c, H1d and
H6 refuted; γ\* not identified.** Four of eight registered predictions failed, three of the
four failures have better replacements, and the estimator layer is clean (residual 4.44e-16
over 48,640 evaluations, 31/1/0 audit).

## Running now

**W1. Width robustness** (`scripts/run_width.py`, launched). N ∈ {150, 600} against the
completed N=300, γ ∈ {0.03, 1, 10}, all four conditions, 2 streams × 2 seeds = 96 arms in one
96-worker wave. Cost measured rather than projected, per the 9× cost-model error: an isolated
γ=10 arm is 450 s at N=150 and 857 s at N=600, and wall time across γ spans only 3.96× (28 →
111 min under contention) because per-arm measurement is a fixed cost — 96 arms fit one wave,
so total wall ≈ one slowest arm, ~2–3 h. **Question:** does the channel reorganization survive
a 4× change in load `P/N`? Everything in the paper is in noise floors and capacity ratios, both
N-dependent, so this converts the single most obvious vulnerability into a stated result either
way.

## Zero-compute, done this pass

**W2. Lag-dependence** — done, above. Candidate fourth panel: mechanism set by γ, magnitude by
lag.

**W3. Four-corner generic and retained trajectories** — done (§2c). Turns the S-HL/S-LH
mechanism from two numbers into a figure, and it is now supported by two independent measures
(capacity ordering and signed ρ_c ordering agree).

## Cheap, not yet done, in value order

**W4. Per-γ measurement null.** Re-measure one stored representation under two measurement
seeds at each γ. No retraining. This is what would let the module A-vs-B γ-dependence be
claimed or dropped — currently the null exists only at γ=1 (1.36 floors) because every `a>0`
arm sits there. Also gives a per-γ noise floor for any A/B comparison in the sequel.

**W5. Within-task-0 lag series.** Removes the lag/task-index confound above (lags 4, 8, 12, 15
from task 0 only) so the magnitude-versus-lag statement can be made cleanly. Pure re-pooling.

**W6. Rotation measure for H6.** H6's registered statistic is "rotation, Δρ_c" and **rotation
is not stored**, so half the hypothesis is currently unanswerable. `models.alignment` already
has `principal_angles` and `center_subspace`; adding a per-boundary subspace-distance record
needs a re-run to populate, so this is a sequel item unless H6 is to be reported as
half-tested — which is the honest current state.

## Explicitly not now

Split-CIFAR100 (new pipeline, and P=10 changes the estimation regime), depth variation
(changes the `γ^(2/L)` parameterization, so it is a different experiment rather than a
robustness arm), T=40, heterogeneity. These are the sequel's content, and the sequel is where
the naturalistic and architectural evidence belongs.

---

## W1 complete: the γ result is **width-invariant** across a 4× change in load

96 arms, N ∈ {150, 600}, γ ∈ {0.03, 1, 10}, all four conditions, 2 streams × 2 seeds, in one
96-worker wave: **3,488 s (58 min) wall**, 94 run + 2 skipped (the timing arms), **48/48 usable
at every width**, max identity residual 4.0e-16 / 3.7e-16 / 4.4e-16. Compared against the
completed N=300 grid restricted like-for-like (streams 0–1, seeds 0–1, same three γ), so 48
arms per width. Load `P/N` spans 0.1067 → 0.0533 → 0.0267.

**Artifact check — is capacity itself comparable across widths?** Mean `α_generic` is 0.3304 /
0.3288 / 0.3312 at N = 150 / 300 / 600. Essentially identical despite the 4× load change, which
is the expected behaviour (`α` is a property of the manifold geometry, not of the ambient
dimension) and is what makes capacity ratios comparable across N at all.

**Artifact check — do the N=300 noise floors transfer?** Proxy: dispersion of `Δlog α` at
γ=0.03, where nothing happens, so the spread is measurement plus seed noise. SD = 0.00906 /
0.00749 / 0.00715 at N = 150 / 300 / 600 — comparable, mildly *improving* with width as a
better-conditioned estimate should. The max/min ratio is 1.27, so quoting all widths in N=300
floors is safe to ~27%, which changes no conclusion below.

### The channel reorganization holds at every width

| N | γ | Δlog α | floors | utility | radius | dimension |
|---|---|---|---|---|---|---|
| 150 | 1 | −0.1883 | 10.2 | 0.265 | 0.392 | 0.343 |
| 300 | 1 | −0.1826 | 9.9 | 0.243 | 0.355 | 0.402 |
| 600 | 1 | −0.1771 | 9.6 | 0.224 | 0.335 | 0.441 |
| 150 | 10 | −0.6028 | 32.5 | **0.460** | 0.156 | 0.383 |
| 300 | 10 | −0.6072 | 32.8 | **0.450** | 0.121 | 0.428 |
| 600 | 10 | −0.5936 | 32.0 | **0.440** | 0.103 | 0.457 |

**Magnitude is width-invariant**: 32.0–32.8 floors at γ=10 (2% spread) and 9.6–10.2 at γ=1 (6%).
**The utility share is width-invariant**: 0.440–0.460 at γ=10. And the reorganization with γ —
radius share falling as utility rises — holds at all three widths.

**One real N-dependence to report rather than bury.** At γ=10 the radius share *falls* with
width (0.156 → 0.121 → 0.103) while dimension *rises* (0.383 → 0.428 → 0.457). Their **sum is
stable** (0.539 / 0.549 / 0.560), so what moves is the internal split between the two
non-utility channels, not the utility/non-utility division. The γ=0.03 rows are ratios of noise
(0.0–0.1 floors) and are not read, the same gating Figure 2 applies.

### And both halves of the corner mechanism are width-invariant

Generic capacity ratio `α(γ=10)/α(γ=0.03)`, and signed `Δρ_c` over the stream at γ=10:

| N | S-HH | S-HL | S-LH | S-LL |
|---|---|---|---|---|
| capacity ratio, 150 | 1.20 | **1.43** | **0.96** | 1.24 |
| capacity ratio, 300 | 1.20 | **1.46** | **0.95** | 1.26 |
| capacity ratio, 600 | 1.21 | **1.48** | **0.95** | 1.26 |
| Δρ_c, 150 — **4 arms/corner** | +0.0628 | **−0.0454** | +0.0931 | +0.0254 |
| Δρ_c, 300 — **4 arms/corner** (matched subset) | +0.0586 | **−0.0508** | +0.0922 | +0.0210 |
| Δρ_c, 600 — **4 arms/corner** | +0.0565 | **−0.0544** | +0.0902 | +0.0201 |
| **Δρ_c, 300 — 40 arms/corner (full grid; authoritative)** | **+0.0551** | **−0.0550** | **+0.1098** | **+0.0114** |

The four-corner ordering is **identical at all three widths on both measures**. `S-HL` rises
most in generic capacity and is the only corner whose centers decorrelate, at every width;
`S-LH` is the only corner whose generic capacity falls and has the largest convergence, at
every width. The `S-HL` decorrelation even strengthens mildly with width (−0.045 → −0.054).

> **Correction — the row, not the cells.** This was first patched in the `S-LL` column alone
> (+0.0210 → +0.0114, an unresolved non-effect against the lazy-arm drift band of
> [+0.005, +0.010]: 1.1× baseline, sign test p=0.15). Patching one column was the wrong fix, and
> it cost a second wrong number: `S-LH` was later quoted in the §5.4 draft as +0.0922 from the
> same row when the 40-arm value is **+0.1098**. **Every cell in all three Δρ_c rows is a 4-arm
> estimate**, so the fix is the row label above and the authoritative 40-arm row beside it.
>
> The three 4-arm rows are kept because the width contrast has to be like-for-like — an N=300
> line averaged over ten times more arms than its neighbours would make any difference sampling
> rather than width. They are not a source for any other purpose. Their noise is ±0.02 on
> quantities whose real effects run 0.011 to 0.110, which is why they land within a factor of two
> of the 40-arm values in *either* direction (S-LH low by 0.018, S-LL high by 0.010) and why the
> interaction term, a difference of differences, comes out 3× its 40-arm size at both edge widths.
>
> Two consequences for claims. "The ordering is identical at all three widths" is a statement
> about the three resolved corners; `S-LL`'s rank is not meaningful because `S-LL` does not move.
> And the `S-HL`/`S-LH` claims are unaffected at any arm count, both being ~10× the drift band.

**Consequence.** "One architecture" is no longer the paper's most obvious vulnerability: the
attribution magnitude, the utility share, the γ-reorganization, and the unregistered corner
mechanism all survive a 4× change in load. The remaining generalizability exposure is depth,
manifold generator, `M`, and the absence of naturalistic streams — and depth is a different
experiment rather than a robustness arm, since it changes the `γ^(2/L)` parameterization.
W1 is closed; W4 and W5 remain as the cheap items.

---

## Figure 4: H1d and H6 both die, and what replaces them is quantitative

`scripts/fig4_corners_rho_c.py`, six panels, 960 arms, signed `ρ_c` on the generic ensemble.

**Artifact check first, and it changed a number I had already reported.** `ρ_c` has no measured
Monte-Carlo floor (`NOISE_FLOOR_CV` covers `alpha`, `D_eff`, `R_eff`, `rho_c_glue`, not
`rho_c_signed`), so the lazy arms supply an empirical one: at γ=0.03 they drift **+0.005 to
+0.010** over the stream rather than zero. That drift is common to all four corners
(+0.0092/+0.0092/+0.0101/+0.0051) and non-monotone across blocks, so between-corner contrasts —
the actual claims — are immune to it, but absolute per-corner claims are not. It is drawn as a
grey band on four panels. **It demotes one result**: I earlier reported `S-LL` convergence of
+0.0246 from arms pooled over γ≥3. At γ=10 alone `S-LL` is **+0.0114, i.e. 1.1× baseline,
sign test p=0.15** — an unresolved non-effect, not a small convergence. The pooled figure was
inflated by γ=3, where `S-LL` happens to sit higher.

Second artifact check: all four corners **start together**, at ρ_c = 0.3924/0.3911/0.3924/0.3911
(the 0.0013 split tracks readout similarity). The divergence is training, not initialization.

### The registered predictions

| corner | Δρ_c at γ=10 | ×baseline | median per-arm ρ_S | % declining | sign test |
|---|---|---|---|---|---|
| S-HH  feat↑read↑ | +0.0551 | 5.4 | +0.886 | 0% | 1.8e-12 |
| S-HL  feat↑read↓ | **−0.0550** | 10.9 | **−0.788** | **100%** | 1.8e-12 |
| S-LH  feat↓read↑ | **+0.1098** | 10.8 | +0.886 | 0% | 1.8e-12 |
| S-LL  feat↓read↓ | +0.0114 | 1.1 | +0.062 | 38% | **0.15 (null)** |

**H1d is refuted.** It predicted that richer training decorrelates task centers. Three corners
converge; only `S-HL` decorrelates. **H6 is refuted.** It predicted the largest |Δρ_c| at
`S-HL`; the largest is at `S-LH`, twice the size (0.110 vs 0.055).

**The `S-HL` decline is progressive, not an endpoint difference.** Per-arm Spearman of ρ_c
against block is negative on **40 of 40 arms** with median −0.788. This is the specific thing
that licenses comparison with practice-related decorrelation in humans, which is a
within-condition claim over time; an endpoint contrast would not have.

### What replaces them: the 2×2 is approximately additive

Read as two main effects on `Δρ_c` (half-differences, so each is the effect of flipping one
factor), at γ=10: **readout similarity +0.1043** (drives convergence), **feature similarity
−0.0606** (drives decorrelation), **interaction +0.0059** — at the resolution limit while the
main effects are 6–10× it. The additive model predicts `S-HH` = +0.0434 against +0.0551
measured. Both main effects survive the 4× load change (readout +0.083/+0.104/+0.090, feature
−0.051/−0.061/−0.054 at N=150/300/600); the interaction is +0.020/+0.006/+0.020, noisiest
where arms are fewest (4 per corner in the width arm against 40 at N=300).

Both effects grow smoothly with γ while holding a roughly constant ratio (feature/readout =
−0.62/−0.66/−0.58 at γ=1/3/10). **Richness sets the gain; the 2×2 sets the sign.** That is a
stronger claim than the corner ordering — an ordering ranks cells, additivity predicts the
fourth cell from the other three and says the two similarity axes act separately.

### The limit on the S-HL result, stated plainly

**At γ=3, `S-HL` decorrelation does not exist**: +0.0034, 0.3× baseline, 50% of arms declining,
p=1.0. The effect appears only at **γ=10, the richest point in the sweep**, so it rests on one
γ value at the edge of the swept range. What makes it credible rather than an edge artifact is
that the *underlying* feature-similarity effect grows smoothly across the whole sweep
(+0.002/−0.003/−0.015/−0.036/−0.061 at γ=0.03/0.3/1/3/10): the `S-HL` sign flip is that smooth
effect overtaking the smooth readout offset in the one cell where the offset is absent, not a
new mechanism switching on. This is the same structure as the "retained Ψ_eff > 1 threshold" —
a smoothly growing quantity crossing a constant — and it should be described the same way,
as a crossing rather than a threshold. A γ point beyond 10 would test it directly.

**Consequence for §5.4.** The Menghi correspondence is now specific enough to state and to
attack: the corner with high feature and low readout similarity — same stimuli, different rules,
the closest analogue to their similar-structure condition — is the one whose centers decorrelate,
progressively over the stream, on every arm, and more strongly at larger width. The
disanalogies stay in the text: their decorrelation is over practice within a condition and ours
is a between-corner contrast at matched training, and ours needs γ=10.

---

## W5: the lag statement, decontaminated — and the confound was bigger than the effect

Pure re-pooling, no compute. Pooling by lag mixes lag with task index, because only task 0
can have lag 15 while lag 4 draws on tasks 0, 4, 8 and 12. Both were measured separately.

**W5a — within task 0 only, lag varying.** Magnitude grows with lag and **saturates**:

| γ | lag 4 | lag 8 | lag 12 | lag 15 |
|---|---|---|---|---|
| 1 | 12.5 floors | 14.9 | 16.7 | 17.0 |
| 3 | 23.8 | 28.4 | 31.3 | 32.0 |
| 10 | 39.0 | 46.9 | 50.7 | 52.1 |

Channel composition is near-invariant to lag: at γ=10 utility runs 0.480 → 0.445, radius
0.101 → 0.122, dimension 0.419 → 0.433 from lag 4 to 15. **The earlier claim — γ sets the
mechanism, lag sets the magnitude — survives with the confound removed**, and now also says
the magnitude is asymptotic rather than linear in lag.

**W5b — the confound, measured.** At lag 4 held fixed, varying which task is being forgotten:
γ=10 gives **39.0 / 27.8 / 23.1 floors for tasks 0 / 4 / 8**, and γ=1 gives 12.5 / 8.6 / 7.0.
So the task-index effect (×1.7) is **larger than the whole lag effect within task 0** (×1.34).
Anything pooled by lag alone therefore *overstates* lag dependence, since high lags are
necessarily early tasks and both push the same way. Composition also shifts with task index at
γ=1 (utility 0.296 → 0.215, radius 0.301 → 0.382) but not at γ=10 (0.480 → 0.464).

**Most of the task-index effect is distance above a common level, not the primacy of task 0.**
Later tasks begin with less retained capacity — at γ=10, α at a task's own boundary is 1.874 /
1.357 / 1.176 / 1.078 for tasks 0 / 4 / 8 / 12 — and after the same lag they land far closer
together (1.100 / 0.967 / 0.874) than they started. Regressing across (task, lag) cell means
gives an implied common asymptote α_∞ ≈ 0.81 at γ=10 and 0.43 at γ=1.

**The number that would have been misleading, and the honest version.** That cell-mean
regression has r² = 0.889, and reporting "89% of forgetting is explained by starting distance"
would have been wrong: cell means average away arm-to-arm variation, so their r² is not
variance explained in the data. **At the arm level the same two predictors give R² = 0.099.**
Both numbers are real and answer different questions — the systematic trend is clean, and it is
a small part of the spread.

**What does explain the spread (γ=10, 1600 observations):**

| predictors | R² |
|---|---|
| distance + lag | 0.099 |
| **condition alone** | **0.814** |
| stream instantiation alone | 0.000 |
| initialization seed alone | 0.005 |
| condition + distance + lag | 0.943 |
| + stream + seed | 0.948 |

**Which corner of the 2×2 you are in explains 81% of how much gets forgotten**; adding distance
and lag reaches 94%. Stream instantiation and seed contribute **nothing measurable** (0.000 and
0.005) — a strong reproducibility statement, and retrospective justification for the paired-init
and shared-stream design. Residual sd is 0.141 log units against a total of 0.620, still 7.6
noise floors, so real structure remains unmodelled; the obvious candidate is the interaction
between corner and task index visible in the γ=1 composition shift above.

W5 is closed. Remaining cheap items: the width Figure-2 equivalent, then W4.

---

## Figure 2 is a task-0 figure, and its caption has to say so

**Lag 12 exists only for task 0.** With measurement boundaries at 0, 4, 8, 12, 15, the only
comparison with twelve intervening tasks is the one from task 0: **960 of 960 lag-12
attribution records are task 0**. Which tasks contribute to each lag, over all 9,600 records:

| lag | 3 | 4 | 7 | 8 | 11 | 12 | 15 |
|---|---|---|---|---|---|---|---|
| tasks | 12 | 0, 4, 8 | 8 | 0, 4 | 4 | **0** | **0** |

Two consequences. Figure 2, drawn at lag 12, is **immune to the W5 lag/task-position confound**
by construction — there is nothing to stratify. And because task 0 is the most-forgotten task
in the stream (W5: 39.0 floors against 23.1 for task 8 at matched lag), the figure reports the
**maximum-forgetting task, not the grid average**, which is a scoping statement the caption must
carry. Both facts are in the script's docstring, and the figure now prints its own task set into
the subtitle (`lag 12, task 0`) so a later `--lag 4` run cannot inherit a task-0 caption.

Figure 2 re-run on the completed grid (960 `a=0` arms, was 776 at the draft): 0.1 / 0.4 / 5.2 /
16.7 / 31.3 / **50.7 floors** at γ = 0.03 → 10, Ψ_eff becoming resolvable and negative between
γ = 0.1 and 0.3. No change in shape from the draft.

## Width robustness in Figure 2's form

`scripts/fig_width_invariance.py`, four panels, matched lag 12 and matched sampling (streams
0–1, seeds 0–1 at every width, so the N=300 line is not an average over 10× more arms than its
neighbours). At matched lag the invariance is **tighter than the lag-pooled version above**:

| N | γ=1 floors | γ=10 floors | utility | radius | dimension | radius+dimension |
|---|---|---|---|---|---|---|
| 150 | 16.3 | **47.9** | 0.472 | 0.139 | 0.389 | 0.528 |
| 300 | 16.6 | **48.2** | 0.460 | 0.118 | 0.422 | 0.540 |
| 600 | 15.1 | **47.5** | 0.438 | 0.112 | 0.450 | 0.562 |

**1.5% spread in magnitude at γ=10** across a 4× load change. The one genuine N-dependence has
its own panel rather than a footnote: the radius share falls (0.139 → 0.112) while dimension
rises (0.389 → 0.450) and **their sum stays put** (0.528 → 0.562). The utility/non-utility
division is width-invariant; what moves is how the non-utility part is spent.

The matched-subset diagnostic is in the figure's JSON and behaves as it should. At γ=1 and 10
the 16-arm subset and the 160-arm full grid agree (Δlog α −0.3074 vs −0.3088 and −0.8922 vs
−0.9402; utility share 0.292 vs 0.284 and 0.460 vs 0.448). At γ=0.03 they disagree wildly
(utility share 0.948 vs 0.430) — the unresolvable cell at 0.1 floors, which is exactly what the
`MIN_FLOORS` gate refuses. A 16-arm subset cannot estimate the composition of a change that did
not happen, and neither can 160.

Two hardcoded numbers in that figure's panel titles ("2% spread", "0.440–0.460") were wrong the
moment the figure moved to matched lag; both now compute from the plotted data.

## Write-up prose has its own file

`docs/07-writeup.md`: the aggregation-artifacts appendix subsection (three artifacts, each with
the wrong number and the right one), the methods note on the variance decomposition and what it
says about the paired-init and shared-stream design, and the Figure 4 caption with §5.4's
argument order and its γ-range limit stated in the same breath.

---

## Structural: off-design arms can no longer be adopted by the registered grid

`grid.load()` now **refuses** any arm in `results/phase1/` whose γ is outside the registered
sweep or whose `N` is not 300, naming the file and telling you to load it via `arms_dir`.

The hazard is not a collision, it is the absence of one. `Phase1Spec.key` carries γ but not
`N`, so an exploratory arm written into the grid directory either overwrites a grid arm — which
at least perturbs a number someone might notice — or, if its γ or `N` differs, quietly *joins*
the grid. Every figure, `phase1_summary.json` and the scope audit then include it, with no
exception, no warning, and no diff to inspect. **Same failure class as the stale fork: nothing
errors and the artifact is wrong.** The refusal deliberately precedes the `usable` filter, so
an unusable off-design arm cannot slip through and set the precedent.

Six tests in `tests/test_grid_offdesign.py` pin it, including the `N=600` case, which is the
nastier one because the filename does not even collide. Registered grid still loads 960 `a=0`
arms over γ ∈ {0.03, 0.1, 0.3, 1, 3, 10}; 130 tests pass.

---

## γ=30 probe: the decorrelation extends, and additivity turns out to be range-bounded

64 arms, four corners × 4 streams × 4 seeds, **741 s wall** — against a projection of 45–55 min.
The projection was 3.8× too pessimistic: it applied the width run's 96-worker contention factor
to a 64-worker wave on 256 cores, and γ=30 arms need very few SGD steps to matched loss. Worth
recording as the *third* cost-model miss in this project, this time in the safe direction.

**Declared artifact checks, all clear.** 64/64 usable; max |identity residual| 4.08e-16; **0 of
1024 tasks failed to converge**; `rho_c_signed` spans [0.2493, 0.5073], inside the [0.043, 0.803]
calibration window, so even center-collapse share stays valid at this richness.

**The regime is sane, not an extrapolation into breakdown.** At lag 12 the Figure 2 story
continues smoothly: 31.3 → 50.7 → **71.1 floors** at γ = 3 → 10 → 30, utility share 0.407 →
0.448 → 0.482, radius 0.164 → 0.121 → 0.102. And the generic-capacity half of the corner
mechanism extends without qualification — `α(γ)/α(0.03)` goes S-HL 1.48 → **1.76**, S-HH 1.22 →
1.35, S-LL 1.28 → 1.42, with **S-LH still the only corner below 1** (0.95 → 0.94).

### Outcome 1 (pre-declared): the S-HL decorrelation extends and strengthens

| γ | Δρ_c at S-HL | ±sem | median per-arm ρ_S | % declining |
|---|---|---|---|---|
| 3 | +0.0034 | 0.0019 | −0.049 | 50% (p=1.0) |
| 10 | −0.0550 | 0.0033 | −0.788 | **100%** |
| 30 | **−0.0956** | 0.0041 | −0.837 | **100%** |

More negative than −0.055, which by the table fixed in `docs/07-writeup.md` before these numbers
were read licenses exactly this: **γ=10 is not an edge artifact**, the decorrelation strengthens
across the two richest points, and it stays one appendix sentence.

### Outcome 2 (pre-declared): additivity is a property of the registered range

| γ | readout | feature | interaction | f/r ratio | additive predicts S-HH | measured |
|---|---|---|---|---|---|---|
| 1 | +0.0243 | −0.0150 | −0.0027 | −0.62 | +0.0426 | +0.0372 |
| 3 | +0.0543 | −0.0359 | −0.0015 | −0.66 | +0.0592 | +0.0562 |
| 10 | +0.1043 | −0.0606 | +0.0059 | −0.58 | +0.0434 | +0.0551 |
| 30 | +0.1258 | −0.0647 | **+0.0160** | −0.51 | +0.0142 | +0.0461 |

The γ=30 interaction is **resolved, not a 16-arm artifact**: 3.9 SEM from zero with a bootstrap
95% CI of [+0.0082, +0.0238], entirely above the drift band's lower edge. The γ=10 value for
comparison is 1.9 SEM with CI [+0.0001, +0.0118], overlapping the band — which is what "at the
resolution limit" meant. The additive prediction for `S-HH` degrades from off-by-0.012 to
**off-by-0.032**.

**Why**: the two main effects *saturate* while the interaction grows. From γ=3 to 10 both roughly
doubled; from 10 to 30 readout gains only 21% and feature only 7%, for a 3× richness increase.
Per the pre-declared table, §5.4's additivity claim therefore gains **"over the registered
range"** — and that is a real restriction, not a hedge.

**A third thing, unpredicted: `S-LL` changes sign.** +0.0379 → +0.0114 → **−0.0149** at γ = 3 →
10 → 30 (75% declining, p = 0.077, marginal). At extreme richness even the low-feature,
low-readout corner trends toward decorrelation. This is the same phenomenon as the growing
interaction seen from a different angle: something beyond the two similarity axes decorrelates
centers at high richness. It is one marginal cell in an appendix probe and is logged, not
claimed.

**The value of having fixed the status first.** The favourable outcome (the decorrelation
extends) arrived together with an unfavourable one (additivity is range-bounded). Had the reading
not been fixed in advance, the temptation to report the first at length and the second in a
subordinate clause would have been real. Both get the same prominence, as committed.

---

## W4: the α floor is γ-flat, and the R_eff floor is wrong in two ways

`scripts/run_measurement_null.py`, 1,188 s. One arm per registered γ, trained once, then the
**same trained network on the same manifolds** re-measured under 4 measurement seeds, so the only
thing varying is the estimator's anchor draw. 6/6 converged.

**How precisely a 4-seed CV can be read.** A coefficient of variation from n=4 carries its own
relative uncertainty of about `1/√(2(n−1))` ≈ 41%. So ratios below roughly 2× are *not* resolved
by this design, and only larger ones should be spoken about. This bounds every reading below.

Registered global floor against measured per-γ floors, retained ensemble — the one attribution
uses, since forgetting is measured on a retained task:

| channel | registered | measured range | γ-variation | vs registered |
|---|---|---|---|---|
| `alpha` | 0.0187 | [0.0149, 0.0203] | 1.4× (**not resolved**) | 1.1× |
| `D_eff` | 0.0126 | [0.0078, 0.0142] | 1.8× (**not resolved**) | 1.1× |
| `R_eff` | 0.0050 | [0.0016, 0.0116] | **6.7×** | **2.3×** |
| `Psi_eff` | *none* | [0.0060, 0.0160] | 2.0× | — |

### The headline claims are safe

**`alpha`'s floor is flat across γ within the resolution of this test, and matches the registered
value to 1.1×.** Every floor-denominated headline number is α-denominated — Figure 2's magnitude
curve, the ±2-floor resolution threshold, the width invariance, the W5 lag and task-position
series. Those all stand as written, and the worst case is a ±10% rescaling of a quantity where
nothing hinges on 50.7 versus 46 floors. `D_eff` is the same story. **No restatement is needed
and I have made no changes.**

### The `R_eff` floor is a real problem, and it is narrow

Two separate defects. It is **2.3× too small** at γ=10 (measured 0.0116 against a registered
0.0050), and it is **γ-dependent by 6.7×** within the retained ensemble (0.0016 at γ=0.03 rising
monotonically to 0.0116 at γ=10) — the one channel where the ruler genuinely stretches with the
thing being measured. In the generic ensemble it is 3.1× too small across the board
([0.0107, 0.0153]).

What this touches, precisely:

- **`attribution.MIN_DLOG_R` = 0.004988**, which is the registered 0.005 in log units. It floors
  the denominator of `center_collapse_share` specifically to stop sub-noise radius changes
  producing unbounded ratios. If the true floor at high γ is 2.3× larger, that guard is **too
  permissive exactly where it is most used** — Figure 2 panel (d) reports center-collapse shares
  at 91% coverage for γ = 3 and 10. This is the same failure mode the guard was introduced to
  fix, one step further out.
- Any statement quoting an **`R_eff` excursion in noise floors**.

What it does **not** touch: channel *shares*. Those are `|term|/Σ|term|` gated on total
`|Δ log α|`, which is α-denominated and safe — so Figure 2 panel (b), the width figure's
radius/dimension split (0.139 → 0.112 with a stable sum), and the composition invariance all
stand independent of the `R_eff` floor.

**No remediation attempted, per instruction.** The options, for the decision: refit `MIN_DLOG_R`
and `NOISE_FLOOR_CV['R_eff']` to the measured per-γ values and re-derive panel (d) — cheap,
because attribution re-derives from stored geometry, so this is a re-summarize and not a re-run —
or drop panel (d) to the appendix and state the radius channel's resolvability only in aggregate.
Also worth noting: `Psi_eff` has **no registered floor at all**, which is why `grid.floors()`
raises rather than borrowing another channel's; W4 now supplies one if it is ever wanted.

---

## Panel (d) refit: the corrected floor is per-γ, and panel (d) survives it

Pre-branched test, outcome rules fixed before the numbers were read: coverage ≥ 70% at γ = 3 and
10 with the growth pattern intact → panel (d) stays in Figure 2 with per-γ floors and a caption
note; coverage collapse or lost pattern → panel (d) moves to the appendix. **Coverage passed and
the shares barely moved, so panel (d) stays.**

Refitted `MIN_DLOG_R` from one constant to the measured per-γ table (pooling both modules, so
8 measurement seeds per γ). Old constant: CV 0.50% at every γ.

| γ₀ | floor CV, old → new | coverage, old → new | median share, old → new |
|---|---|---|---|
| 0.03 | 0.50% → 0.28% | 19% → **53%** | −0.005 → 0.015 |
| 0.1 | 0.50% → 0.31% | 84% → 84% | 0.040 → 0.040 |
| 0.3 | 0.50% → 0.62% | 100% → 100% | 0.124 → 0.124 |
| 1 | 0.50% → 0.81% | 94% → 88% | 0.231 → 0.240 |
| 3 | 0.50% → 0.71% | 91% → **88%** | 0.422 → 0.431 |
| 10 | 0.50% → 1.12% | 91% → **88%** | 0.425 → 0.421 |

Coverage at γ = 3 and 10 is **88%**, a 3-point loss, well clear of the 70% branch. No median
moves by more than 0.010.

**The strict-monotonicity question, answered rather than eyeballed.** The refit flips the sign of
the top step (γ = 3 → 10: +0.004 before, −0.010 after), so `strictly increasing` is now False. But
that step was **never resolved in either fit** — bootstrap on the median difference, 20k
resamples: old +0.004 with CI [−0.031, +0.074], refit −0.010 with CI [−0.038, +0.070]. Both
straddle zero. Every step *below* it is resolved and positive, in the refit: +0.025
[+0.018, +0.039], +0.084 [+0.070, +0.097], +0.116 [+0.089, +0.140], +0.191 [+0.155, +0.219]. So
the correct description was always "rises steeply through γ = 3, then plateaus", and the pre-refit
claim of monotone growth *through the top* was reading an unresolved 0.004 as a trend. The refit
did not break the pattern; it corrected how the pattern was stated.

**Two honest consequences of a γ-dependent gate.** The bars are now gated at different thresholds,
so their coverage percentages are not comparable across bars — printed per bar in the panel and
stated in the caption. And at γ = 0.03 the corrected floor is *smaller* than the constant, so
coverage rises 19% → 53%: the refit admits more cells where there is nothing to attribute (0.1
floors of α change). Those two bars are hatched in the panel as "no resolvable forgetting".

**Verified that the refit touches nothing else.** Re-derived every pooled `dlog_alpha`, term and
share at lag 12 against the committed figure JSON: worst absolute difference **0.00e+00**. Panels
(a)–(c) are bit-identical, as they must be — the floor gates one reported ratio and the identity
is exact. Two tests pin this: one that the same radius change is reported at γ = 0.03 and refused
at γ = 10, off-sweep γ, and unknown γ; one that the three terms and shares are invariant to the
floor.

Three smaller decisions taken with it:

- **Off-sweep γ is not interpolated.** `min_dlog_R_for` returns the largest measured floor for
  unknown γ. Guessing a floor would put a fabricated number inside a guard whose job is refusing
  fabricated numbers, and a ruler that is too short flatters the result.
- **`grid.floors()` is now γ-aware** for channels with a per-γ table, defaulting to the most
  conservative. This changes no live number — every live `floors()` call is `alpha`, which W4
  found γ-flat — but it closes the path by which a future caller silently picks up the old,
  too-permissive `R_eff` value.
- **`NOISE_FLOOR_CV` is left alone.** Its remaining consumer is the warp test, which compares
  trajectories at two different γ and therefore cannot be denominated in either one's floor. The
  per-γ table lives beside it as `PER_GAMMA_FLOOR_CV`, and `attribution` imports it rather than
  restating it.

**The Ψ_eff floor is recorded and not used.** `PER_GAMMA_FLOOR_CV['Psi_eff']` now holds
[0.0115, 0.0240], with a comment saying why it is inert: the utility channel has run un-floored
through every analysis, and retro-fitting a gate would move numbers for no gain at this stage.

Logged as the third instance of the guard-too-permissive family in `07-writeup.md` §A.2, after the
ρ_c clip and the arccos/dual clips. The generalisable point is that each guard was calibrated at
the one setting that motivated it, and testing it at another setting is cheap.

**Where this leaves panel (d):** in Figure 2, but as its weakest element — post-hoc calibration,
a corrected floor, per-bar gating — while (a)–(c) are clean and carry the attribution claim.
§5.1's paragraph 3 stays short regardless, per the standing instruction.

**Side effect: the audit's stale-code check fired, and it was right to.** Editing `attribution.py`
and `pipeline.py` means the 1,280 arms on disk were produced by code that no longer matches the
tree, and `audit_scope` failed on hash equality. Hash equality was the wrong invariant, but the
fix is not to weaken it. The check now (i) reports drift *per module* against a hand-written
declaration naming what changed and why measurement is untouched, with `core` deliberately absent
from that list because geometry cannot be re-derived without re-running; and (ii) adds
`audit_rederivation`, which recomputes the identity from stored geometry with current code and
requires the stored terms back exactly. That is a stronger check than the one it replaces:
**worst |Δ| 0.00e+00 over 25,600 re-derived attributions.** A drift that had touched measurement
would fail there rather than being waved through by a declaration. Audit now 32 pass, 1 warn
(the pre-existing sibling-import warn), 0 fail.

---

## Two cheap checks, and a conflation in Figure 2 that they exposed

### Check 1: the lag-4 companion — composition holds, and the reviewer question is answered

Figure 2 is task 0 at lag 12, by construction the maximum-forgetting cell. Re-ran at lag 4 both
ways — restricted to task 0 (a pure lag change) and pooled over tasks (the grid-average cell),
using a new `--task` flag so the two are separable. Added because a lag comparison pooled over
tasks varies two things at once, task position being the larger of the two (W5).

Shares as plotted (ratio-of-means) agree across all three cells to **≤ 0.033 at every γ**, while
the magnitude changes by 1.3–1.8×:

| γ₀ | lag 12, task 0 | lag 4, task 0 | lag 4, pooled | largest share gap |
|---|---|---|---|---|
| 0.3 | 0.101 / 0.478 / 0.421 | 0.131 / 0.477 / 0.392 | 0.120 / 0.474 / 0.407 | 0.030 |
| 1 | 0.284 / 0.413 / 0.303 | 0.296 / 0.403 / 0.301 | 0.263 / 0.403 / 0.334 | 0.033 |
| 3 | 0.407 / 0.430 / 0.164 | 0.425 / 0.418 / 0.157 | 0.398 / 0.413 / 0.189 | 0.032 |
| 10 | 0.448 / 0.431 / 0.121 | 0.480 / 0.419 / 0.101 | 0.471 / 0.424 / 0.104 | 0.032 |

(Ψ_eff / −D_eff / radius.) Magnitude ratio lag 12 : lag 4 within task 0 is **1.30–1.33×** at every
γ, matching W5's 1.34× for lag 4 → 15; against the task-pooled lag-4 cell the same ratio inflates
to 1.65–1.78×, which is the task confound reappearing exactly as W5 said it would.

A **paired** within-arm test (same arm, lag 4 vs lag 12, 20k bootstrap) does resolve small drifts,
largest at γ=10: utility −0.055 [−0.065, −0.046], dimension +0.031, radius +0.024. So composition
is *near*-invariant, not invariant. The number that matters is the comparison: the γ-dependent
swing being claimed is 0.10 → 0.45 in utility share, a change of **0.35**, against a
cell-choice sensitivity of **≤ 0.055** — six times smaller. (Paired means-of-ratios and the
plotted ratio-of-means differ slightly by construction; both are quoted above as what they are.)

### Check 2: the capacity pair at γ=30 — no surprise waiting out there

Both capacities keep rising past the registered range and retained stays above generic, with the
ratio still growing. Nothing to revise.

| γ₀ | generic α | retained α | ratio |
|---|---|---|---|
| 3 | 0.3404 ± 0.0009 | 0.7261 ± 0.0047 | 2.133 |
| 10 | 0.3750 ± 0.0018 | 1.0519 ± 0.0092 | 2.805 |
| **30** | **0.4159 ± 0.0046** | **1.4175 ± 0.0239** | **3.408** |

Both γ=10 → 30 steps resolved (8.3 and 14.3 SEM). Monotone in γ throughout, retained > generic at
every γ including 30, 64/64 usable, 0 unconverged. Worth noting against the other γ=30 result:
the ρ_c **main effects saturate** above γ=10 while **capacity does not**, so "saturation at γ=30"
is a statement about the center-geometry effects specifically, not about the sweep in general.

### What check 1 turned up on the way: Figure 2 pools a gain with three losses

Following the per-corner breakdown of panel (d) — `S-HH` had a *negative* median share and half
the coverage of every other corner — the cause is not the panel. **`S-HH` does not forget. Its
retained capacity rises.** Mean Δ log α at lag 12, per corner:

| γ₀ | S-HH | S-HL | S-LH | S-LL |
|---|---|---|---|---|
| 0.3 | **+0.173** | −0.283 | −0.071 | −0.204 |
| 1 | **+0.219** | −0.692 | −0.259 | −0.504 |
| 3 | **+0.201** | −1.117 | −0.552 | −0.852 |
| 10 | **+0.114** | −1.540 | −1.035 | −1.301 |

Behavioral forgetting from the accuracy matrix agrees exactly: `S-HH` is **−0.0005 to +0.0001** at
every γ — zero — against +0.40 for `S-HL`, +0.15 for `S-LL`, +0.01 for `S-LH`. And this is the
**registered prediction for that corner coming true**: `01` §3.1 lists `S-HH` (s_f = s_r = 0.9) as
"benign". Its tasks are distinct, so this is backward transfer in a benign corner, not a
degenerate cell.

The consequence is that Figure 2's pooled panels average a capacity *gain* against three capacity
*losses*, and panel (a)'s axis label — "how much retained capacity is lost" — is not true of one
of the four corners it pools. Three concrete symptoms, all of which had been noticed and
misattributed:

- **Panel (d)'s `S-HH` bar was noise over noise.** Its median |Δ log R_eff| is **0.86–0.98× the
  floor** while the other three corners sit at **10–21×**; 50% of its cells are refused and 50–75%
  of those admitted have the *opposite sign*. At a 3× gate `S-HH` empties completely (0/40) and the
  other three corners are untouched (40/40, identical medians). All of panel (d)'s missing coverage
  was this one corner.
- **The γ=0.3 utility-share collapse to 0.101** was read as the utility term crossing zero within
  the pool. Part of it is `S-HH` contributing an opposite-signed utility term: over the three
  losing corners the same share is **0.189**.
- **Panel (b) at γ=1 is a genuine mixture**: pooled utility share 0.284 is *below all four*
  corner values (0.296, 0.330, 0.355, 0.529), because corner-level utility terms partially cancel
  in the pooled mean. Spread across corners is 0.233 in utility and 0.220 in radius at γ=1,
  narrowing to 0.072 and 0.063 by γ=10.

**The headline claim is stronger, not weaker, once the benign corner is separated.** Utility share
rises monotonically with γ in each of the three forgetting corners taken individually
(`S-HL` 0.203 → 0.469, `S-LH` 0.068 → 0.397, `S-LL` 0.212 → 0.466) and falls in `S-HH`. Pooled
over the three that forget:

| γ₀ | Δ log α | floors | Ψ_eff | −D_eff | radius |
|---|---|---|---|---|---|
| 0.3 | −0.186 | 10.0 | 0.189 | 0.443 | 0.368 |
| 1 | −0.485 | 26.2 | 0.332 | 0.421 | 0.246 |
| 3 | −0.840 | 45.3 | 0.414 | 0.436 | 0.150 |
| 10 | −1.292 | 69.7 | 0.449 | 0.437 | 0.114 |

Monotone in both directions — utility up, radius down — with no term crossing zero anywhere in the
range, and every γ resolvable including γ=0.3 at 10.0 floors where the 4-corner pool gives 5.2.
The 4-corner version needs "below floor" annotations at two γ; this one does not.

**Not acted on.** Changing Figure 2's primary object is a figure decision, not a bug fix, and it is
the load-bearing panel four days out. Reported for a decision.

### Decision taken: three-corner Figure 2, gate at 3σ

Both decisions implemented and figures regenerated. `FLOOR_GATE_K = 3.0` in `attribution`, and
`fig2_gamma_sweep.py` now defaults to `--condition forgetting` (the three losing corners named
explicitly, so which conditions are pooled is a stated choice rather than an outcome of the numbers
being reported), with `--condition S-HH` drawing the benign corner alone.

**The three-corner figure at lag 12**, all six γ present:

| γ₀ | floors | Δ log α | Ψ_eff | −D_eff | radius | utility>0 | panel (d) median | coverage |
|---|---|---|---|---|---|---|---|---|
| 0.03 | 0.2 | −0.0033 | — | — | — | 0.42 | n/a | 0% |
| 0.1 | **2.1** | −0.0389 | 0.082 | 0.531 | 0.388 | 0.25 | 0.053 | 58% |
| 0.3 | 10.0 | −0.1859 | 0.189 | 0.443 | 0.368 | 0.08 | 0.122 | 92% |
| 1 | 26.2 | −0.4849 | 0.332 | 0.421 | 0.246 | **0.00** | 0.256 | **100%** |
| 3 | 45.3 | −0.8400 | 0.414 | 0.436 | 0.150 | **0.00** | 0.442 | **100%** |
| 10 | 69.7 | −1.2917 | 0.449 | 0.437 | 0.114 | **0.00** | 0.450 | **100%** |

Every improvement over the 4-corner version comes from removing a sign mixture, not from removing
data: γ=0.1 becomes resolvable at 2.1 floors (was 0.4), the utility term is negative on **100%** of
arms for γ ≥ 1 (was 75%), and panel (d) reaches **100% coverage at γ ≥ 1** — better than the 88%
that the refit alone gave, and better than the 91% we started with. Panel (d)'s γ=3 → 10 step is
still unresolved (0.442 vs 0.450), so the "rises through γ=3, then plateaus" reading is unchanged.

**The benign corner on its own** is worth its own panel rather than a footnote: capacity gain peaks
at **γ=1 (11.8 floors)** and *falls* to 6.2 by γ=10, so backward transfer is strongest at moderate
richness. Its utility and dimension terms are both positive while its **radius term is negative**
at γ ≥ 1, and panel (d) empties at γ ≥ 3 exactly as it should — there is no radius movement to
attribute. At γ=1 its one admitted bar is **negative (−0.50)**, i.e. ρ_c and R_eff moved in
opposite directions; the panel now shows negative bars rather than clipping them to the axis floor,
since clipping would render a sign disagreement as a small positive share.

Three plotting bugs fixed while doing this, each of which had been hiding the conflation:

- **Panel (a) plotted a magnitude.** `G.floors()` returns `abs(Δ log)/floor`, so a corner that
  *gained* capacity plotted as though it had lost that much, under an axis reading "how much
  retained capacity is lost". The panel now plots a signed value and the title follows the sign.
- Panel (d) printed the floor CV in a label reading "gate", when the gate is 3× that.
- The hatched-bar legend line was drawn whether or not any bar was hatched.

Appendix updated: the pooled-sign artifact is now the first and largest of the four aggregation
artifacts in `07-writeup.md` §A.1, and the 1σ → 3σ correction joins the guard family in §A.2 as the
same guard failing in level as well as in γ-dependence. Figure 2's caption drafted in §D.

---

## Write-up scaffolding: drafts, a reproducible figure set, and an inventory that checks itself

Generation stopped. Three artifacts, in the order requested.

**`docs/07-writeup.md` restructured into §5 drafts.** Part 1 is §5.1 (the decomposition), §5.2
(both capacities rise; the two label-agnostic measures disagree between richness levels), §5.3
(what it survives: width, lag, task position, and the measured floors it is denominated in), §5.4
(center correlation across the 2×2), and a §5.5 placeholder for the pre-registration table. Part 2
holds the figure captions and the appendix material. §5.1 is written on the three-corner numbers
and §5.4's limits are stated with the claims they bound, both as decided above.

**`scripts/make_figures.py` regenerates the final set and writes `figures/MANIFEST.json`.** Six
figures, each with the exact command that produced it, the result-set hash it read, the git SHA and
module hashes of the code that ran, and a SHA-256 of every output file. `--check` verifies the
manifest against disk without regenerating and is the form to run before submitting. Anything not
in the declared final set is moved to `figures/superseded/` rather than left in place: **12 files
moved**, all of them four-corner-pooled variants, which is exactly the material that would have
been indistinguishable from current work by filename once the arm count matched.

**`docs/08-inventory.md` is generated by a script that re-reads every number from its source.**
49 entries, each with a getter that reads the value back out of the artifact it came from, compared
**at the precision the paper quotes** — a paper saying "0.2 floors" is making a claim about the
first decimal, and 0.176 satisfies it. It fails rather than warns, so the inventory cannot become
the stale copy. A further 20 numbers whose source is a one-off analysis are listed separately with
their log entry, not machine-checked, and marked as such.

### It caught one on the first run

**`S-LH`'s Δρ_c is +0.110, not +0.092.** The value in the §5.4 draft came from the W1 width table's
N=300 row, which is a **4-arm cell**; the 40-arm value from Figure 4 is +0.1098. This is the same
failure as the `S-LL` +0.0210 → +0.0114 correction, from the *same table row*, which had been
corrected only for `S-LL`. Every entry in that row is a 4-arm estimate: the true 40-arm values are
+0.0551 / −0.0550 / +0.1098 / +0.0114. The row now carries a comment saying so, and §5.4 says
+0.110. The claim it supports is unchanged — `S-LH` is still twice `S-HL` and in the opposite
direction, since 0.1098 / 0.0550 = 2.0.

Two smaller corrections from the same pass. §5.3's width numbers were quoted from the W1 log entry
(all lags) while the width *figure* is lag 12: at γ₀=10 the figure gives 47.5–48.2 floors and
utility 0.438–0.472, not 32.0–32.8 and 0.440–0.460. The draft now quotes the artifact in the final
set and states that the width comparison pools all four conditions on both sides — like-for-like
across widths, so it certifies invariance rather than §5.1's absolute share levels. And the
four-corner starting spread is 0.0014, not 0.0017.

**Remaining before submission**, unchanged: §5.5's pre-registration table, and the two
verifications only Kati can sign — the Graldi §3 base-width derivation, which underlies every γ
label in every figure, and `glue-core-validation.md`, which §4 is written from.

---

## Tier 1 propagation audit: the headline held, §B's meaning did not

`scripts/audit_propagation.py`, one script, `results/audit_propagation.json`. Every quantity that
averaged over the 2×2 recomputed at both groupings from one observation table, so the two cannot
drift apart through two implementations of the same pooling. The reconstruction reproduces the
drafted four-corner values exactly where a script existed to produce them — Figure 2's
0.1/−0.4/−5.2/−16.7/−31.3/−50.7, the width row's 47.9/48.2/47.5 and its utility shares
0.438–0.472, W5a's 39.0/46.9/50.7/52.1 and 1.34×, W5b's 39.0/27.8/23.1 and 1.69×, γ=30's 71.1 —
which is what licenses reading the three-corner column as a change rather than as a difference of
method.

**Marking each row `estimand` or `error` is not cosmetic.** Figure 2 at γ=10 reads 50.7 four-corner
and 69.7 three-corner. Nothing was miscomputed; the pooled population changed, and only one of the
two answers the question the paper asks. The 4-arm row below is the other kind. Presented in one
column without the distinction, a 37% movement in a headline invites "which of these is right",
whose answer differs by row.

### What moved, and by how much

| quantity | 4-corner | 3-corner | kind |
|---|---|---|---|
| Fig 2 magnitude, γ=10 | 50.7 fl | **69.7 fl** | estimand |
| Fig 2 magnitude, γ=0.1 | 0.4 fl (unresolvable) | **2.1 fl (resolvable)** | estimand |
| utility share over the resolvable range | 0.101 → 0.448 (from γ=0.3) | **0.083 → 0.449 (from γ=0.1)** | estimand |
| width magnitude, γ=10, N=150/300/600 | 47.9 / 48.2 / 47.5 | **65.7 / 66.6 / 65.6** | estimand |
| width utility-share spread, γ=10 | 0.034 | **0.024** | estimand |
| W5a lag ratio, γ=10 (lag 4→15) | 1.34× | **1.29×** | estimand |
| W5b task ratio, γ=10 (task 8→0) | 1.69× | **1.56×** | estimand |
| cell-mean r², distance only | 0.885 | **0.878** | estimand |
| arm-level R², distance + lag | 0.092 | **0.589** | estimand |
| condition alone, R² | 0.816 | **0.523** | estimand |
| stream alone / seed alone | 0.000 / 0.002 | **0.000 / 0.007** | estimand |
| implied common asymptote α_∞, γ=10 | 0.815 | **0.526** | estimand |
| γ=30 magnitude, lag 12 | 71.1 fl | **94.5 fl** | estimand |
| panel (d) median share, γ≥1 | 0.249 / 0.442 / 0.450 | **0.256 / 0.442 / 0.450** | estimand |
| panel (d) coverage, γ≥1 | 78 / 75 / 75% | **100 / 100 / 100%** | estimand |
| Δρ_c at N=300, all four corners | 4-arm row | **40-arm row** | **error** |

Three rows of that table need more than a new number.

**§B's variance decomposition changes meaning, not magnitude.** "Which corner of the 2×2 you are in
explains 81% of how much gets forgotten" was measured with one corner having the opposite sign, so a
large part of that 0.814 was condition encoding a *sign*, not a size. Removing it: condition falls
to 0.523 while distance-above-asymptote plus lag rises from 0.092 to **0.589**. The ordering
inverts. The four-corner table said forgetting is mostly about which stream you are in and barely
about where in it you are; the three-corner table says the two are comparable, with distance and lag
slightly ahead. §B is a rewrite rather than a renumbering, and it is held for Kati's read.

**The reproducibility half survives, as predicted.** Stream instantiation 0.000 → 0.000 and seed
0.002 → 0.007: both within-cell quantities, untouched by which cells are pooled. The paired-init and
shared-stream justification stands exactly as drafted.

**§A.1's own worked example was itself a four-corner artifact.** The cell-mean inflation pair was
quoted as 0.889 → 0.099, a factor of nine. At three corners it is 0.878 → **0.589**, a factor of
1.5. Most of the apparent inflation was not cell-averaging at all: it was that a model of pull
toward a common asymptote cannot fit a corner that moves *away* from it, which crushed the arm-level
R² while leaving the ten cell means smooth. The artifact is real and the lesson holds, but the
worked example has to be restated at its true size — and it is a fourth instance of the same
pooling fault, this time inside the appendix that describes the fault.

**Two drafted sentences disagreed with the figure they describe.** The regroup made γ=0.1 resolvable
(2.1 floors, from 0.4), so panel (b) now draws five bars where §5.1 and the caption described four:
the utility share range starts at 0.083, not 0.19, and dimension is not "near 0.44 throughout" but
falls 0.53 → 0.44 across the first step and is flat after. Both are corrected. Neither changes a
claim — the exchange between utility and radius is the same, over one more step of γ.

### The 4-arm row, fixed as a row

All twelve Δρ_c cells in the width block are 4-arm estimates, including the N=300 row, which is a
matched 16-arm subset rather than the 160-arm grid. Patching the one cell we noticed (`S-LL`
+0.0210 → +0.0114) guaranteed meeting the next one, and we did: `S-LH` +0.0922 reached the §5.4
draft where the 40-arm value is +0.1098. The rows now carry their arm count in the row label and the
authoritative 40-arm row sits beside them. Their noise is ±0.02 against real effects of 0.011 to
0.110, which is why they miss in both directions (S-LH low by 0.018, S-LL high by 0.010) and why the
interaction term, a difference of differences, comes out 3× its 40-arm size at both edge widths —
+0.020 against +0.006. The additivity claim uses the 40-arm value and is unaffected.

### Sign audit: the gain is unique to the benign corner

520 cells across the registered grid, the width arms and the γ=30 probe, each cell a
(γ, condition, N, task, lag) mean. 172 have positive mean Δ log α; 98 clear ±2 floors; **all 98 are
`S-HH`, and none of the other three corners produces a resolvable gain anywhere** — not at any
richness, lag, task position or width. The 44 positive cells outside `S-HH` reach at most +0.74
floors, i.e. a third of the resolution gate, and sit almost entirely at γ ≤ 0.1 where nothing moves
in either direction. So backward transfer is a property of that corner rather than a tendency
appearing unresolvably elsewhere, and §5.1 may state it as such.

### One observation surfaced early from Tier 2.1

The gain is **not** the mirror image of a loss. In the three forgetting corners all three terms are
negative at every resolvable γ. In `S-HH` utility and dimension are positive while the radius term
is *negative* from γ=1 onward (−0.0154, −0.0015, −0.0124 at γ=1, 3, 10) — the radius channel opposes
the two that carry the gain, having agreed with them at γ ≤ 0.3. So a corner that gains retained
capacity does so by aligning with labels and shedding dimension *against* a small radius cost, which
is a different channel signature from a loss and not a sign flip of one. Quantification is Tier 2.1;
the qualitative fact bears on how §5.1 frames backward transfer, so it is recorded now.

### A gap this audit closed, and one it opened

Closed: the W5 series and the variance decomposition were computed in throwaway shell sessions and
were not reproducible from any script, which is why the reconstruction agrees with §B's drafted
numbers to within 0.007 rather than exactly. They now have one.

Opened: the width contrast's middle line is a 16-arm matched subset that differs from the full grid
by 4.5% at γ=10, three times the 1.5% spread across widths. "Varies by 1.5%" therefore overstates
the precision available — the honest form is that no width dependence is detectable and 1.5% bounds
one. §5.3 is corrected to say that.

### §B rewritten, and two sections strengthened by their own corrections

**§B now leads with the pair of zeros.** Stream instantiation 0.000 and initialization seed 0.007
are what the note exists for: they are the measurement that turns pairing initializations and
sharing streams from tidiness into a justified choice, and they are untouched by the regroup because
both are within-cell. The condition/position split is demoted to a descriptive sentence — 0.523 for
which corner, 0.589 for distance above the asymptote plus lag, 0.882 together — with **no dominance
claim in either direction**, since the two are close, they overlap through the corner-by-position
interaction, and the design was not built to separate them.

The old sentence said condition explains 81% against 9% for position, a ratio of nine. That was
largely an artifact of the sign: a categorical predictor separating one gain from three losses will
explain most of the variance of a *signed* quantity nearly regardless of what else holds. §B states
that, because a reader comparing this table to an earlier draft deserves the reason rather than a
silent change.

**§A.1's second example is now reported at its true size, as an illustration of §A.1.** The
cell-mean inflation pair was 0.889 → 0.099 and is 0.878 → 0.589, a factor of 1.5 rather than nine.
The diagnosis matters more than the number: the arm-level R² was not crushed by cell-averaging but
by fitting a pull-toward-a-common-asymptote model to a corner that moves *away* from it, for which
no value of "distance above the asymptote" predicts anything. So the appendix documenting the
pooling fault had committed it in sizing its own example. Written that way deliberately — the
section's claim is that pooling is a modelling assumption rather than a neutral step, and the
strongest evidence available is that we made the assumption while writing the section.

**§5.1 states backward transfer categorically and adds the channel signature.** The sign audit
licenses "a property of that corner, not a tendency present elsewhere in weaker form" — 520 cells,
every resolvable gain an `S-HH` cell, largest gain elsewhere 0.74 floors against a gate of 2. And
the gain is a distinct route rather than forgetting in reverse: utility and dimension positive
against a *negative* radius term, so capacity is gained by aligning with labels and shedding
dimension at a small radius cost. That is an argument for the decomposition itself, since a
magnitude-only measure would present gain and loss as one quantity with two signs. The peak's
location is stated only as the largest of six sampled richnesses; Tier 2.3 bounds it, and no
sentence pairs it with the behavioural optimum.

**Width claim tightened.** "Varies by 1.5%" became "no width dependence is detectable, with 1.5%
bounding one rather than measuring one", because the matched middle line differs from the full grid
by 4.5% — three times the spread being reported.

### `notebooks/results.ipynb` — the diagnostic artifact

Thirteen cells, runs top to bottom in about 8 seconds, executed clean in CI-style with
`nbconvert --execute`. Sections 1–3 and 5–10 built; **§4 is a stub** that reports backward transfer
as far as Tier 1 established it — magnitudes across richness, the sign audit, the channel
signature — and names the three things Tier 2 owes, including the pre-committed text rule on the
peak location written into the section header rather than kept in a brief.

The logic lives in `src/analysis/ledger.py` so that the notebook cells are thin and the discipline
is testable. Three properties are enforced by the code rather than asked for:

- **`Table` raises if `n` or the arm-selection rule is missing.** Both are rendered in the output,
  not held in a comment. The 4-arm row reached drafted text twice because its arm count was
  recoverable but not visible, and an optional `n` is an omitted `n`.
- **A value that cannot be re-derived prints "not re-derivable from stored data"** rather than a
  blank, since a blank cell reads as zero.
- **Full precision and the paper's rounding sit in adjacent columns.** The inventory compares at
  quoted precision; the notebook is where the unrounded value lives.

`tests/test_analysis_ledger.py` pins all three, plus that the ledger covers all four fault families,
that every entry names where it is quoted, and that unresolved non-effects are marked so rather than
reading as corrected measurements.

**The notebook found something on its first run, which is the point of it.** §5.3 quoted the
matched-subset-versus-full-grid gap as 5.1% and compared it to a 1.5% spread across widths, calling
it "three times". The 5.1% was four-corner and the 1.5% three-corner, so the ratio was between two
different populations. At a common grouping the gap is 4.5% against a 1.52% spread — still three
times, so the claim survives, but it was only accidentally true. Corrected in `07-writeup.md`,
`make_inventory.py` and here, with the reason recorded in the inventory so the pairing is not
re-split later.

**Also re-derived rather than transcribed:** the H2d stratification, which was previously an ad-hoc
computation recorded only in this log. The notebook reproduces it exactly — pooled −0.128, within-γ
+0.056 / +0.026 / +0.009 / +0.108 / +0.190 / +0.322, fully stratified +0.135 at γ = 10 and near zero
elsewhere. That was the last analysis in the paper with no script behind it.

Two housekeeping consequences: `notebook` is a new optional dependency group, kept out of the
default set so no measurement path can acquire a dependency on notebook tooling; and `flake8` is now
declared in `dev` with a `.flake8` at max-line-length 100, having previously been present only by
accident and pruned by the first `uv sync`.

### Pre-commitment for Tier 2.2, written before the numbers are computed

Kati's point is that 2.2 is the item whose outcome cannot be predicted, and that a gain confined to
task 0 is a different claim from backward transfer generally. So the framing is fixed here, against
the design as it actually exists, before running anything.

**The design.** Ten (task, lag) cells at 40 arms each: task positions {0, 4, 8, 12} and lags
{3, 4, 7, 8, 11, 12, 15}. Lag and position are confounded by construction — only early tasks can
have long lags — so **two matched series carry the whole question** and nothing is pooled across
them:

- **position at matched lag**: tasks 0, 4 and 8, all at lag 4. Task 12 cannot appear; its longest
  lag is 3.
- **lag within one position**: task 0 at lags 4, 8, 12 and 15.

**Resolution rule, same gate as everywhere else.** A cell shows the gain if its mean Δ log α is
positive and at least 2 α noise floors, with a bootstrap CI over its 40 arms excluding zero. Cells
below the gate are *unresolved*, not zero.

**The three outcomes, with the sentence each licenses, fixed now:**

1. **General** — resolvable at all three positions at lag 4 *and* at all four lags within task 0.
   Then §5.1 says: in the benign corner retained capacity rises rather than falls, at every position
   and lag the design can resolve, and backward transfer is a property of the corner.
2. **Primacy** — resolvable at task 0 and at neither of the other two matched positions. Then the
   claim is **not** backward transfer. It is that *the first task of a stream benefits from a
   subsequent stream of similar tasks*, which is a statement about position and would need a
   mechanism about the first task's privileged role. §5.1 would then say that, and would explicitly
   decline "similar streams produce backward transfer" as a corner-level property. Figure 2's benign
   panel would be relabelled to name the task it describes.
3. **Gradient** — resolvable at some positions and not others, monotone in position. Then §5.1
   reports the gain as largest for the earliest task and declining, gives the count, and makes no
   claim beyond the positions where it resolves.

**Two things ruled out in advance.** No claim about lag will be drawn from anything but the
within-task-0 series, and no claim about position from anything but the matched-lag-4 series;
the pooled lag ratio already overstated lag once for exactly this reason (§A.1). And if the outcome
is (2) or (3), the existing §5.1 sentence — "a property of that corner" — is **too strong as
written** and gets narrowed, regardless of how much better the categorical version reads.

### Tier 2 complete — and 2.3 landed the opposite way from the caution

`scripts/tier2_backward_transfer.py`, reading stored geometry, training nothing. Report in
`results/tier2_backward_transfer.json`; notebook §4 is no longer a stub.

**2.2 first, since its framing was pre-committed: general.** All ten (task, lag) cells of the benign
corner resolve as gains, at each of γ = 1, 3 and 10 — thirty cells, no exceptions. That is
pre-committed outcome (1), so §5.1's categorical sentence stands unnarrowed. The two matched series,
never pooled: at lag 4 the gain runs +8.7, +12.6, +12.8 floors for tasks 0, 4, 8 at γ = 10, and
within task 0 it runs +8.7, +9.0, +6.2, +3.1 floors at lags 4, 8, 12, 15. **The primacy reading is
not merely unsupported, it is backwards** — at matched lag the earliest task gains *least*. And the
gain decays with elapsed time rather than accumulating.

**2.1 found something the Tier 1 qualitative note had missed.** The gain is not the losses with all
three signs flipped. For γ ≥ 1 the radius term is negative in **every corner, including the one that
gains**: `R_eff` grows wherever the representation moves, and that costs capacity everywhere. Only
utility and dimension change sign. So the four corners share one channel and differ in two, which is
a sharper statement than "a mechanistically distinct process" and a better argument for the
decomposition — pooling the three losses had hidden the common channel just as it had hidden the
gain.

Reported in **signed** shares, which surfaced a third instance of the §A.3 family. The standard share
is `|term| / Σ|term|`, a magnitude by deliberate choice with a stated justification. In the benign
corner the radius term opposes the other two, so that share reads as an 11% contribution to a gain it
subtracts 11% from. A normalisation by a sum of magnitudes takes an absolute value without spelling
it `abs`. §A.3 now says so.

**2.3 is where the caution's reasoning failed, and the conclusion held anyway.** The predicted
outcome was two unresolved locations. Instead: at lag 12 the peak is *sharply* determined —
interpolated γ = 1.23, paired bootstrap over 40 (seed, stream) units [1.18, 1.28], a fifteenth of a
grid step. Model error across five defensible functional forms is 0.28 steps and dominates the
sampling CI, but even that leaves the location inside a third of a step.

**What decides it is that the peak moves monotonically with lag**: 2.86, 1.75, 1.23, 0.96 at lags 4,
8, 12, 15 — nearly a full grid step end to end. There is no richness at which backward transfer is
strongest; there is one per lag. The question is ill-posed unless the lag is fixed first, so §5.1
reports the shape and no location. Two notes on process. First, an earlier version of the check
called this "stable" because the range fitted inside one grid step; a range test cannot see a
monotone slide, and the flag now tests the trend. Second, **the unconditional text rule was worth
having precisely because its stated reason turned out to be wrong.** Had the rule been conditional on
the CI, a 0.07-step interval landing at 1.23 would have licensed exactly the coincidence sentence the
rule existed to prevent.

**§A.4 added: comparisons between quantities pooled over different populations.** Named by the 5.1%
/ 1.5% pairing the notebook caught, with §5.3's three-corner decomposition beside four-corner ratios
as the second instance. Distinct from §A.1 because each value is correct about its own population, so
no recomputation finds it — only asking of each number what it was computed over. The appendix intro
now says four families.

**§B gained a sentence on executable provenance**, since the stratification script closed the last
gap: no number in the text exists only as a figure produced once or a computation done at a prompt.

Inventory now 68 numbers, 19 prose-pinned. 140 tests, lint clean, notebook 13 cells in 4.8 s.

### §5.5 written — and two registered hypotheses had no verdict until now

The last drafting item. Full table of all thirteen registered entries including H3a/H3b, H4 and H5,
which were never run: a register with the untested rows deleted is not a register.

**H1b had no recorded verdict, and the honest one is "untestable".** It predicted lazy forgetting is
ρ_c-accounted. At γ = 0.03 forgetting is 0.2 floors — under the gate — and the ρ_c-to-radius
attribution is available in **0 of 120 arms**, rising to 58% at γ = 0.1 and 100% from γ = 1. So both
the subject and the instrument are absent exactly where the hypothesis lives. Reported as untestable
rather than null: a null implies we looked and saw nothing, and there was nothing to look at.

**H1a passes its statistic and fails its mechanism, and the statistic is the interesting part.**
Registered measure: the combined radius + utility share, predicted higher in rich. It is, 0.470 at
γ = 0.1 against 0.563 at γ = 10, so the kill does not fire. But three shares sum to one, so
"radius + utility" **is** "one minus dimension" — the registered test reduces to *the dimension share
falls with richness*, by 0.09. And the named mechanism is inverted: radius was predicted to carry
rich forgetting and its share falls 0.387 → 0.114 while utility rises 0.083 → 0.449. A share of a
sum-to-one decomposition has to be registered per channel; pooled across channels it cannot
distinguish the hypothesis from its opposite.

Also stated in §5.5: H2a as the clean case of *prediction wrong, criterion did not fire*; H2c's
statistic becoming undefined before it could be evaluated, so that the strict two-grid-step kill
overstates what was learned; γ\* unidentified and therefore no "X coincides with γ\*" claim anywhere
in the paper; and the list of criteria that were set and did not fire, including the one case (H2d
pooled) where we declined a criterion's verdict with the confound documented rather than asserted.

Nine tested, four failed, three of the four with better replacements.

---

## 2026-08-13 — Submission pass: figure provenance, citations, and an executable checklist

Four items from the team's list. Two are done and checkable; one is done as far as it can be
without the LaTeX skeleton, which is not in this repo; one is a checklist that now runs.

### 1. Figure PDFs and the manifest

`scripts/make_figures.py --check` reports **manifest matches disk**: 6 figures, 18 artifacts, all
at the digest the manifest claims, nothing in `figures/` that the manifest does not name. The 12
superseded four-corner variants are in `figures/superseded/`, moved there by the script rather than
deleted, and none is referenced from the draft or from `paper/figures.tex`.

Regenerating the set exposed **three provenance gaps that a matching manifest did not catch**,
which is the more useful finding, because all three were invisible to `--check` as it stood.

- **`module_hashes` was `{}`.** The field was present, well-formed, and empty. `make_figures.py`
  runs the figure scripts as subprocesses, so nothing they import registers in the parent process,
  and `provenance.code_stamp()` was reading an empty registry. Fixed by importing the analysis
  modules in the parent for their side effect on the registry; the manifest now carries hashes for
  `attribution`, `grid` and `core`. A provenance field that is present and empty is worse than one
  that is absent, because it reads as having been recorded.
- **`fig_width` had no `result_set_sha` and no `n_arms`.** It is the only figure that reads two
  result sets — the width arms and a matched subset of the registered grid — and the script wrote
  neither into its sidecar, though the filename carried a hash of the cell keys. It now stamps
  three: `width_arms_sha`, `grid_matched_subset_sha`, `grid_full_sha`, with `n_arms = 144`. A single
  `result_set_sha` on a two-input figure would name one input and imply it had named them all.
- **The manifest records `git_dirty: true`.** This is the one item that is still open and it is not
  mine to close: the working tree has 46 uncommitted paths including the figure scripts themselves,
  so "regenerable from a committed script" is currently false as a matter of fact. The check now
  says so.

`--check --submission` was added for exactly this: `--check` answers whether the files on disk are
the files the manifest describes, which is a different question from whether they are reproducible.
A figure drawn from an uncommitted tree, or from unnamed inputs, is byte-identical to one that is
neither. The stricter mode fails on an empty `result_set_sha`, a missing `n_arms`, an empty
`module_hashes`, or a dirty tree. It currently fails on the last of those alone.

### 2. Inventory against the draft as it will be submitted

**75/75 numbers match their sources; 25 verified verbatim against the sentence that quotes them.**
Re-run after every prose edit below, including the citation insertions, and clean each time.

One number the inventory structurally could not catch: §B said the inventory pins **thirteen**
sentences, which was true when it was written and is now 25. The inventory verifies numbers against
sources and sentences against numbers; a sentence *describing the inventory* is outside both loops.
Corrected to "each of the 75 numbers ... and 25 of them". Noting the shape rather than the typo —
self-describing prose is a blind spot of any checker that checks everything except itself.

### 3. Citations

The draft named one author (Chou, in the §5.5 table) and otherwise attributed nothing. Five sites
now carry names, chosen as the places where the paper uses someone else's object rather than merely
touches the same topic:

| §5.1 | the three-factor identity | Chou et al. (2026) |
| §5.1 | the γ₀ parameterization of richness | Graldi et al. (2025) |
| §5.1 | the feature × readout 2×2 | Hiratani (2024) |
| §5.2 | the refit-readout probe margin, H2d's registered measure | Johnston & Fusi (2023) |
| §5.4 | the human practice result, twice | Menghi et al. (2025) |

`paper/references.bib` holds 13 entries. **Three carry a `% TODO-VERIFY` marker** — Wakhloo PRL
2023, Johnston & Fusi 2023 and Menghi 2025 are recorded in `docs/03-references.md` with authors,
title, journal and year but no volume or article number, and under the verified/inferred rule a
volume number recalled from memory is inferred. An inferred bibliographic field in a submitted
bibliography is indistinguishable from a correct one, which is precisely why it has to be marked
rather than filled in. `check_submission.py` fails while any marker remains; closing them is a
five-minute lookup by whoever has the papers.

### 4. LaTeX assembly — blocked, and the blocker is real

**There is no LaTeX skeleton in this repository, and no TeX toolchain on this machine.** No `.tex`,
`.sty` or `.bib` existed under `continual_geometry/` before today; `\todo{}`, `\tc{}` and `\cn{}`
appear zero times anywhere in the workspace. The skeleton with the six-beat abstract outline is
somewhere else — Overleaf, most likely — so "assemble into the jmlr skeleton" cannot be done here
without inventing a different skeleton and creating a merge conflict with the real one.

What was built instead is the part that drops into any jmlr skeleton unchanged:

- **`paper/figures.tex`** — `\floatconts` floats for all six figures, captions from §D and §E with
  citations attached, `\includegraphics` naming the manifest filenames verbatim, hashes included.
  The hashes are load-bearing and should not be tidied away: they are what stops a stale PDF being
  swapped in without the filename changing.
- **`paper/references.bib`** — above.
- **`scripts/check_submission.py`** — the checklist, below.

### 5. The submission checklist, as a script

Item 4 on the list was "a submission checklist in the log". A checklist in a log gets read on the
day it is written and ticked from memory on the deadline, so it is a script instead:
`python scripts/check_submission.py`.

It reports three states, and the third is the point. **`UNKNOWN` is not a pass.** Page count and
"bibliography compiles" cannot be verified without a TeX installation, and this machine has none;
printing them green because nothing objected would be the fifth instance of the fault §A.2
documents four times. They print as `UNKNOWN` with the reason and are counted separately, and the
exit code distinguishes them: 0 clean, 1 something failed, 2 nothing failed but something could not
be checked and a human still has to.

Current state — **6 pass, 3 fail, 3 unknown**:

| PASS | placeholder macros | no `\todo`, `\tc`, `\cn`, `\fixme` |
| PASS | watermark removed | |
| PASS | anonymisation | 8 files incl. figure PDF binary metadata; no identifying strings |
| PASS | figures traceable | 6 graphics, all in the manifest at the current digest |
| PASS | citations resolve | 3 keys used so far, all present |
| FAIL | figures from a committed tree | `git_dirty` at 443945dc |
| FAIL | bibliography fields verified | 3 `TODO-VERIFY` markers |
| FAIL | keywords present | no `\begin{keywords}` — needs the skeleton |
| UNKNOWN | main document | no `\begin{document}` — needs the skeleton |
| UNKNOWN | page count | no built PDF |
| UNKNOWN | bibliography compiles | no TeX toolchain here |

The anonymisation check reads the figure PDFs as bytes, not just the sources, because the exposure
is in metadata rather than in text. The result is clean: the PDFs carry only `Creator` (the script
name) and `Producer` (the matplotlib version), no absolute paths, no username, no hostname. Worth
recording that the check includes the authors' own unaccepted arXiv preprint by id — a paper that
cites "our prior work" by number has deanonymised itself as completely as a name on the title page,
and that is the failure most likely to be introduced late, while closing a citation gap.

### 6. Two appendix additions from the team's reading of Tier 2

- **§A.2 gains a fourth guard**, and it is a different shape from the other three. The peak
  stability check tested whether the peak's *range* across lags fitted inside one grid step. It
  does, at 0.90, so the check passed — but the four values are 2.86, 1.75, 1.23, 0.96, a monotone
  slide ordered by lag, which is the dependence the check existed to detect. **A range test cannot
  see a monotone trend**: it collapses an ordered sequence to its two extremes, so a systematic
  drift and an unlucky pair of noisy endpoints are the same observation to it. The general form now
  stated in §A.2 is that a check on a summary statistic inherits that statistic's blindnesses, and
  a check applied to ordered data should use the ordering. Added to `CORRECTIONS` as a `guard`
  entry, so the notebook's §10 shows it beside the other three rather than only as the estimand
  change it caused.
- **§A.3's third instance is rewritten around why it hid longest.** The `|term| / Σ|term|` share is
  now stated as: **a normalisation by a sum of magnitudes takes an absolute value without spelling
  it `abs`.** The first two instances are visible at the point of transformation; this one is
  visible only in the denominator, where it reads as a neutral choice of scale, and the sign is
  discarded in the numerator as a side effect. So the rule about justifying every magnitude has to
  cover the cases where no magnitude appears to be taken.

### Also noted

**The deadline is recorded twice and differently.** `docs/05-implementation-brief.md` says Aug 22
AoE; `docs/09-post-generation-brief.md` says Aug 24 AoE. Today is Aug 13, so the gap is 9 days
against 11 — a difference that matters for whether the Split-CIFAR100 pilot is affordable. Not
resolving it unilaterally; flagging it.

140 tests pass, lint clean, inventory 75/75 with 25 pinned, notebook re-executes clean.

### Follow-up the same day — three decisions taken and acted on

**Deadline: Aug 24 AoE.** Confirmed by Kati. `05-implementation-brief.md` carried Aug 22 and has been
corrected, with the correction dated and the disagreement recorded rather than overwritten. 11 days
from today, not 9.

**LaTeX: a fragment, not a skeleton.** `paper/body.tex` is the paper body with no documentclass, no
preamble and no document environment — it pastes into the jmlr skeleton after the title block.
Written this way on Kati's call, because a second `main.tex` would conflict with the real skeleton
rather than save work. Sections: introduction, setup, the decomposition, the benign corner, the two
capacity measures, center geometry, robustness, discussion. Abstract and `\begin{keywords}` included
so the skeleton's own can be deleted or ours can. Two figures in the body, four in the appendix,
`\floatconts` throughout, five citations resolving against `references.bib`.

**Split-CIFAR100: spec only, nothing run.** `docs/10-cifar-pilot-spec.md`. Two facts checked today
that shape it more than anything scientific:

- **No GPU on this machine** (`nvidia-smi` reports no driver) and **no `torch`** — the stack is pure
  NumPy. A learned convolutional encoder is not feasible.
- So the input representation becomes the main design question. The spec recommends **raw pixels into
  the existing MLP**: no new dependency, architecture identical to the synthetic result, which is
  what makes the comparison a comparison. Accuracy will be poor (order 30--40% on 10-way) and the
  spec says so plainly, because the question is whether the geometry of forgetting survives real
  input statistics, not whether the network is good at CIFAR. The alternative — frozen features from
  a pretrained encoder — buys accuracy and pays with a pretraining confound and with part of the
  measured geometry being the encoder's rather than the learner's.

Three further calls in the spec worth surfacing here:

- **No attempt at the $2\times2$.** On real images feature similarity cannot be set independently of
  class identity, so four corners built that way would not be these four corners. Labelling them the
  same and comparing would be the fifth instance of the population fault in §A.4 and the worst of
  them, since the labels would agree while the constructions did not. The pilot answers one question
  instead: does retained capacity fall with a channel composition that reorganises in the same
  direction? A no is publishable as a boundary.
- **The measurement half of the cost is calibrated and the training half is not.** 21.3 s/eval
  uncontended, $4.72\times$ contention at 128 workers, ~4,600 evals/h — so a 24-arm grid's
  measurement is about six minutes of wall time. Raw pixels multiply the first layer's work by
  $20.5\times$ and leave measurement almost unchanged, which **inverts the current cost structure**.
  The spec refuses to project the training side: that number is what the one measured arm is for, and
  the last projection of this kind was wrong by $9\times$.
- **The floors do not transfer.** Nothing is quoted in noise floors until they are re-measured on
  CIFAR manifolds. $R_{\mathrm{eff}}$'s floor was already found $\gamma$-dependent by $6.7\times$
  within this project, and it is the step most likely to be skipped under time pressure.

### A bug I introduced and the test that now pins it

Worth recording because it is the same fault the appendix documents, committed while building the
checker for it. `check_submission.py` gained `strip_comments()` so that a file whose header explains
it contains no `\begin{document}` would not be read as containing one. That helper was then applied
to the `.bib` as well — where the `TODO-VERIFY` markers live behind a `%`. The bibliography check
went from FAIL to PASS **by no longer looking**, and the transition looked like progress.

Caught by reading the diff of the output rather than the output. Fixed by reading the `.bib` raw, and
pinned by `tests/test_check_submission.py`: 16 tests, every one a case where the check could report
PASS while the condition does not hold. The parametrised exit-code test is the load-bearing one —
`UNKNOWN` must never exit clean, since three items on the venue checklist cannot be verified on a
machine with no TeX toolchain.

156 tests pass. Submission check: **7 pass, 2 fail, 3 unknown**. The two failures are both human:
commit the tree so the figures trace to a commit, and look up three journal volume numbers.

---

## 2026-08-13 (later) — role change, a TeX toolchain, and the record report

`docs/11-role-and-open-questions.md` narrows my remit to the record and to interpretation close to
the data; argument, positioning and weighting move outside the repo. Accepted. Report written to
`docs/12-record-report.md`, which is the deliverable for this round.

### Two corrections to facts I asserted

**GPUs 0–3 exist on `grime`.** `docs/10-cifar-pilot-spec.md` §3 built its entire recommendation on
"there is no GPU", which I verified on this host (`a8000-2409n4`, no driver) and then stated as a
property of the available resources. Corrected in the spec, with both facts recorded. A learned conv
encoder is feasible; the recommendation is still raw pixels *for the pilot*, on this host, because
moving host, adding `torch` and introducing a training loop are three changes at once and none of
them is the one being measured. The no-GPU line earlier in this log is a dated entry and stands as
written.

**I introduced the Chou misattribution.** The note observes that the draft attributes the central
identity to the paper that applies it rather than the one that derives it. That was my `\citet` added
last round — a positioning decision made from inside the repo without the literature in view, which
is the failure mode the note is about.

### The page count, which every length decision was waiting on

`tectonic` 0.15.0, harness at `paper/harness/main.tex` in a subdirectory so the submission checker's
non-recursive `paper/*.tex` glob cannot mistake it for the real document.

**Body text alone 5 pages. Body plus the two main figures 8. Plus references 9. The limit is 4.**

The overrun is not caption text — stubbing both captions leaves it at 8. It is the graphics: `fig2`
is 10.5 × 7.6 in and `fig4` is 15.5 × 8.4, and at `\linewidth` in this class each float exceeds the
text block (`Float too large for page by 65.83pt`), so each takes a page. That is a re-plot to a wider
aspect, not a trim. And body text alone is already over at 5.

### Two things only compiling could have found

**`%` does not start a comment inside a BibTeX entry.** The three `% TODO-VERIFY` markers I added last
round silently deleted their whole entries — Chou, Graldi and Menghi were absent from the compiled
bibliography. The marker existed to be loud and was in the one position where it was mute. Now a
`note` field, so an unverified reference prints in the typeset bibliography.

**`check_submission.py` could not count pages.** Its counter regexed `/Type /Page` over raw PDF bytes,
which returns 0 once the engine packs the page tree into compressed object streams, as `xdvipdfmx`
does. It reported UNKNOWN, so it was honest, but it could never have reported the overrun. Now
inflates object streams and cross-checks `/Count`.

Both are the same family as the `strip_comments` bug above: a check that stops seeing rather than
starts failing. Third instance in two days.

### The inventory now points at the submitted files

Quotes are looked for in `paper/body.tex`, `paper/figures.tex` and `paper/figures-appendix.tex`
first, then `docs/07-writeup.md`. Matching is normalised on both sides — symbols expanded so `γ₀` and
`$\gamma_0$` compare equal, markup stripped, whitespace collapsed, which also fixed the old check's
inability to see a quote spanning a wrapped line (verified on `paper/body.tex:95`, where the 69.7
sentence wraps mid-fragment).

**75/75 verified, 25 pinned: 8 in the paper, 17 in the long draft only.** That split now reports what
a length cut costs. `paper/figures.tex` split so the four supplementary floats live in
`paper/figures-appendix.tex`.

### The finding that needs a decision

`scripts/audit_open_questions.py` → `results/open_questions.json`. Zero new compute; six questions
answered from stored geometry. The one that matters:

**Every value §5.1 quotes for the `S-HH` radius term fails the gate the same analysis applies in
panel (d).** Quoted −0.0154, −0.0015, −0.0124 at γ₀ = 1, 3, 10 against per-richness 3σ gates of
0.0241, 0.0214, 0.0336 — all three below. The γ₀=3 value is a 20/20 sign split, mean/SD −0.13. And
the only `S-HH` radius values that *do* clear the gate are at γ₀ = 0.1 and 0.3, where the term is
**positive**. In the three forgetting corners the same term clears by 5–24 floors.

So the shared-channel reading holds in the corners that forget and is unsupported in the corner that
gains. More arms cannot fix it: the floor is estimator dispersion, not sampling error. This is a new
member of the artifact families — not a guard that failed open, not a sign destroyed by a
transformation, not a comparison across populations, but **a gate applied in one panel and not to a
series quoted in prose.**

Also: the `S-LL` ρ_c value at γ₀=10 is 40 positive and 40 negative, mean +0.0058 against SD 0.0169 —
the unresolved convergence value is an exactly balanced coin flip, not a small effect. The
between/within reversal is general across eight measure pairs and *stronger* in all eight than in the
reported pair, which weakens rather than strengthens the specific claim. The γ=30 saturation is
specific to ρ_c (second step 0.51× the first; every other quantity 1.1–1.3×). The `S-HH` gain does not
depend on the measurement boundary at all (R²=0.003) and `task − lag` explains it as well as two free
slopes do — and the lag decay is γ₀=10-specific, absent at γ₀=1.

### Two corrections inside the report itself

Caught by re-reading the JSON against what I had written: the resolution gate excludes 5 of 24 cells,
not 6; and the panel (d) gate excludes **all 160 arms at γ₀=0.03** and 3–25% elsewhere, not "8–95%,
worst where the effect is largest". Coverage at γ₀=10 is the same 75% as at γ₀=0.1, because the radius
term grows about as fast as its floor. My Q13 recommendation to cut panel (d) was partly built on the
wrong reading and is weaker than first written; the report says so.

### Closed a drift risk

`NOISE_FLOOR_CV` in `src/analysis/timewarp.py` is a hardcoded copy of `results/cost_model.json` at
n_t=200, and every floor-denominated number in the paper divides by it. Nothing connected the two.
All four still agree to their rounding; `tests/test_noise_floor_provenance.py` now asserts it.

### New this round

`docs/12-record-report.md` (the report), `docs/13-experiment-registry.md` (Q1 — every run with
parameters, which did not exist), `scripts/audit_open_questions.py`,
`tests/test_noise_floor_provenance.py`, `paper/harness/`, `paper/figures-appendix.tex`.

**161 tests pass**, lint clean, inventory 75/75. One measurement recommended: a γ₀=5 probe, 64 arms,
**12–13 minutes wall** from the measured γ=30 cost, to halve the 3.3× bracket on the `S-HL`
decorrelation onset. It is the only place where under an hour changes what a sentence can say.

---

## 2026-08-17 — γ₀=5 probe, figure replot, gate-faithful prose

### γ₀=5 (the measurement that was supposed to take twelve minutes)

Already on disk from an earlier run; verified rather than recomputed. 64/64 arms usable, 4
conditions × 4 streams × 4 seeds, 9.51 core-hours, 535 s/arm. Analysis via
`scripts/analyse_gamma5.py` using Figure 4's own `per_arm`/`summarize`/`effects`.

S-HL Δρ_c:

| γ₀ | Δρ_c | declining | sign-test p | n |
|---|---|---|---|---|
| 3 | +0.0034 ± 0.0019 | 20/40 | 1.0 | 40 |
| **5** | **−0.0064 ± 0.0021** | **12/16** | **0.077** | 16 |
| 10 | −0.0550 ± 0.0033 | 40/40 | 1.8e-12 | 40 |
| 30 | −0.0956 ± 0.0041 | 16/16 | 3.1e-5 | 16 |

The mean first goes negative at γ₀=5 (3.1 SEM from zero). The per-arm sign test has not yet
rejected. At γ₀=10 it is unanimous. Onset bracket 3→5, factor **1.67** (was 3.33). Additivity
still holds (error +0.0045 ± 0.0062, not resolved). Written into §5.4 and the γ=3 caption; not
over-read as a located threshold.

### Figure replot

`figstyle.figsize` already authored at print width (6.00 in). Heights cut further this round
(fig2 3.40→3.25, fig4 3.30→3.15) to kill the remaining 11pt overflow. `pdfinfo` after replot:

- body text only: 6 pages (was 5; the γ=5 sentence and two citation keys are the difference)
- body + 2 main figures: 9 pages (was 8)
- Float too large: **gone** (was 65.83pt)

The two-page recovery does **not** appear in the page count. Each float still occupies a full
page because the captions typeset to ~5 in, so graphic + caption ≈ textheight even when the
graphic fits. Recovering those pages is a caption cut, not a further replot. Reported rather
than performed.

### Prose that the gate supports

§5.1's radius sentence and the Discussion close were already rewritten in `paper/body.tex` to
claim no sign for the S-HH radius term, only a bound. The stale claim had survived in one
place: `fig:benign`'s caption ("negative here exactly as it is in the three forgetting
corners"). Rewritten to the bound. The family lives as **§A.5** rather than under §A.3: §A.3 is
sign-destroying transformations, and this fault transforms nothing — a gate applied in a panel
and not to the same quantity in prose. §A.3's own header now says so.

§5.2 already leads with the general between/within reversal (8 of 28 pairs); the probe-margin
instance is labelled weaker than all eight. Left as written.

### Three body/caption discrepancies

1. Center-collapse share "through γ₀=3": caption said 0.45, body said 0.44. Source is 0.442 at
   γ=3 and 0.450 at γ=10. Caption was quoting the γ=10 value for the γ=3 claim. Caption → 0.44.
2. Width caption attributed 1.52% to the *utility share*; body attributes 1.5% to forgetting
   *magnitude* and 0.024 to the utility share. Caption rewritten to match the body.
3. `fig:benign` radius sentence, above.

### Citations

Manifold capacity now cites Chung 2018 and Cohen 2020. The floor paragraph cites Chou 2025
(ICML estimator) and Chou 2025 (GLUE). The identity remains Chou 2026 (ICLR §B.3). Nine keys
resolve. Four bib entries still unused (Atanasov, Bernardi, two Wakhloo).

Inventory 75/75, 25 pinned (10 in the paper, 15 in the long draft only). 163 tests pass.

---

## 2026-08-17 — stream_id was a no-op; γ=5 and width n=40 reopened as unique-n designs

### Arrangement uniqueness (asked before any new arms)

`make_stream` never reads `cfg.stream_id`. `run_arm` passed `paired_init(seed)["stream"]`, so
the 5×8 filenames stored a blocking factor that was not consumed. On disk, at every
(γ, condition) cell of the 960-arm grid:

- 40 files, **8 unique** `(S_f, S_r)` matrices, 5 copies each (one per `stream_id`)
- uniqueness is keyed by **seed**, not `stream_id`
- same seed + condition ⇒ identical arrangements across γ (γ=1 vs γ=10, bitwise)
- conditions differ (Hamming sampling consumes the RNG before `make_arrangement`)

This is **not** n=1 arrangement across 960 arms, and **not** 5 independent streams × 8 inits.
It is 8 unique (init + arrangement + dichotomy) draws per cell, seed-confounded, each written
five times. Headline n=40 is n=8 unique. Stream R²=0.000 is partly tautological. Sign tests
on 40 files treat copies as independent.

γ=5 on disk (now `results/gamma_5_n16_streamid_unused/`): 16 files/corner = 4 unique seeds × 4
copies. Width 2×2 files = 2 unique seeds.

Fix, for new arms only: `pipeline.stream_rng(stream_id)` keys dichotomies and arrangements by
`stream_id`; `paired_init(seed)` still keys `W(0)`. Same `stream_id`, different seed ⇒ same
stream, different init. The 960-arm grid is **not** regenerated. New γ=5 / width cells are
complete unique-n designs in their own directories, not mixed with the duplicate-stream files.

### S-HH radius SEM beside the 3σ floor (gate unchanged)

Lag 12, task 0, S-HH. Files are 5 copies of 8 unique seeds, so SEM40 is optimistic by ~√5.

| γ | mean | SEM40 | SEM8 | 3σ gate | \|m\|/SEM40 | \|m\|/SEM8 | R_eff floors | clears 3σ? |
|---|---|---|---|---|---|---|---|---|
| 0.03 | +0.00590 | 0.00015 | 0.00036 | 0.0084 | 38.3 | 16.2 | +2.11 | no |
| 0.1 | +0.03345 | 0.00054 | 0.00127 | 0.0092 | 62.4 | 26.4 | +10.91 | yes |
| 0.3 | +0.04301 | 0.00184 | 0.00433 | 0.0185 | 23.4 | 9.93 | +6.98 | yes |
| 1 | −0.01543 | 0.00168 | 0.00396 | 0.0241 | 9.19 | 3.89 | −1.92 | no |
| 3 | −0.00153 | 0.00191 | 0.00450 | 0.0214 | 0.80 | 0.34 | −0.22 | no |
| 10 | −0.01237 | 0.00206 | 0.00486 | 0.0336 | 6.01 | 2.54 | −1.11 | no |

As **means** over unique n=8: γ=1 is 3.9 SEM from 0; γ=10 is 2.5 SEM; γ=3 is noise. The 3σ
single-measurement gate is unchanged. `radius_vs_gate_table()` now prints both SEMs.

### Costs (measured s/arm on disk)

**γ=5 to 40 unique/corner.** Filling 4×4→5×8 without the RNG fix would add 96 files and still
leave unique n=4→8, not 40. Honest design: 5×8×4 = **160 new arms** after moving the n=16
duplicate-stream set aside. Mean 535 s/arm (S-HL 695 s). 160 × 535 s = 23.8 core-h; wall
~7–12 min at 160 workers.

**Width to 40 unique/cell at γ=10.** N∈{150,600} × 4 corners × 5×8 = **320 arms** in
`results/width_g10_n40/` (not mixed into `results/width/`). Disk: N=150 γ=10 **470 s/arm**,
N=600 γ=10 **933 s/arm** — not the 1294 s W1 mixed-γ average (that was N=600 γ=1). 160×470 +
160×933 = 62.4 core-h; wall ~20 min at 192 workers.

**Not run:** centre-collapse step, S-LL, Figure 4 in-range interaction, γ=0.1 onset, anything
at γ=0.03.

**Arrangement robustness (costed, not run).** Not n=1, so the reviewer's "one draw of 16
manifolds" is already false — but the 8 draws are confounded with init. Unconfounded
K∈{3,5} arrangement draws at γ∈{1,10}, four corners, 40 unique inits:
2×4×40×K = 960 (K=3) or 1600 (K=5) arms. γ=1 S-HL 2420 s, γ=10 S-HL 2060 s. K=3: ~597
core-h, ~3.1 h wall; K=5: ~996 core-h, ~5.2 h wall.

**D∈{2,4,8} (costed, not run).** One γ, one corner, 40 unique arms. D=R=4 and 1 in every
current arm. Don't reuse the D=4 grid cell (8 unique, seed-confounded). 120 new arms at
γ=10 S-HL (2060 s) = 68.7 core-h, ~21 min wall. Cheap after the two launched runs; needs a
decision.

### Presentational

- Figure 2 panel (b): γ=0.1 hatched as marginal (margin 0.10 floors / 5% sits inside the
  floor's ~41% relative SE). Gate unchanged. Body keeps "between γ₀=0.1 and 0.3".
- §12 onset row: 5% is the binding comparison; 20% is a weaker counterfactual.
- Draft checks: paper quotes centre-collapse 0.44 as the γ=3 endpoint and states the 3→10
  step is unresolved; 0.45 in the paper is the utility share, not that step. "Ten (task, lag)
  cells … thirty cells" is 10 cells × 3 γ, not 400 independent observations.
- No body prose cut. Section typeset lengths measured on a body-only harness (HARNESS
  BODYEND=5).

172 tests pass. Inventory 82/82.

---

## 2026-08-18 — unique n=8 labelled; (a) γ=5 genuine n=40 resolves the onset

Reporting decision (not a re-run of the 960-arm grid): the registered grid is **n=8 unique**
per (γ, condition) cell; new arms after `stream_rng(stream_id)` are genuine n=40.

### What changes status at unique n=8 (α=0.05)

Point estimates invariant. One interpretation change: stream R²=0.000 is tautological
(stream_id unused). Peak paired-bootstrap CI at lag 12: files [1.18, 1.28] n=40 → unique
**[1.12, 1.35] n=8** (0.16 grid steps). Still beside the point (peak moves with lag).

Sign tests that still reject, with p three orders weaker: S-HL γ=10 is 8/8, p=0.0078
(was 40/40, p=1.8e−12). S-LL stays unresolved (3/8, p=0.73). Figure 4 interaction
1.94 SEM → 0.82 SEM, still unresolved. S-HH radius: mean 3.9 SEM8 from 0 at γ=1 and
2.5 at γ=10; 3σ single-measurement gate unchanged.

### (a) γ=5 at genuine 5×8 — 160 arms, 1086 s wall → `results/gamma_5_n40/`

The n=16 directory was restored and not overwritten. Analysed to
`results/gamma5_n40_onset.json` (not `gamma5_onset.json`).

S-HL at γ=5, genuine n=40: **Δρ_c = −0.0102 ± 0.0032, 31/40 declining,
p = 6.8×10⁻⁴**. The duplicate-stream 16-file set was 3/4 unique, p=0.625,
unresolved. **The onset now resolves.** §5.4 quotes the genuine n=40 probe.
Bracket remains 3→5 (factor 1.67).

### Paper / ledger

- Setup: 960 files, inference at unique n=8, arrangement confounded with init.
- §5.1 radius: mean distinguishable, single measurement not; gate unmoved.
- §5.1 peak CI: [1.12, 1.35] at unique n=8.
- §B: stream R² tautological; crossed run is the measurement.
- Discussion: sixth artifact family (filename field stored, never read).
- Appendix A.6 in `docs/07-writeup.md`. Notebook §12 shows
  `unresolved_claims_table` + `unique_n_status_table`.
- Inventory 89 numbers, 38 pinned. 173 tests (17 ledger).

### Campaign

(b) width γ=10 genuine 5×8 → `results/width_g10_n40/` (320 arms) running.
(c) K=3 unconfound → `results/unconfound_k3/` queued after (b). D-sweep skipped.

### Unique-n reading (no leading status flip)

No sign test flips at α=0.05. p-values drop ~three orders, which is the check that unique-n
recompute is doing the work rather than preserving the old inference.

- **S-HL at 8/8, p=0.0078** is the floor of a two-sided sign test at n=8 — maximally
  significant given eight independent draws, not weakly significant.
- **S-LL 15/40 p=0.15 → 3/8 p=0.73.** Already unresolved; now nothing, not marginal.
  §12 quotes the unique-n number.
- Stream R² is the only interpretive change (tautological: streams did not vary).
- Peak CI [1.18, 1.28] → [1.12, 1.35]: still a fraction of a grid step; the location
  claim was already declined because the peak moves with lag.

γ=5 genuine n=40 is the before/after that makes the fix legible: duplicate-stream 3/4
p=0.625 unresolved vs genuine −0.0102 ± 0.0032, 31/40, p=6.8e−4. The 3/4 result stays
in the unique-n table. Onset sentence names the mix: intermediate genuine n; endpoints
unique n=8 from the registered grid.

### K=3 pre-commit (written with 0 files in `results/unconfound_k3/`)

`results/unconfound_k3_precommit.json`. Same discipline as γ=30.

**Can show:** whether channel reorganisation and corner ordering reproduce across three
independent arrangements.

**Cannot show:** arrangement-level variance with any precision. K=3 is detection, not
estimation.

**If it reproduces:** the results are not arrangement-specific.
**If it does not:** that is a finding and the paper's scope narrows.

**Reporting:** out of registration; fixed RNG; not pooled with the grid. Appendix
robustness arm, one sentence in Setup, same status as γ=30 and the width arm.

### Population table

`ledger.population_table()` — every result set, file n, unique n, RNG version, and which
paper claims draw on it. The draft is checked against this the way §12 is checked for
resolution. Four (soon six) populations; a silent genuine-n vs unique-n=8 comparison
would be the §A.4 fault.

Unique-n column is now always `N per cell; T total` — N is the sign-test n on one cell,
T is unique files in the set. Mixing N from one row with T from another was the same
fault one level up.

### γ=30 interaction at unique n=4 — STATUS CHANGED

`scripts/audit_gamma30_unique.py` → `results/gamma30_unique.json`. Same SEM formula as
the grid unique-n pass (half RSS of four cell SEMs).

| | files (16/corner) | unique (4/corner) |
|---|---|---|
| interaction | +0.0160 | +0.0160 (point estimate invariant) |
| SEM ratio | **3.89 SEM — clears** | **1.74 SEM — does not clear** |
| independent-cell bootstrap CI | [+0.0082, +0.0239] excludes 0 | [+0.0006, +0.0319] excludes 0 by a hair |
| paired-seed (n=4 seeds) | — | 1.53 SEM, CI includes 0; 1 of 4 seeds opposite sign |

The 3.9 SEM figure was copy-inflation. The additivity-failure claim ("over the registered
range") is **not licensed** at unique n. Body.tex never stated it; 07-writeup §5.4 and
appendix C now say so. S-HL at γ=30 is 4/4 declining, sign-test p=0.125 — n=4 cannot
reject; the mean −0.096 is still clearly negative as a location.

S-LL current-claim language: no submitted-paper sentence still quotes p=0.15 as the
status; 07 §A.1 now flags that figure as copy-inflated and states unique p=0.73 as
nothing, not near-threshold.

---

## 2026-08-19 — width n40 and K=3 analysed; no further runs

Campaign finished 2026-08-18T13:52Z. 160 + 320 + 960 usable, 0 skipped.

### Width genuine n=40 (`results/width_g10_n40.json`)

N=150 vs 600 at γ=10, three-corner n=120 per width. Forgetting −71.6 vs −72.0 floors
(**0.62%**). Alignment share 0.460 → 0.440 (Δ=0.020). Radius 0.146 → 0.098, dimension
0.394 → 0.463, sum 0.540 → 0.560. S-HH a gain at both (+3.3 floors). Still a bound: two
widths, not a located dependence. N=300 is a different population (grid unique n=8) and
is not in the contrast. §5.3 now quotes these numbers; the 1.5%/4.5% duplicate-stream
n=12 bound is superseded.

### K=3 against precommit (`results/unconfound_k3.json`)

Leading pattern **3/3**: channel reorg (alignment up, radius down), S-LH largest positive
Δρ_c, S-HL decorrelates, S-HH a gain. Verdict: **not arrangement-specific**.

'S-HL the only decorrelating corner' is **2/3**: arrangement 2 has S-LL also
decorrelating (−0.0195, 40/40, p≈10⁻¹²). That is the unresolved S-LL corner, not a
failure of the leading results. Scope does not narrow.

Stream R² at γ=10, three-corner, streams actually vary: **0.001**. Seed R² 0.0008. The
grid's stream R²=0.000 was tautological; this is the measurement. §B quotes 0.001.

### Further runs — none

| candidate | why not |
|---|---|
| K=5 | Precommit: K=3 is detection, not estimation. 3/3 leading pattern. Running K=5 after seeing arrangement 2's S-LL would be fishing. |
| N=300 genuine-n width (160 arms) | Three-corner magnitude is already flat 150↔600. A U-shape on S-HH (3.3 at 150/600 vs ~5 at K=3 N=300) is a different population and not a paper claim. |
| D∈{2,4,8} | No claim depends on D. K=3 closed the "one arrangement" objection. |
| γ=30 genuine n=40 | Would be recovering a fenced additivity-failure claim after unique n said it was not licensed. |

Nothing launched.

---

## 2026-08-19 — project check: the sampling unit, not the count

`scripts/audit_seed_security.py` → `results/seed_security.json`. Write-up:
`docs/14-claim-security.md`. The unique-n audit fixed the *count*; this asks which
**unit** each claim is an inference about. Three units exist — initialisation
(`paired_init(seed)`, which also keys the measurement seed), arrangement
(`stream_rng(stream_id)`, which keys manifolds and dichotomies), and the crossed arm.

### The reassuring half

**The grid's unique-n=8 interval is the right size.** Because `seed` keyed the arrangement
too, the grid's 8 unique arms are 8 independent (arrangement, init) draws, so `SEM8` should
already be arrangement-scale. Checked against K=3's real arrangement axis at γ=10:
`SEM8` / between-arrangement SEM = 1.77 (S-HH), 0.97 (S-HL), 0.94 (S-LH), 0.55 (S-LL).
Same order on all four. The relabelling did not just shrink n; it left correctly scaled
intervals.

Duplication confirmed exactly: worst Δρ_c spread across the five copies of a seed at γ=10 is
**0.00e+00**.

`S-HL` at γ=10 is the best-supported claim in the paper: grid 8/8 unique, **and** K=3
40/40 inits in every one of 3 arrangements. `S-HH` gain sign-stable 3/3 (+11.56/+11.68/+12.27
floors at γ=1; +5.74/+5.17/+4.58 at γ=10).

### Sign-test ceilings (design limits, not results)

n=4 → 0.125 and n=5 → 0.0625: **cannot reach α=0.05 under any outcome.** n=8 → 0.0078
unanimous but **0.0703 after one flip**, so every 8/8 grid result is one seed from not
clearing on grid evidence alone.

### The costly half — γ=5 onset is arrangement-dependent

`gamma_5_n40` and `width_g10_n40` are 5 arrangements × 8 inits. 40 arms, but only 5
arrangement draws, so their arm-level SEM is ~3× too small for an arrangement-level claim.
γ=5 S-HL by arrangement: −0.0114, −0.0169, **+0.0084**, −0.0419, **+0.0109**
(arm signs 8/8, 8/8, 5/8, 8/8, 2/8). The two positives land on the lazy-arm drift band
[+0.005, +0.010]. Arrangement sign test 3/5 p=1.0; SEM 0.0096 vs 0.0032 over arms; 95% t CI
**[−0.0368, +0.0165] includes zero**. By initialisation it is 8/8, p=0.0078.

So `paper/body.tex:216–218` and `figures-appendix.tex:44–45` ("−0.0102±0.0032, 31 of 40,
p=6.8e−4", "onset sits between 3 and 5") are right as arithmetic and wrong as a population
label — the onset is an arrangement-level claim. Honest form: bracket 3→10, and γ=5 is
negative in 3 of 5 arrangements. Same fault family as §A.4, one level deeper.

### Δρ_c magnitudes are soft across arrangements

Between-arrangement SEM is 5.5–10× the within-arrangement SEM for all four corners. Grid
−0.055 for S-HL falls **outside** the three-arrangement range (−0.0500, −0.0511, −0.0262).
Quote the sign and the ordering, not −0.055 as a population value. The γ=10 interaction
(−0.0130 / +0.0050 / −0.0160) and S-LL (+0.0017 / +0.0182 / −0.0195) flip sign across
arrangements — new evidence *for* the existing "unresolved" verdicts.

### Width, paired

Marginal arrangement spread at fixed width is 8.8–10.6%, **17×** the 0.62% width gap. Paired
within arrangement (600 − 150): +0.20, +0.13, +1.57, +0.09, +0.26 floors, mean +0.45, 5/5
same direction (p=0.0625 = the ceiling), 95% CI [−0.34, +1.23] includes zero. Direction
consistent, magnitude under 1%, still a bound. Claim unchanged.

### Costed fix, not yet run

γ=5 at **8 arrangements × 8 inits** = 256 arms ≈ 29 min (observed rate: 160 arms in 1086 s).
n=8 is the smallest arrangement count whose ceiling clears α=0.05. Rule to pre-commit before
the arms exist: ≥7/8 arrangements decorrelating ⇒ the onset sentence stands at arrangement
level; otherwise the paper reports the 3→10 bracket and calls γ=5 arrangement-dependent.
A power fix on an existing claim with a fixed rule, not a search for a better p.

---

## 2026-08-19 — γ=5 8-arrangement pre-commit scored: arrangement-dependent

`results/gamma5_k8_precommit.json` written before any arm. 256/256 usable in 1716 s wall.
Arrangements 0–4 reproduce `gamma_5_n40` bitwise (160/160).

S-HL arrangement means: −0.0114, −0.0169, +0.0084, −0.0419, +0.0109, **+0.0015**,
**+0.0166**, −0.0221. **4/8 negative.** 95% t CI [−0.0234, +0.0097] includes zero.
Sign-test p = 1.0 (ceiling 0.0078). Verdict: **arrangement_dependent**.

§5.4 quotes four of eight and the grid's 3→10 bracket (factor 3.3). The arm-level
p = 6.8e-4 is not the onset evidence. `docs/14-claim-security.md` §7.

---

## 2026-08-19 — width smoke N=32 / N=100 (pre-commit scored)

`results/width_smoke_precommit.json` written before any arm. Init-readout curve
(33 dichotomies, frozen W) then 4 full S-HL arms (stream 0, seed 0), 1209 s wall.
Own directory; not adopted by `grid.load()`.

**Init floor.** Module-A exact-readout median MSE: 0.215 (N=32), 0.116 (50),
0.088 (64), 0.051 (80), 0.037 (100), 0.012 (150), 0.002 (300). At N=32, 0/33
dichotomies have MSE≤0.05 or perfect sign; even concatenated A+B median MSE is
0.098. At N=100, 97% hit MSE 0.05 but only 52% have perfect sign on module A.

**Trained arms.** N=32 γ=10: 16/16 converged. N=32 γ=0.03: 12/16, task 0 miss is
loss 0.057 @ 20k with acc 0.997 — a haircut, not a collapsed classifier; end
‖ΔW‖/‖W‖ = 0.053 vs 0.0015 at N=300 lazy. N=100: both γ 16/16. I6: init clouds
bitwise identical across γ. Identity residual ~10⁻¹⁶.

**Geometry / behaviour, n=1, descriptive.** N=32 D_eff ~1.2–1.5 (packed); lazy
retained α stuck at 0.508. N=100 lazy generic α=0.303 (near 0.33), rich after
task 0 = 0.429. Past-task accuracy at end of stream ~0.53–0.63; at N=300 this
was ~1. PCA of N=32 clouds: init is a 2D blob; after task 0, γ=10 separates
manifolds and γ=0.03 does not.

**Pre-commit verdict.** `n32_go_capacity_limited`: **true**. `n32_stop_broken`:
false. `n100_go_small_robustness`: **false** (letter: rich α@b0=0.429 vs 0.33;
that comparator is after learning, i.e. H2a). Do not spend a four-corner
robustness grid at N=100. Expand N=32 only as a new experiment with re-pinned
stopping; do not pool with N=300.

Artifacts: `results/width_smoke.json`, `fig_init_readout.png`, `fig_clouds_N32.png`.

---

## 2026-08-19 — S-HH at N=32 / 100 (pre-commit scored)

`results/shh_width_precommit.json` written before any arm. 16/16 completed in
1081 s wall (4 seeds × 2 widths × 2 γ, stream 0, S-HH). Own directory.

**Correction of the S-HL smoke reading.** S-HL past-task accuracy ~0.6 is already
the N=300 value for that corner. This run is the S-HH question.

**Pre-commit verdict: `split_breaks`.** N=32 γ=0.03 mean CF 0.086 (range
0.062–0.113), mean past-task acc 0.912. That clears both break thresholds
(mean CF ≥ 0.05, past-task acc < 0.95). Expand is licensed as more
arrangements at N=32 S-HH, not a GLUE grid.

**The break is lazy, not rich.** N=32 γ=10: mean CF 2×10⁻⁵, past-task acc
0.9998, 4/4 usable — in the N≥150 band. N=100 γ=10: the same. N=100 γ=0.03:
mean CF 0.035, past-task acc 0.963, 4/4 usable — intermediate (not ≤0.01, not
≥0.05). Two of four N=32 lazy arms missed loss 0.05; the two that hit it still
have CF 0.078 and 0.062.

Lazy S-HH CF on this arrangement: ~0 (N=300) → 0.035 (N=100) → 0.086 (N=32).

Artifacts: `results/shh_width.json`.

---

## 2026-08-19 — small-N full 2×2 (CF + GLUE, 8 seeds)

`results/small_n_grid_precommit.json` written before any new arm. 128 arms
(N∈{32,100} × γ∈{0.03,10} × 4 corners × seeds 0–7, stream 0). 20 imported
from `shh_width` / `width_smoke`, 108 trained, 3301 s wall. All 128 present.

**Pre-commit verdict.** Rich CF 2×2 **survives** at N=32 and N=100. GLUE Δ log α
signs at γ=10 **match** at both N (S-HH >, other three <). Lazy S-HH CF break
at N=32 **confirmed** at n=8 (mean 0.096). N=32 packing **yes** (D_eff 1.22).

**γ=10 mean CF (8 seeds), vs N=300 ref.** N=32: HH 0.000, HL 0.381, LH 0.013,
LL 0.145. N=100: 0.000, 0.380, 0.009, 0.138. N=300: 0.000, 0.407, 0.013, 0.155.
Order S-HL > S-LL > S-LH > S-HH at both small widths. Δ log α: S-HH +0.116 /
+0.072; loss corners −1.0 to −1.7. All γ=10 cells 8/8 usable.

**γ=0.03.** N=32 usable 4/8 HH, 1/8 HL, 0/8 LH, 0/8 LL — S-LH/S-LL CF are not
matched-loss. N=100 all 8/8; S-HH CF 0.027 (intermediate).

Not pooled with the registered grid. Channel shares at N=32 not a §5 test.

Artifacts: `results/small_n_grid.json`.

---

## 2026-08-19 — N=16 full 2×2 (capacity decline, same grid)

`results/small_n16_precommit.json` written before any N=16 arm. 64 new arms
(N=16 × γ∈{0.03,10} × 4 corners × seeds 0–7, stream 0) into
`results/small_n_grid/` beside N=32/100. 953 s wall, 0 skipped. Init readout
already on disk: module-A median MSE 0.312, A+B 0.199, 0/33 dichotomies at
0.05 or perfect sign. P/N = 1, packing P(D+1)/N = 5.

**Pre-commit verdict.** Rich CF 2×2 **survives**. Rich fit does **not** break
(γ=10 S-HH 8/8 usable, all 16 tasks hit 0.05 on every rich cell). GLUE Δ log α
signs at γ=10 **break** the strict rule: S-HH mean Δ log α = **−0.007** (need
>0). Lazy: **0/8 usable on all four corners** (predicted floor). D_eff packed
at 0.996.

**γ=10 mean CF (8 seeds).** HH −0.000, HL 0.383, LH 0.021, LL 0.151. Same
order as N=32/100/300. S-HH past-task acc 1.000. Loss-corner Δ log α still
negative (HL/LL −0.655, LH −0.593) but **compressed** vs N=32 (−1.4 / −1.0).
S-HH Δ log α is init-noise around zero: 4 seeds +, 3 −, 1 ≈0 (seed 6 α stuck
at 1.011). Do not read that as a 2×2 reversal; CF is still silent on S-HH.

**Capacity decline (the point of the run).** Generic D_eff at b0, S-HH γ=10:
N=16 **0.996** → N=32 1.22 → N=100 2.71. Generic α at b0: **1.005** → 0.505 →
0.413. After S-HL overwrite, retained α at lag 15 floors at **1.00** (N=16),
0.50 (N=32), 0.39 (N=100). Packed GLUE cannot go below one leftover dimension.

**γ=0.03.** Geometry frozen (Δ log α ~ 10⁻¹⁶, retained α stuck at 1.00).
S-HL/LH/LL: 0/16 tasks hit loss 0.05. S-HH: mean 4.1/16 tasks hit, last-task
loss ~0.054, still 0/8 usable. Those CF means are not matched-loss.

Not pooled with the registered grid. Do not quote N=16 channel shares.

Artifacts: `results/small_n_grid.json` (192 arms), `results/small_n16_precommit.json`.



