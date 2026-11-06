# 25 — Agent instructions from here on

```
Status:  RATIFIED by Kati, 2026-11-03. Operating rules for every arm from now.
Top:     docs/24-project-skeleton.md is the scientific hierarchy.
         This file is how work is done inside it.
Keeps:   AGENTS.md invariants I1–I10, §5 hygiene, §8.1 verified/inferred,
         §8.2 artifact check. Nothing here relaxes them.
Amends:  docs/23 (order) runs on the new core learner, not the two-module one.
First:   add one line at the top of AGENTS.md: "Read docs/24 and docs/25 first."
```

---

## 0. The rule that replaces most others

Every arm names its slot in `docs/24` **before** its pre-commit:
§4 core, a §5 ablation row, or a §6 extension axis. No slot, no run.
Validation and robustness results are never reported as findings.

---

## 1. Core rerun — one module, one change

**What changes.** Configuration ~~C2~~ **2M** (two identical modules summed into one
readout) becomes ~~C1~~ **1M**: one hidden layer, `N = 300`. ~~Nothing else changes, so the
new core is comparable to every existing result.~~ The two-module structure was
a vestige of the modularity origin, not a design choice for this question.

**[Amended 2026-11-06, Kati]**
- There is no comparability requirement with existing results (`docs/24` §0a).
- 1M at N = 300 halves total width relative to 2M (2 × 300). Kati's reason, for
  the choices register:
  - N = 300 is the smallest width at which readings were width-invariant
    (N = 150–600), at the lowest cost.
  - `lr0` and the noise floors were established at 300.
  - μP keeps γ₀ comparable across widths.
- Width is tested as an ablation, N ∈ {150, 600} (`docs/24` §5).

**Added to the measurement (same runs, no extra arms):**
- forward-transfer quantities, §4
- measured richness at every boundary, §3 (already logged as a manipulation
  check; make it a first-class output)
- `make_stream(..., rng_dichotomies=...)` on, so task 0 is paired across
  Hamming levels (LOG 2026-10-28: the default left `s_r` cells unpaired)

**Scale.** Design scale, not the v3 reduction: `M = 150`, `n_t = 200`.
v3 (`M = 50`, `n_t = 50`) is a robustness configuration, not paper numbers.

**Default grid (propose changes, do not apply them):**

| factor | levels |
|---|---|
| γ₀ | 0.03, 0.1, 0.3, 1, 3, 10 |
| input change `s_f` | frozen 1.0, drift 0.9, jump 0.1 |
| Hamming to predecessor | 0, 2, 4, 6, 8 |
| population | n = 8 unique (arrangement, init), fresh ids, recorded |

720 arms. Time one measured arm first (AGENTS §5). If the measured cost does
not fit, propose a cut to Kati — γ₀ {0.1, 1, 10} × Hamming {0, 2, 4, 8} is
the fallback (288 arms). Cut levels, never seeds.

~~**Pre-commit before the grid (this is also the C2→C1 check):**
C1–C3 of `docs/24` §3 keep their **sign and ordering** under C1. Magnitudes
are reported side by side with the C2 grid, not pooled. A sign change is a
finding about the two-module learner and stops the arm for a decision.~~

**[Amended 2026-11-06, Kati]** Dropped. Old results are signal, not baselines
(`docs/24` §0a), so the 1M grid needs no agreement check against the 2M grid.
The pre-commit for the grid is the confirmatory-test specification (`docs/26`
A4; drafted for Kati to sign). No grid runs before Kati signs `docs/26` and the
grid.

---

## 2. Choices register — no unexplained choices

Write `docs/reference/choices-register.md`: one row per training or
architecture choice in the core, with (a) the choice, (b) whether it is
standard practice and where, (c) the reason it is needed for this question,
(d) the ablation that tests it, if any.

Seed it with:

| choice | standard? | reason | tested by |
|---|---|---|---|
| one hidden layer ReLU MLP | standard in lazy/rich theory | I4; geometry of one nonlinearity is interpretable | depth L=3 (blocked on μP derivation) |
| μP, readout init 0 | standard for γ₀ (Graldi et al.) | I5, I6 — richness manipulation | — |
| full-batch GD | standard in theory, not in practice | removes SGD noise as a confound | none yet → §6.1 extension |
| MSE on ±1 labels | standard in theory | operating-point problem with BCE (`docs/21`) | CE, closed |
| quadratic μP LR law | non-standard | — | corrected LR: harmless at γ>1 |
| stop at matched loss 0.05 | non-standard, needed | equal current-task fit across γ | — |
| no WD / norm / clipping | deviates from practice | I1, I2, I7 protect the manipulation | — |

A choice without a reason in (c) is a decision for Kati, not for the agent.
Change one choice at a time, as an ablation row, never inside the core.

---

## 3. Measured richness alongside γ₀

**Primary:** relative hidden-weight change after task 0,
`r = ‖W(t₀) − W(0)‖_F / ‖W(0)‖_F`.
**Secondary:** relative NTK (or hidden-kernel) change after task 0.
Both already exist as manipulation checks; report them per arm.

**[Amended 2026-11-06, Kati]** ~~Primary and secondary above.~~ Primary is the
**analytic NTK** on a fixed probe set (e.g. 20 points per manifold), reported
three ways:
- (a) relative change of the full NTK, `‖K(t) − K(0)‖_F / ‖K(0)‖_F`;
- (b) relative change of the readout block `K_u = s²HHᵀ`;
- (c) NTK–label alignment `CKA(K, yyᵀ)` (Chou et al. 2025, Table 1).

Reason for (b): `u(0) = 0` makes the hidden-weight block `K_W` zero at init, so
the full-NTK change includes readout growth even when training is lazy. Check
whether `K_u` separates lazy from rich more cleanly than the full NTK.

Secondary: weight change from init (logged), and the per-task
`r_t = ‖W_end(t) − W_start(t)‖ / ‖W_start(t)‖`, which is new code (store `W` at
each task start). Hidden-Gram CKA is **not** a richness axis, because it
measures the representation, which is the outcome. Neither the NTK nor `r_t`
existed in `src/` on 2026-11-06. The A–B–A notebook's `aba_lib.py` implements
both.

For every claim C1–C3, produce the result **twice**: against γ₀ and against
`r`. Pre-commit, per claim: the reading is the same on both axes (sign and
ordering of conditions). Disagreement is reported as a finding about what the
γ₀ label tracks, not resolved by choosing an axis.

Within-learner, γ₀ stays the manipulated variable. Across learners (§6.1),
`r` is the axis, because γ₀ is not defined, or not equivalent, there
(Adam's `partial_manipulation` gate).

---

## 4. Forward transfer — one geometric, one behavioural measure

**[Amended 2026-11-06, Kati]** This section is superseded by the forward-transfer
definitions in `docs/26` A2 as amended:
- **Geometric, specific:** `Δlog α_{y_{t+1}} − Δlog α_{y*}`, using the held-out
  probe y*, with `s_r(y*, y_t)` recorded. Only `Δlog α_{y_{t+1}}` is decomposed
  into channels.
- **Behavioural:** steps to criterion against a fresh learner from the same init.
- **Zero-shot:** the first loss of each task is stored.
- **Exclusion:** only Hamming 0 with frozen input.

The text below is kept for the record.

At each boundary `t`, on the representation after task `t`:

- **Geometric.** Capacity of the *next* task's dichotomy `y_{t+1}` on task
  `t+1`'s manifolds, relative to init:
  `Δlog α_fwd(t) = log α_{t+1}(h_t) − log α_{t+1}(h_0)`,
  decomposed into the same three channels. Positive means the past made the
  next task more separable.
- **Behavioural.** Steps to criterion on task `t+1`, against task 0's steps in
  the same cell (task 0 is trained from init; this is a proxy, say so).

**The question, and only this one:** which channel change predicts
behavioural forward transfer, **conditional on** Hamming(`t`, `t+1`) and input
change? Hamming 0 is excluded from the reading (identical task; transfer is
trivial).

Guards:
- Stratify by condition. No pooled correlations. The H2d sign reversal under
  pooling (`docs/14` A8) is the reason.
- One primary cell, named in the pre-commit. Everything else is reported, not
  read.
- Do not add a third forward-transfer measure without a slot.

---

## 5. Rerun policy — results produced before a fix

**[Amended 2026-11-06, Kati]** Simplified, and it overrides the text below:
- **No old arm is rerun.** Every claim comes from new runs on fixed code.
- The provenance audit is **one table** that marks every existing result file
  "pre-method, signal only". The `clean` / `unaffected` / `rerun` statuses below
  are not used.
- The error registry stays, but each row's job is a **fix in code with a test**.
  It does not drive reruns.
- Add every item listed in the 2026-11-06 analysis (about 20 beyond the seed).
- Split the two ρ_c errors into separate rows: the `rho_c_glue` calibration
  clip, and the earlier `|ρ_c|` absolute-value convention.
- Correct the `stream_id` row: the affected sets were the 960-arm grid, the
  γ=5 n16 set (4 unique seeds × 4 copies) and the 2×2 width set (2 unique
  seeds). The n=40 sets are the post-fix reruns.
- The live forgetting-formula error (`a_{i,i}` instead of the running max) was
  fixed in `src/pipeline.py` with tests on 2026-11-06.

Build `docs/reference/error-registry.md`: every code error found so far, with
the date and commit of the fix and what it could affect. Seed from LOG:

| error | fixed | affects |
|---|---|---|
| stopping on accuracy, spec said matched loss | 2026-08-11 | anything trained before |
| `rho_c` clip / absolute-value convention | Aug | Δρ_c, Fig 4 |
| `stream_id` stored, never read (n=40 was n=8) | 2026-08-17 | γ=5, width n=40, anything citing n>8 |
| `abs(Δlog α)` in plotting | `docs/09` | figures, any sign read from them |
| pairing across `s_r` assumed, not true | 2026-10-28 (opt-in) | paired readings across Hamming |
| dirty-tree stamps (all 270 v3 arms) | — | code state unprovable |

Complete the table from `LOG.md` and `docs/07` Appendix A before using it.

Then produce `results/provenance_audit.json` and a one-page table: every
result file that feeds a `docs/24` §3 claim or §5 row → code commit, dirty
flag, estimator version, and status:

- `clean` — produced after every relevant fix, clean tree
- `unaffected` — produced before a fix, **with a test or bitwise
  regeneration of a sample** showing the fix does not change it
- `rerun` — affected, or unknown (dirty tree counts as unknown)

Rerun every `rerun` item that feeds a claim. Most are subsumed by the core
rerun (§1); list which. Items that feed nothing go to archive, not rerun.

---

## 6. Open ablations — run on the new core learner

### 6.1 Order (task position) — `docs/23`, amended
Design and readings unchanged, except: the learner is ~~C1~~ **1M**, and the `forward`
schedule comes from the new core grid, not `results/hamming/`. Amend
`results/order_precommit.json` accordingly **before** the one-arm. Primary
cell stays lag 4, presentation index 8, γ₀ = 10.

### 6.2 Exposure spacing — new pre-commit, ~~`docs/26`~~ `docs/27` **[Amended 2026-11-06]**
**Question.** At equal total exposure to a task, does distributing it change
retained capacity and its channel composition?

Candidate design (Kati signs before implementation):
- Target task A gets total budget `B` steps (matched loss is impossible here;
  use a fixed budget and report final loss).
- Schedules: massed (one block) vs spaced (`k` blocks), with intervening tasks
  at fixed Hamming and input level, so only spacing varies.
- Readout: Δlog α of A at a fixed lag after its last exposure; channel shares;
  Δρ_c.
- γ₀ = 10 primary, γ₀ = 1 secondary. n = 8.
- Readings named before numbers, including a null.

Note: a fixed step budget changes the stopping rule, which is itself a core
choice. State it in the choices register as an ablation-only deviation.

---

## 7. Notebooks — one research question per notebook

```
notebooks/
  poc/          four scripts from docs/24 §7 (≤ 60 lines each)
  core/         C1.ipynb  C2.ipynb  C3.ipynb  forward_transfer.ipynb  richness_axis.ipynb
  ablations/    one notebook per docs/24 §5 row
  extensions/   one notebook per docs/24 §6 axis
  archive/      project-overview*.ipynb, results.ipynb
```

Every notebook, in this order:
1. **Question** — one sentence, and its `docs/24` slot.
2. **Pre-commit** — linked and quoted; readings named before numbers.
3. **Setup** — config table printed from the code that runs, not typed.
   Data provenance: commit, clean/dirty, cache digest, n, sampling unit.
4. **Run or load** — either runs the arm in place, or loads a cache and
   asserts its digest and stamps.
5. **Results** — each figure with the stream strip where the x-axis is
   stream position (`docs/24` §8 conventions); a table with every number
   the text quotes.
6. **Reading** — which pre-committed reading fired, verbatim.
7. **Limits** — what this does not license.

Rules: one finding per notebook; no conflating claims. Built from a
`_build_*.py` script so the notebook is reproducible. Under ~2 MB committed:
inline figures at low dpi, publication figures written to `figures/`.

**[Amended 2026-11-06, Kati]** Waiver for `notebooks/core/aba_controlled.ipynb`:
- It has no `_build_*.py`. It comes with `aba_lib.py`, which holds the
  generator, the 1M network, training, NTK and a vendored GLUE copy.
- One notebook cell asserts that the vendored GLUE matches `src.glue.core`
  whenever `src` imports.

---

## 8. Order of work

**[Amended 2026-11-06, Kati]** This order replaces the list below:
1. Fix the forgetting formula in `src`, with a test. (Done 2026-11-06.)
2. The A–B–A notebook in QUICK mode; report the validation results and runtime.
3. Docs amendments.
4. Error registry: fixes plus tests.
5. Choices register.
6. Draft the confirmatory-test specification.
7. Implement measurement on task t+1's manifolds for the stream, then time ONE
   design-scale 1M stream arm with every `docs/26` measurement included.

No stream grid, order or spacing runs until Kati signs `docs/26` and the grid.

1. Error registry and provenance audit (§5). Zero compute.
2. Choices register (§2). Zero compute.
3. Measured richness on all **existing** arms, both axes (§3). Zero compute;
   shows whether the axes agree before anything is rerun.
4. POC scripts (`docs/24` §7), one-module learner.
5. Core rerun: measured arm → cost → Kati signs grid → run (§1, §4).
6. Core notebooks C1–C3, forward transfer, richness axis.
7. `rerun` items not subsumed by the core.
8. Order (§6.1), then spacing (§6.2).

Extensions (`docs/24` §6: learners, EWC/replay, CIFAR/neural) wait until 6 is
done and the method section is drafted. Venue is decided after that.

---

## 9. Will not

A second module; any architecture choice not in the choices register; a
learner, loss or optimiser change inside the core; reading a pooled
correlation; a forward-transfer measure beyond §4; writing into
`results/phase1/`; deleting archived results; reporting a validation as a
finding; starting an extension before §8 step 6.
