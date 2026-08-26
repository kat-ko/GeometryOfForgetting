# 12 — Report against `11-role-and-open-questions.md`

Written 2026-08-13. Answers the thirteen open questions from the record, reports the two items that
came before them, and makes one recommendation with a measured cost. No writing decisions taken.

**Three things in here need a decision before submission, and one of them lands on a sentence the
last round asked to be §5.1's headline. They are collected in "What needs a decision" at the end.**

---

## The boundary

Accepted, and the diagnosis is right. The four-corner/three-corner inconsistency happened because a
document assembled in one place was reasoned about in another, and I contributed to it by writing
argument rather than reporting measurement. What follows is the record and readings close to it;
where a finding bears on what the paper should claim, it is flagged rather than resolved.

One correction to the note itself, in the same spirit. It says the draft "never cites manifold
capacity itself" and attributes the central identity to the paper that applies it rather than the one
that derives it. Both are true of `paper/body.tex` as it stands. I introduced the misattribution: I
added `\citet{chou2026diagnosing}` to §5.1 for "the exact three-factor identity" in the last round.
That was a positioning decision made from inside the repo without the literature in view, which is
exactly the failure mode the note describes.

---

## Before the questions

### The page count — the number everything was waiting on

`tectonic` 0.15.0 installed and working. Harness at `paper/harness/main.tex`, deliberately in a
subdirectory so `scripts/check_submission.py`, which globs `paper/*.tex` non-recursively, cannot
mistake it for the submission. `\documentclass[wcp]{jmlr}`, US letter, class defaults.

| what | pages |
|---|---|
| body text alone, no figures, no references | **5** |
| body text + the two main figures | **8** |
| body + two main figures + references | **9** |
| body text + references, no figures | 6 |

**The limit is 4 pages excluding references and appendices. The paper is at 8. It is exactly twice
its allowance.**

Three details that change what the cut has to be:

- **The overrun is not caption text.** Replacing both captions with a one-line stub leaves the count
  at 8. The figures cost three pages as graphics.
- **The figures are too tall for the format.** `fig2` is 10.5 × 7.6 in (aspect 0.72) and `fig4` is
  15.5 × 8.4 in (0.54). At `\linewidth` in this class each float exceeds the text block; TeX reports
  `Float too large for page by 65.83pt` and gives each one a page of its own. A 2×2 panel grid at
  this aspect cannot share a page with text. This is a re-plot, not a trim.
- **Body text alone is already over.** Even with no figures at all, 5 pages against 4.

So hitting 4 pages with two figures needs the text at roughly 2–2.5 pages and both figures re-laid
out to a wider aspect. I have not cut anything; the ordering of what goes is yours. What I can supply
is which numbers each candidate cut costs, which the inventory now reports directly — see below.

Also found by compiling, which is the only way it could have been found:

- **`%` does not start a comment inside a BibTeX entry.** The three `% TODO-VERIFY` markers I added
  last round silently deleted their entire entries — Chou, Graldi and Menghi were missing from the
  compiled bibliography with no error beyond a BibTeX warning. Fixed by making the marker a real
  `note` field, so an unverified reference is now visible in the typeset bibliography rather than
  hidden in a comment. This is the guards-that-fail-open family again: the marker was there to be
  loud and was in the one position where it was silent.
- **`check_submission.py` could not count pages.** Its counter regexes `/Type /Page` over the raw
  PDF bytes, which returns 0 once the engine packs the page tree into compressed object streams, as
  `xdvipdfmx` does. It reported UNKNOWN rather than wrong, so it was honest, but it could never have
  reported the overrun. Now inflates object streams and cross-checks `/Count`. Verified: 9 and 6
  pages on the two harness outputs.

### The inventory now points at the submitted files

`scripts/make_inventory.py` looks for each pinned fragment in `paper/body.tex`,
`paper/figures.tex` and `paper/figures-appendix.tex` first, and only then in `docs/07-writeup.md`.

Matching had to change to make this work: the fragments were written against markdown and the target
is LaTeX, so both sides are now normalised — symbols expanded (`γ₀` and `$\gamma_0$` compare equal),
LaTeX markup stripped, punctuation and whitespace collapsed. Collapsing whitespace also fixed a
limitation of the old check, which searched line by line and so could not see a quote spanning a
wrapped line. Verified against a real case: `falls by 69.7 floors at γ₀ = 10` now resolves to
`paper/body.tex:95`, where the sentence wraps mid-fragment.

**Result: 75 of 75 numbers verified against their sources; 25 prose-pinned; of those, 8 are in the
submitted paper and 17 only in the long draft.**

That split is now the useful part. A row pinned to `07-writeup.md` is a number that exists in the
draft and *not* in the paper, so the inventory reports directly what a length cut costs and what it
does not. I also split `paper/figures.tex` into the two body floats and
`paper/figures-appendix.tex` for the four supplementary ones, because the page measurement needed
the distinction.

### The GPU premise in the CIFAR spec was wrong

`docs/10-cifar-pilot-spec.md` §3 built its whole recommendation on "there is no GPU", which I
verified on the wrong machine. Corrected in place, with both facts recorded:

- this host is `a8000-2409n4` — `nvidia-smi` reports no driver, 256 cores, 503 GB RAM;
- **GPUs 0–3 are available on `grime`**, a different host.

A learned convolutional encoder is therefore feasible, and the spec now says so. The revised
recommendation is still raw pixels *for the pilot* — on this host, architecture identical to the
synthetic result — with the conv encoder on `grime` as a deliberate second step. The reason is not
cost: it is that moving host, adding `torch` (which the provenance system does not hash), and
introducing a training loop that has never run in this project are three places for a difference to
enter that is not the difference being measured. `results/LOG.md:3370` still states the no-GPU fact
as of that date; it is a dated entry and I have left it, with the correction carried here and in the
spec.

---

## On what exists

### Q1 — is there a single document listing every experiment?

**No, there was not.** `results/LOG.md` is the narrative and is chronological and interleaved with
reasoning; `docs/01-experiments.md` is the registration, not the record of what ran.

Written: **`docs/13-experiment-registry.md`**. Compiled from a full read of the LOG (3414 lines), all
17 `run_*.py` entry points, and the contents of `results/`. Name, purpose, parameters, arm count,
date, results path and whether it fed the paper, for every run — including the eleven validation-tier
runs, the five Phase 0 checks, the killed grid launch, the destroyed pilot, and the five planned
items that never ran.

Two things it surfaces that were not visible in the LOG's chronological form:

- **The validation tier is eleven runs and contributes no quoted number.** It gates numbers that are
  quoted. That is a legitimate structure but it is invisible in the paper, and the registry is the
  artifact that makes it citable in an appendix.
- **`results/scaling_colgen.json` has no named entry point.** It was produced by a modified
  invocation of `run_scaling.py` that is not on disk. It is the one result in the project that is not
  reproducible from a committed script. Nothing in the paper depends on it.

### Q2 — which computed results are in neither the paper nor the notebook?

**Phase 0 is the answer, and it is more complete than "some of it".** Four of the five checks appear
nowhere numerically — not in `paper/body.tex`, not in `docs/07-writeup.md`, not in the notebook:

| Phase 0 check | outcome | where it appears |
|---|---|---|
| 1 — capacity-at-init tracks `a` | α 0.3091 → 0.3940, rank corr +1.00, range 23.97% vs a 3.74% threshold | nowhere |
| 2 — capacity-at-init flat in γ | representations bitwise identical, α = 0.310595 at γ=0.03 and γ=10 | nowhere |
| 3 — richness separation | 2.53 decades in ‖ΔW‖/‖W‖ at matched loss, 39.5× steps | nowhere |
| 4 — `pairwise` vs `full_P` | α agrees to ~1%; D_eff and Ψ_eff diverge ~26% | nowhere |
| 5 — time reparameterisation | DTW residual 11.54 floors, lazy-arm excursion 0.2 floors | §5.5, as "a Phase 0 gate that would have killed H3" — no numbers |

Check 2 is worth a second look. "The representations are bitwise identical across γ at
initialisation, and α = 0.310595 at both ends" is the cleanest statement in the project that the
richness manipulation does nothing before training — which is what licenses reading every later
difference as an effect of training rather than of initialisation. It costs one sentence. The paper
currently asks the reader to take that on trust.

Also unreported anywhere: the eleven validation-tier runs in the registry, and the `run_scaling.py`
and `scaling_colgen` throughput measurements.

### Q3 — what is in `results/` that no script reads?

1476 files, 6 directories. The three arm directories are all live, read via globs through
`grid.load()`: `results/phase1/*.json` (1280), `results/width/*.json` (96),
`results/gamma_ext/*.json` (64).

**Read by current code:** `audit_scope.json`, `audit_propagation.json`, `measurement_null.json`,
`tier2_backward_transfer.json`, `open_questions.json` (new), `glue_core_recovery.json` (by a test),
`timewarp.json` (by `fig_gamma_excursion.py`).

**Orphaned — unreported measurements, not dead weight:** `phase0.json`, `gate_2a.json`,
`psi_eff_diagnostic.json`, `scale_compression.json`, `estimator_followups.json`,
`mode_constancy.json`, `probe_check.json`, `psi_bound_check.json`. These are the validation tier.
Each was read once by a human, decided something, and has had no reader since. The registry now
indexes them, which is the cheapest fix; wiring them into the notebook would be better and is not
worth doing in nine days.

**Orphaned and genuinely dead weight:** eight `.log` files, `phase1_summary_smoke.json`,
`results/figures/gamma_excursion.{pdf,png}` (superseded), `fig_gamma_excursion.csv`,
`phase1_summary.json` (348 KB — nothing reads the full summary; analysis goes through per-arm JSON).

**One orphan was a live risk, now closed.** `results/cost_model.json` holds the registered noise
floors, and those floors are **hardcoded** into `src/analysis/timewarp.py` as `NOISE_FLOOR_CV` rather
than loaded. Every floor-denominated number in the paper divides by one of those four constants, and
nothing connected them to the artifact — regenerating the cost model would not have updated them and
no test compared them. They do all still agree, to the four decimals they are rounded to: α 0.0187 vs
0.018679, `D_eff` 0.0126 vs 0.012647, `R_eff` 0.0050 vs 0.005031, `rho_c_glue` 0.0097 vs 0.009722,
all at the registered n_t=200. `tests/test_noise_floor_provenance.py` now asserts it, so the risk was
drift rather than error and it can no longer happen silently.

---

## On resolution and coverage

Computed by `scripts/audit_open_questions.py` → `results/open_questions.json`. No new training or
measurement; every value re-derived from arms on disk.

### Q4 — which reported numbers sit closest to their resolution limit?

102 pooled per-(corner, richness) quantities across the four channels, each denominated by its own
measured floor — `Ψ_eff` for the utility term, `R_eff` for radius, `D_eff` for dimension, α for the
magnitude. `grid.floors` refuses an unknown channel rather than borrowing a neighbour's floor, which
is what made this table trustworthy to build. **74 of 102 resolve at the project's 2-floor
convention. 19 of those sit within 3× of the gate.**

The tightest — the first is quoted in §5.1; the rest are per-corner channel values that appear in a
figure or in the appendix series:

| quantity | floors | n |
|---|---|---|
| **α, three forgetting corners, γ₀=0.1** | **+2.10** | 120 |
| radius, `S-HH`, γ₀=0.03 | +2.11 | 40 |
| utility, `S-HH`, γ₀=10 | +2.34 | 40 |
| α, `S-LL`, γ₀=0.1 | +2.49 | 40 |
| dimension, `S-HH`, γ₀=0.1 | +2.52 | 40 |
| utility, `S-LL`, γ₀=0.3 | +2.63 | 40 |
| dimension, `S-HL`, γ₀=0.1 | +2.83 | 40 |
| dimension, `S-LH`, γ₀=0.3 | +2.85 | 40 |

**The most fragile number in the paper is the "first resolvable" claim.** §5.1 says forgetting first
becomes resolvable at γ₀=0.1, at 2.10 floors. That is 1.05× the gate. Any change that moves the floor
by 5% — a different measurement-null pooling, a different `k`, one fewer seed — moves the onset from
γ₀=0.1 to γ₀=0.3 and changes a sentence about where forgetting begins. It is also the number a
reviewer is most likely to probe, because it is the one that sets the range.

This does not make it wrong. It makes it the one number whose stated precision should not be
overstated, and it argues for phrasing the onset as "not resolvable below γ₀ ≈ 0.1" rather than as a
located threshold.

### Q5 — where is coverage below 100%, and what is excluded?

Eleven gates. The exclusions are concentrated at the lazy end, which is the benign pattern.

| gate | threshold | applies to | excluded | what goes |
|---|---|---|---|---|
| resolution gate on \|Δlog α\| | 2 floors | 24 (corner, γ) cells at lag 12 | 5 / 24 = **20.8%** | γ₀ ≤ 0.1, where nothing moves either way |
| `R_eff` floor gate, panel (d) | 3σ of the measured `R_eff` floor, per richness | per-arm radius terms, 160 per richness | **γ-dependent, 3.1% – 100%** | arms whose radius barely moved |
| ρ_c calibration domain | tol 0.05 on the fitted range | 36,480 ρ_c values used for conversion | **0.00%** | nothing — no value fell outside |
| arm usability | convergence + artifact checks | 960 / 96 / 64 arms | **0.00%** in all three sets | nothing |

The panel (d) gate per richness, since one number cannot represent it:

| γ₀ | gate on \|Δlog R\| | excluded | coverage |
|---|---|---|---|
| 0.03 | 0.0084 | 160 / 160 | **0%** |
| 0.1 | 0.0092 | 40 / 160 | 75.0% |
| 0.3 | 0.0185 | 5 / 160 | **96.9%** |
| 1 | 0.0241 | 30 / 160 | 81.3% |
| 3 | 0.0214 | 35 / 160 | 78.1% |
| 10 | 0.0336 | 40 / 160 | 75.0% |

Two readings worth having.

**The ρ_c domain gate and the usability gates exclude nothing at all.** Zero of 36,480 ρ_c values fell
outside the calibration domain, and zero of 1120 arms failed a usability check. They are guards that
have never fired — good news about the data, and also a reminder that they are untested in
production: the appendix's point about guards, applied to guards that have not yet had the chance to
fail open.

**The panel (d) gate excludes everything at γ₀=0.03 and between 3% and 25% everywhere else.** So the
panel has one bar with no arms behind it at all and five bars with coverage between 75% and 97%. That
is worth a coverage row in the appendix, and it is a smaller problem than I expected: the gate is not
progressively worse at high richness — coverage at γ₀=10 is the same 75% as at γ₀=0.1, because the
radius term grows roughly as fast as its floor does.

### Q6 — anything pooled that we have not checked for a sign mixture?

The four-corner pooling hid a gain, and that audit covered Δlog α over 520 cells. It had **not** been
run on the three terms separately, on ρ_c, or on the probe margin. Now run on all of them: 51 cells
average values of opposite sign, 38 materially — meaning the minority is at least 10% of the cell and
the mean is within 2 SD of zero.

Most of it is benign and expected: 30 of the 38 are Δρ_c cells, and most sit at γ₀ ≤ 0.1 where
nothing is happening and the sign of a near-zero quantity is arbitrary. Two findings are not benign.

**First: the `S-LL` ρ_c value at γ₀=10 is an exactly balanced sign mixture.** 80 measurements, 40
positive and 40 negative, mean +0.0058 against SD 0.0169. This is the convergence value already
flagged as unresolved. It is now clear *why* it is unresolved: it is not a small effect measured
imprecisely, it is a coin flip. Half the arms converge and half diverge. "Unresolved" is the right
word but "we cannot tell the sign, and the arms disagree evenly" is the more informative one.

**Second, and this is the one that matters: every value quoted for the `S-HH` radius term is below
the gate the same analysis applies elsewhere.** §5.1 states that the radius term is negative from
γ₀=1 onward and quotes −0.0154, −0.0015, −0.0124 at γ₀ = 1, 3, 10. Against the analysis's own
per-richness 3σ `R_eff` gate:

| γ₀ | radius term | the gate | passes? | floors |
|---|---|---|---|---|
| 0.03 | **+0.0059** | 0.0084 | no | +2.11 |
| 0.1 | **+0.0335** | 0.0092 | **yes** | +10.91 |
| 0.3 | **+0.0430** | 0.0185 | **yes** | +6.98 |
| 1 | −0.0154 | 0.0241 | **no** | +1.92 |
| 3 | −0.0015 | 0.0214 | **no** | +0.22 |
| 10 | −0.0124 | 0.0336 | **no** | +1.11 |

All three quoted negatives fail. The γ₀=3 value is a 20/20 sign split with mean/SD = −0.13 — it is
zero. And the only `S-HH` radius values that *do* clear the gate are at γ₀ = 0.1 and 0.3, where the
term is **positive**.

For contrast, in the three forgetting corners the same term at γ₀ ≥ 0.3 clears the gate by 5 to 24
floors (values 0.030–0.174 against gates 0.018–0.034). **The claim is solid in the three corners that
forget and unsupported in the corner that gains.**

This bears directly on the reading the last round asked to be §5.1's sentence — that the radius term
is negative in every corner including the one that gains, so the four corners share one channel and
differ in two. In the three forgetting corners that is well resolved. In `S-HH` it rests on three
numbers that the paper's own gate rejects, and the resolved evidence there points the other way. I
have not changed the sentence; it needs a decision.

It is worth saying what this is an instance of, because it is a fifth member of the appendix's
families and a new one: not a guard that failed open, not a sign destroyed by a transformation, not a
comparison across populations, but **a gate applied in one panel and not to a series quoted in
prose**. Panel (d) applies this exact gate per bar. The channel-signature sentence quotes the
ungated pooled means.

---

## On the specific results

### Q7 — does the lag/position pattern distinguish any candidates?

Yes, and it rules out the one that sounds most natural. Ten (task, lag) cells at γ₀=10, gain in
floors, fitted against each candidate:

| predictor | slope | R² |
|---|---|---|
| task position alone | +0.597 / step | 0.534 |
| lag alone | −0.611 / step | 0.512 |
| **task − lag** | **+0.384 / step** | **0.664** |
| task and lag with free separate slopes | +0.390, −0.377 | 0.664 |
| **boundary (task + lag)** — total training at measurement | +0.047 | **0.003** |
| stream remaining after the task (15 − task) | −0.597 | 0.534 |

Three things follow.

**The gain has nothing to do with how much training has happened when you measure.** Boundary index
explains 0.3% of the variance. Cells at boundary 12 range from +6.16 to +12.75 floors depending on
how that boundary is reached. Whatever the gain is about, it is not cumulative exposure.

**Task position and lag act with equal and opposite strength, and a single contrast captures both.**
`task − lag` with one free slope reaches R²=0.664, exactly what letting the two vary independently
achieves. So the data do not distinguish "two effects" from "one effect of a single contrast", and
the more economical description is that the gain depends on how far into the stream a task sits
relative to how long ago it was learned.

**On the candidate in the question — "how much stream remains after the task" — the data say the
opposite of the intuitive direction.** More stream remaining means *less* gain (slope −0.597). If the
gain came from subsequent similar training continuing to help an earlier task, more remaining stream
should mean more gain.

Two constraints on all of this. Ten cells, and task and lag are correlated at −0.575 in the sampled
cells with 4 of 10 at boundary 15, so the design cannot fully separate them. And **the pattern is
specific to γ₀=10**: at γ₀=1 the lag slope is *positive* (+0.274, R²=0.369) and at γ₀=3 nothing
explains much (best R²=0.173). The quoted series +8.7, +9.0, +6.2, +3.1 is a γ₀=10 series and the
decay it shows does not appear at lower richness. Any sentence about the gain decaying with lag needs
"at high richness" in it.

### Q8 — is the between/within split general?

**It is much more general than one instance, and that turns out to be an argument against the
reading rather than for it.** Seven measures, 28 pairs, between-richness rank correlation of the
per-level means against the mean within-level correlation.

Eight pairs reverse sign strongly:

| pair | between | within |
|---|---|---|
| generic α vs retained α | **+1.000** | −0.283 |
| generic `R_eff` vs retained α | **−1.000** | +0.277 |
| generic ρ_c vs retained α | **−1.000** | +0.247 |
| generic `D_eff` vs retained α | **−1.000** | +0.159 |
| forgetting CFr vs generic α | −0.771 | +0.462 |
| forgetting CFr vs generic ρ_c | +0.771 | −0.431 |
| forgetting CFr vs generic `R_eff` | +0.771 | −0.431 |
| forgetting CFr vs generic `D_eff` | +0.771 | −0.329 |

And the reported instance — generic capacity against probe margin — is **weaker than all eight**:
between −0.429, within +0.180.

The honest reading: richness moves nearly every measure monotonically, so between-level correlations
are driven to ±1 by that single common cause, while within-level correlations reflect arm-to-arm
variation at fixed richness. A reversal between the two is therefore close to the *default* in this
design, not a discovery. It is Simpson's paradox with richness as the confound, and it will appear
for almost any pair of measures that richness affects.

So the answer to "if it is general, that is a stronger claim than one instance" is: it is general, and
generality makes it a weaker claim about capacity and the probe margin specifically. The finding is
real and the mechanism is worth one sentence; what it cannot support is the implication that
something notable is true of *that pair*. Six between-level points also cannot carry a correlation of
−0.429.

### Q9 — is the γ=30 saturation specific to ρ_c?

**Yes. ρ_c is the only quantity that saturates; everything else accelerates slightly.**

| quantity | γ=3 | γ=10 | γ=30 | step 3→10 | step 10→30 | ratio |
|---|---|---|---|---|---|---|
| generic α | 0.3404 | 0.3750 | 0.4159 | +0.0347 | +0.0409 | 1.18 |
| retained α | 0.7261 | 1.0519 | 1.4175 | +0.3258 | +0.3656 | 1.12 |
| generic `D_eff` | 4.2612 | 3.8511 | 3.3181 | −0.4101 | −0.5331 | 1.30 |
| generic `R_eff` | 0.8475 | 0.8245 | 0.7982 | −0.0230 | −0.0263 | 1.14 |
| **generic ρ_c** | 0.4464 | 0.4021 | 0.3794 | −0.0443 | **−0.0227** | **0.51** |

Every other quantity's second step is 1.1–1.3× its first. ρ_c's is 0.51×. So the saturation that
breaks additivity at γ₀=30 is a property of the center correlation and not of the richness range —
capacity, dimension and radius are all still moving at their earlier rate or faster. That makes the
additivity failure a specific fact about ρ_c rather than a general edge-of-sweep effect, which is the
stronger and more interesting version.

### Q10 — is there anything between γ=3 and γ=10?

**No. Nothing in the record lies strictly between them.** Every richness value that exists anywhere:

- registered grid: 0.03, 0.1, 0.3, 1, 3, 10
- width arms: 0.03, 1, 10
- γ=30 probe: 30

The `S-HL` decorrelation onset is therefore located to one grid step spanning a factor of 3.3 in
richness, and no re-analysis can narrow it. This is the one gap in the record that new compute could
close, and it is cheap — see Q11.

---

## On whether to run more

### Q11 — is there a cheap measurement that would materially strengthen a claim?

**Yes, exactly one: a richness probe between γ₀=3 and γ₀=10.**

Cost, from the measured γ=30 probe, which is the same shape: 64 arms (4 conditions × 4 streams × 4
seeds), 741 s wall on 254 workers, 10.2 core-hours, 571 s/arm. A single γ₀=5 probe is
**about 12–13 minutes wall**. Two probes at γ₀=5 and 6 are about 25 minutes. It reuses
`run_gamma_ext.py` unchanged apart from the richness value, writes to a separate directory, and
touches no existing artifact.

What it buys: the `S-HL` decorrelation is absent at γ₀=3, present at γ₀=10 and stronger at 30. Right
now the onset is bracketed by a factor of 3.3 and the paper can only say "between 3 and 10". One probe
halves that bracket and would let the claim be stated as a located onset. It is the only place where
under an hour of compute changes what a sentence can say.

Two candidates I checked and reject:

- **More arms will not rescue the `S-HH` radius term.** The floor is estimator dispersion measured on
  a fixed representation, not sampling error, so adding arms shrinks the confidence interval around a
  mean that stays where it is relative to the floor. The Q6 finding cannot be fixed by compute; it
  has to be restated.
- **More seeds will not firm up the γ₀=0.1 onset** for the same reason. 2.10 floors is 2.10 floors at
  any n.

### Q12 — anything I would want measured that we have not discussed?

Three, in order of how much they bother me. None is a nine-day item; they are for September.

**The `S-HH` radius term, measured rather than inferred.** This is the Q6 finding and it is the one
that has been nagging since Tier 2.1 — the channel signature was reported as a shape (utility and
dimension positive, radius negative) and I quantified it without checking it against the gate that
the same quantity passes through in panel (d). A measurement-null run at higher measurement-seed
count *specifically for `R_eff` at γ₀ ∈ {1,3,10}* would establish whether the `S-HH` radius floor is
genuinely that large or whether the W4 null (one arm per γ, four measurement seeds) is itself too
noisy to denominate this term. That is the honest way to find out whether the sentence can be
recovered, and it is a real experiment rather than a re-pool.

**W6, the rotation measure for H6.** Flagged in the LOG at 2179–2183 as never stored. §5.4 is written
about center geometry and H6 is a rotation hypothesis; the quantity that would test it directly was
never saved. This is the only registered hypothesis with no measurement behind it.

**A second stream realisation at fixed seed.** Stream variance is 0.000 and seed variance 0.007 in
the three-corner decomposition, which is the reproducibility result §B now leads with. Both are
estimated within the existing five streams. One more stream would tell us whether 0.000 is a property
of the design or of these five streams, and it is the cheapest possible check on the paper's
methods claim.

### Q13 — is any reported result thin enough to cut rather than defend?

The page count changes the tenor of this question: at 8 pages against 4, things are going regardless.
Ranked by what I would give up first.

**Cut the center-collapse panel (d).** It is post-hoc calibrated, and its γ₀=0.03 bar has **no arms
behind it at all** — the 3σ gate excludes all 160. The other five bars are better than I expected at
75–97%, so the case here is weaker than I first read it: this is one empty bar and a calibration
domain to explain, not a systematically thin panel. It is also the panel whose top step was already
identified as over-read. If a page has to go, this costs one panel and removes a gate and a
calibration domain from the appendix. If pages were free I would keep it with a coverage row.

**Cut the width arm from the body, keep it in the appendix.** It bounds an effect rather than
measuring one, and the honest phrasing is already "no width dependence is detectable, with 1.5%
bounding one rather than measuring one". Worse, all twelve Δρ_c width cells are 4-arm with ±0.02
noise against effects of 0.011–0.110, and the matched-subset-versus-full-grid gap is 4.5% at γ₀=10 —
three times the spread it is supposed to bound. It is 34.5 core-hours and the second-largest
compute item in the project, which is precisely why it is tempting to keep; that is the sunk-cost
argument and it should not win. One appendix sentence preserves it.

**Do not cut the γ=30 probe.** It is 10 core-hours, it is the only extension beyond the registered
range, and Q9 has just made it more valuable, not less: it now supports a specific claim about ρ_c
saturating while nothing else does.

---

## What needs a decision

Three items, all outside my remit, in order of consequence.

1. **The `S-HH` radius term (Q6).** Every value §5.1 quotes for it fails the analysis's own gate, and
   the resolved values at low richness have the opposite sign. The claim holds in the three
   forgetting corners. Options as I see them: restrict the shared-channel sentence to the corners
   that forget; state the `S-HH` radius term as unresolved and drop the numbers; or run the
   `R_eff`-specific measurement null in Q12 first. I have changed nothing pending your call.
2. **The length cut.** 8 pages against 4, body text alone over at 5, and the figures need re-plotting
   to a wider aspect rather than trimming. The inventory now reports which numbers live only in the
   long draft, so the cost of each candidate cut is available on request.
3. **The between/within reading (Q8).** It is general across eight pairs, more strongly than in the
   reported pair, which weakens the specific claim about capacity and the probe margin. The mechanism
   is worth a sentence; the current framing implies something notable about that pair specifically.

Two smaller things I did unilaterally because they are record-keeping rather than argument, flagged
in case you disagree: the bibliography `TODO-VERIFY` markers are now `note` fields that print, and
`paper/figures.tex` is split so the four supplementary floats sit in `paper/figures-appendix.tex`.

Verification state: **161 tests pass**, lint clean, inventory 75/75 with 25 pinned, scope audit clean,
figure manifest verified. New this round: `scripts/audit_open_questions.py` →
`results/open_questions.json`, `docs/13-experiment-registry.md`,
`tests/test_noise_floor_provenance.py`, and a working TeX toolchain.
