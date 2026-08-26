# Write-up-ready prose

Paper-facing text whose numbers are pinned to specific runs. Everything here is traceable to
`results/LOG.md`; nothing here is a new claim. When a number changes, it changes here too, or
this file becomes the stale copy that gets published.

Three parts: the §5 draft sections, the figure captions, and the appendix material. Every number
in here appears in `docs/08-inventory.md` with the file it comes from.

---

# Part 1 — §5 draft sections

## §5.1 Forgetting decomposes exactly, and the channel carrying it shifts with richness

Retained capacity is the capacity of the representation *for a specific past task's labels*, and
it obeys an exact identity in three factors — a utility term Ψ_eff, a radius term
(1 + R_eff⁻²), and an effective dimension D_eff, due to Chou et al. (2026) — so a change in it can
be attributed without a model of what forgetting is:

    Δ log α = Δ log Ψ_eff + Δ log(1 + R_eff⁻²) − Δ log D_eff

The identity closes to 4.4 × 10⁻¹⁶ across 48,640 evaluations, so the attribution has no residual
to argue about. What varies is which term carries the change.

**Forgetting appears between γ₀ = 0.1 and 0.3 and grows by two orders of magnitude.** Richness is
the feature-learning strength γ₀ in the parameterization of Graldi et al. (2025), swept over six
levels from 0.03 to 10. Measured for task 0 twelve task boundaries after it was learned, retained
capacity is unchanged in the lazy regime (0.2 measurement floors at γ₀ = 0.03, i.e. nothing) and
falls by 69.7 floors at γ₀ = 10.
Quoting this in units of the estimator's own Monte-Carlo dispersion is what makes "unchanged"
a measurement rather than a small number: the floor was measured directly, at every richness, by
re-measuring one trained representation under several measurement seeds, and for α it is flat in
γ (CV 1.49–2.03% against the registered 1.87%).

**The channel changes character across the sweep.** Over the range where forgetting is resolvable
at all — γ₀ = 0.1 to 10, the weakest level sitting just above the two-floor gate at 2.1 — the
utility term grows from 0.08 to 0.45 of total motion while the radius term falls from 0.39 to 0.11.
The dimension term falls from 0.53 to 0.44 across the first step and is then flat, so from γ₀ = 0.3
onward the reorganization is a straight exchange between utility and radius. So lazy-regime and
rich-regime forgetting are not the same
process at different intensities. In the rich regime capacity is lost mostly because the
representation's *alignment with the task's labels* degrades, and in the intermediate regime it is
lost mostly because manifolds inflate relative to their separation. Read the signs alongside the
shares: for γ₀ ≥ 1 the utility term is negative on **100%** of arms, so the growth of its share is
not a sign ambiguity being resolved.

**Part of the radius channel is centers collapsing rather than manifolds inflating.** Converting
the measured change in center correlation ρ_c through a synthetic-manifold calibration and taking
it as a fraction of the observed radius change, the center-attributable share rises from 0.05 to
0.44 through γ₀ = 3 and then flattens; the γ₀ = 3 → 10 step is not resolved (bootstrap CI
[−0.038, +0.070]), so the honest reading is a steep rise followed by a plateau. This is the same
radius–center duality that the manifold-capacity literature describes, appearing here as a
decomposition of forgetting. It is a mechanistic detail rather than a load-bearing claim, and it
is gated conservatively: each richness level uses three times *its own* measured R_eff floor, at
which the three forgetting conditions retain 100% of their cells for γ₀ ≥ 1.

**The corner that does not forget.** These numbers pool three of the four corners of Hiratani's
(2024) feature × readout similarity design — the three that lose retained capacity. The fourth,
`S-HH` — high feature *and* high readout similarity, pre-registered as benign — gains it: Δ log α is positive at every richness, largest of the six sampled richnesses at
γ₀ = 1 with 11.8 floors and falling to 6.2 by γ₀ = 10. Its behavioral forgetting is zero to within
measurement (−0.0005 to +0.0001). This is **a property of that corner and not a tendency present
elsewhere in weaker form**: across 520 cells of the design — every combination of richness, stream
condition, width, task position and lag in the registered grid, the width arms and the γ₀ = 30
probe — every resolvable capacity gain is an `S-HH` cell, and the largest gain anywhere outside it
reaches 0.74 floors, a third of the resolution gate.

**The gain holds at every task position and every lag the design can resolve.** All ten (task, lag)
cells of the benign corner show a resolvable gain, at each of γ₀ = 1, 3 and 10 — thirty cells, no
exceptions. Because lag and task position are confounded by construction, we read each from its own
matched series and pool neither: at matched lag 4 the gain is +8.7, +12.6 and +12.8 floors for tasks
0, 4 and 8 at γ₀ = 10, and within task 0 it runs +8.7, +9.0, +6.2 and +3.1 floors at lags 4, 8, 12
and 15. So this is backward transfer rather than a privilege of the first task — if anything the
first task benefits least — and it weakens with elapsed time rather than accumulating.

**Two channels carry the gain; the third is too small to sign.** In the three corners that forget,
the radius term is large and well resolved for γ₀ ≥ 0.3 — 5.0 to 24.4 floors, reaching −0.131 to
−0.157 at γ₀ = 10 — so there the growth of `R_eff` demonstrably works against retained capacity. In
`S-HH` it is **not resolved at any γ₀ ≥ 1**: −0.015, −0.002 and −0.012 at γ₀ = 1, 3 and 10, which
are 1.9, 0.2 and 1.1 floors against the two-floor convention applied everywhere else in this
document. We therefore claim no sign for it, only a bound — under 2 floors, where the three
forgetting corners carry 11.7 to 14.1, so at most a sixth of theirs.

**An earlier draft of this section said the opposite.** It read "the radius term is negative in every
corner including the one that gains", and quoted the three `S-HH` values as if they were
measurements. They are all below the gate that the same analysis applies to per-arm radius terms in
Figure 2d, and the only `S-HH` radius values that *do* clear it are at γ₀ = 0.1 and 0.3, where the
term is **positive** (+0.033 and +0.043, at 10.9 and 7.0 floors). The claim was solid in the three
corners that forget and unsupported in the corner that gains. §A.3 records the error family.

What is resolved is the other two channels, and both favour the gain: utility and dimension are
negative in the three forgetting corners — the representation loses alignment with the old task's
labels and gains dimension — and positive in `S-HH`, at +2.3 and +5.7 floors at γ₀ = 10. As signed
fractions of the net change at γ₀ = 10, the gain is +0.48 utility and +0.62 dimension against −0.11
radius that the floor does not resolve; the pooled loss is +0.45, +0.44 and +0.11, with all three
resolved.

So a corner that gains retained capacity does so by aligning with the task's labels and shedding
dimension, and the radius cost that dominates the losses is absent from it to within resolution.
**Gain and loss are not one process with two signs**: the gain is carried by two channels and the
third is bounded, not shared. This is the argument for decomposing rather than measuring magnitude —
a magnitude-only account would render them the same quantity with opposite signs and could not see
that they differ in which channels carry them. Pooling them does worse still, and §A.1 records what
it cost us before we separated them.

**Where the gain is largest depends on when you look, so we do not locate it.** At lag 12 the peak
is sharply determined: interpolating in log γ₀ puts it at 1.23 with a paired-bootstrap interval of
[1.12, 1.35] at unique n=8, a sixth of a grid step. That precision is real and it is also beside the point,
because the peak **moves monotonically with lag** — 2.86, 1.75, 1.23 and 0.96 at lags 4, 8, 12
and 15, nearly a full grid step from end to end. There is therefore no richness at which backward
transfer is strongest; there is one per lag, and the question is ill-posed without fixing the lag
first. We report the shape — the gain rises from the lazy regime, peaks in the middle of the sampled
range, and decays — and make no claim about the location of a maximum.

## §5.2 Both capacity measures rise with richness, and two label-agnostic diagnostics disagree

**The registered prediction was a crossing, and there is no crossing.** We predicted that generic
capacity — capacity for arbitrary dichotomies, independent of any task's labels — would *fall*
with richness while retained capacity rose, so that their product would peak at an interior γ
matching the richness that minimizes final error. Generic capacity rises instead: 0.3043 to 0.3750
from γ₀ = 0.03 to 10, monotone, and resolvable at 11.3 noise floors. Retained capacity rises far
faster, 0.3129 to 1.3710, a factor of 4.4. Retained exceeds generic at *every* richness including
the laziest (+0.0086), with the gap widening monotonically to +0.9960. The object the figure was
designed around does not exist.

Two consequences follow, and they are different in kind. H2a is refuted in sign rather than killed
by its own criterion, which named "non-monotone or flat" and does not cover a clean monotone trend
in the opposite direction. H2c is not independently testable at all: its statistic, the argmax of
the capacity product, presupposes an interior maximum, which requires generic capacity to fall. It
sits at the grid edge by construction, two steps from the measured error minimum at γ₀ = 1
(interpolated 1.61). §5.5 records both as *prediction wrong* rather than *kill fired*.

**The rise is not uniform across the design, and one corner behaves as predicted.** `S-LH` is the
only condition whose generic capacity falls with richness (0.3038 → 0.2898); `S-HL`, the predicted
catastrophic corner, rises most (0.4507). So the pooled failure is not an average concealing a
uniform effect — the sign depends on the stream's similarity structure.

**The sharper result is a between/within reversal, and it is general.** Taking the seven measures on
disk pairwise — generic α, retained α, generic `Psi_eff`, `D_eff`, `R_eff`, `rho_c`, the probe margin
and behavioural forgetting — the rank correlation computed *across* richness levels reverses sign
against the mean correlation *within* a level for **8 of 28 pairs**. The strongest reversals are not
subtle:

| pair | between | within |
|---|---|---|
| generic α vs retained α | **+1.000** | −0.283 |
| generic `R_eff` vs retained α | **−1.000** | +0.277 |
| generic `rho_c` vs retained α | **−1.000** | +0.247 |
| generic `D_eff` vs retained α | **−1.000** | +0.159 |
| forgetting CFr vs generic α | −0.771 | +0.462 |
| forgetting CFr vs generic `rho_c` | +0.771 | −0.431 |
| forgetting CFr vs generic `R_eff` | +0.771 | −0.431 |
| forgetting CFr vs generic `D_eff` | +0.771 | −0.329 |

The mechanism is that richness moves nearly every measure monotonically, so that single common cause
drives the between-level correlation to ±1 while the within-level correlation reflects arm-to-arm
variation at fixed richness. **This is Simpson's paradox with richness as the confound**, and in this
design a reversal is closer to the default than to a discovery. Six between-level points cannot carry
a correlation on their own either.

**One instance is worth stating, and it is one instance rather than the finding.** Generic GLUE
capacity rises by 11.3 floors across the sweep. The margin achieved by a refit linear readout on a
random dichotomy — the probe protocol of Johnston & Fusi (2023), which is the measure H2d registered
— *falls* over the same range (0.0673 at γ₀ = 1 to 0.0531 at γ₀ = 10). Yet within a fixed richness
the two correlate positively (+0.322 at γ₀ = 10). Both are standard label-agnostic diagnostics of
representation quality and they disagree about the direction of the effect, which is the practically
useful form of the reversal — but the pair is **weaker than all eight above**, at between −0.429
against a within-level mean of +0.180, and an earlier draft of this section implied something notable
was true of this pair specifically. What generalises is the mechanism.

The caution that survives: **comparing two representation-quality measures across a hyperparameter
that moves both can invert the conclusion drawn at fixed hyperparameter.** That is a comparison the
literature makes routinely, and it does not require the two measures to be this pair.

An appendix probe one log step beyond the registered range (γ₀ = 30) finds both capacities still
rising, both steps resolved, and retained still above generic with the ratio still growing
(2.81 → 3.41). Nothing in this section turns on the range boundary.

## §5.3 What the result survives, and what it is measured against

**Width.** At genuine n=40, a 4× change in load (N = 150, 600 at γ₀ = 10; N=300 is the registered
grid at unique n=8 and is not in this contrast) moves three-corner forgetting by 0.62% (71.6 vs
72.0 floors) and the alignment share by 0.020. Two widths, not a located dependence. The γ
reorganisation holds at both widths, as does S-HH as a gain. The single width dependence worth
reporting is internal to the two non-alignment channels: the radius share falls with width
(0.146 → 0.098) while dimension rises (0.394 → 0.463), with their **sum moving far less**
(0.540 → 0.560). So width moves the radius/dimension boundary, not the alignment/non-alignment
division. The earlier 1.5%/4.5% bound was the duplicate-stream n=12 contrast; it is superseded.

**Lag and task position.** Forgetting grows with elapsed lag, but in a 16-task stream long lags are
available only to early tasks, so lag and task position are confounded by construction. Measured
separately, position is the larger effect: within task 0, forgetting grows 1.29× from lag 4 to 15,
while at fixed lag 4 it grows 1.56× from task 8 to task 0. The main decomposition is drawn at lag
12, which exists only for task 0 and is therefore clean — at the cost of reporting the
most-forgotten task rather than the grid average, which its caption says. The *composition* is
insensitive to this choice: shares agree to within 0.032 at every richness between lag 12 and lag
4, against a richness-driven swing of 0.37 in the utility share.

**What does not vary.** Regressing per-arm forgetting on candidate sources at γ₀ = 10, the
initialization a run was trained from explains none of it (R² = 0.007). Stream identity was stored
in the filename and never consumed, so the quoted stream R² = 0.000 is tautological on that grid.
The crossed arrangement×init run measures stream R² = 0.001. What does explain forgetting are two comparable sources — how far the task sat above
the common asymptote when it was left, with its elapsed lag (0.589), and which corner of the similarity
design the stream occupies (0.523) — which together reach 0.882. §B gives the table and states why
we do not rank the two.

**Everything above is denominated in a measured floor.** Each geometric channel's Monte-Carlo
dispersion was measured directly at every richness by re-measuring a fixed trained representation
under several measurement seeds. For α and D_eff that dispersion is flat in γ and matches the
registered values, so every magnitude in §5.1 and §5.3 stands as quoted. For R_eff it is not flat,
which mattered for one gate and is recorded in §A.2.

## §5.4 Center correlation moves in opposite directions across the similarity design

The two pre-registered predictions about center geometry both fail, and what replaces them is a
more specific claim than either.

**Both predictions fail.** H1d predicted that centers *decorrelate* over a task stream, more
quickly in rich arms, following the progressive decorrelation reported in human sequential
practice by Menghi et al. (2025). Three of the four conditions do the opposite: centers converge. H6 predicted that the
largest center movement would occur in the catastrophic corner `S-HL`; the largest is in `S-LH`
(+0.110), twice `S-HL`'s magnitude (−0.055) and in the opposite direction. All four conditions
begin at the same place (ρ_c = 0.391–0.392), so the split is produced by training rather than by
initialization.

**What replaces them.** Read as a 2×2 rather than as a single ordering, the design separates
cleanly: readout similarity drives center *convergence* (+0.104) and feature similarity drives
center *decorrelation* (−0.061), with an interaction at the resolution limit (+0.006) and an
additive model predicting the held-out corner to within 0.012. Richness sets the gain on both
effects at a roughly constant ratio; the design sets the sign. Both main effects survive the same
4× change in load as §5.3, with the corner ordering identical at all three widths.

**One corner decorrelates, and it does so progressively.** `S-HL` — the same stimuli under
different rules — is the only condition whose centers pull apart, and the movement is spread over
the stream rather than concentrated at its end: the per-arm rank correlation between ρ_c and block
index is negative on 8 of 8 unique seeds (p = 0.0078, the floor of a two-sided sign test at n=8;
40 files are copies), median −0.79. That progressive form is what makes the
correspondence with the Menghi et al. practice result a correspondence and not a coincidence of
endpoints.

**Two limits, stated with the claims they bound.** The decorrelation is absent at γ₀ = 3
(+0.003, 4 of 8 unique, p = 1.0) and present at γ₀ = 10 (−0.055 on 8 of 8 unique), so the
registered grid brackets the onset by a factor of 3.3. The crossed K=3 run settles the sign and
qualifies the size: all 40 initialisations decline in each of the three arrangements, while the
magnitude runs from −0.026 to −0.051 across them, so the claim is the corner ordering rather than
−0.055 as a population value.

**The γ₀ = 5 probe does not narrow the bracket, and the reason is the sampling unit.** The
RNG-fixed set is 5 arrangements × 8 initialisations (the duplicate-stream 16-file predecessor was
3 of 4 unique, p = 0.625). As an inference about *new arms* it is decisive: −0.0102 ± 0.0032,
declining on 31 of 40, sign-test p = 6.8×10⁻⁴. But the onset is a claim about γ₀, hence about new
arrangements, and re-pooled by arrangement the means are −0.0114, −0.0169, **+0.0084**, −0.0419,
**+0.0109** — three of five, with the two positives landing on the lazy-arm drift band
[+0.005, +0.010], where an arm that barely moved is indistinguishable from one that converged.
The SEM over arrangements is 0.0096 against 0.0032 over arms, and the 95% t interval over
arrangements is [−0.037, +0.016], which includes zero. A sign test on five units has a floor of
p = 0.0625, so this design could not have cleared α = 0.05 whatever it found. By initialisation the
effect is 8 of 8 (p = 0.0078): secure against `W(0)`, not established across arrangements.

**The 8-arrangement pre-commit** (`results/gamma5_k8_precommit.json`, written before any of those
256 arms existed; 256/256 usable; arrangements 0–4 reproduce the 5×8 set bitwise). Primary
statistic: 95% t interval on the eight arrangement means of Δρ_c. Decision: ≥7/8 negative *and*
the interval excludes zero ⇒ the onset sentence stands at arrangement level; ≥7/8 but the
interval includes zero ⇒ suggestive only; fewer than 7/8 ⇒ arrangement-dependent, keep the
3 → 10 bracket. Outcome: **4/8 negative**, means −0.0114, −0.0169, +0.0084, −0.0419, +0.0109,
+0.0015, +0.0166, −0.0221; 95% interval **[−0.023, +0.010] includes zero**; sign-test p = 1.0
against a ceiling of 0.0078. Verdict: **arrangement_dependent**. §5.4 quotes four of eight and
the 3 → 10 bracket. The arm-level p = 6.8×10⁻⁴ is not the onset evidence.

The appendix probe at
γ₀ = 30 finds it stronger as a point estimate (−0.096 at unique n=4, 4/4 declining, sign-test
p = 0.125 — n=4 cannot reject), which is why we describe it
as the crossing of two smoothly growing effects rather than a threshold. Within the registered
range the interaction is unresolved ($+0.006$ at 0.82 unique SEM). The $\gamma_0 = 30$ probe's
interaction is larger as a point estimate ($+0.016$) but at unique $n=4$ it is 1.74 SEM and
does not resolve — the 3.9 SEM figure was 16 files / 4 unique copies, and a paired-seed
reading is 1.53 SEM with a bootstrap CI that includes 0. The range-bounded additivity claim
is not licensed at unique $n$. Separately, `S-LL` does not move resolvably (+0.0114 against a lazy-arm drift band of
+0.005 to +0.010, unique sign test p = 0.73), so its rank in any corner ordering carries no information.

## §5.5 Pre-registration outcomes

Every hypothesis was registered with a measure and a disconfirming observation before any arm of
the reported grid was run. We report all of them, including the three that were never tested,
because a register from which the untested entries have been dropped is not a register.

| id | registered prediction | outcome | kill fired? |
|---|---|---|---|
| H1 | forgetting attributes to different geometric channels in rich and lazy training | **supported, and sharpened**: 150× apart in magnitude and qualitatively different in composition | no |
| H1a | rich forgetting is radius- and utility-accounted; radius/utility share higher in rich than lazy | **passes its statistic, fails its mechanism** — see below | no |
| H1b | lazy forgetting is ρ_c-accounted, more so than rich | **untestable**: at γ₀ = 0.03 forgetting is 0.2 floors and ρ_c attribution is available in 0 of 120 arms | n/a |
| H1c | forgetting reproduces the ultra-rich OOD sign pattern of Chou et al. | **reframed**: same currency, different reference point — our Ψ_eff decays from a task-specific peak *above* the random-dichotomy normalization | n/a |
| H1d | task centers progressively decorrelate over the stream, faster in rich (`ρ_c` signed, decreasing) | **refuted**: three corners *converge*; only `S-HL` decorrelates | **yes** ("flat or increasing") |
| H2a | generic capacity falls with richness | **refuted in sign**: it rises monotonically | no — the kill named "non-monotone or flat" |
| H2b | retained capacity rises with richness | **supported**: monotone, 4.4× over the sweep | no |
| H2c | the generic/retained crossing locates the behavioural optimum | **failed, and the statistic is ill-posed**: there is no crossing anywhere in the grid | see below |
| H2d | generic capacity correlates with a probe measure of label-validity | **supported at γ₀ ≥ 1, null at γ₀ ≤ 0.3**; the pooled correlation is negative and is a between-richness confound | no |
| H3a, H3b | division of labour between modules; reliance shifts with similarity | **not run** — the Phase 0 gate passed, so H3 is testable, but Phase 2 was cut | n/a |
| H4 | the rich/lazy separation erodes over a long stream | **not run** — Phase 3 cut | n/a |
| H5 | pair-mean γ does not explain heterogeneous-pair outcomes | **not run** — Phase 2 cut | n/a |
| H6 | the catastrophic corner `S-HL` shows the largest center movement | **refuted**: the largest \|Δρ_c\| is at `S-LH`, about twice `S-HL`'s | **yes** ("max elsewhere") |

Of the nine hypotheses that were tested, four failed. **Three of the four have replacements we
consider better than the original prediction**, which is stated in §5.2 and §5.4 rather than
claimed here.

**A prediction can be wrong without its kill criterion firing, and H2a is the clean case.** We
predicted generic capacity would fall with richness and wrote the disconfirming observation as
"non-monotone or flat". It is monotone and it is not flat: it rises, cleanly, by every measure. The
criterion was written to catch a prediction that dissolved, and the prediction failed instead by
being exactly reversed. We record this as *prediction wrong, criterion did not fire*, because
collapsing the two would let a criterion that never engaged take credit for the refutation.

**H2c failed in a way that its criterion could not express.** The registered statistic — the argmax
of the product of generic and retained capacity — presupposes an interior maximum, which exists only
if one of the two curves falls. Once H2a failed, the product became monotone and its argmax sat at
the grid edge by construction. The strict criterion (separated by more than one grid step) does fire,
at two steps, but reporting that as a refutation would overstate what was learned: the statistic
stopped being defined before it was evaluated. Separately, the behavioural optimum γ\* is **not
identified by this grid** — bootstrap CI [0.339, 4.426], spanning more than a decade — so no
hypothesis of the form "X coincides with γ\*" is testable here, and we make no such claim anywhere in
this paper.

**H1a passes on a statistic that turns out to mean less than its name.** The registered measure is
the share of the change carried by the radius and utility channels together, predicted higher in rich
than lazy. It is: 0.470 at the lowest resolvable richness against 0.563 at the highest. But there are
three channels and the shares sum to one, so "radius plus utility" is exactly "one minus dimension" —
the registered test reduces to *the dimension share falls with richness*, which it does, by 0.09.
Meanwhile the mechanism the hypothesis named is inverted: we predicted radius would carry rich
forgetting, and the radius share **falls** from 0.387 to 0.114 while utility rises from 0.083 to
0.449. The interesting half of H1a — that utility carries rich forgetting — is confirmed and large;
the half that named radius is wrong; and the statistic as registered cannot distinguish the two. The
lesson we take is that a share of a sum-to-one decomposition should be registered per channel, not
pooled across channels.

**H1b could not be tested, and saying so is not a hedge.** It predicted that lazy forgetting is
accounted for by center movement. In the lazy regime there is no forgetting to account for: 0.2 noise
floors at γ₀ = 0.03, below the two-floor gate. The instrument agrees — the ρ_c-to-radius attribution
requires a resolvable radius change, and it is available in 0 of 120 arms at γ₀ = 0.03, 58% at 0.1,
and 100% from γ₀ = 1. Its coverage rises exactly where the hypothesis predicted the effect would be
weakest. We report H1b as untestable on this design rather than as null, because a null would imply
we looked and saw nothing, and what happened is that there was nothing to look at.

**Criteria that were set and did not fire.** Beyond H2a: the Phase 1 fallback (that GLUE attributions
would fail to separate by regime — they separate), the Phase 0 gate that would have killed H3 (the
trajectories do not coincide, at 11.5 and 7.1 noise floors), the capacity-at-init gate (passed
exactly, bitwise identical across richness), and the H2d pooled criterion, which *would* have fired
on the pooled statistic and does not once stratified — the one case where we declined a criterion's
verdict, with the confound identified and documented in §5.2 rather than asserted.

---

# Part 2 — Appendix material and figure captions

## A. Appendix subsection: aggregation artifacts, and guards that failed open

Several of our own numbers were wrong before they were right, and they fall into six families:
quantities averaged over the wrong grouping, guards that admitted values they were built to refuse,
transformations that destroyed the sign of a signed quantity, comparisons between two quantities each
correctly computed over a different population, a resolution gate applied in the panel that
displays a quantity but not to the same quantity quoted in prose, and a field stored in a filename
and never read. We report them because the
alternative — reporting only the surviving numbers — gives a reader no way to judge how carefully the
rest were checked. The families need separating rather than listing, because they fail differently
and are caught differently: the first by re-grouping, the second by testing a guard away from the
setting that motivated it, the third by nothing downstream at all, the fourth only by asking of each
number in a comparison what population it was computed over, the fifth only by re-deriving every
quoted quantity in units of its own floor, and the sixth only by asking whether a filename field is
an argument to anything.

The fifth family changed a claim's direction rather than its magnitude; §A.5 has it. The sixth
changed the $n$ rather than the files; §A.6 has it. Six families is a property of how carefully the
record was kept --- the checks were built as the work went --- not of how unusual the codebase is.
Otherwise a reader may take six documented errors as six errors rather than as six caught errors.

### A.1 Aggregation artifacts

Four numbers were wrong for the same reason: a quantity averaged over one grouping does not
mean what the same quantity means averaged over another. §A.4 covers the adjacent fault of
comparing two quantities that were each pooled correctly but over different populations.

**A gain averaged against three losses (the pooled decomposition figure).** The largest of these,
and the one that reached furthest into the paper. Our main decomposition figure originally pooled
all four corners of the similarity design. But `S-HH` — high feature *and* high readout
similarity, pre-registered as the benign corner — **does not forget**: its retained capacity rises
(Δ log α = +0.11 to +0.22 across richness) and its behavioral forgetting is zero (−0.0005 to
+0.0001), against +0.41 for the catastrophic corner. Pooling a corner that gains capacity with
three that lose it made the figure's own axis label untrue of a quarter of its data, and produced
two derived numbers that looked like findings: the utility share at γ₀ = 0.3 read 0.101 rather
than 0.189, and at γ₀ = 1 the pooled utility share (0.284) fell *below all four* corner values
(0.296, 0.330, 0.355, 0.529) — a mean of opposite-signed terms is smaller in magnitude than the
mean of their magnitudes. We now report the three forgetting corners together and the benign
corner separately. This sharpened the result rather than weakening it: the utility share rises
monotonically with richness in each of the three corners taken alone, one further richness level
becomes resolvable (γ₀ = 0.1, at 2.1 noise floors, where the four-corner pool gave 0.4), and no
term crosses zero anywhere in the range. The lesson is that "pool the conditions" is a modelling
assumption about sign, not a neutral averaging step.

Two independent failures compounded here, and they belong to different families. Pooling a gain
with three losses is the aggregation artifact described above. That it went unnoticed for as long
as it did is the separate fault in §A.3: the panel plotted the *magnitude* of the change, so the
one corner moving the other way was drawn as though it were forgetting.

**Cell-mean r² inflation (r² = 0.878 → R² = 0.589), and a correction that illustrates its own
section.** Forgetting magnitude varies with which task is being forgotten and with how long ago it
was learned. Averaging arms within each (task, lag) cell and regressing the ten cell means on the
task's starting retained capacity gives r² = 0.878, which invites the claim that a single
quantity — distance above a common asymptote — explains most of forgetting. It overstates it:
averaging within cells removes arm-to-arm variance from the denominator, so the r² describes the
smoothness of the cell means rather than the predictability of an arm. The same regression on the
1,200 individual arm observations gives **R² = 0.589**. Both numbers are correct about different
things — the systematic trend is clean, and it is somewhat over half of the spread — and we report
the arm-level figure.

We first reported this pair as 0.889 → 0.099, a factor of nine, and the correction is worth more
than the number. Those values were computed over all four corners of the similarity design, and the
arm-level R² was not low because cell-averaging inflates r²; it was low because a model of pull
toward a *common* asymptote cannot fit a corner that moves away from it. The benign corner gains
capacity, so no value of "distance above the asymptote" predicts its arms at all, which crushed the
arm-level fit while leaving the ten cell means as smooth as before. Restricted to the corners that
actually forget, the inflation is a factor of 1.5 rather than nine. So the artifact is real and the
lesson holds — a cell-mean r² is not variance explained in the data — but **its magnitude had itself
been inflated by the pooling fault described immediately above**, which is one more instance of that
fault and the only one we found inside this appendix. We report it in that form deliberately:
the section's own claim is that pooling across a grouping is a modelling assumption rather than a
neutral step, and the strongest available evidence for it is that we made the same assumption while
writing the section.

**A per-condition effect pooled across richness (Δρ_c = +0.0246 → +0.0114, unresolved).** In
the `S-LL` corner we first reported center convergence of +0.0246 over the stream, pooling arms
with γ ≥ 3. The effect is γ-dependent and the pooled figure was carried by γ = 3. At γ = 10
alone the value is **+0.0114**, which against the drift measured in lazy arms — where the
representation barely moves, and which nevertheless drift +0.005 to +0.010 — is 1.1× baseline
with a copy-inflated sign test p = 0.15. At unique n=8 that is 3/8 declining, p = 0.73 —
nothing, not a near-threshold effect. `S-LL` is an unresolved non-effect, not a small
convergence, so its rank in any corner ordering carries no information.

The instructive part is how we fixed it the first time. The wrong value came from a 4-arm width
cell, and we corrected that one cell — leaving the other three cells of the same row in place, one
of which (`S-LH`, +0.0922 against a 40-arm +0.1098) then reached a later draft. A row estimated at
4 arms per corner carries ±0.02 on quantities whose real effects run 0.011 to 0.110, so *every*
cell in it was unusable outside the like-for-like width contrast it was built for, and correcting
the cell we had noticed guaranteed we would meet the next one. The fix that holds is to label the
row by its arm count and give the authoritative row beside it.

**Lag pooled across task position (1.29× effect inside a 1.56× confound).** Forgetting grows
with the number of intervening tasks. But in a 16-task stream, long lags are available only
for early tasks — lag 15 exists only for task 0 — so pooling by lag alone mixes elapsed time
with task position. Measured separately, the two are comparable in size and act in the same
direction: within task 0, forgetting grows 1.29× from lag 4 to lag 15, while at lag 4 held
fixed it grows **1.56×** from task 8 to task 0. Any lag-pooled magnitude therefore overstates
lag dependence. Our main decomposition figure is drawn at lag 12, which by construction exists
only for task 0 and is therefore unaffected; we state this in the caption rather than relying
on the reader to reconstruct it, and we note the consequence, which is that the figure reports
the most-forgotten task rather than the grid average.

None of the four changed a headline conclusion, and the first of them improved one. That is the
point worth making: they were found because the analysis re-derives every quantity from stored
geometry and gates every share on a measured noise floor, not because a conclusion looked wrong. One
instance was found inside this section, in the size of its own second example, which is the most
direct evidence we can offer that the fault it describes is easy to commit rather than a beginner's
error we are reporting from a distance.

### A.2 Guards that failed open

The second family is more instructive, because in each case a guard existed, ran, and reported
a number anyway. A guard that fails *closed* announces itself: the analysis stops. A guard that
fails *open* returns a plausible value, and the only way to catch it is to test the guard
against something it did not calibrate itself on.

**A silent clip on an out-of-domain conversion.** The center-collapse conversion maps Δρ_c to
the radius change it implies, using a form fitted on synthetic manifolds over ρ ∈ [0.043, 0.803]
that diverges as ρ → 1. Out-of-domain inputs were passed through `np.clip`. On real
representations the input routinely exceeded 1 — because the fit had been made against the
unnormalized convention `rho_c_glue` while the measurement supplied the normalized
`rho_c_signed` — so the clip was silently evaluating the fit at its boundary and returning
finite, reportable numbers. The fix was three-part: refit against the normalized convention,
declare the fitted domain as data, and *refuse* rather than clip outside it. The conversion now
returns `None` with a stated reason, and out-of-domain cells are excluded from coverage.

**Clips absorbing float error indistinguishably from real violations.** Two other clips —
cosines of principal angles into [−1, 1], and non-negativity of the anchor QP's duals — existed
to absorb rounding, but could equally absorb a genuine violation of hundredths, which would mean
the estimator was wrong. They were replaced by `numerics.clip_to_noise`, which clips and asserts
the movement was at float scale, raising otherwise. Both now pass at ~1e-16.

**A floor measured once, applied everywhere (2.3× too permissive where it mattered most).** The
center-collapse share is a ratio whose denominator is a measured radius change, so it is
unbounded when that change is unresolvable; the first smoke grid produced fractions of 42 and
−134 this way. The guard floors the denominator at the `R_eff` Monte-Carlo noise floor, taken as
a single CV of 0.50%. Re-measuring that floor at every richness — the same trained
representation re-measured under several measurement seeds — showed it is **not a constant**: it
rises monotonically from CV 0.28% at γ₀ = 0.03 to 1.13% at γ₀ = 10, so the guard was **2.3× too
permissive at exactly the richness where it is used most**. That is the failure mode the guard
was introduced to prevent, one step further out, and it is the third instance of the same
pattern. Refitting to per-γ floors moved coverage from 91% to 88% and the median share by less
than 0.01 at every richness, so the panel survived its own correction. Both of those percentages
describe the four-corner pool at the 1σ gate then in force, and both are superseded by the panel as
it now stands — three corners at 3σ, where coverage is 100% for γ₀ ≥ 1; they are quoted because the
comparison they make is between two floor settings on one population, which the later regrouping
does not disturb.

**The same guard, too permissive in level as well as in γ (1σ → 3σ).** The refit then exposed the
deeper version of the same mistake. A denominator sitting at 1σ of its own measurement noise
carries about 100% relative error, so the ratio built on it is noise over noise even though the
guard passed it — and the benign corner is exactly that case, with a median |Δ log R_eff| of
0.86–0.98 floors where the three forgetting corners sit at 10–21. Half its cells were admitted,
and 50–75% of those came out with the *opposite sign* to the ρ_c prediction. Raising the gate to
three floors empties that corner completely while leaving the three forgetting corners untouched
(all cells retained, medians unchanged to three decimals), which is the signature of a threshold
that separates signal from noise rather than one that trims a distribution. With the gate at 3σ
and the benign corner reported separately, coverage is **100% at γ₀ ≥ 1**, where the claim lives.
One consequence we state rather than bury: because the floor is γ-dependent, the panel's bars are
gated at *different* thresholds, so their coverage percentages are not comparable across bars, and
both the coverage and the threshold are printed on each bar.

**A guard that could not see the thing it was guarding against.** The last of these was caught in
the final week and is a different shape from the other two. Asking whether the peak of the
backward-transfer curve is a property of the corner or of the lag it is read at, we tested whether
the peak's *range* across lags fitted inside one grid step. It does, at 0.90, so the check passed
and reported the peak as stable. But the four values are 2.86, 1.75, 1.23 and 0.96 — a monotone
slide, ordered by lag, which is precisely the dependence the check existed to detect. **A range test
cannot see a monotone trend**: it collapses an ordered sequence to its two extremes and asks only
how far apart they are, so a systematic drift and an unlucky pair of noisy endpoints are the same
observation to it. Replacing the range with a test for monotonicity reversed the verdict and changed
what §5.1 claims. The general form: a check on a summary statistic inherits that statistic's
blindnesses, and a check applied to data with a natural ordering should use the ordering, because
the failure mode being guarded against is usually systematic rather than dispersed.

The generalisable point across all four is that each guard was written to catch a specific failure
and was itself calibrated at, or shaped by, the one setting that motivated it. Testing a guard at a
setting other than that one is cheap, and it is the only step that would have found any of these —
the floor's level and its γ-dependence were both found by re-measuring it somewhere else, the
pooled-sign artifact in §A.1 was found by asking which condition the guard was refusing, and the
range test was found by looking at the four numbers it had summarised.

### A.3 Transformations that destroyed sign information

The third family is distinct from both of the others and produced the largest single correction in
this project. §A.1's artifacts came from averaging over the wrong grouping and §A.2's from guards
that admitted out-of-domain values. These are neither: a transformation applied for presentational
tidiness erased the very distinction being measured, and every guard downstream of it ran on a
quantity that no longer carried the information it was guarding.

**An absolute value under an axis that named a direction.** The decomposition figure's magnitude
panel plotted `|Δ log α|` beneath an axis reading *how much retained capacity is lost*. For the
three conditions that lose capacity the two agree, so the panel was right about three quarters of
its data and silently inverted about the fourth: `S-HH` gains retained capacity, and a gain of 6.2
noise floors was drawn at the same height as a loss of 6.2. Nothing failed, no guard fired, and the
figure was internally consistent — the axis label was the only thing that was false. What it hid is
the backward-transfer result now in §5.1, and what it produced is a pooled magnitude at γ₀ = 10 of
50.7 floors where the three forgetting conditions give **69.7**: a gain averaged against three
losses, cancelling in the numerator of every derived share. Keeping the sign both revealed a result
and corrected the number it was concealed inside.

**The same mechanism, one project-year earlier.** Center correlation was originally recorded under
an absolute-value convention, `|ρ_c|`, which cannot represent the direction of center movement at
all. H1d and H6 are both claims about direction, so under that convention neither was testable, and
the 2×2 result in §5.4 — readout similarity driving convergence and feature similarity driving
decorrelation, with opposite signs in adjacent corners — would have appeared as four positive
magnitudes of similar size. It was fixed the same way, by keeping the sign.

The generalisable rule we now apply: **an absolute value or magnitude taken of a signed quantity
requires a stated justification at the point it is taken**, because the default outcome is a silent
mixture that no downstream check can detect. Both instances were found by asking what the sign
would have said, not by any test failing.

**A third instance, and the least visible of the three.** The channel shares are `|term| / Σ|term|`,
a magnitude by deliberate choice: opposing factors make shares of a small net difference exceed one
and mislead, so the denominator is total motion. In the three forgetting corners every term has the
same sign and the choice costs nothing. In the benign corner the radius term opposes the other two,
and the share then reports it as contributing 11% of the gain when it *subtracts* 11%. We now report
signed shares wherever the terms disagree in sign, and keep the magnitude share only where they do
not.

The reason this one hid longest is that it does not look like the other two. **A normalisation by a
sum of magnitudes takes an absolute value without spelling it `abs`.** The first two instances are
visible at the point of the transformation — someone reading the line can see the operation being
applied to a signed quantity. This one is visible only in the denominator, where it reads as a
neutral choice about how to scale, and the sign is discarded in the numerator as a side effect. So
the rule about stating a justification wherever a magnitude is taken has to cover the cases where no
magnitude appears to be taken.

### A.4 Comparisons between quantities pooled over different populations

Distinct from §A.1, and it took us longer to see. In §A.1 a single number was averaged over a
grouping that made it mean something other than it appeared to. Here **each number is correct about
its own population** and the fault is in setting them beside each other. Nothing is wrong with either
value, so no recomputation finds it; only asking of each what it was computed over does.

The instance that named the family: §5.3 reported that the matched subset used for the width
contrast differs from the full grid by 5.1%, "three times the 1.5% spread across widths". The 5.1%
was computed over four corners and the 1.5% over three. The ratio was therefore between two
populations, and the sentence claiming one was three times the other had no defined referent —
even though both percentages were right. At a common grouping the numbers are 4.5% and 1.52%, so
the ratio is still three and the conclusion survives. **It was accidentally true**, which is the
part worth reporting: the next such comparison would not be.

A second instance ran through the draft for longer. §5.3's magnitude decomposition was quoted at
three corners while the lag and task-position ratios in the same paragraph were still at four, so
adjacent sentences described different populations. That one was caught by the propagation audit
rather than by any check on the sentences themselves.

The rule we now apply: **a ratio, a "three times", or a "larger than" between two pooled quantities
must state the population both were pooled over, and the two must match.** The number inventory
carries the arm count `n` for every value partly for this reason; a comparison whose two sides
report different `n` over different conditions is the visible form of this fault.

### A.5 A gate applied in one place and not to a series quoted in prose

*Filed as its own family rather than under §A.3, which is specifically about transformations that
destroy sign information. This fault destroys nothing and transforms nothing: the gate exists, is
correct, and is applied — just not everywhere the quantity appears.*

The `R_eff` noise floor is not flat in γ₀, so per-arm radius terms are admitted only when
|Δ log R| clears 3σ of the floor **measured at that richness**. Figure 2d applies that gate, prints
the threshold under every bar, and reports coverage. The same quantity, pooled per corner, was then
quoted as a prose series in §5.1 — −0.0154, −0.0015, −0.0124 at γ₀ = 1, 3 and 10 — with no gate
applied at all. Those are 1.9, 0.2 and 1.1 floors. **Every one of them fails the threshold printed
in the figure two paragraphs away**, and the γ₀ = 3 value is a 20/20 sign split with mean/SD −0.13,
which is to say it is zero.

Worse than unsupported: **misleading about direction.** The only `S-HH` radius values that clear the
gate sit at γ₀ = 0.1 and 0.3 and are *positive* (+0.033 and +0.043, at 10.9 and 7.0 floors). So the
resolved evidence points the opposite way from the series that was quoted, and the sentence built on
it — that the radius channel is shared across all four corners — read a sign off three numbers the
paper's own machinery rejects. The correction is in §5.1.

Why it survived review of the figure *and* review of the prose: **the gate lives in the figure
script, and the prose series was computed by a different code path.** Nothing disagreed, because the
two never met. `audit_open_questions.py` found it only because it re-derived every pooled quantity in
floors and ranked them by margin — that is, by asking of each number the question the gate asks,
rather than checking whether a gate had been called.

The distinguishing feature against the other four families: §A.1 and §A.4 are faults of *grouping*,
§A.2 of a guard admitting what it should reject, §A.3 of a transformation discarding information.
This one is a fault of **coverage of the check itself**. The quantity is right, the gate is right,
the figure is right, and the prose is wrong, because a threshold was treated as a property of a
panel instead of a property of the quantity.

The rule we now apply: **a resolution gate belongs to the quantity, not to the figure that displays
it.** If a number is admitted to a panel only above a threshold, then every value of that number
anywhere in the document — prose, table, caption, abstract — carries the same threshold, and quoting
one below it requires saying so. Operationally: the gate should be computed where the quantity is
computed and travel with it, and the number inventory should record the margin in floors beside the
value, so that a sub-threshold number cannot be quoted without the margin being visible next to it.

### A.6 A field stored in a filename and never read

The sixth family is a parameter that appears in every output path and is never consumed. `make_stream`
does not read `stream_id`. `run_arm` passed `paired_init(seed)["stream"]`, so the filename recorded a
blocking factor the RNG did not use. On disk, every (γ, condition) cell of the registered grid is 40
files, 8 unique (initialisation, arrangement, dichotomy) triples, five identical copies each, keyed by
seed. Arrangement and initialisation are therefore confounded: seed determines both, and no existing
arm of the registered grid separates them.

The files were correct. The n was wrong. Point estimates are invariant to the copies — a mean over
five identical values equals the mean over one — so Figure 2, Figure 4, the width table and the
attribution did not move. Every SEM, confidence interval and sign-test p did: a sign test on 40 files
of which 32 are duplicates is not a test on 40 observations. `S-HL` at γ₀ = 10 is 8 of 8 unique seeds
declining, p = 0.0078, not 40 of 40, p = 1.8×10⁻¹². Stream R² = 0.000 is tautological, because
streams did not vary.

The generalisable point is that a parameter written into a filename is not evidence it was used. The
check that would have caught it is the same shape as the others: ask of a field whether anything reads
it, rather than whether it appears in the path.

The registered grid is reported at unique n = 8. New arms after the RNG fix (`stream_rng(stream_id)`)
are genuine n = 40, and a crossed arrangement × initialisation run is what unconfounds the two.
The first of those new arms, γ₀ = 5 at 5×8, finds S-HL declining on 31 of 40, p = 6.8×10⁻⁴. That
is an arm-level statistic. The 8-arrangement pre-commit (`results/gamma5_k8.json`) then scored
**4/8** arrangement means negative, interval [−0.023, +0.010], verdict arrangement-dependent.
§5.4 quotes that, not the arm-level p. Fixing the count did not by itself fix the *unit* —
see `docs/14-claim-security.md`.
The registered grid, whose 8 unique arms each carry their own arrangement, turns out to be the
better-conditioned set for that question, and its interval is the right size: SEM8 sits within a
factor of two of the K=3 between-arrangement SEM on all four corners (1.77, 0.97, 0.94, 0.55).

**K=3 pre-commit** (`results/unconfound_k3_precommit.json`, written before any unconfound arm
completed). With three arrangements you can detect whether the leading results hold across draws;
you cannot estimate arrangement-level variance. If the channel reorganisation (alignment up,
radius down) and the corner ordering (S-LH largest positive Δρ_c, S-HL the only decorrelating
corner, S-HH a gain) reproduce across three independent arrangements, the results are not
arrangement-specific; if they don't, that is a finding and the paper's scope narrows to the
registered-grid arrangements. K=3 is out of registration, uses the fixed RNG, and is not pooled
with the grid — an appendix robustness arm with the same status as γ=30 and the width arm.

**K=3 outcome** (`results/unconfound_k3.json`; 960/960 usable). Against that pre-commit: channel
reorganisation (alignment up, radius down) and the leading corner ordering (S-LH largest positive
Δρ_c, S-HL the decorrelating corner, S-HH a gain) reproduce on **3/3** arrangements. Verdict: not
arrangement-specific. The paper's scope does not narrow.

The stricter clause "S-HL the *only* decorrelating corner" is **2/3**: arrangement 2 has S-LL also
decorrelating (−0.0195, 40/40, Spearman −0.69). That is the unresolved S-LL corner already on the
registered grid, not a failure of the leading pattern.

Stream R² at γ=10, three-corner, streams actually varying: **0.001**. Seed R² 0.0008. The grid's
stream R²=0.000 was tautological; this is the measurement. §B quotes 0.001.

K=5 was not run. The pre-commit said K=3 is detection, not estimation; 3/3 on the leading pattern
is the licensed stopping rule. Running K=5 after seeing arrangement 2's S-LL would be fishing.
Neither a genuine-n N=300 width arm, a D-sweep, nor γ=30 at genuine n=40 is licensed: the first
chases a U-shape the paper does not claim across mixed populations; the second is closed by K=3;
the third would recover a fenced additivity-failure claim unique n already unlicensed.

---

## B. Methods note: what does not vary, and why the design is paired

At γ = 10, regressing per-arm forgetting on candidate sources, over the three conditions that
lose capacity (1,200 observations):

| predictors | R² |
|---|---|
| **stream instantiation alone** | **0.000** |
| **initialization seed alone** | **0.007** |
| distance above the common asymptote + lag | 0.589 |
| stream condition alone | 0.523 |
| condition + distance + lag | 0.882 |
| + stream + seed | 0.892 |

**The result this note exists for is a pair of zeros, one of them tautological.** The initialization
a run was trained from explains none of how much capacity is lost (R² = 0.007). Stream identity was
stored in the filename and never consumed, so the quoted stream R² = 0.000 is tautological on that
grid. The crossed arrangement×init run measures stream R² = 0.001. Because seed contributes nothing, pairing initializations across conditions means a
condition contrast is not competing with initialization variance. Sharing streams across conditions
was defended on the same measurement; that defence waited on streams actually varying.

Descriptively, the two sources that do explain forgetting are comparable in size: how far the task
sat above the common asymptote when it was left, together with the elapsed lag, accounts for 0.589,
and which corner of the similarity design the stream occupies for 0.523, with 0.882 between them.
Which of the two is nominally larger is not a claim we make — they are close, they overlap through
the corner-by-position interaction visible in the channel composition at γ = 1, and the design was
not built to separate them. Residual dispersion after all five predictors is 0.143 in log units
against a total of 0.436, still 7.7 noise floors, so real structure remains unmodelled.

**Every number in this paper is re-derived by a script from stored geometry.** There is no value in
the text that exists only as a figure produced once, or as a computation done at a prompt: the last
such quantity, the stratified probe correlation in §5.2, was given a script during the final audit.
A one-page inventory lists each of the 75 numbers with the artifact it is read from and the arm
count it is computed over, and 25 of them additionally carry the sentence that quotes them,
verified verbatim. We state this because it is unusual and because it is what made the corrections in §A
findable at all.

**An earlier version of this table said condition dominates, and that was an artifact.** Computed
over all four corners it gave condition 0.816 against 0.092 for distance and lag — a ratio of nine,
which invited the reading that forgetting is mostly about which stream you are in and barely about
where in it you are. It was not. With one corner *gaining* capacity and three losing it, the
condition variable was largely encoding the **direction** of the change rather than its size, and
a categorical predictor that separates a gain from three losses will explain most of the variance of
a signed quantity almost regardless of what else is true. Restricted to corners that all forget,
the apparent dominance disappears. The pair of zeros is untouched by this, both being within-cell
quantities: 0.000 → 0.000 and 0.002 → 0.007.

---

## C. Status of the γ = 30 probe — fixed before its numbers were read

The γ = 30 arms extend the swept richness range by one log step to test whether the `S-HL`
center decorrelation is more than an edge-of-range effect. **It is a post-hoc extension of the
design, run to test a post-hoc finding**, and both halves of that sentence are true regardless
of how it comes out. Its status is therefore fixed here, in advance:

- **Appendix robustness probe.** One sentence in §5.4, one appendix paragraph or table row.
- **Not promotable.** It does not become a main-text panel, a headline number, or part of the
  additive-decomposition claim, however favourable it is. The registered sweep is γ ≤ 10, and
  the paper's γ axis stays that way.
- **Reported either way**, with the same prominence, at the same length.

What each outcome licenses, decided now so that none of them can be spun later:

| outcome at γ = 30 | what §5.4 may say |
|---|---|
| `S-HL` Δρ_c more negative than −0.055 | the decorrelation strengthens across the two richest points, so γ = 10 is not an edge artifact — still one appendix sentence |
| between −0.055 and 0 | the effect is present but non-monotone in richness; say "non-monotone", not "present" |
| ≥ 0 | γ = 10 is a single-point excursion; §5.4 must say the decorrelation is not monotone in richness and rest on the smooth *feature main effect* instead |
| main effects keep growing, interaction stays inside the drift band | additivity is not a γ = 10 coincidence; one appendix sentence |
| interaction grows past the drift band | additivity is γ-limited, and the §5.4 claim gains "over the registered range" |

Declared artifact checks, before the fact: every arm must be `usable` with an identity residual
at float scale, and ρ_c may leave the `[0.04, 0.80]` calibration window at this richness — which
would invalidate any *center-collapse share* at γ = 30 but not Δρ_c itself, since the latter
needs no calibration.

**Outcome, and what it licensed.** Two rows of that table fired in opposite directions *on
the file-level numbers*; unique n revises the second. The `S-HL` decorrelation **extended
and strengthened** as a point estimate (−0.055 at γ = 10 unique n=8 → −0.096 at γ = 30 unique
n=4; 4/4 unique declining, sign-test p = 0.125 — n=4 cannot reject). And the interaction
grew as a point estimate (+0.006 at 0.82 unique SEM to +0.016) but **does not resolve at
unique n=4** (1.74 SEM on the same formula that gave 3.9 SEM on 16 files; paired-seed 1.53
SEM, bootstrap CI includes 0). The pre-declared table licensed "over the registered range"
only if the interaction grew *past the drift band as a resolved effect*. It did not, at
unique n. Additivity-failure is not licensed. Artifact checks were clean: 64/64 usable, 0 of
1024 tasks unconverged, residual 4.1e-16, ρ_c inside the window. The unique-n numbers are in
`results/gamma30_unique.json`.

---

## D. Figure 2 caption draft

> **Forgetting decomposes exactly, and the channel that carries it shifts with richness.**
> Retained capacity `α(·; y_j)` for task 0, measured at its own task boundary and again twelve
> boundaries later, decomposed by the exact identity
> `Δ log α = Δ log Ψ_eff + Δ log(1 + R_eff⁻²) − Δ log D_eff`. Pooled over the three stream
> conditions that lose capacity (`S-HL`, `S-LH`, `S-LL`; 120 arms per richness level); the benign
> corner `S-HH`, whose retained capacity *rises*, is shown separately in Fig. S_. (a) Magnitude in
> units of the estimator's measured Monte-Carlo noise floor, signed: 0.2 floors at γ₀ = 0.03,
> rising to 69.7 at γ₀ = 10. The grey band marks ±2 floors, below which nothing is resolvable.
> (b) Channel composition, `|term| / Σ|term|`, drawn only where the total clears the band — γ₀ = 0.1
> and above; the utility channel grows from 0.08 to 0.45 of total motion while the radius channel
> falls from 0.39 to 0.11, and dimension falls from 0.53 to 0.44 across the first step and is flat
> thereafter. (c) The same terms with their signs, hollow
> markers where a term's sign is not resolved across arms; the utility term is negative on 100% of
> arms for γ₀ ≥ 1. (d) How much of the radius channel is center collapse: the change in `ρ_c`
> converted through a synthetic-manifold calibration, as a fraction of the observed radius change.
> Rising from 0.05 to 0.44 through γ₀ = 3 and then flat — the γ₀ = 3 → 10 step is not resolved
> (bootstrap CI [−0.038, +0.070]). Each bar is gated at three times the `R_eff` noise floor
> measured *at its own richness*, printed on the bar with its coverage; because the threshold
> differs per bar the coverage percentages are not comparable across bars.

**Two facts about the cell, both in the caption rather than left to the reader.** Lag 12 exists
only for task 0 in a 16-task stream with boundaries at 0, 4, 8, 12, 15, so this is a task-0 figure
by construction and is immune to the lag/task-position confound of §A.1 — and because task 0 is
the most-forgotten task, it reports the largest forgetting in the grid rather than its average.
The composition is not a property of that cell: at lag 4 the shares agree to within 0.032 at every
richness while the magnitude changes by 1.27×.

---

## E. Figure 4 caption draft

> **Center correlation moves in opposite directions across the similarity design, and the two
> axes act almost separately.** (a) Signed center correlation ρ_c over the task stream at
> γ₀ = 10, per corner of the feature × readout similarity design; all four corners begin
> together (ρ_c = 0.391–0.392), so the split is produced by training rather than by
> initialization. Three corners converge; `S-HL` — high feature similarity, low readout
> similarity, the same stimuli under different rules — decorrelates. (b) The same endpoint
> difference across the richness sweep, with the grey band showing the drift measured in lazy
> arms (γ₀ = 0.03), where the representation barely moves; the corner ordering is a rich-regime
> effect, and `S-LL` never leaves the band. (c) Per-arm Spearman correlation between ρ_c and
> block index: the `S-HL` decline is progressive rather than an endpoint difference, negative
> on 40 of 40 arms (median −0.79), which is the form the human practice result takes.
> (d) Both pre-registered predictions fail. H1d predicted decorrelation with richness; three of
> four corners converge. H6 predicted the largest |Δρ_c| in the catastrophic corner `S-HL`; the
> largest is `S-LH`, twice the size. (e) Read as two main effects, readout similarity drives
> convergence (+0.104) and feature similarity drives decorrelation (−0.061), with an
> interaction at the resolution limit (+0.006) and an additive model predicting `S-HH` to
> within 0.012; across the registered range both effects grow with richness at a roughly
> constant ratio, so richness sets the gain and the design sets the sign. (f) Both main effects
> survive a 4× change in load (N = 150, 300, 600). Error bars are ±1 SEM over arms; 40 arms per
> corner in (a)–(e), 4 per corner at N = 150 and 600 in (f).

The prose these two paragraphs supported is now §5.4, which states both limits in the same breath
as the claims they bound. Neither limit is a hedge — each names the range over which the sentence
before it holds.
