# 11 — Role Clarification, and Open Questions About the Results

## Why this note

The last deliverable was a full draft — abstract, eight sections, argument, framing.
It is good writing, and that is the problem. The paper's argument is being made in
two places at once: in `paper/body.tex` inside the repo, and in the analysis
conversation outside it. Two authors of the same argument, each with partial view
of the other, is how the four-corner/three-corner inconsistency happened in
`07-writeup.md` — not through carelessness, but because a document assembled in
one place was being reasoned about in another.

So this is a boundary change, not a criticism. The draft you produced stays and
is being edited rather than rewritten. What changes is where new argument comes
from.

---

## What is yours

**The record of what was computed.** This is the part nothing else can do, and it
is the part the paper depends on absolutely.

- Every experiment, with every parameter: grid definition, architecture,
  parameterisation, stopping rule, stream construction, seeds, arm counts,
  what was swept and over what range.
- Every run that happened, including the ones that did not become results:
  Phase 0, W1, W4, W5, W5b, the γ=30 probe, the timing arms, the pilots that were
  killed. What each was for and what it returned.
- Every number, with its source artifact, its n, its selection rule, and its
  resolution status against the relevant floor.
- Every correction: the superseded value, the replacement, and the mechanism that
  produced the error. The artifact families in the appendix are yours.
- Every figure: generated from a committed script, stamped with the result-set
  hash and arm count, regenerable, manifest-verified.
- The verification apparatus: tests, scope audit, inventory, prose-pinning,
  submission checker.

**The standard for this material is that a reader should be able to reconstruct
any number in the paper without asking anyone.** That standard is currently met
and it is unusual; the notebook, the inventory and the manifest are the reason.

## What is shared

**Interpretation, when it is close to the data.** When you say "the radius term is
negative in every corner including the one that gains, so the four corners share
one channel and differ in two" — that is a reading of a measurement, you are the
one who can see it, and it belongs in your reports. Several of the paper's best
sentences came in exactly that form.

**Flagging when prose and data diverge.** You have caught this repeatedly and
should keep doing it: a claim that overstates what the numbers support, a caption
that describes a figure it no longer matches, a comparison between two differently
pooled quantities. That check does not work from outside the repo.

**Drafting to a specification.** If a section is specified — this claim, these
numbers, this length — writing it is efficient and you do it well.

## What is not yours

**Choosing the argument.** Which findings lead, how the contribution is framed,
what the paper claims, what goes in the abstract, how results are ordered, what is
cut for length. This is where the two-author problem bites hardest, and it is
handled outside the repo.

**Positioning against the literature.** Which prior work the paper is situated
against, which citation supports which claim, what is novel relative to what.
The current draft is a demonstration of why: it makes claims across capacity
theory, feature learning and continual learning with five citations, attributes
the central identity to the paper that applies it rather than the one that derives
it, and never cites manifold capacity itself. Those are not writing errors; they
are positioning decisions made without the literature in view.

**Deciding what is a finding.** Whether the S-HH result is a headline or an
appendix note, whether an unregistered result is reported as such, whether a
refutation is a failure or a redirection. You should report what the data shows
and flag the status; the weighting is decided outside.

---

## Immediate consequence

The draft is not discarded. Specific edits are coming from outside: an estimator
paragraph, citation fixes, three numeric discrepancies between body and captions,
a length cut. Apply those as specified.

**Before any of it**, two things:

1. **Get a TeX toolchain and report the actual page count.** `tectonic` is a
   single binary needing no system install. Every length decision waits on that
   number, and it is currently UNKNOWN.
2. **Re-point the inventory at `paper/body.tex` and `paper/figures.tex`.** It
   currently checks `07-writeup.md`, which is not the submitted file, so the
   prose-pinning is checking a document that will diverge.

---

## Open questions about the results

Answer from the record where you can, and say plainly where the record does not
answer. These determine whether anything further needs running.

### On what exists

1. **Is there a single document that states every experiment run in this
   project, with parameters, in one place?** Not the log, which is chronological
   and interleaved with reasoning. A registry: name, purpose, parameters, arm
   count, date, result location, whether it fed the paper. If not, that is
   probably the most valuable thing you could produce next, both for the appendix
   and for September.

2. **Which computed results are not in the paper or the notebook at all?**
   Phase 0 has five results; the notebook has ten sections. What was measured and
   then not reported anywhere?

3. **What is in `results/` that no script currently reads?** Orphaned artifacts
   are either dead weight or unreported measurements, and it matters which.

### On resolution and coverage

4. **Which reported numbers sit closest to their resolution limit?** A ranked
   list of every paper number by margin over its floor. The ones within 2–3× are
   where a reviewer will push, and where a small analysis change could flip a
   claim.

5. **Where is coverage below 100%, and what is excluded?** Panel (d) gates per
   bar; the attribution has a `MIN_FLOORS` gate; the calibration has a domain.
   One table of every gate, its threshold, and what fraction it excludes.

6. **Is there any quantity we report pooled that we have not checked for a sign
   mixture?** The four-corner pooling hid both a gain and a shared channel. The
   sign audit covered Δlog α across 520 cells. Was the same check run on the
   three terms separately, on ρ_c, on the probe margin?

### On the specific results

7. **The S-HH gain decays with lag (+8.7, +9.0, +6.2, +3.1) but increases with
   task position (+8.7, +12.6, +12.8).** Is there a reading of that pattern from
   the data — does it look like the gain is about how much stream remains after
   the task, or about something else? Not asking for a mechanism, asking whether
   the numbers distinguish any candidates.

8. **§5 says generic capacity rises while probe margin falls, between richness
   levels but not within.** Do we have the same between/within split for any
   other pair of measures? If it is general, that is a stronger claim than one
   instance.

9. **The additive decomposition holds over the registered range and fails at
   γ=30 because main effects saturate while the interaction does not.** Is the
   saturation visible in any other quantity at γ=30, or is it specific to ρ_c?

10. **`S-HL` decorrelation is absent at γ=3, present at γ=10, stronger at γ=30.**
    Is there anything between 3 and 10 in the record — width arms, timing arms,
    anything — that would locate the onset better than one grid step?

### On whether to run more

11. **Is there a cheap measurement that would materially strengthen a claim we
    already make?** Cheap means under an hour, and materially means it moves a
    number off its resolution limit or closes a gap a reviewer would find. If the
    answer is no, say so — that is the useful answer with nine days left.

12. **Is there anything you would want measured that we have not discussed?**
    You have the closest view of where the data is thin. If something has been
    bothering you, this is the moment.

13. **Conversely: is any currently-reported result thin enough that you would
    rather cut it than defend it?** The width arm bounds an effect rather than
    measuring one; the centre-collapse panel is post-hoc calibrated with per-bar
    gates. Would the paper be stronger without either?

---

## What I expect back

A written report, not code changes. Where the record answers, the answer with its
source. Where it does not, that stated plainly. Where you think something should
be run, the case for it with a measured cost.

Then we decide together what, if anything, runs — and the writing decisions move
outside the repo.
