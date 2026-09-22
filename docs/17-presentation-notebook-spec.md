# 17 — Presentation Notebook: Specification

```
Target:  notebooks/project-overview.ipynb (rewrite, not patch)
Purpose: a colleague who has never seen this project should be able to run it
         top to bottom and come away able to explain the method and the
         findings to someone else.
Status:  specification. §0 changes what the notebook is for; the rest is how.
```

---

## 0. What is wrong with the current version, and the rule that fixes it

The first four sections work: they build the setup from scratch with code that
runs, and a reader ends up holding the objects the project is about. Then it
stops explaining and starts summarising — the findings arrive as rendered PNGs
loaded from `figures/` with prose around them, and a reader who has not seen the
analysis pipeline has no way to see what was computed.

It also carries the project's history: refuted predictions, a failed crossing, a
plotting bug, a duplication fault. That history is real and belongs in the paper
appendix. In a talk it costs the audience their attention budget before the
result lands.

**The rule for the rewrite:**

> Every claim in the notebook is either computed in front of the reader in a
> cell they can read, or loaded from the repo *with the computation shown on a
> toy case first*. No claim is asserted with only a PNG behind it.

A concrete test: if a reader wanted to check that "the channel mix reorganises
with richness" means what the notebook says, could they point to a cell where a
share is computed from three numbers? Right now, no. After the rewrite, yes.

**And a second rule, on history:**

> Present what the method is and what it found. Do not present how we got there.
> No refuted hypotheses, no pre-registration narrative, no bug stories, no
> stepping stones. One paragraph near the end may say the record exists and
> where it lives.

---

## 1. What a reader must be able to do at the end

Five things. Every section serves one of them, or it is cut.

1. **State what a task is** in this setup, and why the design fixes the inputs
   and varies only the rule.
2. **Explain what manifold capacity measures** in their own words, and why it is
   not accuracy and not CKA.
3. **Read the three-factor decomposition** — what each factor is, and why the
   fact that it is an identity matters.
4. **Say what the difference is between generic and retained capacity**, and why
   that distinction only exists in a sequence.
5. **State the three findings and one limitation each.**

Nothing else earns space.

---

## 2. Structure

Ten sections. Target runtime under three minutes end to end.

### §1 — The question (markdown)

Short. When a network learns one thing then another, what happens to the first?
Existing answers report how much was lost or how far the representation moved.
Neither says what changed. One paragraph on where this comes from
(arXiv:2606.17889 found modularity's benefit is conditional on the
representational regime; this asks what the regime is *doing*).

### §2 — Sixteen clouds (markdown + code)

**Keep the existing generator cell.** It is the best thing in the current
notebook: real, from the source, and it makes the object concrete.

Add one cell that makes the geometry legible before any capacity talk:

```python
# For each manifold: how big is it, and how far is it from the others?
centres = mans.mean(axis=1)
radius  = np.linalg.norm(mans - centres[:, None], axis=2).mean(axis=1)
gaps    = pdist(centres)
print(f"mean within-manifold radius : {radius.mean():.3f}")
print(f"mean between-centre distance: {gaps.mean():.3f}")
print(f"ratio                        : {radius.mean()/gaps.mean():.3f}")
```

Then the two-panel PCA figure (keep as is). The reader should leave §2 knowing
that a manifold has a *size* and a *position*, and that those are separate.

### §3 — A task is a way of splitting them (markdown + code)

Keep the dichotomy cells. Add one thing the current version lacks: **show both
similarity axes changing something visible.** Two arrangements at high and low
centre correlation, side by side, with the realised correlation printed. Right
now feature similarity is described and never shown.

### §4 — What capacity is (markdown + code)

Keep the mini estimator — the LP separability test, the projection, the
bisection. It is the pedagogical core and it runs in five seconds.

**Add the piece the current notebook skips.** After the R and D sweeps, show
that capacity depends on *which* split you ask for:

```python
# Same manifolds. Different questions. Different answers.
m = make_arrangement(rng=np.random.default_rng(1))
for name, y in [("random split",  balanced_dichotomy(16, rng)),
                ("another random", balanced_dichotomy(16, rng))]:
    ...  # N_crit for that single fixed dichotomy
```

This is what makes §6 land. A reader who has seen that capacity is
split-dependent will understand generic-versus-retained without being told.

### §5 — The decomposition (markdown + code)

State the identity. Then **verify it numerically in front of the reader**, on
values loaded from one real result file:

```python
rec = load_one_attribution_record()          # from results/, any arm
lhs = np.log(rec["alpha_after"] / rec["alpha_before"])
rhs = (np.log(rec["psi_after"]/rec["psi_before"])
     + np.log((1+rec["R_after"]**-2)/(1+rec["R_before"]**-2))
     - np.log(rec["D_after"]/rec["D_before"]))
print(f"Δlog α  measured : {lhs:+.6f}")
print(f"        summed   : {rhs:+.6f}")
print(f"        residual : {lhs-rhs:.2e}")
```

A residual at 1e-16 on the screen is worth more than any sentence about
exactness. This is the single highest-value addition to the notebook.

Then show the arithmetic of a share, on the same record, so "the utility channel
carries 45%" is visibly a ratio of three numbers rather than a claim.

### §6 — Generic and retained (markdown + code)

Define both. Then compute both on the toy manifolds, using the estimator from
§4: capacity averaged over random splits, versus capacity for one fixed split.
Show that they differ. Two numbers, and the distinction is made rather than
asserted.

Then state that in a sequence the second is measured on a split the network was
trained on *earlier*, and that its decline is what this project calls forgetting.

### §7 — What was run (markdown, one table)

The setup table from the current version — network, optimiser, knob, stream,
conditions, measurement. Add the sampling unit in one line: how many independent
draws sit behind a reported number. No history, no populations table, no
artifact families.

### §8 — Three findings (markdown + code per finding)

**Each finding gets the same three-part shape**, and this is the part the current
notebook is missing entirely:

1. one sentence of claim,
2. **a table of the numbers the claim rests on**, computed from the result files
   by a visible cell,
3. the figure.

So for finding one, before the PNG appears:

```python
tbl = channel_shares_by_gamma()   # thin wrapper over the existing loader
print(tbl.to_string(index=False))
#   gamma   floors   utility   radius   dimension
#    0.03      0.2        —        —           —     (below resolution)
#     0.3      ...      ...      ...         ...
```

A reader can then look at the PNG and see the same numbers in it. Right now the
PNG is the only evidence and it is not readable as evidence.

The three findings, and nothing else:

- **Forgetting has a composition, and it reorganises with richness.** Shares by
  γ, resolution status stated per row.
- **The stream sets the sign, and the two similarity axes act separately.** The
  four conditions, the two main effects, the interaction with its resolution
  status.
- **One corner gains.** The gain across γ, and the count that makes it specific
  (resolvable gains occur in one condition only).

Where a number is a bound rather than a measurement, the table says so in a
column. That is the whole of the honesty burden — no separate caveats section.

### §9 — What this is allowed to mean (markdown)

The four constraints, compressed to one short paragraph each: content was never
scarce in a one-layer ReLU over separable centres; training was equalised by loss
and not by amount of change; everything is linear readability; the design sits at
one extreme of its own space by choice.

Then the three claims ranked by exposure — stream-sets-the-sign most robust
(network is a constant), channel composition most exposed.

One sentence, no more, that a full record of corrections and sampling-unit
analysis exists in the repo, with the path.

### §10 — Where it goes (markdown)

The five planned axes, one line each. Close with the open question worth putting
to the room: whether the scarcity argument can be made analytically, which would
turn the depth experiment into a test of a prediction rather than an
interpretation of a result.

---

## 3. Repo code: what to use, what to reimplement

**Reimplement in the notebook** (simple enough to read, and the reader needs to
see it):

- the manifold generator
- balanced dichotomies and the two similarity measures
- the LP separability test and the bisection capacity estimator
- the identity check and share arithmetic

**Import thin loaders from the repo** (too large to inline, but must be one
readable call):

- `ledger` or equivalent, for the tables in §8
- the figure sidecar JSONs

**Do not import** the full GLUE core, the training loop, or the attribution
pipeline. If a reader needs to open `src/glue/core.py` to follow the notebook,
the notebook has failed.

**Every loader call in §8 must return a table the reader can see**, not a
pre-rendered figure. If a wrapper does not exist that returns the numbers behind
a figure, write one — the sidecar JSONs already hold them.

---

## 4. Cut from the current version

- the plotting-bug story
- the duplication-fault story and the sampling-unit recomputation
- refuted predictions and the pre-registration narrative
- the artifact families
- the seven-population comparability discussion
- anything phrased as "we then found that we had been wrong about…"

All of it is in `docs/14`, `docs/15` and the paper appendix, and §9 points there
in one sentence.

---

## 5. Acceptance

- Runs top to bottom in a clean kernel, under ten minutes.
- Executes with `nbconvert --execute` with no manual steps.
- Every §8 claim has a visible table above its figure.
- The §5 identity check prints a residual at machine precision.
- Zero mentions of a refuted hypothesis, a bug, or a superseded number.
- A reader who has never seen the project can, after running it, state what a
  task is, what capacity measures, what the three factors are, why the identity
  matters, and what generic-versus-retained means.

That last one is the real test. If it needs the reader to already know the
project, it is not the notebook we need.


Drop it in as `docs/17-presentation-notebook-spec.md`.

The diagnosis in §0 is the part that drives everything else: the notebook stops explaining at exactly the point the findings arrive. Sections 1–4 build objects a reader can hold; sections 5 onward hand over rendered PNGs with prose around them. So the rule is that **no claim gets asserted with only a picture behind it** — either it's computed in front of the reader, or it's loaded with the computation demonstrated on a toy case first.

Four additions do most of the work:

**The identity check in §5** is the single highest-value cell. Load one real attribution record, compute Δlog α two ways, print the residual. A `1e-16` on screen makes "the decomposition is exact" a thing the reader watched happen rather than a claim they were asked to accept. Nothing else in the notebook earns as much per line.

**Split-dependent capacity in §4.** Same manifolds, two different splits, two different answers. Once a reader has seen that, generic-versus-retained needs no explaining — it explains itself.

**Tables before figures in §8.** Every finding gets claim → numbers → figure, in that order. Right now the figure *is* the evidence and it isn't readable as evidence. The sidecar JSONs already hold the numbers, so this is a wrapper, not new analysis.

**Feature similarity shown, not described.** Two arrangements at different centre correlations, side by side. It's currently the only element of the design that's asserted and never displayed.

On the cuts: the bug stories and refuted predictions are genuinely good material and they belong in the paper appendix, where a reviewer is already committed to reading carefully. In a talk they spend the audience's attention before the result lands. One sentence in §9 pointing at `docs/14` and `docs/15` covers it.

The acceptance criterion at the end is the one I'd hold the agent to: a reader who has never seen the project should be able to state what a task is, what capacity measures, what the three factors are, why the identity matters, and what generic-versus-retained means. Everything else in the spec is a means to that.



# On Writing
## Summary of general writing rules (But specifically tailored to the lab internal discourse research blog). Should provide guidance here with phrasing of paragraphs.  

**Structure**

- Post title = project name; opening post of a thread appends "- Introduction".
- Open with origin: who proposed it, which conversation or paper it came from. No throat-clearing.
- Separate literature summary from own reasoning with explicit headers ("Implications for my work", "What this shows", "Findings").
- Numbered sections for background, bullets for extracted claims, tables for two-way comparisons.
- Close with numbered references, full citations.
- Link the repository where experiments live.
- Mark provisional posts explicitly ("Disclaimer: This will be specified further").

**Sentence-level**

- One point per sentence, stated in the fewest words that keep it precise.
- Cut hedging that carries no information; keep hedging that marks genuine uncertainty ("might", "could", "hypothesized").
- No colons mid-sentence. No negated subclauses. No blog-style prose or rhetorical questions used as transitions.
- Present tense for literature findings, future or infinitive for plans.
- → for implication, not prose connectors.
- Bold or italic only for terms being introduced.

**Register**

- Scientific throughout. No pleasantries, no sign-off, no meta-commentary on the writing itself.
- "I" for solo work, "we" when the project has collaborators.
- Name collaborators and their specific contribution when an idea traces to them.
- State what a result means for the next step, not that it is interesting.

**Content discipline**

- Every literature summary must end in a consequence for the project, otherwise it does not belong in the post.
- Describe the form of a claim rather than asserting it absolutely when the evidence is partial.