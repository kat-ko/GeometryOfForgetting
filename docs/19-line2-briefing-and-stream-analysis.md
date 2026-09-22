# 19 — The Line 2 Framework, and What It Asks of This Project

```
Purpose: brief the repo agent on a second line of research it has not seen, then
         specify the analysis that line makes necessary here.
Status:  §1–3 are background. §4 is the work. Zero compute; every quantity is
         already recorded and has never been read.
Rule:    report tables. No figure edits, no writeup changes, no new runs, until
         the report has been read.
```

---

## 1. What the other line argues

There is a parallel paper in this group, a position piece on continual-learning
methodology. Its argument, in one form:

> A benchmark result is a joint outcome of the learner and the setting it was
> measured in. The setting includes how the data stream changes, how similar its
> regions are, how coarsely it was partitioned into tasks, and what the learner
> was told about that partition. Where those are not reported, an outcome cannot
> be attributed to the learner rather than to the setting.

It separates the setting into three layers, by **who fixes each property**:

**Stream structure** — fixed by the data-generating process, independent of any
observer.
- *similarity* — the degree and kind of shared structure between regions of the
  stream. The paper's sharpest point is that similarity must be stated **at a
  level**: inputs may overlap while readouts do not, and components at different
  levels can act in opposite directions.
- *change* — the temporal geometry of the process. Abrupt, gradual, cyclic, or
  compositional.
- *exposure* — the schedule. Blocked or interleaved, in what order, with what
  spacing and recurrence, and how many passes.

**Discretization** — fixed by the observer who decides where tasks begin and end.
- *scope* — the breadth and heterogeneity of experience grouped under one task
  label.

**Learner access** — fixed by the protocol.
- whether task identity is available, whether boundaries are signalled, whether
  state persists across transitions.

The paper audits representative benchmarks against those properties and finds
that **similarity and scope are specified by essentially no benchmark at any
scale**. Its central figure is a thought experiment: one continuous stream,
several partitions laid over it, with quantities defined relative to the
partition (boundary locations, task similarity, what forgetting is measured
against) taking different values under each.

It also makes a second demand, stricter than reporting:

> A property varied without being shown to matter for the learner under test has
> not been probed.

---

## 2. Why this project is that argument's constructive counterpart

This project specifies every property that audit finds missing. `docs/06-scope`
and the notebook's §2 already state them as a table. Similarity is set on two
levels independently. Scope, change, exposure and access are stated. That is
unusual and it is the reason the geometric attribution is possible at all.

**But the audit's standard applies to us too**, and there is one place where we
currently fall short of it.

The parameters `s_f` and `s_r` are **consecutive**. They govern the step from
task `t−1` to task `t`. `docs/15` records this explicitly:

> Not controlled: the full `T×T` similarity; off-diagonal `s_f` is a realized
> cosine.

The headline measurement is **retained capacity on task 0 at lag 12**. What
plausibly determines whether task 0 survives is its relationship to the twelve
intervening tasks. That is a realized property of each stream, it is recorded,
and nobody has looked at it.

So a corner label describes a **generating process**, not a relationship. When
we write "high feature similarity," a reader takes that to describe the
comparison the figure reports. It describes the step size that produced the
sequence.

That gap is exactly what the audit accuses others of. Being the study that reads
its own realized structure rather than trusting its generating parameters is the
standard we are asking of the field.

---

## 3. Three specific reasons this matters here

Stated as hypotheses to test, not as findings.

**3.1 The corners may separate far less at lag 12 than at lag 1.**
If arrangements are AR(1) at correlation `s_f`, then `corr(A_0, A_t) ≈ s_f^t`.
At `s_f = 0.9` that is roughly 0.28 by lag 12. At `s_f = 0.1` it is essentially
zero. The corners still differ, but between 0.28 and 0, not between 0.9 and 0.1.

The dichotomy side may be worse. At `s_r = 0.9` the walk takes two label swaps
per step on the Johnson graph J(16,8). After twelve steps it may have approached
the random-pair baseline, in which case **the high and low readout corners do not
differ at lag 12 at all**.

**3.2 Unintended recurrence is possible in one corner, and it bears on a
headline finding.**
S-HH takes small steps on both axes. A small-step walk over sixteen tasks can
wander back toward `y_0`. If a stream contains a task at position 4 or 8 close to
`y_0`, on an arrangement still partly correlated with `A_0`, then part of the
S-HH gain is **re-training on approximately task 0** rather than backward
transfer.

The recurrence axis found no savings on deliberate re-presentation, which makes
this less likely. It does not make it checked.

**3.3 Change and similarity are entangled by construction.**
At high `s_f` the arrangement drifts slowly across the stream. At low `s_f` it
jumps. So the framework's *change* property co-varies with *similarity* in our
design. The setup table reports change as "abrupt at every boundary," which is
true of the boundary and not of the trajectory.

---

## 4. The analysis

Zero compute. Every quantity below comes from the recorded `T×T` similarity
matrices and the existing per-arm records. Unit throughout is the **unique
(arrangement, init) draw**, not the file.

### 4.0 Verify a load-bearing sentence first, before anything else

Does the manifold arrangement change between tasks within a stream, or is it
fixed for the stream?

`docs/15` says `A_0` is fresh and `A_t` redraws centres correlated at `s_f`.
The notebook §4 and the paper's setup both say the sixteen clouds never change,
so the inputs are identical across tasks and only the required answer changes.

**Those cannot both be true.** If arrangements are redrawn per task, the
rule-only-interference claim holds for `S-fixed-r` — which was registered and
never run — and not for the 2×2 that produced every figure.

Report which it is, from `streams.py`, and list every place the wrong version is
asserted.

### 4.1 Realized similarity, by lag

For each corner: `s_r(0,t)` and `s_f(0,t)` for `t = 1…15`, mean and spread across
unique streams.

Then the two numbers the figures actually depend on: `s_r(0,12)` and `s_f(0,12)`.

**Report the effective separation between corners at lag 12, beside the nominal
separation at lag 1.** If they differ substantially, every sentence describing
the 2×2 needs rewording.

### 4.2 Decay law

Does `s_f(0,t)` follow `s_f^t`? Report measured against predicted.

For dichotomies, report the distribution of `s_r(0,12)` against the
random-balanced-pair baseline, so we know whether the high-readout walk has
saturated. If it has, the readout axis is not a lag-12 variable.

### 4.3 Unintended recurrence

Per stream: `min_t s_r(0,t)` and `max_t s_f(0,t)` over `t = 1…12`. How close does
each stream come back to task 0? Distribution by corner.

Then the load-bearing check: **does the S-HH gain magnitude correlate with the
closest approach?** Report the correlation with a CI at unique n.

Either outcome is reportable. A null rules out an alternative explanation and
strengthens the finding. A positive correlation means part of the gain is
re-training and the finding must be restated.

### 4.4 Which predictor wins

Regress forgetting at lag 12 on, separately:

1. the corner label (categorical)
2. realized `s_r(0,12)`
3. realized `s_f(0,12)`
4. `min_t s_r(0,t)`
5. cumulative displacement over the twelve steps

If a realized quantity predicts better than the label, that is the audit's
argument demonstrated on our own data, and it belongs in the paper rather than
in a log.

### 4.5 Change, as a recorded design fact

Report the per-step arrangement displacement by corner, to make the
change/similarity entanglement quantitative rather than argued.

---

## 5. What not to do

- Do not change any figure, caption, or writeup text on the basis of this report.
- Do not run new arms. Every quantity is recorded.
- Do not treat a realized quantity as a new experimental axis. It is a
  description of what was already run.
- If §4.0 finds the arrangement claim is wrong, flag it and stop rather than
  correcting text — it touches the paper's setup and the notebook's §4, and how
  to state it is a decision outside the repo.

---

## 6. Why this is worth doing now rather than at the sequel

Two reasons.

The 2×2 carries two of the paper's findings. If the corners collapse at lag 12,
those findings are about a smaller contrast than the labels imply, and it is
better to find that ourselves than to have it found.

And the audit's second demand — that a property varied without being shown to
matter has not been probed — is one we currently meet for the two similarity axes
and nowhere else. §4.4 is the check that turns "we specified the stream" into "we
know which specified property did the work."
