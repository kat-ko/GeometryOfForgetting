# 10 — Split-CIFAR100 pilot: spec and cost estimate

```
Status:   SPEC ONLY. Nothing here has been run. Written 2026-08-13 at Kati's instruction
          ("write the pilot spec and cost estimate only, don't run anything yet").
Target:   ICLR 2027 full paper. NOT NeurReps.
Fence:    This work does not touch any NeurReps artifact. See §1.
Basis:    Costs are derived from measured per-eval timings in results/LOG.md
          ("Where the 26 h actually came from"), not from a fresh projection.
```

---

## 0. Why this exists, and what would make it worth running

Every number in the NeurReps abstract is measured on synthetic Gaussian manifolds with
independently controlled feature and readout similarity. That control is the whole reason the
$2\times2$ result is interpretable — it is what lets us say readout similarity drives center
convergence and feature similarity drives decorrelation, as separable main effects.

It is also the first thing an ICLR reviewer will attack. The two claims most exposed are:

1. **The channel reorganisation** — alignment share rising from 0.08 to 0.45 while radius falls from
   0.39 to 0.11 — could be a property of Gaussian manifolds with a fixed covariance structure rather
   than of rich learning.
2. **The benign corner's gain** — retained capacity rising in `S-HH` at every position and lag —
   depends on "high feature similarity" being a controlled quantity. On real images it is not.

The pilot is worth running if and only if it can move one of those two. It is **not** worth running
to show that the estimator executes on CIFAR features, which is not in doubt and is not a result.

**The single question:** at matched richness, does retained capacity for a past Split-CIFAR100 task
fall with a channel composition that reorganises in the same direction as the synthetic result?
A yes strengthens claim 1 substantially. A no is publishable as a boundary on it. Either is worth
four pages of an ICLR paper; neither requires the $2\times2$.

---

## 1. The fence

The NeurReps artifact is frozen. Concretely:

- **New directory `cifar_pilot/`**, sibling to `src/`, with its own `results/`, `scripts/` and
  `tests/`. Nothing in it writes above itself.
- **`src/` is imported read-only.** The estimator, the attribution, the provenance stamping and the
  noise-floor machinery are reused unchanged. If the pilot needs a change to `src/`, that is a
  finding to report, not a patch to apply — a modified `src/` invalidates the manifest hash on every
  NeurReps figure, and `scripts/make_figures.py --check --submission` will say so.
- **`results/arms/`, `results/width/`, `figures/`, `docs/07-writeup.md`, `docs/08-inventory.md` are
  not written to.** Not appended to, not re-summarised.
- **Separate result-set hash.** Pilot arms are never loaded by `src.analysis.grid.load()` without an
  explicit `arms_dir`, which the existing registration guard already enforces and already refuses
  silently-mixed loads for.
- **The pooling firewall applies.** Synthetic and CIFAR arms are different populations. After four
  corrections in this project traceable to comparing differently-pooled quantities (§A.1, §A.4),
  a number pooled across the two would be the fifth and the worst.

---

## 2. The design decision that has to be made first

Split-CIFAR100 does not come with independent feature and readout similarity. Three ways to build
the stream, and they are not equally good:

| option | feature similarity | readout similarity | verdict |
|---|---|---|---|
| **A. Standard split**, 100 classes into 10 tasks of 10, random assignment | uncontrolled, set by which classes co-occur | uncontrolled | **Recommended for the pilot.** Answers the single question in §0 and nothing else. |
| **B. Similarity-stratified split** — use CIFAR-100's 20 coarse superclasses to build high- and low-feature-similarity streams | manipulated, 2 levels, confounded with semantic category | controllable via label permutation | The interesting version, and the one to build *after* A returns a usable arm. |
| **C. Full $2\times2$ replication** | requires a feature-similarity axis orthogonal to everything else | controllable | **Do not attempt.** On real images the feature axis cannot be set independently of class identity, so the corners would not be the corners. Claiming a replication of the $2\times2$ here would be a comparison between differently-constructed populations wearing the same four labels. |

**Recommendation: A for the pilot, B as the paper.** The pilot exists to calibrate cost and confirm
the geometry is measurable and the forgetting is resolvable. Option B's design work is cheap and can
be done while A runs; option C is the trap.

---

## 3. What the architecture has to be

**Corrected 2026-08-13, after this section was first written.** The original version said there is
no GPU and built the whole recommendation on that. That was wrong in the way that matters: it was
true of *this* machine and not of the resources available. Recording both facts, because the
distinction is the point.

- **This host is `a8000-2409n4`:** `nvidia-smi` fails (no driver), 256 CPU cores, 503 GB RAM.
- **GPUs 0--3 are available on `grime`**, a different host. Kati confirmed.
- **There is no `torch` in this environment.** The entire current stack is NumPy, and the provenance
  system hashes NumPy modules.

So a learned convolutional encoder **is** feasible, on `grime`, and the recommendation below changes
accordingly. What does not change is that it is a bigger step than it looks: the pilot would run on
a different host from every existing arm, under a dependency the provenance system does not
currently hash, with a training loop that has never been exercised in this project. Each of those is
a place for a difference to enter that is not the difference being measured.

| input | cost | what it costs scientifically |
|---|---|---|
| **Raw pixels, 3072-d, into the existing MLP** | no new dependency, CPU-feasible here, no host change; input dim $150 \to 3072$ is a **$20.5\times$** increase in the first layer's work | Accuracy will be poor (order 30--40% on 10-way). **This remains acceptable and should be said plainly**: the question is whether the *geometry of forgetting* survives real input statistics, not whether the network is good at CIFAR. |
| **Learned conv encoder on `grime`** | feasible with GPUs 0--3; needs `torch`, a training loop, and provenance hashing extended to cover it | The scientifically strongest version, and the one an ICLR reviewer would expect. Also the one that changes the most variables at once relative to the NeurReps result. |
| **Frozen features from a pretrained encoder** | one-time `torch` dependency; per-arm cost close to current | Introduces a pretraining effect, which Graldi et al. treat as first-order, and makes part of the measured geometry the encoder's rather than the learner's. |

**Revised recommendation: raw pixels for the pilot, conv encoder on `grime` for the paper.** The
pilot's job is to establish that forgetting is resolvable on real data and to measure the cost; both
are answered by the cheap version, on this host, with the architecture held identical to the
synthetic result — which is what makes the comparison a comparison. Moving to `grime` and to `torch`
is then a deliberate second step with one thing changing at a time, rather than the pilot and the
architecture change being confounded from the first arm.

A weak learner that forgets measurably answers the pilot's question; the accuracy caveat is real and
is cheaper to state than the pretraining confound is to control.

---

## 4. The one measured arm, and what it decides

Per the standing rule and the three prior misses: **one arm, measured, before any projection.**

**Arm spec.** Option A stream, $T = 10$ tasks $\times$ 10 classes, $\gamma_0 = 10$ (the most
expensive richness and the one where the synthetic effect is largest), single seed, single stream.
Manifolds are classes: $P = 16$ sampled classes, $M = 150$ exemplars each, $N = 300$, $n_t = 200$ —
every estimator parameter held at its current registered value so that per-eval cost is comparable
to a number we already have.

**Recorded before anything is concluded:**

1. **Wall time, split into training and measurement.** This is the number the arm exists for. The
   26 h overrun was a cost model that measured per-eval cost single-threaded and assumed linear
   scaling; the lesson recorded then was that cutting the grid is a weak lever and the solver is a
   strong one. Raw pixels move the *training* side by $20.5\times$ and the measurement side barely
   at all, which inverts the current cost structure — so the split has to be measured, not assumed.
2. **Whether forgetting is resolvable at all.** Retained capacity change at the longest available
   lag, in units of the measured noise floor, with the floor **re-measured on CIFAR manifolds**. The
   registered synthetic floors do not transfer: floor magnitude depends on the manifold geometry,
   and $R_{\mathrm{eff}}$'s floor was already found to be $\gamma$-dependent by $6.7\times$ within
   this project.
3. **Whether the estimator is in its valid regime.** The §2a mean-field validity gate, and
   $\rho_c$ inside the calibration window. Real-image class manifolds are not Gaussian and are not
   isotropic; if $\rho_c$ lands outside the window where the synthetic calibration was fitted, the
   center-attribution panel does not transfer and must be dropped rather than clipped — this is the
   exact fault recorded in §A.2 as "out-of-domain inputs were clipped rather than refused".
4. **Convergence.** Fraction of tasks reaching target loss. If a substantial fraction do not, the
   forgetting measurement is confounded by never having learned, and the arm has failed in a way
   that a magnitude would hide.

**Gates on that one arm.** Any of these means stop and report rather than scale up:

- Forgetting at the longest lag is under 2 measured floors — nothing to decompose.
- The §2a validity gate fails, or $\rho_c$ is outside the calibration window.
- Fewer than 90% of tasks converge.
- Measured arm cost exceeds 4 h — at which point the grid in §5 needs re-sizing before it is queued,
  not after.

---

## 5. Cost estimate, and which half of it is trustworthy

**The measurement half is calibrated.** From `results/LOG.md`:

| quantity | value | source |
|---|---|---|
| per-eval, `colgen`, uncontended, $P{=}16$, $M{=}150$, $N{=}300$ | 21.3 s | measured on Phase 1 data |
| contention factor at 128 workers | $4.72\times$ | measured, memory-bandwidth bound |
| per-eval, contended | $\approx 100$ s | derived |
| throughput at 128 workers | $\approx 4{,}600$ evals/h | derived |
| evals per arm (synthetic, $T{=}16$) | 38 | 48,640 / 1,280 |

At $T = 10$ the evaluation count per arm falls, since retained capacity is read for each past task at
each boundary: roughly 20 evals per arm against 38. Holding $P$, $M$, $N$ and $n_t$ fixed keeps
per-eval cost in the measured regime, because the estimator's working set is unchanged. So the
**measurement** side of a 24-arm grid is $24 \times 20 = 480$ evals, about **6 minutes of wall time
in one 128-worker wave** — negligible, and the same conclusion the width arms reached.

**The training half is not calibrated and will not be projected here.** Raw pixels multiply the
first layer's work by $20.5\times$, CIFAR-100 has 500 exemplars per class against synthetic sampling
on demand, and the number of steps to target loss on real data is unknown and probably larger. Any
number written here would be a projection of exactly the kind that was wrong by $9\times$ in June.
**The one measured arm produces it.**

**Provisional grid, to be re-sized after the arm.** 4 stream instantiations $\times$ 3 richness
levels ($\gamma_0 \in \{0.1, 1, 10\}$) $\times$ 2 seeds = 24 arms, one 128-worker wave, wall time
$\approx$ one slowest arm. If the measured arm is under 2 h this is a same-day run; if it is 4 h the
grid is queued overnight; if it is more, the richness sweep is cut to $\{1, 10\}$ before anything is
queued.

---

## 6. What is NOT built

- No $2\times2$ similarity design (§2, option C).
- No width sweep. The synthetic width result bounds a dependence rather than measuring one; there is
  no reason to spend real-data compute reproducing a null.
- No behavioural-optimum or $\gamma^*$ claim. $\gamma^*$ was never identified in the synthetic grid
  (bootstrap CI $[0.339, 4.426]$), and a second unidentified location does not make a pair.
- No peak-location claim of any kind. The synthetic peak moves monotonically with lag, so the
  quantity is ill-posed without fixing the lag, and that argument is not data-dependent.
- No cross-module CKA. That needs retained activations and is a separate fenced item.
- No pooling with synthetic arms, ever (§1).

---

## 7. Order of work when this is greenlit

1. `cifar_pilot/` skeleton, data loader, Option A stream construction, tests.
2. Re-measure the noise floors on CIFAR manifolds. Nothing is quoted in floors until this exists.
3. **One arm.** Report §4's four items and the §4 gates before anything else is queued.
4. Only then: re-size and run the §5 grid.
5. Option B's similarity-stratified design, drafted while (3) runs since it costs no compute.

The step most likely to be skipped under time pressure is (2), and it is the one that makes every
subsequent magnitude meaningful. The floors are not optional and they do not transfer.
