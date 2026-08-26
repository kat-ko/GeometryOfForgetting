# 15 — What was actually run: network, optimiser, representations, design

Compiled 2026-08-19 from the code and the registered specs, not from the paper's
narrative. Every statement below is pinned to a file. Where a registered option
exists but was not used, that is stated. This document is the place to rest
interpretation; `docs/14-claim-security.md` is the place to rest *which claims
the numbers support*.

Sources: `src/models/network.py`, `src/models/parameterization.py`,
`src/train/loop.py`, `src/pipeline.py`, `src/manifolds/{generator,streams}.py`,
`src/glue/core.py`, `src/analysis/attribution.py`, `scripts/run_phase1.py`,
`docs/00-math-spec.md`, `docs/01-experiments.md`, `docs/06-scope.md`.

---

## 1. Network

**What it is.** A two-layer ReLU network with a scalar readout, implemented in
numpy float64 with analytic gradients (`src/models/network.py`):

```
h_m(x) = ReLU(β₀ W_m x),     W_m ∈ R^{N×d},  m ∈ {A, B}
f(x)   = Σ_m (β_L / γ_eff) · u_mᵀ h_m(x)
```

- Depth: one hidden layer. Not a deep net, not a conv net, not a transformer.
- Width: `N = 300` per module, input `d = 150` (`Phase1Spec`).
- Two modules, **identical configuration**, summing into **one shared scalar
  readout**. No task-conditioned head (`I8`). `u_m(0) = 0`, so `f(x; θ₀) = 0`
  exactly (`I5`).
- Nonlinearity: ReLU, and only there. The readout is linear in `h`.
- Parameterization: μP of Graldi et al., `parameterization="mup"` (the default;
  NTP is implemented and unused). Weight variance `σ² = 1`. Hidden weights drawn
  from a γ-independent RNG, so `W_m(0)` is bitwise identical across richness
  (`I6`; Phase 0 confirmed representations bitwise identical at init).
- Parameter count: 2 × (`N d + N`) = 2 × 45,300 = **90,600** parameters.
- `06-scope.md`: this is configuration **C2**, a homogeneous control. It is not
  a modularity experiment. Module A and module B are two halves of the same
  network. At `a = 0` they are independent Gaussian draws; at `a > 0` they are
  bitwise identical for the whole run. Headline results are at `a = 0`.
- Single-module C1, heterogeneous C3/C4, frozen-B C5, and i.i.d. joint-training
  C6 are registered in `01` and **were not run**.

**Limits this imposes on interpretability.**

1. **ReLU is piecewise-linear.** Geometry is of the ReLU image of the input
   manifolds, not of a deep nonlinear hierarchy. A change in `D_eff` or `R_eff`
   is a change in how those 4-spheres sit after one ReLU, not evidence about
   CNN/transformer representations.
2. **Capacity here is not parameter-count, not VC dimension, and not test
   accuracy.** The measured `α` is replica-MFT *manifold capacity* of the hidden
   representation (GLUE; Chou et al.): how many balanced dichotomies of the `P`
   manifolds a linear readout on `h` can realise, in the thermodynamic-limit
   estimator. It is a property of `{h(x) : x ∈ manifold_μ}`. It is not "the
   network has capacity N". Finite-`N` corrections exist; the estimator is
   validated at `N = 1000` in Gate 2a and used at `N = 300`.
3. **The identity `α = Ψ_eff (1 + R_eff⁻²) / D_eff` is exact for this
   estimator**, not a model of the network. Interpreting a term as "the
   representation aligned / inflated / collapsed in dimension" is an
   interpretation of GLUE factors, licensed only as far as GLUE's geometry
   matches the trained ReLU image. Phase 0: `pairwise` vs `full_P` agree on `α`
   to ~1% and **diverge on `D_eff` / `Ψ_eff` by 20–25%**. Primary measurement is
   `full_P`. Published GLUE geometries from the pairwise convention are not
   numerically comparable to these `D_eff` numbers.
4. **Scalar binary readout.** Every task is one balanced dichotomy, labels
   constant on a manifold. Nothing here is multi-class, nothing is a structured
   output, nothing is a next-token prediction.
5. **Finite width, one depth, one nonlinearity.** No claim about "neural
   networks" in general survives a change of any of those. The μP γ₀ knob is
   specific to this scaling.

---

## 2. Optimisation

**What it is.** Full-batch gradient descent on MSE, sequential per task
(`src/train/loop.py`, `TwoModuleNet.sgd_step`):

- Loss: `L = mean_b ½ (f(x_b) − y_b)²`, `y_b ∈ {±1}`. Not cross-entropy, not
  logistic. It is regression to the dichotomy.
- Update: `W ← W − η ∇W L`, `u ← u − η ∇u L`, same `η` on both (`00` §4.2:
  changing one without the other changes contribution magnitude, not richness).
- Batch: `batch_size=None` → the whole task, `P · M = 2,400` points. The method
  is named SGD in the config schema; the run is **full-batch GD**.
- Learning rate (μP, `lr_scaling="quadratic"`, the default and the only setting
  used):

      η = lr0 · γ₀² · (N / N_base),   lr0 = 5.0,  N_base = 64

  At the design point `N = 300` this is `η ≈ 23.44 · γ₀²`, so η ≈ 0.021 at
  γ₀ = 0.03 and η ≈ 2.3×10³ at γ₀ = 10. Output scale shrinks as `1/γ_eff` at
  the same time. **The Atanasov "corrected" scaling `γ^(2/L)` for γ > 1 is
  implemented and was not used.** `00` §4.3 flags Graldi's γ² as misspecified
  in the rich regime; that robustness arm does not exist in the result set.
- Stopping: `matched_loss`, `target_loss = 0.05`, cap `steps_per_task = 20,000`.
  Phase 0: train accuracy is unusable (from `u=0`, step 1 is a kernel readout
  and accuracy jumps to ~0.99 at every γ). Loss is what matches progress.
  `lr0 = 5.0` was pinned because `lr0 = 0.2` left γ ≤ 1 at loss ~0.34. **Any
  change of `M`, `P`, or `N` invalidates `lr0` and `target_loss`.**
- First gradient step moves **only the readout** (`∇W ∝ u`, `u(0)=0`). Feature
  learning starts at step 2.
- No momentum, Adam, weight decay, dropout, replay, EWC, or other continual-
  learning regulariser. Tasks are trained in order; the weights of task `t−1`
  are the initialisation of task `t`.
- 0 of 1,280 registered-grid arms failed to reach the loss target.

**Limits this imposes.**

1. **Matched loss is not matched compute.** Steps-to-target span ~39× across γ
   (Phase 1 comment in `run_phase1.py`). Rich arms take fewer steps. A γ contrast
   is a contrast of *regime at matched loss*, not of wall-clock or of step count.
   Phase 0 time-warp: warping trajectories together still leaves a residual of
   11.5 floors (γ=1 vs 10), so it is not "the same path at different speed" —
   but it is also not "the same number of updates".
2. **MSE to ±1 is not classification risk.** Capacity and accuracy can disagree
   (they do, in `S-HH`: capacity gains while behavioural forgetting is ~0).
   Interpreting Δ log α as "how much the network forgot the task" is already a
   step: the behavioural readout is sign accuracy, and it is recorded, but the
   paper's currency is capacity.
3. **Full batch, convex-in-`u` at frozen `W`.** The readout subproblem is a
   linear least squares on ReLU features. Non-convexity is only in `W`. Claims
   about SGD noise, minibatch implicit regularisation, or Adam geometry do not
   apply.
4. **The γ² learning-rate law is a modelling choice, flagged in the spec as
   possibly wrong for γ ≫ 1.** Results at γ₀ = 10 sit in the regime where
   Atanasov says the exponent should be 1 (L=2). We do not know what the
   channel shares would be under the corrected law.
5. **No claim about other continual-learning methods.** There is no replay
   baseline, no regularisation baseline, no i.i.d. joint-training control (C6
   was cut). Sequential GD on MSE is the only optimiser in the result set.

---

## 3. Representations: what is measured, how, and what they are

**The object.** For each module, each measured boundary, and each past task (or
the generic ensemble), the representation is

    { h_m(x) ∈ R^N  :  x ∈ manifold_μ of that task's arrangement },   μ = 1…P

stored as `P` arrays of shape `(N, M)` = `(300, 150)`. That is the ReLU hidden
state of **one module**, not the scalar output, not concatenated modules (except
where `module=None`, which the pipeline does not use for GLUE).

**What the inputs to that map are.** Each task's arrangement is `P = 16`
isotropic **spherical manifolds** of intrinsic dimension `D = 4` and radius
`R = 1` in `R^{150}`, plus isotropic noise `ε = 10⁻²` (Chou et al. ICML 2025
App. D.1.1; `src/manifolds/generator.py`). Centers and axes ~ `N(0, I_d/d)`.
`ρ_A = 0`, `ψ_gen = 0` (no extra center–axis or axis–axis correlation beyond
what consecutive `s_f` imposes by redrawing centers). Consecutive tasks share
a **controlled center correlation** `s_f ∈ {0.1, 0.9}`; axes are redrawn with
the centers.

**What they look like, factually.** We do not have a visualisation in the result
set. What can be said without inventing one:

- At init, `W` is i.i.d. Gaussian, so `h` is a random ReLU feature map of 16
  small 4-spheres in 150-D. Phase 0: this map is bitwise identical across γ.
- After training, richness is read out as `‖ΔW‖_F / ‖W(0)‖_F`. Phase 0 at
  matched loss: 0.0015 (lazy) → 0.503 (rich), 2.53 decades. In the lazy regime
  the representation barely moves; in the rich regime about half the Frobenius
  mass of `W` has changed.
- The paper never plots the point clouds. It plots GLUE summaries of them.

**How they are measured.** GLUE (`src/glue/core.py`, `full_P` mode, `n_t = 200`,
`center_policy="all"`):

1. Draw 200 random directions `t ~ N(0, I_N)` (fixed across boundaries of one
   arm — the measurement RNG is keyed by `(seed, module, task)`, not by
   boundary, so a change is the representation moving, not the estimator
   resampling).
2. For each `t` and a dichotomy `y`, solve the capacity QP; record anchors.
3. Average into `α`, `D_eff`, `R_eff`, `Ψ_eff`, and two ρ_c conventions.
4. Two ensembles: **retained** (`y` frozen at the trained dichotomy of that past
   task) and **generic** (`y` uniform over balanced dichotomies). Attribution
   of forgetting uses retained. H2d used a probe margin, not generic α as the
   probe itself.

**Identity used for forgetting:**

    Δ log α = Δ log Ψ_eff  +  Δ log(1 + R_eff⁻²)  −  Δ log D_eff

named in the paper as alignment, radius, dimension. Residual is at machine
precision (4.4×10⁻¹⁶). `ρ_c` sits *outside* this identity; a synthetic
calibration converts Δρ_c into a share of the radius term and is refused
outside the fitted ρ range.

**Limits this imposes.**

1. **We measure a linear-readout geometry of ReLU features of synthetic
   4-spheres.** We do not measure "concepts", "features" in the CNN sense, or
   the network's output geometry (the output is a scalar).
2. **Generic vs retained is an ensemble choice, not two networks.** Same `h`,
   different `Y`. A rise in generic α with richness (H2a failed: it rises) is
   not a rise in "generalisation" in the held-out-data sense; it is a rise in
   how many random dichotomies of these 16 manifolds a linear readout on `h`
   can realise.
3. **`n_t = 200` is Monte Carlo inside one evaluation**, not independent
   experimental trials. Dispersion of that estimator is the "floor" everything
   is denominated in (α CV ~1.9%; `R_eff` CV 0.28–1.13% across γ, 4 seeds).
4. **No CCGP in the codebase.** The config schema lists `ccgp_enabled: true`;
   `src/` contains no CCGP. Factor structure exists in the labeling
   (`P = 16 = 2^4`) and is **not** used as the training tasks — tasks are
   random balanced dichotomies with Hamming-controlled consecutive similarity.
   Abstraction-over-factors is untested.
5. **Pairwise GLUE (lab standard) is not the reported geometry.** `full_P` is
   mandatory after Phase 0 found a 9.7-floor disagreement on Δ log D_eff.

---

## 4. Experimental setup: inputs, outputs, trials, runs

### One task

| | value | source |
|---|---|---|
| Input points | `P × M = 16 × 150 = 2,400` vectors in `R^{150}` | `Phase1Spec` |
| Structure | 16 manifolds × 150 points, D=4, R=1, ε=10⁻² | `generator.py` |
| Output | one balanced dichotomy `y ∈ {±1}^{16}`, **constant on each manifold** | `loop.py` |
| Loss targets | the 2,400 labels `repeat(y, M)` | `flatten_task` |
| Consecutive control | `s_f` (center correlation) and `s_r` (Hamming of dichotomies) | `streams.py` |
| Not controlled | the full `T×T` similarity; off-diagonal `s_f` is a realized cosine | `streams.py` |

### One stream (one arm's data)

- `T = 16` tasks in order.
- Dichotomies: `y_0` random balanced; `y_t` at Hamming distance set by `s_r`.
- Arrangements: `A_0` fresh; `A_t` redraws centers correlated at `s_f`.
- Held-out **probe** dichotomy `y*`, not among the training `y_t` (up to sign),
  used for H2d / probe margin, not for training.
- Condition is the Hiratani 2×2:

  | ID | s_f | s_r |
  |---|---|---|
  | S-HH | 0.9 | 0.9 |
  | S-HL | 0.9 | 0.1 |
  | S-LH | 0.1 | 0.9 |
  | S-LL | 0.1 | 0.1 |

- Factor labeling is constructed and unused as a task family. `S-fixed-r`,
  blocked/shuffled, XOR, single-factor, Split-CIFAR: registered or mentioned,
  **not in the result set**.

### One arm (one run)

`Phase1Spec`: `(γ₀, a, condition, stream_id, seed)` plus the frozen design
point `(T, P, d, M, D, R, N, n_t, lr0, target_loss, …)`.

Sequence: for `t = 0…15`, GD on task `t` to loss 0.05; then, if `t` is a
measurement boundary `{0, 4, 8, 12, 15}`, GLUE on each tracked past task
`j ∈ {0, 4, 8, 12} ∩ [0, t]` and on the generic ensemble, for modules A and B.
That is the "widening lag" design: task 0 is seen at lags 0, 4, 8, 12, 15;
task 12 only at lag 0 and 3.

**"Trials" in this project are not a single n.**

| unit | what it is | n on the registered grid |
|---|---|---|
| arm / file | one trained stream | 960 at `a=0` (1,280 with `a` sweep) |
| unique seed | `paired_init(seed)`; on the grid this also drew the arrangement | **8 per (γ, condition)** |
| `stream_id` | intended arrangement draw; **unread on the grid** | 5 labels, 0 information |
| task | one dichotomy in the stream | 16 per arm |
| geometry evaluation | one (module, boundary, task-or-generic) GLUE call | 40 per arm (budget) |
| `n_t` | Monte Carlo directions inside one GLUE call | 200 |
| measurement seed | RNG for those directions, derived from `seed` | 1 per (module, task) per arm |

Headline inference on the grid is at **unique n = 8**, with arrangement and
initialisation confounded. After the RNG fix, `stream_id` draws the arrangement
and `seed` draws `W(0)` and the measurement seed. See `docs/14-claim-security.md`.

CIFAR was specified (`docs/10-cifar-pilot-spec.md`) and **not run**.

---

## 5. Along which axes anything was varied

### Varied in the registered grid (what Figures 2 and 4 rest on)

| axis | values | notes |
|---|---|---|
| richness γ₀ | 0.03, 0.1, 0.3, 1, 3, 10 | μP; log-spaced ×~3.33 |
| stream condition | S-HH, S-HL, S-LH, S-LL | consecutive (s_f, s_r) only |
| seed | 0…7 | init (and, on the grid, arrangement) |

Homogeneous `a ∈ {0.5, 1}` at γ₀ = 1 only: 320 files, **not in the headline
forgetting numbers**, and at `a > 0` modules A and B are copies.

### Varied off-design (appendix / robustness; not pooled with the grid)

| axis | values | n / design | status |
|---|---|---|---|
| width N | 150, 600 at γ₀=10 | 5 arr × 8 init (RNG-fixed) | bound: 0.62%; two widths |
| γ₀ | 5 | 8 arr × 8 init (pre-committed) | arrangement-dependent, 4/8 |
| γ₀ | 30 | 4 unique × 4 copies | fenced; unique n=4 |
| arrangement (unconfound) | 3 arr × 40 init, γ∈{1,10} | 960 arms | leading pattern 3/3 |

### Held fixed (and therefore not licensed as general)

| axis | frozen at | consequence |
|---|---|---|
| architecture | two-module ReLU, depth 2, shared scalar readout | not "networks" |
| width (grid) | N = 300 | width arm is a bound at two other values, γ=10 only |
| input dim, P, M | d=150, P=16, M=150 | |
| manifold D, R, ε | 4, 1, 10⁻² | no D-sweep; 4-spheres only |
| stream length | T=16 | lag and position confounded |
| dichotomy family | Hamming-controlled random balanced | not factors, not XOR, not class-incremental |
| consecutive similarities | {0.1, 0.9} only | no 0.5, no continuum |
| optimiser | full-batch GD, MSE, μP γ² LR | no Adam, no corrected LR |
| stopping | matched loss 0.05, lr0=5 | not matched steps |
| estimation | GLUE `full_P`, n_t=200, center_policy=all | not pairwise, not CKA-as-primary |
| data | synthetic spheres | not images, not language |

### Registered and cut (do not interpret silence as a negative result)

C1, C3, C4, C5, C6; Phase 2 (H3, H5); Phase 3 (H4, long stream); CIFAR;
`S-fixed-r`; blocked/shuffled; `lr_scaling="corrected"` robustness; CCGP;
gaussianized capacity mode as a reported channel.

---

## 6. What a finding is allowed to mean, given §1–5

A licensed finding is a statement about **this** network, **this** optimiser,
**this** synthetic stream, **this** estimator. The geometry claims are about
GLUE factors of ReLU images of 4-spheres under sequential full-batch MSE.

They are not, without a further argument that this document does not contain:

- a statement about deep nets, conv nets, or trained-from-pixels representations;
- a statement about SGD noise or adaptive optimisers;
- a statement about class-incremental image benchmarks;
- a statement that "capacity" means anything other than GLUE α of these
  manifolds;
- a statement that two modules did anything modular (they did not; C2 is a
  control, and C3 was cut);
- a statement that γ₀ = 5 is where decorrelation begins (arrangement-level
  4/8; see `docs/14-claim-security.md`).
