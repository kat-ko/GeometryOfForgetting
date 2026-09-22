# Reference sheet — μP base-width normalisation (Graldi)

```
Source:       Graldi, Breccia, Lanzillotta, Hofmann & Noci, "The Importance of
              Being Lazy: Scaling Limits of Continual Learning", ICML 2025 —
              arXiv 2506.16884v2, Table 1 + Appendix A.2, A.3, A.4.
Version:      arXiv:2506.16884v2
Derived from: Table 1 cells (human, 2026-09-01). Appendix A.2–A.4 pasted from
              the PDF (human, 2026-09-01). NOT copied from 00-math-spec.md.
Transcribed by: agent, 2026-09-01, from the human paste
Verified by (human): Kati, 2026-09-01 — Table 1 four cells, A.3 text, and the
                     independent derivation below (Verification 1 signed off).
Status: verified. A.3 does not state the constant. Table 1 caption's
        "Further details are in App. A.3" is a dangling reference. The
        (N/N_base) factors are an independent derivation from Table 1,
        confirmed against the agent's, unit-tested.
```

---

## Table 1 — signed 2026-09-01

| | NTP | μP |
|---|---|---|
| Branch scale `β_ℓ` | `N^{-1/2}` (ℓ>0), `D^{-1/2}` (ℓ=0) | same |
| Output scale `γ` | `1` | `γ₀ · N^{1/2}` |
| LR `η(t)` | `η₀(t)` | `η₀(t) · γ₀² · N` |
| Weight variance `σ_ℓ²` | `1` | `1` |

**Graldi's `D` is the input dimension** — our `d`. `D` means intrinsic manifold
dimension everywhere else here.

**`σ² = 1` in both.** I6 is confirmed at source.

Table 1 caption: μP is normalized to be equivalent to NTP at a base width of
`N = 64`; further details in App. A.3.

---

## A.3 — transcribed. The caption's forward reference is dangling.

A.3 is three sentences. It does **not** write `N/N₀`. The base-width
normalisation is stated **nowhere in the paper**.

> Note that many equivalent parameterizations can achieve the same functional
> behavior (Yang & Hu, 2020; Bordelon & Pehlevan, 2022; Yang et al., 2023;
> Bordelon et al., 2023). In Tab. 1 we report the notation of (Bordelon et al.,
> 2023, Tab. 1), but with the NTP from (Yang & Hu, 2020, Tab. 1) for
> implementational simplicity. Also, note that the standard parameterization
> (SP) of PyTorch (i.e. the SP column in (Bordelon et al., 2023, Tab. 1)) is
> equivalent to the NTP used here.

A.3 identifies the table (Bordelon 2023 notation, Yang–Hu 2020 NTP, PyTorch
SP ≡ this NTP). Verification 1 is therefore not "check the agent's derivation
against a stated constant." It is "derive independently and confirm the
agent's matches."

A.2 still supplies the number: "The base configuration uses `N = 64` and
`L = 6`."

---

## Independent derivation — from Table 1

Set μP equal to NTP and solve.

- Output scale: `γ₀ N^{1/2} = 1` ⇒ `γ₀ = N^{-1/2}`
- LR: `γ₀² N = 1` ⇒ `γ₀ = N^{-1/2}`

Both rows give **the same condition**. That consistency is the check that
matters: the two μP extra factors are not two constraints.

At `N₀ = 64` that is `γ₀ = 1/8`. To make `γ₀ = 1` the NTP point instead,
rescale `γ₀ → γ₀ / √N₀`, which is equivalent to **replacing `N` with `N/N₀`
in both the output scale and the LR**:

```
γ_μP  = γ₀ · (N / N₀)^{1/2}     →  1 at γ₀ = 1, N = 64
η_μP  = η₀ · γ₀² · (N / N₀)     →  η₀ at γ₀ = 1, N = 64
N₀    = 64
```

Held fixed as width scales: `η₀`. This is what
`src/models/parameterization.py` implements (`N_BASE = 64`). Pinned by
`tests/test_models_parameterization.py::test_base_width_equivalence`.

**The agent's derivation is confirmed.** Its stated reasoning — requiring
`μP(γ₀=1, N=64) ≡ NTP(N=64)` forces the width factors to be relative to the
base width — is the same algebra. The spec's `η = lr0 · γ₀² · (N/N_base)`
follows.

**Verification 1 signed off on this basis**, 2026-09-01.

**What would have been the N vs N₀ conflation.** Implementing Table 1's `N`
as raw width. At `N = 64`, `γ₀ = 1` that gives output scale 8 and LR `64 η₀`,
not NTP.

---

## NTP-equivalent γ₀ on our sweep (computed, not interpreted)

Under this normalisation the NTP-equivalent point at width `N` sits at

```
γ₀^{NTP}(N) = (N₀ / N)^{1/2}
```

At `N = 300`: `(64/300)^{1/2} = 0.461880… ≈ 0.462`.

That is the point on the sweep where the network behaves like PyTorch's
default parameterization (A.3: SP ≡ NTP).

Grid `{0.03, 0.1, 0.3, 1, 3, 10}`: **0.462 falls between 0.3 and 1.**

Relative to documented resolution thresholds, not as a coincidence:

- Forgetting: §5.1, resolvable from `γ₀ = 0.1` at 2.1 floors (appears between
  0.1 and 0.3). LOG also records first resolvable between 0.3 and 1.
- `Ψ_eff`: becomes resolvable (and negative) between `γ₀ = 0.3` and `1`
  (`results/LOG.md`; §5.1, channel reorganisation from 0.3 onward).

One line in the paper: where the standard parameterization sits on the sweep.
No coincidence claim. Three prior γ-coincidence readings in this project were
refuted; this number is analytically determined, so a reader can place it
without a CI.

---

## Depth

A.2's base depth is `L = 6` for their ResNet. A.3 does not extend Table 1 to
μP+1/√L. Sequential L=3 arms still wait on a separate human derivation.

---

## Setup alignment with Graldi's infinite-width arm (A.4)

Graldi's infinite-width arm is a **2-layer non-linear ReLU perceptron**,
full-batch GD on **MSE**, last layer initialized to **0** (Yang et al. 2022b
App. D.2), **SGD without momentum or weight decay**. That is our architecture,
loss, batching, readout init (I5), and optimizer (I1, I3). One sentence in the
paper: the setup is the same model class they use for the theoretical arm, not
an idiosyncratic simplification.

**Deviations to state:**

1. They use a cosine LR schedule without warmup, restarted at each task. We
   use constant LR with matched-loss stopping.
2. Their MLP arm is 30 MNIST samples, 2 tasks, similarity `ρ = 0`. `γ₀* ≈ 0.1`
   comes from their **ResNet** experiments, not the MLP. Any comparison of our
   `γ*` to theirs is across architectures.
