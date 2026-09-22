# Hamming × input-change slice

Unique n=8, module A, task 0, lag 12 unless noted. 95% CI = mean ± 1.96 SEM. The arm worked. The raw table says more than the pre-committed readings.

**n_arms = 192, usable = 192, missed = 0.**

**Finding 3 recovery** (s_r=1, γ=10): yes. drift +6.7 fl, jump -55.1 fl.

## The structure is a cliff, not a dose

Cliff = floors at s_r=1.0 minus floors at 0.75 (1-decimal). Post-cliff range = |floors(0.75) − floors(0.25)|.

### γ=10

| input | cliff (1.0 → 0.75) | range 0.75 → 0.25 | 0.75 vs 0.25 CI |
|---|---|---|---|
| frozen | +75.3 | 0.7 | overlap |
| drift | +68.0 | 13.3 | **resolved** |
| jump | +21.8 | 0.7 | overlap |

### γ=1

| input | cliff (1.0 → 0.75) | range 0.75 → 0.25 | 0.75 vs 0.25 CI |
|---|---|---|---|
| frozen | +35.0 | 0.4 | overlap |
| drift | +36.4 | 10.0 | **resolved** |
| jump | +12.3 | 2.5 | overlap |

Task change is a threshold at zero everywhere except under drift, where a graded component appears and is resolved (drift 0.75 vs 0.25 CIs do not overlap at either γ). How much the task changed only matters when the input distribution is also changing slowly.

## Crossover

Frozen is flat and drift is graded, so they cross. At γ=10, s_r=0.75: drift -61.3, frozen -68.2 (drift better). At s_r=0.25: drift -74.6, frozen -68.9 (frozen better). A slowly drifting input helps when the task barely changes and hurts when it changes a lot. That is why `frozen_outside` fired: frozen is insensitive to task change while drift is not.

## Pre-commit readings (historical; they failed to discriminate)

- Primary as written: `mixed` (γ=10 `nonmonotone`, γ=1 `monotone_dose`).
- Frozen level: `frozen_outside` — fired correctly; the reason is the crossover above.
- `monotone_dose` fires on a step-then-flat series, so it cannot distinguish a dose from a threshold.
- `mixed` fires on a 0.2-floor unresolved wobble in γ=10 frozen (−69.1 → −68.9, CIs overlap). Both γ show the same cliff-plus-drift-gradation. Going forward: a monotonicity reading requires the post-threshold range to clear the floor.

## γ=10, Δ log α, task 0 lag 12

| input \ s_r | 1.00 | 0.75 | 0.50 | 0.25 |
|---|---|---|---|---|
| frozen | +7.1 fl  (+0.1321 [+0.1220, +0.1422]  n=8  8+/0-) | -68.2 fl  (-1.2642 [-1.4074, -1.1209]  n=8  0+/8-) | -69.1 fl  (-1.2805 [-1.3790, -1.1820]  n=8  0+/8-) | -68.9 fl  (-1.2772 [-1.4048, -1.1496]  n=8  0+/8-) |
| drift | +6.7 fl  (+0.1248 [+0.0950, +0.1546]  n=8  8+/0-) | -61.3 fl  (-1.1356 [-1.2166, -1.0546]  n=8  0+/8-) | -70.3 fl  (-1.3026 [-1.4046, -1.2007]  n=8  0+/8-) | -74.6 fl  (-1.3816 [-1.4905, -1.2727]  n=8  0+/8-) |
| jump | -55.1 fl  (-1.0203 [-1.1056, -0.9350]  n=8  0+/8-) | -76.9 fl  (-1.4250 [-1.4904, -1.3596]  n=8  0+/8-) | -78.0 fl  (-1.4447 [-1.5056, -1.3838]  n=8  0+/8-) | -77.6 fl  (-1.4386 [-1.5557, -1.3215]  n=8  0+/8-) |

## γ=1, same cell

| input \ s_r | 1.00 | 0.75 | 0.50 | 0.25 |
|---|---|---|---|---|
| frozen | +1.0 fl  (+0.0191 [+0.0182, +0.0201]  n=8  8+/0-) | -34.0 fl  (-0.6295 [-0.6619, -0.5972]  n=8  0+/8-) | -34.0 fl  (-0.6299 [-0.6825, -0.5772]  n=8  0+/8-) | -34.4 fl  (-0.6372 [-0.6997, -0.5748]  n=8  0+/8-) |
| drift | +12.7 fl  (+0.2361 [+0.2160, +0.2563]  n=8  8+/0-) | -23.7 fl  (-0.4390 [-0.5098, -0.3682]  n=8  0+/8-) | -33.3 fl  (-0.6167 [-0.6953, -0.5382]  n=8  0+/8-) | -33.7 fl  (-0.6250 [-0.6802, -0.5698]  n=8  0+/8-) |
| jump | -14.0 fl  (-0.2586 [-0.2976, -0.2196]  n=8  0+/8-) | -26.3 fl  (-0.4863 [-0.5189, -0.4538]  n=8  0+/8-) | -29.5 fl  (-0.5459 [-0.6056, -0.4861]  n=8  0+/8-) | -28.8 fl  (-0.5331 [-0.5934, -0.4728]  n=8  0+/8-) |

## Δρ_c, retained task 0, lag 12 (signed; no ρ_c floor)

Finding 4's recast now has four Hamming levels. Capacity and centre geometry come apart: frozen capacity is a cliff, frozen Δρ_c is graded and the 0.75 vs 0.25 CIs do not overlap. Drift capacity is graded; drift Δρ_c is a cliff then unresolved. Jump Δρ_c is already large at s_r=1. Populations named, not pooled: registered 2×2 drift at a repeated task is +0.055; reserved Hamming drift at s_r=1 is +0.002 (CI includes 0, 5+/3−). Capacity on the reserved same cells recovered (+6.7 vs −55.1). Δρ_c did not.

### γ=10

| input \ s_r | 1.00 | 0.75 | 0.50 | 0.25 |
|---|---|---|---|---|
| frozen | -0.0076  [-0.0104, -0.0048]  0+/8- | -0.0525  [-0.0644, -0.0406]  0+/8- | -0.0674  [-0.0776, -0.0572]  0+/8- | -0.0868  [-0.1036, -0.0699]  0+/8- |
| drift | +0.0017  [-0.0068, +0.0103]  5+/3- | +0.0679  [+0.0427, +0.0930]  7+/1- | +0.0746  [+0.0586, +0.0907]  8+/0- | +0.0788  [+0.0605, +0.0972]  8+/0- |
| jump | +0.1039  [+0.0933, +0.1145]  8+/0- | +0.1101  [+0.0979, +0.1223]  8+/0- | +0.1065  [+0.0935, +0.1194]  8+/0- | +0.1131  [+0.0996, +0.1266]  8+/0- |

### γ=1

| input \ s_r | 1.00 | 0.75 | 0.50 | 0.25 |
|---|---|---|---|---|
| frozen | -0.0013  [-0.0015, -0.0011]  0+/8- | -0.0203  [-0.0253, -0.0154]  0+/8- | -0.0236  [-0.0275, -0.0196]  0+/8- | -0.0297  [-0.0346, -0.0248]  0+/8- |
| drift | -0.0227  [-0.0270, -0.0185]  0+/8- | +0.0163  [+0.0098, +0.0228]  7+/1- | +0.0309  [+0.0278, +0.0341]  8+/0- | +0.0360  [+0.0312, +0.0408]  8+/0- |
| jump | +0.0341  [+0.0312, +0.0370]  8+/0- | +0.0346  [+0.0297, +0.0395]  8+/0- | +0.0369  [+0.0336, +0.0402]  8+/0- | +0.0400  [+0.0359, +0.0442]  8+/0- |

## Lag 4 across task position (γ=10, floors)

First output from schedule B. W5b on the registered grid was ×1.7 (tasks 0/4/8 at lag 4). Here the ratio |task 0 / task 8| is cell-dependent: ~1.2–1.4× on most forgetting cells, ×3.5 on jump at s_r=1, and inverted (later tasks gain more) on drift at s_r=1.

| input | s_r | t=0 | t=2 | t=4 | t=6 | t=8 | t=10 | |t0/t8| |
|---|---|---|---|---|---|---|---|---|
| frozen | 1 | +3.3 | +2.8 | +2.3 | +1.9 | +1.6 | +1.4 | 2.00× |
| frozen | 0.75 | -72.4 | -64.6 | -63.9 | -53.4 | -53.9 | -53.3 | 1.34× |
| frozen | 0.5 | -68.9 | -60.3 | -66.2 | -62.2 | -57.3 | -55.8 | 1.20× |
| frozen | 0.25 | -75.3 | -66.2 | -63.5 | -51.1 | -57.5 | -55.2 | 1.31× |
| drift | 1 | +8.0 | +12.1 | +13.5 | +10.8 | +11.4 | +11.6 | 0.70× |
| drift | 0.75 | -54.5 | -54.4 | -50.8 | -42.9 | -39.2 | -45.1 | 1.39× |
| drift | 0.5 | -70.8 | -57.3 | -66.0 | -57.0 | -52.5 | -53.5 | 1.35× |
| drift | 0.25 | -74.0 | -69.8 | -62.2 | -54.0 | -58.3 | -60.0 | 1.27× |
| jump | 1 | -39.1 | -27.5 | -15.9 | -16.3 | -11.3 | -7.5 | 3.45× |
| jump | 0.75 | -45.6 | -33.2 | -29.5 | -24.7 | -18.4 | -21.1 | 2.48× |
| jump | 0.5 | -45.8 | -35.9 | -31.4 | -31.7 | -32.9 | -29.0 | 1.39× |
| jump | 0.25 | -55.7 | -36.2 | -45.2 | -34.4 | -27.6 | -27.3 | 2.02× |

P=32 is a motivated cliff-resolution arm (h=2 is 6.25% of labels vs 12.5% at P=16). It still needs its own §2a gate and is a scope change. Not next. Next is a second learner (binary cross-entropy) on these streams; see `docs/21-second-learner-ce.md`.
