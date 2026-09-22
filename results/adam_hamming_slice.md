# Adam Hamming-only slice (γ=10)

Unique n=8, module A, task 0, lag 12. 95% CI = mean ± 1.96 SEM. Adam, MSE, `lr0=0.0002`. **This arm cannot speak to γ.**

**n_arms = 96, usable = 96, missed = 0.**

**Finding 3 recovery** (s_r=1, γ=10): NO — stop. drift -0.3 fl, jump -88.4 fl.

**Primary reading: `finding3_fails_to_recover`.** Stop. The Adam streams are not the same object. Hamming-axis flags were computed and are not a reading.

## Cliff, post-cliff range

Cliff = floors at s_r=1.0 minus floors at 0.75 (1-decimal). Post-cliff range = |floors(0.75) − floors(0.25)|.

| input | cliff (1.0 → 0.75) | range 0.75 → 0.25 | 0.75 vs 0.25 CI |
|---|---|---|---|
| frozen | +69.5 | 5.9 | overlap |
| drift | +86.5 | 17.4 | **resolved** |
| jump | +18.8 | 3.1 | overlap |

## Crossover

s_r=0.75: drift -86.8, frozen -63.4. s_r=0.25: drift -104.2, frozen -69.3.

## Δ log α, task 0 lag 12, γ=10

| input \ s_r | 1.00 | 0.75 | 0.50 | 0.25 |
|---|---|---|---|---|
| frozen | +6.1 fl  (+0.1124 [+0.0719, +0.1529]  n=8  8+/0-) | -63.4 fl  (-1.1744 [-1.2935, -1.0554]  n=8  0+/8-) | -64.7 fl  (-1.1982 [-1.2582, -1.1382]  n=8  0+/8-) | -69.3 fl  (-1.2847 [-1.3515, -1.2178]  n=8  0+/8-) |
| drift | -0.3 fl  (-0.0055 [-0.0452, +0.0343]  n=8  3+/5-) | -86.8 fl  (-1.6087 [-1.6777, -1.5397]  n=8  0+/8-) | -95.2 fl  (-1.7642 [-1.8517, -1.6766]  n=8  0+/8-) | -104.2 fl  (-1.9310 [-2.0563, -1.8057]  n=8  0+/8-) |
| jump | -88.4 fl  (-1.6375 [-1.7242, -1.5509]  n=8  0+/8-) | -107.2 fl  (-1.9857 [-2.0960, -1.8754]  n=8  0+/8-) | -101.3 fl  (-1.8774 [-2.0088, -1.7460]  n=8  0+/8-) | -104.1 fl  (-1.9283 [-2.0965, -1.7601]  n=8  0+/8-) |

## Δρ_c, retained task 0, lag 12 (reported, not a reading)

| input \ s_r | 1.00 | 0.75 | 0.50 | 0.25 |
|---|---|---|---|---|
| frozen | -0.0154  [-0.0296, -0.0013]  2+/6- | -0.1488  [-0.1650, -0.1326]  0+/8- | -0.1803  [-0.1981, -0.1626]  0+/8- | -0.1998  [-0.2172, -0.1824]  0+/8- |
| drift | -0.0028  [-0.0134, +0.0078]  3+/5- | -0.0105  [-0.0367, +0.0157]  3+/5- | -0.0131  [-0.0333, +0.0070]  2+/6- | -0.0208  [-0.0384, -0.0031]  2+/6- |
| jump | +0.1386  [+0.1183, +0.1588]  8+/0- | +0.1264  [+0.0991, +0.1536]  8+/0- | +0.1120  [+0.0879, +0.1360]  8+/0- | +0.1102  [+0.0939, +0.1265]  8+/0- |

Cannot speak to: γ; finding 1; channel reorganisation; γ-dependence of cliff sizes; a second learner agreeing; the original five 192-arm readings.

Source: `results/adam_hamming_slice.json`. Pre-commit: `results/adam_hamming_precommit.json`.
