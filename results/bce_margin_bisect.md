# BCE margin bisect — no shared operating point

MSE four-stream band: **[0.8308, 0.8582]**, spread 0.0274, grand mean 0.844.
Eight rounds on the Hamming one-arm cells (γ=10 and γ=1, frozen × s_r=0.5,
stream 10000). `lr0=5` reached every candidate. Both-in-band: never.

Reading: **`no_shared_operating_point`**. No pin. No `ce_precommit.json`.
No slice.

| round | L | m γ=10 | p05 | p25 | m γ=1 | p05 | p25 | gap | both in band |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.517 | 0.598 | −0.51 | 0.04 | 0.543 | −0.58 | 0.03 | 0.055 | no |
| 2 | 0.433 | 0.937 | −0.47 | 0.22 | 0.865 | −0.59 | 0.19 | 0.072 | no |
| 3 | 0.475 | 0.766 | −0.50 | 0.12 | 0.697 | −0.60 | 0.10 | 0.069 | no |
| 4 | 0.454 | **0.850** | −0.49 | 0.16 | 0.777 | −0.60 | 0.14 | 0.073 | no (γ=1 below) |
| 5 | 0.444 | 0.890 | −0.48 | 0.19 | 0.820 | −0.60 | 0.17 | 0.070 | no |
| 6 | 0.449 | 0.868 | −0.49 | 0.17 | 0.799 | −0.60 | 0.15 | 0.068 | no |
| 7 | 0.446 | 0.878 | −0.48 | 0.18 | 0.810 | −0.60 | 0.16 | 0.068 | no |
| 8 | 0.445 | 0.888 | −0.48 | 0.19 | 0.815 | −0.60 | 0.16 | 0.073 | no |

When γ=10 sits in the MSE band (round 4), γ=1 is 0.07 below it. The BCE
between-γ gap at matched L is ~0.07 against an MSE band 0.027 wide. One
number cannot place both.

p05 is negative at every L tried (−0.47 to −0.60). MSE-stop p05 was
~+0.33. At the L that matches γ=10's *mean* to MSE, a tail of points is
on the wrong side of `f=0`, and p25 is ~0.16 against MSE ~0.71. That is
the distribution caveat firing, not a reason to re-pin. The mean already
failed to match across γ.

One-arm probe L=0.35: mean 1.31, p05 −0.33, 80 steps, 15 s, `lr0=5`
reached. Wall for the bisect: 1190 s.

Do not pin per-γ. The CE arm cannot test stream vs learner without a
shared operating point. The failure is the measurement: the two
objectives cannot be placed at comparable progress. That sentence
replaces §9's unmeasured "MSE to ±1 is not classification risk" for this
architecture, these streams, and this `lr0`.

Sources: `results/bce_margin_bisect.json`,
`results/bce_margin_one_arm.json`,
`results/mse_stop_margin_four_streams.json`.
