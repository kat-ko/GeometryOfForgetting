# MSE-stop readout margin (training-only replay)

Hamming one-arm cell: frozen, s_r=0.5, stream 10000, seed=0. γ ∈ {1, 10}.
No geometry. Identity: `steps_taken` matches the stored Hamming JSON on
both cells.

`m = mean_b [y_b f(x_b)]` on the current-task batch at the state
`train_task` returns. Naive homogeneous conversion of MSE 0.05 is
`1 − √0.1 ≈ 0.684` — an assumption, not this measurement.

| γ | mean m over 16 tasks | range | task 0 mean | task 0 p05 | mean p05 | steps match |
|---|---|---|---|---|---|---|
| 10 | 0.8527 | 0.779–0.873 | 0.779 | 0.463 | 0.323 | yes |
| 1 | 0.8316 | 0.745–0.848 | 0.745 | 0.444 | 0.342 | yes |

Not frozen. n=2 cells. The two task-averages differ by 0.021. Whether one
BCE number can hit both is the next pilot, not a gate fitted to this
table. Hard-point p05 sits far below the mean (~0.33 vs ~0.84); matching
the mean does not automatically match the tail. The pin remains the mean,
as written before these numbers.

Source: `results/mse_stop_margin_one_arm.json`. Wall 153 s. Do not write
`ce_precommit.json` from this file.
