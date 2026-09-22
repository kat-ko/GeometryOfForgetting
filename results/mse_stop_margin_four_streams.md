# MSE-stop readout margin — four reserved streams

Frozen × s_r=0.5, seed=0, γ ∈ {1, 10}, stream_ids 10000–10003. Training-only
replay. All eight `steps_taken` lists match stored Hamming JSON. Wall 252 s.

`m` = mean current-task `y f` at the MSE-stop state. Tolerance for the BCE
pilot is the range of these eight cell means, not a number we choose.

| γ | streams | cell-mean m | p05 | p25 |
|---|---|---|---|---|
| 10 | 10000–10003 | 0.8527–0.8582 | 0.30–0.34 | 0.71–0.73 |
| 1 | 10000–10003 | 0.8308–0.8373 | 0.33–0.34 | 0.66–0.67 |

- grand mean **0.8442**
- band **[0.8308, 0.8582]**
- spread **0.0274**

Within-γ spread is ~0.006. Between-γ is the rest: the two γ sit at
opposite ends of the band. Homogeneous conversion of 0.05 MSE remains
0.684, ~20% shallower than this measurement.

p05 stays ~0.33 against mean ~0.84. Report mean/p05/p25 on the BCE
pilot; do not re-pin on the tail.

Source: `results/mse_stop_margin_four_streams.json`. Not frozen. No
`ce_precommit.json`.
