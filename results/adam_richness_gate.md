# Adam richness gate — frozen × s_r=0.5, unique n=8, task 0

Training-only. Mean ‖ΔW‖/‖W‖ (module A): γ=10 **1.8357**, γ=1 **0.5758**, ratio **3.188** against a one-decade bar.

Reading: **`partial_manipulation`**.

Mean steps: γ=10 24.0, γ=1 501.5 (step ratio γ=1/γ=10 = 20.90).

Missed target: 0 / 16. Wall 42.1 s.

The 192-arm slice launches only on `clears_decade`. `partial_manipulation` is a finding, not a failed check: step counts separate 20.9× while weight change separates 3.19×. Do not launch 192. The arm that can still run is Hamming-only at γ=10 (`results/adam_hamming_precommit.json`).

Source: `results/adam_richness_gate.json`.

Identity: stream 10000 γ=10 `‖ΔW‖/‖W‖` = 1.8291011753468978, bitwise match
to the one-arm task-0 record.

| stream | γ=10 ΔW/W (steps) | γ=1 ΔW/W (steps) |
|---|---|---|
| 10000 | 1.829 (24) | 0.576 (504) |
| 10001 | 1.839 (24) | 0.587 (509) |
| 10002 | 1.852 (24) | 0.584 (513) |
| 10003 | 1.841 (24) | 0.576 (498) |
| 10004 | 1.852 (24) | 0.577 (499) |
| 10005 | 1.912 (26) | 0.580 (521) |
| 10006 | 1.772 (23) | 0.562 (487) |
| 10007 | 1.789 (23) | 0.563 (481) |

192-arm runner refuses: `partial_manipulation`.
