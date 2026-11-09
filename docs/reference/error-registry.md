# Error registry

Every **code** error found in this project, with the fix and the test that guards it.

**Policy** (`docs/25` §5, amended 2026-11-06):
- Each row's job is a fix in code with a test.
- The registry does not drive reruns. Old results are signal, not baselines (`docs/24` §0a), and no
  old arm is rerun.
- A row whose test column says "none found" is a coverage gap.
- A row whose fix column says **open** is a live defect.

Rows are added, never edited in place. A later correction is a new dated note under the row's
number.

**Conventions:**
- **found:** the date, then the place it was recorded. LOG means `results/LOG.md`.
- **fixed:** the date, then the commit.
  - Git history starts on 2026-08-26 with a bulk import (`2be1728`). A fix logged before that date
    is marked "pre-history". It lives in `2be1728`, but its commit cannot be shown.
  - Several undated LOG sections sit between the 2026-08-11 and 2026-08-13 headers. They are marked
    "~08-11/12".

Compiled 2026-11-09 (backlog item 2, `docs/25` §8 step 4). Sources:
- `results/LOG.md`;
- `docs/07-writeup.md` App. A;
- `src/analysis/ledger.py::CORRECTIONS`;
- `docs/16` amendments;
- `docs/reference/choices-register.md`, `docs/reference/protocol-deviations.md`;
- git.

Commits and test names were checked against the tree.

> **Coverage caveat.** `docs/25` §5 asks to "add every item listed in the 2026-11-06 analysis (about
> 20 beyond the seed)". That list is not in the repo. The 11-06 LOG entry (lines 4693–4731) records
> only four doc-vs-code facts, and they are not code errors. The table below was rebuilt from the
> sources above. Kati should confirm that it covers the 11-06 list.

## 1. Code errors that changed or could change a number

| # | error | found | fixed | guarding test | affects |
|---|---|---|---|---|---|
| E1 | Training stopped on train **accuracy**; the spec (`01` §1) said matched loss. Accuracy is ≈ 0.99 after one step at every γ, so training halted before feature learning. | 2026-08-11, LOG 611 | 2026-08-11, pre-history | `tests/test_train_loop.py::test_accuracy_saturates_at_step_one_so_stopping_must_use_loss` | Phase 0 check 3, first pass (rerun under matched loss). No grid existed yet. |
| E2 | `src` `train_xy` checked the loss **after** the update. Every task took one update past criterion, `steps_taken` ≥ 1 even at criterion, and `final_loss` was not the loss of the returned weights (choices register C5). | 2026-11-08, C5 | 2026-11-09, `8bc9a49` | `tests/test_train_loop.py::test_task_at_criterion_takes_zero_steps_and_leaves_weights`, `::test_final_loss_is_the_loss_of_the_returned_weights`, `::test_stops_at_the_first_update_count_at_target`, `::test_fixed_steps_applies_exactly_that_many_updates`, `::test_grads_then_apply_grads_equals_step` | Every `src`-trained arm before 11-09 ends one step past criterion. Step-count readings: H3 warp, recurrence Δsteps, steps-to-target tables, lr0 calibration. `aba_lib` has its own correct loop and is unaffected. |
| E3 | GLUE ensembles drew `y` and `t` from **one RNG stream**, so comparisons across β mixed Monte-Carlo noise in `t` (~10% of α at n_t = 150) into the effect. | 2026-08-11, LOG 228 | 2026-08-11 (`rng.spawn(2)`), pre-history | none found (indirect: `tests/test_glue_core.py::test_tilt_interpolates_generic_to_retained`) | Pre-grid β comparisons (tilted vs retained). |
| E4 | `timewarp.measured_warp_scale` returned the **inverse** of the convention `residual` uses. | 2026-08-11, LOG 819 | 2026-08-11, pre-history | `tests/test_analysis_timewarp.py::test_measured_warp_uses_loss_not_a_fit`, `::test_pure_rate_rescaling_is_detected_and_undone` | H3 time-reparameterisation test. |
| E5 | DTW required coverage in **either** series, not both. A static trajectory could match the other's flat opening (residual 1.18; 11.54 after the fix). | 2026-08-11, LOG 821 | 2026-08-11, pre-history | `tests/test_analysis_timewarp.py::test_genuinely_different_shapes_are_not_warped_away`, `::test_non_overlapping_supports_are_infinite_not_silently_zero`, `::test_dtw_is_no_worse_than_the_best_rate_warp` | H3 verdict (a false "H3 is dead"). |
| E6 | **Stale fork.** Pilot workers ran the pre-change solver because `fork` handed them the parent's imported module. | 2026-08-11, LOG 1039 | 2026-08-11 (`spawn`, provenance stamps), pre-history | `tests/test_provenance.py::test_a_forked_child_inherits_the_stale_stamp`, `::test_hash_is_captured_at_import_not_recomputed`, `::test_editing_a_registered_module_is_detected` | Pilot timings. Geometry is unaffected (colgen = reference to 3.7e-15). |
| E7 | **`rho_c_glue` calibration clip.** `radius_from_rho` was fitted on unnormalised `rho_c_glue`, which reaches 1.49. `np.clip(rho, 0, 0.995)` silently returned finite numbers (centre-collapse shares 15–38). | 2026-08-11, LOG 1057; `07` §A.2 | 2026-08-11 (refit on `rho_c_signed`, refusal outside `RHO_FIT_RANGE`), pre-history | `tests/test_numerics.py::test_the_rho_c_bug_would_now_be_caught_at_the_clip`; `tests/test_analysis_attribution.py::test_the_calibration_uses_the_normalized_convention`, `::test_out_of_domain_rho_raises_rather_than_clamping`, `::test_rho_outside_the_fitted_range_declines_to_report` | Δρ_c → R_eff conversion; Fig 2(d) centre-collapse share (pilot draft only). |
| E8 | **`\|ρ_c\|` absolute-value convention.** Centre correlation was recorded as \|ρ_c\|, which cannot represent direction, so H1d and H6 were untestable. It was inherited from the predecessor project. | 2026-08-10/11, LOG 36, 113; `07` §A.3 | 2026-08-11: signed `rho_c_signed` from the first `src/glue/core.py`, pre-history | `tests/test_glue_core.py::test_rho_c_signed_detects_anticorrelation_that_abs_hides`, `::test_rho_c_conventions_are_distinct_and_correct` | Δρ_c sign, H1d/H6, Fig 4. No result in this repo was found to have been produced under \|ρ_c\|. |
| E9 | Two clips absorbed **real violations** along with rounding: `alignment.py` `arccos(clip(s, −1, 1))` and `core.py` `maximum(res.x, 0)` on the duals. | ~08-11/12, LOG 1171; `07` §A.2 | same pass (`numerics.clip_to_noise`), pre-history | `tests/test_numerics.py::test_a_real_violation_raises_instead_of_being_clipped`, `::test_float_error_is_absorbed_silently`, `::test_principal_angles_rejects_non_orthonormal_input` | None observed (both at ~1e-16). |
| E10 | Channel shares were drawn where the **total was below resolution** (γ 0.03/0.1 at 0.1/0.5 floors showed utility 0.52/0.60). | ~08-11/12, LOG 1305 | same pass (`MIN_FLOORS` ≥ 2-floor gate), pre-history | none found | Fig 2(b), low-γ shares. |
| E11 | `attribution.MIN_DLOG_R` was **one constant** (CV 0.50%). The R_eff floor depends on γ (0.28–1.13%), so the guard was 2.3× too permissive at γ = 10. | 2026-08-12, LOG 2551 | 2026-08-12 (`PER_GAMMA_FLOOR_CV`, `min_dlog_R_for`), pre-history | `tests/test_analysis_attribution.py::test_the_radius_floor_is_per_gamma_and_defaults_conservatively`, `::test_the_radius_floor_does_not_touch_the_three_terms` | Fig 2(d) coverage and medians; R_eff excursions quoted in floors. |
| E12 | **`abs(Δlog α)` in plotting.** Fig 2(a) plotted `G.floors()`, which returns \|Δlog\|/floor, under "capacity lost". The S-HH gain was drawn as a loss. | ~08-12, LOG 2833; `docs/09` §0; `07` §A.3 | ~08-12 (signed panel), pre-history | none found | Fig 2(a); pooled magnitude 50.7 vs 69.7 floors; any sign read from figures. |
| E13 | `grid.load()` **silently adopted off-design arms**. `Phase1Spec.key` has γ but not N. | ~08-12, LOG 2469 | same pass (refusal), pre-history | `tests/test_grid_offdesign.py::test_unregistered_width_is_refused`, `::test_unregistered_gamma_is_refused`, `::test_unusable_off_design_arm_still_refused` (+3) | Latent; no adopted arm reported. |
| E14 | The peak-stability check used a **range test**, which cannot see the monotone slide 2.86 → 0.96 across lags, so it reported "stable". | ~08-12/13, LOG 3158, 3327; `07` §A.2 | same pass (monotonicity flag, `scripts/tier2_backward_transfer.py`), pre-history | none found | §5.1 peak-location claim. |
| E15 | **`stream_id` stored, never read.** `make_stream` ignored `cfg.stream_id`, and `run_arm` passed `paired_init(seed)["stream"]`, so the seed keyed both init and arrangement. | 2026-08-17, LOG 3594; `07` §A.6 | 2026-08-17 (`pipeline.stream_rng(stream_id)`, new arms only), pre-history | `tests/test_pipeline.py::test_stream_id_keys_the_arrangement_not_the_init_seed` | See the list after this table. |
| E16 | **Pairing across s_r assumed, not true.** `make_stream` draws dichotomies before arrangements from one generator, and `sample_at_hamming` uses a variable number of draws. So the task-0 arrangement differs across s_r. | 2026-10-28, LOG 4434 | partly: opt-in `rng_dichotomies` in `eebb6bf` (2026-11-02). **The default is unchanged.** `docs/25` §1 requires it on for the core. | `tests/test_manifolds_dichotomies.py::test_separate_dichotomy_rng_pairs_task0_across_readout_similarity`, `::test_default_stream_unchanged_by_dichotomy_rng_option` | Paired readings across Hamming/s_r: `results/hamming_slice.*` (extra variance, no bias); probably `adam_hamming_slice` and order's forward schedule. |
| E17 | **Forgetting formula.** `pipeline.forgetting_metrics` used the peak `acc[j,j]` (a_{i,i}) instead of the running max over t ∈ {i..T−1} of a_{t,i}. | 2026-09-01 (`cl-metrics.md`; protocol-deviations D.2; `07` §A.7) | 2026-11-06, `698e1c4` | `tests/test_pipeline.py::test_forgetting_peak_is_running_max_when_accuracy_recovers`, `::test_forgetting_peak_excludes_final_row_so_backward_transfer_is_negative` | Behavioural CF/CFr on every arm before 11-06, wherever past-task accuracy recovers (S-HH, drift). |

**E15, the sets affected.** Correcting the `docs/25` §5 seed row:
- the 960-arm a = 0 grid: 40 files are 8 unique draws. Its SEM, CI, sign-test p, the stream R² = 0.000 and the peak CI are all affected.
- the γ = 5 n16 set: 4 unique seeds × 4 copies (`results/gamma_5_n16_streamid_unused/`).
- the 2×2 width set W1: 2 unique seeds (`results/width/`).
- **added here:** the γ = 30 probe, 4 unique per corner (LOG 3764; `results/gamma30_unique.json`).
- **probably:** the 320 a > 0 arms of Phase 1. They went through the same keying; the LOG has no uniqueness count for them.

The n = 40 sets (`gamma_5_n40`, `width_g10_n40`, `unconfound_k3`, `gamma5_k8`) are post-fix reruns.

## 2. Infrastructure, tooling and provenance errors (no measured value changed)

| # | error | found | fixed | guarding test | affects |
|---|---|---|---|---|---|
| I1 | The `run_phase1` default branch **deleted every arm** (`--resume` was opt-in). A resume test wiped the 8 pilot arms. | 2026-08-11, LOG 1198 | same day (resume by default, `--fresh` moves to `.trash-*`), pre-history | `tests/test_phase1_resume.py::test_the_default_invocation_does_not_delete_existing_arms`, `::test_fresh_moves_to_trash_rather_than_deleting` | 8 pilot arms lost; summary kept. |
| I2 | `run_scaling.py` measured at n_t = 20; the grid runs at 200. | 2026-08-11, LOG 1139 | same day, pre-history | none found | `results/scaling.json` (superseded); the 128-worker cap. |
| I3 | `audit_scope` v1 grepped whole files. It false-FAILed on docstrings and could miss real machinery. | ~08-11/12, LOG 1252 | same pass (AST, identifiers only), pre-history | none found | `results/audit_scope.json`. |
| I4 | The reachability audit filtered imports to `continual_geometry/`, so it would have skipped a sibling-project import. | ~08-12, LOG 1524 | same pass, pre-history | none found | audit_scope "no a1b2 machinery". |
| I5 | Fig 2(d) printed the floor CV under a label reading "gate" and always drew the hatched-bar legend. `fig_width_invariance.py` titles hard-coded "2% spread" and "0.440–0.460". | ~08-12, LOG 2831, 2457 | same pass, pre-history | none found | Figure labels and titles only. |
| I6 | `make_figures.py` wrote `module_hashes: {}`, because figure scripts run as subprocesses. The `fig_width` sidecar lacked `result_set_sha` and `n_arms`. | 2026-08-13, LOG 3221, 3227 | same day, pre-history | none found | `figures/MANIFEST.json` provenance. |
| I7 | `check_submission.py`: `strip_comments()` on the `.bib` hid `% TODO-VERIFY` (FAIL → PASS). Its page counter returned 0 on compressed object streams. | 2026-08-13, LOG 3398, 3457 | same day, pre-history | `tests/test_check_submission.py::test_an_unverified_bib_field_fails_even_though_its_marker_is_behind_a_percent` (+15); page count: none found | Submission checklist only. |
| I8 | `v3_lib.calibrate` first encoded the Adam re-pin lr0 rule instead of the Phase 0 practice. | 2026-10-29, LOG 4478 | before any rung was read, `eebb6bf` | none found | None (scratch run deleted unread). |
| I9 | `parallel_map` set the BLAS thread variables **after** NumPy was imported. Workers under Linux `fork` would run multithreaded. | 2026-11-02, LOG 4593 | `eebb6bf` (variables set before `import numpy`). `parallel_map` still sets them late; this is redundant. | none found | Linux/fork runs; none of the 270 laptop arms. |
| I10 | `code_digest` hashes bytes, and `src` was CRLF on disk, so a Linux checkout hashed differently. | 2026-11-02, LOG 4607 | `.gitattributes` (`0e65c3f`), LF normalisation `f4265a6` (2026-11-06) | none found | v3 cache digest, cross-machine resume. |
| I11 | **Dirty-tree stamps.** All 270 v3 arms are stamped dirty, and `figures/MANIFEST.json` at 443945dc was `git_dirty: true`. | 2026-11-02, LOG 4547; LOG 3232 | No code fix. A–B–A arms carry clean stamps and a dirty-tree guard from `52f5567` on. | none found | The code state of the v3 270 and the 08-13 figure set cannot be proved. |
| I12 | The arm timer uses `time.time()`, which counts laptop suspend. 11 arms show 71–74k s, so the 282.8 h sum is not a compute cost. | 2026-10-30, LOG 4562 | **open** (`notebooks/v3_lib.py`). `aba_lib` uses `perf_counter`. | none | `wall_seconds` in the v3 cache; no measured values. |
| I13 | `sbatch/run.sh` and `sbatch/timing.sh` still run `ENTRYPOINT_PLACEHOLDER` / `TIMING_ENTRYPOINT_PLACEHOLDER`. | 2026-11-03, LOG 4658 | **open** | none | Job 3570 failed after 1 s; no arm ran. |
| I14 | **cp1252 decode on Windows.** Repo JSON is read with `read_text()` without `encoding="utf-8"`: `src/analysis/ledger.py` (~25 sites), `src/analysis/grid.py`, `src/reservations.py`. Tests fail unless `PYTHONUTF8=1`. | 2026-10-28, LOG 4463; LOG 4683 | **open**, backlog item 4 | none yet | Windows test runs and `grid.load` (they crash; no values are wrong). |

## 3. Not code errors (recorded elsewhere)

These were considered and left out on purpose, so the omission is visible.
- **Documentation and pre-commit errors.**
  - `docs/28` §5 quoted t ≈ 4.03 where the rule gives 3.636. The erratum is `a86a2e4`, choices
    register C12. The code used 3.636.
  - The `docs/26` statements the 11-06 review found wrong about the code: L(0), `weight_change`
    from init, NTK/CKA code, `MODULES`.
  - The `docs/16` A11 "100×" slip.
  - The width-row values S-LH +0.0922 and S-LL +0.0210, which were read from the wrong table.
- **Design and threshold choices that were later changed** (in `docs/07` App. A, the ledger or the
  choices register):
  - the 1σ → 3σ floor gate;
  - four-corner pooling;
  - \|term\|/Σ\|term\| shares;
  - the A.8 statistics;
  - the A.9 operating-point obstructions;
  - the under-training step budget, which is guarded by
    `tests/test_train_loop.py::test_under_training_is_flagged_not_silent`;
  - frozen s_f = 1 re-realising points (A–B–A now uses an exact copy, C3);
  - the anchored y* forward-transfer baseline (C6, replaced by neutral probes in `ee2c1b6`).
- **Third-party and provenance:**
  - the vendored `manifold_simcap_analysis` draws its own labels with a global seed; an adapter
    handles it (LOG 262);
  - v3 stamps hash only `(core, pipeline, attribution)` (LOG 4573);
  - the `NOISE_FLOOR_CV` copy is a drift risk only (`tests/test_noise_floor_provenance.py`).

## 4. Open items

- **I14 (encoding):** backlog item 4.
- **E16:** `rng_dichotomies` has to be on by default for the core (`docs/25` §1). It is currently
  opt-in.
- **I12, I13:** small fixes. Each needs a test or a removal.
- **Coverage gaps:** the rows in §1 marked "none found" (E3, E10, E12, E14). Tests should be added
  where the code still exists.

## 5. Updates (dated; rows above are not edited)

- **2026-11-09, I14 fixed.** All 309 text-mode `open` / `read_text` / `write_text` calls in `src/`,
  `scripts/`, `tests/` and `notebooks/core/aba_cluster.py` now pass `encoding="utf-8"`. They were
  rewritten mechanically with an AST-guided insertion and checked by re-parsing. `aba_lib.py` had no
  such call and is unchanged (hash `4e110835b8fc2863`). Guard:
  `tests/test_encoding.py::test_every_text_io_call_names_its_encoding`. Before the fix, the suite had
  29 failures without `PYTHONUTF8=1`. After it: 287 passed, 1 skipped, without the variable.
- **2026-11-09, E18 (new): replication timing cell hard-coded seed 0.**
  - **Error.** `aba_controlled.ipynb` cell 17, the timing display, built `ArmSpec(0, …)`. The
    `seed_offset` change (`80044e4`) replaced seed 0 in cells 5, 6, 13, 14, 19 and 31, but missed
    this one.
  - **Found.** 2026-11-09, by `notebooks/core/aba_summary.py`: the replication checkpoint
    `FULL_4e110835b8fc2863.jsonl` holds 220 rows, one of them seed 0, γ0 0.1, h 8, jump, stamped
    `8407a7d`.
  - **Affects.** No reading. The readings cell filters on the run's spec keys (`WANT`), and docs/28
    §9.3 re-derives exactly from seeds 8–15 alone (`aba_summary.py --check`). The only effect: the
    replication *computed* one seed-0 arm, and the notebook printed it as the timing arm.
  - **Fixed.** 2026-11-09: `ArmSpec(S0, …)`. The notebook is not hashed, so the lib hash is
    unchanged.
  - **Test.** None. Notebook cells are not under `tests/` (coverage gap).
