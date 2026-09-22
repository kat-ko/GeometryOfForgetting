# src/glue/adapters — corrections layer over vendored estimators

Vendored code in `third_party/` is kept **byte-identical to upstream** (see
`third_party/VENDORED.md`). This layer is where we make it conform to
`docs/AGENTS.md` §4 without mutating the pinned source.

## What every adapter here must own

1. **RNG isolation (AGENTS.md §4).** Upstream uses global `np.random.seed(...)`:
   - `correlated_capacity/capacity/manifold_simcap_analysis.py:173`
   - `correlated_capacity/capacity/sphere_sim_capacity.py:194`
   Adapters must accept an explicit `numpy.random.Generator` and route all
   stochastic draws through it. No global seeding, no bare `np.random.*`.
2. **float64** throughout the geometry pipeline (pseudo-inverses of near-singular
   Gram matrices; float32 is insufficient).
3. **Explicit `rcond`** on every pseudo-inverse, with the effective rank logged.
4. **κ = 0** asserted on entry to any capacity call (see `docs/00-math-spec.md` §6).

## Planned adapters (not yet implemented)

| Adapter | Wraps | Emits |
|---|---|---|
| `replica_mft.py` | `replicaMFT.mftma.manifold_analysis_correlation.manifold_analysis_corr` | `α_mf, R, D, ρ_c` |
| `sim_capacity.py` | `correlated_capacity.capacity.manifold_simcap_analysis` | `α_sim` (general point-cloud, κ=0) |
| `correlated_capacity.py` | `correlated_capacity.capacity.replica_correlations` | duality checks |

Implementation is deferred until the estimator is wired (Phase 0). This README
fixes the contract so the correspondence between `third_party/` and its pinned SHA
is never broken by a "quick fix" in vendored source.
