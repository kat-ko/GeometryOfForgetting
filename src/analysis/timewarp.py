"""Time-reparameterization test (`00` §12) — the H3 gate.

Atanasov et al. find that networks at different **large** `γ` optimize along
similar trajectories up to a reparameterization of time. If that holds here, two
modules at different large `γ` are one trajectory at two speeds and the
division-of-labour hypothesis is void.

The test asks whether a monotone warp `τ` can bring
`(α, R_eff, D_eff, ρ_c)(γ₁, t)` onto `(α, R_eff, D_eff, ρ_c)(γ₂, τ(t))` to within
the trajectory noise floor. Three warps of increasing generosity are tried:

1. **measured** — `τ(t) = t / c` with `c` the observed steps-to-target ratio
   between the two runs (`01` Phase 0 measured ≈ 39.5× across the full γ range).
   No free parameters, so this is the honest first test.
2. **best rate** — one free parameter, the scale that minimizes the residual.
3. **best monotone** — DTW, the most generous monotone warp there is.

The logic runs one way only. If even the *best monotone* warp leaves a residual
above the noise floor, the trajectories genuinely differ and H3 is testable. If
the measured warp already collapses them, H3 is dead. Anything between is
reported as-is: a warp with free parameters that fits is weaker evidence for
coincidence than the parameter-free one.

Residuals are in units of the **noise floor**, so 1.0 is the decision boundary
and the number is directly interpretable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .trajectories import CHANNELS, GeometryTrajectory

# Monte-Carlo CV per channel at n_t = 200 (results/cost_model.json). Converted to
# a log-space scale, which is what `GeometryTrajectory.matrix` returns.
#
# These are the *registered* floors, measured at one setting. They remain the right object for
# the warp test, which compares trajectories at two different γ and so cannot be denominated in
# either one's floor. Where a floor is applied within a single γ, use `PER_GAMMA_FLOOR_CV`.
NOISE_FLOOR_CV = {"alpha": 0.0187, "D_eff": 0.0126, "R_eff": 0.0050,
                  "rho_c_glue": 0.0097}

# Per-γ floors from the measurement null (`scripts/run_measurement_null.py`, W4): the same
# trained representation re-measured under several measurement seeds, at every registered γ.
#
# `alpha` and `D_eff` came back γ-flat within the design's own resolution (a CV from 4 seeds
# carries ~40% relative uncertainty, so ratios under ~2× are unresolved), which is why every
# floor-denominated headline in this project is α-denominated and stands as registered.
#
# `R_eff` did not: it rises monotonically with richness and reaches 2.3× the registered value
# at γ=10. Its only live consumer is `attribution.min_dlog_R_for`.
#
# `Psi_eff` had no registered floor. **This one is recorded, not used.** Every analysis to date
# ran the utility channel un-floored, and retro-fitting a gate would move published numbers for
# no gain; it is here so the next version of this work starts with it.
PER_GAMMA_FLOOR_CV = {
    "R_eff": {0.03: 0.00280, 0.1: 0.00307, 0.3: 0.00618,
              1.0: 0.00808, 3.0: 0.00715, 10.0: 0.01125},
    "Psi_eff": {0.03: 0.01152, 0.1: 0.01286, 0.3: 0.01660,
                1.0: 0.01831, 3.0: 0.02187, 10.0: 0.02395},
}


def floor_vector(channels: tuple[str, ...] = CHANNELS) -> np.ndarray:
    return np.array([np.log1p(NOISE_FLOOR_CV[c]) for c in channels])


@dataclass
class WarpResult:
    name: str
    residual_in_floors: float
    per_channel: dict[str, float]
    scale: float | None = None
    coincide: bool = False

    def to_dict(self) -> dict:
        return {"name": self.name, "residual_in_floors": self.residual_in_floors,
                "per_channel": self.per_channel, "scale": self.scale,
                "coincide": self.coincide}


def _resample(traj: GeometryTrajectory, grid: np.ndarray,
              channels: tuple[str, ...]) -> np.ndarray:
    """Interpolate a trajectory onto `grid` (step counts) in log-step, log-value."""
    src = np.log(traj.steps.astype(float))
    tgt = np.log(grid.astype(float))
    M = traj.matrix(channels)
    return np.column_stack([np.interp(tgt, src, M[:, j]) for j in range(M.shape[1])])


def _common_grid(a: GeometryTrajectory, b: GeometryTrajectory, scale: float,
                 n: int = 24) -> np.ndarray | None:
    """Steps where `a(s)` and `b(s/scale)` both have support."""
    lo = max(a.steps.min(), b.steps.min() * scale)
    hi = min(a.steps.max(), b.steps.max() * scale)
    if not (hi > lo):
        return None
    return np.geomspace(lo, hi, n)


def residual(a: GeometryTrajectory, b: GeometryTrajectory, scale: float,
             channels: tuple[str, ...] = CHANNELS) -> tuple[float, dict[str, float]]:
    """RMS distance between `a` and time-warped `b`, in units of the noise floor."""
    grid = _common_grid(a, b, scale)
    if grid is None:
        return float("inf"), {c: float("inf") for c in channels}
    A = _resample(a, grid, channels)
    B = _resample(b, grid / scale, channels)
    diff = np.abs(A - B) / floor_vector(channels)[None, :]
    per = {c: float(np.sqrt(np.mean(diff[:, j] ** 2)))
           for j, c in enumerate(channels)}
    return float(np.sqrt(np.mean(diff**2))), per


def measured_warp_scale(a: GeometryTrajectory, b: GeometryTrajectory,
                        target_loss: float) -> float:
    """Ratio of steps each run needs to reach a common loss — a warp with no fit.

    Convention matches `residual`, which compares `a(s)` against `b(s / scale)`:
    if `b` is 8× faster then `scale = 8`, i.e. `steps_to(a) / steps_to(b)`.
    """
    def steps_to(t):
        below = np.where(t.loss <= target_loss)[0]
        return float(t.steps[below[0]]) if below.size else float(t.steps[-1])
    return steps_to(a) / steps_to(b)


def best_rate_warp(a: GeometryTrajectory, b: GeometryTrajectory,
                   channels: tuple[str, ...] = CHANNELS,
                   n_scan: int = 241) -> tuple[float, float, dict[str, float]]:
    """One free parameter: the rate that minimizes the residual."""
    scales = np.geomspace(1e-3, 1e3, n_scan)
    best = (float("inf"), 1.0, {})
    for s in scales:
        r, per = residual(a, b, s, channels)
        if r < best[0]:
            best = (r, float(s), per)
    return best[1], best[0], best[2]


def dtw_residual(a: GeometryTrajectory, b: GeometryTrajectory,
                 channels: tuple[str, ...] = CHANNELS,
                 min_coverage: float = 0.5,
                 resolution: int = 40) -> tuple[float, dict[str, float]]:
    """Most generous monotone warp: DTW with a fixed start and a free end.

    **Fixed start** is legitimate rather than a convenience: the paired-init design
    gives every `γ` bitwise identical hidden weights and a zero readout, so all
    trajectories begin at the same geometry (verified — `01` Phase 0, "flat in γ").
    **Free end** is necessary: a faster run has travelled further along the same
    curve by the last checkpoint, so forcing the ends together would penalize the
    very rescaling being tested and make DTW *less* generous than a plain rate
    warp — the opposite of its purpose here.

    `min_coverage` requires the alignment to reach at least this fraction of both
    series, which stops the degenerate "match the first two points and stop"
    solution from reporting a near-zero residual.

    This is a lower bound on what *any* monotone reparameterization achieves, so a
    value above 1.0 is the strong form of "the trajectories genuinely differ".
    """
    # Densify first. DTW restricted to the raw checkpoints can only pair sampled
    # times, which at a large rate difference leaves it *less* free than the rate
    # warp (which interpolates onto a common grid). Interpolating both series to a
    # fine grid over their own support restores DTW's role as the upper bound on
    # what a monotone warp can do.
    A = _resample(a, np.geomspace(a.steps.min(), a.steps.max(), resolution), channels)
    B = _resample(b, np.geomspace(b.steps.min(), b.steps.max(), resolution), channels)
    f = floor_vector(channels)[None, :]
    n, m = len(A), len(B)
    cost = np.sqrt((((A[:, None, :] - B[None, :, :]) / f) ** 2).mean(axis=2))

    acc = np.full((n, m), np.inf)
    length = np.zeros((n, m), dtype=int)
    acc[0, 0], length[0, 0] = cost[0, 0], 1
    for i in range(n):
        for j in range(m):
            if i == 0 and j == 0:
                continue
            cands = []
            for pi, pj in ((i - 1, j - 1), (i - 1, j), (i, j - 1)):
                if pi >= 0 and pj >= 0 and np.isfinite(acc[pi, pj]):
                    cands.append((acc[pi, pj], length[pi, pj], pi, pj))
            if not cands:
                continue
            best = min(cands, key=lambda c: (c[0] + cost[i, j]) / (c[1] + 1))
            acc[i, j] = best[0] + cost[i, j]
            length[i, j] = best[1] + 1

    # Coverage is required in BOTH series. With `or`, a trajectory that barely
    # moves could be matched against the other's flat opening segment, satisfy
    # coverage on its own axis alone, and report a near-zero residual — a
    # degenerate warp masquerading as coincidence.
    i_min, j_min = int(min_coverage * (n - 1)), int(min_coverage * (m - 1))
    ends = [(i, j) for i in range(n) for j in range(m)
            if i >= i_min and j >= j_min and np.isfinite(acc[i, j])
            and (i == n - 1 or j == m - 1)]
    if not ends:
        ends = [(n - 1, m - 1)]
    i, j = min(ends, key=lambda e: acc[e] / length[e])

    path = []
    while (i, j) != (0, 0):
        path.append((i, j))
        cands = [(acc[pi, pj], pi, pj) for pi, pj in
                 ((i - 1, j - 1), (i - 1, j), (i, j - 1))
                 if pi >= 0 and pj >= 0 and np.isfinite(acc[pi, pj])]
        _, i, j = min(cands)
    path.append((0, 0))

    idx = np.array(path)
    diff = np.abs(A[idx[:, 0]] - B[idx[:, 1]]) / f
    per = {c: float(np.sqrt(np.mean(diff[:, k] ** 2)))
           for k, c in enumerate(channels)}
    return float(np.sqrt(np.mean(diff**2))), per


def time_reparameterization_test(
    a: GeometryTrajectory,
    b: GeometryTrajectory,
    *,
    target_loss: float = 0.05,
    channels: tuple[str, ...] = CHANNELS,
    min_excursion: float = 3.0,
) -> dict:
    """Run all three warps and return the verdict for one `(γ₁, γ₂)` pair.

    If either trajectory moves less than `min_excursion` noise floors over the
    whole window, the test is reported **vacuous** rather than as coincidence: a
    trajectory that does not move is trivially the slowed-down opening of any
    other, so "they coincide" carries no information about shape.
    """
    excursions = {f"gamma_{t.gamma}": t.excursion_in_floors(NOISE_FLOOR_CV, channels)
                  for t in (a, b)}
    vacuous = min(excursions.values()) < min_excursion

    results = []

    c = measured_warp_scale(a, b, target_loss)
    r, per = residual(a, b, c, channels)
    results.append(WarpResult("measured", r, per, scale=c, coincide=r <= 1.0))

    s, r, per = best_rate_warp(a, b, channels)
    results.append(WarpResult("best_rate", r, per, scale=s, coincide=r <= 1.0))

    r, per = dtw_residual(a, b, channels)
    results.append(WarpResult("best_monotone_dtw", r, per, coincide=r <= 1.0))

    # H3 survives only if *no* warp aligns them. Any one succeeding means the
    # trajectories coincide up to a reparameterization of time, which is the claim.
    # DTW's fixed start is valid for our runs (shared init) but not in general, so
    # it is not privileged over the others; the minimum decides.
    aligned = [w for w in results if w.coincide]
    if vacuous:
        verdict = ("VACUOUS — a trajectory barely moves, so coincidence is "
                   "uninformative about shape")
    elif aligned:
        verdict = (f"COINCIDE under the {aligned[0].name} warp — "
                   "H3 is dead in this gamma range")
    else:
        verdict = "DIFFER — no monotone warp aligns them; H3 is testable"

    return {"gamma_a": a.gamma, "gamma_b": b.gamma,
            "warps": [w.to_dict() for w in results],
            "min_residual_in_floors": min(w.residual_in_floors for w in results),
            "excursions_in_floors": excursions,
            "vacuous": bool(vacuous),
            "verdict": verdict,
            "h3_testable": bool(not aligned and not vacuous)}
