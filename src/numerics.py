"""Clipping that checks its own premise.

The ρ_c calibration bug was not a wrong formula. It was `np.clip(rho, 0, 0.995)`, written
for numerical safety, applied to values that reached 1.49 — so a domain violation of
order 0.5 was absorbed exactly as if it were float error of order 1e-16, and returned a
plausible number. Center-collapse shares of 15–38 followed, and read as a result.

Every clip in a measurement path carries an unstated premise: *the violation being
absorbed is at noise scale*. When that premise holds, clipping is right and the
alternative (propagating a NaN from `arccos(1 + 2e-16)`) is worse. When it fails, the
clip manufactures a plausible value from an invalid one, and nothing downstream can tell.

`clip_to_noise` states the premise as a tolerance and fails when it is violated, so a
clip can no longer quietly change its own meaning from "absorb rounding" to "invent a
number".
"""

from __future__ import annotations

import numpy as np


def clip_to_noise(
    x: np.ndarray | float,
    lo: float | None,
    hi: float | None,
    *,
    atol: float = 1e-9,
    what: str = "value",
) -> np.ndarray | float:
    """Clip to `[lo, hi]`, asserting nothing was moved by more than `atol`.

    Use where a quantity is analytically in range and float error can put it marginally
    outside — a singular value of an orthonormal product slightly above 1, a
    constrained solver returning −1e-17. Do **not** use to bring genuinely
    out-of-range input into a function's domain; that case needs a refusal, since the
    result would be a guess (see `analysis.attribution.radius_from_rho`).
    """
    a = np.asarray(x, dtype=np.float64)
    over = 0.0 if hi is None else float(np.max(a - hi, initial=0.0))
    under = 0.0 if lo is None else float(np.max(lo - a, initial=0.0))
    worst = max(over, under)
    if worst > atol:
        raise ValueError(
            f"{what} is {worst:.3e} outside [{lo}, {hi}], far beyond the {atol:.1e} "
            f"tolerance for float error. Clipping here would absorb a real violation "
            f"into a plausible number rather than round-off; the input is wrong, or "
            f"this range is not the right one."
        )
    out = np.clip(a, -np.inf if lo is None else lo, np.inf if hi is None else hi)
    return float(out) if np.isscalar(x) or out.ndim == 0 else out
