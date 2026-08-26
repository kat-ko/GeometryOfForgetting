"""Geometry trajectories through training (`00` §12, `01` Phase 0).

Records `(α, R_eff, D_eff, ρ_c)` at checkpoints while a task trains, so that two
runs at different `γ` can be compared as *trajectories* rather than as endpoints.

Checkpoints are placed logarithmically in step count. Under μP the function-space
rate is γ-independent but feature learning is not, and the measured spread in
steps-to-target across `γ` is ~40× (`01` Phase 0), so a linear grid would put
almost all of a rich run's samples after it had finished moving.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..glue import core
from ..train.loop import accuracy, flatten_task

CHANNELS = ("alpha", "D_eff", "R_eff", "rho_c_glue")


@dataclass
class GeometryTrajectory:
    """Geometry and loss at a common set of step checkpoints."""

    gamma: float
    seed: int
    steps: np.ndarray
    loss: np.ndarray
    channels: dict[str, np.ndarray]
    module: str = "A"
    meta: dict = field(default_factory=dict)

    def matrix(self, channels: tuple[str, ...] = CHANNELS) -> np.ndarray:
        """`(n_checkpoints, n_channels)` in **log** space.

        Log space because the attribution is additive in logs (`00` §8) and
        because it makes the per-channel noise floors, which are relative, into a
        common unit.
        """
        return np.column_stack([np.log(self.channels[c]) for c in channels])

    def to_dict(self) -> dict:
        return {"gamma": self.gamma, "seed": self.seed, "module": self.module,
                "steps": self.steps.tolist(), "loss": self.loss.tolist(),
                "channels": {k: v.tolist() for k, v in self.channels.items()},
                **self.meta}

    @classmethod
    def from_dict(cls, d: dict) -> "GeometryTrajectory":
        known = {"gamma", "seed", "module", "steps", "loss", "channels"}
        return cls(gamma=d["gamma"], seed=d["seed"], module=d.get("module", "A"),
                   steps=np.asarray(d["steps"]), loss=np.asarray(d["loss"]),
                   channels={k: np.asarray(v) for k, v in d["channels"].items()},
                   meta={k: v for k, v in d.items() if k not in known})

    def excursion_in_floors(self, floors: dict[str, float],
                            channels: tuple[str, ...] = CHANNELS) -> float:
        """How far the geometry actually moved, in units of the noise floor.

        A trajectory that barely moves is trivially "the initial segment of" any
        other trajectory under a sufficiently slow warp, so the reparameterization
        test is **vacuous** for it. This is the number that says so.
        """
        return float(np.sqrt(np.mean([
            (np.log(self.channels[c]).max() - np.log(self.channels[c]).min())
            ** 2 / np.log1p(floors[c]) ** 2 for c in channels])))


def log_checkpoints(max_steps: int, n: int = 8) -> np.ndarray:
    """`n` distinct step counts spaced logarithmically in `[1, max_steps]`."""
    raw = np.unique(np.round(np.geomspace(1, max_steps, n)).astype(int))
    return raw


def record_geometry_trajectory(
    model,
    points: np.ndarray,
    y: np.ndarray,
    rng: np.random.Generator,
    *,
    checkpoints: np.ndarray,
    module: str = "A",
    n_t: int = 200,
    measure_rng_seed: int = 0,
) -> GeometryTrajectory:
    """Train on one task, measuring geometry at `checkpoints` (in step counts).

    The measurement RNG is fixed across checkpoints and across runs, so successive
    points on a trajectory differ because the representation changed and not
    because the estimator resampled `(y, t)`. Without this, the trajectory noise
    floor would swamp the drift being measured.
    """
    X, target = flatten_task(points, y)
    gamma = model.cfg[module].gamma_0

    steps, losses = [], []
    chans: dict[str, list[float]] = {c: [] for c in CHANNELS}
    step = 0
    for target_step in checkpoints:
        while step < target_step:
            model.sgd_step(X, target)
            step += 1
        reps = model.manifold_representation(points, module=module)
        r = core.glue_measures(reps, np.random.default_rng(measure_rng_seed), n_t=n_t)
        steps.append(step)
        losses.append(float(0.5 * np.mean((model.forward(X) - target) ** 2)))
        for c in CHANNELS:
            chans[c].append(float(getattr(r, c)))

    return GeometryTrajectory(
        gamma=float(gamma), seed=int(measure_rng_seed),
        steps=np.array(steps), loss=np.array(losses),
        channels={c: np.array(v) for c, v in chans.items()},
        module=module,
        meta={"final_accuracy": accuracy(model, X, target), "n_t": n_t},
    )
