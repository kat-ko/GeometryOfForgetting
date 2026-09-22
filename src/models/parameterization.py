"""NTP / μP scaling and the base-width normalization (`00` §4.2).

Source: Graldi et al. (ICML 2025), Table 1.

| | NTP | μP (mean field) |
|---|---|---|
| branch scale `β_ℓ` | `N^{-1/2}` (ℓ>0), `d^{-1/2}` (ℓ=0) | same |
| output scale `γ` | `1` | `γ₀ · N^{1/2}` |
| LR `η(t)` | `η₀(t)` | `η₀(t) · γ₀² · N` |
| weight variance `σ_ℓ²` | `1` | `1` |

**Base-width constant (`00` §4.2, Verification 1 signed 2026-09-01).** Table 1's
four cells and A.3 text are human-signed. A.3 identifies the table (Bordelon
2023 notation, Yang–Hu 2020 NTP, PyTorch SP ≡ NTP) and does not write `N/N₀`;
the caption's forward reference is dangling. Independent derivation: set μP =
NTP; both rows give `γ₀ = N^{-1/2}`; rescaling so `γ₀ = 1` at `N₀ = 64` is
`N → N/N_base`. Confirmed against the agent's. See
`docs/reference/parameterization-derivation.md`. Pinned by
`tests/test_models_parameterization.py::test_base_width_equivalence`.
At `N = 300`, NTP-equivalent `γ₀ ≈ 0.462` (computed, not interpreted).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

N_BASE = 64
Parameterization = Literal["ntp", "mup"]
LRScaling = Literal["quadratic", "corrected"]


@dataclass(frozen=True)
class ScalingConfig:
    """Per-module scaling. `gamma_0` is richness; `N` is module width."""

    N: int
    d: int
    gamma_0: float = 1.0
    parameterization: Parameterization = "mup"
    lr_scaling: LRScaling = "quadratic"
    lr0: float = 0.01
    depth: int = 2
    n_base: int = N_BASE

    def __post_init__(self) -> None:
        if self.gamma_0 <= 0:
            raise ValueError(f"gamma_0 must be positive; got {self.gamma_0}")
        if self.parameterization not in ("ntp", "mup"):
            raise ValueError(f"unknown parameterization {self.parameterization!r}")

    @property
    def beta_0(self) -> float:
        """Input branch scale `d^{-1/2}`."""
        return self.d ** -0.5

    @property
    def beta_L(self) -> float:
        """Output branch scale `N^{-1/2}`."""
        return self.N ** -0.5

    @property
    def beta_hid(self) -> float:
        """Hidden-to-hidden branch scale `N^{-1/2}` (ℓ > 0). Same in NTP and μP."""
        return self.N ** -0.5

    @property
    def gamma_eff(self) -> float:
        """Output scale appearing as `β_L / γ_eff` in the forward pass."""
        if self.parameterization == "ntp":
            return 1.0
        return self.gamma_0 * (self.N / self.n_base) ** 0.5

    @property
    def output_scale(self) -> float:
        return self.beta_L / self.gamma_eff

    @property
    def lr(self) -> float:
        """Per-module learning rate, applied to `W_m`, `u_m`, and `W²_m` (`00` §4.2).

        Changing one without the others changes contribution magnitude, not richness.
        One η is the unsigned L=3 hypothesis (`docs/16` A21), not a signed finding.
        """
        if self.parameterization == "ntp":
            return self.lr0
        width = self.N / self.n_base
        if self.lr_scaling == "corrected" and self.gamma_0 > 1.0:
            # Atanasov et al. (ICLR 2025): eta* ~ gamma^(2/L) for gamma >> 1;
            # Graldi's gamma^2 is misspecified in the rich regime (`00` §4.3).
            return self.lr0 * self.gamma_0 ** (2.0 / self.depth) * width
        return self.lr0 * self.gamma_0**2 * width

    def describe(self) -> dict:
        return {
            "N": self.N, "d": self.d, "gamma_0": self.gamma_0,
            "parameterization": self.parameterization,
            "lr_scaling": self.lr_scaling, "lr0": self.lr0,
            "n_base": self.n_base, "beta_0": self.beta_0, "beta_L": self.beta_L,
            "gamma_eff": self.gamma_eff, "output_scale": self.output_scale,
            "lr": self.lr, "depth": self.depth,
        }


def module_param_count(N: int, d: int, n_hidden_layers: int = 1) -> int:
    """Parameters in one module: first layer N×d, optional hidden N×N, readout N."""
    if n_hidden_layers < 1:
        raise ValueError(f"n_hidden_layers must be ≥ 1; got {n_hidden_layers}")
    count = N * d
    count += N * N * (n_hidden_layers - 1)
    count += N
    return count


def width_matching_param_count(N_ref: int, d: int, n_hidden_layers: int) -> int:
    """Integer width whose parameter count is nearest `module_param_count(N_ref, d, 1)`.

    Matching N across depth would confound depth with capacity (`docs/16` §Amendments A6).
    """
    target = module_param_count(N_ref, d, 1)
    best, best_err = N_ref, abs(module_param_count(N_ref, d, n_hidden_layers) - target)
    for N in range(1, N_ref * 4 + 1):
        err = abs(module_param_count(N, d, n_hidden_layers) - target)
        if err < best_err:
            best, best_err = N, err
    return best
