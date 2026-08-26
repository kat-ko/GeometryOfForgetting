"""NTP / μP scaling and the base-width normalization (`00` §4.2).

Source: Graldi et al. (ICML 2025), Table 1.

| | NTP | μP (mean field) |
|---|---|---|
| branch scale `β_ℓ` | `N^{-1/2}` (ℓ>0), `d^{-1/2}` (ℓ=0) | same |
| output scale `γ` | `1` | `γ₀ · N^{1/2}` |
| LR `η(t)` | `η₀(t)` | `η₀(t) · γ₀² · N` |
| weight variance `σ_ℓ²` | `1` | `1` |

**The base-width constant (`00` §4.2 UNCERTAIN).** Graldi et al. normalize μP to
agree with NTP at base width `N = 64` but do not write the constant out. Requiring
`μP(γ₀ = 1, N = N_base) ≡ NTP(N = N_base)` fixes it uniquely: the width factors
must be *relative* to the base width, i.e. `N^{1/2} → (N/N_base)^{1/2}` in the
output scale and `N → N/N_base` in the learning rate. At `γ₀ = 1, N = N_base` both
reduce to `γ_eff = 1` and `η = η₀`, which is NTP exactly. Verified numerically on
the forward pass and the first gradient step by
`tests/test_models_parameterization.py::test_base_width_equivalence` (`02` §3).

AGENT-DERIVED, NOT HUMAN-VERIFIED — `docs/reference/parameterization-derivation.md`
is human-owned and still unwritten. The unit test pins the behaviour; the
*justification* above is ours and should be checked against Graldi §3 before the
paper cites it.
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
        """Per-module learning rate, applied to **both** `W_m` and `u_m` (`00` §4.2).

        Changing one without the other changes contribution magnitude, not richness.
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
            "lr": self.lr,
        }
