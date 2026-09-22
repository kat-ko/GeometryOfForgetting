"""Two-module network, its NTP/μP scaling, and the alignment initialization."""

from . import alignment, network, parameterization
from .network import MODULES, TwoModuleNet, paired_init, projection_rng
from .parameterization import ScalingConfig, module_param_count, width_matching_param_count

__all__ = [
    "alignment",
    "network",
    "parameterization",
    "MODULES",
    "ScalingConfig",
    "TwoModuleNet",
    "paired_init",
    "projection_rng",
    "module_param_count",
    "width_matching_param_count",
]
