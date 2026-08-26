"""Synthetic manifold generator, factorial labeling, dichotomies, and streams.

Implements Chou et al. ICML 2025 App. D.1.1 and `docs/00-math-spec.md` §§1–3.
"""

from .dichotomies import (
    hamming_distance,
    readout_similarity,
    sample_at_hamming,
    sample_balanced,
)
from .generator import Arrangement, make_arrangement, resample_test_points
from .labeling import FactorLabeling, make_labeling
from .streams import Stream, StreamConfig, make_stream

__all__ = [
    "Arrangement",
    "FactorLabeling",
    "Stream",
    "StreamConfig",
    "hamming_distance",
    "make_arrangement",
    "make_labeling",
    "make_stream",
    "readout_similarity",
    "resample_test_points",
    "sample_at_hamming",
    "sample_balanced",
]
