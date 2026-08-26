"""Factorial labeling of P=16 manifolds — labels only, centers random.

Spec: `docs/00-math-spec.md` §2.1–§2.2.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FactorLabeling:
    """Bijective map manifold index → four binary factors.

    ``factors[i] = (f1, f2, f3, f4) ∈ {0,1}^4``.
    Factor structure lives in the labeling only — never in geometry.
    """

    factors: np.ndarray  # (P, n_factors) int8, entries in {0,1}
    n_factors: int
    P: int

    def single_factor_dichotomy(self, j: int) -> np.ndarray:
        """y_i = 2·f_j(i) − 1 ∈ {±1}^P."""
        if not (0 <= j < self.n_factors):
            raise IndexError(j)
        return (2 * self.factors[:, j] - 1).astype(np.int8)

    def xor_dichotomy(self, j: int, k: int) -> np.ndarray:
        """y_i = 2·(f_j(i) ⊕ f_k(i)) − 1."""
        if j >= k:
            raise ValueError("require j < k")
        xor = np.bitwise_xor(self.factors[:, j], self.factors[:, k])
        return (2 * xor - 1).astype(np.int8)

    def all_single_factor(self) -> list[np.ndarray]:
        return [self.single_factor_dichotomy(j) for j in range(self.n_factors)]

    def all_xor(self) -> list[np.ndarray]:
        out: list[np.ndarray] = []
        for j in range(self.n_factors):
            for k in range(j + 1, self.n_factors):
                out.append(self.xor_dichotomy(j, k))
        return out


def make_labeling(P: int = 16, n_factors: int | None = None) -> FactorLabeling:
    """Standard binary counting: manifold i ↔ bit-tuple of i.

    For P=16 this is four factors; for P=32, five. ``n_factors`` defaults to
    ``log2(P)`` and requires P to be a power of two.
    """
    if n_factors is None:
        if P & (P - 1) != 0 or P < 2:
            raise ValueError(f"P must be a power of two when n_factors is None; got {P}")
        n_factors = int(np.log2(P))
    if 2**n_factors != P:
        raise ValueError(f"P={P} is not 2^{n_factors}")
    indices = np.arange(P, dtype=np.int32)
    factors = np.zeros((P, n_factors), dtype=np.int8)
    for j in range(n_factors):
        factors[:, j] = ((indices >> j) & 1).astype(np.int8)
    return FactorLabeling(factors=factors, n_factors=n_factors, P=P)
