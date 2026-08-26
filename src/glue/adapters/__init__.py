"""Thin wrappers around vendored estimators.

`third_party/` stays byte-identical to upstream (`third_party/VENDORED.md`); every
correction — RNG-Generator injection instead of global `np.random.seed`, float64
enforcement, explicit pseudo-inverse rcond — lives here.
"""

from . import simcap

__all__ = ["simcap"]
