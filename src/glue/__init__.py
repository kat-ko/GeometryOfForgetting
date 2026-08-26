"""GLUE geometry estimators.

`core` is **our** implementation of the three-factor capacity decomposition
(estimator string ``glue_core@<sha>``); `adapters` wraps vendored third-party
estimators without modifying them. The two are distinct estimators under the
`docs/02-validation-suite.md` §9 pooling firewall — never pool their numbers.
"""

from . import core

__all__ = ["core"]
