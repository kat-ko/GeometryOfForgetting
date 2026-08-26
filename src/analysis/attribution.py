"""Attribution of forgetting to the three capacity factors (`00` §8).

Two things happen here.

**The decomposition.** `α = Ψ_eff · (1 + R_eff⁻²) / D_eff` is exact, so the change in
retained capacity between two boundaries splits additively in log space with **zero
residual**:

    Δ log α = Δ log Ψ_eff + Δ log(1 + R_eff⁻²) − Δ log D_eff

No cross-terms, no leftover. The residual is computed anyway and asserted small,
because it is the cheapest available check that the two geometry measurements being
differenced came from the same estimator configuration — a nonzero residual means
something upstream is inconsistent, not that the identity failed.

**The center-collapse share of the radius channel.** `ρ_c` sits *outside* the
identity, yet Wakhloo, Sussman & Chung (PRL 2023) show center correlation is
effectively equivalent to shrunk center separation — which is to say it arrives in
`R_eff`. So a radius contribution is not automatically an anisotropy story: some of
it is centers falling together. `00` §8 already requires reporting `ρ_c` alongside
`R_eff`; this makes the joint report quantitative by asking what fraction of an
observed `Δ log R_eff` the observed `Δ ρ_c` already accounts for.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from src import provenance

from . import timewarp

_SOURCE = provenance.register(__file__)

FACTORS = ("utility", "radius", "dimension")
GEOMETRY = ("alpha", "D_eff", "R_eff", "Psi_eff", "rho_c_glue", "rho_c_signed")

# Measured on the `00` §6.1 center-correlation sweep of the B.5 recovery suite
# (`results/glue_core_recovery.json`, `center_correlation`, n_t=200, policy=all;
# P=2, M=200, N=1000, D=4, R=1). Ground-truth ρ_C 0 → 0.8 drives R_eff 1.02 → 1.82
# while D_eff stays flat to 0.38%, so on synthetic manifolds the radius absorbs
# center correlation essentially alone, exactly as the duality says.
#
#     log R_eff = RHO_R_INTERCEPT + RHO_R_EXPONENT · (−log(1 − ρ_c_signed))
#
# i.e. R_eff ∝ (1 − ρ_c)^(−0.365), fitting the five sweep points at R² = 0.9997.
#
# **The convention is `rho_c_signed`, and that is load-bearing rather than a
# preference.** `rho_c_glue` is unnormalized by definition (`00` §6.1 C3), so it is not
# a correlation and is not confined to [0, 1): measured on Phase 1 representations it
# runs to **1.49**, where `1 − ρ` is negative and this form is undefined. It stayed
# inside [0, 1) on the synthetic sweep only because there the two conventions nearly
# coincide (0.038/0.043, 0.220/0.236, …), which is exactly why fitting on the wrong one
# survived review. `rho_c_signed` is normalized to [−1, 1] and sits at 0.33–0.48 on the
# same representations — inside the fitted range.
RHO_R_EXPONENT = 0.36483
RHO_R_INTERCEPT = 0.002911
RHO_R_R2 = 0.99970
RHO_R_CONVENTION = "rho_c_signed"
RHO_R_PROVENANCE = "glue_core_recovery.json:center_correlation/n_t=200,policy=all"
# The sweep's ρ range. Outside it the conversion extrapolates a form that diverges at
# ρ → 1, so it is refused rather than reported.
RHO_FIT_RANGE = (0.0427, 0.8029)
# Margin past the fitted endpoints still treated as interpolation.
RHO_DOMAIN_TOL = 0.05

# `R_eff` Monte-Carlo noise floor, in log units. Radius changes below it are unresolvable, so
# they cannot be a ratio's denominator.
#
# **This floor depends on γ, and a single constant was wrong.** The original value came from
# one setting (CV 0.50%, `results/cost_model.json`). The per-γ measurement null
# (`scripts/run_measurement_null.py`, W4 in `results/LOG.md`) re-measured the *same* trained
# representation under several measurement seeds at every registered γ and found the retained
# ensemble's `R_eff` dispersion rising monotonically with richness, from CV 0.28% to 1.13% —
# 2.3× the old constant at γ=10, and 4× across the sweep. A single floor was therefore too
# permissive exactly where this guard is used most, which is the same failure mode the guard
# was introduced to prevent, one step further out. Values pool both modules, and live in the
# floor registry beside the registered ones rather than being restated here.
R_EFF_FLOOR_CV = timewarp.PER_GAMMA_FLOOR_CV["R_eff"]

# The gate is set at **three** floors, not one. A denominator at 1σ of its own measurement noise
# carries ~100% relative error, so the ratio built on it is noise over noise even though the guard
# passed it. That is not hypothetical: in the one stream condition whose radius does not move
# (`S-HH`, the registered benign corner), the median |Δ log R_eff| sits at 0.86–0.98 floors, half
# its cells were admitted, and 50–75% of those came out with the *opposite sign* to the ρ_c
# prediction. At 3σ that condition empties completely while the three conditions whose radius does
# move are untouched — 40/40 cells each, identical medians — which is the signature of a threshold
# that separates signal from noise rather than one that trims a distribution.
FLOOR_GATE_K = 3.0

# Used when γ is unknown or off-sweep: the largest measured floor, since a guard that fails
# open is worse than one that occasionally refuses a resolvable change.
MIN_DLOG_R = FLOOR_GATE_K * float(np.log1p(max(R_EFF_FLOOR_CV.values())))


def min_dlog_R_for(gamma: float | None) -> float:
    """The smallest `Δ log R_eff` usable as a denominator at this γ, in log units.

    `FLOOR_GATE_K` measurement floors at the γ in question. Off-sweep γ gets the most
    conservative measured floor rather than an interpolation: the floor is a property of the
    estimator at a given richness, and guessing it would put a fabricated number in a guard
    whose whole job is to refuse fabricated numbers.
    """
    if gamma is None or gamma not in R_EFF_FLOOR_CV:
        return MIN_DLOG_R
    return FLOOR_GATE_K * float(np.log1p(R_EFF_FLOOR_CV[gamma]))


@dataclass(frozen=True)
class GeometryPoint:
    """One `(module, boundary, past-task)` geometry measurement.

    `alpha` is **retained** capacity `α(·; y_j)` for the past task being attributed,
    not generic capacity — the identity holds for whichever ensemble `Y` was used,
    but a decomposition of generic capacity answers a different question.
    """

    alpha: float
    D_eff: float
    R_eff: float
    Psi_eff: float
    rho_c_glue: float
    rho_c_signed: float
    label: str = ""

    @classmethod
    def from_result(cls, r, label: str = "") -> "GeometryPoint":
        get = r.get if isinstance(r, dict) else lambda k: getattr(r, k)
        return cls(*(float(get(k)) for k in GEOMETRY), label=label)

    @property
    def alpha_from_identity(self) -> float:
        return self.Psi_eff * (1.0 + self.R_eff**-2) / self.D_eff


@dataclass(frozen=True)
class Attribution:
    """Signed log-space contributions. Positive = pushed capacity up."""

    dlog_alpha: float
    terms: dict[str, float]
    shares: dict[str, float]
    residual: float
    dominant: str
    center: dict[str, float] = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def attribute(
    before: GeometryPoint,
    after: GeometryPoint,
    *,
    tol: float = 1e-8,
    strict: bool = True,
    gamma: float | None = None,
) -> Attribution:
    """Decompose `Δ log α` from `before` to `after` into the three factors.

    `gamma` selects the per-γ `R_eff` noise floor for the center-collapse share; omitting it
    uses the most conservative measured floor. It affects nothing else — the three terms and
    their shares are exact and need no floor.

    Sign convention: each term is the amount that factor **pushed capacity up**, so
    the dimension term carries the minus sign internally and the three terms sum to
    `Δ log α` as reported. Forgetting is `dlog_alpha < 0`, and the factor
    responsible is the most negative term.

    `shares` are `|term| / Σ|term|` — a fraction of *total motion*, not of the net
    change, because factors routinely oppose each other and shares of a small net
    difference of large opposing terms exceed 1 and mislead. Read them with the
    signs, and read `cancellation` in `meta` first: when it is high the factors are
    fighting and "which factor explains the change" is the wrong question.
    """
    d = lambda f, g: float(np.log(g) - np.log(f))  # noqa: E731
    terms = {
        "utility": d(before.Psi_eff, after.Psi_eff),
        "radius": d(1.0 + before.R_eff**-2, 1.0 + after.R_eff**-2),
        "dimension": -d(before.D_eff, after.D_eff),
    }
    dlog_alpha = d(before.alpha, after.alpha)
    residual = dlog_alpha - sum(terms.values())

    if strict and abs(residual) > max(tol, 1e-6 * abs(dlog_alpha)):
        raise ValueError(
            f"identity residual {residual:.3e} exceeds tolerance — the three-factor "
            f"identity is exact, so this means the two points were measured under "
            f"different estimator settings, or `alpha` is not the same ensemble's "
            f"capacity as the factors ({before.label!r} -> {after.label!r})"
        )

    total = sum(abs(v) for v in terms.values())
    shares = {k: (abs(v) / total if total > 0 else 0.0) for k, v in terms.items()}
    dominant = max(terms, key=lambda k: abs(terms[k])) if total > 0 else "none"

    return Attribution(
        dlog_alpha=dlog_alpha,
        terms=terms,
        shares=shares,
        residual=residual,
        dominant=dominant,
        center=center_collapse_share(before, after, min_dlog_R=min_dlog_R_for(gamma)),
        meta={
            "cancellation": float(1.0 - abs(dlog_alpha) / total) if total > 0 else 0.0,
            "before": before.label,
            "after": after.label,
        },
    )


def rho_in_domain(rho: float, *, tol: float = RHO_DOMAIN_TOL) -> bool:
    """Is `rho` close enough to the sweep that the calibration is interpolation?

    `tol` allows a small margin past the fitted endpoints; beyond that the form is
    extrapolating toward its divergence at ρ → 1 and its output is not a measurement.
    """
    lo, hi = RHO_FIT_RANGE
    return bool(lo - tol <= rho <= hi + tol)


def radius_from_rho(rho: float) -> float:
    """`R_eff` predicted by center correlation alone, per the measured calibration.

    Raises outside the fitted range rather than clipping. An earlier version clamped
    to 0.995, which turned an out-of-domain input into a plausible finite number: fed
    `rho_c_glue` values up to 1.49 it silently returned the value at 0.995 and reported
    center-collapse shares of 15–38, which read as a substantive result and were
    entirely an artifact of the clamp. Callers that expect out-of-range values should
    test with `rho_in_domain` first.
    """
    rho = float(rho)
    if not rho_in_domain(rho):
        lo, hi = RHO_FIT_RANGE
        raise ValueError(
            f"rho={rho:.4f} is outside the calibration range [{lo:.4f}, {hi:.4f}]; the "
            f"conversion diverges at rho -> 1 and would not be a measurement here. "
            f"Note {RHO_R_CONVENTION} is the required convention — rho_c_glue is "
            f"unnormalized and routinely exceeds 1."
        )
    return float(np.exp(RHO_R_INTERCEPT + RHO_R_EXPONENT * -np.log1p(-rho)))


def center_collapse_share(
    before: GeometryPoint, after: GeometryPoint, *, min_dlog_R: float = MIN_DLOG_R
) -> dict:
    """How much of the observed `Δ log R_eff` does the observed `Δ ρ_c` account for?

    Uses the synthetic-manifold calibration above to convert the measured change in
    center correlation into the radius change it would have produced on its own, and
    reports that as a fraction of the radius change actually observed:

        attributable = Δ log R_eff(ρ_c) / Δ log R_eff observed

    Read it as follows. Near **1**, the radius channel *is* the center channel and
    reporting them as separate mechanisms would double-count. Near **0** with `ρ_c`
    moving, centers moved and the radius did not follow, so the radius change is
    anisotropy rather than center collapse. **Above 1** means centers collapsed more
    than the radius reflects, so something else offset it.

    Four caveats, all of which keep this a reported attribution rather than a
    correction.

    The calibration is fitted on synthetic manifolds where `ρ_C` is the *only* thing
    varying, so it is an upper bound on how much a representation's `ρ_c` movement can
    explain — in a representation, radius and centers move together.

    **It requires `rho_c_signed`, the normalized convention.** `rho_c_glue` is
    unnormalized and exceeds 1 on real representations, where `(1 − ρ)^−k` is
    undefined; see the note on `RHO_R_CONVENTION`.

    It is monotone in `ρ_c` only, so it says nothing when `ρ_c` is unchanged;
    `attributable` is then `None` rather than 0. It is likewise `None` when either
    endpoint falls outside the fitted range, since the fitted form diverges at ρ → 1
    and extrapolating it produces large numbers that look like findings.

    **The denominator is floored at the `R_eff` noise floor.** A ratio whose
    denominator is a radius change smaller than the estimator can resolve is
    unbounded — the first smoke grid produced fractions of 42 and −134 this way, from
    `Δ log R_eff` of order 1e-3. Below the floor the fraction is `None` with
    `reason: "radius change below noise floor"`, which is the honest statement: the
    radius did not measurably move, so there is nothing to attribute.
    """
    dlog_R = float(np.log(after.R_eff) - np.log(before.R_eff))
    rho_before, rho_after = before.rho_c_signed, after.rho_c_signed
    drho = float(rho_after - rho_before)

    out = {
        "dlog_R_eff_observed": dlog_R,
        "d_rho_c_signed": drho,
        "d_rho_c_glue": float(after.rho_c_glue - before.rho_c_glue),
        "calibration_convention": RHO_R_CONVENTION,
        "min_dlog_R": float(min_dlog_R),
    }

    in_domain = rho_in_domain(rho_before) and rho_in_domain(rho_after)
    out["rho_in_calibration_range"] = in_domain
    predicted = (float(np.log(radius_from_rho(rho_after))
                       - np.log(radius_from_rho(rho_before))) if in_domain else None)
    out["dlog_R_eff_predicted_from_rho_c"] = predicted

    if not in_domain:
        lo, hi = RHO_FIT_RANGE
        out["attributable_fraction"] = None
        out["reason"] = (f"rho_c_signed outside calibration range [{lo:.3f}, {hi:.3f}] "
                         f"(before {rho_before:.3f}, after {rho_after:.3f})")
    elif abs(drho) < 1e-12:
        out["attributable_fraction"] = None
        out["reason"] = "rho_c unchanged — no conversion to report"
    elif abs(dlog_R) < min_dlog_R:
        out["attributable_fraction"] = None
        out["reason"] = "radius change below noise floor"
    else:
        out["attributable_fraction"] = predicted / dlog_R
        out["same_direction"] = bool(np.sign(predicted) == np.sign(dlog_R))
    return out


def attribution_table(rows: list[Attribution]) -> dict:
    """Pool a set of attributions — per stream condition, γ arm, or module.

    Averages in log space, which is where the identity is additive; the mean of the
    three term-means therefore still equals the mean `Δ log α` exactly. `shares` are
    recomputed from the pooled terms rather than averaged, since a mean of ratios is
    not the ratio of means.
    """
    if not rows:
        raise ValueError("no attributions to pool")
    terms = {k: float(np.mean([r.terms[k] for r in rows])) for k in FACTORS}
    dlog_alpha = float(np.mean([r.dlog_alpha for r in rows]))
    total = sum(abs(v) for v in terms.values())
    frac = [r.center["attributable_fraction"] for r in rows
            if r.center.get("attributable_fraction") is not None]
    return {
        "n": len(rows),
        "dlog_alpha": dlog_alpha,
        "dlog_alpha_sem": float(np.std([r.dlog_alpha for r in rows], ddof=1)
                                / np.sqrt(len(rows))) if len(rows) > 1 else 0.0,
        "terms": terms,
        "term_sem": {
            k: float(np.std([r.terms[k] for r in rows], ddof=1) / np.sqrt(len(rows)))
            if len(rows) > 1 else 0.0 for k in FACTORS
        },
        "shares": {k: (abs(v) / total if total else 0.0) for k, v in terms.items()},
        "dominant": max(terms, key=lambda k: abs(terms[k])) if total else "none",
        "max_residual": float(max(abs(r.residual) for r in rows)),
        "center_attributable_median": float(np.median(frac)) if frac else None,
        "center_attributable_n": len(frac),
    }
