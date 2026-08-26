"""Figure geometry, measured from the venue class rather than guessed.

Every figure was authored on a 10.5--15.5 inch canvas and then included at `width=\\linewidth`,
which is 6.00 inches here. Two consequences, both found by compiling the paper for the first time:

1. **Every label shrank by 43%.** Fonts set at 6--12pt render at 3.4--6.9pt. Below legibility, and
   invisible while the figures were only ever viewed as standalone PNGs.
2. **Each float took a page of its own.** `Float too large for page by 65.83pt`. Body plus two
   figures came to 8 pages against a 4-page limit.

So: author at the size the figure will be printed, and let `width=\\linewidth` be a no-op. Then a
font size in the script is the font size on paper, and the height is a page budget rather than an
accident.

Measured from `\\the\\textwidth` and `\\the\\textheight` under `\\documentclass[wcp]{jmlr}`.
"""

from __future__ import annotations

PT_PER_INCH = 72.27

TEXT_WIDTH_PT = 433.62
TEXT_HEIGHT_PT = 614.295

TEXT_WIDTH_IN = TEXT_WIDTH_PT / PT_PER_INCH      # 6.00
TEXT_HEIGHT_IN = TEXT_HEIGHT_PT / PT_PER_INCH    # 8.50

# Measured, not estimated. The two body captions run 312 and 257 words and typeset to roughly 5.0
# and 4.6 inches -- more than a words-per-line estimate suggested, which is why a 4.2 inch budget
# still produced `Float too large for page by 55.5pt`. Budget for the larger of the two.
CAPTION_BUDGET_IN = 5.0
MAX_GRAPHIC_HEIGHT_IN = TEXT_HEIGHT_IN - CAPTION_BUDGET_IN   # 3.50


# The font sizes in the figure scripts were chosen against a 10.5 inch canvas. Rendering the same
# point sizes on a 6 inch canvas makes them 1.75x too large for their panels and they collide. Scale
# them, with a floor: below about 6pt a label is present but not readable, and a figure that cannot
# be read at print size is worse than one that takes an extra page.
FONT_SCALE = TEXT_WIDTH_IN / 10.5
MIN_POINTS = 6.0


def fs(points: float) -> float:
    """A font size authored for the old canvas, converted to print size."""
    return round(max(MIN_POINTS, points * FONT_SCALE), 1)


def apply() -> None:
    """Set the matplotlib defaults that go with authoring at print size."""
    import matplotlib as mpl

    mpl.rcParams.update({
        "font.size": fs(9),
        "axes.titlesize": fs(10),
        "axes.labelsize": fs(9),
        "xtick.labelsize": fs(8),
        "ytick.labelsize": fs(8),
        "legend.fontsize": fs(8),
        "figure.titlesize": fs(11),
        "axes.linewidth": 0.6,
        "lines.linewidth": 1.1,
        "lines.markersize": 3.2,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        # constrained_layout resolves the collisions that tight_layout cannot once panels are this
        # small, and it accounts for suptitles and outside-axes legends.
        "figure.constrained_layout.use": True,
        "figure.constrained_layout.h_pad": 0.03,
        "figure.constrained_layout.w_pad": 0.03,
    })


def wrap(text: str, width: int = 95) -> str:
    """Wrap a long title so it does not run off a 6 inch canvas."""
    import textwrap

    return "\n".join(textwrap.wrap(text, width=width))


def figsize(height_in: float, *, width_frac: float = 1.0) -> tuple[float, float]:
    """Figure size in inches for a graphic included at `width_frac` of `\\linewidth`.

    Raises rather than silently returning something that will be given its own page: a figure that
    does not fit is the failure this module exists to prevent, and it stays invisible until the
    paper is compiled.
    """
    if height_in > MAX_GRAPHIC_HEIGHT_IN:
        raise ValueError(
            f"a graphic {height_in:.2f} in tall leaves less than {CAPTION_BUDGET_IN:g} in for its "
            f"caption in a {TEXT_HEIGHT_IN:.2f} in text block, so LaTeX will float it onto a page "
            f"of its own. Keep it under {MAX_GRAPHIC_HEIGHT_IN:.2f} in.")
    return (TEXT_WIDTH_IN * width_frac, height_in)
