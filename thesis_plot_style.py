"""
thesis_plot_style.py  
Reusable plotting style template for the Master's thesis on CMB delensing.  
Import the module once at the start of a notebook or script to set a
consistent, publication-quality look across all figures.

Example
-------
>>> import thesis_plot_style as tps
>>> tps.apply_style()
>>> fig, ax = plt.subplots()
>>> ax.plot(x, y, label="simulation")
>>> ax.set_xlabel(r"$\ell$")
>>> ax.set_ylabel(r"$C_{\ell}$ [$10^{-2}$]")
>>> ax.legend()
>>> tps.savefig(fig, "fig1_spectrum.pdf")
>>> with tps.context_two_panel(filename="fig2_dual.pdf", heights=(2,1)) as (fig, (ax1, ax2)):
...     ax1.plot(x, y1)
...     ax2.plot(x, y2)
"""
from __future__ import annotations

from contextlib import contextmanager
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

__all__ = [
    "apply_style",
    "context_figure",
    "context_two_panel",
    "savefig",
    "CB_PALETTE",
]

# -----------------------------------------------------------------------------
# Core style definition
# -----------------------------------------------------------------------------
CB_PALETTE = sns.color_palette("colorblind")  # colour-blind-friendly palette

THESIS_RCPARAMS = {
    # Figure geometry ---------------------------------------------------------
    "figure.figsize": (5, 4),        #  single-column width in inches
    "figure.dpi": 100,
    "savefig.dpi": 300,
    # Fonts -------------------------------------------------------------------
    "font.family": "serif",
    "font.size": 10,
    "text.usetex": False,               #  flip to True if LaTeX installed
    # Axes --------------------------------------------------------------------
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "axes.linewidth": 0.8,
    "axes.edgecolor": "black",
    "axes.grid": False,
    # Ticks -------------------------------------------------------------------
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    # Lines / markers ---------------------------------------------------------
    "lines.linewidth": 1.6,
    "errorbar.capsize": 2,
    # Legend ------------------------------------------------------------------
    "legend.fontsize": 9,
    "legend.frameon": False,
}


def apply_style(column: str = "single", update: dict | None = None) -> None:
    """Apply the thesis Matplotlib style globally.

    Parameters
    ----------
    column : {"single", "double"}
        Width preset.  "double" sets a wider default figure size (7.2 × 3 in).
    update : dict, optional
        Extra rcParams to override the defaults.
    """
    rc = THESIS_RCPARAMS.copy()
    if column == "double":
        rc["figure.figsize"] = (7.2, 3.0)
    if update:
        rc.update(update)

    mpl.rcParams.update(rc)
    sns.set_palette(CB_PALETTE)


def savefig(fig: mpl.figure.Figure, path: str, **kwargs):
    """Save *fig* to *path* with tight layout and vector back-end by default."""
    fig.tight_layout()
    defaults = dict(bbox_inches="tight", transparent=False)
    defaults.update(kwargs)
    fig.savefig(path, **defaults)

@contextmanager
def context_figure(column: str = "single", **save_kwargs):
    """Context manager that yields an (fig, ax) pair and auto-saves.

    Usage
    -----
    >>> with context_figure(filename="fig1.pdf") as (fig, ax):
    ...     ax.plot(...)
    ...     ax.set_xlabel("$x$")
    """
    apply_style(column)
    fig, ax = plt.subplots()
    yield fig, ax
    filename = save_kwargs.pop("filename", None)
    if filename is not None:
        savefig(fig, filename, **save_kwargs)
    plt.close(fig)

@contextmanager
def context_two_panel(
    filename: str,
    column: str = "single",
    heights: tuple[float, float] = (1,1),
    hspace: float = 0,
    **save_kwargs
):
    """Context manager for two vertically stacked panels sharing x-axis.

    Yields (fig, (ax1, ax2)).

    Parameters
    ----------
    filename : str
        Path to save the figure. If None, figure is not saved.
    column : {"single", "double"}
        Width preset from apply_style.
    heights : tuple of two floats
        Relative heights of the top and bottom subplots. E.g., (2,1) makes
        the first plot twice as tall as the second.
    """
    apply_style(column)
    w, h = THESIS_RCPARAMS["figure.figsize"]
    fig, (ax1, ax2) = plt.subplots(
        2, 1, sharex=True,
        figsize=(w, h * sum(heights)),
        gridspec_kw={
            "height_ratios": heights,
            "hspace": hspace
        }
    )
    yield fig, (ax1, ax2)
    if filename:
        savefig(fig, filename, **save_kwargs)
    plt.close(fig)
