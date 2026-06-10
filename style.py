#!/usr/bin/env python3
"""
style.py — the single source of house style. EVERY renderer calls apply_house_style()
so varied figures stay visually uniform and publication-grade. Edit here, not per-figure.
"""
import matplotlib as mpl
import matplotlib.pyplot as plt

# A readable, perceptually-sensible snowfall ramp: pale lavender -> blues -> purple
# -> magenta -> deep red for extremes. Hand-built because snowfall has a domain-standard
# "more = cooler-then-hot" reading that no single matplotlib default matches.
SNOW_COLORS = [
    "#f7f7fb", "#e6e6f5", "#c9d6f0", "#9ec0e8", "#6ba3da", "#4682c4",
    "#3a5fae", "#3f3f9e", "#5a2d91", "#7b2382", "#a01f7a", "#c41f6b",
    "#d83a4e", "#e35d3a", "#ec8b2f", "#f0b429",
]

def apply_house_style():
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 300,            # publication raster
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.08,    # tighten whitespace around the figure
        "font.family": "DejaVu Sans",  # swap to a journal font if required
        "font.size": 10,
        "font.weight": "bold",         # bold everything (ticks inherit this)
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "axes.labelweight": "bold",
        "axes.linewidth": 1.0,
        "axes.edgecolor": "#333333",
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.frameon": False,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    })

def annotate_provenance(fig, text, *, fontsize=6):
    """Stamp source/provenance in the figure footer — every figure carries its origin."""
    fig.text(0.005, 0.005, text, ha="left", va="bottom",
             fontsize=fontsize, color="#888888", family="monospace")
