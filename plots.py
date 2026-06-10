#!/usr/bin/env python3
"""
plots.py — VETTED, version-controlled renderers. Each is "dumb": it receives
already-computed, already-validated arrays and only handles aesthetics. No unit math,
no aggregation, no science here — that all happened upstream in the executor/prep.
This is what keeps even an LLM-selected figure low-risk: it just draws vetted numbers.

Geographic boundaries come from a BUNDLED local file (us_states.geojson), not a
runtime download — so these render identically in my sandbox, on mlm-stormy, AND on
an air-gapped AWARE compute node. cartopy supplies the projection MATH only (local);
it never phones home for features.
"""
from __future__ import annotations
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import BoundaryNorm, ListedColormap
import cartopy.crs as ccrs
from cartopy.feature import ShapelyFeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from shapely.geometry import shape

from style import apply_house_style, SNOW_COLORS, annotate_provenance

# Service-level style overrides (presets + user rcParams). The service's
# style_context() sets this for the duration of a request; exported figure
# scripts set it directly. Applied AFTER apply_house_style() inside every
# renderer, so these win over house defaults.
STYLE_OVERRIDES: dict = {}


def _style():
    """House style, then any service/user overrides on top."""
    apply_house_style()
    if STYLE_OVERRIDES:
        plt.rcParams.update(STYLE_OVERRIDES)


def load_boundaries(path: str):
    """Read a GeoJSON of polygons into shapely geometries (for offline overlay)."""
    gj = json.load(open(path))
    return [shape(f["geometry"]) for f in gj["features"]]


def gridded_field_map(
    *,
    data: np.ndarray,          # 2D field (already in display units, already masked)
    lats: np.ndarray,          # 1D or 2D degrees_north
    lons: np.ndarray,          # 1D or 2D degrees_east
    units: str,
    title: str,
    levels: list[float] | None = None,   # class-bin edges (BoundaryNorm)
    cmap=None,                            # ListedColormap or name
    extent: tuple | None = None,          # (lon_min, lon_max, lat_min, lat_max)
    boundaries=None,                      # list of shapely geoms from load_boundaries()
    huc_boundaries=None,                  # list of shapely geoms (watershed overlay)
    huc_level: int | None = None,         # 4/6/8/10/12 — sets line weight + label
    central_lon: float = -96.0,
    standard_parallels=(33.0, 45.0),
    cbar_label: str | None = None,
    figsize=None,
    xticks: list[float] | None = None,
    yticks: list[float] | None = None,
):
    """Filled map of a gridded geophysical field on a Lambert Conformal CONUS projection.
    Returns (fig, ax). Draws only — caller supplies computed data and styling params."""
    _style()

    proj = ccrs.LambertConformal(central_longitude=central_lon,
                                 standard_parallels=standard_parallels)
    data_crs = ccrs.PlateCarree()       # data is on regular lat/lon

    # size the figure so the AXES box matches the map's aspect (colorbar/title live
    # OUTSIDE the axes), which removes the empty space GeoAxes leaves when its box
    # aspect != the map aspect.
    if figsize is None and extent is not None:
        lo0, lo1, la0, la1 = extent
        midlat = np.deg2rad((la0 + la1) / 2)
        agg = (abs(lo1 - lo0) * np.cos(midlat)) / max(abs(la1 - la0), 1e-6)  # map w/h
        base = 8.0
        if agg >= 1:
            axes_w, axes_h = base, base / agg
        else:
            axes_h, axes_w = base, base * agg
        figsize = (axes_w + 2.2, axes_h + 0.9)        # +cbar/labels, +title
    elif figsize is None:
        figsize = (10, 7)

    fig = plt.figure(figsize=figsize)
    ax = plt.axes(projection=proj)
    if extent is not None:
        ax.set_extent(extent, crs=data_crs)

    Lon, Lat = (np.meshgrid(lons, lats) if lons.ndim == 1 else (lons, lats))

    if levels is not None:
        N = len(levels) - 1  # number of bins (N+1 boundaries -> N intervals)
        if cmap is None:
            # snowfall default palette (truncate if user asked for fewer bins)
            cmap_obj = ListedColormap(SNOW_COLORS[:N])
        elif isinstance(cmap, list):
            cmap_obj = ListedColormap(cmap[:N])
        elif isinstance(cmap, str):
            # Named cmap (e.g. "BrBG", "viridis"). MUST discretize to N colors —
            # otherwise BoundaryNorm(levels, ncolors=N) maps values to indices 0..N-1,
            # but a 256-color LinearSegmentedColormap returns the first N of 256
            # (i.e. one end of the colormap), collapsing the figure to one hue.
            cmap_obj = plt.get_cmap(cmap, N)
        else:
            # already a Colormap object — resample if it doesn't match N
            cmap_obj = (cmap.resampled(N)
                        if hasattr(cmap, "resampled") and getattr(cmap, "N", N) != N
                        else cmap)
        norm = BoundaryNorm(levels, cmap_obj.N)
        mesh = ax.pcolormesh(Lon, Lat, data, transform=data_crs,
                             cmap=cmap_obj, norm=norm, shading="nearest")
        # extend="both" whenever the colorbar spans negative values (anomalies, diverging
        # linear scales). Sequential positive-only fields stay extend="max".
        extend = "both" if levels[0] < 0 else "max"
        cbar = fig.colorbar(mesh, ax=ax, orientation="vertical", shrink=0.7,
                            pad=0.02, ticks=levels, extend=extend, drawedges=True)
    else:
        mesh = ax.pcolormesh(Lon, Lat, data, transform=data_crs,
                             cmap=(cmap or "viridis"), shading="nearest")
        cbar = fig.colorbar(mesh, ax=ax, orientation="vertical", shrink=0.7,
                            pad=0.02, drawedges=True)
    cbar.dividers.set_color("black")          # crisp lines between discrete classes
    cbar.dividers.set_linewidth(0.6)
    cbar.outline.set_linewidth(0.8)
    cbar.set_label(cbar_label or f"{title} [{units}]", fontsize=9, weight="bold")
    cbar.ax.tick_params(labelsize=7)
    for t in cbar.ax.get_yticklabels():
        t.set_fontweight("bold")

    if boundaries:
        ax.add_feature(ShapelyFeature(boundaries, data_crs, facecolor="none",
                                      edgecolor="#444444", linewidth=0.4))

    # HUC watershed overlay — finer levels have more polygons, so thinner lines.
    if huc_boundaries:
        lw = {4: 1.1, 6: 0.9, 8: 0.7, 10: 0.5, 12: 0.35}.get(huc_level, 0.7)
        ax.add_feature(ShapelyFeature(huc_boundaries, data_crs, facecolor="none",
                                      edgecolor="#0b3d4f", linewidth=lw, alpha=0.9))
        # legend proxy so the overlay is labeled
        from matplotlib.lines import Line2D
        ax.legend([Line2D([0], [0], color="#0b3d4f", lw=max(lw, 0.8))],
                  [f"HUC-{huc_level} watershed boundaries"],
                  loc="lower left", fontsize=7, framealpha=0.85).get_frame().set_edgecolor("none")

    # graticule: longitude labels on the BOTTOM border, latitude on the LEFT —
    # explicit locators keep them on the axes instead of floating along curved meridians.
    gl = ax.gridlines(crs=data_crs, draw_labels=True, linewidth=0.3,
                      color="gray", alpha=0.4, linestyle=":",
                      x_inline=False, y_inline=False, rotate_labels=False)
    gl.top_labels = False
    gl.right_labels = False
    gl.bottom_labels = True
    gl.left_labels = True
    if xticks is not None:
        gl.xlocator = mticker.FixedLocator(xticks)
    if yticks is not None:
        gl.ylocator = mticker.FixedLocator(yticks)
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {"size": 8, "weight": "bold", "color": "#333333"}
    gl.ylabel_style = {"size": 8, "weight": "bold", "color": "#333333"}

    ax.set_title(title, pad=8)
    return fig, ax


# ----- post-render quality checks (cheap assertions; catch silent breakage) -----
def check_figure(fig, ax) -> list[str]:
    warnings = []
    if not ax.get_title():
        warnings.append("no title")
    if not fig.axes or len(fig.axes) < 2:
        warnings.append("no colorbar axis found")
    return warnings


def watershed_series_plot(
    *,
    years: list[int],
    values: dict,                 # {year: value}
    yoy: dict | None = None,      # {year: yoy_change} -> adds a second panel
    units: str,
    watershed_name: str,
    huc8: str | None = None,
    statistic: str = "mean",
    title: str | None = None,
    highlight_years: list | None = None,
):
    """Annual basin-statistic bars (+ optional year-over-year change panel). Dumb renderer:
    receives the computed series and draws. Wettest/driest years highlighted; period mean
    drawn as a reference line."""
    _style()
    yrs = list(years)
    vals = np.array([values[y] for y in yrs], dtype=float)
    mean = float(np.nanmean(vals))

    two = yoy is not None
    fig, axes = plt.subplots(
        2 if two else 1, 1, sharex=True,
        figsize=(max(8.0, len(yrs) * 0.55), 6.6 if two else 4.2),
        gridspec_kw={"height_ratios": [2, 1]} if two else None,
    )
    axes = list(np.atleast_1d(axes))
    ax0 = axes[0]

    # annual bars, with wettest (purple) and driest (orange) highlighted
    base, wet, dry, hot = "#3a7bbf", "#6a51a3", "#cc7a3b", "#e35d3a"
    colors = [base] * len(yrs)
    if highlight_years:
        for i, y in enumerate(yrs):
            if y in highlight_years:
                colors[i] = hot
    else:
        colors[int(np.nanargmax(vals))] = wet
        colors[int(np.nanargmin(vals))] = dry
    ax0.bar(yrs, vals, color=colors, edgecolor="#10263d", linewidth=0.6, width=0.72, zorder=3)
    ax0.axhline(mean, ls="--", lw=1.2, color="#444444", zorder=4)
    ax0.text(0.995, mean, f"mean {mean:.0f}", transform=ax0.get_yaxis_transform(),
             va="bottom", ha="right", fontsize=8, fontweight="bold", color="#444444")
    ax0.set_ylabel(f"Basin-{statistic} snowfall ({units})", fontweight="bold")
    ax0.grid(axis="y", alpha=0.3)
    ax0.margins(x=0.01)

    if title is None:
        title = (f"{watershed_name}" + (f" (HUC-8 {huc8})" if huc8 else "")
                 + f" — Basin-{statistic} Snowfall, WY{yrs[0]}–WY{yrs[-1]}")
    ax0.set_title(title, fontweight="bold", pad=8)

    if two:
        ax1 = axes[1]
        yy = [yoy.get(y, np.nan) for y in yrs]
        bar_colors = ["#b2182b" if (v is not None and v < 0) else "#2166ac"
                      for v in yy]
        ax1.bar(yrs, [0 if v is None else v for v in yy], color=bar_colors,
                edgecolor="#333333", linewidth=0.4, width=0.72, zorder=3)
        ax1.axhline(0, color="#333333", lw=0.8)
        ax1.set_ylabel(f"YoY change ({units})", fontweight="bold")
        ax1.grid(axis="y", alpha=0.3)

    bottom = axes[-1]
    bottom.set_xlabel("Water year", fontweight="bold")
    bottom.set_xticks(yrs)
    bottom.set_xticklabels([str(y) for y in yrs], rotation=45, ha="right",
                           fontsize=8, fontweight="bold")
    for a in axes:
        for lbl in a.get_yticklabels():
            lbl.set_fontweight("bold")
    fig.tight_layout()
    return fig, axes


def ranked_bar_plot(*, labels, values, units, title, statistic="mean"):
    """Horizontal sorted bars — multi-watershed comparison."""
    _style()
    order = np.argsort([(-1e18 if v is None else v) for v in values])[::-1]
    labels = [labels[i] for i in order]; values = [values[i] for i in order]
    fig, ax = plt.subplots(figsize=(8.5, max(3.2, 0.5 * len(labels) + 1.2)))
    colors = ["#3a7bbf"] * len(values)
    if values:
        colors[0] = "#6a51a3"; colors[-1] = "#cc7a3b"
    y = np.arange(len(labels))[::-1]
    ax.barh(y, [0 if v is None else v for v in values], color=colors,
            edgecolor="#10263d", linewidth=0.6, zorder=3)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontweight="bold", fontsize=9)
    ax.set_xlabel(f"Basin-{statistic} snowfall ({units})", fontweight="bold")
    ax.set_title(title, fontweight="bold", pad=8); ax.grid(axis="x", alpha=0.3)
    for lbl in ax.get_xticklabels(): lbl.set_fontweight("bold")
    fig.tight_layout(); return fig, ax


def box_distribution_plot(*, values, units, title, name, statistic="mean"):
    """Box-whisker of the interannual distribution, with jittered yearly points."""
    _style()
    vals = np.array([v for v in values if v is not None], float)
    fig, ax = plt.subplots(figsize=(5.5, 6))
    bp = ax.boxplot(vals, widths=0.5, patch_artist=True, showmeans=True,
                    medianprops=dict(color="#10263d", linewidth=2),
                    meanprops=dict(marker="D", markerfacecolor="#cc7a3b",
                                   markeredgecolor="#10263d", markersize=8))
    for b in bp["boxes"]:
        b.set(facecolor="#9ec0e8", edgecolor="#10263d")
    x = np.random.default_rng(0).normal(1, 0.04, len(vals))
    ax.scatter(x, vals, s=24, color="#3a5fae", alpha=0.75, zorder=3,
               edgecolor="white", linewidth=0.5)
    ax.set_xticks([1]); ax.set_xticklabels([name], fontweight="bold")
    ax.set_ylabel(f"Basin-{statistic} snowfall ({units})", fontweight="bold")
    ax.set_title(title, fontweight="bold", pad=8); ax.grid(axis="y", alpha=0.3)
    for lbl in ax.get_yticklabels(): lbl.set_fontweight("bold")
    fig.tight_layout(); return fig, ax


def trend_plot(*, years, values, trend, units, title, statistic="mean"):
    """Scatter of annual values + OLS fit line + trend statistics annotation."""
    _style()
    yrs = np.array(years); vals = np.array(values, float)
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.scatter(yrs, vals, s=52, color="#3a5fae", zorder=3, edgecolor="white", linewidth=0.6)
    xs = np.array([yrs.min(), yrs.max()]); ys = trend["intercept"] + trend["slope_per_year"] * xs
    ax.plot(xs, ys, "--", color="#cc7a3b", lw=2, zorder=2)
    txt = (f"{trend['slope_per_decade']:+.1f} {units}/decade\n"
           f"R\u00b2={trend['r_squared']:.2f}   OLS p={trend['p_value_ols']:.2f}\n"
           f"Mann\u2013Kendall p={trend['p_value_mannkendall']:.2f}")
    ax.text(0.025, 0.97, txt, transform=ax.transAxes, va="top", ha="left",
            fontsize=9, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#cc7a3b", alpha=0.9))
    ax.set_xlabel("Water year", fontweight="bold")
    ax.set_ylabel(f"Basin-{statistic} snowfall ({units})", fontweight="bold")
    ax.set_title(title, fontweight="bold", pad=8); ax.grid(alpha=0.3)
    ax.set_xticks(yrs); ax.set_xticklabels([str(y) for y in yrs], rotation=45, ha="right")
    for lbl in ax.get_xticklabels() + ax.get_yticklabels(): lbl.set_fontweight("bold")
    fig.tight_layout(); return fig, ax
