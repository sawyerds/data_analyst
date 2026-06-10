#!/usr/bin/env python3
"""
render.py — the generic entry point. Open ANY recognized netCDF, auto-detect its
DatasetProfile, prep per the profile, and render at the requested region. No per-file
params: the profile supplies colormap/units/levels, the region supplies extent/projection.

    python render.py sfav2.nc                      # default region (CONUS)
    python render.py sfav2.nc west_coast           # zoom to a named region
    python render.py sfav2.nc california out.png   # region + output path

The `cmap` parameter on field_map_figure() lets callers (e.g. service_snowfall.py)
override the profile's default colormap from spec.colormap.
"""
from __future__ import annotations
import sys
import numpy as np
from netCDF4 import Dataset

from profiles import identify
from regions import get_region
from plots import gridded_field_map, load_boundaries, check_figure
from style import annotate_provenance
from watersheds import load_huc8_in_extent   # real WBD HUC-8, same source as analysis

BOUNDARIES = "us_states.geojson"


def _read_coords(ds, coord_kind):
    if coord_kind == "regular_1d":
        return ds.variables["lat"][:], ds.variables["lon"][:]
    if coord_kind == "wrf_curvilinear":          # 2D XLAT/XLONG (time 0)
        la = np.asarray(ds.variables["XLAT"][:]); lo = np.asarray(ds.variables["XLONG"][:])
        return (la[0] if la.ndim == 3 else la), (lo[0] if lo.ndim == 3 else lo)
    raise ValueError(f"unhandled coord_kind {coord_kind!r}")


def _nice_ticks(extent):
    """Pick clean tick spacing from the domain width (so labels aren't crowded)."""
    lo0, lo1, la0, la1 = extent
    def step(span):
        for s in (2, 5, 10, 15):
            if span / s <= 7:
                return s
        return 20
    import math
    dx, dy = step(abs(lo1 - lo0)), step(abs(la1 - la0))
    xs = list(range(int(math.ceil(lo0 / dx) * dx), int(lo1) + 1, dx))
    ys = list(range(int(math.ceil(la0 / dy) * dy), int(la1) + 1, dy))
    return xs, ys


def field_map_figure(path: str, region: str | None = None, huc_level: int | None = None,
                     cmap: str | None = None, levels: list | None = None):
    """Build (and return) the field-map figure without saving. Returns (fig, ax, meta).

    `cmap`:   when provided (non-empty), overrides the profile's default colormap.
              When None or empty, falls back to `prof.cmap`.
    `levels`: when provided, overrides the profile's default contour levels (so the
              user can control the colorbar range / bin count from the worksheet).
              When None, falls back to `prof.levels`.
    """
    ds = Dataset(path)
    prof = identify(ds)
    if prof is None:
        ds.close()
        raise ValueError(f"no DatasetProfile matched {path}. "
                         f"Add one in profiles.py to support this data type.")
    lats, lons = _read_coords(ds, prof.coord_kind)
    raw = ds.variables[prof.field_var][:]
    title = prof.title(ds)
    ds.close()

    data = np.ma.masked_invalid(np.ma.filled(raw, np.nan).astype("f8"))
    data = prof.convert(data)                       # numeric work (units) here, not in renderer

    region = region or prof.default_region
    reg = get_region(region)
    xs, ys = _nice_ticks(reg["extent"])

    huc_geoms = None
    if huc_level:
        huc_geoms = load_huc8_in_extent(reg["extent"])   # real WBD, clipped to region

    eff_levels = levels if levels is not None else prof.levels
    eff_cmap = cmap or prof.cmap
    full_title = title if region == "conus" else f"{title}\n{reg['label']}"
    fig, ax = gridded_field_map(
        data=data, lats=lats, lons=lons,
        units=prof.display_units, cbar_label=prof.cbar_label,
        title=full_title, levels=eff_levels, cmap=eff_cmap,
        extent=reg["extent"], central_lon=reg["central_lon"],
        standard_parallels=reg["parallels"],
        boundaries=load_boundaries(BOUNDARIES),
        huc_boundaries=huc_geoms, huc_level=8 if huc_geoms else None,
        xticks=xs, yticks=ys,
    )
    prov = f"profile: {prof.name} | region: {region} | source: {path}"
    if huc_geoms:
        prov += f" | HUC-8 overlay ({len(huc_geoms)} basins)"
    if cmap:
        prov += f" | cmap: {cmap}"
    if levels is not None:
        prov += f" | levels: {len(levels)-1} bins ({eff_levels[0]:.1f}\u2026{eff_levels[-1]:.1f})"
    annotate_provenance(fig, prov)
    meta = {"profile": prof.name, "region": region, "field": prof.field_var,
            "n_huc8": len(huc_geoms) if huc_geoms else 0,
            "cmap": eff_cmap, "n_levels": len(eff_levels) - 1 if eff_levels else None}
    return fig, ax, meta


def render_file(path: str, region: str | None = None, out: str | None = None,
                huc_level: int | None = None, cmap: str | None = None,
                levels: list | None = None):
    fig, ax, meta = field_map_figure(path, region, huc_level, cmap=cmap, levels=levels)
    if meta["n_huc8"]:
        print(f"HUC-8 overlay: {meta['n_huc8']} real CA watersheds in region")
    w = check_figure(fig, ax)
    print("post-render checks:", w or "clean")
    region = region or meta["region"]
    out = out or f"{meta['field']}_{region}{('_huc'+str(huc_level)) if huc_level else ''}.png"
    fig.savefig(out, dpi=200)
    print(f"identified profile: {meta['profile']}")
    print(f"wrote {out}")
    return out


if __name__ == "__main__":
    pos = [a for a in sys.argv[1:] if not a.startswith("--")]
    huc = None
    cmap = None
    for a in sys.argv[1:]:
        if a.startswith("--huc") and "=" in a:
            huc = int(a.split("=")[1])
        elif a.startswith("--cmap") and "=" in a:
            cmap = a.split("=", 1)[1]
    path = pos[0] if pos else "sfav2.nc"
    region = pos[1] if len(pos) > 1 else None
    out = None
    if len(pos) > 2:                      # 3rd positional: HUC level if numeric, else out path
        if pos[2].isdigit():
            huc = huc or int(pos[2])
        else:
            out = pos[2]
    if len(pos) > 3 and pos[3].isdigit():
        huc = huc or int(pos[3])
    render_file(path, region, out, huc_level=huc, cmap=cmap)
