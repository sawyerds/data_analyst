#!/usr/bin/env python3
"""
zonal.py — spatial statistics of a gridded field WITHIN a polygon (watershed scale).
Deterministic 'execute' step for watershed analyses. Cell-center point-in-polygon
after a bbox subset for speed.
"""
from __future__ import annotations
import numpy as np
import shapely


def _cells_inside(lat, lon, data, polygon):
    minx, miny, maxx, maxy = polygon.bounds
    li = np.where((lon >= minx) & (lon <= maxx))[0]
    lj = np.where((lat >= miny) & (lat <= maxy))[0]
    if li.size == 0 or lj.size == 0:
        return np.array([]), np.array([]), 0
    sub = np.ma.masked_invalid(np.asarray(data)[np.ix_(lj, li)])
    Lon, Lat = np.meshgrid(lon[li], lat[lj])
    pts = shapely.points(Lon.ravel(), Lat.ravel())
    inside = shapely.contains(polygon, pts).reshape(Lon.shape)
    sel = inside & ~np.ma.getmaskarray(sub)
    vals = np.asarray(sub)[sel]
    return vals[np.isfinite(vals)], Lat, int(inside.sum())


def zonal_stats(lat, lon, data, polygon, stats=("mean", "max", "min"),
                threshold: float | None = None, min_coverage: float = 0.0) -> dict:
    """Aggregate `data` over cells whose centers fall in `polygon`.
    Supported stat keys: mean, max, min, median, std, p10, p25, p75, p90, p95, p99,
    sum, range. If `threshold` given, also returns frac_above / count_above."""
    vals, _, cells_in = _cells_inside(lat, lon, data, polygon)
    out = {"n_cells": int(vals.size), "cells_in_polygon": cells_in}
    if cells_in and vals.size / cells_in < min_coverage:
        out["note"] = f"coverage {vals.size}/{cells_in} below min {min_coverage}"
        return out
    if vals.size == 0:
        out["note"] = "no finite cells in polygon"
        return out

    funcs = {
        "mean": np.mean, "max": np.max, "min": np.min, "median": np.median,
        "std": np.std, "sum": np.sum,
        "range": lambda a: float(np.max(a) - np.min(a)),
    }
    for s in stats:
        if s in funcs:
            out[s] = float(funcs[s](vals))
        elif s.startswith("p") and s[1:].isdigit():
            out[s] = float(np.percentile(vals, int(s[1:])))
    # always include the common bundle for convenience
    for s in ("mean", "max", "min", "median", "std"):
        out.setdefault(s, float(funcs[s](vals)))
    if threshold is not None:
        above = int(np.count_nonzero(vals >= threshold))
        out["count_above"] = above
        out["frac_above"] = float(above / vals.size)
    return out


def pick(stat: str, result: dict):
    """Return the value for a requested statistic key, handling area_above->frac_above."""
    if stat == "area_above":
        return result.get("frac_above")
    return result.get(stat)


if __name__ == "__main__":
    import numpy as np
    from netCDF4 import Dataset
    from watersheds import load_polygon
    ds = Dataset("sfav2.nc"); lat = ds.variables["lat"][:]; lon = ds.variables["lon"][:]
    snow = np.ma.filled(ds.variables["Data"][:], np.nan).astype("f8") * 39.3701; ds.close()
    code, name, geom = load_polygon("Upper Yuba")
    s = zonal_stats(lat, lon, snow, geom, stats=("mean", "max", "p90", "median"), threshold=100)
    print(name, {k: (round(v, 2) if isinstance(v, float) else v) for k, v in s.items()})
