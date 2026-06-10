#!/usr/bin/env python3
"""
watersheds.py — resolve California HUC-8 watersheds by name and load their polygons.

  name <-> huc8 code : from ca_huc8_watersheds.csv (also the worksheet dropdown source)
  polygon            : from per-HUC8 shapefiles in HUC8_SHP_DIR (USGS WBD)

Reads local files only (works air-gapped on AWARE). Point at them with:
  WXAGENT_HUC8_DIR  (default ./huc8_shp)   — directory of {code}.shp
  WXAGENT_HUC8_CSV  (default ./ca_huc8_watersheds.csv)
"""
from __future__ import annotations
import os, csv, functools
import geopandas as gpd

HUC8_SHP_DIR = os.environ.get("WXAGENT_HUC8_DIR", "./huc8_shp")
HUC8_CSV = os.environ.get("WXAGENT_HUC8_CSV", "./ca_huc8_watersheds.csv")


@functools.lru_cache(maxsize=1)
def _tables():
    rows = list(csv.DictReader(open(HUC8_CSV)))
    by_name = {r["name"].lower(): r["huc8"] for r in rows}
    by_code = {r["huc8"]: r["name"] for r in rows}
    return by_name, by_code


def list_watersheds() -> list[tuple[str, str]]:
    """(code, name) for every CA HUC-8 — the worksheet dropdown."""
    _, by_code = _tables()
    return sorted(by_code.items(), key=lambda x: x[1])


def resolve(name_or_code: str) -> tuple[str, str]:
    """Return (huc8_code, name). Accepts exact code, exact name, or unique substring."""
    by_name, by_code = _tables()
    s = str(name_or_code).strip()
    if s in by_code:
        return s, by_code[s]
    if s.lower() in by_name:
        code = by_name[s.lower()]
        return code, by_code[code]
    hits = [(c, by_code[c]) for n, c in by_name.items() if s.lower() in n]
    if len(hits) == 1:
        return hits[0]
    raise KeyError(f"watershed {name_or_code!r} not found or ambiguous; "
                   f"candidates: {[n for _, n in hits][:6] or 'none'}")


def load_polygon(name_or_code: str, to_crs: str = "EPSG:4326"):
    """Return (code, name, shapely geometry in to_crs) for one HUC-8."""
    code, name = resolve(name_or_code)
    shp = os.path.join(HUC8_SHP_DIR, f"{code}.shp")
    if not os.path.exists(shp):
        raise FileNotFoundError(f"no shapefile {shp} for HUC-8 {code} ({name})")
    g = gpd.read_file(shp)
    if to_crs and g.crs and g.crs.to_string() != to_crs:
        g = g.to_crs(to_crs)
    geom = g.geometry.union_all() if hasattr(g.geometry, "union_all") else g.geometry.unary_union
    return code, name, geom


_COMBINED_CACHE = os.environ.get("WXAGENT_HUC8_CACHE", "ca_huc8_combined.geojson")

@functools.lru_cache(maxsize=1)
def ca_huc8_layer():
    """All California HUC-8 polygons as one GeoDataFrame in WGS84. Built once from the
    per-HUC8 shapefiles (slow), then cached to disk + memory so map overlays are cheap.
    This is the SAME real WBD source the zonal analysis uses — no more synthetic overlay."""
    import pandas as pd
    if os.path.exists(_COMBINED_CACHE):
        return gpd.read_file(_COMBINED_CACHE)
    _, by_code = _tables()
    parts = []
    for code in by_code:
        shp = os.path.join(HUC8_SHP_DIR, f"{code}.shp")
        if os.path.exists(shp):
            parts.append(gpd.read_file(shp)[["huc8", "name", "geometry"]])
    layer = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=parts[0].crs)
    layer = layer.to_crs("EPSG:4326")
    try:
        layer.to_file(_COMBINED_CACHE, driver="GeoJSON")   # cache for next time
    except Exception:
        pass
    return layer


def load_huc8_in_extent(extent: tuple) -> list:
    """Real CA HUC-8 polygons intersecting extent=(lon_min,lon_max,lat_min,lat_max).
    For map overlays — same WBD data as the analysis, clipped to the map region."""
    from shapely.geometry import box
    layer = ca_huc8_layer()
    clip = box(extent[0], extent[2], extent[1], extent[3])
    return [g for g in layer.geometry if g.intersects(clip)]


if __name__ == "__main__":
    import sys
    q = sys.argv[1] if len(sys.argv) > 1 else "Upper Yuba"
    code, name, geom = load_polygon(q)
    print(f"{name} (HUC-8 {code}) | bounds {tuple(round(b,3) for b in geom.bounds)}")
    print(f"total CA HUC-8 watersheds available: {len(list_watersheds())}")
