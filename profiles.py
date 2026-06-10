#!/usr/bin/env python3
"""
profiles.py — couple a dataset/data-type to its render settings. A DatasetProfile
fingerprints a file (matches) and supplies frozen presets: which variable is the
field, unit conversion, colormap, class levels, title, default region, coord handling.

identify(ds) returns the first matching profile, so opening a recognized file
auto-configures the render — no manual params. This is the data-source-adapter
pattern applied to plotting: add a new product = add one DatasetProfile entry.

Two coord_kind values are handled downstream:
  "regular_1d"      -> 1D lat/lon (NOHRSC, most analysis grids)
  "wrf_curvilinear" -> 2D XLAT/XLONG (WRF; field chosen per-run, see note below)

For single-field products (sfav2) the profile fully specifies the look. For WRF the
profile handles coords, and PER-VARIABLE presets (reflectivity/precip/temp colormaps)
layer on top — that's the generalization seam, stubbed in VARIABLE_PRESETS below.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Optional
import numpy as np
from netCDF4 import Dataset

from style import SNOW_COLORS

M_TO_IN = 39.3701
SNOW_BINS_IN = [0.1, 1, 3, 6, 12, 18, 24, 36, 48, 72, 96, 144, 192, 288, 480, 720, 1080]


@dataclass
class DatasetProfile:
    name: str
    coord_kind: str                 # "regular_1d" | "wrf_curvilinear"
    field_var: str
    display_units: str
    cbar_label: str
    levels: Optional[list]
    cmap: object
    default_region: str
    _match: Callable[[Dataset], bool]
    _convert: Callable[[np.ndarray], np.ndarray]
    _title: Callable[[Dataset], str]

    def matches(self, ds) -> bool: return self._match(ds)
    def convert(self, a): return self._convert(a)
    def title(self, ds) -> str: return self._title(ds)


# ---- NOHRSC National Snowfall Analysis v2 (sfav2) ----
def _nohrsc_match(ds) -> bool:
    fv = str(getattr(ds, "format_version", "")) + str(getattr(ds, "title", ""))
    has_field = "Data" in ds.variables and \
        getattr(ds.variables.get("Data"), "standard_name", "") == "thickness_of_snowfall_amount"
    return ("NOHRSC" in fv) or has_field

def _nohrsc_title(ds) -> str:
    d = ds.variables["Data"]
    s = str(getattr(d, "start_date", ""))[:10]
    e = str(getattr(d, "stop_date", ""))[:10]
    return f"NOHRSC National Snowfall Analysis v2.1\n{s} to {e}"

NOHRSC_SFAV2 = DatasetProfile(
    name="NOHRSC National Snowfall Analysis v2 (sfav2)",
    coord_kind="regular_1d",
    field_var="Data",
    display_units="in",
    cbar_label="Snowfall accumulation (in)",
    levels=SNOW_BINS_IN,
    cmap=SNOW_COLORS,
    default_region="conus",
    _match=_nohrsc_match,
    _convert=lambda a: a * M_TO_IN,
    _title=_nohrsc_title,
)


# registry — ordered; first match wins
PROFILES: list[DatasetProfile] = [NOHRSC_SFAV2]

def identify(ds) -> Optional[DatasetProfile]:
    for p in PROFILES:
        try:
            if p.matches(ds):
                return p
        except Exception:
            continue
    return None


# ---- per-variable render presets (the WRF generalization seam; not yet wired) ----
# When WRF support lands, the profile gives coords and these give each field's look.
VARIABLE_PRESETS = {
    "REFL_10CM": {"display_units": "dBZ",   "levels": [5,10,15,20,25,30,35,40,45,50,55,60,65,70]},
    "RAINNC":    {"display_units": "mm",     "levels": [0.1,1,2,5,10,15,25,50,75,100,150,200,300]},
    "T2":        {"display_units": "degC",   "levels": None},   # continuous cmap
}


if __name__ == "__main__":
    import sys
    ds = Dataset(sys.argv[1] if len(sys.argv) > 1 else "sfav2.nc")
    p = identify(ds)
    print("identified:", p.name if p else "NO MATCH")
    if p:
        print("  field:", p.field_var, "| units:", p.display_units,
              "| coord_kind:", p.coord_kind, "| default region:", p.default_region)
        print("  title:", p.title(ds).replace("\n", " / "))
    ds.close()
