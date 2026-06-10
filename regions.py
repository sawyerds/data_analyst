#!/usr/bin/env python3
"""
regions.py — named geographic domains. Pass region="california" (etc.) to any renderer
to zoom; each carries an extent plus Lambert-Conformal projection params tuned so the
zoomed map stays undistorted. West-Coast / CW3E regions included since that's the focus.

extent = (lon_min, lon_max, lat_min, lat_max); parallels = LCC standard parallels.
"""

REGIONS = {
    "conus":         {"extent": (-125, -66.5, 23, 50),    "central_lon": -96.0,  "parallels": (33, 45),   "label": "CONUS"},
    "west":          {"extent": (-125, -102, 31, 49.5),   "central_lon": -113.0, "parallels": (33, 45),   "label": "Western U.S."},
    "west_coast":    {"extent": (-125.5, -114, 32, 49),   "central_lon": -120.0, "parallels": (34, 46),   "label": "West Coast"},
    "california":    {"extent": (-124.6, -113.5, 32, 42.3),"central_lon": -119.5,"parallels": (33, 40),   "label": "California"},
    "norcal":        {"extent": (-124.6, -119.5, 36.5, 42.2),"central_lon":-122.0,"parallels": (37, 41),  "label": "Northern California"},
    "sierra_nevada": {"extent": (-122, -117.5, 35.5, 40),  "central_lon": -119.5, "parallels": (36, 39),   "label": "Sierra Nevada"},
    "pacific_nw":    {"extent": (-125, -116, 41.5, 49.2),  "central_lon": -120.5, "parallels": (43, 47),   "label": "Pacific Northwest"},
    "rockies":       {"extent": (-116, -104, 35, 49.3),    "central_lon": -110.0, "parallels": (37, 46),   "label": "Rockies"},
    "southwest":     {"extent": (-120, -103, 31, 42),      "central_lon": -111.0, "parallels": (33, 40),   "label": "Southwest"},
    "great_lakes":   {"extent": (-93, -75.5, 40, 49.2),    "central_lon": -84.0,  "parallels": (41, 47),   "label": "Great Lakes"},
    "northeast":     {"extent": (-80.5, -66.5, 39, 47.6),  "central_lon": -73.0,  "parallels": (40, 46),   "label": "Northeast"},
    # tight CW3E/AQPI domain
    "russian_river": {"extent": (-124.2, -122.0, 38.0, 39.7),"central_lon": -123.1,"parallels": (38.3, 39.3),"label": "Russian River"},
}

def get_region(name: str) -> dict:
    if name not in REGIONS:
        raise KeyError(f"unknown region {name!r}; choices: {sorted(REGIONS)}")
    return REGIONS[name]
