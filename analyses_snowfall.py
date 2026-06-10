#!/usr/bin/env python3
"""
analyses_snowfall.py — NOHRSC sfav2 snowfall analyses.

This module owns everything snowfall-specific: file discovery, climatology engine,
spec definition, worksheet schema, and the execute() that produces the analysis
result (numbers; rendering lives in service_snowfall.py).

Pre-refactor this was just analyses.py. The public dispatcher analyses.py now
re-exports the names that app.py / service_snowfall.py used to import directly,
so external callers don't break.
"""
from __future__ import annotations
import os, re, glob
from typing import Literal, Optional

import numpy as np
from netCDF4 import Dataset
from pydantic import model_validator

from watersheds import load_polygon, list_watersheds, resolve as resolve_watershed
from zonal import zonal_stats, pick
from regions import REGIONS
from analyses_common import (
    NotWired, BaseAnalysisFields, _opt,
    map_style_field_specs, meta_style_field_specs, output_field_specs,
)

M_TO_IN = 39.3701


# ----------------------------- file discovery -----------------------------
def water_year_from_name(fn: str) -> Optional[int]:
    m = re.findall(r"(\d{10})", os.path.basename(fn))
    return int(m[1][:4]) if len(m) >= 2 else (int(m[0][:4]) if m else None)


def find_sfav2_files(directory: str) -> dict[int, str]:
    out = {}
    for f in glob.glob(os.path.join(directory, "*.nc")):
        wy = water_year_from_name(f)
        if wy:
            out[wy] = f
    return dict(sorted(out.items()))


def _read_snow_inches(path):
    ds = Dataset(path)
    lat = ds.variables["lat"][:]; lon = ds.variables["lon"][:]
    snow = np.ma.masked_invalid(np.ma.filled(ds.variables["Data"][:], np.nan).astype("f8"))
    ds.close()
    return lat, lon, snow * M_TO_IN


# ----------------------------- climatology engine -----------------------------
def basin_series(watershed, statistic, files, threshold=None, min_coverage=0.0):
    """Per-water-year basin statistic -> (code, name, geom, {year: value})."""
    code, name, geom = load_polygon(watershed)
    base_stat = "mean" if statistic == "area_above" else statistic
    series = {}
    for wy, path in files.items():
        lat, lon, snow = _read_snow_inches(path)
        s = zonal_stats(lat, lon, snow, geom, stats=(base_stat,),
                        threshold=threshold, min_coverage=min_coverage)
        series[wy] = pick(statistic, s)
    return code, name, geom, dict(sorted(series.items()))


def climatology(series, baseline=None):
    yrs = sorted(y for y in series if series[y] is not None)
    if baseline:
        a, b = baseline; yrs = [y for y in yrs if a <= y <= b]
    vals = np.array([series[y] for y in yrs], float)
    return {"normal_mean": float(vals.mean()), "normal_median": float(np.median(vals)),
            "std": float(vals.std()), "min": float(vals.min()), "max": float(vals.max()),
            "n_years": len(vals), "baseline": [yrs[0], yrs[-1]],
            "wettest_year": int(max(yrs, key=lambda y: series[y])),
            "driest_year": int(min(yrs, key=lambda y: series[y]))}


def anomaly_for_year(series, year, anomaly_type, baseline=None):
    clim = climatology(series, baseline)
    v = series[year]
    if anomaly_type == "absolute":
        val = v - clim["normal_mean"]
    elif anomaly_type == "percent":
        val = 100.0 * v / clim["normal_mean"]
    else:  # standardized
        val = (v - clim["normal_mean"]) / clim["std"] if clim["std"] else None
    allvals = sorted(s for s in series.values() if s is not None)
    pct_rank = 100.0 * sum(1 for x in allvals if x <= v) / len(allvals)
    return {"year": year, "value": v, "anomaly_type": anomaly_type, "anomaly": val,
            "percentile_rank": pct_rank, "climatology": clim}


def trend(series):
    from scipy import stats
    yrs = np.array(sorted(y for y in series if series[y] is not None))
    vals = np.array([series[y] for y in yrs], float)
    lr = stats.linregress(yrs, vals)
    tau, p_mk = stats.kendalltau(yrs, vals)
    return {"slope_per_year": float(lr.slope), "slope_per_decade": float(lr.slope * 10),
            "intercept": float(lr.intercept), "r_squared": float(lr.rvalue ** 2),
            "p_value_ols": float(lr.pvalue), "kendall_tau": float(tau),
            "p_value_mannkendall": float(p_mk), "n_years": int(len(yrs)),
            "direction": "increasing" if lr.slope > 0 else "decreasing"}


def cell_climatology(files, baseline=None):
    """Per-cell mean + std across water years (streaming). Returns lat,lon,mean,std,n."""
    yrs = sorted(y for y in files if (not baseline or baseline[0] <= y <= baseline[1]))
    s = ss = c = LAT = LON = None
    for wy in yrs:
        lat, lon, snow = _read_snow_inches(files[wy])
        arr = np.ma.filled(snow, np.nan); m = np.isfinite(arr)
        a0 = np.where(m, arr, 0.0)
        if s is None:
            s = a0.copy(); ss = a0 * a0; c = m.astype("f8"); LAT, LON = lat, lon
        else:
            s += a0; ss += a0 * a0; c += m
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(c > 0, s / c, np.nan)
        var = np.where(c > 0, ss / np.where(c > 0, c, 1) - mean ** 2, np.nan)
    std = np.sqrt(np.clip(var, 0, None))
    return LAT, LON, mean, std, len(yrs)


# ----------------------------- spec -----------------------------
class SnowfallSpec(BaseAnalysisFields):
    """Snowfall analyses. data_source discriminator pins this to the snowfall branch."""
    data_source: Literal["snowfall"] = "snowfall"

    analysis_type: Literal[
        "map", "map_anomaly", "watershed_mean", "watershed_yoy", "watershed_anomaly",
        "watershed_climatology", "watershed_trend", "watershed_distribution",
        "multi_watershed", "elevation_profile", "climate_index_correlation"]

    # spatial
    region: str = "norcal"
    huc8_overlay: bool = False
    show_states: bool = True
    watershed: Optional[str] = None
    watersheds: Optional[list[str]] = None
    # statistic
    statistic: Literal["mean", "max", "min", "median", "std", "p90", "p95", "area_above"] = "mean"
    threshold: Optional[float] = None
    min_coverage: float = 0.0
    # temporal
    water_year: Optional[int] = None
    years: Literal["all", "range"] = "all"
    year_range: Optional[tuple[int, int]] = None
    baseline_start: Optional[int] = None
    baseline_end: Optional[int] = None
    anomaly_type: Literal["absolute", "percent", "standardized"] = "absolute"
    # plot styling — only the snowfall-specific bits stay here.
    # colormap / color_scale / vmin / vmax / level_step are now inherited
    # from BaseAnalysisFields so every data source's spec gets them for free.
    show_extremes: bool = True
    reference_line: Literal["mean", "median", "none"] = "mean"
    # data-gated
    climate_index: Optional[str] = None
    elevation_min: Optional[float] = None
    elevation_max: Optional[float] = None

    @model_validator(mode="after")
    def _check(self):
        single = ("watershed_mean", "watershed_yoy", "watershed_anomaly",
                  "watershed_climatology", "watershed_trend", "watershed_distribution",
                  "elevation_profile", "climate_index_correlation")
        if self.analysis_type in single:
            if not self.watershed:
                raise ValueError(f"{self.analysis_type} requires a 'watershed'")
            _, name = resolve_watershed(self.watershed)
            self.watershed = name
        if self.analysis_type == "multi_watershed":
            if not self.watersheds or len(self.watersheds) < 2:
                raise ValueError("multi_watershed requires 2+ 'watersheds'")
            self.watersheds = [resolve_watershed(w)[1] for w in self.watersheds]
        if self.analysis_type == "map" and self.region not in REGIONS:
            raise ValueError(f"unknown region {self.region!r}")
        if self.statistic == "area_above" and self.threshold is None:
            raise ValueError("statistic 'area_above' requires a 'threshold'")
        if not (1 <= self.dpi <= 600):
            raise ValueError("dpi must be 1\u2013600")
        return self

    def baseline(self):
        if self.baseline_start and self.baseline_end:
            return (self.baseline_start, self.baseline_end)
        return None


# ----------------------------- worksheet schema -----------------------------
def build_snowfall_worksheet() -> dict:
    ws = [_opt(c, n) for c, n in list_watersheds()]
    regs = [_opt(r, REGIONS[r]["label"]) for r in REGIONS]
    single = ["watershed_mean", "watershed_yoy", "watershed_anomaly",
              "watershed_climatology", "watershed_trend", "watershed_distribution"]
    map_types = ["map", "map_anomaly"]
    anom_types = ["watershed_anomaly", "map_anomaly"]
    clim_ref = ["watershed_anomaly", "watershed_climatology", "map_anomaly"]

    f = []
    f.append({"key": "analysis_type", "label": "Analysis type", "widget": "select", "group": "Analysis", "options": [
        _opt("map", "Field map"), _opt("map_anomaly", "Field map \u2014 anomaly vs climatology"),
        _opt("watershed_mean", "Watershed statistic (one year)"),
        _opt("watershed_yoy", "Watershed year-over-year"),
        _opt("watershed_anomaly", "Watershed anomaly (one year)"),
        _opt("watershed_climatology", "Watershed climatology / normals"),
        _opt("watershed_trend", "Watershed trend"),
        _opt("watershed_distribution", "Watershed distribution (box plot)"),
        _opt("multi_watershed", "Multi-watershed comparison"),
        _opt("elevation_profile", "Elevation-band profile  [needs DEM]"),
        _opt("climate_index_correlation", "Climate-index correlation  [needs index]")]})

    # spatial
    f.append({"key": "region", "label": "Region", "widget": "select", "group": "Spatial",
              "default": "norcal", "options": regs, "depends_on": {"field": "analysis_type", "in": map_types}})
    f.append({"key": "huc8_overlay", "label": "Overlay HUC-8 watersheds", "widget": "toggle", "group": "Spatial",
              "default": False, "depends_on": {"field": "analysis_type", "in": map_types}})
    f.append({"key": "show_states", "label": "State boundaries", "widget": "toggle", "group": "Spatial",
              "default": True, "depends_on": {"field": "analysis_type", "in": map_types}})
    f.append({"key": "watershed", "label": "Watershed (HUC-8)", "widget": "select", "group": "Spatial",
              "options": ws, "depends_on": {"field": "analysis_type", "in": single + ["elevation_profile", "climate_index_correlation"]}})
    f.append({"key": "watersheds", "label": "Watersheds (pick 2+)", "widget": "multiselect", "group": "Spatial",
              "options": ws, "depends_on": {"field": "analysis_type", "equals": "multi_watershed"}})

    # statistic
    stat_types = single + ["multi_watershed"]
    f.append({"key": "statistic", "label": "Spatial statistic", "widget": "select", "group": "Statistic",
              "default": "mean", "options": [
                  _opt("mean", "Basin mean"), _opt("max", "Basin maximum"), _opt("min", "Basin minimum"),
                  _opt("median", "Basin median"), _opt("std", "Basin std dev"),
                  _opt("p90", "90th percentile"), _opt("p95", "95th percentile"),
                  _opt("area_above", "Fraction above threshold")],
              "depends_on": {"field": "analysis_type", "in": stat_types}})
    f.append({"key": "threshold", "label": "Threshold (in)", "widget": "number", "group": "Statistic",
              "default": 100, "depends_on": {"field": "statistic", "equals": "area_above"}})
    f.append({"key": "min_coverage", "label": "Min valid coverage (0\u20131)", "widget": "number", "group": "Statistic",
              "default": 0, "depends_on": {"field": "analysis_type", "in": stat_types}})

    # temporal
    f.append({"key": "water_year", "label": "Water year", "widget": "select", "group": "Temporal",
              "depends_on": {"field": "analysis_type", "in": ["watershed_mean", "watershed_anomaly", "map", "map_anomaly", "multi_watershed"]},
              "help": "Populated from the data directory."})
    f.append({"key": "years", "label": "Water years", "widget": "select", "group": "Temporal", "default": "all",
              "options": [_opt("all", "All available"), _opt("range", "Range")],
              "depends_on": {"field": "analysis_type", "in": ["watershed_yoy", "watershed_trend", "watershed_distribution"]}})
    f.append({"key": "year_start", "label": "Start WY", "widget": "number", "group": "Temporal",
              "depends_on": {"field": "years", "equals": "range"}})
    f.append({"key": "year_end", "label": "End WY", "widget": "number", "group": "Temporal",
              "depends_on": {"field": "years", "equals": "range"}})
    f.append({"key": "baseline_start", "label": "Baseline start WY", "widget": "number", "group": "Temporal",
              "depends_on": {"field": "analysis_type", "in": clim_ref}})
    f.append({"key": "baseline_end", "label": "Baseline end WY", "widget": "number", "group": "Temporal",
              "depends_on": {"field": "analysis_type", "in": clim_ref}})
    f.append({"key": "anomaly_type", "label": "Anomaly type", "widget": "select", "group": "Temporal", "default": "absolute",
              "options": [_opt("absolute", "Absolute (in)"), _opt("percent", "Percent of normal"),
                          _opt("standardized", "Standardized (z-score)")],
              "depends_on": {"field": "analysis_type", "in": anom_types}})

    # plot styling — shared map controls (colormap, color_scale, vmin, vmax, level_step)
    # come from the analyses_common helper so every data source's worksheet exposes them
    # identically. Snowfall-specific bits (show_extremes, reference_line) follow.
    f.extend(map_style_field_specs(map_types, default_cmap_label="Default (NWS snowfall)"))
    f.append({"key": "show_extremes", "label": "Highlight wettest/driest", "widget": "toggle", "group": "Style",
              "default": True, "depends_on": {"field": "analysis_type", "in": ["watershed_yoy", "watershed_trend", "multi_watershed"]}})
    f.append({"key": "reference_line", "label": "Reference line", "widget": "select", "group": "Style", "default": "mean",
              "options": [_opt("mean", "Mean"), _opt("median", "Median"), _opt("none", "None")],
              "depends_on": {"field": "analysis_type", "in": ["watershed_yoy", "watershed_trend"]}})
    f.extend(meta_style_field_specs())

    # data-gated
    f.append({"key": "climate_index", "label": "Climate index", "widget": "select", "group": "Climate index [needs data]",
              "options": [_opt("oni", "ONI / ENSO"), _opt("pdo", "PDO"), _opt("ao", "AO")],
              "depends_on": {"field": "analysis_type", "equals": "climate_index_correlation"}})
    f.append({"key": "elevation_min", "label": "Elevation min (m)", "widget": "number", "group": "Elevation [needs DEM]",
              "depends_on": {"field": "analysis_type", "equals": "elevation_profile"}})
    f.append({"key": "elevation_max", "label": "Elevation max (m)", "widget": "number", "group": "Elevation [needs DEM]",
              "depends_on": {"field": "analysis_type", "equals": "elevation_profile"}})

    # output (always shown) — comes from the analyses_common helper
    f.extend(output_field_specs())

    return {"pipeline": "gridded_snowfall", "fields": f,
            "groups": ["Analysis", "Spatial", "Statistic", "Temporal", "Style",
                       "Climate index [needs data]", "Elevation [needs DEM]", "Output"]}


# ----------------------------- execute (numbers only; figures in service_snowfall.py) -----------------------------
def execute_snowfall(spec: SnowfallSpec, data_dir: str = ".") -> dict:
    files = find_sfav2_files(data_dir)
    at = spec.analysis_type

    if at in ("elevation_profile", "climate_index_correlation"):
        need = "a DEM grid" if at == "elevation_profile" else "a climate-index table (ONI/PDO/AO)"
        raise NotWired(f"'{at}' needs {need}, which isn't wired yet. "
                       f"It's in the schema as the planner's action space; add the data source to enable it.")

    if at == "watershed_mean":
        code, name, geom, series = basin_series(spec.watershed, spec.statistic, files,
                                                threshold=spec.threshold, min_coverage=spec.min_coverage)
        wy = spec.water_year or max(series)
        return {"analysis": at, "watershed": name, "huc8": code, "water_year": wy,
                "statistic": spec.statistic, "units": "in", "value": series[wy],
                "provenance": {"source": os.path.basename(files[wy]),
                               "method": f"zonal {spec.statistic} within HUC-8 {code}"}}

    if at == "watershed_yoy":
        code, name, geom, series = basin_series(spec.watershed, spec.statistic, files)
        if spec.years == "range" and spec.year_range:
            a, b = spec.year_range; series = {y: v for y, v in series.items() if a <= y <= b}
        yrs = sorted(series)
        yoy = {yrs[i]: (None if series[yrs[i-1]] in (None, 0) else round(series[yrs[i]] - series[yrs[i-1]], 2))
               for i in range(1, len(yrs))}
        return {"analysis": at, "watershed": name, "huc8": code, "statistic": spec.statistic, "units": "in",
                "series_by_year": {y: round(v, 2) for y, v in series.items()}, "yoy_change": yoy,
                "provenance": {"sources": [os.path.basename(files[y]) for y in yrs],
                               "method": f"basin {spec.statistic} per WY, differenced YoY; HUC-8 {code}"}}

    if at == "watershed_anomaly":
        code, name, geom, series = basin_series(spec.watershed, spec.statistic, files)
        wy = spec.water_year or max(series)
        a = anomaly_for_year(series, wy, spec.anomaly_type, spec.baseline())
        return {"analysis": at, "watershed": name, "huc8": code, "statistic": spec.statistic,
                "units": "in", **a, "series_by_year": {y: round(v, 2) for y, v in series.items()},
                "provenance": {"sources": [os.path.basename(f) for f in files.values()],
                               "method": f"{spec.anomaly_type} anomaly of basin {spec.statistic} vs baseline {a['climatology']['baseline']}"}}

    if at == "watershed_climatology":
        code, name, geom, series = basin_series(spec.watershed, spec.statistic, files)
        clim = climatology(series, spec.baseline())
        return {"analysis": at, "watershed": name, "huc8": code, "statistic": spec.statistic,
                "units": "in", "climatology": clim, "series_by_year": {y: round(v, 2) for y, v in series.items()},
                "provenance": {"sources": [os.path.basename(f) for f in files.values()],
                               "method": f"climatology of basin {spec.statistic}, baseline {clim['baseline']}"}}

    if at == "watershed_trend":
        code, name, geom, series = basin_series(spec.watershed, spec.statistic, files)
        if spec.years == "range" and spec.year_range:
            a, b = spec.year_range; series = {y: v for y, v in series.items() if a <= y <= b}
        tr = trend(series)
        return {"analysis": at, "watershed": name, "huc8": code, "statistic": spec.statistic,
                "units": "in", "trend": tr, "series_by_year": {y: round(v, 2) for y, v in series.items()},
                "provenance": {"sources": [os.path.basename(f) for f in files.values()],
                               "method": f"OLS + Mann-Kendall trend of basin {spec.statistic}"}}

    if at == "watershed_distribution":
        code, name, geom, series = basin_series(spec.watershed, spec.statistic, files)
        if spec.years == "range" and spec.year_range:
            a, b = spec.year_range; series = {y: v for y, v in series.items() if a <= y <= b}
        clim = climatology(series)
        return {"analysis": at, "watershed": name, "huc8": code, "statistic": spec.statistic,
                "units": "in", "summary_stats": clim, "series_by_year": {y: round(v, 2) for y, v in series.items()},
                "provenance": {"sources": [os.path.basename(f) for f in files.values()],
                               "method": f"interannual distribution of basin {spec.statistic}"}}

    if at == "multi_watershed":
        wy = spec.water_year or max(files)
        rows = []
        for w in spec.watersheds:
            code, name, geom, series = basin_series(w, spec.statistic, files,
                                                    threshold=spec.threshold)
            rows.append({"watershed": name, "huc8": code, "value": series.get(wy)})
        rows.sort(key=lambda r: (r["value"] is None, -(r["value"] or 0)))
        return {"analysis": at, "water_year": wy, "statistic": spec.statistic, "units": "in",
                "watersheds": rows,
                "provenance": {"source": os.path.basename(files[wy]),
                               "method": f"basin {spec.statistic} across {len(rows)} HUC-8 watersheds, WY{wy}"}}

    # map / map_anomaly handled in service_snowfall.py (need rendering); return directive
    return {"analysis": at, "directive": "render", "region": spec.region}


if __name__ == "__main__":
    ws = build_snowfall_worksheet()
    print("analysis types:", len(ws["fields"][0]["options"]))
    print("total worksheet fields:", len(ws["fields"]))
    print("groups:", ws["groups"])
