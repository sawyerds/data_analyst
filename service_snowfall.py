#!/usr/bin/env python3
"""
service_snowfall.py — render + package outputs for snowfall analyses.

The thin dispatcher `service.run_analysis()` parses an answers dict into a
discriminated SnowfallSpec, enters style_context(), and forwards here.

Every figure branch builds its renderer kwargs in a `kw` dict so the same dict
can feed (a) the live render and (b) the optional reproducible-script export
(spec.export_script) without drift between the two.
"""
import matplotlib
matplotlib.use("Agg")
import io, os, csv, base64

import numpy as np
from netCDF4 import Dataset
import matplotlib.pyplot as plt

from analyses_snowfall import (
    SnowfallSpec, execute_snowfall, find_sfav2_files,
    cell_climatology, _read_snow_inches,
)
from service_common import (_nice_step, _resolve_levels, resolve_rc,
                            figure_script_bundle)
from render import field_map_figure
from plots import (gridded_field_map, load_boundaries, watershed_series_plot,
                   ranked_bar_plot, box_distribution_plot, trend_plot)
from style import annotate_provenance
from watersheds import load_polygon, load_huc8_in_extent
from profiles import SNOW_BINS_IN

M_TO_IN = 39.3701
MIME = {"png": "image/png", "pdf": "application/pdf", "svg": "image/svg+xml"}


# ---------------- output packaging ----------------
def _figure_outputs(fig, spec, default_name):
    disp = io.BytesIO()
    fig.savefig(disp, format="png", dpi=110, bbox_inches="tight")
    image_b64 = base64.b64encode(disp.getvalue()).decode()

    fmt = spec.output_format
    art = io.BytesIO()
    fig.savefig(art, format=fmt, dpi=(spec.dpi if fmt == "png" else 200), bbox_inches="tight")
    raw = art.getvalue()
    plt.close(fig)

    fname = spec.filename or default_name
    if not fname.lower().endswith("." + fmt):
        fname += "." + fmt
    saved_path = None
    if spec.save_dir:
        os.makedirs(spec.save_dir, exist_ok=True)
        saved_path = os.path.join(spec.save_dir, fname)
        with open(saved_path, "wb") as fh:
            fh.write(raw)
    download = {"b64": base64.b64encode(raw).decode(), "filename": fname, "mime": MIME[fmt]}
    return image_b64, download, saved_path


def _csv_from_result(result) -> str:
    s = io.StringIO(); w = csv.writer(s)
    stat = result.get("statistic", "value")
    if "series_by_year" in result:
        w.writerow(["water_year", f"{stat}_in"])
        for y, v in result["series_by_year"].items():
            w.writerow([y, v])
    elif "watersheds" in result:
        w.writerow(["watershed", "huc8", f"{stat}_in"])
        for r in result["watersheds"]:
            w.writerow([r["watershed"], r["huc8"], r["value"]])
    elif "value" in result:
        w.writerow([f"{stat}_in"]); w.writerow([result["value"]])
    return s.getvalue()


def _footer(spec, base):
    txt = base
    if spec.caption:
        txt += f" | {spec.caption}"
    return txt


# ---------------- script-export packaging ----------------
def _script_dpi(spec):
    return spec.dpi if spec.output_format == "png" else 200


def _prov_post_lines(spec, footer_txt):
    """Post-render lines that mirror the service's provenance annotation."""
    if not (spec.provenance_footer and footer_txt):
        return ()
    return (f"annotate_provenance(fig, {footer_txt!r})",)


def _attach_script(result, spec, bundle):
    """Attach a figure_script_bundle to the response (and save_dir if set)."""
    if not bundle:
        return
    extras = result.setdefault("extra_downloads", [])
    extras.append({
        "b64": base64.b64encode(bundle["script"].encode()).decode(),
        "filename": bundle["script_name"],
        "mime": "text/x-python",
    })
    if bundle["npz"] is not None:
        extras.append({
            "b64": base64.b64encode(bundle["npz"]).decode(),
            "filename": bundle["npz_name"],
            "mime": "application/octet-stream",
        })
    if spec.save_dir:
        os.makedirs(spec.save_dir, exist_ok=True)
        sp = os.path.join(spec.save_dir, bundle["script_name"])
        with open(sp, "w") as fh:
            fh.write(bundle["script"])
        result["saved_script_path"] = sp
        if bundle["npz"] is not None:
            with open(os.path.join(spec.save_dir, bundle["npz_name"]), "wb") as fh:
                fh.write(bundle["npz"])


_PROV_IMPORT = ("from style import annotate_provenance",)


# ---------------- per-type figures ----------------
def _basin_map(name, code, geom, lat, lon, snow_in, stat_val, statistic, wy, spec):
    """Returns (fig, script_parts) — script_parts feeds figure_script_bundle."""
    minx, miny, maxx, maxy = geom.bounds; m = 0.35
    title = spec.title or f"WY{wy} Snowfall \u2014 {name} (HUC-8 {code})"
    kw = dict(units="in", cbar_label="Snowfall accumulation (in)", title=title,
              levels=list(SNOW_BINS_IN),
              extent=(minx - m, maxx + m, miny - m, maxy + m),
              central_lon=(minx + maxx) / 2, standard_parallels=(miny, maxy),
              huc_level=8)
    fig, ax = gridded_field_map(
        data=snow_in, lats=lat, lons=lon,
        boundaries=load_boundaries("us_states.geojson") if spec.show_states else None,
        huc_boundaries=[geom], **kw)
    stat_line = (f'ax.text(0.025, 0.97, "Basin {statistic}: {stat_val:.0f} in", '
                 f'transform=ax.transAxes, va="top", ha="left", fontsize=9, '
                 f'fontweight="bold", bbox=dict(boxstyle="round,pad=0.4", '
                 f'facecolor="white", edgecolor="#0b3d4f", alpha=0.9))')
    ax.text(0.025, 0.97, f"Basin {statistic}: {stat_val:.0f} in", transform=ax.transAxes,
            va="top", ha="left", fontsize=9, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#0b3d4f", alpha=0.9))
    rec = {"huc_boundaries": f"[load_polygon({name!r})[2]]"}
    imports = ["from watersheds import load_polygon"]
    if spec.show_states:
        rec["boundaries"] = 'load_boundaries("us_states.geojson")'
        imports.append("from plots import load_boundaries")
    script_parts = dict(
        func_module="plots", func_name="gridded_field_map",
        array_kwargs={"data": snow_in, "lats": lat, "lons": lon},
        literal_kwargs=kw, reconstruct_kwargs=rec,
        extra_imports=tuple(imports),
        post_lines=("ax = result[1]", stat_line),
    )
    return fig, script_parts


def _anomaly_map(spec, files, wy):
    """Returns (fig, script_parts)."""
    lat, lon, clim_mean, clim_std, n = cell_climatology(files, spec.baseline())
    _, _, snow = _read_snow_inches(files[wy])
    arr = np.ma.filled(snow, np.nan)
    if spec.anomaly_type == "standardized":
        anom = np.where(clim_std > 0, (arr - clim_mean) / clim_std, np.nan)
        unit, lbl = "\u03c3", "Standardized anomaly (\u03c3)"
    else:  # absolute (percent maps fall back to absolute; use watershed_anomaly for %)
        anom = arr - clim_mean
        unit, lbl = "in", "Snowfall anomaly (in)"
    # Default: symmetric levels at the 98th percentile of |anom| (13 boundaries -> 12 bins).
    default_M = float(np.nanpercentile(np.abs(anom), 98))
    default_levels = list(np.linspace(-default_M, default_M, 13))
    levels = _resolve_levels(spec, data=anom, default_levels=default_levels)
    from regions import get_region
    reg = get_region(spec.region)
    title = spec.title or f"WY{wy} Snowfall Anomaly vs {n}-yr Climatology \u2014 {reg['label']}"
    kw = dict(units=unit, cbar_label=lbl, title=title,
              levels=[float(v) for v in levels], cmap=spec.colormap or "RdBu_r",
              extent=tuple(reg["extent"]), central_lon=reg["central_lon"],
              standard_parallels=tuple(reg["parallels"]),
              huc_level=8 if spec.huc8_overlay else None)
    fig, ax = gridded_field_map(
        data=anom, lats=lat, lons=lon,
        boundaries=load_boundaries("us_states.geojson") if spec.show_states else None,
        huc_boundaries=load_huc8_in_extent(reg["extent"]) if spec.huc8_overlay else None,
        **kw)
    rec = {}
    imports = []
    if spec.show_states:
        rec["boundaries"] = 'load_boundaries("us_states.geojson")'
        imports.append("from plots import load_boundaries")
    if spec.huc8_overlay:
        rec["huc_boundaries"] = f"load_huc8_in_extent({tuple(reg['extent'])!r})"
        imports.append("from watersheds import load_huc8_in_extent")
    script_parts = dict(
        func_module="plots", func_name="gridded_field_map",
        array_kwargs={"data": anom, "lats": lat, "lons": lon},
        literal_kwargs=kw, reconstruct_kwargs=rec, extra_imports=tuple(imports),
        post_lines=(),
    )
    return fig, script_parts


def _series_script_parts(func_name, kw):
    """Script parts for the all-literal series/bar/box/trend renderers."""
    return dict(func_module="plots", func_name=func_name,
                array_kwargs=None, literal_kwargs=kw,
                reconstruct_kwargs=None, extra_imports=(), post_lines=())


def _maybe_export_script(result, spec, script_parts, out_stem, footer_txt=None):
    if not (spec.export_script and script_parts):
        return
    post = tuple(script_parts.get("post_lines", ()))
    imports = tuple(script_parts.get("extra_imports", ()))
    if spec.provenance_footer and footer_txt:
        post = post + _prov_post_lines(spec, footer_txt)
        imports = imports + _PROV_IMPORT
    bundle = figure_script_bundle(
        analysis=result.get("analysis", "analysis"),
        func_module=script_parts["func_module"],
        func_name=script_parts["func_name"],
        literal_kwargs=script_parts.get("literal_kwargs"),
        array_kwargs=script_parts.get("array_kwargs"),
        reconstruct_kwargs=script_parts.get("reconstruct_kwargs"),
        extra_imports=imports,
        post_lines=post,
        rc=resolve_rc(spec),
        out_stem=out_stem,
        output_format=spec.output_format,
        dpi=_script_dpi(spec),
    )
    _attach_script(result, spec, bundle)


# ---------------- entry point ----------------
def run_snowfall(spec: SnowfallSpec, data_dir: str = ".") -> dict:
    """Render the analysis described by `spec`. spec is already validated upstream."""
    files = find_sfav2_files(data_dir)
    at = spec.analysis_type

    # ----- map types (rendered here) -----
    if at == "map":
        if not files:
            raise FileNotFoundError(f"no sfav2 files in {data_dir}")
        wy = spec.water_year or max(files)
        levels_override = _resolve_levels(spec, default_levels=None)
        fm_kw = dict(path=files[wy], region=spec.region,
                     huc_level=8 if spec.huc8_overlay else None,
                     cmap=spec.colormap, levels=levels_override)
        fig, ax, meta = field_map_figure(**fm_kw)
        footer = _footer(spec, f"region {spec.region} | WY{wy}")
        if spec.provenance_footer:
            annotate_provenance(fig, footer)
        out_stem = f"map_{spec.region}_wy{wy}"
        result = {"analysis": at, "region": spec.region, "water_year": wy, "meta": meta,
                  "summary": f"{meta['field']} over {spec.region}, WY{wy}"
                             + (f" \u2014 {meta['n_huc8']} HUC-8 basins" if meta["n_huc8"] else "")}
        _maybe_export_script(result, spec,
                             dict(func_module="render", func_name="field_map_figure",
                                  literal_kwargs=fm_kw, array_kwargs=None,
                                  reconstruct_kwargs=None, extra_imports=(), post_lines=()),
                             out_stem, footer_txt=footer)
        img, dl, saved = _figure_outputs(fig, spec, out_stem)
        result.update({"image_b64": img, "download": dl, "saved_path": saved})
        return result

    if at == "map_anomaly":
        if not files:
            raise FileNotFoundError("no sfav2 files")
        wy = spec.water_year or max(files)
        fig, script_parts = _anomaly_map(spec, files, wy)
        footer = _footer(spec, f"{spec.anomaly_type} anomaly | region {spec.region} | WY{wy}")
        if spec.provenance_footer:
            annotate_provenance(fig, footer)
        out_stem = f"anomaly_{spec.region}_wy{wy}"
        result = {"analysis": at, "region": spec.region, "water_year": wy,
                  "anomaly_type": spec.anomaly_type,
                  "summary": f"WY{wy} {spec.anomaly_type} snowfall anomaly over {spec.region}"}
        _maybe_export_script(result, spec, script_parts, out_stem, footer_txt=footer)
        img, dl, saved = _figure_outputs(fig, spec, out_stem)
        result.update({"image_b64": img, "download": dl, "saved_path": saved})
        return result

    # ----- watershed / multi types: compute then render -----
    result = execute_snowfall(spec, data_dir)
    footer = None

    if at == "watershed_mean":
        wy = result["water_year"]
        code, name, geom = load_polygon(spec.watershed)
        lat, lon, snow = _read_snow_inches(files[wy])
        fig, script_parts = _basin_map(name, code, geom, lat, lon, snow,
                                       result["value"], spec.statistic, wy, spec)
        footer = _footer(spec, f"basin {spec.statistic} | HUC-8 {code} | {result['provenance']['source']}")
        out_stem = f"{code}_{spec.statistic}_wy{wy}"
        result["summary"] = f"{name}: WY{wy} basin {spec.statistic} = {result['value']:.0f} in"

    elif at == "watershed_yoy":
        series = {int(y): v for y, v in result["series_by_year"].items() if v is not None}
        yoy = {int(y): v for y, v in result["yoy_change"].items()}
        years = sorted(series)
        kw = dict(years=years, values=series, yoy=yoy if len(years) > 1 else None,
                  units="in", watershed_name=result["watershed"],
                  huc8=result["huc8"], statistic=spec.statistic, title=spec.title)
        fig, _ = watershed_series_plot(**kw)
        script_parts = _series_script_parts("watershed_series_plot", kw)
        footer = _footer(spec, f"HUC-8 {result['huc8']} | {len(years)} WY")
        out_stem = f"{result['huc8']}_{spec.statistic}_yoy"
        if years:
            w = max(series, key=series.get); d = min(series, key=series.get)
            result["summary"] = (f"{result['watershed']} WY{years[0]}\u2013{years[-1]}: "
                                 f"wettest {w} ({series[w]:.0f} in), driest {d} ({series[d]:.0f} in)")

    elif at == "watershed_anomaly":
        series = {int(y): v for y, v in result["series_by_year"].items() if v is not None}
        years = sorted(series)
        title = spec.title or (f"{result['watershed']} \u2014 WY{result['year']} "
                               f"{spec.anomaly_type} anomaly of basin {spec.statistic}")
        kw = dict(years=years, values=series, units="in",
                  watershed_name=result["watershed"], huc8=result["huc8"],
                  statistic=spec.statistic, title=title,
                  highlight_years=[result["year"]])
        fig, _ = watershed_series_plot(**kw)
        script_parts = _series_script_parts("watershed_series_plot", kw)
        base = (f"WY{result['year']} = {result['value']:.0f} in | "
                f"{spec.anomaly_type} anomaly {result['anomaly']:.1f} | "
                f"pctile {result['percentile_rank']:.0f} | normal {result['climatology']['normal_mean']:.0f}")
        footer = _footer(spec, base)
        out_stem = f"{result['huc8']}_anomaly_wy{result['year']}"
        a = result["anomaly"]
        unit = {"absolute": "in", "percent": "% of normal", "standardized": "\u03c3"}[spec.anomaly_type]
        result["summary"] = (f"{result['watershed']} WY{result['year']}: "
                             f"{result['value']:.0f} in = {a:.0f} {unit} "
                             f"({result['percentile_rank']:.0f}th percentile)")

    elif at == "watershed_climatology":
        series = {int(y): v for y, v in result["series_by_year"].items() if v is not None}
        years = sorted(series); clim = result["climatology"]
        title = spec.title or (f"{result['watershed']} \u2014 Basin-{spec.statistic} Snowfall Climatology "
                               f"(WY{clim['baseline'][0]}\u2013{clim['baseline'][1]})")
        kw = dict(years=years, values=series, units="in",
                  watershed_name=result["watershed"], huc8=result["huc8"],
                  statistic=spec.statistic, title=title)
        fig, _ = watershed_series_plot(**kw)
        script_parts = _series_script_parts("watershed_series_plot", kw)
        footer = _footer(spec, f"normal {clim['normal_mean']:.0f} in | median {clim['normal_median']:.0f} | std {clim['std']:.0f}")
        out_stem = f"{result['huc8']}_climatology"
        result["summary"] = (f"{result['watershed']}: normal {clim['normal_mean']:.0f} in "
                             f"(median {clim['normal_median']:.0f}, \u03c3 {clim['std']:.0f}); "
                             f"wettest {clim['wettest_year']}, driest {clim['driest_year']}")

    elif at == "watershed_trend":
        series = {int(y): v for y, v in result["series_by_year"].items() if v is not None}
        years = sorted(series); vals = [series[y] for y in years]
        title = spec.title or f"{result['watershed']} \u2014 Basin-{spec.statistic} Snowfall Trend"
        kw = dict(years=years, values=vals, trend=result["trend"], units="in",
                  title=title, statistic=spec.statistic)
        fig, _ = trend_plot(**kw)
        script_parts = _series_script_parts("trend_plot", kw)
        footer = _footer(spec, f"HUC-8 {result['huc8']} | OLS + Mann\u2013Kendall")
        out_stem = f"{result['huc8']}_trend"
        t = result["trend"]
        result["summary"] = (f"{result['watershed']}: {t['slope_per_decade']:+.1f} in/decade "
                             f"({t['direction']}), p={t['p_value_ols']:.2f} (OLS), "
                             f"{t['p_value_mannkendall']:.2f} (MK)")

    elif at == "watershed_distribution":
        series = {int(y): v for y, v in result["series_by_year"].items() if v is not None}
        vals = [series[y] for y in sorted(series)]
        title = spec.title or f"{result['watershed']} \u2014 Basin-{spec.statistic} Snowfall Distribution"
        kw = dict(values=vals, units="in", title=title,
                  name=result["watershed"], statistic=spec.statistic)
        fig, _ = box_distribution_plot(**kw)
        script_parts = _series_script_parts("box_distribution_plot", kw)
        c = result["summary_stats"]
        footer = _footer(spec, f"n={c['n_years']} WY | median {c['normal_median']:.0f} | mean {c['normal_mean']:.0f}")
        out_stem = f"{result['huc8']}_distribution"
        result["summary"] = (f"{result['watershed']}: median {c['normal_median']:.0f} in, "
                             f"range {c['min']:.0f}\u2013{c['max']:.0f} over {c['n_years']} WY")

    elif at == "multi_watershed":
        rows = result["watersheds"]
        title = spec.title or f"Basin-{spec.statistic} Snowfall by Watershed \u2014 WY{result['water_year']}"
        kw = dict(labels=[r["watershed"] for r in rows],
                  values=[r["value"] for r in rows], units="in",
                  title=title, statistic=spec.statistic)
        fig, _ = ranked_bar_plot(**kw)
        script_parts = _series_script_parts("ranked_bar_plot", kw)
        footer = _footer(spec, f"{len(rows)} HUC-8 watersheds | WY{result['water_year']}")
        out_stem = f"multi_{spec.statistic}_wy{result['water_year']}"
        top = rows[0]
        result["summary"] = (f"WY{result['water_year']} {spec.statistic}: highest {top['watershed']} "
                             f"({top['value']:.0f} in) of {len(rows)} basins")
    else:
        raise ValueError(f"no renderer for {at}")

    if spec.provenance_footer and footer:
        annotate_provenance(fig, footer)
    _maybe_export_script(result, spec, script_parts, out_stem, footer_txt=footer)
    img, dl, saved = _figure_outputs(fig, spec, out_stem)

    result["image_b64"] = img
    result["download"] = dl
    result["saved_path"] = saved
    if spec.export_data:
        result["data_csv"] = _csv_from_result(result)
    return result
