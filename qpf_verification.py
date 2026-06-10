"""
wxagent/analyses/qpf_verification.py

WxAgent analysis type: "wrf_qpf_verification"

Plugs into the existing WxAgent scaffold (analyses.py / plots.py / service.py)
following the same plan→execute→synthesize pattern as the snowfall pipeline.

Integration steps (in your existing repo on mlm-stormy):
  1. Drop this file into:   wxagent/analyses/qpf_verification.py
  2. In wxagent/analyses.py:
        from .analyses.qpf_verification import (
            SCHEMA as QPF_VERIFICATION_SCHEMA,
            run as run_qpf_verification,
        )
        ANALYSIS_TYPES["wrf_qpf_verification"] = {
            "label": "WRF QPF vs Stage IV (gridded + basin)",
            "schema": QPF_VERIFICATION_SCHEMA,
            "runner": run_qpf_verification,
        }
  3. In wxagent/service.py — the existing run_analysis dispatcher already
     iterates ANALYSIS_TYPES, so no change needed unless you want a
     dedicated route prefix.
  4. The UI picks the new type up automatically from the schema endpoint.

Real data dependencies:
  - geopandas (for WBD shapefile reads)
  - cfgrib + eccodes (for Stage IV)
  - scipy, numpy, netCDF4, matplotlib (already in wxagent env)
"""

from __future__ import annotations

import io
import json
import warnings
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from netCDF4 import Dataset
from scipy.interpolate import griddata
import cfgrib

warnings.filterwarnings('ignore')


# =============================================================================
# SCHEMA — published to the UI via /api/schema, drives the form
# Follows the conditional-group pattern used by the snowfall pipeline.
# =============================================================================
SCHEMA: dict[str, Any] = {
    "type": "object",
    "title": "WRF QPF Verification against Stage IV",
    "description": (
        "Verify a WRF forecast (RAINNC + RAINC accumulated total) against "
        "the corresponding Stage IV 6-h QPE files for the forecast window. "
        "Produces global d03 maps + per-basin diagnostic panels."
    ),
    "groups": [
        {
            "title": "Inputs",
            "fields": [
                {"name": "wrfout_path", "type": "string", "required": True,
                 "label": "Path to extracted wrfout NetCDF",
                 "placeholder": "/Users/sawyersmith/Desktop/wrfout_d03_ISHMAEL_SUBSET.nc"},
                {"name": "stageiv_dir", "type": "string", "required": True,
                 "label": "Directory containing Stage IV 6-h grib2 files",
                 "default": "/Users/sawyersmith/Desktop/datasets/stage_iv/"},
                {"name": "auto_window_from_wrfout", "type": "boolean", "default": True,
                 "label": "Auto-discover Stage IV window from wrfout START_DATE → valid_time"},
            ],
        },
        {
            "title": "Manual time window (only if auto-discover is off)",
            "show_when": {"auto_window_from_wrfout": False},
            "fields": [
                {"name": "window_start", "type": "string",
                 "label": "Window start (UTC ISO)", "placeholder": "2023-01-04T00:00:00"},
                {"name": "window_end", "type": "string",
                 "label": "Window end (UTC ISO)", "placeholder": "2023-01-06T00:00:00"},
            ],
        },
        {
            "title": "Basin selection",
            "fields": [
                {"name": "huc8_codes", "type": "array", "items": {"type": "string"},
                 "label": "HUC-8 codes (comma-separated)",
                 "default": ["18010110", "18020125"],
                 "help": "Russian River = 18010110, Upper Yuba = 18020125"},
                {"name": "huc8_dir", "type": "string",
                 "label": "WBD shapefile directory",
                 "default": "/Users/sawyersmith/Desktop/datasets/huc8/"},
                {"name": "huc8_combined_geojson", "type": "string",
                 "label": "Pre-built combined HUC-8 GeoJSON (optional cache)",
                 "default": "/Users/sawyersmith/Desktop/datasets/huc8/ca_huc8_combined.geojson"},
            ],
        },
        {
            "title": "Verification options",
            "fields": [
                {"name": "regrid_method", "type": "enum", "default": "linear",
                 "options": ["linear", "nearest"],
                 "label": "Stage IV → WRF regrid method"},
                {"name": "thresholds_mm", "type": "array", "items": {"type": "number"},
                 "default": [1, 5, 10, 25, 50, 75, 100],
                 "label": "Categorical thresholds (mm)"},
                {"name": "elevation_bins_m", "type": "array", "items": {"type": "number"},
                 "default": [0, 100, 250, 500, 750, 1000, 1250, 1500, 2000, 2500, 3000, 4000],
                 "label": "Elevation bin edges (m) for hypsometric bias"},
            ],
        },
        {
            "title": "Output",
            "fields": [
                {"name": "format", "type": "enum", "default": "png",
                 "options": ["png", "pdf", "svg"], "label": "Figure format"},
                {"name": "dpi", "type": "number", "default": 130, "min": 50, "max": 600,
                 "label": "DPI"},
                {"name": "save_dir", "type": "string",
                 "label": "Server save directory (optional)",
                 "default": "/Users/sawyersmith/Desktop/wxagent_outputs/qpf_verification/"},
                {"name": "filename_stem", "type": "string",
                 "label": "Output filename stem",
                 "default": "qpf_verification"},
                {"name": "export_csv", "type": "boolean", "default": True,
                 "label": "Also export per-basin stats CSV"},
            ],
        },
    ],
}


# =============================================================================
# Core compute primitives — independent of the WxAgent service layer so they
# can also be invoked from notebooks, tests, or Celery tasks.
# =============================================================================
@dataclass
class WRFFields:
    total: np.ndarray
    rainnc: np.ndarray
    rainc: np.ndarray
    lat: np.ndarray
    lon: np.ndarray
    hgt: np.ndarray
    valid_time: str
    start_date: str
    mp_physics: int
    bl_pbl_physics: int
    dx_m: float


def read_wrfout(path: str | Path) -> WRFFields:
    ds = Dataset(str(path))
    def unmask(a):
        return np.asarray(a.filled(np.nan)) if hasattr(a, 'filled') else np.asarray(a, dtype=float)
    rainnc = unmask(ds.variables['RAINNC'][0, :, :])
    rainc = unmask(ds.variables['RAINC'][0, :, :])
    out = WRFFields(
        total=rainnc + rainc,
        rainnc=rainnc, rainc=rainc,
        lat=unmask(ds.variables['XLAT'][0, :, :]),
        lon=unmask(ds.variables['XLONG'][0, :, :]),
        hgt=unmask(ds.variables['HGT'][0, :, :]),
        valid_time=b''.join(ds.variables['Times'][0]).decode('utf-8'),
        start_date=str(getattr(ds, 'START_DATE', '')),
        mp_physics=int(getattr(ds, 'MP_PHYSICS', -1)),
        bl_pbl_physics=int(getattr(ds, 'BL_PBL_PHYSICS', -1)),
        dx_m=float(getattr(ds, 'DX', float('nan'))),
    )
    ds.close()
    return out


def discover_stageiv_window(wrf: WRFFields, stageiv_dir: str | Path) -> list[Path]:
    """List the Stage IV 6-h grib2 files whose valid time falls inside
    (start_date, valid_time]. Filenames follow st4_conus_YYYYMMDDHH_06h.grb2."""
    from datetime import datetime, timedelta
    fmt = "%Y-%m-%d_%H:%M:%S"
    t_start = datetime.strptime(wrf.start_date, fmt)
    t_end = datetime.strptime(wrf.valid_time, fmt)
    out = []
    t = t_start + timedelta(hours=6)
    while t <= t_end:
        fname = f"st4_conus_{t:%Y%m%d%H}_06h.grb2"
        p = Path(stageiv_dir) / fname
        if p.exists():
            out.append(p)
        t += timedelta(hours=6)
    return out


def sum_stageiv(paths: list[Path], indexpath_dir: str = '/tmp/cfgrib_idx'):
    Path(indexpath_dir).mkdir(parents=True, exist_ok=True)
    total = None; src_lat = src_lon = None; count = None; valid_times = []
    for p in sorted(paths):
        ds = cfgrib.open_datasets(str(p), backend_kwargs={
            'indexpath': str(Path(indexpath_dir) / (Path(p).name + '.idx'))
        })[0]
        tp_mm = ds['tp'].values.astype(float)  # Stage IV tp is kg/m**2 = mm
        tp_mm = np.where(tp_mm > 1e10, np.nan, tp_mm)
        valid_times.append(str(ds['valid_time'].values))
        if total is None:
            total = np.where(np.isfinite(tp_mm), tp_mm, 0.0)
            count = np.isfinite(tp_mm).astype(int)
            src_lat = ds['latitude'].values
            src_lon = ds['longitude'].values
            src_lon = np.where(src_lon > 180, src_lon - 360, src_lon)
        else:
            total = total + np.where(np.isfinite(tp_mm), tp_mm, 0.0)
            count = count + np.isfinite(tp_mm).astype(int)
    return np.where(count > 0, total, np.nan), src_lat, src_lon, valid_times


def regrid_to_wrf(src_lat, src_lon, src_data, wrf_lat, wrf_lon, method='linear'):
    pts = np.column_stack([src_lat.ravel(), src_lon.ravel()])
    vals = src_data.ravel()
    fin = np.isfinite(vals)
    return griddata(pts[fin], vals[fin], (wrf_lat, wrf_lon), method=method, fill_value=np.nan)


# =============================================================================
# Basin geometry: load HUC-8 polygons from real WBD shapefiles or, for testing,
# from in-memory demo polygons.
# =============================================================================
DEMO_BASIN_POLYGONS = {
    "18010110": {
        "name": "Russian River",
        "vertices": [
            (-123.55, 38.40), (-123.55, 38.85), (-123.30, 39.10), (-123.00, 39.40),
            (-122.85, 39.35), (-122.70, 39.10), (-122.65, 38.75), (-122.75, 38.45),
            (-122.95, 38.30), (-123.20, 38.30), (-123.55, 38.40),
        ],
    },
    "18020125": {
        "name": "Upper Yuba",
        "vertices": [
            (-121.20, 39.40), (-121.15, 39.55), (-121.00, 39.65), (-120.60, 39.65),
            (-120.30, 39.55), (-120.30, 39.30), (-120.50, 39.15), (-120.85, 39.10),
            (-121.10, 39.15), (-121.20, 39.30), (-121.20, 39.40),
        ],
    },
}


def load_basin_polygons(huc8_codes: list[str], huc8_dir: str | None,
                        combined_geojson: str | None = None) -> dict:
    """Return {huc8: {'name': str, 'vertices': [(lon, lat), ...]}}.

    Loads from real WBD shapefiles via geopandas if available; falls back to
    demo polygons for codes that aren't found. The demo path lets unit tests
    and dev work happen without the full WBD installed.
    """
    out = {}
    try:
        import geopandas as gpd
        gdf = None
        if combined_geojson and Path(combined_geojson).exists():
            gdf = gpd.read_file(combined_geojson)
        elif huc8_dir and Path(huc8_dir).exists():
            # WBD HUC-8 layer is typically named WBDHU8.shp inside the HU2 folders
            shps = list(Path(huc8_dir).rglob('WBDHU8.shp'))
            if shps:
                gdf = gpd.read_file(shps[0])
                for s in shps[1:]:
                    gdf = gpd.GeoDataFrame.from_features(
                        list(gdf.iterfeatures()) + list(gpd.read_file(s).iterfeatures())
                    )
        if gdf is not None and 'huc8' in [c.lower() for c in gdf.columns]:
            huc_col = [c for c in gdf.columns if c.lower() == 'huc8'][0]
            name_col = next((c for c in gdf.columns if c.lower() == 'name'), None)
            for code in huc8_codes:
                row = gdf[gdf[huc_col] == code]
                if len(row) > 0:
                    geom = row.iloc[0].geometry
                    name = str(row.iloc[0][name_col]) if name_col else f"HUC8 {code}"
                    # Reduce MultiPolygon to outermost ring of largest part
                    parts = list(geom.geoms) if geom.geom_type == 'MultiPolygon' else [geom]
                    largest = max(parts, key=lambda g: g.area)
                    coords = list(largest.exterior.coords)
                    out[code] = {'name': name, 'vertices': coords}
    except ImportError:
        pass

    # Fill in any missing codes from the demo set (with a warning marker)
    for code in huc8_codes:
        if code not in out and code in DEMO_BASIN_POLYGONS:
            d = DEMO_BASIN_POLYGONS[code]
            out[code] = {'name': d['name'] + ' (DEMO polygon)', 'vertices': d['vertices']}

    return out


def build_basin_mask(lon2d, lat2d, vertices) -> np.ndarray:
    poly = MplPath(np.asarray(vertices))
    pts = np.column_stack([lon2d.ravel(), lat2d.ravel()])
    return poly.contains_points(pts).reshape(lon2d.shape)


# =============================================================================
# Statistics
# =============================================================================
def verification_stats(obs, fcst, mask, thresholds_mm) -> dict:
    valid = mask & np.isfinite(obs) & np.isfinite(fcst)
    if valid.sum() < 5:
        return {'n_paired_points': int(valid.sum()), 'note': 'too few paired cells'}
    o = obs[valid]; f = fcst[valid]
    out = {
        'n_paired_points': int(valid.sum()),
        'obs_mean_mm': float(o.mean()),
        'fcst_mean_mm': float(f.mean()),
        'mean_bias_mm': float((f - o).mean()),
        'percent_bias': float((f.mean() - o.mean()) / o.mean() * 100) if o.mean() > 0 else None,
        'rmse_mm': float(np.sqrt(((f - o) ** 2).mean())),
        'mae_mm': float(np.abs(f - o).mean()),
        'pearson_r': float(np.corrcoef(o, f)[0, 1]),
        'obs_max_mm': float(o.max()),
        'fcst_max_mm': float(f.max()),
        'obs_percentiles_mm': {str(p): float(np.percentile(o, p)) for p in (50, 75, 90, 95, 99)},
        'fcst_percentiles_mm': {str(p): float(np.percentile(f, p)) for p in (50, 75, 90, 95, 99)},
    }
    cat = {}
    for thr in thresholds_mm:
        hits = int(((o >= thr) & (f >= thr)).sum())
        misses = int(((o >= thr) & (f < thr)).sum())
        falses = int(((o < thr) & (f >= thr)).sum())
        denom = hits + misses + falses
        cat[str(thr)] = {
            'hits': hits, 'misses': misses, 'false_alarms': falses,
            'csi': hits / denom if denom > 0 else None,
            'pod': hits / (hits + misses) if (hits + misses) > 0 else None,
            'far': falses / (hits + falses) if (hits + falses) > 0 else None,
            'frequency_bias': (hits + falses) / (hits + misses) if (hits + misses) > 0 else None,
        }
    out['categorical'] = cat
    return out


def hypsometric_bias(obs, fcst, hgt, mask, bin_edges):
    valid = mask & np.isfinite(obs) & np.isfinite(fcst)
    o = obs[valid]; f = fcst[valid]; h = hgt[valid]
    bias = f - o
    centers, means, p25s, p75s, ns = [], [], [], [], []
    for i in range(len(bin_edges) - 1):
        sel = (h >= bin_edges[i]) & (h < bin_edges[i+1])
        if sel.sum() >= 10:
            centers.append((bin_edges[i] + bin_edges[i+1]) / 2)
            means.append(float(np.mean(bias[sel])))
            p25s.append(float(np.percentile(bias[sel], 25)))
            p75s.append(float(np.percentile(bias[sel], 75)))
            ns.append(int(sel.sum()))
    return {'bin_centers_m': centers, 'mean_bias_mm': means,
            'p25_bias_mm': p25s, 'p75_bias_mm': p75s, 'n_cells': ns}


# =============================================================================
# Plot renderers — return raw bytes so the service layer can stream/save them.
# Two plots per run: overview, then one per basin.
# =============================================================================
def render_overview(wrf: WRFFields, st4_on_wrf, bias, basins: dict,
                    global_stats: dict, fmt: str = 'png', dpi: int = 130) -> bytes:
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.7), dpi=dpi)
    levels = [0, 1, 5, 10, 20, 40, 60, 80, 100, 125, 150, 175, 200, 250]
    bmax = max(abs(np.nanpercentile(bias, 2)), abs(np.nanpercentile(bias, 98)), 30)
    blevels = np.linspace(-bmax, bmax, 21)
    basin_colors = ['#0C447C', '#993556', '#3B6D11', '#854F0B', '#3C3489']

    for ax, fld, title, cmap, lv, ext in [
        (axes[0], wrf.total, 'WRF d03 (RAINNC + RAINC)', 'YlGnBu', levels, 'max'),
        (axes[1], st4_on_wrf, 'Stage IV → d03', 'YlGnBu', levels, 'max'),
        (axes[2], bias, 'Bias (WRF − Stage IV)', 'RdBu_r', blevels, 'both'),
    ]:
        cf = ax.contourf(wrf.lon, wrf.lat, fld, levels=lv, cmap=cmap, extend=ext)
        plt.colorbar(cf, ax=ax, label='mm')
        ax.set_title(title, fontsize=11)
        ax.set_xlabel('Longitude'); ax.set_aspect('equal')
        for i, (code, b) in enumerate(basins.items()):
            v = np.array(b['vertices'])
            ax.plot(v[:, 0], v[:, 1], color=basin_colors[i % len(basin_colors)], lw=2, zorder=10)
    axes[0].set_ylabel('Latitude')

    plt.suptitle(
        f"WRF QPF verification  |  init {wrf.start_date[:13]} → valid {wrf.valid_time[:13]}\n"
        f"Global: bias {global_stats['mean_bias_mm']:+.1f} mm  |  "
        f"RMSE {global_stats['rmse_mm']:.1f} mm  |  r {global_stats['pearson_r']:.3f}  |  "
        f"n = {global_stats['n_paired_points']:,}",
        fontsize=11, y=1.01,
    )
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format=fmt, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    return buf.getvalue()


def render_basin(wrf: WRFFields, st4_on_wrf, bias, code: str, basin: dict,
                 mask: np.ndarray, stats: dict, hyps: dict, color: str,
                 fmt: str = 'png', dpi: int = 130) -> bytes:
    fig = plt.figure(figsize=(14, 9), dpi=dpi)
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], width_ratios=[1.1, 1.1, 1])
    v = np.array(basin['vertices'])
    pad = 0.15
    xmin, xmax = v[:, 0].min() - pad, v[:, 0].max() + pad
    ymin, ymax = v[:, 1].min() - pad, v[:, 1].max() + pad
    levels = [0, 1, 5, 10, 20, 40, 60, 80, 100, 125, 150, 175, 200]
    blevels = np.linspace(-50, 50, 21)

    for col, (fld, title, cmap, lv, ext, lbl) in enumerate([
        (np.where(mask, wrf.total, np.nan),     'WRF QPF (basin)',   'YlGnBu', levels, 'max', 'mm'),
        (np.where(mask, st4_on_wrf, np.nan),    'Stage IV QPE',      'YlGnBu', levels, 'max', 'mm'),
        (np.where(mask, bias, np.nan),          'Bias (WRF − ST4)',  'RdBu_r', blevels, 'both', 'mm'),
    ]):
        ax = fig.add_subplot(gs[0, col])
        if col < 2:  # show out-of-basin faintly for context
            ax.contourf(wrf.lon, wrf.lat,
                        np.where(~mask, wrf.total if col == 0 else st4_on_wrf, np.nan),
                        levels=levels, cmap=cmap, alpha=0.20)
        cf = ax.contourf(wrf.lon, wrf.lat, fld, levels=lv, cmap=cmap, extend=ext)
        plt.colorbar(cf, ax=ax, label=lbl, shrink=0.85)
        ax.plot(v[:, 0], v[:, 1], color=color, lw=2.0)
        ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax); ax.set_aspect('equal')
        ax.set_title(title, fontsize=10.5)
        ax.set_xlabel('Lon');
        if col == 0:
            ax.set_ylabel('Lat')

    # Density scatter
    ax = fig.add_subplot(gs[1, 0])
    valid = mask & np.isfinite(st4_on_wrf) & np.isfinite(wrf.total)
    if valid.sum() > 0:
        o = st4_on_wrf[valid]; f = wrf.total[valid]
        lim = max(np.percentile(o, 99), np.percentile(f, 99)) * 1.1
        ax.hist2d(o, f, bins=40, range=[[0, lim], [0, lim]], cmap='magma_r',
                  norm=matplotlib.colors.LogNorm())
        ax.plot([0, lim], [0, lim], 'w--', lw=1)
        ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel('Stage IV (mm)'); ax.set_ylabel('WRF (mm)')
    ax.set_title('Paired density', fontsize=10.5); ax.set_aspect('equal')

    # Hypsometric
    ax = fig.add_subplot(gs[1, 1])
    if hyps['bin_centers_m']:
        bc = np.array(hyps['bin_centers_m'])
        mb = np.array(hyps['mean_bias_mm'])
        p25 = np.array(hyps['p25_bias_mm']); p75 = np.array(hyps['p75_bias_mm'])
        ax.fill_between(bc, p25, p75, alpha=0.25, color=color, label='IQR')
        ax.plot(bc, mb, 'o-', color=color, label='mean bias')
        ax.axhline(0, color='k', lw=0.8)
        ax.legend(fontsize=9)
    ax.set_xlabel('Terrain height (m)')
    ax.set_ylabel('Bias WRF − Stage IV (mm)')
    ax.set_title('Bias vs elevation', fontsize=10.5)
    ax.grid(alpha=0.3, linestyle=':')

    # Stats card
    ax = fig.add_subplot(gs[1, 2]); ax.axis('off')
    cat = stats.get('categorical', {})
    pct_bias = stats.get('percent_bias')
    lines = [
        f"n cells: {stats.get('n_paired_points', 0):,}",
        '',
        f"Basin-mean QPE: {stats.get('obs_mean_mm', 0):.1f} mm",
        f"Basin-mean QPF: {stats.get('fcst_mean_mm', 0):.1f} mm",
        f"Mean bias:      {stats.get('mean_bias_mm', 0):+.1f} mm",
        f"Percent bias:   {pct_bias:+.1f} %" if pct_bias is not None else "Percent bias:   n/a",
        f"RMSE:           {stats.get('rmse_mm', 0):.1f} mm",
        f"MAE:            {stats.get('mae_mm', 0):.1f} mm",
        f"Pearson r:      {stats.get('pearson_r', 0):.3f}",
        '',
        'CSI / POD / FAR / FB',
    ]
    for thr in ['10', '25', '50', '75']:
        c = cat.get(thr, {})
        if c and c.get('csi') is not None:
            lines.append(
                f"  ≥{thr:>3} mm: {c['csi']:.2f} / {c['pod']:.2f} / {c['far']:.2f} / {c['frequency_bias']:.2f}"
            )
    ax.text(0.02, 0.98, "\n".join(lines), transform=ax.transAxes,
            va='top', ha='left', fontfamily='monospace', fontsize=10.5)
    ax.set_title('Summary', fontsize=10.5, loc='left')

    plt.suptitle(
        f"{basin['name']} (HUC8 {code})  |  WRF QPF  |  "
        f"init {wrf.start_date[:13]} → valid {wrf.valid_time[:13]}",
        fontsize=11.5, y=1.00,
    )
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format=fmt, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    return buf.getvalue()


# =============================================================================
# Top-level runner — called by service.py via the analysis dispatcher.
# Returns the same shape as the snowfall pipeline: figure bytes, CSV bytes,
# JSON payload, list of (filename, mimetype, bytes) for the response packager.
# =============================================================================
@dataclass
class RunResult:
    figures: list[tuple[str, str, bytes]] = field(default_factory=list)
    csv_files: list[tuple[str, bytes]] = field(default_factory=list)
    stats: dict = field(default_factory=dict)


def run(plan: dict) -> RunResult:
    """plan = the validated dict produced from the schema. See run_demo() below
    for a working example invocation."""
    # ---- read WRF ----
    wrf = read_wrfout(plan['wrfout_path'])

    # ---- discover Stage IV files ----
    if plan.get('auto_window_from_wrfout', True):
        st4_paths = discover_stageiv_window(wrf, plan['stageiv_dir'])
    else:
        from datetime import datetime, timedelta
        t0 = datetime.fromisoformat(plan['window_start'])
        t1 = datetime.fromisoformat(plan['window_end'])
        st4_paths = []
        t = t0 + timedelta(hours=6)
        while t <= t1:
            p = Path(plan['stageiv_dir']) / f"st4_conus_{t:%Y%m%d%H}_06h.grb2"
            if p.exists():
                st4_paths.append(p)
            t += timedelta(hours=6)
    if not st4_paths:
        raise FileNotFoundError(f"No Stage IV files found in {plan['stageiv_dir']} for window")

    # ---- compute ----
    st4, st4_lat, st4_lon, valid_times = sum_stageiv(st4_paths)
    st4_on_wrf = regrid_to_wrf(st4_lat, st4_lon, st4, wrf.lat, wrf.lon,
                                method=plan.get('regrid_method', 'linear'))
    bias = wrf.total - st4_on_wrf

    # ---- basins ----
    basins = load_basin_polygons(
        plan.get('huc8_codes', []),
        plan.get('huc8_dir'),
        plan.get('huc8_combined_geojson'),
    )

    # ---- global stats ----
    global_mask = np.ones_like(wrf.total, dtype=bool)
    thr = plan.get('thresholds_mm', [1, 5, 10, 25, 50, 75, 100])
    elev_bins = plan.get('elevation_bins_m',
                          [0, 100, 250, 500, 750, 1000, 1250, 1500, 2000, 2500, 3000, 4000])
    g_stats = verification_stats(st4_on_wrf, wrf.total, global_mask, thr)

    result = RunResult()
    fmt = plan.get('format', 'png')
    dpi = int(plan.get('dpi', 130))
    stem = plan.get('filename_stem', 'qpf_verification')

    overview_bytes = render_overview(wrf, st4_on_wrf, bias, basins, g_stats, fmt=fmt, dpi=dpi)
    result.figures.append((f"{stem}_overview.{fmt}", f"image/{fmt}", overview_bytes))

    # ---- per-basin ----
    basin_colors = ['#0C447C', '#993556', '#3B6D11', '#854F0B', '#3C3489']
    basin_payload = {}
    csv_rows = ["huc8,name,n_cells,obs_mean_mm,fcst_mean_mm,mean_bias_mm,percent_bias,"
                "rmse_mm,mae_mm,pearson_r,csi_25mm,csi_50mm,csi_75mm"]
    for i, (code, basin) in enumerate(basins.items()):
        mask = build_basin_mask(wrf.lon, wrf.lat, basin['vertices'])
        bstats = verification_stats(st4_on_wrf, wrf.total, mask, thr)
        hyps = hypsometric_bias(st4_on_wrf, wrf.total, wrf.hgt, mask, elev_bins)
        color = basin_colors[i % len(basin_colors)]
        b_bytes = render_basin(wrf, st4_on_wrf, bias, code, basin, mask, bstats, hyps,
                                color=color, fmt=fmt, dpi=dpi)
        result.figures.append((f"{stem}_basin_{code}.{fmt}", f"image/{fmt}", b_bytes))
        basin_payload[code] = {'name': basin['name'], 'stats': bstats, 'hypsometric': hyps}

        cat = bstats.get('categorical', {})
        def _csi(t): return cat.get(str(t), {}).get('csi')
        csv_rows.append(
            f"{code},{basin['name']},{bstats.get('n_paired_points', 0)},"
            f"{bstats.get('obs_mean_mm', 0):.2f},{bstats.get('fcst_mean_mm', 0):.2f},"
            f"{bstats.get('mean_bias_mm', 0):+.2f},"
            f"{bstats.get('percent_bias') if bstats.get('percent_bias') is not None else ''},"
            f"{bstats.get('rmse_mm', 0):.2f},{bstats.get('mae_mm', 0):.2f},"
            f"{bstats.get('pearson_r', 0):.3f},"
            f"{_csi(25) if _csi(25) is not None else ''},"
            f"{_csi(50) if _csi(50) is not None else ''},"
            f"{_csi(75) if _csi(75) is not None else ''}"
        )

    if plan.get('export_csv', True):
        result.csv_files.append((f"{stem}_basin_stats.csv", "\n".join(csv_rows).encode('utf-8')))

    result.stats = {
        'wrf_meta': {
            'mp_physics': wrf.mp_physics,
            'bl_pbl_physics': wrf.bl_pbl_physics,
            'dx_m': wrf.dx_m,
            'start_date': wrf.start_date,
            'valid_time': wrf.valid_time,
        },
        'stageiv_files': [str(p.name) for p in st4_paths],
        'stageiv_valid_times': valid_times,
        'global': g_stats,
        'basins': basin_payload,
    }

    # ---- optional server-side save ----
    save_dir = plan.get('save_dir')
    if save_dir:
        sd = Path(save_dir); sd.mkdir(parents=True, exist_ok=True)
        for fname, _mime, data in result.figures:
            (sd / fname).write_bytes(data)
        for fname, data in result.csv_files:
            (sd / fname).write_bytes(data)
        (sd / f"{stem}_stats.json").write_text(json.dumps(result.stats, indent=2, default=str))

    return result


# =============================================================================
# Synthesize step: deterministic computation of structured findings the LLM
# turns into prose. NEVER pass raw arrays to the LLM — only this payload.
# =============================================================================
def build_synthesize_payload(result: RunResult) -> dict:
    """Reshape stats into a flat, LLM-friendly findings dict."""
    g = result.stats.get('global', {})
    findings = {
        'forecast_meta': result.stats.get('wrf_meta', {}),
        'global': {
            'n': g.get('n_paired_points'),
            'mean_bias_mm': g.get('mean_bias_mm'),
            'percent_bias': g.get('percent_bias'),
            'rmse_mm': g.get('rmse_mm'),
            'pearson_r': g.get('pearson_r'),
        },
        'basins': [],
    }
    for code, b in result.stats.get('basins', {}).items():
        s = b.get('stats', {})
        h = b.get('hypsometric', {})
        # Find the elevation band with the most negative and most positive bias
        bands = h.get('bin_centers_m') or []
        biases = h.get('mean_bias_mm') or []
        worst_dry = None; worst_wet = None
        if bands and biases:
            i_dry = int(np.argmin(biases))
            i_wet = int(np.argmax(biases))
            worst_dry = {'elevation_m': bands[i_dry], 'bias_mm': biases[i_dry]}
            worst_wet = {'elevation_m': bands[i_wet], 'bias_mm': biases[i_wet]}
        findings['basins'].append({
            'huc8': code, 'name': b.get('name'),
            'basin_mean_obs_mm': s.get('obs_mean_mm'),
            'basin_mean_fcst_mm': s.get('fcst_mean_mm'),
            'mean_bias_mm': s.get('mean_bias_mm'),
            'percent_bias': s.get('percent_bias'),
            'rmse_mm': s.get('rmse_mm'),
            'pearson_r': s.get('pearson_r'),
            'csi_25mm': s.get('categorical', {}).get('25', {}).get('csi'),
            'csi_50mm': s.get('categorical', {}).get('50', {}).get('csi'),
            'csi_75mm': s.get('categorical', {}).get('75', {}).get('csi'),
            'worst_dry_band': worst_dry,
            'worst_wet_band': worst_wet,
        })
    return findings


# =============================================================================
# Example invocation — same plan our exploration used, runnable as a script.
# =============================================================================
def run_demo():
    plan = {
        'wrfout_path': '/mnt/user-data/uploads/wrfout_d03_ISHMAEL_SUBSET.nc',
        'stageiv_dir': '/mnt/user-data/uploads',
        'auto_window_from_wrfout': True,
        'huc8_codes': ['18010110', '18020125'],
        'huc8_dir': None,   # use demo polygons
        'regrid_method': 'linear',
        'thresholds_mm': [1, 5, 10, 25, 50, 75, 100],
        'elevation_bins_m': [0, 100, 250, 500, 750, 1000, 1250, 1500, 2000, 2500, 3000, 4000],
        'format': 'png', 'dpi': 130,
        'filename_stem': 'qpf_verification',
        'save_dir': '/home/claude/wxagent_demo_out',
        'export_csv': True,
    }
    result = run(plan)
    findings = build_synthesize_payload(result)
    print(json.dumps(findings, indent=2, default=str))
    print(f"\nFigures: {[f[0] for f in result.figures]}")
    print(f"CSV: {[f[0] for f in result.csv_files]}")


if __name__ == '__main__':
    run_demo()
