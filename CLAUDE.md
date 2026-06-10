# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A WxSmith/WxAgent **gridded-snowfall analysis service**: a FastAPI app over an analysis engine that turns NOHRSC sfav2 snowfall netCDF files into maps, watershed statistics, climatologies, anomalies, and trends. It is built as "a thin product on a shared platform" — a multi-data-source pipeline where snowfall is the only source wired today (`StageIVSpec` is the planned Phase 2 addition).

There is a **second, independent app** in the same directory: the QPF verification tool (`qpf_*.py`), described under "The QPF verification app" below. It does not share the dispatcher/spec machinery of the snowfall app.

## Running

```bash
conda activate wxagent                                   # the project's conda env
WXAGENT_DATA_DIR=. uvicorn app:app --reload --port 8000  # snowfall app -> http://localhost:8000
uvicorn qpf_quick:app --port 8011                         # QPF app (separate process)
```

No `requirements.txt` / `environment.yml` exists — dependencies live in the `wxagent` conda env: `fastapi uvicorn matplotlib numpy netCDF4 pydantic(v2) scipy shapely geopandas cartopy pandas`, plus `cfgrib`/`eccodes` for the QPF app.

There is no test suite, linter config, or build step. Exercise the pipeline directly:

```bash
python worksheet_run.py --schema              # print the conditional worksheet schema
python worksheet_run.py                       # interactive: only shows fields whose depends_on is met
python render.py sfav2_CONUS_2024....nc california out.png   # standalone generic renderer
python render.py <file.nc> norcal --huc=8     # with HUC-8 overlay
python analyses_snowfall.py                   # smoke-print the worksheet field counts
python zonal.py / python watersheds.py / python profiles.py  # each has a __main__ smoke test
```

## Environment variables (data location)

- `WXAGENT_DATA_DIR` (default `.`) — directory of `sfav2_CONUS_*.nc` files. The `*.nc` files are committed here; water year is parsed from the second 10-digit timestamp in the filename.
- `WXAGENT_HUC8_DIR` (default `./huc8_shp`) — per-HUC8 `{code}.shp` shapefiles (USGS WBD).
- `WXAGENT_HUC8_CSV` (default `./ca_huc8_watersheds.csv`) — HUC-8 name↔code table; also the worksheet dropdown source.
- `WXAGENT_HUC8_CACHE` (default `ca_huc8_combined.geojson`) — cached combined HUC-8 layer for map overlays (this file IS committed).

**Important:** `huc8_shp/` and `ca_huc8_watersheds.csv` are **not present** in this repo. Map and anomaly analyses work from the committed `.nc` + `ca_huc8_combined.geojson`, but any `watershed_*` / `multi_watershed` analysis calls `watersheds.load_polygon()`, which needs the shapefiles and CSV and will raise `FileNotFoundError`/`KeyError` without them.

## Architecture

The core abstraction is a **three-stage split — compute (numbers) → render (figures) → draw (vetted, dumb renderers)** — fronted by a **discriminated-union dispatcher** keyed on `data_source`.

### Request flow
```
app.py (FastAPI routes)
  GET /api/schema -> analyses.build_analysis_worksheet()   # drives the conditional form
  GET /api/years  -> analyses.find_sfav2_files()
  POST /api/run   -> service.run_analysis(answers, DATA_DIR)
        |
        +-- analyses.parse_spec()      # validate answers dict -> SnowfallSpec (Pydantic v2)
        +-- service_common.style_context(spec)   # apply preset + rcParams for the whole request
        +-- service_snowfall.run_snowfall(spec)
              +-- map / map_anomaly: render directly (needs the grid)
              +-- watershed/*: analyses_snowfall.execute_snowfall() for NUMBERS, then render
```

### The `*_common` / `*_snowfall` split (key to the whole design)
Each layer is split into a **source-agnostic common module** and a **snowfall-specific module**, so adding a data source (e.g. Stage IV) means adding `analyses_stageiv.py` + `service_stageiv.py` and one `Union[...]` arm — not editing existing logic:

- `analyses.py` — public dispatcher + the `GriddedAnalysisSpec = Annotated[Union[SnowfallSpec], Field(discriminator="data_source")]`. Re-exports snowfall helpers so older `from analyses import ...` imports keep working. `parse_spec()` defaults missing `data_source` to `"snowfall"`.
- `analyses_common.py` — `BaseAnalysisFields` (shared spec fields: styling, output, annotation) and worksheet-fragment helpers (`map_style_field_specs`, `meta_style_field_specs`, `output_field_specs`) that every source composes into its worksheet so Style/Output groups are identical across sources. `NotWired` exception for schema'd-but-unwired analyses.
- `analyses_snowfall.py` — owns sfav2 everything: file discovery, the climatology/anomaly/trend engine, `SnowfallSpec`, `build_snowfall_worksheet()`, and `execute_snowfall()` which returns **numbers only** (no figures). Map types return a `{"directive": "render"}` stub for the service to handle.
- `service.py` — thin dispatcher: parse → `style_context` → route by `data_source`.
- `service_common.py` — source-agnostic render utilities, all duck-typed on the spec: `_resolve_levels()` (translates `color_scale`/`vmin`/`vmax`/`level_step` into contour levels), `resolve_rc()` + `STYLE_PRESETS` + `style_context()`, and `figure_script_bundle()` (the reproducible-script exporter).
- `service_snowfall.py` — builds each figure type and packages outputs (base64 PNG for display, format-specific download, optional CSV/script export). Every branch assembles a `kw` dict reused by both the live render and the exported script, so the two never drift.

### Rendering primitives
- `plots.py` — **VETTED, "dumb" renderers** (`gridded_field_map`, `watershed_series_plot`, `ranked_bar_plot`, `box_distribution_plot`, `trend_plot`). They receive already-computed, already-validated arrays and only handle aesthetics — **no unit math, aggregation, or science here**. This is what keeps an LLM-selected figure low-risk. `STYLE_OVERRIDES` (a module global) is set by `style_context()` and applied after the house style so user/preset rcParams win.
- `style.py` — single source of house style (`apply_house_style()`, `SNOW_COLORS` ramp, `annotate_provenance()` footer).
- `profiles.py` — `DatasetProfile` data-source-adapter: `identify(ds)` fingerprints a netCDF and supplies frozen presets (field var, unit conversion, colormap, levels, default region, coord kind). Adding a recognizable product = adding one `DatasetProfile` to the `PROFILES` registry. `render.py` is the generic CLI/entry point that uses it for any recognized file.
- `regions.py` — named domains with Lambert-Conformal projection params (West-Coast/CW3E focused).
- `watersheds.py` — resolve CA HUC-8 by name/code/substring and load polygons (local files only, air-gap friendly).
- `zonal.py` — point-in-polygon zonal statistics of a grid within a watershed (the deterministic "execute" for watershed analyses).
- `index.html` — single-file worksheet UI; reads `/api/schema` and renders fields, honoring each field's `depends_on` visibility rule.

### Two cross-cutting features to preserve when editing service code
1. **Reproducible figure export** (`spec.export_script`): emits a standalone `.py` (+ `.npz` of arrays) that re-imports the *same* vetted renderer and re-creates the figure exactly. Live render and exported script are driven from one shared `kw` dict — keep them in sync.
2. **Conditional worksheet schema**: worksheet fields carry `depends_on` rules (`{"field": ..., "equals"/"in": ...}`) that both `index.html` and `worksheet_run.py` evaluate. The schema also intentionally lists *unwired* analyses (`elevation_profile`, `climate_index_correlation`) as the planner's action space — `execute_snowfall` raises `NotWired` for them.

## The QPF verification app (separate)

`qpf_verification.py`, `qpf_verification_extensions.py`, and `qpf_quick.py` are a **standalone** WRF-QPF-vs-Stage-IV verification tool (FastAPI on port 8011). It follows an older `SCHEMA`/`ANALYSIS_TYPES` plan→execute→synthesize convention (not the discriminated-spec dispatcher of the snowfall app), reads WRF output + Stage IV GRIB2 (`cfgrib`), and has its own LLM-synthesis report path. The header comments describe integrating it into a separate `wxagent/` repo on `mlm-stormy`; treat it as adjacent, not part of the snowfall pipeline.

## Notes

- `_backup_phase1/` and `_backup_phase1_6/` are pre-refactor snapshots; `__pycache__/` is build artifacts. Don't edit these as live code.
- Pydantic is **v2** (discriminated unions need the `Literal` discriminator on each concrete subclass, `TypeAdapter`, `model_validator(mode="after")`).
- All rendering forces the `Agg` matplotlib backend at import time — keep that when adding new entry points.
