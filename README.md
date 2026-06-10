# Gridded Snowfall Analysis

A FastAPI service over an analysis engine that turns NOHRSC sfav2 snowfall netCDF
grids into maps, watershed statistics, climatologies, anomalies, and trends. Also
bundles a separate WRF-QPF-vs-Stage-IV verification app (`qpf_*.py`). See
[CLAUDE.md](CLAUDE.md) for the architecture.

## Setup

```bash
conda env create -f environment.yml   # recreates the `wxagent` env
conda activate wxagent
```

(Pure-pip alternative: `python3.12 -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt`.
The QPF app additionally needs the `eccodes` C library, which `environment.yml` installs via conda.)

## Run

```bash
WXAGENT_DATA_DIR=. uvicorn app:app --reload --port 8000   # snowfall app -> http://localhost:8000
uvicorn qpf_quick:app --port 8011                          # QPF verification app (separate process)
```

## Getting the data

The data files are **not** stored in git (they're `.gitignore`d to keep the repo
small), so a fresh clone has the code but none of the inputs. Copy them onto the
machine separately — a cloud folder or `scp` from a machine that already has them
works fine — and point the environment variables below at their locations.

| Data | Needed for | Default location | How to obtain |
|------|-----------|------------------|---------------|
| `sfav2_CONUS_*.nc` (~28 MB total) | every snowfall analysis | `WXAGENT_DATA_DIR` (default `.`) | copy over, or re-download the NOHRSC National Snowfall Analysis v2 water-year files |
| `ca_huc8_combined.geojson` (~62 MB) | HUC-8 map overlays | `WXAGENT_HUC8_CACHE` (default `./ca_huc8_combined.geojson`) | copy over, **or** let it auto-regenerate from `huc8_shp/` on first overlay |
| `huc8_shp/` (per-HUC8 `{code}.shp`) | all `watershed_*` analyses | `WXAGENT_HUC8_DIR` (default `./huc8_shp`) | USGS Watershed Boundary Dataset (WBD), California HUC-8s |
| `ca_huc8_watersheds.csv` (name↔code table) | watershed name lookup + worksheet dropdown | `WXAGENT_HUC8_CSV` (default `./ca_huc8_watersheds.csv`) | derived from the WBD attributes |

Environment variables (all optional; shown with defaults):

```bash
export WXAGENT_DATA_DIR=.                          # dir of sfav2_CONUS_*.nc files
export WXAGENT_HUC8_DIR=./huc8_shp                 # per-HUC8 shapefiles
export WXAGENT_HUC8_CSV=./ca_huc8_watersheds.csv   # HUC-8 name<->code table
export WXAGENT_HUC8_CACHE=./ca_huc8_combined.geojson  # combined-overlay cache
```

**What works without which data:**

- **Map / anomaly analyses** need only the `sfav2_*.nc` files (plus the committed
  `us_states.geojson` for state boundaries). HUC-8 overlays additionally need either
  `ca_huc8_combined.geojson` or `huc8_shp/`.
- **`watershed_*` and `multi_watershed` analyses** require `huc8_shp/` **and**
  `ca_huc8_watersheds.csv`; without them, `watersheds.load_polygon()` raises
  `FileNotFoundError` / `KeyError`.

> Note: `huc8_shp/` and `ca_huc8_watersheds.csv` are not currently present in this
> project on any machine — they must be sourced from the USGS WBD before watershed
> analyses will run.
