#!/usr/bin/env python3
"""
analyses_common.py — shared base spec fields + worksheet helpers for the
multi-source pipeline.

Both SnowfallSpec (analyses_snowfall.py) and StageIVSpec (analyses_stageiv.py, Phase 2)
inherit BaseAnalysisFields. The discriminator field `data_source` is declared on each
subclass as a Literal, not here — Pydantic v2 discriminated unions require the
discriminator on the concrete subclass.

The worksheet helpers (map_style_field_specs, meta_style_field_specs,
output_field_specs) emit the field-dict fragments that get composed into each
per-source build_*_worksheet(). Keeping them here means the Style and Output groups
are visually and behaviourally identical across all data sources.
"""
from __future__ import annotations
from typing import Literal, Optional

from pydantic import BaseModel, field_validator


class NotWired(ValueError):
    """Raised when an analysis needs a data source not yet ingested."""


class BaseAnalysisFields(BaseModel):
    """Output, annotation, and universal map-styling fields shared by every spec.

    Everything here is independent of data source. Per-source style choices
    (e.g. `show_extremes` on bar charts, `reference_line` on YoY plots) stay on
    the subclass because their applicability is analysis-family-specific.
    """
    # plot annotation (shared)
    title: Optional[str] = None
    caption: Optional[str] = None
    provenance_footer: bool = True

    # map styling — applies to any gridded-field renderer
    colormap: Optional[str] = None
    color_scale: Literal["auto", "linear", "diverging"] = "auto"
    vmin: Optional[float] = None
    vmax: Optional[float] = None
    level_step: Optional[float] = None

    # global figure styling — applies to EVERY plot type
    style_preset: Literal["default", "publication", "presentation", "dark"] = "default"
    rc_overrides: Optional[dict] = None   # raw matplotlib rcParams; JSON str accepted

    # output packaging (shared)
    output_format: Literal["png", "pdf", "svg"] = "png"
    dpi: int = 200
    save_dir: Optional[str] = None
    filename: Optional[str] = None
    export_data: bool = False
    export_script: bool = False           # ship a reproducible .py (+ .npz) per figure

    @field_validator("rc_overrides", mode="before")
    @classmethod
    def _parse_rc_overrides(cls, v):
        """Accept a JSON string from the web form, a dict from API callers."""
        if v is None or v == "":
            return None
        if isinstance(v, str):
            import json
            try:
                parsed = json.loads(v)
            except json.JSONDecodeError as e:
                raise ValueError(f"rc_overrides must be valid JSON: {e}")
            if not isinstance(parsed, dict):
                raise ValueError("rc_overrides JSON must be an object, e.g. "
                                 '{"font.weight": "bold"}')
            return parsed
        return v


def _opt(v, l):
    """Shorthand for {value, label} dicts in worksheet schemas."""
    return {"value": v, "label": l}


# Default cmap palette offered in the UI. Per-source builders can override this list
# (e.g. Stage IV may prefer precip-specific palettes).
DEFAULT_CMAPS = [
    ("", "Default"),
    ("viridis", "viridis"),
    ("Blues", "Blues"),
    ("RdBu_r", "RdBu (diverging)"),
    ("BrBG", "BrBG (diverging)"),
    ("magma", "magma"),
]


def map_style_field_specs(map_types, *, cmaps=None, default_cmap_label=None):
    """Worksheet field dicts for the universal map-styling controls.

    Returns the 5 fields in canonical order: colormap, color_scale, vmin, vmax,
    level_step. All gated on `analysis_type in map_types`, with vmin/vmax/level_step
    additionally gated on color_scale being linear or diverging.

    Args:
        map_types: list of analysis_type values this Style block applies to.
        cmaps: optional list of (value, label) tuples; defaults to DEFAULT_CMAPS.
        default_cmap_label: overrides the first cmap option's label
            (e.g. 'Default (NWS snowfall)' for the snowfall worksheet).
    """
    cmaps = list(cmaps) if cmaps else list(DEFAULT_CMAPS)
    if default_cmap_label is not None and cmaps:
        cmaps[0] = (cmaps[0][0], default_cmap_label)
    return [
        {"key": "colormap", "label": "Colormap", "widget": "select", "group": "Style",
         "options": [_opt(v, l) for v, l in cmaps],
         "depends_on": {"field": "analysis_type", "in": list(map_types)}},
        {"key": "color_scale", "label": "Color scale", "widget": "select", "group": "Style",
         "default": "auto",
         "options": [_opt("auto", "Auto"),
                     _opt("linear", "Linear continuous"),
                     _opt("diverging", "Diverging (centered 0)")],
         "depends_on": {"field": "analysis_type", "in": list(map_types)}},
        {"key": "vmin", "label": "Color min", "widget": "number", "group": "Style",
         "depends_on": {"field": "color_scale", "in": ["linear", "diverging"]}},
        {"key": "vmax", "label": "Color max", "widget": "number", "group": "Style",
         "depends_on": {"field": "color_scale", "in": ["linear", "diverging"]}},
        {"key": "level_step", "label": "Level step", "widget": "number", "group": "Style",
         "depends_on": {"field": "color_scale", "in": ["auto", "linear", "diverging"]},
         "help": "Increment between contour boundaries (in field units \u2014 in for snowfall, "
                 "\u03c3 for standardized). Leave blank to use the default. In auto mode this "
                 "snaps the data-derived range to multiples of step (e.g. step=1 \u2192 whole-number ticks)."},
    ]


def meta_style_field_specs():
    """Worksheet field dicts for preset/rc/title/caption/provenance (always shown)."""
    return [
        {"key": "style_preset", "label": "Style preset", "widget": "select", "group": "Style",
         "default": "default",
         "options": [_opt("default", "House default"),
                     _opt("publication", "Publication (AMS, bold)"),
                     _opt("presentation", "Presentation (large type)"),
                     _opt("dark", "WxSmith dark")],
         "help": "Named rcParams bundle applied on top of the house style. "
                 "Affects every plot type, not just maps."},
        {"key": "rc_overrides", "label": "Matplotlib rcParams overrides (JSON)",
         "widget": "textarea", "group": "Style",
         "placeholder": '{"font.weight": "bold", "xtick.labelsize": 9}',
         "help": "Applied after the preset \u2014 any matplotlib rcParams key works "
                 "(fonts, ticks, grids, line widths, legend, figure\u2026). "
                 "Unknown keys are rejected with a clear error."},
        {"key": "title", "label": "Custom title (optional)", "widget": "text", "group": "Style"},
        {"key": "caption", "label": "Caption (optional)", "widget": "text", "group": "Style"},
        {"key": "provenance_footer", "label": "Provenance footer", "widget": "toggle",
         "group": "Style", "default": True},
    ]


def output_field_specs():
    """Worksheet field dicts for output format/dpi/save/export (always shown)."""
    return [
        {"key": "output_format", "label": "Output format", "widget": "select", "group": "Output",
         "default": "png",
         "options": [_opt("png", "PNG"), _opt("pdf", "PDF"), _opt("svg", "SVG")]},
        {"key": "dpi", "label": "Resolution (DPI)", "widget": "number", "group": "Output",
         "default": 200,
         "depends_on": {"field": "output_format", "equals": "png"}},
        {"key": "save_dir", "label": "Save to directory (server-side, optional)",
         "widget": "text", "group": "Output"},
        {"key": "filename", "label": "Filename (optional)", "widget": "text", "group": "Output"},
        {"key": "export_data", "label": "Also export computed data (CSV)",
         "widget": "toggle", "group": "Output", "default": False},
        {"key": "export_script", "label": "Export reproducible figure script (.py + data)",
         "widget": "toggle", "group": "Output", "default": False,
         "help": "Download a standalone Python script (plus an .npz of the computed "
                 "arrays when needed) that re-creates this figure with the vetted "
                 "renderer. Edit anything matplotlib offers and re-run."},
    ]
