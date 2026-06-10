#!/usr/bin/env python3
"""
analyses.py — multi-source dispatcher.

This module is the public surface that app.py and service.py import against.
It does three things:

  1. Declares the discriminated union `GriddedAnalysisSpec` over per-source
     specs (currently SnowfallSpec; StageIVSpec lands in Phase 2).
  2. Provides `parse_spec()` to validate a raw answers dict into the right
     concrete spec, and `execute()` to dispatch to the per-source executor.
  3. Re-exports a handful of snowfall helpers that pre-refactor callers
     (notably app.py and service_snowfall.py) import as `from analyses import ...`,
     so external import statements don't have to change.
"""
from __future__ import annotations
from typing import Annotated, Union
from pydantic import Field, TypeAdapter

from analyses_common import NotWired, BaseAnalysisFields, _opt
from analyses_snowfall import (
    SnowfallSpec,
    find_sfav2_files,
    _read_snow_inches,
    cell_climatology,
    build_snowfall_worksheet,
    execute_snowfall,
)

# ---- discriminated union -----------------------------------------------------
# Phase 2 will extend Union[...] with StageIVSpec; the discriminator on the
# `data_source` field of each concrete spec routes parsing.
GriddedAnalysisSpec = Annotated[
    Union[SnowfallSpec],
    Field(discriminator="data_source"),
]

_SpecAdapter = TypeAdapter(GriddedAnalysisSpec)


# ---- public dispatchers ------------------------------------------------------
def parse_spec(answers: dict):
    """Validate a raw answers dict into the appropriate per-source spec.

    For back-compat: if the caller didn't include `data_source`, we default to
    `snowfall` (the original behaviour of the pre-refactor app).
    """
    if "data_source" not in answers:
        answers = {**answers, "data_source": "snowfall"}
    return _SpecAdapter.validate_python(answers)


def execute(spec, data_dir: str = ".") -> dict:
    """Dispatch numeric execution to the per-source module by `spec.data_source`."""
    if spec.data_source == "snowfall":
        return execute_snowfall(spec, data_dir)
    raise ValueError(f"no executor registered for data_source={spec.data_source!r}")


def build_analysis_worksheet() -> dict:
    """Return the worksheet schema for the UI. Phase 1: snowfall only.

    Phase 2 will merge per-source worksheets behind a top-level `data_source`
    selector and gate all per-source fields with `depends_on: data_source==...`.
    """
    return build_snowfall_worksheet()


__all__ = [
    # public API
    "GriddedAnalysisSpec", "parse_spec", "execute", "build_analysis_worksheet",
    # shared helpers
    "NotWired", "BaseAnalysisFields",
    # snowfall back-compat re-exports (app.py, service_snowfall.py)
    "find_sfav2_files", "_read_snow_inches", "cell_climatology",
    "SnowfallSpec", "execute_snowfall",
]
