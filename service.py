#!/usr/bin/env python3
"""
service.py — multi-source dispatcher: parse answers + route to per-source renderer.

Pre-refactor this module owned all the snowfall rendering. That now lives in
service_snowfall.py. This file is just the public entry point.

The style_context() wrapper applies the spec's style preset + rc overrides for
the whole request (render AND savefig), for every data source, in one place.
"""
import matplotlib
matplotlib.use("Agg")

from analyses import parse_spec
from service_common import style_context
from service_snowfall import run_snowfall


def run_analysis(answers: dict, data_dir: str = ".") -> dict:
    """Validate the answers dict and dispatch to the per-source service.

    Raises pydantic.ValidationError on bad spec (handled by app.py route),
    and ValueError / FileNotFoundError on missing data (also handled there).
    """
    spec = parse_spec(answers)
    with style_context(spec):
        if spec.data_source == "snowfall":
            return run_snowfall(spec, data_dir)
        raise ValueError(f"no service registered for data_source={spec.data_source!r}")
