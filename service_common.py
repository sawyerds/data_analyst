#!/usr/bin/env python3
"""
service_common.py — rendering utilities shared by every per-source service module.

These functions deliberately take a duck-typed `spec` (anything with the relevant
attributes from BaseAnalysisFields: color_scale, vmin, vmax, level_step). That keeps
them independent of which discriminated spec is in play — SnowfallSpec today,
StageIVSpec next, and whatever lands after.
"""
from __future__ import annotations
import math
import numpy as np


def _nice_step(range_size, target_n: int = 10) -> float:
    """Pick a 'nice' step (1/2/5 \u00d7 10^k) giving roughly target_n divisions.

    Used when the user is in linear/diverging mode but leaves level_step blank.
    Returns a positive float; falls back to 1.0 on degenerate inputs.
    """
    if range_size <= 0 or not math.isfinite(range_size):
        return 1.0
    raw = range_size / max(target_n, 1)
    magnitude = 10 ** math.floor(math.log10(raw))
    normalized = raw / magnitude
    if normalized < 1.5:
        return 1 * magnitude
    if normalized < 3.5:
        return 2 * magnitude
    if normalized < 7.5:
        return 5 * magnitude
    return 10 * magnitude


def _resolve_levels(spec, *, data=None, default_levels=None):
    """Translate spec.color_scale / vmin / vmax / level_step into a contour-level list.

    Modes:
      - 'auto'      -> return default_levels (caller's fallback; preserves prof.levels
                       behaviour for snowfall maps and the 98th-pct anomaly default)
      - 'linear'    -> arange snapped to multiples of step covering [vmin, vmax]
      - 'diverging' -> zero-centered [-k*s, ..., 0, ..., k*s], k = ceil(M/step)
                       M = |vmax| if given, else 98th pct of |data|

    Step rules:
      - level_step provided -> use it
      - level_step blank    -> _nice_step() picks one targeting ~10 divisions
    """
    cs = getattr(spec, "color_scale", None) or "auto"
    user_step = float(spec.level_step) if spec.level_step else None

    if cs == "auto":
        # If the user provided a step in auto mode, snap the default range to it.
        # This is how the user gets clean whole-number ticks without changing the
        # data-derived range that auto mode computes (e.g. \u00b198th pct of anom).
        if user_step and default_levels and len(default_levels) >= 2:
            lo = float(default_levels[0])
            hi = float(default_levels[-1])
            lo_snap = math.floor(lo / user_step) * user_step
            hi_snap = math.ceil(hi / user_step) * user_step
            n_bins = int(round((hi_snap - lo_snap) / user_step))
            if n_bins >= 1:
                return [lo_snap + i * user_step for i in range(n_bins + 1)]
        return default_levels

    if cs == "diverging":
        if spec.vmax is not None:
            M = abs(float(spec.vmax))
        elif data is not None:
            M = float(np.nanpercentile(np.abs(data), 98))
        else:
            return default_levels
        if M <= 0:
            return default_levels
        step = user_step if user_step else _nice_step(2 * M)
        k = int(math.ceil(M / step))
        if k <= 0:
            return default_levels
        return [(i - k) * step for i in range(2 * k + 1)]

    # linear
    if spec.vmin is not None and spec.vmax is not None:
        vmin, vmax = float(spec.vmin), float(spec.vmax)
    elif data is not None:
        vmin = float(spec.vmin) if spec.vmin is not None else float(np.nanmin(data))
        vmax = float(spec.vmax) if spec.vmax is not None else float(np.nanmax(data))
    else:
        return default_levels
    if vmax <= vmin:
        return default_levels
    step = user_step if user_step else _nice_step(vmax - vmin)
    if step <= 0:
        return default_levels
    lo = math.floor(vmin / step) * step
    hi = math.ceil(vmax / step) * step
    n_bins = int(round((hi - lo) / step))
    if n_bins < 1:
        return default_levels
    return [lo + i * step for i in range(n_bins + 1)]


# =====================================================================
# Layer 1+2: style presets and rcParams resolution
# =====================================================================

STYLE_PRESETS: dict[str, dict] = {
    "default": {},
    # AMS-publication look: bold everything, ticks-in, 300 dpi.
    "publication": {
        "font.weight": "bold",
        "axes.labelweight": "bold",
        "axes.titleweight": "bold",
        "font.size": 10,
        "axes.labelsize": 10.5,
        "axes.titlesize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "axes.linewidth": 0.8,
        "legend.frameon": False,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    },
    # Slides: big type, thick lines.
    "presentation": {
        "font.size": 15,
        "font.weight": "bold",
        "axes.labelweight": "bold",
        "axes.titleweight": "bold",
        "axes.titlesize": 19,
        "axes.labelsize": 16,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 13,
        "lines.linewidth": 2.5,
        "lines.markersize": 8,
        "axes.linewidth": 1.4,
        "savefig.dpi": 200,
    },
    # WxSmith UI palette. Best-effort: vetted renderers hardcode some artist
    # colors (e.g. annotation boxes), which won't follow the dark background.
    "dark": {
        "figure.facecolor": "#0b1220",
        "savefig.facecolor": "#0b1220",
        "axes.facecolor": "#121c2e",
        "axes.edgecolor": "#8aa0bd",
        "text.color": "#e8eef7",
        "axes.labelcolor": "#e8eef7",
        "xtick.color": "#8aa0bd",
        "ytick.color": "#8aa0bd",
        "grid.color": "#22304a",
        "legend.frameon": False,
    },
}


def resolve_rc(spec) -> dict:
    """Merge style preset + user rc_overrides into one validated rcParams dict.

    Order: preset first, then user overrides on top (user wins).
    Raises ValueError naming any keys matplotlib doesn't recognize, so the
    web UI gets a clear 400 instead of a mid-render crash.
    """
    merged: dict = {}
    preset = getattr(spec, "style_preset", None) or "default"
    merged.update(STYLE_PRESETS.get(preset, {}))
    user = getattr(spec, "rc_overrides", None)
    if user:
        merged.update(user)
    if not merged:
        return {}
    import matplotlib.pyplot as plt
    bad = sorted(k for k in merged if k not in plt.rcParams)
    if bad:
        raise ValueError(
            "Unknown matplotlib rcParams key(s): " + ", ".join(bad)
            + ". Valid keys are those in matplotlib.rcParams."
        )
    return merged


from contextlib import contextmanager


@contextmanager
def style_context(spec):
    """Apply the spec's preset + rc overrides for the duration of one request.

    Implementation note: every vetted renderer calls plots._style(), which is
    apply_house_style() followed by plots.STYLE_OVERRIDES. Setting the module
    attribute here (instead of plt.rc_context(merged)) is what lets user
    overrides WIN over the house style rather than being stomped by it.
    The surrounding rc_context() snapshots/restores global rcParams state.
    """
    merged = resolve_rc(spec)
    if not merged:
        yield {}
        return
    import matplotlib.pyplot as plt
    import plots
    with plt.rc_context():
        plots.STYLE_OVERRIDES = dict(merged)
        try:
            yield merged
        finally:
            plots.STYLE_OVERRIDES = {}


# =====================================================================
# Layer 3: reproducible figure-script export
# =====================================================================

def _pylit(v):
    """Recursively convert numpy scalars to Python natives so repr() emits
    clean literals that eval identically in the exported script."""
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, dict):
        return {_pylit(k): _pylit(x) for k, x in v.items()}
    if isinstance(v, tuple):
        return tuple(_pylit(x) for x in v)
    if isinstance(v, list):
        return [_pylit(x) for x in v]
    return v


def figure_script_bundle(
    *,
    analysis: str,
    func_module: str,
    func_name: str,
    literal_kwargs: dict | None = None,
    array_kwargs: dict | None = None,
    reconstruct_kwargs: dict | None = None,   # kwarg name -> python source expr
    extra_imports: tuple = (),
    post_lines: tuple = (),                   # raw code lines after fig creation
    rc: dict | None = None,
    out_stem: str,
    output_format: str = "png",
    dpi: int = 200,
    provenance: str | None = None,
) -> dict:
    """Build a standalone .py (+ optional .npz) that reproduces a figure exactly.

    The script imports the SAME vetted renderer from the user's local modules,
    loads the SAME computed arrays from the .npz, applies the SAME style
    overrides, and savefigs. The user edits anything and re-runs.

    Returns {"script": str, "script_name": str, "npz": bytes|None, "npz_name": str|None}.
    """
    import io
    import pprint
    from datetime import datetime, timezone

    literal_kwargs = dict(literal_kwargs or {})
    array_kwargs = dict(array_kwargs or {})
    reconstruct_kwargs = dict(reconstruct_kwargs or {})
    rc = dict(rc or {})

    script_name = f"{out_stem}.py"
    npz_name = f"{out_stem}_data.npz" if array_kwargs else None

    # ---- arrays -> npz bytes (masked arrays: fill with NaN, re-mask on load)
    npz_bytes = None
    masked_names = []
    if array_kwargs:
        store = {}
        for name, arr in array_kwargs.items():
            if isinstance(arr, np.ma.MaskedArray):
                masked_names.append(name)
                store[name] = np.ma.filled(arr.astype("f8"), np.nan)
            else:
                store[name] = np.asarray(arr)
        buf = io.BytesIO()
        np.savez_compressed(buf, **store)
        npz_bytes = buf.getvalue()

    # ---- assemble the script line by line (no template-brace pitfalls)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%MZ")
    L: list[str] = []
    L.append("#!/usr/bin/env python3")
    L.append('"""')
    L.append(f"Reproducible figure script — analysis: {analysis}")
    L.append(f"Generated by the WxSmith gridded-analysis service, {ts}.")
    L.append("")
    L.append(f"Re-creates the figure EXACTLY: same vetted renderer "
             f"({func_module}.{func_name}),")
    if npz_name:
        L.append(f"same computed arrays (loaded from {npz_name}, keep it next to this file).")
    else:
        L.append("same inputs (embedded below as literals).")
    L.append("")
    L.append("Edit anything and re-run from the gridded_snowfall_analysis directory:")
    L.append("")
    L.append("    conda activate wxagent")
    L.append(f"    python {script_name}")
    L.append('"""')
    L.append("import numpy as np")
    L.append("import matplotlib")
    L.append('matplotlib.use("Agg")')
    L.append("import matplotlib.pyplot as plt")
    L.append("from pathlib import Path")
    L.append("")
    L.append("import plots")
    L.append(f"from {func_module} import {func_name}")
    for imp in extra_imports:
        L.append(imp)
    L.append("")
    L.append("# ---- styling: preset + overrides captured at generation time ----------")
    L.append("# Applied on top of the house style inside the renderer. Edit freely;")
    L.append("# any matplotlib rcParams key works. Empty dict = pure house style.")
    L.append("RC = " + pprint.pformat(_pylit(rc), sort_dicts=True))
    L.append("plots.STYLE_OVERRIDES = RC")
    L.append("")
    if array_kwargs:
        L.append("# ---- computed data (the exact arrays the service rendered) ------------")
        L.append(f'_d = np.load(Path(__file__).with_name("{npz_name}"))')
        for name in array_kwargs:
            if name in masked_names:
                L.append(f'{name} = np.ma.masked_invalid(_d["{name}"])  '
                         f"# was masked; NaN-filled in the npz")
            else:
                L.append(f'{name} = _d["{name}"]')
        L.append("")
    L.append("# ---- renderer inputs ---------------------------------------------------")
    L.append("kwargs = dict(")
    for name in array_kwargs:
        L.append(f"    {name}={name},")
    for name, v in literal_kwargs.items():
        L.append(f"    {name}={repr(_pylit(v))},")
    for name, expr in reconstruct_kwargs.items():
        L.append(f"    {name}={expr},")
    L.append(")")
    L.append("")
    L.append("# ---- render --------------------------------------------------------------")
    L.append(f"result = {func_name}(**kwargs)")
    L.append("fig = result[0] if isinstance(result, tuple) else result")
    for ln in post_lines:
        L.append(ln)
    L.append("")
    L.append("# ---- save ------------------------------------------------------------------")
    L.append(f'out = Path(__file__).with_suffix(".{output_format}")')
    L.append(f'fig.savefig(out, dpi={int(dpi)}, bbox_inches="tight")')
    L.append('print(f"wrote {out}")')
    L.append("")

    return {"script": "\n".join(L), "script_name": script_name,
            "npz": npz_bytes, "npz_name": npz_name}
