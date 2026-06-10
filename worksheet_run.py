#!/usr/bin/env python3
"""
worksheet_run.py — drive the gridded-analysis worksheet end to end:
    schema -> fill -> validate -> execute -> render

Two modes:
  * interactive:  python worksheet_run.py            (prompts; only shows fields whose
                                                       depends_on conditions are met)
  * scripted:     run_worksheet(answers={...})        (pre-filled; for tests/automation)
  * inspect:      python worksheet_run.py --schema    (print the conditional schema)

This is the single entry point a UI (or later the LLM) targets. It evaluates the
worksheet's depends_on rules itself, so the conditional structure is actually exercised.
"""
from __future__ import annotations
import sys, json
from analyses import build_analysis_worksheet, GriddedAnalysisSpec, execute, find_sfav2_files


# ---------- conditional-visibility evaluation (the worksheet's core logic) ----------
def _visible(field: dict, answers: dict) -> bool:
    dep = field.get("depends_on")
    if not dep:
        return True
    val = answers.get(dep["field"])
    if "equals" in dep:
        return val == dep["equals"]
    if "in" in dep:
        return val in dep["in"]
    return True


def show_schema():
    ws = build_analysis_worksheet()
    print(f"pipeline: {ws['pipeline']}\n")
    for f in ws["fields"]:
        dep = f.get("depends_on")
        cond = ""
        if dep:
            tgt = dep.get("equals", dep.get("in"))
            cond = f"   [shown when {dep['field']} = {tgt}]"
        opts = ""
        if f.get("options"):
            n = len(f["options"])
            opts = (" {" + ", ".join(o["value"] for o in f["options"]) + "}"
                    if n <= 8 else f" {{{n} options}}")
        print(f"  • {f['key']} ({f['widget']}){opts}{cond}")


# ---------- interactive prompting ----------
def _ask_select(field: dict, answers: dict, data_dir: str):
    opts = field.get("options")
    if field["key"] == "water_year":                       # dynamic from data dir
        yrs = sorted(find_sfav2_files(data_dir))
        opts = [{"value": y, "label": f"WY{y}"} for y in yrs]
    label = field["label"]

    if opts and len(opts) > 15:                            # too many to list -> type it
        print(f"\n{label} — type a name (substring ok, e.g. 'Yuba'):")
        return input("> ").strip()

    print(f"\n{label}:")
    for i, o in enumerate(opts, 1):
        mark = "  (default)" if o["value"] == field.get("default") else ""
        print(f"  {i}. {o['label']}{mark}")
    raw = input("> ").strip()
    if raw == "" and field.get("default") is not None:
        return field["default"]
    if raw.isdigit() and 1 <= int(raw) <= len(opts):
        return opts[int(raw) - 1]["value"]
    return raw


def _ask_toggle(field: dict) -> bool:
    raw = input(f"\n{field['label']} [y/N]: ").strip().lower()
    return raw in ("y", "yes", "1", "true")


def prompt_worksheet(data_dir: str = ".") -> dict:
    ws = build_analysis_worksheet()
    answers: dict = {}
    for field in ws["fields"]:
        if not _visible(field, answers):
            continue                                       # conditional field hidden
        try:
            if field["widget"] == "toggle":
                answers[field["key"]] = _ask_toggle(field)
            else:
                v = _ask_select(field, answers, data_dir)
                if v != "":
                    answers[field["key"]] = v
        except EOFError:
            break
    return answers


# ---------- the single end-to-end entry point ----------
def run_worksheet(answers: dict | None = None, data_dir: str = "."):
    if answers is None:
        answers = prompt_worksheet(data_dir)
    if "water_year" in answers and str(answers["water_year"]).isdigit():
        answers["water_year"] = int(answers["water_year"])

    print("\n— filled worksheet —")
    print(json.dumps(answers, indent=2))

    spec = GriddedAnalysisSpec(**answers)                  # VALIDATE
    print(f"\nvalidated: analysis_type={spec.analysis_type}")

    if spec.analysis_type == "map":
        from render import render_file
        files = find_sfav2_files(data_dir)
        if not files:
            raise FileNotFoundError(f"no sfav2 files in {data_dir}")
        path = files[max(files)]                           # most recent WY for the map
        out = render_file(path, region=spec.region,
                          huc_level=8 if spec.huc8_overlay else None)
        return out

    result = execute(spec, data_dir)                       # EXECUTE
    print("\n— result —")
    print(json.dumps(result, indent=2))

    if spec.analysis_type == "watershed_yoy":              # RENDER
        from plots import watershed_series_plot
        from style import annotate_provenance
        series = {int(y): v for y, v in result["series_by_year"].items() if v is not None}
        yoy = {int(y): v for y, v in result["yoy_change"].items()}
        years = sorted(series)
        fig, _ = watershed_series_plot(
            years=years, values=series, yoy=yoy, units=result["units"],
            watershed_name=result["watershed"], huc8=result["huc8"],
            statistic=spec.statistic)
        annotate_provenance(fig, f"via worksheet | HUC-8 {result['huc8']} | "
                                 f"{len(result['provenance']['sources'])} sfav2 files")
        out = f"{result['huc8']}_{spec.statistic}_yoy.png"
        fig.savefig(out, dpi=200)
        print(f"\nrendered: {out}")
        return out

    return result                                          # watershed_mean -> numbers


if __name__ == "__main__":
    if "--schema" in sys.argv:
        show_schema()
    else:
        ddir = next((a for a in sys.argv[1:] if not a.startswith("--")), ".")
        run_worksheet(data_dir=ddir)
