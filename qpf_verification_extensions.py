from __future__ import annotations

import io
import json
import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from qpf_verification import (
    run,
    build_synthesize_payload,
    SCHEMA as QPF_VERIFICATION_SCHEMA,
    RunResult,
)


SYNTHESIZE_SYSTEM_PROMPT = """You are a senior atmospheric scientist writing a verification report.
You will receive a JSON findings payload from a deterministic verification run.
Rules:
1. Reference ONLY numbers present in the payload. Never invent or estimate values.
2. Use the exact basin names and HUC-8 codes from the payload.
3. Lead with the global picture, then go basin by basin.
4. When a basin shows compensating errors (low percent_bias but low pearson_r), call it out explicitly.
5. When worst_dry_band and worst_wet_band have opposite signs, describe the dipole.
6. Use AMS-style scientific prose. Sentence case. No emoji. No marketing language.
7. End with a one-line caveat about the limitations of single-event verification."""


def synthesize_report(findings: dict, model: str = "claude-sonnet-4-5",
                       max_tokens: int = 1500) -> str:
    import anthropic
    client = anthropic.Anthropic()
    msg = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=SYNTHESIZE_SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": (
                "Verification findings payload follows. Write the report.\n\n"
                f"```json\n{json.dumps(findings, indent=2, default=str)}\n```"
            ),
        }],
    )
    return "".join(block.text for block in msg.content if block.type == "text")


def synthesize_report_offline(findings: dict) -> str:
    g = findings.get('global', {})
    meta = findings.get('forecast_meta', {})
    lines = [
        f"WRF QPF verification report — init {meta.get('start_date', '?')[:13]} → valid "
        f"{meta.get('valid_time', '?')[:13]} (MP_PHYSICS={meta.get('mp_physics')}, "
        f"BL_PBL_PHYSICS={meta.get('bl_pbl_physics')}, DX={meta.get('dx_m', 0):.0f} m).",
        "",
        f"Across the full d03 domain (n={g.get('n')} paired cells), the forecast carried a "
        f"mean bias of {g.get('mean_bias_mm', 0):+.1f} mm ({g.get('percent_bias', 0):+.1f}%) "
        f"with RMSE {g.get('rmse_mm', 0):.1f} mm and spatial Pearson correlation of "
        f"{g.get('pearson_r', 0):.3f}.",
        "",
    ]
    for b in findings.get('basins', []):
        pb = b.get('percent_bias') or 0.0
        r = b.get('pearson_r') or 0.0
        dry = b.get('worst_dry_band') or {}
        wet = b.get('worst_wet_band') or {}
        lines.append(
            f"{b['name']} (HUC8 {b['huc8']}): basin-mean QPE {b.get('basin_mean_obs_mm', 0):.1f} mm, "
            f"QPF {b.get('basin_mean_fcst_mm', 0):.1f} mm, bias {b.get('mean_bias_mm', 0):+.1f} mm "
            f"({pb:+.1f}%), RMSE {b.get('rmse_mm', 0):.1f} mm, r {r:.3f}, "
            f"CSI >=25/50/75 mm = {b.get('csi_25mm') or 0:.2f}/{b.get('csi_50mm') or 0:.2f}/"
            f"{b.get('csi_75mm') or 0:.2f}."
        )
        if dry and wet and dry.get('bias_mm', 0) * wet.get('bias_mm', 0) < 0:
            lines.append(
                f"  Elevation dipole present: dry by {dry['bias_mm']:.1f} mm near "
                f"{dry['elevation_m']:.0f} m, wet by {wet['bias_mm']:+.1f} mm near "
                f"{wet['elevation_m']:.0f} m."
            )
        if abs(pb) < 5 and r < 0.5:
            lines.append("  Compensating spatial errors: basin total agrees but pattern skill is weak.")
        lines.append("")
    lines.append("Caveat: results reflect a single forecast and event; "
                 "skill estimates carry large sampling uncertainty until aggregated across cases.")
    return "\n".join(lines)


@dataclass
class EnsembleMember:
    name: str
    wrfout_path: str
    mp_physics: int | None = None
    pbl_physics: int | None = None


@dataclass
class EnsembleResult:
    members: list[str] = field(default_factory=list)
    per_member_stats: dict[str, dict] = field(default_factory=dict)
    figures: list[tuple[str, str, bytes]] = field(default_factory=list)
    csv_files: list[tuple[str, bytes]] = field(default_factory=list)


def run_ensemble(members: list[EnsembleMember], base_plan: dict,
                  comparison_basins: list[str] | None = None) -> EnsembleResult:
    er = EnsembleResult()
    for m in members:
        plan = dict(base_plan)
        plan['wrfout_path'] = m.wrfout_path
        plan['filename_stem'] = f"qpf_verification_{m.name}"
        plan['save_dir'] = str(Path(base_plan.get('save_dir', '/tmp/wxagent_ensemble')) / m.name)
        result: RunResult = run(plan)
        findings = build_synthesize_payload(result)
        er.members.append(m.name)
        er.per_member_stats[m.name] = findings
        er.figures.extend([(f"{m.name}__{n}", t, b) for (n, t, b) in result.figures])
        er.csv_files.extend([(f"{m.name}__{n}", b) for (n, b) in result.csv_files])

    fmt = base_plan.get('format', 'png')
    dpi = int(base_plan.get('dpi', 130))

    basins_seen: list[tuple[str, str]] = []
    for m in er.members:
        for b in er.per_member_stats[m].get('basins', []):
            key = (b['huc8'], b['name'])
            if key not in basins_seen:
                basins_seen.append(key)
    if comparison_basins:
        basins_seen = [b for b in basins_seen if b[0] in comparison_basins]

    def collect(metric: str, basin_huc8: str | None = None):
        vals = []
        for m in er.members:
            if basin_huc8 is None:
                vals.append(er.per_member_stats[m].get('global', {}).get(metric))
            else:
                found = False
                for b in er.per_member_stats[m].get('basins', []):
                    if b['huc8'] == basin_huc8:
                        vals.append(b.get(metric))
                        found = True
                        break
                if not found:
                    vals.append(None)
        return [v if v is not None else np.nan for v in vals]

    n_basins = len(basins_seen) + 1
    n_rows = 3
    fig, axes = plt.subplots(n_rows, n_basins, figsize=(4.0 * n_basins, 3.4 * n_rows), dpi=dpi,
                              sharey='row')
    if n_basins == 1:
        axes = axes.reshape(n_rows, 1)
    x = np.arange(len(er.members))

    panels = [('Global', None)] + [(name, huc) for (huc, name) in basins_seen]
    metrics = [
        ('mean_bias_mm', 'Mean bias (mm)'),
        ('rmse_mm', 'RMSE (mm)'),
        ('pearson_r', 'Pearson r'),
    ]
    color_cycle = ['#0C447C', '#993556', '#3B6D11', '#854F0B', '#3C3489',
                    '#1D9E75', '#A32D2D', '#BA7517', '#534AB7', '#185FA5']

    for col, (panel_name, huc) in enumerate(panels):
        for row, (key, ylabel) in enumerate(metrics):
            ax = axes[row, col]
            vals = collect(key, basin_huc8=huc)
            ax.bar(x, vals, color=[color_cycle[i % len(color_cycle)] for i in range(len(er.members))])
            ax.set_xticks(x)
            ax.set_xticklabels(er.members, rotation=45, ha='right', fontsize=8)
            if key == 'mean_bias_mm':
                ax.axhline(0, color='k', lw=0.6)
            ax.grid(alpha=0.25, linestyle=':', axis='y')
            if col == 0:
                ax.set_ylabel(ylabel, fontsize=10)
            if row == 0:
                ax.set_title(panel_name, fontsize=10.5)
            for xi, v in enumerate(vals):
                if np.isfinite(v):
                    ax.text(xi, v, f"{v:+.1f}" if key == 'mean_bias_mm' else f"{v:.2f}",
                             ha='center', va='bottom' if v >= 0 else 'top',
                             fontsize=7.5)

    plt.suptitle("Ensemble comparison - WRF QPF vs Stage IV", fontsize=12, y=1.00)
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format=fmt, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    er.figures.insert(0, (f"ensemble_comparison.{fmt}", f"image/{fmt}", buf.getvalue()))

    n_members = len(er.members)
    n_basin_maps = len(basins_seen)
    if n_basin_maps > 0:
        fig, axes = plt.subplots(n_basin_maps, n_members,
                                  figsize=(2.8 * n_members, 2.8 * n_basin_maps),
                                  dpi=dpi, squeeze=False)
        for col, m in enumerate(er.members):
            for row, (huc, name) in enumerate(basins_seen):
                ax = axes[row, col]
                bstats = next((b for b in er.per_member_stats[m].get('basins', [])
                                if b['huc8'] == huc), None)
                if bstats:
                    pb = bstats.get('percent_bias') or 0.0
                    mb = bstats.get('mean_bias_mm') or 0.0
                    r = bstats.get('pearson_r') or 0.0
                    ax.text(0.5, 0.62, f"{mb:+.1f} mm",
                             transform=ax.transAxes, ha='center', va='center',
                             fontsize=14, fontweight='bold',
                             color='#A32D2D' if mb > 0 else '#0C447C')
                    ax.text(0.5, 0.38, f"({pb:+.1f}%)  r={r:.2f}",
                             transform=ax.transAxes, ha='center', va='center',
                             fontsize=9, color='#444441')
                ax.set_xticks([]); ax.set_yticks([])
                if row == 0:
                    ax.set_title(m, fontsize=10)
                if col == 0:
                    ax.set_ylabel(name.split(' (')[0], fontsize=10)
                for spine in ax.spines.values():
                    spine.set_edgecolor('#888780')
                    spine.set_linewidth(0.5)
        plt.suptitle("Per-basin bias matrix", fontsize=12, y=1.00)
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format=fmt, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        er.figures.insert(1, (f"ensemble_basin_matrix.{fmt}", f"image/{fmt}", buf.getvalue()))

    header = ["member"]
    header += [f"global_{k}" for k in ("mean_bias_mm", "rmse_mm", "pearson_r")]
    for huc, name in basins_seen:
        nm = name.split(' (')[0].replace(' ', '_').lower()
        header += [f"{nm}_mean_bias_mm", f"{nm}_percent_bias",
                   f"{nm}_rmse_mm", f"{nm}_pearson_r",
                   f"{nm}_csi_50mm", f"{nm}_csi_75mm"]
    rows = [",".join(header)]
    for m in er.members:
        g = er.per_member_stats[m].get('global', {})
        row = [m,
                f"{g.get('mean_bias_mm', 0):+.2f}",
                f"{g.get('rmse_mm', 0):.2f}",
                f"{g.get('pearson_r', 0):.3f}"]
        for huc, _name in basins_seen:
            b = next((bb for bb in er.per_member_stats[m].get('basins', [])
                       if bb['huc8'] == huc), {})
            row += [
                f"{b.get('mean_bias_mm', 0):+.2f}" if b else "",
                f"{b.get('percent_bias', 0):+.2f}" if b and b.get('percent_bias') is not None else "",
                f"{b.get('rmse_mm', 0):.2f}" if b else "",
                f"{b.get('pearson_r', 0):.3f}" if b else "",
                f"{b.get('csi_50mm') or 0:.3f}" if b else "",
                f"{b.get('csi_75mm') or 0:.3f}" if b else "",
            ]
        rows.append(",".join(row))
    er.csv_files.insert(0, ("ensemble_summary.csv", "\n".join(rows).encode("utf-8")))
    return er


def parse_aware_uri(uri: str) -> tuple[str, str, str]:
    if not uri.startswith("aware://"):
        raise ValueError(f"not an aware URI: {uri}")
    rest = uri[len("aware://"):]
    if "@" in rest:
        userhost, path = rest.split("/", 1) if "/" in rest else (rest, "")
        user, host = userhost.split("@")
    else:
        host, path = rest.split("/", 1) if "/" in rest else (rest, "")
        user = os.environ.get("AWARE_USER", "sas042")
    return user, host, "/" + path


def scp_from_aware(uri: str, local_dir: str | Path,
                    ssh_key: str | None = None) -> Path:
    user, host, remote_path = parse_aware_uri(uri)
    local_dir = Path(local_dir); local_dir.mkdir(parents=True, exist_ok=True)
    local_path = local_dir / Path(remote_path).name
    cmd = ["scp"]
    if ssh_key:
        cmd += ["-i", ssh_key]
    cmd += [f"{user}@{host}:{remote_path}", str(local_path)]
    subprocess.run(cmd, check=True)
    return local_path


def resolve_wrfout_path(path_or_uri: str, scratch_dir: str = "/tmp/wxagent_aware") -> str:
    if path_or_uri.startswith("aware://"):
        return str(scp_from_aware(path_or_uri, scratch_dir))
    return path_or_uri


try:
    from celery import Celery, states
    from celery.exceptions import Ignore
    celery_app = Celery(
        "wxagent",
        broker=os.environ.get("CELERY_BROKER", "redis://redis:6379/0"),
        backend=os.environ.get("CELERY_BACKEND", "redis://redis:6379/1"),
    )

    @celery_app.task(bind=True, name="wxagent.qpf_verification.run")
    def run_qpf_verification_task(self, plan: dict, do_synthesize: bool = False,
                                    synthesize_model: str = "claude-sonnet-4-5"):
        try:
            self.update_state(state="PROGRESS", meta={"phase": "resolving_inputs"})
            plan = dict(plan)
            plan["wrfout_path"] = resolve_wrfout_path(plan["wrfout_path"])
            stageiv_dir = plan.get("stageiv_dir", "")
            if stageiv_dir.startswith("aware://"):
                user, host, remote_root = parse_aware_uri(stageiv_dir)
                local_dir = Path("/tmp/wxagent_aware/stageiv")
                local_dir.mkdir(parents=True, exist_ok=True)
                subprocess.run(["rsync", "-az",
                                 f"{user}@{host}:{remote_root}/",
                                 f"{local_dir}/"], check=True)
                plan["stageiv_dir"] = str(local_dir)

            self.update_state(state="PROGRESS", meta={"phase": "computing_verification"})
            result = run(plan)
            findings = build_synthesize_payload(result)

            prose = None
            if do_synthesize:
                self.update_state(state="PROGRESS", meta={"phase": "synthesizing"})
                try:
                    prose = synthesize_report(findings, model=synthesize_model)
                except Exception:
                    prose = synthesize_report_offline(findings)

            artifact_dir = Path(plan.get("save_dir", f"/tmp/wxagent_task_{self.request.id}"))
            artifact_dir.mkdir(parents=True, exist_ok=True)
            for fname, _mime, data in result.figures:
                (artifact_dir / fname).write_bytes(data)
            for fname, data in result.csv_files:
                (artifact_dir / fname).write_bytes(data)
            (artifact_dir / "findings.json").write_text(json.dumps(findings, indent=2, default=str))
            if prose:
                (artifact_dir / "report.md").write_text(prose)

            return {
                "task_id": self.request.id,
                "artifact_dir": str(artifact_dir),
                "figures": [f[0] for f in result.figures],
                "csv_files": [f[0] for f in result.csv_files],
                "findings": findings,
                "report_markdown": prose,
            }
        except Exception as exc:
            self.update_state(state=states.FAILURE,
                               meta={"exc_type": type(exc).__name__, "exc_message": str(exc)})
            raise Ignore()

    @celery_app.task(bind=True, name="wxagent.qpf_verification.run_ensemble")
    def run_ensemble_task(self, members_spec: list[dict], base_plan: dict,
                            comparison_basins: list[str] | None = None,
                            do_synthesize: bool = False,
                            synthesize_model: str = "claude-sonnet-4-5"):
        try:
            members: list[EnsembleMember] = []
            for m in members_spec:
                resolved = resolve_wrfout_path(m["wrfout_path"])
                members.append(EnsembleMember(
                    name=m["name"], wrfout_path=resolved,
                    mp_physics=m.get("mp_physics"), pbl_physics=m.get("pbl_physics"),
                ))
            self.update_state(state="PROGRESS",
                               meta={"phase": "ensemble", "n_members": len(members)})
            er = run_ensemble(members, base_plan, comparison_basins=comparison_basins)

            prose = None
            if do_synthesize:
                aggregate = {"members": er.members,
                              "per_member": {m: er.per_member_stats[m] for m in er.members}}
                try:
                    import anthropic
                    client = anthropic.Anthropic()
                    msg = client.messages.create(
                        model=synthesize_model, max_tokens=2500,
                        system=SYNTHESIZE_SYSTEM_PROMPT + "\nYou are now writing an ENSEMBLE report. "
                                "Identify the best and worst members per basin, by metric.",
                        messages=[{"role": "user",
                                    "content": f"Ensemble findings:\n```json\n"
                                                f"{json.dumps(aggregate, indent=2, default=str)}\n```"}],
                    )
                    prose = "".join(b.text for b in msg.content if b.type == "text")
                except Exception:
                    prose = None

            artifact_dir = Path(base_plan.get("save_dir", f"/tmp/wxagent_ens_{self.request.id}"))
            artifact_dir.mkdir(parents=True, exist_ok=True)
            for fname, _mime, data in er.figures:
                (artifact_dir / fname).write_bytes(data)
            for fname, data in er.csv_files:
                (artifact_dir / fname).write_bytes(data)
            (artifact_dir / "ensemble_findings.json").write_text(
                json.dumps({"members": er.members,
                             "per_member": er.per_member_stats}, indent=2, default=str))
            if prose:
                (artifact_dir / "ensemble_report.md").write_text(prose)

            return {
                "task_id": self.request.id,
                "artifact_dir": str(artifact_dir),
                "members": er.members,
                "figures": [f[0] for f in er.figures],
                "csv_files": [f[0] for f in er.csv_files],
                "report_markdown": prose,
            }
        except Exception as exc:
            self.update_state(state=states.FAILURE,
                               meta={"exc_type": type(exc).__name__, "exc_message": str(exc)})
            raise Ignore()
except ImportError:
    celery_app = None
    run_qpf_verification_task = None
    run_ensemble_task = None


def run_demo():
    members = [
        EnsembleMember(name="ishmael",
                        wrfout_path="/mnt/user-data/uploads/wrfout_d03_ISHMAEL_SUBSET.nc",
                        mp_physics=55, pbl_physics=1),
    ]
    base_plan = {
        'stageiv_dir': '/mnt/user-data/uploads',
        'auto_window_from_wrfout': True,
        'huc8_codes': ['18010110', '18020125'],
        'huc8_dir': None,
        'regrid_method': 'linear',
        'thresholds_mm': [1, 5, 10, 25, 50, 75, 100],
        'elevation_bins_m': [0, 100, 250, 500, 750, 1000, 1250, 1500, 2000, 2500, 3000, 4000],
        'format': 'png', 'dpi': 130,
        'save_dir': '/home/claude/wxagent_ensemble_demo',
        'export_csv': True,
    }
    er = run_ensemble(members, base_plan, comparison_basins=['18010110', '18020125'])
    Path(base_plan['save_dir']).mkdir(parents=True, exist_ok=True)
    for fname, _mime, data in er.figures:
        (Path(base_plan['save_dir']) / fname).write_bytes(data)
    for fname, data in er.csv_files:
        (Path(base_plan['save_dir']) / fname).write_bytes(data)

    findings = er.per_member_stats[er.members[0]]
    prose = synthesize_report_offline(findings)
    (Path(base_plan['save_dir']) / "report_offline.md").write_text(prose)
    print(prose)
    print("\nArtifacts written to:", base_plan['save_dir'])


if __name__ == "__main__":
    run_demo()
