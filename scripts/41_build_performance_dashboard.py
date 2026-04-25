#!/usr/bin/env python3
"""Build performance KPI dashboard from workflow profile history."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from plasmid_priority.config import build_context
from plasmid_priority.reporting import ManagedScriptRun
from plasmid_priority.utils.files import atomic_write_json, ensure_directory

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HISTORY = PROJECT_ROOT / "data" / "tmp" / "logs" / "workflow_profile_history.jsonl"


def _read_history(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except ValueError:
            continue
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _mode_summary(rows: list[dict[str, Any]], *, mode: str) -> dict[str, Any]:
    filtered = [row for row in rows if str(row.get("budget_mode", "")) == mode]
    filtered.sort(key=lambda row: str(row.get("generated_at", "")))
    if not filtered:
        return {"mode": mode, "n_runs": 0}
    latest = filtered[-1]
    previous = filtered[-2] if len(filtered) > 1 else None
    latest_runtime = float(latest.get("total_duration_seconds", 0.0))
    if previous is None:
        delta_seconds = None
        delta_pct = None
    else:
        previous_runtime = float(previous.get("total_duration_seconds", 0.0))
        delta_seconds = latest_runtime - previous_runtime
        delta_pct = (delta_seconds / previous_runtime * 100.0) if previous_runtime > 0 else None
    return {
        "mode": mode,
        "n_runs": len(filtered),
        "latest_generated_at": str(latest.get("generated_at", "")),
        "latest_git_commit": str(latest.get("git_commit", "")),
        "latest_total_duration_seconds": latest_runtime,
        "delta_seconds_vs_previous": delta_seconds,
        "delta_pct_vs_previous": delta_pct,
    }


def _render_markdown(summaries: list[dict[str, Any]]) -> str:
    lines = [
        "# Workflow Performance Dashboard",
        "",
        f"Generated at: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        "",
        "| Mode | Runs | Latest Duration (s) | Delta vs Previous (s) | Delta vs Previous (%) | Latest Commit |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for summary in summaries:
        mode = str(summary.get("mode", ""))
        n_runs = int(summary.get("n_runs", 0) or 0)
        latest = summary.get("latest_total_duration_seconds")
        latest_text = f"{float(latest):.2f}" if isinstance(latest, (int, float)) else "-"
        delta_s = summary.get("delta_seconds_vs_previous")
        delta_s_text = f"{float(delta_s):+.2f}" if isinstance(delta_s, (int, float)) else "-"
        delta_p = summary.get("delta_pct_vs_previous")
        delta_p_text = f"{float(delta_p):+.2f}" if isinstance(delta_p, (int, float)) else "-"
        commit = str(summary.get("latest_git_commit", ""))[:12] or "-"
        lines.append(
            f"| {mode} | {n_runs} | {latest_text} | {delta_s_text} | {delta_p_text} | `{commit}` |"
        )
    lines.append("")
    return "\n".join(lines)


def _render_html(summaries: list[dict[str, Any]]) -> str:
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    rows: list[str] = []
    for summary in summaries:
        mode = str(summary.get("mode", ""))
        n_runs = int(summary.get("n_runs", 0) or 0)
        latest = summary.get("latest_total_duration_seconds")
        latest_text = f"{float(latest):.2f}" if isinstance(latest, (int, float)) else "-"
        delta_s = summary.get("delta_seconds_vs_previous")
        delta_s_text = f"{float(delta_s):+.2f}" if isinstance(delta_s, (int, float)) else "-"
        delta_p = summary.get("delta_pct_vs_previous")
        delta_p_text = f"{float(delta_p):+.2f}" if isinstance(delta_p, (int, float)) else "-"
        commit = str(summary.get("latest_git_commit", ""))[:12] or "-"
        rows.append(
            "<tr>"
            f"<td>{mode}</td><td>{n_runs}</td><td>{latest_text}</td>"
            f"<td>{delta_s_text}</td><td>{delta_p_text}</td><td><code>{commit}</code></td>"
            "</tr>"
        )
    return "\n".join(
        [
            "<!doctype html>",
            '<html lang="en"><head><meta charset="utf-8">',
            "<title>Workflow Performance Dashboard</title>",
            "<style>body{font-family:system-ui,sans-serif;margin:24px;}table{border-collapse:collapse;"
            "width:100%;max-width:980px;}th,td{border:1px solid #ddd;padding:8px;text-align:left;}"
            "th{background:#f5f5f5;}code{font-size:0.9em;}</style>",
            "</head><body>",
            "<h1>Workflow Performance Dashboard</h1>",
            f"<p>Generated at: <code>{generated_at}</code></p>",
            "<table><thead><tr><th>Mode</th><th>Runs</th><th>Latest Duration (s)</th>"
            "<th>Delta vs Previous (s)</th><th>Delta vs Previous (%)</th><th>Latest Commit</th></tr>"
            "</thead><tbody>",
            *rows,
            "</tbody></table>",
            "<script>document.querySelectorAll('tbody tr').forEach((row)=>{"
            "const d=parseFloat(row.children[3].textContent);"
            "if(!Number.isNaN(d)&&d>0){row.style.background='#fff7e6';}"
            "});</script>",
            "</body></html>",
        ]
    )


def _render_latex(summaries: list[dict[str, Any]]) -> str:
    rows: list[str] = []
    for summary in summaries:
        mode = str(summary.get("mode", "")).replace("_", "\\_")
        n_runs = int(summary.get("n_runs", 0) or 0)
        latest = summary.get("latest_total_duration_seconds")
        latest_text = f"{float(latest):.2f}" if isinstance(latest, (int, float)) else "-"
        delta_s = summary.get("delta_seconds_vs_previous")
        delta_s_text = f"{float(delta_s):+.2f}" if isinstance(delta_s, (int, float)) else "-"
        rows.append(f"{mode} & {n_runs} & {latest_text} & {delta_s_text} \\\\")
    return "\n".join(
        [
            "\\documentclass{article}",
            "\\usepackage[margin=1in]{geometry}",
            "\\begin{document}",
            "\\section*{Workflow Performance Dashboard}",
            "\\begin{tabular}{lrrr}",
            "Mode & Runs & Latest Duration (s) & Delta vs Previous (s) \\\\",
            "\\hline",
            *rows,
            "\\end{tabular}",
            "\\end{document}",
            "",
        ]
    )


def _compile_pdf(tex_path: Path) -> Path | None:
    pdflatex = shutil.which("pdflatex")
    if pdflatex is None:
        return None
    command = [
        pdflatex,
        "-interaction=nonstopmode",
        "-halt-on-error",
        tex_path.name,
    ]
    completed = subprocess.run(
        command,
        cwd=tex_path.parent,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if int(completed.returncode) != 0:
        return None
    pdf_path = tex_path.with_suffix(".pdf")
    if not pdf_path.exists():
        return None
    return pdf_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", type=Path, default=DEFAULT_HISTORY)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "reports" / "performance",
    )
    parser.add_argument(
        "--render-pdf",
        action="store_true",
        help="Compile LaTeX output to PDF when pdflatex is available.",
    )
    args = parser.parse_args(argv)

    context = build_context(PROJECT_ROOT)
    with ManagedScriptRun(context, "41_build_performance_dashboard") as run:
        if args.history.exists():
            run.record_input(args.history)
        rows = _read_history(args.history)
        output_dir = ensure_directory(args.output_dir)
        json_path = output_dir / "workflow_performance_dashboard.json"
        md_path = output_dir / "workflow_performance_dashboard.md"
        html_path = output_dir / "workflow_performance_dashboard.html"
        tex_path = output_dir / "workflow_performance_dashboard.tex"
        for path in (json_path, md_path, html_path, tex_path):
            run.record_output(path)
        modes = ["smoke-local", "dev-refresh", "model-refresh", "report-refresh", "release-full"]
        summaries = [_mode_summary(rows, mode=mode) for mode in modes]
        dashboard_json = {
            "history_path": str(args.history.resolve()),
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "summaries": summaries,
        }
        atomic_write_json(json_path, dashboard_json)
        md_path.write_text(
            _render_markdown(summaries),
            encoding="utf-8",
        )
        html_path.write_text(
            _render_html(summaries),
            encoding="utf-8",
        )
        tex_path.write_text(_render_latex(summaries), encoding="utf-8")
        if args.render_pdf:
            pdf_path = _compile_pdf(tex_path)
            if pdf_path is not None:
                run.record_output(pdf_path)
        run.set_metric("history_rows", len(rows))
        run.set_metric("mode_count", len(modes))
        if not rows:
            run.note("Workflow history is empty; dashboard rendered with zero-run summaries.")
        print(f"Wrote performance dashboard to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
