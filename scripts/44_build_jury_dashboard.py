#!/usr/bin/env python3
"""Build jury-facing interactive HTML dashboard with local RAG snippets."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from plasmid_priority.reporting.rag import load_rag_chunks, retrieve_rag
from plasmid_priority.utils.files import atomic_write_json, ensure_directory

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports" / "release"


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _resolve_corpus_paths(project_root: Path) -> list[Path]:
    cfg_path = project_root / "config" / "rag_corpus.yaml"
    try:
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        cfg = {}
    raw_paths = cfg.get("paths", []) if isinstance(cfg, dict) else []
    paths: list[Path] = []
    for item in raw_paths:
        rel = Path(str(item))
        target = rel if rel.is_absolute() else (project_root / rel)
        if target.exists() and target.is_file():
            paths.append(target.resolve())
    return paths


def _render_html(payload: dict[str, Any]) -> str:
    qa_rows: list[str] = []
    for qa in payload.get("qa", []):
        question = str(qa.get("question", ""))
        hits = qa.get("hits", [])
        if not isinstance(hits, list):
            continue
        li_items: list[str] = []
        for hit in hits:
            if not isinstance(hit, dict):
                continue
            title = str(hit.get("title", ""))
            score = hit.get("score", "-")
            snippet = str(hit.get("snippet", ""))
            li_items.append(
                f"<li><strong>{title}</strong> <em>(score={score})</em><br>{snippet}</li>"
            )
        qa_rows.append(f"<h3>{question}</h3><ul>{''.join(li_items) or '<li>No matches</li>'}</ul>")
    return "\n".join(
        [
            "<!doctype html>",
            '<html lang="en"><head><meta charset="utf-8"><title>Jury Dashboard</title>',
            "<style>body{font-family:system-ui,sans-serif;margin:24px;max-width:1100px;}code{font-size:0.9em;}"
            ".grid{display:grid;grid-template-columns:1fr 1fr;gap:16px;} .card{border:1px solid #ddd;"
            "padding:12px;border-radius:10px;background:#fff;} ul{padding-left:20px;} </style>",
            "</head><body>",
            "<h1>Jury Dashboard</h1>",
            f"<p>Generated at: <code>{payload.get('generated_at','')}</code></p>",
            "<div class='grid'>",
            "<div class='card'><h2>Release Readiness</h2>"
            f"<pre>{json.dumps(payload.get('release_readiness', {}), indent=2)}</pre></div>",
            "<div class='card'><h2>Performance Summary</h2>"
            f"<pre>{json.dumps(payload.get('performance_summary', {}), indent=2)}</pre></div>",
            "</div>",
            "<div class='card'><h2>RAG Q&A Evidence</h2>",
            *qa_rows,
            "</div>",
            "</body></html>",
        ]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)

    project_root = args.project_root.resolve()
    output_dir = ensure_directory(args.output_dir)

    release_readiness = _read_json(output_dir / "release_readiness_report.json")
    performance_summary = _read_json(
        project_root / "reports" / "performance" / "workflow_performance_dashboard.json"
    )
    rag_paths = _resolve_corpus_paths(project_root)
    rag_chunks = load_rag_chunks(rag_paths)

    questions = [
        "What evidence supports external validation and claim levels?",
        "What are the strongest release readiness checks and failures?",
        "What are the key reproducibility and protocol guarantees?",
    ]
    qa_payload = [
        {"question": question, "hits": retrieve_rag(rag_chunks, question, top_k=4)}
        for question in questions
    ]
    dashboard_payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "release_readiness": release_readiness,
        "performance_summary": performance_summary,
        "rag_corpus_files": [str(path) for path in rag_paths],
        "qa": qa_payload,
    }

    atomic_write_json(output_dir / "jury_dashboard.json", dashboard_payload)
    (output_dir / "jury_dashboard.html").write_text(
        _render_html(dashboard_payload),
        encoding="utf-8",
    )
    print(f"Wrote jury dashboard to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
