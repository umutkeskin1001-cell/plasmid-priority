"""Minimal FastAPI REST API for Plasmid Priority.

Provides read-only endpoints for:
- ``GET /health`` – liveness check
- ``GET /config`` – project configuration summary
- ``POST /score`` – compute priority scores for a backbone table

Run with::

    uvicorn plasmid_priority.api.app:app --reload

Requires the ``api`` extra dependency group.
"""

from __future__ import annotations

import logging
from typing import Any

_log = logging.getLogger(__name__)

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel

    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

    # Stubs so the module can be imported without FastAPI installed
    class FastAPI:  # type: ignore[no-redef]
        def get(self, _path: str, **_kw: Any) -> Any:
            return lambda f: f

        def post(self, _path: str, **_kw: Any) -> Any:
            return lambda f: f

    class BaseModel:  # type: ignore[no-redef]
        pass

    class HTTPException(Exception):  # type: ignore[no-redef]
        pass


app = FastAPI(title="Plasmid Priority API", version="0.3.0") if FASTAPI_AVAILABLE else FastAPI()


class ScoreRequest(BaseModel):
    """Request body for the ``/score`` endpoint."""

    backbone_ids: list[str]
    config_key: str = "geo_spread"


class ScoreResponse(BaseModel):
    """Response body for the ``/score`` endpoint."""

    scores: list[dict[str, Any]]
    status: str = "ok"


@app.get("/health")
def health_check() -> dict[str, str]:
    """Liveness probe."""
    return {"status": "alive"}


@app.get("/config")
def get_config_summary() -> dict[str, Any]:
    """Return project configuration summary."""
    if not FASTAPI_AVAILABLE:
        return {"status": "error", "reason": "fastapi_not_installed"}
    from plasmid_priority.config import build_context

    ctx = build_context()
    return {"config_keys": list(ctx.config.keys()) if isinstance(ctx.config, dict) else []}


@app.post("/score", response_model=ScoreResponse)
def compute_scores(request: ScoreRequest) -> ScoreResponse:
    """Compute priority scores for the given backbone IDs.

    This is a placeholder that returns empty scores until the scoring
    pipeline is wired in.
    """
    if not FASTAPI_AVAILABLE:
        return ScoreResponse(scores=[], status="error: fastapi not installed")
    _log.info("Score request for %d backbones (%s)", len(request.backbone_ids), request.config_key)
    return ScoreResponse(
        scores=[{"backbone_id": bid, "priority_index": 0.0} for bid in request.backbone_ids],
        status="ok",
    )
