"""Minimal FastAPI REST API for Plasmid Priority.

Provides read-only endpoints for:
- ``GET /health`` – liveness check
- ``GET /config`` – project configuration summary
- ``GET /models`` – artifact-backed model metadata
- ``POST /score`` – compute priority scores for a backbone table
- ``POST /score/backbones`` – artifact-backed backbone scoring
- ``GET /explain/{backbone_id}`` – fetch score component explanation
- ``GET /evidence/{backbone_id}`` – fetch evidence metadata for a backbone

Run with::

    uvicorn plasmid_priority.api.app:app --reload

Requires the ``api`` extra dependency group.
"""

from __future__ import annotations

import hmac
import logging
import time
from collections import deque
from threading import Lock
from typing import Any, Callable, Literal, cast

from plasmid_priority.settings import load_api_settings

_log = logging.getLogger(__name__)

try:
    from fastapi import Depends, FastAPI, HTTPException, Request
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

    class Request:  # type: ignore[no-redef]
        headers: dict[str, str]
        method: str
        client: Any
        url: Any

        async def body(self) -> bytes:
            return b""

    class HTTPException(Exception):  # type: ignore[no-redef]
        pass

    def Depends(
        dependency: Callable[..., Any] | None = None,
        *,
        use_cache: bool = True,
        scope: Literal["function", "request"] | None = None,
    ) -> Any:
        _ = dependency, use_cache, scope
        return None


app = FastAPI(title="Plasmid Priority API", version="0.3.0") if FASTAPI_AVAILABLE else FastAPI()
_REQUEST_BUCKETS: dict[str, deque[float]] = {}
_REQUEST_BUCKETS_LOCK = Lock()


def _fallback_score(backbone_id: str) -> dict[str, Any]:
    return {"backbone_id": str(backbone_id), "priority_index": 0.0}


def _normalize_score_row(backbone_id: str, row: dict[str, Any]) -> dict[str, object]:
    try:
        priority_index = float(row.get("priority_index", 0.0))
    except (TypeError, ValueError):
        priority_index = 0.0
    normalized: dict[str, object] = {
        "backbone_id": str(backbone_id),
        "priority_index": priority_index,
    }
    for key in (
        "operational_priority_index",
        "bio_priority_index",
        "evidence_support_index",
    ):
        if key not in row:
            continue
        try:
            normalized[key] = float(row.get(key, 0.0))
        except (TypeError, ValueError):
            normalized[key] = 0.0
    return normalized


def _build_artifact_registry() -> Any | None:
    try:
        from plasmid_priority.api.artifact_registry import ArtifactRegistry
    except ImportError:
        return None
    try:
        return ArtifactRegistry()
    except Exception as exc:  # pragma: no cover - defensive fallback
        import logging
        logging.getLogger(__name__).warning("Caught suppressed exception: %s", exc, exc_info=True)
        _log.exception("Failed to initialize ArtifactRegistry")
        return None


def _is_artifact_unavailable_error(exc: Exception) -> bool:
    try:
        from plasmid_priority.api.artifact_registry import ArtifactUnavailableError
    except ImportError:
        return False
    return isinstance(exc, ArtifactUnavailableError)


def _require_artifact_registry() -> Any:
    registry = _build_artifact_registry()
    if registry is None:
        raise HTTPException(status_code=503, detail="Artifact registry unavailable")
    return registry


def _extract_api_key(request: Request) -> str:
    x_api_key = str(request.headers.get("x-api-key", "")).strip()
    if x_api_key:
        return x_api_key
    authorization = str(request.headers.get("authorization", ""))
    bearer_prefix = "Bearer "
    if authorization.startswith(bearer_prefix):
        return authorization[len(bearer_prefix) :].strip()
    return ""


def _enforce_rate_limit(request: Request, limit_per_minute: int) -> None:
    if limit_per_minute <= 0:
        return
    client_host = "unknown"
    if getattr(request, "client", None) is not None:
        client_host = str(getattr(request.client, "host", "unknown"))
    path = str(getattr(getattr(request, "url", None), "path", "unknown"))
    bucket_key = f"{client_host}:{path}"
    now = time.monotonic()
    window_start = now - 60.0
    with _REQUEST_BUCKETS_LOCK:
        bucket = _REQUEST_BUCKETS.setdefault(bucket_key, deque())
        while bucket and bucket[0] <= window_start:
            bucket.popleft()
        if len(bucket) >= limit_per_minute:
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded: max {limit_per_minute} requests/minute",
            )
        bucket.append(now)


async def _enforce_request_size(request: Request, max_bytes: int) -> None:
    if max_bytes <= 0:
        return
    if str(getattr(request, "method", "GET")).upper() not in {"POST", "PUT", "PATCH"}:
        return
    content_length = request.headers.get("content-length")
    if content_length:
        try:
            declared_size = int(content_length)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="Invalid Content-Length header") from exc
        if declared_size > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"Request payload too large (max {max_bytes} bytes)",
            )
    body = await request.body()
    if len(body) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"Request payload too large (max {max_bytes} bytes)",
        )


async def _protect_api_surface(request: Request) -> None:
    settings = load_api_settings()
    if not settings.api_key:
        raise HTTPException(
            status_code=500,
            detail="API key not configured on server",
        )
    if not hmac.compare_digest(_extract_api_key(request), settings.api_key):
        raise HTTPException(
            status_code=401,
            detail="Missing or invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    await _enforce_request_size(request, settings.max_request_bytes)
    _enforce_rate_limit(request, settings.rate_limit_per_minute)


class ModelRegistry:
    """Model registry for API operations."""

    @staticmethod
    def list_models() -> list[dict[str, Any]]:
        """List available models."""
        registry = _build_artifact_registry()
        if registry is None:
            return []
        try:
            models = registry.list_models()
        except Exception as exc:  # pragma: no cover - defensive fallback
            if not _is_artifact_unavailable_error(exc):
                _log.exception("Unexpected error while listing artifact-backed models")
            return []
        return [row for row in models if isinstance(row, dict)]

    @staticmethod
    def score_backbones(backbone_ids: list[str]) -> list[dict[str, object]]:
        """Score backbones and return priorities."""
        if not backbone_ids:
            return []
        registry = _require_artifact_registry()
        try:
            artifact_scores = registry.score_backbones(backbone_ids)
        except Exception as exc:
            if _is_artifact_unavailable_error(exc):
                raise HTTPException(
                    status_code=503,
                    detail="Scoring artifacts unavailable",
                ) from exc
            _log.exception("Unexpected error while scoring artifact-backed backbones")
            raise HTTPException(status_code=500, detail="Failed to score backbones") from exc
        score_map: dict[str, dict[str, Any]] = {}
        for row in artifact_scores:
            if not isinstance(row, dict):
                continue
            score_map[str(row.get("backbone_id", ""))] = row
        merged_scores: list[dict[str, object]] = []
        for backbone_id in backbone_ids:
            row = score_map.get(str(backbone_id))
            if row is None:
                merged_scores.append(_fallback_score(backbone_id))
                continue
            merged_scores.append(_normalize_score_row(backbone_id, row))
        return merged_scores


REGISTRY = ModelRegistry()


def _artifact_payload(
    backbone_id: str,
    *,
    operation_name: str,
    missing_detail: str,
    unavailable_detail: str,
    failure_detail: str,
) -> dict[str, Any]:
    registry = _require_artifact_registry()
    operation = getattr(registry, operation_name)
    try:
        return cast(dict[str, Any], operation(backbone_id))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=missing_detail) from exc
    except Exception as exc:
        if _is_artifact_unavailable_error(exc):
            raise HTTPException(status_code=503, detail=unavailable_detail) from exc
        _log.exception(
            "Unexpected error while running %s for backbone %s",
            operation_name,
            backbone_id,
        )
        raise HTTPException(status_code=500, detail=failure_detail) from exc


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


@app.get("/models", dependencies=[Depends(_protect_api_surface)])
def list_models() -> dict[str, Any]:
    """Return artifact-backed model metadata."""
    if not FASTAPI_AVAILABLE:
        return {"status": "error", "reason": "fastapi_not_installed"}
    return {"status": "ok", "models": REGISTRY.list_models()}


@app.post("/score", response_model=ScoreResponse, dependencies=[Depends(_protect_api_surface)])
def compute_scores(request: ScoreRequest) -> ScoreResponse:
    """Compute priority scores for the given backbone IDs.

    Fail closed when scoring artifacts are unavailable.
    Zero-fill unresolved backbone IDs when artifact scoring succeeds.
    """
    if not FASTAPI_AVAILABLE:
        return ScoreResponse(scores=[], status="error: fastapi not installed")
    _log.info("Score request for %d backbones (%s)", len(request.backbone_ids), request.config_key)
    scores = REGISTRY.score_backbones(request.backbone_ids)
    return ScoreResponse(
        scores=scores,
        status="ok",
    )


@app.post(
    "/score/backbones",
    response_model=ScoreResponse,
    dependencies=[Depends(_protect_api_surface)],
)
def score_backbones(request: ScoreRequest) -> ScoreResponse:
    """Artifact-backed scoring endpoint for product API clients."""
    return compute_scores(request)


@app.get("/explain/{backbone_id}", dependencies=[Depends(_protect_api_surface)])
def explain_backbone(backbone_id: str) -> dict[str, Any]:
    """Return score component explanation for a backbone."""
    if not FASTAPI_AVAILABLE:
        return {"status": "error", "reason": "fastapi_not_installed"}
    return _artifact_payload(
        backbone_id,
        operation_name="explain_backbone",
        missing_detail=f"Backbone not found: {backbone_id}",
        unavailable_detail="Scoring artifacts unavailable",
        failure_detail="Failed to explain backbone",
    )


@app.get("/evidence/{backbone_id}", dependencies=[Depends(_protect_api_surface)])
def get_backbone_evidence(backbone_id: str) -> dict[str, Any]:
    """Return evidence payload for a backbone."""
    if not FASTAPI_AVAILABLE:
        return {"status": "error", "reason": "fastapi_not_installed"}
    return _artifact_payload(
        backbone_id,
        operation_name="get_evidence",
        missing_detail=f"Backbone not found: {backbone_id}",
        unavailable_detail="Scoring artifacts unavailable",
        failure_detail="Failed to fetch backbone evidence",
    )
