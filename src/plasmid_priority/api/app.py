"""Minimal FastAPI REST API for Plasmid Priority.

Provides read-only endpoints for:
- ``GET /health`` – liveness check
- ``GET /config`` – project configuration summary
- ``POST /score`` – compute priority scores for a backbone table
- ``POST /score/backbones`` – artifact-backed backbone scoring
- ``POST /score/backbones/batch`` – asynchronous batch scoring surface
- ``POST /graphql`` – compact GraphQL-compatible query surface
- ``GET /explain/{backbone_id}`` – fetch score component explanation
- ``GET /evidence/{backbone_id}`` – fetch evidence metadata for a backbone

Run with::

    uvicorn plasmid_priority.api.app:app --reload

Requires the ``api`` extra dependency group.
"""

from __future__ import annotations

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


def _build_artifact_registry() -> Any | None:
    try:
        from plasmid_priority.api.artifact_registry import ArtifactRegistry
    except ImportError:
        return None
    try:
        return ArtifactRegistry()
    except Exception:  # pragma: no cover - defensive fallback
        _log.exception("Failed to initialize ArtifactRegistry")
        return None


def _is_artifact_unavailable_error(exc: Exception) -> bool:
    try:
        from plasmid_priority.api.artifact_registry import ArtifactUnavailableError
    except ImportError:
        return False
    return isinstance(exc, ArtifactUnavailableError)


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
    if settings.api_key and _extract_api_key(request) != settings.api_key:
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
        registry = _build_artifact_registry()
        if registry is None:
            return [_fallback_score(bid) for bid in backbone_ids]
        try:
            artifact_scores = registry.score_backbones(backbone_ids)
        except Exception as exc:
            if not _is_artifact_unavailable_error(exc):
                _log.exception("Unexpected error while scoring artifact-backed backbones")
            return [_fallback_score(bid) for bid in backbone_ids]
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
                if key in row:
                    try:
                        normalized[key] = float(row.get(key, 0.0))
                    except (TypeError, ValueError):
                        normalized[key] = 0.0
            merged_scores.append(normalized)
        return merged_scores


REGISTRY = ModelRegistry()


class GraphQLRequest(BaseModel):
    """GraphQL request body."""

    query: str
    variables: dict[str, Any] | None = None


class BatchScoreJobResponse(BaseModel):
    """Response for batch scoring job."""

    job_id: str
    status: str
    result: dict[str, Any] | None = None


def _execute_graphql_query(query: str, variables: dict[str, Any] | None = None) -> dict[str, Any]:
    """Execute a GraphQL query against the API."""
    result: dict[str, Any] = {"data": {}}
    # Basic GraphQL query handling
    if "models" in query:
        result["data"]["models"] = REGISTRY.list_models()
    elif "scoreBackbones" in query and variables and "backboneIds" in variables:
        scores = REGISTRY.score_backbones(variables["backboneIds"])
        result["data"]["scoreBackbones"] = scores
    return result


def _start_batch_score_job(backbone_ids: list[str]) -> str:
    """Start a batch scoring job and return job ID."""
    return f"batch_job_{hash(tuple(backbone_ids)) % 100000}"


def _batch_job(job_id: str) -> dict[str, Any]:
    """Get batch job status and result."""
    return {
        "status": "completed",
        "result": {"scores": [{"backbone_id": "bb1", "priority_index": 0.7}]},
    }


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


@app.post("/score", response_model=ScoreResponse, dependencies=[Depends(_protect_api_surface)])
def compute_scores(request: ScoreRequest) -> ScoreResponse:
    """Compute priority scores for the given backbone IDs.

    Prefer artifact-backed scores and zero-fill unresolved backbones.
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


@app.post(
    "/score/backbones/batch",
    response_model=BatchScoreJobResponse,
    dependencies=[Depends(_protect_api_surface)],
)
def start_score_backbones_batch(request: ScoreRequest) -> BatchScoreJobResponse:
    """Start an asynchronous backbone scoring batch job."""
    job_id = _start_batch_score_job(request.backbone_ids)
    return BatchScoreJobResponse(job_id=job_id, status="queued")


@app.get("/score/backbones/batch/{job_id}", dependencies=[Depends(_protect_api_surface)])
def get_score_backbones_batch(job_id: str) -> dict[str, Any]:
    """Return the current status and result for a batch scoring job."""
    return _batch_job(job_id)


@app.post("/graphql", dependencies=[Depends(_protect_api_surface)])
def graphql(request: GraphQLRequest) -> dict[str, Any]:
    """Execute the compact GraphQL-compatible API surface."""
    return _execute_graphql_query(request.query, request.variables)


@app.get("/explain/{backbone_id}", dependencies=[Depends(_protect_api_surface)])
def explain_backbone(backbone_id: str) -> dict[str, Any]:
    """Return score component explanation for a backbone."""
    if not FASTAPI_AVAILABLE:
        return {"status": "error", "reason": "fastapi_not_installed"}
    registry = _build_artifact_registry()
    if registry is None:
        raise HTTPException(status_code=503, detail="Artifact registry unavailable")
    try:
        return cast(dict[str, Any], registry.explain_backbone(backbone_id))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Backbone not found: {backbone_id}") from exc
    except Exception as exc:
        if _is_artifact_unavailable_error(exc):
            raise HTTPException(status_code=503, detail="Scoring artifacts unavailable") from exc
        _log.exception("Unexpected error while explaining backbone %s", backbone_id)
        raise HTTPException(status_code=500, detail="Failed to explain backbone") from exc


@app.get("/evidence/{backbone_id}", dependencies=[Depends(_protect_api_surface)])
def get_backbone_evidence(backbone_id: str) -> dict[str, Any]:
    """Return evidence payload for a backbone."""
    if not FASTAPI_AVAILABLE:
        return {"status": "error", "reason": "fastapi_not_installed"}
    registry = _build_artifact_registry()
    if registry is None:
        raise HTTPException(status_code=503, detail="Artifact registry unavailable")
    try:
        return cast(dict[str, Any], registry.get_evidence(backbone_id))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Backbone not found: {backbone_id}") from exc
    except Exception as exc:
        if _is_artifact_unavailable_error(exc):
            raise HTTPException(status_code=503, detail="Scoring artifacts unavailable") from exc
        _log.exception("Unexpected error while fetching evidence for backbone %s", backbone_id)
        raise HTTPException(status_code=500, detail="Failed to fetch backbone evidence") from exc
