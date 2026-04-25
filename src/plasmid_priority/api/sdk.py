"""Lightweight Python SDK for the Plasmid Priority API."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


class ApiClientError(RuntimeError):
    """Raised when the API request fails."""


@dataclass
class PlasmidPrioritySDK:
    """Small sync SDK covering REST, async batch, and GraphQL endpoints."""

    base_url: str = "http://127.0.0.1:8000"
    timeout_seconds: float = 30.0

    def _request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        url = f"{self.base_url.rstrip('/')}{path}"
        body: bytes | None = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            body = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(url, method=method, data=body, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
                raw = resp.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            raise ApiClientError(f"API HTTP error {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise ApiClientError(f"API request failed: {exc.reason}") from exc
        try:
            parsed = json.loads(raw)
        except ValueError as exc:
            raise ApiClientError("API response is not valid JSON.") from exc
        if not isinstance(parsed, dict):
            raise ApiClientError("API response must be a JSON object.")
        return parsed

    def health(self) -> dict[str, Any]:
        return self._request("GET", "/health")

    def models(self) -> dict[str, Any]:
        return self._request("GET", "/models")

    def score_backbones(self, backbone_ids: list[str]) -> dict[str, Any]:
        return self._request(
            "POST",
            "/score/backbones",
            {"backbone_ids": list(backbone_ids)},
        )

    def start_score_batch(self, backbone_ids: list[str]) -> dict[str, Any]:
        return self._request(
            "POST",
            "/score/backbones/batch",
            {"backbone_ids": list(backbone_ids)},
        )

    def batch_status(self, job_id: str) -> dict[str, Any]:
        return self._request("GET", f"/score/backbones/batch/{job_id}")

    def wait_for_batch(
        self,
        job_id: str,
        *,
        timeout_seconds: float = 60.0,
        poll_interval_seconds: float = 0.2,
    ) -> dict[str, Any]:
        deadline = time.time() + float(timeout_seconds)
        while time.time() <= deadline:
            payload = self.batch_status(job_id)
            status = str(payload.get("status", "")).lower()
            if status in {"completed", "failed"}:
                return payload
            time.sleep(max(0.01, float(poll_interval_seconds)))
        raise ApiClientError(f"Batch job timed out: {job_id}")

    def graphql(self, query: str, variables: dict[str, Any] | None = None) -> dict[str, Any]:
        return self._request(
            "POST",
            "/graphql",
            {"query": query, "variables": variables or {}},
        )
