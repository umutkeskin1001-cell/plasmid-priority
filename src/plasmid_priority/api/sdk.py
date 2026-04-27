"""Lightweight Python SDK for the Plasmid Priority API."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


class ApiClientError(RuntimeError):
    """Raised when the API request fails."""


@dataclass
class PlasmidPrioritySDK:
    """Small sync SDK covering the stable REST API surface."""

    base_url: str = "http://127.0.0.1:8000"
    timeout_seconds: float = 30.0
    api_key: str | None = None

    def _request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        url = f"{self.base_url.rstrip('/')}{path}"
        body: bytes | None = None
        headers = {"Accept": "application/json"}
        if self.api_key:
            headers["x-api-key"] = str(self.api_key)
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

    def config(self) -> dict[str, Any]:
        return self._request("GET", "/config")

    def models(self) -> dict[str, Any]:
        return self._request("GET", "/models")

    def score_backbones(
        self,
        backbone_ids: list[str],
        *,
        config_key: str = "geo_spread",
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            "/score/backbones",
            {"backbone_ids": list(backbone_ids), "config_key": config_key},
        )

    def explain_backbone(self, backbone_id: str) -> dict[str, Any]:
        return self._request("GET", f"/explain/{backbone_id}")

    def get_evidence(self, backbone_id: str) -> dict[str, Any]:
        return self._request("GET", f"/evidence/{backbone_id}")
