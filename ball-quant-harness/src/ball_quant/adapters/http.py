from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional
from urllib import parse, request

_logger = logging.getLogger(__name__)


class HttpError(RuntimeError):
    pass


def get_json(
    base_url: str,
    path: str,
    params: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, str]] = None,
    timeout: int = 12,
    cache_path: Optional[Path] = None,
) -> Any:
    if cache_path and cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))

    url = build_url(base_url, path, params)
    request_headers = {"User-Agent": "ball-quant/0.1"}
    request_headers.update(headers or {})
    req = request.Request(url, headers=request_headers)
    try:
        with request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
    except Exception as exc:  # pragma: no cover - network varies by environment.
        # Log before re-raising so ops dashboards can correlate URL + error without
        # needing to reconstruct the request from the exception chain.
        _logger.warning("HTTP request failed: url=%s error=%r", url, exc)
        raise HttpError(f"GET {url} failed: {exc}") from exc

    try:
        payload = json.loads(body)
    except json.JSONDecodeError as exc:
        _logger.warning("HTTP non-JSON response: url=%s", url)
        raise HttpError(f"GET {url} returned non-JSON response") from exc

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    time.sleep(0.05)
    return payload


def get_text(
    base_url: str,
    path: str,
    params: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, str]] = None,
    timeout: int = 12,
) -> str:
    url = build_url(base_url, path, params)
    request_headers = {"User-Agent": "ball-quant/0.1"}
    request_headers.update(headers or {})
    req = request.Request(url, headers=request_headers)
    try:
        with request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
    except Exception as exc:  # pragma: no cover - network varies by environment.
        raise HttpError(f"GET {url} failed: {exc}") from exc
    time.sleep(0.05)
    return body


def build_url(base_url: str, path: str, params: Optional[Dict[str, Any]] = None) -> str:
    if not path.startswith("/"):
        path = "/" + path
    query = ""
    if params:
        clean = {key: value for key, value in params.items() if value is not None}
        query = "?" + parse.urlencode(clean, doseq=True) if clean else ""
    return base_url.rstrip("/") + path + query
