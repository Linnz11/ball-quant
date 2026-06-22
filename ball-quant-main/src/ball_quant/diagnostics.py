from __future__ import annotations

import time
import urllib.request
from typing import Any, Dict

from ball_quant.adapters.http import HttpError, get_json


def polymarket_doctor(timeout: int = 15) -> Dict[str, Any]:
    proxies = urllib.request.getproxies()
    gamma = timed_json(
        "https://gamma-api.polymarket.com",
        "/public-search",
        {"q": "Germany Curacao", "events_status": "active", "limit_per_type": 3},
        timeout,
    )
    clob = timed_json(
        "https://clob.polymarket.com",
        "/markets",
        {"limit": 1},
        timeout,
    )
    return {"proxies": proxies, "gamma": gamma, "clob": clob}


def timed_json(base_url: str, path: str, params: Dict[str, Any], timeout: int) -> Dict[str, Any]:
    start = time.time()
    try:
        payload = get_json(base_url, path, params=params, timeout=timeout)
    except HttpError as exc:
        return {"ok": False, "seconds": round(time.time() - start, 3), "error": str(exc)}
    return {
        "ok": True,
        "seconds": round(time.time() - start, 3),
        "summary": summarize_payload(payload),
    }


def summarize_payload(payload: Any) -> str:
    if isinstance(payload, list):
        return f"list[{len(payload)}]"
    if isinstance(payload, dict):
        if isinstance(payload.get("events"), list):
            first = payload["events"][0].get("slug") if payload["events"] else "empty"
            return f"events[{len(payload['events'])}] first={first}"
        if isinstance(payload.get("data"), list):
            return f"data[{len(payload['data'])}]"
        return f"dict keys={','.join(sorted(str(k) for k in payload.keys())[:6])}"
    return type(payload).__name__
