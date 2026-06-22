"""
Settings — centralised runtime configuration for ball-quant.

Merge order (later wins):
  code defaults <- optional JSON file <- environment variables (prefix BALLQ_)

Environment variable mapping (all uppercase after prefix):
  BALLQ_STORE_ROOT         -> store_root
  BALLQ_CACHE_DIR          -> cache_dir
  BALLQ_REPORTS_DIR        -> reports_dir
  BALLQ_LIVE_REPORTS_DIR   -> live_reports_dir
  BALLQ_DEFAULT_BUDGET     -> default_budget   (float)
  BALLQ_DEFAULT_BANKROLL   -> default_bankroll (float)
  BALLQ_TIMEZONE           -> timezone
  BALLQ_WORLD_CUP_TAG_ID   -> world_cup_tag_id (int)
  BALLQ_HTTP_TIMEOUT       -> http_timeout     (int)
  BALLQ_LOG_LEVEL          -> log_level
  BALLQ_CACHE_TTL_SECONDS  -> cache_ttl_seconds (int)
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


# Canonical mapping: JSON / dataclass field name -> env var suffix (after BALLQ_)
_FIELD_ENV: Dict[str, str] = {
    "store_root": "STORE_ROOT",
    "cache_dir": "CACHE_DIR",
    "reports_dir": "REPORTS_DIR",
    "live_reports_dir": "LIVE_REPORTS_DIR",
    "default_budget": "DEFAULT_BUDGET",
    "default_bankroll": "DEFAULT_BANKROLL",
    "timezone": "TIMEZONE",
    "world_cup_tag_id": "WORLD_CUP_TAG_ID",
    "http_timeout": "HTTP_TIMEOUT",
    "log_level": "LOG_LEVEL",
    "cache_ttl_seconds": "CACHE_TTL_SECONDS",
}

# Type coercions for fields that are not str.
_FIELD_TYPES: Dict[str, type] = {
    "default_budget": float,
    "default_bankroll": float,
    "world_cup_tag_id": int,
    "http_timeout": int,
    "cache_ttl_seconds": int,
}


@dataclass
class Settings:
    store_root: str = "data/store"
    cache_dir: str = "data/cache"
    reports_dir: str = "reports"
    live_reports_dir: str = "reports/live"
    default_budget: float = 200.0
    default_bankroll: float = 1000.0
    timezone: str = "Asia/Shanghai"
    world_cup_tag_id: int = 102232
    http_timeout: int = 12
    log_level: str = "INFO"
    cache_ttl_seconds: int = 3600

    # -------------------------------------------------------------------------
    # Factory
    # -------------------------------------------------------------------------

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "Settings":
        """Build Settings by merging defaults <- JSON file <- env vars.

        Unknown keys in the JSON file raise ValueError immediately so callers
        never silently proceed with a mis-typed config.
        """
        known_fields = set(_FIELD_ENV.keys())

        # Start from code defaults.
        values: Dict[str, Any] = {f: getattr(cls(), f) for f in known_fields}

        # Layer 1: optional JSON file.
        if path is not None:
            raw = json.loads(Path(path).read_text(encoding="utf-8"))
            unknown = set(raw.keys()) - known_fields
            if unknown:
                raise ValueError(
                    f"Unknown key(s) in settings file {path}: {sorted(unknown)}. "
                    f"Known fields: {sorted(known_fields)}"
                )
            for k, v in raw.items():
                # Coerce to the expected type if specified.
                if k in _FIELD_TYPES:
                    values[k] = _FIELD_TYPES[k](v)
                else:
                    values[k] = v

        # Layer 2: environment variables override everything.
        for field_name, env_suffix in _FIELD_ENV.items():
            env_key = f"BALLQ_{env_suffix}"
            raw_val = os.environ.get(env_key)
            if raw_val is not None:
                if field_name in _FIELD_TYPES:
                    values[field_name] = _FIELD_TYPES[field_name](raw_val)
                else:
                    values[field_name] = raw_val

        return cls(**values)

    # -------------------------------------------------------------------------
    # Serialisation
    # -------------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Return a plain JSON-serialisable dict of all settings."""
        return {f: getattr(self, f) for f in _FIELD_ENV}
