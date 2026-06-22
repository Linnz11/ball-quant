"""
Logging setup for ball-quant.

Call configure_logging() once at the top of main() to establish a consistent
format across all modules.  Module-level loggers are obtained via get_logger().

Format: timestamp  LEVEL  logger.name  message
"""
from __future__ import annotations

import logging
from typing import Optional


_CONFIGURED = False

_FORMAT = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s"
_DATE_FORMAT = "%Y-%m-%dT%H:%M:%S"


def configure_logging(level: str = "INFO") -> None:
    """Configure root logger with a human-readable timestamped format.

    Idempotent — calling twice with the same level is safe; calling with a
    different level updates the root logger level in place.
    """
    global _CONFIGURED
    numeric = getattr(logging, level.upper(), logging.INFO)

    if not _CONFIGURED:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATE_FORMAT))
        logging.root.addHandler(handler)
        _CONFIGURED = True

    logging.root.setLevel(numeric)


def get_logger(name: str) -> logging.Logger:
    """Return a named logger; configure_logging() should be called first."""
    return logging.getLogger(name)
