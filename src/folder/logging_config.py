"""Structured logging setup 
"""

from __future__ import annotations
import logging
import sys
from folder.config import get_settings

_JSON_FORMAT = (
    '{"time":"%(asctime)s","level":"%(levelname)s","logger":"%(name)s","message":"%(message)s"}'
)
_TEXT_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"

_configured = False


def configure_logging() -> None:
    global _configured
    if _configured:
        return

    settings = get_settings()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(_JSON_FORMAT if settings.log_json else _TEXT_FORMAT))

    root = logging.getLogger("RePaAs")
    root.setLevel(settings.log_level.upper())
    root.handlers.clear()
    root.addHandler(handler)
    root.propagate = False

    # Quiet noisy third-party loggers by default.
    for noisy in ("httpx", "httpcore", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    configure_logging()
    return logging.getLogger(f"RePaAs.{name}")
