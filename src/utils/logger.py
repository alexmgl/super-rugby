"""Application-wide logger.

Usage:
    from src.utils.logger import get_logger
    log = get_logger(__name__)
    log.info("hello")

Configuration:
    LOG_LEVEL  env var overrides the default (INFO).
    Logs stream to stdout and to <repo>/logs/app.log (rotated at 10 MB, 5 backups).
"""

from __future__ import annotations

import logging
import os
import sys
from logging.handlers import RotatingFileHandler

from src.utils.paths import LOGS_DIR

_FMT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"

_configured = False


def _configure_root() -> None:
    global _configured
    if _configured:
        return

    level = os.environ.get("LOG_LEVEL", "INFO").upper()
    formatter = logging.Formatter(_FMT, _DATEFMT)

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    file_handler = RotatingFileHandler(
        LOGS_DIR / "app.log",
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()
    root.addHandler(console)
    root.addHandler(file_handler)

    _configured = True


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a configured logger. Safe to call repeatedly — handlers attach once."""
    _configure_root()
    return logging.getLogger(name)
