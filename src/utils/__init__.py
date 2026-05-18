"""Project utilities: logger, paths, decorators, IO helpers."""

from src.utils.logger import get_logger
from src.utils.paths import (
    BRONZE_DIR,
    DATA_DIR,
    GOLD_DIR,
    LOGS_DIR,
    ROOT_DIR,
    SILVER_DIR,
)

__all__ = [
    "get_logger",
    "ROOT_DIR",
    "DATA_DIR",
    "BRONZE_DIR",
    "SILVER_DIR",
    "GOLD_DIR",
    "LOGS_DIR",
]
