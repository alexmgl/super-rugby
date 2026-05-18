"""Project path constants. Anchored to the repo root via __file__."""

from pathlib import Path

# This file lives at src/utils/paths.py — parents[2] is the repo root.
ROOT_DIR: Path = Path(__file__).resolve().parents[2]
DATA_DIR: Path = ROOT_DIR / "data"

# Medallion layers
BRONZE_DIR: Path = DATA_DIR / "bronze"
SILVER_DIR: Path = DATA_DIR / "silver"
GOLD_DIR: Path = DATA_DIR / "gold"

LOGS_DIR: Path = ROOT_DIR / "logs"
