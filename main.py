"""Application entrypoint.

Run from the repo root:
    python main.py
"""

from __future__ import annotations

import sys

from src.utils.logger import get_logger

log = get_logger("super_rugby")


def main() -> int:
    log.info("=== super-rugby pipeline start ===")
    try:
        from src.scraping.bronze import main as scrape_bronze
        from src.etl.silver import main as build_silver
        from src.etl.gold import main as build_gold
        from src.ml.walk_forward import main as run_walkforward

        log.info("Stage: bronze (playfantasyrugby + ultimate rugby scrape)")
        scrape_bronze()
        log.info("Stage: silver (player_rounds + players_dim)")
        build_silver()
        log.info("Stage: gold (feature panel)")
        build_gold()
        log.info("Stage: ml (quantile model + MILP + walk-forward + round-15 optimum)")
        run_walkforward()
        log.info("=== pipeline complete ===")
        return 0
    except Exception:
        log.exception("Pipeline failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
