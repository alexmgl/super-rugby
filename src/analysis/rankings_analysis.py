"""Summarise PlayFantasyRugby per-round + overall leaderboard data.

Run after `python -m src.scraping.fantasy_rankings --rounds all`. Reads the per-round
JSON files in `data/bronze/raw/fantasy_rankings_round*.json` and the overall
CSV, then writes:

  data/gold/league_round_summary.parquet  — round-level stats (mean/median/percentiles)
  data/gold/league_manager_panel.parquet  — long-form (user_id, round, points, rank, ...)

Plus prints headline numbers so you can read patterns directly from the terminal.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from src.utils.logger import get_logger
from src.utils.paths import BRONZE_DIR, GOLD_DIR

log = get_logger(__name__)


def load_per_round() -> pd.DataFrame:
    raw = BRONZE_DIR / "raw"
    rows: list[dict] = []
    for fp in sorted(raw.glob("fantasy_rankings_round*.json")):
        m = re.search(r"round(\d+)\.json$", fp.name)
        if not m:
            continue
        round_id = int(m.group(1))
        data = json.loads(fp.read_text(encoding="utf-8"))
        for r in data:
            r = dict(r)
            r["_round"] = round_id
            rows.append(r)
    return pd.DataFrame(rows)


def main() -> None:
    GOLD_DIR.mkdir(parents=True, exist_ok=True)

    per = load_per_round()
    if per.empty:
        log.warning("no per-round files found in %s", BRONZE_DIR / "raw")
        return

    overall_csv = BRONZE_DIR / "fantasy_rankings_overall.csv"
    overall = pd.read_csv(overall_csv) if overall_csv.exists() else pd.DataFrame()

    # --- round-level summary ---
    rs = per.groupby("_round").agg(
        n_managers=("userId", "nunique"),
        max_pts=("points", "max"),
        mean_pts=("points", "mean"),
        median_pts=("points", "median"),
        p10=("points", lambda s: s.quantile(0.10)),
        p25=("points", lambda s: s.quantile(0.25)),
        p75=("points", lambda s: s.quantile(0.75)),
        p90=("points", lambda s: s.quantile(0.90)),
        p99=("points", lambda s: s.quantile(0.99)),
        min_pts=("points", "min"),
    ).reset_index().rename(columns={"_round": "round"})

    rs.to_parquet(GOLD_DIR / "league_round_summary.parquet", index=False)
    log.info("\n=== per-round leaderboard summary ===")
    log.info("\n%s", rs.round(0).to_string(index=False))

    # --- long-form manager panel ---
    keep = ["userId", "userName", "leagueId", "_round", "rank",
            "points", "overallRank", "overallPoints", "averagePoints"]
    keep = [c for c in keep if c in per.columns]
    panel = per[keep].rename(columns={"_round": "round"})
    panel.to_parquet(GOLD_DIR / "league_manager_panel.parquet", index=False)
    log.info("\nwrote manager panel: %d rows across %d rounds, %d unique managers",
             len(panel), panel["round"].nunique(), panel["userId"].nunique())

    # --- patterns to print directly ---
    if not overall.empty:
        log.info("\n=== leaderboard vs model (R1-R%d) ===", int(panel["round"].max()))
        log.info("overall rank-1   : %s pts", int(overall["overallPoints"].max()))
        log.info("overall rank-100 : %s pts", int(overall.iloc[99].overallPoints))
        log.info("overall rank-1000: %s pts", int(overall.iloc[999].overallPoints))
        log.info("overall rank-2000: %s pts", int(overall["overallPoints"].min()))

    # --- weekly variance among the elite ---
    top_users = panel.groupby("userId")["points"].sum().nlargest(50).index
    elite = panel[panel["userId"].isin(top_users)]
    weekly = elite.groupby("_round" if "_round" in elite.columns else "round")["points"]
    if not elite.empty:
        log.info("\n=== top-50 manager round-to-round variance ===")
        rng = elite.groupby("round")["points"].agg(["mean", "std", "min", "max"]).round(1)
        log.info("\n%s", rng.to_string())


if __name__ == "__main__":
    main()
