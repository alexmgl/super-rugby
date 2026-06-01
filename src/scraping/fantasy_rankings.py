"""Scrape PlayFantasyRugby manager rankings (overall + per-round).

The leaderboard endpoint is auth-gated:
    GET /api/en/fantasy/ranking?page=N           -> overall top 2000 (20 per page, 100 pages)
    GET /api/en/fantasy/ranking?page=N&round=R   -> per-round top 2000

PFR caps each leaderboard at 2,000 entries. Total active managers is larger
(>14k inferred from `rank` field in latest-round records).

Writes:
  data/bronze/raw/fantasy_rankings_overall.json
  data/bronze/raw/fantasy_rankings_round{R}.json
  data/bronze/fantasy_rankings_overall.csv
  data/bronze/fantasy_rankings_per_round.csv     # long form: one row per (user, round)
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from src.utils.logger import get_logger
from src.utils.paths import BRONZE_DIR
from src.utils.pfr_auth import login

log = get_logger(__name__)

RANKING_URL = "https://www.playfantasyrugby.com/api/en/fantasy/ranking"
MAX_PAGES = 100  # PFR caps at 2000 entries = 100 pages of 20
MAX_WORKERS = 16  # concurrent page requests per round (requests.Session is thread-safe for GETs)

RAW_DIR = BRONZE_DIR / "raw"


def fetch_page(session, page: int, round_id: int | None = None, retries: int = 3):
    import time as _time
    params = {"page": page}
    if round_id is not None:
        params["round"] = round_id
    last_err = None
    for attempt in range(retries):
        try:
            r = session.get(RANKING_URL, params=params, timeout=30)
            r.raise_for_status()
            return r.json()["success"]
        except Exception as e:
            last_err = e
            _time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"page {page} round {round_id} failed: {last_err}")


def fetch_all_pages(session, round_id: int | None = None) -> list[dict]:
    """Fetch all 100 pages concurrently via a thread pool, preserving page order.

    PFR caps the leaderboard at 2000 entries (= 100 pages of 20), so we fire all
    MAX_PAGES requests in parallel rather than paginating serially. Pages past
    the actual cutoff return empty `rankings` arrays and are dropped.

    Using requests.Session across threads is safe for GETs — the adapter pool
    holds connections; concurrent reads do not mutate session state.
    """
    results: dict[int, list[dict]] = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {
            ex.submit(fetch_page, session, p, round_id): p
            for p in range(1, MAX_PAGES + 1)
        }
        for fut in as_completed(futures):
            p = futures[fut]
            try:
                data = fut.result()
                results[p] = data.get("rankings", [])
            except Exception as e:
                log.warning("page %d round %s failed: %s", p, round_id, e)
                results[p] = []

    # Re-assemble in page order so caller sees the ranked list in rank order.
    rows: list[dict] = []
    for p in sorted(results):
        rows.extend(results[p])
    return rows


def scrape_overall(session) -> pd.DataFrame:
    log.info("scraping overall leaderboard...")
    rows = fetch_all_pages(session, round_id=None)
    log.info("overall: %d rows", len(rows))
    (RAW_DIR / "fantasy_rankings_overall.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )
    df = pd.DataFrame(rows)
    df.to_csv(BRONZE_DIR / "fantasy_rankings_overall.csv", index=False)
    return df


def scrape_per_round(session, rounds: list[int]) -> pd.DataFrame:
    frames = []
    for r in rounds:
        log.info("scraping round %d leaderboard...", r)
        rows = fetch_all_pages(session, round_id=r)
        log.info("  round %d: %d rows", r, len(rows))
        (RAW_DIR / f"fantasy_rankings_round{r}.json").write_text(
            json.dumps(rows, indent=2), encoding="utf-8"
        )
        df = pd.DataFrame(rows)
        df["_round"] = r
        frames.append(df)
    long_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not long_df.empty:
        long_df.to_csv(BRONZE_DIR / "fantasy_rankings_per_round.csv", index=False)
    return long_df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=str, default="",
                    help="comma-separated round ids to scrape per-round, or 'all' for 1-14 "
                         "(default: skip — overall leaderboard alone is enough for rank lookups)")
    ap.add_argument("--no-overall", action="store_true", help="skip overall leaderboard")
    args = ap.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    BRONZE_DIR.mkdir(parents=True, exist_ok=True)

    s = login()

    if not args.no_overall:
        scrape_overall(s)

    if args.rounds:
        if args.rounds.lower() == "all":
            # Dynamic: every round that has played at least once, per fantasy_rounds.csv.
            rounds_path = BRONZE_DIR / "fantasy_rounds.csv"
            if rounds_path.exists():
                rdf = pd.read_csv(rounds_path)
                rounds = sorted(
                    rdf.loc[rdf["status"].isin(["completed", "playing"]), "number"].astype(int).tolist()
                )
                log.info("--rounds all -> %s (status in completed/playing)", rounds)
            else:
                rounds = list(range(1, 16))
                log.warning("no fantasy_rounds.csv; defaulting to 1..15")
        else:
            rounds = [int(x) for x in args.rounds.split(",") if x.strip()]
        scrape_per_round(s, rounds)

    log.info("done — bronze: %s", BRONZE_DIR)


if __name__ == "__main__":
    main()
