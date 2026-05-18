"""Submit an optimised team to PlayFantasyRugby.

Reads one of the optimal-team parquets produced by `src.ml.walk_forward` and
POSTs it to the authenticated team-update endpoint.

Endpoint:
    POST /api/en/fantasy/team/update
Payload shape:
    {
        "captainId": int,
        "lineup": {
            "prop":           [pid, pid],
            "hooker":         [pid],
            "lock":           [pid, pid],
            "loose_forward":  [pid, pid, pid],
            "scrum_half":     [pid],
            "fly_half":       [pid],
            "center":         [pid, pid],
            "outside_back":   [pid, pid, pid]
        }
    }

Defaults to a dry-run (prints payload + summary). Pass `--submit` to actually POST.
"""

from __future__ import annotations

import argparse
import json

import pandas as pd

from src.utils.logger import get_logger
from src.utils.paths import GOLD_DIR
from src.utils.pfr_auth import login

log = get_logger(__name__)

UPDATE_URL = "https://www.playfantasyrugby.com/api/en/fantasy/team/update"

EXPECTED_COUNTS = {
    "prop": 2, "hooker": 1, "lock": 2, "loose_forward": 3,
    "scrum_half": 1, "fly_half": 1, "center": 2, "outside_back": 3,
}
POSITION_ORDER = ["prop", "hooker", "lock", "loose_forward",
                  "scrum_half", "fly_half", "center", "outside_back"]


def build_payload(team: pd.DataFrame) -> dict:
    """Convert the optimal-team DataFrame into the API payload shape."""
    if len(team) != 15:
        raise ValueError(f"team must have 15 players, got {len(team)}")

    counts = team["position"].value_counts().to_dict()
    for pos, expected in EXPECTED_COUNTS.items():
        if counts.get(pos, 0) != expected:
            raise ValueError(
                f"position {pos}: have {counts.get(pos, 0)}, need {expected}"
            )

    captains = team[team["is_captain"]]
    if len(captains) != 1:
        raise ValueError(
            f"need exactly 1 captain for this endpoint, got {len(captains)} "
            f"(co_captains chip parquets won't submit via this endpoint as-is)"
        )

    lineup = {
        pos: [int(pid) for pid in team.loc[team["position"] == pos, "player_id"]]
        for pos in POSITION_ORDER
    }
    return {
        "captainId": int(captains["player_id"].iloc[0]),
        "lineup": lineup,
    }


def submit(payload: dict) -> dict:
    """Authenticate and POST the team. Returns parsed response body."""
    s = login()
    r = s.post(UPDATE_URL, json=payload, timeout=30)
    log.info("update status: %s", r.status_code)
    try:
        body = r.json()
    except ValueError:
        body = {"_raw": r.text}
    r.raise_for_status()
    return body


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--chip", default="normal",
        choices=["normal", "triple_captain", "limitless", "co_captains"],
        help="which round-15 optimal team to submit (default: normal)",
    )
    ap.add_argument(
        "--parquet", default=None,
        help="explicit parquet path; overrides --chip if given",
    )
    ap.add_argument("--submit", action="store_true",
                    help="actually POST the payload (default: dry-run)")
    ap.add_argument("--yes", "-y", action="store_true",
                    help="skip the y/N confirmation prompt before submitting")
    args = ap.parse_args()

    path = args.parquet or str(GOLD_DIR / f"optimal_team_round15_{args.chip}.parquet")
    team = pd.read_parquet(path)
    log.info("loaded team from %s (%d players)", path, len(team))

    payload = build_payload(team)

    cap_row = team[team["is_captain"]].iloc[0]
    log.info("\n--- payload preview ---")
    log.info("captain: %s %s (id=%d, %s)",
             cap_row["first_name"], cap_row["last_name"],
             payload["captainId"], cap_row["squad_abbr"])
    for pos in POSITION_ORDER:
        names = team.loc[team["position"] == pos, ["first_name", "last_name", "player_id"]]
        joined = ", ".join(f"{r.first_name} {r.last_name} ({r.player_id})"
                           for _, r in names.iterrows())
        log.info("  %s: %s", pos, joined)
    log.info("\nfull JSON:\n%s", json.dumps(payload, indent=2))

    if not args.submit:
        log.info("\n[dry-run] re-run with --submit to POST this to PFR")
        return

    if not args.yes:
        try:
            confirm = input(
                f"\nSubmit this team to PFR for {cap_row['first_name']} "
                f"{cap_row['last_name']} as captain? [y/N]: "
            ).strip().lower()
        except EOFError:
            confirm = ""
        if confirm != "y":
            log.info("aborted by user")
            return

    log.info("\nsubmitting...")
    body = submit(payload)
    log.info("response: %s", json.dumps(body, indent=2)[:1000])


if __name__ == "__main__":
    main()
