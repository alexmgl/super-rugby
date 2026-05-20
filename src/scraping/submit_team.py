"""Submit an optimised team (and chip booster) to PlayFantasyRugby.

Reads one of the optimal-team parquets produced by `src.ml.walk_forward` and:
  1. computes the trades needed to morph your current squad into the target,
  2. POSTs the trades to `/api/en/fantasy/trade/make`,
  3. POSTs the lineup + (primary) captain to `/api/en/fantasy/team/update`,
  4. (if a chip is selected) POSTs the booster to `/api/en/fantasy/team/booster`.

Endpoints:
    POST /api/en/fantasy/trade/make
        {"tradePairs": [{"out": pid, "in": pid}, ...]}
    POST /api/en/fantasy/team/update
        {"captainId": int,
         "lineup": {"prop":[..], "hooker":[..], "lock":[..], "loose_forward":[..],
                    "scrum_half":[..], "fly_half":[..], "center":[..], "outside_back":[..]}}
    POST /api/en/fantasy/team/booster
        {"type": "triple_captain", "playerId": int}    # 3x captain
        {"type": "limitless"}                          # unlimited budget that round
        {"type": "co_captain",     "playerId": int}    # second player at 2x

Defaults to a dry-run (prints everything that would happen). Pass `--submit` to
actually fire the requests. The user gets a y/N prompt before any POST unless
`--yes` is passed.
"""

from __future__ import annotations

import argparse
import json

import pandas as pd

from src.utils.logger import get_logger
from src.utils.paths import GOLD_DIR
from src.utils.pfr_auth import fetch_team_state, login

log = get_logger(__name__)

TRADE_URL = "https://www.playfantasyrugby.com/api/en/fantasy/trade/make"
UPDATE_URL = "https://www.playfantasyrugby.com/api/en/fantasy/team/update"
BOOSTER_URL = "https://www.playfantasyrugby.com/api/en/fantasy/team/booster"
AUTOPICK_URL = "https://www.playfantasyrugby.com/api/en/fantasy/team/autopick"

# Empty autopick payload: 0 = "server picks this slot". Partial-locked autopick
# is supported by replacing some zeros with real player IDs (the server fills
# the remaining slots within the budget + position constraints).
AUTOPICK_EMPTY = {
    "captainId": 0,
    "lineup": {
        "prop": [0, 0], "hooker": [0], "lock": [0, 0],
        "loose_forward": [0, 0, 0], "scrum_half": [0], "fly_half": [0],
        "center": [0, 0], "outside_back": [0, 0, 0],
    },
}

EXPECTED_COUNTS = {
    "prop": 2, "hooker": 1, "lock": 2, "loose_forward": 3,
    "scrum_half": 1, "fly_half": 1, "center": 2, "outside_back": 3,
}
POSITION_ORDER = ["prop", "hooker", "lock", "loose_forward",
                  "scrum_half", "fly_half", "center", "outside_back"]


def build_payload(team: pd.DataFrame) -> tuple[dict, int | None]:
    """Build the team/update payload + return (payload, co_captain_id_or_None).

    - 1 captain in the parquet  -> (payload, None)
    - 2 captains (co_captains)  -> primary = higher P80; co-captain id returned separately
    """
    if len(team) != 15:
        raise ValueError(f"team must have 15 players, got {len(team)}")

    counts = team["position"].value_counts().to_dict()
    for pos, expected in EXPECTED_COUNTS.items():
        if counts.get(pos, 0) != expected:
            raise ValueError(f"position {pos}: have {counts.get(pos, 0)}, need {expected}")

    captains = team[team["is_captain"]]
    if len(captains) == 1:
        primary = captains.iloc[0]
        co_captain_id: int | None = None
    elif len(captains) == 2:
        captains_sorted = captains.sort_values("p80", ascending=False)
        primary = captains_sorted.iloc[0]
        co_captain_id = int(captains_sorted.iloc[1]["player_id"])
    else:
        raise ValueError(f"expected 1 or 2 captains, got {len(captains)}")

    lineup = {
        pos: [int(pid) for pid in team.loc[team["position"] == pos, "player_id"]]
        for pos in POSITION_ORDER
    }
    payload = {"captainId": int(primary["player_id"]), "lineup": lineup}
    return payload, co_captain_id


def compute_trades(current_lineup: dict, target_team: pd.DataFrame) -> list[dict]:
    """Return position-matched [{"out": pid, "in": pid}, ...] to morph current -> target.

    Trades stay within position so the squad remains valid mid-flight. PFR's
    trade endpoint processes pairs atomically so order does not matter.
    """
    trades = []
    for pos in POSITION_ORDER:
        current_ids = set(int(pid) for pid in current_lineup.get(pos, []))
        target_ids = set(
            int(pid) for pid in target_team.loc[target_team["position"] == pos, "player_id"]
        )
        to_out = sorted(current_ids - target_ids)
        to_in = sorted(target_ids - current_ids)
        if len(to_out) != len(to_in):
            raise ValueError(
                f"position {pos}: count mismatch out={to_out} in={to_in}"
            )
        for o, i in zip(to_out, to_in):
            trades.append({"out": o, "in": i})
    return trades


def submit_trades(session, pairs: list[dict]) -> dict:
    if not pairs:
        log.info("no trades needed (current squad already matches target)")
        return {}
    r = session.post(TRADE_URL, json={"tradePairs": pairs}, timeout=30)
    log.info("trade/make status: %s", r.status_code)
    try:
        body = r.json()
    except ValueError:
        body = {"_raw": r.text}
    if not r.ok:
        log.error("error body: %s", json.dumps(body)[:1000])
    r.raise_for_status()
    return body


def submit_update(session, payload: dict) -> dict:
    r = session.post(UPDATE_URL, json=payload, timeout=30)
    log.info("team/update status: %s", r.status_code)
    try:
        body = r.json()
    except ValueError:
        body = {"_raw": r.text}
    if not r.ok:
        log.error("error body: %s", json.dumps(body)[:1000])
    r.raise_for_status()
    return body


def submit_booster(session, chip: str, primary_captain_id: int,
                   co_captain_id: int | None = None) -> dict | None:
    """POST /api/en/fantasy/team/booster for the chip in use. `normal` -> no-op.

    Returns the parsed response body. If activation succeeded, the response will
    typically include a booster instance id (e.g. 168616) which can be passed to
    `cancel_booster()` to undo.
    """
    if chip == "normal":
        return None
    if chip == "triple_captain":
        payload = {"type": "triple_captain", "playerId": primary_captain_id}
    elif chip == "limitless":
        payload = {"type": "limitless"}
    elif chip == "co_captains":
        if co_captain_id is None:
            raise ValueError("co_captains chip needs a second captain")
        payload = {"type": "co_captain", "playerId": co_captain_id}
    else:
        raise ValueError(f"unknown chip: {chip}")

    log.info("booster payload: %s", payload)
    r = session.post(BOOSTER_URL, json=payload, timeout=30)
    log.info("team/booster status: %s", r.status_code)
    try:
        body = r.json()
    except ValueError:
        body = {"_raw": r.text}
    if not r.ok:
        log.error("error body: %s", json.dumps(body)[:1000])
    r.raise_for_status()
    return body


def submit_autopick(session, payload: dict | None = None) -> dict:
    """POST /api/en/fantasy/team/autopick.

    With an all-zeros payload (the default), the server picks the entire team
    automatically subject to budget + position constraints. Partial autopick is
    supported by passing a payload with some real player IDs locked in and the
    rest left as zero.
    """
    body_payload = payload if payload is not None else AUTOPICK_EMPTY
    r = session.post(AUTOPICK_URL, json=body_payload, timeout=30)
    log.info("team/autopick status: %s", r.status_code)
    try:
        body = r.json()
    except ValueError:
        body = {"_raw": r.text}
    if not r.ok:
        log.error("error body: %s", json.dumps(body)[:1000])
    r.raise_for_status()
    return body


def cancel_booster(session, booster_id: int) -> dict:
    """Cancel a previously-activated booster by its instance id.

    PFR uses an oddly-shaped GET-as-cancel convention:
        GET /api/en/fantasy/team/booster/<id>

    The id comes from the response body of the POST that activated it, or from
    `team.boosters[*].id` on the show-my endpoint.
    """
    url = f"{BOOSTER_URL}/{booster_id}"
    r = session.get(url, timeout=30)
    log.info("cancel booster %d -> status %s", booster_id, r.status_code)
    try:
        body = r.json()
    except ValueError:
        body = {"_raw": r.text}
    if not r.ok:
        log.error("error body: %s", json.dumps(body)[:1000])
    r.raise_for_status()
    return body


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--chip", default="normal",
        choices=["normal", "triple_captain", "limitless", "co_captains"],
        help="which round-15 optimal team to submit (default: normal)",
    )
    ap.add_argument("--parquet", default=None,
                    help="explicit parquet path; overrides --chip if given")
    ap.add_argument("--submit", action="store_true",
                    help="actually POST trades/update/booster (default: dry-run)")
    ap.add_argument("--yes", "-y", action="store_true",
                    help="skip the y/N confirmation prompt before submitting")
    ap.add_argument("--cancel-booster", type=int, default=None, metavar="ID",
                    help="GET /api/en/fantasy/team/booster/<ID> to cancel an "
                         "active booster (e.g. 168616). Skips the rest of the flow.")
    ap.add_argument("--no-trades", action="store_true",
                    help="skip /trade/make and POST /team/update directly. "
                         "Use when update is enough to swap the full lineup atomically.")
    ap.add_argument("--autopick", action="store_true",
                    help="POST /team/autopick with an all-zeros payload "
                         "(server fills the entire team). Skips the rest of the flow.")
    args = ap.parse_args()

    if args.cancel_booster is not None:
        s = login()
        body = cancel_booster(s, args.cancel_booster)
        log.info("cancel response: %s", json.dumps(body)[:500])
        return

    if args.autopick:
        log.info("autopick payload: %s", json.dumps(AUTOPICK_EMPTY))
        if not args.submit:
            log.info("[dry-run] re-run with --submit to actually POST")
            return
        if not args.yes:
            try:
                confirm = input("\nPOST /team/autopick (server will replace your team)? [y/N]: ").strip().lower()
            except EOFError:
                confirm = ""
            if confirm != "y":
                log.info("aborted by user")
                return
        s = login()
        body = submit_autopick(s)
        log.info("autopick response: %s", json.dumps(body)[:500])
        return

    path = args.parquet or str(GOLD_DIR / f"optimal_team_round15_{args.chip}.parquet")
    team = pd.read_parquet(path)
    log.info("loaded target team from %s (%d players)", path, len(team))

    payload, co_captain_id = build_payload(team)

    # Need a session early so we can fetch current state.
    session = login()
    current = fetch_team_state(session)
    current_lineup = current["lineup"]
    log.info("current squad value $%.2fM, salary cap $%.2fM",
             current["value"] / 1e6, current["salaryCap"] / 1e6)

    trades = [] if args.no_trades else compute_trades(current_lineup, team)

    # --- show everything we're about to do ---
    primary = team[team["is_captain"]].sort_values("p80", ascending=False).iloc[0]
    log.info("\n--- plan ---")
    log.info("chip          : %s", args.chip)
    log.info("primary captain: %s %s (id=%d)",
             primary["first_name"], primary["last_name"], payload["captainId"])
    if co_captain_id is not None:
        co = team[team["player_id"] == co_captain_id].iloc[0]
        log.info("co-captain    : %s %s (id=%d)",
                 co["first_name"], co["last_name"], co_captain_id)

    if trades:
        log.info("\ntrades (%d):", len(trades))
        for t in trades:
            in_row = team[team["player_id"] == t["in"]].iloc[0]
            log.info("  out %5d  ->  in %5d  (%s %s, %s)",
                     t["out"], t["in"], in_row["first_name"], in_row["last_name"],
                     in_row["position"])
    elif args.no_trades:
        log.info("\n--no-trades: skipping /trade/make, will POST /team/update directly")
    else:
        log.info("\nno trades needed")

    log.info("\nlineup:")
    for pos in POSITION_ORDER:
        names = team.loc[team["position"] == pos, ["first_name", "last_name", "player_id"]]
        joined = ", ".join(f"{r.first_name} {r.last_name} ({r.player_id})"
                           for _, r in names.iterrows())
        log.info("  %s: %s", pos, joined)

    log.info("\nfull payloads that will be POSTed:")
    if trades:
        log.info("  trade/make : %s", json.dumps({"tradePairs": trades}))
    log.info("  team/update: %s", json.dumps(payload))
    if args.chip != "normal":
        booster_preview = (
            {"type": "co_captain", "playerId": co_captain_id} if args.chip == "co_captains"
            else {"type": "triple_captain", "playerId": payload["captainId"]} if args.chip == "triple_captain"
            else {"type": "limitless"}
        )
        log.info("  team/booster: %s", json.dumps(booster_preview))

    if not args.submit:
        log.info("\n[dry-run] re-run with --submit to actually POST")
        return

    if not args.yes:
        try:
            confirm = input(
                f"\nFire {len(trades)} trades + lineup update + "
                f"{'no booster' if args.chip == 'normal' else args.chip + ' booster'}? [y/N]: "
            ).strip().lower()
        except EOFError:
            confirm = ""
        if confirm != "y":
            log.info("aborted by user")
            return

    log.info("\n=== submitting ===")
    if trades:
        body = submit_trades(session, trades)
        log.info("trade response: %s", json.dumps(body)[:500])

    body = submit_update(session, payload)
    log.info("update response: %s", json.dumps(body)[:500])

    body = submit_booster(session, args.chip, payload["captainId"], co_captain_id)
    if body is not None:
        log.info("booster response: %s", json.dumps(body)[:500])
        # Try to surface the booster instance id so the user can cancel it later if needed.
        booster_id = None
        if isinstance(body, dict):
            success = body.get("success") if isinstance(body.get("success"), dict) else None
            for candidate in (success, body):
                if isinstance(candidate, dict):
                    booster_id = booster_id or candidate.get("id") or candidate.get("boosterId")
        if booster_id is not None:
            log.info("booster instance id: %s  "
                     "(cancel via: python -m src.scraping.submit_team --cancel-booster %s)",
                     booster_id, booster_id)

    log.info("\ndone")


if __name__ == "__main__":
    main()
