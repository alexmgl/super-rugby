"""PlayFantasyRugby authentication.

Login to PFR with email/password from environment and return a requests.Session
carrying the X-SID cookie used by authenticated endpoints (rankings, my team, etc.).

Credentials live in .env (gitignored):
    PFR_EMAIL=...
    PFR_PASSWORD=...
"""

from __future__ import annotations

import os

import requests
from dotenv import load_dotenv

from src.utils.logger import get_logger

log = get_logger(__name__)

LOGIN_URL = "https://www.playfantasyrugby.com/api/en/auth/login"

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/148.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Content-Type": "application/json",
    "Origin": "https://www.playfantasyrugby.com",
    "Referer": "https://www.playfantasyrugby.com/super",
}


def login(session: requests.Session | None = None) -> requests.Session:
    """Authenticate and return a session with the X-SID cookie set."""
    load_dotenv()
    email = os.environ.get("PFR_EMAIL")
    password = os.environ.get("PFR_PASSWORD")
    if not email or not password:
        raise RuntimeError("PFR_EMAIL / PFR_PASSWORD must be set in .env")

    s = session or requests.Session()
    s.headers.update(DEFAULT_HEADERS)
    r = s.post(LOGIN_URL, json={"email": email, "password": password}, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"login failed: {r.status_code} {r.text[:200]}")
    body = r.json()
    if body.get("errors"):
        raise RuntimeError(f"login errored: {body['errors']}")
    if "X-SID" not in s.cookies:
        raise RuntimeError("login returned 200 but no X-SID cookie set")
    log.info("logged in as %s", email)
    return s


# Authenticated endpoint that returns the user's current team state.
# Response shape:
#   {"success": {"team": {
#       "id": int, "lineup": {<position>: [pid, ...]}, "captainId": int,
#       "value": int,        # current spent (team valuation)
#       "salaryCap": int,    # hard cap — this is what the MILP `budget` should use
#       "boosters": [...], "startRoundId": int, "isComplete": bool,
#   }}}
SHOW_MY_TEAM_URL = "https://www.playfantasyrugby.com/api/en/fantasy/team/show-my"


def fetch_team_state(session: requests.Session) -> dict:
    """Return the parsed `team` object from PFR's show-my-team endpoint."""
    r = session.get(SHOW_MY_TEAM_URL, timeout=30)
    r.raise_for_status()
    body = r.json()
    if body.get("errors"):
        raise RuntimeError(f"show-my-team errored: {body['errors']}")
    return body["success"]["team"]


def fetch_budget(session: requests.Session, default: float = 100_000_000) -> float:
    """Return the user's salary cap from PFR (the MILP budget constraint).

    Falls back to `default` ($100M) on any error so the optimiser still runs offline.
    """
    try:
        team = fetch_team_state(session)
        cap = float(team["salaryCap"])
        log.info("fetched live salary cap: $%.2fM (current team value $%.2fM)",
                 cap / 1e6, team.get("value", 0) / 1e6)
        return cap
    except Exception as e:
        log.warning("budget fetch failed (%s) — falling back to $%.1fM",
                    e, default / 1e6)
        return default
