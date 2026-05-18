"""
super_rugby_data_scraper.py

Pulls every publicly-available piece of Super Rugby Pacific data useful for a
fantasy points predictor:

  1. PlayFantasyRugby JSON feeds         (fantasy cost, ownership, fantasy points,
                                          per-round scores, season aggregate stats)
  2. Ultimate Rugby player profiles      (DOB, height, weight, position, bio,
                                          career history)
  3. Ultimate Rugby fixtures + lineups   (date, venue, score, starting 15 + bench)
  4. Ultimate Rugby match-level stats    (possession, territory, attack, defence)

Outputs (in data/bronze/, source-prefixed):
    raw/fantasy_<name>.json            raw PlayFantasyRugby JSON dumps
    raw/ur_<name>.json                 raw Ultimate Rugby intermediate dumps
    raw/html/                          UR HTML cache (persistent across runs)

    fantasy_squads.csv                 11 clubs
    fantasy_rounds.csv                 16 rounds
    fantasy_fixtures.csv               all season matches with scores/venues
    fantasy_players.csv                fantasy fields + joined UR bio fields
    fantasy_player_round_scores.csv    long: player_id, round, fantasy_points
    fantasy_player_round_prices.csv    long: player_id, round, price
    fantasy_player_round_ownership.csv long: player_id, round, selected_pct
    fantasy_player_season_stats.csv    tries, tackles, metres etc per player

    ur_player_bios.csv                 DOB, height, weight, position, bio text
    ur_player_career.csv               long: prior clubs per player (year_start/end)
    ur_match_lineups.csv               long: starting 15 + bench per match
    ur_match_team_stats.csv            long: team-level match stats (empty — JS-rendered)

Dependencies:
    pip install requests beautifulsoup4 lxml

Usage:
    python super_rugby_data_scraper.py
    python super_rugby_data_scraper.py --no-bios     # skip Ultimate Rugby profiles
    python super_rugby_data_scraper.py --no-matches  # skip lineup/stats scrape
"""

from __future__ import annotations

import argparse
import csv
import difflib
import json
import re
import sys
import time
import unicodedata
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable

import requests
from bs4 import BeautifulSoup

from src.utils.logger import get_logger
from src.utils.paths import BRONZE_DIR

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

FANTASY_BASE = "https://www.playfantasyrugby.com"
FANTASY_JSON = f"{FANTASY_BASE}/json/fantasy"
FANTASY_MEDIA = f"{FANTASY_BASE}/media"

UR_BASE = "https://www.ultimaterugby.com"
UR_COMP_RESULTS = f"{UR_BASE}/app/public/index.php/super-rugby-pacific-2026/results"
UR_COMP_FIXTURES = f"{UR_BASE}/app/public/index.php/super-rugby-pacific-2026/matches"

# Primary squad-page slug for each fantasy team (used for /<slug>/squad fetches).
SQUAD_ABBR_TO_UR_SLUG = {
    "HIGH": "highlanders",
    "CRUS": "crusaders",
    "WARA": "nsw-waratahs",
    "REDS": "queensland-reds",
    "FDRU": "fiji-warriors",
    "MOPA": "moana-pasifika",
    "BLUE": "auckland-blues",
    "CHIE": "chiefs",
    "FORC": "western-force",
    "BRUM": "brumbies",
    "HURR": "hurricanes",
}

# Any UR team-slug variant -> fantasy squad abbreviation. UR is inconsistent across
# squad pages and match URLs (e.g. /brumbies/squad but match slug uses "act-brumbies";
# /auckland-blues/squad but match slug uses "blues"). Used by parse_match_teams +
# by silver to map lineup rows back to fantasy squads.
UR_TEAM_SLUG_TO_ABBR = {
    "highlanders":            "HIGH",
    "crusaders":              "CRUS",
    "nsw-waratahs":           "WARA",
    "queensland-reds":        "REDS",
    "fiji-warriors":          "FDRU",
    "fijian-drua":            "FDRU",
    "moana-pasifika":         "MOPA",
    "moana-pasifika-rugby":   "MOPA",
    "auckland-blues":         "BLUE",
    "blues":                  "BLUE",
    "chiefs":                 "CHIE",
    "western-force":          "FORC",
    "brumbies":               "BRUM",
    "act-brumbies":           "BRUM",
    "hurricanes":             "HURR",
}

ENDPOINTS = {
    "checksums":    f"{FANTASY_JSON}/checksums.json",
    "squads":       f"{FANTASY_JSON}/squads.json",
    "rounds":       f"{FANTASY_JSON}/rounds.json",
    "players":      f"{FANTASY_JSON}/players.json",
    "player_stats": f"{FANTASY_JSON}/player_stats.json",
}

OUT_DIR = BRONZE_DIR
RAW_DIR = OUT_DIR / "raw"
HTML_CACHE = RAW_DIR / "html"

HEADERS = {
    "User-Agent": "super-rugby-pacific-scraper/1.0 (+research)",
    "Accept": "text/html,application/json,*/*",
    "Accept-Language": "en-US,en;q=0.9",
}

HTTP_WORKERS = 6
HTTP_DELAY = 0.25  # seconds between requests per worker

MONTHS = {m: i + 1 for i, m in enumerate(
    ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul",
     "Aug", "Sep", "Oct", "Nov", "Dec"])}


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def ensure_dirs() -> None:
    for d in (OUT_DIR, RAW_DIR, HTML_CACHE):
        d.mkdir(parents=True, exist_ok=True)


def normalise(name: str) -> str:
    """Lowercase, strip accents/punctuation, collapse to alpha-only."""
    if not name:
        return ""
    nfkd = unicodedata.normalize("NFKD", name)
    ascii_only = "".join(c for c in nfkd if not unicodedata.combining(c))
    return re.sub(r"[^a-z]+", "", ascii_only.lower())


def fetch_json(url: str, session: requests.Session, retries: int = 3) -> Any:
    last = None
    for i in range(retries):
        try:
            r = session.get(url, headers=HEADERS, timeout=30)
            r.raise_for_status()
            return r.json()
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(1.5 * (i + 1))
    raise RuntimeError(f"GET {url} failed: {last}")


def fetch_html(url: str, session: requests.Session, retries: int = 3) -> str | None:
    cache_path = HTML_CACHE / (re.sub(r"[^a-zA-Z0-9_-]+", "_", url)[:180] + ".html")
    if cache_path.exists():
        return cache_path.read_text(encoding="utf-8")
    last = None
    for i in range(retries):
        try:
            r = session.get(url, headers=HEADERS, timeout=30, allow_redirects=True)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            if "ultimaterugby.com" in url and "Page not Found" in r.text:
                return None
            cache_path.write_text(r.text, encoding="utf-8")
            time.sleep(HTTP_DELAY)
            return r.text
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(1.0 * (i + 1))
    log.warning("GET %s failed: %s", url, last)
    return None


def write_csv(path: Path, rows: Iterable[dict], fieldnames: list[str]) -> None:
    rows = list(rows)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    log.info("wrote %s (%d rows)", path, len(rows))


# ---------------------------------------------------------------------------
# 1. Fantasy data
# ---------------------------------------------------------------------------

def fetch_fantasy_data() -> dict[str, Any]:
    log.info("[1/5] Fetching PlayFantasyRugby static JSON...")
    with requests.Session() as s:
        data = {n: fetch_json(u, s) for n, u in ENDPOINTS.items()}
    for n, p in data.items():
        (RAW_DIR / f"fantasy_{n}.json").write_text(
            json.dumps(p, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    log.info("checksums=%s", data["checksums"])
    log.info("squads=%d rounds=%d players=%d player_stats=%d",
             len(data["squads"]), len(data["rounds"]),
             len(data["players"]), len(data["player_stats"]))
    return data


# ---------------------------------------------------------------------------
# 2. Ultimate Rugby — team squad pages -> player slug index
# ---------------------------------------------------------------------------

def fetch_ur_squad_slugs(session: requests.Session) -> dict[str, list[str]]:
    log.info("[2/5] Fetching Ultimate Rugby squad rosters...")
    out: dict[str, list[str]] = {}
    for abbr, team_slug in SQUAD_ABBR_TO_UR_SLUG.items():
        html = fetch_html(f"{UR_BASE}/{team_slug}/squad", session)
        if not html:
            log.warning("no squad page for %s", team_slug)
            out[team_slug] = []
            continue
        soup = BeautifulSoup(html, "lxml")
        slugs: list[str] = []
        for a in soup.select("a[href^='/']"):
            href = a.get("href", "")
            # Allow %-encoded characters (e.g. %27 for apostrophe in Polynesian surnames).
            if (re.fullmatch(r"/[a-z][a-z0-9%-]+", href)
                    and href.strip("/") not in {team_slug, "privacy", "teams"}
                    and not href.startswith(f"/{team_slug}")):
                slugs.append(urllib.parse.unquote(href.strip("/")))
        seen, ordered = set(), []
        for s in slugs:
            if s not in seen:
                seen.add(s)
                ordered.append(s)
        out[team_slug] = ordered
        log.info("%-5s (%s): %d players", abbr, team_slug, len(ordered))
    return out


# ---------------------------------------------------------------------------
# 3. Match fantasy players to UR slugs
# ---------------------------------------------------------------------------

def parse_match_teams(match_slug: str) -> tuple[str, str]:
    """Extract (home_team_slug, away_team_slug) from a UR match slug.

    Recognises every UR team-slug variant in UR_TEAM_SLUG_TO_ABBR.

    Example: 'act-brumbies-vs-blues-at-gio-stadium-28th-feb-2026'
                -> ('act-brumbies', 'blues')

    Returns ('', '') if either segment isn't a known UR team slug.
    """
    if not match_slug or "-vs-" not in match_slug:
        return ("", "")
    decoded = urllib.parse.unquote(match_slug)
    home_part, rest = decoded.split("-vs-", 1)
    away_part = rest.split("-at-", 1)[0] if "-at-" in rest else rest
    if home_part in UR_TEAM_SLUG_TO_ABBR and away_part in UR_TEAM_SLUG_TO_ABBR:
        return (home_part, away_part)
    return ("", "")


def _build_lineup_indexes(
    lineup_rows: list[dict] | None,
) -> tuple[dict[tuple[str, str], str], dict[str, set[str]]]:
    """Return (team_name_idx, name_idx) built from match-lineup rows.

    - team_name_idx: (team_slug, normalised_name) -> slug
    - name_idx     : normalised_name -> {slug, ...}
    """
    team_name_idx: dict[tuple[str, str], str] = {}
    name_idx: dict[str, set[str]] = {}
    for r in lineup_rows or []:
        name = r.get("player_name", "")
        slug = r.get("player_slug", "")
        if not name or not slug:
            continue
        norm = normalise(name)
        name_idx.setdefault(norm, set()).add(slug)
        home_t, away_t = parse_match_teams(r.get("match_slug", ""))
        side = r.get("team_side", "")
        this_team = home_t if side == "home" else away_t if side == "away" else ""
        if this_team:
            team_name_idx[(this_team, norm)] = slug
    return team_name_idx, name_idx


def match_players_to_slugs(
    players: list[dict],
    fantasy_squads_by_id: dict[int, dict],
    squad_slugs: dict[str, list[str]],
    lineup_rows: list[dict] | None = None,
) -> dict[int, str]:
    """Match fantasy players to Ultimate Rugby slugs using multiple sources.

    Heuristic cascade (first hit wins):
      1. Squad-roster exact full-name match (any team)
      2. Lineup match by (team, full name)              [fixes MOPA roster gaps]
      3. Squad-roster last-name match within team
      4. Squad-roster first-initial + last name within team
      5. Squad-roster loose substring within team
      6. Lineup unique full-name globally
      7. Fuzzy match within team's squad roster         [fixes spelling drift]
      8. Fuzzy match within team's lineup names         [fixes drift + missing rosters]
    """
    log.info("[3/5] Matching fantasy players to Ultimate Rugby slugs...")

    # Squad-roster index
    norm_to_slug: dict[str, str] = {}
    for slugs in squad_slugs.values():
        for s in slugs:
            norm_to_slug[normalise(s.replace("-", ""))] = s

    # Lineup-derived indexes (empty if lineups not provided)
    team_name_idx, name_idx = _build_lineup_indexes(lineup_rows)

    matches: dict[int, str] = {}
    source: dict[str, int] = {}
    unmatched: list[dict] = []

    def _record(pid: int, slug: str, src: str) -> None:
        matches[pid] = slug
        source[src] = source.get(src, 0) + 1

    for p in players:
        team_abbr = fantasy_squads_by_id.get(p["squadId"], {}).get("abbreviation", "")
        team_slug = SQUAD_ABBR_TO_UR_SLUG.get(team_abbr)
        if not team_slug:
            unmatched.append(p)
            continue
        fn, ln = p.get("firstName", ""), p.get("lastName", "")
        full_norm = normalise(fn + ln)
        last_norm = normalise(ln)
        team_roster = squad_slugs.get(team_slug, [])

        # 1) Squad-roster exact full-name match (any team)
        if full_norm in norm_to_slug:
            _record(p["id"], norm_to_slug[full_norm], "squad_full")
            continue

        # 2) Lineup match by (team, full name) — strongest signal for missing rosters
        if (team_slug, full_norm) in team_name_idx:
            _record(p["id"], team_name_idx[(team_slug, full_norm)], "lineup_team_full")
            continue

        # 3) Last-name suffix within team roster
        candidates = [s for s in team_roster if normalise(s.split("-")[-1]) == last_norm]
        if len(candidates) == 1:
            _record(p["id"], candidates[0], "squad_lastname")
            continue

        # 4) First-initial + last-name within team
        if candidates:
            fi = normalise(fn)[:1]
            filtered = [s for s in candidates if normalise(s.split("-")[0]).startswith(fi)]
            if len(filtered) == 1:
                _record(p["id"], filtered[0], "squad_initial_last")
                continue

        # 5) Loose substring within team
        loose = [s for s in team_roster
                 if last_norm and last_norm in normalise(s.replace("-", ""))]
        if len(loose) == 1:
            _record(p["id"], loose[0], "squad_substring")
            continue

        # 6) Globally unique lineup full-name
        slugs_global = name_idx.get(full_norm, set())
        if len(slugs_global) == 1:
            _record(p["id"], next(iter(slugs_global)), "lineup_global_unique")
            continue

        # 7) Fuzzy within team's squad roster (catches spelling drift like Proffit/Profit)
        team_norm_pairs = [(normalise(s.replace("-", "")), s) for s in team_roster]
        team_norms = [pair[0] for pair in team_norm_pairs]
        close = difflib.get_close_matches(full_norm, team_norms, n=1, cutoff=0.85)
        if close:
            for nrm, slg in team_norm_pairs:
                if nrm == close[0]:
                    _record(p["id"], slg, "fuzzy_squad")
                    break
            continue

        # 8) Fuzzy within team's lineup names
        team_lineup_norms = [norm for (tt, norm) in team_name_idx if tt == team_slug]
        close = difflib.get_close_matches(full_norm, team_lineup_norms, n=1, cutoff=0.85)
        if close:
            _record(p["id"], team_name_idx[(team_slug, close[0])], "fuzzy_lineup")
            continue

        unmatched.append(p)

    log.info("matched %d/%d players to UR slugs", len(matches), len(players))
    log.info("match sources: %s", dict(sorted(source.items())))
    if unmatched:
        sample = ", ".join(f"{p['firstName']} {p['lastName']}"
                           for p in unmatched[:15])
        log.info("unmatched (%d): %s%s", len(unmatched), sample,
                 "..." if len(unmatched) > 15 else "")
    return matches


# ---------------------------------------------------------------------------
# 4. UR player profile parser
# ---------------------------------------------------------------------------

# "31st Aug 2002 1.86m/105kg Outside Centre"
RE_PROFILE_HEADER = re.compile(
    r"(\d{1,2}(?:st|nd|rd|th)?\s+[A-Za-z]+\s+\d{4})\s+"
    r"([\d.]+)m/(\d+)kg\s+"
    r"(.+?)\s*$",
    re.MULTILINE,
)
RE_DATE = re.compile(r"(\d{1,2})(?:st|nd|rd|th)?\s+([A-Za-z]+)\s+(\d{4})")


def _date_to_iso(s: str) -> str | None:
    m = RE_DATE.search(s)
    if not m:
        return None
    d, mon, y = m.groups()
    mon_num = MONTHS.get(mon[:3].title())
    if not mon_num:
        return None
    return f"{y}-{mon_num:02d}-{int(d):02d}"


def parse_ur_profile(html: str, slug: str) -> dict[str, Any] | None:
    if not html:
        return None
    soup = BeautifulSoup(html, "lxml")

    # Name from og:title
    name = ""
    if (m := soup.find("meta", attrs={"property": "og:title"})):
        name = (m.get("content") or "").strip()

    # Internal numeric id from ios deep link
    ur_id = None
    if (m := soup.find("meta", attrs={"property": "al:ios:url"})):
        if (mm := re.search(r"player/(\d+)", m.get("content", ""))):
            ur_id = int(mm.group(1))

    # Image
    image_url = ""
    if (m := soup.find("meta", attrs={"property": "og:image"})):
        image_url = (m.get("content") or "").strip()

    text = soup.get_text("\n", strip=True)

    profile: dict[str, Any] = {}
    if (m := RE_PROFILE_HEADER.search(text)):
        date_str, height_m, weight_kg, position = m.groups()
        if (iso := _date_to_iso(date_str)):
            profile["dob"] = iso
        profile["height_m"] = float(height_m)
        profile["weight_kg"] = int(weight_kg)
        profile["position"] = position.strip()

    # Bio paragraphs (everything between Bio header and Career block)
    paragraphs: list[str] = []
    for p in soup.find_all("p"):
        t = p.get_text(" ", strip=True)
        if not t:
            continue
        if "Privacy Policy" in t or "Download" in t or "Ultimate Rugby Ltd" in t:
            continue
        paragraphs.append(t)
    bio_text = "\n\n".join(paragraphs[:8])

    # Career block — UR renders three lines per entry: team, position, year-range.
    # Walk the full document text and look for (team, position, "YYYY - YYYY|present") triples.
    career: list[dict] = []
    text_lines = [ln.strip() for ln in text.split("\n")]
    yr_re = re.compile(r"^(\d{4})\s*-\s*(present|\d{4})\s*$")
    i = 0
    while i < len(text_lines) - 2:
        yrs = text_lines[i + 2]
        mm = yr_re.match(yrs)
        if mm:
            team = text_lines[i]
            position_line = text_lines[i + 1]
            if team and position_line and len(team) < 60 and len(position_line) < 60 \
                    and not re.search(r"\d{4}", team):
                career.append({
                    "team": team,
                    "position": position_line,
                    "year_start": int(mm.group(1)),
                    "year_end": None if mm.group(2) == "present" else int(mm.group(2)),
                    "years": yrs.replace(" ", ""),
                })
                i += 3
                continue
        i += 1

    return {
        "slug": slug,
        "name": name,
        "ur_id": ur_id,
        "image_url": image_url,
        "dob": profile.get("dob", ""),
        "height_m": profile.get("height_m"),
        "weight_kg": profile.get("weight_kg"),
        "position": profile.get("position", ""),
        "bio": bio_text,
        "career": career,
    }


def fetch_player_bios(slug_list: list[str],
                      session: requests.Session) -> dict[str, dict]:
    log.info("[4/5] Fetching %d Ultimate Rugby profiles...", len(slug_list))
    out: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=HTTP_WORKERS) as ex:
        futs = {ex.submit(fetch_html, f"{UR_BASE}/{slug}", session): slug
                for slug in slug_list}
        done = 0
        for fut in as_completed(futs):
            slug = futs[fut]
            done += 1
            if done % 50 == 0:
                log.info("profile fetch %d/%d", done, len(slug_list))
            try:
                html = fut.result()
            except Exception as e:  # noqa: BLE001
                log.warning("profile fetch failed for %s: %s", slug, e)
                continue
            if not html:
                continue
            parsed = parse_ur_profile(html, slug)
            if parsed:
                out[slug] = parsed
    log.info("parsed %d profiles", len(out))
    return out


# ---------------------------------------------------------------------------
# 5. UR match data — fixture list, lineups, team-level stats
# ---------------------------------------------------------------------------

RE_MATCH_HREF = re.compile(
    r"/app/public/index\.php/match/([a-z0-9%\-\.']+?)/(\d+)\b"
)


def fetch_match_ids(session: requests.Session) -> list[tuple[str, str]]:
    log.info("[5/5a] Fetching Super Rugby Pacific match list...")
    ids: dict[str, str] = {}
    for url in (UR_COMP_RESULTS, UR_COMP_FIXTURES):
        for page in range(1, 10):
            page_url = f"{url}?page={page}" if page > 1 else url
            html = fetch_html(page_url, session)
            if not html:
                break
            before = len(ids)
            for m in RE_MATCH_HREF.finditer(html):
                slug, mid = m.group(1), m.group(2)
                ids.setdefault(mid, slug)
            if len(ids) == before:
                break
    log.info("found %d matches", len(ids))
    return list(ids.items())


def _slug_from_href(href: str) -> str:
    """Extract the player slug from a UR href, URL-decoding it.

    Handles both top-level (`/dalton-papali%27i`) and app-path
    (`/app/public/index.php/dalton-papali%27i`) forms.
    """
    if not href:
        return ""
    tail = href.rstrip("/").rsplit("/", 1)[-1]
    return urllib.parse.unquote(tail)


def parse_match_lineup(html: str) -> list[dict]:
    """Return rows: {team_side, role, position, player_name, player_slug}.

    UR renders the lineup as a single table:
      - separator rows have one cell ("Starting 15", "Substitutes")
      - data rows have three cells: home_player | position | away_player
    """
    if not html:
        return []
    soup = BeautifulSoup(html, "lxml")
    rows: list[dict] = []

    for tbl in soup.find_all("table"):
        role = "start"
        for tr in tbl.find_all("tr"):
            cells = tr.find_all(["td", "th"])

            if len(cells) == 1:
                marker = cells[0].get_text(strip=True).lower()
                if "substitute" in marker:
                    role = "bench"
                elif "starting" in marker:
                    role = "start"
                continue

            if len(cells) != 3:
                continue

            home_cell, pos_cell, away_cell = cells
            position = pos_cell.get_text(" ", strip=True)
            # Starter rows always have a position label; bench rows leave it empty.
            # Without this, all 8 substitutes per team get dropped.
            if role == "start" and not position:
                continue
            if not position:
                position = "Substitute"

            for side, cell in (("home", home_cell), ("away", away_cell)):
                a = cell.find("a")
                href = a["href"] if a and a.has_attr("href") else ""
                slug = _slug_from_href(href)
                name = cell.get_text(" ", strip=True)
                if not (slug or name):
                    continue
                rows.append({
                    "team_side": side,
                    "role": role,
                    "position": position,
                    "player_name": name,
                    "player_slug": slug,
                })

    # Dedupe (multiple tables can render the same fixture, e.g. mobile/desktop variants).
    seen, out = set(), []
    for r in rows:
        key = (r["team_side"], r["role"], r["position"], r["player_slug"])
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


STAT_SECTIONS = {"POSSESSION", "TERRITORY", "ATTACK", "DEFENCE",
                 "DISCIPLINE", "SET PIECE", "KICKING"}


def parse_match_team_stats(html: str) -> list[dict]:
    """Return rows: {section, stat_name, home_value, away_value}."""
    if not html:
        return []
    soup = BeautifulSoup(html, "lxml")
    section = ""
    rows: list[dict] = []
    # Walk in document order so we always know the current section heading.
    for el in soup.find_all(True):
        text = el.get_text(" ", strip=True)
        if not text:
            continue
        up = text.strip().upper()
        if up in STAT_SECTIONS and len(text) < 30:
            section = up
            continue
        # Stat rows have the shape "<num/percent> <label> <num/percent>"
        if section and len(text) < 80:
            m = re.match(
                r"^\s*(-?\d+(?:\.\d+)?%?)\s+(.+?)\s+(-?\d+(?:\.\d+)?%?)\s*$",
                text,
            )
            if m:
                home_v, label, away_v = m.groups()
                label = label.strip()
                # Filter out the "1st Half / 2nd Half" overlays under
                # POSSESSION/TERRITORY (kept by giving them a sub-label).
                rows.append({
                    "section": section,
                    "stat_name": label,
                    "home_value": home_v,
                    "away_value": away_v,
                })

    # Dedupe — multiple wrappers around the same stat element are common.
    seen, out = set(), []
    for r in rows:
        key = (r["section"], r["stat_name"], r["home_value"], r["away_value"])
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def fetch_match_pages(
    match_ids: list[tuple[str, str]],
    session: requests.Session,
) -> tuple[list[dict], list[dict]]:
    log.info("[5/5b] Fetching %d match lineup + stats pages...", len(match_ids))
    lineup_rows: list[dict] = []
    stat_rows: list[dict] = []
    for idx, (mid, slug) in enumerate(match_ids, 1):
        if idx % 20 == 0:
            log.info("match fetch %d/%d", idx, len(match_ids))
        lineup_url = f"{UR_BASE}/app/public/index.php/match/{slug}/{mid}/lineup"
        stats_url  = f"{UR_BASE}/app/public/index.php/match/{slug}/{mid}/chart"
        lh = fetch_html(lineup_url, session)
        sh = fetch_html(stats_url, session)
        if lh:
            for r in parse_match_lineup(lh):
                r["match_id"] = mid
                r["match_slug"] = slug
                lineup_rows.append(r)
        if sh:
            for r in parse_match_team_stats(sh):
                r["match_id"] = mid
                r["match_slug"] = slug
                stat_rows.append(r)
    log.info("collected %d lineup rows, %d stat rows", len(lineup_rows), len(stat_rows))
    if lineup_rows == [] and match_ids:
        log.warning("no lineup rows parsed across %d matches — check parser/page structure", len(match_ids))
    if stat_rows == [] and match_ids:
        log.warning("no team-stat rows parsed across %d matches — UR /chart pages appear to be "
                    "client-rendered (no stats in raw HTML); a different source is needed.",
                    len(match_ids))
    return lineup_rows, stat_rows


# ---------------------------------------------------------------------------
# 6. CSV writers
# ---------------------------------------------------------------------------

def write_squads(squads: list[dict]) -> None:
    write_csv(
        OUT_DIR / "fantasy_squads.csv",
        [{
            "id": s["id"],
            "name": s["name"],
            "abbreviation": s["abbreviation"],
            "ur_team_slug": SQUAD_ABBR_TO_UR_SLUG.get(s["abbreviation"], ""),
            "badge_url": f"{FANTASY_MEDIA}/{s['badge']}"
                          if s.get("badge") else "",
        } for s in squads],
        ["id", "name", "abbreviation", "ur_team_slug", "badge_url"],
    )


def write_rounds_and_fixtures(rounds: list[dict]) -> None:
    round_rows, fixture_rows = [], []
    for r in rounds:
        round_rows.append({
            "id": r["id"],
            "number": r["number"],
            "status": r["status"],
            "is_locked_for_updating": r["isLockedForUpdating"],
            "start_date": r["startDate"],
            "end_date": r["endDate"],
            "fixture_count": len(r.get("tournaments", [])),
        })
        for t in r.get("tournaments", []):
            fixture_rows.append({
                "fixture_id": t["id"],
                "round_id": r["id"],
                "round_number": r["number"],
                "date": t["date"],
                "status": t["status"],
                "venue_id": t.get("venueId"),
                "venue_name": t.get("venueName"),
                "home_squad_id": t["homeSquadId"],
                "home_squad_abbr": t["homeSquadAbbr"],
                "home_squad_name": t["homeSquadName"],
                "home_score": t.get("homeScore"),
                "away_squad_id": t["awaySquadId"],
                "away_squad_abbr": t["awaySquadAbbr"],
                "away_squad_name": t["awaySquadName"],
                "away_score": t.get("awayScore"),
            })
    write_csv(OUT_DIR / "fantasy_rounds.csv", round_rows,
              ["id", "number", "status", "is_locked_for_updating",
               "start_date", "end_date", "fixture_count"])
    write_csv(OUT_DIR / "fantasy_fixtures.csv", fixture_rows,
              ["fixture_id", "round_id", "round_number", "date", "status",
               "venue_id", "venue_name",
               "home_squad_id", "home_squad_abbr", "home_squad_name",
               "home_score",
               "away_squad_id", "away_squad_abbr", "away_squad_name",
               "away_score"])


def write_players(
    players: list[dict],
    squads_by_id: dict[int, dict],
    bios: dict[str, dict],
    player_id_to_slug: dict[int, str],
) -> None:
    rows = []
    for p in players:
        stats = p.get("stats") or {}
        squad = squads_by_id.get(p.get("squadId"), {})
        slug = player_id_to_slug.get(p["id"], "")
        bio = bios.get(slug, {}) if slug else {}
        rows.append({
            "id": p["id"],
            "feed_id": p.get("feedId"),
            "first_name": p.get("firstName"),
            "last_name": p.get("lastName"),
            "position": p.get("position"),
            "squad_id": p.get("squadId"),
            "squad_name": squad.get("name", ""),
            "squad_abbr": squad.get("abbreviation", ""),
            "cost": p.get("cost"),
            "status": p.get("status"),
            "is_locked": p.get("isLocked"),
            "image_profile_url": f"{FANTASY_MEDIA}/{p['imageProfile']}"
                                  if p.get("imageProfile") else "",
            "image_pitch_url": f"{FANTASY_MEDIA}/{p['imagePitch']}"
                                if p.get("imagePitch") else "",
            "avg_points": stats.get("avgPoints"),
            "total_points": stats.get("totalPoints"),
            "last_round_points": stats.get("lastRoundPoints"),
            "position_rank": stats.get("positionRank"),
            "next_fixture_id": stats.get("nextFixture"),
            # Joined Ultimate Rugby bio fields (blank if no match found)
            "ur_slug": slug,
            "ur_id": bio.get("ur_id"),
            "dob": bio.get("dob", ""),
            "height_m": bio.get("height_m"),
            "weight_kg": bio.get("weight_kg"),
            "ur_position": bio.get("position", ""),
            "ur_image_url": bio.get("image_url", ""),
        })
    write_csv(
        OUT_DIR / "fantasy_players.csv",
        rows,
        ["id", "feed_id", "first_name", "last_name", "position",
         "squad_id", "squad_name", "squad_abbr",
         "cost", "status", "is_locked",
         "image_profile_url", "image_pitch_url",
         "avg_points", "total_points", "last_round_points",
         "position_rank", "next_fixture_id",
         "ur_slug", "ur_id", "dob", "height_m", "weight_kg",
         "ur_position", "ur_image_url"],
    )


def write_player_bios(bios: dict[str, dict]) -> None:
    rows = []
    for slug, b in bios.items():
        rows.append({
            "slug": slug,
            "ur_id": b.get("ur_id"),
            "name": b.get("name", ""),
            "dob": b.get("dob", ""),
            "height_m": b.get("height_m"),
            "weight_kg": b.get("weight_kg"),
            "position": b.get("position", ""),
            "image_url": b.get("image_url", ""),
            "bio": b.get("bio", ""),
        })
    write_csv(
        OUT_DIR / "ur_player_bios.csv",
        rows,
        ["slug", "ur_id", "name", "dob", "height_m", "weight_kg",
         "position", "image_url", "bio"],
    )


def write_player_career(bios: dict[str, dict]) -> None:
    rows = []
    for slug, b in bios.items():
        for c in b.get("career", []):
            rows.append({
                "slug": slug,
                "team": c.get("team", ""),
                "position": c.get("position", ""),
                "year_start": c.get("year_start"),
                "year_end": c.get("year_end"),
                "years": c.get("years", ""),
            })
    write_csv(
        OUT_DIR / "ur_player_career.csv",
        rows,
        ["slug", "team", "position", "year_start", "year_end", "years"],
    )


def write_player_round_tables(players: list[dict]) -> None:
    scores, prices, ownership = [], [], []
    for p in players:
        pid = p["id"]
        for rnd, pts in (p.get("stats", {}).get("scores") or {}).items():
            scores.append({"player_id": pid, "round": int(rnd), "points": pts})
        for rnd, price in (p.get("priceHistory") or {}).items():
            prices.append({"player_id": pid, "round": int(rnd), "price": price})
        for rnd, sel in (p.get("selected") or {}).items():
            ownership.append({"player_id": pid, "round": int(rnd),
                              "selected_pct": sel})
    scores.sort(key=lambda r: (r["player_id"], r["round"]))
    prices.sort(key=lambda r: (r["player_id"], r["round"]))
    ownership.sort(key=lambda r: (r["player_id"], r["round"]))
    write_csv(OUT_DIR / "fantasy_player_round_scores.csv", scores,
              ["player_id", "round", "points"])
    write_csv(OUT_DIR / "fantasy_player_round_prices.csv", prices,
              ["player_id", "round", "price"])
    write_csv(OUT_DIR / "fantasy_player_round_ownership.csv", ownership,
              ["player_id", "round", "selected_pct"])


def write_player_season_stats(player_stats: dict[str, dict]) -> None:
    cols = ["player_id", "gamesPlayed", "tries", "assists", "conversions",
            "penalties", "dropGoals", "tackles", "linebreaks",
            "linebreakAssists", "turnovers", "interceptions",
            "defendersBeaten", "lineoutsWon", "scrumsWon", "metresGained",
            "penaltiesConceded", "errors", "yellowCards", "redCards",
            "offloads", "kick5022"]
    rows = []
    for pid, st in player_stats.items():
        row = {"player_id": int(pid)}
        row.update(st)
        rows.append(row)
    rows.sort(key=lambda r: r["player_id"])
    write_csv(OUT_DIR / "fantasy_player_season_stats.csv", rows, cols)


def write_match_lineups(rows: list[dict]) -> None:
    write_csv(
        OUT_DIR / "ur_match_lineups.csv",
        rows,
        ["match_id", "match_slug", "team_side", "role",
         "position", "player_name", "player_slug"],
    )


def write_match_team_stats(rows: list[dict]) -> None:
    write_csv(
        OUT_DIR / "ur_match_team_stats.csv",
        rows,
        ["match_id", "match_slug", "section", "stat_name",
         "home_value", "away_value"],
    )


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(no_bios: bool = False, no_matches: bool = False) -> None:
    ensure_dirs()

    # 1) Fantasy
    fantasy = fetch_fantasy_data()
    squads_by_id = {s["id"]: s for s in fantasy["squads"]}

    # Fantasy-only outputs first so a partial run remains useful.
    write_squads(fantasy["squads"])
    write_rounds_and_fixtures(fantasy["rounds"])
    write_player_round_tables(fantasy["players"])
    write_player_season_stats(fantasy["player_stats"])

    bios: dict[str, dict] = {}
    player_id_to_slug: dict[int, str] = {}
    lineup_rows: list[dict] = []
    stat_rows: list[dict] = []

    with requests.Session() as session:
        squad_slugs = fetch_ur_squad_slugs(session)

        # Fetch match lineups BEFORE matching so lineup-derived slugs feed the matcher
        # (this is what fixes the MOPA gap and most spelling-drift cases).
        if not no_matches:
            match_ids = fetch_match_ids(session)
            lineup_rows, stat_rows = fetch_match_pages(match_ids, session)
            write_match_lineups(lineup_rows)
            write_match_team_stats(stat_rows)
        else:
            # --no-matches: still feed the matcher cached lineups if they exist on
            # disk, so we don't lose the ~48 lineup-augmented PFR↔UR matches
            # (mostly MOPA, plus Polynesian-name spelling drift).
            cached = OUT_DIR / "ur_match_lineups.csv"
            if cached.exists():
                import csv as _csv
                with cached.open("r", encoding="utf-8") as _f:
                    lineup_rows = list(_csv.DictReader(_f))
                log.info("loaded %d cached lineup rows for matcher (--no-matches)",
                         len(lineup_rows))

        player_id_to_slug = match_players_to_slugs(
            fantasy["players"], squads_by_id, squad_slugs, lineup_rows,
        )
        (RAW_DIR / "ur_player_id_to_slug.json").write_text(
            json.dumps({str(k): v for k, v in player_id_to_slug.items()}, indent=2),
            encoding="utf-8",
        )

        if not no_bios:
            wanted_slugs = sorted(set(player_id_to_slug.values()))
            bios = fetch_player_bios(wanted_slugs, session)
            (RAW_DIR / "ur_bios.json").write_text(
                json.dumps(bios, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            write_player_bios(bios)
            write_player_career(bios)

    # Joined fantasy_players.csv (after bios known).
    write_players(fantasy["players"], squads_by_id, bios, player_id_to_slug)

    log.info("bronze scrape complete (output in %s)", OUT_DIR)


def _cli() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-bios", action="store_true",
                    help="Skip Ultimate Rugby player profile scrape")
    ap.add_argument("--no-matches", action="store_true",
                    help="Skip Ultimate Rugby lineup/stats scrape")
    args = ap.parse_args()
    main(no_bios=args.no_bios, no_matches=args.no_matches)


if __name__ == "__main__":
    _cli()