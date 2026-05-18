"""Build silver-layer tables from bronze CSVs.

Outputs (under data/silver/):
    fixtures.parquet       — typed fixture rows
    players_dim.parquet    — player dimension enriched with UR bio + career features
    team_rounds.parquet    — one row per (squad_id, round) with points scored/conceded + team fantasy total
    player_rounds.parquet  — main player x round panel with fixture context, per-round
                              scores/prices/ownership, AND lineup_role (start/bench/dnp)
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.scraping.bronze import (
    SQUAD_ABBR_TO_UR_SLUG,
    UR_TEAM_SLUG_TO_ABBR,
    parse_match_teams,
)
from src.utils.logger import get_logger
from src.utils.paths import BRONZE_DIR, ROOT_DIR, SILVER_DIR

log = get_logger(__name__)


def _load_stadium_config() -> dict:
    """Stadium + team-home GPS coords for travel-distance features."""
    p = ROOT_DIR / "data" / "stadium_locations.json"
    if not p.exists():
        log.warning("stadium_locations.json not found; travel features will be NaN")
        return {"venues": {}, "team_home": {}}
    return json.loads(p.read_text(encoding="utf-8"))


def _haversine_km(lat1, lon1, lat2, lon2):
    """Vectorised great-circle distance in km. NaN-safe."""
    R = 6371.0  # Earth radius km
    lat1 = np.radians(np.asarray(lat1, dtype="float64"))
    lon1 = np.radians(np.asarray(lon1, dtype="float64"))
    lat2 = np.radians(np.asarray(lat2, dtype="float64"))
    lon2 = np.radians(np.asarray(lon2, dtype="float64"))
    dphi = lat2 - lat1
    dlam = lon2 - lon1
    a = np.sin(dphi / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlam / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(a))


def _add_travel_features(
    panel: pd.DataFrame,
    fixtures: pd.DataFrame,
    squads: pd.DataFrame,
) -> pd.DataFrame:
    """Add own_travel_km, opp_travel_km, travel_advantage_km to player_rounds.

    own_travel_km     = haversine(this team's home, venue)             (0 for home games)
    opp_travel_km     = haversine(opponent's home, venue)              (0 if opp is home)
    travel_advantage_km = opp_travel - own_travel                       (+ve = we have less travel)
    """
    cfg = _load_stadium_config()
    if not cfg.get("venues") or not cfg.get("team_home"):
        return panel

    # venue_id -> (lat, lon) via venue_name
    venue_latlon: dict[int, tuple[float, float]] = {}
    for row in fixtures.itertuples(index=False):
        v = cfg["venues"].get(row.venue_name)
        if v is not None:
            venue_latlon[int(row.venue_id)] = (float(v["lat"]), float(v["lon"]))

    # squad_id -> (home_lat, home_lon) via squad_abbr
    abbr_by_id = squads.set_index("id")["abbreviation"].to_dict()
    home_latlon: dict[int, tuple[float, float]] = {}
    for sid, abbr in abbr_by_id.items():
        h = cfg["team_home"].get(abbr)
        if h is not None:
            home_latlon[int(sid)] = (float(h["lat"]), float(h["lon"]))

    # Map into the panel
    def _map_lat(d, key):
        return panel[key].map(lambda x: d.get(x, (np.nan, np.nan))[0] if pd.notna(x) else np.nan)

    def _map_lon(d, key):
        return panel[key].map(lambda x: d.get(x, (np.nan, np.nan))[1] if pd.notna(x) else np.nan)

    venue_lat = _map_lat(venue_latlon, "venue_id")
    venue_lon = _map_lon(venue_latlon, "venue_id")
    own_lat = _map_lat(home_latlon, "squad_id")
    own_lon = _map_lon(home_latlon, "squad_id")
    opp_lat = _map_lat(home_latlon, "opponent_squad_id")
    opp_lon = _map_lon(home_latlon, "opponent_squad_id")

    panel = panel.copy()
    panel["own_travel_km"] = _haversine_km(own_lat, own_lon, venue_lat, venue_lon)
    panel["opp_travel_km"] = _haversine_km(opp_lat, opp_lon, venue_lat, venue_lon)
    panel["travel_advantage_km"] = panel["opp_travel_km"] - panel["own_travel_km"]

    n_with_travel = panel["own_travel_km"].notna().sum()
    log.info("travel features computed for %d/%d panel rows", n_with_travel, len(panel))
    return panel

# Teams whose career-entries count as international appearances.
# Anything else (clubs, NPC sides, age-grade non-national) is excluded.
INTL_TEAMS = {
    "New Zealand", "New Zealand U20", "New Zealand U20's", "All Blacks", "All Blacks XV",
    "Maori All Blacks", "Australia", "Australia U20's", "Australia A", "Australia U20",
    "Fiji", "Fiji U20", "Tonga", "Samoa",
    "England", "Wales", "Scotland", "Ireland", "France", "South Africa", "Argentina",
}


def _load_bronze() -> dict[str, pd.DataFrame]:
    def _maybe(name: str) -> pd.DataFrame | None:
        p = BRONZE_DIR / name
        return pd.read_csv(p) if p.exists() else None

    return {
        "squads":    pd.read_csv(BRONZE_DIR / "fantasy_squads.csv"),
        "rounds":    pd.read_csv(BRONZE_DIR / "fantasy_rounds.csv"),
        "fixtures":  pd.read_csv(BRONZE_DIR / "fantasy_fixtures.csv"),
        "players":   pd.read_csv(BRONZE_DIR / "fantasy_players.csv"),
        "scores":    pd.read_csv(BRONZE_DIR / "fantasy_player_round_scores.csv"),
        "prices":    pd.read_csv(BRONZE_DIR / "fantasy_player_round_prices.csv"),
        "ownership": pd.read_csv(BRONZE_DIR / "fantasy_player_round_ownership.csv"),
        "bios":      _maybe("ur_player_bios.csv"),
        "career":    _maybe("ur_player_career.csv"),
        "lineups":   _maybe("ur_match_lineups.csv"),
    }


# Historical priors from rugbypy were removed: the rugbypy data repo is
# downstream of a private S3 bucket we don't control, so we cannot guarantee
# we can keep pulling fresh data going forward. Per project policy, we don't
# integrate features we can't reliably refresh. To re-enable, build a scraper
# against the actual upstream (likely ESPN StatsGuru) and write our own
# `historical_priors` table to silver, then reinstate this loader.


def build_fixtures(fixtures: pd.DataFrame) -> pd.DataFrame:
    df = fixtures.copy()
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    return df


def _compute_bio_features(bios: pd.DataFrame, today: pd.Timestamp) -> pd.DataFrame:
    df = bios.copy()
    df["dob_parsed"] = pd.to_datetime(df["dob"], errors="coerce")
    df["age_yrs"] = (today - df["dob_parsed"]).dt.days / 365.25
    # Drop nonsense DOBs (UR placeholders / data-entry errors).
    df.loc[(df["age_yrs"] < 16) | (df["age_yrs"] > 45), "age_yrs"] = pd.NA
    df["bmi"] = df["weight_kg"] / (df["height_m"] ** 2)
    return df[["slug", "age_yrs", "height_m", "weight_kg", "bmi"]]


def _compute_career_features(career: pd.DataFrame) -> pd.DataFrame:
    df = career.copy()
    df["year_end_eff"] = df["year_end"].fillna(2026).astype(int)
    df["is_intl"] = df["team"].isin(INTL_TEAMS)
    feats = df.groupby("slug").agg(
        career_first_year=("year_start", "min"),
        career_last_year=("year_end_eff", "max"),
        n_career_teams=("team", "nunique"),
        has_intl=("is_intl", "max"),
        n_intl_appearances=("is_intl", "sum"),
    ).reset_index()
    feats["career_years"] = feats["career_last_year"] - feats["career_first_year"]
    feats["has_intl"] = feats["has_intl"].astype("Int64")
    return feats[["slug", "career_years", "n_career_teams", "has_intl", "n_intl_appearances"]]


def build_players_dim(
    players: pd.DataFrame,
    bios: pd.DataFrame | None,
    career: pd.DataFrame | None,
    today: pd.Timestamp,
) -> pd.DataFrame:
    base = players[[
        "id", "feed_id", "first_name", "last_name", "position",
        "squad_id", "squad_abbr", "squad_name", "cost", "status", "ur_slug",
    ]].rename(columns={"id": "player_id"})

    if bios is not None and len(bios):
        bio_feats = _compute_bio_features(bios, today=today)
        base = base.merge(bio_feats, left_on="ur_slug", right_on="slug", how="left").drop(columns=["slug"])

    if career is not None and len(career):
        career_feats = _compute_career_features(career)
        base = base.merge(career_feats, left_on="ur_slug", right_on="slug", how="left").drop(columns=["slug"])

    # Fill missing intl counts/flags with 0 for players with no career table entry
    # (they may exist on UR but have no listed career — treat as no intl history).
    if "has_intl" in base.columns:
        base["has_intl"] = base["has_intl"].fillna(0).astype("Int64")
        base["n_intl_appearances"] = base["n_intl_appearances"].fillna(0).astype("Int64")
        base["n_career_teams"] = base["n_career_teams"].fillna(0).astype("Int64")

    return base


def build_team_rounds(fixtures: pd.DataFrame, scores: pd.DataFrame, players: pd.DataFrame) -> pd.DataFrame:
    f = fixtures.copy()
    home = f[["round_number", "home_squad_id", "away_squad_id", "home_score", "away_score", "date"]].rename(
        columns={
            "round_number": "round",
            "home_squad_id": "squad_id",
            "away_squad_id": "opponent_squad_id",
            "home_score": "team_points_scored",
            "away_score": "team_points_conceded",
        }
    )
    home["is_home"] = True
    away = f[["round_number", "away_squad_id", "home_squad_id", "away_score", "home_score", "date"]].rename(
        columns={
            "round_number": "round",
            "away_squad_id": "squad_id",
            "home_squad_id": "opponent_squad_id",
            "away_score": "team_points_scored",
            "home_score": "team_points_conceded",
        }
    )
    away["is_home"] = False
    teams = pd.concat([home, away], ignore_index=True)
    teams["date"] = pd.to_datetime(teams["date"], utc=True, errors="coerce")

    scores_squad = scores.merge(
        players[["id", "squad_id"]].rename(columns={"id": "player_id"}),
        on="player_id", how="left",
    )
    team_fantasy = (
        scores_squad.groupby(["squad_id", "round"], as_index=False)["points"]
        .sum()
        .rename(columns={"points": "team_fantasy_points"})
    )
    teams = teams.merge(team_fantasy, on=["squad_id", "round"], how="left")
    return teams.sort_values(["squad_id", "round"]).reset_index(drop=True)


def build_lineup_role_map(
    lineups: pd.DataFrame,
    fixtures: pd.DataFrame,
    squads: pd.DataFrame,
    players: pd.DataFrame,
) -> tuple[pd.DataFrame, set[tuple[int, int]]]:
    """Map UR lineup rows -> (player_id, round, lineup_role).

    Returns:
        - DataFrame with columns [player_id, round, lineup_role]
        - Set of (round, squad_id) pairs for which lineup data exists.
          Used downstream to decide which panel rows can safely default to 'dnp'.
    """
    ur_to_abbr = UR_TEAM_SLUG_TO_ABBR
    abbr_to_squad_id = squads.set_index("abbreviation")["id"].to_dict()
    slug_to_player_id = (
        players[players["ur_slug"].notna()]
        .drop_duplicates("ur_slug")
        .set_index("ur_slug")["id"]
        .to_dict()
    )

    # Parse home/away team from each unique match_slug.
    match_meta = lineups[["match_id", "match_slug"]].drop_duplicates().copy()
    parsed = match_meta["match_slug"].apply(lambda s: pd.Series(parse_match_teams(s), index=["home_slug", "away_slug"]))
    match_meta = pd.concat([match_meta.reset_index(drop=True), parsed.reset_index(drop=True)], axis=1)
    match_meta["home_squad_id"] = match_meta["home_slug"].map(lambda s: abbr_to_squad_id.get(ur_to_abbr.get(s, ""), None))
    match_meta["away_squad_id"] = match_meta["away_slug"].map(lambda s: abbr_to_squad_id.get(ur_to_abbr.get(s, ""), None))
    match_meta = match_meta.dropna(subset=["home_squad_id", "away_squad_id"]).copy()
    match_meta["home_squad_id"] = match_meta["home_squad_id"].astype(int)
    match_meta["away_squad_id"] = match_meta["away_squad_id"].astype(int)

    # Join to fantasy fixtures to get the round number.
    fx_keys = fixtures[["round_number", "home_squad_id", "away_squad_id"]].drop_duplicates()
    match_meta = match_meta.merge(fx_keys, on=["home_squad_id", "away_squad_id"], how="inner")

    log.info("matched %d UR matches to fantasy fixtures (of %d total UR matches)",
             match_meta["match_id"].nunique(), lineups["match_id"].nunique())

    # Attach round + squad_id to each lineup row.
    enriched = lineups.merge(
        match_meta[["match_id", "home_squad_id", "away_squad_id", "round_number"]],
        on="match_id", how="inner",
    )
    enriched["squad_id"] = enriched.apply(
        lambda r: r["home_squad_id"] if r["team_side"] == "home" else r["away_squad_id"], axis=1,
    ).astype(int)
    enriched["player_id"] = enriched["player_slug"].map(slug_to_player_id)
    enriched = enriched.dropna(subset=["player_id"]).copy()
    enriched["player_id"] = enriched["player_id"].astype(int)

    role_map = (
        enriched[["player_id", "round_number", "role", "position"]]
        .rename(columns={
            "round_number": "round",
            "role": "lineup_role",
            "position": "ur_position_lineup",
        })
        .drop_duplicates(subset=["player_id", "round"])
    )

    covered = set(zip(enriched["round_number"].astype(int), enriched["squad_id"]))
    return role_map, covered


def build_player_rounds(
    fixtures: pd.DataFrame,
    scores: pd.DataFrame,
    prices: pd.DataFrame,
    ownership: pd.DataFrame,
    players: pd.DataFrame,
    role_map: pd.DataFrame | None,
    covered: set[tuple[int, int]] | None,
) -> pd.DataFrame:
    f = fixtures.rename(columns={"round_number": "round"}).copy()
    f["date"] = pd.to_datetime(f["date"], utc=True, errors="coerce")

    home_ctx = f[["round", "home_squad_id", "away_squad_id", "venue_id", "date", "fixture_id"]].rename(
        columns={"home_squad_id": "squad_id", "away_squad_id": "opponent_squad_id"}
    )
    home_ctx["is_home"] = True
    away_ctx = f[["round", "away_squad_id", "home_squad_id", "venue_id", "date", "fixture_id"]].rename(
        columns={"away_squad_id": "squad_id", "home_squad_id": "opponent_squad_id"}
    )
    away_ctx["is_home"] = False
    fixture_ctx = pd.concat([home_ctx, away_ctx], ignore_index=True)

    panel = players[["id", "squad_id", "position"]].rename(columns={"id": "player_id"})
    panel = panel.merge(fixture_ctx[["squad_id", "round"]].drop_duplicates(), on="squad_id", how="inner")
    panel = panel.sort_values(["player_id", "round"]).reset_index(drop=True)

    panel = panel.merge(scores,    on=["player_id", "round"], how="left")
    panel = panel.merge(prices,    on=["player_id", "round"], how="left")
    panel = panel.merge(ownership, on=["player_id", "round"], how="left")
    panel = panel.merge(fixture_ctx, on=["squad_id", "round"], how="left")

    # Attach lineup role. For panel rows where (round, squad_id) has lineup coverage
    # but the player isn't listed, default to 'dnp'. For rounds without any coverage
    # (e.g. future rounds), leave lineup_role NaN so it doesn't bias lagged features.
    if role_map is not None and len(role_map):
        panel = panel.merge(role_map, on=["player_id", "round"], how="left")
        if covered:
            covered_df = pd.DataFrame(list(covered), columns=["round", "squad_id"])
            covered_df["_covered"] = True
            panel = panel.merge(covered_df, on=["round", "squad_id"], how="left")
            mask = panel["_covered"].fillna(False) & panel["lineup_role"].isna()
            panel.loc[mask, "lineup_role"] = "dnp"
            panel = panel.drop(columns=["_covered"])
    else:
        panel["lineup_role"] = pd.NA

    return panel


def main() -> None:
    log.info("=== silver layer ===")
    SILVER_DIR.mkdir(parents=True, exist_ok=True)
    bronze = _load_bronze()
    today = pd.Timestamp("2026-05-18")

    fixtures_clean = build_fixtures(bronze["fixtures"])
    fixtures_clean.to_parquet(SILVER_DIR / "fixtures.parquet", index=False)
    log.info("wrote silver/fixtures.parquet (%d rows)", len(fixtures_clean))

    players_dim = build_players_dim(
        bronze["players"], bios=bronze["bios"], career=bronze["career"], today=today,
    )
    players_dim.to_parquet(SILVER_DIR / "players_dim.parquet", index=False)
    log.info("wrote silver/players_dim.parquet (%d rows, %d cols)", len(players_dim), players_dim.shape[1])

    team_rounds = build_team_rounds(bronze["fixtures"], bronze["scores"], bronze["players"])
    team_rounds.to_parquet(SILVER_DIR / "team_rounds.parquet", index=False)
    log.info("wrote silver/team_rounds.parquet (%d rows)", len(team_rounds))

    role_map, covered = (None, None)
    if bronze["lineups"] is not None and len(bronze["lineups"]):
        role_map, covered = build_lineup_role_map(
            bronze["lineups"], bronze["fixtures"], bronze["squads"], bronze["players"],
        )

    player_rounds = build_player_rounds(
        bronze["fixtures"], bronze["scores"], bronze["prices"],
        bronze["ownership"], bronze["players"], role_map, covered,
    )
    player_rounds = _add_travel_features(player_rounds, bronze["fixtures"], bronze["squads"])
    player_rounds.to_parquet(SILVER_DIR / "player_rounds.parquet", index=False)
    log.info("wrote silver/player_rounds.parquet (%d rows, %d cols)", len(player_rounds), player_rounds.shape[1])

    if "lineup_role" in player_rounds.columns:
        role_counts = player_rounds["lineup_role"].value_counts(dropna=False).to_dict()
        log.info("lineup_role distribution: %s", role_counts)


if __name__ == "__main__":
    main()
