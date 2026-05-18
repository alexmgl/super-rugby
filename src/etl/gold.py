"""Build gold-layer feature panel for ML.

LEAKAGE POLICY: every time-varying feature at round r uses ONLY data from rounds < r.
Pattern: groupby(player_id).transform(lambda s: s.shift(1).rolling(...).<agg>())
                                                ^^^^^^^^^^^ — shift FIRST, then roll.

Fixture-context features (is_home, opponent, days_rest) are pre-round-known and used at round r.

Output: data/gold/feature_panel.parquet — one row per (player_id, round), with target_points + features.
"""

from __future__ import annotations

import pandas as pd

from src.utils.logger import get_logger
from src.utils.paths import GOLD_DIR, SILVER_DIR

log = get_logger(__name__)


def add_player_form_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["player_id", "round"]).reset_index(drop=True).copy()
    g = df.groupby("player_id", sort=False)["points"]

    df["pts_lag1"] = g.shift(1)
    df["pts_lag2"] = g.shift(2)
    df["pts_lag3"] = g.shift(3)
    df["pts_roll3_mean"] = g.transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    df["pts_roll5_mean"] = g.transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
    df["pts_roll3_std"]  = g.transform(lambda s: s.shift(1).rolling(3, min_periods=2).std())
    df["pts_roll5_max"]  = g.transform(lambda s: s.shift(1).rolling(5, min_periods=1).max())
    df["pts_roll5_min"]  = g.transform(lambda s: s.shift(1).rolling(5, min_periods=1).min())

    # Season-to-date totals (lagged so they include only rounds strictly before current).
    # fillna(0) before cumsum so that a DNP round contributes 0 (rather than propagating NaN).
    df["pts_season_total"] = g.transform(lambda s: s.shift(1).fillna(0).cumsum())

    played = df["points"].notna().astype(int)
    df["games_played"] = (
        played.groupby(df["player_id"], sort=False)
              .transform(lambda s: s.shift(1, fill_value=0).cumsum())
    )
    df["pts_per_game_season"] = df["pts_season_total"] / df["games_played"].where(df["games_played"] > 0)
    return df


def add_market_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["player_id", "round"]).reset_index(drop=True).copy()
    gp = df.groupby("player_id", sort=False)["price"]
    go = df.groupby("player_id", sort=False)["selected_pct"]

    # All market features are lagged: round-r feature uses pre-round-r data.
    df["price_lag1"] = gp.shift(1)
    df["price_delta_3"] = gp.shift(1) - gp.shift(4)   # change between rounds r-1 and r-4
    df["ownership_lag1"] = go.shift(1)
    df["ownership_delta_3"] = go.shift(1) - go.shift(4)
    return df


def add_schedule_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["squad_id", "round"]).reset_index(drop=True).copy()
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    prev_date = df.groupby("squad_id", sort=False)["date"].shift(1)
    df["days_rest"] = (df["date"] - prev_date).dt.days.astype("Float64")
    return df


def add_lineup_features(df: pd.DataFrame) -> pd.DataFrame:
    """Lagged role indicators derived from ur_match_lineups.

    All features at round r use ONLY lineup_role from rounds < r (shift(1) before rolling).
    """
    df = df.sort_values(["player_id", "round"]).reset_index(drop=True).copy()
    # Per-row binary role indicators (raw, not yet lagged).
    role = df.get("lineup_role")
    if role is None:
        # No lineup data — set all role features to NaN.
        for c in ["is_starter_last_match", "appeared_last_match",
                  "start_share_l3", "start_share_l5",
                  "bench_share_l3", "dnp_share_l3"]:
            df[c] = pd.NA
        return df

    df["_is_start"] = (role == "start").astype("Float64").where(role.notna())
    df["_is_bench"] = (role == "bench").astype("Float64").where(role.notna())
    df["_is_dnp"]   = (role == "dnp").astype("Float64").where(role.notna())
    df["_appeared"] = ((role == "start") | (role == "bench")).astype("Float64").where(role.notna())

    g_start    = df.groupby("player_id", sort=False)["_is_start"]
    g_bench    = df.groupby("player_id", sort=False)["_is_bench"]
    g_dnp      = df.groupby("player_id", sort=False)["_is_dnp"]
    g_appear   = df.groupby("player_id", sort=False)["_appeared"]

    df["is_starter_last_match"] = g_start.shift(1)
    df["appeared_last_match"]   = g_appear.shift(1)
    df["start_share_l3"] = g_start.transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    df["start_share_l5"] = g_start.transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
    df["bench_share_l3"] = g_bench.transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    df["dnp_share_l3"]   = g_dnp.transform(  lambda s: s.shift(1).rolling(3, min_periods=1).mean())

    df = df.drop(columns=["_is_start", "_is_bench", "_is_dnp", "_appeared"])
    return df


FANTASY_FORWARD_POSITIONS = {"prop", "hooker", "lock", "loose_forward"}


def add_team_composition_features(
    panel: pd.DataFrame,
    players_dim: pd.DataFrame,
    team_rounds: pd.DataFrame,
) -> pd.DataFrame:
    """Team-composition features at round r derived from each team's most recent
    starting XV in rounds < r.

    Implementation:
      1. Compute per-(squad, round) aggregates from the actual starting XV at that round.
      2. Use team_rounds as the "every scheduled round per squad" backbone — merge aggs.
      3. shift(1) within squad + ffill: each round inherits the team's most recent prior
         starting-XV aggregates. Handles byes and future rounds (e.g. round 15) cleanly.
      4. Merge to panel for own team and opponent; compute team − opp differentials.

    Bio fields (weight/height/age) are imputed by position mean before aggregation so
    teams with patchy bio coverage (e.g. MOPA) don't get NaN sums.
    """
    if "lineup_role" not in panel.columns:
        return panel
    starts = panel[panel["lineup_role"] == "start"].copy()
    if starts.empty:
        return panel

    # Bio columns are already on the panel (joined by add_static_player_features).
    # Impute NaN with position-mean computed from the full players_dim so MOPA-like
    # patchy coverage doesn't truncate pack sums.
    for col in ["height_m", "weight_kg", "age_yrs"]:
        if col not in starts.columns or col not in players_dim.columns:
            continue
        mean_map = players_dim.groupby("position")[col].mean().to_dict()
        starts[col] = starts[col].fillna(starts["position"].astype(str).map(mean_map))

    starts["is_forward"] = starts["position"].isin(FANTASY_FORWARD_POSITIONS)
    # has_intl can be nullable Int — cast to float for aggregation
    if "has_intl" in starts.columns:
        starts["has_intl"] = starts["has_intl"].astype("Float64")

    # Per-(squad, round) actual aggregates
    forwards = starts[starts["is_forward"]]
    pack = forwards.groupby(["squad_id", "round"], as_index=False).agg(
        pack_weight=("weight_kg", "sum"),
        pack_height=("height_m", "mean"),
        pack_age=("age_yrs", "mean"),
        pack_intl=("has_intl", "sum"),
    )
    backs = starts[~starts["is_forward"]]
    back_aggs = backs.groupby(["squad_id", "round"], as_index=False).agg(
        back_weight_mean=("weight_kg", "mean"),
        back_age=("age_yrs", "mean"),
        back_intl=("has_intl", "sum"),
    )
    xv = starts.groupby(["squad_id", "round"], as_index=False).agg(
        xv_intl=("has_intl", "sum"),
        xv_age=("age_yrs", "mean"),
    )
    aggs = pack.merge(back_aggs, on=["squad_id", "round"], how="outer") \
               .merge(xv, on=["squad_id", "round"], how="outer")

    # Lag at the team level using team_rounds as the scheduled-rounds backbone
    backbone = team_rounds[["squad_id", "round"]].drop_duplicates()
    feats = backbone.merge(aggs, on=["squad_id", "round"], how="left")
    feats = feats.sort_values(["squad_id", "round"]).reset_index(drop=True)
    agg_cols = [c for c in feats.columns if c not in ("squad_id", "round")]
    for col in agg_cols:
        feats[col] = feats.groupby("squad_id", sort=False)[col].transform(lambda s: s.shift(1).ffill())

    # Rename to *_lag
    rename_team = {c: f"team_{c}_lag" for c in agg_cols}
    feats_team = feats.rename(columns=rename_team)

    panel = panel.merge(feats_team, on=["squad_id", "round"], how="left")

    # Opponent
    rename_opp = {"squad_id": "opponent_squad_id"}
    rename_opp.update({f"team_{c}_lag": f"opp_{c}_lag" for c in agg_cols})
    feats_opp = feats_team.rename(columns=rename_opp)
    panel = panel.merge(feats_opp, on=["opponent_squad_id", "round"], how="left")

    # Differentials (team − opp), only for features we'd genuinely contrast
    diff_cols = ["pack_weight", "pack_height", "pack_intl", "xv_intl", "xv_age", "back_intl"]
    for c in diff_cols:
        t_col, o_col = f"team_{c}_lag", f"opp_{c}_lag"
        if t_col in panel.columns and o_col in panel.columns:
            panel[f"{c}_advantage"] = panel[t_col] - panel[o_col]

    return panel


def add_fixture_style_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cheap derived features from already-lagged team/opp form rolls."""
    if "team_attack_3" in df.columns and "opp_attack_3" in df.columns:
        df["fixture_total_implied"] = df["team_attack_3"] + df["opp_attack_3"]
    if all(c in df.columns for c in ["team_attack_3", "team_defense_3", "opp_attack_3", "opp_defense_3"]):
        df["fixture_margin_implied"] = (
            (df["team_attack_3"] - df["team_defense_3"])
            - (df["opp_attack_3"] - df["opp_defense_3"])
        )
    return df


def add_positional_features(df: pd.DataFrame) -> pd.DataFrame:
    """Static handle for forward/back interactions in tree splits."""
    df["is_forward"] = df["position"].astype(str).isin(FANTASY_FORWARD_POSITIONS).astype("Int64")
    return df


def add_static_player_features(df: pd.DataFrame, players_dim: pd.DataFrame) -> pd.DataFrame:
    """Join static UR bio + career features onto each panel row.

    All features here are static per player (no time dimension, no leakage).
    """
    static_cols = [
        "player_id", "age_yrs", "height_m", "weight_kg", "bmi",
        "career_years", "n_career_teams", "has_intl", "n_intl_appearances",
        "status",  # carried through for the round-15 status filter (not used as a feature)
        "cost",    # the LATEST PFR cost; used as the live-price fallback for future rounds
    ]
    available = [c for c in static_cols if c in players_dim.columns]
    if "player_id" not in available:
        return df
    return df.merge(players_dim[available].drop_duplicates("player_id"), on="player_id", how="left")


def add_team_features(df: pd.DataFrame, team_rounds: pd.DataFrame) -> pd.DataFrame:
    tr = team_rounds.sort_values(["squad_id", "round"]).reset_index(drop=True).copy()
    gp_scored   = tr.groupby("squad_id", sort=False)["team_points_scored"]
    gp_conceded = tr.groupby("squad_id", sort=False)["team_points_conceded"]
    gp_fantasy  = tr.groupby("squad_id", sort=False)["team_fantasy_points"]

    tr["team_attack_3"]  = gp_scored.transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    tr["team_defense_3"] = gp_conceded.transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    tr["team_fantasy_3"] = gp_fantasy.transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())

    feats = tr[["squad_id", "round", "team_attack_3", "team_defense_3", "team_fantasy_3"]]
    df = df.merge(feats, on=["squad_id", "round"], how="left")

    opp = feats.rename(columns={
        "squad_id": "opponent_squad_id",
        "team_attack_3": "opp_attack_3",
        "team_defense_3": "opp_defense_3",
        "team_fantasy_3": "opp_fantasy_3",
    })
    df = df.merge(opp, on=["opponent_squad_id", "round"], how="left")
    return df


def build_feature_panel(
    player_rounds: pd.DataFrame,
    team_rounds: pd.DataFrame,
    players_dim: pd.DataFrame,
) -> pd.DataFrame:
    df = player_rounds.copy()
    df = add_player_form_features(df)
    df = add_market_features(df)
    df = add_schedule_features(df)
    df = add_team_features(df, team_rounds)
    df = add_lineup_features(df)
    df = add_static_player_features(df, players_dim)
    df = add_team_composition_features(df, players_dim, team_rounds)
    df = add_fixture_style_features(df)
    df = add_positional_features(df)

    df["target_points"] = df["points"]

    # XGBoost handles pandas categoricals natively with enable_categorical=True.
    for col in ["position", "squad_id", "opponent_squad_id"]:
        if col in df.columns:
            df[col] = df[col].astype("category")

    # Cast nullable-integer feature columns to float so xgb's NaN handling kicks in.
    for col in ["has_intl", "n_intl_appearances", "n_career_teams", "career_years"]:
        if col in df.columns:
            df[col] = df[col].astype("Float64")

    return df


def main() -> None:
    log.info("=== gold layer ===")
    GOLD_DIR.mkdir(parents=True, exist_ok=True)
    player_rounds = pd.read_parquet(SILVER_DIR / "player_rounds.parquet")
    team_rounds   = pd.read_parquet(SILVER_DIR / "team_rounds.parquet")
    players_dim   = pd.read_parquet(SILVER_DIR / "players_dim.parquet")

    panel = build_feature_panel(player_rounds, team_rounds, players_dim)
    out = GOLD_DIR / "feature_panel.parquet"
    panel.to_parquet(out, index=False)
    log.info("wrote %s (%d rows, %d cols)", out, len(panel), panel.shape[1])


if __name__ == "__main__":
    main()
