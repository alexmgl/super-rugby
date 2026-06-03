"""Walk-forward expanding-window evaluation of the quantile model + MILP optimiser.

For each target round r in 2..14:
  - Train 5 quantile models on rounds < r (excluding rows with unknown lineup_role)
  - Predict P50/60/70/80/90 for round r
  - Run MILP using P60 for team / 2·P80−P60 for captain
  - Look up actual fantasy points for the chosen team
  - Accumulate

Reports a per-round table and cumulative actuals. Also reports an "oracle" upper
bound: the MILP solved with perfect foresight (p60=p80=actual_points).

Round 15 (future): trains on rounds 1..14, generates the next-GW optimal team.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.ml.quantile_model import (
    FEATURES,
    calibration_report,
    ensure_categoricals,
    fit_for_round_calibrated,
    fit_per_position_for_round,
    make_uncond_target,
    predict_quantiles,
    predict_quantiles_per_position,
)
from src.milp.team_optimiser import SCENARIOS, build_optimal_team, compare_scenarios
from src.utils.logger import get_logger
from src.utils.paths import GOLD_DIR, SILVER_DIR


# PFR status values that we should filter OUT of the candidate pool at inference.
BANNED_STATUSES = {"injured", "not-selected", "eliminated"}

# If non-empty, restricts the round-15 candidate pool to ONLY these statuses.
# Once squads are announced, PFR populates "starting" / "bench" on each player.
# Set this to the announced-only set so the optimiser does not consider rotated
# players. Earlier in the week (status="uncertain"), leave as set() and the
# BANNED_STATUSES list above is used instead.
ALLOWED_STATUSES: set[str] = {"starting", "bench"}

# Selective calibration: only calibrate the captain quantile (P80). Team (P60) stays
# raw to preserve mid-tier discrimination — full calibration creates isotonic plateaus
# that collapse mid-tier predictions onto a few discrete levels, pushing the optimiser
# toward a barbell roster (premiums + cheap fillers). Set to None to calibrate all
# quantiles; set to set() to disable calibration entirely.
#
# Empirical (with travel features, 13-round backtest):
#   set()  (no calibration)          : 47.3% of oracle — Fineanganofo captain (correct)
#   {0.8}  (selective P80)           : 47.3% — but isotonic plateau breaks captain pick
# Default: no calibration. Same backtest, better captain on round 15.
CALIBRATE_QUANTILES: set[float] | None = set()

# Which quantile column to use for captain selection.
# A/B-tested 2026-06-01: p90 beat p80 by +1.4pp on full-season walk-forward
# (57.62% vs 56.23% of oracle). The higher ceiling catches boom potential —
# wins like McKenzie@R9 (+12pp) and Fineanganofo@R11 (+8pp) outweigh the losses
# from extra captain churn.
CAPTAIN_QUANTILE_COL = "p90"

# If True, fit one set of quantile models per position. Thin-position positions
# (<50 train rows) fall back to a globally trained model. Empirically tested in
# experiment: see ./walk_forward.py:main() output when toggled.
PER_POSITION = False

log = get_logger(__name__)


def _cost_basis(pred_pool: pd.DataFrame) -> pd.Series:
    """Cost for MILP at target round.

    Fallback chain:
      1. `price`         — observed round-r price (populated for past rounds)
      2. `cost`          — the LATEST PFR cost from the most recent scrape;
                            this is the LIVE price for the next-GW (round 15+)
      3. `price_lag1`    — round (r-1)'s price as a last resort
      4. median of remaining costs

    The `cost` step is critical for round-15+ where `price` isn't yet populated
    in fantasy_player_round_prices.csv but the PFR feed has rolled prices forward
    on the players record.
    """
    cost = pred_pool["price"].copy() if "price" in pred_pool.columns else pd.Series(np.nan, index=pred_pool.index)
    if "cost" in pred_pool.columns:
        cost = cost.fillna(pred_pool["cost"])
    if "price_lag1" in pred_pool.columns:
        cost = cost.fillna(pred_pool["price_lag1"])
    cost = cost.fillna(cost.median())
    return cost


def _load_manual_exclusions(target_round: int) -> set[int]:
    """Read data/manual_exclusions.json and return player_ids excluded for this round.

    Used to override the PFR `status` field when PFR's UI blocks a player but the
    JSON status doesn't reflect it (e.g. rotation rest, weekly squad omission).
    """
    from src.utils.paths import ROOT_DIR
    p = ROOT_DIR / "data" / "manual_exclusions.json"
    if not p.exists():
        return set()
    data = json.loads(p.read_text(encoding="utf-8"))
    out: set[int] = set()
    for entry in data.get("exclusions", []):
        if target_round in entry.get("rounds", []):
            out.add(int(entry["player_id"]))
    return out


def _enrich_team(team: pd.DataFrame, players_dim: pd.DataFrame, pred_pool: pd.DataFrame) -> pd.DataFrame:
    """Attach player names and actual fantasy points (where observed)."""
    team = team.merge(
        players_dim[["player_id", "first_name", "last_name", "squad_abbr"]],
        on="player_id", how="left",
    )
    actuals = pred_pool[["player_id", "target_points"]].dropna(subset=["target_points"]).set_index("player_id")
    team["actual_pts"] = team["player_id"].map(actuals["target_points"])
    team["actual_pts_filled"] = team["actual_pts"].fillna(0)  # DNP -> 0
    team["actual_with_captain"] = np.where(
        team["is_captain"], 2 * team["actual_pts_filled"], team["actual_pts_filled"]
    )
    return team


def _oracle_team(pred_pool: pd.DataFrame) -> pd.DataFrame | None:
    """Upper-bound MILP — solve as if we knew the actuals (p60=p80=actual)."""
    actuals = pred_pool[pred_pool["target_points"].notna()].copy()
    if len(actuals) < 15:
        return None
    actuals["cost"] = _cost_basis(actuals)
    actuals["p60"] = actuals["target_points"]
    actuals["p80"] = actuals["target_points"]
    optim_in = actuals[["player_id", "position", "squad_id", "cost", "p60", "p80"]].dropna()
    if len(optim_in) < 15:
        return None
    try:
        return build_optimal_team(optim_in)
    except Exception as e:
        log.warning("oracle solve failed: %s", e)
        return None


def select_team_for_round(
    panel: pd.DataFrame, target_round: int, players_dim: pd.DataFrame,
    apply_status_filter: bool = False,
    return_models: bool = False,
    filter_to_observed: bool = False,
    budget: float | None = None,
) -> tuple[pd.DataFrame | None, dict | None, pd.DataFrame | None]:
    """Train quantile model on rounds < target_round (with isotonic calibration when
    enough rounds exist), run MILP, return (team, info, oracle).

    apply_status_filter=True drops players currently marked injured/not-selected/eliminated
    from the candidate pool. Use for round-15 inference; leave off for historical walk-forward
    so the status snapshot doesn't retroactively prune past rounds.

    filter_to_observed=True drops players whose target_points are NaN for the target round
    — i.e. drops the players who didn't end up playing. This encodes the assumption that we
    knew the starting lineup at MILP time (lineup announcement before kickoff). Used in
    walk-forward to evaluate the model's signal quality independent of DNP risk.
    """
    panel = ensure_categoricals(panel.copy())
    if "target_uncond" not in panel.columns:
        panel["target_uncond"] = make_uncond_target(panel)

    try:
        if PER_POSITION:
            models, calibrators, info = fit_per_position_for_round(
                panel, target_round, calibrate_quantiles=CALIBRATE_QUANTILES,
            )
        else:
            models, calibrators, info = fit_for_round_calibrated(
                panel, target_round, calibrate_quantiles=CALIBRATE_QUANTILES,
            )
    except ValueError as e:
        log.warning("round %d: %s", target_round, e)
        return None, None, None

    pred_pool = panel[panel["round"] == target_round].copy()
    if pred_pool.empty:
        return None, None, None

    feats = info.get("active_features", FEATURES) if info else FEATURES
    if PER_POSITION:
        qs = predict_quantiles_per_position(models, calibrators, pred_pool, features=feats)
    else:
        qs = predict_quantiles(models, pred_pool[feats], calibrators=calibrators)
    pred = pd.concat([pred_pool.reset_index(drop=True), qs.reset_index(drop=True)], axis=1)
    pred["cost"] = _cost_basis(pred)

    captain_col = CAPTAIN_QUANTILE_COL
    proj_cols = ["player_id", "position", "squad_id", "cost", "p60", captain_col]
    if captain_col == "p80":
        optim_in = pred[proj_cols].dropna()
    else:
        # Carry both p80 (for enrichment/logging) and the chosen captain column.
        optim_in = pred[proj_cols + (["p80"] if "p80" not in proj_cols else [])].dropna()

    if filter_to_observed:
        # Assume we knew the starting lineup at MILP time — drop players who DNP'd this round.
        observed_ids = set(pred.loc[pred["target_points"].notna(), "player_id"].tolist())
        n_before = len(optim_in)
        optim_in = optim_in[optim_in["player_id"].isin(observed_ids)]
        log.info("known-starters filter for round %d: %d -> %d candidates (dropped %d DNPs)",
                 target_round, n_before, len(optim_in), n_before - len(optim_in))

    if apply_status_filter:
        if ALLOWED_STATUSES:
            active = players_dim[players_dim["status"].fillna("").isin(ALLOWED_STATUSES)]
            label = f"in {sorted(ALLOWED_STATUSES)}"
        else:
            active = players_dim[~players_dim["status"].fillna("").isin(BANNED_STATUSES)]
            label = f"not in {sorted(BANNED_STATUSES)}"
        n_before = len(optim_in)
        optim_in = optim_in[optim_in["player_id"].isin(active["player_id"])]
        log.info("status filter (%s): %d -> %d candidates", label, n_before, len(optim_in))

        manual = _load_manual_exclusions(target_round)
        if manual:
            n_before = len(optim_in)
            optim_in = optim_in[~optim_in["player_id"].isin(manual)]
            log.info("manual exclusions for round %d: %d -> %d candidates (dropped %s)",
                     target_round, n_before, len(optim_in), sorted(manual))

    if len(optim_in) < 15:
        log.warning("round %d: only %d viable players", target_round, len(optim_in))
        return None, info, None

    build_kwargs = {} if budget is None else {"budget": budget}
    if captain_col != "p80":
        build_kwargs["p80_col"] = captain_col
    team = build_optimal_team(optim_in, **build_kwargs)
    team = _enrich_team(team, players_dim, pred)
    info["optim_pool"] = optim_in  # keep for scenario comparison

    # Oracle (only meaningful for past rounds where we have actuals)
    oracle = _oracle_team(pred)
    if oracle is not None:
        oracle = _enrich_team(oracle, players_dim, pred)

    if return_models:
        info["models"] = models
        info["calibrators"] = calibrators
        info["pred"] = pred  # for downstream calibration reporting

    return team, info, oracle


def walk_forward(panel: pd.DataFrame, players_dim: pd.DataFrame,
                 start_round: int = 2, end_round: int = 14) -> pd.DataFrame:
    rows = []
    for r in range(start_round, end_round + 1):
        # Past rounds: we know who actually started. Filter MILP candidates to them so
        # the model is judged on its picking signal, not on captain-DNP unluck.
        team, info, oracle = select_team_for_round(
            panel, r, players_dim, filter_to_observed=True,
        )
        if team is None:
            continue

        captain = team[team["is_captain"]].iloc[0]
        pred_pts = float(team["objective_pts"].sum())
        actual_pts = float(team["actual_with_captain"].sum())
        n_observed = int(team["actual_pts"].notna().sum())
        oracle_pts = float(oracle["actual_with_captain"].sum()) if oracle is not None else float("nan")

        rows.append({
            "round": r,
            "train_rows": info.get("train_rows") if info else None,
            "predicted_pts": pred_pts,
            "actual_pts": actual_pts,
            "oracle_pts": oracle_pts,
            "actual_pct_of_oracle": 100 * actual_pts / oracle_pts if oracle_pts else float("nan"),
            "budget_used": float(team["cost"].sum()),
            "players_who_played": n_observed,
            "captain": f"{captain['first_name']} {captain['last_name']} ({captain['squad_abbr']})",
            "captain_actual_pts": float(captain["actual_pts_filled"]),
        })
        log.info("  pred=%.1f actual=%.1f oracle=%.1f (%.0f%% of oracle) | captain=%s (%.0f pts)",
                 pred_pts, actual_pts, oracle_pts,
                 rows[-1]["actual_pct_of_oracle"],
                 rows[-1]["captain"], rows[-1]["captain_actual_pts"])

    summary = pd.DataFrame(rows)
    if not summary.empty:
        summary["cumulative_actual"] = summary["actual_pts"].cumsum()
        summary["cumulative_oracle"] = summary["oracle_pts"].cumsum()
        summary["cumulative_predicted"] = summary["predicted_pts"].cumsum()
    return summary


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=None,
                    help="salary cap in NZD (e.g. 100000000). "
                         "Default: try PFR live budget, fall back to $100M.")
    ap.add_argument("--skip-live-budget", action="store_true",
                    help="don't hit PFR for live budget; use default/passed value")
    ap.add_argument("--per-position", action="store_true",
                    help="fit one quantile-model set per position (experimental)")
    args = ap.parse_args()
    if args.per_position:
        global PER_POSITION
        PER_POSITION = True
        log.info("per-position training: ON")

    log.info("=== walk-forward + round-15 optimiser ===")
    panel = pd.read_parquet(GOLD_DIR / "feature_panel.parquet")
    players_dim = pd.read_parquet(SILVER_DIR / "players_dim.parquet")

    budget = args.budget
    if budget is None and not args.skip_live_budget:
        try:
            from src.utils.pfr_auth import fetch_budget, login
            budget = fetch_budget(login())
        except Exception as e:
            log.warning("live budget fetch failed: %s — using optimiser default", e)
            budget = None
    log.info("budget for round-15 optimiser: %s",
             f"${budget/1e6:.1f}M" if budget else "optimiser default")

    # 1) walk-forward through completed rounds 2..16 (full season)
    log.info("\n--- WALK-FORWARD ---")
    summary = walk_forward(panel, players_dim, start_round=2, end_round=16)
    if not summary.empty:
        out_path = GOLD_DIR / "walk_forward_summary.parquet"
        summary.to_parquet(out_path, index=False)
        log.info("\n=== WALK-FORWARD SUMMARY ===")
        cols = ["round", "predicted_pts", "actual_pts", "oracle_pts",
                "actual_pct_of_oracle", "budget_used", "players_who_played", "captain"]
        log.info("\n%s", summary[cols].round(2).to_string(index=False))
        log.info("")
        log.info("Total actual : %.1f", summary["actual_pts"].sum())
        log.info("Total oracle : %.1f  (perfect-foresight upper bound)", summary["oracle_pts"].sum())
        log.info("Total pred   : %.1f  (model's own expectation)", summary["predicted_pts"].sum())
        log.info("Model captured %.1f%% of the oracle's points across %d rounds",
                 100 * summary["actual_pts"].sum() / summary["oracle_pts"].sum(), len(summary))

    # 2) Round 16 — the next-GW recommendation, with status filter + scenario comparison
    log.info("\n--- predicting round 16 (next GW) — status filter ON ---")
    team_15, info_15, _ = select_team_for_round(
        panel, 16, players_dim,
        apply_status_filter=True, return_models=True,
        budget=budget,
    )
    if team_15 is not None:
        out_path = GOLD_DIR / "optimal_team_round16.parquet"
        team_15.to_parquet(out_path, index=False)
        captain = team_15[team_15["is_captain"]].iloc[0]
        log.info("[normal] objective=%.1f budget=$%.1fM captain=%s %s (2*P80=%.1f)",
                 team_15["objective_pts"].sum(), team_15["cost"].sum() / 1e6,
                 captain["first_name"], captain["last_name"], 2 * captain["p80"])

        # Calibration report on round-14 (the calibration round itself) — sanity only
        if info_15 and info_15.get("calibrators") and info_15.get("pred") is not None:
            log.info("\n[calibration] applied per-quantile isotonic from round %d",
                     info_15.get("calib_round"))
            # On round 15 we have no actuals; just log mean predicted per quantile pre/post.
            pred = info_15["pred"]
            for col in ["p50", "p60", "p70", "p80", "p90"]:
                if col in pred.columns:
                    log.info("  mean %s = %.2f", col, pred[col].mean())

        # 3) Scenario comparison
        log.info("\n--- round 15 scenario comparison ---")
        optim_in = info_15["optim_pool"] if info_15 else None
        if optim_in is not None:
            scenario_kwargs = {"budget": budget} if budget else {}
            if CAPTAIN_QUANTILE_COL != "p80":
                scenario_kwargs["p80_col"] = CAPTAIN_QUANTILE_COL
            scenarios = compare_scenarios(optim_in, **scenario_kwargs)
            rows = []
            for name, team in scenarios.items():
                team_e = _enrich_team(team, players_dim, info_15["pred"])
                team_e.to_parquet(GOLD_DIR / f"optimal_team_round16_{name}.parquet", index=False)
                caps = team_e[team_e["is_captain"]]
                rows.append({
                    "scenario": name,
                    "total_objective": float(team_e["objective_pts"].sum()),
                    "budget_used_M": float(team_e["cost"].sum() / 1e6),
                    "captain(s)": ", ".join(f"{r.first_name} {r.last_name} ({r.squad_abbr})"
                                            for _, r in caps.iterrows()),
                    "captain_contribution": float(caps["objective_pts"].sum()),
                    "team_avg_p60": float(team_e[~team_e["is_captain"]]["p60"].mean()),
                })
            summary = pd.DataFrame(rows)
            log.info("\n%s", summary.round(1).to_string(index=False))

            # Per-scenario team details
            for name, team in scenarios.items():
                team_e = _enrich_team(team, players_dim, info_15["pred"])
                log.info("\n[%s] team:", name)
                cols = ["first_name", "last_name", "squad_abbr", "position",
                        "cost", "p60", "p80", "is_captain", "objective_pts"]
                log.info("\n%s", team_e[cols].to_string(index=False))


if __name__ == "__main__":
    main()
