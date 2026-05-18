"""Train quantile-regression XGBoost models predicting P50/P60/P70/P80/P90 of
unconditional fantasy points (= actual pts if played, 0 if DNP).

Quantile target captures the distribution: P60 = conservative team-points pick,
P80 = aggressive captain pick. The MILP downstream uses both.

Trains on rows where lineup_role is known (avoids polluting the target with rows
where we genuinely don't know if the player played).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression

from src.utils.logger import get_logger
from src.utils.paths import GOLD_DIR

log = get_logger(__name__)


FEATURES = [
    "position", "squad_id", "opponent_squad_id", "is_home",
    "pts_lag1", "pts_lag2", "pts_lag3",
    "pts_roll3_mean", "pts_roll5_mean", "pts_roll3_std",
    "pts_roll5_max", "pts_roll5_min",
    "pts_season_total", "games_played", "pts_per_game_season",
    "price_lag1", "price_delta_3", "ownership_lag1", "ownership_delta_3",
    "team_attack_3", "team_defense_3", "team_fantasy_3",
    "opp_attack_3", "opp_defense_3", "opp_fantasy_3",
    "days_rest",
    "is_starter_last_match", "appeared_last_match",
    "start_share_l3", "start_share_l5", "bench_share_l3", "dnp_share_l3",
    "age_yrs", "height_m", "weight_kg", "bmi",
    "career_years", "n_career_teams", "has_intl", "n_intl_appearances",
    "team_pack_weight_lag", "team_pack_intl_lag", "team_xv_intl_lag",
    "opp_pack_weight_lag", "opp_pack_intl_lag", "opp_xv_intl_lag",
    "pack_weight_advantage", "pack_height_advantage", "pack_intl_advantage",
    "xv_intl_advantage", "xv_age_advantage", "back_intl_advantage",
    "fixture_total_implied", "fixture_margin_implied", "is_forward",
    # Travel — haversine km from each team's home stadium to this fixture's venue
    "own_travel_km", "opp_travel_km", "travel_advantage_km",
]
CATEGORICAL = ["position", "squad_id", "opponent_squad_id"]
QUANTILES = (0.5, 0.6, 0.7, 0.8, 0.9)

BASE_PARAMS = dict(
    n_estimators=500,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    reg_lambda=1.0,
    enable_categorical=True,
    early_stopping_rounds=40,
    eval_metric="rmse",
    random_state=42,
    n_jobs=-1,
)


def ensure_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    for col in CATEGORICAL:
        if col in df.columns and not isinstance(df[col].dtype, pd.CategoricalDtype):
            df[col] = df[col].astype("category")
    return df


def make_uncond_target(panel: pd.DataFrame) -> pd.Series:
    """Unconditional target: actual pts if observed, 0 if dnp confirmed, NaN otherwise.

    Rows with unknown lineup_role (e.g. future rounds) get NaN and are excluded from training.
    Rows where lineup_role indicates the player played (start/bench) but target_points
    is NaN are treated as 0 (rare; mostly bench cameos with no scoring events).
    """
    t = panel["target_points"].astype("float64").copy()
    role_known = panel["lineup_role"].notna() & (panel["lineup_role"] != "")
    is_dnp = panel["lineup_role"] == "dnp"
    played_no_pts = role_known & ~is_dnp & t.isna()

    t = t.where(~is_dnp, 0.0)
    t = t.where(~played_no_pts, 0.0)
    # Rows with unknown lineup status stay NaN (won't be trained on)
    return t


def train_quantile_models(
    X_tr: pd.DataFrame, y_tr: pd.Series,
    X_va: pd.DataFrame, y_va: pd.Series,
    quantiles=QUANTILES,
    base_params: dict | None = None,
) -> dict[float, xgb.XGBRegressor]:
    base = dict(base_params or BASE_PARAMS)
    models: dict[float, xgb.XGBRegressor] = {}
    for q in quantiles:
        params = dict(base)
        params.update(objective="reg:quantileerror", quantile_alpha=q)
        m = xgb.XGBRegressor(**params)
        m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        models[q] = m
    return models


def predict_quantiles(
    models: dict[float, xgb.XGBRegressor],
    X: pd.DataFrame,
    calibrators: dict[float, IsotonicRegression] | None = None,
    enforce_monotone: bool = True,
) -> pd.DataFrame:
    """Predict P-quantiles, optionally passing each quantile through its calibrator."""
    cols = []
    preds = {}
    for q in sorted(models.keys()):
        raw = models[q].predict(X)
        if calibrators is not None and q in calibrators:
            raw = calibrators[q].predict(raw)
        name = f"p{int(round(q * 100))}"
        preds[name] = np.clip(raw, 0, None)
        cols.append(name)
    df = pd.DataFrame(preds, index=X.index)
    if enforce_monotone:
        # Quantile-crossing fix: ensure non-decreasing across quantiles row-wise.
        arr = df[cols].values
        arr = np.maximum.accumulate(arr, axis=1)
        df = pd.DataFrame(arr, columns=cols, index=df.index)
    return df


def fit_calibrators(
    models: dict[float, xgb.XGBRegressor],
    X_calib: pd.DataFrame,
    y_calib: pd.Series,
    only_quantiles: set[float] | None = None,
) -> dict[float, IsotonicRegression]:
    """Per-quantile isotonic calibration of predicted -> actual on a held-out round.

    For each quantile q in `models` (filtered by `only_quantiles` if supplied) we
    fit a monotonic non-decreasing map (raw_pred -> actual). Selective calibration
    is supported by passing e.g. `only_quantiles={0.8}` — calibrate only the captain
    quantile and leave team (P60) predictions raw, which avoids isotonic plateaus
    collapsing mid-tier discrimination.
    """
    out: dict[float, IsotonicRegression] = {}
    y_arr = np.asarray(y_calib, dtype=float)
    for q, model in models.items():
        if only_quantiles is not None and q not in only_quantiles:
            continue
        raw = model.predict(X_calib)
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0)
        iso.fit(raw, y_arr)
        out[q] = iso
    return out


def calibration_report(
    models: dict[float, xgb.XGBRegressor],
    X: pd.DataFrame,
    y: pd.Series,
    calibrators: dict[float, IsotonicRegression] | None = None,
    label: str = "",
) -> dict[float, dict]:
    """For each quantile q, report empirical coverage (% of actuals <= predicted q)
    and mean predicted vs actual. Coverage should be close to q for well-calibrated models.
    """
    y_arr = np.asarray(y, dtype=float)
    out = {}
    for q in sorted(models.keys()):
        raw = models[q].predict(X)
        pred = calibrators[q].predict(raw) if calibrators and q in calibrators else raw
        pred = np.clip(pred, 0, None)
        coverage = float((y_arr <= pred).mean())
        out[q] = {
            "target_coverage": q,
            "empirical_coverage": coverage,
            "coverage_gap": coverage - q,
            "mean_pred": float(pred.mean()),
            "mean_actual": float(y_arr.mean()),
        }
    log.info("--- calibration %s (target_q | empirical_coverage | gap | mean_pred) ---", label)
    for q, m in out.items():
        log.info("  q=%.2f  cov=%.3f  gap=%+.3f  pred_mean=%.2f  actual_mean=%.2f",
                 q, m["empirical_coverage"], m["coverage_gap"],
                 m["mean_pred"], m["mean_actual"])
    return out


def _inner_split(train_pool: pd.DataFrame):
    """Use the latest training round as inner-val for early stopping."""
    rounds = sorted(train_pool["round"].unique())
    if len(rounds) < 2:
        # Fallback: stratified random split
        from sklearn.model_selection import train_test_split
        tr, va = train_test_split(train_pool, test_size=0.25, random_state=42)
        return tr, va
    inner_va_round = rounds[-1]
    tr = train_pool[train_pool["round"] != inner_va_round]
    va = train_pool[train_pool["round"] == inner_va_round]
    if len(va) < 30:  # too thin a val — back off one round
        return train_pool[:int(0.8 * len(train_pool))], train_pool[int(0.8 * len(train_pool)):]
    return tr, va


def fit_for_round(panel: pd.DataFrame, target_round: int) -> tuple[dict, dict]:
    """Train quantile models using rounds < target_round. No calibration."""
    panel = ensure_categoricals(panel)
    if "target_uncond" not in panel.columns:
        panel = panel.copy()
        panel["target_uncond"] = make_uncond_target(panel)

    train_pool = panel[(panel["round"] < target_round) & panel["target_uncond"].notna()].copy()
    if len(train_pool) < 50:
        raise ValueError(f"too little training data for round {target_round}: only {len(train_pool)} rows")

    tr, va = _inner_split(train_pool)
    X_tr, y_tr = tr[FEATURES], tr["target_uncond"]
    X_va, y_va = va[FEATURES], va["target_uncond"]

    models = train_quantile_models(X_tr, y_tr, X_va, y_va)
    info = {
        "target_round": target_round,
        "train_rows": len(X_tr),
        "val_rows": len(X_va),
        "rounds_in_train": sorted(tr["round"].unique().tolist()),
        "val_round": int(va["round"].iloc[0]) if len(va) else None,
        "calibrated": False,
    }
    return models, info


def fit_for_round_calibrated(
    panel: pd.DataFrame, target_round: int,
    calibrate_quantiles: set[float] | None = None,
) -> tuple[dict, dict | None, dict]:
    """Train quantile models + fit isotonic calibrators (optionally only for some quantiles).

    Walk-forward split:
        train rounds       : 2 .. r-3  (model fitting)
        inner-val round    : r-2       (early stopping)
        calibration round  : r-1       (isotonic per-quantile)
        target round       : r         (prediction)

    `calibrate_quantiles` controls which quantiles get calibrated.
        - None (default) -> all of them
        - {0.8}          -> selective: only the captain quantile is calibrated;
                            team P60 stays raw to preserve mid-tier discrimination

    Falls back to uncalibrated `fit_for_round` if fewer than 3 prior rounds are available.
    Returns (models, calibrators_or_None, info).
    """
    panel = ensure_categoricals(panel)
    if "target_uncond" not in panel.columns:
        panel = panel.copy()
        panel["target_uncond"] = make_uncond_target(panel)

    train_pool = panel[(panel["round"] < target_round) & panel["target_uncond"].notna()].copy()
    rounds = sorted(train_pool["round"].unique())
    if len(rounds) < 3:
        models, info = fit_for_round(panel, target_round)
        return models, None, info

    calib_round = rounds[-1]
    val_round = rounds[-2]
    train_rounds = rounds[:-2]

    tr = train_pool[train_pool["round"].isin(train_rounds)]
    va = train_pool[train_pool["round"] == val_round]
    ca = train_pool[train_pool["round"] == calib_round]

    X_tr, y_tr = tr[FEATURES], tr["target_uncond"]
    X_va, y_va = va[FEATURES], va["target_uncond"]
    X_ca, y_ca = ca[FEATURES], ca["target_uncond"]

    models = train_quantile_models(X_tr, y_tr, X_va, y_va)
    calibrators = fit_calibrators(models, X_ca, y_ca, only_quantiles=calibrate_quantiles)

    info = {
        "target_round": target_round,
        "train_rows": len(X_tr),
        "val_rows": len(X_va),
        "calib_rows": len(X_ca),
        "rounds_in_train": train_rounds,
        "val_round": int(val_round),
        "calib_round": int(calib_round),
        "calibrated": True,
        "calibrated_quantiles": sorted(calibrators.keys()),
    }
    return models, calibrators, info


def fit_per_position_for_round(
    panel: pd.DataFrame, target_round: int,
    calibrate_quantiles: set[float] | None = None,
    min_train_rows: int = 50,
) -> tuple[dict[str, dict], dict[str, dict | None], dict]:
    """Per-position variant of `fit_for_round_calibrated`.

    Trains an independent set of 5 quantile models for each position. At inference,
    each player's predictions come from their position's model only.

    Returns:
        models_by_pos     : {position: {q: XGBRegressor}}
        calibrators_by_pos: {position: {q: IsotonicRegression} or None}
        info              : aggregate metadata (rows per position, fallback list, ...)

    If a position has fewer than `min_train_rows` trainable rows for this target,
    it falls back to a globally-trained quantile model (same as the non-per-position
    path) so we don't dump cheap depth at thin-data positions.
    """
    panel = ensure_categoricals(panel)
    if "target_uncond" not in panel.columns:
        panel = panel.copy()
        panel["target_uncond"] = make_uncond_target(panel)

    train_pool = panel[(panel["round"] < target_round) & panel["target_uncond"].notna()].copy()
    rounds = sorted(train_pool["round"].unique())
    if len(rounds) < 3:
        # Not enough to do per-position splits — return global as a single 'all' bucket
        models, info = fit_for_round(panel, target_round)
        return {"_all": models}, {"_all": None}, info

    calib_round = rounds[-1]
    val_round = rounds[-2]
    train_rounds = rounds[:-2]

    # First, fit a global model as the fallback for thin positions.
    tr_g = train_pool[train_pool["round"].isin(train_rounds)]
    va_g = train_pool[train_pool["round"] == val_round]
    ca_g = train_pool[train_pool["round"] == calib_round]
    global_models = train_quantile_models(tr_g[FEATURES], tr_g["target_uncond"],
                                          va_g[FEATURES], va_g["target_uncond"])
    global_calib = fit_calibrators(global_models, ca_g[FEATURES], ca_g["target_uncond"],
                                   only_quantiles=calibrate_quantiles)

    models_by_pos: dict[str, dict] = {}
    calib_by_pos: dict[str, dict | None] = {}
    rows_by_pos: dict[str, int] = {}
    fallback_positions: list[str] = []

    positions = sorted(panel["position"].astype(str).unique())
    for pos in positions:
        tr_p = tr_g[tr_g["position"].astype(str) == pos]
        va_p = va_g[va_g["position"].astype(str) == pos]
        ca_p = ca_g[ca_g["position"].astype(str) == pos]
        rows_by_pos[pos] = len(tr_p)
        if len(tr_p) < min_train_rows or len(va_p) < 5:
            models_by_pos[pos] = global_models
            calib_by_pos[pos] = global_calib
            fallback_positions.append(pos)
            continue

        models = train_quantile_models(tr_p[FEATURES], tr_p["target_uncond"],
                                       va_p[FEATURES], va_p["target_uncond"])
        if len(ca_p) >= 5:
            calibs = fit_calibrators(models, ca_p[FEATURES], ca_p["target_uncond"],
                                     only_quantiles=calibrate_quantiles)
        else:
            calibs = None
        models_by_pos[pos] = models
        calib_by_pos[pos] = calibs

    info = {
        "target_round": target_round,
        "train_rows": len(tr_g),
        "val_rows": len(va_g),
        "calib_rows": len(ca_g),
        "rounds_in_train": train_rounds,
        "val_round": int(val_round),
        "calib_round": int(calib_round),
        "calibrated": calibrate_quantiles != set(),
        "per_position": True,
        "rows_by_position": rows_by_pos,
        "fallback_positions": fallback_positions,
    }
    return models_by_pos, calib_by_pos, info


def predict_quantiles_per_position(
    models_by_pos: dict[str, dict],
    calibrators_by_pos: dict[str, dict | None],
    panel_subset: pd.DataFrame,
    enforce_monotone: bool = True,
) -> pd.DataFrame:
    """Predict P-quantiles routing each row to the model trained for its position.

    Returns a DataFrame indexed like `panel_subset` with p50..p90 columns.
    """
    out_chunks = []
    for pos, group in panel_subset.groupby(panel_subset["position"].astype(str), sort=False):
        models = models_by_pos.get(pos) or models_by_pos["_all"]
        calibs = calibrators_by_pos.get(pos) if calibrators_by_pos else None
        preds = predict_quantiles(models, group[FEATURES], calibrators=calibs,
                                  enforce_monotone=enforce_monotone)
        preds.index = group.index
        out_chunks.append(preds)
    full = pd.concat(out_chunks).reindex(panel_subset.index)
    return full


def main() -> None:
    """Train + emit round-15 quantile predictions."""
    log.info("=== quantile model: P50/60/70/80/90 ===")
    panel = pd.read_parquet(GOLD_DIR / "feature_panel.parquet")
    panel = ensure_categoricals(panel)
    panel["target_uncond"] = make_uncond_target(panel)

    log.info("target stats (training pool):")
    tu = panel.loc[panel["target_uncond"].notna(), "target_uncond"]
    log.info("  n=%d mean=%.2f std=%.2f min=%.1f max=%.1f pct_zero=%.1f%%",
             len(tu), tu.mean(), tu.std(), tu.min(), tu.max(),
             100 * (tu == 0).mean())

    models, info = fit_for_round(panel, target_round=15)
    log.info("trained on %d rows (rounds %s), inner-val on round %d (%d rows)",
             info["train_rows"], info["rounds_in_train"], info["val_round"], info["val_rows"])

    # Predict round 15
    pred_pool = panel[panel["round"] == 15].copy()
    quantile_preds = predict_quantiles(models, pred_pool[FEATURES])
    out = pd.concat([
        pred_pool[["player_id", "squad_id", "position", "is_home", "opponent_squad_id", "price"]].reset_index(drop=True),
        quantile_preds.reset_index(drop=True),
    ], axis=1)
    out_path = GOLD_DIR / "predictions_round15_quantile.parquet"
    out.to_parquet(out_path, index=False)
    log.info("wrote %s (%d players)", out_path, len(out))

    # Quick sanity: quantile means + sample
    log.info("quantile means across round-15 players:")
    log.info("%s", out[["p50", "p60", "p70", "p80", "p90"]].mean().round(2).to_string())
    log.info("top 10 by P60:\n%s",
             out.nlargest(10, "p60")[["player_id", "position", "p50", "p60", "p70", "p80", "p90"]].to_string(index=False))


if __name__ == "__main__":
    main()
