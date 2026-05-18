"""Tasks 4 + 5: objective bake-off + model bake-off.

Trains a matrix of (model x objective) on the enriched gold panel and reports
side-by-side eval metrics. Picks the winning combination for downstream use.

Models : xgboost, lightgbm, catboost
Objectives: tweedie (current), squarederror/rmse, pseudo-huber

Uses the same time-based train (rounds 2-12) / eval (rounds 13-14) split as train.py
for apples-to-apples comparison.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore", category=UserWarning)

from src.utils.logger import get_logger
from src.utils.paths import GOLD_DIR

log = get_logger(__name__)

# Use the enriched feature set from train.py
ENRICHED_FEATURES = [
    "position", "squad_id", "opponent_squad_id", "is_home",
    "pts_lag1", "pts_lag2", "pts_lag3",
    "pts_roll3_mean", "pts_roll5_mean", "pts_roll3_std",
    "pts_roll5_max", "pts_roll5_min",
    "pts_season_total", "games_played", "pts_per_game_season",
    "price_lag1", "price_delta_3", "ownership_lag1", "ownership_delta_3",
    "team_attack_3", "team_defense_3", "team_fantasy_3",
    "opp_attack_3",  "opp_defense_3",  "opp_fantasy_3",
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
]
CATEGORICAL = ["position", "squad_id", "opponent_squad_id"]

TRAIN_ROUNDS = list(range(2, 13))
EVAL_ROUNDS = [13, 14]


def metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "MAE":      float(mean_absolute_error(y_true, y_pred)),
        "RMSE":     float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MedAE":    float(np.median(np.abs(y_pred - y_true))),
        "Spearman": float(spearmanr(y_true, y_pred).statistic),
        "R2":       float(r2_score(y_true, y_pred)),
        "bias":     float((y_pred - y_true).mean()),
    }


def _prepare_panel():
    panel = pd.read_parquet(GOLD_DIR / "feature_panel.parquet")
    for col in CATEGORICAL:
        if col in panel.columns and not isinstance(panel[col].dtype, pd.CategoricalDtype):
            panel[col] = panel[col].astype("category")

    obs = panel[panel["target_points"].notna()].copy()
    y_min = float(obs["target_points"].min())
    shift = max(0.0, -y_min)
    obs["y_shifted"] = obs["target_points"] + shift

    tr_mask = obs["round"].isin(TRAIN_ROUNDS)
    ev_mask = obs["round"].isin(EVAL_ROUNDS)
    X_tr = obs.loc[tr_mask, ENRICHED_FEATURES]
    y_tr_shift = obs.loc[tr_mask, "y_shifted"]
    y_tr_orig = obs.loc[tr_mask, "target_points"]
    X_ev = obs.loc[ev_mask, ENRICHED_FEATURES]
    y_ev_shift = obs.loc[ev_mask, "y_shifted"]
    y_ev_orig = obs.loc[ev_mask, "target_points"].to_numpy()
    return X_tr, y_tr_shift, y_tr_orig, X_ev, y_ev_shift, y_ev_orig, shift, y_min


# --- XGBoost variants ---

def fit_xgb(X_tr, y_tr, X_ev, y_ev, objective: str) -> xgb.XGBRegressor:
    extra = {}
    if objective == "reg:tweedie":
        extra["tweedie_variance_power"] = 1.5
    elif objective == "reg:pseudohubererror":
        extra["huber_slope"] = 5.0
    model = xgb.XGBRegressor(
        objective=objective,
        n_estimators=800, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
        reg_lambda=1.0, enable_categorical=True,
        early_stopping_rounds=40, eval_metric="rmse",
        random_state=42, n_jobs=-1, **extra,
    )
    model.fit(X_tr, y_tr, eval_set=[(X_ev, y_ev)], verbose=False)
    return model


# --- LightGBM ---

def fit_lgbm(X_tr, y_tr, X_ev, y_ev, objective: str):
    import lightgbm as lgb
    # LGBM wants integer codes for categoricals
    X_tr_lgb = X_tr.copy()
    X_ev_lgb = X_ev.copy()
    cat_idx = []
    for i, c in enumerate(X_tr_lgb.columns):
        if isinstance(X_tr_lgb[c].dtype, pd.CategoricalDtype):
            cat_idx.append(c)
    extra = {}
    if objective == "tweedie":
        extra["tweedie_variance_power"] = 1.5
    elif objective == "huber":
        extra["alpha"] = 5.0
    model = lgb.LGBMRegressor(
        objective=objective,
        n_estimators=800, max_depth=-1, num_leaves=31, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_samples=3,
        reg_lambda=1.0, random_state=42, n_jobs=-1, verbose=-1, **extra,
    )
    model.fit(
        X_tr_lgb, y_tr,
        eval_set=[(X_ev_lgb, y_ev)],
        categorical_feature=cat_idx,
        callbacks=[lgb.early_stopping(40, verbose=False)],
    )
    return model


# --- CatBoost ---

def fit_catboost(X_tr, y_tr, X_ev, y_ev, objective: str):
    from catboost import CatBoostRegressor, Pool
    cat_idx = [i for i, c in enumerate(X_tr.columns)
               if isinstance(X_tr[c].dtype, pd.CategoricalDtype)]
    # CatBoost wants categoricals as strings or ints, not pd.Categorical with NaN.
    X_tr_cb = X_tr.copy()
    X_ev_cb = X_ev.copy()
    for c in CATEGORICAL:
        if c in X_tr_cb.columns:
            X_tr_cb[c] = X_tr_cb[c].astype(str).fillna("nan")
            X_ev_cb[c] = X_ev_cb[c].astype(str).fillna("nan")

    extra = {}
    if objective == "Tweedie:variance_power=1.5":
        pass  # objective string encodes the parameter
    model = CatBoostRegressor(
        loss_function=objective,
        iterations=800, depth=6, learning_rate=0.05,
        l2_leaf_reg=3.0, random_seed=42, verbose=False,
        early_stopping_rounds=40,
    )
    train_pool = Pool(X_tr_cb, y_tr, cat_features=cat_idx)
    eval_pool = Pool(X_ev_cb, y_ev, cat_features=cat_idx)
    model.fit(train_pool, eval_set=eval_pool, use_best_model=True)
    return model, X_ev_cb


def predict_unshift(model, X, shift, y_min, kind="sklearn") -> np.ndarray:
    pred = model.predict(X) - shift
    return np.clip(pred, y_min, None)


def main() -> None:
    log.info("=== BAKE-OFF: models x objectives ===")
    X_tr, y_tr_s, y_tr_o, X_ev, y_ev_s, y_ev_o, shift, y_min = _prepare_panel()
    log.info("train=%d eval=%d features=%d shift=%.1f",
             len(X_tr), len(X_ev), len(ENRICHED_FEATURES), shift)

    results = {}

    # --- XGBoost: 3 objectives ---
    log.info("\n--- XGBoost ---")
    # Tweedie needs shifted (non-negative) target
    model = fit_xgb(X_tr, y_tr_s, X_ev, y_ev_s, "reg:tweedie")
    pred = predict_unshift(model, X_ev, shift, y_min)
    results["xgb_tweedie"] = metrics(y_ev_o, pred)
    log.info("xgb_tweedie best_iter=%d", int(model.best_iteration))

    # Squared error works on original target
    model = fit_xgb(X_tr, y_tr_o, X_ev, y_ev_o, "reg:squarederror")
    pred = np.clip(model.predict(X_ev), y_min, None)
    results["xgb_squarederror"] = metrics(y_ev_o, pred)
    log.info("xgb_squarederror best_iter=%d", int(model.best_iteration))

    # Pseudo-Huber (robust to outliers)
    model = fit_xgb(X_tr, y_tr_o, X_ev, y_ev_o, "reg:pseudohubererror")
    pred = np.clip(model.predict(X_ev), y_min, None)
    results["xgb_pseudohuber"] = metrics(y_ev_o, pred)
    log.info("xgb_pseudohuber best_iter=%d", int(model.best_iteration))

    # --- LightGBM: 3 objectives ---
    log.info("\n--- LightGBM ---")
    model = fit_lgbm(X_tr, y_tr_s, X_ev, y_ev_s, "tweedie")
    pred = np.clip(model.predict(X_ev) - shift, y_min, None)
    results["lgbm_tweedie"] = metrics(y_ev_o, pred)
    log.info("lgbm_tweedie best_iter=%d", int(model.best_iteration_))

    model = fit_lgbm(X_tr, y_tr_o, X_ev, y_ev_o, "regression")  # RMSE
    pred = np.clip(model.predict(X_ev), y_min, None)
    results["lgbm_rmse"] = metrics(y_ev_o, pred)
    log.info("lgbm_rmse best_iter=%d", int(model.best_iteration_))

    model = fit_lgbm(X_tr, y_tr_o, X_ev, y_ev_o, "huber")
    pred = np.clip(model.predict(X_ev), y_min, None)
    results["lgbm_huber"] = metrics(y_ev_o, pred)
    log.info("lgbm_huber best_iter=%d", int(model.best_iteration_))

    # --- CatBoost: 3 objectives ---
    log.info("\n--- CatBoost ---")
    model, X_ev_cb = fit_catboost(X_tr, y_tr_s, X_ev, y_ev_s, "Tweedie:variance_power=1.5")
    pred = np.clip(model.predict(X_ev_cb) - shift, y_min, None)
    results["cb_tweedie"] = metrics(y_ev_o, pred)

    model, X_ev_cb = fit_catboost(X_tr, y_tr_o, X_ev, y_ev_o, "RMSE")
    pred = np.clip(model.predict(X_ev_cb), y_min, None)
    results["cb_rmse"] = metrics(y_ev_o, pred)

    model, X_ev_cb = fit_catboost(X_tr, y_tr_o, X_ev, y_ev_o, "Huber:delta=5.0")
    pred = np.clip(model.predict(X_ev_cb), y_min, None)
    results["cb_huber"] = metrics(y_ev_o, pred)

    # --- Simple equal-weight ensemble of best objective per model ---
    log.info("\n--- Ensemble (avg of best objective per model) ---")
    # Pick best objective per model by Spearman
    def best_for(prefix):
        keys = [k for k in results if k.startswith(prefix)]
        return max(keys, key=lambda k: results[k]["Spearman"])

    log.info("best per model (by Spearman):")
    best_xgb = best_for("xgb_")
    best_lgbm = best_for("lgbm_")
    best_cb = best_for("cb_")
    log.info("  xgb=%s, lgbm=%s, cb=%s", best_xgb, best_lgbm, best_cb)

    # Refit + predict in parallel for the three winners
    # XGB winner
    obj = {"xgb_tweedie": "reg:tweedie", "xgb_squarederror": "reg:squarederror",
           "xgb_pseudohuber": "reg:pseudohubererror"}[best_xgb]
    use_shift_xgb = obj == "reg:tweedie"
    y_tr_x = y_tr_s if use_shift_xgb else y_tr_o
    y_ev_x = y_ev_s if use_shift_xgb else y_ev_o
    m_xgb = fit_xgb(X_tr, y_tr_x, X_ev, y_ev_x, obj)
    p_xgb = m_xgb.predict(X_ev) - (shift if use_shift_xgb else 0)

    # LGBM winner
    obj = {"lgbm_tweedie": "tweedie", "lgbm_rmse": "regression", "lgbm_huber": "huber"}[best_lgbm]
    use_shift_lgbm = obj == "tweedie"
    y_tr_l = y_tr_s if use_shift_lgbm else y_tr_o
    y_ev_l = y_ev_s if use_shift_lgbm else y_ev_o
    m_lgbm = fit_lgbm(X_tr, y_tr_l, X_ev, y_ev_l, obj)
    p_lgbm = m_lgbm.predict(X_ev) - (shift if use_shift_lgbm else 0)

    # CatBoost winner
    obj = {"cb_tweedie": "Tweedie:variance_power=1.5", "cb_rmse": "RMSE",
           "cb_huber": "Huber:delta=5.0"}[best_cb]
    use_shift_cb = obj.startswith("Tweedie")
    y_tr_c = y_tr_s if use_shift_cb else y_tr_o
    y_ev_c = y_ev_s if use_shift_cb else y_ev_o
    m_cb, X_ev_cb = fit_catboost(X_tr, y_tr_c, X_ev, y_ev_c, obj)
    p_cb = m_cb.predict(X_ev_cb) - (shift if use_shift_cb else 0)

    # Clip + ensemble
    p_xgb = np.clip(p_xgb, y_min, None)
    p_lgbm = np.clip(p_lgbm, y_min, None)
    p_cb = np.clip(p_cb, y_min, None)
    p_ensemble = (p_xgb + p_lgbm + p_cb) / 3.0

    results["ensemble_mean"] = metrics(y_ev_o, p_ensemble)

    # Naive baseline for reference
    naive_ppg = X_ev["pts_per_game_season"].fillna(y_ev_o.mean()).to_numpy()
    results["naive_ppg"] = metrics(y_ev_o, naive_ppg)

    # --- Comparison table ---
    cmp = pd.DataFrame(results).T[["MAE", "RMSE", "MedAE", "Spearman", "R2", "bias"]]
    cmp = cmp.sort_values("Spearman", ascending=False)
    log.info("\n=== EVAL METRICS (rounds %s) ===\n%s", EVAL_ROUNDS, cmp.round(3).to_string())

    # Overall winner
    winner = cmp.index[0]
    log.info("\nOverall winner by Spearman: %s", winner)


if __name__ == "__main__":
    main()
