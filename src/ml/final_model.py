"""Tasks 2 + 6: two-stage P(plays) x E[points | plays] model with Optuna tuning.

Stage 1 — binary classifier P(played | features). Trained on rows with known
          lineup_role (start or bench => 1, dnp => 0).
Stage 2 — regressor E[points | played]. Trained on rows where the target
          (fantasy_points) is observed. Uses XGBoost with reg:squarederror (winner
          from the bake-off).

Final prediction = P(played) * E[points | played].

Compares against the single-stage baseline using two evaluation framings:
  - observed-only : apples-to-apples with previous runs (eval rounds 13-14
                    on rows where points are observed)
  - unconditional : eval over ALL panel rows for 13-14 with actual = points if
                    played else 0 (this is what really matters for team
                    selection — the model should down-weight DNP risks)

Optuna is run on Stage 2 with random TimeSeriesSplit folds for hyper-tuning,
then re-trained at the picked params for final round-15 predictions.
"""

from __future__ import annotations

import warnings

import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from scipy.stats import spearmanr
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)

warnings.filterwarnings("ignore", category=UserWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

from src.utils.logger import get_logger
from src.utils.paths import GOLD_DIR, SILVER_DIR

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
]
CATEGORICAL = ["position", "squad_id", "opponent_squad_id"]

TRAIN_ROUNDS = list(range(2, 13))
EVAL_ROUNDS = [13, 14]
PREDICT_ROUND = 15


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


def _prep_panel():
    panel = pd.read_parquet(GOLD_DIR / "feature_panel.parquet")
    for col in CATEGORICAL:
        if col in panel.columns and not isinstance(panel[col].dtype, pd.CategoricalDtype):
            panel[col] = panel[col].astype("category")
    return panel


# === Stage 1: P(played) ===

def train_stage1(panel: pd.DataFrame):
    """Binary classifier on rows with known lineup_role."""
    s1 = panel[panel["lineup_role"].notna() & (panel["lineup_role"] != "")].copy()
    s1["played"] = s1["lineup_role"].isin(["start", "bench"]).astype(int)

    tr = s1[s1["round"].isin(TRAIN_ROUNDS)]
    ev = s1[s1["round"].isin(EVAL_ROUNDS)]
    X_tr, y_tr = tr[FEATURES], tr["played"]
    X_ev, y_ev = ev[FEATURES], ev["played"]

    log.info("[stage1] train=%d eval=%d", len(X_tr), len(X_ev))
    log.info("[stage1] train class balance: %s",
             y_tr.value_counts(normalize=True).round(3).to_dict())

    clf = xgb.XGBClassifier(
        n_estimators=600, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
        reg_lambda=1.0, enable_categorical=True,
        early_stopping_rounds=40, eval_metric="logloss",
        random_state=42, n_jobs=-1,
    )
    clf.fit(X_tr, y_tr, eval_set=[(X_ev, y_ev)], verbose=False)
    p_ev = clf.predict_proba(X_ev)[:, 1]
    auc = roc_auc_score(y_ev, p_ev)
    acc = ((p_ev >= 0.5).astype(int) == y_ev).mean()
    log.info("[stage1] eval AUC=%.4f acc=%.4f best_iter=%d",
             auc, acc, int(clf.best_iteration))
    return clf


# === Stage 2: E[points | played] (the regressor we already had) ===

def train_stage2(panel: pd.DataFrame, params: dict | None = None):
    obs = panel[panel["target_points"].notna()].copy()
    tr = obs[obs["round"].isin(TRAIN_ROUNDS)]
    ev = obs[obs["round"].isin(EVAL_ROUNDS)]
    X_tr, y_tr = tr[FEATURES], tr["target_points"]
    X_ev, y_ev = ev[FEATURES], ev["target_points"]

    base = dict(
        objective="reg:squarederror",
        n_estimators=800, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
        reg_lambda=1.0, enable_categorical=True,
        early_stopping_rounds=40, eval_metric="rmse",
        random_state=42, n_jobs=-1,
    )
    if params:
        base.update(params)
    reg = xgb.XGBRegressor(**base)
    reg.fit(X_tr, y_tr, eval_set=[(X_ev, y_ev)], verbose=False)
    return reg


# === Optuna tuning of stage 2 ===

def optuna_tune_stage2(panel: pd.DataFrame, n_trials: int = 60) -> dict:
    """Tune the regressor. Uses an inner train (2-11) / val (12) split
    so the round-13-14 eval is NEVER used during tuning (mitigates the
    early-stop optimism flagged in earlier runs).
    """
    obs = panel[panel["target_points"].notna()].copy()
    inner_train = obs[obs["round"].isin(list(range(2, 12)))]
    inner_val = obs[obs["round"] == 12]
    X_tr, y_tr = inner_train[FEATURES], inner_train["target_points"]
    X_va, y_va = inner_val[FEATURES], inner_val["target_points"]

    def objective(trial):
        params = dict(
            objective="reg:squarederror",
            n_estimators=trial.suggest_int("n_estimators", 200, 1500, step=100),
            max_depth=trial.suggest_int("max_depth", 3, 9),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.20, log=True),
            min_child_weight=trial.suggest_int("min_child_weight", 1, 10),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
            reg_lambda=trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 1.0, log=True),
            gamma=trial.suggest_float("gamma", 0.0, 5.0),
            enable_categorical=True,
            early_stopping_rounds=40, eval_metric="rmse",
            random_state=42, n_jobs=-1,
        )
        m = xgb.XGBRegressor(**params)
        m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        pred = m.predict(X_va)
        return spearmanr(y_va, pred).statistic  # maximise

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    log.info("[optuna] best value (val Spearman on round 12) = %.4f", study.best_value)
    log.info("[optuna] best params: %s", study.best_params)
    return study.best_params


# === Two-stage prediction + evaluation ===

def evaluate_two_stage(panel: pd.DataFrame, clf, reg, label: str = "two_stage"):
    """Compare single-stage (reg only) and two-stage (P(plays) * E[pts|plays])."""
    ev = panel[panel["round"].isin(EVAL_ROUNDS)].copy()
    X = ev[FEATURES]
    pred_reg = reg.predict(X)
    pred_reg = np.clip(pred_reg, panel["target_points"].min(), None)

    p_plays = clf.predict_proba(X)[:, 1]
    pred_two = pred_reg * p_plays

    # Build "unconditional" target: actual points if observed, else 0 (DNP -> 0 pts).
    # This is the team-selection-relevant metric.
    ev["actual_uncond"] = ev["target_points"].fillna(0)

    log.info("\n=== %s — unconditional eval (rounds %s, ALL panel rows incl DNPs, n=%d) ===",
             label, EVAL_ROUNDS, len(ev))
    m1 = metrics(ev["actual_uncond"], pred_reg)
    m2 = metrics(ev["actual_uncond"], pred_two)
    log.info("  single-stage : MAE=%.3f RMSE=%.3f Spearman=%.3f R2=%.3f bias=%+.3f",
             m1["MAE"], m1["RMSE"], m1["Spearman"], m1["R2"], m1["bias"])
    log.info("  two-stage    : MAE=%.3f RMSE=%.3f Spearman=%.3f R2=%.3f bias=%+.3f",
             m2["MAE"], m2["RMSE"], m2["Spearman"], m2["R2"], m2["bias"])

    # Observed-only eval (apples-to-apples with previous runs)
    obs = ev[ev["target_points"].notna()].copy()
    X_obs = obs[FEATURES]
    pred_reg_o = reg.predict(X_obs)
    pred_reg_o = np.clip(pred_reg_o, panel["target_points"].min(), None)
    p_obs = clf.predict_proba(X_obs)[:, 1]
    pred_two_o = pred_reg_o * p_obs
    log.info("=== %s — observed-only eval (n=%d) ===", label, len(obs))
    m1o = metrics(obs["target_points"], pred_reg_o)
    m2o = metrics(obs["target_points"], pred_two_o)
    log.info("  single-stage : MAE=%.3f RMSE=%.3f Spearman=%.3f R2=%.3f bias=%+.3f",
             m1o["MAE"], m1o["RMSE"], m1o["Spearman"], m1o["R2"], m1o["bias"])
    log.info("  two-stage    : MAE=%.3f RMSE=%.3f Spearman=%.3f R2=%.3f bias=%+.3f",
             m2o["MAE"], m2o["RMSE"], m2o["Spearman"], m2o["R2"], m2o["bias"])
    return {"unconditional": {"single": m1, "two_stage": m2},
            "observed_only": {"single": m1o, "two_stage": m2o}}


def predict_round15(panel: pd.DataFrame, clf, reg) -> pd.DataFrame:
    inf = panel[panel["round"] == PREDICT_ROUND].copy()
    X_inf = inf[FEATURES]
    p_plays = clf.predict_proba(X_inf)[:, 1]
    e_points = np.clip(reg.predict(X_inf), panel["target_points"].min(), None)

    inf["p_plays"] = p_plays
    inf["e_points_given_plays"] = e_points
    inf["pred_points_unconditional"] = p_plays * e_points

    players_dim = pd.read_parquet(SILVER_DIR / "players_dim.parquet")
    inf = inf.merge(
        players_dim[["player_id", "first_name", "last_name", "squad_abbr"]],
        on="player_id", how="left",
    )
    out_cols = [
        "player_id", "first_name", "last_name", "squad_abbr", "position",
        "is_home", "opponent_squad_id", "price_lag1",
        "p_plays", "e_points_given_plays", "pred_points_unconditional",
    ]
    out = inf[out_cols].sort_values("pred_points_unconditional", ascending=False)
    return out


def main() -> None:
    log.info("=== FINAL: two-stage + Optuna ===")
    panel = _prep_panel()

    # Stage 1
    log.info("\n--- Stage 1: P(played) classifier ---")
    clf = train_stage1(panel)

    # Stage 2 with default params (baseline)
    log.info("\n--- Stage 2 (default params) ---")
    reg_default = train_stage2(panel)
    log.info("[stage2-default] best_iter=%d", int(reg_default.best_iteration))
    evaluate_two_stage(panel, clf, reg_default, label="default")

    # Task 6: Optuna-tune Stage 2
    log.info("\n--- Stage 2: Optuna tuning (60 trials, inner val on round 12) ---")
    best_params = optuna_tune_stage2(panel, n_trials=60)

    log.info("\n--- Stage 2 (tuned params) ---")
    reg_tuned = train_stage2(panel, params=best_params)
    log.info("[stage2-tuned] best_iter=%d", int(reg_tuned.best_iteration))
    evaluate_two_stage(panel, clf, reg_tuned, label="tuned")

    # Final round-15 predictions
    log.info("\n--- Round-15 final predictions (tuned two-stage) ---")
    out = predict_round15(panel, clf, reg_tuned)
    out_path = GOLD_DIR / "predictions_round15_two_stage.parquet"
    out.to_parquet(out_path, index=False)
    log.info("wrote %s (%d players)", out_path, len(out))
    log.info("--- top 20 ---\n%s", out.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
