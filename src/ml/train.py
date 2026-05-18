"""Train XGBoost models to predict next-round fantasy points.

Trains two variants for an apples-to-apples comparison:
  - baseline : the original 26 features (fantasy + team/opponent form)
  - enriched : baseline + 13 new features sourced from Ultimate Rugby
               (bio + career + match-lineup signals)

Diagnostics on the enriched feature set:
  - Multicollinearity (Pearson |r| > 0.8 among numeric features)
  - Mutual information (sklearn.feature_selection.mutual_info_regression)
  - SHAP (XGBoost native pred_contribs)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import spearmanr
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.utils.logger import get_logger
from src.utils.paths import GOLD_DIR, SILVER_DIR

log = get_logger(__name__)


# --- feature sets ---------------------------------------------------------------

BASELINE_FEATURES = [
    # Static / fixture context
    "position", "squad_id", "opponent_squad_id", "is_home",
    # Player form (lagged)
    "pts_lag1", "pts_lag2", "pts_lag3",
    "pts_roll3_mean", "pts_roll5_mean", "pts_roll3_std",
    "pts_roll5_max", "pts_roll5_min",
    "pts_season_total", "games_played", "pts_per_game_season",
    # Market (lagged)
    "price_lag1", "price_delta_3", "ownership_lag1", "ownership_delta_3",
    # Team form (lagged)
    "team_attack_3", "team_defense_3", "team_fantasy_3",
    "opp_attack_3",  "opp_defense_3",  "opp_fantasy_3",
    # Schedule
    "days_rest",
]

NEW_FEATURES = [
    # Lineup-derived (lagged)
    "is_starter_last_match", "appeared_last_match",
    "start_share_l3", "start_share_l5", "bench_share_l3", "dnp_share_l3",
    # Bio (static)
    "age_yrs", "height_m", "weight_kg", "bmi",
    # Career (static)
    "career_years", "n_career_teams", "has_intl", "n_intl_appearances",
    # Team composition — lagged starting XV aggregates (own team)
    "team_pack_weight_lag", "team_pack_intl_lag", "team_xv_intl_lag",
    # Opponent composition — lagged
    "opp_pack_weight_lag", "opp_pack_intl_lag", "opp_xv_intl_lag",
    # Team-vs-team differentials (signed: positive = own team has the edge)
    "pack_weight_advantage", "pack_height_advantage", "pack_intl_advantage",
    "xv_intl_advantage", "xv_age_advantage", "back_intl_advantage",
    # Fixture style
    "fixture_total_implied", "fixture_margin_implied",
    # Forward/back interaction handle
    "is_forward",
]

ENRICHED_FEATURES = BASELINE_FEATURES + NEW_FEATURES

TRAIN_ROUNDS = list(range(2, 13))   # 2..12
EVAL_ROUNDS = [13, 14]
PREDICT_ROUND = 15

XGB_PARAMS = dict(
    objective="reg:tweedie",
    tweedie_variance_power=1.5,
    n_estimators=800,
    max_depth=5,
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


# --- helpers --------------------------------------------------------------------

def _ensure_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    for col in ["position", "squad_id", "opponent_squad_id"]:
        if col in df.columns and not isinstance(df[col].dtype, pd.CategoricalDtype):
            df[col] = df[col].astype("category")
    return df


def describe_target(y: np.ndarray, label: str = "target") -> None:
    log.info("--- %s distribution ---", label)
    log.info("n=%d mean=%.2f std=%.2f min=%.1f max=%.1f",
             len(y), float(y.mean()), float(y.std()), float(y.min()), float(y.max()))
    q25, q50, q75, q95 = np.percentile(y, [25, 50, 75, 95])
    log.info("median=%.1f q25=%.1f q75=%.1f q95=%.1f", q50, q25, q75, q95)
    log.info("pct_zero=%.2f%% pct_neg=%.2f%% pct_>50=%.2f%% pct_>70=%.2f%%",
             100 * (y == 0).mean(), 100 * (y < 0).mean(),
             100 * (y > 50).mean(), 100 * (y > 70).mean())


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "MAE":      float(mean_absolute_error(y_true, y_pred)),
        "RMSE":     float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MedAE":    float(np.median(np.abs(y_pred - y_true))),
        "Spearman": float(spearmanr(y_true, y_pred).statistic),
        "R2":       float(r2_score(y_true, y_pred)),
        "bias":     float((y_pred - y_true).mean()),
        "n":        int(len(y_true)),
    }


# --- diagnostics ----------------------------------------------------------------

def multicollinearity_report(X: pd.DataFrame, threshold: float = 0.8) -> None:
    numeric = X.select_dtypes(exclude=["category", "object"])
    # Convert pandas nullable dtypes to numpy float for .corr()
    numeric = numeric.astype("float64")
    if numeric.empty:
        log.info("multicollinearity: no numeric columns")
        return
    corr = numeric.corr(method="pearson").abs()
    upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
    pairs = []
    for c in upper.columns:
        for r in upper.index:
            v = upper.at[r, c]
            if pd.notna(v) and v > threshold:
                pairs.append((r, c, float(v)))
    pairs.sort(key=lambda x: -x[2])
    log.info("--- multicollinearity (Pearson |r| > %.2f, top 20) ---", threshold)
    if not pairs:
        log.info("  none above threshold")
    for r, c, v in pairs[:20]:
        log.info("  r=%.3f : %-26s <-> %s", v, r, c)


def mutual_information_report(X: pd.DataFrame, y: pd.Series, random_state: int = 42) -> pd.Series:
    X_enc = X.copy()
    discrete = []
    for col in X_enc.columns:
        if isinstance(X_enc[col].dtype, pd.CategoricalDtype):
            X_enc[col] = X_enc[col].cat.codes.replace(-1, np.nan)
            discrete.append(True)
        else:
            discrete.append(False)
    # Median-impute NaNs (MI doesn't accept NaN).
    for col in X_enc.columns:
        if X_enc[col].isna().any():
            med = X_enc[col].median()
            X_enc[col] = X_enc[col].fillna(0 if pd.isna(med) else med)
    X_enc = X_enc.astype("float64")
    mi = mutual_info_regression(X_enc, y, discrete_features=discrete, random_state=random_state)
    mi_ser = pd.Series(mi, index=X.columns).sort_values(ascending=False)
    log.info("--- mutual information vs target (top 20) ---")
    for feat, val in mi_ser.head(20).items():
        log.info("  MI=%.4f  %s", val, feat)
    return mi_ser


def shap_report(model: xgb.XGBRegressor, X: pd.DataFrame) -> pd.Series:
    booster = model.get_booster()
    dmat = xgb.DMatrix(X, enable_categorical=True)
    shap_vals = booster.predict(dmat, pred_contribs=True)
    # shap_vals[:, -1] is the bias term; drop it.
    feature_shap = shap_vals[:, :-1]
    mean_abs = np.abs(feature_shap).mean(axis=0)
    shap_ser = pd.Series(mean_abs, index=X.columns).sort_values(ascending=False)
    log.info("--- SHAP mean(|value|) per feature (top 20) ---")
    for feat, val in shap_ser.head(20).items():
        log.info("  |SHAP|=%.3f  %s", val, feat)
    return shap_ser


# --- training -------------------------------------------------------------------

def fit_and_eval(
    train_pool: pd.DataFrame,
    feature_cols: list[str],
    shift: float,
    label: str,
) -> tuple[xgb.XGBRegressor, dict]:
    tr_mask = train_pool["round"].isin(TRAIN_ROUNDS)
    ev_mask = train_pool["round"].isin(EVAL_ROUNDS)

    X_tr = train_pool.loc[tr_mask, feature_cols]
    y_tr = train_pool.loc[tr_mask, "y_shifted"]
    X_ev = train_pool.loc[ev_mask, feature_cols]
    y_ev = train_pool.loc[ev_mask, "y_shifted"]
    y_ev_orig = train_pool.loc[ev_mask, "target_points"].to_numpy()

    log.info("[%s] train=%d eval=%d features=%d", label, len(X_tr), len(X_ev), len(feature_cols))

    model = xgb.XGBRegressor(**XGB_PARAMS)
    model.fit(X_tr, y_tr, eval_set=[(X_ev, y_ev)], verbose=False)

    pred = model.predict(X_ev) - shift
    pred = np.clip(pred, train_pool["target_points"].min(), None)
    m = metrics(y_ev_orig, pred)
    m["best_iteration"] = int(model.best_iteration)
    return model, m


# --- main -----------------------------------------------------------------------

def main() -> None:
    log.info("=== ML training: XGBoost (Tweedie), baseline vs enriched ===")
    panel = pd.read_parquet(GOLD_DIR / "feature_panel.parquet")
    panel = _ensure_categoricals(panel)

    train_pool = panel[panel["target_points"].notna()].copy()
    log.info("rows with observed target: %d (rounds %s)",
             len(train_pool), sorted(train_pool["round"].unique()))

    describe_target(train_pool["target_points"].to_numpy(), "target_points")

    y_min = float(train_pool["target_points"].min())
    shift = max(0.0, -y_min)
    log.info("target shift for Tweedie: +%.1f (raw min=%.1f)", shift, y_min)
    train_pool["y_shifted"] = train_pool["target_points"] + shift

    # Sanity-check feature presence
    for label, cols in [("baseline", BASELINE_FEATURES), ("enriched", ENRICHED_FEATURES)]:
        missing = [c for c in cols if c not in train_pool.columns]
        if missing:
            raise RuntimeError(f"{label} features missing from panel: {missing}")

    # === Diagnostics on the enriched feature set ===
    log.info("")
    log.info("============== DIAGNOSTICS (enriched) ==============")
    ev_for_diag = train_pool[train_pool["round"].isin(EVAL_ROUNDS)]
    X_diag = train_pool[ENRICHED_FEATURES]
    y_diag = train_pool["target_points"]

    multicollinearity_report(X_diag, threshold=0.8)
    log.info("")
    mi = mutual_information_report(X_diag, y_diag)

    # === Train both variants ===
    log.info("")
    log.info("============== TRAINING ==============")
    _model_b, m_baseline = fit_and_eval(train_pool, BASELINE_FEATURES, shift, "baseline")
    model_e, m_enriched = fit_and_eval(train_pool, ENRICHED_FEATURES, shift, "enriched")

    # Naive baselines (for reference)
    X_ev = train_pool.loc[train_pool["round"].isin(EVAL_ROUNDS), ENRICHED_FEATURES]
    y_ev_orig = train_pool.loc[train_pool["round"].isin(EVAL_ROUNDS), "target_points"].to_numpy()
    naive_lag1 = X_ev["pts_lag1"].fillna(train_pool["y_shifted"].mean() - shift).to_numpy()
    naive_ppg = X_ev["pts_per_game_season"].fillna(train_pool["y_shifted"].mean() - shift).to_numpy()
    m_naive_lag1 = metrics(y_ev_orig, naive_lag1)
    m_naive_ppg = metrics(y_ev_orig, naive_ppg)

    # === Side-by-side comparison ===
    log.info("")
    log.info("============== EVAL METRICS (rounds %s) ==============", EVAL_ROUNDS)
    rows = {
        "naive_lag1":   m_naive_lag1,
        "naive_ppg":    m_naive_ppg,
        "xgb_baseline": m_baseline,
        "xgb_enriched": m_enriched,
    }
    cmp_df = pd.DataFrame(rows).T[["MAE", "RMSE", "MedAE", "Spearman", "R2", "bias"]]
    log.info("\n%s", cmp_df.round(3).to_string())

    # Improvement of enriched vs baseline
    log.info("")
    log.info("--- enriched vs baseline (deltas; negative = improvement for error metrics) ---")
    delta_pct = {}
    for k in ["MAE", "RMSE", "MedAE", "Spearman", "R2"]:
        b, e = m_baseline[k], m_enriched[k]
        if k in ("MAE", "RMSE", "MedAE"):
            pct = 100 * (e - b) / b
            log.info("  %-8s baseline=%.3f  enriched=%.3f  %+.1f%%", k, b, e, pct)
        else:
            log.info("  %-8s baseline=%.3f  enriched=%.3f  +%.3f", k, b, e, e - b)

    # === SHAP on enriched model ===
    log.info("")
    log.info("============== SHAP (enriched, eval set) ==============")
    X_shap = train_pool.loc[train_pool["round"].isin(EVAL_ROUNDS), ENRICHED_FEATURES]
    shap_ser = shap_report(model_e, X_shap)

    # === Round-15 predictions (enriched model) ===
    inf_pool = panel[panel["round"] == PREDICT_ROUND].copy()
    log.info("")
    log.info("round-%d inference pool: %d players", PREDICT_ROUND, len(inf_pool))
    if len(inf_pool):
        X_inf = inf_pool[ENRICHED_FEATURES]
        inf_pool["pred_points"] = np.clip(
            model_e.predict(X_inf) - shift, y_min, None,
        )
        players_dim = pd.read_parquet(SILVER_DIR / "players_dim.parquet")
        inf_pool = inf_pool.merge(
            players_dim[["player_id", "first_name", "last_name", "squad_abbr"]],
            on="player_id", how="left",
        )
        out_cols = [
            "player_id", "first_name", "last_name", "squad_abbr", "position",
            "is_home", "opponent_squad_id", "price_lag1", "pred_points",
        ]
        out = inf_pool[out_cols].sort_values("pred_points", ascending=False)
        out_path = GOLD_DIR / "predictions_round15.parquet"
        out.to_parquet(out_path, index=False)
        log.info("wrote %s (%d players)", out_path, len(out))
        log.info("--- top 20 round-%d predictions ---\n%s",
                 PREDICT_ROUND, out.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
