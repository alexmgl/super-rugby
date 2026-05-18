"""MILP solver for the fantasy team selection with captain.

Objective:
    Maximise   Σ p60_i · x_i + Σ (2·p80_c − p60_c) · is_captain_c
    so the captain's contribution becomes 2·p80 (not p60+p80; the captain doesn't
    also count as a regular p60 player). Decomposes nicely as
    total = Σ_{non-captain} p60_i + 2·p80_captain.

Decision variables:
    x_i           ∈ {0,1}   — player i selected for team
    is_captain_i  ∈ {0,1}   — player i is captain

Constraints:
    - Σ x_i = team_size (default 15)
    - Σ is_captain_i = 1
    - is_captain_i ≤ x_i (captain must be selected)
    - Σ (x_i · cost_i) ≤ budget
    - For each position p: Σ x_i [pos_i = p] = required_count(p)
    - For each squad s : Σ x_i [squad_i = s] ≤ max_per_squad
"""

from __future__ import annotations

import logging

import pandas as pd
import pyomo.environ as pyo

from src.utils.logger import get_logger

log = get_logger(__name__)

# HiGHS solver chats a lot through pyomo's appsi logger; quiet it.
logging.getLogger("pyomo.contrib.appsi.solvers.highs").setLevel(logging.WARNING)


# Default 8-forwards / 7-backs starting XV
DEFAULT_POSITION_COUNTS = {
    "prop":          2,
    "hooker":        1,
    "lock":          2,
    "loose_forward": 3,
    "scrum_half":    1,
    "fly_half":      1,
    "center":        2,
    "outside_back":  3,
}
DEFAULT_TEAM_SIZE = sum(DEFAULT_POSITION_COUNTS.values())  # 15
DEFAULT_BUDGET = 100_000_000       # NZ$100M — PFR opening cap; configurable
DEFAULT_MAX_PER_SQUAD = 5          # standard fantasy cap


def build_optimal_team(
    players: pd.DataFrame,
    *,
    budget: float = DEFAULT_BUDGET,
    position_counts: dict[str, int] | None = None,
    max_per_squad: int = DEFAULT_MAX_PER_SQUAD,
    captain_multiplier: float = 2.0,
    n_captains: int = 1,
    p60_col: str = "p60",
    p80_col: str = "p80",
    cost_col: str = "cost",
    position_col: str = "position",
    squad_col: str = "squad_id",
    id_col: str = "player_id",
    solver_name: str = "appsi_highs",
) -> pd.DataFrame:
    """Solve the captain-augmented team-selection MILP.

    Args:
        players: candidate pool with columns ['player_id', 'position', 'squad_id',
                 'cost', 'p60', 'p80'] (names overridable via kwargs).
                 Rows with NaN p60/p80/cost are dropped silently.
        budget: salary-cap upper bound.
        position_counts: required # of starters per position. Must sum to team_size.
        max_per_squad: max players from any single squad.

    Returns:
        DataFrame of the 15 chosen players with `is_captain` and `objective_pts` columns,
        sorted by objective_pts descending.
    """
    pos_req = dict(position_counts or DEFAULT_POSITION_COUNTS)
    team_size = sum(pos_req.values())

    df = players.dropna(subset=[p60_col, p80_col, cost_col]).reset_index(drop=True).copy()
    if len(df) < team_size:
        raise ValueError(f"only {len(df)} candidates after dropna; need at least {team_size}")

    # Sanity: enough players per position
    for pos, need in pos_req.items():
        have = (df[position_col] == pos).sum()
        if have < need:
            raise ValueError(f"position {pos!r}: need {need}, only {have} candidates available")

    n = len(df)
    idx = list(range(n))
    p60 = df[p60_col].to_numpy()
    p80 = df[p80_col].to_numpy()
    cost = df[cost_col].to_numpy()
    pos = df[position_col].to_numpy()
    squad = df[squad_col].to_numpy()

    m = pyo.ConcreteModel()
    m.x = pyo.Var(idx, domain=pyo.Binary)
    m.c = pyo.Var(idx, domain=pyo.Binary)

    # Objective: Σ p60·x  +  Σ (captain_mult·p80 − p60)·c
    # captain_mult=2 (normal), 3 (triple captain). Captain replaces their p60 with
    # captain_mult·p80, so the marginal value of being captain is (mult·p80 − p60).
    m.obj = pyo.Objective(
        expr=sum(p60[i] * m.x[i] for i in idx)
             + sum((captain_multiplier * p80[i] - p60[i]) * m.c[i] for i in idx),
        sense=pyo.maximize,
    )

    # Captain must be in team; n_captains captains total (1 normal, 2 co-captains)
    m.cap_in_team = pyo.ConstraintList()
    for i in idx:
        m.cap_in_team.add(m.c[i] <= m.x[i])
    m.n_captains = pyo.Constraint(expr=sum(m.c[i] for i in idx) == n_captains)

    # Team size
    m.team_size = pyo.Constraint(expr=sum(m.x[i] for i in idx) == team_size)

    # Position composition (exact counts — equality, not inequality)
    m.position_constraints = pyo.ConstraintList()
    for p, need in pos_req.items():
        ix = [i for i in idx if pos[i] == p]
        m.position_constraints.add(sum(m.x[i] for i in ix) == need)

    # Budget
    m.budget = pyo.Constraint(expr=sum(cost[i] * m.x[i] for i in idx) <= budget)

    # Max per squad
    m.squad_constraints = pyo.ConstraintList()
    for sid in set(squad):
        ix = [i for i in idx if squad[i] == sid]
        m.squad_constraints.add(sum(m.x[i] for i in ix) <= max_per_squad)

    solver = pyo.SolverFactory(solver_name)
    res = solver.solve(m, tee=False)
    status = str(res.solver.termination_condition)
    if status not in ("optimal", "TerminationCondition.optimal", "feasible"):
        log.warning("solver status: %s", status)

    chosen_idx = [i for i in idx if pyo.value(m.x[i]) > 0.5]
    captain_idx = [i for i in idx if pyo.value(m.c[i]) > 0.5]
    if len(captain_idx) != n_captains:
        raise RuntimeError(f"expected {n_captains} captain(s), got {len(captain_idx)}")
    captain_positions = set(captain_idx)

    out = df.iloc[chosen_idx].copy()
    out["is_captain"] = out.index.isin(captain_positions)
    out["objective_pts"] = out.apply(
        lambda r: captain_multiplier * r[p80_col] if r["is_captain"] else r[p60_col],
        axis=1,
    )
    out = out.sort_values("objective_pts", ascending=False).reset_index(drop=True)
    return out


SCENARIOS = {
    "normal":         dict(captain_multiplier=2.0, n_captains=1),
    "triple_captain": dict(captain_multiplier=3.0, n_captains=1),
    "limitless":      dict(captain_multiplier=2.0, n_captains=1, budget=10**12),
    "co_captains":    dict(captain_multiplier=2.0, n_captains=2),
}


def compare_scenarios(players: pd.DataFrame, **base_kwargs) -> dict[str, pd.DataFrame]:
    """Solve all four scenarios on the same player pool. Returns dict[name -> team]."""
    out = {}
    for name, overrides in SCENARIOS.items():
        merged = dict(base_kwargs)
        merged.update(overrides)
        try:
            out[name] = build_optimal_team(players, **merged)
        except Exception as e:
            log.warning("scenario %s failed: %s", name, e)
    return out


def summarise(team: pd.DataFrame) -> dict:
    """Roll up a chosen-team DataFrame to a single dict of headline metrics."""
    captain_row = team[team["is_captain"]].iloc[0]
    return {
        "n_players": len(team),
        "total_cost": float(team["cost"].sum()),
        "total_objective_pts": float(team["objective_pts"].sum()),
        "captain_player_id": captain_row["player_id"],
        "captain_p80": float(captain_row["p80"]),
        "captain_objective_contribution": float(2 * captain_row["p80"]),
    }
