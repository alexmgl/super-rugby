"""MILP solver for Fantasy Super Rugby Pacific 2026.

Picks a 15-player squad that maximises expected fantasy points, subject to:
  - exact position counts (2 props, 1 hooker, 2 locks, 3 loose forwards,
    1 scrum half, 1 fly half, 2 centers, 3 outside backs)
  - budget (default 100m)
  - max 4 players from any one Super Rugby Pacific club
  - exactly one captain (2x points)

There is no bench in Super Rugby Pacific fantasy — all 15 players score.

Solves each round independently if multiple rounds are present in the data
(via a `round` column). Most callers will pass a single-round predictions
table.
"""

from __future__ import annotations

import pandas as pd
import pyomo.environ as pyo
from pyomo.environ import Binary, Constraint, Objective, SolverFactory, Var, maximize, value
from pyomo.opt import SolverStatus, TerminationCondition

from src.milp.config import (
    CAPTAIN_BONUS,
    CAPTAIN_MULTIPLIER,
    MAX_PLAYERS_PER_CLUB,
    POSITION_COUNTS,
    POSITIONS,
    SQUAD_SIZE,
    STARTING_BUDGET,
)
from src.utils.logger import get_logger

log = get_logger(__name__)


class SuperRugbySolver:
    """Pick the optimal 15-player Fantasy Super Rugby Pacific squad.

    Each round in the input data is solved independently. The caller can
    override any default constraint via the constructor (e.g. set
    `budget=float('inf')` to model the Limitless booster, or
    `captain_multiplier=3` for Triple Captain).
    """

    def __init__(
        self,
        budget: float = STARTING_BUDGET,
        squad_size: int = SQUAD_SIZE,
        max_per_club: int = MAX_PLAYERS_PER_CLUB,
        position_counts: dict = POSITION_COUNTS,
        captain_multiplier: float = CAPTAIN_MULTIPLIER,
        captain_bonus: float = CAPTAIN_BONUS,
        positions=POSITIONS,
        squad_must_include=None,
        squad_must_exclude=None,
        set_captain: int | None = None,
        n_captains: int = 1,
    ):
        """
        :param budget: max total squad cost per round (raw units, 1m = 1_000_000)
        :param squad_size: total players in the squad (15)
        :param max_per_club: max players from any one club (4)
        :param position_counts: exact count required per position
        :param captain_multiplier: points multiplier for captain (2x; 3x for Triple Captain)
        :param captain_bonus: flat bonus points for the captain (default 0)
        :param positions: set of valid position labels
        :param squad_must_include: list of player_ids forced into the squad
        :param squad_must_exclude: list of player_ids forbidden from the squad
        :param set_captain: player_id forced to be (a) captain
        :param n_captains: number of captains per round (1 normally, 2 for Co-Captains booster)
        """
        self.data: pd.DataFrame | None = None

        self.budget = budget
        self.squad_size = squad_size
        self.max_per_club = max_per_club
        self.position_counts = position_counts
        self.positions = positions

        self.captain_multiplier = captain_multiplier
        self.captain_bonus = captain_bonus
        self.n_captains = n_captains
        self.set_captain = set_captain

        self.squad_must_include = squad_must_include or []
        self.squad_must_exclude = squad_must_exclude or []

        self.m = pyo.ConcreteModel()
        self.solver_status = None
        self.solver_results = None

    # ------------------------------------------------------------------ data

    def load(self, data: pd.DataFrame) -> None:
        """Load player data.

        Required columns:
            player_id : unique player ID
            club      : club identifier (any hashable; squad_abbr or squad_id both fine)
            position  : one of `self.positions`
            cost      : player price in raw units
            points    : expected fantasy points (objective)

        Optional:
            round     : if present, each round is optimised independently.
                        Defaults to 1.
        """
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Data must be a pandas DataFrame.")
        self.data = data.copy()

        required = {"player_id", "club", "position", "cost", "points"}
        missing = required - set(self.data.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

        if "round" not in self.data.columns:
            self.data["round"] = 1

        unknown = set(self.data["position"].unique()) - set(self.positions)
        if unknown:
            raise ValueError(f"Unknown position labels in data: {sorted(unknown)}")

        if sum(self.position_counts.values()) != self.squad_size:
            raise ValueError(
                f"position_counts sum {sum(self.position_counts.values())} != squad_size {self.squad_size}"
            )

    # ----------------------------------------------------------------- build

    def __set_params(self) -> None:
        if self.data is None:
            raise RuntimeError("Call load(data) before build_model().")

        player_keys = [tuple(r) for r in self.data[["player_id", "round"]].drop_duplicates().values.tolist()]
        self.m.player_keys = pyo.Set(initialize=player_keys)

        rounds = sorted(self.data["round"].unique())
        self.m.rounds = pyo.Set(initialize=rounds)

        self.m.round_players = {}
        for pid, rnd in self.m.player_keys:
            self.m.round_players.setdefault(rnd, []).append(pid)

        self.m.prices    = self.data.set_index(["player_id", "round"])["cost"].to_dict()
        self.m.points    = self.data.set_index(["player_id", "round"])["points"].to_dict()
        self.m.clubs     = self.data.set_index(["player_id", "round"])["club"].to_dict()
        self.m.player_pos = self.data.set_index(["player_id", "round"])["position"].to_dict()

    def __set_vars(self) -> None:
        self.m.squad   = Var(self.m.player_keys, domain=Binary)  # 1 if picked
        self.m.captain = Var(self.m.player_keys, domain=Binary)  # 1 if captain

    def __set_objective(self) -> None:
        def rule(_):
            base = sum(self.m.squad[i, r]   * self.m.points[i, r] for i, r in self.m.player_keys)
            cap  = sum(self.m.captain[i, r] * self.m.points[i, r] * (self.captain_multiplier - 1)
                       for i, r in self.m.player_keys)
            bonus = sum(self.m.captain[i, r] * self.captain_bonus for i, r in self.m.player_keys)
            return base + cap + bonus

        self.m.objective = Objective(rule=rule, sense=maximize)

    def __set_constraints(self) -> None:
        m = self.m

        # Exactly squad_size players per round
        m.c_squad_size = Constraint(
            m.rounds,
            rule=lambda _m, r: sum(m.squad[p, r] for p in m.round_players[r]) == self.squad_size,
        )

        # Exact position counts
        m.c_positions = pyo.ConstraintList()
        for r in m.rounds:
            players_r = m.round_players[r]
            for pos, count in self.position_counts.items():
                in_pos = [p for p in players_r if m.player_pos[(p, r)] == pos]
                m.c_positions.add(sum(m.squad[p, r] for p in in_pos) == count)

        # Max players per club
        m.c_clubs = pyo.ConstraintList()
        for (club, rnd), pids in self.data.groupby(["club", "round"])["player_id"].apply(lambda s: list(set(s))).items():
            m.c_clubs.add(sum(m.squad[p, rnd] for p in pids) <= self.max_per_club)

        # Budget
        m.c_budget = Constraint(
            m.rounds,
            rule=lambda _m, r: sum(m.squad[p, r] * m.prices[(p, r)] for p in m.round_players[r]) <= self.budget,
        )

        # Captain must be in squad; exactly n_captains per round
        m.c_captain_in_squad = Constraint(m.player_keys, rule=lambda _m, p, r: m.captain[p, r] <= m.squad[p, r])
        m.c_captain_count = Constraint(
            m.rounds,
            rule=lambda _m, r: sum(m.captain[p, r] for p in m.round_players[r]) == self.n_captains,
        )

        # Must include / exclude / set_captain
        m.c_must = pyo.ConstraintList()
        for r in m.rounds:
            for pid in self.squad_must_include:
                if pid in m.round_players[r]:
                    m.c_must.add(m.squad[pid, r] == 1)
            for pid in self.squad_must_exclude:
                if pid in m.round_players[r]:
                    m.c_must.add(m.squad[pid, r] == 0)
            if self.set_captain is not None and self.set_captain in m.round_players[r]:
                m.c_must.add(m.captain[self.set_captain, r] == 1)

    def build_model(self) -> None:
        self.__set_params()
        self.__set_vars()
        self.__set_objective()
        self.__set_constraints()

    # ----------------------------------------------------------------- solve

    def solve_model(self, solver_name: str = "appsi_highs"):
        """Solve the model. Default solver is HiGHS (`appsi_highs`), which is
        installed via the `highspy` package in requirements.txt."""
        if self.m is None:
            raise ValueError("Model not built. Call build_model() first.")

        log.info("solving with %s", solver_name)
        solver = SolverFactory(solver_name)
        result = solver.solve(self.m, tee=False)

        self.solver_status = result.solver.status
        self.solver_results = result

        if result.solver.status == SolverStatus.ok and \
                result.solver.termination_condition == TerminationCondition.optimal:
            log.info("optimal solution found")
        else:
            log.warning("solver finished with status=%s termination=%s",
                        result.solver.status, result.solver.termination_condition)
        return result

    # --------------------------------------------------------------- results

    def get_results_summary(self) -> list[dict]:
        if self.solver_status != SolverStatus.ok:
            return []

        results = []
        for r in self.m.rounds:
            squad = []
            captains: list[int] = []

            for pid in self.m.round_players[r]:
                if value(self.m.squad[pid, r]) > 0.5:
                    squad.append({
                        "id":       pid,
                        "position": self.m.player_pos[(pid, r)],
                        "club":     self.m.clubs[(pid, r)],
                        "cost":     self.m.prices[(pid, r)],
                        "points":   self.m.points[(pid, r)],
                    })
                    if value(self.m.captain[pid, r]) > 0.5:
                        captains.append(pid)

            total_points = sum(p["points"] for p in squad)
            cap_extra = sum(
                self.m.points[(pid, r)] * (self.captain_multiplier - 1) + self.captain_bonus
                for pid in captains
            )
            total_points += cap_extra
            total_cost = sum(p["cost"] for p in squad)

            results.append({
                "round":        r,
                "squad":        squad,
                "captains":     captains,
                "captain":      captains[0] if len(captains) == 1 else None,
                "total_cost":   total_cost,
                "total_points": total_points,
            })

        return results


if __name__ == "__main__":
    SuperRugbySolver()
