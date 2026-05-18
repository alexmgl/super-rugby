"""Driver script for the Fantasy Super Rugby Pacific 2026 optimiser.

Reads the round-level predictions written by `src.ml.train`, runs the
SuperRugbySolver, and pretty-prints the optimal 15-player squad with captain.
"""

from __future__ import annotations

from collections import Counter
from typing import Optional

import pandas as pd
from pyomo.opt import SolverStatus
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from src.milp.config import POSITION_COUNTS
from src.milp.solver import SuperRugbySolver
from src.utils.paths import GOLD_DIR, SILVER_DIR

console = Console()

POSITION_ORDER = list(POSITION_COUNTS.keys())
POSITION_LABELS = {
    "prop":          "Prop",
    "hooker":        "Hooker",
    "lock":          "Lock",
    "loose_forward": "Loose Fwd",
    "scrum_half":    "Half Back",
    "fly_half":      "Fly Half",
    "center":        "Midfield",
    "outside_back":  "Back Three",
}


# ---------- Data prep ----------

def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise a predictions table into the columns the solver expects.

    Expected input columns (from gold/predictions_round*.parquet):
        player_id, squad_abbr, position, price_lag1, pred_points

    Output columns: player_id, club, position, cost, points, round
    """
    out = df.copy()
    rename = {}
    if "pred_points" in out.columns and "points" not in out.columns:
        rename["pred_points"] = "points"
    if "price_lag1" in out.columns and "cost" not in out.columns:
        rename["price_lag1"] = "cost"
    if "squad_abbr" in out.columns and "club" not in out.columns:
        rename["squad_abbr"] = "club"
    out = out.rename(columns=rename)

    if "round" not in out.columns:
        out["round"] = 1

    out["cost"] = pd.to_numeric(out["cost"], errors="coerce")
    out["points"] = pd.to_numeric(out["points"], errors="coerce")
    out["position"] = out["position"].astype(str)

    # Drop rows with no price (can't cost them) or no prediction.
    out = out.dropna(subset=["cost", "points", "club", "position"]).copy()
    return out


def load_player_names(silver_dir=SILVER_DIR) -> dict[int, str]:
    """Build {player_id: "First Last"} from silver/players_dim.parquet."""
    path = silver_dir / "players_dim.parquet"
    if not path.exists():
        console.print(f"[yellow]players_dim not found at {path}[/yellow]")
        return {}
    dim = pd.read_parquet(path, columns=["player_id", "first_name", "last_name"])
    dim["full"] = (dim["first_name"].fillna("") + " " + dim["last_name"].fillna("")).str.strip()
    return dict(zip(dim["player_id"], dim["full"]))


# ---------- Optimisation ----------

def run_optimization(data: pd.DataFrame, constraints: Optional[dict] = None, solver_name: str = "appsi_highs"):
    """Build and solve. Returns the solver's results list, or None on failure."""
    constraints = constraints or {}
    solver = SuperRugbySolver(**constraints)
    solver.load(data)
    solver.build_model()
    result = solver.solve_model(solver_name=solver_name)

    if result.solver.status == SolverStatus.ok:
        return solver.get_results_summary()
    console.print(f"[red]Solver failed: {result.solver.status}[/red]")
    return None


# ---------- Rich display ----------

def _name(pid: int, names: dict[int, str]) -> str:
    return names.get(pid, f"ID {pid}") if names else f"ID {pid}"


def _squad_table(title: str) -> Table:
    t = Table(title=title, box=box.SIMPLE_HEAVY)
    t.add_column("Position", style="cyan", no_wrap=True)
    t.add_column("Name", style="bold")
    t.add_column("Club")
    t.add_column("Cost (m)", justify="right")
    t.add_column("Exp Pts", justify="right")
    t.add_column("", justify="center")  # captain flag
    return t


def _sorted_squad(squad: list[dict]) -> list[dict]:
    rank = {p: i for i, p in enumerate(POSITION_ORDER)}
    return sorted(squad, key=lambda x: (rank.get(x["position"], 99), -x["points"]))


def display_results(results: list[dict], names: Optional[dict[int, str]] = None) -> None:
    names = names or {}
    for res in results:
        rnd = res["round"]
        cap_ids = set(res.get("captains") or ([res["captain"]] if res.get("captain") else []))

        header = Text.assemble(
            ("  ROUND ", "bold white"),
            (str(rnd), "bold yellow"),
            ("  |  ", "dim"),
            ("Squad 15 (all start)", "cyan"),
        )
        console.rule(header)

        summary = Table.grid(padding=(0, 2))
        summary.add_column(justify="left")
        summary.add_column(justify="right")
        summary.add_row("Total Cost", f"{res['total_cost'] / 1_000_000:0.2f}m")
        summary.add_row("Expected Points", f"{res['total_points']:0.2f}")
        pos_counts = Counter(p["position"] for p in res["squad"])
        summary.add_row(
            "Composition",
            " / ".join(f"{pos_counts.get(p, 0)} {POSITION_LABELS.get(p, p)}" for p in POSITION_ORDER),
        )
        console.print(Panel(summary, title="Summary", border_style="green", box=box.ROUNDED))

        table = _squad_table("Squad")
        for p in _sorted_squad(res["squad"]):
            flag = "[yellow]C[/yellow]" if p["id"] in cap_ids else ""
            table.add_row(
                POSITION_LABELS.get(p["position"], p["position"]),
                _name(p["id"], names),
                str(p["club"]),
                f"{p['cost'] / 1_000_000:0.2f}",
                f"{p['points']:0.2f}",
                flag,
            )
        console.print(table)

        cap_line = ", ".join(_name(c, names) for c in sorted(cap_ids)) if cap_ids else "—"
        console.print(Panel.fit(f"[bold]Captain:[/bold] {cap_line}", border_style="yellow", box=box.ROUNDED))


def get_squad_list(results: list[dict]):
    """Return (squad_ids, captain_ids, position_counts) for the first round."""
    first = results[0]
    ids = [p["id"] for p in first["squad"]]
    captains = first.get("captains") or ([first["captain"]] if first.get("captain") else [])
    counts = Counter(p["position"] for p in first["squad"])
    return ids, captains, dict(counts)


# ---------- Entry ----------

if __name__ == "__main__":
    console.rule("[bold white]Fantasy Super Rugby Pacific — optimisation[/bold white]")

    predictions_path = GOLD_DIR / "predictions_round15.parquet"
    raw = pd.read_parquet(predictions_path)
    data = prepare_data(raw)
    console.print(f"loaded {len(data)} candidate players from {predictions_path.name}")

    names = load_player_names()

    constraints = {
        "squad_must_include": [],
        "squad_must_exclude": [],
        "set_captain": None,
        # "budget": float("inf"),  # uncomment to model the Limitless booster
        # "captain_multiplier": 3,  # uncomment for Triple Captain
        # "n_captains": 2,          # uncomment for Co-Captains
    }

    results = run_optimization(data, constraints, solver_name="appsi_highs")
    if not results:
        raise SystemExit(1)

    display_results(results, names=names)

    squad_ids, captain_ids, pos_counts = get_squad_list(results)
    console.print(
        Panel.fit(
            f"Squad IDs: {squad_ids}\nCaptain IDs: {captain_ids}\nPosition counts: {pos_counts}",
            title="IDs",
            border_style="blue",
            box=box.ROUNDED,
        )
    )
