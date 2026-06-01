"""Cumulative-points chart: walk-forward model vs season leader (rank-1) per round.

For each round R, we look up the manager with overallRank=1 in that round's
leaderboard JSON. Their `overallPoints` IS their season-to-date cumulative
through R — i.e. who is leading the league at the close of GW-R.

Then we overlay the model's walk-forward cumulative actual (rounds 2 onwards).

Outputs:
    data/gold/cumulative_vs_leader.png
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.utils.logger import get_logger
from src.utils.paths import BRONZE_DIR, GOLD_DIR

log = get_logger(__name__)


def load_leader_curve() -> pd.DataFrame:
    """Cumulative-leader curve.

    The per-round JSON is sorted by THAT round's rank, not by overall, so the true
    overallRank=1 may not be in any given round's top-2000 if they had a quiet
    round. We take max(overallPoints) per round JSON — this is the cumulative
    leader AMONG users present in that round's leaderboard, a tight near-leader
    in practice (true leaders typically score in the round top-2000 each week).
    """
    raw = BRONZE_DIR / "raw"
    rows = []
    for fp in sorted(raw.glob("fantasy_rankings_round*.json")):
        m = re.search(r"round(\d+)\.json$", fp.name)
        if not m:
            continue
        rid = int(m.group(1))
        data = json.loads(fp.read_text(encoding="utf-8"))
        if not data:
            continue
        leader = max(data, key=lambda u: u.get("overallPoints") or 0)
        rows.append({
            "round": rid,
            "leader_userName": leader.get("userName"),
            "leader_overallRank": leader.get("overallRank"),
            "leader_overallPts": leader.get("overallPoints"),
            "leader_roundPts": leader.get("points"),
        })
    return pd.DataFrame(rows).sort_values("round").reset_index(drop=True)


def load_rank_band_curve(low: int, high: int, label: str) -> pd.DataFrame:
    """Cumulative-points curve from the rank-band's best member each round.

    `low`/`high` are overallRank bounds (inclusive). For each round we keep the
    user whose overallRank lies in that band and has the LOWEST rank — i.e. the
    band's top of the season-to-date order present in that round's leaderboard.
    """
    raw = BRONZE_DIR / "raw"
    rows = []
    for fp in sorted(raw.glob("fantasy_rankings_round*.json")):
        m = re.search(r"round(\d+)\.json$", fp.name)
        if not m:
            continue
        rid = int(m.group(1))
        data = json.loads(fp.read_text(encoding="utf-8"))
        band = [u for u in data if u.get("overallRank") and low <= u["overallRank"] <= high]
        if not band:
            continue
        # The member with the LOWEST overallRank in the band = best of the band
        rep = min(band, key=lambda u: u["overallRank"])
        rows.append({"round": rid, f"{label}_overallPts": rep.get("overallPoints")})
    return pd.DataFrame(rows).sort_values("round").reset_index(drop=True)


def main() -> None:
    GOLD_DIR.mkdir(parents=True, exist_ok=True)

    leader = load_leader_curve()
    if leader.empty:
        log.warning("no per-round leaderboard files found")
        return

    top100 = load_rank_band_curve(95, 105, "top100")
    top1000 = load_rank_band_curve(995, 1005, "top1000")
    log.info("leader curve (%d rounds):\n%s", len(leader), leader.to_string(index=False))

    # Model walk-forward cumulative
    wf_path = GOLD_DIR / "walk_forward_summary.parquet"
    if not wf_path.exists():
        log.warning("walk_forward_summary.parquet missing; re-run walk_forward first")
        return
    wf = pd.read_parquet(wf_path).sort_values("round").reset_index(drop=True)
    wf["cumulative_actual"] = wf["actual_pts"].cumsum()

    # ---- plot ----
    fig, ax = plt.subplots(figsize=(11, 6.5))

    ax.plot(leader["round"], leader["leader_overallPts"], marker="o", markersize=7,
            color="#d62728", linewidth=2.2, label="Cumulative leader (max overallPts)")
    if not top100.empty:
        ax.plot(top100["round"], top100["top100_overallPts"], marker="s", markersize=5,
                color="#9467bd", linewidth=1.6, alpha=0.85, label="~rank 100")
    if not top1000.empty:
        ax.plot(top1000["round"], top1000["top1000_overallPts"], marker="s", markersize=5,
                color="#ff7f0e", linewidth=1.6, alpha=0.85, label="~rank 1000")
    ax.plot(wf["round"], wf["cumulative_actual"], marker="X", markersize=8,
            color="#1f77b4", linewidth=2.2, label="Model walk-forward (R2+)")

    # Annotate the gap at the most-recent round
    if len(leader) and len(wf):
        last_round = min(leader["round"].max(), wf["round"].max())
        ldr_pts = leader.loc[leader["round"] == last_round, "leader_overallPts"].iloc[0]
        wf_pts = wf.loc[wf["round"] == last_round, "cumulative_actual"].iloc[0]
        gap = ldr_pts - wf_pts
        ax.annotate(
            f"GW{int(last_round)} gap: {int(gap)} pts ({100*wf_pts/ldr_pts:.0f}% of leader)",
            xy=(last_round, (ldr_pts + wf_pts) / 2),
            xytext=(-8, 0), textcoords="offset points",
            ha="right", color="#444", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", fc="#fff", ec="#ccc"),
        )

    ax.set_xlabel("Round")
    ax.set_ylabel("Cumulative points")
    ax.set_title(f"Cumulative points: model vs season leader (R1-R{int(leader['round'].max())})")
    ax.set_xticks(range(1, int(leader["round"].max()) + 1))
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left")

    plt.tight_layout()
    out = GOLD_DIR / "cumulative_vs_leader.png"
    plt.savefig(out, dpi=140)
    log.info("wrote %s", out)


if __name__ == "__main__":
    main()
