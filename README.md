# super-rugby

End-to-end fantasy pipeline for [PlayFantasyRugby Super Rugby Pacific](https://www.playfantasyrugby.com/super/my-team):
scrape, build features, train a quantile model, optimise a salary-capped XV via MILP,
and (if you want) submit the team back to the PFR API.

TEAM: https://www.playfantasyrugby.com/super/my-team
RULES: https://www.playfantasyrugby.com/super/help/game-guidelines

## Requirements

- Python 3.12
- Windows, macOS, or Linux
- A PlayFantasyRugby account (only needed for live budget fetch, leaderboards, and team submission; the data pipeline itself runs on public endpoints)

## Setup

Clone and enter the repo:

```bash
git clone https://github.com/alexmgl/super-rugby.git
cd super-rugby
```

Create a virtual environment and install the dependencies. The project was developed with [`uv`](https://github.com/astral-sh/uv) but plain `venv` works too.

Using `uv`:

```bash
uv venv --python 3.12
uv pip install -r requirements.txt
```

Using stock `venv`:

```bash
python -m venv .venv
.venv/Scripts/activate            # Windows PowerShell: .venv\Scripts\Activate.ps1
.venv/bin/activate                # macOS / Linux
pip install -r requirements.txt
```

Copy the example env file and fill in your PlayFantasyRugby credentials. This file is gitignored.

```bash
cp .env.example .env
```

```
PFR_EMAIL=you@example.com
PFR_PASSWORD=your-password
```

## Running the pipeline

The full pipeline (scrape, ETL, ML, optimiser) runs from the repo root:

```bash
python main.py
```

Stages, in order (todo - will need to be updated in future gameweeks):

1. **Bronze**: scrape PlayFantasyRugby and Ultimate Rugby into `data/bronze/`.
2. **Silver**: clean and join into player and fixture tables under `data/silver/`.
3. **Gold**: build the feature panel at `data/gold/feature_panel.parquet`.
4. **ML + MILP**: walk-forward evaluation across past rounds plus the next round-15 optimal team, saved to `data/gold/optimal_team_round15_*.parquet`.

### Running individual stages

```bash
python -m src.scraping.bronze            # refresh raw data
python -m src.etl.silver                 # silver layer only
python -m src.etl.gold                   # gold feature panel only
python -m src.ml.walk_forward            # train + walk-forward + round-15 picks
```

### Useful flags for `walk_forward`

```bash
python -m src.ml.walk_forward --budget 100000000    # override salary cap
python -m src.ml.walk_forward --skip-live-budget    # do not call PFR
python -m src.ml.walk_forward --per-position        # experimental per-position models
```

By default the script logs into PlayFantasyRugby and reads your live salary cap; the `--skip-live-budget` flag falls back to the optimiser default ($100M).

## Auxiliary scripts

Pull the top-2000 manager leaderboard into `data/bronze/fantasy_rankings_overall.csv`:

```bash
python -m src.scraping.fantasy_rankings
```

Dry-run the team submission (prints the JSON payload only):

```bash
python -m src.scraping.submit_team --chip normal
```

POST the team to PFR (prompts for `y/N` confirmation first):

```bash
python -m src.scraping.submit_team --chip normal --submit
```

Chip options: `normal`, `triple_captain`, `limitless`, `co_captains`. Pass `-y` to skip the confirmation prompt.

## Output locations

- `data/bronze/` raw scraped tables and JSON
- `data/silver/` cleaned tables (`players_dim.parquet`, `player_rounds.parquet`, etc.)
- `data/gold/` feature panel, walk-forward summary, round-15 optimal teams per chip
- `logs/` rotating application logs

## Notes

- The walk-forward results assume known starting line-ups for past rounds (`filter_to_observed=True`). Three days before a round, ~75% of players carry an `uncertain` status; rerun closer to kick-off once the squads are announced for a tighter candidate pool.
- The PFR salary cap is read live each run. Override with `--budget` if you want to test a hypothetical.
- See [TODO.md](TODO.md) for open work, including the browser-based historical-stats scraper and volatility features.
