# TODO

## Revisit calibration once we have more data

Current state: **selective P80-only isotonic calibration** is on by default
(`CALIBRATE_QUANTILES = {0.8}` in [src/ml/walk_forward.py](src/ml/walk_forward.py)).
On the 13-round backtest this gave **44.9% of oracle**, beating both uncalibrated
(44.3%) and full-quantile calibration (44.0%). The reason full calibration loses:
the calibration set is one round (~330 rows), so isotonic builds step-function
plateaus that collapse mid-tier P60 predictions onto a few discrete levels —
mid-tier discrimination dies and the optimiser falls back to a barbell roster.

When data is richer (say ~30+ completed rounds, ~10k+ rows in the calibration pool),
isotonic should smooth out and full-quantile calibration may start winning. **Revisit
both:**

1. **Switch `CALIBRATE_QUANTILES` back to `None`** (calibrate all quantiles) and
   re-run walk-forward. If % of oracle improves over the selective baseline,
   adopt full calibration as default.
2. **Try other calibration methods** — quantile-specific Platt / sigmoid scaling
   (smoother than isotonic, less prone to plateaus), conformalised quantile
   regression (CQR), or simple per-quantile bias offset (cheap, robust on thin
   data).
3. **Re-evaluate the captain-specific model** (deleted [src/ml/captain_model.py]
   in this session — it cost 2.1pp at current data size). Same dynamic: the
   start-only training pool was too thin. With more rounds it might earn its
   keep.

Re-check the trigger: when the calibration-round size in
`fit_for_round_calibrated.info["calib_rows"]` is consistently ≥1,000, run the
comparison.

When re-running each calibration variant, **regenerate all four chip teams**
(normal / triple captain / limitless / co-captains) and compare composition +
captain pick across configurations — the chip teams are the visible
decision-relevant output and shifts in roster shape (e.g. barbell vs balanced)
reveal calibration artefacts faster than a single Spearman / oracle-% number.
Walk-forward already saves these to `data/gold/optimal_team_round15_<chip>.parquet`
each run; just diff between calibration variants.

## Revisit per-position models with richer data

Tested 2026-05-18 against the global-model baseline of 55.6% of oracle:
**per-position came in at 54.0% (−1.6pp)** despite each position having 450–950
cumulative training rows by R14. The fattest position (loose_forward, 952 rows)
already splits across 5 quantile models; thin positions (hooker, fly_half,
scrum_half at ~450 rows) overfit and produce volatile captain picks (round 10
dropped to 40% of oracle vs 62% for the global model).

Code/flag is preserved — `PER_POSITION` in [src/ml/walk_forward.py](src/ml/walk_forward.py)
plus `fit_per_position_for_round` / `predict_quantiles_per_position` in
[src/ml/quantile_model.py](src/ml/quantile_model.py). Re-run via
`python -m src.ml.walk_forward --per-position`.

Revisit when each position has ~1,500+ cumulative training rows (likely after
one more full season) — at that point the position-specific signal should
overcome the row-budget penalty.

## Volatility + box-score features (highest-leverage remaining model gain)

Currently `pts_roll3_std` is the only volatility signal. Adding `pts_roll5_std`,
`boom_share_l5` (fraction of last 5 games ≥50 pts), `bust_share_l5` (fraction ≤10),
and `pts_coefficient_of_variation` would sharpen captain-vs-floor discrimination
substantially — these directly encode the boom/bust shape that captain P80 wants.

Per-round box-score components (tries, tackles, kicks, metres) are unavailable
from PFR's JSON. They'd be a much bigger lift if recoverable — but require the
browser-based scraper below to land first.



## Browser-based scraper to bypass Cloudflare (historical SRP stats)

**Why.** Measured a +2.6pp lift (44.3% → 46.9% of oracle in walk-forward) from
adding historical per-match player stats as static priors. That data came from
the `rugbypy` package, which was removed per the project's "known APIs only"
policy. The actual upstream (ESPN StatsGuru) is now closed to plain HTTP:

| Endpoint | Result |
|---|---|
| `stats.espnscrum.com` (TLS) | TCP connection refused |
| `en.espn.co.uk/statsguru/*` | 403 Forbidden |
| `en.espn.co.uk/scrum/*` | 403 Forbidden |
| `www.espn.com/rugby/*` | 202 (Cloudflare JS challenge, empty body) |
| `super.rugby/superrugby/{fixtures-and-results,players-stats,match-centre}/` | 503 (anti-bot) |
| `rugbypass.com/super-rugby/stats` | 200 — season totals only, no per-match drill-down |
| `rugbypass.com/live/<teams>/?g=<match_id>` | 200 — team-level only, no per-player breakdown in HTML |

Plain `requests` cannot get through. To recover the lift without a paid API we
need a real browser session.

**Goal.** A scraper that pulls per-match per-player stats for SRP 2022+ from a
public source (no paid API), writes to silver in a stable schema, and runs
reliably enough to refresh weekly.

**Approach options**, in order of likely-to-work-first:

1. **Playwright** (Python) with headless Chromium + stealth plugins. Generally
   beats Cloudflare's JS challenge better than Selenium and has a cleaner API.
2. **`undetected-chromedriver`** + Selenium — well-documented Cloudflare bypass
   pattern, lots of community support.
3. **`cloudscraper`** — pure-Python; lightest weight; works only on simpler
   challenges, often not enough for full JS challenges.
4. **FlareSolverr** sidecar service — local proxy that solves challenges and
   exposes a clean HTTP endpoint. Useful if scraping at any scale.

**Target sources**, ranked by data quality vs effort:

1. **RugbyPass `/live/<teams>/?g=<match_id>`** — post-match pages likely have
   richer player data than the pre-match preview I tested. Cloudflare-protected.
   Most realistic and least obviously TOS-breaking.
2. **super.rugby Match Centre** — anti-bot returning 503. Worth retrying with
   a real browser session; if it works it's the most "official" source.
3. **ESPN StatsGuru** — the original column-format match. Server refusing
   connections; might be permanently dead. Try a residential proxy or shelve.

**Integration hook.** Write to `data/silver/historical_player_match_stats.parquet`
with columns at minimum:
`pfr_player_id, game_date, season, position, est_fantasy_points, started, points_for, points_against`.
Then re-enable `_compute_historical_priors()` in [src/etl/silver.py](src/etl/silver.py)
(comment block in that file points to this TODO).

**Constraints.**
- Respect each source's `robots.txt` and TOS. RugbyPass and ESPN generally do
  not permit automated scraping in their TOS — read first.
- Rate-limit (≥1 req / 2-5s, randomised) and cache HTML to
  `data/bronze/raw/html_<source>/` like [src/scraping/bronze.py](src/scraping/bronze.py) does.
- If this turns out to need more than weekly refresh or repeated anti-bot
  re-engineering, that's the signal to bite the bullet and pay for Stats
  Perform or Highlightly's paid tier instead.

**Effort.** 1-3 days depending on which option/target works first. RugbyPass
match pages are the highest-fidelity output and also the hardest path.

**Acceptance.**
- ≥3 full seasons of SRP per-match per-player stats land in silver.
- Walk-forward % of oracle recovers to ≥46% (matches the lift we measured).
- Pipeline is idempotent: re-runs are cheap (HTML cache) and incrementally pick
  up newly-played rounds.
