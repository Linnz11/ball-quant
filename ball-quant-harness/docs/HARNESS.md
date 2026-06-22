---
title: ball-quant research harness
tldr: |
  Four CLI commands form the research loop: capture (fetch+store odds), settle (load results),
  backtest (calibration+PnL report), optimize (walk-forward param search).
  Zero runtime dependencies — stdlib only.
---

# ball-quant Research Harness

## Research Loop

```
capture  →  settle  →  backtest  →  optimize
  (live)      (post)     (review)    (tune)
```

### `ballq capture`

Fetches a match's `EventMarketMatrix` from Polymarket (or an offline JSON fixture), optionally attaches a `MatchSP` from the China Sports Lottery SP file, and writes a `bq.snapshot.v1` record to the store.

```sh
ballq capture --slug fifwc-nld-jpn-2026-06-14 [--sp-file sp.csv] [--competition world-cup] [--store-root data/store]
```

Run hourly (or on-demand) before kick-off so the store has multiple pre-match snapshots. See `scripts/cron_refresh.sh`.

### `ballq settle`

Parses a results CSV (`match_id,home_score,away_score[,void]`) into `MatchOutcome` objects and persists them as JSON to `<store>/outcomes/results.json`.

```sh
ballq settle --results results.csv [--store-root data/store] [--out path/to/outcomes.json]
```

Run after each match day closes. The JSON can be passed as `--results` to backtest/optimize.

### `ballq backtest`

Replays all snapshots in a date range against known outcomes, grades selections and combos, and emits calibration + PnL + Kelly metrics in a Markdown report.

```sh
ballq backtest --from 2026-06-01 --to 2026-06-14 \
  --results data/store/outcomes/results.json \
  --report-out reports/backtest.md
```

See `scripts/nightly_backtest.sh` for the automated 30-day rolling run.

### `ballq optimize`

Walk-forward parameter search over `StrategyParams`. Splits the date range into train/test folds (no lookahead), scores each trial OOS, and picks the best by the chosen metric.

```sh
ballq optimize --from 2026-06-01 --to 2026-06-14 \
  --space '{"fractional_kelly":[0.1,0.2,0.25,0.3]}' \
  --metric brier --search grid --folds 3 \
  --results data/store/outcomes/results.json
```

For random search pass `--search random --max-trials 50`.

---

## StrategyParams Knobs

Key fields (see `core/params.py` for the full set):

| Field | Default | Effect |
|---|---:|---|
| `fractional_kelly` | 0.25 | Fraction of Kelly stake to bet — primary risk dial |
| `budget_a/b/c` | 0.60/0.30/0.10 | Budget allocation across stake tiers (A=high conf, C=speculative) |
| `cap_a/b/c` | 0.35/0.20/0.075 | Max fraction of budget per single bet in each tier |
| `total_hint_nudge` | 0.70 | Weight of Polymarket total-goals line on Poisson calibration |
| `calib_primary_iters` | 90 | Gradient iterations aligning score grid to moneyline constraint |
| `cs_mass_cap` | 0.85 | Max probability mass allowed on top correct-score outcomes |
| `typec_prob_lo/hi` | 0.05/0.12 | Type-C (speculative) selection probability window |

---

## Metrics

| Metric | Direction | Meaning |
|---|---|---|
| `brier` | min | Mean squared error of probability forecasts (0 = perfect, 1 = worst) |
| `log_loss` | min | Cross-entropy loss; penalises confident wrong predictions more than Brier |
| `ece` | min | Expected Calibration Error — bin-level reliability gap |
| `net_pnl` | max | Net profit/loss in stake units after all bets |
| `roi` | max | Net PnL / total stake — efficiency measure independent of scale |
| `geometric_growth_rate` | max | Kelly criterion: per-bet bankroll geometric mean growth rate |

A healthy run has `brier < 0.22`, `ece < 0.05`, positive `net_pnl`, and `geometric_growth_rate > 1.000`.

---

## Walk-Forward Discipline

`optimize_params` uses `walk_forward_splits` to ensure test folds are strictly **after** train folds.

Rules:
1. Never fit on data from the same match day as the test window.
2. `best_params` are selected by OOS score, not in-sample.
3. `overfit_gap > 0.05` triggers a report warning — investigate before deploying.
4. After each tournament phase, run settle → backtest → optimize → redeploy in sequence.

---

## Zero Runtime-Dep Guarantee

`ball-quant` uses **stdlib only** at runtime (Python 3.9+):
- HTTP: `urllib.request`
- JSON: `json`
- CSV: `csv`
- Logging: `logging`
- Config: `os`, `json`, `dataclasses`

`pytest` and `coverage` are dev-only (listed under `[project.optional-dependencies].dev`). No third-party packages are imported at any call site that runs in production.
