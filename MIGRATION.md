# ball-quant Codex migration guide

This bundle is designed to move the project into another Codex workspace for continued optimization.

## Install from wheel

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install dist/ball_quant-0.1.0-py3-none-any.whl
ballq --help
```

## Develop from source

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e .
PYTHONPATH=src python -m unittest discover -s tests
```

## Refresh live Polymarket data

```bash
PYTHONPATH=src python -m ball_quant auto-refresh --timezone Asia/Shanghai --lookahead-hours 36
```

Important live outputs:

```text
data/cache/poly_worldcup_active_schedule.json
data/cache/poly_auto_refresh_status.json
data/cache/live/live_probability_summary.csv
data/cache/live/live_probability_snapshots.json
data/cache/live/*_probability.json
reports/live/
```

## Recreate Codex hourly automation

Create a Codex cron automation in the target workspace that runs:

```bash
PYTHONPATH=src python -m ball_quant auto-refresh --timezone Asia/Shanghai --lookahead-hours 36
```

Recommended schedule: hourly.

## What is intentionally excluded

The migration source bundle excludes large/generated live data:

- `data/cache/`
- `reports/live/`
- Python caches
- virtual environments

Run `auto-refresh` in the new workspace to regenerate them from live Polymarket data.

