#!/usr/bin/env bash
# nightly_backtest.sh — rolling-window backtest run at end of each match day.
#
# WHY: Running backtest nightly over a fixed look-back window (default 30 days) lets
# the team detect calibration drift early and trigger optimize before the live window.
# The report is written to reports/ so the next morning review can open it directly.
#
# Usage (cron example — runs nightly at 06:00 local time after Asian match day closes):
#   0 6 * * * /path/to/ball-quant/scripts/nightly_backtest.sh >> /var/log/ballq/backtest.log 2>&1
#
# Environment:
#   BALLQ_STORE_ROOT  — override snapshot store (default: data/store)
#   RESULTS_PATH      — path to outcomes CSV or JSON (required to grade matches)
#   LOOKBACK_DAYS     — rolling window in days (default: 30)
#   PROJECT_DIR       — project root; defaults to parent of this script

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$PROJECT_DIR"

export PYTHONPATH="${PYTHONPATH:-}:${PROJECT_DIR}/src"

BALLQ="python3 -m ball_quant"
LOOKBACK_DAYS="${LOOKBACK_DAYS:-30}"
RESULTS_PATH="${RESULTS_PATH:-}"

# Compute rolling window: today - LOOKBACK_DAYS to today.
DATE_TO=$(date +%Y-%m-%d)
DATE_FROM=$(python3 -c "
from datetime import date, timedelta
print((date.today() - timedelta(days=int('$LOOKBACK_DAYS'))).isoformat())
")

echo "[$(date -Iseconds)] Nightly backtest: from=$DATE_FROM to=$DATE_TO lookback=${LOOKBACK_DAYS}d"

if [ -z "$RESULTS_PATH" ]; then
  echo "[$(date -Iseconds)] RESULTS_PATH not set; using store outcomes if present"
  RESULTS_ARG=""
else
  RESULTS_ARG="--results $RESULTS_PATH"
fi

$BALLQ backtest \
  --from "$DATE_FROM" \
  --to "$DATE_TO" \
  ${BALLQ_STORE_ROOT:+--store-root "$BALLQ_STORE_ROOT"} \
  $RESULTS_ARG

echo "[$(date -Iseconds)] Nightly backtest complete"
