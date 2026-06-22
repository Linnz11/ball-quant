#!/usr/bin/env bash
# cron_refresh.sh — hourly ops loop: refresh active schedule + capture live snapshots.
#
# WHY: Polymarket odds shift between matches; capturing at regular intervals lets the
# backtest compare odds at different points in the pre-match window.
#
# Usage (cron example — runs every hour):
#   0 * * * * /path/to/ball-quant/scripts/cron_refresh.sh >> /var/log/ballq/refresh.log 2>&1
#
# Environment:
#   BALLQ_STORE_ROOT  — override snapshot store (default: data/store)
#   ACTIVE_SLUGS      — space-separated list of slugs to capture; empty = no captures
#   PROJECT_DIR       — project root; defaults to parent of this script

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$PROJECT_DIR"

# Ensure the package is importable regardless of install state.
export PYTHONPATH="${PYTHONPATH:-}:${PROJECT_DIR}/src"

BALLQ="python3 -m ball_quant"

echo "[$(date -Iseconds)] Starting cron_refresh"

# Step 1: refresh active schedule and live match matrices + probability reports.
# This updates data/cache/poly_worldcup_active_schedule.json and reports/live/.
$BALLQ auto-refresh \
  --lookahead-hours 36 \
  --expire-after-hours 3

echo "[$(date -Iseconds)] auto-refresh complete"

# Step 2: capture snapshots for any active slugs listed in ACTIVE_SLUGS.
# Example: export ACTIVE_SLUGS="fifwc-nld-jpn-2026-06-14 fifwc-eng-usa-2026-06-14"
if [ -n "${ACTIVE_SLUGS:-}" ]; then
  for slug in $ACTIVE_SLUGS; do
    echo "[$(date -Iseconds)] Capturing snapshot for slug=$slug"
    $BALLQ capture --slug "$slug" ${BALLQ_STORE_ROOT:+--store-root "$BALLQ_STORE_ROOT"}
  done
  echo "[$(date -Iseconds)] Captures complete"
else
  echo "[$(date -Iseconds)] ACTIVE_SLUGS not set — skipping capture step"
fi

echo "[$(date -Iseconds)] cron_refresh done"
