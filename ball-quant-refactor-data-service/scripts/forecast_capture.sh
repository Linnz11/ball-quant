#!/usr/bin/env bash
# Pre-kickoff forecast capture for the closed-loop calibration ledger (#22).
#
# Cron runs this a few times/day. `bundle --forecast-ledger` persists ONE
# forecast record per match to data/forecasts/ledger.jsonl (each tagged with a
# pre_kickoff flag from the slug kickoff date). grade_forecasts dedups to the
# latest pre_kickoff record per match, so running this repeatedly is safe — the
# closest-to-kickoff valid snapshot wins. A forecast MUST be captured pre-kickoff
# to count; post-kickoff captures are flagged and excluded at grade time.
set -uo pipefail
cd "$(dirname "$0")/.."
D="$(TZ=Asia/Shanghai date +%F)"
.venv/bin/ballq bundle --date "$D" --c500-live --forecast-ledger data/forecasts/ledger.jsonl >> logs/forecast-cron.log 2>&1
