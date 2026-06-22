"""
CLI harness smoke tests (Phase 5).

Tests exercise the four new subcommands through cli.main() — no subprocess,
no network.  Capture is tested via the underlying capture_snapshot function
(the CLI capture subcommand requires a live slug or offline cache fixture
that is not part of this repo).  Backtest and optimize are tested end-to-end
through cli.main(argv).
"""
from __future__ import annotations

import csv
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

from ball_quant.cli import main as cli_main
from ball_quant.core.settlement import MatchOutcome
from ball_quant.data.capture import capture_snapshot
from ball_quant.data.store import read_snapshot
from ball_quant.models import EventMarketMatrix, MarketQuote, MatchSP


# ---------------------------------------------------------------------------
# Shared fixture factory
# ---------------------------------------------------------------------------

def _make_match() -> MatchSP:
    return MatchSP(
        match_id="smoke-001",
        date="2026-06-01",
        home="TeamA",
        away="TeamB",
        spf_home=1.90,
        spf_draw=3.40,
        spf_away=4.50,
        handicap=-1,
        rq_home=2.80,
        rq_draw=3.50,
        rq_away=2.10,
    )


def _make_matrix() -> EventMarketMatrix:
    return EventMarketMatrix(
        match_id="smoke-001",
        home="TeamA",
        away="TeamB",
        markets=[
            MarketQuote("mq1", "winner", "moneyline", "home", 0.60, spread=0.02, liquidity=9000),
            MarketQuote("mq1", "winner", "moneyline", "draw", 0.23, spread=0.02, liquidity=9000),
            MarketQuote("mq1", "winner", "moneyline", "away", 0.17, spread=0.02, liquidity=9000),
            MarketQuote("mq2", "TeamA -1.5", "handicap", "TeamA -1.5", 0.36, spread=0.03, liquidity=5000),
        ],
    )


def _seed_store(tmp_path: Path, n_snapshots: int = 1) -> tuple[Path, Path]:
    """
    Write n_snapshots snapshots into tmp_path/store and a results CSV.

    n_snapshots >= 3 is required when the optimize command is being tested with
    n_folds=2 (walk_forward_splits needs at least n_folds+1 records).  Each
    snapshot gets its own match_id so the results CSV can cover all of them.

    Returns (store_root, results_csv_path).
    """
    store_root = tmp_path / "store"

    results_csv = tmp_path / "results.csv"
    rows: list = [["match_id", "home_score", "away_score"]]

    for i in range(n_snapshots):
        mid = f"smoke-{i:03d}"
        match = MatchSP(
            match_id=mid,
            date="2026-06-01",
            home="TeamA",
            away="TeamB",
            spf_home=1.90,
            spf_draw=3.40,
            spf_away=4.50,
            handicap=-1,
            rq_home=2.80,
            rq_draw=3.50,
            rq_away=2.10,
        )
        matrix = EventMarketMatrix(
            match_id=mid,
            home="TeamA",
            away="TeamB",
            markets=[
                MarketQuote("mq1", "winner", "moneyline", "home", 0.60, spread=0.02, liquidity=9000),
                MarketQuote("mq1", "winner", "moneyline", "draw", 0.23, spread=0.02, liquidity=9000),
                MarketQuote("mq1", "winner", "moneyline", "away", 0.17, spread=0.02, liquidity=9000),
            ],
        )
        capture_snapshot(
            matrix=matrix,
            match_sp=match,
            root=store_root,
            # Distinct timestamps so walk_forward_splits can sort them correctly.
            captured_at=datetime(2026, 6, 1, 10 + i, 0, 0, tzinfo=timezone.utc),
            competition="smoke-test",
        )
        rows.append([mid, "2", "0"])

    with results_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    return store_root, results_csv


# ---------------------------------------------------------------------------
# settle subcommand
# ---------------------------------------------------------------------------

class TestSettleCommand:
    def test_settle_writes_json_and_exits_zero(self, tmp_path):
        _, results_csv = _seed_store(tmp_path)
        out_json = tmp_path / "outcomes.json"
        rc = cli_main([
            "settle",
            "--results", str(results_csv),
            "--out", str(out_json),
        ])
        assert rc == 0
        assert out_json.exists()
        data = json.loads(out_json.read_text(encoding="utf-8"))
        assert "smoke-000" in data

    def test_settle_default_store_path(self, tmp_path, monkeypatch):
        """When --out is not given, outcomes land in <store-root>/outcomes/results.json."""
        _, results_csv = _seed_store(tmp_path)
        store_root = tmp_path / "store"
        rc = cli_main([
            "settle",
            "--results", str(results_csv),
            "--store-root", str(store_root),
        ])
        assert rc == 0
        default_out = store_root / "outcomes" / "results.json"
        assert default_out.exists()


# ---------------------------------------------------------------------------
# backtest subcommand (main smoke path)
# ---------------------------------------------------------------------------

class TestBacktestCommand:
    def test_backtest_produces_report_file(self, tmp_path):
        store_root, results_csv = _seed_store(tmp_path)
        report_out = tmp_path / "bt_report.md"
        rc = cli_main([
            "backtest",
            "--from", "2026-06-01",
            "--to", "2026-06-01",
            "--store-root", str(store_root),
            "--results", str(results_csv),
            "--report-out", str(report_out),
        ])
        assert rc == 0, "backtest command must exit 0"
        assert report_out.exists(), "report file must be written"

    def test_backtest_report_contains_metric(self, tmp_path):
        store_root, results_csv = _seed_store(tmp_path)
        report_out = tmp_path / "bt_report.md"
        cli_main([
            "backtest",
            "--from", "2026-06-01",
            "--to", "2026-06-01",
            "--store-root", str(store_root),
            "--results", str(results_csv),
            "--report-out", str(report_out),
        ])
        content = report_out.read_text(encoding="utf-8")
        # Report must contain calibration section header.
        assert "Calibration" in content or "校准" in content, \
            f"Expected calibration block in report. Got prefix: {content[:400]}"

    def test_backtest_no_results_still_exits_zero(self, tmp_path):
        """Without outcomes the engine skips all records but must not crash."""
        store_root, _ = _seed_store(tmp_path)
        report_out = tmp_path / "bt_empty.md"
        rc = cli_main([
            "backtest",
            "--from", "2026-06-01",
            "--to", "2026-06-01",
            "--store-root", str(store_root),
            "--report-out", str(report_out),
        ])
        assert rc == 0
        assert report_out.exists()

    def test_backtest_empty_date_range_exits_zero(self, tmp_path):
        """A range that matches no snapshots should produce an empty report, not crash."""
        store_root, results_csv = _seed_store(tmp_path)
        report_out = tmp_path / "bt_empty_range.md"
        rc = cli_main([
            "backtest",
            "--from", "2020-01-01",
            "--to", "2020-01-01",
            "--store-root", str(store_root),
            "--results", str(results_csv),
            "--report-out", str(report_out),
        ])
        assert rc == 0
        assert report_out.exists()


# ---------------------------------------------------------------------------
# optimize subcommand (main smoke path)
# ---------------------------------------------------------------------------

class TestOptimizeCommand:
    """
    optimize needs at least n_folds+1 snapshot records for walk_forward_splits.
    We seed 3 snapshots and use n_folds=2 (requires 3 records: 2+1=3 chunks).
    """

    def _space_2pt(self) -> str:
        """A two-point grid space: fractional_kelly in [0.20, 0.25]."""
        return json.dumps({"fractional_kelly": [0.20, 0.25]})

    def test_optimize_produces_report_file(self, tmp_path):
        store_root, results_csv = _seed_store(tmp_path, n_snapshots=3)
        report_out = tmp_path / "opt_report.md"
        rc = cli_main([
            "optimize",
            "--from", "2026-06-01",
            "--to", "2026-06-01T99",
            "--space", self._space_2pt(),
            "--metric", "brier",
            "--search", "grid",
            "--folds", "2",
            "--store-root", str(store_root),
            "--results", str(results_csv),
            "--report-out", str(report_out),
        ])
        assert rc == 0, "optimize command must exit 0"
        assert report_out.exists(), "optimize report must be written"

    def test_optimize_report_contains_best_params_section(self, tmp_path):
        store_root, results_csv = _seed_store(tmp_path, n_snapshots=3)
        report_out = tmp_path / "opt_report.md"
        cli_main([
            "optimize",
            "--from", "2026-06-01",
            "--to", "2026-06-01T99",
            "--space", self._space_2pt(),
            "--metric", "brier",
            "--search", "grid",
            "--folds", "2",
            "--store-root", str(store_root),
            "--results", str(results_csv),
            "--report-out", str(report_out),
        ])
        content = report_out.read_text(encoding="utf-8")
        # Must contain best-params section header.
        assert "Best" in content or "最优" in content, \
            f"Expected best-params section. Got prefix: {content[:400]}"

    def test_optimize_random_search_with_max_trials(self, tmp_path):
        """Random search requires max_trials; must exit 0 and write a report."""
        store_root, results_csv = _seed_store(tmp_path, n_snapshots=3)
        report_out = tmp_path / "opt_random.md"
        # For random search the space values are [lo, hi] pairs encoded as lists.
        space = json.dumps({"fractional_kelly": [0.10, 0.30]})
        rc = cli_main([
            "optimize",
            "--from", "2026-06-01",
            "--to", "2026-06-01T99",
            "--space", space,
            "--metric", "brier",
            "--search", "random",
            "--max-trials", "3",
            "--folds", "2",
            "--store-root", str(store_root),
            "--results", str(results_csv),
            "--report-out", str(report_out),
        ])
        assert rc == 0
        assert report_out.exists()


# ---------------------------------------------------------------------------
# capture subcommand (function-level test — avoids network)
# ---------------------------------------------------------------------------

class TestCaptureFunction:
    """Tests the capture_snapshot function directly instead of the CLI subcommand
    to avoid the network dependency on Polymarket slug resolution."""

    def test_capture_writes_snapshot_and_reads_back(self, tmp_path):
        store_root = tmp_path / "store"
        matrix = _make_matrix()
        match = _make_match()
        snap_path = capture_snapshot(
            matrix=matrix,
            match_sp=match,
            root=store_root,
            captured_at=datetime(2026, 6, 1, 9, 0, 0, tzinfo=timezone.utc),
            competition="smoke",
        )
        assert snap_path.exists()
        record = read_snapshot(snap_path)
        assert record["match_id"] == "smoke-001"
        assert record["schema"] == "bq.snapshot.v1"
        assert record["sp"] is not None
        assert record["sp"]["spf_home"] == pytest.approx(1.90)

    def test_capture_without_sp(self, tmp_path):
        store_root = tmp_path / "store"
        matrix = _make_matrix()
        snap_path = capture_snapshot(
            matrix=matrix,
            match_sp=None,
            root=store_root,
            captured_at=datetime(2026, 6, 1, 8, 0, 0, tzinfo=timezone.utc),
        )
        record = read_snapshot(snap_path)
        assert record["sp"] is None
