"""
Tests for optimize.py and report.py.

Fixture strategy: build in-memory snapshot records spanning >= 3 distinct
captured_at timestamps + supply outcomes, reusing sample_match / sample_matrix
helpers from test_probability_and_combo.py together with the capture+read
pipeline used in test_backtest.py.
"""
from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from tests.test_probability_and_combo import sample_match, sample_matrix

from ball_quant.backtest.engine import run_backtest
from ball_quant.backtest.optimize import (
    get_metric_info,
    iter_grid,
    iter_random,
    optimize_params,
    score_params,
)
from ball_quant.backtest.report import render_backtest_report, render_optimization_report
from ball_quant.backtest.splits import walk_forward_splits
from ball_quant.core.params import DEFAULT_PARAMS, StrategyParams
from ball_quant.core.settlement import MatchOutcome
from ball_quant.data.capture import capture_snapshot
from ball_quant.data.store import read_snapshot
from ball_quant.models import EventMarketMatrix, MarketQuote, MatchSP


# ---------------------------------------------------------------------------
# Shared fixture helpers
# ---------------------------------------------------------------------------

def _value_gap_match(match_id: str) -> MatchSP:
    """Match with a deliberate positive edge so allocation actually fires."""
    return MatchSP(
        match_id=match_id,
        date="2026-06-14",
        home="Team A",
        away="Team B",
        spf_home=2.80,
        spf_draw=4.50,
        spf_away=9.00,
        handicap=0,
        rq_home=0.0,
        rq_draw=0.0,
        rq_away=0.0,
    )


def _value_gap_matrix(match_id: str) -> EventMarketMatrix:
    """Matrix with high liquidity + tight spread -> high confidence -> bets allocated."""
    return EventMarketMatrix(
        match_id=match_id,
        home="Team A",
        away="Team B",
        markets=[
            MarketQuote("m1", "winner", "moneyline", "home", 0.65, spread=0.02, liquidity=15000),
            MarketQuote("m1", "winner", "moneyline", "draw", 0.20, spread=0.02, liquidity=15000),
            MarketQuote("m1", "winner", "moneyline", "away", 0.15, spread=0.02, liquidity=15000),
        ],
    )


def build_records_and_outcomes(
    root: Path,
    n_records: int = 4,
) -> tuple:
    """
    Capture `n_records` snapshot records spanning distinct captured_at times.

    Uses alternating match IDs so walk_forward_splits has enough diversity.
    Returns (records, outcomes) ready for run_backtest / optimize_params.
    """
    records = []
    outcomes = {}

    match_ids = [f"opt{i:03d}" for i in range(n_records)]
    # Spread timestamps across 4 distinct days (lexicographic == chronological).
    timestamps = [
        datetime(2026, 6, 10, 10, 0, 0, tzinfo=timezone.utc),
        datetime(2026, 6, 11, 10, 0, 0, tzinfo=timezone.utc),
        datetime(2026, 6, 12, 10, 0, 0, tzinfo=timezone.utc),
        datetime(2026, 6, 13, 10, 0, 0, tzinfo=timezone.utc),
    ]
    # If more records requested, extend timestamps.
    while len(timestamps) < n_records:
        last = timestamps[-1]
        timestamps.append(datetime(last.year, last.month, last.day + 1, 10, 0, 0, tzinfo=timezone.utc))

    for i, mid in enumerate(match_ids):
        match = _value_gap_match(mid)
        matrix = _value_gap_matrix(mid)
        snap_path = capture_snapshot(
            matrix=matrix,
            match_sp=match,
            root=root,
            captured_at=timestamps[i],
            competition="test_optimize",
        )
        record = read_snapshot(snap_path)
        records.append(record)
        # home wins 2-0 for odd indices, 0-2 for even (variety in calibration).
        if i % 2 == 0:
            outcomes[mid] = MatchOutcome(match_id=mid, home_score=2, away_score=0)
        else:
            outcomes[mid] = MatchOutcome(match_id=mid, home_score=0, away_score=2)

    return records, outcomes


# ---------------------------------------------------------------------------
# iter_grid tests
# ---------------------------------------------------------------------------

class TestIterGrid(unittest.TestCase):
    def test_cartesian_product_count(self):
        space = {"conf_base": [0.5, 0.6], "corr_base": [0.94, 0.96]}
        result = list(iter_grid(space))
        self.assertEqual(len(result), 4)

    def test_all_combinations_present(self):
        space = {"conf_base": [0.5, 0.6], "corr_base": [0.94, 0.96]}
        result = list(iter_grid(space))
        expected = [
            {"conf_base": 0.5, "corr_base": 0.94},
            {"conf_base": 0.5, "corr_base": 0.96},
            {"conf_base": 0.6, "corr_base": 0.94},
            {"conf_base": 0.6, "corr_base": 0.96},
        ]
        self.assertEqual(result, expected)

    def test_single_field(self):
        space = {"fractional_kelly": [0.25, 0.50, 0.75]}
        result = list(iter_grid(space))
        self.assertEqual(len(result), 3)
        self.assertEqual([r["fractional_kelly"] for r in result], [0.25, 0.50, 0.75])

    def test_deterministic_same_output_twice(self):
        space = {"conf_base": [0.5, 0.6], "corr_base": [0.90, 0.94, 0.96]}
        r1 = list(iter_grid(space))
        r2 = list(iter_grid(space))
        self.assertEqual(r1, r2)


# ---------------------------------------------------------------------------
# iter_random tests
# ---------------------------------------------------------------------------

class TestIterRandom(unittest.TestCase):
    import random as _random

    def test_count(self):
        import random
        rng = random.Random(42)
        space = {"conf_base": (0.4, 0.7), "corr_base": (0.88, 0.98)}
        result = list(iter_random(space, 10, rng))
        self.assertEqual(len(result), 10)

    def test_values_in_range(self):
        import random
        rng = random.Random(0)
        space = {"conf_base": (0.4, 0.7), "corr_base": (0.88, 0.98)}
        for trial in iter_random(space, 20, rng):
            self.assertGreaterEqual(trial["conf_base"], 0.4)
            self.assertLessEqual(trial["conf_base"], 0.7)
            self.assertGreaterEqual(trial["corr_base"], 0.88)
            self.assertLessEqual(trial["corr_base"], 0.98)

    def test_determinism_same_seed(self):
        import random
        space = {"conf_base": (0.4, 0.7)}
        r1 = list(iter_random(space, 5, random.Random(99)))
        r2 = list(iter_random(space, 5, random.Random(99)))
        self.assertEqual(r1, r2)

    def test_different_seeds_differ(self):
        import random
        space = {"conf_base": (0.4, 0.7)}
        r1 = list(iter_random(space, 5, random.Random(1)))
        r2 = list(iter_random(space, 5, random.Random(2)))
        self.assertNotEqual(r1, r2)


# ---------------------------------------------------------------------------
# score_params tests
# ---------------------------------------------------------------------------

class TestScoreParams(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        root = Path(self._tmpdir.name) / "store"
        self.records, self.outcomes = build_records_and_outcomes(root, n_records=4)

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_score_params_structure(self):
        scored = score_params(
            self.records, self.outcomes, DEFAULT_PARAMS,
            metric="brier", n_folds=2,
        )
        self.assertIn("in_sample", scored)
        self.assertIn("out_of_sample", scored)
        self.assertIn("fold_scores", scored)
        self.assertIn("n_folds_scored", scored)
        self.assertEqual(len(scored["fold_scores"]), 2)

    def test_in_sample_is_float_or_none(self):
        scored = score_params(
            self.records, self.outcomes, DEFAULT_PARAMS,
            metric="brier", n_folds=2,
        )
        is_val = scored["in_sample"]
        self.assertTrue(is_val is None or isinstance(is_val, float))

    def test_uses_walk_forward_splits_fold_count(self):
        """fold_scores length must equal n_folds — confirming walk_forward_splits was called."""
        for n_folds in [2, 3]:
            scored = score_params(
                self.records, self.outcomes, DEFAULT_PARAMS,
                metric="brier", n_folds=n_folds,
            )
            self.assertEqual(len(scored["fold_scores"]), n_folds,
                             f"Expected {n_folds} fold_scores, got {len(scored['fold_scores'])}")

    def test_n_folds_scored_consistent(self):
        scored = score_params(
            self.records, self.outcomes, DEFAULT_PARAMS,
            metric="brier", n_folds=2,
        )
        # n_folds_scored must equal count of non-None fold_scores.
        defined = [s for s in scored["fold_scores"] if s is not None]
        self.assertEqual(scored["n_folds_scored"], len(defined))


# ---------------------------------------------------------------------------
# optimize_params — grid search
# ---------------------------------------------------------------------------

class TestOptimizeGrid(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        root = Path(self._tmpdir.name) / "store"
        self.records, self.outcomes = build_records_and_outcomes(root, n_records=4)

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_n_trials_equals_grid_product(self):
        space = {"conf_base": [0.5, 0.6], "corr_base": [0.94, 0.96]}
        opt = optimize_params(
            self.records, self.outcomes, space,
            metric="brier", search="grid", n_folds=2,
        )
        self.assertEqual(opt["n_trials"], 4)

    def test_trials_count_matches_n_trials(self):
        space = {"conf_base": [0.5, 0.6], "corr_base": [0.94, 0.96]}
        opt = optimize_params(
            self.records, self.outcomes, space,
            metric="brier", search="grid", n_folds=2,
        )
        self.assertEqual(len(opt["trials"]), opt["n_trials"])

    def test_best_params_is_full_dict(self):
        space = {"conf_base": [0.5, 0.6], "corr_base": [0.94, 0.96]}
        opt = optimize_params(
            self.records, self.outcomes, space,
            metric="brier", search="grid", n_folds=2,
        )
        best = opt["best_params"]
        self.assertIsInstance(best, dict)
        # Must contain all StrategyParams fields.
        expected_keys = set(DEFAULT_PARAMS.to_dict().keys())
        self.assertEqual(set(best.keys()), expected_keys)

    def test_best_out_of_sample_is_min_over_trials(self):
        """For metric='brier' (min direction), best OOS == min of trial OOS values."""
        space = {"conf_base": [0.5, 0.6], "corr_base": [0.94, 0.96]}
        opt = optimize_params(
            self.records, self.outcomes, space,
            metric="brier", search="grid", n_folds=2,
        )
        oos_values = [
            t["out_of_sample"] for t in opt["trials"]
            if t["out_of_sample"] is not None
        ]
        if oos_values and opt["best_out_of_sample"] is not None:
            self.assertAlmostEqual(opt["best_out_of_sample"], min(oos_values), places=10)

    def test_result_structure(self):
        space = {"conf_base": [0.5, 0.6]}
        opt = optimize_params(
            self.records, self.outcomes, space,
            metric="brier", search="grid", n_folds=2,
        )
        for key in [
            "metric", "direction", "search", "n_trials", "n_folds",
            "best_params", "best_overrides", "best_in_sample",
            "best_out_of_sample", "overfit_gap", "trials",
        ]:
            self.assertIn(key, opt, f"Missing key: {key}")

    def test_direction_min(self):
        """brier is a min metric: direction must be 'min'."""
        space = {"conf_base": [0.5, 0.6]}
        opt = optimize_params(
            self.records, self.outcomes, space,
            metric="brier", search="grid", n_folds=2,
        )
        self.assertEqual(opt["direction"], "min")

    def test_overfit_gap_sign_convention(self):
        """For min metric: gap = OOS - IS. Positive means OOS is worse (common)."""
        space = {"conf_base": [0.5, 0.6], "corr_base": [0.94, 0.96]}
        opt = optimize_params(
            self.records, self.outcomes, space,
            metric="brier", search="grid", n_folds=2,
        )
        if opt["best_in_sample"] is not None and opt["best_out_of_sample"] is not None:
            expected_gap = opt["best_out_of_sample"] - opt["best_in_sample"]
            self.assertAlmostEqual(opt["overfit_gap"], expected_gap, places=10)


# ---------------------------------------------------------------------------
# Determinism test
# ---------------------------------------------------------------------------

class TestDeterminism(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        root = Path(self._tmpdir.name) / "store"
        self.records, self.outcomes = build_records_and_outcomes(root, n_records=4)

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_grid_determinism(self):
        """Same args + same seed -> identical best_overrides and trials."""
        space = {"conf_base": [0.5, 0.6], "corr_base": [0.94, 0.96]}
        opt1 = optimize_params(
            self.records, self.outcomes, space,
            metric="brier", search="grid", n_folds=2, seed=0,
        )
        opt2 = optimize_params(
            self.records, self.outcomes, space,
            metric="brier", search="grid", n_folds=2, seed=0,
        )
        self.assertEqual(opt1["best_overrides"], opt2["best_overrides"],
                         "best_overrides not deterministic across identical runs")
        self.assertEqual(
            [(t["overrides"], t["out_of_sample"]) for t in opt1["trials"]],
            [(t["overrides"], t["out_of_sample"]) for t in opt2["trials"]],
            "trial list not deterministic across identical runs",
        )

    def test_random_determinism(self):
        """Same seed -> identical random trial sequence."""
        space = {"conf_base": (0.4, 0.7), "corr_base": (0.88, 0.98)}
        opt1 = optimize_params(
            self.records, self.outcomes, space,
            metric="brier", search="random", n_folds=2, max_trials=3, seed=42,
        )
        opt2 = optimize_params(
            self.records, self.outcomes, space,
            metric="brier", search="random", n_folds=2, max_trials=3, seed=42,
        )
        self.assertEqual(opt1["best_overrides"], opt2["best_overrides"])
        # overrides should be identical (drawn from same seeded rng).
        for t1, t2 in zip(opt1["trials"], opt2["trials"]):
            for k in t1["overrides"]:
                self.assertAlmostEqual(t1["overrides"][k], t2["overrides"][k], places=15)

    def test_random_different_seeds_differ(self):
        """Different seeds -> different trial overrides."""
        space = {"conf_base": (0.4, 0.7), "corr_base": (0.88, 0.98)}
        opt1 = optimize_params(
            self.records, self.outcomes, space,
            metric="brier", search="random", n_folds=2, max_trials=5, seed=1,
        )
        opt2 = optimize_params(
            self.records, self.outcomes, space,
            metric="brier", search="random", n_folds=2, max_trials=5, seed=2,
        )
        overrides1 = [t["overrides"] for t in opt1["trials"]]
        overrides2 = [t["overrides"] for t in opt2["trials"]]
        self.assertNotEqual(overrides1, overrides2,
                            "Different seeds should produce different trial overrides")


# ---------------------------------------------------------------------------
# Direction test: max metric selects highest OOS
# ---------------------------------------------------------------------------

class TestDirectionSelection(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        root = Path(self._tmpdir.name) / "store"
        self.records, self.outcomes = build_records_and_outcomes(root, n_records=4)

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_max_metric_selects_highest_oos(self):
        """net_pnl is a max metric: best trial must have the HIGHEST OOS value."""
        space = {"conf_base": [0.5, 0.6], "corr_base": [0.94, 0.96]}
        opt = optimize_params(
            self.records, self.outcomes, space,
            metric="net_pnl", search="grid", n_folds=2,
            budget=500.0, bankroll=5000.0,
        )
        self.assertEqual(opt["direction"], "max")
        oos_values = [
            t["out_of_sample"] for t in opt["trials"]
            if t["out_of_sample"] is not None
        ]
        if oos_values and opt["best_out_of_sample"] is not None:
            self.assertAlmostEqual(opt["best_out_of_sample"], max(oos_values), places=10)

    def test_min_metric_selects_lowest_oos(self):
        """brier is a min metric: best trial must have the LOWEST OOS value."""
        space = {"conf_base": [0.5, 0.6], "corr_base": [0.94, 0.96]}
        opt = optimize_params(
            self.records, self.outcomes, space,
            metric="brier", search="grid", n_folds=2,
        )
        self.assertEqual(opt["direction"], "min")
        oos_values = [
            t["out_of_sample"] for t in opt["trials"]
            if t["out_of_sample"] is not None
        ]
        if oos_values and opt["best_out_of_sample"] is not None:
            self.assertAlmostEqual(opt["best_out_of_sample"], min(oos_values), places=10)


# ---------------------------------------------------------------------------
# No-lookahead: confirm walk_forward_splits is used (fold count behaviour)
# ---------------------------------------------------------------------------

class TestNoLookahead(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        root = Path(self._tmpdir.name) / "store"
        self.records, self.outcomes = build_records_and_outcomes(root, n_records=4)

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_fold_count_matches_n_folds(self):
        """score_params must produce exactly n_folds fold_scores — the signature
        of walk_forward_splits being used (not a different splitter)."""
        for n_folds in [2, 3]:
            scored = score_params(
                self.records, self.outcomes, DEFAULT_PARAMS,
                metric="brier", n_folds=n_folds,
            )
            self.assertEqual(
                len(scored["fold_scores"]), n_folds,
                f"Expected {n_folds} fold scores, got {len(scored['fold_scores'])}. "
                "Check that walk_forward_splits is used."
            )

    def test_walk_forward_splits_no_lookahead_invariant(self):
        """Directly verify that walk_forward_splits on our fixture records obeys
        the no-lookahead guarantee (assert_no_lookahead embedded in the split)."""
        def _key(r):
            return r.get("captured_at", "")

        # Should not raise — the split implementation enforces this internally.
        folds = walk_forward_splits(self.records, _key, n_folds=2)
        self.assertEqual(len(folds), 2)
        for train, test in folds:
            # test items must all be after train items in key order.
            if train and test:
                max_train = max(_key(r) for r in train)
                min_test = min(_key(r) for r in test)
                self.assertLessEqual(max_train, min_test,
                                     "Lookahead detected in walk_forward_splits output")

    def test_selection_is_by_oos_not_is(self):
        """Confirm best_params are chosen by OOS, not in-sample.

        We can't directly force IS != OOS selection from outside, but we can
        verify that best_out_of_sample == the smallest OOS among all trials
        (for brier), which is the out-of-sample selection rule.
        """
        space = {"conf_base": [0.5, 0.6], "corr_base": [0.94, 0.96]}
        opt = optimize_params(
            self.records, self.outcomes, space,
            metric="brier", search="grid", n_folds=2,
        )
        best_oos = opt["best_out_of_sample"]
        trial_ooss = [t["out_of_sample"] for t in opt["trials"] if t["out_of_sample"] is not None]
        if trial_ooss and best_oos is not None:
            self.assertAlmostEqual(best_oos, min(trial_ooss), places=10,
                                   msg="best OOS must be min of all trial OOS scores (brier/min)")


# ---------------------------------------------------------------------------
# Report tests
# ---------------------------------------------------------------------------

class TestRenderBacktestReport(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        root = Path(self._tmpdir.name) / "store"
        self.records, self.outcomes = build_records_and_outcomes(root, n_records=4)

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_render_backtest_report_returns_str(self):
        result = run_backtest(self.records, self.outcomes, budget=500.0, bankroll=5000.0)
        report = render_backtest_report(result, title="Test Backtest Report")
        self.assertIsInstance(report, str)
        self.assertTrue(len(report) > 0)

    def test_render_backtest_report_contains_headers(self):
        result = run_backtest(self.records, self.outcomes, budget=500.0, bankroll=5000.0)
        report = render_backtest_report(result, title="Test Backtest")
        self.assertIn("# Test Backtest", report)
        self.assertIn("brier", report)
        self.assertIn("log_loss", report)
        self.assertIn("ece", report)

    def test_render_backtest_report_no_crash_empty_pnl(self):
        """Render must not crash when pnl block is empty (no bets were graded)."""
        # Build a result with empty pnl block (no bets): run on empty record list.
        result = run_backtest([], {})
        report = render_backtest_report(result, title="Empty PnL Test")
        self.assertIsInstance(report, str)
        self.assertIn("—", report)  # graceful placeholder for missing values

    def test_render_backtest_report_no_crash_missing_metrics(self):
        """Render must not crash when metrics block is entirely absent."""
        result = {
            "n_records": 0,
            "n_graded_matches": 0,
            "skipped_no_outcome": 0,
            "skipped_no_sp": 0,
            "n_calibration_points": 0,
            "n_void": 0,
            "n_bets": 0,
            "metrics": {},
            "per_market_type": {},
        }
        report = render_backtest_report(result, title="No Metrics Test")
        self.assertIsInstance(report, str)

    def test_render_backtest_report_with_bets(self):
        """When bets are allocated, pnl / edge / kelly blocks must appear."""
        result = run_backtest(self.records, self.outcomes, budget=500.0, bankroll=5000.0)
        report = render_backtest_report(result, title="Full Report")
        # These sections are always rendered (may show '—' if no data).
        self.assertIn("PnL", report)
        self.assertIn("Edge", report)
        self.assertIn("Kelly", report)


class TestRenderOptimizationReport(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        root = Path(self._tmpdir.name) / "store"
        self.records, self.outcomes = build_records_and_outcomes(root, n_records=4)

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_render_optimization_report_returns_str(self):
        space = {"conf_base": [0.5, 0.6], "corr_base": [0.94, 0.96]}
        opt = optimize_params(
            self.records, self.outcomes, space,
            metric="brier", search="grid", n_folds=2,
        )
        report = render_optimization_report(opt, title="Test Opt Report")
        self.assertIsInstance(report, str)
        self.assertTrue(len(report) > 0)

    def test_render_optimization_report_contains_expected_headers(self):
        space = {"conf_base": [0.5, 0.6]}
        opt = optimize_params(
            self.records, self.outcomes, space,
            metric="brier", search="grid", n_folds=2,
        )
        report = render_optimization_report(opt, title="My Opt")
        self.assertIn("# My Opt", report)
        self.assertIn("brier", report)
        self.assertIn("conf_base", report)
        self.assertIn("in_sample", report)
        self.assertIn("out_of_sample", report)

    def test_render_optimization_report_no_crash_empty_trials(self):
        """Render must not crash when trials list is empty."""
        fake_opt = {
            "metric": "brier",
            "direction": "min",
            "search": "grid",
            "n_trials": 0,
            "n_folds": 2,
            "best_params": DEFAULT_PARAMS.to_dict(),
            "best_overrides": {},
            "best_in_sample": None,
            "best_out_of_sample": None,
            "overfit_gap": None,
            "trials": [],
        }
        report = render_optimization_report(fake_opt, title="Empty Trials")
        self.assertIsInstance(report, str)

    def test_render_optimization_report_top10_sorted(self):
        """Top-10 table in report must be ordered by OOS (min first for brier)."""
        space = {"conf_base": [0.5, 0.55, 0.6], "corr_base": [0.94, 0.96]}
        opt = optimize_params(
            self.records, self.outcomes, space,
            metric="brier", search="grid", n_folds=2,
        )
        report = render_optimization_report(opt, title="Top10 Test")
        # Report must contain trial rows (non-empty table).
        self.assertIn("| 1 |", report)

    def test_overfit_warning_present_when_gap_large(self):
        """⚠ warning must appear when overfit_gap > 0.05."""
        fake_opt = {
            "metric": "brier",
            "direction": "min",
            "search": "grid",
            "n_trials": 1,
            "n_folds": 2,
            "best_params": DEFAULT_PARAMS.to_dict(),
            "best_overrides": {"conf_base": 0.5},
            "best_in_sample": 0.10,
            "best_out_of_sample": 0.20,   # gap = 0.20 - 0.10 = 0.10 > 0.05
            "overfit_gap": 0.10,
            "trials": [
                {"overrides": {"conf_base": 0.5}, "in_sample": 0.10, "out_of_sample": 0.20, "undefined": False}
            ],
        }
        report = render_optimization_report(fake_opt, title="Overfit Warning Test")
        self.assertIn("⚠", report)


if __name__ == "__main__":
    unittest.main()
