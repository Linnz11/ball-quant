"""
Tests for the backtest spine: replay, splits, engine.

The end-to-end seam test exercises:
  capture_snapshot -> read_snapshot -> run_backtest -> metrics
proving that capture, reconstruct, analyze, settlement_key, grade, and metrics
all compose correctly from a single test fixture.
"""
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

import pytest

from tests.test_probability_and_combo import sample_match, sample_matrix

from ball_quant.backtest.engine import grade_combo, run_backtest
from ball_quant.backtest.splits import assert_no_lookahead, rolling_splits, walk_forward_splits
from ball_quant.core.settlement import MatchOutcome
from ball_quant.data.capture import capture_snapshot
from ball_quant.data.store import read_snapshot
from ball_quant.models import EventMarketMatrix, MarketQuote, MatchSP, Selection, SettlementKey, TeamFacts


# ---------------------------------------------------------------------------
# Helpers shared across tests
# ---------------------------------------------------------------------------

def _make_sel(match_id: str, sp: float, prob: float, market_type: str = "spf", side: str = "home") -> Selection:
    """Minimal Selection with a valid settlement_key for grade_combo tests."""
    edge = prob * sp - 1.0
    return Selection(
        match_id=match_id,
        home="A",
        away="B",
        play=market_type,
        outcome=side,
        condition=f"{side} wins",
        probability=prob,
        sp=sp,
        fair_odds=1.0 / prob,
        break_even=1.0 / sp,
        edge=edge,
        kelly=max(0.0, (prob * (sp - 1) - (1 - prob)) / (sp - 1)),
        confidence=0.70,
        risk_label="价值保留",
        settlement_key=SettlementKey(market_type=market_type, side=side),
    )


# ---------------------------------------------------------------------------
# splits tests
# ---------------------------------------------------------------------------

class TestSplits(unittest.TestCase):
    def _items(self):
        # Out-of-order captured_at strings — split must sort them.
        return [
            {"id": i, "captured_at": f"2026-01-{i:02d}T12:00:00Z"}
            for i in [5, 3, 1, 4, 2, 6, 7, 8, 9]
        ]

    def _key(self, item):
        return item["captured_at"]

    def test_walk_forward_splits_time_ordered(self):
        items = self._items()
        folds = walk_forward_splits(items, self._key, n_folds=2)
        self.assertEqual(len(folds), 2)
        for train, test in folds:
            # Time ordering must hold: every test item is >= all train items.
            assert_no_lookahead(train, test, self._key)
            self.assertGreater(len(train), 0)
            self.assertGreater(len(test), 0)

    def test_walk_forward_splits_expanding_train(self):
        items = self._items()
        folds = walk_forward_splits(items, self._key, n_folds=2)
        # Fold 1 train must be smaller than fold 2 train (expanding window).
        self.assertLess(len(folds[0][0]), len(folds[1][0]))

    def test_assert_no_lookahead_passes_on_correct_split(self):
        items = self._items()
        folds = walk_forward_splits(items, self._key, n_folds=2)
        for train, test in folds:
            # Should not raise.
            assert_no_lookahead(train, test, self._key)

    def test_assert_no_lookahead_raises_on_leaky_split(self):
        items = self._items()
        sorted_items = sorted(items, key=self._key)
        # Deliberately put a future item into train and a past item into test.
        train = sorted_items[3:]   # later items
        test = sorted_items[:3]    # earlier items — this is a lookahead
        with self.assertRaises(AssertionError):
            assert_no_lookahead(train, test, self._key)

    def test_rolling_splits_sizes(self):
        items = [{"id": i, "captured_at": f"2026-01-{i:02d}T00:00:00Z"} for i in range(1, 11)]
        folds = rolling_splits(items, self._key, train_size=5, test_size=2, step=2)
        for train, test in folds:
            self.assertEqual(len(train), 5)
            self.assertEqual(len(test), 2)
        # With 10 items, train_size=5, test_size=2, step=2:
        # fold 0: train=[1-5], test=[6-7]; fold 1: train=[3-7], test=[8-9]; fold 2: would need [10,11] -> stops.
        self.assertEqual(len(folds), 2)

    def test_rolling_splits_no_lookahead(self):
        items = [{"id": i, "captured_at": f"2026-01-{i:02d}T00:00:00Z"} for i in range(1, 15)]
        folds = rolling_splits(items, self._key, train_size=5, test_size=3, step=2)
        for train, test in folds:
            assert_no_lookahead(train, test, self._key)


# ---------------------------------------------------------------------------
# grade_combo tests
# ---------------------------------------------------------------------------

class TestGradeCombo(unittest.TestCase):
    def _make_outcome(self, home_score=2, away_score=0):
        return MatchOutcome(match_id="001", home_score=home_score, away_score=away_score)

    def test_all_win_legs_gives_win_with_product_odds(self):
        # home wins (2-0), both legs are spf:home -> both WIN
        sel1 = _make_sel("001", sp=1.8, prob=0.62, market_type="spf", side="home")
        sel2 = _make_sel("001", sp=2.1, prob=0.55, market_type="spf", side="home")

        class FakeCombo:
            selections = [sel1, sel2]
            odds = 1.8 * 2.1

        outcome = self._make_outcome(2, 0)
        result, eff_odds = grade_combo(FakeCombo(), outcome)
        self.assertEqual(result, "WIN")
        self.assertAlmostEqual(eff_odds, 1.8 * 2.1, places=6)

    def test_one_loss_leg_gives_loss(self):
        # home wins (2-0)
        # sel1: spf:home -> WIN
        # sel2: spf:away -> LOSS
        sel1 = _make_sel("001", sp=1.8, prob=0.62, market_type="spf", side="home")
        sel2 = _make_sel("001", sp=5.0, prob=0.15, market_type="spf", side="away")

        class FakeCombo:
            selections = [sel1, sel2]
            odds = 1.8 * 5.0

        outcome = self._make_outcome(2, 0)
        result, eff_odds = grade_combo(FakeCombo(), outcome)
        self.assertEqual(result, "LOSS")

    def test_void_leg_dropped_odds_recomputed(self):
        # Match 2-2: totals over 2 -> VOID (integer push), spf:home -> LOSS
        # But let's use btts:yes -> WIN and totals:over:2 -> VOID.
        # Score 1-1: home < away is False, btts -> both scored (yes) -> WIN.
        # totals over 1 (line=1.0) with score=2 total -> WIN (2>1) but let's force VOID:
        # Use score 2-0, totals over/under line=2.0 (push) -> VOID; spf:home -> WIN.
        sel_win = _make_sel("001", sp=1.8, prob=0.62, market_type="spf", side="home")
        sel_void = Selection(
            match_id="001",
            home="A", away="B",
            play="totals", outcome="over",
            condition="over 2",
            probability=0.55, sp=1.9, fair_odds=1.82, break_even=0.53,
            edge=0.045, kelly=0.05, confidence=0.65, risk_label="观察",
            settlement_key=SettlementKey(market_type="totals", side="over", line=2.0),
        )

        class FakeCombo:
            selections = [sel_win, sel_void]
            odds = 1.8 * 1.9

        # Score 2-0: total goals = 2, line = 2.0 -> push -> VOID; spf:home -> WIN.
        outcome = self._make_outcome(2, 0)
        result, eff_odds = grade_combo(FakeCombo(), outcome)
        # The VOID leg is dropped; remaining WIN leg -> combo is WIN.
        self.assertEqual(result, "WIN")
        # effective_odds = product of surviving WIN leg SPs = sel_win.sp = 1.8
        self.assertAlmostEqual(eff_odds, 1.8, places=6)


# ---------------------------------------------------------------------------
# END-TO-END seam test: capture -> store -> run_backtest -> metrics
# ---------------------------------------------------------------------------

class TestEndToEndBacktest(unittest.TestCase):
    def test_capture_to_backtest_produces_graded_metrics(self):
        match = sample_match()   # match_id="001", Netherlands vs Japan
        matrix = sample_matrix()

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "store"

            # 1. Capture snapshot to the tmp store.
            snap_path = capture_snapshot(
                matrix=matrix,
                match_sp=match,
                root=root,
                captured_at=datetime(2026, 6, 14, 10, 0, 0, tzinfo=timezone.utc),
                competition="test",
            )

            # 2. Read back the record.
            record = read_snapshot(snap_path)
            self.assertEqual(record["match_id"], "001")

            # 3. Provide the actual outcome (Netherlands 2-0 Japan: home wins).
            outcomes = {
                "001": MatchOutcome(
                    match_id="001",
                    home_score=2,
                    away_score=0,
                )
            }

            # 4. Run backtest over the single record.
            result = run_backtest(
                records=[record],
                outcomes=outcomes,
                budget=100.0,
                bankroll=1000.0,
            )

            # ---- structural assertions ----
            self.assertEqual(result["n_records"], 1)
            self.assertEqual(result["n_graded_matches"], 1)
            self.assertEqual(result["skipped_no_outcome"], 0)

            # At least one calibration point must have been graded (not all VOID).
            self.assertGreater(result["n_calibration_points"], 0,
                               "Expected at least one graded calibration point")

            # Brier score must be a float in [0, 1].
            metrics = result["metrics"]
            self.assertIn("calibration", metrics)
            calib = metrics["calibration"]
            self.assertIn("brier", calib)
            brier = calib["brier"]
            self.assertIsInstance(brier, float)
            self.assertGreaterEqual(brier, 0.0)
            self.assertLessEqual(brier, 1.0)

            # PnL block must be present (may be empty dict if no non-VOID bets, but key exists).
            self.assertIn("pnl", metrics)

            # Per-market-type breakdown must contain at least one group.
            self.assertGreater(
                len(result["per_market_type"]), 0,
                "Expected at least one market type in the breakdown"
            )

    def test_missing_outcome_increments_skipped(self):
        """A record with no matching outcome must be skipped, not crashed."""
        match = sample_match()
        matrix = sample_matrix()

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "store"
            snap_path = capture_snapshot(
                matrix=matrix,
                match_sp=match,
                root=root,
                captured_at=datetime(2026, 6, 14, 11, 0, 0, tzinfo=timezone.utc),
            )
            record = read_snapshot(snap_path)

            # Pass empty outcomes so the match has no outcome.
            result = run_backtest(records=[record], outcomes={})

            self.assertEqual(result["n_records"], 1)
            self.assertEqual(result["n_graded_matches"], 0)
            self.assertEqual(result["skipped_no_outcome"], 1)


# ---------------------------------------------------------------------------
# Money-path test: verify run_backtest allocates real bets and produces
# hand-verifiable PnL when the fixture has a clear positive-edge value gap.
# ---------------------------------------------------------------------------

class TestAllocatedBetsMoneyPath(unittest.TestCase):
    """
    The existing e2e test uses sample_match() whose SPs are tight vs. model
    probs, so allocate_stakes returns zero allocated bets and the
    engine's bet-building / pnl_ledger code path never runs.

    This test engineers a deliberate VALUE GAP:
      - Matrix moneyline: home=0.65, draw=0.20, away=0.15
        (model probability for home win is 0.65)
      - MatchSP.spf_home = 2.80  =>  implied_prob = 1/2.80 ≈ 0.357
        =>  edge = 0.65 × 2.80 − 1 = +0.82  (large positive edge)
      - MatchSP.spf_away = 9.00  =>  implied_prob = 1/9 ≈ 0.111
        =>  edge = 0.15 × 9.00 − 1 = +0.35  (positive edge)

    Outcome: home wins 2-0, so spf:home is a WIN and spf:away is a LOSS.

    Hand-computed expected PnL (verified by sanity run before writing this test):
      spf:home: stake=56, odds=2.80 → net = 56 × (2.80−1) = +100.80
      spf:away: stake=4,  odds=9.00 → net = −4.00
      net_pnl = 100.80 − 4.00 = 96.80
      total_stake_at_risk = 56 + 4 = 60.00
      roi = 96.80 / 60.00 ≈ 1.6133

    These numbers come from the real allocate_stakes call path — not injected.
    """

    def _value_gap_match(self) -> MatchSP:
        # SPs are generous vs. model probs; handicap=0 means rq unused here.
        return MatchSP(
            match_id="vg001",
            date="2026-06-14",
            home="Team A",
            away="Team B",
            spf_home=2.80,   # edge vs 0.65 model prob = +0.82
            spf_draw=4.50,   # edge vs 0.20 = −0.10 (negative, filtered out by Kelly=0)
            spf_away=9.00,   # edge vs 0.15 = +0.35
            handicap=0,
            rq_home=0.0,
            rq_draw=0.0,
            rq_away=0.0,
        )

    def _value_gap_matrix(self) -> EventMarketMatrix:
        return EventMarketMatrix(
            match_id="vg001",
            home="Team A",
            away="Team B",
            markets=[
                # Tight spread + high liquidity → high confidence score → combos pass filters.
                MarketQuote("m1", "winner", "moneyline", "home", 0.65, spread=0.02, liquidity=15000),
                MarketQuote("m1", "winner", "moneyline", "draw", 0.20, spread=0.02, liquidity=15000),
                MarketQuote("m1", "winner", "moneyline", "away", 0.15, spread=0.02, liquidity=15000),
            ],
        )

    def test_run_backtest_allocates_bets_and_populates_pnl(self):
        match = self._value_gap_match()
        matrix = self._value_gap_matrix()

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "store"

            snap_path = capture_snapshot(
                matrix=matrix,
                match_sp=match,
                root=root,
                captured_at=datetime(2026, 6, 14, 12, 0, 0, tzinfo=timezone.utc),
                competition="test_value_gap",
            )
            record = read_snapshot(snap_path)

            # Home wins 2-0 → spf:home = WIN, spf:away = LOSS.
            outcomes = {
                "vg001": MatchOutcome(
                    match_id="vg001",
                    home_score=2,
                    away_score=0,
                )
            }

            # Use a large budget so type caps don't squeeze stakes to zero.
            # Budget=500 means cap_a=0.35 → type_cap=175, kelly_cap≈57 is binding.
            result = run_backtest(
                records=[record],
                outcomes=outcomes,
                budget=500.0,
                bankroll=5000.0,
            )

            # --- core invariant: bets were ACTUALLY allocated --------------------
            # This is the whole point: the money-path that was previously untested.
            self.assertGreaterEqual(
                result["n_bets"], 1,
                "Expected at least 1 allocated bet; value gap should force allocation",
            )

            metrics = result["metrics"]

            # --- pnl block is non-empty and has the right keys ------------------
            pnl = metrics.get("pnl", {})
            self.assertIn("n_bets",             pnl, "pnl block must contain n_bets")
            self.assertIn("net_pnl",            pnl, "pnl block must contain net_pnl")
            self.assertIn("roi",                pnl, "pnl block must contain roi")
            self.assertIn("total_stake_at_risk", pnl, "pnl block must contain total_stake_at_risk")

            # --- hand-computed net_pnl (see class docstring) --------------------
            # spf:home: stake=56, WIN  => net = 56*(2.80−1) = +100.80
            # spf:away: stake=4,  LOSS => net = −4.00
            # net_pnl = 96.80   total_stake_at_risk = 60.00
            expected_net_pnl = 56.0 * (2.80 - 1.0) + (-4.0)   # = 96.80
            expected_stake_at_risk = 56.0 + 4.0                 # = 60.00

            self.assertAlmostEqual(
                pnl["net_pnl"], expected_net_pnl,
                places=6,
                msg=f"net_pnl {pnl['net_pnl']!r} != hand-computed {expected_net_pnl}",
            )
            self.assertAlmostEqual(
                pnl["total_stake_at_risk"], expected_stake_at_risk,
                places=6,
                msg=f"total_stake_at_risk {pnl['total_stake_at_risk']!r} != {expected_stake_at_risk}",
            )

            # --- edge and kelly blocks prove aggregate ran the full money path --
            edge_block = metrics.get("edge", {})
            self.assertTrue(
                bool(edge_block),
                "metrics['edge'] must be non-empty; proving edge_realization ran",
            )

            kelly_block = metrics.get("kelly", {})
            self.assertTrue(
                bool(kelly_block),
                "metrics['kelly'] must be non-empty; proving kelly_growth ran",
            )


if __name__ == "__main__":
    unittest.main()
