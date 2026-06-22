"""Regression tests against the real captured Polymarket snapshot for match 351726
(Sweden vs Tunisia, 24 market categories, 590 quotes).

The snapshot was captured ~90 min post-match (endDate 2026-06-15T02:00Z, captured
2026-06-15T03:38Z), so near-settled tokens show Polymarket floor-price asks (0.001).
These were previously priced as 1/0.001 = 1000-odds, producing phantom edges up to 5.0.

The fix (core/value.py _MIN_VIABLE_PRICE = 0.02) skips legs where both ask AND
probability are below 2%, ensuring no selection can carry sp >= 50.
"""

from __future__ import annotations

import glob
import math
import unittest
from pathlib import Path

from ball_quant.core.analysis import analyze_match
from ball_quant.backtest.replay import neutral_facts
from ball_quant.core.value import _MIN_VIABLE_PRICE, _price_from_matrix
from ball_quant.data.store import read_snapshot, reconstruct_matrix
from ball_quant.models import Branch, EventMarketMatrix, MarketQuote, MatchSP

# ---------------------------------------------------------------------------
# Helper: build a synthetic zero-SP MatchSP for a pure-Polymarket snapshot
# (SP block is None when China Sports Lottery SP was not present at capture time)
# ---------------------------------------------------------------------------

_SNAPSHOT_GLOB = "data/store/snapshots/351726__*.json"


def _load_snapshot():
    files = glob.glob(_SNAPSHOT_GLOB)
    if not files:
        return None, None
    record = read_snapshot(Path(files[0]))
    matrix = reconstruct_matrix(record)
    match = MatchSP(
        match_id=record["match_id"],
        date=record["captured_at"][:10],
        home=record["home"],
        away=record["away"],
        spf_home=0.0,
        spf_draw=0.0,
        spf_away=0.0,
        handicap=0,
        rq_home=0.0,
        rq_draw=0.0,
        rq_away=0.0,
    )
    return match, matrix


# ---------------------------------------------------------------------------
# 1. Real-snapshot pricing regression tests
# ---------------------------------------------------------------------------

class TestRealSnapshotPricing(unittest.TestCase):
    """Run the full engine against the captured 351726 snapshot."""

    @classmethod
    def setUpClass(cls):
        match, matrix = _load_snapshot()
        if match is None:
            cls._skip = True
            return
        cls._skip = False
        facts = neutral_facts(match)
        cls.analysis = analyze_match(match, matrix, facts)
        cls.selections = cls.analysis.selections

    def _check_available(self):
        if self._skip:
            self.skipTest(f"Snapshot not found at {_SNAPSHOT_GLOB}")

    def test_no_phantom_sp_ge_100(self):
        """No selection may have sp >= 100 (covers the 1000-odds phantom bug)."""
        self._check_available()
        phantoms = [s for s in self.selections if s.sp >= 100]
        self.assertEqual(
            phantoms, [],
            msg=f"Phantom sp>=100 selections: "
                + ", ".join(f"{s.play}:{s.outcome} sp={s.sp:.1f}" for s in phantoms),
        )

    def test_all_sp_finite_and_above_one(self):
        """Every emitted selection must have finite sp > 1.0."""
        self._check_available()
        bad = [s for s in self.selections if not math.isfinite(s.sp) or s.sp <= 1.0]
        self.assertEqual(
            bad, [],
            msg="Selections with sp not finite or <= 1: "
                + ", ".join(f"{s.play}:{s.outcome} sp={s.sp}" for s in bad),
        )

    def test_high_probability_outcome_has_no_absurd_edge(self):
        """A leg where p_model > 0.90 must not show edge > 1.0.

        Before the fix, near-certain outcomes (over 0.5 etc.) sometimes had
        phantom large edges because the opposite side's ask was 0.001 and the
        over side landed the phantom price indirectly.
        """
        self._check_available()
        absurd = [
            s for s in self.selections if s.probability > 0.90 and s.edge > 1.0
        ]
        self.assertEqual(
            absurd, [],
            msg="High-prob selections with edge>1.0: "
                + ", ".join(
                    f"{s.play}:{s.outcome} p={s.probability:.3f} edge={s.edge:.3f}"
                    for s in absurd
                ),
        )

    def test_under_2_5_sweden_no_phantom(self):
        """team_total(Sweden,2.5) under must either be absent or carry a real price.

        Before the fix this produced sp=1000 via ask=0.001 with p_model=0.006,
        yielding edge=5.14.
        """
        self._check_available()
        culprits = [
            s
            for s in self.selections
            if s.play == "team_total(Sweden,2.5)" and s.outcome == "under"
        ]
        for s in culprits:
            self.assertLess(
                s.sp,
                100,
                f"team_total(Sweden,2.5) under has phantom sp={s.sp:.1f}",
            )
            self.assertLess(
                s.edge,
                1.0,
                f"team_total(Sweden,2.5) under has phantom edge={s.edge:.2f}",
            )

    def test_selection_count_sane(self):
        """After filtering phantoms, we should have fewer selections than before the fix
        (post-match near-settled market has few tradeable legs).  The count must be
        in [0, 50] — a loose sanity bound.
        """
        self._check_available()
        n = len(self.selections)
        self.assertLessEqual(n, 50, f"Suspiciously many selections: {n}")


# ---------------------------------------------------------------------------
# 2. Synthetic outcome-alignment tests (unit-level, no snapshot required)
# ---------------------------------------------------------------------------

def _make_branch(play: str, outcome: str) -> Branch:
    return Branch(
        match_id="test",
        play=play,
        outcome=outcome,
        condition="",
        probability=0.5,
        source="test",
    )


def _make_matrix_with_team_total(over_ask, under_ask, over_prob, under_prob):
    """Synthetic team_total matrix for Sweden over/under 2.5."""
    markets = []
    if over_prob is not None:
        markets.append(
            MarketQuote(
                market_id="q-over",
                question="Sweden goals",
                category="team_total",
                outcome="Sweden over 2.5",
                probability=over_prob,
                ask=over_ask,
                bid=None,
                line=2.5,
                entity="Sweden",
            )
        )
    if under_prob is not None:
        markets.append(
            MarketQuote(
                market_id="q-under",
                question="Sweden goals",
                category="team_total",
                outcome="Sweden under 2.5",
                probability=under_prob,
                ask=under_ask,
                bid=None,
                line=2.5,
                entity="Sweden",
            )
        )
    return EventMarketMatrix(
        match_id="test",
        home="Sweden",
        away="Tunisia",
        markets=markets,
    )


class TestOutcomeAlignment(unittest.TestCase):
    """Verify that the 'under' branch is priced from the 'under' quote, not the 'over' quote,
    and that floor-price asks are skipped, not used."""

    def test_under_branch_uses_under_quote_ask(self):
        """under branch with viable under-ask (0.25) must price off that ask."""
        branch = _make_branch("team_total(Sweden,2.5)", "under")
        matrix = _make_matrix_with_team_total(
            over_ask=0.75, under_ask=0.25, over_prob=0.75, under_prob=0.25
        )
        price = _price_from_matrix(branch, matrix)
        self.assertIsNotNone(price)
        self.assertAlmostEqual(price, 1.0 / 0.25, places=5,
                               msg="under branch must use the under-quote ask")

    def test_over_branch_uses_over_quote_ask(self):
        """over branch with viable over-ask (0.75) must price off that ask."""
        branch = _make_branch("team_total(Sweden,2.5)", "over")
        matrix = _make_matrix_with_team_total(
            over_ask=0.75, under_ask=0.25, over_prob=0.75, under_prob=0.25
        )
        price = _price_from_matrix(branch, matrix)
        self.assertIsNotNone(price)
        self.assertAlmostEqual(price, 1.0 / 0.75, places=5,
                               msg="over branch must use the over-quote ask")

    def test_floor_ask_skipped_returns_none(self):
        """Under branch with only a floor-tick ask (0.001) must return None (no phantom)."""
        branch = _make_branch("team_total(Sweden,2.5)", "under")
        matrix = _make_matrix_with_team_total(
            over_ask=None, under_ask=0.001, over_prob=0.999, under_prob=0.001
        )
        price = _price_from_matrix(branch, matrix)
        self.assertIsNone(
            price,
            f"Floor-tick ask=0.001 must be skipped, not priced as sp=1000; got {price}",
        )

    def test_floor_probability_fallback_also_skipped(self):
        """Under branch where ask=None and probability=0.001 must also return None."""
        branch = _make_branch("team_total(Sweden,2.5)", "under")
        matrix = _make_matrix_with_team_total(
            over_ask=None, under_ask=None, over_prob=0.999, under_prob=0.001
        )
        price = _price_from_matrix(branch, matrix)
        self.assertIsNone(
            price,
            f"Floor-level probability=0.001 fallback must be skipped; got {price}",
        )

    def test_viable_probability_fallback_when_ask_none(self):
        """Under branch where ask=None but probability=0.25 must use 1/probability."""
        branch = _make_branch("team_total(Sweden,2.5)", "under")
        matrix = _make_matrix_with_team_total(
            over_ask=None, under_ask=None, over_prob=0.75, under_prob=0.25
        )
        price = _price_from_matrix(branch, matrix)
        self.assertIsNotNone(price)
        self.assertAlmostEqual(price, 1.0 / 0.25, places=5,
                               msg="Viable probability fallback must yield 1/prob")

    def test_min_viable_price_constant(self):
        """_MIN_VIABLE_PRICE must be exactly 0.02 (the documented floor)."""
        self.assertEqual(_MIN_VIABLE_PRICE, 0.02)

    def test_totals_under_floor_ask_skipped(self):
        """totals(2.5) under with ask=0.001 must return None."""
        branch = _make_branch("totals(2.5)", "under")
        matrix = EventMarketMatrix(
            match_id="test",
            home="A",
            away="B",
            markets=[
                MarketQuote(
                    market_id="q1",
                    question="total goals",
                    category="total_goals",
                    outcome="over 2.5",
                    probability=0.999,
                    ask=None,
                    bid=0.998,
                    line=2.5,
                ),
                MarketQuote(
                    market_id="q2",
                    question="total goals",
                    category="total_goals",
                    outcome="under 2.5",
                    probability=0.001,
                    ask=0.001,
                    bid=None,
                    line=2.5,
                ),
            ],
        )
        price = _price_from_matrix(branch, matrix)
        self.assertIsNone(price, f"Floor-tick totals under must be skipped; got {price}")

    def test_btts_no_floor_ask_skipped(self):
        """btts no with ask=0.001 must return None."""
        branch = _make_branch("btts", "no")
        matrix = EventMarketMatrix(
            match_id="test",
            home="A",
            away="B",
            markets=[
                MarketQuote(
                    market_id="q1",
                    question="BTTS",
                    category="btts",
                    outcome="yes",
                    probability=0.999,
                    ask=None,
                    bid=0.998,
                ),
                MarketQuote(
                    market_id="q2",
                    question="BTTS",
                    category="btts",
                    outcome="no",
                    probability=0.001,
                    ask=0.001,
                    bid=None,
                ),
            ],
        )
        price = _price_from_matrix(branch, matrix)
        self.assertIsNone(price, f"Floor-tick btts no must be skipped; got {price}")

    def test_correct_score_floor_ask_skipped(self):
        """correct_score 0-0 with ask=0.001 must return None."""
        branch = _make_branch("correct_score", "0-0")
        matrix = EventMarketMatrix(
            match_id="test",
            home="A",
            away="B",
            markets=[
                MarketQuote(
                    market_id="q1",
                    question="score",
                    category="correct_score",
                    outcome="0-0",
                    probability=0.001,
                    ask=0.001,
                    bid=None,
                ),
            ],
        )
        price = _price_from_matrix(branch, matrix)
        self.assertIsNone(price, f"Floor-tick correct_score must be skipped; got {price}")


if __name__ == "__main__":
    unittest.main()
