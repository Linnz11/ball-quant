"""End-to-end test: devig_method toggle must reach the real pipeline.

Before the fix, build_probability_context stored params but ProbabilityContext
did not, so probability_for_spf / moneyline_margin_* always used the
DEFAULT_PARAMS (proportional) even when shin was requested.  This file proves
the plumbing is now connected.
"""
from __future__ import annotations

import dataclasses
import unittest

from ball_quant.core.params import DEFAULT_PARAMS, StrategyParams
from ball_quant.core.probability import (
    build_probability_context,
    match_branches,
)
from ball_quant.models import EventMarketMatrix, MarketQuote, MatchSP


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

def _vig_matrix() -> EventMarketMatrix:
    """Moneyline with booksum > 1 (implied 0.50+0.30+0.32 = 1.12).

    The vig is large enough that Shin and proportional devig give
    meaningfully different fair probabilities for home/draw/away.
    """
    return EventMarketMatrix(
        match_id="e2e",
        home="Home",
        away="Away",
        markets=[
            MarketQuote("q1", "ml", "moneyline", "home", 0.50, spread=0.02, liquidity=10000),
            MarketQuote("q2", "ml", "moneyline", "draw", 0.30, spread=0.02, liquidity=10000),
            MarketQuote("q3", "ml", "moneyline", "away", 0.32, spread=0.02, liquidity=10000),
        ],
    )


def _match() -> MatchSP:
    return MatchSP(
        match_id="e2e",
        date="2026-06-14",
        home="Home",
        away="Away",
        spf_home=2.0,
        spf_draw=3.3,
        spf_away=3.1,
        handicap=0,
        rq_home=2.0,
        rq_draw=3.3,
        rq_away=3.1,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class DevigE2ETest(unittest.TestCase):
    """Prove devig_method="shin" now produces different SPF probabilities
    than the default "proportional" path when going through the full pipeline:
    build_probability_context → match_branches → probability_for_spf.
    """

    def setUp(self):
        self.match = _match()
        self.matrix = _vig_matrix()
        self.shin_params = dataclasses.replace(DEFAULT_PARAMS, devig_method="shin")

        # Build contexts under both methods.
        self.ctx_prop = build_probability_context(self.match, self.matrix)  # default: proportional
        self.ctx_shin = build_probability_context(self.match, self.matrix, params=self.shin_params)

        # Extract branches.
        self.branches_prop = {b.outcome: b for b in match_branches(self.match, self.ctx_prop)
                               if b.play == "spf"}
        self.branches_shin = {b.outcome: b for b in match_branches(self.match, self.ctx_shin)
                               if b.play == "spf"}

    # --- The headline assertion: shin and proportional must diverge -------

    def test_spf_home_prob_differs_between_shin_and_proportional(self):
        """This would have caught the original bug (identical Brier scores)."""
        p_prop = self.branches_prop["home"].probability
        p_shin = self.branches_shin["home"].probability
        self.assertIsNotNone(p_prop)
        self.assertIsNotNone(p_shin)
        self.assertNotAlmostEqual(
            p_prop, p_shin, places=6,
            msg=(
                f"SPF home prob is identical under proportional ({p_prop:.8f}) "
                f"and shin ({p_shin:.8f}); devig_method toggle is still dead end-to-end"
            ),
        )

    def test_spf_draw_prob_differs_between_shin_and_proportional(self):
        p_prop = self.branches_prop["draw"].probability
        p_shin = self.branches_shin["draw"].probability
        self.assertNotAlmostEqual(p_prop, p_shin, places=6)

    def test_spf_away_prob_differs_between_shin_and_proportional(self):
        p_prop = self.branches_prop["away"].probability
        p_shin = self.branches_shin["away"].probability
        self.assertNotAlmostEqual(p_prop, p_shin, places=6)

    # --- Proportional path must remain byte-identical to pre-fix ----------

    def test_proportional_spf_home_matches_expected_value(self):
        """Proportional devig: fair prob = raw / booksum.
        raw = [0.50, 0.30, 0.32], booksum = 1.12.
        market_probability_value clamps to [0.005, 0.995] and then divides.
        With spread=0.02 the probability stored is 0.50/0.30/0.32 directly,
        so expected home = 0.50/1.12 (subject to market_probability_value floor).
        """
        booksum = 0.50 + 0.30 + 0.32
        expected_home = 0.50 / booksum
        p_prop = self.branches_prop["home"].probability
        self.assertAlmostEqual(p_prop, expected_home, places=6,
                               msg="Default proportional path changed — regression")

    def test_proportional_spf_probs_sum_to_one(self):
        total = sum(b.probability for b in self.branches_prop.values())
        self.assertAlmostEqual(total, 1.0, places=6)

    def test_shin_spf_probs_sum_to_one(self):
        total = sum(b.probability for b in self.branches_shin.values())
        self.assertAlmostEqual(total, 1.0, places=6)

    # --- ProbabilityContext carries params --------------------------------

    def test_context_stores_params(self):
        """ProbabilityContext.params must equal what was passed in."""
        self.assertEqual(self.ctx_prop.params.devig_method, "proportional")
        self.assertEqual(self.ctx_shin.params.devig_method, "shin")

    def test_default_context_params_is_default_params(self):
        """Constructing context without explicit params must default correctly."""
        self.assertIs(self.ctx_prop.params, DEFAULT_PARAMS)

    # --- Shin is directionally correct (longshot bias correction) ----------

    def test_shin_lifts_favourite_relative_to_proportional(self):
        """Shin (1992): informed bettors concentrate on longshots, so the book
        over-prices longshots (shrinks their fair prob) and under-prices the
        favourite — Shin corrects for this, lifting the favourite.
        home has raw 0.50 / booksum 1.12, making it the favourite; Shin should
        assign it *more* probability than proportional scaling.
        """
        p_prop = self.branches_prop["home"].probability
        p_shin = self.branches_shin["home"].probability
        self.assertGreater(
            p_shin, p_prop,
            msg=(
                f"Shin should lift the favourite vs proportional; "
                f"got shin={p_shin:.6f}, prop={p_prop:.6f}"
            ),
        )


if __name__ == "__main__":
    unittest.main()
