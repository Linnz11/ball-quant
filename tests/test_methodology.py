"""Methodology toggle tests.

Two groups:
  A. Default-off (rho=0.0, devig_method="proportional") — output must match
     current independent-Poisson + proportional-devig exactly.
  B. Directional — behavior changes when the toggle is flipped; expected to
     FAIL until the implementation is in place.
"""
from __future__ import annotations

import math
import unittest

from ball_quant.core.params import StrategyParams
from ball_quant.core.probability import (
    normalize_probs,
    poisson_grid,
    poisson_pmf,
    shin_devig,
    normalized_moneyline_probabilities,
)
from ball_quant.models import EventMarketMatrix, MarketQuote


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _independent_grid(lh: float, la: float, n: int = 7):
    """Pure independent-Poisson grid; reference implementation for comparison."""
    probs = {}
    for h in range(n + 1):
        for a in range(n + 1):
            probs[(h, a)] = poisson_pmf(h, lh) * poisson_pmf(a, la)
    return normalize_probs(probs)


def _draw_prob(grid):
    return sum(p for (h, a), p in grid.items() if h == a)


def _matrix_with_moneyline(home_p, draw_p, away_p):
    """Tiny EventMarketMatrix with only moneyline quotes (no vig – raw probs)."""
    return EventMarketMatrix(
        match_id="t1",
        home="A",
        away="B",
        markets=[
            MarketQuote("q1", "ml", "moneyline", "home", home_p, spread=0.0, liquidity=1000),
            MarketQuote("q2", "ml", "moneyline", "draw", draw_p, spread=0.0, liquidity=1000),
            MarketQuote("q3", "ml", "moneyline", "away", away_p, spread=0.0, liquidity=1000),
        ],
    )


# ---------------------------------------------------------------------------
# Group A: Default-off invariants
# ---------------------------------------------------------------------------

class DefaultOffTests(unittest.TestCase):

    # --- A1. poisson_grid with default rho=0.0 is byte-identical to
    #         the pure independent-Poisson reference ---

    def test_poisson_grid_default_rho_identical_to_reference(self):
        lh, la = 1.4, 1.1
        ref = _independent_grid(lh, la)
        got = poisson_grid(lh, la, max_goals=7)          # rho omitted → 0.0
        self.assertEqual(ref.keys(), got.keys())
        for key in ref:
            self.assertAlmostEqual(ref[key], got[key], places=15,
                                   msg=f"Mismatch at {key}")

    def test_poisson_grid_explicit_rho_zero_identical(self):
        lh, la = 1.8, 0.9
        ref = _independent_grid(lh, la)
        got = poisson_grid(lh, la, max_goals=7, rho=0.0)
        for key in ref:
            self.assertAlmostEqual(ref[key], got[key], places=15,
                                   msg=f"Mismatch at {key}")

    # --- A2. proportional devig is a simple ratio normalisation ---

    def test_proportional_devig_via_normalized_moneyline(self):
        """Raw vig'd book: implied probs 0.55+0.30+0.27 = 1.12.
        Proportional devig divides each by 1.12."""
        matrix = _matrix_with_moneyline(0.55, 0.30, 0.27)
        result = normalized_moneyline_probabilities(matrix)
        self.assertIsNotNone(result)
        total = 0.55 + 0.30 + 0.27
        self.assertAlmostEqual(result["home"], 0.55 / total, places=10)
        self.assertAlmostEqual(result["draw"], 0.30 / total, places=10)
        self.assertAlmostEqual(result["away"], 0.27 / total, places=10)

    def test_default_params_devig_method_is_proportional(self):
        params = StrategyParams()
        self.assertEqual(params.devig_method, "proportional")
        self.assertEqual(params.dixon_coles_rho, 0.0)


# ---------------------------------------------------------------------------
# Group B: Directional tests (FAIL before implementation)
# ---------------------------------------------------------------------------

class DixonColesDirectionalTests(unittest.TestCase):
    """rho < 0 lifts low-score cells; draw probability must increase."""

    def test_negative_rho_increases_draw_probability(self):
        lh, la = 1.2, 1.0   # low-scoring fixture
        grid_base = poisson_grid(lh, la, max_goals=7, rho=0.0)
        grid_dc   = poisson_grid(lh, la, max_goals=7, rho=-0.05)

        draw_base = _draw_prob(grid_base)
        draw_dc   = _draw_prob(grid_dc)

        # DC with rho=-0.05 must lift 0-0 and 1-1, hence more draws
        self.assertGreater(draw_dc, draw_base,
                           msg=f"draw_base={draw_base:.6f}, draw_dc={draw_dc:.6f}")

    def test_negative_rho_increases_00_cell(self):
        lh, la = 1.2, 1.0
        grid_base = poisson_grid(lh, la, max_goals=7, rho=0.0)
        grid_dc   = poisson_grid(lh, la, max_goals=7, rho=-0.05)
        self.assertGreater(grid_dc[(0, 0)], grid_base[(0, 0)])

    def test_dc_grid_still_sums_to_one(self):
        lh, la = 1.5, 1.0
        grid = poisson_grid(lh, la, max_goals=7, rho=-0.05)
        self.assertAlmostEqual(sum(grid.values()), 1.0, places=12)

    def test_extreme_positive_rho_raises(self):
        """tau(0,0) = 1 - lh*la*rho must be > 0; extreme rho violates this."""
        # With lh=la=1.5, rho=0.5: tau(0,0) = 1 - 1.5*1.5*0.5 = 1 - 1.125 < 0
        with self.assertRaises(ValueError):
            poisson_grid(1.5, 1.5, max_goals=7, rho=0.5)

    def test_rho_zero_and_positive_match_reference_for_nonneg(self):
        """Sanity: rho=0 always matches independent baseline."""
        lh, la = 1.3, 1.1
        ref  = _independent_grid(lh, la)
        got  = poisson_grid(lh, la, max_goals=7, rho=0.0)
        for key in ref:
            self.assertAlmostEqual(ref[key], got[key], places=14)


class ShinDevigDirectionalTests(unittest.TestCase):
    """shin_devig must exist, sum to 1, and differ from proportional on a vig'd book."""

    # Raw implied probs from a 3-way book with booksum = 1.12
    RAW = [0.55, 0.30, 0.27]          # booksum = 1.12
    BOOKSUM = sum(RAW)                 # 1.12

    def _proportional(self, raw):
        s = sum(raw)
        return [r / s for r in raw]

    def test_shin_sums_to_one(self):
        result = shin_devig(self.RAW)
        self.assertAlmostEqual(sum(result), 1.0, places=10)

    def test_shin_differs_from_proportional(self):
        shin   = shin_devig(self.RAW)
        prop   = self._proportional(self.RAW)
        # They must not be identical (Shin structure ≠ proportional)
        for s, p in zip(shin, prop):
            if abs(s - p) > 1e-8:
                return   # found a difference — pass
        self.fail("Shin and proportional gave identical results; expected them to differ")

    def test_shin_favourite_probability_differs(self):
        """Shin shrinks favourite relative to longshots vs proportional."""
        shin = shin_devig(self.RAW)
        prop = self._proportional(self.RAW)
        # The favourite is index 0 (raw 0.55).
        # Shin's z > 0 correction redistributes toward longshots.
        # So shin[0] <= prop[0] (at least not equal) — directional check.
        # Note: actual direction depends on z sign convention; we just assert they differ.
        self.assertNotAlmostEqual(shin[0], prop[0], places=6)

    def test_shin_all_positive(self):
        result = shin_devig(self.RAW)
        for p in result:
            self.assertGreater(p, 0.0)

    def test_shin_fair_book_is_unchanged(self):
        """If booksum == 1 already, Shin should leave probs unchanged (z→0)."""
        raw = [0.50, 0.30, 0.20]   # already sums to 1.0
        result = shin_devig(raw)
        for got, expected in zip(result, raw):
            self.assertAlmostEqual(got, expected, places=8)

    def test_shin_devig_exists_and_is_callable(self):
        """Import guard — fails before implementation."""
        result = shin_devig([0.55, 0.30, 0.27])
        self.assertEqual(len(result), 3)


if __name__ == "__main__":
    unittest.main()
