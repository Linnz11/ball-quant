"""Tests for the fundamental (Elo) cross-check block in build_bundle.

All tests are network-free: Elo ratings are injected as plain dicts; MarketQuote
/ EventMarketMatrix / TicaiOdds are built from minimal synthetic data.

Three scenarios:
  1. WITH elo_ratings: fundamental block present; probs sum to 1; stronger team
     has higher Elo-implied p; delta_home == poly_p_home - p_home_elo.
  2. UNRATED team: home_rated=False/away_rated=False; block still present (no crash,
     no fabricated rating — z=0 fallback used).
  3. WITHOUT elo_ratings (None): no fundamental block (back-compat; existing code
     not broken).
"""
from __future__ import annotations

import pytest

from ball_quant.core.bundle import build_bundle
from ball_quant.core.match_join import normalize_team
from ball_quant.core.params import StrategyParams
from ball_quant.models import EventMarketMatrix, MarketQuote, TicaiOdds


# ---------------------------------------------------------------------------
# Shared synthetic fixtures
# ---------------------------------------------------------------------------

def _make_ticai(home: str = "Spain", away: str = "England") -> TicaiOdds:
    """Minimal TicaiOdds — only the fields build_bundle touches."""
    return TicaiOdds(
        match_id="test-001",
        match_date="2026-06-20",
        league="World Cup",
        home=home,
        away=away,
        match_num="周五001",
        spf={"home": 1.8, "draw": 3.5, "away": 4.0},
        handicap_line=-1.0,
        rqspf={"home": 1.9, "draw": 3.4, "away": 3.8},
        correct_score={"1-0": 7.0, "0-0": 11.0},
        total_goals={0: 8.0, 1: 4.5, 2: 3.2, 3: 4.0},
        hafu={"hh": 3.5, "hd": 7.0, "ha": 17.0, "dh": 4.5, "dd": 5.0,
              "da": 9.0, "ah": 7.5, "ad": 8.0, "aa": 11.0},
    )


def _make_matrix(home: str = "Spain", away: str = "England",
                 poly_home_prob: float = 0.60,
                 poly_draw_prob: float = 0.20,
                 poly_away_prob: float = 0.20) -> EventMarketMatrix:
    """EventMarketMatrix with a moneyline angle so _poly_1x2 can extract probs."""
    quotes = [
        MarketQuote(
            market_id="ml-home",
            question="Will Spain win?",
            category="moneyline",
            outcome="home",
            probability=poly_home_prob,
            liquidity=50000.0,
            spread=0.02,
        ),
        MarketQuote(
            market_id="ml-draw",
            question="Will there be a draw?",
            category="moneyline",
            outcome="draw",
            probability=poly_draw_prob,
            liquidity=50000.0,
            spread=0.03,
        ),
        MarketQuote(
            market_id="ml-away",
            question="Will England win?",
            category="moneyline",
            outcome="away",
            probability=poly_away_prob,
            liquidity=50000.0,
            spread=0.02,
        ),
    ]
    return EventMarketMatrix(
        match_id="test-001",
        home=home,
        away=away,
        event_id="evt-001",
        event_slug="spain-vs-england-2026-06-20",
        markets=quotes,
    )


# Elo ratings keyed by normalized name (as the adapter produces).
# Spain (2129) vs England (2050): Spain is clearly stronger.
_ELO_RATINGS = {
    normalize_team("Spain"): 2129.0,
    normalize_team("England"): 2050.0,
    normalize_team("Brazil"): 2060.0,
    normalize_team("France"): 2084.0,
    normalize_team("Argentina"): 2128.0,
}

_PARAMS = StrategyParams()


# ---------------------------------------------------------------------------
# 1. WITH elo_ratings: fundamental block content
# ---------------------------------------------------------------------------

class TestFundamentalBlockPresent:
    """Each helper returns the full bundle dict; tests extract 'fundamental' as needed."""

    def _build(self, home="Spain", away="England",
               poly_home=0.60, poly_draw=0.20, poly_away=0.20) -> dict:
        """Return the full bundle dict (index 0) with Elo injected."""
        pairs = [(_make_ticai(home, away), _make_matrix(home, away, poly_home, poly_draw, poly_away))]
        return build_bundle(pairs, elo_ratings=_ELO_RATINGS, params=_PARAMS)[0]

    def _fund(self, **kwargs) -> dict:
        """Convenience: return the fundamental sub-block."""
        return self._build(**kwargs)["fundamental"]

    def test_fundamental_block_present(self):
        """fundamental block is present when elo_ratings are injected."""
        b = self._build()
        assert "fundamental" in b

    def test_fundamental_source_field(self):
        """source field must be 'elo'."""
        assert self._fund()["source"] == "elo"

    def test_elo_probs_sum_to_one(self):
        """Elo-implied p_home + p_draw + p_away must sum to ≈ 1.0."""
        fund = self._fund()
        total = fund["p_home"] + fund["p_draw"] + fund["p_away"]
        assert total == pytest.approx(1.0, abs=1e-4)

    def test_stronger_team_higher_elo_prob(self):
        """Spain (Elo 2129) vs England (2050): Elo-implied p_home > p_away."""
        fund = self._fund()
        # Spain is home (higher Elo) → should win more often than England
        assert fund["p_home"] > fund["p_away"], (
            f"Spain (home, stronger) p_home={fund['p_home']} should > "
            f"England (away) p_away={fund['p_away']}"
        )

    def test_lambdas_present(self):
        """lam_home and lam_away fields are present and positive."""
        fund = self._fund()
        assert "lam_home" in fund and "lam_away" in fund
        assert fund["lam_home"] > 0
        assert fund["lam_away"] > 0

    def test_stronger_team_higher_lambda(self):
        """Spain (home, stronger) → lam_home > lam_away."""
        fund = self._fund()
        assert fund["lam_home"] > fund["lam_away"]

    def test_delta_home_equals_poly_minus_elo(self):
        """delta_home must equal poly_p_home - p_home_elo exactly."""
        fund = self._fund(poly_home=0.77, poly_draw=0.16, poly_away=0.07)
        expected_delta = round(fund["poly_p_home"] - fund["p_home"], 4)
        assert fund["delta_home"] == pytest.approx(expected_delta, abs=1e-4)

    def test_delta_away_equals_poly_minus_elo(self):
        """delta_away must equal poly_p_away - p_away_elo."""
        fund = self._fund(poly_home=0.77, poly_draw=0.16, poly_away=0.07)
        # poly_p_away = elo_p_away + delta_away; should recover injected 0.07
        poly_p_away = fund["p_away"] + fund["delta_away"]
        assert poly_p_away == pytest.approx(0.07, abs=0.001)

    def test_poly_p_home_stored(self):
        """poly_p_home should match the injected Polymarket moneyline home prob."""
        fund = self._fund(poly_home=0.77, poly_draw=0.16, poly_away=0.07)
        assert fund["poly_p_home"] == pytest.approx(0.77, abs=0.001)

    def test_rated_flags_true_for_known_teams(self):
        """home_rated=True, away_rated=True when both teams are in elo_ratings."""
        fund = self._fund()
        assert fund["home_rated"] is True
        assert fund["away_rated"] is True


# ---------------------------------------------------------------------------
# 2. UNRATED team: home_rated / away_rated = False
# ---------------------------------------------------------------------------

class TestUnratedTeam:
    def test_unrated_home_flagged(self):
        """Team absent from elo_ratings → home_rated=False; block still present."""
        pairs = [(_make_ticai("Atlantis FC", "Spain"),
                  _make_matrix("Atlantis FC", "Spain"))]
        b = build_bundle(pairs, elo_ratings=_ELO_RATINGS, params=_PARAMS)[0]
        assert "fundamental" in b
        fund = b["fundamental"]
        assert fund["home_rated"] is False
        assert fund["away_rated"] is True  # Spain is rated

    def test_unrated_away_flagged(self):
        """Away team absent from elo_ratings → away_rated=False."""
        pairs = [(_make_ticai("Spain", "Unknown United"),
                  _make_matrix("Spain", "Unknown United"))]
        b = build_bundle(pairs, elo_ratings=_ELO_RATINGS, params=_PARAMS)[0]
        fund = b["fundamental"]
        assert fund["home_rated"] is True   # Spain is rated
        assert fund["away_rated"] is False

    def test_unrated_probs_still_sum_to_one(self):
        """Even with z=0 fallback for missing team, probs must sum to 1."""
        pairs = [(_make_ticai("Atlantis FC", "Spain"),
                  _make_matrix("Atlantis FC", "Spain"))]
        b = build_bundle(pairs, elo_ratings=_ELO_RATINGS, params=_PARAMS)[0]
        fund = b["fundamental"]
        total = fund["p_home"] + fund["p_draw"] + fund["p_away"]
        assert total == pytest.approx(1.0, abs=1e-4)

    def test_unrated_no_fabricated_elo(self):
        """Missing team must use z=0 fallback — its lambda must equal baseline
        when both teams are missing (all-zero z-scores → equal strength → baseline).
        """
        pairs = [(_make_ticai("Atlantis FC", "Unknown United"),
                  _make_matrix("Atlantis FC", "Unknown United"))]
        # Inject a ratings dict that has NEITHER team so both use z=0
        sparse_ratings = {normalize_team("Brazil"): 2060.0}
        b = build_bundle(pairs, elo_ratings=sparse_ratings, params=_PARAMS)[0]
        fund = b["fundamental"]
        # With z=0 for both, sup=0 → lam_home == lam_away == baseline
        baseline = _PARAMS.elo_baseline_goals
        assert fund["lam_home"] == pytest.approx(baseline, rel=1e-6)
        assert fund["lam_away"] == pytest.approx(baseline, rel=1e-6)
        assert fund["home_rated"] is False
        assert fund["away_rated"] is False


# ---------------------------------------------------------------------------
# 3. WITHOUT elo_ratings (None): no fundamental block — back-compat
# ---------------------------------------------------------------------------

class TestNoEloRatings:
    def test_no_fundamental_block_when_elo_ratings_none(self):
        """When elo_ratings is not passed (None default), fundamental must be absent."""
        pairs = [(_make_ticai(), _make_matrix())]
        b = build_bundle(pairs)[0]
        assert "fundamental" not in b

    def test_existing_bundle_keys_still_present(self):
        """Without elo_ratings, all other bundle keys remain intact."""
        pairs = [(_make_ticai(), _make_matrix())]
        b = build_bundle(pairs)[0]
        for key in ("poly_home", "poly_away", "poly", "ticai", "kg", "poly_liquidity"):
            assert key in b, f"Expected bundle key {key!r} missing"
