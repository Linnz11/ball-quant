"""Tests for HAFU (半全场 / half-time–full-time) support.

Covers:
1. hafu_probabilities sums to ~1.0; all 9 keys present.
2. Sanity ordering: with home-favored lambdas, "hh" (lead-and-win) > "ah" (comeback).
3. FT-result marginal of the HT/FT model ≈ main-grid FT result (within tolerance).
4. analyze_ticai emits hafu selections priced off ticai.hafu odds; edge = P×O-1.
5. Settlement: "dh" → WIN on (HT 0-0, FT 2-1); LOSS on (HT 1-0, FT 2-1); VOID when HT absent.
6. first_half_goal_share calibration: shifts from default when Polymarket 1H market present.
"""

from __future__ import annotations

import dataclasses
import unittest

from ball_quant.core.params import DEFAULT_PARAMS, StrategyParams
from ball_quant.core.settlement import LOSS, VOID, WIN, MatchOutcome, grade
from ball_quant.core.ticai_engine import analyze_ticai, hafu_probabilities
from ball_quant.models import (
    EventMarketMatrix,
    MarketQuote,
    SettlementKey,
    TicaiOdds,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _home_favored_params() -> StrategyParams:
    return DEFAULT_PARAMS


def _base_ticai() -> TicaiOdds:
    return TicaiOdds(
        match_id="H001",
        match_date="2026-06-14",
        league="World Cup",
        home="Brazil",
        away="Korea",
        match_num="周三001",
        spf={"home": 1.80, "draw": 3.60, "away": 4.50},
        handicap_line=-1.0,
        rqspf={"home": 2.50, "draw": 3.40, "away": 2.20},
        correct_score={"1-0": 6.5, "2-0": 7.0, "2-1": 8.5},
        total_goals={0: 18.0, 1: 8.0, 2: 4.5, 3: 3.8, 4: 5.5, 5: 10.0, 6: 20.0, 7: 35.0},
        hafu={
            "hh": 2.0,   # HT home / FT home (home leads and wins)
            "hd": 8.0,   # HT home / FT draw (home leads then draws)
            "ha": 18.0,  # HT home / FT away (home leads then loses)
            "dh": 6.0,   # HT draw / FT home
            "dd": 6.5,   # HT draw / FT draw
            "da": 14.0,  # HT draw / FT away
            "ah": 14.0,  # HT away / FT home (comeback)
            "ad": 12.0,  # HT away / FT draw
            "aa": 6.0,   # HT away / FT away (away leads and wins)
        },
    )


def _liquid_matrix(with_1h_market: bool = False) -> EventMarketMatrix:
    """Polymarket matrix with good moneyline liquidity; optionally includes 1H total."""
    markets = [
        MarketQuote("m1", "winner", "moneyline", "home", 0.62,
                    bid=0.61, ask=0.63, spread=0.02, liquidity=12000),
        MarketQuote("m2", "winner", "moneyline", "draw", 0.22,
                    bid=0.21, ask=0.23, spread=0.02, liquidity=12000),
        MarketQuote("m3", "winner", "moneyline", "away", 0.16,
                    bid=0.15, ask=0.17, spread=0.02, liquidity=12000),
    ]
    if with_1h_market:
        # 1H total goals over 0.5 implied prob = 0.80.
        # parse_total_quote reads (question, outcome): outcome has "over", question has "0.5".
        # Implied ht_total ≈ 0.5 + (0.80-0.5)*0.7 = 0.71; share = 0.71/(1.50+0.95)=0.29
        # → clamped to _HAFU_SHARE_CLAMP_LO=0.30. Default share=0.45 → shift is observable.
        markets.append(
            MarketQuote("m4", "over 0.5", "first_half_total_goals", "over", 0.80,
                        bid=0.79, ask=0.81, spread=0.02, liquidity=3000),
        )
    return EventMarketMatrix(
        match_id="H001",
        home="Brazil",
        away="Korea",
        markets=markets,
    )


def make_outcome(h, a, ht_h=None, ht_a=None):
    return MatchOutcome(
        match_id="H001",
        home_score=h,
        away_score=a,
        ht_home_score=ht_h,
        ht_away_score=ht_a,
    )


# ---------------------------------------------------------------------------
# 1. hafu_probabilities — sum, keys, no crash
# ---------------------------------------------------------------------------

class TestHafuProbabilitiesBasic(unittest.TestCase):
    """hafu_probabilities returns all 9 buckets summing to 1."""

    def setUp(self):
        self.params = _home_favored_params()
        # Home-favored: home_lambda > away_lambda
        self.probs = hafu_probabilities(
            home_lambda=1.50, away_lambda=0.95, params=self.params
        )

    def test_nine_keys_present(self):
        expected = {"hh", "hd", "ha", "dh", "dd", "da", "ah", "ad", "aa"}
        self.assertEqual(set(self.probs.keys()), expected)

    def test_sums_to_one(self):
        total = sum(self.probs.values())
        self.assertAlmostEqual(total, 1.0, places=9,
                               msg=f"hafu_probabilities must sum to 1.0, got {total}")

    def test_all_probabilities_nonnegative(self):
        for key, p in self.probs.items():
            self.assertGreaterEqual(p, 0.0, f"P({key}) must be non-negative")

    def test_all_probabilities_at_most_one(self):
        for key, p in self.probs.items():
            self.assertLessEqual(p, 1.0, f"P({key}) must be ≤ 1.0")


# ---------------------------------------------------------------------------
# 2. Sanity ordering with home-favored lambdas
# ---------------------------------------------------------------------------

class TestHafuSanityOrdering(unittest.TestCase):
    """With home-favored lambdas, verify intuitive ordering of bucket probabilities."""

    def setUp(self):
        self.params = _home_favored_params()
        self.probs = hafu_probabilities(
            home_lambda=1.60, away_lambda=0.80, params=self.params
        )

    def test_hh_greater_than_ah(self):
        """Lead-and-win (hh) must be more probable than comeback (ah)
        when home team is heavily favored."""
        self.assertGreater(
            self.probs["hh"], self.probs["ah"],
            f"P(hh)={self.probs['hh']:.4f} should exceed P(ah)={self.probs['ah']:.4f}",
        )

    def test_aa_greater_than_ha(self):
        """Away-lead-and-win (aa) should exceed home-lead-then-lose (ha).
        Even though home is favored overall, conditioning on away leading at HT
        makes aa (stays ahead) more likely than ha (collapse to home win)."""
        # WHY: P(away win FT | away winning HT) > P(home win FT | away winning HT)
        # because the home team must mount a multi-goal comeback from deficit.
        self.assertGreater(
            self.probs["aa"], self.probs["ha"],
            f"P(aa)={self.probs['aa']:.4f} should exceed P(ha)={self.probs['ha']:.4f}",
        )

    def test_hh_is_largest_bucket(self):
        """For a strongly home-favored match, hh (home leads and wins) is the
        most likely single bucket."""
        max_key = max(self.probs, key=self.probs.get)
        self.assertEqual(
            max_key, "hh",
            f"Expected hh to be largest bucket; got {max_key} "
            f"(probs={self.probs})",
        )


# ---------------------------------------------------------------------------
# 3. FT marginal ≈ main-grid FT result probs (same lambdas)
# ---------------------------------------------------------------------------

class TestHafuFTMarginalConsistency(unittest.TestCase):
    """FT-result marginal of the HT/FT model should be close to the FT grid."""

    def setUp(self):
        from ball_quant.core.probability import poisson_grid
        self.params = _home_favored_params()
        self.home_lambda = 1.50
        self.away_lambda = 0.95
        self.hafu_probs = hafu_probabilities(
            self.home_lambda, self.away_lambda, params=self.params
        )
        # Build main FT grid for comparison (independent Poisson, rho=0)
        raw_grid = poisson_grid(self.home_lambda, self.away_lambda, self.params.max_goals)
        self.ft_home = sum(p for (h, a), p in raw_grid.items() if h > a)
        self.ft_draw = sum(p for (h, a), p in raw_grid.items() if h == a)
        self.ft_away = sum(p for (h, a), p in raw_grid.items() if h < a)

    def _hafu_ft_marginal(self):
        """Sum hafu buckets to recover FT result probabilities."""
        ft_home = self.hafu_probs["hh"] + self.hafu_probs["dh"] + self.hafu_probs["ah"]
        ft_draw = self.hafu_probs["hd"] + self.hafu_probs["dd"] + self.hafu_probs["ad"]
        ft_away = self.hafu_probs["ha"] + self.hafu_probs["da"] + self.hafu_probs["aa"]
        return ft_home, ft_draw, ft_away

    def test_ft_home_marginal_close(self):
        hafu_ft_home, _, _ = self._hafu_ft_marginal()
        self.assertAlmostEqual(
            hafu_ft_home, self.ft_home, delta=0.06,
            msg=f"HAFU FT-home marginal {hafu_ft_home:.4f} differs from "
                f"main-grid {self.ft_home:.4f} by more than 6pp",
        )

    def test_ft_draw_marginal_close(self):
        _, hafu_ft_draw, _ = self._hafu_ft_marginal()
        self.assertAlmostEqual(
            hafu_ft_draw, self.ft_draw, delta=0.06,
            msg=f"HAFU FT-draw marginal {hafu_ft_draw:.4f} differs from "
                f"main-grid {self.ft_draw:.4f} by more than 6pp",
        )

    def test_ft_away_marginal_close(self):
        _, _, hafu_ft_away = self._hafu_ft_marginal()
        self.assertAlmostEqual(
            hafu_ft_away, self.ft_away, delta=0.06,
            msg=f"HAFU FT-away marginal {hafu_ft_away:.4f} differs from "
                f"main-grid {self.ft_away:.4f} by more than 6pp",
        )


# ---------------------------------------------------------------------------
# 4. analyze_ticai emits hafu selections with correct edge
# ---------------------------------------------------------------------------

class TestAnalyzeTicaiHafu(unittest.TestCase):
    """analyze_ticai must emit hafu selections using 体彩 posted odds."""

    def setUp(self):
        self.ticai = _base_ticai()
        self.matrix = _liquid_matrix()
        self.selections, self.skipped = analyze_ticai(self.ticai, self.matrix)
        self.hafu_sels = [s for s in self.selections if s.play == "hafu"]

    def test_hafu_selections_present(self):
        """All 9 hafu keys with valid odds must produce a selection."""
        hafu_outcomes = {s.outcome for s in self.hafu_sels}
        for key in ("hh", "hd", "ha", "dh", "dd", "da", "ah", "ad", "aa"):
            self.assertIn(key, hafu_outcomes,
                          f"hafu:{key} selection must be emitted")

    def test_hafu_edge_formula(self):
        """edge = P_polymarket × O_体彩 − 1 for every hafu leg."""
        for sel in self.hafu_sels:
            self.assertAlmostEqual(
                sel.edge,
                sel.probability * sel.sp - 1.0,
                places=10,
                msg=f"edge formula must hold for hafu:{sel.outcome}",
            )

    def test_hafu_sp_equals_ticai_odds(self):
        """sp field must equal the 体彩 posted hafu odds (never Polymarket)."""
        for sel in self.hafu_sels:
            expected_sp = self.ticai.hafu[sel.outcome]
            self.assertAlmostEqual(
                sel.sp, expected_sp, places=8,
                msg=f"hafu:{sel.outcome} sp={sel.sp} must equal ticai.hafu[{sel.outcome!r}]={expected_sp}",
            )

    def test_hafu_settlement_keys_set(self):
        """Every hafu selection must have a settlement_key with market_type='hafu'."""
        for sel in self.hafu_sels:
            self.assertIsNotNone(sel.settlement_key)
            self.assertEqual(sel.settlement_key.market_type, "hafu")
            self.assertEqual(sel.settlement_key.side, sel.outcome)

    def test_hafu_not_skipped_when_valid(self):
        """With a full hafu dict, no hafu entries should appear in skipped."""
        hafu_skips = [n for n in self.skipped if "hafu" in n.lower()]
        # Only invalid-odds skips are permitted; the 9 valid keys must not be skipped.
        for note in hafu_skips:
            self.assertIn("missing or invalid", note.lower(),
                          f"Unexpected hafu skip: {note!r}")

    def test_hafu_missing_odds_skipped(self):
        """A hafu key with odds ≤ 1 must be skipped, not crash."""
        ticai_bad = dataclasses.replace(
            self.ticai,
            hafu={"hh": 0.5, "hd": 8.0},  # hh has invalid odds ≤1
        )
        sels, skipped = analyze_ticai(ticai_bad, self.matrix)
        hafu_sels = [s for s in sels if s.play == "hafu"]
        # Only "hd" should produce a selection; "hh" should be skipped
        self.assertTrue(any(s.outcome == "hd" for s in hafu_sels))
        self.assertTrue(any("hafu:hh" in n for n in skipped))


# ---------------------------------------------------------------------------
# 5. Settlement grading for hafu
# ---------------------------------------------------------------------------

class TestHafuSettlement(unittest.TestCase):
    """Grade hafu bets: WIN/LOSS/VOID scenarios."""

    def _key(self, side: str) -> SettlementKey:
        return SettlementKey(market_type="hafu", side=side)

    def test_dh_win_on_ht_draw_ft_home(self):
        """'dh' (HT draw, FT home win) must grade WIN on (HT 0-0, FT 2-1)."""
        key = self._key("dh")
        outcome = make_outcome(h=2, a=1, ht_h=0, ht_a=0)
        self.assertEqual(grade(key, outcome), WIN)

    def test_dh_loss_on_ht_home_ft_home(self):
        """'dh' must grade LOSS on (HT 1-0, FT 2-1) — HT result is 'h', not 'd'."""
        key = self._key("dh")
        outcome = make_outcome(h=2, a=1, ht_h=1, ht_a=0)
        self.assertEqual(grade(key, outcome), LOSS)

    def test_void_when_ht_scores_absent(self):
        """hafu bet grades VOID when ht scores are not provided."""
        key = self._key("dh")
        outcome = make_outcome(h=2, a=1)  # ht_h and ht_a default to None
        self.assertEqual(grade(key, outcome), VOID)

    def test_hh_win_on_ht_home_ft_home(self):
        """'hh' wins when home leads at HT and wins FT."""
        key = self._key("hh")
        outcome = make_outcome(h=2, a=0, ht_h=1, ht_a=0)
        self.assertEqual(grade(key, outcome), WIN)

    def test_hh_loss_on_ht_draw_ft_home(self):
        """'hh' loses when HT is draw even if FT is home win."""
        key = self._key("hh")
        outcome = make_outcome(h=2, a=1, ht_h=0, ht_a=0)
        self.assertEqual(grade(key, outcome), LOSS)

    def test_aa_win_on_ht_away_ft_away(self):
        """'aa' wins when away leads at HT and wins FT."""
        key = self._key("aa")
        outcome = make_outcome(h=0, a=2, ht_h=0, ht_a=1)
        self.assertEqual(grade(key, outcome), WIN)

    def test_ah_win_on_comeback(self):
        """'ah' (HT away, FT home) wins on a home comeback."""
        key = self._key("ah")
        outcome = make_outcome(h=2, a=1, ht_h=0, ht_a=1)
        self.assertEqual(grade(key, outcome), WIN)

    def test_void_outcome_propagates(self):
        """A voided match grades all hafu bets as VOID."""
        key = self._key("hh")
        outcome = MatchOutcome(match_id="H001", home_score=2, away_score=0, void=True,
                               ht_home_score=1, ht_away_score=0)
        self.assertEqual(grade(key, outcome), VOID)

    def test_invalid_side_void(self):
        """A malformed 2-char side (unknown chars) grades VOID."""
        key = self._key("xy")
        outcome = make_outcome(h=2, a=1, ht_h=1, ht_a=0)
        self.assertEqual(grade(key, outcome), VOID)


# ---------------------------------------------------------------------------
# 6. first_half_goal_share calibration
# ---------------------------------------------------------------------------

class TestFirstHalfGoalShareCalibration(unittest.TestCase):
    """Verify share calibration from Polymarket 1H market."""

    def test_no_1h_market_uses_default(self):
        """Without a 1H market, hafu_probabilities uses the default share=0.45."""
        params = DEFAULT_PARAMS
        # Call without matrix — share must come from params.first_half_goal_share=0.45
        probs_default = hafu_probabilities(1.50, 0.95, params=params, matrix=None)
        # Call with a matrix that has NO first_half_total_goals quotes
        matrix_no_1h = _liquid_matrix(with_1h_market=False)
        probs_no_1h = hafu_probabilities(1.50, 0.95, params=params, matrix=matrix_no_1h)
        # Both should produce identical results (same code path)
        for key in probs_default:
            self.assertAlmostEqual(
                probs_default[key], probs_no_1h[key], places=12,
                msg=f"P({key}) should be identical when no 1H market present",
            )

    def test_1h_market_shifts_share_from_default(self):
        """When a Polymarket 1H total market is present, the calibrated share
        should differ from the default 0.45 default."""
        params = DEFAULT_PARAMS
        # Matrix WITHOUT 1H market — uses default share
        matrix_no_1h = _liquid_matrix(with_1h_market=False)
        probs_without = hafu_probabilities(1.50, 0.95, params=params, matrix=matrix_no_1h)

        # Matrix WITH a 1H over 0.5 @ 0.80 market — implied ht_total is lower
        # than 0.45 * (1.50+0.95) = 1.1025; so share should move toward 0.3 clamp.
        matrix_with_1h = _liquid_matrix(with_1h_market=True)
        probs_with = hafu_probabilities(1.50, 0.95, params=params, matrix=matrix_with_1h)

        # The two distributions must differ meaningfully (share has shifted)
        max_diff = max(abs(probs_with[k] - probs_without[k]) for k in probs_with)
        self.assertGreater(
            max_diff, 1e-6,
            "Presence of a 1H market should shift hafu bucket probabilities"
        )

    def test_default_share_param_value(self):
        """DEFAULT_PARAMS.first_half_goal_share == 0.45."""
        self.assertAlmostEqual(DEFAULT_PARAMS.first_half_goal_share, 0.45, places=9)

    def test_custom_share_param_respected(self):
        """A custom first_half_goal_share propagates to the model."""
        params_high = dataclasses.replace(DEFAULT_PARAMS, first_half_goal_share=0.55)
        params_low = dataclasses.replace(DEFAULT_PARAMS, first_half_goal_share=0.35)
        probs_high = hafu_probabilities(1.50, 0.95, params=params_high)
        probs_low = hafu_probabilities(1.50, 0.95, params=params_low)
        # With a higher first-half share, the HT distribution has more goals → "hh" should
        # be relatively higher (more chance of home lead at HT compared to low share).
        # With lower share, 1H goals are scarcer → more HT-draw paths (dh, dd, da).
        dd_high = probs_high["dd"]
        dd_low = probs_low["dd"]
        self.assertGreater(
            dd_low, dd_high,
            "Lower 1H share → fewer 1H goals → more HT draws → higher P(dd)",
        )


# ---------------------------------------------------------------------------
# 7. FT-marginal rescale: Fix A correctness tests
# ---------------------------------------------------------------------------

class TestHafuFTMarginalRescale(unittest.TestCase):
    """After Fix A, the 9 hafu buckets must exactly match the calibrated
    score_distribution FT marginals — not just approximately."""

    def _build_probs_with_grid(self, home_lambda: float, away_lambda: float) -> dict:
        from ball_quant.core.probability import poisson_grid, ScoreDistribution
        params = DEFAULT_PARAMS
        raw = poisson_grid(home_lambda, away_lambda, params.max_goals)
        dist = ScoreDistribution(raw, max_goals=params.max_goals)
        return hafu_probabilities(
            home_lambda, away_lambda, params,
            matrix=None, score_distribution=dist,
        ), dist

    def test_ft_home_group_equals_grid_marginal(self):
        """sum(hh,dh,ah) must equal grid P(FT home) to floating-point precision."""
        probs, dist = self._build_probs_with_grid(1.50, 0.95)
        ft_home_hafu = probs["hh"] + probs["dh"] + probs["ah"]
        ft_home_grid = dist.probability(lambda h, a: h > a)
        self.assertAlmostEqual(
            ft_home_hafu, ft_home_grid, places=10,
            msg=f"FT-home hafu group {ft_home_hafu:.8f} != grid {ft_home_grid:.8f}",
        )

    def test_ft_draw_group_equals_grid_marginal(self):
        """sum(hd,dd,ad) must equal grid P(FT draw) to floating-point precision."""
        probs, dist = self._build_probs_with_grid(1.50, 0.95)
        ft_draw_hafu = probs["hd"] + probs["dd"] + probs["ad"]
        ft_draw_grid = dist.probability(lambda h, a: h == a)
        self.assertAlmostEqual(
            ft_draw_hafu, ft_draw_grid, places=10,
            msg=f"FT-draw hafu group {ft_draw_hafu:.8f} != grid {ft_draw_grid:.8f}",
        )

    def test_ft_away_group_equals_grid_marginal(self):
        """sum(ha,da,aa) must equal grid P(FT away) to floating-point precision."""
        probs, dist = self._build_probs_with_grid(1.50, 0.95)
        ft_away_hafu = probs["ha"] + probs["da"] + probs["aa"]
        ft_away_grid = dist.probability(lambda h, a: h < a)
        self.assertAlmostEqual(
            ft_away_hafu, ft_away_grid, places=10,
            msg=f"FT-away hafu group {ft_away_hafu:.8f} != grid {ft_away_grid:.8f}",
        )

    def test_no_hafu_subtype_exceeds_its_ft_marginal(self):
        """Every hafu key must be <= P(its FT result) — the Spain/Cape Verde bug."""
        for home_lam, away_lam in [(0.5, 2.5), (1.5, 0.95), (1.2, 1.2), (2.0, 0.3)]:
            probs, dist = self._build_probs_with_grid(home_lam, away_lam)
            ft_home = dist.probability(lambda h, a: h > a)
            ft_draw = dist.probability(lambda h, a: h == a)
            ft_away = dist.probability(lambda h, a: h < a)
            for key in ("hh", "dh", "ah"):
                self.assertLessEqual(
                    probs[key], ft_home + 1e-9,
                    f"P(hafu:{key})={probs[key]:.6f} > P(FT home)={ft_home:.6f} "
                    f"(lambdas={home_lam},{away_lam})",
                )
            for key in ("hd", "dd", "ad"):
                self.assertLessEqual(
                    probs[key], ft_draw + 1e-9,
                    f"P(hafu:{key})={probs[key]:.6f} > P(FT draw)={ft_draw:.6f} "
                    f"(lambdas={home_lam},{away_lam})",
                )
            for key in ("ha", "da", "aa"):
                self.assertLessEqual(
                    probs[key], ft_away + 1e-9,
                    f"P(hafu:{key})={probs[key]:.6f} > P(FT away)={ft_away:.6f} "
                    f"(lambdas={home_lam},{away_lam})",
                )

    def test_analyze_ticai_hafu_respects_ft_marginal(self):
        """In the full pipeline (analyze_ticai), hafu:aa probability must be
        ≤ the grid P(FT away) for a match where the away team is a heavy underdog
        (replicating the Spain vs Cape Verde scenario)."""
        from ball_quant.core.ticai_engine import analyze_ticai
        from ball_quant.models import EventMarketMatrix, MarketQuote, TicaiOdds
        from ball_quant.core.probability import build_probability_context

        # Underdog scenario: P(FT away) ≈ 2.35% (Spain-heavy match)
        ticai = TicaiOdds(
            match_id="FIX_A",
            match_date="2026-06-15",
            league="World Cup",
            home="Spain",
            away="Cape Verde",
            match_num="周日001",
            spf={"home": 1.25, "draw": 5.5, "away": 12.0},
            handicap_line=-2.0,
            rqspf={"home": 2.10, "draw": 3.20, "away": 3.50},
            correct_score={},
            total_goals={},
            hafu={
                "hh": 1.8, "hd": 9.0, "ha": 30.0,
                "dh": 5.0, "dd": 9.0, "da": 30.0,
                "ah": 25.0, "ad": 30.0, "aa": 30.0,
            },
        )
        # Moneyline reflecting ~80% home / ~4% away
        matrix = EventMarketMatrix(
            match_id="FIX_A",
            home="Spain",
            away="Cape Verde",
            markets=[
                MarketQuote("m1", "winner", "moneyline", "home", 0.80,
                            bid=0.79, ask=0.81, spread=0.02, liquidity=20000),
                MarketQuote("m2", "winner", "moneyline", "draw", 0.16,
                            bid=0.15, ask=0.17, spread=0.02, liquidity=20000),
                MarketQuote("m3", "winner", "moneyline", "away", 0.04,
                            bid=0.03, ask=0.05, spread=0.02, liquidity=20000),
            ],
        )
        selections, _ = analyze_ticai(ticai, matrix)
        hafu_sels = {s.outcome: s for s in selections if s.play == "hafu"}

        # Get calibrated FT-away prob from context
        from ball_quant.core.ticai_engine import _synthetic_match_sp
        from ball_quant.core.params import DEFAULT_PARAMS
        m = _synthetic_match_sp(ticai)
        ctx = build_probability_context(m, matrix, DEFAULT_PARAMS)
        ft_away = ctx.score_distribution.probability(lambda h, a: h < a)

        for key in ("ha", "da", "aa"):
            if key in hafu_sels:
                p = hafu_sels[key].probability
                self.assertLessEqual(
                    p, ft_away + 1e-9,
                    f"hafu:{key} P={p:.6f} > P(FT away)={ft_away:.6f} — "
                    "impossible sub-event exceeds its FT marginal",
                )


if __name__ == "__main__":
    unittest.main()
