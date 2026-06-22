"""Spec-first tests for full-market betting engine (totals / btts / team_total / correct_score).

Design contract (STANDING ORDERS):
- bet_markets default = all 6 → match_branches emits totals/btts/correct_score branches.
- bet_markets=("spf","handicap") → branch set byte-identical to old behaviour.
- Each new branch → selections_from_branches produces Selection with finite sp from ask.
- settlement.grade grades new-market Selections correctly end-to-end.
- generate_combos accepts new selection types without error.
"""

from __future__ import annotations

import math
import unittest

from ball_quant.core.combo import generate_combos
from ball_quant.core.params import StrategyParams
from ball_quant.core.probability import build_probability_context, match_branches
from ball_quant.core.settlement import LOSS, WIN, MatchOutcome, grade
from ball_quant.core.value import selections_from_branches
from ball_quant.models import (
    EventMarketMatrix,
    MarketQuote,
    MatchSP,
    SettlementKey,
    TeamFacts,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _base_match() -> MatchSP:
    return MatchSP(
        match_id="m01",
        date="2026-06-14",
        home="Netherlands",
        away="Japan",
        spf_home=1.55,
        spf_draw=3.90,
        spf_away=5.60,
        handicap=-1,
        rq_home=2.78,
        rq_draw=3.55,
        rq_away=2.05,
    )


def _full_matrix() -> EventMarketMatrix:
    """Matrix that covers all 6 market categories used in the tests."""
    return EventMarketMatrix(
        match_id="m01",
        home="Netherlands",
        away="Japan",
        markets=[
            # moneyline
            MarketQuote("q1", "winner", "moneyline", "home", 0.62,
                        bid=0.61, ask=0.63, spread=0.02, liquidity=10000),
            MarketQuote("q2", "winner", "moneyline", "draw", 0.22,
                        bid=0.21, ask=0.23, spread=0.02, liquidity=10000),
            MarketQuote("q3", "winner", "moneyline", "away", 0.16,
                        bid=0.15, ask=0.17, spread=0.02, liquidity=10000),
            # handicap
            MarketQuote("q4", "Netherlands -1.5", "handicap", "Netherlands -1.5",
                        0.34, bid=0.33, ask=0.35, spread=0.02, liquidity=8000),
            MarketQuote("q5", "Japan +0.5", "handicap", "Japan +0.5",
                        0.38, bid=0.37, ask=0.39, spread=0.02, liquidity=8000),
            # totals
            MarketQuote("q6", "O/U 2.5", "total_goals", "over 2.5", 0.58,
                        bid=0.57, ask=0.59, line=2.5, spread=0.02, liquidity=9000),
            MarketQuote("q7", "O/U 2.5", "total_goals", "under 2.5", 0.42,
                        bid=0.41, ask=0.43, line=2.5, spread=0.02, liquidity=9000),
            # btts
            MarketQuote("q8", "Both teams to score", "btts", "yes", 0.55,
                        bid=0.54, ask=0.56, spread=0.02, liquidity=7000),
            MarketQuote("q9", "Both teams to score", "btts", "no", 0.45,
                        bid=0.44, ask=0.46, spread=0.02, liquidity=7000),
            # team_total (home side)
            MarketQuote("q10", "Netherlands O/U 1.5", "team_total",
                        "Netherlands over 1.5", 0.52,
                        bid=0.51, ask=0.53, line=1.5, spread=0.02, liquidity=5000,
                        entity="Netherlands"),
            MarketQuote("q11", "Netherlands O/U 1.5", "team_total",
                        "Netherlands under 1.5", 0.48,
                        bid=0.47, ask=0.49, line=1.5, spread=0.02, liquidity=5000,
                        entity="Netherlands"),
            # correct score
            MarketQuote("q12", "Correct Score 2-1", "correct_score", "2-1", 0.12,
                        bid=0.11, ask=0.13, spread=0.02, liquidity=3000),
        ],
    )


def _facts() -> TeamFacts:
    return TeamFacts(match_id="m01", source="test",
                     home_summary="strong", away_summary="weak")


# ---------------------------------------------------------------------------
# 1. Branch emission tests
# ---------------------------------------------------------------------------

class TestMatchBranchesNewMarkets(unittest.TestCase):

    def _branches_by_play(self, params=None):
        match = _base_match()
        matrix = _full_matrix()
        if params is None:
            context = build_probability_context(match, matrix)
        else:
            context = build_probability_context(match, matrix, params=params)
        branches = match_branches(match, context)
        result: dict = {}
        for b in branches:
            result.setdefault(b.play, []).append(b)
        return result

    def test_default_params_emits_totals_branch(self):
        by_play = self._branches_by_play()
        totals_plays = [p for p in by_play if p.startswith("totals(")]
        self.assertTrue(len(totals_plays) > 0,
                        f"Expected totals branches; found plays: {list(by_play)}")

    def test_default_params_emits_btts_branches(self):
        by_play = self._branches_by_play()
        self.assertIn("btts", by_play,
                      f"Expected btts branches; found plays: {list(by_play)}")
        sides = {b.outcome for b in by_play["btts"]}
        self.assertIn("yes", sides)
        self.assertIn("no", sides)

    def test_default_params_emits_correct_score_branch(self):
        by_play = self._branches_by_play()
        self.assertIn("correct_score", by_play,
                      f"Expected correct_score branches; found plays: {list(by_play)}")
        outcomes = {b.outcome for b in by_play["correct_score"]}
        self.assertIn("2-1", outcomes)

    def test_default_params_emits_team_total_branch(self):
        by_play = self._branches_by_play()
        tt_plays = [p for p in by_play if p.startswith("team_total(")]
        self.assertTrue(len(tt_plays) > 0,
                        f"Expected team_total branches; found plays: {list(by_play)}")

    def test_correct_score_branch_tagged_exact_margin(self):
        by_play = self._branches_by_play()
        cs_branches = by_play.get("correct_score", [])
        self.assertTrue(len(cs_branches) > 0)
        for b in cs_branches:
            self.assertIn("exact_margin", b.tags,
                          f"correct_score branch {b.outcome} missing exact_margin tag")

    def test_totals_branch_probability_from_grid(self):
        """Probability of over 2.5 must come from score_distribution, not raw quote."""
        match = _base_match()
        matrix = _full_matrix()
        context = build_probability_context(match, matrix)
        branches = match_branches(match, context)
        over_branches = [b for b in branches if b.play == "totals(2.5)" and b.outcome == "over"]
        self.assertEqual(len(over_branches), 1)
        grid_prob = context.score_distribution.probability(lambda h, a: h + a > 2.5)
        self.assertAlmostEqual(over_branches[0].probability, grid_prob, places=10)

    # ---- byte-identical gate: restricting bet_markets reproduces old set ----

    def test_restricted_bet_markets_reproduces_old_branch_set(self):
        """When bet_markets=('spf','handicap') only spf/rq branches are emitted."""
        old_params = StrategyParams(bet_markets=("spf", "handicap"))
        by_play = self._branches_by_play(params=old_params)
        plays = set(by_play.keys())
        # Only spf and rq(...) should appear
        for p in plays:
            self.assertTrue(
                p == "spf" or p.startswith("rq("),
                f"Unexpected play {p!r} with restricted bet_markets",
            )
        # Must still have all 3 spf outcomes
        self.assertEqual(len(by_play.get("spf", [])), 3)
        # Must have 3 handicap outcomes
        rq_plays = [p for p in plays if p.startswith("rq(")]
        self.assertEqual(len(rq_plays), 1)
        self.assertEqual(len(by_play[rq_plays[0]]), 3)

    def test_restricted_bet_markets_branch_count_exact(self):
        """Exactly 6 branches (3 spf + 3 rq) when restricted."""
        old_params = StrategyParams(bet_markets=("spf", "handicap"))
        match = _base_match()
        matrix = _full_matrix()
        context = build_probability_context(match, matrix, params=old_params)
        branches = match_branches(match, context)
        self.assertEqual(len(branches), 6, branches)


# ---------------------------------------------------------------------------
# 2. Price lookup + settlement_key tests
# ---------------------------------------------------------------------------

class TestSelectionsFromBranchesNewMarkets(unittest.TestCase):

    def _run(self, params=None):
        match = _base_match()
        matrix = _full_matrix()
        facts = _facts()
        if params is None:
            context = build_probability_context(match, matrix)
        else:
            context = build_probability_context(match, matrix, params=params)
        branches = match_branches(match, context)
        return selections_from_branches(match, matrix, facts, branches, params=params or StrategyParams())

    def test_totals_selection_has_finite_sp(self):
        sels = self._run()
        totals = [s for s in sels if s.play.startswith("totals(")]
        self.assertTrue(len(totals) > 0, "No totals selection produced")
        for sel in totals:
            self.assertTrue(math.isfinite(sel.sp) and sel.sp > 1,
                            f"Bad sp={sel.sp} for {sel.play}:{sel.outcome}")

    def test_totals_selection_sp_from_ask(self):
        """over 2.5 ask=0.59 → decimal_odds = 1/0.59 ≈ 1.6949."""
        sels = self._run()
        over = [s for s in sels if s.play == "totals(2.5)" and s.outcome == "over"]
        self.assertTrue(len(over) > 0)
        self.assertAlmostEqual(over[0].sp, 1.0 / 0.59, places=5)

    def test_totals_selection_has_settlement_key(self):
        sels = self._run()
        totals = [s for s in sels if s.play.startswith("totals(")]
        for sel in totals:
            self.assertIsNotNone(sel.settlement_key,
                                 f"settlement_key None for {sel.play}:{sel.outcome}")
            self.assertEqual(sel.settlement_key.market_type, "totals")
            self.assertIsNotNone(sel.settlement_key.line)

    def test_btts_selection_has_settlement_key_and_finite_sp(self):
        sels = self._run()
        btts = [s for s in sels if s.play == "btts"]
        self.assertTrue(len(btts) >= 1, "No btts selection produced")
        for sel in btts:
            self.assertIsNotNone(sel.settlement_key)
            self.assertEqual(sel.settlement_key.market_type, "btts")
            self.assertTrue(math.isfinite(sel.sp) and sel.sp > 1)

    def test_btts_yes_sp_from_ask(self):
        """btts yes ask=0.56 → decimal_odds = 1/0.56 ≈ 1.7857."""
        sels = self._run()
        yes = [s for s in sels if s.play == "btts" and s.outcome == "yes"]
        self.assertTrue(len(yes) > 0)
        self.assertAlmostEqual(yes[0].sp, 1.0 / 0.56, places=5)

    def test_correct_score_selection_has_settlement_key_and_finite_sp(self):
        sels = self._run()
        cs = [s for s in sels if s.play == "correct_score"]
        self.assertTrue(len(cs) >= 1, "No correct_score selection produced")
        for sel in cs:
            self.assertIsNotNone(sel.settlement_key)
            self.assertEqual(sel.settlement_key.market_type, "correct_score")
            self.assertTrue(math.isfinite(sel.sp) and sel.sp > 1)

    def test_team_total_selection_has_settlement_key_and_finite_sp(self):
        sels = self._run()
        tt = [s for s in sels if s.play.startswith("team_total(")]
        self.assertTrue(len(tt) >= 1, "No team_total selection produced")
        for sel in tt:
            self.assertIsNotNone(sel.settlement_key)
            self.assertEqual(sel.settlement_key.market_type, "team_total")
            self.assertIsNotNone(sel.settlement_key.line)
            self.assertTrue(math.isfinite(sel.sp) and sel.sp > 1)

    def test_no_market_quote_means_no_selection(self):
        """If the matrix has no totals/btts/correct_score quotes the engine
        must not fabricate prices — those branches are simply skipped."""
        match = _base_match()
        # Matrix with only moneyline and handicap
        sparse_matrix = EventMarketMatrix(
            match_id="m01",
            home="Netherlands",
            away="Japan",
            markets=[
                MarketQuote("q1", "winner", "moneyline", "home", 0.62,
                            bid=0.61, ask=0.63, spread=0.02, liquidity=10000),
                MarketQuote("q2", "winner", "moneyline", "draw", 0.22,
                            bid=0.21, ask=0.23, spread=0.02, liquidity=10000),
                MarketQuote("q3", "winner", "moneyline", "away", 0.16,
                            bid=0.15, ask=0.17, spread=0.02, liquidity=10000),
            ],
        )
        facts = _facts()
        context = build_probability_context(match, sparse_matrix)
        branches = match_branches(match, context)
        sels = selections_from_branches(match, sparse_matrix, facts, branches)
        # All selections must be spf or rq — no extra market types
        for sel in sels:
            self.assertTrue(
                sel.play == "spf" or sel.play.startswith("rq("),
                f"Unexpected play={sel.play!r} from sparse matrix",
            )


# ---------------------------------------------------------------------------
# 3. End-to-end settlement grading
# ---------------------------------------------------------------------------

def _mo(h: int, a: int) -> MatchOutcome:
    """Small module-level helper — avoids name collision with MatchOutcome in test class."""
    return MatchOutcome(match_id="m01", home_score=h, away_score=a)


class TestNewMarketSettlementGrade(unittest.TestCase):

    def test_totals_over_25_win_on_3_goals(self):
        key = SettlementKey(market_type="totals", side="over", line=2.5)
        # 2-1 = 3 goals total > 2.5
        self.assertEqual(grade(key, _mo(2, 1)), WIN)

    def test_totals_over_25_loss_on_2_goals(self):
        key = SettlementKey(market_type="totals", side="over", line=2.5)
        # 1-1 = 2 goals < 2.5
        self.assertEqual(grade(key, _mo(1, 1)), LOSS)

    def test_totals_under_25_win_on_2_goals(self):
        key = SettlementKey(market_type="totals", side="under", line=2.5)
        self.assertEqual(grade(key, _mo(1, 1)), WIN)

    def test_correct_score_win(self):
        key = SettlementKey(market_type="correct_score", side="2-1")
        self.assertEqual(grade(key, _mo(2, 1)), WIN)

    def test_correct_score_loss_on_wrong_score(self):
        key = SettlementKey(market_type="correct_score", side="2-1")
        self.assertEqual(grade(key, _mo(1, 1)), LOSS)

    def test_btts_yes_win(self):
        key = SettlementKey(market_type="btts", side="yes")
        self.assertEqual(grade(key, _mo(1, 1)), WIN)

    def test_btts_yes_loss(self):
        key = SettlementKey(market_type="btts", side="yes")
        self.assertEqual(grade(key, _mo(1, 0)), LOSS)

    def test_btts_no_win(self):
        key = SettlementKey(market_type="btts", side="no")
        self.assertEqual(grade(key, _mo(2, 0)), WIN)

    def test_team_total_home_over_win(self):
        key = SettlementKey(market_type="team_total", side="over", line=1.5, entity="home")
        self.assertEqual(grade(key, _mo(2, 0)), WIN)

    def test_team_total_home_over_loss(self):
        key = SettlementKey(market_type="team_total", side="over", line=1.5, entity="home")
        self.assertEqual(grade(key, _mo(1, 2)), LOSS)

    def test_end_to_end_totals_via_selection(self):
        """Full pipeline: match_branches → selections_from_branches → grade."""
        match = _base_match()
        matrix = _full_matrix()
        facts = _facts()
        context = build_probability_context(match, matrix)
        branches = match_branches(match, context)
        sels = selections_from_branches(match, matrix, facts, branches)
        over_sels = [s for s in sels if s.play == "totals(2.5)" and s.outcome == "over"]
        self.assertTrue(len(over_sels) > 0, "No totals(2.5) over selection produced")
        result = grade(over_sels[0], _mo(2, 1))   # 3 goals > 2.5
        self.assertEqual(result, WIN)

    def test_end_to_end_correct_score_via_selection(self):
        match = _base_match()
        matrix = _full_matrix()
        facts = _facts()
        context = build_probability_context(match, matrix)
        branches = match_branches(match, context)
        sels = selections_from_branches(match, matrix, facts, branches)
        cs_21 = [s for s in sels if s.play == "correct_score" and s.outcome == "2-1"]
        self.assertTrue(len(cs_21) > 0, "No correct_score 2-1 selection produced")
        # 2-1 result → WIN
        self.assertEqual(grade(cs_21[0], _mo(2, 1)), WIN)
        # 1-1 result → LOSS
        self.assertEqual(grade(cs_21[0], _mo(1, 1)), LOSS)


# ---------------------------------------------------------------------------
# 4. generate_combos accepts new selection types
# ---------------------------------------------------------------------------

class TestCombosWithNewMarkets(unittest.TestCase):

    def test_generate_combos_includes_new_market_types_cross_match(self):
        """generate_combos must build combos spanning new market types across
        different matches without error."""
        match = _base_match()
        matrix = _full_matrix()
        facts = _facts()
        context = build_probability_context(match, matrix)
        branches = match_branches(match, context)
        sels_m01 = selections_from_branches(match, matrix, facts, branches)

        # Build a second match with a different match_id
        match2 = MatchSP(
            match_id="m02",
            date="2026-06-14",
            home="France",
            away="Germany",
            spf_home=1.70,
            spf_draw=3.60,
            spf_away=4.80,
            handicap=0,
            rq_home=1.70,
            rq_draw=3.60,
            rq_away=4.80,
        )
        matrix2 = EventMarketMatrix(
            match_id="m02",
            home="France",
            away="Germany",
            markets=[
                MarketQuote("r1", "winner", "moneyline", "home", 0.55,
                            bid=0.54, ask=0.56, spread=0.02, liquidity=10000),
                MarketQuote("r2", "winner", "moneyline", "draw", 0.25,
                            bid=0.24, ask=0.26, spread=0.02, liquidity=10000),
                MarketQuote("r3", "winner", "moneyline", "away", 0.20,
                            bid=0.19, ask=0.21, spread=0.02, liquidity=10000),
                MarketQuote("r4", "O/U 2.5", "total_goals", "over 2.5", 0.55,
                            bid=0.54, ask=0.56, line=2.5, spread=0.02, liquidity=9000),
                MarketQuote("r5", "O/U 2.5", "total_goals", "under 2.5", 0.45,
                            bid=0.44, ask=0.46, line=2.5, spread=0.02, liquidity=9000),
            ],
        )
        facts2 = TeamFacts(match_id="m02", source="test",
                           home_summary="strong", away_summary="weak")
        context2 = build_probability_context(match2, matrix2)
        branches2 = match_branches(match2, context2)
        sels_m02 = selections_from_branches(match2, matrix2, facts2, branches2)

        all_sels = sels_m01 + sels_m02
        self.assertTrue(len(all_sels) > 0, "No selections produced")

        # Must not raise
        groups = generate_combos(all_sels, max_size=2)

        # Check that at least some combos (or deleted ones) exist
        total = sum(len(v) for v in groups.values())
        self.assertGreater(total, 0, "generate_combos produced no combos at all")

        # Confirm no same-match legs in A/B/C combos (correlation gate)
        for bucket in ("A", "B", "C"):
            for combo in groups[bucket]:
                match_ids = [s.match_id for s in combo.selections]
                self.assertEqual(
                    len(match_ids), len(set(match_ids)),
                    f"Same-match leg in combo {combo.name}",
                )


# ---------------------------------------------------------------------------
# 5. bet_markets round-trip via params
# ---------------------------------------------------------------------------

class TestBetMarketsParamsRoundTrip(unittest.TestCase):

    def test_default_tuple_preserved_in_to_dict(self):
        p = StrategyParams()
        d = p.to_dict()
        self.assertIn("bet_markets", d)
        self.assertIsInstance(d["bet_markets"], list)  # JSON-serialisable

    def test_from_dict_coerces_list_to_tuple(self):
        p = StrategyParams()
        d = p.to_dict()
        p2 = StrategyParams.from_dict(d)
        self.assertIsInstance(p2.bet_markets, tuple)
        self.assertEqual(p2.bet_markets, p.bet_markets)

    def test_custom_bet_markets_round_trips(self):
        p = StrategyParams(bet_markets=("spf", "handicap"))
        p2 = StrategyParams.from_dict(p.to_dict())
        self.assertEqual(p2.bet_markets, ("spf", "handicap"))


if __name__ == "__main__":
    unittest.main()
