"""Tests for the Phase 1B settlement layer.

Covers:
- SettlementKey construction
- grade() for all score-derivable market types
- VOID paths (void outcome, no poly resolution, unknown market)
- grade_selections() convenience
- settlement_key backfill via selections_from_branches
- adapters/results.py CSV round-trip and JSON cache
"""

from __future__ import annotations

import io
import os
import tempfile
import unittest

from ball_quant.core.handicap import handicap_result
from ball_quant.core.settlement import (
    LOSS,
    VOID,
    WIN,
    MatchOutcome,
    grade,
    grade_selections,
)
from ball_quant.core.value import selections_from_branches
from ball_quant.models import (
    Branch,
    EventMarketMatrix,
    MarketQuote,
    MatchSP,
    Selection,
    SettlementKey,
    TeamFacts,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_outcome(match_id="001", h=0, a=0, void=False, poly=None):
    return MatchOutcome(
        match_id=match_id,
        home_score=h,
        away_score=a,
        void=void,
        poly_resolutions=poly or {},
    )


def make_key(market_type, side, line=None, entity=None):
    return SettlementKey(market_type=market_type, side=side, line=line, entity=entity)


# ---------------------------------------------------------------------------
# SPF grading
# ---------------------------------------------------------------------------

class TestSPFGrade(unittest.TestCase):
    def test_home_win(self):
        key = make_key("spf", "home")
        self.assertEqual(grade(key, make_outcome(h=2, a=1)), WIN)

    def test_home_loss(self):
        key = make_key("spf", "home")
        self.assertEqual(grade(key, make_outcome(h=0, a=2)), LOSS)

    def test_draw(self):
        key = make_key("spf", "draw")
        self.assertEqual(grade(key, make_outcome(h=1, a=1)), WIN)

    def test_away_win(self):
        key = make_key("spf", "away")
        self.assertEqual(grade(key, make_outcome(h=0, a=2)), WIN)

    def test_spf_uses_handicap_result(self):
        # Verify our grading delegates to handicap_result(h,a,0), not a reimplementation.
        for h, a in [(2, 0), (1, 1), (0, 3)]:
            expected_side = handicap_result(h, a, 0)
            key = make_key("spf", expected_side)
            self.assertEqual(grade(key, make_outcome(h=h, a=a)), WIN)


# ---------------------------------------------------------------------------
# Handicap grading
# ---------------------------------------------------------------------------

class TestHandicapGrade(unittest.TestCase):
    def test_home_minus1_cover(self):
        # home -1: adjusted_home = 2-1=1 > 0 => "home"
        key = make_key("handicap", "home", line=-1.0)
        self.assertEqual(grade(key, make_outcome(h=2, a=0)), WIN)

    def test_home_minus1_push(self):
        # home -1: adjusted_home = 1-1=0 == 0 => "draw" (push in handicap sense)
        key = make_key("handicap", "home", line=-1.0)
        self.assertEqual(grade(key, make_outcome(h=1, a=0)), LOSS)

    def test_home_minus1_no_cover(self):
        # home -1: adjusted_home = 1-1=0, then compared to away=1 => away wins
        key = make_key("handicap", "home", line=-1.0)
        self.assertEqual(grade(key, make_outcome(h=1, a=1)), LOSS)

    def test_handicap_draw_side(self):
        # If betting on "draw" outcome of handicap: home -1, score 1-0 => adjusted=0==0 => draw -> WIN
        key = make_key("handicap", "draw", line=-1.0)
        self.assertEqual(grade(key, make_outcome(h=1, a=0)), WIN)

    def test_handicap_missing_line_void(self):
        key = make_key("handicap", "home", line=None)
        self.assertEqual(grade(key, make_outcome(h=2, a=0)), VOID)


# ---------------------------------------------------------------------------
# Correct score grading
# ---------------------------------------------------------------------------

class TestCorrectScoreGrade(unittest.TestCase):
    def test_exact_hit(self):
        key = make_key("correct_score", "2-1")
        self.assertEqual(grade(key, make_outcome(h=2, a=1)), WIN)

    def test_miss(self):
        key = make_key("correct_score", "2-1")
        self.assertEqual(grade(key, make_outcome(h=1, a=1)), LOSS)

    def test_malformed_side_void(self):
        key = make_key("correct_score", "2x1")  # bad separator
        self.assertEqual(grade(key, make_outcome(h=2, a=1)), VOID)


# ---------------------------------------------------------------------------
# Totals grading
# ---------------------------------------------------------------------------

class TestTotalsGrade(unittest.TestCase):
    def test_over_win(self):
        key = make_key("totals", "over", line=2.5)
        self.assertEqual(grade(key, make_outcome(h=2, a=1)), WIN)

    def test_under_win(self):
        key = make_key("totals", "under", line=2.5)
        self.assertEqual(grade(key, make_outcome(h=1, a=1)), WIN)

    def test_over_loss(self):
        key = make_key("totals", "over", line=2.5)
        self.assertEqual(grade(key, make_outcome(h=1, a=1)), LOSS)

    def test_integer_line_push_void(self):
        # 2 goals vs line 2 => push => VOID
        key = make_key("totals", "over", line=2.0)
        self.assertEqual(grade(key, make_outcome(h=1, a=1)), VOID)

    def test_missing_line_void(self):
        key = make_key("totals", "over", line=None)
        self.assertEqual(grade(key, make_outcome(h=2, a=1)), VOID)


# ---------------------------------------------------------------------------
# Team total grading
# ---------------------------------------------------------------------------

class TestTeamTotalGrade(unittest.TestCase):
    def test_home_over_win(self):
        key = make_key("team_total", "over", line=1.5, entity="home")
        self.assertEqual(grade(key, make_outcome(h=2, a=0)), WIN)

    def test_home_over_loss(self):
        key = make_key("team_total", "over", line=1.5, entity="home")
        self.assertEqual(grade(key, make_outcome(h=1, a=2)), LOSS)

    def test_away_under_win(self):
        key = make_key("team_total", "under", line=1.5, entity="away")
        self.assertEqual(grade(key, make_outcome(h=3, a=1)), WIN)

    def test_integer_push_void(self):
        key = make_key("team_total", "over", line=1.0, entity="home")
        self.assertEqual(grade(key, make_outcome(h=1, a=0)), VOID)

    def test_unknown_entity_void(self):
        # entity not "home"/"away" and we have no team-name resolution
        key = make_key("team_total", "over", line=1.5, entity="Netherlands")
        self.assertEqual(grade(key, make_outcome(h=2, a=0)), VOID)


# ---------------------------------------------------------------------------
# BTTS grading
# ---------------------------------------------------------------------------

class TestBttsGrade(unittest.TestCase):
    def test_btts_yes_win(self):
        key = make_key("btts", "yes")
        self.assertEqual(grade(key, make_outcome(h=1, a=1)), WIN)

    def test_btts_yes_loss(self):
        key = make_key("btts", "yes")
        self.assertEqual(grade(key, make_outcome(h=1, a=0)), LOSS)

    def test_btts_no_win(self):
        key = make_key("btts", "no")
        self.assertEqual(grade(key, make_outcome(h=1, a=0)), WIN)

    def test_btts_no_loss(self):
        key = make_key("btts", "no")
        self.assertEqual(grade(key, make_outcome(h=1, a=1)), LOSS)


# ---------------------------------------------------------------------------
# moneyline_not grading
# ---------------------------------------------------------------------------

class TestMoneylineNotGrade(unittest.TestCase):
    def test_not_home_win_on_draw(self):
        # bet: "not home" wins if result is not home
        key = make_key("moneyline_not", "home")
        self.assertEqual(grade(key, make_outcome(h=1, a=1)), WIN)

    def test_not_home_loss_on_home(self):
        key = make_key("moneyline_not", "home")
        self.assertEqual(grade(key, make_outcome(h=2, a=0)), LOSS)


# ---------------------------------------------------------------------------
# Non-score prop via poly_resolutions
# ---------------------------------------------------------------------------

class TestNonScoreProps(unittest.TestCase):
    def test_no_resolution_void(self):
        key = make_key("player_goals", "Mbappe", entity="Mbappe")
        self.assertEqual(grade(key, make_outcome()), VOID)

    def test_poly_resolution_yes_win(self):
        key = make_key("player_goals", "Mbappe", entity="Mbappe")
        outcome = make_outcome(poly={"Mbappe": "YES"})
        self.assertEqual(grade(key, outcome), WIN)

    def test_poly_resolution_no_loss(self):
        key = make_key("player_goals", "Mbappe", entity="Mbappe")
        outcome = make_outcome(poly={"Mbappe": "NO"})
        self.assertEqual(grade(key, outcome), LOSS)

    def test_corner_prop_no_resolution_void(self):
        key = make_key("total_corners", "over", line=9.5, entity="total_corners:over:9.5")
        self.assertEqual(grade(key, make_outcome()), VOID)


# ---------------------------------------------------------------------------
# outcome.void -> all VOID
# ---------------------------------------------------------------------------

class TestVoidOutcome(unittest.TestCase):
    def test_spf_void_match(self):
        key = make_key("spf", "home")
        self.assertEqual(grade(key, make_outcome(h=2, a=1, void=True)), VOID)

    def test_handicap_void_match(self):
        key = make_key("handicap", "home", line=-1.0)
        self.assertEqual(grade(key, make_outcome(h=2, a=1, void=True)), VOID)

    def test_none_key_void(self):
        # Selection with settlement_key=None always VOID
        sel = Selection(
            match_id="001", home="A", away="B", play="spf", outcome="home",
            condition="A wins", probability=0.6, sp=1.8, fair_odds=1.67,
            break_even=0.56, edge=0.08, kelly=0.04, confidence=0.7,
            risk_label="价值保留",
        )
        self.assertIsNone(sel.settlement_key)
        self.assertEqual(grade(sel, make_outcome(h=2, a=1)), VOID)


# ---------------------------------------------------------------------------
# grade() accepting Selection directly
# ---------------------------------------------------------------------------

class TestGradeFromSelection(unittest.TestCase):
    def _make_sel(self, market_type, side, line=None):
        key = SettlementKey(market_type=market_type, side=side, line=line)
        return Selection(
            match_id="001", home="A", away="B", play="spf", outcome="home",
            condition="", probability=0.6, sp=1.8, fair_odds=1.67,
            break_even=0.56, edge=0.08, kelly=0.04, confidence=0.7,
            risk_label="价值保留", settlement_key=key,
        )

    def test_win_via_selection(self):
        sel = self._make_sel("spf", "home")
        self.assertEqual(grade(sel, make_outcome(h=2, a=0)), WIN)

    def test_grade_selections_list(self):
        sels = [self._make_sel("spf", "home"), self._make_sel("spf", "away")]
        results = grade_selections(sels, make_outcome(h=2, a=0))
        self.assertEqual(results[0][1], WIN)
        self.assertEqual(results[1][1], LOSS)


# ---------------------------------------------------------------------------
# settlement_key backfill via selections_from_branches
# ---------------------------------------------------------------------------

class TestBackfill(unittest.TestCase):
    def _sample_match(self):
        return MatchSP(
            match_id="001",
            date="2026-06-14",
            home="Netherlands",
            away="Japan",
            spf_home=1.55,
            spf_draw=3.9,
            spf_away=5.6,
            handicap=-1,
            rq_home=2.78,
            rq_draw=3.55,
            rq_away=2.05,
        )

    def _sample_matrix(self):
        return EventMarketMatrix(
            match_id="001",
            home="Netherlands",
            away="Japan",
            markets=[
                MarketQuote("m1", "winner", "moneyline", "home", 0.62, spread=0.02, liquidity=10000),
                MarketQuote("m1", "winner", "moneyline", "draw", 0.22, spread=0.02, liquidity=10000),
                MarketQuote("m1", "winner", "moneyline", "away", 0.16, spread=0.02, liquidity=10000),
            ],
        )

    def _sample_facts(self):
        return TeamFacts(
            match_id="001",
            source="test",
            home_summary="Strong attack",
            away_summary="Defensive",
        )

    def test_spf_branches_get_settlement_key(self):
        match = self._sample_match()
        matrix = self._sample_matrix()
        facts = self._sample_facts()
        branches = [
            Branch(match_id="001", play="spf", outcome="home",
                   condition="NL wins", probability=0.62, source="test"),
        ]
        sels = selections_from_branches(match, matrix, facts, branches)
        self.assertTrue(len(sels) > 0, "Expected at least one selection")
        sel = sels[0]
        self.assertIsNotNone(sel.settlement_key)
        self.assertEqual(sel.settlement_key.market_type, "spf")
        self.assertEqual(sel.settlement_key.side, "home")

    def test_rq_branches_get_handicap_key(self):
        match = self._sample_match()
        matrix = self._sample_matrix()
        facts = self._sample_facts()
        branches = [
            Branch(match_id="001", play="rq(-1)", outcome="home",
                   condition="NL wins by 2+", probability=0.34, source="test"),
        ]
        sels = selections_from_branches(match, matrix, facts, branches)
        self.assertTrue(len(sels) > 0, "Expected at least one selection")
        sel = sels[0]
        self.assertIsNotNone(sel.settlement_key)
        self.assertEqual(sel.settlement_key.market_type, "handicap")
        self.assertEqual(sel.settlement_key.line, -1.0)
        self.assertEqual(sel.settlement_key.side, "home")

    def test_all_score_derivable_branches_have_key(self):
        """All branches produced by match_branches have a non-None settlement_key."""
        from ball_quant.core.probability import build_probability_context, match_branches
        match = self._sample_match()
        matrix = self._sample_matrix()
        facts = self._sample_facts()
        context = build_probability_context(match, matrix)
        branches = match_branches(match, context)
        sels = selections_from_branches(match, matrix, facts, branches)
        for sel in sels:
            self.assertIsNotNone(
                sel.settlement_key,
                f"settlement_key is None for {sel.play}:{sel.outcome}",
            )


# ---------------------------------------------------------------------------
# adapters/results.py
# ---------------------------------------------------------------------------

class TestResultsAdapter(unittest.TestCase):
    def _write_csv(self, content: str) -> str:
        fd, path = tempfile.mkstemp(suffix=".csv")
        os.close(fd)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return path

    def test_load_results_basic(self):
        from ball_quant.adapters.results import load_results
        path = self._write_csv("match_id,home_score,away_score\n001,2,1\n002,0,0\n")
        outcomes = load_results(path)
        os.unlink(path)
        self.assertEqual(outcomes["001"].home_score, 2)
        self.assertEqual(outcomes["001"].away_score, 1)
        self.assertFalse(outcomes["001"].void)
        self.assertEqual(outcomes["002"].home_score, 0)

    def test_load_results_void_flag(self):
        from ball_quant.adapters.results import load_results
        path = self._write_csv("match_id,home_score,away_score,void\n001,2,1,true\n")
        outcomes = load_results(path)
        os.unlink(path)
        self.assertTrue(outcomes["001"].void)

    def test_load_results_bad_row_raises(self):
        from ball_quant.adapters.results import load_results
        path = self._write_csv("match_id,home_score,away_score\n001,two,1\n")
        with self.assertRaises(ValueError):
            load_results(path)
        os.unlink(path)

    def test_json_round_trip(self):
        from ball_quant.adapters.results import load_results_json, save_results
        outcomes = {
            "001": MatchOutcome(match_id="001", home_score=3, away_score=2),
            "002": MatchOutcome(match_id="002", home_score=0, away_score=1, void=True),
        }
        fd, path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        save_results(outcomes, path)
        loaded = load_results_json(path)
        os.unlink(path)
        self.assertEqual(loaded["001"].home_score, 3)
        self.assertEqual(loaded["002"].void, True)


if __name__ == "__main__":
    unittest.main()
