import unittest

from ball_quant.adapters.polymarket import market_to_quotes
from ball_quant.core.analysis import analyze_match
from ball_quant.core.combo import generate_combos
from ball_quant.core.probability import (
    build_probability_context,
    normalized_moneyline_probabilities,
    probability_for_handicap,
    total_goal_hint,
)
from ball_quant.core.staking import allocate_stakes
from ball_quant.models import Combo, EventMarketMatrix, MarketQuote, MatchSP, Selection, TeamFacts


def sample_match():
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


def sample_matrix():
    return EventMarketMatrix(
        match_id="001",
        home="Netherlands",
        away="Japan",
        markets=[
            MarketQuote("m1", "winner", "moneyline", "home", 0.62, spread=0.02, liquidity=10000),
            MarketQuote("m1", "winner", "moneyline", "draw", 0.22, spread=0.02, liquidity=10000),
            MarketQuote("m1", "winner", "moneyline", "away", 0.16, spread=0.02, liquidity=10000),
            MarketQuote("m2", "Netherlands -1.5", "handicap", "Netherlands -1.5", 0.34, spread=0.02, liquidity=8000),
            MarketQuote("m3", "Japan +0.5", "handicap", "Japan +0.5", 0.38, spread=0.02, liquidity=8000),
        ],
    )


class ProbabilityAndComboTest(unittest.TestCase):
    def test_handicap_branch_from_polymarket_lines(self):
        match = sample_match()
        context = build_probability_context(match, sample_matrix())
        self.assertAlmostEqual(
            probability_for_handicap(context, -1, "home", match.home, match.away),
            0.34,
        )
        self.assertAlmostEqual(
            probability_for_handicap(context, -1, "away", match.home, match.away),
            0.38,
        )
        self.assertAlmostEqual(
            probability_for_handicap(context, -1, "draw", match.home, match.away),
            0.28,
        )

    def test_sports_spread_uses_outcome_line_not_question_line(self):
        match = sample_match()
        markets = [
            MarketQuote("m1", "winner", "moneyline", "home", 0.475, spread=0.01, liquidity=20000),
            MarketQuote("m1", "winner", "moneyline", "draw", 0.275, spread=0.01, liquidity=20000),
            MarketQuote("m1", "winner", "moneyline", "away", 0.265, spread=0.01, liquidity=20000),
        ]
        markets.extend(
            market_to_quotes(
                {
                    "id": "spread1",
                    "question": "Spread: Netherlands (-1.5)",
                    "sportsMarketType": "spreads",
                    "outcomes": ["Netherlands", "Japan"],
                    "outcomePrices": ["0.235", "0.765"],
                    "line": -1.5,
                    "groupItemTitle": "Netherlands (-1.5)",
                    "bestBid": 0.23,
                    "bestAsk": 0.24,
                },
                "Netherlands",
                "Japan",
            )
        )
        context = build_probability_context(
            match,
            EventMarketMatrix(match_id="001", home=match.home, away=match.away, markets=markets),
        )
        self.assertAlmostEqual(
            probability_for_handicap(context, -1, "home", match.home, match.away),
            0.235,
        )
        self.assertAlmostEqual(
            probability_for_handicap(context, -1, "away", match.home, match.away),
            (0.275 + 0.265) / (0.475 + 0.275 + 0.265),
        )
        self.assertAlmostEqual(
            probability_for_handicap(context, -1, "draw", match.home, match.away),
            1.0 - 0.235 - (0.275 + 0.265) / (0.475 + 0.275 + 0.265),
        )

    def test_moneyline_probabilities_are_normalized_before_branch_use(self):
        matrix = EventMarketMatrix(
            match_id="001",
            home="Netherlands",
            away="Japan",
            markets=[
                MarketQuote("m1", "winner", "moneyline", "home", 0.475),
                MarketQuote("m2", "draw", "moneyline", "draw", 0.275),
                MarketQuote("m3", "away", "moneyline", "away", 0.265),
            ],
        )
        probs = normalized_moneyline_probabilities(matrix)
        self.assertIsNotNone(probs)
        self.assertAlmostEqual(sum((probs or {}).values()), 1.0)
        self.assertAlmostEqual((probs or {})["home"], 0.475 / 1.015)

    def test_closed_moneyline_is_not_treated_as_live_probability(self):
        matrix = EventMarketMatrix(
            match_id="001",
            home="Australia",
            away="Türkiye",
            markets=[
                MarketQuote("m1", "winner", "moneyline", "home", 1.0, closed=True, accepting_orders=False),
                MarketQuote("m2", "draw", "moneyline", "draw", 0.0, closed=True, accepting_orders=False),
                MarketQuote("m3", "away", "moneyline", "away", 0.0, closed=True, accepting_orders=False),
            ],
        )
        self.assertIsNone(normalized_moneyline_probabilities(matrix))

    def test_team_total_uses_market_subject_not_first_team_in_question(self):
        quotes = market_to_quotes(
            {
                "id": "tt1",
                "question": "Netherlands vs. Japan: Japan O/U 0.5",
                "sportsMarketType": "soccer_team_totals",
                "outcomes": ["Over", "Under"],
                "outcomePrices": ["0.665", "0.335"],
                "line": 0.5,
                "groupItemTitle": "Japan O/U 0.5",
            },
            "Netherlands",
            "Japan",
        )
        self.assertEqual(quotes[0].outcome, "Japan over 0.5")
        self.assertEqual(quotes[0].entity, "Japan")

    def test_total_goal_hint_uses_balanced_line_not_average_of_all_alt_lines(self):
        markets = []
        for line, over, under in [
            (0.5, 0.925, 0.075),
            (1.5, 0.735, 0.265),
            (2.5, 0.475, 0.525),
            (3.5, 0.265, 0.735),
            (8.5, 0.0025, 0.9975),
        ]:
            markets.append(MarketQuote(f"o{line}", "O/U", "total_goals", f"over {line}", over, line=line))
            markets.append(MarketQuote(f"u{line}", "O/U", "total_goals", f"under {line}", under, line=line))
        matrix = EventMarketMatrix(match_id="001", home="Netherlands", away="Japan", markets=markets)
        self.assertAlmostEqual(total_goal_hint(matrix) or 0.0, 2.4825)

    def test_combo_generation_marks_low_probability_deleted(self):
        facts = TeamFacts("001", "test", "home", "away")
        analysis = analyze_match(sample_match(), sample_matrix(), facts)
        groups = generate_combos(analysis.selections, max_size=2)
        self.assertIn("deleted", groups)
        self.assertTrue(all(combo.probability >= 0.08 for combo in groups["A"]))

    def test_score_distribution_uses_totals_and_btts_constraints(self):
        match = sample_match()
        matrix = EventMarketMatrix(
            match_id="001",
            home="Netherlands",
            away="Japan",
            markets=[
                MarketQuote("m1", "winner", "moneyline", "home", 0.50, spread=0.02, liquidity=10000),
                MarketQuote("m1", "winner", "moneyline", "draw", 0.25, spread=0.02, liquidity=10000),
                MarketQuote("m1", "winner", "moneyline", "away", 0.25, spread=0.02, liquidity=10000),
                MarketQuote("m2", "Total goals Over/Under 2.5", "total_goals", "Over", 0.72, spread=0.02, liquidity=10000),
                MarketQuote("m3", "Both teams to score", "btts", "yes", 0.66, spread=0.02, liquidity=10000),
            ],
        )
        context = build_probability_context(match, matrix)
        over_25 = context.score_distribution.probability(lambda h, a: h + a > 2.5)
        btts = context.score_distribution.probability(lambda h, a: h > 0 and a > 0)
        self.assertGreater(over_25, 0.62)
        self.assertGreater(btts, 0.56)

    def test_primary_moneyline_dominates_shape_constraints(self):
        match = sample_match()
        matrix = EventMarketMatrix(
            match_id="001",
            home="Netherlands",
            away="Japan",
            markets=[
                MarketQuote("m1", "winner", "moneyline", "home", 0.70, spread=0.02, liquidity=20000),
                MarketQuote("m1", "winner", "moneyline", "draw", 0.20, spread=0.02, liquidity=20000),
                MarketQuote("m1", "winner", "moneyline", "away", 0.10, spread=0.02, liquidity=20000),
                MarketQuote("m2", "Total goals Over/Under 0.5", "total_goals", "Under", 0.80, spread=0.02, liquidity=20000),
                MarketQuote("m3", "Both teams to score", "btts", "no", 0.90, spread=0.02, liquidity=20000),
            ],
        )
        context = build_probability_context(match, matrix)
        self.assertAlmostEqual(context.score_distribution.probability(lambda h, a: h > a), 0.70, delta=0.05)
        self.assertEqual(context.matrix.implied_probability("moneyline", "home"), 0.70)

    def test_negative_ev_combo_is_deleted_and_not_staked(self):
        first = Selection(
            match_id="001",
            home="A",
            away="B",
            play="spf",
            outcome="home",
            condition="A wins",
            probability=0.60,
            sp=1.40,
            fair_odds=1.67,
            break_even=0.71,
            edge=-0.16,
            kelly=0.0,
            confidence=0.8,
            risk_label="赔率不足",
        )
        second = Selection(
            match_id="002",
            home="C",
            away="D",
            play="spf",
            outcome="home",
            condition="C wins",
            probability=0.60,
            sp=1.40,
            fair_odds=1.67,
            break_even=0.71,
            edge=-0.16,
            kelly=0.0,
            confidence=0.8,
            risk_label="赔率不足",
        )
        groups = generate_combos([first, second], max_size=2)
        self.assertEqual(len(groups["A"]), 0)
        self.assertEqual(groups["deleted"][0].deletion_reason, "组合EV不为正，概率与赔率不匹配")
        self.assertEqual(allocate_stakes(groups, 200), [])

    def test_combo_groups_do_not_reuse_same_selection(self):
        selections = [
            Selection(
                match_id=f"00{i}",
                home=f"H{i}",
                away=f"A{i}",
                play="spf",
                outcome="home",
                condition="home wins",
                probability=0.62,
                sp=1.9,
                fair_odds=1.61,
                break_even=0.53,
                edge=0.178,
                kelly=0.20,
                confidence=0.75,
                risk_label="价值保留",
            )
            for i in range(1, 6)
        ]
        groups = generate_combos(selections, max_size=2)
        used = []
        for key in ("A", "B", "C"):
            for combo in groups[key]:
                used.extend(selection.key for selection in combo.selections)
        self.assertEqual(len(used), len(set(used)))

    def test_sub_eight_percent_combo_only_allowed_in_lottery_bucket(self):
        first = Selection(
            match_id="001",
            home="A",
            away="B",
            play="spf",
            outcome="home",
            condition="A wins",
            probability=0.25,
            sp=4.0,
            fair_odds=4.0,
            break_even=0.25,
            edge=0.0,
            kelly=0.0,
            confidence=0.65,
            risk_label="观察",
        )
        second = Selection(
            match_id="002",
            home="C",
            away="D",
            play="spf",
            outcome="home",
            condition="C wins",
            probability=0.24,
            sp=4.8,
            fair_odds=4.17,
            break_even=0.21,
            edge=0.152,
            kelly=0.05,
            confidence=0.65,
            risk_label="价值保留",
        )
        groups = generate_combos([first, second], max_size=2)
        self.assertFalse(any(combo.probability < 0.08 for combo in groups["A"] + groups["B"]))


if __name__ == "__main__":
    unittest.main()
