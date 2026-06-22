import unittest

from ball_quant.adapters.polymarket import market_to_quotes
from ball_quant.core.analysis import analyze_match
from ball_quant.core.combo import correlation_discount, generate_combos
from ball_quant.core.probability import (
    build_probability_context,
    normalized_moneyline_probabilities,
    prior_lambdas,
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

    # ------------------------------------------------------------------
    # FIX 1 — prior_lambdas lambda-sum conservation
    # ------------------------------------------------------------------
    def test_prior_lambdas_sum_equals_base_total_in_normal_regime(self):
        """home_lambda + away_lambda must equal base_total exactly when
        base_total >= 2 * lambda_floor (the normal, non-degenerate case).
        Before the fix the floor was applied independently so the sum could
        drift above base_total."""
        # Use a matrix that has a two-sided total-goals market at 2.5 so the
        # hint is well above 2 * lambda_floor (2 * 0.25 = 0.50).
        markets = [
            MarketQuote("o25", "O/U", "total_goals", "over 2.5", 0.50, line=2.5),
            MarketQuote("u25", "O/U", "total_goals", "under 2.5", 0.50, line=2.5),
            MarketQuote("m1", "winner", "moneyline", "home", 0.55, spread=0.01, liquidity=5000),
            MarketQuote("m1", "winner", "moneyline", "draw", 0.25, spread=0.01, liquidity=5000),
            MarketQuote("m1", "winner", "moneyline", "away", 0.20, spread=0.01, liquidity=5000),
        ]
        matrix = EventMarketMatrix(match_id="001", home="A", away="B", markets=markets)
        home_l, away_l = prior_lambdas(matrix)
        base_total = total_goal_hint(matrix)
        self.assertIsNotNone(base_total)
        # Sum must equal base_total within floating-point rounding.
        self.assertAlmostEqual(home_l + away_l, base_total, places=10)

    def test_prior_lambdas_sum_preserved_across_several_home_shares(self):
        """For a range of moneyline skews (different home_share values) the
        sum must remain == base_total as long as both legs stay above the floor."""
        from ball_quant.core.params import StrategyParams
        base_total = 2.50  # well above 2 * 0.25
        params = StrategyParams()
        for home_share_raw in [0.30, 0.40, 0.50, 0.60, 0.70]:
            # Manufacture a matrix whose moneyline implies this home_share.
            # diff = home_share - 0.5, so home_win - away_win = diff / home_share_coeff
            diff = home_share_raw - 0.5
            delta = diff / params.home_share_coeff
            # home_win and away_win don't need to sum to 1; they get normalised.
            home_win_raw = 0.5 + delta / 2
            away_win_raw = 0.5 - delta / 2
            markets = [
                MarketQuote("ov", "O/U", "total_goals", "over 2.5", 0.50, line=2.5),
                MarketQuote("un", "O/U", "total_goals", "under 2.5", 0.50, line=2.5),
                MarketQuote("m1", "w", "moneyline", "home", max(0.05, min(0.90, home_win_raw)), spread=0.01, liquidity=5000),
                MarketQuote("m1", "w", "moneyline", "draw", 0.25, spread=0.01, liquidity=5000),
                MarketQuote("m1", "w", "moneyline", "away", max(0.05, min(0.90, away_win_raw)), spread=0.01, liquidity=5000),
            ]
            matrix = EventMarketMatrix(match_id="001", home="A", away="B", markets=markets)
            home_l, away_l = prior_lambdas(matrix)
            hint = total_goal_hint(matrix)
            self.assertIsNotNone(hint)
            # Only assert sum == base_total when both legs are above the floor.
            if home_l > params.lambda_floor and away_l > params.lambda_floor:
                self.assertAlmostEqual(
                    home_l + away_l,
                    hint,
                    places=10,
                    msg=f"Sum drifted for home_share_raw={home_share_raw}",
                )

    def test_prior_lambdas_floor_only_fires_on_degenerate_small_total(self):
        """When base_total < 2*lambda_floor the floor guard may push the sum
        above base_total — that is explicitly allowed (Poisson needs lambda>0)
        but must not happen in the normal regime."""
        from ball_quant.core.params import StrategyParams
        params = StrategyParams()
        threshold = 2 * params.lambda_floor  # 0.50 with defaults
        # Degenerate: base_total=0.30 < threshold
        markets_degen = [
            MarketQuote("ov", "O/U", "total_goals", "over 0.5", 0.10, line=0.5),
            MarketQuote("un", "O/U", "total_goals", "under 0.5", 0.90, line=0.5),
        ]
        matrix_degen = EventMarketMatrix(match_id="x", home="A", away="B", markets=markets_degen)
        hint_degen = total_goal_hint(matrix_degen)
        if hint_degen is not None and hint_degen < threshold:
            home_l, away_l = prior_lambdas(matrix_degen)
            # Floor may inflate sum — that is OK.
            self.assertGreaterEqual(home_l, params.lambda_floor)
            self.assertGreaterEqual(away_l, params.lambda_floor)
        # Normal: base_total=2.5 >= threshold → sum must be exact.
        markets_normal = [
            MarketQuote("ov", "O/U", "total_goals", "over 2.5", 0.50, line=2.5),
            MarketQuote("un", "O/U", "total_goals", "under 2.5", 0.50, line=2.5),
        ]
        matrix_normal = EventMarketMatrix(match_id="y", home="A", away="B", markets=markets_normal)
        home_l, away_l = prior_lambdas(matrix_normal)
        hint_normal = total_goal_hint(matrix_normal)
        self.assertAlmostEqual(home_l + away_l, hint_normal, places=10)

    # ------------------------------------------------------------------
    # FIX 2 — total_goal_hint one-sided devig
    # ------------------------------------------------------------------
    def test_total_goal_hint_one_sided_over_is_debiased(self):
        """A single over quote at a vig-inflated price (e.g. 0.56 raw implied
        prob at a true 50/50 line) must yield a de-biased hint that is closer
        to 0.50 than the raw price, not the raw price itself.
        Before the fix p_over was used raw."""
        # True balanced line is 2.5; over is priced at 0.56 (6-point overround
        # absorbed on the over side only — one-sided market).
        markets = [
            MarketQuote("o25", "O/U", "total_goals", "over 2.5", 0.56, line=2.5),
        ]
        matrix = EventMarketMatrix(match_id="001", home="A", away="B", markets=markets)
        hint = total_goal_hint(matrix)
        self.assertIsNotNone(hint)
        # The raw-price hint at line 2.5 with p_over=0.56 would be:
        #   2.5 + (0.56 - 0.50) * nudge  →  2.5 + 0.06 * 0.7 = 2.542
        # The de-biased hint must be strictly less than this (shrunk toward 2.5).
        raw_hint = 2.5 + (0.56 - 0.50) * 0.7  # 2.542
        self.assertLess(hint, raw_hint)

    def test_total_goal_hint_one_sided_under_is_debiased(self):
        """Single under quote at vig-inflated price must yield a de-biased
        p_over that is closer to 0.50 than the raw complement (1 - under)."""
        markets = [
            MarketQuote("u25", "O/U", "total_goals", "under 2.5", 0.56, line=2.5),
        ]
        matrix = EventMarketMatrix(match_id="001", home="A", away="B", markets=markets)
        hint = total_goal_hint(matrix)
        self.assertIsNotNone(hint)
        # Raw: p_over = 1 - 0.56 = 0.44, raw_hint = 2.5 + (0.44 - 0.50)*0.7 = 2.458
        raw_hint = 2.5 + (1.0 - 0.56 - 0.50) * 0.7  # 2.458
        self.assertGreater(hint, raw_hint)  # de-biased toward 2.5

    def test_total_goal_hint_two_sided_path_unchanged(self):
        """Two-sided devig must produce the same result as before; this
        regression-pins the existing behavior."""
        markets = [
            MarketQuote("o25", "O/U", "total_goals", "over 2.5", 0.54, line=2.5),
            MarketQuote("u25", "O/U", "total_goals", "under 2.5", 0.54, line=2.5),
        ]
        matrix = EventMarketMatrix(match_id="001", home="A", away="B", markets=markets)
        hint = total_goal_hint(matrix)
        # Two-sided devig: p_over = 0.54 / (0.54+0.54) = 0.50 → hint = 2.5
        self.assertAlmostEqual(hint, 2.5, places=6)

    # ------------------------------------------------------------------
    # FIX 3 — combo correlation_discount is a safety haircut, not signed corr
    # ------------------------------------------------------------------
    def test_correlation_discount_is_le_one_always(self):
        """discount must be in (0, 1] regardless of how many legs or tags."""
        sel = Selection(
            match_id="001", home="A", away="B", play="spf", outcome="home",
            condition="A wins", probability=0.60, sp=1.80, fair_odds=1.67,
            break_even=0.56, edge=0.07, kelly=0.12, confidence=0.80,
            risk_label="价值保留",
        )
        sel_low = Selection(
            match_id="002", home="C", away="D", play="spf", outcome="home",
            condition="C wins", probability=0.40, sp=2.60, fair_odds=2.50,
            break_even=0.38, edge=0.06, kelly=0.05, confidence=0.45,
            risk_label="观察",
        )
        for legs in [[sel], [sel, sel_low], [sel, sel, sel_low]]:
            d = correlation_discount(legs)
            self.assertGreater(d, 0.0)
            self.assertLessEqual(d, 1.0)

    def test_correlation_discount_single_leg_is_one(self):
        sel = Selection(
            match_id="001", home="A", away="B", play="spf", outcome="home",
            condition="A wins", probability=0.60, sp=1.80, fair_odds=1.67,
            break_even=0.56, edge=0.07, kelly=0.12, confidence=0.80,
            risk_label="价值保留",
        )
        self.assertEqual(correlation_discount([sel]), 1.0)

    def test_correlation_discount_params_tunable(self):
        """corr_* values must come from StrategyParams so they are adjustable."""
        from ball_quant.core.params import StrategyParams
        sel = Selection(
            match_id="001", home="A", away="B", play="spf", outcome="home",
            condition="A wins", probability=0.60, sp=1.80, fair_odds=1.67,
            break_even=0.56, edge=0.07, kelly=0.12, confidence=0.80,
            risk_label="价值保留",
        )
        sel2 = Selection(
            match_id="002", home="C", away="D", play="spf", outcome="home",
            condition="C wins", probability=0.55, sp=2.00, fair_odds=1.82,
            break_even=0.50, edge=0.10, kelly=0.15, confidence=0.75,
            risk_label="价值保留",
        )
        params_tight = StrategyParams(corr_base=0.80)
        params_loose = StrategyParams(corr_base=0.99)
        d_tight = correlation_discount([sel, sel2], params=params_tight)
        d_loose = correlation_discount([sel, sel2], params=params_loose)
        self.assertLess(d_tight, d_loose)


if __name__ == "__main__":
    unittest.main()
