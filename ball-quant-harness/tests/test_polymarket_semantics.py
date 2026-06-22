import json
import unittest

from ball_quant.adapters.polymarket import (
    classify_market,
    event_to_quotes,
    extract_sports_event_from_next_data,
    infer_match_teams,
    matrix_to_inventory,
    market_to_quotes,
)
from ball_quant.core.probability import parse_score
from ball_quant.models import EventMarketMatrix, normalize_key


class PolymarketSemanticsTest(unittest.TestCase):
    def test_binary_win_market_no_is_not_away_win(self):
        market = {
            "id": "m1",
            "question": "Will Germany win against Japan?",
            "outcomes": '["Yes","No"]',
            "outcomePrices": '["0.62","0.38"]',
        }
        quotes = market_to_quotes(market, "Germany", "Japan")
        self.assertEqual(quotes[0].outcome, "home")
        self.assertEqual(quotes[1].outcome, "not_home")

    def test_binary_draw_market_yes_maps_to_draw_first(self):
        market = {
            "id": "m2",
            "question": "Will Germany vs Japan be a draw?",
            "outcomes": '["Yes","No"]',
            "outcomePrices": '["0.24","0.76"]',
        }
        quotes = market_to_quotes(market, "Germany", "Japan")
        self.assertEqual(quotes[0].outcome, "draw")
        self.assertEqual(quotes[1].outcome, "not_draw")

    def test_binary_handicap_market_labels_positive_and_complement(self):
        market = {
            "id": "m3",
            "question": "Germany -1.5 handicap vs Japan",
            "outcomes": '["Yes","No"]',
            "outcomePrices": '["0.41","0.59"]',
        }
        quotes = market_to_quotes(market, "Germany", "Japan")
        self.assertEqual(quotes[0].outcome, "Germany -1.5")
        self.assertEqual(quotes[1].outcome, "not:Germany -1.5")
        self.assertFalse(quotes[0].is_complement)
        self.assertTrue(quotes[1].is_complement)

    def test_date_is_not_classified_as_correct_score(self):
        market = {
            "id": "m4",
            "question": "Will Germany win on 2026-06-14?",
            "outcomes": '["Yes","No"]',
            "outcomePrices": '["0.94","0.06"]',
        }
        quotes = market_to_quotes(market, "Germany", "Curaçao")
        self.assertEqual(quotes[0].category, "moneyline")
        self.assertEqual(quotes[0].outcome, "home")
        self.assertIsNone(parse_score("Will Germany win on 2026-06-14?"))

    def test_world_cup_long_term_market_classification(self):
        self.assertEqual(classify_market("World Cup Winner"), "tournament_winner")
        self.assertEqual(classify_market("World Cup: Nation to Reach Final"), "stage_advancement")
        self.assertEqual(classify_market("World Cup: Team to advance to Knockout Stages"), "group_advancement")
        self.assertEqual(classify_market("Will Kylian Mbappe win?", event_title="World Cup: Golden Boot Winner"), "player_award")
        self.assertEqual(classify_market("Will USA be eliminated in the Round of 16?", event_title="World Cup: USA Stage of Elimination"), "stage_elimination")
        self.assertEqual(classify_market("Will Germany finish second place in Group E?"), "group_position")
        self.assertEqual(classify_market("Will Lionel Messi play in the World Cup?"), "player_future")
        self.assertEqual(classify_market("World Cup Goals H2H: Messi vs. Ronaldo"), "player_h2h")
        self.assertEqual(classify_market("Will Ronaldo Cry at the World Cup?"), "culture_future")
        self.assertEqual(classify_market("Will Any Team Participate in 3+ Penalty Shootouts in the Knockout Phase?"), "record_future")
        self.assertEqual(classify_market("Will 5+ matches go to extra time during the 2026 FIFA World Cup?"), "record_future")
        self.assertEqual(classify_market("Will Memphis Depay be in Netherlands's Starting 11?"), "starting_lineup")
        self.assertEqual(classify_market("Will Lionel Messi win the Golden Ball?", event_title="World Cup Golden Ball"), "player_award")
        self.assertEqual(classify_market("Will Xavi Simons record the most assists at the 2026 FIFA World Cup?"), "player_award")
        self.assertEqual(classify_market("World Cup Fair Play Award"), "team_prop")

    def test_match_team_inference_strips_market_suffixes(self):
        self.assertEqual(
            infer_match_teams("Algeria vs. Austria - More Markets"),
            ("Algeria", "Austria"),
        )
        self.assertEqual(
            infer_match_teams("Netherlands vs. Japan - Total Corners"),
            ("Netherlands", "Japan"),
        )

    def test_accent_insensitive_team_match(self):
        self.assertEqual(normalize_key("Curaçao"), normalize_key("Curacao"))

    def test_inventory_contains_normalized_quote_rows(self):
        event = {
            "id": "e1",
            "slug": "fifwc-ger-kor-2026-06-14",
            "title": "Germany vs. Curaçao",
            "eventDate": "2026-06-14",
            "startTime": "2026-06-14T17:00:00Z",
            "closed": False,
            "markets": [
                {
                    "id": "m1",
                    "question": "Will Germany win on 2026-06-14?",
                    "outcomes": '["Yes","No"]',
                    "outcomePrices": '["0.94","0.06"]',
                    "liquidity": "1000",
                    "volume": "5000",
                }
            ],
        }
        matrix = EventMarketMatrix(
            match_id="e1",
            home="Germany",
            away="Curaçao",
            event_id="e1",
            event_slug=event["slug"],
            markets=event_to_quotes(event, "Germany", "Curaçao"),
            raw_event=event,
        )
        inventory = matrix_to_inventory(matrix)
        self.assertEqual(inventory["category_counts"]["moneyline"], 2)
        self.assertEqual(inventory["quotes"][0]["fair_odds"], 1 / 0.94)
        self.assertEqual(inventory["quotes"][0]["causal_layer"], "same_match_result")
        self.assertEqual(inventory["quotes"][0]["horizon"], "today_match")
        self.assertEqual(inventory["quotes"][0]["model_weight"], 1.0)
        self.assertEqual(inventory["polymarket_date"], "2026-06-14")
        self.assertEqual(inventory["start_time_utc"], "2026-06-14T17:00:00Z")
        self.assertEqual(inventory["event_closed"], False)
        self.assertEqual(inventory["quotes"][0]["polymarket_date"], "2026-06-14")

    def test_world_cup_future_labels_and_downweights(self):
        market = {
            "id": "f1",
            "question": "Will USA be eliminated in the Round of 16?",
            "outcomes": '["Yes","No"]',
            "outcomePrices": '["0.31","0.69"]',
        }
        quotes = market_to_quotes(market, "", "", event_title="World Cup: USA Stage of Elimination")
        self.assertEqual(quotes[0].category, "stage_elimination")
        self.assertEqual(quotes[0].outcome, "USA Round of 16")
        self.assertEqual(quotes[0].causal_layer, "tournament_path")
        self.assertEqual(quotes[0].horizon, "medium_future")
        self.assertLess(quotes[0].model_weight or 0.0, 0.25)

    def test_more_player_prop_sports_types(self):
        shots_target = market_to_quotes(
            {
                "id": "p1",
                "question": "Ayoub El Kaabi: 1+ shots on target",
                "sportsMarketType": "soccer_player_shots_on_target",
                "outcomes": '["Yes","No"]',
                "outcomePrices": '["0.44","0.56"]',
                "line": 0.5,
            },
            "Brazil",
            "Morocco",
        )
        self.assertEqual(shots_target[0].category, "player_shots_on_target")
        self.assertEqual(shots_target[0].outcome, "Ayoub El Kaabi shots_on_target over 0.5")
        self.assertEqual(shots_target[0].causal_layer, "player_prop")

        saves = market_to_quotes(
            {
                "id": "p2",
                "question": "Alisson: 4+ saves",
                "sportsMarketType": "soccer_player_goalkeeper_saves",
                "outcomes": '["Yes","No"]',
                "outcomePrices": '["0.21","0.79"]',
                "line": 3.5,
            },
            "Brazil",
            "Morocco",
        )
        self.assertEqual(saves[0].category, "goalkeeper_saves")
        self.assertEqual(saves[0].outcome, "Alisson saves over 3.5")

    def test_starting_lineup_market_metadata(self):
        quotes = market_to_quotes(
            {
                "id": "l1",
                "question": "Will Memphis Depay be in Netherlands's Starting 11?",
                "outcomes": '["Yes","No"]',
                "outcomePrices": '["0.72","0.28"]',
            },
            "Netherlands",
            "Japan",
        )
        self.assertEqual(quotes[0].category, "starting_lineup")
        self.assertEqual(quotes[0].entity, "Memphis Depay")
        self.assertEqual(quotes[0].causal_layer, "lineup_signal")
        self.assertEqual(quotes[0].horizon, "today_match")

    def test_sports_market_type_spreads_totals_and_btts(self):
        spread = market_to_quotes(
            {
                "id": "m5",
                "question": "Spread: FC Bayern München (-2.5)",
                "sportsMarketType": "spreads",
                "outcomes": '["FC Bayern München","RU Saint-Gilloise"]',
                "outcomePrices": '["0.44","0.56"]',
                "groupItemTitle": "FC Bayern München (-2.5)",
            },
            "FC Bayern München",
            "RU Saint-Gilloise",
        )
        self.assertEqual(spread[0].category, "handicap")
        self.assertEqual(spread[0].outcome, "FC Bayern München -2.5")
        self.assertEqual(spread[1].outcome, "RU Saint-Gilloise +2.5")

        total = market_to_quotes(
            {
                "id": "m6",
                "question": "FC Bayern München vs. RU Saint-Gilloise: O/U 2.5",
                "sportsMarketType": "totals",
                "outcomes": '["Over","Under"]',
                "outcomePrices": '["0.61","0.39"]',
                "groupItemTitle": "O/U 2.5",
            },
            "FC Bayern München",
            "RU Saint-Gilloise",
        )
        self.assertEqual(total[0].category, "total_goals")
        self.assertEqual(total[0].outcome, "over 2.5")
        self.assertEqual(total[1].outcome, "under 2.5")

        btts = market_to_quotes(
            {
                "id": "m7",
                "question": "FC Bayern München vs. RU Saint-Gilloise: Both Teams to Score",
                "sportsMarketType": "both_teams_to_score",
                "outcomes": '["Yes","No"]',
                "outcomePrices": '["0.52","0.48"]',
            },
            "FC Bayern München",
            "RU Saint-Gilloise",
        )
        self.assertEqual(btts[0].category, "btts")
        self.assertEqual(btts[0].outcome, "yes")

    def test_sports_page_payload_contains_full_market_matrix(self):
        next_payload = {
            "buildId": "test-build",
            "props": {
                "pageProps": {
                    "serverDate": "2026-06-14T07:30:41Z",
                    "sportsEvent": {
                        "id": "351724",
                        "slug": "fifwc-nld-jpn-2026-06-14",
                        "title": "Netherlands vs. Japan",
                        "markets": [
                            {
                                "id": "shot1",
                                "question": "Memphis Depay: 1+ shots",
                                "sportsMarketType": "soccer_player_shots",
                                "outcomes": ["Yes", "No"],
                                "outcomePrices": ["0.97", "0.03"],
                                "clobTokenIds": ["yes-token", "no-token"],
                                "line": 0.5,
                                "bestBid": 0.96,
                                "bestAsk": 0.98,
                            },
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
                            {
                                "id": "total1",
                                "question": "Netherlands vs. Japan: O/U 2.5",
                                "sportsMarketType": "totals",
                                "outcomes": ["Over", "Under"],
                                "outcomePrices": ["0.475", "0.525"],
                                "line": 2.5,
                                "groupItemTitle": "O/U 2.5",
                                "bestBid": 0.47,
                                "bestAsk": 0.48,
                            },
                            {
                                "id": "score1",
                                "question": "Exact Score: Netherlands 2 - 1 Japan?",
                                "sportsMarketType": "soccer_exact_score",
                                "outcomes": ["Yes", "No"],
                                "outcomePrices": ["0.11", "0.89"],
                            },
                        ],
                    },
                }
            },
        }
        html = (
            '<script id="__NEXT_DATA__" type="application/json">'
            + json.dumps(next_payload)
            + "</script>"
        )
        event = extract_sports_event_from_next_data(html)
        quotes = event_to_quotes(event, "Netherlands", "Japan")

        shot_yes = quotes[0]
        self.assertEqual(shot_yes.category, "player_shots")
        self.assertEqual(shot_yes.outcome, "Memphis Depay shots over 0.5")
        self.assertEqual(shot_yes.entity, "Memphis Depay")
        self.assertEqual(shot_yes.side, "over")
        self.assertEqual(shot_yes.line, 0.5)
        self.assertEqual(shot_yes.causal_layer, "player_prop")
        self.assertEqual(shot_yes.horizon, "today_match")
        self.assertLess(shot_yes.model_weight or 0.0, 0.25)
        self.assertAlmostEqual(shot_yes.spread or 0.0, 0.02)

        shot_no = quotes[1]
        self.assertEqual(shot_no.outcome, "Memphis Depay shots under 0.5")
        self.assertAlmostEqual(shot_no.bid or 0.0, 0.02)
        self.assertAlmostEqual(shot_no.ask or 0.0, 0.04)

        spread_away = quotes[3]
        self.assertEqual(spread_away.category, "handicap")
        self.assertEqual(spread_away.outcome, "Japan +1.5")
        self.assertEqual(spread_away.line, 1.5)

        total_over = quotes[4]
        self.assertEqual(total_over.category, "total_goals")
        self.assertEqual(total_over.outcome, "over 2.5")

        exact = quotes[6]
        self.assertEqual(exact.category, "correct_score")
        self.assertEqual(exact.outcome, "2-1")


if __name__ == "__main__":
    unittest.main()
