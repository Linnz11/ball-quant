import unittest

from ball_quant.adapters.polymarket import event_to_quotes
from ball_quant.core.snapshot import build_live_probability_snapshot
from ball_quant.models import EventMarketMatrix


class SnapshotTest(unittest.TestCase):
    def test_live_probability_snapshot_contains_adaptive_causal_chain(self):
        event = {
            "id": "351724",
            "slug": "fifwc-nld-jpn-2026-06-14",
            "title": "Netherlands vs. Japan",
            "eventDate": "2026-06-14",
            "startTime": "2026-06-14T20:00:00Z",
            "active": True,
            "closed": False,
            "markets": [
                {
                    "id": "ml",
                    "question": "Netherlands vs. Japan",
                    "sportsMarketType": "moneyline",
                    "outcomes": ["Netherlands", "Draw", "Japan"],
                    "outcomePrices": ["0.48", "0.27", "0.25"],
                    "bestBid": 0.47,
                    "bestAsk": 0.49,
                    "liquidity": "200000",
                },
                {
                    "id": "sp",
                    "question": "Spread: Netherlands (-1.5)",
                    "sportsMarketType": "spreads",
                    "outcomes": ["Netherlands", "Japan"],
                    "outcomePrices": ["0.23", "0.77"],
                    "groupItemTitle": "Netherlands (-1.5)",
                    "line": -1.5,
                    "bestBid": 0.22,
                    "bestAsk": 0.24,
                    "liquidity": "50000",
                },
                {
                    "id": "tot",
                    "question": "Netherlands vs. Japan: O/U 2.5",
                    "sportsMarketType": "totals",
                    "outcomes": ["Over", "Under"],
                    "outcomePrices": ["0.48", "0.52"],
                    "groupItemTitle": "O/U 2.5",
                    "line": 2.5,
                    "bestBid": 0.47,
                    "bestAsk": 0.49,
                    "liquidity": "30000",
                },
            ],
        }
        matrix = EventMarketMatrix(
            match_id="351724",
            home="Netherlands",
            away="Japan",
            event_id=event["id"],
            event_slug=event["slug"],
            markets=event_to_quotes(event, "Netherlands", "Japan"),
            raw_event=event,
        )
        snapshot = build_live_probability_snapshot(matrix)

        self.assertEqual(snapshot["match"]["polymarket_date"], "2026-06-14")
        self.assertGreater(snapshot["market_state"]["usable_quote_count"], 0)
        self.assertTrue(snapshot["signal_layers"])
        self.assertTrue(snapshot["collapse_layers"])
        self.assertIn("influence_share", snapshot["collapse_layers"][0])
        self.assertTrue(snapshot["collapse_constraints"])
        self.assertIn("gap_after", snapshot["collapse_constraints"][0])
        self.assertTrue(snapshot["probabilities"]["handicap"])
        self.assertTrue(snapshot["candidate_paths"])


if __name__ == "__main__":
    unittest.main()
