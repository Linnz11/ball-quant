import unittest
from datetime import datetime, timezone

from ball_quant.core.schedule import (
    is_core_match_event,
    schedule_row,
    select_schedule_rows,
)


class ScheduleTest(unittest.TestCase):
    def test_polymarket_date_and_local_date_are_both_preserved(self):
        event = {
            "id": "351729",
            "slug": "fifwc-irn-nzl-2026-06-15",
            "title": "IR Iran vs. New Zealand",
            "eventDate": "2026-06-15",
            "startTime": "2026-06-16T01:00:00Z",
            "active": True,
            "closed": False,
        }
        row = schedule_row(
            event,
            timezone_name="Asia/Shanghai",
            now_utc=datetime(2026, 6, 15, 12, tzinfo=timezone.utc),
        )
        self.assertEqual(row["polymarket_date"], "2026-06-15")
        self.assertEqual(row["local_date"], "2026-06-16")
        self.assertEqual(row["local_time"], "09:00")
        self.assertEqual(row["status"], "upcoming")

    def test_derivative_markets_are_not_core_matches(self):
        self.assertFalse(
            is_core_match_event(
                {
                    "slug": "fifwc-nld-jpn-2026-06-14-more-markets",
                    "title": "Netherlands vs. Japan - More Markets",
                }
            )
        )
        self.assertTrue(
            is_core_match_event(
                {
                    "slug": "fifwc-nld-jpn-2026-06-14",
                    "title": "Netherlands vs. Japan",
                }
            )
        )

    def test_select_by_polymarket_date_not_local_date(self):
        events = [
            {
                "id": "1",
                "slug": "fifwc-esp-cvi-2026-06-15",
                "title": "Spain vs. Cabo Verde",
                "eventDate": "2026-06-15",
                "startTime": "2026-06-15T16:00:00Z",
                "closed": False,
            },
            {
                "id": "2",
                "slug": "fifwc-ger-kor-2026-06-14",
                "title": "Germany vs. Curaçao",
                "eventDate": "2026-06-14",
                "startTime": "2026-06-14T17:00:00Z",
                "closed": False,
            },
        ]
        rows = select_schedule_rows(
            events,
            timezone_name="Asia/Shanghai",
            date="2026-06-15",
            date_mode="poly",
            now_utc=datetime(2026, 6, 15, 8, tzinfo=timezone.utc),
        )
        self.assertEqual([row["event_slug"] for row in rows], ["fifwc-esp-cvi-2026-06-15"])

    def test_expired_rows_are_pruned_by_default(self):
        events = [
            {
                "id": "1",
                "slug": "fifwc-aus-tur-2026-06-14",
                "title": "Australia vs. Türkiye",
                "eventDate": "2026-06-14",
                "startTime": "2026-06-14T04:00:00Z",
                "closed": True,
            }
        ]
        rows = select_schedule_rows(
            events,
            now_utc=datetime(2026, 6, 14, 8, tzinfo=timezone.utc),
        )
        self.assertEqual(rows, [])
        kept = select_schedule_rows(
            events,
            include_expired=True,
            now_utc=datetime(2026, 6, 14, 8, tzinfo=timezone.utc),
        )
        self.assertEqual(kept[0]["status"], "expired")


if __name__ == "__main__":
    unittest.main()

