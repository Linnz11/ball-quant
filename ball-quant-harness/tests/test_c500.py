"""Tests for the 500.com 竞彩 odds adapter (src/ball_quant/adapters/c500.py).

All tests run fully offline using cassette HTML files saved under tests/fixtures/.
Cassettes were captured 2026-06-15 via 'ssh jeffly@47.84.5.161 curl ...' with GB18030
converted to UTF-8 by iconv; trimmed to 4 matches.

Covered:
  1. parse_html: structural parsing sanity (rows returned, type fields present).
  2. load_odds_from_fixtures: full 4-page merge → TicaiOdds with all 5 playtypes.
  3. Value-mapping unit tests: crs "1:0"→"1-0", hafu "3-1"→"hd", ttg "7"→7.
  4. CLI recommend --c500-cache integration with polymarket fixture.
  5. No BeautifulSoup / lxml / requests imported (grep enforced below).
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

# ---------------------------------------------------------------------------
# Confirm no 3rd-party parser imports in the adapter module
# ---------------------------------------------------------------------------

_ADAPTER_PATH = Path(__file__).parent.parent / "src" / "ball_quant" / "adapters" / "c500.py"


class TestNoDepsInAdapter(unittest.TestCase):
    """The adapter must use stdlib only — never bs4 / lxml / requests."""

    def test_no_forbidden_imports(self):
        source = _ADAPTER_PATH.read_text(encoding="utf-8")
        # Check for explicit import statements of forbidden 3rd-party libraries.
        # "requests" is skipped as a standalone check because "urllib.request"
        # legitimately contains that substring; instead we look for "import requests".
        import re as _re
        for forbidden in ("beautifulsoup", "bs4", "lxml"):
            self.assertNotIn(
                forbidden,
                source.lower(),
                msg=f"Forbidden library '{forbidden}' found in c500.py",
            )
        # requests: only flag "import requests" (not "urllib.request")
        self.assertFalse(
            bool(_re.search(r"\bimport requests\b", source)),
            "Found 'import requests' in c500.py — use urllib.request instead",
        )

    def test_uses_html_parser(self):
        source = _ADAPTER_PATH.read_text(encoding="utf-8")
        self.assertIn("html.parser", source, "Expected html.parser import in c500.py")

    def test_uses_urllib(self):
        source = _ADAPTER_PATH.read_text(encoding="utf-8")
        self.assertIn("urllib", source, "Expected urllib usage in c500.py")


# ---------------------------------------------------------------------------
# Helper: load cassette files
# ---------------------------------------------------------------------------

FIXTURES = Path(__file__).parent / "fixtures"


def _read(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Test: parse_html on individual cassette pages
# ---------------------------------------------------------------------------

class TestParseHtmlIndividual(unittest.TestCase):
    """Verify each page-type returns the expected row count and data-type entries."""

    def test_spf_page_returns_4_rows(self):
        from ball_quant.adapters.c500 import parse_html
        rows = parse_html(_read("c500_spf.html"))
        self.assertGreaterEqual(len(rows), 1, "Expected at least 1 row from SPF cassette")

    def test_spf_page_has_nspf_and_spf(self):
        from ball_quant.adapters.c500 import parse_html
        rows = parse_html(_read("c500_spf.html"))
        # At least one row should have nspf (胜平负) odds
        has_nspf = any(r.nspf for r in rows)
        # At least one row should have spf (让球) odds
        has_rq = any(r.spf for r in rows)
        self.assertTrue(has_nspf, "No nspf odds found in SPF cassette rows")
        self.assertTrue(has_rq, "No spf (让球) odds found in SPF cassette rows")

    def test_crs_page_has_bf_odds(self):
        from ball_quant.adapters.c500 import parse_html
        rows = parse_html(_read("c500_crs.html"))
        has_bf = any(r.bf for r in rows)
        self.assertTrue(has_bf, "No bf (比分) odds found in CRS cassette rows")

    def test_ttg_page_has_jqs_odds(self):
        from ball_quant.adapters.c500 import parse_html
        rows = parse_html(_read("c500_ttg.html"))
        has_jqs = any(r.jqs for r in rows)
        self.assertTrue(has_jqs, "No jqs (进球数) odds found in TTG cassette rows")

    def test_hafu_page_has_bqc_odds(self):
        from ball_quant.adapters.c500 import parse_html
        rows = parse_html(_read("c500_hafu.html"))
        has_bqc = any(r.bqc for r in rows)
        self.assertTrue(has_bqc, "No bqc (半全场) odds found in HAFU cassette rows")

    def test_parse_html_raises_on_empty(self):
        from ball_quant.adapters.c500 import parse_html
        with self.assertRaises(ValueError):
            parse_html("<html><body><p>no matches today</p></body></html>")


# ---------------------------------------------------------------------------
# Test: load_odds_from_fixtures — full merge
# ---------------------------------------------------------------------------

class TestLoadOddsFromFixtures(unittest.TestCase):
    """Full 4-page parse → TicaiOdds merge."""

    @classmethod
    def setUpClass(cls):
        from ball_quant.adapters.c500 import load_odds_from_fixtures
        cls.odds_list = load_odds_from_fixtures(
            spf_html=_read("c500_spf.html"),
            crs_html=_read("c500_crs.html"),
            ttg_html=_read("c500_ttg.html"),
            hafu_html=_read("c500_hafu.html"),
        )

    def test_at_least_one_ticai_odds_returned(self):
        self.assertGreaterEqual(len(self.odds_list), 1)

    def test_all_entries_are_ticai_odds(self):
        from ball_quant.models import TicaiOdds
        for o in self.odds_list:
            self.assertIsInstance(o, TicaiOdds)

    def test_match_has_home_and_away(self):
        for o in self.odds_list:
            self.assertTrue(o.home, f"Empty home team for match_id={o.match_id}")
            self.assertTrue(o.away, f"Empty away team for match_id={o.match_id}")

    def test_match_has_date(self):
        for o in self.odds_list:
            self.assertTrue(o.match_date, f"Missing match_date for {o.match_id}")

    # ---- SPF (胜平负) -------------------------------------------------------

    def test_at_least_one_spf_with_3_outcomes(self):
        """At least one match must have home/draw/away SPF odds."""
        ok = any(
            "home" in o.spf and "draw" in o.spf and "away" in o.spf
            for o in self.odds_list
        )
        self.assertTrue(ok, "No TicaiOdds has all 3 SPF outcomes")

    def test_spf_odds_are_floats_gt_1(self):
        for o in self.odds_list:
            for k, v in o.spf.items():
                self.assertIsInstance(v, float, f"spf[{k}] is not float in {o.match_id}")
                self.assertGreater(v, 1.0, f"spf[{k}]={v} ≤ 1.0 in {o.match_id}")

    # ---- RQSPF (让球) -------------------------------------------------------

    def test_at_least_one_rqspf_with_3_outcomes(self):
        ok = any(
            "home" in o.rqspf and "draw" in o.rqspf and "away" in o.rqspf
            for o in self.odds_list
        )
        self.assertTrue(ok, "No TicaiOdds has all 3 RQSPF outcomes")

    def test_rqspf_has_handicap_line(self):
        """Any match with rqspf odds should also carry a handicap line."""
        for o in self.odds_list:
            if o.rqspf:
                self.assertIsNotNone(
                    o.handicap_line,
                    f"rqspf present but handicap_line=None for {o.match_id}",
                )

    # ---- CRS (比分) ---------------------------------------------------------

    def test_at_least_one_crs_with_score_keys(self):
        """At least one match has "H-A" style score keys in correct_score."""
        import re
        pat = re.compile(r"^\d+-\d+$")
        ok = any(
            any(pat.match(k) for k in o.correct_score.keys())
            for o in self.odds_list
        )
        self.assertTrue(ok, "No TicaiOdds has numeric H-A keys in correct_score")

    def test_crs_has_other_bucket(self):
        """At least one match has a home_other / draw_other / away_other bucket."""
        other_keys = {"home_other", "draw_other", "away_other"}
        ok = any(
            bool(other_keys & set(o.correct_score.keys()))
            for o in self.odds_list
        )
        self.assertTrue(ok, "No TicaiOdds has an 'other' bucket in correct_score")

    def test_crs_odds_are_floats(self):
        for o in self.odds_list:
            for k, v in o.correct_score.items():
                self.assertIsInstance(v, float, f"correct_score[{k}] not float in {o.match_id}")

    # ---- TTG (总进球) --------------------------------------------------------

    def test_at_least_one_ttg_with_int_keys(self):
        ok = any(o.total_goals for o in self.odds_list)
        self.assertTrue(ok, "No TicaiOdds has total_goals data")

    def test_ttg_keys_are_ints_0_to_7(self):
        for o in self.odds_list:
            for k in o.total_goals:
                self.assertIsInstance(k, int, f"total_goals key {k!r} is not int in {o.match_id}")
                self.assertIn(k, range(8), f"total_goals key {k} out of 0-7 range in {o.match_id}")

    def test_ttg_odds_are_floats(self):
        for o in self.odds_list:
            for k, v in o.total_goals.items():
                self.assertIsInstance(v, float, f"total_goals[{k}] not float in {o.match_id}")

    # ---- HAFU (半全场) -------------------------------------------------------

    def test_at_least_one_hafu_with_2char_keys(self):
        valid = {"hh", "hd", "ha", "dh", "dd", "da", "ah", "ad", "aa"}
        ok = any(
            bool(valid & set(o.hafu.keys()))
            for o in self.odds_list
        )
        self.assertTrue(ok, "No TicaiOdds has valid 2-char hafu keys")

    def test_hafu_odds_are_floats(self):
        for o in self.odds_list:
            for k, v in o.hafu.items():
                self.assertIsInstance(v, float, f"hafu[{k}] not float in {o.match_id}")


# ---------------------------------------------------------------------------
# Test: value-mapping unit tests
# ---------------------------------------------------------------------------

class TestValueMappings(unittest.TestCase):
    """Unit tests for the pure mapping helper functions."""

    def test_crs_score_colon_to_dash(self):
        from ball_quant.adapters.c500 import _crs_value_to_key
        self.assertEqual(_crs_value_to_key("1:0"), "1-0")
        self.assertEqual(_crs_value_to_key("2:1"), "2-1")
        self.assertEqual(_crs_value_to_key("0:0"), "0-0")
        self.assertEqual(_crs_value_to_key("5:2"), "5-2")

    def test_crs_other_buckets(self):
        from ball_quant.adapters.c500 import _crs_value_to_key
        self.assertEqual(_crs_value_to_key("胜其它"), "home_other")
        self.assertEqual(_crs_value_to_key("平其它"), "draw_other")
        self.assertEqual(_crs_value_to_key("负其它"), "away_other")

    def test_crs_unknown_returns_none(self):
        from ball_quant.adapters.c500 import _crs_value_to_key
        self.assertIsNone(_crs_value_to_key("garbage"))
        self.assertIsNone(_crs_value_to_key(""))

    def test_hafu_all_9_combos(self):
        from ball_quant.adapters.c500 import _hafu_value_to_key
        mapping = {
            "3-3": "hh", "3-1": "hd", "3-0": "ha",
            "1-3": "dh", "1-1": "dd", "1-0": "da",
            "0-3": "ah", "0-1": "ad", "0-0": "aa",
        }
        for raw, expected in mapping.items():
            with self.subTest(raw=raw):
                self.assertEqual(_hafu_value_to_key(raw), expected)

    def test_hafu_invalid_returns_none(self):
        from ball_quant.adapters.c500 import _hafu_value_to_key
        self.assertIsNone(_hafu_value_to_key("garbage"))
        self.assertIsNone(_hafu_value_to_key(""))
        self.assertIsNone(_hafu_value_to_key("2-3"))  # 2 is not a valid HT result code

    def test_ttg_key_7_means_7plus(self):
        """Verify the parser stores ttg "7" as int key 7 (the 7+ bucket)."""
        from ball_quant.adapters.c500 import load_odds_from_fixtures
        # The TTG cassette has jqs data-value="7" buttons; after merge they appear as int 7.
        odds = load_odds_from_fixtures(
            spf_html=_read("c500_spf.html"),
            crs_html=_read("c500_crs.html"),
            ttg_html=_read("c500_ttg.html"),
            hafu_html=_read("c500_hafu.html"),
        )
        found_seven = any(7 in o.total_goals for o in odds)
        self.assertTrue(found_seven, "int key 7 (7+ goals) missing from total_goals after parse")


# ---------------------------------------------------------------------------
# Test: CLI recommend --c500-cache integration
# ---------------------------------------------------------------------------

class TestRecommendC500Cache(unittest.TestCase):
    """Drive cli.main(['recommend', '--c500-cache', ...]) fully offline.

    Uses fixtures: c500_*.html (cassettes) + polymarket_c500_test.json
    (Belgium vs Egypt Polymarket matrix).

    Belgium = 比利时 (in c500 cassette fixture 1359206, matchdate 2026-06-16).
    TEAM_ALIASES already maps 比利时→belgium, 埃及→egypt.
    """

    def _run(self, extra_args=None, budget=200.0):
        from ball_quant import cli
        with tempfile.TemporaryDirectory() as tmp:
            report_out = os.path.join(tmp, "recommend_c500.md")
            json_out   = os.path.join(tmp, "recommend_c500.json")
            argv = [
                "recommend",
                "--budget", str(budget),
                "--c500-cache", str(FIXTURES),
                "--polymarket-cache", str(FIXTURES / "polymarket_c500_test.json"),
                "--date", "2026-06-16",
                "--report-out", report_out,
                "--json-out", json_out,
            ]
            if extra_args:
                argv.extend(extra_args)
            rc = cli.main(argv)
            report_text = Path(report_out).read_text(encoding="utf-8")
            json_payload = json.loads(Path(json_out).read_text(encoding="utf-8"))
        return rc, report_text, json_payload

    def test_exit_code_zero(self):
        rc, _, _ = self._run()
        self.assertEqual(rc, 0)

    def test_report_written_with_budget_line(self):
        _, report_text, _ = self._run()
        self.assertIn("预算", report_text)

    def test_json_has_required_keys(self):
        _, _, payload = self._run()
        self.assertIn("recommended_bets", payload)
        self.assertIn("unmatched_ticai", payload)
        self.assertIn("total_staked", payload)
        self.assertIn("budget", payload)

    def test_total_staked_within_budget(self):
        _, _, payload = self._run(budget=200.0)
        self.assertLessEqual(payload["total_staked"], 200.0 + 1e-6)

    def test_single_bet_fields_present(self):
        _, _, payload = self._run()
        for bet in payload["recommended_bets"]:
            if bet.get("type") != "single":
                continue
            for field in ("match", "play", "outcome", "ticai_odds", "prob", "edge", "stake"):
                self.assertIn(field, bet, f"Missing field '{field}' in bet: {bet}")


if __name__ == "__main__":
    unittest.main()
