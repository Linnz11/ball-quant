"""Tests for the sporttery (中国竞彩) odds + results adapter.

All tests run purely from cassette fixtures — no live network.
Cassette: tests/fixtures/sporttery_calc.json (2 matches, all 5 pool types)
          tests/fixtures/sporttery_results.json (2 match results)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import pytest

from ball_quant.adapters.sporttery import (
    fetch_odds_raw,
    load_odds,
    parse_odds,
    parse_results,
)
from ball_quant.core.settlement import MatchOutcome
from ball_quant.models import TicaiOdds

FIXTURES = Path(__file__).parent / "fixtures"
CALC_CASSETTE = FIXTURES / "sporttery_calc.json"
RESULTS_CASSETTE = FIXTURES / "sporttery_results.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_cassette(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# parse_odds — structural + field assertions
# ---------------------------------------------------------------------------

class TestParseOdds:
    def setup_method(self):
        raw = _load_cassette(CALC_CASSETTE)
        self.odds: list[TicaiOdds] = parse_odds(raw)

    def test_two_matches_returned(self):
        assert len(self.odds) == 2

    def test_match_ids(self):
        ids = {o.match_id for o in self.odds}
        assert ids == {"M001", "M002"}

    def test_match_num_preserved(self):
        m001 = next(o for o in self.odds if o.match_id == "M001")
        assert m001.match_num == "周日009"

    def test_match_date(self):
        m001 = next(o for o in self.odds if o.match_id == "M001")
        assert m001.match_date == "2026-06-14"

    def test_league_home_away(self):
        m001 = next(o for o in self.odds if o.match_id == "M001")
        assert m001.league == "英超"
        assert m001.home == "曼城"
        assert m001.away == "阿森纳"

    # --- SPF (HAD) ---

    def test_spf_keys_and_types(self):
        for o in self.odds:
            assert set(o.spf.keys()) == {"home", "draw", "away"}
            for v in o.spf.values():
                assert isinstance(v, float)

    def test_spf_values_m001(self):
        m001 = next(o for o in self.odds if o.match_id == "M001")
        assert m001.spf["home"] == pytest.approx(2.10)
        assert m001.spf["draw"] == pytest.approx(3.40)
        assert m001.spf["away"] == pytest.approx(3.20)

    # --- HHAD (handicap) ---

    def test_handicap_line_m001(self):
        m001 = next(o for o in self.odds if o.match_id == "M001")
        assert m001.handicap_line == pytest.approx(-1.0)

    def test_handicap_line_m002_zero(self):
        m002 = next(o for o in self.odds if o.match_id == "M002")
        assert m002.handicap_line == pytest.approx(0.0)

    def test_rqspf_keys_and_types(self):
        for o in self.odds:
            assert set(o.rqspf.keys()) == {"home", "draw", "away"}
            for v in o.rqspf.values():
                assert isinstance(v, float)

    def test_rqspf_values_m001(self):
        m001 = next(o for o in self.odds if o.match_id == "M001")
        assert m001.rqspf["home"] == pytest.approx(2.40)
        assert m001.rqspf["draw"] == pytest.approx(3.10)
        assert m001.rqspf["away"] == pytest.approx(2.80)

    # --- Correct Score (CRS) ---

    def test_crs_standard_score_keys_converted(self):
        m001 = next(o for o in self.odds if o.match_id == "M001")
        cs = m001.correct_score
        # s01s00 → "1-0", s00s00 → "0-0", s01s01 → "1-1"
        assert "1-0" in cs
        assert "0-0" in cs
        assert "1-1" in cs

    def test_crs_leading_zeros_stripped(self):
        """s02s01 must become "2-1", not "02-01"."""
        m001 = next(o for o in self.odds if o.match_id == "M001")
        cs = m001.correct_score
        assert "2-1" in cs
        assert cs["2-1"] == pytest.approx(8.50)
        assert cs["1-0"] == pytest.approx(6.50)
        assert cs["0-0"] == pytest.approx(7.00)
        assert cs["1-1"] == pytest.approx(8.00)

    def test_crs_other_buckets_preserved_as_raw_keys(self):
        """Non-standard keys (home_other etc.) must be kept under their raw key."""
        m001 = next(o for o in self.odds if o.match_id == "M001")
        cs = m001.correct_score
        assert "home_other" in cs
        assert "draw_other" in cs
        assert "away_other" in cs
        assert cs["home_other"] == pytest.approx(18.00)

    def test_crs_all_odds_floats(self):
        for o in self.odds:
            for v in o.correct_score.values():
                assert isinstance(v, float)

    # --- Total Goals (TTG) ---

    def test_ttg_int_keys_0_to_7(self):
        for o in self.odds:
            assert set(o.total_goals.keys()) == set(range(8))

    def test_ttg_key_7_is_seven_plus(self):
        """Key 7 (s7 in the API) represents 7+ goals."""
        m001 = next(o for o in self.odds if o.match_id == "M001")
        assert m001.total_goals[7] == pytest.approx(30.00)

    def test_ttg_values_m001(self):
        m001 = next(o for o in self.odds if o.match_id == "M001")
        assert m001.total_goals[0] == pytest.approx(12.00)
        assert m001.total_goals[1] == pytest.approx(5.50)
        assert m001.total_goals[2] == pytest.approx(3.20)

    def test_ttg_all_odds_floats(self):
        for o in self.odds:
            for v in o.total_goals.values():
                assert isinstance(v, float)

    # --- Half/Full (HAFU) ---

    _HAFU_KEYS = {"hh", "hd", "ha", "dh", "dd", "da", "ah", "ad", "aa"}

    def test_hafu_nine_keys(self):
        for o in self.odds:
            assert set(o.hafu.keys()) == self._HAFU_KEYS

    def test_hafu_values_m001(self):
        m001 = next(o for o in self.odds if o.match_id == "M001")
        assert m001.hafu["hh"] == pytest.approx(3.20)
        assert m001.hafu["aa"] == pytest.approx(4.20)
        assert m001.hafu["dd"] == pytest.approx(5.50)

    def test_hafu_all_odds_floats(self):
        for o in self.odds:
            for v in o.hafu.values():
                assert isinstance(v, float)

    # --- Frozen dataclass ---

    def test_ticai_odds_is_frozen(self):
        m001 = next(o for o in self.odds if o.match_id == "M001")
        with pytest.raises((AttributeError, TypeError)):
            m001.match_id = "MUTATED"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Error handling — structural breaks
# ---------------------------------------------------------------------------

class TestParseOddsErrors:
    def test_missing_matchInfoList_raises_ValueError(self):
        broken = {"payload": {"value": {}}}
        with pytest.raises(ValueError, match="matchInfoList"):
            parse_odds(broken)

    def test_completely_empty_payload_raises_ValueError(self):
        with pytest.raises(ValueError):
            parse_odds({})

    def test_match_without_matchId_is_skipped(self):
        """A subMatch entry missing matchId should be silently dropped."""
        raw = {
            "payload": {
                "value": {
                    "matchInfoList": [
                        {
                            "subMatchList": [
                                {
                                    # no matchId
                                    "had": {"h": "2.00", "d": "3.00", "a": "4.00"},
                                }
                            ]
                        }
                    ]
                }
            }
        }
        result = parse_odds(raw)
        assert result == []

    def test_non_numeric_odds_field_skipped(self):
        """A non-numeric odds value in had should be excluded rather than raise."""
        raw = {
            "payload": {
                "value": {
                    "matchInfoList": [
                        {
                            "subMatchList": [
                                {
                                    "matchId": "TEST",
                                    "had": {"h": "BAD", "d": "3.00", "a": "4.00"},
                                }
                            ]
                        }
                    ]
                }
            }
        }
        result = parse_odds(raw)
        assert len(result) == 1
        # "home" absent (bad value skipped), others present
        assert "home" not in result[0].spf
        assert result[0].spf.get("draw") == pytest.approx(3.00)


# ---------------------------------------------------------------------------
# parse_results — MatchOutcome
# ---------------------------------------------------------------------------

class TestParseResults:
    def setup_method(self):
        raw = _load_cassette(RESULTS_CASSETTE)
        self.outcomes: Dict[str, MatchOutcome] = parse_results(raw)

    def test_two_outcomes(self):
        assert len(self.outcomes) == 2

    def test_match_ids_present(self):
        assert "M001" in self.outcomes
        assert "M002" in self.outcomes

    def test_scores_m001(self):
        o = self.outcomes["M001"]
        assert o.home_score == 2
        assert o.away_score == 1

    def test_scores_m002(self):
        o = self.outcomes["M002"]
        assert o.home_score == 0
        assert o.away_score == 3

    def test_settled_flag(self):
        for o in self.outcomes.values():
            assert o.settled is True

    def test_outcome_types(self):
        for o in self.outcomes.values():
            assert isinstance(o, MatchOutcome)
            assert isinstance(o.home_score, int)
            assert isinstance(o.away_score, int)

    def test_missing_list_raises(self):
        with pytest.raises(ValueError, match="list"):
            parse_results({"payload": {"value": {}}})


# ---------------------------------------------------------------------------
# Integration — fetch_odds_raw + parse_odds via cache_path (cassette replay)
# ---------------------------------------------------------------------------

class TestCassetteFetchIntegration:
    def test_fetch_odds_raw_reads_cassette(self):
        """fetch_odds_raw with cache_path pointing at the cassette never hits network."""
        raw = fetch_odds_raw(cache_path=CALC_CASSETTE)
        # Structural check: the key we need is present
        assert "payload" in raw
        assert "matchInfoList" in raw["payload"]["value"]

    def test_load_odds_end_to_end(self):
        """load_odds(cache_path) → 2 TicaiOdds without network."""
        odds = load_odds(CALC_CASSETTE)
        assert len(odds) == 2
        assert all(isinstance(o, TicaiOdds) for o in odds)

    def test_load_odds_m001_spf_via_cassette(self):
        odds = load_odds(CALC_CASSETTE)
        m001 = next(o for o in odds if o.match_id == "M001")
        assert m001.spf["home"] == pytest.approx(2.10)
        assert m001.handicap_line == pytest.approx(-1.0)
        assert m001.total_goals[7] == pytest.approx(30.00)
        assert len(m001.hafu) == 9

    def test_adapter_uses_sporttery_headers(self):
        """Confirm DEFAULT_HEADERS contains all four required sporttery headers."""
        from ball_quant.adapters.sporttery import DEFAULT_HEADERS
        assert "User-Agent" in DEFAULT_HEADERS
        assert "Referer" in DEFAULT_HEADERS
        assert "Origin" in DEFAULT_HEADERS
        assert "Accept" in DEFAULT_HEADERS
        assert "sporttery.cn" in DEFAULT_HEADERS["Referer"]
        assert "sporttery.cn" in DEFAULT_HEADERS["Origin"]
