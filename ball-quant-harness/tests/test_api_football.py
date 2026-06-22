"""Tests for ball_quant.adapters.api_football — deterministic, offline."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict
from unittest.mock import patch

import pytest

from ball_quant.adapters.api_football import APIFootballClient, summarize_facts, unavailable_facts
from ball_quant.models import MatchSP, TeamFacts


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_match(match_id: str = "001", date: str = "2026-06-14") -> MatchSP:
    return MatchSP(
        match_id=match_id,
        date=date,
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


def _cassette(cassette_dir: Path, name: str) -> Dict[str, Any]:
    return json.loads((cassette_dir / name).read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# unavailable_facts — missing API key
# ---------------------------------------------------------------------------

class TestUnavailableFacts:
    def test_missing_env_key_returns_unavailable(self, monkeypatch):
        monkeypatch.delenv("API_FOOTBALL_KEY", raising=False)
        match = _make_match()
        client = APIFootballClient(api_key=None)
        facts = client.facts_for_match(match)
        assert facts.source == "unavailable"

    def test_missing_env_key_confidence_adjustment_is_minus_015(self, monkeypatch):
        monkeypatch.delenv("API_FOOTBALL_KEY", raising=False)
        match = _make_match()
        client = APIFootballClient(api_key=None)
        facts = client.facts_for_match(match)
        assert facts.confidence_adjustment == pytest.approx(-0.15)

    def test_unavailable_facts_helper_directly(self):
        match = _make_match()
        facts = unavailable_facts(match, "test reason")
        assert facts.source == "unavailable"
        assert facts.confidence_adjustment == pytest.approx(-0.15)
        assert "test reason" in facts.warnings

    def test_offline_mode_returns_unavailable(self, monkeypatch):
        monkeypatch.setenv("API_FOOTBALL_KEY", "dummy")
        match = _make_match()
        client = APIFootballClient(api_key="dummy", offline=True)
        facts = client.facts_for_match(match)
        assert facts.source == "unavailable"


# ---------------------------------------------------------------------------
# facts_for_match — monkeypatched get_json via cassettes
# ---------------------------------------------------------------------------

class TestFactsForMatchCassette:
    def _cassette_payload(self, cassette_dir: Path) -> Dict[str, Any]:
        return {
            "fixture": _cassette(cassette_dir, "api_football_fixture.json")["response"][0],
            "lineups": _cassette(cassette_dir, "api_football_lineups.json"),
            "injuries": _cassette(cassette_dir, "api_football_injuries.json"),
            "statistics": _cassette(cassette_dir, "api_football_stats.json"),
        }

    def test_parses_cassette_into_team_facts(self, cassette_dir: Path):
        match = _make_match()
        payload = self._cassette_payload(cassette_dir)

        # Simulate find_fixture returning the fixture entry, and each subsequent
        # get_endpoint returning the cassette data.
        fixture_response = _cassette(cassette_dir, "api_football_fixture.json")
        lineups = _cassette(cassette_dir, "api_football_lineups.json")
        injuries = _cassette(cassette_dir, "api_football_injuries.json")
        stats = _cassette(cassette_dir, "api_football_stats.json")

        call_sequence = [fixture_response, lineups, injuries, stats]
        call_iter = iter(call_sequence)

        def fake_get_json(base_url, path, params=None, headers=None, **kwargs):
            return next(call_iter)

        with patch("ball_quant.adapters.api_football.get_json", side_effect=fake_get_json):
            client = APIFootballClient(api_key="test-key")
            facts = client.facts_for_match(match)

        assert isinstance(facts, TeamFacts)
        assert facts.source == "api-football"
        assert facts.match_id == "001"

    def test_injuries_parsed_into_list(self, cassette_dir: Path):
        match = _make_match()
        fixture_response = _cassette(cassette_dir, "api_football_fixture.json")
        lineups = _cassette(cassette_dir, "api_football_lineups.json")
        injuries = _cassette(cassette_dir, "api_football_injuries.json")
        stats = _cassette(cassette_dir, "api_football_stats.json")

        call_sequence = [fixture_response, lineups, injuries, stats]
        call_iter = iter(call_sequence)

        def fake_get_json(base_url, path, params=None, headers=None, **kwargs):
            return next(call_iter)

        with patch("ball_quant.adapters.api_football.get_json", side_effect=fake_get_json):
            client = APIFootballClient(api_key="test-key")
            facts = client.facts_for_match(match)

        assert len(facts.injuries) == 2
        # Both injury entries should be present
        injury_text = " ".join(facts.injuries)
        assert "van Dijk" in injury_text or "Minamino" in injury_text

    def test_xg_present_gives_positive_confidence_adjustment(self, cassette_dir: Path):
        match = _make_match()
        fixture_response = _cassette(cassette_dir, "api_football_fixture.json")
        lineups = _cassette(cassette_dir, "api_football_lineups.json")
        injuries = _cassette(cassette_dir, "api_football_injuries.json")
        stats = _cassette(cassette_dir, "api_football_stats.json")

        call_sequence = [fixture_response, lineups, injuries, stats]
        call_iter = iter(call_sequence)

        def fake_get_json(base_url, path, params=None, headers=None, **kwargs):
            return next(call_iter)

        with patch("ball_quant.adapters.api_football.get_json", side_effect=fake_get_json):
            client = APIFootballClient(api_key="test-key")
            facts = client.facts_for_match(match)

        # Stats cassette has expected_goals -> confidence_adjustment stays at 0.05 (no penalty)
        assert facts.confidence_adjustment == pytest.approx(0.05)

    def test_fixture_not_found_returns_unavailable(self):
        match = _make_match()
        # Return a response with no candidates -> find_fixture returns None
        empty_response = {"response": []}

        with patch("ball_quant.adapters.api_football.get_json", return_value=empty_response):
            client = APIFootballClient(api_key="test-key")
            facts = client.facts_for_match(match)

        assert facts.source == "unavailable"
        assert facts.confidence_adjustment == pytest.approx(-0.15)

    def test_cached_facts_loaded_from_cache_dir(self, tmp_path: Path):
        match = _make_match()
        expected_facts = TeamFacts(
            match_id="001",
            source="api-football",
            home_summary="cached home",
            away_summary="cached away",
        )
        cache_dir = tmp_path / "facts_cache"
        cache_dir.mkdir()
        cache_file = cache_dir / f"facts_{match.date}_{match.match_id}.json"
        cache_file.write_text(json.dumps(expected_facts.__dict__), encoding="utf-8")

        # With a valid cache, get_json must NOT be called
        with patch("ball_quant.adapters.api_football.get_json") as mock_gj:
            client = APIFootballClient(api_key="test-key", cache_dir=cache_dir)
            facts = client.facts_for_match(match)

        mock_gj.assert_not_called()
        assert facts.home_summary == "cached home"
        assert facts.source == "api-football"


# ---------------------------------------------------------------------------
# summarize_facts — unit tests against cassette payloads
# ---------------------------------------------------------------------------

class TestSummarizeFacts:
    def test_home_summary_contains_team_name(self, cassette_dir: Path):
        match = _make_match()
        injuries = _cassette(cassette_dir, "api_football_injuries.json")
        stats = _cassette(cassette_dir, "api_football_stats.json")
        fixture = _cassette(cassette_dir, "api_football_fixture.json")["response"][0]
        payload = {
            "fixture": fixture,
            "lineups": _cassette(cassette_dir, "api_football_lineups.json"),
            "injuries": injuries,
            "statistics": stats,
        }
        facts = summarize_facts(match, payload)
        assert "Netherlands" in facts.home_summary
        assert "Japan" in facts.away_summary

    def test_no_xg_in_stats_reduces_confidence(self):
        match = _make_match()
        payload = {
            "fixture": {},
            "lineups": {},
            "injuries": {},
            "statistics": {"response": []},
        }
        facts = summarize_facts(match, payload)
        # No xG in empty stats -> confidence_adjustment < 0.05
        assert facts.confidence_adjustment < 0.05
