from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from ball_quant.adapters.http import HttpError, get_json
from ball_quant.models import MatchSP, TeamFacts, normalize_key


API_FOOTBALL_BASE_URL = "https://v3.football.api-sports.io"


class APIFootballClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = API_FOOTBALL_BASE_URL,
        cache_dir: Optional[Path] = None,
        offline: bool = False,
    ) -> None:
        self.api_key = api_key or os.environ.get("API_FOOTBALL_KEY")
        self.base_url = base_url
        self.cache_dir = cache_dir
        self.offline = offline

    def facts_for_match(self, match: MatchSP) -> TeamFacts:
        cached = self.load_cached_facts(match)
        if cached:
            return cached
        if self.offline or not self.api_key:
            return unavailable_facts(match, "API_FOOTBALL_KEY missing or offline mode enabled")

        try:
            fixture = self.find_fixture(match)
            if not fixture:
                return unavailable_facts(match, "API-Football fixture not found")
            fixture_id = fixture.get("fixture", {}).get("id")
            payload = {
                "fixture": fixture,
                "lineups": self.get_endpoint("/fixtures/lineups", {"fixture": fixture_id}),
                "injuries": self.get_endpoint("/injuries", {"fixture": fixture_id}),
                "statistics": self.get_endpoint("/fixtures/statistics", {"fixture": fixture_id}),
            }
        except HttpError as exc:
            return unavailable_facts(match, f"API-Football network error: {exc}")

        facts = summarize_facts(match, payload)
        self.write_cached_facts(match, facts)
        return facts

    def find_fixture(self, match: MatchSP) -> Optional[Dict[str, Any]]:
        payload = self.get_endpoint("/fixtures", {"date": match.date})
        candidates = payload.get("response", []) if isinstance(payload, dict) else []
        home_key = normalize_key(match.home)
        away_key = normalize_key(match.away)
        best = None
        best_score = -1
        for item in candidates:
            teams = item.get("teams", {})
            home = normalize_key(teams.get("home", {}).get("name", ""))
            away = normalize_key(teams.get("away", {}).get("name", ""))
            score = 0
            if home_key in home or home in home_key:
                score += 4
            if away_key in away or away in away_key:
                score += 4
            if score > best_score:
                best_score = score
                best = item
        return best if best_score >= 6 else None

    def get_endpoint(self, path: str, params: Dict[str, Any]) -> Dict[str, Any]:
        headers = {"x-apisports-key": self.api_key or ""}
        return get_json(self.base_url, path, params=params, headers=headers)

    def load_cached_facts(self, match: MatchSP) -> Optional[TeamFacts]:
        if not self.cache_dir:
            return None
        path = self.cache_dir / f"facts_{match.date}_{match.match_id}.json"
        if not path.exists():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        return TeamFacts(**payload)

    def write_cached_facts(self, match: MatchSP, facts: TeamFacts) -> None:
        if not self.cache_dir:
            return
        path = self.cache_dir / f"facts_{match.date}_{match.match_id}.json"
        path.write_text(json.dumps(facts.__dict__, ensure_ascii=False, indent=2), encoding="utf-8")


def summarize_facts(match: MatchSP, payload: Dict[str, Any]) -> TeamFacts:
    injuries = extract_injuries(payload.get("injuries", {}))
    statistics = payload.get("statistics", {})
    has_xg = "expected_goals" in json.dumps(statistics).lower()
    warnings: List[str] = []
    confidence_adjustment = 0.05
    if not has_xg:
        warnings.append("API-Football xG coverage incomplete; using shots/possession/recent form proxy where available")
        confidence_adjustment -= 0.08

    home_summary = f"{match.home}: API-Football fixture/statistics loaded"
    away_summary = f"{match.away}: API-Football fixture/statistics loaded"
    if injuries:
        home_summary += f"; injuries/suspensions listed: {len(injuries)} total"
    return TeamFacts(
        match_id=match.match_id,
        source="api-football",
        home_summary=home_summary,
        away_summary=away_summary,
        tactical_notes="Use fixture statistics, lineups and injuries as explanatory layer; do not override market probability mechanically.",
        motivation_notes="Tournament motivation should be reviewed with group table context when available.",
        injuries=injuries,
        warnings=warnings,
        confidence_adjustment=confidence_adjustment,
        raw=payload,
    )


def extract_injuries(payload: Dict[str, Any]) -> List[str]:
    response = payload.get("response", []) if isinstance(payload, dict) else []
    injuries = []
    for item in response:
        player = item.get("player", {}).get("name", "Unknown")
        team = item.get("team", {}).get("name", "Unknown team")
        reason = item.get("player", {}).get("reason") or item.get("reason") or "listed"
        injuries.append(f"{team} - {player}: {reason}")
    return injuries


def unavailable_facts(match: MatchSP, reason: str) -> TeamFacts:
    return TeamFacts(
        match_id=match.match_id,
        source="unavailable",
        home_summary=f"{match.home}: fact feed unavailable",
        away_summary=f"{match.away}: fact feed unavailable",
        warnings=[reason],
        confidence_adjustment=-0.15,
    )
