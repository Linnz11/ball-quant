"""Sporttery (中国竞彩) odds + results adapter.

Ingests ALL bet types in one API call:
  HAD (胜平负) · HHAD (让球胜平负) · CRS (比分) · TTG (进球数) · HAFU (半全场)

Live fetch: GET https://webapi.sporttery.cn/gateway/uniform/football/getMatchCalculatorV1.qry
  → BLOCKED from non-China IPs (HTTP 567).  Tests replay from a cassette; live fetch
  is the user's responsibility on a China-IP machine.

All network I/O goes through ball_quant.adapters.http.get_json (proxy-aware, cassette-replay).
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ball_quant.adapters.http import get_json
from ball_quant.core.settlement import MatchOutcome
from ball_quant.models import TicaiOdds

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE_URL = "https://webapi.sporttery.cn"

DEFAULT_HEADERS: Dict[str, str] = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Referer": "https://www.sporttery.cn/",
    "Origin": "https://www.sporttery.cn",
    "Accept": "application/json, text/plain, */*",
}

# Regex that matches a properly-formed CRS score key, e.g. "s01s00"
_CRS_KEY_RE = re.compile(r"^s(\d{2})s(\d{2})$")

_TTG_KEYS = ("s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7")  # s7 = 7+


# ---------------------------------------------------------------------------
# Fetch layer
# ---------------------------------------------------------------------------

def fetch_odds_raw(
    pool_codes: Tuple[str, ...] = ("had", "hhad", "crs", "ttg", "hafu"),
    channel: str = "c",
    cache_path: Optional[Path] = None,
    timeout: int = 15,
) -> dict:
    """Fetch raw calculator response from sporttery.

    Passes cache_path to http.get_json: if the file exists it is read instead
    of making a live request (cassette replay).  If absent, the live response
    is written there for future replays.

    NOTE: live fetch fails with HTTP 567 outside China — use cache_path with a
    pre-captured cassette or run on a China-IP machine.
    """
    params = {
        "channel": channel,
        "poolCode": ",".join(pool_codes),
    }
    return get_json(
        BASE_URL,
        "/gateway/uniform/football/getMatchCalculatorV1.qry",
        params=params,
        headers=DEFAULT_HEADERS,
        timeout=timeout,
        cache_path=cache_path,
    )


def fetch_results_raw(
    begin: str,
    end: str,
    cache_path: Optional[Path] = None,
    timeout: int = 15,
) -> dict:
    """Fetch raw match-result response.

    begin/end: "YYYY-MM-DD".  Pages at up to 100 results.
    Assumed response structure: payload.value.list[{matchId, homeScore, awayScore}]
    (documented as analogous to the odds endpoint; path documented inline in parse_results).
    """
    params = {
        "matchBeginDate": begin,
        "matchEndDate": end,
        "pageSize": 100,
        "pageNo": 1,
    }
    return get_json(
        BASE_URL,
        "/gateway/uniform/football/getUniformMatchResultV1.qry",
        params=params,
        headers=DEFAULT_HEADERS,
        timeout=timeout,
        cache_path=cache_path,
    )


# ---------------------------------------------------------------------------
# Parse layer — odds
# ---------------------------------------------------------------------------

def parse_odds(raw: dict) -> List[TicaiOdds]:
    """Navigate matchInfoList → subMatchList and build TicaiOdds objects.

    Raises ValueError if the top-level structure is absent (broken payload).
    Skips / returns None for individual fields that are missing or non-numeric
    (field-level defensive parse) without swallowing structural breaks.
    """
    try:
        match_info_list = raw["payload"]["value"]["matchInfoList"]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            f"Sporttery odds payload missing expected structure "
            f"(payload.value.matchInfoList): {exc}"
        ) from exc

    results: List[TicaiOdds] = []
    for info_item in match_info_list:
        sub_list = info_item.get("subMatchList") or []
        for match in sub_list:
            odds = _parse_match(match)
            if odds is not None:
                results.append(odds)
    return results


def _parse_match(m: dict) -> Optional[TicaiOdds]:
    """Parse one subMatchList entry.  Returns None if match_id is absent."""
    match_id = m.get("matchId")
    if not match_id:
        return None

    return TicaiOdds(
        match_id=str(match_id),
        match_date=m.get("matchDate", ""),
        league=m.get("leagueAbbName", ""),
        home=m.get("homeTeamAbbName", ""),
        away=m.get("awayTeamAbbName", ""),
        match_num=m.get("matchNumStr"),
        spf=_parse_spf(m.get("had") or {}),
        handicap_line=_parse_handicap_line(m.get("hhad")),
        rqspf=_parse_spf(m.get("hhad") or {}),
        correct_score=_parse_crs(m.get("crs") or {}),
        total_goals=_parse_ttg(m.get("ttg") or {}),
        hafu=_parse_hafu(m.get("hafu") or {}),
    )


def _parse_spf(had: dict) -> Dict[str, float]:
    """Parse HAD / HHAD home-draw-away odds into {home, draw, away}."""
    result: Dict[str, float] = {}
    for out_key, api_key in (("home", "h"), ("draw", "d"), ("away", "a")):
        val = _safe_float(had.get(api_key))
        if val is not None:
            result[out_key] = val
    return result


def _parse_handicap_line(hhad: Optional[dict]) -> Optional[float]:
    """Extract the goalLine float from the HHAD block; None if absent or unparseable."""
    if not hhad:
        return None
    return _safe_float(hhad.get("goalLine"))


def _parse_crs(crs: dict) -> Dict[str, float]:
    """Parse correct-score dict.

    Standard keys: s{HH}s{AA} zero-padded, e.g. "s01s00" → "1-0".
    Any key not matching s\\d{2}s\\d{2} is kept under its raw key unchanged
    (these are the "other" catch-all buckets: home_other, draw_other, away_other,
    or any future variants the API adds).
    """
    result: Dict[str, float] = {}
    for k, v in crs.items():
        val = _safe_float(v)
        if val is None:
            continue
        m = _CRS_KEY_RE.match(k)
        if m:
            # Strip leading zeros: "s01s00" → "1-0"
            home_g = int(m.group(1))
            away_g = int(m.group(2))
            score_key = f"{home_g}-{away_g}"
        else:
            # Catch-all / "other" bucket — keep raw key (e.g. "home_other")
            score_key = k
        result[score_key] = val
    return result


def _parse_ttg(ttg: dict) -> Dict[int, float]:
    """Parse total-goals dict.

    Keys s0..s7 → int keys 0..7.  s7 represents 7+ goals.
    Missing keys are silently skipped.
    """
    result: Dict[int, float] = {}
    for i, k in enumerate(_TTG_KEYS):
        val = _safe_float(ttg.get(k))
        if val is not None:
            result[i] = val
    return result


def _parse_hafu(hafu: dict) -> Dict[str, float]:
    """Parse half/full dict.  Keys hh,hd,ha,dh,dd,da,ah,ad,aa preserved as-is."""
    result: Dict[str, float] = {}
    for k in ("hh", "hd", "ha", "dh", "dd", "da", "ah", "ad", "aa"):
        val = _safe_float(hafu.get(k))
        if val is not None:
            result[k] = val
    return result


# ---------------------------------------------------------------------------
# Parse layer — results
# ---------------------------------------------------------------------------

def parse_results(raw: dict) -> Dict[str, MatchOutcome]:
    """Parse getUniformMatchResultV1 response into {match_id: MatchOutcome}.

    Assumed path: raw["payload"]["value"]["list"][{matchId, homeScore, awayScore}].
    This assumption is documented because the exact results endpoint schema was
    labelled UNVERIFIED in the task spec; we parse defensively and skip entries
    with unparseable scores rather than raising.

    homeScore / awayScore are string integers in the cassette; we int() them.
    """
    try:
        result_list = raw["payload"]["value"]["list"]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            f"Sporttery results payload missing expected structure "
            f"(payload.value.list): {exc}"
        ) from exc

    outcomes: Dict[str, MatchOutcome] = {}
    for entry in result_list:
        match_id = entry.get("matchId")
        if not match_id:
            continue
        home_score = _safe_int(entry.get("homeScore"))
        away_score = _safe_int(entry.get("awayScore"))
        if home_score is None or away_score is None:
            # Incomplete result — skip rather than fabricate
            continue
        # Half-time scores — parsed when present; left None otherwise.
        # The sporttery results API does not guarantee an HT score field;
        # candidate field names observed in the wild: "halfHomeScore" /
        # "halfAwayScore" (getUniformMatchResultV1 schema [UNVERIFIED]).
        # When absent, hafu bets grade as VOID rather than fabricating a score.
        ht_home = _safe_int(entry.get("halfHomeScore"))
        ht_away = _safe_int(entry.get("halfAwayScore"))
        outcomes[str(match_id)] = MatchOutcome(
            match_id=str(match_id),
            home_score=home_score,
            away_score=away_score,
            settled=True,
            ht_home_score=ht_home,
            ht_away_score=ht_away,
        )
    return outcomes


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------

def load_odds(cache_path: Path) -> List[TicaiOdds]:
    """Fetch (or replay from cassette) and parse odds in one call."""
    raw = fetch_odds_raw(cache_path=cache_path)
    return parse_odds(raw)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_float(value: object) -> Optional[float]:
    """Parse a decimal-odds string/number to float; return None on failure."""
    if value is None:
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def _safe_int(value: object) -> Optional[int]:
    """Parse a score string/number to int; return None on failure."""
    if value is None:
        return None
    try:
        return int(str(value))
    except (ValueError, TypeError):
        return None
