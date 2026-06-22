"""Elo adapter — fetch World Football Elo ratings from eloratings.net.

Data source: https://eloratings.net/World.tsv
Format: tab-separated, NO header row.  Columns (positional, from ratings.js):
  0  current_rank   1  rank_tie   2  country_code (site-internal, e.g. "ES", "EN")
  3  current_elo    4+ per-period delta pairs, win/draw/loss counts (unused here)

Country codes are site-internal, NOT ISO 3166 (e.g. "EN" = England, not "GB").
We resolve them via https://eloratings.net/en.teams.tsv which maps code → full
English name (one name per line after the code, tab-separated).

WHY two-step: the per-country lookup resolves "EN" → "England" reliably without
hard-coding a map that silently drifts when countries rename/split.  If the teams
file is unavailable we fall back to the raw code string (callers see a warning).

Live-fetch status: VERIFIED accessible from macOS env (no geo-block, no auth).
"""
from __future__ import annotations

import csv
import logging
import os
from pathlib import Path
from typing import Dict, Optional
from urllib.request import Request, urlopen

from ball_quant.core.match_join import normalize_team

logger = logging.getLogger(__name__)

_WORLD_TSV_URL = "https://eloratings.net/World.tsv"
_TEAMS_TSV_URL = "https://eloratings.net/en.teams.tsv"

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/tab-separated-values,text/plain,*/*",
    "Referer": "https://eloratings.net/",
}

# Column indices in World.tsv (0-based, no header)
_COL_CODE = 2
_COL_ELO = 3


def _fetch_text(url: str, cache_path: Optional[Path] = None) -> str:
    """Fetch URL and return UTF-8 text.

    Cassette pattern: if cache_path exists, return its content and skip the
    network.  On live fetch, write result to cache_path for future replays.
    Raises urllib.error.URLError / OSError on network failure — no silent swallow.
    """
    if cache_path and cache_path.exists():
        return cache_path.read_text(encoding="utf-8", errors="replace")

    req = Request(url, headers=_HEADERS)
    with urlopen(req, timeout=30) as resp:
        raw = resp.read()

    # eloratings.net serves UTF-8 for .tsv files
    text = raw.decode("utf-8", errors="replace")

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(text, encoding="utf-8")

    return text


def _parse_teams_tsv(text: str) -> Dict[str, str]:
    """Parse en.teams.tsv → {site_code: full_english_name}.

    Format: code<TAB>name[<TAB>alias...] one country per line.
    We only use the first name column; aliases are ignored (they are alternate
    spellings, not what appears in World.tsv).
    """
    mapping: Dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) >= 2:
            code = parts[0].strip()
            name = parts[1].strip()
            if code and name:
                mapping[code] = name
    return mapping


def _parse_world_tsv(
    text: str,
    code_to_name: Dict[str, str],
) -> Dict[str, float]:
    """Parse World.tsv and return {canonical_team_name: elo_rating}.

    canonical_team_name is produced by normalize_team() from match_join so names
    join cleanly against the bundle's team set.  Rows where the Elo column is not
    a valid integer are skipped with a warning (malformed lines, comment rows).
    """
    ratings: Dict[str, float] = {}
    reader = csv.reader(text.splitlines(), delimiter="\t")
    for row in reader:
        if len(row) <= _COL_ELO:
            continue
        code = row[_COL_CODE].strip()
        elo_str = row[_COL_ELO].strip()
        try:
            elo = float(elo_str)
        except ValueError:
            logger.debug("Skipping unparseable Elo row: %r", row)
            continue

        # Resolve site code → display name, fall back to the raw code so the
        # caller gets something rather than silently losing the row.
        display_name = code_to_name.get(code, code)
        canonical = normalize_team(display_name)
        if not canonical:
            logger.debug("normalize_team returned empty for %r (%r)", display_name, code)
            continue

        ratings[canonical] = elo

    return ratings


def fetch_elo_ratings(
    cache_dir: Optional[str] = None,
) -> Dict[str, float]:
    """Fetch and return current World Elo ratings keyed by canonical team name.

    cache_dir: path to a directory for cassette files (same pattern as c500 adapter).
    Pass None to always fetch live.

    Returns: {canonical_team_name: float} — Elo values typically in [1200, 2200].

    Raises on any network or parse failure.  No fabricated fallback ratings are
    returned; the caller must decide how to handle a partial or failed fetch.
    """
    world_cache: Optional[Path] = None
    teams_cache: Optional[Path] = None
    if cache_dir:
        p = Path(cache_dir)
        world_cache = p / "elo_world.tsv"
        teams_cache = p / "elo_teams.tsv"

    teams_text = _fetch_text(_TEAMS_TSV_URL, teams_cache)
    code_to_name = _parse_teams_tsv(teams_text)
    if not code_to_name:
        logger.warning(
            "en.teams.tsv returned empty — Elo country codes will not be resolved "
            "to full names.  Ratings keyed by raw site code."
        )

    world_text = _fetch_text(_WORLD_TSV_URL, world_cache)
    ratings = _parse_world_tsv(world_text, code_to_name)

    if not ratings:
        raise ValueError(
            "Parsed zero Elo ratings from World.tsv — check network or cassette."
        )

    logger.info("Fetched %d Elo ratings", len(ratings))
    return ratings


def load_elo_ratings_from_fixtures(
    world_tsv_text: str,
    teams_tsv_text: str,
) -> Dict[str, float]:
    """Test helper: parse from in-memory strings instead of fetching.

    world_tsv_text: content of World.tsv (no header, tab-separated)
    teams_tsv_text: content of en.teams.tsv (code<TAB>name[<TAB>alias...])

    This function is the seam for unit tests — passes the same parser as the live
    path, zero network calls.
    """
    code_to_name = _parse_teams_tsv(teams_tsv_text)
    return _parse_world_tsv(world_tsv_text, code_to_name)
