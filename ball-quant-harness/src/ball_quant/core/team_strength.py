"""team_strength.py — derive team strength signals from Polymarket futures markets.

CAUSAL LOGIC
------------
A futures market (tournament_winner / group_winner / group_advancement) is a
real-money prediction market aggregating the collective real-money view of a
team's probability of achieving the outcome — independent of any single match
line.  After devigging (which removes the bookmaker overround), the remaining
probability is the market's best estimate of true team strength on the relevant
dimension.

- strength_win  ← devigged tournament_winner probability across all teams
- strength_advance ← devigged group_advancement probability (or group_winner
                     if group_advancement quotes are unavailable)
- strength_rank ← ordinal rank by strength_win (1 = highest)

Keys are canonicalised through normalize_team so they match the rest of the KG.

NON-TEAM OUTCOME FILTERING (tournament_winner)
----------------------------------------------
Polymarket hosts several tournament_winner-tagged markets that are NOT
national-team markets:
  - "Which continent will win the World Cup?" → outcomes are confederation
    slugs (europeuefa, northamericaconcacaf, africacaf, asiaafc, oceaniaofc,
    southamericaconmebol, anothercontinent …)
  - "World Cup Winner" market also contains placeholder slots for teams that
    have not yet qualified, named "Team AM", "Team AI" etc. → normalized to
    teamam, teamai … (pattern: "team" followed by 2 uppercase letters)
  - "Any Other Team" / "anyotherteam" catch-all bucket

All of these must be stripped before devigging so the devig pool contains
only genuine national teams (~48 qualified nations).  The discriminator is
applied at the canonical (normalized) name level:
  1. Confederation slugs — name contains any of the confederation substrings
  2. Placeholder team slots — name matches /^team[a-z]{2}$/ (exactly 6 chars
     starting with "team")
  3. Catch-all bucket — name is "anyotherteam" or "anyother" or "thefield"
     or "otherteam"
"""
from __future__ import annotations

import datetime
import re
from typing import Dict, List, Optional

from ball_quant.models import MarketQuote
from ball_quant.core.match_join import normalize_team
from ball_quant.core.knowledge_graph import Team, upsert_team

# We need the raw list-based devig helpers; re-implement proportional and
# re-use shin_devig from probability.py (stdlib-only, no 3rd-party).
from ball_quant.core.probability import shin_devig


# ---------------------------------------------------------------------------
# Non-national-team discriminator
# ---------------------------------------------------------------------------

# Confederation/region substrings that appear in normalized outcome names from
# "Which continent will win?" and similar meta markets.
# Substring search is safe for these because no real national team name contains
# these strings:
#   - "uefa", "conmebol", "concacaf", "afc", "ofc", "ocf": pure acronyms
#   - "oceania": no real team is named "oceania*"
#   - "northamerica", "southamerica", "centralamerica": composite region names
# Deliberately excluded from substring search (would false-positive on real teams):
#   - "africa" → would match "southafrica" (real team)
#   - "europe" → would match "northern/eastern europe" hypotheticals
#   - "caf" → not in "southafrica"; "africacaf" is already caught by "afc"...
#              actually "africacaf" contains "afc" → caught.
_CONFEDERATION_SLUG_SUBSTRINGS = frozenset({
    "uefa",           # europeuefa
    "conmebol",       # southamericaconmebol
    "concacaf",       # northamericaconcacaf
    "afc",            # asiaafc — NOTE: "southafrica" does NOT contain "afc"
    "ofc",            # oceaniaofc (correct Polymarket spelling)
    "ocf",            # oceaniaocf (Polymarket typo: OCF instead of OFC)
    "oceania",        # any "oceania*" form; no real team starts with "oceania"
    "northamerica",   # northamericaconcacaf
    "southamerica",   # southamericaconmebol
    "centralamerica", # any CONCACAF variant
})

# Exact-match non-team slugs (catch-all buckets, meta entries, and confederation
# names that cannot be safely detected by substring due to real-team collisions).
_NON_TEAM_EXACT = frozenset({
    "anyotherteam",
    "anyother",
    "thefield",
    "otherteam",
    "anothercontinent",
    "another",
    "africacaf",       # "Which continent?" Africa outcome — not caught by "afc" alone
                       # because "africacaf" contains "afc" but we must confirm
                       # (it does: a-f-c-a-f → contains "afc" at index 4). Belt-and-suspenders.
    "europeuefa",      # "Which continent?" Europe outcome — belt-and-suspenders
    "asia",            # standalone "Asia" outcome — exact match safe (no real team)
    "asiaafc",         # "Asia (AFC)"
    "europe",          # standalone "Europe" — exact match
    "africa",          # standalone "Africa" — exact match; no real team named exactly "africa"
})

# Pattern for unresolved qualification slots: "team" + exactly 2 alpha chars
# e.g. "Team AM" → "teamam", "Team AI" → "teamai"
# Real national teams never have this shape (shortest real slug is "iran", 4 chars
# that does NOT start with "team").
_PLACEHOLDER_TEAM_RE = re.compile(r"^team[a-z]{2}$")


def _is_non_team(canonical: str) -> bool:
    """Return True if the normalized outcome name is NOT a national team.

    Covers:
    - Confederation/region outcomes from "which continent wins" market
    - Unresolved qualification placeholder slots ("Team AM" → "teamam")
    - Meta catch-all buckets ("Any Other Team", "The Field", …)
    """
    if canonical in _NON_TEAM_EXACT:
        return True
    # Confederation slug substring in the name (e.g. "europeuefa", "southamericaconmebol",
    # "oceaniaocf" — includes Polymarket's OCF typo for OFC).
    for slug in _CONFEDERATION_SLUG_SUBSTRINGS:
        if slug in canonical:
            return True
    # Placeholder pattern: exactly "team" + 2 letters
    if _PLACEHOLDER_TEAM_RE.match(canonical):
        return True
    return False


# ---------------------------------------------------------------------------
# Internal devig dispatcher (list-based, not score-distribution-based)
# ---------------------------------------------------------------------------

def _proportional_devig(raw: List[float]) -> List[float]:
    """Proportional (normalisation) devig: divide each probability by the booksum."""
    total = sum(raw)
    if total <= 0:
        return [0.0] * len(raw)
    return [p / total for p in raw]


def _devig_list(raw: List[float], method: str) -> List[float]:
    """Dispatch to proportional or Shin devig.  Returns a list that sums to 1.

    Proportional devig scales all probs uniformly — correct when the
    overround is spread evenly.  Shin (1992) devig accounts for informed
    insider trading and redistributes slightly from short-priced favourites
    to longer-priced runners — more accurate for multi-outcome futures where
    tail teams are systematically underpriced by bookmakers.

    Shin devig requires booksum > 1 (overround).  When booksum ≤ 1 the
    input is already fair-or-under-priced — we fall back to proportional
    normalisation so the output still sums to 1 regardless of method choice.
    """
    if not raw or all(p <= 0 for p in raw):
        return [0.0] * len(raw)
    if method == "shin":
        booksum = sum(raw)
        if booksum <= 1.0:
            # No vig to remove — just normalise proportionally.
            return _proportional_devig(raw)
        return shin_devig(raw)
    # Default: proportional
    return _proportional_devig(raw)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def strength_from_futures(
    quotes: List[MarketQuote],
    devig: str = "proportional",
) -> Dict[str, dict]:
    """Derive team strength dict from a flat list of MarketQuote objects.

    Iterates quotes, groups by category, deviggs each category independently,
    then returns a dict keyed by canonical team name:

        {
            "brazil": {"strength_win": 0.18, "strength_rank": 1,
                       "strength_advance": 0.82},
            ...
        }

    Only tournament_winner, group_winner, and group_advancement categories
    are consumed.  Other categories are silently ignored.
    """
    # Collect raw probabilities per category × team
    tw_quotes: List[MarketQuote] = []   # tournament_winner
    ga_quotes: List[MarketQuote] = []   # group_advancement
    gw_quotes: List[MarketQuote] = []   # group_winner (fallback for advance)

    for q in quotes:
        if q.probability is None:
            continue
        # Filter complement / "not-X" outcomes before any devigging.
        # Polymarket binary markets ("Will Italy win?") produce a NO-side token
        # whose outcome normalises to e.g. "notitaly" or "not:italy".  These are
        # hedge complements, not real team entities; including them in the devig
        # pool pollutes the entity set and collapses all real-team magnitudes.
        # Primary guard: the adapter already sets is_complement=True for these.
        # Secondary guard: outcome key starts with "not" after normalize_team —
        # catches any future naming variants the adapter hasn't flagged yet.
        if q.is_complement:
            continue
        canonical = normalize_team(q.outcome)
        if canonical.startswith("not"):
            continue
        # Drop non-national-team outcomes: confederation/region slugs from
        # "which continent wins" market, unresolved qualification placeholders
        # ("Team AM" → teamam), and catch-all buckets ("Any Other Team").
        # Applied only to tournament_winner to avoid false-drops in group markets
        # where the question is different.
        if q.category == "tournament_winner" and _is_non_team(canonical):
            continue
        if q.category == "tournament_winner":
            tw_quotes.append(q)
        elif q.category == "group_advancement":
            ga_quotes.append(q)
        elif q.category == "group_winner":
            gw_quotes.append(q)

    result: Dict[str, dict] = {}

    # --- Tournament winner → strength_win + strength_rank -------------------
    if tw_quotes:
        # Deduplicate by canonical team name, keeping the highest raw probability
        # entry for each team.  Polymarket sometimes lists the same national team
        # twice (different question phrasings map to the same canonical key).
        # Without dedup the rank_map dict overrides with the lower-probability
        # duplicate, causing rank gaps in the final output.
        seen: Dict[str, MarketQuote] = {}
        for q in tw_quotes:
            can = normalize_team(q.outcome)
            if can not in seen or (q.probability or 0) > (seen[can].probability or 0):
                seen[can] = q
        tw_quotes = list(seen.values())

        teams = [normalize_team(q.outcome) for q in tw_quotes]
        raw = [q.probability for q in tw_quotes]
        fair = _devig_list(raw, devig)
        # Sort descending for ranking (best = rank 1)
        ranked = sorted(zip(teams, fair), key=lambda x: -x[1])
        rank_map = {t: i + 1 for i, (t, _) in enumerate(ranked)}
        for team, p in zip(teams, fair):
            if team not in result:
                result[team] = {}
            result[team]["strength_win"] = p
            result[team]["strength_rank"] = rank_map[team]

    # --- Group advancement (preferred) or group_winner (fallback) → strength_advance ---
    advance_quotes = ga_quotes if ga_quotes else gw_quotes
    if advance_quotes:
        # Group markets quote each team independently (binary yes/no per team),
        # so devig over the full multi-team pool (sum should be ~2.0 for 4-team
        # groups with 2 advancing).  Proportional devig makes sense here.
        teams = [normalize_team(q.outcome) for q in advance_quotes]
        raw = [q.probability for q in advance_quotes]
        fair = _devig_list(raw, devig)
        for team, p in zip(teams, fair):
            if team not in result:
                result[team] = {}
            result[team]["strength_advance"] = p

    return result


def update_kg_from_futures(
    quotes: List[MarketQuote],
    kg: Dict,
    devig: str = "proportional",
) -> Dict:
    """Derive strength signals from futures quotes and upsert them into the KG.

    Qualitative fields already in the KG are preserved — only strength_win,
    strength_advance, strength_rank, and updated_at are overwritten.

    Returns the kg dict (mutated in place) for chaining.
    """
    strengths = strength_from_futures(quotes, devig=devig)
    now = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    for canonical_name, signals in strengths.items():
        team = Team(
            name=canonical_name,
            strength_win=signals.get("strength_win"),
            strength_advance=signals.get("strength_advance"),
            strength_rank=signals.get("strength_rank"),
            updated_at=now,
        )
        upsert_team(team, kg)
    return kg
