from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

from ball_quant.core.handicap import handicap_result
from ball_quant.models import Selection, SettlementKey

# Grade literals
WIN = "WIN"
LOSS = "LOSS"
VOID = "VOID"

Grade = str  # "WIN" | "LOSS" | "VOID"

# Market types whose outcome is derivable purely from (home_score, away_score).
_SCORE_DERIVABLE: frozenset = frozenset(
    {"spf", "handicap", "correct_score", "totals", "team_total", "btts", "moneyline_not"}
)


@dataclass(frozen=True)
class MatchOutcome:
    match_id: str
    home_score: int
    away_score: int
    settled: bool = True
    void: bool = False
    # Polymarket token/market_id -> "YES" | "NO" for non-score props.
    # Key must match SettlementKey.entity or a market_id used by the caller.
    poly_resolutions: Dict[str, str] = field(default_factory=dict)
    # Half-time scores — required for hafu (半全场) grading; None when the
    # result feed does not include HT data (grade returns VOID in that case).
    ht_home_score: Optional[int] = None
    ht_away_score: Optional[int] = None


def grade(
    selection_or_key: Union[Selection, SettlementKey],
    outcome: MatchOutcome,
) -> Grade:
    """Grade a single bet leg against a final MatchOutcome.

    Accepts either a Selection (uses its settlement_key) or a bare SettlementKey.
    Returns "WIN", "LOSS", or "VOID".  VOID is always explicit — never fabricated.
    """
    if isinstance(selection_or_key, Selection):
        key = selection_or_key.settlement_key
    else:
        key = selection_or_key

    # Void match or absent key -> VOID without fabrication
    if outcome.void:
        return VOID
    if key is None:
        return VOID

    return _grade_key(key, outcome)


def grade_selections(
    selections: List[Selection],
    outcome: MatchOutcome,
) -> List[Tuple[Selection, Grade]]:
    """Grade every selection in a list; returns (selection, grade) pairs."""
    return [(sel, grade(sel, outcome)) for sel in selections]


# ---------------------------------------------------------------------------
# Internal routing
# ---------------------------------------------------------------------------

def _grade_key(key: SettlementKey, outcome: MatchOutcome) -> Grade:
    h = outcome.home_score
    a = outcome.away_score
    mt = key.market_type
    side = key.side

    if mt == "spf":
        # handicap_result(h, a, 0) replicates spf logic (home wins if h>a, etc.)
        # Reuses handicap.py:8 — do NOT duplicate.
        result = handicap_result(h, a, 0)
        return WIN if result == side else LOSS

    if mt == "handicap":
        if key.line is None:
            return VOID
        # handicap_result accepts int; line stored as float for generality.
        # Integer-line push ("draw" per handicap_result semantics) is handled
        # by handicap_result itself — we match its output, never override it.
        result = handicap_result(h, a, int(key.line))
        return WIN if result == side else LOSS

    if mt == "correct_score":
        # side format is "h-a" (e.g. "2-1")
        parsed = _parse_score_side(side)
        if parsed is None:
            return VOID
        target_h, target_a = parsed
        return WIN if (h == target_h and a == target_a) else LOSS

    if mt == "totals":
        if key.line is None:
            return VOID
        total = h + a
        line = key.line
        if total == line:
            # Integer line push — e.g. over/under 2 when goals=2 -> VOID
            return VOID
        if side == "over":
            return WIN if total > line else LOSS
        if side == "under":
            return WIN if total < line else LOSS
        return VOID

    if mt == "team_total":
        if key.line is None:
            return VOID
        # entity names the team ("home"/"away" or team name); side is "over"/"under"
        # We normalise: "home" entity -> h, "away" entity -> a.
        # Callers may also embed team-name in entity; fall back to VOID if unknown.
        team_goals = _resolve_team_goals(key.entity, h, a)
        if team_goals is None:
            return VOID
        total = team_goals
        line = key.line
        if total == line:
            return VOID  # integer-line push same as totals
        if side == "over":
            return WIN if total > line else LOSS
        if side == "under":
            return WIN if total < line else LOSS
        return VOID

    if mt == "btts":
        both_scored = h > 0 and a > 0
        if side == "yes":
            return WIN if both_scored else LOSS
        if side == "no":
            return WIN if not both_scored else LOSS
        return VOID

    if mt == "moneyline_not":
        # Inverse SPF — e.g. "not_home" wins if result is draw or away.
        # side stores the excluded outcome ("home"/"draw"/"away").
        result = handicap_result(h, a, 0)
        return WIN if result != side else LOSS

    if mt == "hafu":
        # 半全场 (half-time / full-time double result).
        # side is a 2-char key: first char = HT result, second = FT result.
        # h/d/a encodes home-win / draw / away-win for each half.
        # Requires HT scores; if absent the bet cannot be graded → VOID.
        if outcome.ht_home_score is None or outcome.ht_away_score is None:
            return VOID
        if len(side) != 2 or side[0] not in "hda" or side[1] not in "hda":
            return VOID
        ht_result = handicap_result(outcome.ht_home_score, outcome.ht_away_score, 0)
        ft_result = handicap_result(h, a, 0)
        _result_char = {"home": "h", "draw": "d", "away": "a"}
        actual_key = _result_char.get(ht_result, "?") + _result_char.get(ft_result, "?")
        return WIN if actual_key == side else LOSS

    # Non-score prop (player_*, corners, futures, etc.)
    # Must be resolved via poly_resolutions; if absent -> VOID (no fabrication).
    resolution_key = key.entity or key.side
    resolution = outcome.poly_resolutions.get(resolution_key)
    if resolution is None:
        return VOID
    if resolution == "YES":
        return WIN
    if resolution == "NO":
        return LOSS
    return VOID


def _parse_score_side(side: str) -> Optional[Tuple[int, int]]:
    """Parse "h-a" string into (home_goals, away_goals); returns None on failure."""
    parts = side.split("-")
    if len(parts) != 2:
        return None
    try:
        return int(parts[0]), int(parts[1])
    except ValueError:
        return None


def _resolve_team_goals(entity: Optional[str], home_goals: int, away_goals: int) -> Optional[int]:
    """Return the goal count for the named entity.

    entity is expected to be "home", "away", or a team name stored in SettlementKey.
    Returns None if we cannot determine which side to use.
    """
    if entity is None:
        return None
    if entity.lower() == "home":
        return home_goals
    if entity.lower() == "away":
        return away_goals
    # Caller may pass a team name — we cannot resolve without match context here.
    # The adapters/backtest layer should normalise to "home"/"away" before grading.
    return None
