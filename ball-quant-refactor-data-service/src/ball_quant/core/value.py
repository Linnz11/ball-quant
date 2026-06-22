from __future__ import annotations

import re
from typing import List, Optional

from ball_quant.core.params import DEFAULT_PARAMS, StrategyParams
from ball_quant.models import Branch, EventMarketMatrix, MatchSP, Selection, SettlementKey, TeamFacts


# Minimum viable ask/probability for a Polymarket quote to be usable as a
# bet price.  Polymarket's CLOB minimum tick is 0.001 (0.1 cents per dollar).
# Quotes at the floor (ask ≤ 0.01) are near-settled / effectively worthless
# tokens; pricing 1/0.001 = 1000-odds against a non-negligible model prob
# produces phantom edges.  We require ask ≥ 0.02 (≤ 50× odds) before treating
# the ask as a real tradeable price.  The same floor applies to the probability
# mid-price fallback so that 1/probability is also capped at ≤ 50×.
_MIN_VIABLE_PRICE: float = 0.02

# RQ play strings look like "rq(-1)" or "rq(+2)".  The signed integer inside
# the parens is the handicap applied to the home side (same sign convention as
# MatchSP.handicap and handicap_result).
_RQ_PATTERN = re.compile(r"^rq\(([+-]?\d+)\)$")

# totals play: "totals(2.5)"
_TOTALS_PATTERN = re.compile(r"^totals\((\d+(?:\.\d+)?)\)$")

# team_total play: "team_total(Netherlands,1.5)"
_TEAM_TOTAL_PATTERN = re.compile(r"^team_total\((.+),(\d+(?:\.\d+)?)\)$")

# Score-derivable market types that match_branches generates.
# play value -> SettlementKey market_type mapping.
_PLAY_TO_MARKET_TYPE = {
    "spf": "spf",
    # rq(...) keys are matched dynamically below
}

# causal.py CAUSAL_PROFILES keys that are NOT derivable from home/away score.
_NON_SCORE_CATEGORIES: frozenset = frozenset(
    {
        "total_corners",
        "team_total_corners",
        "first_half_total_corners",
        "second_half_total_corners",
        "corners_odd_even",
        "first_corner",
        "player_goals",
        "player_shots",
        "player_shots_on_target",
        "player_assists",
        "player_goal_contributions",
        "goalkeeper_saves",
        "starting_lineup",
        "group_winner",
        "group_advancement",
        "group_position",
        "stage_advancement",
        "stage_elimination",
        "tournament_winner",
        "team_prop",
        "player_award",
        "player_h2h",
        "player_future",
        "continent_future",
        "record_future",
        "culture_future",
    }
)


def _settlement_key_for_branch(branch: Branch) -> Optional[SettlementKey]:
    """Derive a SettlementKey from the Branch's play/outcome/tags.

    Uses the REAL play/outcome string values produced by probability.py:
      - play="spf",  outcome="home"|"draw"|"away"
      - play="rq(<signed_int>)", outcome="home"|"draw"|"away"
      - play="totals(<line>)", outcome="over"|"under"
      - play="btts", outcome="yes"|"no"
      - play="team_total(<team>,<line>)", outcome="over"|"under"
      - play="correct_score", outcome="<h>-<a>"

    Returns None only when the play type is explicitly non-score-derivable,
    so the downstream grader can route to poly_resolutions or return VOID.
    """
    play = branch.play
    outcome = branch.outcome

    # SPF (moneyline win/draw/lose) — graded via handicap_result(h,a,0)
    if play == "spf":
        if outcome in ("home", "draw", "away"):
            return SettlementKey(market_type="spf", side=outcome)

    # RQ (Asian-style handicap) — play is "rq(-1)", "rq(+2)", etc.
    rq_match = _RQ_PATTERN.match(play)
    if rq_match:
        handicap_val = int(rq_match.group(1))
        if outcome in ("home", "draw", "away"):
            return SettlementKey(
                market_type="handicap",
                side=outcome,
                line=float(handicap_val),
            )

    # Totals (over/under total goals) — play="totals(2.5)", outcome="over"|"under"
    totals_match = _TOTALS_PATTERN.match(play)
    if totals_match:
        line = float(totals_match.group(1))
        if outcome in ("over", "under"):
            return SettlementKey(market_type="totals", side=outcome, line=line)

    # BTTS — play="btts", outcome="yes"|"no"
    if play == "btts":
        if outcome in ("yes", "no"):
            return SettlementKey(market_type="btts", side=outcome)

    # Team total — play="team_total(<team>,<line>)", outcome="over"|"under"
    tt_match = _TEAM_TOTAL_PATTERN.match(play)
    if tt_match:
        team = tt_match.group(1)
        line = float(tt_match.group(2))
        if outcome in ("over", "under"):
            # entity "home"/"away" lets settlement.py resolve via h or a.
            # We carry the raw team name; adapters normalise to "home"/"away"
            # if needed.  settlement._resolve_team_goals handles "home"/"away"
            # directly, so pass the literal team name so the caller can enrich.
            return SettlementKey(market_type="team_total", side=outcome, line=line, entity=team)

    # Correct score — play="correct_score", outcome="<h>-<a>"
    if play == "correct_score":
        # outcome is already "h-a" format; settlement._parse_score_side handles it.
        return SettlementKey(market_type="correct_score", side=outcome)

    # Non-score props — will resolve via poly_resolutions or VOID
    if play in _NON_SCORE_CATEGORIES:
        return SettlementKey(market_type=play, side=outcome)

    # Unknown play type — surface it so it grades VOID without fabrication
    return SettlementKey(market_type=play, side=outcome)


def _price_from_matrix(branch: Branch, matrix: EventMarketMatrix) -> Optional[float]:
    """Return decimal odds for a branch whose price lives in the market matrix.

    For score-derivable extras (totals, btts, team_total, correct_score) we
    fetch the Polymarket quote and compute:
        decimal_odds = 1 / ask          (prefer ask — what you actually pay)
        fallback:      1 / probability  (mid-price when ask is None)

    Using ask rather than mid avoids overstating edge by the half-spread.
    Returns None when no usable quote exists (caller skips the branch).
    SPF and handicap prices come from MatchSP fields, not this helper.
    """
    from ball_quant.core.probability import best_usable_quote, quote_is_usable

    play = branch.play
    outcome = branch.outcome

    # Totals: look up the quote whose outcome matches "over <line>" / "under <line>"
    totals_match = _TOTALS_PATTERN.match(play)
    if totals_match:
        line = float(totals_match.group(1))
        # Try "over <line>" / "under <line>" style outcomes first; fall through to
        # iterating all total_goals quotes to find the matching line+side.
        for q in matrix.quotes("total_goals"):
            if not quote_is_usable(q):
                continue
            if q.outcome.startswith("not:"):
                continue
            # Match by explicit line field or by parsing the outcome string
            q_line = q.line
            if q_line is None:
                import re as _re
                lm = _re.search(r"(\d+(?:\.\d+)?)", q.outcome)
                q_line = float(lm.group(1)) if lm else None
            if q_line is None or abs(q_line - line) > 1e-9:
                continue
            q_side = "over" if "over" in q.outcome.lower() else "under" if "under" in q.outcome.lower() else None
            if q_side != outcome:
                continue
            # Found the quote — price at ask (must be a real tradeable price,
            # not Polymarket's floor-tick phantom), fall back to mid-price.
            if q.ask is not None and _MIN_VIABLE_PRICE <= q.ask <= 1:
                return 1.0 / q.ask
            if q.probability is not None and _MIN_VIABLE_PRICE <= q.probability <= 1:
                return 1.0 / q.probability
        return None

    # BTTS
    if play == "btts":
        q = best_usable_quote(matrix, "btts", outcome)
        if q is None:
            return None
        if q.ask is not None and _MIN_VIABLE_PRICE <= q.ask <= 1:
            return 1.0 / q.ask
        if q.probability is not None and _MIN_VIABLE_PRICE <= q.probability <= 1:
            return 1.0 / q.probability
        return None

    # Team total: outcome encodes team+line in play string
    tt_match = _TEAM_TOTAL_PATTERN.match(play)
    if tt_match:
        team = tt_match.group(1)
        line = float(tt_match.group(2))
        from ball_quant.models import normalize_key
        for q in matrix.quotes("team_total"):
            if not quote_is_usable(q):
                continue
            if q.outcome.startswith("not:"):
                continue
            # Match team via entity or outcome text
            entity_key = normalize_key(q.entity or "")
            if entity_key and entity_key != normalize_key(team):
                continue
            if not entity_key and normalize_key(team) not in normalize_key(q.outcome):
                continue
            # Match line
            q_line = q.line
            if q_line is None:
                import re as _re
                lm = _re.search(r"(\d+(?:\.\d+)?)", q.outcome)
                q_line = float(lm.group(1)) if lm else None
            if q_line is None or abs(q_line - line) > 1e-9:
                continue
            q_side = "over" if "over" in q.outcome.lower() else "under" if "under" in q.outcome.lower() else None
            if q_side != outcome:
                continue
            if q.ask is not None and _MIN_VIABLE_PRICE <= q.ask <= 1:
                return 1.0 / q.ask
            if q.probability is not None and _MIN_VIABLE_PRICE <= q.probability <= 1:
                return 1.0 / q.probability
        return None

    # Correct score: outcome is "h-a"
    if play == "correct_score":
        q = best_usable_quote(matrix, "correct_score", outcome)
        if q is None:
            # Also try "h-a" in question+outcome concatenation (some quotes use
            # question as the score slug)
            from ball_quant.core.probability import parse_score
            for q2 in matrix.quotes("correct_score"):
                if not quote_is_usable(q2):
                    continue
                if q2.outcome.startswith("not:"):
                    continue
                score = parse_score(f"{q2.question} {q2.outcome}")
                if score is not None and f"{score[0]}-{score[1]}" == outcome:
                    q = q2
                    break
        if q is None:
            return None
        if q.ask is not None and _MIN_VIABLE_PRICE <= q.ask <= 1:
            return 1.0 / q.ask
        if q.probability is not None and _MIN_VIABLE_PRICE <= q.probability <= 1:
            return 1.0 / q.probability
        return None

    return None


def selections_from_branches(
    match: MatchSP,
    matrix: EventMarketMatrix,
    facts: TeamFacts,
    branches: List[Branch],
    params: StrategyParams = DEFAULT_PARAMS,
) -> List[Selection]:
    result: List[Selection] = []
    # Hard prices for spf and handicap come from MatchSP (exchange-verified).
    sp_legacy = {
        ("spf", "home"): match.spf_home,
        ("spf", "draw"): match.spf_draw,
        ("spf", "away"): match.spf_away,
        (f"rq({match.handicap:+d})", "home"): match.rq_home,
        (f"rq({match.handicap:+d})", "draw"): match.rq_draw,
        (f"rq({match.handicap:+d})", "away"): match.rq_away,
    }
    avg_spread, total_liquidity = matrix.liquidity_snapshot()
    for branch in branches:
        # For spf/handicap use the MatchSP fields (unchanged legacy path).
        # For all new market types derive the price from the matrix quote.
        legacy_sp = sp_legacy.get((branch.play, branch.outcome))
        if legacy_sp is not None:
            sp = legacy_sp
        else:
            # New market: look up price from the Polymarket matrix quote.
            # Skip the branch entirely if no usable quote is found (no fabrication).
            matrix_sp = _price_from_matrix(branch, matrix)
            if matrix_sp is None:
                continue
            sp = matrix_sp
        if branch.probability is None or branch.probability <= 0 or sp <= 1:
            continue
        p = branch.probability
        fair_odds = 1.0 / p if p > 0 else float("inf")
        break_even = 1.0 / sp
        edge = p * sp - 1.0
        kelly = kelly_fraction(p, sp)
        confidence = confidence_score(
            probability=p,
            spread=avg_spread,
            liquidity=total_liquidity,
            facts_adjustment=facts.confidence_adjustment,
            source=branch.source,
            params=params,
        )
        result.append(
            Selection(
                match_id=match.match_id,
                home=match.home,
                away=match.away,
                play=branch.play,
                outcome=branch.outcome,
                condition=branch.condition,
                probability=p,
                sp=sp,
                fair_odds=fair_odds,
                break_even=break_even,
                edge=edge,
                kelly=kelly,
                confidence=confidence,
                risk_label=risk_label(edge, confidence, branch.tags),
                tags=branch.tags,
                source=branch.source,
                settlement_key=_settlement_key_for_branch(branch),
            )
        )
    return result


def kelly_fraction(probability: float, decimal_odds: float) -> float:
    b = decimal_odds - 1.0
    if b <= 0:
        return 0.0
    q = 1.0 - probability
    return max(0.0, (probability * b - q) / b)


def confidence_score(
    probability: float,
    spread: Optional[float],
    liquidity: Optional[float],
    facts_adjustment: float,
    source: str,
    params: StrategyParams = DEFAULT_PARAMS,
) -> float:
    score = params.conf_base
    if source.startswith("polymarket"):
        score += params.conf_poly_bonus
    if probability >= 0.5:
        score += 0.06
    if spread is not None:
        if spread <= 0.03:
            score += 0.08
        elif spread >= 0.10:
            score -= 0.12
    else:
        score -= 0.04
    if liquidity is not None:
        if liquidity >= 10000:
            score += 0.08
        elif liquidity < 1000:
            score -= 0.10
    else:
        score -= 0.06
    score += facts_adjustment
    return max(0.0, min(1.0, score))


def risk_label(edge: float, confidence: float, tags: List[str]) -> str:
    if "exact_margin" in tags:
        return "精准分支"
    if edge > 0.08 and confidence >= 0.6:
        return "价值保留"
    if edge >= 0 and confidence >= 0.45:
        return "可防守"
    if edge < -0.12:
        return "赔率不足"
    return "观察"
