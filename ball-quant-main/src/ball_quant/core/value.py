from __future__ import annotations

from typing import List, Optional

from ball_quant.models import Branch, EventMarketMatrix, MatchSP, Selection, TeamFacts


def selections_from_branches(
    match: MatchSP,
    matrix: EventMarketMatrix,
    facts: TeamFacts,
    branches: List[Branch],
) -> List[Selection]:
    result: List[Selection] = []
    sp_lookup = {
        ("spf", "home"): match.spf_home,
        ("spf", "draw"): match.spf_draw,
        ("spf", "away"): match.spf_away,
        (f"rq({match.handicap:+d})", "home"): match.rq_home,
        (f"rq({match.handicap:+d})", "draw"): match.rq_draw,
        (f"rq({match.handicap:+d})", "away"): match.rq_away,
    }
    avg_spread, total_liquidity = matrix.liquidity_snapshot()
    for branch in branches:
        sp = sp_lookup.get((branch.play, branch.outcome), 0.0)
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
) -> float:
    score = 0.55
    if source.startswith("polymarket"):
        score += 0.12
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
