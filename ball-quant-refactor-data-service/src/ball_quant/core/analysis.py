from __future__ import annotations

from typing import Iterable, List

from ball_quant.core.params import DEFAULT_PARAMS, StrategyParams
from ball_quant.core.probability import build_probability_context, match_branches
from ball_quant.core.value import selections_from_branches
from ball_quant.models import EventMarketMatrix, MatchAnalysis, MatchSP, TeamFacts


def analyze_match(
    match: MatchSP,
    matrix: EventMarketMatrix,
    facts: TeamFacts,
    params: StrategyParams = DEFAULT_PARAMS,
) -> MatchAnalysis:
    context = build_probability_context(match, matrix, params=params)
    branches = match_branches(match, context)
    selections = selections_from_branches(match, matrix, facts, branches, params=params)
    sp_lookup = {
        ("spf", "home"): match.spf_home,
        ("spf", "draw"): match.spf_draw,
        ("spf", "away"): match.spf_away,
        (f"rq({match.handicap:+d})", "home"): match.rq_home,
        (f"rq({match.handicap:+d})", "draw"): match.rq_draw,
        (f"rq({match.handicap:+d})", "away"): match.rq_away,
    }
    deleted_paths = [
        f"{branch.play}:{branch.outcome} 概率缺失或体彩 SP 不可用"
        for branch in branches
        if branch.probability is None
    ]
    deleted_paths.extend(
        f"{branch.play}:{branch.outcome} 体彩 SP 缺失，仅可做 Polymarket 概率观察，不能计算 EV/RR"
        for branch in branches
        if branch.probability is not None and sp_lookup.get((branch.play, branch.outcome), 0.0) <= 1
    )
    return MatchAnalysis(
        match=match,
        matrix=matrix,
        facts=facts,
        branches=branches,
        selections=selections,
        deleted_paths=deleted_paths,
    )


def flatten_selections(analyses: Iterable[MatchAnalysis]) -> List:
    result = []
    for analysis in analyses:
        result.extend(analysis.selections)
    return result
