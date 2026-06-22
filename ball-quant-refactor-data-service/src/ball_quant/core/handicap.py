from __future__ import annotations

from typing import Literal

HandicapOutcome = Literal["home", "draw", "away"]


def handicap_result(home_score: int, away_score: int, handicap: int) -> HandicapOutcome:
    adjusted_home_score = home_score + handicap
    if adjusted_home_score > away_score:
        return "home"
    if adjusted_home_score == away_score:
        return "draw"
    return "away"


def handicap_condition(home: str, away: str, handicap: int, outcome: HandicapOutcome) -> str:
    target_margin = -handicap
    if outcome == "home":
        if target_margin >= 0:
            return f"{home} 净胜 {target_margin + 1} 球或更多"
        return f"{home} 不败" if target_margin == -1 else f"{home} 最多输 {abs(target_margin) - 1} 球"
    if outcome == "draw":
        if target_margin > 0:
            return f"{home} 刚好赢 {target_margin} 球"
        if target_margin == 0:
            return "双方打平"
        return f"{home} 刚好输 {abs(target_margin)} 球"
    if target_margin > 0:
        return f"{home} 净胜 {target_margin - 1} 球以内或不胜"
    if target_margin == 0:
        return f"{home} 不胜"
    return f"{home} 输 {abs(target_margin) + 1} 球或更多"


def spf_condition(home: str, away: str, outcome: HandicapOutcome) -> str:
    if outcome == "home":
        return f"{home} 胜"
    if outcome == "draw":
        return f"{home} 与 {away} 平"
    return f"{away} 胜"
