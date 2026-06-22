from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional

from ball_quant.models import MarketQuote


@dataclass(frozen=True)
class CausalProfile:
    causal_layer: str
    horizon: str
    model_weight: float


CAUSAL_PROFILES: Dict[str, CausalProfile] = {
    "moneyline": CausalProfile("same_match_result", "today_match", 1.00),
    "handicap": CausalProfile("same_match_margin", "today_match", 0.95),
    "total_goals": CausalProfile("same_match_goal_environment", "today_match", 0.72),
    "team_total": CausalProfile("same_match_team_goal_shape", "today_match", 0.62),
    "btts": CausalProfile("same_match_goal_correlation", "today_match", 0.58),
    "correct_score": CausalProfile("same_match_exact_score_tail", "today_match", 0.42),
    "first_half_team_total": CausalProfile("period_goal_shape", "today_match", 0.36),
    "second_half_team_total": CausalProfile("period_goal_shape", "today_match", 0.34),
    "first_half_total_goals": CausalProfile("period_goal_shape", "today_match", 0.36),
    "second_half_total_goals": CausalProfile("period_goal_shape", "today_match", 0.34),
    "halftime_result": CausalProfile("period_result", "today_match", 0.36),
    "second_half_result": CausalProfile("period_result", "today_match", 0.34),
    "btts_first_half": CausalProfile("period_goal_correlation", "today_match", 0.30),
    "btts_second_half": CausalProfile("period_goal_correlation", "today_match", 0.28),
    "first_to_score": CausalProfile("same_match_sequence", "today_match", 0.32),
    "total_corners": CausalProfile("non_goal_match_prop", "today_match", 0.22),
    "team_total_corners": CausalProfile("non_goal_team_prop", "today_match", 0.20),
    "first_half_total_corners": CausalProfile("non_goal_period_prop", "today_match", 0.16),
    "second_half_total_corners": CausalProfile("non_goal_period_prop", "today_match", 0.15),
    "corners_odd_even": CausalProfile("non_goal_tail_prop", "today_match", 0.08),
    "first_corner": CausalProfile("non_goal_tail_prop", "today_match", 0.10),
    "player_goals": CausalProfile("player_prop", "today_match", 0.26),
    "player_shots": CausalProfile("player_prop", "today_match", 0.18),
    "player_shots_on_target": CausalProfile("player_prop", "today_match", 0.16),
    "player_assists": CausalProfile("player_prop", "today_match", 0.16),
    "player_goal_contributions": CausalProfile("player_prop", "today_match", 0.18),
    "goalkeeper_saves": CausalProfile("player_prop", "today_match", 0.14),
    "starting_lineup": CausalProfile("lineup_signal", "today_match", 0.24),
    "group_winner": CausalProfile("group_path", "near_future", 0.26),
    "group_advancement": CausalProfile("group_path", "near_future", 0.28),
    "group_position": CausalProfile("group_path", "near_future", 0.20),
    "stage_advancement": CausalProfile("tournament_path", "medium_future", 0.20),
    "stage_elimination": CausalProfile("tournament_path", "medium_future", 0.18),
    "tournament_winner": CausalProfile("tournament_winner", "long_future", 0.12),
    "team_prop": CausalProfile("tournament_team_prop", "long_future", 0.12),
    "player_award": CausalProfile("tournament_player_prop", "long_future", 0.10),
    "player_h2h": CausalProfile("tournament_player_h2h", "long_future", 0.09),
    "player_future": CausalProfile("tournament_player_prop", "long_future", 0.08),
    "continent_future": CausalProfile("tournament_meta_prop", "long_future", 0.08),
    "record_future": CausalProfile("tournament_tail_prop", "long_future", 0.07),
    "culture_future": CausalProfile("non_sport_context", "long_future", 0.03),
    "other": CausalProfile("unmapped_market", "unknown", 0.05),
}


def causal_profile_for_category(category: str) -> CausalProfile:
    return CAUSAL_PROFILES.get(category, CAUSAL_PROFILES["other"])


def quote_constraint_strength(quote: Optional[MarketQuote]) -> float:
    if quote is None:
        return 0.20
    reliability = quote_market_reliability(quote)
    profile_weight = quote.model_weight
    if profile_weight is None:
        profile_weight = causal_profile_for_category(quote.category).model_weight
    return max(0.03, min(1.0, reliability * max(0.03, min(1.0, profile_weight))))


def quote_market_reliability(quote: MarketQuote) -> float:
    reliability = 0.72
    if quote.spread is not None:
        if quote.spread <= 0.02:
            reliability += 0.18
        elif quote.spread <= 0.06:
            reliability += 0.08
        elif quote.spread >= 0.50:
            reliability -= 0.55
        elif quote.spread >= 0.15:
            reliability -= 0.30
        elif quote.spread >= 0.10:
            reliability -= 0.18
    else:
        reliability -= 0.08
    if quote.probability is not None and (quote.probability <= 0.005 or quote.probability >= 0.995):
        reliability -= 0.35
    if quote.liquidity is not None:
        if quote.liquidity >= 100000:
            reliability += 0.12
        elif quote.liquidity >= 10000:
            reliability += 0.08
        elif quote.liquidity < 500:
            reliability -= 0.18
        elif quote.liquidity < 1000:
            reliability -= 0.12
    else:
        reliability -= 0.06
    if quote.volume is not None:
        if quote.volume >= 100000:
            reliability += 0.08
        elif quote.volume < 500:
            reliability -= 0.06
    return max(0.05, min(1.0, reliability))


def causal_layer_summary(quotes: Iterable[MarketQuote]) -> Dict[str, Dict[str, float]]:
    summary: Dict[str, Dict[str, float]] = {}
    for quote in quotes:
        layer = quote.causal_layer or causal_profile_for_category(quote.category).causal_layer
        item = summary.setdefault(layer, {"quotes": 0.0, "avg_weight": 0.0})
        item["quotes"] += 1.0
        weight = quote.model_weight
        if weight is None:
            weight = causal_profile_for_category(quote.category).model_weight
        item["avg_weight"] += weight
    for item in summary.values():
        if item["quotes"] > 0:
            item["avg_weight"] /= item["quotes"]
    return summary
