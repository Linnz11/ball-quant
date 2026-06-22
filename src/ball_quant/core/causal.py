from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional

from ball_quant.models import MarketQuote
from ball_quant.core.params import DEFAULT_PARAMS, StrategyParams


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


def quote_constraint_strength(quote: Optional[MarketQuote], params: StrategyParams = DEFAULT_PARAMS) -> float:
    if quote is None:
        return 0.20
    reliability = quote_market_reliability(quote, params)
    profile_weight = quote.model_weight
    if profile_weight is None:
        profile_weight = causal_profile_for_category(quote.category).model_weight
    # apply optional per-category weight scaling from params
    if params.profile_weight_scale is not None and quote.category in params.profile_weight_scale:
        profile_weight = profile_weight * params.profile_weight_scale[quote.category]
    return max(0.03, min(1.0, reliability * max(0.03, min(1.0, profile_weight))))


def quote_inverse_variance_reliability(quote: MarketQuote) -> float:
    """Compute a (0, 1] reliability from the quote's bid-ask spread using an
    inverse-variance weighting rationale.

    WHY spread as the variance proxy:
      A market quote is a noisy estimate of the true probability.  The bid-ask
      spread is the market-maker's own measure of uncertainty: a tighter spread
      means the maker is confident (low variance); a wider spread means the maker
      demands compensation for higher uncertainty.  So weight ∝ 1/variance, where
      variance ≈ spread².  The Lorentzian form  w = 1/(1 + (s/s_ref)²)  maps
      spread=0 → 1.0 and spread=s_ref → 0.5, giving a smooth monotone-decreasing
      curve bounded in (0, 1] without requiring tuned step thresholds.

    s_ref = 0.05 (5 % spread): chosen as a typical mid-market spread on liquid
      prediction markets.  At this reference point the weight is 0.5, which keeps
      the scale intuitive — tight markets score near 1, liquid-but-normal markets
      score ~0.5, wide illiquid markets score near 0.

    WHY fold in liquidity as a mild multiplier:
      Liquidity is a secondary signal of information quality (higher depth ⇒
      more informed flow ⇒ lower adverse-selection risk).  The multiplier is
      kept gentle (0.9–1.1 range) so spread remains the dominant driver.

    None spread: no information about uncertainty → fall back to 0.5 (low
      confidence, not zero, because the quote may still carry useful signal).
    """
    _S_REF = 0.05  # reference spread where weight = 0.5

    if quote.spread is None:
        base = 0.5  # low-confidence constant; no spread data
    else:
        s = quote.spread
        base = 1.0 / (1.0 + (s / _S_REF) ** 2)

    # Optional mild liquidity multiplier: clips to [0.9, 1.1] so spread dominates.
    # WHY: high liquidity confirms tight spread is real (informed market); low
    # liquidity makes even a tight spread suspect (thin book can be spoofed).
    if quote.liquidity is not None:
        if quote.liquidity >= 50000:
            liq_mult = 1.10
        elif quote.liquidity >= 5000:
            liq_mult = 1.00
        elif quote.liquidity < 500:
            liq_mult = 0.90
        else:
            liq_mult = 0.95
        base = base * liq_mult

    return max(1e-6, min(1.0, base))


def quote_market_reliability(quote: MarketQuote, params: StrategyParams = DEFAULT_PARAMS) -> float:
    """Return a [0.05, 1.0] reliability for a market quote.

    Branches on params.weight_scheme:
      "heuristic"        — original hand-tuned step-function adjustments (unchanged).
      "inverse_variance" — spread-based Lorentzian; see quote_inverse_variance_reliability.

    WHY two schemes coexist: the inverse-variance formula is theoretically grounded
    but changes quote weights non-trivially.  The toggle lets researchers A/B-backtest
    without touching production defaults.
    """
    if params.weight_scheme == "heuristic":
        reliability = params.reliability_base
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

    elif params.weight_scheme == "inverse_variance":
        # Spread is the primary variance proxy; liquidity is a mild secondary signal.
        # probability and volume adjustments are intentionally omitted here: they are
        # quote-characteristic signals that belong to the heuristic regime's design.
        # IV reliability is a single interpretable quantity — keep it pure.
        return quote_inverse_variance_reliability(quote)

    else:
        raise ValueError(
            f"Unknown weight_scheme={params.weight_scheme!r}. "
            "Valid values: 'heuristic', 'inverse_variance'."
        )


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
