from __future__ import annotations

from math import sqrt
from typing import Dict, List

from ball_quant.core.params import DEFAULT_PARAMS, StrategyParams
from ball_quant.models import Combo


def allocate_stakes(
    combo_groups: Dict[str, List[Combo]],
    budget: float,
    unit: int = 2,
    params: StrategyParams = DEFAULT_PARAMS,
) -> List[Combo]:
    budgets = {
        "A": budget * params.budget_a,
        "B": budget * params.budget_b,
        "C": min(budget * params.budget_c, budget * 0.15),
    }
    allocated: List[Combo] = []
    for key in ("A", "B", "C"):
        combos = combo_groups.get(key, [])
        if not combos:
            continue
        weights = [combo_weight(combo, key) for combo in combos]
        total = sum(weights) or 1.0
        for combo, weight in zip(combos, weights):
            raw_stake = budgets[key] * weight / total
            kelly_cap = budget * max(0.0, combo.kelly) * params.fractional_kelly
            type_cap = max_type_stake(budget, key, params=params)
            stake = round_down_unit(min(raw_stake, kelly_cap, type_cap), unit)
            combo.stake = stake
            combo.payout = stake * combo.odds
            combo.profit = combo.payout - stake
            allocated.append(combo)
    trim_to_budget(allocated, budget, unit)
    return [combo for combo in allocated if combo.stake > 0]


def combo_weight(combo: Combo, key: str) -> float:
    if combo.expected_return <= 0 or combo.kelly <= 0:
        return 0.0
    volatility = sqrt(max(0.001, combo.probability * (1.0 - combo.probability)))
    if key == "A":
        return combo.probability * combo.kelly / volatility
    if key == "B":
        return combo.risk_reward * combo.kelly * combo.probability / volatility
    return min(combo.odds / 10.0, 2.0) * combo.kelly * combo.probability


def max_type_stake(
    budget: float,
    key: str,
    params: StrategyParams = DEFAULT_PARAMS,
) -> float:
    if key == "A":
        return budget * params.cap_a
    if key == "B":
        return budget * params.cap_b
    return budget * params.cap_c


def trim_to_budget(combos: List[Combo], budget: float, unit: int) -> None:
    while sum(combo.stake for combo in combos) > budget and combos:
        target = min((combo for combo in combos if combo.stake > 0), key=lambda c: c.probability)
        target.stake = max(0.0, target.stake - unit)
        target.payout = target.stake * target.odds
        target.profit = target.payout - target.stake


def round_down_unit(value: float, unit: int) -> float:
    return float(int(value // unit) * unit)
