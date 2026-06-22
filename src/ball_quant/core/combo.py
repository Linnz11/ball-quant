from __future__ import annotations

from itertools import combinations
from math import sqrt
from typing import Dict, Iterable, List, Sequence

from ball_quant.core.params import DEFAULT_PARAMS, StrategyParams
from ball_quant.models import Combo, Selection


def generate_combos(
    selections: Sequence[Selection],
    max_size: int = 3,
    params: StrategyParams = DEFAULT_PARAMS,
) -> Dict[str, List[Combo]]:
    kept = [selection for selection in selections if selection.probability > 0 and selection.sp > 1]
    combos: List[Combo] = []
    for size in range(1, max_size + 1):
        for group in combinations(kept, size):
            if has_same_match_conflict(group):
                continue
            combo = build_combo(group, params=params)
            reason = deletion_reason(combo)
            if reason:
                combo.deletion_reason = reason
            combos.append(combo)

    active = [combo for combo in combos if combo.deletion_reason is None]
    used_selection_keys = set()
    type_a = pick_type_a(active, used_selection_keys)
    used_selection_keys.update(selection_keys(type_a))
    type_b = pick_type_b(active, used_selection_keys)
    used_selection_keys.update(selection_keys(type_b))
    type_c = pick_type_c(active, used_selection_keys, params=params)
    return {
        "A": type_a,
        "B": type_b,
        "C": type_c,
        "deleted": [combo for combo in combos if combo.deletion_reason is not None],
    }


def build_combo(group: Iterable[Selection], params: StrategyParams = DEFAULT_PARAMS) -> Combo:
    selections = list(group)
    probability = product(selection.probability for selection in selections) * correlation_discount(selections, params=params)
    odds = product(selection.sp for selection in selections)
    expected_return = probability * odds - 1.0
    kelly = combo_kelly(probability, odds)
    risk_reward = expected_return / max(0.001, 1.0 - probability)
    return Combo(
        name=" × ".join(selection.key for selection in selections),
        selections=selections,
        probability=probability,
        odds=odds,
        expected_return=expected_return,
        combo_type="",
        kelly=kelly,
        risk_reward=risk_reward,
    )


def deletion_reason(combo: Combo) -> str:
    precision_count = sum(1 for selection in combo.selections if "exact_margin" in selection.tags)
    if combo.expected_return <= 0:
        return "组合EV不为正，概率与赔率不匹配"
    if precision_count > 1 and combo.probability < 0.08:
        return "两个以上精准分支且组合概率低于 8%"
    if combo.probability < 0.05:
        return "组合概率低于 5%，只可小搏或删除"
    if combo.probability < 0.08 and precision_count >= 1:
        return "精准分支组合概率低于 8%"
    if all(selection.edge < -0.12 for selection in combo.selections):
        return "全部选择赔率不足，无价值路径"
    if combo.odds >= 12 and average_confidence(combo) < 0.55:
        return "高赔率但置信度不足，疑似赔率彩票路径"
    return ""


def has_same_match_conflict(group: Sequence[Selection]) -> bool:
    seen = set()
    for selection in group:
        if selection.match_id in seen:
            return True
        seen.add(selection.match_id)
    return False


def pick_type_a(combos: List[Combo], used_selection_keys: set) -> List[Combo]:
    candidates = [
        combo
        for combo in combos
        if combo.probability >= 0.08 and combo.expected_return > 0 and average_confidence(combo) >= 0.45
        and not is_lottery_combo(combo)
        and not conflicts_with_used(combo, used_selection_keys)
    ]
    candidates.sort(key=lambda c: (c.probability, c.risk_reward), reverse=True)
    return mark(pick_disjoint(candidates, limit=3), "A 高概率配平版")


def pick_type_b(combos: List[Combo], used_selection_keys: set) -> List[Combo]:
    candidates = [
        combo
        for combo in combos
        if combo.probability >= 0.08 and combo.expected_return > 0 and combo.kelly > 0
        and not is_lottery_combo(combo)
        and not conflicts_with_used(combo, used_selection_keys)
    ]
    candidates.sort(key=lambda c: (c.risk_reward, c.expected_return), reverse=True)
    return mark(pick_disjoint(candidates, limit=3), "B RR优化版")


def pick_type_c(
    combos: List[Combo],
    used_selection_keys: set,
    params: StrategyParams = DEFAULT_PARAMS,
) -> List[Combo]:
    candidates = [
        combo
        for combo in combos
        if params.typec_prob_lo <= combo.probability < params.typec_prob_hi
        and combo.odds >= params.typec_odds_min
        and combo.expected_return > 0
        and average_confidence(combo) >= 0.40
        and not conflicts_with_used(combo, used_selection_keys)
    ]
    candidates.sort(key=lambda c: (c.risk_reward, average_confidence(c), c.expected_return), reverse=True)
    return mark(pick_disjoint(candidates, limit=2), "C 高赔率小搏版")


def mark(combos: List[Combo], combo_type: str) -> List[Combo]:
    for combo in combos:
        combo.combo_type = combo_type
    return combos


def average_confidence(combo: Combo) -> float:
    return sum(selection.confidence for selection in combo.selections) / len(combo.selections)


def is_lottery_combo(combo: Combo) -> bool:
    return combo.odds >= 5.0 and combo.probability < 0.12


def pick_disjoint(combos: List[Combo], limit: int) -> List[Combo]:
    selected: List[Combo] = []
    used = set()
    for combo in combos:
        keys = {selection.key for selection in combo.selections}
        if keys & used:
            continue
        selected.append(combo)
        used.update(keys)
        if len(selected) >= limit:
            break
    return selected


def conflicts_with_used(combo: Combo, used_selection_keys: set) -> bool:
    return bool({selection.key for selection in combo.selections} & used_selection_keys)


def selection_keys(combos: List[Combo]) -> set:
    keys = set()
    for combo in combos:
        keys.update(selection.key for selection in combo.selections)
    return keys


def correlation_discount(
    selections: Sequence[Selection],
    params: StrategyParams = DEFAULT_PARAMS,
) -> float:
    """Return a conservative safety haircut on the independent-product parlay
    probability.

    This is NOT a signed correlation model.  The function always returns a
    value in (0, 1], unconditionally reducing the joint probability below the
    naive independent product.  It is a margin-of-safety buffer for unmodeled
    positive correlation among parlay legs (e.g. both legs tied to the same
    match dynamics, weather, or referee).  For negatively correlated legs it
    would push the joint probability the wrong way — we accept that conservatism
    because the primary defense against wrong-direction bets is edge/EV
    filtering, not correlation sign estimation.

    All coefficient inputs (corr_base, corr_lowconf, corr_exact, corr_floor)
    are StrategyParams-tunable so they can be adjusted without code changes.
    The floor (corr_floor) guarantees the haircut never exceeds 100%.
    """
    if len(selections) <= 1:
        # Single-leg: no multi-leg correlation risk; no haircut applied.
        return 1.0
    # Base haircut grows geometrically with each additional leg — each extra
    # leg adds another uncaptured dependency source.
    discount = params.corr_base ** (len(selections) - 1)
    # Low-confidence legs carry extra unmodeled variance; apply additional
    # per-leg haircut as a further conservative buffer.
    low_confidence_count = sum(1 for selection in selections if selection.confidence < 0.50)
    if low_confidence_count:
        discount *= params.corr_lowconf ** low_confidence_count
    # Exact-margin outcomes (e.g. correct score) have highly path-dependent
    # joint distributions; apply the steepest per-leg haircut for these.
    exact_margin_count = sum(1 for selection in selections if "exact_margin" in selection.tags)
    if exact_margin_count:
        discount *= params.corr_exact ** exact_margin_count
    # Floor prevents the haircut from zeroing out any parlay regardless of size.
    return max(params.corr_floor, discount)


def combo_kelly(probability: float, decimal_odds: float) -> float:
    b = decimal_odds - 1.0
    if b <= 0:
        return 0.0
    q = 1.0 - probability
    return max(0.0, (probability * b - q) / b)


def volatility_penalty(combo: Combo) -> float:
    return sqrt(max(0.001, combo.probability * (1.0 - combo.probability)))


def product(values: Iterable[float]) -> float:
    result = 1.0
    for value in values:
        result *= value
    return result
