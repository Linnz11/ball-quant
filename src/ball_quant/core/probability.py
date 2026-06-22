from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Tuple

from ball_quant.core.causal import quote_constraint_strength
from ball_quant.core.handicap import handicap_condition, spf_condition
from ball_quant.models import Branch, EventMarketMatrix, MarketQuote, MatchSP, normalize_key


@dataclass
class ProbabilityContext:
    matrix: EventMarketMatrix
    score_distribution: "ScoreDistribution"


@dataclass
class MarketConstraint:
    label: str
    target: float
    predicate: Callable[[int, int], bool]
    strength: float
    source: str
    tier: str = "shape"


class ScoreDistribution:
    def __init__(self, probs: Dict[Tuple[int, int], float], max_goals: int = 7) -> None:
        self.probs = normalize_probs(probs)
        self.max_goals = max_goals

    def probability(self, predicate) -> float:
        return sum(prob for score, prob in self.probs.items() if predicate(score[0], score[1]))

    def margin_probability(self, predicate) -> float:
        return self.probability(lambda home, away: predicate(home - away))


def build_probability_context(match: MatchSP, matrix: EventMarketMatrix) -> ProbabilityContext:
    distribution = fit_score_distribution(match, matrix)
    return ProbabilityContext(matrix=matrix, score_distribution=distribution)


def match_branches(match: MatchSP, context: ProbabilityContext) -> List[Branch]:
    branches = []
    for outcome in ("home", "draw", "away"):
        prob = probability_for_spf(context, outcome)
        branches.append(
            Branch(
                match_id=match.match_id,
                play="spf",
                outcome=outcome,
                condition=spf_condition(match.home, match.away, outcome),
                probability=prob,
                source=probability_source(context, "moneyline", outcome),
            )
        )
    for outcome in ("home", "draw", "away"):
        prob = probability_for_handicap(context, match.handicap, outcome, match.home, match.away)
        tags = ["exact_margin"] if outcome == "draw" else []
        branches.append(
            Branch(
                match_id=match.match_id,
                play=f"rq({match.handicap:+d})",
                outcome=outcome,
                condition=handicap_condition(match.home, match.away, match.handicap, outcome),
                probability=prob,
                source=probability_source(context, "handicap", outcome),
                tags=tags,
            )
        )
    return branches


def probability_for_spf(context: ProbabilityContext, outcome: str) -> Optional[float]:
    direct = normalized_moneyline_probability(context.matrix, outcome)
    if direct is not None:
        return clamp_probability(direct)
    if outcome == "home":
        return context.score_distribution.probability(lambda h, a: h > a)
    if outcome == "draw":
        return context.score_distribution.probability(lambda h, a: h == a)
    return context.score_distribution.probability(lambda h, a: h < a)


def direct_moneyline_probability(matrix: EventMarketMatrix, outcome: str) -> Optional[float]:
    direct_quote = best_usable_quote(matrix, "moneyline", outcome)
    if direct_quote and direct_quote.probability is not None:
        return market_probability(direct_quote)
    positive = best_usable_quote(matrix, "moneyline", f"not_{outcome}")
    if positive and positive.probability is not None:
        return 1.0 - market_probability(positive)
    return None


def normalized_moneyline_probabilities(matrix: EventMarketMatrix) -> Optional[Dict[str, float]]:
    raw = {
        outcome: direct_moneyline_probability(matrix, outcome)
        for outcome in ("home", "draw", "away")
    }
    if any(value is None for value in raw.values()):
        return None
    total = sum(market_probability_value(value or 0.0) for value in raw.values())
    if total <= 0:
        return None
    return {outcome: market_probability_value(value or 0.0) / total for outcome, value in raw.items()}


def normalized_moneyline_probability(matrix: EventMarketMatrix, outcome: str) -> Optional[float]:
    normalized = normalized_moneyline_probabilities(matrix)
    if normalized is None:
        return direct_moneyline_probability(matrix, outcome)
    return normalized.get(outcome)


def probability_for_handicap(
    context: ProbabilityContext,
    handicap: int,
    outcome: str,
    home: str,
    away: str,
) -> Optional[float]:
    target_margin = -handicap
    if outcome == "home":
        direct = probability_margin_greater_than(context.matrix, home, away, target_margin)
        if direct is not None:
            return clamp_probability(direct)
        return context.score_distribution.margin_probability(lambda m: m > target_margin)
    if outcome == "away":
        direct = probability_margin_less_than(context.matrix, home, away, target_margin)
        if direct is not None:
            return clamp_probability(direct)
        return context.score_distribution.margin_probability(lambda m: m < target_margin)

    greater = probability_for_handicap(context, handicap, "home", home, away)
    less = probability_for_handicap(context, handicap, "away", home, away)
    if greater is not None and less is not None:
        return clamp_probability(1.0 - greater - less)
    return context.score_distribution.margin_probability(lambda m: m == target_margin)


def probability_margin_greater_than(
    matrix: EventMarketMatrix,
    home: str,
    away: str,
    target_margin: int,
) -> Optional[float]:
    line = -(target_margin + 0.5)
    direct = find_team_handicap_quote(matrix, home, line)
    if direct is not None:
        return direct
    return moneyline_margin_greater_than(matrix, target_margin)


def probability_margin_less_than(
    matrix: EventMarketMatrix,
    home: str,
    away: str,
    target_margin: int,
) -> Optional[float]:
    line = target_margin - 0.5
    direct = find_team_handicap_quote(matrix, away, line)
    if direct is not None:
        return direct
    return moneyline_margin_less_than(matrix, target_margin)


def find_team_handicap_quote(matrix: EventMarketMatrix, team: str, line: float) -> Optional[float]:
    quote = exact_handicap_quote(matrix, team, line)
    if quote is None or quote.probability is None:
        return None
    opposite = opposite_team(matrix, team)
    complement = exact_handicap_quote(matrix, opposite, -line) if opposite else None
    if complement and complement.probability is not None:
        return normalize_binary_side(quote, complement)
    return market_probability(quote)


def exact_handicap_quote(matrix: EventMarketMatrix, team: str, line: float) -> Optional[MarketQuote]:
    team_key = normalize_key(team)
    candidates = []
    for quote in matrix.quotes("handicap"):
        if not quote_is_usable(quote):
            continue
        if quote.outcome.startswith("not:"):
            continue
        entity_key = normalize_key(quote.entity or "")
        outcome_key = normalize_key(quote.outcome)
        if entity_key and entity_key != team_key:
            continue
        if not entity_key and team_key not in outcome_key:
            continue
        quote_line = quote.line if quote.line is not None else parse_signed_line(quote.outcome)
        if quote_line is not None and abs(quote_line - line) < 1e-9:
            candidates.append(quote)
    return best_quality_quote(candidates)


def opposite_team(matrix: EventMarketMatrix, team: str) -> Optional[str]:
    team_key = normalize_key(team)
    if normalize_key(matrix.home) == team_key:
        return matrix.away
    if normalize_key(matrix.away) == team_key:
        return matrix.home
    return None


def moneyline_margin_greater_than(matrix: EventMarketMatrix, target_margin: int) -> Optional[float]:
    home = normalized_moneyline_probability(matrix, "home")
    draw = normalized_moneyline_probability(matrix, "draw")
    if target_margin == 0 and home is not None:
        return home
    if target_margin == -1 and home is not None and draw is not None:
        return clamp_probability(home + draw)
    return None


def moneyline_margin_less_than(matrix: EventMarketMatrix, target_margin: int) -> Optional[float]:
    away = normalized_moneyline_probability(matrix, "away")
    draw = normalized_moneyline_probability(matrix, "draw")
    if target_margin == 0 and away is not None:
        return away
    if target_margin == 1 and away is not None and draw is not None:
        return clamp_probability(away + draw)
    return None


def fit_score_distribution(match: MatchSP, matrix: EventMarketMatrix, max_goals: int = 7) -> ScoreDistribution:
    base = poisson_grid(*prior_lambdas(matrix), max_goals)
    constraints = build_market_constraints(match, matrix)
    if not constraints:
        return ScoreDistribution(base, max_goals=max_goals)
    calibrated = calibrate_distribution(base, constraints)
    return ScoreDistribution(calibrated, max_goals=max_goals)


def prior_lambdas(matrix: EventMarketMatrix) -> Tuple[float, float]:
    home_win = normalized_moneyline_probability(matrix, "home")
    away_win = normalized_moneyline_probability(matrix, "away")
    total_hint = total_goal_hint(matrix)
    base_total = total_hint if total_hint is not None else 2.45
    home_share = 0.5
    if home_win is not None and away_win is not None:
        diff = clamp_probability(home_win) - clamp_probability(away_win)
        home_share = max(0.25, min(0.75, 0.5 + diff * 0.35))
    home_lambda = max(0.25, base_total * home_share)
    away_lambda = max(0.25, base_total - home_lambda)
    return home_lambda, away_lambda


def total_goal_hint(matrix: EventMarketMatrix) -> Optional[float]:
    by_line: Dict[float, Dict[str, float]] = {}
    for quote in matrix.quotes("total_goals"):
        if not quote_is_usable(quote):
            continue
        parsed = parse_total_quote(quote)
        if parsed and quote.probability is not None:
            side, line, _ = parsed
            p = clamp_probability(quote.probability)
            by_line.setdefault(line, {})[side] = p
    candidates: List[Tuple[float, float]] = []
    for line, sides in by_line.items():
        over = sides.get("over")
        under = sides.get("under")
        if over is not None and under is not None and over + under > 0:
            p_over = over / (over + under)
        elif over is not None:
            p_over = over
        elif under is not None:
            p_over = 1.0 - under
        else:
            continue
        candidates.append((abs(p_over - 0.50), line + (p_over - 0.50) * 0.7))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return max(0.5, candidates[0][1])


def build_market_constraints(match: MatchSP, matrix: EventMarketMatrix) -> List[MarketConstraint]:
    constraints: List[MarketConstraint] = []
    constraints.extend(moneyline_constraints(matrix))
    constraints.extend(handicap_constraints(matrix, match.home, match.away))
    constraints.extend(total_goal_constraints(matrix))
    constraints.extend(team_total_constraints(matrix, match.home, match.away))
    constraints.extend(btts_constraints(matrix))
    constraints.extend(correct_score_constraints(matrix))
    return constraints


def moneyline_constraints(matrix: EventMarketMatrix) -> List[MarketConstraint]:
    quotes = {
        outcome: best_usable_quote(matrix, "moneyline", outcome)
        for outcome in ("home", "draw", "away")
    }
    targets = normalized_moneyline_probabilities(matrix)
    if not targets:
        return []
    predicates = {
        "home": lambda h, a: h > a,
        "draw": lambda h, a: h == a,
        "away": lambda h, a: h < a,
    }
    return [
        MarketConstraint(
            label=f"moneyline:{outcome}",
            target=target,
            predicate=predicates[outcome],
            strength=quote_quality(quotes[outcome] or best_usable_quote(matrix, "moneyline", f"not_{outcome}")),
            source="polymarket:moneyline",
            tier="primary",
        )
        for outcome, target in targets.items()
    ]


def handicap_constraints(matrix: EventMarketMatrix, home: str, away: str) -> List[MarketConstraint]:
    groups: Dict[Tuple[str, float], Tuple[MarketQuote, float]] = {}
    for quote in matrix.quotes("handicap"):
        if not quote_is_usable(quote):
            continue
        parsed = parse_handicap_quote(quote, home, away)
        if not parsed or quote.probability is None:
            continue
        team, line, is_complement = parsed
        if line > 0:
            continue
        target = market_probability(quote)
        if is_complement:
            target = 1.0 - target
        key = (team, line)
        existing = groups.get(key)
        if existing is None or quote_quality(quote) > quote_quality(existing[0]):
            groups[key] = (quote, target)
    constraints = []
    for (team, line), (quote, target) in groups.items():
        if normalize_key(team) == normalize_key(home):
            predicate = lambda h, a, line=line: h + line > a
        elif normalize_key(team) == normalize_key(away):
            predicate = lambda h, a, line=line: a + line > h
        else:
            continue
        constraints.append(
            MarketConstraint(
                label=f"handicap:{team}:{line:+.1f}",
                target=target,
                predicate=predicate,
                strength=quote_quality(quote),
                source="polymarket:handicap",
                tier="primary",
            )
        )
    return constraints


def total_goal_constraints(matrix: EventMarketMatrix) -> List[MarketConstraint]:
    constraints = []
    groups: Dict[float, Dict[str, MarketQuote]] = {}
    for quote in matrix.quotes("total_goals"):
        if not quote_is_usable(quote):
            continue
        parsed = parse_total_quote(quote)
        if not parsed or quote.probability is None:
            continue
        side, line, is_complement = parsed
        if is_complement:
            side = "under" if side == "over" else "over"
        previous = groups.setdefault(line, {}).get(side)
        if previous is None or quote_quality(quote) > quote_quality(previous):
            groups[line][side] = quote
    for line, sides in groups.items():
        target = binary_target(sides.get("over"), sides.get("under"))
        if target is None:
            continue
        strength = paired_quote_quality(sides.get("over"), sides.get("under"))
        constraints.append(
            MarketConstraint(
                label=f"total:over:{line:.1f}",
                target=target,
                predicate=lambda h, a, line=line: h + a > line,
                strength=strength,
                source="polymarket:total_goals",
                tier="shape",
            )
        )
    return constraints


def team_total_constraints(matrix: EventMarketMatrix, home: str, away: str) -> List[MarketConstraint]:
    groups: Dict[Tuple[str, float], Dict[str, MarketQuote]] = {}
    for quote in matrix.quotes("team_total"):
        if not quote_is_usable(quote):
            continue
        parsed = parse_team_total_quote(quote, home, away)
        if not parsed or quote.probability is None:
            continue
        team, side, line, is_complement = parsed
        if is_complement:
            side = "under" if side == "over" else "over"
        key = (team, line)
        previous = groups.setdefault(key, {}).get(side)
        if previous is None or quote_quality(quote) > quote_quality(previous):
            groups[key][side] = quote
    constraints = []
    for (team, line), sides in groups.items():
        target = binary_target(sides.get("over"), sides.get("under"))
        if target is None:
            continue
        is_home = normalize_key(team) == normalize_key(home)
        predicate = (
            (lambda h, a, line=line: h > line)
            if is_home
            else (lambda h, a, line=line: a > line)
        )
        constraints.append(
            MarketConstraint(
                label=f"team_total:{team}:over:{line:.1f}",
                target=target,
                predicate=predicate,
                strength=paired_quote_quality(sides.get("over"), sides.get("under")),
                source="polymarket:team_total",
                tier="shape",
            )
        )
    return constraints


def btts_constraints(matrix: EventMarketMatrix) -> List[MarketConstraint]:
    yes = best_usable_quote(matrix, "btts", "yes")
    no = best_usable_quote(matrix, "btts", "no")
    target = binary_target(yes, no)
    if target is None:
        return []
    return [
        MarketConstraint(
            label="btts:yes",
            target=target,
            predicate=lambda h, a: h > 0 and a > 0,
            strength=paired_quote_quality(yes, no),
            source="polymarket:btts",
            tier="shape",
        )
    ]


def correct_score_constraints(matrix: EventMarketMatrix) -> List[MarketConstraint]:
    score_probs = correct_score_probs(matrix)
    total = sum(score_probs.values())
    if total > 0.85:
        score_probs = {score: prob * 0.85 / total for score, prob in score_probs.items()}
    constraints = []
    for score, target in score_probs.items():
        quote = best_usable_quote(matrix, "correct_score", f"{score[0]}-{score[1]}")
        constraints.append(
            MarketConstraint(
                label=f"correct_score:{score[0]}-{score[1]}",
                target=target,
                predicate=lambda h, a, score=score: (h, a) == score,
                strength=min(0.26, quote_quality(quote) * 0.65),
                source="polymarket:correct_score",
            )
        )
    return constraints


def calibrate_distribution(
    probs: Dict[Tuple[int, int], float],
    constraints: List[MarketConstraint],
) -> Dict[Tuple[int, int], float]:
    calibrated = normalize_probs(probs)
    primary = [constraint for constraint in constraints if constraint.tier == "primary"]
    shape = [constraint for constraint in constraints if constraint.tier != "primary"]
    for _ in range(90):
        for constraint in primary:
            calibrated = apply_constraint(calibrated, constraint, tier_multiplier=1.0)
    for _ in range(25):
        for constraint in shape:
            calibrated = apply_constraint(calibrated, constraint, tier_multiplier=0.30)
        for constraint in primary:
            calibrated = apply_constraint(calibrated, constraint, tier_multiplier=0.75)
    for _ in range(20):
        for constraint in primary:
            calibrated = apply_constraint(calibrated, constraint, tier_multiplier=1.0)
    return normalize_probs(calibrated)


def apply_constraint(
    probs: Dict[Tuple[int, int], float],
    constraint: MarketConstraint,
    tier_multiplier: float = 1.0,
) -> Dict[Tuple[int, int], float]:
    target = min(0.995, max(0.005, constraint.target))
    current = sum(prob for score, prob in probs.items() if constraint.predicate(score[0], score[1]))
    current = min(0.995, max(0.005, current))
    strength = max(0.02, min(1.0, constraint.strength * tier_multiplier))
    in_scale = (target / current) ** strength
    out_scale = ((1.0 - target) / (1.0 - current)) ** strength
    adjusted = {}
    for score, prob in probs.items():
        adjusted[score] = prob * (in_scale if constraint.predicate(score[0], score[1]) else out_scale)
    return normalize_probs(adjusted)


def quote_quality(quote: Optional[MarketQuote]) -> float:
    return quote_constraint_strength(quote)


def quote_is_usable(quote: Optional[MarketQuote]) -> bool:
    if quote is None or quote.probability is None:
        return False
    if quote.closed is True:
        return False
    if quote.accepting_orders is False:
        return False
    if quote.active is False:
        return False
    if quote.liquidity is not None and quote.liquidity <= 0 and quote.bid is None and quote.ask is None:
        return False
    return True


def best_usable_quote(matrix: EventMarketMatrix, category: str, outcome: str) -> Optional[MarketQuote]:
    outcome_key = normalize_key(outcome)
    candidates = [
        quote
        for quote in matrix.quotes(category)
        if quote_is_usable(quote) and normalize_key(quote.outcome) == outcome_key
    ]
    return best_quality_quote(candidates)


def market_probability(quote: MarketQuote) -> float:
    return market_probability_value(quote.probability or 0.0)


def market_probability_value(value: float, floor: float = 0.005) -> float:
    return min(1.0 - floor, max(floor, clamp_probability(value)))


def normalize_binary_side(positive: MarketQuote, negative: MarketQuote) -> Optional[float]:
    if positive.probability is None:
        return None
    pos = market_probability(positive)
    neg = market_probability(negative)
    total = pos + neg
    if total <= 0:
        return None
    return pos / total


def binary_target(positive: Optional[MarketQuote], negative: Optional[MarketQuote]) -> Optional[float]:
    if positive and positive.probability is not None and negative and negative.probability is not None:
        return normalize_binary_side(positive, negative)
    if positive and positive.probability is not None:
        return market_probability(positive)
    if negative and negative.probability is not None:
        return 1.0 - market_probability(negative)
    return None


def paired_quote_quality(positive: Optional[MarketQuote], negative: Optional[MarketQuote]) -> float:
    qualities = [quote_quality(quote) for quote in (positive, negative) if quote is not None]
    if not qualities:
        return 0.12
    return sum(qualities) / len(qualities)


def best_quality_quote(quotes: Iterable[MarketQuote]) -> Optional[MarketQuote]:
    candidates = list(quotes)
    if not candidates:
        return None
    return max(candidates, key=quote_quality)


def parse_handicap_quote(quote: MarketQuote, home: str, away: str) -> Optional[Tuple[str, float, bool]]:
    line = quote.line
    if line is None:
        line = parse_signed_line(quote.outcome)
    if line is None:
        line = parse_signed_line(quote.question)
    if line is None:
        return None
    outcome_key = normalize_key(quote.outcome)
    entity_key = normalize_key(quote.entity or "")
    team = None
    if entity_key == normalize_key(home) or normalize_key(home) in outcome_key:
        team = home
    if entity_key == normalize_key(away) or normalize_key(away) in outcome_key:
        team = away
    if team is None:
        return None
    return team, line, quote.outcome.startswith("not:")


def parse_total_quote(quote: MarketQuote) -> Optional[Tuple[str, float, bool]]:
    parsed = parse_market_side_line(quote)
    if not parsed:
        return None
    side, line = parsed
    return side, line, quote.outcome.startswith("not:")


def parse_team_total_quote(
    quote: MarketQuote,
    home: str,
    away: str,
) -> Optional[Tuple[str, str, float, bool]]:
    parsed = parse_market_side_line(quote)
    if not parsed:
        return None
    side, line = parsed
    entity_key = normalize_key(quote.entity or "")
    if entity_key == normalize_key(home):
        return home, side, line, quote.outcome.startswith("not:")
    if entity_key == normalize_key(away):
        return away, side, line, quote.outcome.startswith("not:")
    blob = f"{quote.outcome}"
    blob_key = normalize_key(blob)
    if normalize_key(home) in blob_key:
        return home, side, line, quote.outcome.startswith("not:")
    if normalize_key(away) in blob_key:
        return away, side, line, quote.outcome.startswith("not:")
    return None


def parse_signed_line(text: str) -> Optional[float]:
    match = re.search(r"(^|[\s(])([+-]\d+(?:\.\d+)?)(?=$|[\s)])", text)
    return float(match.group(2)) if match else None


def parse_side_line(text: str) -> Optional[Tuple[str, float]]:
    match = re.search(r"\b(over|under)\s*(\d+(?:\.\d+)?)", text, flags=re.IGNORECASE)
    if not match:
        return None
    return match.group(1).lower(), float(match.group(2))


def parse_market_side_line(quote: MarketQuote) -> Optional[Tuple[str, float]]:
    outcome_side = re.search(r"\b(over|under)\b", quote.outcome, flags=re.IGNORECASE)
    outcome_line = re.search(r"(\d+(?:\.\d+)?)", quote.outcome)
    question_line = re.search(r"(\d+(?:\.\d+)?)", quote.question)
    if outcome_side:
        line_match = outcome_line or question_line
        if line_match:
            return outcome_side.group(1).lower(), float(line_match.group(1))
    return parse_side_line(f"{quote.outcome} {quote.question}")


def poisson_grid(home_lambda: float, away_lambda: float, max_goals: int) -> Dict[Tuple[int, int], float]:
    probs = {}
    for home in range(max_goals + 1):
        for away in range(max_goals + 1):
            probs[(home, away)] = poisson_pmf(home, home_lambda) * poisson_pmf(away, away_lambda)
    return normalize_probs(probs)


def poisson_pmf(k: int, lam: float) -> float:
    return math.exp(-lam) * (lam**k) / math.factorial(k)


def correct_score_probs(matrix: EventMarketMatrix) -> Dict[Tuple[int, int], float]:
    probs: Dict[Tuple[int, int], float] = {}
    for quote in matrix.quotes("correct_score"):
        if not quote_is_usable(quote):
            continue
        if quote.outcome.startswith("not:"):
            continue
        score = parse_score(f"{quote.question} {quote.outcome}")
        if score and quote.probability is not None:
            probs[score] = max(probs.get(score, 0.0), clamp_probability(quote.probability))
    return probs


def parse_score(text: str) -> Optional[Tuple[int, int]]:
    for match in re.finditer(r"(?<![\d-])(\d{1,2})\s*[-:]\s*(\d{1,2})(?![\d-])", text):
        home_score = int(match.group(1))
        away_score = int(match.group(2))
        if 0 <= home_score <= 15 and 0 <= away_score <= 15:
            return home_score, away_score
    return None


def normalize_probs(probs: Dict[Tuple[int, int], float]) -> Dict[Tuple[int, int], float]:
    total = sum(max(0.0, prob) for prob in probs.values())
    if total <= 0:
        return {}
    return {score: max(0.0, prob) / total for score, prob in probs.items()}


def clamp_probability(value: float) -> float:
    return max(0.0, min(1.0, value))


def format_line(line: float) -> str:
    sign = "+" if line > 0 else ""
    return f"{sign}{line:.1f}"


def probability_source(context: ProbabilityContext, category: str, outcome: str) -> str:
    quote = best_usable_quote(context.matrix, category, outcome)
    if quote:
        return f"polymarket:{category}"
    return "score-distribution"
