from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Tuple

from ball_quant.core.causal import quote_constraint_strength
from ball_quant.core.handicap import handicap_condition, spf_condition
from ball_quant.core.params import DEFAULT_PARAMS, StrategyParams
from ball_quant.models import Branch, EventMarketMatrix, MarketQuote, MatchSP, normalize_key

# When only one side of a total-goals market is quoted we cannot run a proper
# two-sided devig.  The raw implied probability is vig-contaminated: a
# bookmaker absorbs the full overround on the one side available, so the raw
# price systematically overstates the true implied probability.  We shrink the
# distance from 0.5 by this factor as a conservative bias correction.  A value
# of 0.95 corresponds roughly to a 5% half-overround on a single binary side,
# consistent with typical exchange/sportsbook spreads.
_ONE_SIDED_DEVIG_SHRINK = 0.95


@dataclass
class ProbabilityContext:
    matrix: EventMarketMatrix
    score_distribution: "ScoreDistribution"
    # params is stored here so every branch-probability helper can forward
    # devig_method (and other strategy knobs) without a separate argument.
    params: StrategyParams = None  # type: ignore[assignment]  # set in build_probability_context

    def __post_init__(self) -> None:
        # Guard against accidental None — default to DEFAULT_PARAMS so that
        # code that constructs ProbabilityContext directly still works.
        if self.params is None:
            self.params = DEFAULT_PARAMS


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


def build_probability_context(
    match: MatchSP,
    matrix: EventMarketMatrix,
    params: StrategyParams = DEFAULT_PARAMS,
) -> ProbabilityContext:
    distribution = fit_score_distribution(match, matrix, params=params)
    # Store params so downstream helpers (probability_for_spf, moneyline_margin_*)
    # can forward devig_method without a separate call-site argument.
    return ProbabilityContext(matrix=matrix, score_distribution=distribution, params=params)


def match_branches(match: MatchSP, context: ProbabilityContext) -> List[Branch]:
    enabled = context.params.bet_markets
    branches = []

    # ---- SPF (moneyline win/draw/lose) ----
    if "spf" in enabled:
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

    # ---- Handicap / Asian spread ----
    if "handicap" in enabled:
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

    # ---- Totals (over/under total goals) ----
    # We emit one branch per (line, side) that appears in the matrix.
    # Grid probability comes directly from the calibrated score distribution.
    if "totals" in enabled:
        seen_totals: set = set()
        for quote in context.matrix.quotes("total_goals"):
            if not quote_is_usable(quote):
                continue
            parsed = parse_total_quote(quote)
            if parsed is None:
                continue
            side, line, is_complement = parsed
            if is_complement:
                # "not:over 2.5" encodes the under; flip before deduplication
                side = "under" if side == "over" else "over"
            key = (line, side)
            if key in seen_totals:
                continue
            seen_totals.add(key)
            if side == "over":
                prob = context.score_distribution.probability(lambda h, a, L=line: h + a > L)
            else:
                prob = context.score_distribution.probability(lambda h, a, L=line: h + a < L)
            branches.append(
                Branch(
                    match_id=match.match_id,
                    # play encodes the line; value.py uses it to look up the quote
                    play=f"totals({line})",
                    outcome=side,
                    condition=f"total goals {side} {line}",
                    probability=prob,
                    source="score-distribution",
                )
            )

    # ---- BTTS (both teams to score) ----
    if "btts" in enabled:
        yes_quote = best_usable_quote(context.matrix, "btts", "yes")
        no_quote = best_usable_quote(context.matrix, "btts", "no")
        if yes_quote is not None or no_quote is not None:
            prob_yes = context.score_distribution.probability(lambda h, a: h > 0 and a > 0)
            for side, prob in (("yes", prob_yes), ("no", 1.0 - prob_yes)):
                branches.append(
                    Branch(
                        match_id=match.match_id,
                        play="btts",
                        outcome=side,
                        condition=f"both teams to score: {side}",
                        probability=prob,
                        source="score-distribution",
                    )
                )

    # ---- Team total (over/under per team) ----
    if "team_total" in enabled:
        seen_tt: set = set()
        for quote in context.matrix.quotes("team_total"):
            if not quote_is_usable(quote):
                continue
            parsed_tt = parse_team_total_quote(quote, match.home, match.away)
            if parsed_tt is None:
                continue
            team, side, line, is_complement = parsed_tt
            if is_complement:
                side = "under" if side == "over" else "over"
            key = (team, line, side)
            if key in seen_tt:
                continue
            seen_tt.add(key)
            is_home = normalize_key(team) == normalize_key(match.home)
            if side == "over":
                if is_home:
                    prob = context.score_distribution.probability(lambda h, a, L=line: h > L)
                else:
                    prob = context.score_distribution.probability(lambda h, a, L=line: a > L)
            else:
                if is_home:
                    prob = context.score_distribution.probability(lambda h, a, L=line: h < L)
                else:
                    prob = context.score_distribution.probability(lambda h, a, L=line: a < L)
            # Encode team+line+side in play so value.py can recover the quote.
            branches.append(
                Branch(
                    match_id=match.match_id,
                    play=f"team_total({team},{line})",
                    outcome=side,
                    condition=f"{team} goals {side} {line}",
                    probability=prob,
                    source="score-distribution",
                )
            )

    # ---- Correct score ----
    # "exact_margin" tag triggers combo stacking penalty in correlation_discount.
    if "correct_score" in enabled:
        seen_cs: set = set()
        for quote in context.matrix.quotes("correct_score"):
            if not quote_is_usable(quote):
                continue
            if quote.outcome.startswith("not:"):
                continue
            score = parse_score(f"{quote.question} {quote.outcome}")
            if score is None:
                continue
            if score in seen_cs:
                continue
            seen_cs.add(score)
            x, y = score
            prob = context.score_distribution.probability(lambda h, a, sx=x, sy=y: h == sx and a == sy)
            branches.append(
                Branch(
                    match_id=match.match_id,
                    play="correct_score",
                    outcome=f"{x}-{y}",
                    condition=f"correct score {x}-{y}",
                    probability=prob,
                    source="score-distribution",
                    tags=["exact_margin"],
                )
            )

    return branches


def probability_for_spf(context: ProbabilityContext, outcome: str) -> Optional[float]:
    # Forward context.params so devig_method reaches normalized_moneyline_probabilities.
    direct = normalized_moneyline_probability(context.matrix, outcome, params=context.params)
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


def normalized_moneyline_probabilities(
    matrix: EventMarketMatrix,
    params: StrategyParams = DEFAULT_PARAMS,
) -> Optional[Dict[str, float]]:
    raw = {
        outcome: direct_moneyline_probability(matrix, outcome)
        for outcome in ("home", "draw", "away")
    }
    if any(value is None for value in raw.values()):
        return None
    raw_vals = [market_probability_value(value or 0.0) for value in raw.values()]
    total = sum(raw_vals)
    if total <= 0:
        return None
    outcomes = list(raw.keys())
    if params.devig_method == "shin":
        # Shin (1992) devig: accounts for informed-bettor markup structure.
        # Recovers fair probs p_i such that sum p_i = 1 with insider proportion z.
        fair = shin_devig(raw_vals)
        return dict(zip(outcomes, fair))
    elif params.devig_method == "proportional":
        # Proportional: divide each raw implied prob by the booksum (current default).
        return {outcome: market_probability_value(value or 0.0) / total for outcome, value in raw.items()}
    else:
        raise ValueError(f"Unknown devig_method: {params.devig_method!r}; expected 'proportional' or 'shin'")


def normalized_moneyline_probability(
    matrix: EventMarketMatrix,
    outcome: str,
    params: StrategyParams = DEFAULT_PARAMS,
) -> Optional[float]:
    normalized = normalized_moneyline_probabilities(matrix, params=params)
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
        # Forward context.params so devig_method reaches the moneyline fallback.
        direct = probability_margin_greater_than(context.matrix, home, away, target_margin, params=context.params)
        if direct is not None:
            return clamp_probability(direct)
        return context.score_distribution.margin_probability(lambda m: m > target_margin)
    if outcome == "away":
        # Forward context.params so devig_method reaches the moneyline fallback.
        direct = probability_margin_less_than(context.matrix, home, away, target_margin, params=context.params)
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
    params: StrategyParams = DEFAULT_PARAMS,
) -> Optional[float]:
    line = -(target_margin + 0.5)
    direct = find_team_handicap_quote(matrix, home, line)
    if direct is not None:
        return direct
    # Forward params so moneyline fallback uses the correct devig_method.
    return moneyline_margin_greater_than(matrix, target_margin, params=params)


def probability_margin_less_than(
    matrix: EventMarketMatrix,
    home: str,
    away: str,
    target_margin: int,
    params: StrategyParams = DEFAULT_PARAMS,
) -> Optional[float]:
    line = target_margin - 0.5
    direct = find_team_handicap_quote(matrix, away, line)
    if direct is not None:
        return direct
    # Forward params so moneyline fallback uses the correct devig_method.
    return moneyline_margin_less_than(matrix, target_margin, params=params)


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


def moneyline_margin_greater_than(
    matrix: EventMarketMatrix,
    target_margin: int,
    params: StrategyParams = DEFAULT_PARAMS,
) -> Optional[float]:
    # params must reach here so devig_method applies to the moneyline fallback.
    home = normalized_moneyline_probability(matrix, "home", params=params)
    draw = normalized_moneyline_probability(matrix, "draw", params=params)
    if target_margin == 0 and home is not None:
        return home
    if target_margin == -1 and home is not None and draw is not None:
        return clamp_probability(home + draw)
    return None


def moneyline_margin_less_than(
    matrix: EventMarketMatrix,
    target_margin: int,
    params: StrategyParams = DEFAULT_PARAMS,
) -> Optional[float]:
    # params must reach here so devig_method applies to the moneyline fallback.
    away = normalized_moneyline_probability(matrix, "away", params=params)
    draw = normalized_moneyline_probability(matrix, "draw", params=params)
    if target_margin == 0 and away is not None:
        return away
    if target_margin == 1 and away is not None and draw is not None:
        return clamp_probability(away + draw)
    return None


def fit_score_distribution(
    match: MatchSP,
    matrix: EventMarketMatrix,
    params: StrategyParams = DEFAULT_PARAMS,
) -> ScoreDistribution:
    home_lambda, away_lambda = prior_lambdas(matrix, params=params)
    # Pass Dixon-Coles rho so the low-score tau correction fires when toggled.
    # rho=0.0 (default) leaves the grid byte-identical to independent Poisson.
    base = poisson_grid(home_lambda, away_lambda, params.max_goals, rho=params.dixon_coles_rho)
    constraints = build_market_constraints(match, matrix, params=params)
    if not constraints:
        return ScoreDistribution(base, max_goals=params.max_goals)
    calibrated = calibrate_distribution(base, constraints, params=params)
    return ScoreDistribution(calibrated, max_goals=params.max_goals)


def prior_lambdas(
    matrix: EventMarketMatrix,
    params: StrategyParams = DEFAULT_PARAMS,
) -> Tuple[float, float]:
    # Forward params so devig_method is consistent throughout lambda estimation.
    home_win = normalized_moneyline_probability(matrix, "home", params=params)
    away_win = normalized_moneyline_probability(matrix, "away", params=params)
    total_hint = total_goal_hint(matrix, params=params)
    base_total = total_hint if total_hint is not None else params.base_total_fallback
    home_share = 0.5
    if home_win is not None and away_win is not None:
        diff = clamp_probability(home_win) - clamp_probability(away_win)
        home_share = max(
            params.home_share_floor,
            min(params.home_share_cap, 0.5 + diff * params.home_share_coeff),
        )
    # Compute both legs from the SAME share so they sum exactly to base_total.
    # Previously away_lambda = max(floor, base_total - home_lambda) meant that
    # when the floor bit on either leg independently the sum drifted above
    # base_total, inflating the grid's expected-total relative to the seeding
    # market.  The floor is now a degenerate guard only: in the normal regime
    # (base_total >= 2*lambda_floor) it never fires and sum == base_total.
    # In the pathological regime (very small base_total) the sum may exceed
    # base_total, but that is unavoidable — a valid Poisson must have lambda > 0.
    home_lambda = base_total * home_share
    away_lambda = base_total * (1.0 - home_share)
    home_lambda = max(params.lambda_floor, home_lambda)
    away_lambda = max(params.lambda_floor, away_lambda)
    return home_lambda, away_lambda


def total_goal_hint(
    matrix: EventMarketMatrix,
    params: StrategyParams = DEFAULT_PARAMS,
) -> Optional[float]:
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
            # Two-sided: standard proportional devig — both legs present so
            # overround cancels cleanly.
            p_over = over / (over + under)
        elif over is not None:
            # One-sided only: raw implied prob is vig-contaminated (bookmaker
            # absorbs full spread on this side).  Shrink the deviation from 0.5
            # by _ONE_SIDED_DEVIG_SHRINK to partially correct the overround
            # bias.  This is conservative — not a full devig — because the true
            # overround is unknown without the complement quote.
            p_over = 0.5 + (over - 0.5) * _ONE_SIDED_DEVIG_SHRINK
        elif under is not None:
            # Mirror of the over-only case: 1-under is the raw complement, also
            # vig-contaminated.  Same shrink applied symmetrically.
            p_over = 0.5 + ((1.0 - under) - 0.5) * _ONE_SIDED_DEVIG_SHRINK
        else:
            continue
        candidates.append((abs(p_over - 0.50), line + (p_over - 0.50) * params.total_hint_nudge))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return max(0.5, candidates[0][1])


def build_market_constraints(
    match: MatchSP,
    matrix: EventMarketMatrix,
    params: StrategyParams = DEFAULT_PARAMS,
) -> List[MarketConstraint]:
    constraints: List[MarketConstraint] = []
    constraints.extend(moneyline_constraints(matrix, params=params))
    constraints.extend(handicap_constraints(matrix, match.home, match.away))
    constraints.extend(total_goal_constraints(matrix))
    constraints.extend(team_total_constraints(matrix, match.home, match.away))
    constraints.extend(btts_constraints(matrix))
    constraints.extend(correct_score_constraints(matrix, params=params))
    return constraints


def moneyline_constraints(
    matrix: EventMarketMatrix,
    params: StrategyParams = DEFAULT_PARAMS,
) -> List[MarketConstraint]:
    quotes = {
        outcome: best_usable_quote(matrix, "moneyline", outcome)
        for outcome in ("home", "draw", "away")
    }
    targets = normalized_moneyline_probabilities(matrix, params=params)
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


def correct_score_constraints(
    matrix: EventMarketMatrix,
    params: StrategyParams = DEFAULT_PARAMS,
) -> List[MarketConstraint]:
    score_probs = correct_score_probs(matrix)
    total = sum(score_probs.values())
    if total > params.cs_mass_cap:
        score_probs = {score: prob * params.cs_mass_cap / total for score, prob in score_probs.items()}
    constraints = []
    for score, target in score_probs.items():
        quote = best_usable_quote(matrix, "correct_score", f"{score[0]}-{score[1]}")
        constraints.append(
            MarketConstraint(
                label=f"correct_score:{score[0]}-{score[1]}",
                target=target,
                predicate=lambda h, a, score=score: (h, a) == score,
                strength=min(params.cs_strength_cap, quote_quality(quote) * params.cs_strength_coeff),
                source="polymarket:correct_score",
            )
        )
    return constraints


def calibrate_distribution(
    probs: Dict[Tuple[int, int], float],
    constraints: List[MarketConstraint],
    params: StrategyParams = DEFAULT_PARAMS,
) -> Dict[Tuple[int, int], float]:
    calibrated = normalize_probs(probs)
    primary = [constraint for constraint in constraints if constraint.tier == "primary"]
    shape = [constraint for constraint in constraints if constraint.tier != "primary"]
    for _ in range(params.calib_primary_iters):
        for constraint in primary:
            calibrated = apply_constraint(calibrated, constraint, tier_multiplier=1.0)
    for _ in range(params.calib_shape_iters):
        for constraint in shape:
            calibrated = apply_constraint(calibrated, constraint, tier_multiplier=params.calib_shape_mult)
        for constraint in primary:
            calibrated = apply_constraint(calibrated, constraint, tier_multiplier=params.calib_primary_in_shape_mult)
    for _ in range(params.calib_final_iters):
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


def quote_quality(quote: Optional[MarketQuote], params: StrategyParams = DEFAULT_PARAMS) -> float:
    return quote_constraint_strength(quote, params)


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


def poisson_grid(
    home_lambda: float,
    away_lambda: float,
    max_goals: int,
    rho: float = 0.0,
) -> Dict[Tuple[int, int], float]:
    """Build a (max_goals+1)^2 joint score probability grid.

    When rho=0.0 the grid is pure independent Poisson — identical to the
    previous implementation (default-off guarantee).  When rho != 0 the
    Dixon-Coles (1997) low-score tau correction is applied to the four cells
    {(0,0), (0,1), (1,0), (1,1)} before renormalisation.  rho < 0 lifts 0-0
    and 1-1 (draws / low totals), which is the typical empirically fitted sign.

    Raises ValueError if any tau factor is <= 0 (rho too extreme for these
    lambdas), so the caller (optimizer) learns the region is infeasible rather
    than silently clamping to a wrong distribution.
    """
    probs = {}
    for home in range(max_goals + 1):
        for away in range(max_goals + 1):
            probs[(home, away)] = poisson_pmf(home, home_lambda) * poisson_pmf(away, away_lambda)

    if rho != 0.0:
        # Dixon-Coles tau factors — only the four low-score cells deviate from 1.
        tau = {
            (0, 0): 1.0 - home_lambda * away_lambda * rho,
            (0, 1): 1.0 + home_lambda * rho,
            (1, 0): 1.0 + away_lambda * rho,
            (1, 1): 1.0 - rho,
        }
        for cell, factor in tau.items():
            if factor <= 0.0:
                raise ValueError(
                    f"Dixon-Coles tau({cell[0]},{cell[1]}) = {factor:.6f} <= 0 "
                    f"for home_lambda={home_lambda}, away_lambda={away_lambda}, rho={rho}. "
                    "Choose a rho closer to 0 or reduce lambda magnitudes."
                )
            probs[cell] *= factor

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


def shin_devig(raw_probs: List[float]) -> List[float]:
    """Shin (1992) devig: recover fair probabilities from vig'd implied probs.

    Given raw implied probs r_i (= 1/decimal_odds, booksum B = sum(r_i) > 1),
    find insider-trading proportion z in [0, 1) such that the fair probs

        p_i = ( sqrt(z^2 + 4*(1-z)*r_i^2 / B) - z ) / ( 2*(1-z) )

    satisfy sum_i p_i = 1.  z is solved by bisection on the sum residual.

    When B = 1 (no vig) z = 0 and p_i = r_i (no change).  When B > 1 the
    solution z > 0 and the formula redistributes probability away from
    favourites toward longshots, unlike proportional scaling.

    This is the helper; routing is done by normalized_moneyline_probabilities
    when params.devig_method == "shin".
    """
    n = len(raw_probs)
    if n == 0:
        return []
    booksum = sum(raw_probs)

    def fair(z: float) -> List[float]:
        if abs(1.0 - z) < 1e-14:
            # z→1 is degenerate; treat as uniform
            return [1.0 / n] * n
        return [
            (math.sqrt(z * z + 4.0 * (1.0 - z) * r * r / booksum) - z) / (2.0 * (1.0 - z))
            for r in raw_probs
        ]

    def residual(z: float) -> float:
        return sum(fair(z)) - 1.0

    # z=0 gives sum > 1 when booksum > 1; z approaching 1 gives sum → n/n=1
    # but residual at z=0 is already ~0 if booksum≈1, else positive.
    # Bisect in [0, 1-eps).
    lo, hi = 0.0, 1.0 - 1e-12
    res_lo = residual(lo)
    if abs(res_lo) < 1e-12:
        # No vig or trivially fair — z=0 is the solution.
        return fair(0.0)

    for _ in range(64):   # 64 bisection steps → ~1e-19 precision
        mid = (lo + hi) * 0.5
        if residual(mid) * res_lo > 0:
            lo = mid
        else:
            hi = mid

    return fair((lo + hi) * 0.5)


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
