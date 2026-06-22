from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

from ball_quant.core.causal import (
    causal_profile_for_category,
    quote_constraint_strength,
    quote_market_reliability,
)
from ball_quant.core.handicap import handicap_condition
from ball_quant.core.probability import (
    ScoreDistribution,
    build_market_constraints,
    build_probability_context,
    parse_team_total_quote,
    parse_total_quote,
    poisson_grid,
    prior_lambdas,
    probability_for_handicap,
    probability_for_spf,
    quote_is_usable,
)
from ball_quant.models import EventMarketMatrix, MarketQuote, MatchSP, normalize_key


def build_live_probability_snapshot(
    matrix: EventMarketMatrix,
    generated_at: Optional[datetime] = None,
    local_schedule: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    match = MatchSP(
        match_id=matrix.match_id,
        date=str(matrix.raw_event.get("eventDate") or "live"),
        home=matrix.home,
        away=matrix.away,
        spf_home=0.0,
        spf_draw=0.0,
        spf_away=0.0,
        handicap=0,
        rq_home=0.0,
        rq_draw=0.0,
        rq_away=0.0,
    )
    context = build_probability_context(match, matrix)
    prior = ScoreDistribution(poisson_grid(*prior_lambdas(matrix), context.score_distribution.max_goals))
    constraints = build_market_constraints(match, matrix)
    generated = generated_at or datetime.now(timezone.utc)
    payload = {
        "generated_at": generated.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
        "match": match_metadata(matrix, local_schedule),
        "market_state": market_state(matrix),
        "collapse_constraints": collapse_constraints(constraints, prior, context.score_distribution),
        "collapse_layers": collapse_layers(constraints),
        "signal_layers": adaptive_causal_layers(matrix.markets),
        "probabilities": {
            "moneyline": moneyline_snapshot(context, matrix),
            "handicap": handicap_snapshot(context, matrix),
            "totals": totals_snapshot(context, matrix),
            "team_totals": team_totals_snapshot(context, matrix),
            "btts": btts_snapshot(context),
            "top_scores": top_scores_snapshot(context.score_distribution),
            "high_probability_quotes": high_probability_quotes_snapshot(matrix.markets),
        },
        "candidate_paths": candidate_paths(context, matrix),
    }
    return payload


def match_metadata(matrix: EventMarketMatrix, local_schedule: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    metadata = {
        "event_id": matrix.event_id,
        "event_slug": matrix.event_slug,
        "event_title": matrix.raw_event.get("title"),
        "home": matrix.home,
        "away": matrix.away,
        "polymarket_date": matrix.raw_event.get("eventDate"),
        "start_time_utc": matrix.raw_event.get("startTime")
        or matrix.raw_event.get("endDate")
        or matrix.raw_event.get("startDate"),
        "active": matrix.raw_event.get("active"),
        "closed": matrix.raw_event.get("closed"),
        "ended": matrix.raw_event.get("ended"),
        "updated_at": matrix.raw_event.get("updatedAt"),
    }
    if local_schedule:
        metadata.update(
            {
                "local_timezone": local_schedule.get("local_timezone"),
                "local_date": local_schedule.get("local_date"),
                "local_time": local_schedule.get("local_time"),
                "status": local_schedule.get("status"),
            }
        )
    return metadata


def market_state(matrix: EventMarketMatrix) -> Dict[str, Any]:
    category_counts: Dict[str, int] = {}
    usable_counts: Dict[str, int] = {}
    for quote in matrix.markets:
        category_counts[quote.category] = category_counts.get(quote.category, 0) + 1
        if quote_is_usable(quote):
            usable_counts[quote.category] = usable_counts.get(quote.category, 0) + 1
    avg_spread, total_liquidity = matrix.liquidity_snapshot()
    return {
        "quote_count": len(matrix.markets),
        "usable_quote_count": sum(1 for quote in matrix.markets if quote_is_usable(quote)),
        "market_count": len(matrix.raw_event.get("markets") or []),
        "category_counts": category_counts,
        "usable_category_counts": usable_counts,
        "avg_spread": avg_spread,
        "total_liquidity": total_liquidity,
    }


def adaptive_causal_layers(quotes: Iterable[MarketQuote]) -> List[Dict[str, Any]]:
    by_layer: Dict[str, Dict[str, float]] = {}
    for quote in quotes:
        layer = quote.causal_layer or causal_profile_for_category(quote.category).causal_layer
        item = by_layer.setdefault(
            layer,
            {
                "quotes": 0.0,
                "usable_quotes": 0.0,
                "avg_base_weight": 0.0,
                "avg_reliability": 0.0,
                "avg_effective_strength": 0.0,
                "total_effective_strength": 0.0,
            },
        )
        item["quotes"] += 1.0
        base_weight = quote.model_weight
        if base_weight is None:
            base_weight = causal_profile_for_category(quote.category).model_weight
        item["avg_base_weight"] += base_weight
        reliability = quote_market_reliability(quote)
        strength = quote_constraint_strength(quote)
        item["avg_reliability"] += reliability
        item["avg_effective_strength"] += strength
        if quote_is_usable(quote):
            item["usable_quotes"] += 1.0
            item["total_effective_strength"] += strength
    total_strength = sum(item["total_effective_strength"] for item in by_layer.values()) or 1.0
    rows = []
    for layer, item in by_layer.items():
        quotes_count = item["quotes"] or 1.0
        rows.append(
            {
                "layer": layer,
                "quotes": int(item["quotes"]),
                "usable_quotes": int(item["usable_quotes"]),
                "avg_base_weight": item["avg_base_weight"] / quotes_count,
                "avg_reliability": item["avg_reliability"] / quotes_count,
                "avg_effective_strength": item["avg_effective_strength"] / quotes_count,
                "total_effective_strength": item["total_effective_strength"],
                "influence_share": item["total_effective_strength"] / total_strength,
            }
        )
    rows.sort(key=lambda row: row["total_effective_strength"], reverse=True)
    return rows


def collapse_constraints(
    constraints: Iterable[Any],
    prior: ScoreDistribution,
    final: ScoreDistribution,
) -> List[Dict[str, Any]]:
    rows = []
    for constraint in constraints:
        prior_prob = prior.probability(constraint.predicate)
        final_prob = final.probability(constraint.predicate)
        rows.append(
            {
                "label": constraint.label,
                "source": constraint.source,
                "tier": constraint.tier,
                "target": constraint.target,
                "prior_probability": prior_prob,
                "final_probability": final_prob,
                "strength": constraint.strength,
                "gap_before": constraint.target - prior_prob,
                "gap_after": constraint.target - final_prob,
                "improvement": abs(constraint.target - prior_prob) - abs(constraint.target - final_prob),
            }
        )
    rows.sort(key=lambda row: (row["tier"] != "primary", -abs(row["improvement"])))
    return rows[:80]


def collapse_layers(constraints: Iterable[Any]) -> List[Dict[str, Any]]:
    by_layer: Dict[str, Dict[str, float]] = {}
    for constraint in constraints:
        layer = layer_from_constraint_source(constraint.source)
        item = by_layer.setdefault(
            layer,
            {
                "constraints": 0.0,
                "avg_strength": 0.0,
                "total_strength": 0.0,
                "primary_constraints": 0.0,
            },
        )
        item["constraints"] += 1.0
        item["avg_strength"] += constraint.strength
        item["total_strength"] += constraint.strength
        if constraint.tier == "primary":
            item["primary_constraints"] += 1.0
    total = sum(item["total_strength"] for item in by_layer.values()) or 1.0
    rows = []
    for layer, item in by_layer.items():
        count = item["constraints"] or 1.0
        rows.append(
            {
                "layer": layer,
                "constraints": int(item["constraints"]),
                "primary_constraints": int(item["primary_constraints"]),
                "avg_strength": item["avg_strength"] / count,
                "total_strength": item["total_strength"],
                "influence_share": item["total_strength"] / total,
            }
        )
    rows.sort(key=lambda row: row["total_strength"], reverse=True)
    return rows


def layer_from_constraint_source(source: str) -> str:
    if source.endswith(":moneyline"):
        return "same_match_result"
    if source.endswith(":handicap"):
        return "same_match_margin"
    if source.endswith(":total_goals"):
        return "same_match_goal_environment"
    if source.endswith(":team_total"):
        return "same_match_team_goal_shape"
    if source.endswith(":btts"):
        return "same_match_goal_correlation"
    if source.endswith(":correct_score"):
        return "same_match_exact_score_tail"
    return "other_constraint"


def moneyline_snapshot(context: Any, matrix: EventMarketMatrix) -> List[Dict[str, Any]]:
    labels = {"home": matrix.home, "draw": "Draw", "away": matrix.away}
    rows = []
    for outcome in ("home", "draw", "away"):
        probability = probability_for_spf(context, outcome) or 0.0
        quote = best_quote_for(matrix, "moneyline", outcome)
        rows.append(
            {
                "outcome": outcome,
                "label": labels[outcome],
                "probability": probability,
                "fair_odds": fair_odds(probability),
                "source_probability": quote.probability if quote else None,
                "spread": quote.spread if quote else None,
                "liquidity": quote.liquidity if quote else None,
                "effective_strength": quote_constraint_strength(quote) if quote else None,
            }
        )
    return rows


def handicap_snapshot(context: Any, matrix: EventMarketMatrix) -> List[Dict[str, Any]]:
    rows = []
    for handicap in handicap_lines(matrix):
        for outcome, label in (("home", "让胜"), ("draw", "让平"), ("away", "让负")):
            probability = probability_for_handicap(context, handicap, outcome, matrix.home, matrix.away) or 0.0
            rows.append(
                {
                    "handicap": handicap,
                    "outcome": outcome,
                    "label": label,
                    "condition": handicap_condition(matrix.home, matrix.away, handicap, outcome),
                    "probability": probability,
                    "fair_odds": fair_odds(probability),
                }
            )
    return rows


def handicap_lines(matrix: EventMarketMatrix) -> List[int]:
    lines = {-3, -2, -1, 1, 2, 3}
    for quote in matrix.quotes("handicap"):
        if not quote_is_usable(quote) or quote.line is None:
            continue
        if normalize_key(quote.entity or "") != normalize_key(matrix.home):
            continue
        derived = int(quote.line + 0.5)
        if derived != 0 and -5 <= derived <= 5:
            lines.add(derived)
    return sorted(lines)


def totals_snapshot(context: Any, matrix: EventMarketMatrix) -> List[Dict[str, Any]]:
    lines = sorted({parsed[1] for quote in matrix.quotes("total_goals") if (parsed := parse_total_quote(quote))})
    rows = []
    for line in lines:
        over = context.score_distribution.probability(lambda h, a, line=line: h + a > line)
        rows.append(
            {
                "line": line,
                "over_probability": over,
                "under_probability": 1.0 - over,
                "over_fair_odds": fair_odds(over),
                "under_fair_odds": fair_odds(1.0 - over),
            }
        )
    return rows


def team_totals_snapshot(context: Any, matrix: EventMarketMatrix) -> List[Dict[str, Any]]:
    entries: set[Tuple[str, float]] = set()
    for quote in matrix.quotes("team_total"):
        parsed = parse_team_total_quote(quote, matrix.home, matrix.away)
        if parsed:
            team, _, line, _ = parsed
            entries.add((team, line))
    rows = []
    for team, line in sorted(entries, key=lambda item: (normalize_key(item[0]), item[1])):
        is_home = normalize_key(team) == normalize_key(matrix.home)
        over = context.score_distribution.probability(
            (lambda h, a, line=line: h > line)
            if is_home
            else (lambda h, a, line=line: a > line)
        )
        rows.append(
            {
                "team": team,
                "line": line,
                "over_probability": over,
                "under_probability": 1.0 - over,
                "over_fair_odds": fair_odds(over),
                "under_fair_odds": fair_odds(1.0 - over),
            }
        )
    return rows


def btts_snapshot(context: Any) -> Dict[str, Any]:
    yes = context.score_distribution.probability(lambda h, a: h > 0 and a > 0)
    return {
        "yes_probability": yes,
        "no_probability": 1.0 - yes,
        "yes_fair_odds": fair_odds(yes),
        "no_fair_odds": fair_odds(1.0 - yes),
    }


def top_scores_snapshot(distribution: ScoreDistribution, limit: int = 12) -> List[Dict[str, Any]]:
    rows = []
    for score, probability in sorted(distribution.probs.items(), key=lambda item: item[1], reverse=True)[:limit]:
        rows.append(
            {
                "score": f"{score[0]}-{score[1]}",
                "home_goals": score[0],
                "away_goals": score[1],
                "probability": probability,
                "fair_odds": fair_odds(probability),
            }
        )
    return rows


def high_probability_quotes_snapshot(quotes: Iterable[MarketQuote], limit: int = 20) -> List[Dict[str, Any]]:
    core_categories = {
        "moneyline",
        "handicap",
        "total_goals",
        "team_total",
        "btts",
        "starting_lineup",
        "halftime_result",
        "second_half_result",
    }
    candidates = [
        quote
        for quote in quotes
        if quote.category in core_categories
        and quote_is_usable(quote)
        and quote.probability is not None
        and 0.60 <= quote.probability < 0.995
        and not quote.is_complement
    ]
    candidates.sort(key=lambda quote: (quote_constraint_strength(quote), quote.probability or 0.0), reverse=True)
    return [
        {
            "category": quote.category,
            "outcome": quote.outcome,
            "probability": quote.probability,
            "fair_odds": fair_odds(quote.probability or 0.0),
            "spread": quote.spread,
            "liquidity": quote.liquidity,
            "causal_layer": quote.causal_layer,
            "effective_strength": quote_constraint_strength(quote),
        }
        for quote in candidates[:limit]
    ]


def candidate_paths(context: Any, matrix: EventMarketMatrix) -> List[Dict[str, Any]]:
    rows = []
    for item in moneyline_snapshot(context, matrix):
        rows.append(
            {
                "play": "spf",
                "outcome": item["outcome"],
                "label": item["label"],
                "probability": item["probability"],
                "fair_odds": item["fair_odds"],
                "risk": probability_risk(item["probability"]),
            }
        )
    for item in handicap_snapshot(context, matrix):
        rows.append(
            {
                "play": f"rq({item['handicap']:+d})",
                "outcome": item["outcome"],
                "label": item["label"],
                "probability": item["probability"],
                "fair_odds": item["fair_odds"],
                "condition": item["condition"],
                "risk": "精准分支" if item["outcome"] == "draw" else probability_risk(item["probability"]),
            }
        )
    rows.sort(key=lambda row: row["probability"], reverse=True)
    return rows


def best_quote_for(matrix: EventMarketMatrix, category: str, outcome: str) -> Optional[MarketQuote]:
    outcome_key = normalize_key(outcome)
    candidates = [
        quote
        for quote in matrix.quotes(category)
        if quote_is_usable(quote) and normalize_key(quote.outcome) == outcome_key
    ]
    if not candidates:
        return None
    return max(candidates, key=quote_constraint_strength)


def probability_risk(probability: float) -> str:
    if probability >= 0.60:
        return "主概率路径"
    if probability >= 0.45:
        return "可保留路径"
    if probability >= 0.25:
        return "次级路径"
    return "小搏/删除候选"


def fair_odds(probability: float) -> Optional[float]:
    if probability <= 0:
        return None
    return 1.0 / probability
