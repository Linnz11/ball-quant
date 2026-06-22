"""ticai_engine.py — 中国体彩 World Cup betting engine.

SCOPE — EXACTLY the five 竞彩 玩法 covered:
    1. 胜平负       (SPF / HAD)
    2. 让球胜平负   (HHAD / rqspf)
    3. 比分         (correct_score / CRS)
    4. 总进球数     (total_goals / TTG)
    5. 半全场       (HAFU / half-time–full-time)

All five settle on 90-minute regulation time.

NOT 竞彩 (these are Polymarket calibration inputs ONLY — never emitted as 体彩 bets):
    btts / team_total / over_under

CORE STRATEGY:
    Polymarket = PROBABILITY oracle (real-money crowd).
    体彩 = where we BET, at 体彩's posted odds.
    edge = P_polymarket × O_体彩 − 1

    We NEVER use Polymarket odds as the sp on a Selection.  The sp field is
    ALWAYS 体彩's posted decimal odds.  This is asserted in tests.

ARCHITECTURE:
    analyze_ticai()       → List[Selection]  — per-leg P/O/edge/kelly
    rank_recommendations() → dict             — gated + payoff-tilted ranking
    recommend_portfolio()  → dict             — staked 串关 slip
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from ball_quant.core.causal import quote_constraint_strength
from ball_quant.core.combo import generate_combos
from ball_quant.core.params import DEFAULT_PARAMS, StrategyParams
from ball_quant.core.probability import (
    build_probability_context,
    best_usable_quote,
    prior_lambdas,
    probability_for_handicap,
    probability_for_spf,
    quote_constraint_strength as _qcs_alias,  # same as causal.quote_constraint_strength
)
from ball_quant.core.staking import allocate_stakes
from ball_quant.core.value import (
    _settlement_key_for_branch,
    confidence_score,
    kelly_fraction,
    risk_label,
)
from ball_quant.models import (
    Branch,
    EventMarketMatrix,
    MatchSP,
    MarketQuote,
    Selection,
    SettlementKey,
    TeamFacts,
    TicaiOdds,
)

# ---------------------------------------------------------------------------
# HAFU (半全场) first-half goal share calibration bounds
# ---------------------------------------------------------------------------

# Clamp the Polymarket-inferred first_half_goal_share into this range so that
# a sparse/illiquid market cannot produce a degenerate prior for the half model.
# WHY [0.3, 0.6]: a share below 0.3 would imply the first half averages fewer
# than half the typical rate, contradicting decades of data; above 0.6 is
# equally implausible.  Real-world values cluster around 0.42–0.49.
_HAFU_SHARE_CLAMP_LO: float = 0.3
_HAFU_SHARE_CLAMP_HI: float = 0.6

# ---------------------------------------------------------------------------
# Module-level constants (documented with WHY so they can be moved to
# StrategyParams later without hunting for magic numbers)
# ---------------------------------------------------------------------------

# Minimum Polymarket market strength for a quote's probability to be
# "trustworthy enough" to place a 体彩 bet on.
# WHY 0.15: quote_constraint_strength returns ~0.05 for a null quote and
# ~0.20 for a thin/illiquid quote.  0.15 is the floor that requires at least
# some real liquidity; it gates out quotes that are essentially untraded.
_POLY_STRENGTH_FLOOR: float = 0.15

# Minimum total Polymarket liquidity (USD) for the match's moneyline market
# to be used as the grid anchor for grid-derived legs (total_goals,
# correct_score).  WHY 1000: below this threshold the calibration grid is
# mostly prior — probabilities are unreliable for betting purposes.
_MONO_LIQUIDITY_FLOOR: float = 1_000.0

# Minimum probability floor to bet on a leg.  Below this the tail mass is
# too uncertain for a sharp bet even if edge is nominally positive.
# Exception: we do NOT apply this to correct_score "other" bucket because
# that IS the intentional residual tail.
_MIN_PROB_FLOOR: float = 0.02

# Payoff-tilt exponent α for rank_recommendations.
# score = edge × O^α  (α ∈ (0, 1) so higher-odds legs rank up without
# swamping low-odds high-edge legs completely).
# WHY 0.5: square-root blending gives a mild tilt; 1.0 would be proportional
# to odds (dangerous — a 99× long with edge=0.01 would dominate); 0.0 is
# pure-EV.  0.5 is the standard geometric mean compromise.
_PAYOFF_ALPHA: float = 0.5


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_score_key(key: str) -> Optional[Tuple[int, int]]:
    """Parse a 'h-a' score string into (home_goals, away_goals)."""
    m = re.fullmatch(r"(\d{1,2})-(\d{1,2})", key.strip())
    if m:
        return int(m.group(1)), int(m.group(2))
    return None


def _synthetic_match_sp(ticai: TicaiOdds) -> MatchSP:
    """Build a minimal MatchSP from TicaiOdds so build_probability_context
    gets a well-typed input.

    WHY: build_probability_context expects a MatchSP for team names and the
    handicap integer.  The Polymarket calibration grid (from the matrix)
    supplies the actual probability signal; the MatchSP fields (spf_*/rq_*)
    are only used as fallback prices inside probability.py when no Polymarket
    moneyline quote exists.  We pass 体彩 odds into those fallback fields so
    the fallback is at least roughly calibrated to bookmaker priors.
    """
    handicap_int = int(round(ticai.handicap_line or 0.0))
    spf = ticai.spf
    rqspf = ticai.rqspf
    return MatchSP(
        match_id=ticai.match_id,
        date=ticai.match_date,
        home=ticai.home,
        away=ticai.away,
        spf_home=spf.get("home", 1.0),
        spf_draw=spf.get("draw", 1.0),
        spf_away=spf.get("away", 1.0),
        handicap=handicap_int,
        rq_home=rqspf.get("home", 1.0),
        rq_draw=rqspf.get("draw", 1.0),
        rq_away=rqspf.get("away", 1.0),
    )


def _moneyline_liquidity(matrix: EventMarketMatrix) -> float:
    """Total Polymarket moneyline liquidity across home/draw/away quotes."""
    total = 0.0
    for outcome in ("home", "draw", "away"):
        q = best_usable_quote(matrix, "moneyline", outcome)
        if q and q.liquidity is not None:
            total += q.liquidity
    return total


def _make_selection(
    match: MatchSP,
    play: str,
    outcome: str,
    condition: str,
    probability: float,
    sp: float,
    source: str,
    tags: List[str],
    avg_spread: Optional[float],
    total_liquidity: Optional[float],
    facts_adjustment: float,
    params: StrategyParams,
    settlement_key: Optional[SettlementKey],
) -> Selection:
    """Build a Selection with edge = P × O_体彩 − 1 (NEVER Polymarket odds)."""
    # INVARIANT: sp is always the 体彩 posted decimal odds.
    p = probability
    fair_odds = 1.0 / p if p > 0 else float("inf")
    break_even = 1.0 / sp
    edge = p * sp - 1.0  # core formula — body of CORE STRATEGY
    kelly = kelly_fraction(p, sp)
    conf = confidence_score(
        probability=p,
        spread=avg_spread,
        liquidity=total_liquidity,
        facts_adjustment=facts_adjustment,
        source=source,
        params=params,
    )

    # Build a minimal Branch so _settlement_key_for_branch resolves correctly.
    branch = Branch(
        match_id=match.match_id,
        play=play,
        outcome=outcome,
        condition=condition,
        probability=probability,
        source=source,
        tags=tags,
    )
    sk = settlement_key if settlement_key is not None else _settlement_key_for_branch(branch)

    return Selection(
        match_id=match.match_id,
        home=match.home,
        away=match.away,
        play=play,
        outcome=outcome,
        condition=condition,
        probability=p,
        sp=sp,           # ← 体彩 odds — DO NOT CHANGE
        fair_odds=fair_odds,
        break_even=break_even,
        edge=edge,       # = P_poly × O_体彩 − 1
        kelly=kelly,
        confidence=conf,
        risk_label=risk_label(edge, conf, tags),
        tags=list(tags),
        source=source,
        settlement_key=sk,
    )


# ---------------------------------------------------------------------------
# HAFU (半全场) half-time / full-time model
# ---------------------------------------------------------------------------

def _calibrate_first_half_share(
    matrix: EventMarketMatrix,
    ft_lambda_total: float,
    params: StrategyParams,
) -> float:
    """Infer first-half goal share from Polymarket first_half_total_goals market.

    WHY: when Polymarket is quoting a first_half_total_goals over/under, the
    implied 1H expected goals directly constrains the half-split.  We extract
    the best available implied 1H total by finding the over/under pair whose
    line is closest to the market's median (lowest |p_over − 0.5|), compute
    share = 1H_total / FT_total, and clamp to [0.3, 0.6].

    If no usable first_half_total_goals quote exists, return params.first_half_goal_share
    (the statistically-calibrated default of 0.45).
    """
    from ball_quant.core.probability import (
        best_usable_quote,
        clamp_probability,
        market_probability,
        parse_total_quote,
        quote_is_usable,
    )

    by_line: Dict[float, Dict[str, float]] = {}
    for quote in matrix.quotes("first_half_total_goals"):
        if not quote_is_usable(quote):
            continue
        parsed = parse_total_quote(quote)
        if parsed is None or quote.probability is None:
            continue
        side, line, is_complement = parsed
        if is_complement:
            side = "under" if side == "over" else "over"
        p = clamp_probability(market_probability(quote))
        by_line.setdefault(line, {})[side] = p

    if not by_line:
        # No Polymarket 1H market — use default prior.
        return params.first_half_goal_share

    # Select the most informative line: lowest |p_over - 0.5| (closest to median)
    best_line: Optional[float] = None
    best_p_over: Optional[float] = None
    best_distance = float("inf")
    for line, sides in by_line.items():
        over = sides.get("over")
        under = sides.get("under")
        if over is not None and under is not None:
            p_over = over / (over + under)
        elif over is not None:
            p_over = over
        elif under is not None:
            p_over = 1.0 - under
        else:
            continue
        dist = abs(p_over - 0.5)
        if dist < best_distance:
            best_distance = dist
            best_line = line
            best_p_over = p_over

    if best_line is None or best_p_over is None or ft_lambda_total <= 0:
        return params.first_half_goal_share

    # Infer expected 1H total from (line, p_over) under Poisson assumption.
    # For a Poisson(mu) total, P(X > line) = p_over => mu ≈ line + (p_over - 0.5) * nudge.
    # We use the same nudge weight as total_goal_hint for consistency.
    from ball_quant.core.params import DEFAULT_PARAMS
    nudge = params.total_hint_nudge  # default 0.7
    ht_total = max(0.1, best_line + (best_p_over - 0.5) * nudge)
    share = ht_total / ft_lambda_total
    # Clamp to defensible range so sparse quotes can't produce absurd splits.
    return max(_HAFU_SHARE_CLAMP_LO, min(_HAFU_SHARE_CLAMP_HI, share))


def hafu_probabilities(
    home_lambda: float,
    away_lambda: float,
    params: StrategyParams,
    matrix: Optional[EventMarketMatrix] = None,
    score_distribution=None,
) -> Dict[str, float]:
    """Compute 9-bucket HT/FT double-result probabilities.

    METHODOLOGY:
        1. Split FT lambdas into two independent half-lambdas using
           first_half_goal_share (calibrated from Polymarket if available).
        2. Build Poisson grids for 1H and 2H independently (max_goals = params.max_goals).
        3. Enumerate all (h1, a1) × (h2, a2) combinations; FT score = (h1+h2, a1+a2).
        4. Accumulate P(h1,a1) × P(h2,a2) into the 9 HAFU buckets.
        5. MARGINAL RESCALE: anchor each FT-result group (home/draw/away) to the
           calibrated grid's FT marginal so no hafu sub-event can exceed its FT
           marginal.  WHY: the independent-halves convolution produces FT marginals
           that differ from the main calibrated grid (which is fit to Polymarket
           moneyline + totals + correct_score).  Rescaling preserves the HT-conditional
           SHAPE from the convolution while anchoring the FT margin — this guarantees
           P(hafu key) ≤ P(its FT result) for all 9 buckets.

    APPROXIMATION NOTE:
        The two halves are modelled as independent Poisson — this ignores
        in-game state correlation (a team winning 1-0 at HT may defend more,
        depressing 2H goal rates).  This is acceptable for 体彩 hafu pricing;
        after the FT-marginal rescale the probabilities respect the calibrated grid.

    Args:
        score_distribution: Optional pre-built ScoreDistribution from the calibrated
            grid (passed by analyze_ticai to avoid re-fitting).  When None and a
            matrix is provided, marginals come from an independent Poisson grid at
            the same lambdas (slight discrepancy from main grid is acceptable).

    Returns:
        Dict with 9 keys (hh, hd, ha, dh, dd, da, ah, ad, aa), values summing to ~1.0.
        FT groups: sum(hh,dh,ah) == P(FT home), sum(hd,dd,ad) == P(FT draw),
                   sum(ha,da,aa) == P(FT away).
    """
    from ball_quant.core.probability import poisson_grid, clamp_probability

    ft_total = home_lambda + away_lambda

    # Calibrate share from Polymarket 1H market if a matrix is provided.
    if matrix is not None:
        share = _calibrate_first_half_share(matrix, ft_total, params)
    else:
        share = params.first_half_goal_share

    # Split lambdas into halves.
    lam_home_1h = home_lambda * share
    lam_away_1h = away_lambda * share
    lam_home_2h = home_lambda * (1.0 - share)
    lam_away_2h = away_lambda * (1.0 - share)

    # Clamp to lambda_floor so Poisson grids are always valid.
    lam_home_1h = max(params.lambda_floor, lam_home_1h)
    lam_away_1h = max(params.lambda_floor, lam_away_1h)
    lam_home_2h = max(params.lambda_floor, lam_home_2h)
    lam_away_2h = max(params.lambda_floor, lam_away_2h)

    # Build independent Poisson grids for each half.
    # WHY max_goals for each half: the sum h1+h2 can reach 2×max_goals, but
    # in practice P(each half > max_goals) is negligible for normal lambdas.
    # Using the same max_goals ceiling as the FT grid keeps computation O(n^4)
    # but n is small (7 by default) so it is fast enough.
    grid_1h = poisson_grid(lam_home_1h, lam_away_1h, params.max_goals)
    grid_2h = poisson_grid(lam_home_2h, lam_away_2h, params.max_goals)

    buckets: Dict[str, float] = {k: 0.0 for k in ("hh", "hd", "ha", "dh", "dd", "da", "ah", "ad", "aa")}

    def _result_char(home_g: int, away_g: int) -> str:
        if home_g > away_g:
            return "h"
        if home_g == away_g:
            return "d"
        return "a"

    for (h1, a1), p1 in grid_1h.items():
        if p1 <= 0.0:
            continue
        ht_char = _result_char(h1, a1)
        for (h2, a2), p2 in grid_2h.items():
            if p2 <= 0.0:
                continue
            ft_char = _result_char(h1 + h2, a1 + a2)
            key = ht_char + ft_char
            buckets[key] += p1 * p2

    # Normalise so rounding doesn't leave sum slightly off 1.0.
    total = sum(buckets.values())
    if total > 0:
        buckets = {k: v / total for k, v in buckets.items()}

    # ------------------------------------------------------------------
    # FT-MARGINAL RESCALE (Fix A — respect-probability)
    # WHY: the independent-halves convolution's FT marginals differ from the
    # main calibrated grid.  Without this, a hafu sub-event probability can
    # exceed its FT marginal (e.g. P(hafu:aa) > P(FT away)) — impossible.
    # We anchor each FT group to the calibrated grid's FT marginal while
    # preserving the HT-conditional shape from the convolution.
    # ------------------------------------------------------------------
    if score_distribution is not None:
        ft_home_target = score_distribution.probability(lambda h, a: h > a)
        ft_draw_target = score_distribution.probability(lambda h, a: h == a)
        ft_away_target = score_distribution.probability(lambda h, a: h < a)
    else:
        # Fallback: compute FT marginals from the same independent Poisson grid
        # used for the FT lambdas (slight discrepancy from the calibrated grid
        # when rho/Dixon-Coles are active, but keeps the guarantee intact).
        ft_grid = poisson_grid(
            max(params.lambda_floor, home_lambda),
            max(params.lambda_floor, away_lambda),
            params.max_goals,
        )
        ft_home_target = sum(p for (h, a), p in ft_grid.items() if h > a)
        ft_draw_target = sum(p for (h, a), p in ft_grid.items() if h == a)
        ft_away_target = sum(p for (h, a), p in ft_grid.items() if h < a)

    # FT-home group: hh, dh, ah  |  FT-draw group: hd, dd, ad  |  FT-away: ha, da, aa
    _FT_GROUPS = {
        "home": (("hh", "dh", "ah"), ft_home_target),
        "draw": (("hd", "dd", "ad"), ft_draw_target),
        "away": (("ha", "da", "aa"), ft_away_target),
    }
    for _ft_result, (keys, ft_target) in _FT_GROUPS.items():
        group_sum = sum(buckets[k] for k in keys)
        if group_sum > 0.0 and ft_target > 0.0:
            scale = ft_target / group_sum
            for k in keys:
                buckets[k] *= scale
        elif ft_target == 0.0:
            # Edge case: target is zero → zero out all keys in group
            for k in keys:
                buckets[k] = 0.0

    return buckets


# ---------------------------------------------------------------------------
# 1. analyze_ticai
# ---------------------------------------------------------------------------

def analyze_ticai(
    ticai: TicaiOdds,
    matrix: EventMarketMatrix,
    facts: Optional[TeamFacts] = None,
    params: StrategyParams = DEFAULT_PARAMS,
) -> Tuple[List[Selection], List[str]]:
    """Produce a Selection for every 体彩 bet type that has both:
      - a valid 体彩 odds (>1)
      - a Polymarket-derived probability

    Returns (selections, skipped_notes).

    edge = P_polymarket × O_体彩 − 1  for every selection.
    """
    facts_adj = facts.confidence_adjustment if facts is not None else 0.0
    match = _synthetic_match_sp(ticai)
    context = build_probability_context(match, matrix, params)
    grid = context.score_distribution

    avg_spread, total_liquidity = matrix.liquidity_snapshot()
    mono_liquidity = _moneyline_liquidity(matrix)

    selections: List[Selection] = []
    skipped: List[str] = []

    # ------------------------------------------------------------------
    # A. SPF (胜平负)
    # ------------------------------------------------------------------
    for outcome in ("home", "draw", "away"):
        sp = ticai.spf.get(outcome)
        if not sp or sp <= 1.0:
            skipped.append(f"spf:{outcome} — missing or invalid 体彩 odds")
            continue
        prob = probability_for_spf(context, outcome)
        if prob is None:
            skipped.append(f"spf:{outcome} — no Polymarket probability")
            continue
        # Determine source: direct Polymarket moneyline or score-distribution fallback
        q = best_usable_quote(matrix, "moneyline", outcome)
        source = "polymarket:moneyline" if q else "score-distribution"
        selections.append(
            _make_selection(
                match=match,
                play="spf",
                outcome=outcome,
                condition=f"spf {outcome}",
                probability=prob,
                sp=sp,
                source=source,
                tags=[],
                avg_spread=avg_spread,
                total_liquidity=total_liquidity,
                facts_adjustment=facts_adj,
                params=params,
                settlement_key=SettlementKey(market_type="spf", side=outcome),
            )
        )

    # ------------------------------------------------------------------
    # B. Handicap / 让球 (rqspf)
    # ------------------------------------------------------------------
    if ticai.handicap_line is not None:
        handicap_int = int(round(ticai.handicap_line))
        for outcome in ("home", "draw", "away"):
            sp = ticai.rqspf.get(outcome)
            if not sp or sp <= 1.0:
                skipped.append(f"handicap:{outcome} — missing or invalid 体彩 odds")
                continue
            prob = probability_for_handicap(
                context, handicap_int, outcome, match.home, match.away
            )
            if prob is None:
                skipped.append(f"handicap:{outcome} — no Polymarket probability")
                continue
            source = "score-distribution"
            play_str = f"rq({handicap_int:+d})"
            selections.append(
                _make_selection(
                    match=match,
                    play=play_str,
                    outcome=outcome,
                    condition=f"rq({handicap_int:+d}) {outcome}",
                    probability=prob,
                    sp=sp,
                    source=source,
                    tags=["exact_margin"] if outcome == "draw" else [],
                    avg_spread=avg_spread,
                    total_liquidity=total_liquidity,
                    facts_adjustment=facts_adj,
                    params=params,
                    settlement_key=SettlementKey(
                        market_type="handicap",
                        side=outcome,
                        line=float(handicap_int),
                    ),
                )
            )
    else:
        skipped.append("handicap — no handicap_line in TicaiOdds")

    # ------------------------------------------------------------------
    # C. Correct Score (比分)
    # ------------------------------------------------------------------
    # Collect all individually named "h-a" scores first so we can compute
    # the tail residual for any "other" bucket.
    named_scores: Dict[Tuple[int, int], float] = {}
    other_keys: List[str] = []

    for key, sp in ticai.correct_score.items():
        parsed = _parse_score_key(key)
        if parsed is not None:
            named_scores[parsed] = sp
        else:
            other_keys.append(key)

    # Emit individual score legs
    for (h, a), sp in named_scores.items():
        if sp <= 1.0:
            skipped.append(f"correct_score:{h}-{a} — invalid 体彩 odds")
            continue
        prob = grid.probability(lambda H, A, sx=h, sy=a: H == sx and A == sy)
        selections.append(
            _make_selection(
                match=match,
                play="correct_score",
                outcome=f"{h}-{a}",
                condition=f"correct score {h}-{a}",
                probability=prob,
                sp=sp,
                source="score-distribution",
                tags=["exact_margin"],
                avg_spread=avg_spread,
                total_liquidity=total_liquidity,
                facts_adjustment=facts_adj,
                params=params,
                settlement_key=SettlementKey(
                    market_type="correct_score", side=f"{h}-{a}"
                ),
            )
        )

    # Emit "other" bucket(s) — P is the residual tail mass not captured by
    # the explicitly listed scores IN THAT OUTCOME CLASS.
    #
    # The 体彩 "other" buckets cover:
    #   home_other  → all home-win scores not in the explicit home-win list
    #   draw_other  → all draw scores not in the explicit draw list (0-0,1-1,2-2,3-3)
    #   away_other  → all away-win scores not in the explicit away-win list
    # (A bare "other" key with no class prefix is treated as generic residual
    # across all outcome classes and uses 1 − Σ all listed named scores.)
    #
    # CORRECT FORMULA per class:
    #   P(home_other) = P(home wins) − Σ P(listed home-win scores)   [clamp ≥ 0]
    #   P(draw_other) = P(draw)      − Σ P(listed draw scores)       [clamp ≥ 0]
    #   P(away_other) = P(away wins) − Σ P(listed away-win scores)   [clamp ≥ 0]
    #
    # WHY per-class residual (not 1 − Σ all named): assigning the complement of
    # ALL named scores to each "other" bucket inflates it by the probability mass
    # of OTHER classes' listed scores — e.g. draw_other gets credit for home-win
    # listed scores, producing a grossly inflated probability (~0.25 instead of
    # ~0.001) and a fake +466% edge.

    # Pre-compute per-class named-score probability sums.
    p_home_class = grid.probability(lambda H, A: H > A)   # P(any home win)
    p_draw_class = grid.probability(lambda H, A: H == A)  # P(any draw)
    p_away_class = grid.probability(lambda H, A: A > H)   # P(any away win)

    named_home_sum = sum(
        grid.probability(lambda H, A, sx=h, sy=a: H == sx and A == sy)
        for (h, a) in named_scores if h > a
    )
    named_draw_sum = sum(
        grid.probability(lambda H, A, sx=h, sy=a: H == sx and A == sy)
        for (h, a) in named_scores if h == a
    )
    named_away_sum = sum(
        grid.probability(lambda H, A, sx=h, sy=a: H == sx and A == sy)
        for (h, a) in named_scores if a > h
    )
    # Full-complement fallback for bare "other" key (not class-prefixed)
    named_all_sum = named_home_sum + named_draw_sum + named_away_sum

    def _other_residual(key: str) -> float:
        """Return the correct per-class residual probability for an 'other' bucket key."""
        k = key.lower()
        if "home" in k:
            return max(0.0, p_home_class - named_home_sum)
        if "draw" in k:
            return max(0.0, p_draw_class - named_draw_sum)
        if "away" in k:
            return max(0.0, p_away_class - named_away_sum)
        # Generic / bare "other": full complement across all classes
        return max(0.0, 1.0 - named_all_sum)

    for key in other_keys:
        sp = ticai.correct_score.get(key, 0.0)
        if sp <= 1.0:
            skipped.append(f"correct_score:{key} — invalid 体彩 odds")
            continue
        residual = _other_residual(key)
        selections.append(
            _make_selection(
                match=match,
                play="correct_score",
                outcome=key,   # e.g. "other_home", "other_away", "other"
                condition=f"correct score other ({key})",
                probability=residual,
                sp=sp,
                source="score-distribution:residual",
                tags=[],       # NOT exact_margin — this is a bucket, not a single score
                avg_spread=avg_spread,
                total_liquidity=total_liquidity,
                facts_adjustment=facts_adj,
                params=params,
                settlement_key=SettlementKey(
                    market_type="correct_score", side=key
                ),
            )
        )

    # ------------------------------------------------------------------
    # D. Total Goals (进球数)
    # ------------------------------------------------------------------
    for count, sp in ticai.total_goals.items():
        if sp <= 1.0:
            skipped.append(f"total_goals:{count} — invalid 体彩 odds")
            continue
        # count 7 means "7 or more goals" in 体彩 convention
        if count == 7:
            prob = grid.probability(lambda H, A: H + A >= 7)
            outcome_label = "7+"
        else:
            prob = grid.probability(lambda H, A, c=count: H + A == c)
            outcome_label = str(count)
        play_str = f"totals_exact({count})"
        selections.append(
            _make_selection(
                match=match,
                play=play_str,
                outcome=outcome_label,
                condition=f"total goals {outcome_label}",
                probability=prob,
                sp=sp,
                source="score-distribution",
                tags=[],
                avg_spread=avg_spread,
                total_liquidity=total_liquidity,
                facts_adjustment=facts_adj,
                params=params,
                # 体彩 total_goals is a FIXED-count market, not an over/under.
                # Settlement key uses "totals" market_type with the count as
                # side; the grading engine will need to handle this variant.
                settlement_key=SettlementKey(
                    market_type="totals_exact", side=outcome_label
                ),
            )
        )

    # ------------------------------------------------------------------
    # E. HAFU (半全场) — half-time / full-time double result
    # ------------------------------------------------------------------
    # Requires a half-time Poisson model.  We derive FT lambdas from the
    # calibrated grid's prior_lambdas (same source as the FT grid anchor),
    # then split into 1H/2H using first_half_goal_share (calibrated from
    # Polymarket first_half_total_goals when present, else 0.45 default).
    # The two halves are modelled as independent Poisson; their convolution
    # gives the FT distribution used to assign HT+FT bucket probabilities.
    if ticai.hafu:
        home_lambda, away_lambda = prior_lambdas(matrix, params)
        hafu_probs = hafu_probabilities(
            home_lambda, away_lambda, params,
            matrix=matrix,
            score_distribution=grid,  # pass calibrated grid for FT-marginal anchor
        )
        for key, sp in ticai.hafu.items():
            if not sp or sp <= 1.0:
                skipped.append(f"hafu:{key} — missing or invalid 体彩 odds")
                continue
            prob = hafu_probs.get(key)
            if prob is None:
                skipped.append(f"hafu:{key} — unknown key, not in 9-bucket model")
                continue
            selections.append(
                _make_selection(
                    match=match,
                    play="hafu",
                    outcome=key,
                    condition=f"hafu {key}",
                    probability=prob,
                    sp=sp,
                    source="hafu-model:poisson-split",
                    tags=[],
                    avg_spread=avg_spread,
                    total_liquidity=total_liquidity,
                    facts_adjustment=facts_adj,
                    params=params,
                    settlement_key=SettlementKey(market_type="hafu", side=key),
                )
            )

    return selections, skipped


# ---------------------------------------------------------------------------
# 2. rank_recommendations
# ---------------------------------------------------------------------------

def rank_recommendations(
    selections: List[Selection],
    matrix: EventMarketMatrix,
    params: StrategyParams = DEFAULT_PARAMS,
    min_edge: float = 0.0,
) -> Dict[str, Any]:
    """Gate + rank 体彩 bet selections.

    PROBABILITY-RESPECT GATE (hard — both conditions must hold):
      1. edge > min_edge (positive EV)
      2. Probability is trustworthy:
         - For SPF/handicap legs: the relevant Polymarket moneyline quote
           must have quote_constraint_strength ≥ _POLY_STRENGTH_FLOOR.
         - For grid-derived legs (total_goals, correct_score): the total
           moneyline liquidity across home/draw/away must be ≥
           _MONO_LIQUIDITY_FLOOR (the grid is only as good as its anchor).
         - Any leg whose probability is below _MIN_PROB_FLOOR is also gated
           out UNLESS it is a correct_score "other" bucket (intentional tail).
         - Negative-edge legs are NEVER ranked, even if they have high odds.

    PAYOFF-RATIO RANKING (among gated +EV legs):
      score = edge × (O_体彩 ^ _PAYOFF_ALPHA)
      Higher-odds +EV legs rank above low-odds +EV legs with equal edge.
      A negative-edge leg can never have a positive score (edge < 0 means
      the whole expression is negative — mathematically excluded).

    Returns:
      {
        "ranked":      [...] payoff-tilted order (best first),
        "by_pure_ev":  [...] pure-edge order (for reference),
        "gated_out":   [{"selection": ..., "reason": str}, ...],
        "skipped_hafu": bool,
      }
    """
    mono_liquidity = _moneyline_liquidity(matrix)

    gated: List[Selection] = []
    gated_out: List[Dict[str, Any]] = []

    for sel in selections:
        reason = _gate_check(sel, matrix, mono_liquidity, min_edge, params)
        if reason:
            gated_out.append({"selection": sel, "reason": reason})
        else:
            gated.append(sel)

    def payoff_score(s: Selection) -> float:
        # edge is already positive for gated selections; alpha-tilt by odds
        return s.edge * (s.sp ** _PAYOFF_ALPHA)

    ranked = sorted(gated, key=payoff_score, reverse=True)
    by_pure_ev = sorted(gated, key=lambda s: s.edge, reverse=True)

    return {
        "ranked": ranked,
        "by_pure_ev": by_pure_ev,
        "gated_out": gated_out,
    }


def _has_polymarket_category(matrix: EventMarketMatrix, *categories: str) -> bool:
    """Return True if any quote in matrix belongs to one of the given categories.

    WHY: tail 玩法 (totals_exact / correct_score / hafu) are calibrated by grid
    extrapolation from the moneyline anchor.  When no Polymarket market directly
    constrains these 玩法, the probabilities are prior-only guesses — we must not
    recommend them as real bets.  This check gates out prior-only extrapolations.
    """
    for quote in matrix.quotes():
        if quote.category in categories:
            return True
    return False


def _gate_check(
    sel: Selection,
    matrix: EventMarketMatrix,
    mono_liquidity: float,
    min_edge: float,
    params: StrategyParams,
) -> str:
    """Return a non-empty reason string if the selection should be gated out."""
    # Hard EV gate
    if sel.edge <= min_edge:
        return f"edge {sel.edge:.4f} ≤ min_edge {min_edge}"

    is_score_bucket = (
        sel.play == "correct_score"
        and not re.fullmatch(r"\d{1,2}-\d{1,2}", sel.outcome)
    )

    # Determine which quote drives this leg's probability
    is_spf = sel.play == "spf"
    is_handicap = sel.play.startswith("rq(")

    # RESPECT-PROBABILITY GATE (Polymarket anchor): run BEFORE probability floor
    # so the anchor reason is the authoritative gate for tail 玩法 — the prob
    # floor is a secondary quality guard for thin priors, not the root cause.
    if sel.play.startswith("totals_exact"):
        if not _has_polymarket_category(matrix, "total_goals"):
            return "no_polymarket_anchor: totals_exact requires Polymarket total_goals market"
    elif sel.play == "correct_score":
        if not _has_polymarket_category(matrix, "correct_score"):
            return "no_polymarket_anchor: correct_score requires Polymarket correct_score market"
    elif sel.play == "hafu":
        if not _has_polymarket_category(matrix, "first_half_total_goals", "halftime_result", "first_half_result"):
            return "no_polymarket_anchor: hafu requires Polymarket first_half_total_goals or halftime_result market"

    # Probability floor (skip for "other" buckets — they are intentional tails)
    if not is_score_bucket and sel.probability < _MIN_PROB_FLOOR:
        return (
            f"probability {sel.probability:.4f} < floor {_MIN_PROB_FLOOR} "
            "(too thin a tail for sharp bet)"
        )

    if is_spf:
        # Check moneyline quote strength for the specific outcome
        q = best_usable_quote(matrix, "moneyline", sel.outcome)
        strength = quote_constraint_strength(q, params)
        if strength < _POLY_STRENGTH_FLOOR:
            return (
                f"Polymarket moneyline:{sel.outcome} strength {strength:.3f} "
                f"< floor {_POLY_STRENGTH_FLOOR} (too thin — probability unreliable)"
            )
    elif is_handicap:
        # Handicap probability anchored by moneyline; use total moneyline liquidity
        if mono_liquidity < _MONO_LIQUIDITY_FLOOR:
            return (
                f"moneyline liquidity ${mono_liquidity:.0f} "
                f"< floor ${_MONO_LIQUIDITY_FLOOR:.0f} "
                "(grid unreliable — handicap probability not trustworthy)"
            )
    else:
        # Grid-derived legs (totals_exact, correct_score, hafu, and others):
        # Polymarket anchor check already handled above; now check moneyline
        # liquidity as the grid calibration floor.
        if mono_liquidity < _MONO_LIQUIDITY_FLOOR:
            return (
                f"moneyline liquidity ${mono_liquidity:.0f} "
                f"< floor ${_MONO_LIQUIDITY_FLOOR:.0f} "
                "(grid-derived probability not trustworthy)"
            )

    return ""  # passed all gates


# ---------------------------------------------------------------------------
# 3. recommend_portfolio
# ---------------------------------------------------------------------------

def recommend_portfolio(
    ranked_selections: List[Selection],
    budget: float,
    params: StrategyParams = DEFAULT_PARAMS,
) -> Dict[str, Any]:
    """Generate a staked 体彩 betting slip (singles + 串关).

    Uses generate_combos + allocate_stakes on the gated selections.
    Returns:
      {
        "combos": List[Combo],          — all combos with stake > 0
        "singles": List[Selection],     — high-盈亏比 singles (type A)
        "total_stake": float,
        "expected_profit": float,       — Σ(stake × P × O) − total_stake
      }
    """
    if not ranked_selections:
        return {
            "combos": [],
            "singles": [],
            "total_stake": 0.0,
            "expected_profit": 0.0,
            "note": "No gated selections — nothing to stake.",
        }

    combo_groups = generate_combos(ranked_selections, params=params)
    staked = allocate_stakes(combo_groups, budget=budget, params=params)

    singles = [
        combo for combo in staked
        if len(combo.selections) == 1
    ]
    total_stake = sum(c.stake for c in staked)
    expected_profit = sum(
        c.stake * c.probability * c.odds - c.stake for c in staked
    )

    return {
        "combos": staked,
        "singles": [c.selections[0] for c in singles],
        "total_stake": total_stake,
        "expected_profit": expected_profit,
    }
