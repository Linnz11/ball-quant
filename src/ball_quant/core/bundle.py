"""bundle.py — the per-match RAW-DATA bundle for the LLM analyst.

This is a DATA SERVICE, not a decision engine. It gathers, per paired match:
  - Polymarket: the full angle set (devigged probs + liquidity, grouped by market
    category) with a per-market `thin` flag (oracle-quality signal),
  - 体彩: all-5-玩法 odds + 单关/过关 sale flags,
  - KG: team strength / injuries / tactics,
  - fundamental: Elo-implied λ pair → 1X2 probs + divergence vs Polymarket
    (cross-check axis; separate from the market signal).

It does NOT compute edge, does NOT pick bets, does NOT build a slip — the LLM
(controller) does that. The whole point of the architecture is: code gathers raw
data; the model reasons over it. (See REFACTOR_PLAN.md §0.)
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ball_quant.adapters.c500 import fetch_odds, load_odds
from ball_quant.core.knowledge_graph import Team, get_team, load_kg
from ball_quant.core.match_join import normalize_team, pair_all
from ball_quant.models import EventMarketMatrix, TicaiOdds

logger = logging.getLogger(__name__)

WORLD_CUP_TAG_ID = 102232

# Poly markets below this liquidity (USD) are flagged thin → their probs are
# low-confidence (per §1.7: thin Poly may be the soft side; don't trust it as oracle).
_THIN_LIQUIDITY = 20000.0

# A main match event slug ends with the date; derivative sub-markets
# (…-halftime-result, …-exact-score, …-more-markets) have a suffix after it.
_MAIN_SLUG_RE = re.compile(r"-\d{4}-\d{2}-\d{2}$")


# ---------------------------------------------------------------------------
# Polymarket matrix loader (main events only, full-market enriched)
# ---------------------------------------------------------------------------

def load_world_cup_matrices(client: Any) -> List[EventMarketMatrix]:
    """Build one EventMarketMatrix per MAIN World Cup match event.

    Polymarket now publishes derivative markets in separate sub-events whose
    slugs share the same stem as the main event but carry a suffix such as
    -halftime-result, -exact-score, -more-markets, -first-to-score,
    -second-half-result, -total-corners.  The main event (slug ends with the
    ISO date) holds only 3 moneyline markets.  We merge all sub-event markets
    into the same MatchMatrix so downstream bundles see the full handicap /
    total_goals / correct_score / etc. set.

    player-props sub-events are excluded to keep the bundle focused on
    match-level markets; add back if player prop analysis is needed.
    """
    from ball_quant.adapters.polymarket import event_to_quotes, infer_match_teams

    # Derivative slug suffix pattern: captures the date-terminated stem.
    _SUB_SUFFIX_RE = re.compile(
        r"^(.*-\d{4}-\d{2}-\d{2})"
        r"-(halftime-result|exact-score|more-markets|first-to-score"
        r"|second-half-result|total-corners)$"
    )

    events = client.fetch_world_cup_events(
        tag_id=WORLD_CUP_TAG_ID, max_events=700, include_closed=False
    )

    # Group events by stem (the date-terminated base slug).
    # main_events: stem -> event dict
    # sub_events:  stem -> list of derivative event dicts
    main_events: Dict[str, Any] = {}
    sub_events: Dict[str, List[Any]] = {}
    for event in events:
        slug = event.get("slug") or ""
        if _MAIN_SLUG_RE.search(slug):
            main_events[slug] = event
        else:
            m = _SUB_SUFFIX_RE.match(slug)
            if m:
                stem = m.group(1)  # e.g. "fifwc-esp-ksa-2026-06-21"
                sub_events.setdefault(stem, []).append(event)

    matrices: List[EventMarketMatrix] = []
    for stem, event in main_events.items():
        enriched = client.prefer_sports_event(event)
        title = str(enriched.get("title") or enriched.get("slug") or "")
        home, away = infer_match_teams(title)

        # Start with the main event's markets (moneyline).
        all_quotes = event_to_quotes(enriched, home, away)

        # Merge in markets from each derivative sub-event.
        for sub_event in sub_events.get(stem, []):
            all_quotes.extend(event_to_quotes(sub_event, home, away))

        matrices.append(
            EventMarketMatrix(
                match_id=str(enriched.get("id") or enriched.get("slug") or ""),
                home=home,
                away=away,
                event_id=str(enriched.get("id") or ""),
                event_slug=enriched.get("slug") or "",
                markets=all_quotes,
                raw_event=enriched,
            )
        )
    return matrices


# ---------------------------------------------------------------------------
# Per-match bundle assembly (pure formatting — no edge, no picks)
# ---------------------------------------------------------------------------

def _poly_angles(matrix: EventMarketMatrix) -> Dict[str, List[Dict[str, Any]]]:
    """Group Poly quotes by category → raw devigged probs + liquidity + thin-flag."""
    angles: Dict[str, List[Dict[str, Any]]] = {}
    for q in matrix.markets:
        if q.probability is None:
            continue
        angles.setdefault(q.category, []).append(
            {
                "outcome": q.outcome,
                "prob": round(q.probability, 4),
                "line": q.line,
                "liquidity": q.liquidity,
                "spread": q.spread,
                "thin": (q.liquidity is not None and q.liquidity < _THIN_LIQUIDITY),
            }
        )
    return angles


def _sale_label(danjuan: bool, guoguan: bool) -> str:
    if danjuan:
        return "单关+过关" if guoguan else "单关"
    return "仅过关" if guoguan else "未开售"


def _ticai_block(t: TicaiOdds) -> Dict[str, Any]:
    """体彩 odds + 单关/过关 sale flags, structured by 玩法."""
    return {
        "胜平负": {"odds": t.spf, "sale": _sale_label(t.spf_danjuan, t.spf_guoguan)},
        "让球胜平负": {
            "line": t.handicap_line,
            "odds": t.rqspf,
            "sale": _sale_label(t.rqspf_danjuan, t.rqspf_guoguan),
        },
        "比分": {
            "odds": t.correct_score,
            "sale": _sale_label(t.correct_score_danjuan, t.correct_score_guoguan),
        },
        "总进球": {
            "odds": t.total_goals,
            "sale": _sale_label(t.total_goals_danjuan, t.total_goals_guoguan),
        },
        "半全场": {"odds": t.hafu, "sale": _sale_label(t.hafu_danjuan, t.hafu_guoguan)},
    }


def _kg_block(team: Optional[Team]) -> Dict[str, Any]:
    if team is None:
        return {"known": False}
    return {
        "known": True,
        "strength_win": team.strength_win,
        "rank": team.strength_rank,
        "recent_form": team.recent_form,
        "key_players": team.key_players,
        "injuries": team.injuries,
        "tactics": team.tactical_notes,
    }


def _poly_1x2(angles: Dict[str, List[Dict[str, Any]]]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Extract devigged home/draw/away probs from the moneyline angle block.

    Returns (p_home, p_draw, p_away) or (None, None, None) if the moneyline
    block is absent or incomplete.  The values come directly from Poly's
    already-devigged probabilities stored by _poly_angles — no re-devigging here.
    """
    rows = angles.get("moneyline", [])
    # Map outcome labels → prob; Polymarket uses "home"/"draw"/"away" strings.
    prob_map: Dict[str, float] = {}
    for r in rows:
        outcome = str(r.get("outcome") or "").lower()
        prob = r.get("prob")
        if prob is not None and outcome in ("home", "draw", "away"):
            prob_map[outcome] = float(prob)
    if len(prob_map) < 3:
        return None, None, None
    return prob_map["home"], prob_map["draw"], prob_map["away"]


def _elo_1x2(lam_home: float, lam_away: float, max_goals: int) -> Tuple[float, float, float]:
    """Convert Elo-implied λ pair → 1X2 probabilities using the engine's Poisson grid.

    WHY reuse poisson_grid: the probability engine uses the same independent-Poisson
    grid (rho=0 → no Dixon-Coles correction) when computing score distributions
    for the Elo-only prior.  This keeps the Elo-implied 1X2 apples-to-apples with
    how the main engine treats any (λ_home, λ_away) pair — both sum over the same
    (max_goals+1)^2 grid.  stdlib-only: math.exp/factorial used inside poisson_grid.
    """
    # Lazy import avoids a circular import (probability imports params; bundle imports params).
    from ball_quant.core.probability import poisson_grid

    grid = poisson_grid(lam_home, lam_away, max_goals, rho=0.0)
    p_home = sum(p for (h, a), p in grid.items() if h > a)
    p_draw = sum(p for (h, a), p in grid.items() if h == a)
    p_away = sum(p for (h, a), p in grid.items() if h < a)
    return p_home, p_draw, p_away


def _fundamental_block(
    matrix: EventMarketMatrix,
    angles: Dict[str, List[Dict[str, Any]]],
    elo_ratings: Dict[str, float],
    params: "StrategyParams",
) -> Dict[str, Any]:
    """Build the fundamental (Elo) cross-check block for one match.

    WHY a cross-check block: the LLM analyst sees both axes side-by-side:
      - market axis: Polymarket devigged moneyline probs (crowd signal)
      - fundamental axis: Elo-implied 1X2 (structural prior, orthogonal to market)
    Δ = poly - elo shows whether the market is more or less bullish than the
    fundamental prior suggests, flagging structural divergences the model should
    reason about.

    home_rated / away_rated flags: elo_lambda_prior falls back to z=0 (average)
    for missing teams — which looks like an "even" read.  We expose the flag so
    the LLM (and rendering layer) can mark unrated teams rather than misreading
    a z=0 fallback as a meaningful rating.
    """
    from ball_quant.core.strength_prior import elo_lambda_prior

    # Normalize to match the key space the adapter produces.
    home_key = normalize_team(matrix.home)
    away_key = normalize_team(matrix.away)
    home_rated = home_key in elo_ratings
    away_rated = away_key in elo_ratings

    lam_home, lam_away = elo_lambda_prior(matrix.home, matrix.away, elo_ratings, params)
    p_home_elo, p_draw_elo, p_away_elo = _elo_1x2(lam_home, lam_away, params.max_goals)

    poly_p_home, poly_p_draw, poly_p_away = _poly_1x2(angles)

    block: Dict[str, Any] = {
        "source": "elo",
        "lam_home": round(lam_home, 4),
        "lam_away": round(lam_away, 4),
        "p_home": round(p_home_elo, 4),
        "p_draw": round(p_draw_elo, 4),
        "p_away": round(p_away_elo, 4),
        "home_rated": home_rated,
        "away_rated": away_rated,
    }

    if poly_p_home is not None:
        # Positive delta = market more bullish on home than fundamental warrants.
        block["poly_p_home"] = round(poly_p_home, 4)
        block["delta_home"] = round(poly_p_home - p_home_elo, 4)
        block["delta_away"] = round(poly_p_away - p_away_elo, 4)  # type: ignore[arg-type]

    return block


def build_bundle(
    matched_pairs: Sequence[Tuple[TicaiOdds, EventMarketMatrix]],
    kg: Optional[Dict[str, Team]] = None,
    elo_ratings: Optional[Dict[str, float]] = None,
    params: Optional["StrategyParams"] = None,
) -> List[Dict[str, Any]]:
    """Assemble per-match raw-data bundles. NO edge, NO slip — data only.

    elo_ratings: if provided, each bundle gains a ``fundamental`` cross-check block
        with Elo-implied λ → 1X2 and divergence vs Polymarket moneyline.
        When None the block is omitted entirely so existing callers are unaffected.
    params: StrategyParams consumed by the Elo prior (max_goals, elo_baseline_goals,
        elo_supremacy_coeff).  Defaults to DEFAULT_PARAMS when omitted.
    """
    from ball_quant.core.params import DEFAULT_PARAMS

    kg = kg or {}
    effective_params = params if params is not None else DEFAULT_PARAMS
    bundles: List[Dict[str, Any]] = []
    for ticai, matrix in matched_pairs:
        avg_spread, total_liq = matrix.liquidity_snapshot()
        angles = _poly_angles(matrix)
        entry: Dict[str, Any] = {
            "poly_home": matrix.home,
            "poly_away": matrix.away,
            "ticai_home": ticai.home,
            "ticai_away": ticai.away,
            "match_date": ticai.match_date,
            "kickoff": ticai.kickoff,    # "YYYY-MM-DDTHH:MM" CST; None when unavailable
            "bet_close": ticai.bet_close,  # "YYYY-MM-DDTHH:MM:SS" CST 停售; trigger = bet_close − 70min
            "match_num": ticai.match_num,
            "event_slug": matrix.event_slug,
            "poly_liquidity": {"avg_spread": avg_spread, "total": total_liq},
            "poly": angles,
            "ticai": _ticai_block(ticai),
            "kg": {
                "home": _kg_block(get_team(matrix.home, kg)),
                "away": _kg_block(get_team(matrix.away, kg)),
            },
        }
        if elo_ratings is not None:
            entry["fundamental"] = _fundamental_block(
                matrix, angles, elo_ratings, effective_params
            )
        bundles.append(entry)
    return bundles


# ---------------------------------------------------------------------------
# Compact markdown render (for human/LLM reading; JSON carries everything)
# ---------------------------------------------------------------------------

def _fmt_odds(d: Dict[Any, float], top: int = 0) -> str:
    items = list(d.items())
    if top:
        items = sorted(items, key=lambda kv: kv[1])[:top]
    return " ".join(f"{k}={v}" for k, v in items)


def render_bundle_markdown(bundles: Sequence[Dict[str, Any]], date: str) -> str:
    lines = [f"# 数据 bundle {date} — {len(bundles)} 场（原始数据，未算 edge）", ""]
    for b in bundles:
        liq = b["poly_liquidity"]
        kickoff_str = f" ｜ 开赛 {b['kickoff']} CST" if b.get("kickoff") else ""
        # Render 停售 as "YYYY-MM-DD HH:MM" (drop seconds for compactness)
        raw_bc = b.get("bet_close") or ""
        bet_close_str = (
            f" ｜ 停售 {raw_bc[:10]} {raw_bc[11:16]}"
            if raw_bc
            else ""
        )
        lines.append(
            f"## {b['poly_home']} vs {b['poly_away']}"
            f"  ｜ {b['ticai_home']}vs{b['ticai_away']} ｜ {b['match_num']}"
            f"{kickoff_str}{bet_close_str} ｜ {b['event_slug']}"
        )
        lines.append(
            f"- Poly 流动性: total=${liq['total']} avg_spread={liq['avg_spread']}"
        )
        # Poly main angles
        for cat in ("moneyline", "handicap", "total_goals", "correct_score", "btts", "halftime_result"):
            rows = b["poly"].get(cat)
            if not rows:
                continue
            thin = " ⚠️薄" if any(r["thin"] for r in rows) else ""
            shown = "; ".join(
                f"{r['outcome']}"
                + (f"({r['line']})" if r["line"] is not None else "")
                + f" {r['prob']}"
                for r in rows[:8]
            )
            lines.append(f"- Poly·{cat}{thin}: {shown}")
        # 体彩
        t = b["ticai"]
        lines.append(
            f"- 体彩 胜平负[{t['胜平负']['sale']}]: {_fmt_odds(t['胜平负']['odds'])}"
        )
        lines.append(
            f"- 体彩 让球({t['让球胜平负']['line']})[{t['让球胜平负']['sale']}]: "
            f"{_fmt_odds(t['让球胜平负']['odds'])}"
        )
        lines.append(
            f"- 体彩 总进球[{t['总进球']['sale']}]: {_fmt_odds(t['总进球']['odds'])}"
        )
        lines.append(
            f"- 体彩 比分[{t['比分']['sale']}] / 半全场[{t['半全场']['sale']}] (见 JSON)"
        )
        # KG
        kh, ka = b["kg"]["home"], b["kg"]["away"]
        lines.append(
            f"- KG: {b['poly_home']} str={kh.get('strength_win')} rank={kh.get('rank')} "
            f"inj={kh.get('injuries')} ｜ {b['poly_away']} str={ka.get('strength_win')} "
            f"rank={ka.get('rank')} inj={ka.get('injuries')}"
        )
        # Fundamental (Elo) cross-check — only present when elo_ratings were injected.
        fund = b.get("fundamental")
        if fund:
            # Flag unrated teams so the LLM knows the z=0 fallback was used.
            home_tag = "" if fund["home_rated"] else " ⚠️未评级"
            away_tag = "" if fund["away_rated"] else " ⚠️未评级"
            elo_line = (
                f"- Elo 基本面{home_tag}{away_tag}: "
                f"λ {fund['lam_home']}/{fund['lam_away']} "
                f"→ P(主/平/客) {fund['p_home']}/{fund['p_draw']}/{fund['p_away']}"
            )
            if "poly_p_home" in fund:
                sign = "+" if fund["delta_home"] >= 0 else ""
                direction = "市场更看好主" if fund["delta_home"] > 0 else ("市场更看好客" if fund["delta_home"] < 0 else "持平")
                # poly_p_away = elo p_away + delta_away (delta = poly - elo for each side)
                poly_p_away = round(fund["p_away"] + fund["delta_away"], 4)
                elo_line += (
                    f" ｜ vs Poly {fund['poly_p_home']}/—/{poly_p_away}"
                    f" ｜ Δ主={sign}{fund['delta_home']} ({direction})"
                )
            lines.append(elo_line)
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Full pipeline (load → pair → assemble)
# ---------------------------------------------------------------------------

def run_bundle(
    date: str,
    c500_cache: Optional[str] = None,
    kg_path: Optional[str] = None,
    client: Any = None,
    params: Optional["StrategyParams"] = None,
) -> Tuple[List[Dict[str, Any]], List[TicaiOdds]]:
    """End-to-end: load 体彩 + Poly + KG + Elo, pair, assemble bundles.

    Returns (bundles, unmatched_ticai). Raises on data-source failure (no silent
    swallow). `client` lets tests inject a fake PolymarketClient.

    Elo ratings are fetched from the cached adapter.  Fetch failure is non-fatal:
    a WARNING is logged and elo_ratings is passed as {} so each bundle gets a
    fundamental block with home_rated=False / away_rated=False — the analyst sees
    the absence clearly rather than silently missing the cross-check entirely.
    We do NOT fabricate ratings: the {} path only triggers the rated=False flags.
    """
    from ball_quant.adapters.elo import fetch_elo_ratings

    if c500_cache:
        ticai_list = load_odds(Path(c500_cache), date=date)
    else:
        ticai_list = fetch_odds(date)

    if client is None:
        from ball_quant.adapters.polymarket import PolymarketClient

        client = PolymarketClient(
            cache_dir=Path("data/cache"), refresh=True, enrich_sports_payload=True
        )
    matrices = load_world_cup_matrices(client)

    kg = load_kg(Path(kg_path)) if kg_path else load_kg()

    # Fetch Elo ratings (cached).  Failure is non-fatal: log + use empty dict so
    # the fundamental block is still present but home_rated/away_rated = False.
    try:
        elo_ratings: Dict[str, float] = fetch_elo_ratings(Path("data/cache"))
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Elo fetch failed — fundamental block will show home_rated=away_rated=False. "
            "Error: %s",
            exc,
        )
        elo_ratings = {}

    matched, unmatched = pair_all(ticai_list, matrices, date_tolerance_days=1)
    bundles = build_bundle(matched, kg, elo_ratings=elo_ratings, params=params)
    return bundles, unmatched
