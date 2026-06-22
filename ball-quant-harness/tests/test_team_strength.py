"""Tests for core/team_strength.py.

Coverage:
- strength_from_futures: devigged probs sum to 1.0; shorter-odds team gets
  higher strength_win and lower rank.
- Team keys canonicalised through normalize_team.
- group_advancement / group_winner → strength_advance.
- Quotes with None probability are skipped.
- update_kg_from_futures: populated KG; qualitative fields preserved.
- CLI smoke: kg-build --poly-dump with a minimal fixture → teams populated.
"""
import json
import tempfile
from pathlib import Path
from typing import List

import pytest

from ball_quant.models import MarketQuote
from ball_quant.core.team_strength import strength_from_futures, update_kg_from_futures
from ball_quant.core.knowledge_graph import load_kg, save_kg, Team


# ---------------------------------------------------------------------------
# Helper — build a MarketQuote with minimal required fields
# ---------------------------------------------------------------------------

def _tw_quote(outcome: str, probability: float) -> MarketQuote:
    """Create a tournament_winner MarketQuote."""
    return MarketQuote(
        market_id="test",
        question="Who wins the World Cup?",
        category="tournament_winner",
        outcome=outcome,
        probability=probability,
    )


def _ga_quote(outcome: str, probability: float) -> MarketQuote:
    """Create a group_advancement MarketQuote."""
    return MarketQuote(
        market_id="test_ga",
        question="Will Brazil advance from group?",
        category="group_advancement",
        outcome=outcome,
        probability=probability,
    )


def _gw_quote(outcome: str, probability: float) -> MarketQuote:
    """Create a group_winner MarketQuote."""
    return MarketQuote(
        market_id="test_gw",
        question="Group D winner?",
        category="group_winner",
        outcome=outcome,
        probability=probability,
    )


# ---------------------------------------------------------------------------
# Core derivation
# ---------------------------------------------------------------------------

def _construct_tournament_market() -> List[MarketQuote]:
    """A realistic WC futures market (raw implied probs sum > 1 = vig present).

    Prices as raw implied (= 1/decimal_odds) before devig.
    Short list of 8 teams for test efficiency.
    """
    return [
        _tw_quote("Brazil", 0.21),
        _tw_quote("Spain", 0.16),
        _tw_quote("France", 0.14),
        _tw_quote("England", 0.11),
        _tw_quote("Argentina", 0.10),
        _tw_quote("Germany", 0.09),
        _tw_quote("Portugal", 0.07),
        _tw_quote("Netherlands", 0.06),
        # Deliberately make booksum ≈ 0.94 to exercise devig normalisation
        # (a devig-by-normalisation should push all probs up proportionally)
    ]


def test_strength_win_sums_to_1():
    """Devigged tournament_winner probabilities must sum to 1.0."""
    quotes = _construct_tournament_market()
    result = strength_from_futures(quotes, devig="proportional")
    total = sum(v["strength_win"] for v in result.values() if "strength_win" in v)
    assert abs(total - 1.0) < 1e-6, f"sum={total}"


def test_strength_win_sums_to_1_shin():
    """Shin devig also produces probs that sum to 1.0 (overround market)."""
    # Deliberate overround (booksum ≈ 1.06) — exercises the Shin bisection path.
    quotes = [
        _tw_quote("Brazil", 0.22),
        _tw_quote("Spain", 0.17),
        _tw_quote("France", 0.15),
        _tw_quote("England", 0.12),
        _tw_quote("Argentina", 0.11),
        _tw_quote("Germany", 0.10),
        _tw_quote("Portugal", 0.08),
        _tw_quote("Netherlands", 0.07),
        _tw_quote("Morocco", 0.04),
    ]  # sum ≈ 1.06 — a genuine bookmaker overround
    result = strength_from_futures(quotes, devig="shin")
    total = sum(v["strength_win"] for v in result.values() if "strength_win" in v)
    assert abs(total - 1.0) < 1e-5, f"sum={total}"


def test_shorter_odds_higher_strength():
    """Brazil (highest raw prob) should have the highest strength_win after devig."""
    quotes = _construct_tournament_market()
    result = strength_from_futures(quotes)
    brazil_win = result["brazil"]["strength_win"]
    spain_win = result["spain"]["strength_win"]
    netherlands_win = result["netherlands"]["strength_win"]
    assert brazil_win > spain_win > netherlands_win


def test_rank_order_correct():
    """Brazil should have rank=1 (highest strength), Netherlands rank=8."""
    quotes = _construct_tournament_market()
    result = strength_from_futures(quotes)
    assert result["brazil"]["strength_rank"] == 1
    assert result["netherlands"]["strength_rank"] == 8


def test_team_keys_canonicalized():
    """Outcome names from Polymarket (title-cased) are normalised to canonical keys."""
    quotes = [
        _tw_quote("Brazil", 0.30),
        _tw_quote("巴西", 0.25),   # Chinese alias — both should collapse to "brazil"
    ]
    result = strength_from_futures(quotes)
    # Both map to "brazil" — only one entry after canonicalisation
    assert "brazil" in result
    # The key "巴西" (normalized → "brazil") should NOT appear as a separate key
    assert "巴西" not in result


def test_none_probability_skipped():
    """Quotes with probability=None are ignored gracefully."""
    quotes = [
        _tw_quote("Brazil", 0.30),
        MarketQuote(
            market_id="x",
            question="q",
            category="tournament_winner",
            outcome="Spain",
            probability=None,
        ),
        _tw_quote("France", 0.20),
    ]
    result = strength_from_futures(quotes)
    assert "brazil" in result
    # "france" maps to canonical "france"
    assert "france" in result
    # spain had None probability — may or may not appear but should not crash
    # (if it does appear, its strength_win should be 0)
    if "spain" in result:
        assert result["spain"]["strength_win"] == 0.0


def test_empty_quotes_returns_empty():
    """No quotes → empty result, no crash."""
    result = strength_from_futures([])
    assert result == {}


# ---------------------------------------------------------------------------
# Group advancement
# ---------------------------------------------------------------------------

def test_group_advancement_populates_strength_advance():
    """group_advancement quotes → strength_advance on each team."""
    quotes = [
        _ga_quote("Brazil", 0.82),
        _ga_quote("Spain", 0.75),
        _ga_quote("Cameroon", 0.35),
        _ga_quote("Serbia", 0.28),
    ]
    result = strength_from_futures(quotes)
    assert "brazil" in result
    assert "strength_advance" in result["brazil"]
    assert result["brazil"]["strength_advance"] > result["serbia"]["strength_advance"]


def test_group_winner_fallback_for_advance():
    """group_winner is used as fallback when no group_advancement quotes exist."""
    quotes = [
        _gw_quote("Brazil", 0.50),
        _gw_quote("Spain", 0.30),
    ]
    result = strength_from_futures(quotes)
    assert "brazil" in result
    assert "strength_advance" in result["brazil"]


def test_group_advancement_preferred_over_group_winner():
    """When both ga and gw quotes exist, ga is used for strength_advance."""
    quotes = [
        _ga_quote("Brazil", 0.80),
        _ga_quote("Spain", 0.72),
        _gw_quote("Brazil", 0.55),  # should be ignored for strength_advance
        _gw_quote("Spain", 0.35),
    ]
    result = strength_from_futures(quotes)
    # With ga devig: brazil's strength_advance should be close to 80/(80+72)
    assert "strength_advance" in result["brazil"]
    # Ensure it's NOT the gw price (0.55 raw) but the ga price (0.80 raw devigged)
    # The ga probabilities are higher raw than gw — so the devigged advance > gw raw
    ga_raw_total = 0.80 + 0.72
    expected_brazil_adv = 0.80 / ga_raw_total
    assert abs(result["brazil"]["strength_advance"] - expected_brazil_adv) < 1e-6


# ---------------------------------------------------------------------------
# Combined tournament_winner + group_advancement
# ---------------------------------------------------------------------------

def test_combined_tw_and_ga():
    """tournament_winner → strength_win; group_advancement → strength_advance; both present."""
    quotes = [
        _tw_quote("Brazil", 0.20),
        _tw_quote("Spain", 0.15),
        _ga_quote("Brazil", 0.80),
        _ga_quote("Spain", 0.70),
    ]
    result = strength_from_futures(quotes)
    assert "strength_win" in result["brazil"]
    assert "strength_advance" in result["brazil"]
    assert "strength_win" in result["spain"]
    assert "strength_advance" in result["spain"]


# ---------------------------------------------------------------------------
# update_kg_from_futures
# ---------------------------------------------------------------------------

def test_update_kg_populates_empty_kg():
    """update_kg_from_futures on empty KG creates team entries."""
    kg = {}
    quotes = [
        _tw_quote("Brazil", 0.20),
        _tw_quote("Spain", 0.15),
        _ga_quote("Brazil", 0.80),
    ]
    update_kg_from_futures(quotes, kg)
    assert "brazil" in kg
    assert "spain" in kg
    assert kg["brazil"].strength_win is not None
    assert kg["brazil"].strength_advance is not None


def test_update_kg_preserves_qualitative():
    """Strength refresh via update_kg_from_futures preserves existing qualitative fields."""
    from ball_quant.core.knowledge_graph import upsert_team
    kg = {}
    # Manually insert an analyst-enriched entry
    analyst_team = Team(
        name="brazil",
        recent_form="WWWDW",
        key_players=["Vinicius"],
        injuries=["Neymar out"],
        tactical_notes="High press block",
    )
    upsert_team(analyst_team, kg)

    # Run automated strength update
    quotes = [
        _tw_quote("Brazil", 0.25),
        _tw_quote("Spain", 0.18),
    ]
    update_kg_from_futures(quotes, kg)

    assert kg["brazil"].recent_form == "WWWDW"
    assert kg["brazil"].key_players == ["Vinicius"]
    assert kg["brazil"].injuries == ["Neymar out"]
    assert kg["brazil"].tactical_notes == "High press block"
    assert kg["brazil"].strength_win is not None


def test_update_kg_returns_kg():
    """update_kg_from_futures returns the mutated kg dict."""
    kg = {}
    quotes = [_tw_quote("Brazil", 0.20)]
    returned = update_kg_from_futures(quotes, kg)
    assert returned is kg


# ---------------------------------------------------------------------------
# CLI smoke — kg-build --poly-dump
# ---------------------------------------------------------------------------

def _make_poly_dump_fixture(tmp_path: Path) -> Path:
    """Build a minimal poly-dump JSON with tournament_winner and group_advancement markets."""
    dump = {
        "fetched_at": 1234567890,
        "events": [
            {
                "title": "2026 FIFA World Cup Winner",
                "quote_count": 4,
                "markets": [
                    {
                        "market_id": "tw1",
                        "question": "Will Brazil win the World Cup?",
                        "category": "tournament_winner",
                        "outcome": "Brazil",
                        "probability": 0.20,
                        "token_id": None,
                        "bid": None,
                        "ask": None,
                        "spread": None,
                        "liquidity": None,
                        "volume": None,
                        "sports_type": None,
                        "line": None,
                        "period": None,
                        "side": None,
                        "entity": None,
                        "scope": None,
                        "horizon": None,
                        "causal_layer": None,
                        "model_weight": None,
                        "is_complement": False,
                        "active": True,
                        "closed": False,
                        "accepting_orders": True,
                        "raw": {},
                    },
                    {
                        "market_id": "tw2",
                        "question": "Will Spain win the World Cup?",
                        "category": "tournament_winner",
                        "outcome": "Spain",
                        "probability": 0.15,
                        "token_id": None,
                        "bid": None,
                        "ask": None,
                        "spread": None,
                        "liquidity": None,
                        "volume": None,
                        "sports_type": None,
                        "line": None,
                        "period": None,
                        "side": None,
                        "entity": None,
                        "scope": None,
                        "horizon": None,
                        "causal_layer": None,
                        "model_weight": None,
                        "is_complement": False,
                        "active": True,
                        "closed": False,
                        "accepting_orders": True,
                        "raw": {},
                    },
                    {
                        "market_id": "ga1",
                        "question": "Will Brazil advance from group?",
                        "category": "group_advancement",
                        "outcome": "Brazil",
                        "probability": 0.82,
                        "token_id": None,
                        "bid": None,
                        "ask": None,
                        "spread": None,
                        "liquidity": None,
                        "volume": None,
                        "sports_type": None,
                        "line": None,
                        "period": None,
                        "side": None,
                        "entity": None,
                        "scope": None,
                        "horizon": None,
                        "causal_layer": None,
                        "model_weight": None,
                        "is_complement": False,
                        "active": True,
                        "closed": False,
                        "accepting_orders": True,
                        "raw": {},
                    },
                ],
            }
        ],
    }
    dump_path = tmp_path / "poly_dump.json"
    dump_path.write_text(json.dumps(dump), encoding="utf-8")
    return dump_path


def test_cli_kg_build_offline(tmp_path, capsys):
    """kg-build --poly-dump populates the KG from the fixture."""
    from ball_quant.cli import cmd_kg_build
    import argparse

    dump_path = _make_poly_dump_fixture(tmp_path)
    kg_out = tmp_path / "teams.json"

    args = argparse.Namespace(
        poly_dump=str(dump_path),
        live=False,
        devig="proportional",
        kg_out=str(kg_out),
        world_cup_tag_id=102232,
        max_world_cup_events=700,
    )
    rc = cmd_kg_build(args)
    assert rc == 0, "cmd_kg_build returned non-zero exit code"

    # Check the KG file was written
    assert kg_out.exists()
    kg = load_kg(kg_out)
    assert "brazil" in kg
    assert "spain" in kg
    assert kg["brazil"].strength_win is not None
    assert kg["spain"].strength_win is not None
    # Brazil should be stronger (higher raw prob before devig)
    assert kg["brazil"].strength_win > kg["spain"].strength_win
    # Brazil should have strength_advance from the ga quote
    assert kg["brazil"].strength_advance is not None

    captured = capsys.readouterr()
    assert "teams in KG" in captured.out
    assert "KG written" in captured.out


def test_cli_kg_build_top10_printed(tmp_path, capsys):
    """kg-build prints the top-10 ranking if teams were found."""
    from ball_quant.cli import cmd_kg_build
    import argparse

    dump_path = _make_poly_dump_fixture(tmp_path)
    kg_out = tmp_path / "teams.json"

    args = argparse.Namespace(
        poly_dump=str(dump_path),
        live=False,
        devig="proportional",
        kg_out=str(kg_out),
        world_cup_tag_id=102232,
        max_world_cup_events=700,
    )
    cmd_kg_build(args)
    captured = capsys.readouterr()
    assert "Top-10" in captured.out
    assert "brazil" in captured.out.lower()


def test_cli_kg_build_no_futures_quotes(tmp_path, capsys):
    """kg-build with a dump containing no futures markets produces an empty KG without crashing."""
    from ball_quant.cli import cmd_kg_build
    import argparse

    # Dump with only moneyline quotes — no futures
    dump = {
        "events": [
            {
                "markets": [
                    {
                        "market_id": "m1",
                        "question": "Brazil vs Spain",
                        "category": "moneyline",
                        "outcome": "Brazil",
                        "probability": 0.60,
                        "token_id": None,
                        "bid": None,
                        "ask": None,
                        "spread": None,
                        "liquidity": None,
                        "volume": None,
                        "sports_type": None,
                        "line": None,
                        "period": None,
                        "side": None,
                        "entity": None,
                        "scope": None,
                        "horizon": None,
                        "causal_layer": None,
                        "model_weight": None,
                        "is_complement": False,
                        "active": True,
                        "closed": False,
                        "accepting_orders": True,
                        "raw": {},
                    }
                ]
            }
        ]
    }
    dump_path = tmp_path / "dump.json"
    dump_path.write_text(json.dumps(dump), encoding="utf-8")
    kg_out = tmp_path / "teams.json"

    args = argparse.Namespace(
        poly_dump=str(dump_path),
        live=False,
        devig="proportional",
        kg_out=str(kg_out),
        world_cup_tag_id=102232,
        max_world_cup_events=700,
    )
    rc = cmd_kg_build(args)
    assert rc == 0
    kg = load_kg(kg_out)
    assert kg == {}  # moneyline quotes don't populate the KG


# ---------------------------------------------------------------------------
# Complement / "not*" outcome filtering
# ---------------------------------------------------------------------------

def test_complement_outcomes_excluded():
    """tournament_winner market with not*/complement outcomes must exclude them.

    Polymarket binary "Will X win?" markets produce a YES token (real team) and
    a NO token whose outcome normalises to e.g. "notitaly".  The NO-side must be
    dropped before devigging so the entity set and magnitudes are correct.
    """
    # Simulate a 3-team market where each team has a paired not* outcome.
    # Raw probs for YES sides sum to ~0.36; if NO sides were included the
    # devigged probs would be tiny (~0.06 each instead of ~0.33).
    from ball_quant.models import MarketQuote as MQ

    def _yes(team: str, p: float) -> MQ:
        return MQ(
            market_id="test_comp",
            question=f"Will {team} win the World Cup?",
            category="tournament_winner",
            outcome=team,
            probability=p,
            is_complement=False,
        )

    def _no(team: str, p: float) -> MQ:
        # Mirrors what normalize_outcome() produces for the NO-side token
        return MQ(
            market_id="test_comp",
            question=f"Will {team} win the World Cup?",
            category="tournament_winner",
            outcome=f"not:{team}",
            probability=p,
            is_complement=True,
        )

    quotes = [
        _yes("Brazil",   0.20),
        _no("Brazil",    0.80),
        _yes("Spain",    0.14),
        _no("Spain",     0.86),
        _yes("France",   0.12),
        _no("France",    0.88),
    ]

    result = strength_from_futures(quotes, devig="proportional")

    # Only real teams should appear — no "not*" entries
    for key in result:
        assert not key.startswith("not"), f"complement key leaked: {key!r}"

    assert "brazil" in result
    assert "spain" in result
    assert "france" in result

    # Devigged probs over the 3-team cleaned set should sum to 1
    total = sum(v["strength_win"] for v in result.values() if "strength_win" in v)
    assert abs(total - 1.0) < 1e-6, f"sum={total}"

    # Favourite (Brazil, highest raw prob) must have sane magnitude
    # In a 3-team market the winner should have strength_win > 0.30
    assert result["brazil"]["strength_win"] > 0.30, (
        f"Brazil strength_win={result['brazil']['strength_win']:.4f} too small — "
        "complement filtering may have failed"
    )

    # Brazil should be rank 1
    assert result["brazil"]["strength_rank"] == 1, (
        f"Brazil rank={result['brazil']['strength_rank']}, expected 1"
    )


def test_not_prefix_name_excluded_even_without_is_complement_flag():
    """Secondary guard: outcome whose normalized name starts with 'not' is filtered
    even if is_complement=False (defensive fallback for unknown naming variants).
    """
    from ball_quant.models import MarketQuote as MQ

    quotes = [
        MQ(market_id="x", question="q", category="tournament_winner",
           outcome="Brazil",    probability=0.20, is_complement=False),
        MQ(market_id="x", question="q", category="tournament_winner",
           outcome="notbrazil", probability=0.80, is_complement=False),  # not flagged
        MQ(market_id="x", question="q", category="tournament_winner",
           outcome="Spain",     probability=0.15, is_complement=False),
    ]
    result = strength_from_futures(quotes, devig="proportional")
    assert "notbrazil" not in result
    assert "brazil" in result
    assert "spain" in result
    total = sum(v["strength_win"] for v in result.values() if "strength_win" in v)
    assert abs(total - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# Non-national-team outcome filtering (continent / placeholder / catch-all)
# ---------------------------------------------------------------------------

def test_continent_outcomes_excluded():
    """tournament_winner quotes mixing real teams + continent outcomes must
    exclude confederation/region entries (europeuefa, northamericaconcacaf,
    africacaf, asiaafc, oceaniaofc, anothercontinent) and devig only over
    real teams.

    Mirrors the live Polymarket "Which continent will win?" market which is
    tagged tournament_winner and pollutes the devig pool.
    """
    from ball_quant.models import MarketQuote as MQ

    def _tw(outcome: str, p: float) -> MQ:
        return MQ(market_id="t", question="q", category="tournament_winner",
                  outcome=outcome, probability=p, is_complement=False)

    quotes = [
        # Real teams
        _tw("Spain",   0.20),
        _tw("France",  0.18),
        _tw("Brazil",  0.22),
        _tw("England", 0.12),
        # Continent / confederation outcomes (from the "Which continent?" market)
        _tw("Europe (UEFA)",           0.70),
        _tw("South America (CONMEBOL)",0.20),
        _tw("North America (CONCACAF)",0.04),
        _tw("Africa (CAF)",            0.03),
        _tw("Asia (AFC)",              0.02),
        _tw("Oceania (OFC)",           0.005),
        _tw("another continent",       0.005),
    ]
    result = strength_from_futures(quotes, devig="proportional")

    # Continent entries must be absent
    for bad in ("europeuefa", "southamericaconmebol", "northamericaconcacaf",
                "africacaf", "asiaafc", "oceaniaofc", "anothercontinent",
                "another", "anothercontinent"):
        assert bad not in result, f"non-team key leaked: {bad!r}"

    # Real teams must be present
    for team in ("spain", "france", "brazil", "england"):
        assert team in result, f"real team missing: {team!r}"

    # Devig over only the 4 real teams must sum to 1
    total = sum(v["strength_win"] for v in result.values() if "strength_win" in v)
    assert abs(total - 1.0) < 1e-6, f"strength_win sum={total}"

    # Brazil (highest raw prob among real teams) must be rank 1
    assert result["brazil"]["strength_rank"] == 1

    # Favourite magnitude must be sane — in a 4-team pool, Brazil ~0.30
    assert result["brazil"]["strength_win"] > 0.25, (
        f"Brazil strength_win={result['brazil']['strength_win']:.4f} — too low, "
        "continent outcomes may still be polluting the pool"
    )


def test_placeholder_team_slots_excluded():
    """tournament_winner quotes with unresolved qualification placeholders
    ("Team AM", "Team AI", "Team AG" …) must be excluded.

    These appear in Polymarket's "World Cup Winner" market for spots not yet
    decided by qualification; they normalize to teamam, teamai, teamag etc.
    Including them inflates the outcome count and depresses real-team magnitudes.
    """
    from ball_quant.models import MarketQuote as MQ

    def _tw(outcome: str, p: float) -> MQ:
        return MQ(market_id="t", question="q", category="tournament_winner",
                  outcome=outcome, probability=p, is_complement=False)

    quotes = [
        _tw("Spain",        0.20),
        _tw("France",       0.18),
        _tw("Brazil",       0.22),
        # Placeholder slots — not yet qualified teams
        _tw("Team AM",      0.50),
        _tw("Team AI",      0.50),
        _tw("Team AG",      0.50),
        _tw("Any Other Team", 0.05),
    ]
    result = strength_from_futures(quotes, devig="proportional")

    # Placeholder / catch-all must be absent
    for bad in ("teamam", "teamai", "teamag", "anyotherteam", "anyother"):
        assert bad not in result, f"non-team key leaked: {bad!r}"

    # Real teams present
    for team in ("spain", "france", "brazil"):
        assert team in result, f"real team missing: {team!r}"

    # Sum over 3 real teams = 1
    total = sum(v["strength_win"] for v in result.values() if "strength_win" in v)
    assert abs(total - 1.0) < 1e-6, f"strength_win sum={total}"

    # Brazil is rank 1
    assert result["brazil"]["strength_rank"] == 1

    # In a 3-team pool Brazil (0.22/0.60) ~ 0.367
    assert result["brazil"]["strength_win"] > 0.30


def test_mixed_pollution_full_scenario():
    """Simulates the exact pollution seen on live data: a pool of real teams
    mixed with continent outcomes, placeholder slots, and any-other-team —
    all tagged tournament_winner.  Asserts all non-team entries are excluded,
    devig sums to 1 over real teams only, and favourite magnitude is sane.
    """
    from ball_quant.models import MarketQuote as MQ

    def _yes(o: str, p: float) -> MQ:
        return MQ(market_id="wc", question="q", category="tournament_winner",
                  outcome=o, probability=p, is_complement=False)
    def _no(o: str, p: float) -> MQ:
        return MQ(market_id="wc", question="q", category="tournament_winner",
                  outcome=f"not:{o}", probability=p, is_complement=True)

    real_teams = [
        ("Spain",      0.1625),
        ("France",     0.1605),
        ("Brazil",     0.1505),
        ("England",    0.0965),
        ("Argentina",  0.0905),
        ("Germany",    0.0705),
        ("Portugal",   0.0605),
        ("Netherlands",0.0505),
    ]
    non_teams = [
        ("Team AM",               0.50),
        ("Team AI",               0.50),
        ("Any Other Team",        0.05),
        ("Europe (UEFA)",         0.72),
        ("North America (CONCACAF)", 0.04),
        ("Africa (CAF)",          0.03),
        ("another continent",     0.01),
    ]

    quotes = []
    for name, p in real_teams:
        quotes.append(_yes(name, p))
        quotes.append(_no(name, 1 - p))
    for name, p in non_teams:
        quotes.append(_yes(name, p))
        quotes.append(_no(name, 1 - p))

    result = strength_from_futures(quotes, devig="proportional")

    # No non-team entries
    bad_slugs = {"teamam", "teamai", "anyotherteam", "europeuefa",
                 "northamericaconcacaf", "africacaf", "anothercontinent"}
    for bad in bad_slugs:
        assert bad not in result, f"non-team key leaked: {bad!r}"

    # All 8 real teams present
    for name, _ in real_teams:
        key = name.lower().replace(" ", "")
        assert key in result, f"real team missing: {key!r}"

    # Sum over 8 real teams = 1
    total = sum(v["strength_win"] for v in result.values() if "strength_win" in v)
    assert abs(total - 1.0) < 1e-6, f"strength_win sum={total}"

    # Spain is rank 1 (highest raw prob among real teams)
    assert result["spain"]["strength_rank"] == 1

    # Spain strength_win in an 8-team pool: 0.1625 / sum(real_probs) ~ 0.19
    spain_win = result["spain"]["strength_win"]
    assert 0.10 < spain_win < 0.35, (
        f"Spain strength_win={spain_win:.4f} out of sane range [0.10, 0.35]; "
        "non-team pollution likely"
    )


# ---------------------------------------------------------------------------
# stdlib-only sanity
# ---------------------------------------------------------------------------

def test_no_third_party_imports():
    """team_strength and knowledge_graph must not import 3rd-party packages."""
    import importlib, sys
    for mod_name in ("ball_quant.core.team_strength", "ball_quant.core.knowledge_graph"):
        mod = importlib.import_module(mod_name)
        src_file = mod.__file__
        with open(src_file, encoding="utf-8") as f:
            source = f.read()
        # Sanity: no numpy, pandas, scipy, requests, etc.
        forbidden = ["import numpy", "import pandas", "import scipy", "import requests",
                     "from numpy", "from pandas", "from scipy"]
        for lib in forbidden:
            assert lib not in source, f"{lib!r} found in {src_file}"
