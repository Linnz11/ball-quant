"""
Snapshot replay layer — Phase 2 backtest.

Reconstructs market state from a persisted bq.snapshot.v1 record and runs the
full analyze->combo->stake pipeline under given StrategyParams without any live
API calls.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from ball_quant.core.analysis import analyze_match
from ball_quant.core.combo import generate_combos
from ball_quant.core.params import DEFAULT_PARAMS, StrategyParams
from ball_quant.core.staking import allocate_stakes
from ball_quant.data.store import read_snapshot, reconstruct_match_sp, reconstruct_matrix
from ball_quant.models import TeamFacts


def neutral_facts(match) -> TeamFacts:
    """Minimal neutral TeamFacts so analyze_match runs without API-Football.

    Mirrors adapters/api_football.py:133 unavailable_facts but uses a neutral
    confidence_adjustment (0.0) instead of -0.15, because we are deliberately
    running without facts rather than experiencing a network failure.  Callers
    that do have real facts should pass them explicitly via replay_snapshot.
    """
    return TeamFacts(
        match_id=match.match_id,
        source="neutral",
        home_summary=f"{match.home}: no team facts (backtest neutral)",
        away_summary=f"{match.away}: no team facts (backtest neutral)",
        warnings=["running in backtest-neutral mode — no injury/lineup data"],
        confidence_adjustment=0.0,
    )


def replay_snapshot(
    record: dict,
    params: StrategyParams = DEFAULT_PARAMS,
    facts: Optional[TeamFacts] = None,
    budget: float = 100.0,
) -> dict:
    """Replay a bq.snapshot.v1 record through the full pipeline.

    Raises ValueError if the snapshot has no SP block (sp is None) because
    edge/Kelly computation is undefined without bookmaker prices.  Callers
    that want to skip SP-less snapshots should catch ValueError and count it.

    Returns a dict with keys:
        match_id, match, matrix, analysis, selections, combo_groups, allocated
    """
    match_sp = reconstruct_match_sp(record)
    if match_sp is None:
        raise ValueError(
            f"Snapshot {record.get('match_id')!r} has no SP block — "
            "cannot compute edge or Kelly without bookmaker prices"
        )

    matrix = reconstruct_matrix(record)

    resolved_facts = facts if facts is not None else neutral_facts(match_sp)

    analysis = analyze_match(match_sp, matrix, resolved_facts, params=params)
    combo_groups = generate_combos(analysis.selections, params=params)
    allocated = allocate_stakes(combo_groups, budget, params=params)

    return {
        "match_id": record["match_id"],
        "match": match_sp,
        "matrix": matrix,
        "analysis": analysis,
        "selections": analysis.selections,
        "combo_groups": combo_groups,
        "allocated": allocated,
    }


def replay_path(
    path,
    params: StrategyParams = DEFAULT_PARAMS,
    facts: Optional[TeamFacts] = None,
    budget: float = 100.0,
) -> dict:
    """Thin wrapper: load snapshot from *path* then replay it."""
    record = read_snapshot(Path(path))
    return replay_snapshot(record, params=params, facts=facts, budget=budget)
