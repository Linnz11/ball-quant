"""
Backtest engine — Phase 2.

Replays a list of snapshot records, grades each bet against supplied outcomes,
and aggregates calibration + PnL metrics.

Grading rules for multi-leg Combos:
  - All legs WIN   -> WIN, effective_odds = product of all leg SPs.
  - Any leg LOSS   -> LOSS immediately (effective_odds = combo.odds, not recomputed).
  - VOID legs are dropped; effective_odds recomputed from survivors.
  - All legs VOID  -> VOID (stake refunded, excluded from PnL).
"""
from __future__ import annotations

from typing import Dict, Optional

from ball_quant.core.metrics import aggregate
from ball_quant.core.params import DEFAULT_PARAMS, StrategyParams
from ball_quant.core.settlement import MatchOutcome, grade
from ball_quant.backtest.replay import replay_snapshot


# ---------------------------------------------------------------------------
# Combo grading
# ---------------------------------------------------------------------------

def grade_combo(combo, outcome: MatchOutcome):
    """Grade a Combo (one or more legs) against a MatchOutcome.

    Returns (grade_str, effective_odds) where:
      grade_str     : "WIN" | "LOSS" | "VOID"
      effective_odds: product of odds of non-VOID legs (or original combo.odds
                      on LOSS — the effective odds on a loss do not matter for
                      PnL, but we carry the original for auditing).
    """
    leg_grades = [(sel, grade(sel, outcome)) for sel in combo.selections]

    # Any LOSS kills the combo immediately.
    if any(g == "LOSS" for _, g in leg_grades):
        return "LOSS", combo.odds

    surviving = [(sel, g) for sel, g in leg_grades if g == "WIN"]
    voided = [g for _, g in leg_grades if g == "VOID"]

    # All legs VOID -> VOID.
    if len(voided) == len(leg_grades):
        return "VOID", combo.odds

    # At least one WIN and possibly some VOID.
    # Effective odds = product of surviving WIN leg SPs.
    # A VOID leg is dropped — it is as if that leg never existed.
    effective_odds = 1.0
    for sel, _ in surviving:
        effective_odds *= sel.sp

    return "WIN", effective_odds


# ---------------------------------------------------------------------------
# run_backtest
# ---------------------------------------------------------------------------

def run_backtest(
    records: list,
    outcomes: Dict[str, MatchOutcome],
    params: StrategyParams = DEFAULT_PARAMS,
    budget: float = 100.0,
    bankroll: float = 1000.0,
    profiles=None,  # Optional[ParamProfiles] — not imported at module level to avoid circular deps
) -> dict:
    """Run the full backtest over *records* against *outcomes*.

    For each record:
      - If match_id not in outcomes  -> increment skipped_no_outcome, skip.
      - If SP block missing          -> increment skipped_no_sp, skip.
      - Otherwise: replay_snapshot, grade selections (calibration points) and
        allocated combos (bets), accumulate into aggregate metrics.

    Returns a summary dict with counts and the metrics block from
    core/metrics.aggregate.

    Skipped records are always reported (never silently dropped).

    profiles (ParamProfiles | None):
        When supplied, each record's effective StrategyParams is resolved via
        profiles.resolve(record.get("competition")).  This allows per-competition
        parameter tuning without any change to the grading or metrics logic.
        When None (the default), the single *params* argument is used for every
        record — behaviour is byte-identical to the pre-profiles implementation.
    """
    n_records = len(records)
    n_graded_matches = 0
    skipped_no_outcome = 0
    skipped_no_sp = 0
    n_void = 0

    calibration_points: list = []
    bets: list = []

    # market_type -> list of calibration points for per-market breakdown
    per_market_type_points: Dict[str, list] = {}

    for record in records:
        match_id = record.get("match_id")
        if match_id not in outcomes:
            skipped_no_outcome += 1
            continue

        outcome = outcomes[match_id]

        # Resolve per-record params: profiles path or single-params path.
        # The single-params path (profiles is None) is the original code path —
        # zero structural change so existing callers see identical behaviour.
        effective_params = (
            profiles.resolve(record.get("competition"))
            if profiles is not None
            else params
        )

        try:
            replayed = replay_snapshot(record, params=effective_params, budget=budget)
        except ValueError:
            # SP block is missing — reported but not a crash.
            skipped_no_sp += 1
            continue

        n_graded_matches += 1
        selections = replayed["selections"]
        allocated = replayed["allocated"]

        # ---- calibration points from raw selections (one per leg) ----
        for sel in selections:
            result = grade(sel, outcome)
            if result == "VOID":
                n_void += 1
                continue
            y = 1 if result == "WIN" else 0
            point = {"prob": sel.probability, "y": y}
            calibration_points.append(point)

            # Per-market-type breakdown
            mt = sel.settlement_key.market_type if sel.settlement_key else "unknown"
            per_market_type_points.setdefault(mt, []).append(point)

        # ---- bets from allocated combos ----
        for combo in allocated:
            if combo.stake <= 0:
                continue
            result, effective_odds = grade_combo(combo, outcome)
            # Edge for the combo: use the combo's expected_return as the
            # predicted edge (consistent with how staking was done).
            bets.append({
                "stake": combo.stake,
                "odds": effective_odds,
                "result": result,
                "prob": combo.probability,
                "edge": combo.expected_return,
            })

    # Aggregate calibration + PnL via the metrics module.
    metrics = aggregate(calibration_points, bets, bankroll)

    # Per-market-type breakdown: brier + n per group (skip empty groups).
    per_market_type: Dict[str, dict] = {}
    for mt, pts in per_market_type_points.items():
        if not pts:
            continue
        from ball_quant.core.metrics import brier_score
        per_market_type[mt] = {
            "brier": brier_score(pts),
            "n": len(pts),
        }

    return {
        "n_records": n_records,
        "n_graded_matches": n_graded_matches,
        "skipped_no_outcome": skipped_no_outcome,
        "skipped_no_sp": skipped_no_sp,
        "n_calibration_points": len(calibration_points),
        "n_void": n_void,
        "n_bets": len(bets),
        "metrics": metrics,
        "per_market_type": per_market_type,
    }
