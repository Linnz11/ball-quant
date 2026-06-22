"""
Evaluation metrics for the backtest harness.

All functions consume plain dicts so the engine can feed them without
importing domain models. stdlib math only — no numpy.

Input record shapes:
  CalibrationPoint: {"prob": float in [0,1], "y": int in {0,1}}
  Bet: {"stake": float>0, "odds": float>1 (decimal),
        "result": "WIN"|"LOSS"|"VOID", "prob": float, "edge": float}
"""

import math
from typing import List


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def brier_score(points: list) -> float:
    """Mean squared error between predicted probability and binary outcome."""
    if not points:
        raise ValueError("brier_score requires at least one point")
    return sum((p["prob"] - p["y"]) ** 2 for p in points) / len(points)


def log_loss(points: list, eps: float = 1e-15) -> float:
    """
    Binary cross-entropy.
    Clip p to [eps, 1-eps] to avoid -inf when the model assigns 0 or 1
    probability to the wrong outcome — a degenerate but valid model state.
    """
    if not points:
        raise ValueError("log_loss requires at least one point")
    total = 0.0
    for p in points:
        prob = max(eps, min(1.0 - eps, p["prob"]))
        y = p["y"]
        total += y * math.log(prob) + (1 - y) * math.log(1.0 - prob)
    return -total / len(points)


def reliability_bins(points: list, n_bins: int = 10) -> List[dict]:
    """
    Equal-width bins over [0, 1].  Only non-empty bins are returned.
    Each bin dict: {"bin_lo", "bin_hi", "mean_pred", "emp_freq", "n"}.
    """
    width = 1.0 / n_bins
    bins: list = [[] for _ in range(n_bins)]
    for p in points:
        # clamp to handle prob == 1.0 exactly
        idx = min(int(p["prob"] / width), n_bins - 1)
        bins[idx].append(p)

    result = []
    for i, bucket in enumerate(bins):
        if not bucket:
            continue
        lo = i * width
        hi = lo + width
        mean_pred = sum(b["prob"] for b in bucket) / len(bucket)
        emp_freq = sum(b["y"] for b in bucket) / len(bucket)
        result.append({
            "bin_lo": lo,
            "bin_hi": hi,
            "mean_pred": mean_pred,
            "emp_freq": emp_freq,
            "n": len(bucket),
        })
    return result


def expected_calibration_error(points: list, n_bins: int = 10) -> float:
    """
    Weighted mean absolute difference between predicted and empirical
    frequency across reliability bins.
    """
    if not points:
        raise ValueError("expected_calibration_error requires at least one point")
    n_total = len(points)
    bins = reliability_bins(points, n_bins=n_bins)
    return sum(
        (b["n"] / n_total) * abs(b["mean_pred"] - b["emp_freq"])
        for b in bins
    )


# ---------------------------------------------------------------------------
# PnL
# ---------------------------------------------------------------------------

def pnl_ledger(bets: list) -> dict:
    """
    Settlement:
      WIN  -> return = stake * odds  (gross), net = stake * (odds - 1)
      LOSS -> return = 0,            net = -stake
      VOID -> return = stake,        net = 0   (excluded from at_risk)
    roi = net_pnl / total_stake_at_risk; raises if at_risk == 0.
    """
    n_win = n_loss = n_void = 0
    total_stake_at_risk = 0.0
    total_return = 0.0
    net_pnl = 0.0

    for b in bets:
        stake = b["stake"]
        odds = b["odds"]
        result = b["result"]
        if result == "WIN":
            n_win += 1
            total_stake_at_risk += stake
            total_return += stake * odds
            net_pnl += stake * (odds - 1)
        elif result == "LOSS":
            n_loss += 1
            total_stake_at_risk += stake
            total_return += 0.0
            net_pnl -= stake
        elif result == "VOID":
            n_void += 1
            total_return += stake  # refund
            # net contribution = 0; not at risk

    if total_stake_at_risk == 0.0:
        raise ValueError("roi undefined: no non-VOID bets")

    return {
        "n_bets": len(bets),
        "n_win": n_win,
        "n_loss": n_loss,
        "n_void": n_void,
        "total_stake_at_risk": total_stake_at_risk,
        "total_return": total_return,
        "net_pnl": net_pnl,
        "roi": net_pnl / total_stake_at_risk,
    }


# ---------------------------------------------------------------------------
# Edge realization
# ---------------------------------------------------------------------------

def _pearson_r(xs: list, ys: list) -> float:
    """
    Pearson correlation.  Returns 0.0 if either series has zero variance
    (degenerate case — correlation undefined, but we must not crash; the
    caller documents this in the return dict note field).
    """
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    var_x = sum((x - mx) ** 2 for x in xs)
    var_y = sum((y - my) ** 2 for y in ys)
    denom = math.sqrt(var_x * var_y)
    if denom == 0.0:
        return 0.0
    return cov / denom


def edge_realization(bets: list) -> dict:
    """
    Compares predicted edge to realized return per unit on non-VOID bets.
    realized_return_per_unit: WIN -> (odds-1), LOSS -> -1.0
    correlation = Pearson r(predicted_edge, realized_return_per_unit).
    Zero variance -> correlation = 0.0 (no crash).
    """
    non_void = [b for b in bets if b["result"] != "VOID"]
    n = len(non_void)
    if n == 0:
        raise ValueError("edge_realization requires at least one non-VOID bet")

    edges = []
    realized = []
    for b in non_void:
        edges.append(b["edge"])
        if b["result"] == "WIN":
            realized.append(b["odds"] - 1.0)
        else:  # LOSS
            realized.append(-1.0)

    return {
        "mean_predicted_edge": sum(edges) / n,
        "mean_realized_return_per_unit": sum(realized) / n,
        "correlation": _pearson_r(edges, realized),
        "n": n,
    }


# ---------------------------------------------------------------------------
# Kelly growth
# ---------------------------------------------------------------------------

def kelly_growth(bets: list, bankroll: float) -> dict:
    """
    Log-growth accounting for each non-VOID bet.
    f = stake / bankroll (fraction of bankroll wagered).
    WIN:  log_return = ln(1 + f*(odds-1))
    LOSS: log_return = ln(1 - f)
          Raises ValueError if f >= 1 (total ruin — log(0) or log of negative
          is mathematically ill-defined and signals a catastrophic sizing error).
    geometric_growth_rate = exp(sum_log_growth / n) - 1.
    """
    if bankroll <= 0:
        raise ValueError("bankroll must be positive")

    non_void = [b for b in bets if b["result"] != "VOID"]
    n = len(non_void)
    if n == 0:
        raise ValueError("kelly_growth requires at least one non-VOID bet")

    sum_log = 0.0
    for b in non_void:
        f = b["stake"] / bankroll
        if b["result"] == "WIN":
            sum_log += math.log(1.0 + f * (b["odds"] - 1.0))
        else:  # LOSS
            if f >= 1.0:
                raise ValueError(
                    f"fraction f={f} >= 1 on a LOSS bet — total ruin, ill-defined"
                )
            sum_log += math.log(1.0 - f)

    return {
        "sum_log_growth": sum_log,
        "geometric_growth_rate": math.exp(sum_log / n) - 1.0,
        "n": n,
    }


# ---------------------------------------------------------------------------
# Aggregate convenience wrapper
# ---------------------------------------------------------------------------

def aggregate(points: list, bets: list, bankroll: float) -> dict:
    """
    Tolerant top-level wrapper.  Empty points -> calibration: {}.
    All-VOID or empty bets -> pnl/edge/kelly: {}.
    Individual sub-block failures do not propagate.
    """
    result: dict = {}

    # --- calibration ---
    if points:
        try:
            result["calibration"] = {
                "brier": brier_score(points),
                "log_loss": log_loss(points),
                "ece": expected_calibration_error(points),
                "reliability": reliability_bins(points),
            }
        except Exception:
            result["calibration"] = {}
    else:
        result["calibration"] = {}

    # --- pnl ---
    if bets:
        try:
            result["pnl"] = pnl_ledger(bets)
        except Exception:
            result["pnl"] = {}
    else:
        result["pnl"] = {}

    # --- edge ---
    if bets:
        try:
            result["edge"] = edge_realization(bets)
        except Exception:
            result["edge"] = {}
    else:
        result["edge"] = {}

    # --- kelly ---
    if bets:
        try:
            result["kelly"] = kelly_growth(bets, bankroll)
        except Exception:
            result["kelly"] = {}
    else:
        result["kelly"] = {}

    return result
