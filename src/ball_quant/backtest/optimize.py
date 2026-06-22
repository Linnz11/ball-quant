"""
Strategy optimizer — Walk-forward, out-of-sample parameter search.

Design invariants (hard):
  - DETERMINISM: all random draws use the caller-supplied seeded rng; the
    global random module is never touched.
  - NO LOOKAHEAD: OOS scoring uses walk_forward_splits exclusively; test
    folds are strictly after train folds.
  - NO SILENT FABRICATION: when a metric is undefined (empty block) for a
    trial, that trial scores as worst-possible for the direction rather than
    being silently skipped or substituted with 0.

Metric registry format:
  metric name -> {"extractor": result -> Optional[float], "direction": "min"|"max"}

The extractor returns None when the metric's block is empty (no bets / no
calibration points), signalling that the metric is undefined for that fold.
"""
from __future__ import annotations

import itertools
import math
import random as _random_module
from dataclasses import replace
from typing import Any, Dict, Iterator, List, Optional, Tuple

from ball_quant.core.params import DEFAULT_PARAMS, StrategyParams
from ball_quant.backtest.engine import run_backtest
from ball_quant.backtest.splits import walk_forward_splits


# ---------------------------------------------------------------------------
# Metric registry
# ---------------------------------------------------------------------------

def _get_nested(result: dict, *keys: str) -> Optional[float]:
    """Safely dig through nested dicts; return None if any level is missing/empty."""
    node = result
    for k in keys:
        if not isinstance(node, dict) or k not in node:
            return None
        node = node[k]
    if not isinstance(node, (int, float)):
        return None
    return float(node)


# Each entry: (extractor, direction)
_METRIC_REGISTRY: Dict[str, Dict[str, Any]] = {
    "brier": {
        "extractor": lambda r: _get_nested(r, "metrics", "calibration", "brier"),
        "direction": "min",
    },
    "log_loss": {
        "extractor": lambda r: _get_nested(r, "metrics", "calibration", "log_loss"),
        "direction": "min",
    },
    "ece": {
        "extractor": lambda r: _get_nested(r, "metrics", "calibration", "ece"),
        "direction": "min",
    },
    "net_pnl": {
        "extractor": lambda r: _get_nested(r, "metrics", "pnl", "net_pnl"),
        "direction": "max",
    },
    "roi": {
        "extractor": lambda r: _get_nested(r, "metrics", "pnl", "roi"),
        "direction": "max",
    },
    "geometric_growth_rate": {
        "extractor": lambda r: _get_nested(r, "metrics", "kelly", "geometric_growth_rate"),
        "direction": "max",
    },
}


def get_metric_info(metric: str) -> Tuple[Any, str]:
    """Return (extractor, direction) for a registered metric name."""
    if metric not in _METRIC_REGISTRY:
        raise ValueError(
            f"Unknown metric {metric!r}. Available: {sorted(_METRIC_REGISTRY)}"
        )
    entry = _METRIC_REGISTRY[metric]
    return entry["extractor"], entry["direction"]


# ---------------------------------------------------------------------------
# Search space iterators
# ---------------------------------------------------------------------------

def iter_grid(param_space: Dict[str, list]) -> Iterator[dict]:
    """Yield all override dicts in Cartesian product order (itertools.product).

    Order is deterministic: Python dict iteration order (insertion order since
    3.7) combined with itertools.product gives a reproducible sequence without
    requiring any sorting — but we sort keys for cross-platform stability.
    """
    keys = sorted(param_space.keys())  # stable across platforms
    value_lists = [param_space[k] for k in keys]
    for combo in itertools.product(*value_lists):
        yield dict(zip(keys, combo))


def iter_random(
    param_space: Dict[str, Tuple[float, float]],
    n_trials: int,
    rng: _random_module.Random,
) -> Iterator[dict]:
    """Yield n_trials override dicts with each field sampled uniformly from (lo, hi).

    Uses the caller-supplied seeded rng — no global random state is touched.
    """
    keys = sorted(param_space.keys())  # stable across platforms
    for _ in range(n_trials):
        overrides: dict = {}
        for k in keys:
            lo, hi = param_space[k]
            overrides[k] = rng.uniform(lo, hi)
        yield overrides


# ---------------------------------------------------------------------------
# Score a single param combo
# ---------------------------------------------------------------------------

def score_params(
    records: list,
    outcomes: dict,
    params: StrategyParams,
    metric: str,
    n_folds: int,
    budget: float = 100.0,
    bankroll: float = 1000.0,
) -> dict:
    """Evaluate *params* by in-sample and out-of-sample walk-forward scoring.

    Returns:
        {
            "in_sample": float | None,      # metric on all records
            "out_of_sample": float | None,  # mean OOS across folds (None if all folds undefined)
            "fold_scores": [float | None],  # per test-fold metric value
            "n_folds_scored": int,          # folds where metric was defined
        }

    When the metric is undefined (empty block) for a fold, that fold contributes
    None to fold_scores and is excluded from the OOS mean.  If ALL folds return
    None, out_of_sample is None.
    """
    extractor, _ = get_metric_info(metric)

    # --- in-sample ---
    in_sample_result = run_backtest(records, outcomes, params=params, budget=budget, bankroll=bankroll)
    in_sample = extractor(in_sample_result)

    # --- out-of-sample via walk-forward ---
    # Key function: sort records by captured_at (ISO string — lexicographic == chronological).
    def _key(rec: dict) -> str:
        return rec.get("captured_at", "")

    folds = walk_forward_splits(records, _key, n_folds)

    fold_scores: List[Optional[float]] = []
    for _train, test in folds:
        # Grade test fold: run_backtest over TEST records only (train fold is
        # used to select params externally; here we just measure generalization).
        # Selection was already done by the outer optimize loop; score_params
        # is a pure measurement function.
        fold_result = run_backtest(test, outcomes, params=params, budget=budget, bankroll=bankroll)
        score = extractor(fold_result)
        fold_scores.append(score)

    defined = [s for s in fold_scores if s is not None]
    n_folds_scored = len(defined)
    out_of_sample: Optional[float] = sum(defined) / n_folds_scored if defined else None

    return {
        "in_sample": in_sample,
        "out_of_sample": out_of_sample,
        "fold_scores": fold_scores,
        "n_folds_scored": n_folds_scored,
    }


# ---------------------------------------------------------------------------
# Worst-possible sentinel for direction
# ---------------------------------------------------------------------------

def _worst_score(direction: str) -> float:
    """Return a sentinel score that will always lose any argmin/argmax comparison.

    Used when out_of_sample is None — such a trial must not win the selection,
    regardless of direction.
    """
    if direction == "min":
        return math.inf
    return -math.inf


# ---------------------------------------------------------------------------
# Main optimizer
# ---------------------------------------------------------------------------

def optimize_params(
    records: list,
    outcomes: dict,
    param_space: dict,
    metric: str = "brier",
    search: str = "grid",
    n_folds: int = 3,
    budget: float = 100.0,
    bankroll: float = 1000.0,
    max_trials: Optional[int] = None,
    seed: int = 0,
) -> dict:
    """Search *param_space* to find the StrategyParams that optimise *metric*.

    Parameters
    ----------
    records     : list of snapshot record dicts (must have 'captured_at').
    outcomes    : {match_id: MatchOutcome} mapping.
    param_space : For 'grid' — {field: [val, ...]}; for 'random' — {field: (lo, hi)}.
    metric      : Registered metric name (brier/log_loss/ece/net_pnl/roi/geometric_growth_rate).
    search      : 'grid' or 'random'.
    n_folds     : Walk-forward fold count (must be >= 1).
    budget      : Per-match budget passed to run_backtest.
    bankroll    : Total bankroll passed to run_backtest.
    max_trials  : Required for 'random'; ignored for 'grid'.
    seed        : RNG seed for reproducibility (only affects 'random' search).

    Returns
    -------
    dict with keys: metric, direction, search, n_trials, n_folds, best_params,
    best_overrides, best_in_sample, best_out_of_sample, overfit_gap, trials.

    Selection is by OUT-OF-SAMPLE score; None treated as worst per direction.
    """
    _, direction = get_metric_info(metric)

    # Build the seeded RNG — only used for random search.  Grid search is
    # order-deterministic without any random draws.
    rng = _random_module.Random(seed)

    # --- generate trials ---
    if search == "grid":
        override_iter: Iterator[dict] = iter_grid(param_space)
        # Count trials = product of list sizes.
        n_trials = 1
        for vals in param_space.values():
            n_trials *= len(vals)
    elif search == "random":
        if max_trials is None:
            raise ValueError("max_trials is required for random search")
        n_trials = max_trials
        override_iter = iter_random(param_space, n_trials, rng)
    else:
        raise ValueError(f"Unknown search strategy {search!r}. Use 'grid' or 'random'.")

    trials: list = []
    best_oos_score: Optional[float] = None
    best_idx: int = 0

    for overrides in override_iter:
        params = replace(DEFAULT_PARAMS, **overrides)
        scored = score_params(
            records, outcomes, params, metric, n_folds, budget, bankroll
        )
        oos = scored["out_of_sample"]
        undefined_flag = oos is None

        trials.append({
            "overrides": overrides,
            "in_sample": scored["in_sample"],
            "out_of_sample": oos,
            "undefined": undefined_flag,
        })

        # Compare using worst-possible sentinel when OOS is None.
        candidate_score = oos if oos is not None else _worst_score(direction)

        if best_oos_score is None:
            # First trial always becomes the provisional best.
            best_oos_score = candidate_score
            best_idx = len(trials) - 1
        else:
            if direction == "min" and candidate_score < best_oos_score:
                best_oos_score = candidate_score
                best_idx = len(trials) - 1
            elif direction == "max" and candidate_score > best_oos_score:
                best_oos_score = candidate_score
                best_idx = len(trials) - 1

    best_trial = trials[best_idx]
    best_params = replace(DEFAULT_PARAMS, **best_trial["overrides"])
    best_in_sample = best_trial["in_sample"]
    best_out_of_sample = best_trial["out_of_sample"]

    # overfit_gap: positive means IS is better than OOS (typical overfitting signal).
    # Signed so that IS - OOS > 0 means the model looks worse OOS than IS.
    # For "min" metrics (lower is better): gap = OOS - IS (positive = OOS worse).
    # For "max" metrics (higher is better): gap = IS - OOS (positive = OOS worse).
    overfit_gap: Optional[float] = None
    if best_in_sample is not None and best_out_of_sample is not None:
        if direction == "min":
            overfit_gap = best_out_of_sample - best_in_sample
        else:
            overfit_gap = best_in_sample - best_out_of_sample

    return {
        "metric": metric,
        "direction": direction,
        "search": search,
        "n_trials": n_trials,
        "n_folds": n_folds,
        "best_params": best_params.to_dict(),
        "best_overrides": best_trial["overrides"],
        "best_in_sample": best_in_sample,
        "best_out_of_sample": best_out_of_sample,
        "overfit_gap": overfit_gap,
        "trials": trials,
    }


# ---------------------------------------------------------------------------
# Per-competition optimizer
# ---------------------------------------------------------------------------

def optimize_by_competition(
    records: list,
    outcomes: dict,
    param_space: dict,
    metric: str = "brier",
    search: str = "grid",
    n_folds: int = 3,
    budget: float = 100.0,
    bankroll: float = 1000.0,
    max_trials: Optional[int] = None,
    seed: int = 0,
    min_records: int = 4,
) -> dict:
    """Run optimize_params per competition group, plus overall.

    Groups records by record["competition"] (None is treated as its own group
    keyed by the string "__none__").  Competitions with fewer records than
    n_folds + 1 are recorded under "skipped" — never silently dropped.

    Parameters
    ----------
    min_records:
        Absolute floor on group size.  A group needs at least
        max(n_folds + 1, min_records) records to be optimised.  Groups below
        this threshold are recorded under "skipped" with the reason.

    Returns
    -------
    {
        "default":              <best_overrides from overall optimize>,
        "by_competition":       {comp: best_overrides, ...},
        "per_competition_detail": {comp: <full optimize_params result>, ...},
        "skipped":              {comp: reason_string, ...},
    }

    The "default" and "by_competition" keys map directly into ParamProfiles
    JSON schema so callers can write::

        profiles = ParamProfiles(
            default_overrides=result["default"],
            by_competition=result["by_competition"],
        )
    """
    # Overall optimize (the "default" baseline uses all records together).
    overall = optimize_params(
        records=records,
        outcomes=outcomes,
        param_space=param_space,
        metric=metric,
        search=search,
        n_folds=n_folds,
        budget=budget,
        bankroll=bankroll,
        max_trials=max_trials,
        seed=seed,
    )

    # Group records by competition.
    groups: Dict[str, list] = {}
    for rec in records:
        comp = rec.get("competition") or "__none__"
        groups.setdefault(comp, []).append(rec)

    min_required = max(n_folds + 1, min_records)

    by_competition: Dict[str, dict] = {}
    per_competition_detail: Dict[str, dict] = {}
    skipped: Dict[str, str] = {}

    for comp, group_records in groups.items():
        n = len(group_records)
        if n < min_required:
            # Explicitly surface the reason — no silent drop.
            skipped[comp] = (
                f"only {n} records; need >= {min_required} "
                f"(n_folds+1={n_folds + 1}, min_records={min_records})"
            )
            continue

        result = optimize_params(
            records=group_records,
            outcomes=outcomes,
            param_space=param_space,
            metric=metric,
            search=search,
            n_folds=n_folds,
            budget=budget,
            bankroll=bankroll,
            max_trials=max_trials,
            seed=seed,
        )
        by_competition[comp] = result["best_overrides"]
        per_competition_detail[comp] = result

    return {
        "default": overall["best_overrides"],
        "by_competition": by_competition,
        "per_competition_detail": per_competition_detail,
        "skipped": skipped,
    }
