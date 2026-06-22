"""strength_prior.py — fundamental strength prior from World Football Elo.

WHY this exists: at WC Matchday 1 there is zero in-tournament form data and
Polymarket WC-futures odds have been used as the KG strength_win axis (market
axis).  This module provides an orthogonal NON-market fundamental axis derived
from Elo ratings, so the system has a credible Poisson λ pair before any live
market signal is available.

Math:
  s_i = z-scored Elo across the field
  sup  = c * (s_home - s_away)      # signed log-rate supremacy
  λ_h  = baseline * exp( sup / 2)
  λ_a  = baseline * exp(-sup / 2)
  => λ_h * λ_a = baseline^2 exactly (product preserved regardless of sup)

  Blend weight at N in-tournament games:
    α(N) = κ / (κ + N)              # pure prior at N=0, decays toward market

All functions are pure (no I/O, no random state) and accept an injected ratings
dict so they are testable without a network call.

NOTE: The supremacy coefficient c=0.40 and baseline=1.25 are DEFAULT starting
values, NOT empirically calibrated to WC2026 data.  They should be tuned via
backtest (#22) once in-tournament results accumulate.  Do not treat the defaults
as validated parameters.
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

from ball_quant.core.match_join import normalize_team
from ball_quant.core.params import StrategyParams


# ---------------------------------------------------------------------------
# Z-score helpers
# ---------------------------------------------------------------------------

def _mean(values: list) -> float:
    return sum(values) / len(values)


def _std(values: list, mu: float) -> float:
    variance = sum((v - mu) ** 2 for v in values) / len(values)
    return math.sqrt(variance)


def elo_z_scores(ratings: Dict[str, float]) -> Dict[str, float]:
    """Return z-scored Elo across the provided field.

    ratings: {canonical_team_name: elo_float}

    Returns {canonical_team_name: z_score}.

    Degenerate case (all equal or single team): returns zeros rather than NaN /
    division-by-zero.  This is correct: equal strength → equal λ → baseline for
    both, which is exactly what strength_to_lambda returns when sup=0.
    """
    if not ratings:
        return {}

    values = list(ratings.values())
    mu = _mean(values)
    sigma = _std(values, mu)

    if sigma == 0.0:
        # All teams have the same Elo — z-scores are all 0 (no information)
        return {team: 0.0 for team in ratings}

    return {team: (elo - mu) / sigma for team, elo in ratings.items()}


# ---------------------------------------------------------------------------
# λ pair from supremacy
# ---------------------------------------------------------------------------

def strength_to_lambda(
    s_home: float,
    s_away: float,
    params: StrategyParams,
) -> Tuple[float, float]:
    """Map z-score pair → (λ_home, λ_away).

    sup = c * (s_home - s_away)
    λ_h = baseline * exp( sup/2)
    λ_a = baseline * exp(-sup/2)

    Invariant: λ_h * λ_a = baseline^2 for any sup (product preserved — total
    expected goals stays at 2 * baseline when teams are equal, and shifts only
    slightly for mismatches because exp(x)*exp(-x)=1).

    params fields used:
      elo_baseline_goals   — expected goals per team for equal-strength match
      elo_supremacy_coeff  — c: sensitivity of log-rate to z-score gap
    """
    baseline = params.elo_baseline_goals
    c = params.elo_supremacy_coeff

    sup = c * (s_home - s_away)
    lam_home = baseline * math.exp(sup / 2.0)
    lam_away = baseline * math.exp(-sup / 2.0)
    return lam_home, lam_away


# ---------------------------------------------------------------------------
# Blend weight
# ---------------------------------------------------------------------------

def prior_blend_alpha(n_games: float, kappa: float) -> float:
    """Weight to place on the Elo prior vs accumulated in-tournament data.

    α(N) = κ / (κ + N)

    N=0  → α=1.0 (pure prior — MD1 cold start)
    N→∞  → α→0.0 (market/form data takes over)

    kappa controls how quickly in-tournament evidence displaces the prior.
    Default κ≈3.5 means 3-4 in-tournament games halve the prior weight.

    WHY Bayesian pseudo-count framing: each in-WC game provides one new sample;
    kappa is the equivalent number of pre-tournament samples the prior is worth.
    The exact kappa should be tuned to historical WC data (#22).
    """
    if kappa <= 0:
        raise ValueError(f"kappa must be positive, got {kappa!r}")
    if n_games < 0:
        raise ValueError(f"n_games must be non-negative, got {n_games!r}")
    return kappa / (kappa + n_games)


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

def elo_lambda_prior(
    home: str,
    away: str,
    ratings: Dict[str, float],
    params: StrategyParams,
) -> Tuple[float, float]:
    """Return (λ_home, λ_away) from Elo fundamental prior.

    home, away: canonical team names (as returned by normalize_team).
    ratings: {canonical_team_name: elo_float} — the full field's ratings.
             Should cover at minimum the teams in the current tournament.
    params: StrategyParams — uses elo_baseline_goals, elo_supremacy_coeff.

    If either team is missing from ratings, uses z-score=0 (average team) for
    that side and logs a warning.  This degrades gracefully rather than crashing
    — the caller still gets a usable baseline λ pair.

    Raises ValueError if ratings is empty (no basis for z-scores at all).
    """
    if not ratings:
        raise ValueError("ratings dict is empty — cannot compute Elo prior")

    import logging
    logger = logging.getLogger(__name__)

    z_scores = elo_z_scores(ratings)

    # Normalize display names so they match adapter-stored normalized keys.
    home_key = normalize_team(home)
    away_key = normalize_team(away)

    if home_key not in z_scores:
        logger.warning(
            "Team %r (normalized: %r) not found in Elo ratings — using z=0 (average) for home side",
            home,
            home_key,
        )
    if away_key not in z_scores:
        logger.warning(
            "Team %r (normalized: %r) not found in Elo ratings — using z=0 (average) for away side",
            away,
            away_key,
        )

    s_home = z_scores.get(home_key, 0.0)
    s_away = z_scores.get(away_key, 0.0)

    return strength_to_lambda(s_home, s_away, params)
