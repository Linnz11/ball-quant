from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple


@dataclass(frozen=True)
class StrategyParams:
    # Score grid ceiling — must match poisson_grid call sites
    max_goals: int = 7

    # prior_lambdas: fallback total and home-share formula
    base_total_fallback: float = 2.45
    home_share_coeff: float = 0.35
    home_share_floor: float = 0.25
    home_share_cap: float = 0.75
    lambda_floor: float = 0.25

    # total_goal_hint nudge weight
    total_hint_nudge: float = 0.7

    # calibrate_distribution iteration counts and multipliers
    calib_primary_iters: int = 90
    calib_shape_iters: int = 25
    calib_final_iters: int = 20
    calib_shape_mult: float = 0.30
    calib_primary_in_shape_mult: float = 0.75

    # correct_score_constraints caps
    cs_mass_cap: float = 0.85
    cs_strength_cap: float = 0.26
    cs_strength_coeff: float = 0.65

    # confidence_score base and polymarket bonus
    conf_base: float = 0.55
    conf_poly_bonus: float = 0.12

    # causal reliability base
    reliability_base: float = 0.72

    # correlation_discount bases
    corr_base: float = 0.96
    corr_lowconf: float = 0.94
    corr_exact: float = 0.92
    corr_floor: float = 0.70

    # type-C bucket thresholds
    typec_prob_lo: float = 0.05
    typec_prob_hi: float = 0.12
    typec_odds_min: float = 5.0

    # staking Kelly fraction and budget splits
    fractional_kelly: float = 0.25
    budget_a: float = 0.60
    budget_b: float = 0.30
    budget_c: float = 0.10

    # per-type stake caps
    cap_a: float = 0.35
    cap_b: float = 0.20
    cap_c: float = 0.075

    # optional per-category multiplier on causal profile model_weight (None = no scaling)
    profile_weight_scale: Optional[Dict[str, float]] = None

    # methodology toggles — defaults preserve current behavior; flip + backtest to evaluate
    dixon_coles_rho: float = 0.0          # 0.0 = independent Poisson (off); <0 lifts low scores/draws
    devig_method: str = "proportional"    # "proportional" (current) | "shin"
    weight_scheme: str = "heuristic"      # "heuristic" (current) | "inverse_variance"

    # Half-time goal share for the HAFU (半全场) model.
    # WHY 0.45: empirically, first halves in top football produce ~45% of match
    # goals; second halves are slightly more open (fatigue, substitutions, game
    # state chasing) → remaining 55%.  This is the prior when no Polymarket
    # first_half_total_goals market is available.  When that market IS present,
    # the calibration routine infers the implied 1H expected total and adjusts
    # the share accordingly (clamped to [0.3, 0.6] to prevent extreme priors
    # from sparse/illiquid markets from distorting the half-split).
    first_half_goal_share: float = 0.45

    # Which market categories to emit as bet candidates.  Tuple (immutable,
    # frozen-dataclass-safe) of canonical category names.  Restricting this to
    # ("spf", "handicap") reproduces the old branch set byte-for-byte.
    # All 6 score-derivable types are enabled by default so the engine ranks
    # totals/btts/team_total/correct_score alongside moneyline/handicap.
    bet_markets: Tuple[str, ...] = (
        "spf",
        "handicap",
        "totals",
        "btts",
        "team_total",
        "correct_score",
    )

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable plain dict (no nested dataclasses)."""
        return {
            "max_goals": self.max_goals,
            "base_total_fallback": self.base_total_fallback,
            "home_share_coeff": self.home_share_coeff,
            "home_share_floor": self.home_share_floor,
            "home_share_cap": self.home_share_cap,
            "lambda_floor": self.lambda_floor,
            "total_hint_nudge": self.total_hint_nudge,
            "calib_primary_iters": self.calib_primary_iters,
            "calib_shape_iters": self.calib_shape_iters,
            "calib_final_iters": self.calib_final_iters,
            "calib_shape_mult": self.calib_shape_mult,
            "calib_primary_in_shape_mult": self.calib_primary_in_shape_mult,
            "cs_mass_cap": self.cs_mass_cap,
            "cs_strength_cap": self.cs_strength_cap,
            "cs_strength_coeff": self.cs_strength_coeff,
            "conf_base": self.conf_base,
            "conf_poly_bonus": self.conf_poly_bonus,
            "reliability_base": self.reliability_base,
            "corr_base": self.corr_base,
            "corr_lowconf": self.corr_lowconf,
            "corr_exact": self.corr_exact,
            "corr_floor": self.corr_floor,
            "typec_prob_lo": self.typec_prob_lo,
            "typec_prob_hi": self.typec_prob_hi,
            "typec_odds_min": self.typec_odds_min,
            "fractional_kelly": self.fractional_kelly,
            "budget_a": self.budget_a,
            "budget_b": self.budget_b,
            "budget_c": self.budget_c,
            "cap_a": self.cap_a,
            "cap_b": self.cap_b,
            "cap_c": self.cap_c,
            "profile_weight_scale": self.profile_weight_scale,
            "dixon_coles_rho": self.dixon_coles_rho,
            "devig_method": self.devig_method,
            "weight_scheme": self.weight_scheme,
            "first_half_goal_share": self.first_half_goal_share,
            "bet_markets": list(self.bet_markets),  # list for JSON round-trip
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "StrategyParams":
        """Construct from a plain dict; raises ValueError on unknown keys."""
        known = cls().to_dict().keys()
        unknown = set(d) - set(known)
        if unknown:
            raise ValueError(f"Unknown StrategyParams keys: {unknown}")
        coerced = dict(d)
        # JSON encodes tuples as lists; coerce back so the frozen dataclass
        # receives the expected immutable type.
        if "bet_markets" in coerced and not isinstance(coerced["bet_markets"], tuple):
            coerced["bet_markets"] = tuple(coerced["bet_markets"])
        return cls(**coerced)


DEFAULT_PARAMS = StrategyParams()
