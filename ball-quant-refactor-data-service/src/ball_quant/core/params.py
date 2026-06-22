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

    # -------------------------------------------------------------------------
    # Fundamental strength prior (strength_prior.py).
    # Used at WC Matchday 1 cold start when no in-tournament form is available.
    # The prior provides a Poisson λ pair from World Football Elo ratings —
    # an orthogonal NON-market axis independent of Polymarket futures odds.
    #
    # elo_baseline_goals: expected goals per team in an equal-strength match.
    #   WHY 1.25: matches the existing prior_lambdas convention (~2.45 total /
    #   ~1.225 per side) and is consistent with historical WC goal averages.
    # elo_supremacy_coeff (c): sensitivity of log-rate to z-score gap.
    #   WHY 0.40: a 1-sigma Elo difference yields ~exp(0.20)≈1.22x advantage,
    #   roughly consistent with top-vs-mid Elo gaps driving ~20-25% win-rate
    #   swings.  NOT empirically calibrated — tune via backtest (#22).
    # elo_prior_kappa: pseudo-count for blend weight α=κ/(κ+N).
    #   WHY 3.5: prior is worth ~3-4 in-tournament games before market/form
    #   takes over.  Rough estimate; calibrate once WC group-stage data exists.
    # strength_w_elo/mv/qual: blend weights for multi-axis strength (future).
    #   Only elo wired now; mv (market value) and qual (qualification path)
    #   placeholders for later integration.  Weights sum to 1.0 by convention.
    # -------------------------------------------------------------------------
    elo_baseline_goals: float = 1.25
    elo_supremacy_coeff: float = 0.40
    elo_prior_kappa: float = 3.5
    strength_w_elo: float = 0.50    # future: fraction of composite score from Elo
    strength_w_mv: float = 0.35     # future: fraction from market value (transfermarkt)
    strength_w_qual: float = 0.15   # future: fraction from qualification path difficulty

    # ---------------------------------------------------------------------
    # Soft-calibration path (REFACTOR_PLAN §1b).  OFF BY DEFAULT: when
    # use_softcal=False the engine takes the legacy IPF route and output is
    # byte-identical.  Flip use_softcal=True (or call
    # fit_score_distribution_soft / calibrate_distribution_soft directly) to
    # replace multiplicative IPF with KL-regularised exponentiated-gradient
    # mirror descent under reliability- and dependency-weighted soft
    # constraints.  WHY a separate path rather than a parameterised IPF: the
    # objective is fundamentally different (a single regularised loss vs. a
    # sequence of exact marginal projections), so the two cannot share a loop
    # without one masking the other's failure modes.
    use_softcal: bool = False

    # Reliability variance budget: sigma^2_g = c_spread*spread^2
    #   + c_depth/log(1+depth) + c_vol/log(1+vol) + c_age*age
    #   + c_devig*devig_var + sigma_floor^2.  These coefficients set the
    # relative trust between spread, market depth, traded volume, staleness and
    # cross-devig-map disagreement.  c_* are in (prob-units)^2 so the terms are
    # directly comparable; sigma_floor caps how confident any single book can be.
    softcal_c_spread: float = 0.50
    softcal_c_depth: float = 0.020
    softcal_c_vol: float = 0.020
    softcal_c_age: float = 0.0010
    softcal_c_devig: float = 1.0
    softcal_sigma_floor: float = 0.02

    # Staleness decay in alpha_g = (1/sigma^2_g)*exp(-age/half_life).  half_life
    # in the SAME time units as the quote age (seconds if fetched_at is epoch).
    # WHY default very large: absent a per-quote timestamp source the age term
    # is 0 and this decay is identity — see fetched_at handling in
    # reliability_weight().  Set to a real horizon once quotes carry fetch time.
    softcal_half_life: float = 1.0e9

    # Correlation caps on the total reliability mass.  per_market caps any
    # single book's alpha; family caps the SUM of alpha within one ladder
    # (总进球 / 让球) because adjacent lines are near-collinear, NOT independent
    # observations — without this an N-line ladder would get N* the weight of a
    # single equally-reliable market and dominate the fit.
    #
    # SCALE (why these numbers): alpha = 1/sigma^2.  With sigma_floor=0.02 the
    # variance floor is 4e-4, so the theoretical MAX alpha (a perfect market) is
    # ~2500; a typical liquid book lands sigma~0.07 -> alpha~200, a thin book
    # sigma~0.3 -> alpha~11.  The per-market cap is therefore set as an OUTLIER
    # guard ABOVE the typical liquid alpha (so the reliability gradient
    # thin<liquid survives — flattening every market to the cap would destroy
    # the §1b differentiation, the whole point).  The family cap lets a few
    # liquid rungs count for ~2 independent markets' worth but clamps a long
    # ladder hard.
    softcal_alpha_per_market_cap: float = 1500.0
    softcal_alpha_family_cap: float = 400.0

    # Mirror-descent solver.  eta = the (constant) exponentiated-gradient step;
    # the objective is smooth so a constant step converges and the convergence
    # test (max|delta q| < tol) is honest.  max_iters caps work; tol is the grid
    # movement below which the fit is declared converged.  kl_weight =
    # lambda_reg, the strength of the KL(q||q0) regulariser: small => reliable
    # books (large alpha_g) move q well off the prior; large => q stays near q0.
    # WHY small eta default: with reliability alpha on the order of tens, a
    # larger step overshoots and oscillates (the 1/(bq(1-bq)) gradient factor is
    # steep) — 0.05 converges across the families seen in practice.
    softcal_eta: float = 0.05
    softcal_max_iters: int = 5000
    softcal_tol: float = 1.0e-7
    softcal_kl_weight: float = 0.05

    # Huber-logit transition for BINARY markets (totals / handicap / btts /
    # team_total ladder rungs).  Residuals in logit space below delta are
    # quadratic (KL-like), above delta are linear — so a single badly-priced
    # book bends the fit linearly instead of KL's exponential over-reaction.
    softcal_huber_delta: float = 0.75

    # Thin/stale shrinkage: p_used = beta*p_poly + (1-beta)*B_g q0.  A market is
    # thin when its reliability alpha < softcal_thin_alpha (sparse/illiquid), in
    # which case its target is pulled beta of the way from the noisy book price
    # toward the model prior projection.  beta=1 trusts the book fully; beta<1
    # borrows strength from q0 exactly where the book is least trustworthy.
    softcal_thin_alpha: float = 1.0
    softcal_shrink_beta: float = 0.5

    # Explicit tail bucket: scores with home+away >= softcal_tail_threshold are
    # aggregated into one "8+" cell that is carried through (NOT silently
    # truncated + renormalised) because the score layer is sensitive to tail
    # mass.  The grid itself is still built to max_goals; this controls the
    # reported tail aggregate in the projections.
    softcal_tail_threshold: int = 8

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
            # soft-calibration path (§1b)
            "use_softcal": self.use_softcal,
            "softcal_c_spread": self.softcal_c_spread,
            "softcal_c_depth": self.softcal_c_depth,
            "softcal_c_vol": self.softcal_c_vol,
            "softcal_c_age": self.softcal_c_age,
            "softcal_c_devig": self.softcal_c_devig,
            "softcal_sigma_floor": self.softcal_sigma_floor,
            "softcal_half_life": self.softcal_half_life,
            "softcal_alpha_per_market_cap": self.softcal_alpha_per_market_cap,
            "softcal_alpha_family_cap": self.softcal_alpha_family_cap,
            "softcal_eta": self.softcal_eta,
            "softcal_max_iters": self.softcal_max_iters,
            "softcal_tol": self.softcal_tol,
            "softcal_kl_weight": self.softcal_kl_weight,
            "softcal_huber_delta": self.softcal_huber_delta,
            "softcal_thin_alpha": self.softcal_thin_alpha,
            "softcal_shrink_beta": self.softcal_shrink_beta,
            "softcal_tail_threshold": self.softcal_tail_threshold,
            # fundamental strength prior
            "elo_baseline_goals": self.elo_baseline_goals,
            "elo_supremacy_coeff": self.elo_supremacy_coeff,
            "elo_prior_kappa": self.elo_prior_kappa,
            "strength_w_elo": self.strength_w_elo,
            "strength_w_mv": self.strength_w_mv,
            "strength_w_qual": self.strength_w_qual,
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
