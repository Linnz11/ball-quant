"""Soft-calibration path tests (REFACTOR_PLAN §1b).

Covers the opt-in mirror-descent calibration engine: it must converge to a
normalised, strictly-positive grid; the per-family reliability cap must
actually bind; the Dixon-Coles low-score correction must move the right cells;
the explicit 8+ tail bucket must be carried (not silently truncated); and the
thin-market shrinkage must pull a thin target toward the prior projection.

The new path is OFF BY DEFAULT — a separate suite-wide invariant (asserted in
test_default_path_unchanged) guarantees use_softcal=False leaves the legacy IPF
output byte-identical, so none of these tests can mask a regression in the
default engine.
"""
from __future__ import annotations

import dataclasses
import math
import unittest

from ball_quant.core.params import DEFAULT_PARAMS, StrategyParams
from ball_quant.core import probability as P
from ball_quant.models import EventMarketMatrix, MarketQuote, MatchSP


# ---------------------------------------------------------------------------
# Fixtures / builders
# ---------------------------------------------------------------------------

def _match() -> MatchSP:
    return MatchSP(
        match_id="m1",
        date="2026-06-17",
        home="Home FC",
        away="Away FC",
        spf_home=2.0,
        spf_draw=3.3,
        spf_away=3.8,
        handicap=None,
        rq_home=None,
        rq_draw=None,
        rq_away=None,
    )


def _quote(category, outcome, prob, question="", liquidity=None, volume=None, spread=None):
    return MarketQuote(
        market_id=f"{category}:{outcome}:{question}",
        question=question,
        category=category,
        outcome=outcome,
        probability=prob,
        liquidity=liquidity,
        volume=volume,
        spread=spread,
        active=True,
        closed=False,
        accepting_orders=True,
    )


def _liquid_matrix() -> EventMarketMatrix:
    """A match with a deep, multi-rung totals ladder + 1X2 + btts."""
    quotes = [
        _quote("moneyline", "home", 0.52, liquidity=5000, volume=20000, spread=0.02),
        _quote("moneyline", "draw", 0.27, liquidity=4000, volume=15000, spread=0.02),
        _quote("moneyline", "away", 0.26, liquidity=4000, volume=15000, spread=0.02),
        _quote("total_goals", "over", 0.62, "Over 1.5", liquidity=6000, volume=30000, spread=0.015),
        _quote("total_goals", "under", 0.40, "Under 1.5", liquidity=6000, volume=30000, spread=0.015),
        _quote("total_goals", "over", 0.48, "Over 2.5", liquidity=8000, volume=40000, spread=0.01),
        _quote("total_goals", "under", 0.54, "Under 2.5", liquidity=8000, volume=40000, spread=0.01),
        _quote("total_goals", "over", 0.30, "Over 3.5", liquidity=3000, volume=12000, spread=0.03),
        _quote("total_goals", "under", 0.72, "Under 3.5", liquidity=3000, volume=12000, spread=0.03),
        _quote("btts", "yes", 0.55, liquidity=4000, volume=10000, spread=0.02),
        _quote("btts", "no", 0.47, liquidity=4000, volume=10000, spread=0.02),
    ]
    return EventMarketMatrix(match_id="m1", home="Home FC", away="Away FC", markets=quotes)


def _softcal_params(**overrides) -> StrategyParams:
    base = dataclasses.replace(DEFAULT_PARAMS, use_softcal=True)
    return dataclasses.replace(base, **overrides) if overrides else base


# ---------------------------------------------------------------------------
# 1. Mirror-descent convergence + normalisation
# ---------------------------------------------------------------------------

class MirrorDescentConvergenceTests(unittest.TestCase):
    def test_grid_is_normalised_and_strictly_positive(self):
        res = P.fit_score_distribution_soft(_match(), _liquid_matrix(), params=_softcal_params())
        grid = res.distribution.probs
        self.assertEqual(len(grid), (DEFAULT_PARAMS.max_goals + 1) ** 2)
        self.assertAlmostEqual(sum(grid.values()), 1.0, places=12)
        # Exponentiated-gradient keeps every cell strictly positive (never 0).
        self.assertTrue(all(v > 0.0 for v in grid.values()))

    def test_solver_converges_flag_trips(self):
        res = P.fit_score_distribution_soft(_match(), _liquid_matrix(), params=_softcal_params())
        self.assertTrue(res.converged, "solver should reach the convergence tolerance")
        self.assertLess(res.iterations, DEFAULT_PARAMS.softcal_max_iters)

    def test_fixed_point_is_eta_independent(self):
        # The objective is convex with a unique optimum; different (stable) step
        # sizes must reach the SAME B_g projections.  If they don't, the iterate
        # is oscillating (coherent noise) rather than converging.
        mx = _liquid_matrix()
        lh, la = P.prior_lambdas(mx, params=_softcal_params())
        q0 = P.dixon_coles_prior(lh, la, _softcal_params())
        markets = P.apply_family_caps(P.build_soft_markets(_match(), mx, _softcal_params()), _softcal_params())

        def projections(eta):
            q, _, conv = P.calibrate_distribution_soft(q0, markets, _softcal_params(softcal_eta=eta))
            self.assertTrue(conv)
            return [P._project(q, m.predicate) for m in markets]

        # Sweep within the stable step band (eta below the constant-step
        # stability ceiling; larger eta oscillates and is correctly NOT used as
        # a default).  Every stable eta must reach the same optimum.
        ref = projections(0.05)
        for eta in (0.02, 0.08, 0.10):
            other = projections(eta)
            drift = max(abs(a - b) for a, b in zip(ref, other))
            self.assertLess(drift, 1e-4, f"fixed point drifted {drift} at eta={eta}")

    def test_calibration_pulls_projections_toward_books(self):
        # After calibration the model projections should sit close to the devig'd
        # book targets (the solver actually fits the data, not just the prior).
        res = P.fit_score_distribution_soft(_match(), _liquid_matrix(), params=_softcal_params())
        residuals = [abs(pr.model_prob - pr.book_prob) for pr in res.projections]
        self.assertLess(max(residuals), 0.05, "no market should be wildly off its book")
        self.assertLess(sum(residuals) / len(residuals), 0.02)

    def test_trust_region_clip_does_not_bind_at_optimum(self):
        # At the converged optimum the per-cell log step must be well below the
        # trust radius — if the clip were binding, convergence would be fake.
        mx = _liquid_matrix()
        params = _softcal_params()
        lh, la = P.prior_lambdas(mx, params=params)
        q0 = P.dixon_coles_prior(lh, la, params)
        markets = P.apply_family_caps(P.build_soft_markets(_match(), mx, params), params)
        q, _, conv = P.calibrate_distribution_soft(q0, markets, params)
        self.assertTrue(conv)
        # One more manual EG step from the converged q: its max log-move must be
        # tiny (far under the clip), proving the rail is inactive at the optimum.
        log_q0 = {s: math.log(p) for s, p in q0.items()}
        active = [m for m in markets if m.alpha > 0.0]
        mean_alpha = sum(m.alpha for m in active) / len(active)
        grad = {s: 0.0 for s in q}
        for m in active:
            a = m.alpha / mean_alpha
            bq = P._project(q, m.predicate)
            if m.is_binary:
                g = a * P._huber_logit_grad(bq, m.target, params.softcal_huber_delta)
            else:
                g = a * P._kl_binary_grad(bq, m.target)
            for s in grad:
                if m.predicate(s[0], s[1]):
                    grad[s] += g
        max_step = max(
            abs(-params.softcal_eta * (params.softcal_kl_weight * (math.log(q[s]) - log_q0[s]) + grad[s]))
            for s in q
        )
        self.assertLess(max_step, P._SOFTCAL_TRUST_RADIUS,
                        "trust-region clip must not bind at the converged optimum")


# ---------------------------------------------------------------------------
# 2. Family cap actually binds
# ---------------------------------------------------------------------------

class FamilyCapTests(unittest.TestCase):
    def test_family_cap_bounds_summed_alpha(self):
        # A totals ladder with many liquid rungs would, uncapped, contribute
        # sum(alpha) >> a single market.  The family cap must clamp the SUM.
        mx = _liquid_matrix()
        params = _softcal_params()
        raw = P.build_soft_markets(_match(), mx, params)
        capped = P.apply_family_caps(raw, params)

        totals_raw = [m for m in raw if m.family == "totals"]
        totals_capped = [m for m in capped if m.family == "totals"]
        self.assertGreaterEqual(len(totals_raw), 3, "need a multi-rung ladder for this test")

        raw_sum = sum(m.alpha for m in totals_raw)
        capped_sum = sum(m.alpha for m in totals_capped)
        self.assertGreater(raw_sum, params.softcal_alpha_family_cap,
                           "precondition: uncapped totals mass must exceed the cap")
        self.assertLessEqual(capped_sum, params.softcal_alpha_family_cap + 1e-9,
                             "family cap must bind: summed alpha <= family_cap")

    def test_family_cap_preserves_relative_weights_within_family(self):
        # The FAMILY scaling step multiplies every rung in a family by one common
        # factor, so it preserves the ratio between two rungs.  (The per-market
        # cap is a separate per-rung clamp applied first; the invariant under
        # test is the family step alone, so we compare against the
        # post-per-market-cap alphas.)
        mx = _liquid_matrix()
        params = _softcal_params()
        raw = {m.label: m for m in P.build_soft_markets(_match(), mx, params)}
        post_market = {
            lab: min(m.alpha, params.softcal_alpha_per_market_cap) for lab, m in raw.items()
        }
        capped = {m.label: m for m in P.apply_family_caps(list(raw.values()), params)}
        totals_labels = [lab for lab, m in raw.items() if m.family == "totals"]
        a, b = totals_labels[0], totals_labels[1]
        self.assertAlmostEqual(
            post_market[a] / post_market[b],
            capped[a].alpha / capped[b].alpha,
            places=9,
        )

    def test_per_market_cap_bounds_single_alpha(self):
        # A single ultra-liquid, zero-spread quote would have enormous inverse
        # variance; the per-market cap must clamp it.
        params = _softcal_params(softcal_alpha_per_market_cap=2.0, softcal_alpha_family_cap=1e9)
        quotes = [
            _quote("btts", "yes", 0.55, liquidity=1e9, volume=1e9, spread=0.0),
            _quote("btts", "no", 0.47, liquidity=1e9, volume=1e9, spread=0.0),
        ]
        mx = EventMarketMatrix(match_id="m1", home="Home FC", away="Away FC", markets=quotes)
        capped = P.apply_family_caps(P.build_soft_markets(_match(), mx, params), params)
        self.assertTrue(capped)
        for m in capped:
            self.assertLessEqual(m.alpha, 2.0 + 1e-9)


# ---------------------------------------------------------------------------
# 3. Dixon-Coles low-score correction shifts the right cells
# ---------------------------------------------------------------------------

class DixonColesPriorTests(unittest.TestCase):
    def test_dc_rho_negative_lifts_low_score_draws(self):
        # rho < 0 is the empirically typical sign: it lifts 0-0 and 1-1 (and the
        # overall draw mass) relative to independent Poisson, and the change is
        # confined to the four low-score cells before renormalisation.
        lh, la = 1.4, 1.1
        indep = P.dixon_coles_prior(lh, la, _softcal_params(dixon_coles_rho=0.0))
        dc = P.dixon_coles_prior(lh, la, _softcal_params(dixon_coles_rho=-0.08))

        # 0-0 and 1-1 lift; 1-0 and 0-1 fall (the four tau cells move).
        self.assertGreater(dc[(0, 0)], indep[(0, 0)])
        self.assertGreater(dc[(1, 1)], indep[(1, 1)])
        self.assertLess(dc[(1, 0)], indep[(1, 0)])
        self.assertLess(dc[(0, 1)], indep[(0, 1)])

        # Total draw mass increases under rho<0.
        draw_indep = sum(p for (h, a), p in indep.items() if h == a)
        draw_dc = sum(p for (h, a), p in dc.items() if h == a)
        self.assertGreater(draw_dc, draw_indep)

    def test_dc_only_touches_low_score_cells(self):
        # Cells outside {(0,0),(0,1),(1,0),(1,1)} change ONLY through
        # renormalisation (a single common factor), so their pairwise ratios are
        # preserved between the independent and DC grids.
        lh, la = 1.6, 1.2
        indep = P.dixon_coles_prior(lh, la, _softcal_params(dixon_coles_rho=0.0))
        dc = P.dixon_coles_prior(lh, la, _softcal_params(dixon_coles_rho=-0.06))
        low = {(0, 0), (0, 1), (1, 0), (1, 1)}
        high_cells = [c for c in indep if c not in low and indep[c] > 0 and dc[c] > 0]
        ratios = [dc[c] / indep[c] for c in high_cells]
        # All high-score cells scaled by the SAME renormalisation constant.
        self.assertLess(max(ratios) - min(ratios), 1e-9)

    def test_default_softcal_prior_is_independent_poisson(self):
        # dixon_coles_rho defaults to 0.0, so the soft prior equals the pure
        # independent-Poisson grid (DC off unless explicitly enabled).
        lh, la = 1.5, 1.0
        prior = P.dixon_coles_prior(lh, la, _softcal_params())
        reference = P.poisson_grid(lh, la, DEFAULT_PARAMS.max_goals, rho=0.0)
        for cell in reference:
            self.assertAlmostEqual(prior[cell], reference[cell], places=12)


# ---------------------------------------------------------------------------
# 4. Explicit 8+ tail bucket
# ---------------------------------------------------------------------------

class TailBucketTests(unittest.TestCase):
    def test_tail_probability_is_explicit_and_matches_grid(self):
        res = P.fit_score_distribution_soft(_match(), _liquid_matrix(), params=_softcal_params())
        threshold = DEFAULT_PARAMS.softcal_tail_threshold
        manual = sum(p for (h, a), p in res.distribution.probs.items() if h + a >= threshold)
        self.assertAlmostEqual(res.tail_prob, manual, places=12)
        # The bucket is carried, not zeroed: with max_goals=7 the only way to
        # reach total>=8 is the (7,>=1)/(>=1,7) shoulder, which is small but
        # nonzero — it must NOT be silently truncated to 0.
        self.assertGreater(res.tail_prob, 0.0)

    def test_tail_helper_counts_at_or_above_threshold(self):
        grid = {(0, 0): 0.5, (4, 4): 0.2, (7, 1): 0.2, (3, 4): 0.1}
        # threshold 8: cells with h+a >= 8 are (4,4)=8 and (7,1)=8 -> 0.4.
        self.assertAlmostEqual(P._tail_probability(grid, 8), 0.4, places=12)
        # (3,4)=7 is below 8 and excluded.
        self.assertAlmostEqual(P._tail_probability(grid, 9), 0.0, places=12)


# ---------------------------------------------------------------------------
# 5. Thin shrinkage pulls toward q0
# ---------------------------------------------------------------------------

class ThinShrinkageTests(unittest.TestCase):
    def test_thin_target_moves_toward_prior_projection(self):
        # A thin market's used target is a beta-blend of book price and prior
        # projection, so it sits strictly between the two (for 0<beta<1) and
        # closer to the prior than the raw book price is.
        mx = _liquid_matrix()
        params = _softcal_params(softcal_shrink_beta=0.5)
        lh, la = P.prior_lambdas(mx, params=params)
        q0 = P.dixon_coles_prior(lh, la, params)

        # Build a deliberately-thin market: book says 0.80 but prior projects low.
        pred = lambda h, a: h + a > 4.5  # rare event -> low prior mass
        thin = P.SoftMarket(
            family="totals", is_binary=True, target=0.80, predicate=pred,
            alpha=0.001, spread=0.2, sigma=1.0, label="thin", thin=True,
        )
        prior_proj = P._project(q0, pred)
        used = P._shrink_target(thin, q0, params)
        expected = 0.5 * 0.80 + 0.5 * prior_proj
        self.assertAlmostEqual(used, expected, places=12)
        # Shrunk target lies strictly between book and prior, pulled off the book.
        self.assertLess(used, 0.80)
        self.assertGreater(used, prior_proj)

    def test_non_thin_target_is_untouched(self):
        mx = _liquid_matrix()
        params = _softcal_params()
        lh, la = P.prior_lambdas(mx, params=params)
        q0 = P.dixon_coles_prior(lh, la, params)
        pred = lambda h, a: h > a
        liquid = P.SoftMarket(
            family="1x2", is_binary=False, target=0.55, predicate=pred,
            alpha=100.0, spread=0.01, sigma=0.05, label="liquid", thin=False,
        )
        self.assertEqual(P._shrink_target(liquid, q0, params), 0.55)

    def test_all_thin_markets_flag_no_bet_reason(self):
        # When every market is thin/stale the result must flag that the grid is
        # mostly prior (a honesty signal for the reference layer).
        params = _softcal_params(softcal_thin_alpha=1e9)  # force everything thin
        res = P.fit_score_distribution_soft(_match(), _liquid_matrix(), params=params)
        self.assertIsNotNone(res.no_bet_reason)
        self.assertIn("thin", res.no_bet_reason.lower())


# ---------------------------------------------------------------------------
# 6. Output bundle + devig-variance machinery
# ---------------------------------------------------------------------------

class SoftOutputTests(unittest.TestCase):
    def test_projections_carry_z_residual_and_influence(self):
        res = P.fit_score_distribution_soft(_match(), _liquid_matrix(), params=_softcal_params())
        self.assertGreater(len(res.projections), 0)
        total_influence = sum(pr.market_influence for pr in res.projections)
        # market_influence is alpha_g / total_alpha -> sums to 1 across markets.
        self.assertAlmostEqual(total_influence, 1.0, places=9)
        for pr in res.projections:
            self.assertIsNotNone(pr.z_residual)
            # z = |model - book| / sigma >= 0.
            self.assertGreaterEqual(pr.z_residual, 0.0)
            self.assertGreater(pr.sigma, 0.0)

    def test_empty_market_returns_prior_only(self):
        mx = EventMarketMatrix(match_id="m1", home="Home FC", away="Away FC", markets=[])
        res = P.fit_score_distribution_soft(_match(), mx, params=_softcal_params())
        self.assertEqual(res.projections, [])
        self.assertIsNotNone(res.no_bet_reason)
        self.assertAlmostEqual(sum(res.distribution.probs.values()), 1.0, places=12)

    def test_logit_devig_recovers_known_values(self):
        # Additive-logit devig solves sum_i sigmoid(logit(r_i) - c) = 1.  For
        # [0.6, 0.5] (booksum 1.1) the bisection yields c such that the fair pair
        # is [0.5505102572..., 0.4494897427...].  Assert the exact root value, so
        # a regression in the solver (or a collapse back to proportional) fails.
        out = P.logit_devig([0.6, 0.5])
        self.assertAlmostEqual(sum(out), 1.0, places=12)
        self.assertAlmostEqual(out[0], 0.5505102572168219, places=10)
        self.assertAlmostEqual(out[1], 0.4494897427831782, places=10)

    def test_logit_devig_is_distinct_from_proportional(self):
        # The map only adds value as a THIRD independent devig if it genuinely
        # differs from proportional — otherwise the cross-map variance term is
        # biased low.  Lock that distinctness in.
        raw = [0.50, 0.30, 0.28]
        booksum = sum(raw)
        proportional = [r / booksum for r in raw]
        logit = P.logit_devig(raw)
        self.assertGreater(max(abs(a - b) for a, b in zip(logit, proportional)), 1e-4)

    def test_logit_devig_identity_when_no_vig(self):
        # A fair book (booksum 1.0) must come back unchanged (c -> 0).
        out = P.logit_devig([0.7, 0.3])
        self.assertAlmostEqual(out[0], 0.7, places=10)
        self.assertAlmostEqual(out[1], 0.3, places=10)

    def test_devig_variance_zero_for_agreeing_maps(self):
        self.assertEqual(P.devig_variance([0.5]), 0.0)
        self.assertEqual(P.devig_variance([0.5, 0.5, 0.5]), 0.0)
        self.assertGreater(P.devig_variance([0.50, 0.55, 0.60]), 0.0)

    def test_reliability_weight_higher_for_tighter_spread(self):
        # Lower spread -> lower variance -> higher alpha (more trust).
        params = _softcal_params()
        tight = _quote("btts", "yes", 0.5, liquidity=5000, volume=20000, spread=0.005)
        wide = _quote("btts", "yes", 0.5, liquidity=5000, volume=20000, spread=0.20)
        a_tight, s_tight, _ = P.reliability_weight(tight, 0.0, params)
        a_wide, s_wide, _ = P.reliability_weight(wide, 0.0, params)
        self.assertGreater(a_tight, a_wide)
        self.assertLess(s_tight, s_wide)


# ---------------------------------------------------------------------------
# 7. Default-off invariant (the regression firewall)
# ---------------------------------------------------------------------------

class DefaultPathUnchangedTests(unittest.TestCase):
    def test_use_softcal_defaults_off(self):
        self.assertFalse(DEFAULT_PARAMS.use_softcal)

    def test_default_fit_uses_legacy_ipf_not_soft(self):
        # With use_softcal=False the public fitter must produce EXACTLY the
        # legacy IPF grid — byte-identical, never touching the soft path.
        match, mx = _match(), _liquid_matrix()
        legacy = P.fit_score_distribution(match, mx, params=DEFAULT_PARAMS)

        # Reconstruct the legacy result independently (prior -> IPF) and compare.
        lh, la = P.prior_lambdas(mx, params=DEFAULT_PARAMS)
        base = P.poisson_grid(lh, la, DEFAULT_PARAMS.max_goals, rho=DEFAULT_PARAMS.dixon_coles_rho)
        constraints = P.build_market_constraints(match, mx, params=DEFAULT_PARAMS)
        expected = P.ScoreDistribution(
            P.calibrate_distribution(base, constraints, params=DEFAULT_PARAMS),
            max_goals=DEFAULT_PARAMS.max_goals,
        )
        for cell in expected.probs:
            self.assertAlmostEqual(legacy.probs[cell], expected.probs[cell], places=12)

    def test_soft_and_legacy_differ(self):
        # Sanity: the two paths are genuinely different engines (so the opt-in
        # is meaningful), not accidentally identical.
        match, mx = _match(), _liquid_matrix()
        legacy = P.fit_score_distribution(match, mx, params=DEFAULT_PARAMS)
        soft = P.fit_score_distribution(match, mx, params=_softcal_params())
        diff = max(abs(legacy.probs[c] - soft.probs[c]) for c in legacy.probs)
        self.assertGreater(diff, 1e-6)


if __name__ == "__main__":
    unittest.main()
