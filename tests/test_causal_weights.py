"""
Spec tests for the optional inverse-variance constraint-weighting scheme.

Three groups:
  A. Heuristic reproduction — default params must produce the pre-change number exactly.
  B. Inverse-variance directional — monotonicity and toggle-routing.
  C. Edge cases — None spread, unknown weight_scheme raises.
"""
from __future__ import annotations

import pytest

from ball_quant.models import MarketQuote
from ball_quant.core.params import StrategyParams, DEFAULT_PARAMS
from ball_quant.core.causal import (
    quote_market_reliability,
    quote_constraint_strength,
    quote_inverse_variance_reliability,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _make_quote(
    *,
    category: str = "moneyline",
    spread: float | None = 0.04,
    probability: float | None = 0.50,
    liquidity: float | None = 5000.0,
    volume: float | None = 10000.0,
    model_weight: float | None = None,
) -> MarketQuote:
    return MarketQuote(
        market_id="test",
        question="test",
        category=category,
        outcome="home",
        probability=probability,
        spread=spread,
        liquidity=liquidity,
        volume=volume,
        model_weight=model_weight,
    )


# ---------------------------------------------------------------------------
# A. Heuristic reproduction (default weight_scheme="heuristic")
# ---------------------------------------------------------------------------

class TestHeuristicReproduction:
    """Default params must produce the EXACT pre-change numbers."""

    def test_default_weight_scheme_is_heuristic(self):
        assert DEFAULT_PARAMS.weight_scheme == "heuristic"

    def test_reliability_tight_spread_high_liquidity(self):
        # spread=0.02 → +0.18; liquidity=100_000 → +0.12; volume=100_000 → +0.08
        # base=0.72 → 0.72+0.18+0.12+0.08 = 1.10 → clamped 1.0
        q = _make_quote(spread=0.02, liquidity=100_000, volume=100_000)
        assert quote_market_reliability(q, DEFAULT_PARAMS) == pytest.approx(1.0)

    def test_reliability_wide_spread_low_liquidity(self):
        # spread=0.50 → -0.55; liquidity=400 → -0.18; volume=400 → -0.06
        # 0.72 - 0.55 - 0.18 - 0.06 = -0.07 → clamped 0.05
        q = _make_quote(spread=0.50, liquidity=400, volume=400)
        assert quote_market_reliability(q, DEFAULT_PARAMS) == pytest.approx(0.05)

    def test_reliability_none_spread(self):
        # spread None → -0.08; liquidity None → -0.06; volume None ignored
        # 0.72 - 0.08 - 0.06 = 0.58
        q = _make_quote(spread=None, liquidity=None, volume=None)
        assert quote_market_reliability(q, DEFAULT_PARAMS) == pytest.approx(0.58)

    def test_constraint_strength_moneyline(self):
        # moneyline model_weight=1.0; spread=0.04 → +0.08; liq=5000 → none; vol=10000 → none
        # 0.72 + 0.08 = 0.80; 0.80 * 1.0 = 0.80
        q = _make_quote(category="moneyline", spread=0.04, liquidity=5000, volume=10000)
        assert quote_constraint_strength(q, DEFAULT_PARAMS) == pytest.approx(0.80)

    def test_constraint_strength_none_quote(self):
        assert quote_constraint_strength(None, DEFAULT_PARAMS) == pytest.approx(0.20)

    def test_inverse_variance_params_does_not_affect_heuristic(self):
        """Explicitly confirm that changing weight_scheme doesn't touch heuristic path."""
        iv_params = StrategyParams(weight_scheme="inverse_variance")
        heuristic_params = StrategyParams(weight_scheme="heuristic")
        q = _make_quote(spread=0.04)
        # heuristic path unchanged regardless of which params object we use for heuristic
        assert quote_market_reliability(q, heuristic_params) == pytest.approx(
            quote_market_reliability(q, DEFAULT_PARAMS)
        )
        # inverse_variance path DIFFERS
        iv_rel = quote_market_reliability(q, iv_params)
        h_rel = quote_market_reliability(q, heuristic_params)
        assert iv_rel != pytest.approx(h_rel), (
            "inverse_variance and heuristic must diverge on the same quote"
        )


# ---------------------------------------------------------------------------
# B. Inverse-variance directional tests
# ---------------------------------------------------------------------------

IV_PARAMS = StrategyParams(weight_scheme="inverse_variance")


class TestInverseVarianceDirectional:
    """Monotonicity and toggle-routing for weight_scheme='inverse_variance'."""

    # --- quote_inverse_variance_reliability ---

    def test_tight_spread_higher_reliability_than_wide(self):
        # Monotone decreasing in spread: spread=0.02 > spread=0.20
        r_tight = quote_inverse_variance_reliability(_make_quote(spread=0.02))
        r_wide = quote_inverse_variance_reliability(_make_quote(spread=0.20))
        assert r_tight > r_wide, (
            f"Expected tight-spread reliability {r_tight:.4f} > wide-spread {r_wide:.4f}"
        )

    def test_reliability_bounded_01(self):
        for spread in (0.001, 0.05, 0.50, 1.0):
            r = quote_inverse_variance_reliability(_make_quote(spread=spread))
            assert 0.0 < r <= 1.0, f"reliability {r} out of (0,1] for spread={spread}"

    def test_none_spread_returns_low_confidence_constant(self):
        r_none = quote_inverse_variance_reliability(_make_quote(spread=None))
        r_tight = quote_inverse_variance_reliability(_make_quote(spread=0.01))
        # None spread falls back to a constant below the tight-spread value
        assert r_none < r_tight

    def test_zero_spread_approaches_maximum(self):
        # At spread=0 the formula should give ≈1.0 (or at least > 0.95)
        r = quote_inverse_variance_reliability(_make_quote(spread=0.0))
        assert r > 0.95, f"Expected near-1.0 for zero spread, got {r:.4f}"

    # --- quote_constraint_strength with IV params ---

    def test_constraint_strength_monotone_in_spread(self):
        """Core directional test: spread=0.02 → strictly higher strength than spread=0.20."""
        q_tight = _make_quote(category="moneyline", spread=0.02)
        q_wide = _make_quote(category="moneyline", spread=0.20)
        s_tight = quote_constraint_strength(q_tight, IV_PARAMS)
        s_wide = quote_constraint_strength(q_wide, IV_PARAMS)
        assert s_tight > s_wide, (
            f"Expected tight-spread strength {s_tight:.4f} > wide-spread {s_wide:.4f}"
        )

    def test_toggle_diverges_from_heuristic(self):
        """IV toggle produces a DIFFERENT number than heuristic on the same quote."""
        q = _make_quote(category="moneyline", spread=0.04)
        s_iv = quote_constraint_strength(q, IV_PARAMS)
        s_h = quote_constraint_strength(q, DEFAULT_PARAMS)
        assert s_iv != pytest.approx(s_h), (
            f"Expected IV ({s_iv:.4f}) ≠ heuristic ({s_h:.4f})"
        )

    def test_profile_weight_still_applied_in_iv(self):
        """profile model_weight must still scale the IV reliability (not bypassed)."""
        q_mono = _make_quote(category="moneyline", spread=0.04)   # model_weight=1.0
        q_cs = _make_quote(category="correct_score", spread=0.04)  # model_weight=0.42
        s_mono = quote_constraint_strength(q_mono, IV_PARAMS)
        s_cs = quote_constraint_strength(q_cs, IV_PARAMS)
        assert s_mono > s_cs, (
            "moneyline (higher causal weight) must beat correct_score in IV mode"
        )

    def test_iv_strength_tight_vs_wide_numeric_snapshot(self):
        """Record the actual numbers so regressions are obvious; update intentionally."""
        q_tight = _make_quote(category="moneyline", spread=0.02, liquidity=5000, volume=10000)
        q_wide = _make_quote(category="moneyline", spread=0.20, liquidity=5000, volume=10000)
        s_tight = quote_constraint_strength(q_tight, IV_PARAMS)
        s_wide = quote_constraint_strength(q_wide, IV_PARAMS)
        # Not asserting exact values — just that tight > wide AND both in (0,1]
        assert 0.0 < s_wide < s_tight <= 1.0, (
            f"Snapshot: tight={s_tight:.4f}, wide={s_wide:.4f} — one is out of range or ordering wrong"
        )


# ---------------------------------------------------------------------------
# C. Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_unknown_weight_scheme_raises(self):
        bad_params = StrategyParams(weight_scheme="bogus")
        q = _make_quote()
        with pytest.raises(ValueError, match="weight_scheme"):
            quote_market_reliability(q, bad_params)

    def test_none_quote_is_scheme_agnostic(self):
        """quote=None short-circuits before scheme dispatch → same 0.20 either way."""
        iv = StrategyParams(weight_scheme="inverse_variance")
        assert quote_constraint_strength(None, iv) == pytest.approx(0.20)
        assert quote_constraint_strength(None, DEFAULT_PARAMS) == pytest.approx(0.20)
