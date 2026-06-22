"""Tests for strength_prior.py (pure math) and elo.py adapter parser.

All tests are network-free: the adapter parse test injects a small TSV fixture
that matches the real eloratings.net format (verified from live fetch).
"""
from __future__ import annotations

import math
import pytest

from ball_quant.core.params import StrategyParams
from ball_quant.core.strength_prior import (
    elo_lambda_prior,
    elo_z_scores,
    prior_blend_alpha,
    strength_to_lambda,
)
from ball_quant.adapters.elo import load_elo_ratings_from_fixtures

# ---------------------------------------------------------------------------
# Minimal fixture matching eloratings.net format
# World.tsv: rank  tie  code  elo  [peak_elo  ...  more columns]  (no header)
# en.teams.tsv: code<TAB>name
# ---------------------------------------------------------------------------

_WORLD_TSV_FIXTURE = """\
1\t1\tES\t2129\t1\t2189\t7\t1946\t19\t0\t-43\t-26\t-5\t-13\t+73\t783\t341\t302\t140\t1426
2\t2\tAR\t2128\t1\t2172\t5\t1987\t26\t0\t+1\t-2\t+11\t-9\t+73\t836\t330\t279\t227\t1435
3\t3\tFR\t2084\t1\t2135\t16\t1795\t40\t0\t-2\t+4\t-5\t-28\t+52\t756\t365\t224\t167\t1345
4\t4\tBR\t2060\t1\t2164\t12\t1946\t78\t0\t+25\t+24\t+27\t+23\t-28\t884\t444\t259\t181\t1587
5\t5\tEN\t2050\t1\t2079\t1\t2021\t3\t0\t-11\t-20\t-16\t+16\t-13\t637\t329\t157\t151\t1117
"""

_TEAMS_TSV_FIXTURE = """\
ES\tSpain\tEspaña
AR\tArgentina
FR\tFrance\tFrance national football team
BR\tBrazil\tBrasil
EN\tEngland
"""

_DEFAULT_PARAMS = StrategyParams()


# ---------------------------------------------------------------------------
# z-score tests
# ---------------------------------------------------------------------------

class TestEloZScores:
    def test_correctness(self):
        """Z-scores of known values should match hand calculation."""
        ratings = {"A": 2000.0, "B": 1800.0, "C": 1600.0}
        mu = (2000 + 1800 + 1600) / 3  # 1800
        variance = ((200**2) + (0**2) + (200**2)) / 3
        sigma = math.sqrt(variance)  # ~163.3

        z = elo_z_scores(ratings)
        assert abs(z["A"] - (2000 - mu) / sigma) < 1e-9
        assert abs(z["B"] - (1800 - mu) / sigma) < 1e-9
        assert abs(z["C"] - (1600 - mu) / sigma) < 1e-9

    def test_all_equal_degenerate(self):
        """All-equal Elo → all z-scores are 0.0 (no division by zero)."""
        ratings = {"A": 1800.0, "B": 1800.0, "C": 1800.0}
        z = elo_z_scores(ratings)
        assert all(v == 0.0 for v in z.values())

    def test_single_team(self):
        """Single-team field: z-score is 0.0."""
        z = elo_z_scores({"X": 2000.0})
        assert z["X"] == 0.0

    def test_empty(self):
        assert elo_z_scores({}) == {}

    def test_z_scores_mean_zero(self):
        """Z-scores across the field should sum to ~0 (within float error)."""
        ratings = {"A": 2100.0, "B": 1900.0, "C": 1800.0, "D": 1600.0}
        z = elo_z_scores(ratings)
        assert abs(sum(z.values())) < 1e-9


# ---------------------------------------------------------------------------
# strength_to_lambda tests
# ---------------------------------------------------------------------------

class TestStrengthToLambda:
    def test_symmetry_equal_strength(self):
        """Equal z-scores → both lambdas equal baseline exactly."""
        lh, la = strength_to_lambda(0.0, 0.0, _DEFAULT_PARAMS)
        assert lh == pytest.approx(_DEFAULT_PARAMS.elo_baseline_goals)
        assert la == pytest.approx(_DEFAULT_PARAMS.elo_baseline_goals)

    def test_baseline_preservation(self):
        """λ_h * λ_a = baseline^2 for arbitrary z-scores (product invariant)."""
        b = _DEFAULT_PARAMS.elo_baseline_goals
        for s_home, s_away in [(1.5, -0.5), (-2.0, 1.0), (0.3, 0.3), (3.0, -3.0)]:
            lh, la = strength_to_lambda(s_home, s_away, _DEFAULT_PARAMS)
            assert lh * la == pytest.approx(b ** 2, rel=1e-9)

    def test_monotonic_stronger_home(self):
        """Stronger home team → higher λ_home, lower λ_away."""
        lh0, la0 = strength_to_lambda(0.0, 0.0, _DEFAULT_PARAMS)
        lh1, la1 = strength_to_lambda(1.0, 0.0, _DEFAULT_PARAMS)
        lh2, la2 = strength_to_lambda(2.0, 0.0, _DEFAULT_PARAMS)
        assert lh2 > lh1 > lh0
        assert la2 < la1 < la0

    def test_monotonic_stronger_away(self):
        """Stronger away team → lower λ_home, higher λ_away."""
        lh0, la0 = strength_to_lambda(0.0, 0.0, _DEFAULT_PARAMS)
        lh1, la1 = strength_to_lambda(0.0, 1.0, _DEFAULT_PARAMS)
        assert lh1 < lh0
        assert la1 > la0

    def test_antisymmetry(self):
        """Flipping home/away swaps the lambdas exactly."""
        lh, la = strength_to_lambda(1.5, -0.5, _DEFAULT_PARAMS)
        lh2, la2 = strength_to_lambda(-0.5, 1.5, _DEFAULT_PARAMS)
        assert lh == pytest.approx(la2, rel=1e-9)
        assert la == pytest.approx(lh2, rel=1e-9)

    def test_custom_params(self):
        """Changing baseline scales both lambdas proportionally."""
        p = StrategyParams(elo_baseline_goals=1.5, elo_supremacy_coeff=0.40)
        lh, la = strength_to_lambda(0.0, 0.0, p)
        assert lh == pytest.approx(1.5)
        assert la == pytest.approx(1.5)


# ---------------------------------------------------------------------------
# prior_blend_alpha tests
# ---------------------------------------------------------------------------

class TestPriorBlendAlpha:
    def test_pure_prior_at_n_zero(self):
        """α=1.0 when N=0 — cold start is pure Elo prior."""
        assert prior_blend_alpha(0, _DEFAULT_PARAMS.elo_prior_kappa) == pytest.approx(1.0)

    def test_exact_formula(self):
        """α = κ/(κ+N) exactly for several N values."""
        kappa = _DEFAULT_PARAMS.elo_prior_kappa
        for n in [0, 1, 3, 7, 14, 100]:
            expected = kappa / (kappa + n)
            assert prior_blend_alpha(n, kappa) == pytest.approx(expected, rel=1e-9)

    def test_strictly_decreasing(self):
        """α is strictly decreasing with N > 0."""
        kappa = _DEFAULT_PARAMS.elo_prior_kappa
        alphas = [prior_blend_alpha(n, kappa) for n in range(0, 20)]
        for a, b in zip(alphas, alphas[1:]):
            assert a > b

    def test_approaches_zero(self):
        """α → 0 for large N."""
        kappa = _DEFAULT_PARAMS.elo_prior_kappa
        assert prior_blend_alpha(10000, kappa) < 0.001

    def test_invalid_kappa_raises(self):
        with pytest.raises(ValueError, match="kappa"):
            prior_blend_alpha(0, 0.0)

    def test_invalid_n_raises(self):
        with pytest.raises(ValueError, match="n_games"):
            prior_blend_alpha(-1, 3.5)


# ---------------------------------------------------------------------------
# elo_lambda_prior integration test (pure dict injection)
# ---------------------------------------------------------------------------

class TestEloLambdaPrior:
    def test_equal_teams_returns_baseline(self):
        """Equal Elo → both λ == baseline regardless of absolute rating."""
        ratings = {"spain": 2129.0, "argentina": 2129.0}
        lh, la = elo_lambda_prior("spain", "argentina", ratings, _DEFAULT_PARAMS)
        b = _DEFAULT_PARAMS.elo_baseline_goals
        assert lh == pytest.approx(b, rel=1e-6)
        assert la == pytest.approx(b, rel=1e-6)

    def test_stronger_home_higher_lambda(self):
        ratings = {"spain": 2129.0, "england": 1900.0}
        lh, la = elo_lambda_prior("spain", "england", ratings, _DEFAULT_PARAMS)
        assert lh > la

    def test_empty_ratings_raises(self):
        with pytest.raises(ValueError, match="empty"):
            elo_lambda_prior("spain", "england", {}, _DEFAULT_PARAMS)

    def test_missing_team_uses_average(self, caplog):
        """Unknown team falls back to z=0 (average), does not raise."""
        import logging
        ratings = {"spain": 2100.0, "france": 2000.0, "brazil": 1900.0}
        with caplog.at_level(logging.WARNING, logger="ball_quant.core.strength_prior"):
            lh, la = elo_lambda_prior("unknown_fc", "spain", ratings, _DEFAULT_PARAMS)
        # unknown_fc gets z=0, spain gets positive z → spain (away) > unknown (home)
        assert la > lh
        assert "unknown_fc" in caplog.text


# ---------------------------------------------------------------------------
# Adapter parser test — fixture of real eloratings.net TSV format, no network
# ---------------------------------------------------------------------------

class TestEloAdapterParser:
    def test_parse_world_tsv_fixture(self):
        """Parser extracts correct {canonical_name: elo} from fixture TSV."""
        result = load_elo_ratings_from_fixtures(
            _WORLD_TSV_FIXTURE,
            _TEAMS_TSV_FIXTURE,
        )
        # All 5 countries should be parsed
        assert len(result) == 5

        # Spot-check canonical names (normalize_team lowercases + alnum-normalises)
        assert "spain" in result
        assert "argentina" in result
        assert "france" in result
        assert "brazil" in result
        assert "england" in result

        # Elo values match fixture
        assert result["spain"] == pytest.approx(2129.0)
        assert result["argentina"] == pytest.approx(2128.0)
        assert result["france"] == pytest.approx(2084.0)
        assert result["brazil"] == pytest.approx(2060.0)
        assert result["england"] == pytest.approx(2050.0)

    def test_parse_empty_returns_empty(self):
        result = load_elo_ratings_from_fixtures("", "")
        assert result == {}

    def test_parse_malformed_elo_column_skipped(self):
        """Rows with non-numeric Elo are skipped (no crash)."""
        tsv = "1\t1\tES\tNOT_A_NUMBER\t2189\n2\t2\tAR\t2128\t2172\n"
        teams = "ES\tSpain\nAR\tArgentina\n"
        result = load_elo_ratings_from_fixtures(tsv, teams)
        assert "spain" not in result
        assert "argentina" in result
        assert result["argentina"] == pytest.approx(2128.0)

    def test_parse_missing_code_in_teams_uses_raw_code(self):
        """Country code absent from en.teams.tsv → keyed by raw code (normalised)."""
        tsv = "1\t1\tXX\t1999\t2100\n"
        teams = ""  # empty teams mapping
        result = load_elo_ratings_from_fixtures(tsv, teams)
        # normalize_team("XX") → "xx" or similar; either way not empty
        assert len(result) == 1


# ---------------------------------------------------------------------------
# Regression: display-name → normalized-key lookup bug
# ---------------------------------------------------------------------------

class TestEloLambdaPriorNormalizationRegression:
    """Regression for the display-name vs normalized-key mismatch.

    The adapter stores ratings keyed by normalize_team(name) (e.g. "portugal").
    elo_lambda_prior must normalize its home/away args before lookup so the keys
    match — otherwise every team misses and collapses to z=0 / baseline.
    """

    def _make_ratings(self):
        """Return a ratings dict keyed by normalized names (as adapter produces)."""
        from ball_quant.core.match_join import normalize_team
        return {
            normalize_team("Portugal"): 2100.0,
            normalize_team("DR Congo"): 1500.0,
            normalize_team("England"): 2050.0,
            normalize_team("Croatia"): 1900.0,
            normalize_team("Brazil"): 2060.0,
            normalize_team("Haiti"): 1400.0,
        }

    def test_display_names_resolve_not_baseline(self):
        """Display names (e.g. 'Portugal') must resolve to real z-scores.

        If the bug is present both λ values collapse to baseline (1.25) because
        every lookup misses.  After the fix, the stronger team must produce a
        higher λ and at least one side must differ from baseline.
        """
        ratings = self._make_ratings()
        p = StrategyParams()
        baseline = p.elo_baseline_goals

        lh, la = elo_lambda_prior("Portugal", "DR Congo", ratings, p)
        # Both must differ from baseline (real signal, not flat prior)
        assert lh != pytest.approx(baseline, rel=1e-3), (
            "Portugal λ_home collapsed to baseline — normalization fix missing"
        )
        assert la != pytest.approx(baseline, rel=1e-3), (
            "DR Congo λ_away collapsed to baseline — normalization fix missing"
        )
        # Portugal (Elo 2100) is far stronger than DR Congo (1500) → home gets higher λ
        assert lh > la, f"Expected Portugal λ_home > DR Congo λ_away, got {lh:.4f} vs {la:.4f}"

    def test_stronger_home_gets_higher_lambda_display_names(self):
        """With display names: stronger team must always produce the higher λ."""
        ratings = self._make_ratings()
        p = StrategyParams()

        for home, away in [("Portugal", "DR Congo"), ("England", "Croatia"), ("Brazil", "Haiti")]:
            lh, la = elo_lambda_prior(home, away, ratings, p)
            assert lh > la, (
                f"{home} vs {away}: expected lam_home ({lh:.4f}) > lam_away ({la:.4f})"
            )
            assert lh != la, f"{home} vs {away}: lambdas should not be equal"

    def test_genuinely_absent_team_still_falls_back_to_zero(self, caplog):
        """A team truly absent after normalization must still fall back to z=0."""
        import logging
        ratings = self._make_ratings()
        p = StrategyParams()

        with caplog.at_level(logging.WARNING, logger="ball_quant.core.strength_prior"):
            lh, la = elo_lambda_prior("Atlantis FC", "Portugal", ratings, p)

        # Warning must be emitted for the missing team
        assert "Atlantis" in caplog.text or "atlantisfc" in caplog.text or "atlantis" in caplog.text
        # Atlantis (z=0, average) vs Portugal (high Elo, positive z) → away should be higher
        assert la > lh, (
            f"Portugal (away, strong) should have higher λ than Atlantis (home, z=0): "
            f"lh={lh:.4f} la={la:.4f}"
        )
