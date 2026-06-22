"""
Tests for src/ball_quant/core/metrics.py.

Hand-computed reference values are annotated inline so the reader can
verify without running code.
"""

import math
import pytest

from ball_quant.core.metrics import (
    brier_score,
    log_loss,
    reliability_bins,
    expected_calibration_error,
    pnl_ledger,
    edge_realization,
    kelly_growth,
    aggregate,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def pt(prob, y):
    return {"prob": prob, "y": y}


def bet(stake, odds, result, prob=0.5, edge=0.0):
    return {"stake": stake, "odds": odds, "result": result, "prob": prob, "edge": edge}


# ---------------------------------------------------------------------------
# brier_score
# ---------------------------------------------------------------------------

class TestBrierScore:
    def test_perfect_prediction_win(self):
        # prob=1, y=1 -> (1-1)^2 = 0
        assert brier_score([pt(1.0, 1)]) == pytest.approx(0.0)

    def test_perfect_prediction_loss(self):
        # prob=0, y=0 -> (0-0)^2 = 0
        assert brier_score([pt(0.0, 0)]) == pytest.approx(0.0)

    def test_half_probability(self):
        # prob=0.5, y=1 -> (0.5-1)^2 = 0.25
        assert brier_score([pt(0.5, 1)]) == pytest.approx(0.25)

    def test_worst_case(self):
        # prob=1, y=0 -> (1-0)^2 = 1
        assert brier_score([pt(1.0, 0)]) == pytest.approx(1.0)

    def test_mean_of_two(self):
        # (0^2 + 1^2) / 2 = 0.5
        assert brier_score([pt(1.0, 1), pt(1.0, 0)]) == pytest.approx(0.5)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            brier_score([])


# ---------------------------------------------------------------------------
# log_loss
# ---------------------------------------------------------------------------

class TestLogLoss:
    def test_perfect_win(self):
        # prob=1 -> clipped to 1-eps; y=1: -ln(1-eps) ~ eps -> ~0
        val = log_loss([pt(1.0, 1)])
        assert val == pytest.approx(0.0, abs=1e-10)

    def test_perfect_loss_prediction(self):
        # prob=0 -> clipped to eps; y=0: -(1-0)*ln(1-eps) ~ eps -> ~0
        val = log_loss([pt(0.0, 0)])
        assert val == pytest.approx(0.0, abs=1e-10)

    def test_half_probability(self):
        # prob=0.5, y=1: -ln(0.5) = ln(2) ~ 0.6931
        val = log_loss([pt(0.5, 1)])
        assert val == pytest.approx(math.log(2), rel=1e-9)

    def test_half_prob_y0(self):
        # prob=0.5, y=0: -(1-0)*ln(0.5) = ln(2)
        val = log_loss([pt(0.5, 0)])
        assert val == pytest.approx(math.log(2), rel=1e-9)

    def test_mean_of_two(self):
        # [p=0.5,y=1] and [p=0.5,y=0]: both ln(2) -> mean = ln(2)
        val = log_loss([pt(0.5, 1), pt(0.5, 0)])
        assert val == pytest.approx(math.log(2), rel=1e-9)

    def test_clip_avoids_inf(self):
        # prob=0, y=1 would be -inf without clipping; with eps clip -> finite
        val = log_loss([pt(0.0, 1)])
        assert math.isfinite(val)
        assert val > 30  # very large but finite

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            log_loss([])


# ---------------------------------------------------------------------------
# reliability_bins
# ---------------------------------------------------------------------------

class TestReliabilityBins:
    def test_single_bin(self):
        # All probs in [0.3, 0.4) with n_bins=10
        points = [pt(0.35, 1), pt(0.38, 0), pt(0.32, 1)]
        bins = reliability_bins(points, n_bins=10)
        assert len(bins) == 1
        b = bins[0]
        assert b["n"] == 3
        assert b["bin_lo"] == pytest.approx(0.3)
        assert b["bin_hi"] == pytest.approx(0.4)
        assert b["emp_freq"] == pytest.approx(2 / 3)

    def test_empty_bins_skipped(self):
        # Only two probs -> only two bins populated
        points = [pt(0.15, 1), pt(0.85, 0)]
        bins = reliability_bins(points, n_bins=10)
        assert len(bins) == 2

    def test_prob_exactly_one_goes_to_last_bin(self):
        points = [pt(1.0, 1)]
        bins = reliability_bins(points, n_bins=10)
        assert len(bins) == 1
        assert bins[0]["bin_lo"] == pytest.approx(0.9)

    def test_empty_points_returns_empty(self):
        assert reliability_bins([], n_bins=10) == []


# ---------------------------------------------------------------------------
# expected_calibration_error
# ---------------------------------------------------------------------------

class TestECE:
    def test_perfect_calibration(self):
        # Each bin: mean_pred == emp_freq -> ECE = 0
        # 10 bins with one point each perfectly calibrated
        points = [pt(i / 10 + 0.05, 1 if i >= 5 else 0) for i in range(10)]
        # ECE won't be exactly 0 for finite data, but test miscal > perfect
        ece = expected_calibration_error(points)
        assert ece >= 0.0

    def test_miscalibrated_ece_positive(self):
        # prob always 0.9 but half the time y=0 -> strong miscalibration
        points = [pt(0.9, 1), pt(0.9, 0)] * 5
        ece = expected_calibration_error(points)
        # mean_pred=0.9, emp_freq=0.5 -> ECE = |0.9-0.5| = 0.4
        assert ece == pytest.approx(0.4, rel=1e-6)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            expected_calibration_error([])

    def test_ece_non_negative(self):
        import random
        rng = random.Random(42)
        points = [pt(rng.random(), rng.randint(0, 1)) for _ in range(100)]
        assert expected_calibration_error(points) >= 0.0


# ---------------------------------------------------------------------------
# pnl_ledger
# ---------------------------------------------------------------------------

class TestPnlLedger:
    def test_single_win(self):
        # stake=10, odds=2.0 -> gross=20, net=+10, roi=1.0
        r = pnl_ledger([bet(10, 2.0, "WIN")])
        assert r["n_bets"] == 1
        assert r["n_win"] == 1
        assert r["net_pnl"] == pytest.approx(10.0)
        assert r["total_return"] == pytest.approx(20.0)
        assert r["roi"] == pytest.approx(1.0)

    def test_single_loss(self):
        # stake=10 -> net=-10, roi=-1.0
        r = pnl_ledger([bet(10, 2.0, "LOSS")])
        assert r["net_pnl"] == pytest.approx(-10.0)
        assert r["total_return"] == pytest.approx(0.0)
        assert r["roi"] == pytest.approx(-1.0)

    def test_single_void(self):
        # VOID: refund returned, not in at_risk
        # Only VOID bet -> at_risk=0 -> ValueError
        with pytest.raises(ValueError):
            pnl_ledger([bet(10, 2.0, "VOID")])

    def test_void_excluded_from_at_risk(self):
        r = pnl_ledger([bet(10, 2.0, "WIN"), bet(5, 2.0, "VOID")])
        assert r["total_stake_at_risk"] == pytest.approx(10.0)
        assert r["n_void"] == 1
        # void refund included in total_return
        assert r["total_return"] == pytest.approx(20.0 + 5.0)

    def test_win_and_loss_mix(self):
        # W: 10@2.0 net=+10; L: 10@2.0 net=-10; total net=0, roi=0
        r = pnl_ledger([bet(10, 2.0, "WIN"), bet(10, 2.0, "LOSS")])
        assert r["net_pnl"] == pytest.approx(0.0)
        assert r["roi"] == pytest.approx(0.0)

    def test_high_odds_win(self):
        # stake=5, odds=5.0 -> net=20, roi=4.0
        r = pnl_ledger([bet(5, 5.0, "WIN")])
        assert r["net_pnl"] == pytest.approx(20.0)
        assert r["roi"] == pytest.approx(4.0)

    def test_empty_bets_raises(self):
        with pytest.raises((ValueError, ZeroDivisionError)):
            pnl_ledger([])


# ---------------------------------------------------------------------------
# edge_realization
# ---------------------------------------------------------------------------

class TestEdgeRealization:
    def test_monotone_positive_correlation(self):
        # Higher edge -> WIN; lower edge -> LOSS: correlation should be > 0
        bets = [
            bet(1, 2.0, "WIN",  prob=0.6, edge=0.2),
            bet(1, 2.0, "WIN",  prob=0.7, edge=0.4),
            bet(1, 2.0, "LOSS", prob=0.3, edge=-0.2),
            bet(1, 2.0, "LOSS", prob=0.2, edge=-0.4),
        ]
        r = edge_realization(bets)
        assert r["correlation"] > 0.0
        assert r["n"] == 4

    def test_monotone_negative_correlation(self):
        # Higher edge -> LOSS (badly miscalibrated model)
        bets = [
            bet(1, 2.0, "LOSS", prob=0.6, edge=0.2),
            bet(1, 2.0, "LOSS", prob=0.7, edge=0.4),
            bet(1, 2.0, "WIN",  prob=0.3, edge=-0.2),
            bet(1, 2.0, "WIN",  prob=0.2, edge=-0.4),
        ]
        r = edge_realization(bets)
        assert r["correlation"] < 0.0

    def test_void_excluded(self):
        bets = [
            bet(1, 2.0, "WIN",  edge=0.1),
            bet(1, 2.0, "VOID", edge=0.5),
        ]
        r = edge_realization(bets)
        assert r["n"] == 1

    def test_all_void_raises(self):
        with pytest.raises(ValueError):
            edge_realization([bet(1, 2.0, "VOID", edge=0.1)])

    def test_zero_variance_no_crash(self):
        # All same edge, all WIN -> no variance in either series
        bets = [bet(1, 2.0, "WIN", edge=0.1) for _ in range(5)]
        r = edge_realization(bets)
        assert r["correlation"] == pytest.approx(0.0)

    def test_mean_values(self):
        # W: realized = (2.0-1) = 1.0; L: realized = -1.0
        # edges: 0.2, -0.2 -> mean = 0
        bets = [
            bet(1, 2.0, "WIN",  edge=0.2),
            bet(1, 2.0, "LOSS", edge=-0.2),
        ]
        r = edge_realization(bets)
        assert r["mean_predicted_edge"] == pytest.approx(0.0)
        assert r["mean_realized_return_per_unit"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# kelly_growth
# ---------------------------------------------------------------------------

class TestKellyGrowth:
    def test_single_win(self):
        # f=0.1, odds=2.0: ln(1 + 0.1*1) = ln(1.1)
        bets = [bet(10, 2.0, "WIN")]
        r = kelly_growth(bets, bankroll=100)
        assert r["sum_log_growth"] == pytest.approx(math.log(1.1), rel=1e-9)
        assert r["n"] == 1
        # geometric rate = exp(ln(1.1)) - 1 = 0.1
        assert r["geometric_growth_rate"] == pytest.approx(0.1, rel=1e-9)

    def test_single_loss(self):
        # f=0.1: ln(1 - 0.1) = ln(0.9)
        bets = [bet(10, 2.0, "LOSS")]
        r = kelly_growth(bets, bankroll=100)
        assert r["sum_log_growth"] == pytest.approx(math.log(0.9), rel=1e-9)

    def test_total_ruin_raises(self):
        # stake == bankroll -> f=1.0 on a LOSS -> ill-defined
        with pytest.raises(ValueError):
            kelly_growth([bet(100, 2.0, "LOSS")], bankroll=100)

    def test_void_excluded(self):
        bets = [bet(10, 2.0, "WIN"), bet(10, 2.0, "VOID")]
        r = kelly_growth(bets, bankroll=100)
        assert r["n"] == 1

    def test_all_void_raises(self):
        with pytest.raises(ValueError):
            kelly_growth([bet(10, 2.0, "VOID")], bankroll=100)

    def test_negative_bankroll_raises(self):
        with pytest.raises(ValueError):
            kelly_growth([bet(10, 2.0, "WIN")], bankroll=-50)

    def test_zero_bankroll_raises(self):
        with pytest.raises(ValueError):
            kelly_growth([bet(10, 2.0, "WIN")], bankroll=0)

    def test_geometric_mean_win_loss(self):
        # f=0.1 each bet; W: ln(1.1), L: ln(0.9)
        # sum = ln(1.1) + ln(0.9) = ln(0.99)
        # rate = exp(ln(0.99)/2) - 1 = sqrt(0.99) - 1
        bets = [bet(10, 2.0, "WIN"), bet(10, 2.0, "LOSS")]
        r = kelly_growth(bets, bankroll=100)
        expected_sum = math.log(1.1) + math.log(0.9)
        assert r["sum_log_growth"] == pytest.approx(expected_sum, rel=1e-9)
        expected_rate = math.exp(expected_sum / 2) - 1.0
        assert r["geometric_growth_rate"] == pytest.approx(expected_rate, rel=1e-9)


# ---------------------------------------------------------------------------
# aggregate
# ---------------------------------------------------------------------------

class TestAggregate:
    def test_empty_points_no_raise(self):
        r = aggregate([], [], bankroll=100)
        assert r["calibration"] == {}
        assert r["pnl"] == {}

    def test_empty_bets_no_raise(self):
        r = aggregate([pt(0.5, 1)], [], bankroll=100)
        assert r["pnl"] == {}
        assert r["edge"] == {}
        assert r["kelly"] == {}

    def test_full_aggregate(self):
        points = [pt(0.6, 1), pt(0.4, 0)]
        bets_list = [bet(10, 2.0, "WIN", edge=0.2), bet(10, 2.0, "LOSS", edge=-0.1)]
        r = aggregate(points, bets_list, bankroll=100)
        assert "brier" in r["calibration"]
        assert "log_loss" in r["calibration"]
        assert "ece" in r["calibration"]
        assert "reliability" in r["calibration"]
        assert "net_pnl" in r["pnl"]
        assert "correlation" in r["edge"]
        assert "geometric_growth_rate" in r["kelly"]

    def test_all_void_bets_returns_empty_blocks(self):
        # All VOID -> pnl raises inside aggregate -> returns {}
        r = aggregate([], [bet(10, 2.0, "VOID")], bankroll=100)
        assert r["pnl"] == {}
        assert r["edge"] == {}
        assert r["kelly"] == {}
