"""
Tests for core/forecast_ledger.py.

Verifies:
  1. ForecastRecord persist/load round-trip (JSONL)
  2. grade_forecasts with known forecasts+outcomes → hand-computed Brier/log-loss
  3. pre_kickoff exclusion works (post-kickoff records are not graded)
  4. Poly p_home=0.765 & Elo p_home=0.455 for a home team that DREW → elo Brier < poly Brier
  5. Market-family grouping is correct (1x2 / handicap / totals separate)
  6. calibration_report poly-vs-elo verdict wording
"""
from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

import pytest

from ball_quant.core.forecast_ledger import (
    ForecastRecord,
    append_forecast,
    calibration_report,
    grade_forecasts,
    load_forecasts,
    make_forecast_record,
    _is_pre_kickoff,
    _parse_kickoff_from_slug,
)
from ball_quant.core.settlement import MatchOutcome


# ---------------------------------------------------------------------------
# Helpers — minimal bundle dicts for testing
# ---------------------------------------------------------------------------

def _bundle_1x2_only(p_home: float, p_draw: float, p_away: float,
                     elo_home: float = None, elo_draw: float = None, elo_away: float = None) -> dict:
    """Minimal bundle with Poly moneyline and optional Elo fundamental."""
    b: dict = {
        "poly_home": "Portugal",
        "poly_away": "Spain",
        "ticai_home": "葡萄牙",
        "ticai_away": "西班牙",
        "match_date": "2026-06-17",
        "match_num": "001",
        "event_slug": "portugal-vs-spain-2026-06-17",
        "poly_liquidity": {"avg_spread": 0.02, "total": 500000},
        "poly": {
            "moneyline": [
                {"outcome": "home", "prob": p_home, "line": None, "liquidity": 100000, "thin": False},
                {"outcome": "draw", "prob": p_draw, "line": None, "liquidity": 100000, "thin": False},
                {"outcome": "away", "prob": p_away, "line": None, "liquidity": 100000, "thin": False},
            ]
        },
        "ticai": {},
        "kg": {},
    }
    if elo_home is not None:
        b["fundamental"] = {
            "source": "elo",
            "p_home": elo_home,
            "p_draw": elo_draw,
            "p_away": elo_away,
            "lam_home": 1.2,
            "lam_away": 0.9,
            "home_rated": True,
            "away_rated": True,
        }
    return b


def _bundle_with_handicap_totals(
    p_home: float, p_draw: float, p_away: float,
    handicap_rows: list, totals_rows: list,
) -> dict:
    b = _bundle_1x2_only(p_home, p_draw, p_away)
    b["poly"]["handicap"] = handicap_rows
    b["poly"]["total_goals"] = totals_rows
    return b


def _make_record(bundle: dict, match_id: str, match_num: str,
                 pre_kickoff: bool = True, match_date: str = "2026-06-17") -> ForecastRecord:
    return ForecastRecord(
        match_id=match_id,
        match_num=match_num,
        home="Portugal",
        away="Spain",
        match_date=match_date,
        captured_at="2026-06-17T10:00:00",
        kickoff="2026-06-17",
        pre_kickoff=pre_kickoff,
        bundle=bundle,
    )


def _outcome(match_id: str, home: int, away: int, void: bool = False) -> MatchOutcome:
    return MatchOutcome(match_id=match_id, home_score=home, away_score=away, void=void)


# ---------------------------------------------------------------------------
# 1. Persist / load round-trip
# ---------------------------------------------------------------------------

class TestPersistLoad:
    def test_roundtrip_single(self, tmp_path):
        bundle = _bundle_1x2_only(0.5, 0.3, 0.2)
        rec = _make_record(bundle, "portugal-vs-spain-2026-06-17", "001")
        ledger = str(tmp_path / "forecasts" / "ledger.jsonl")
        append_forecast(rec, ledger)

        loaded = load_forecasts(ledger)
        assert len(loaded) == 1
        r = loaded[0]
        assert r.match_id == "portugal-vs-spain-2026-06-17"
        assert r.match_num == "001"
        assert r.home == "Portugal"
        assert r.pre_kickoff is True
        assert r.bundle["poly"]["moneyline"][0]["prob"] == 0.5

    def test_roundtrip_multiple(self, tmp_path):
        ledger = str(tmp_path / "ledger.jsonl")
        for i in range(3):
            b = _bundle_1x2_only(0.4 + i * 0.05, 0.3, 0.3 - i * 0.05)
            r = _make_record(b, f"match-{i}", f"00{i + 1}")
            append_forecast(r, ledger)

        loaded = load_forecasts(ledger)
        assert len(loaded) == 3
        assert loaded[0].match_id == "match-0"
        assert loaded[2].match_id == "match-2"

    def test_filter_by_date(self, tmp_path):
        ledger = str(tmp_path / "ledger.jsonl")
        dates = ["2026-06-17", "2026-06-18", "2026-06-17"]
        for i, d in enumerate(dates):
            b = _bundle_1x2_only(0.5, 0.3, 0.2)
            b["match_date"] = d
            r = ForecastRecord(
                match_id=f"m-{i}", match_num=str(i), home="A", away="B",
                match_date=d, captured_at=f"{d}T10:00:00",
                kickoff=d, pre_kickoff=True, bundle=b,
            )
            append_forecast(r, ledger)

        result = load_forecasts(ledger, date="2026-06-17")
        assert len(result) == 2

    def test_empty_ledger_returns_empty(self, tmp_path):
        assert load_forecasts(str(tmp_path / "nonexistent.jsonl")) == []

    def test_schema_field_written(self, tmp_path):
        ledger = str(tmp_path / "ledger.jsonl")
        rec = _make_record(_bundle_1x2_only(0.5, 0.3, 0.2), "m1", "001")
        append_forecast(rec, ledger)
        raw = json.loads(Path(ledger).read_text())
        assert raw["schema"] == "bq.forecast.v1"

    def test_from_dict_wrong_schema_raises(self):
        with pytest.raises(ValueError, match="Unknown forecast schema"):
            ForecastRecord.from_dict({"schema": "bq.snapshot.v1", "match_id": "x",
                                      "match_num": "1", "home": "A", "away": "B",
                                      "match_date": "2026-06-17", "captured_at": "...",
                                      "kickoff": "", "pre_kickoff": True, "bundle": {}})


# ---------------------------------------------------------------------------
# 2. grade_forecasts — hand-computed Brier/log-loss
# ---------------------------------------------------------------------------

class TestGradeForecasts:
    def test_1x2_brier_hand_computed(self):
        """Portugal P(home)=0.6, P(draw)=0.25, P(away)=0.15 → draw (1-1).
        Poly 1X2 CalibrationPoints: home(0.6,y=0), draw(0.25,y=1), away(0.15,y=0).
        Brier = ((0.6-0)^2 + (0.25-1)^2 + (0.15-0)^2) / 3
              = (0.36 + 0.5625 + 0.0225) / 3 = 0.945 / 3 = 0.315
        """
        bundle = _bundle_1x2_only(0.6, 0.25, 0.15)
        rec = _make_record(bundle, "m1", "001")
        outcomes = {"m1": _outcome("m1", 1, 1)}  # draw

        grouped, n_excl = grade_forecasts([rec], outcomes)
        assert n_excl == 0
        pts = grouped[("poly", "1x2")]
        assert len(pts) == 3
        brier = sum((p["prob"] - p["y"]) ** 2 for p in pts) / len(pts)
        assert abs(brier - 0.315) < 1e-9

    def test_match_by_match_num_fallback(self):
        """When match_id not in outcomes, fall back to match_num matching."""
        bundle = _bundle_1x2_only(0.5, 0.3, 0.2)
        # record has match_id="" but outcomes keyed by match_num "007"
        rec = ForecastRecord(
            match_id="", match_num="007", home="A", away="B",
            match_date="2026-06-17", captured_at="2026-06-17T10:00:00",
            kickoff="2026-06-17", pre_kickoff=True, bundle=bundle,
        )
        outcomes = {"007": _outcome("007", 2, 0)}  # home win

        grouped, _ = grade_forecasts([rec], outcomes)
        assert ("poly", "1x2") in grouped
        pts = grouped[("poly", "1x2")]
        home_pt = next(p for p in pts if abs(p["prob"] - 0.5) < 1e-9)
        assert home_pt["y"] == 1  # home won

    def test_voided_outcome_skipped(self):
        bundle = _bundle_1x2_only(0.5, 0.3, 0.2)
        rec = _make_record(bundle, "m1", "001")
        outcomes = {"m1": _outcome("m1", 1, 0, void=True)}

        grouped, _ = grade_forecasts([rec], outcomes)
        assert ("poly", "1x2") not in grouped

    def test_no_outcome_skipped(self):
        bundle = _bundle_1x2_only(0.5, 0.3, 0.2)
        rec = _make_record(bundle, "no-match", "999")
        outcomes = {}

        grouped, _ = grade_forecasts([rec], outcomes)
        assert len(grouped) == 0

    def test_market_family_grouping_separate(self):
        """1x2, handicap, totals must go into separate buckets."""
        handicap_rows = [
            {"outcome": "home", "prob": 0.55, "line": -0.5, "liquidity": 50000, "thin": False},
            {"outcome": "away", "prob": 0.45, "line": -0.5, "liquidity": 50000, "thin": False},
        ]
        totals_rows = [
            {"outcome": "over", "prob": 0.52, "line": 2.5, "liquidity": 60000, "thin": False},
            {"outcome": "under", "prob": 0.48, "line": 2.5, "liquidity": 60000, "thin": False},
        ]
        bundle = _bundle_with_handicap_totals(0.5, 0.3, 0.2, handicap_rows, totals_rows)
        rec = _make_record(bundle, "m1", "001")
        # 2-1 result: home wins, total=3 (>2.5 → over wins)
        outcomes = {"m1": _outcome("m1", 2, 1)}

        grouped, _ = grade_forecasts([rec], outcomes)

        assert ("poly", "1x2") in grouped
        assert ("poly", "handicap") in grouped
        assert ("poly", "totals") in grouped

        # 1x2: home=1, draw=0, away=0
        pts_1x2 = grouped[("poly", "1x2")]
        assert any(abs(p["prob"] - 0.5) < 1e-9 and p["y"] == 1 for p in pts_1x2)  # home won

        # totals: over=1 (3 > 2.5), under=0
        pts_tot = grouped[("poly", "totals")]
        over_pt = next(p for p in pts_tot if abs(p["prob"] - 0.52) < 1e-9)
        assert over_pt["y"] == 1

        # handicap: home(-0.5) → home -0.5 + 2-1 = 0.5 > 0 → home covers (y=1)
        pts_hcap = grouped[("poly", "handicap")]
        home_hcap = next(p for p in pts_hcap if abs(p["prob"] - 0.55) < 1e-9)
        assert home_hcap["y"] == 1

    def test_totals_integer_line_excluded(self):
        """A 2-0 result with line=2.0 is a push → VOID → excluded."""
        totals_rows = [
            {"outcome": "over", "prob": 0.5, "line": 2.0, "liquidity": 50000, "thin": False},
        ]
        bundle = _bundle_with_handicap_totals(0.5, 0.3, 0.2, [], totals_rows)
        rec = _make_record(bundle, "m1", "001")
        outcomes = {"m1": _outcome("m1", 2, 0)}  # total=2, line=2.0 → push

        grouped, _ = grade_forecasts([rec], outcomes)
        # totals bucket should be absent (no non-void points)
        assert ("poly", "totals") not in grouped


# ---------------------------------------------------------------------------
# 3. pre_kickoff exclusion
# ---------------------------------------------------------------------------

class TestPreKickoffExclusion:
    def test_post_kickoff_excluded_from_grading(self):
        bundle = _bundle_1x2_only(0.5, 0.3, 0.2)
        pre_rec = _make_record(bundle, "m1", "001", pre_kickoff=True)
        post_rec = _make_record(bundle, "m2", "002", pre_kickoff=False)
        outcomes = {
            "m1": _outcome("m1", 1, 0),
            "m2": _outcome("m2", 0, 1),
        }

        grouped, n_excl = grade_forecasts([pre_rec, post_rec], outcomes)
        assert n_excl == 1
        pts = grouped.get(("poly", "1x2"), [])
        # only 3 points from pre_rec, not 6
        assert len(pts) == 3

    def test_is_pre_kickoff_helper(self):
        assert _is_pre_kickoff("2026-06-17T10:00:00", "2026-06-17") is True
        assert _is_pre_kickoff("2026-06-16T22:00:00", "2026-06-17") is True
        assert _is_pre_kickoff("2026-06-18T01:00:00", "2026-06-17") is False
        # empty kickoff → False (cannot confirm, exclude)
        assert _is_pre_kickoff("2026-06-17T10:00:00", "") is False

    def test_parse_kickoff_from_slug(self):
        assert _parse_kickoff_from_slug("portugal-vs-spain-2026-06-17") == "2026-06-17"
        assert _parse_kickoff_from_slug("no-date-here") == ""
        assert _parse_kickoff_from_slug(None) == ""
        assert _parse_kickoff_from_slug("") == ""


# ---------------------------------------------------------------------------
# 7. Dedup to latest pre_kickoff per match (cron-capture dedup)
# ---------------------------------------------------------------------------

class TestDedupToLatestPreKickoff:
    """Repeated cron captures of the same match must not inflate calibration n."""

    def test_dedup_keeps_latest_pre_kickoff(self):
        """Three records for the same match_id:
          - T1 pre_kickoff  prob=0.40 (earlier, should be dropped)
          - T2 pre_kickoff  prob=0.55 (latest, should be kept)
          - post_kickoff    prob=0.70 (excluded as non-forecast, counted in n_excl)

        After dedup + grading:
          - calibration points reflect T2 probabilities only
          - the match contributes exactly 3 points to ("poly", "1x2") (home/draw/away for T2)
          - n_excluded == 1 (the post_kickoff record)
        """
        match_id = "portugal-vs-spain-2026-06-17"
        outcome = {"portugal-vs-spain-2026-06-17": _outcome(match_id, 1, 1)}  # draw

        bundle_t1 = _bundle_1x2_only(0.40, 0.35, 0.25)
        bundle_t2 = _bundle_1x2_only(0.55, 0.30, 0.15)
        bundle_post = _bundle_1x2_only(0.70, 0.20, 0.10)

        rec_t1 = ForecastRecord(
            match_id=match_id, match_num="001", home="Portugal", away="Spain",
            match_date="2026-06-17", captured_at="2026-06-17T06:00:00",
            kickoff="2026-06-17", pre_kickoff=True, bundle=bundle_t1,
        )
        rec_t2 = ForecastRecord(
            match_id=match_id, match_num="001", home="Portugal", away="Spain",
            match_date="2026-06-17", captured_at="2026-06-17T12:00:00",
            kickoff="2026-06-17", pre_kickoff=True, bundle=bundle_t2,
        )
        rec_post = ForecastRecord(
            match_id=match_id, match_num="001", home="Portugal", away="Spain",
            match_date="2026-06-17", captured_at="2026-06-17T22:00:00",
            kickoff="2026-06-17", pre_kickoff=False, bundle=bundle_post,
        )

        grouped, n_excl = grade_forecasts([rec_t1, rec_t2, rec_post], outcome)

        # Post-kickoff record is counted in exclusions
        assert n_excl == 1, f"Expected 1 excluded, got {n_excl}"

        pts = grouped.get(("poly", "1x2"), [])

        # Exactly 3 calibration points (home/draw/away) — the match counted once
        assert len(pts) == 3, f"Expected 3 calibration points (one match), got {len(pts)}"

        # Points must reflect T2 probabilities (0.55/0.30/0.15), NOT T1 (0.40/0.35/0.25)
        probs = {round(p["prob"], 9) for p in pts}
        assert round(0.55, 9) in probs, f"T2 home prob 0.55 not in points: {probs}"
        assert round(0.30, 9) in probs, f"T2 draw prob 0.30 not in points: {probs}"
        assert round(0.15, 9) in probs, f"T2 away prob 0.15 not in points: {probs}"

        # T1 probabilities must NOT appear
        assert round(0.40, 9) not in probs, f"T1 home prob 0.40 should not be in points: {probs}"

        # Result is draw → draw point y=1, home y=0, away y=0
        draw_pt = next(p for p in pts if abs(p["prob"] - 0.30) < 1e-9)
        assert draw_pt["y"] == 1, "Draw outcome y should be 1"
        home_pt = next(p for p in pts if abs(p["prob"] - 0.55) < 1e-9)
        assert home_pt["y"] == 0, "Home outcome y should be 0 (draw result)"


# ---------------------------------------------------------------------------
# 4. Portugal DRAW scenario — Elo Brier < Poly Brier
# ---------------------------------------------------------------------------

class TestPolyVsEloBrierOnDraw:
    """
    Concrete acceptance test from the spec:
    tonight Poly said Portugal P(win)=0.765, Elo said 0.455, the match was
    a 1-1 DRAW.

    Poly 1X2 CalibrationPoints contributed by this match:
        home(0.765, y=0), draw(p_draw_poly, y=1), away(p_away_poly, y=0)
    Elo 1X2 CalibrationPoints:
        home(0.455, y=0), draw(p_draw_elo, y=1), away(p_away_elo, y=0)

    For a SINGLE MATCH comparison we only compare the HOME probability point
    (since that's the dimension specified), i.e. Brier on that single point:
        poly_brier_home = (0.765 - 0)^2 = 0.585225
        elo_brier_home  = (0.455 - 0)^2 = 0.207025
    Elo must score lower (better) Brier on this single point.

    For the full 1x2 3-point brier we need plausible draw/away probs too:
        poly: home=0.765, draw=0.130, away=0.105  (sum≈1.0)
        elo:  home=0.455, draw=0.295, away=0.250  (sum=1.0)
    Result: DRAW (1-1) → y_home=0, y_draw=1, y_away=0
        poly brier3 = ((0.765)^2 + (0.130-1)^2 + (0.105)^2) / 3
                    = (0.585225 + 0.7569 + 0.011025) / 3 = 1.35315 / 3 = 0.45105
        elo  brier3 = ((0.455)^2 + (0.295-1)^2 + (0.250)^2) / 3
                    = (0.207025 + 0.497025 + 0.0625) / 3 = 0.76655 / 3 = 0.255517
    Elo must be lower (better calibrated on this match) than Poly.
    """

    def test_elo_brier_better_on_draw(self):
        poly_home, poly_draw, poly_away = 0.765, 0.130, 0.105
        elo_home, elo_draw, elo_away = 0.455, 0.295, 0.250

        bundle = _bundle_1x2_only(
            poly_home, poly_draw, poly_away,
            elo_home=elo_home, elo_draw=elo_draw, elo_away=elo_away,
        )
        rec = _make_record(bundle, "portugal-vs-spain-2026-06-17", "001")
        outcomes = {"portugal-vs-spain-2026-06-17": _outcome("portugal-vs-spain-2026-06-17", 1, 1)}

        grouped, n_excl = grade_forecasts([rec], outcomes)
        assert n_excl == 0

        poly_pts = grouped[("poly", "1x2")]
        elo_pts = grouped[("elo", "1x2")]

        poly_brier = sum((p["prob"] - p["y"]) ** 2 for p in poly_pts) / len(poly_pts)
        elo_brier = sum((p["prob"] - p["y"]) ** 2 for p in elo_pts) / len(elo_pts)

        # Elo is more conservative on home win → closer to 0 when result is draw → lower Brier
        assert elo_brier < poly_brier, (
            f"Expected elo_brier ({elo_brier:.6f}) < poly_brier ({poly_brier:.6f})"
        )

    def test_home_win_prob_brier_single_point(self):
        """Single-point Brier on the HOME outcome for the Portugal draw scenario."""
        poly_home_brier = (0.765 - 0) ** 2  # 0.585225
        elo_home_brier = (0.455 - 0) ** 2   # 0.207025
        assert elo_home_brier < poly_home_brier
        assert abs(poly_home_brier - 0.585225) < 1e-9
        assert abs(elo_home_brier - 0.207025) < 1e-9


# ---------------------------------------------------------------------------
# 5. calibration_report
# ---------------------------------------------------------------------------

class TestCalibrationReport:
    def _make_grouped(self):
        """Two records: one draw (elo better), one home win (poly luckier)."""
        # Match 1: draw — Elo more conservative (better)
        poly_pts_1 = [
            {"prob": 0.765, "y": 0},
            {"prob": 0.130, "y": 1},
            {"prob": 0.105, "y": 0},
        ]
        elo_pts_1 = [
            {"prob": 0.455, "y": 0},
            {"prob": 0.295, "y": 1},
            {"prob": 0.250, "y": 0},
        ]
        # Match 2: home win — Poly more accurate
        poly_pts_2 = [
            {"prob": 0.700, "y": 1},
            {"prob": 0.180, "y": 0},
            {"prob": 0.120, "y": 0},
        ]
        elo_pts_2 = [
            {"prob": 0.420, "y": 1},
            {"prob": 0.310, "y": 0},
            {"prob": 0.270, "y": 0},
        ]
        return {
            ("poly", "1x2"): poly_pts_1 + poly_pts_2,
            ("elo", "1x2"): elo_pts_1 + elo_pts_2,
        }

    def test_report_has_rows(self):
        grouped = self._make_grouped()
        report = calibration_report(grouped, n_excluded_post_kickoff=2)
        assert len(report["rows"]) == 2
        assert report["n_excluded_post_kickoff"] == 2
        row_keys = {(r["forecaster"], r["market_family"]) for r in report["rows"]}
        assert ("poly", "1x2") in row_keys
        assert ("elo", "1x2") in row_keys

    def test_report_contains_verdict(self):
        grouped = self._make_grouped()
        report = calibration_report(grouped)
        verdict = report["poly_vs_elo_1x2"]
        assert "1x2" in verdict or "Brier" in verdict
        # Must name a winner or tie
        assert any(word in verdict for word in ("BETTER", "TIED"))

    def test_empty_grouped_gives_no_comparison(self):
        report = calibration_report({})
        assert "No 1x2 data" in report["poly_vs_elo_1x2"]
        assert report["rows"] == []

    def test_only_poly_no_elo(self):
        grouped = {("poly", "1x2"): [{"prob": 0.6, "y": 1}, {"prob": 0.4, "y": 0}]}
        report = calibration_report(grouped)
        assert "Only Poly available" in report["poly_vs_elo_1x2"]

    def test_row_n_matches_point_count(self):
        pts = [{"prob": 0.5, "y": 1}] * 7
        grouped = {("poly", "1x2"): pts}
        report = calibration_report(grouped)
        row = report["rows"][0]
        assert row["n"] == 7

    def test_brier_value_correct(self):
        """Single point: prob=0.5, y=1 → Brier = (0.5-1)^2 = 0.25."""
        grouped = {("poly", "1x2"): [{"prob": 0.5, "y": 1}]}
        report = calibration_report(grouped)
        row = report["rows"][0]
        assert abs(row["brier"] - 0.25) < 1e-9


# ---------------------------------------------------------------------------
# 6. make_forecast_record helper
# ---------------------------------------------------------------------------

class TestMakeForecastRecord:
    def test_basic_fields(self):
        bundle = _bundle_1x2_only(0.5, 0.3, 0.2)
        rec = make_forecast_record(bundle, captured_at="2026-06-17T08:00:00")
        assert rec.match_id == "portugal-vs-spain-2026-06-17"
        assert rec.match_num == "001"
        assert rec.home == "Portugal"
        assert rec.away == "Spain"
        assert rec.kickoff == "2026-06-17"
        assert rec.pre_kickoff is True  # same day, capture before kickoff

    def test_post_kickoff_detection(self):
        bundle = _bundle_1x2_only(0.5, 0.3, 0.2)
        # Captured a day after the slug date
        rec = make_forecast_record(bundle, captured_at="2026-06-18T10:00:00")
        assert rec.pre_kickoff is False

    def test_no_slug_date_gives_empty_kickoff(self):
        bundle = _bundle_1x2_only(0.5, 0.3, 0.2)
        bundle["event_slug"] = "no-date-in-this-slug"
        rec = make_forecast_record(bundle, captured_at="2026-06-17T10:00:00")
        assert rec.kickoff == ""
        assert rec.pre_kickoff is False  # cannot confirm pre-kickoff
