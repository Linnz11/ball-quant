"""
Tests for ParamProfiles: resolver, serialisation, engine integration,
and optimize_by_competition.

Coverage:
  1. resolve: no profiles -> DEFAULT_PARAMS; default_overrides applied;
     by_competition overrides layer on top; unknown key raises;
     round-trip to_json / from_json.
  2. run_backtest with profiles: two competitions resolve to different params;
     profiles=None reproduces single-params behaviour.
  3. optimize_by_competition: returns by_competition best_overrides + records
     skipped competitions; deterministic under seed.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from tests.test_probability_and_combo import sample_match, sample_matrix
from tests.test_optimize import _value_gap_match, _value_gap_matrix

from ball_quant.backtest.engine import run_backtest
from ball_quant.backtest.optimize import optimize_by_competition
from ball_quant.core.params import DEFAULT_PARAMS, StrategyParams
from ball_quant.core.profiles import ParamProfiles
from ball_quant.core.settlement import MatchOutcome
from ball_quant.data.capture import capture_snapshot
from ball_quant.data.store import read_snapshot
from ball_quant.models import EventMarketMatrix, MarketQuote, MatchSP


# ---------------------------------------------------------------------------
# Section 1: ParamProfiles.resolve
# ---------------------------------------------------------------------------

class TestParamProfilesResolve(unittest.TestCase):

    def test_empty_profiles_returns_default_params(self):
        """Empty ParamProfiles must yield DEFAULT_PARAMS — zero behaviour change."""
        profiles = ParamProfiles()
        resolved = profiles.resolve(None)
        self.assertEqual(resolved, DEFAULT_PARAMS)

    def test_empty_profiles_with_competition_returns_default_params(self):
        profiles = ParamProfiles()
        resolved = profiles.resolve("Premier League")
        self.assertEqual(resolved, DEFAULT_PARAMS)

    def test_default_overrides_applied(self):
        """default_overrides are applied on top of DEFAULT_PARAMS."""
        profiles = ParamProfiles(default_overrides={"fractional_kelly": 0.10})
        resolved = profiles.resolve(None)
        self.assertAlmostEqual(resolved.fractional_kelly, 0.10)
        # All other fields still match DEFAULT_PARAMS.
        self.assertEqual(resolved.max_goals, DEFAULT_PARAMS.max_goals)

    def test_competition_overrides_layer_on_default_overrides(self):
        """by_competition overrides layer on top of default_overrides."""
        profiles = ParamProfiles(
            default_overrides={"fractional_kelly": 0.10},
            by_competition={"PL": {"fractional_kelly": 0.20, "conf_base": 0.60}},
        )
        # Unknown competition falls back to default-overrides base.
        base = profiles.resolve("La Liga")
        self.assertAlmostEqual(base.fractional_kelly, 0.10)

        # Known competition applies its overrides on top of default_overrides base.
        pl = profiles.resolve("PL")
        self.assertAlmostEqual(pl.fractional_kelly, 0.20)
        self.assertAlmostEqual(pl.conf_base, 0.60)

    def test_unknown_override_key_raises(self):
        """Unknown field in default_overrides must raise ValueError immediately."""
        with self.assertRaises(ValueError):
            ParamProfiles(default_overrides={"nonexistent_field": 0.5}).resolve(None)

    def test_unknown_competition_override_key_raises_on_resolve(self):
        """Unknown field in by_competition entry must raise on resolve."""
        profiles = ParamProfiles(by_competition={"PL": {"bad_key": 99}})
        with self.assertRaises(ValueError):
            profiles.resolve("PL")

    def test_resolve_none_equals_resolve_unknown_competition(self):
        """resolve(None) and resolve('unknown') must return the same object value."""
        profiles = ParamProfiles(default_overrides={"fractional_kelly": 0.15})
        self.assertEqual(profiles.resolve(None), profiles.resolve("unknown_comp"))

    def test_new_fields_handled_generically(self):
        """Override the newly-added fields without hardcoding them."""
        overrides = {
            "dixon_coles_rho": -0.1,
            "devig_method": "shin",
            "weight_scheme": "inverse_variance",
        }
        profiles = ParamProfiles(default_overrides=overrides)
        resolved = profiles.resolve(None)
        self.assertAlmostEqual(resolved.dixon_coles_rho, -0.1)
        self.assertEqual(resolved.devig_method, "shin")
        self.assertEqual(resolved.weight_scheme, "inverse_variance")


# ---------------------------------------------------------------------------
# Section 1b: JSON round-trip
# ---------------------------------------------------------------------------

class TestParamProfilesJSON(unittest.TestCase):

    def test_round_trip_empty(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "profiles.json"
            orig = ParamProfiles()
            orig.to_json(p)
            loaded = ParamProfiles.from_json(p)
            self.assertEqual(loaded.default_overrides, {})
            self.assertEqual(loaded.by_competition, {})
            self.assertEqual(loaded.resolve(None), DEFAULT_PARAMS)

    def test_round_trip_with_values(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "profiles.json"
            orig = ParamProfiles(
                default_overrides={"fractional_kelly": 0.15},
                by_competition={"PL": {"conf_base": 0.60}},
            )
            orig.to_json(p)
            loaded = ParamProfiles.from_json(p)
            self.assertAlmostEqual(loaded.resolve(None).fractional_kelly, 0.15)
            self.assertAlmostEqual(loaded.resolve("PL").conf_base, 0.60)
            self.assertAlmostEqual(loaded.resolve("PL").fractional_kelly, 0.15)

    def test_json_schema_keys(self):
        """Emitted JSON must have 'default' and 'by_competition' top-level keys."""
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "profiles.json"
            ParamProfiles(
                default_overrides={"conf_base": 0.55},
                by_competition={"PL": {"conf_base": 0.60}},
            ).to_json(p)
            raw = json.loads(p.read_text())
            self.assertIn("default", raw)
            self.assertIn("by_competition", raw)

    def test_from_json_raises_on_unknown_key(self):
        """from_json must raise on unknown override keys at load time."""
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "bad.json"
            p.write_text(json.dumps({"default": {"bad_key": 1}, "by_competition": {}}))
            with self.assertRaises(ValueError):
                ParamProfiles.from_json(p)


# ---------------------------------------------------------------------------
# Section 2: run_backtest with profiles
# ---------------------------------------------------------------------------

def _capture_record(root: Path, match_id: str, ts: datetime, competition: str) -> dict:
    """Capture a value-gap snapshot and return the loaded record dict."""
    snap_path = capture_snapshot(
        matrix=_value_gap_matrix(match_id),
        match_sp=_value_gap_match(match_id),
        root=root,
        captured_at=ts,
        competition=competition,
    )
    return read_snapshot(snap_path)


class TestRunBacktestWithProfiles(unittest.TestCase):

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self._tmpdir.name) / "store"

        # Two records with different competitions.
        self.record_a = _capture_record(
            self.root, "p001",
            datetime(2026, 6, 10, 10, 0, 0, tzinfo=timezone.utc),
            "CompA",
        )
        self.record_b = _capture_record(
            self.root, "p002",
            datetime(2026, 6, 11, 10, 0, 0, tzinfo=timezone.utc),
            "CompB",
        )
        self.outcomes = {
            "p001": MatchOutcome(match_id="p001", home_score=2, away_score=0),
            "p002": MatchOutcome(match_id="p002", home_score=2, away_score=0),
        }

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_profiles_none_matches_single_params_result(self):
        """profiles=None path must be byte-identical to the single params path."""
        records = [self.record_a, self.record_b]
        result_old = run_backtest(records, self.outcomes, params=DEFAULT_PARAMS,
                                  budget=500.0, bankroll=5000.0)
        result_new = run_backtest(records, self.outcomes, params=DEFAULT_PARAMS,
                                  budget=500.0, bankroll=5000.0, profiles=None)
        self.assertEqual(result_old["n_graded_matches"], result_new["n_graded_matches"])
        self.assertEqual(result_old["n_bets"], result_new["n_bets"])
        # Brier score must be identical.
        brier_old = result_old["metrics"]["calibration"]["brier"]
        brier_new = result_new["metrics"]["calibration"]["brier"]
        self.assertAlmostEqual(brier_old, brier_new, places=10)

    def test_competition_specific_params_resolve_differently(self):
        """Two competitions with different fractional_kelly produce different bets/PnL."""
        # CompA: low kelly -> stake is smaller.
        # CompB: high kelly -> stake is larger.
        # We confirm this by comparing per-competition PnL via separate run_backtest
        # calls and asserting they differ when different params are applied.
        profiles_uniform = ParamProfiles()  # both competitions see DEFAULT_PARAMS
        profiles_varied = ParamProfiles(
            by_competition={
                "CompA": {"fractional_kelly": 0.05},   # very small stake
                "CompB": {"fractional_kelly": 0.50},   # large stake
            }
        )

        # Separate runs for each record to isolate competition effects.
        result_a_uniform = run_backtest(
            [self.record_a], self.outcomes, profiles=profiles_uniform,
            budget=500.0, bankroll=5000.0,
        )
        result_a_varied = run_backtest(
            [self.record_a], self.outcomes, profiles=profiles_varied,
            budget=500.0, bankroll=5000.0,
        )
        pnl_uniform = result_a_uniform["metrics"].get("pnl", {}).get("net_pnl")
        pnl_varied = result_a_varied["metrics"].get("pnl", {}).get("net_pnl")

        # Both should have bets (value-gap fixture).
        self.assertGreaterEqual(result_a_uniform["n_bets"], 1)
        self.assertGreaterEqual(result_a_varied["n_bets"], 1)

        # Different kelly fractions -> different stake sizes -> different PnL.
        self.assertNotAlmostEqual(
            pnl_uniform, pnl_varied, places=4,
            msg="Different fractional_kelly must produce different PnL",
        )

    def test_profiles_competition_resolution_correct_per_record(self):
        """resolve() is called with the competition from each record."""
        profiles = ParamProfiles(
            by_competition={
                "CompA": {"fractional_kelly": 0.50},
                "CompB": {"fractional_kelly": 0.05},
            }
        )
        result_a = run_backtest([self.record_a], self.outcomes, profiles=profiles,
                                budget=500.0, bankroll=5000.0)
        result_b = run_backtest([self.record_b], self.outcomes, profiles=profiles,
                                budget=500.0, bankroll=5000.0)

        pnl_a = result_a["metrics"].get("pnl", {}).get("net_pnl", 0.0)
        pnl_b = result_b["metrics"].get("pnl", {}).get("net_pnl", 0.0)
        # CompA has 10× the kelly fraction of CompB, so its stakes (and PnL) are larger.
        self.assertGreater(abs(pnl_a), abs(pnl_b),
                           msg="CompA (high kelly) must produce larger |PnL| than CompB (low kelly)")

    def test_empty_profiles_same_as_no_profiles(self):
        """Empty ParamProfiles must give byte-identical results to profiles=None."""
        records = [self.record_a, self.record_b]
        result_none = run_backtest(records, self.outcomes, profiles=None,
                                   budget=500.0, bankroll=5000.0)
        result_empty = run_backtest(records, self.outcomes, profiles=ParamProfiles(),
                                    budget=500.0, bankroll=5000.0)
        brier_none = result_none["metrics"]["calibration"]["brier"]
        brier_empty = result_empty["metrics"]["calibration"]["brier"]
        self.assertAlmostEqual(brier_none, brier_empty, places=10)


# ---------------------------------------------------------------------------
# Section 3: optimize_by_competition
# ---------------------------------------------------------------------------

def _build_multi_comp_records(root: Path) -> tuple:
    """
    Build a record set with two competitions and one "__none__"-competition group.

    CompA: 5 records (enough for n_folds=3 with min_records=4).
    CompB: 2 records (below threshold -> skipped).
    None:  3 records (tagged as __none__ internally, below threshold -> skipped).
    Overall: 10 records (enough for overall optimize with n_folds=3).
    """
    records = []
    outcomes = {}
    idx = 0

    def _add(comp, n):
        nonlocal idx
        for i in range(n):
            mid = f"obc{idx:03d}"
            idx += 1
            day = 1 + idx  # unique day offset
            ts = datetime(2026, 6, 1 + (idx % 28), 10, 0, 0, tzinfo=timezone.utc)
            snap_path = capture_snapshot(
                matrix=_value_gap_matrix(mid),
                match_sp=_value_gap_match(mid),
                root=root,
                captured_at=ts,
                competition=comp,
            )
            record = read_snapshot(snap_path)
            records.append(record)
            outcomes[mid] = MatchOutcome(match_id=mid, home_score=2, away_score=0)

    _add("CompA", 5)
    _add("CompB", 2)
    _add(None, 3)

    return records, outcomes


class TestOptimizeByCompetition(unittest.TestCase):

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self._tmpdir.name) / "store"
        self.records, self.outcomes = _build_multi_comp_records(self.root)
        self.param_space = {"fractional_kelly": [0.20, 0.25]}

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_return_shape(self):
        """Result must have the four required top-level keys."""
        result = optimize_by_competition(
            records=self.records,
            outcomes=self.outcomes,
            param_space=self.param_space,
            metric="brier",
            n_folds=3,
            seed=42,
        )
        for key in ("default", "by_competition", "per_competition_detail", "skipped"):
            self.assertIn(key, result, f"Missing key: {key}")

    def test_default_is_dict(self):
        result = optimize_by_competition(
            records=self.records, outcomes=self.outcomes,
            param_space=self.param_space, metric="brier", n_folds=3, seed=42,
        )
        self.assertIsInstance(result["default"], dict)

    def test_compa_optimized_compb_skipped(self):
        """CompA (5 records >= n_folds+1=4) is optimized; CompB (2) is skipped."""
        result = optimize_by_competition(
            records=self.records, outcomes=self.outcomes,
            param_space=self.param_space, metric="brier", n_folds=3, seed=42,
        )
        self.assertIn("CompA", result["by_competition"],
                      "CompA has enough records and must be in by_competition")
        self.assertIn("CompB", result["skipped"],
                      "CompB has too few records and must be in skipped")
        # None-competition group also has too few records.
        self.assertIn("__none__", result["skipped"])

    def test_skipped_reason_is_string(self):
        """Skipped entry must be a non-empty string explaining why."""
        result = optimize_by_competition(
            records=self.records, outcomes=self.outcomes,
            param_space=self.param_space, metric="brier", n_folds=3, seed=42,
        )
        for comp, reason in result["skipped"].items():
            self.assertIsInstance(reason, str)
            self.assertGreater(len(reason), 0)

    def test_deterministic_under_seed(self):
        """Two calls with the same seed must return the same result."""
        kwargs = dict(
            records=self.records, outcomes=self.outcomes,
            param_space=self.param_space, metric="brier", n_folds=3, seed=99,
        )
        r1 = optimize_by_competition(**kwargs)
        r2 = optimize_by_competition(**kwargs)
        self.assertEqual(r1["default"], r2["default"])
        self.assertEqual(r1["by_competition"], r2["by_competition"])
        self.assertEqual(r1["skipped"], r2["skipped"])

    def test_by_competition_overrides_are_valid_param_keys(self):
        """All keys in by_competition overrides must be valid StrategyParams fields."""
        result = optimize_by_competition(
            records=self.records, outcomes=self.outcomes,
            param_space=self.param_space, metric="brier", n_folds=3, seed=42,
        )
        valid_keys = set(DEFAULT_PARAMS.to_dict().keys())
        for comp, overrides in result["by_competition"].items():
            for k in overrides:
                self.assertIn(k, valid_keys, f"Unknown key {k!r} in by_competition[{comp!r}]")

    def test_profiles_roundtrip_from_result(self):
        """Result maps cleanly into a ParamProfiles that serialises/deserialises."""
        result = optimize_by_competition(
            records=self.records, outcomes=self.outcomes,
            param_space=self.param_space, metric="brier", n_folds=3, seed=42,
        )
        profiles = ParamProfiles(
            default_overrides=result["default"],
            by_competition=result["by_competition"],
        )
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "profiles.json"
            profiles.to_json(p)
            loaded = ParamProfiles.from_json(p)
        self.assertEqual(loaded.default_overrides, profiles.default_overrides)
        self.assertEqual(loaded.by_competition, profiles.by_competition)


if __name__ == "__main__":
    unittest.main()
