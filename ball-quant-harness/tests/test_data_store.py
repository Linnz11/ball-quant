"""
Phase 1A — lossless snapshot store + capture tests.

Core guarantee: every MarketQuote field must survive a full write → read →
reconstruct round-trip unchanged.  A lossy store silently breaks the optimizer.
"""
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from ball_quant.data import capture, store
from ball_quant.models import EventMarketMatrix, MarketQuote, MatchSP


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_matrix() -> EventMarketMatrix:
    """Two quotes with non-trivial values across every field."""
    q1 = MarketQuote(
        market_id="mkt-001",
        question="Netherlands vs Japan — 1X2",
        category="moneyline",
        outcome="Netherlands",
        probability=0.4825,
        token_id="tok-abc",
        bid=0.47,
        ask=0.495,
        spread=0.025,
        liquidity=198500.0,
        volume=3_200_000.0,
        sports_type="soccer",
        line=None,
        period="FT",
        side="home",
        entity="Netherlands",
        scope="match",
        horizon="90min",
        causal_layer="market",
        model_weight=0.72,
        is_complement=False,
        active=True,
        closed=False,
        accepting_orders=True,
        raw={"extra": "data", "nested": {"v": 1}},
    )
    q2 = MarketQuote(
        market_id="mkt-002",
        question="Netherlands vs Japan — Spread -1.5",
        category="spreads",
        outcome="Japan",
        probability=0.77,
        token_id="tok-xyz",
        bid=0.76,
        ask=0.78,
        spread=0.02,
        liquidity=49_000.0,
        volume=880_000.0,
        sports_type="soccer",
        line=-1.5,
        period="FT",
        side="away",
        entity="Japan",
        scope="match",
        horizon="90min",
        causal_layer="market",
        model_weight=0.55,
        is_complement=True,
        active=True,
        closed=False,
        accepting_orders=True,
        raw={"raw_field": True},
    )
    return EventMarketMatrix(
        match_id="351724",
        home="Netherlands",
        away="Japan",
        event_id="poly-evt-9",
        event_slug="fifwc-nld-jpn-2026-06-14",
        markets=[q1, q2],
        raw_event={"startTime": "2026-06-14T20:00:00Z", "active": True},
    )


def _make_sp() -> MatchSP:
    return MatchSP(
        match_id="351724",
        date="2026-06-14",
        home="Netherlands",
        away="Japan",
        spf_home=1.55,
        spf_draw=3.90,
        spf_away=5.60,
        handicap=-1,
        rq_home=2.78,
        rq_draw=3.55,
        rq_away=2.05,
    )


_TS = datetime(2026, 6, 14, 18, 0, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Round-trip fidelity
# ---------------------------------------------------------------------------

class TestRoundTrip(unittest.TestCase):
    """Every MarketQuote field must survive write → read → reconstruct intact."""

    def test_all_market_quote_fields_survive(self):
        matrix = _make_matrix()
        sp = _make_sp()

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snap_path = capture.capture_snapshot(matrix, sp, root=root, captured_at=_TS)

            record = store.read_snapshot(snap_path)
            rebuilt = store.reconstruct_matrix(record)

        self.assertEqual(rebuilt.match_id, matrix.match_id)
        self.assertEqual(rebuilt.home, matrix.home)
        self.assertEqual(rebuilt.away, matrix.away)
        self.assertEqual(rebuilt.event_id, matrix.event_id)
        self.assertEqual(rebuilt.event_slug, matrix.event_slug)
        self.assertEqual(len(rebuilt.markets), len(matrix.markets))

        for original, recovered in zip(matrix.markets, rebuilt.markets):
            # Assert every declared field so a future model addition alerts here.
            self.assertEqual(recovered.market_id, original.market_id, "market_id")
            self.assertEqual(recovered.question, original.question, "question")
            self.assertEqual(recovered.category, original.category, "category")
            self.assertEqual(recovered.outcome, original.outcome, "outcome")
            self.assertAlmostEqual(recovered.probability, original.probability, places=9, msg="probability")
            self.assertEqual(recovered.token_id, original.token_id, "token_id")
            self.assertAlmostEqual(recovered.bid, original.bid, places=9, msg="bid")
            self.assertAlmostEqual(recovered.ask, original.ask, places=9, msg="ask")
            self.assertAlmostEqual(recovered.spread, original.spread, places=9, msg="spread")
            self.assertAlmostEqual(recovered.liquidity, original.liquidity, places=9, msg="liquidity")
            self.assertAlmostEqual(recovered.volume, original.volume, places=9, msg="volume")
            self.assertEqual(recovered.sports_type, original.sports_type, "sports_type")
            self.assertEqual(recovered.line, original.line, "line")
            self.assertEqual(recovered.period, original.period, "period")
            self.assertEqual(recovered.side, original.side, "side")
            self.assertEqual(recovered.entity, original.entity, "entity")
            self.assertEqual(recovered.scope, original.scope, "scope")
            self.assertEqual(recovered.horizon, original.horizon, "horizon")
            self.assertEqual(recovered.causal_layer, original.causal_layer, "causal_layer")
            self.assertAlmostEqual(recovered.model_weight, original.model_weight, places=9, msg="model_weight")
            self.assertEqual(recovered.is_complement, original.is_complement, "is_complement")
            self.assertEqual(recovered.active, original.active, "active")
            self.assertEqual(recovered.closed, original.closed, "closed")
            self.assertEqual(recovered.accepting_orders, original.accepting_orders, "accepting_orders")
            self.assertEqual(recovered.raw, original.raw, "raw")

    def test_raw_event_survives(self):
        matrix = _make_matrix()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snap_path = capture.capture_snapshot(matrix, None, root=root, captured_at=_TS)
            record = store.read_snapshot(snap_path)
            rebuilt = store.reconstruct_matrix(record)
        self.assertEqual(rebuilt.raw_event, matrix.raw_event)


# ---------------------------------------------------------------------------
# SP block
# ---------------------------------------------------------------------------

class TestSPBlock(unittest.TestCase):

    def test_sp_round_trips(self):
        matrix = _make_matrix()
        sp = _make_sp()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snap_path = capture.capture_snapshot(matrix, sp, root=root, captured_at=_TS)
            record = store.read_snapshot(snap_path)
            recovered_sp = store.reconstruct_match_sp(record)

        self.assertIsNotNone(recovered_sp)
        self.assertAlmostEqual(recovered_sp.spf_home, sp.spf_home, places=9)
        self.assertAlmostEqual(recovered_sp.spf_draw, sp.spf_draw, places=9)
        self.assertAlmostEqual(recovered_sp.spf_away, sp.spf_away, places=9)
        self.assertEqual(recovered_sp.handicap, sp.handicap)
        self.assertAlmostEqual(recovered_sp.rq_home, sp.rq_home, places=9)
        self.assertAlmostEqual(recovered_sp.rq_draw, sp.rq_draw, places=9)
        self.assertAlmostEqual(recovered_sp.rq_away, sp.rq_away, places=9)

    def test_sp_none_yields_null(self):
        matrix = _make_matrix()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snap_path = capture.capture_snapshot(matrix, None, root=root, captured_at=_TS)
            record = store.read_snapshot(snap_path)

        self.assertIsNone(record["sp"])
        self.assertIsNone(store.reconstruct_match_sp(record))


# ---------------------------------------------------------------------------
# Manifest + list_snapshots
# ---------------------------------------------------------------------------

class TestManifestAndListing(unittest.TestCase):

    def _write_two(self, root: Path):
        m1 = _make_matrix()
        ts1 = datetime(2026, 6, 14, 10, 0, 0, tzinfo=timezone.utc)

        m2 = EventMarketMatrix(
            match_id="999888",
            home="Brazil",
            away="Argentina",
            markets=[],
        )
        ts2 = datetime(2026, 6, 14, 15, 0, 0, tzinfo=timezone.utc)

        capture.capture_snapshot(m1, None, root=root, captured_at=ts1)
        capture.capture_snapshot(m2, None, root=root, captured_at=ts2)
        return ts1, ts2

    def test_list_all(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_two(root)
            entries = store.list_snapshots(root)
        self.assertEqual(len(entries), 2)

    def test_filter_by_match_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_two(root)
            entries = store.list_snapshots(root, match_id="351724")
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["match_id"], "351724")

    def test_filter_since(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_two(root)
            # since=12:00 UTC should exclude the 10:00 snapshot
            entries = store.list_snapshots(root, since="2026-06-14T12:00:00+00:00")
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["match_id"], "999888")

    def test_filter_until(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_two(root)
            # until=12:00 UTC should exclude the 15:00 snapshot
            entries = store.list_snapshots(root, until="2026-06-14T12:00:00+00:00")
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["match_id"], "351724")

    def test_empty_root_returns_empty_list(self):
        with tempfile.TemporaryDirectory() as tmp:
            entries = store.list_snapshots(Path(tmp))
        self.assertEqual(entries, [])


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------

class TestValidation(unittest.TestCase):

    def test_wrong_schema_raises(self):
        record = {
            "schema": "wrong.schema",
            "captured_at": "2026-06-14T10:00:00+00:00",
            "match_id": "001",
            "home": "A",
            "away": "B",
            "matrix": {},
        }
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                store.write_snapshot(record, root=Path(tmp))

    def test_missing_key_raises(self):
        record = {
            "schema": "bq.snapshot.v1",
            # missing captured_at, match_id, home, away, matrix
        }
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                store.write_snapshot(record, root=Path(tmp))

    def test_naive_datetime_raises(self):
        matrix = _make_matrix()
        naive = datetime(2026, 6, 14, 10, 0, 0)  # no tzinfo
        with self.assertRaises(ValueError):
            capture.build_snapshot_record(matrix, None, naive)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class TestHelpers(unittest.TestCase):

    def test_snapshot_id_deterministic(self):
        sid1 = store.snapshot_id("match-123", "2026-06-14T10:00:00+00:00")
        sid2 = store.snapshot_id("match-123", "2026-06-14T10:00:00+00:00")
        self.assertEqual(sid1, sid2)

    def test_snapshot_id_filesystem_safe(self):
        sid = store.snapshot_id("a/b:c?d", "2026-06-14T10:00:00+00:00")
        for bad_char in r'\/:*?"<>|':
            self.assertNotIn(bad_char, sid, f"unsafe char {bad_char!r} in id")

    def test_cache_key_stable(self):
        k1 = store.cache_key("match-1", "moneyline", "Netherlands")
        k2 = store.cache_key("match-1", "moneyline", "Netherlands")
        self.assertEqual(k1, k2)
        self.assertEqual(len(k1), 40)  # SHA-1 hex is always 40 chars

    def test_is_fresh_nonexistent(self):
        self.assertFalse(store.is_fresh(Path("/nonexistent/path.json"), 60))

    def test_competition_stored(self):
        matrix = _make_matrix()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snap_path = capture.capture_snapshot(
                matrix, None, root=root, captured_at=_TS, competition="FIFA WC 2026"
            )
            record = store.read_snapshot(snap_path)
        self.assertEqual(record["competition"], "FIFA WC 2026")


if __name__ == "__main__":
    unittest.main()
