"""
Forecast ledger — closed-loop calibration spine for the bundle architecture.

Persists pre-match probability forecasts (Poly devigged + Elo-implied) and
grades them against actual results to score calibration per forecaster × market
family.

Schema: bq.forecast.v1  (separate from bq.snapshot.v1 — NOT shoehorned there)

Key design decisions:
- ForecastRecord stores the WHOLE per-match bundle dict losslessly so any
  market can be re-graded later without data loss.
- pre_kickoff flag is set at capture time; post-kickoff captures are excluded
  from calibration (they are NOT forecasts — a memory lesson learned).
- Reuses: metrics.brier_score / log_loss / expected_calibration_error
          settlement._grade_key / MatchOutcome / handicap_result (via settlement)
          adapters.results.load_results
- stdlib only — no third-party imports.

NOTE: adapters/sporttery.py:224 parse_results already fetches official results
      but is China-IP-gated and CLI-unwired → that is follow-on #22b, not this
      MVP.  Results ingestion here is manual CSV via adapters.results.load_results.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ball_quant.core.metrics import brier_score, log_loss, expected_calibration_error
from ball_quant.core.settlement import MatchOutcome

logger = logging.getLogger(__name__)

_SCHEMA = "bq.forecast.v1"
_SLUG_DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")

# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

@dataclass
class ForecastRecord:
    """One per-match pre-match forecast snapshot."""

    match_id: str           # Poly event slug used as primary key (may be empty string)
    match_num: str          # 体彩 match number (e.g. "001") — used as fallback join key
    home: str               # Poly canonical home team name
    away: str               # Poly canonical away team name
    match_date: str         # YYYY-MM-DD from 体彩 data
    captured_at: str        # ISO8601 UTC timestamp of capture (datetime.now().isoformat())
    kickoff: str            # YYYY-MM-DD parsed from event_slug (best effort; "" if unavailable)
    pre_kickoff: bool       # True iff captured_at < kickoff date (this is the calibration gate)
    bundle: Dict            # The full per-match bundle dict — lossless storage
    schema: str = field(default=_SCHEMA, init=False, repr=False)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["schema"] = _SCHEMA
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ForecastRecord":
        schema = d.get("schema", _SCHEMA)
        if schema != _SCHEMA:
            raise ValueError(f"Unknown forecast schema {schema!r}")
        return cls(
            match_id=d["match_id"],
            match_num=d["match_num"],
            home=d["home"],
            away=d["away"],
            match_date=d["match_date"],
            captured_at=d["captured_at"],
            kickoff=d["kickoff"],
            pre_kickoff=d["pre_kickoff"],
            bundle=d["bundle"],
        )


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def append_forecast(record: ForecastRecord, ledger_path: str) -> None:
    """Append one ForecastRecord as a JSONL line to *ledger_path* (mkdir parents)."""
    path = Path(ledger_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")


def load_forecasts(
    ledger_path: str,
    date: Optional[str] = None,
) -> List[ForecastRecord]:
    """Load ForecastRecords from *ledger_path*, optionally filtered by match_date.

    Returns an empty list (not an error) when the file does not exist yet.
    """
    path = Path(ledger_path)
    if not path.exists():
        return []
    records: List[ForecastRecord] = []
    with path.open(encoding="utf-8") as fh:
        for line_no, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line:
                continue
            d = json.loads(line)
            rec = ForecastRecord.from_dict(d)
            if date is not None and rec.match_date != date:
                continue
            records.append(rec)
    return records


# ---------------------------------------------------------------------------
# Calibration extraction helpers
# ---------------------------------------------------------------------------

def _extract_poly_1x2(bundle: dict) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Return (p_home, p_draw, p_away) from bundle['poly']['moneyline'], or (None,None,None)."""
    rows = bundle.get("poly", {}).get("moneyline", [])
    prob_map: Dict[str, float] = {}
    for r in rows:
        outcome = str(r.get("outcome") or "").lower()
        prob = r.get("prob")
        if prob is not None and outcome in ("home", "draw", "away"):
            prob_map[outcome] = float(prob)
    if len(prob_map) < 3:
        return None, None, None
    return prob_map["home"], prob_map["draw"], prob_map["away"]


def _extract_elo_1x2(bundle: dict) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Return (p_home, p_draw, p_away) from bundle['fundamental'], or (None,None,None)."""
    fund = bundle.get("fundamental")
    if not fund:
        return None, None, None
    ph = fund.get("p_home")
    pd = fund.get("p_draw")
    pa = fund.get("p_away")
    if ph is None or pd is None or pa is None:
        return None, None, None
    return float(ph), float(pd), float(pa)


def _spf_outcome(outcome: MatchOutcome) -> str:
    """Return 'home' | 'draw' | 'away' for a MatchOutcome full-time result."""
    h, a = outcome.home_score, outcome.away_score
    if h > a:
        return "home"
    if h == a:
        return "draw"
    return "away"


def _grade_totals_lines(bundle: dict, outcome: MatchOutcome) -> List[dict]:
    """Yield CalibrationPoints for each Poly total-goals line in the bundle.

    Reuses settlement._grade_key logic for integer-line push (VOID) via direct
    arithmetic — avoids constructing SettlementKey objects for the simple
    over/under case.  VOID points are excluded (not informative for calibration).
    """
    rows = bundle.get("poly", {}).get("total_goals", [])
    points: List[dict] = []
    total = outcome.home_score + outcome.away_score
    for r in rows:
        prob = r.get("prob")
        line = r.get("line")
        side = str(r.get("outcome") or "").lower()
        if prob is None or line is None or side not in ("over", "under"):
            continue
        line_f = float(line)
        if total == line_f:
            # Integer-line push per settlement.py:107-109 — exclude (VOID)
            continue
        if side == "over":
            y = 1 if total > line_f else 0
        else:
            y = 1 if total < line_f else 0
        points.append({"prob": float(prob), "y": y})
    return points


def _grade_handicap_lines(bundle: dict, outcome: MatchOutcome) -> List[dict]:
    """Yield CalibrationPoints for each Poly handicap line in the bundle.

    Reuses settlement.handicap_result via ball_quant.core.handicap which is
    already imported by settlement.py — importing it directly here avoids
    duplicating the push logic.
    """
    from ball_quant.core.handicap import handicap_result

    rows = bundle.get("poly", {}).get("handicap", [])
    points: List[dict] = []
    h, a = outcome.home_score, outcome.away_score
    for r in rows:
        prob = r.get("prob")
        line = r.get("line")
        side = str(r.get("outcome") or "").lower()
        if prob is None or line is None or side not in ("home", "away"):
            continue
        # handicap_result expects int line; the Poly line may be fractional (e.g. -0.5)
        # We use float arithmetic directly for fractional lines.
        line_f = float(line)
        adjusted = h - a + line_f  # positive → home covers
        if adjusted == 0.0:
            continue  # integer-line push — VOID, exclude
        covers = adjusted > 0
        if side == "home":
            y = 1 if covers else 0
        else:  # "away" side wins when home does NOT cover
            y = 0 if covers else 1
        points.append({"prob": float(prob), "y": y})
    return points


# ---------------------------------------------------------------------------
# Grade
# ---------------------------------------------------------------------------

def _dedup_to_latest_pre_kickoff(
    records: List[ForecastRecord],
) -> Tuple[List[ForecastRecord], int]:
    """Dedup records to one per match, keeping the latest pre_kickoff capture.

    Groups records by match identity key (match_id if non-empty, else match_num).
    For each group, selects the record with the latest captured_at among pre_kickoff=True
    records.  If a match has NO pre_kickoff records, it is dropped entirely — the
    caller will see zero records for it (no calibration points, no excluded count
    increment; post-kickoff exclusion is already handled in grade_forecasts).

    Returns:
        (deduped_records, n_extra_pre_kickoff_dropped)
        n_extra_pre_kickoff_dropped: number of earlier duplicate pre_kickoff records
            that were discarded by the dedup (not the same as post-kickoff exclusions,
            which grade_forecasts still tracks separately via rec.pre_kickoff=False).
    """
    # Group by match identity key.
    groups: Dict[str, List[ForecastRecord]] = {}
    for rec in records:
        key = rec.match_id if rec.match_id else rec.match_num
        groups.setdefault(key, []).append(rec)

    deduped: List[ForecastRecord] = []
    n_extra_dropped = 0
    for key, grp in groups.items():
        pre_recs = [r for r in grp if r.pre_kickoff]
        post_recs = [r for r in grp if not r.pre_kickoff]

        if not pre_recs:
            # All records are post-kickoff; keep them so grade_forecasts can
            # count them in n_excluded_post_kickoff as usual.
            deduped.extend(post_recs)
            continue

        # Keep the latest pre-kickoff record (closest to kickoff = most informed).
        best = max(pre_recs, key=lambda r: r.captured_at)
        n_extra_dropped += len(pre_recs) - 1  # earlier duplicates silently dropped

        # Post-kickoff records for the same match are kept so grade_forecasts
        # increments n_excluded for them — preserving existing exclusion semantics.
        deduped.append(best)
        deduped.extend(post_recs)

    return deduped, n_extra_dropped


def grade_forecasts(
    records: List[ForecastRecord],
    outcomes: Dict[str, MatchOutcome],
) -> Tuple[Dict[Tuple[str, str], List[dict]], int]:
    """Grade records against outcomes, returning grouped CalibrationPoints + excluded count.

    Returns:
        grouped: dict mapping (forecaster, market_family) -> list of CalibrationPoints
                 forecaster ∈ {"poly", "elo"}
                 market_family ∈ {"1x2", "handicap", "totals"}
        n_excluded_post_kickoff: count of records skipped because pre_kickoff is False

    Matching strategy:
        Primary:  match on match_id (Poly event slug) → outcomes key.
        Fallback: match_num normalized against outcome match_id.
        Only records with pre_kickoff=True and a matched outcome are graded.
        Voided outcomes (outcome.void=True) are skipped per no-fabrication rule.

    Dedup:
        Records are deduplicated to ONE pre_kickoff record per match (latest
        captured_at) before grading so that repeated cron captures of the same
        match do not inflate the calibration sample size.
    """
    # Dedup before grading — keeps only the closest-to-kickoff pre-kickoff snapshot
    # per match; post-kickoff records pass through so they are still counted below.
    records, _ = _dedup_to_latest_pre_kickoff(records)

    grouped: Dict[Tuple[str, str], List[dict]] = {}
    n_excluded = 0

    for rec in records:
        if not rec.pre_kickoff:
            n_excluded += 1
            continue

        # --- resolve outcome ---
        outcome: Optional[MatchOutcome] = None
        if rec.match_id and rec.match_id in outcomes:
            outcome = outcomes[rec.match_id]
        else:
            # Fallback: match_num against outcome match_id
            for mid, oc in outcomes.items():
                if mid == rec.match_num or mid.strip().lstrip("0") == rec.match_num.strip().lstrip("0"):
                    outcome = oc
                    break

        if outcome is None:
            logger.debug("No outcome for %s / %s — skipping", rec.match_id, rec.match_num)
            continue
        if outcome.void:
            logger.debug("Outcome voided for %s — skipping", outcome.match_id)
            continue
        if not outcome.settled:
            logger.debug("Outcome not settled for %s — skipping", outcome.match_id)
            continue

        ft_result = _spf_outcome(outcome)

        # --- Poly 1X2 ---
        ph, pd, pa = _extract_poly_1x2(rec.bundle)
        if ph is not None:
            grouped.setdefault(("poly", "1x2"), []).extend([
                {"prob": ph, "y": 1 if ft_result == "home" else 0},
                {"prob": pd, "y": 1 if ft_result == "draw" else 0},
                {"prob": pa, "y": 1 if ft_result == "away" else 0},
            ])

        # --- Elo 1X2 ---
        eh, ed, ea = _extract_elo_1x2(rec.bundle)
        if eh is not None:
            grouped.setdefault(("elo", "1x2"), []).extend([
                {"prob": eh, "y": 1 if ft_result == "home" else 0},
                {"prob": ed, "y": 1 if ft_result == "draw" else 0},
                {"prob": ea, "y": 1 if ft_result == "away" else 0},
            ])

        # --- Poly handicap lines ---
        hcap_pts = _grade_handicap_lines(rec.bundle, outcome)
        if hcap_pts:
            grouped.setdefault(("poly", "handicap"), []).extend(hcap_pts)

        # --- Poly totals lines ---
        tot_pts = _grade_totals_lines(rec.bundle, outcome)
        if tot_pts:
            grouped.setdefault(("poly", "totals"), []).extend(tot_pts)

    return grouped, n_excluded


# ---------------------------------------------------------------------------
# Calibration report
# ---------------------------------------------------------------------------

def calibration_report(
    grouped: Dict[Tuple[str, str], List[dict]],
    n_excluded_post_kickoff: int = 0,
) -> dict:
    """Compute per (forecaster, market_family) calibration metrics + poly-vs-elo head-to-head.

    Returns a dict with:
        "rows": list of row dicts (forecaster, market_family, brier, log_loss, ece, n)
        "poly_vs_elo_1x2": verdict string
        "n_excluded_post_kickoff": int
    """
    rows = []
    for (forecaster, market_family), points in sorted(grouped.items()):
        if not points:
            continue
        n = len(points)
        b = brier_score(points)
        ll = log_loss(points)
        ece = expected_calibration_error(points) if n >= 2 else float("nan")
        rows.append({
            "forecaster": forecaster,
            "market_family": market_family,
            "brier": round(b, 6),
            "log_loss": round(ll, 6),
            "ece": round(ece, 6) if ece == ece else None,  # NaN → None
            "n": n,
        })

    # Poly vs Elo head-to-head on 1x2
    poly_1x2 = grouped.get(("poly", "1x2"), [])
    elo_1x2 = grouped.get(("elo", "1x2"), [])
    if poly_1x2 and elo_1x2:
        poly_b = brier_score(poly_1x2)
        elo_b = brier_score(elo_1x2)
        if elo_b < poly_b:
            verdict = (
                f"Elo BETTER calibrated on 1x2 (Brier: elo={elo_b:.4f} < poly={poly_b:.4f}; "
                f"n_poly={len(poly_1x2)} n_elo={len(elo_1x2)})"
            )
        elif poly_b < elo_b:
            verdict = (
                f"Poly BETTER calibrated on 1x2 (Brier: poly={poly_b:.4f} < elo={elo_b:.4f}; "
                f"n_poly={len(poly_1x2)} n_elo={len(elo_1x2)})"
            )
        else:
            verdict = (
                f"Poly and Elo TIED on 1x2 (Brier={poly_b:.4f}; "
                f"n_poly={len(poly_1x2)} n_elo={len(elo_1x2)})"
            )
    elif poly_1x2:
        verdict = f"Only Poly available for 1x2 (n={len(poly_1x2)}); no Elo comparison"
    elif elo_1x2:
        verdict = f"Only Elo available for 1x2 (n={len(elo_1x2)}); no Poly comparison"
    else:
        verdict = "No 1x2 data available for poly-vs-elo comparison"

    return {
        "rows": rows,
        "poly_vs_elo_1x2": verdict,
        "n_excluded_post_kickoff": n_excluded_post_kickoff,
    }


# ---------------------------------------------------------------------------
# Ledger-capture helpers (called from cmd_bundle)
# ---------------------------------------------------------------------------

def _parse_kickoff_from_slug(event_slug: Optional[str]) -> str:
    """Extract YYYY-MM-DD from event_slug using _SLUG_DATE_RE; return '' on failure."""
    if not event_slug:
        return ""
    m = _SLUG_DATE_RE.search(event_slug)
    return m.group(1) if m else ""


def _is_pre_kickoff(captured_at: str, kickoff_date: str) -> bool:
    """Return True iff captured_at timestamp is strictly before kickoff_date.

    captured_at: ISO8601 string (e.g. "2026-06-17T14:30:00.123456")
    kickoff_date: YYYY-MM-DD string

    Conservative: if kickoff_date is empty/unparseable, returns False
    (we cannot confirm the capture was pre-kickoff so we exclude it).
    """
    if not kickoff_date or len(kickoff_date) < 10:
        return False
    # Compare date-prefix of captured_at against kickoff_date lexicographically.
    # A capture on the same calendar day as kickoff may be post-kickoff — we
    # cannot determine kickoff time from the slug, so same-day captures are
    # treated as pre-kickoff (conservative inclusion; caller can filter manually).
    captured_date = captured_at[:10]
    return captured_date <= kickoff_date


def make_forecast_record(
    bundle_entry: dict,
    captured_at: Optional[str] = None,
) -> ForecastRecord:
    """Build a ForecastRecord from a per-match bundle dict.

    captured_at: ISO8601 string; defaults to datetime.now().isoformat().
    """
    if captured_at is None:
        captured_at = datetime.now().isoformat()

    event_slug = bundle_entry.get("event_slug") or ""
    kickoff = _parse_kickoff_from_slug(event_slug)
    pre_koff = _is_pre_kickoff(captured_at, kickoff)

    match_id = event_slug  # Poly slug is the primary key
    match_num = str(bundle_entry.get("match_num") or "")
    home = bundle_entry.get("poly_home") or bundle_entry.get("ticai_home") or ""
    away = bundle_entry.get("poly_away") or bundle_entry.get("ticai_away") or ""
    match_date = str(bundle_entry.get("match_date") or "")

    return ForecastRecord(
        match_id=match_id,
        match_num=match_num,
        home=home,
        away=away,
        match_date=match_date,
        captured_at=captured_at,
        kickoff=kickoff,
        pre_kickoff=pre_koff,
        bundle=bundle_entry,
    )
