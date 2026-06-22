from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict

from ball_quant.core.settlement import MatchOutcome


def load_results(path: str) -> Dict[str, MatchOutcome]:
    """Parse a CSV of final match scores into a match_id -> MatchOutcome mapping.

    Expected header: match_id,home_score,away_score[,void]
    Bad rows raise ValueError — no silent skipping (standing order: no wide except).
    """
    file_path = Path(path)
    outcomes: Dict[str, MatchOutcome] = {}
    with file_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row_num, row in enumerate(reader, start=2):  # row 1 = header
            try:
                match_id = row["match_id"].strip()
                home_score = int(row["home_score"].strip())
                away_score = int(row["away_score"].strip())
            except (KeyError, ValueError) as exc:
                raise ValueError(
                    f"results CSV row {row_num} is malformed: {exc!r} in {dict(row)}"
                ) from exc
            void_raw = row.get("void", "").strip().lower()
            void = void_raw in ("1", "true", "yes")
            outcomes[match_id] = MatchOutcome(
                match_id=match_id,
                home_score=home_score,
                away_score=away_score,
                void=void,
            )
    return outcomes


def save_results(outcomes: Dict[str, MatchOutcome], path: str) -> None:
    """Serialise a match_id -> MatchOutcome mapping to JSON for caching."""
    file_path = Path(path)
    data = {
        mid: {
            "match_id": o.match_id,
            "home_score": o.home_score,
            "away_score": o.away_score,
            "settled": o.settled,
            "void": o.void,
            "poly_resolutions": dict(o.poly_resolutions),
        }
        for mid, o in outcomes.items()
    }
    file_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def load_results_json(path: str) -> Dict[str, MatchOutcome]:
    """Load a JSON outcome cache previously written by save_results."""
    file_path = Path(path)
    raw = json.loads(file_path.read_text(encoding="utf-8"))
    return {
        mid: MatchOutcome(
            match_id=entry["match_id"],
            home_score=entry["home_score"],
            away_score=entry["away_score"],
            settled=entry.get("settled", True),
            void=entry.get("void", False),
            poly_resolutions=entry.get("poly_resolutions", {}),
        )
        for mid, entry in raw.items()
    }
