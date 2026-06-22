from __future__ import annotations

import csv
from html.parser import HTMLParser
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from ball_quant.models import MatchSP


FIELD_ALIASES = {
    "match_id": ("match_id", "编号", "场次", "赛事编号"),
    "date": ("date", "日期", "比赛日期"),
    "home": ("home", "主队", "主"),
    "away": ("away", "客队", "客"),
    "spf_home": ("spf_home", "主胜", "胜", "普通主胜"),
    "spf_draw": ("spf_draw", "平", "平局", "普通平"),
    "spf_away": ("spf_away", "主负", "负", "客胜", "普通主负"),
    "handicap": ("handicap", "让球", "让球数"),
    "rq_home": ("rq_home", "让胜", "让球胜"),
    "rq_draw": ("rq_draw", "让平", "让球平"),
    "rq_away": ("rq_away", "让负", "让球负"),
}


def load_ticai_matches(path: str) -> List[MatchSP]:
    file_path = Path(path)
    suffix = file_path.suffix.lower()
    if suffix in (".csv", ".tsv"):
        delimiter = "\t" if suffix == ".tsv" else ","
        return parse_csv(file_path, delimiter=delimiter)
    if suffix in (".html", ".htm"):
        return parse_html(file_path)
    raise ValueError(f"Unsupported SP input file: {file_path}")


def parse_csv(path: Path, delimiter: str = ",") -> List[MatchSP]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        rows = [normalize_row(row) for row in reader]
    return [row_to_match_sp(row) for row in rows]


def parse_html(path: Path) -> List[MatchSP]:
    parser = TableParser()
    parser.feed(path.read_text(encoding="utf-8"))
    rows = parser.rows
    if not rows:
        return []
    header = rows[0]
    matches: List[MatchSP] = []
    for cells in rows[1:]:
        if len(cells) < len(header):
            continue
        row = {header[i]: cells[i] for i in range(len(header))}
        matches.append(row_to_match_sp(normalize_row(row)))
    return matches


def normalize_row(row: Dict[str, str]) -> Dict[str, str]:
    normalized: Dict[str, str] = {}
    for canonical, aliases in FIELD_ALIASES.items():
        value = first_present(row, aliases)
        if value is not None:
            normalized[canonical] = value.strip()
    return normalized


def first_present(row: Dict[str, str], aliases: Iterable[str]) -> Optional[str]:
    direct = {key.strip(): value for key, value in row.items() if key is not None}
    lower = {key.lower(): value for key, value in direct.items()}
    for alias in aliases:
        if alias in direct:
            return direct[alias]
        if alias.lower() in lower:
            return lower[alias.lower()]
    return None


def row_to_match_sp(row: Dict[str, str]) -> MatchSP:
    required = [key for key in FIELD_ALIASES if key not in row]
    if required:
        raise ValueError(f"Missing required SP fields: {', '.join(required)} in row {row}")

    return MatchSP(
        match_id=row["match_id"],
        date=row["date"],
        home=row["home"],
        away=row["away"],
        spf_home=parse_float(row["spf_home"]),
        spf_draw=parse_float(row["spf_draw"]),
        spf_away=parse_float(row["spf_away"]),
        handicap=int(parse_float(row["handicap"])),
        rq_home=parse_float(row["rq_home"]),
        rq_draw=parse_float(row["rq_draw"]),
        rq_away=parse_float(row["rq_away"]),
    )


def parse_float(value: str) -> float:
    cleaned = value.replace(",", "").replace(" ", "")
    if cleaned in ("", "-", "--", "null", "None"):
        return 0.0
    return float(cleaned)


class TableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.rows: List[List[str]] = []
        self._in_cell = False
        self._current_row: List[str] = []
        self._current_cell: List[str] = []

    def handle_starttag(self, tag: str, attrs: List[tuple]) -> None:
        if tag.lower() == "tr":
            self._current_row = []
        if tag.lower() in ("td", "th"):
            self._in_cell = True
            self._current_cell = []

    def handle_data(self, data: str) -> None:
        if self._in_cell:
            text = data.strip()
            if text:
                self._current_cell.append(text)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in ("td", "th") and self._in_cell:
            self._current_row.append(" ".join(self._current_cell).strip())
            self._in_cell = False
        if tag == "tr" and self._current_row:
            self.rows.append(self._current_row)
