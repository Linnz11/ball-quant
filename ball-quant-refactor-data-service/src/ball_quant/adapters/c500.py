"""500.com (500彩票网) 竞彩 odds adapter.

Scrapes trade.500.com for 中国竞彩 fixed-odds across all 5 play-types:
  SPF (胜平负)       playid=269 data-type="nspf"
  RQSPF (让球)       playid=269 data-type="spf"   (same page, different div)
  CRS (比分)         playid=271 data-type="bf"
  TTG (总进球)       playid=270 data-type="jqs"
  HAFU (半全场)      playid=272 data-type="bqc"

Note: playid=269&vtype=spf returns BOTH nspf (胜平负) AND spf (让球) buttons on
the same page; no separate fetch needed for playid=354.

All I/O uses stdlib only: urllib.request + html.parser.
Page encoding is GB18030; decoded with errors="replace" to survive malformed sequences.
"""
from __future__ import annotations

import os
import re
from html.parser import HTMLParser
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.request import Request, urlopen

from ball_quant.models import TicaiOdds

# ---------------------------------------------------------------------------
# Fetch constants
# ---------------------------------------------------------------------------

_BASE = "https://trade.500.com/jczq/"
_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,*/*",
    "Referer": "https://trade.500.com/",
}

# playid → (data-type attribute value we care about, vtype query param or "")
_PLAY_PAGES: List[Tuple[str, str, str]] = [
    # (url_param_string, data-type, label)
    ("playid=269&g=2&vtype=spf", "nspf_spf",   "spf+rqspf"),  # one page, two types
    ("playid=271&g=2",           "bf",          "crs"),
    ("playid=270&g=2",           "jqs",         "ttg"),
    ("playid=272&g=2",           "bqc",         "hafu"),
]


# ---------------------------------------------------------------------------
# HTML fetch helper
# ---------------------------------------------------------------------------

def _fetch_html(url: str, cache_path: Optional[Path] = None) -> str:
    """Fetch URL and return decoded text.

    If cache_path points to an existing file the file is returned instead of
    making a network request (cassette replay).  If it does not exist the live
    response is written there for future replays.

    Raises urllib.error.URLError / OSError on network failure.
    Page is GB18030; decode with errors="replace" to tolerate partial content.
    """
    if cache_path and cache_path.exists():
        # Cassette replay — read pre-saved UTF-8 (iconv-converted) file
        return cache_path.read_text(encoding="utf-8", errors="replace")

    req = Request(url, headers=_HEADERS)
    with urlopen(req, timeout=45) as resp:
        raw_bytes = resp.read()

    text = raw_bytes.decode("gb18030", errors="replace")

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(text, encoding="utf-8")

    return text


# ---------------------------------------------------------------------------
# HTML parser
# ---------------------------------------------------------------------------

class _MatchRow:
    """Accumulator for one <tr class="bet-tb-tr"> block."""

    __slots__ = (
        "fid", "home", "away", "match_date", "match_time", "buy_end_time",
        "match_num", "league", "handicap_line", "subactive",
        "nspf", "spf", "bf", "jqs", "bqc",
    )

    def __init__(self, attrs: Dict[str, str]) -> None:
        self.fid = attrs.get("data-fixtureid", "")
        self.home = attrs.get("data-homesxname", "")
        self.away = attrs.get("data-awaysxname", "")
        self.match_date = attrs.get("data-matchdate", "")
        # data-matchtime: "HH:MM" in CST — present on all live 500.com rows.
        # Empty string when absent (e.g. old cassettes); combined with match_date
        # → "YYYY-MM-DDTHH:MM" ISO-8601 kickoff in _rows_to_ticai.
        self.match_time = attrs.get("data-matchtime", "")
        # data-buyendtime: "YYYY-MM-DD HH:MM:SS" in CST — betting-close deadline (停售).
        # The analysis pipeline trigger fires at buyendtime − 70min.
        # Empty string when absent; normalized to ISO-8601 "YYYY-MM-DDTHH:MM:SS" in
        # _rows_to_ticai (space → T).
        self.buy_end_time = attrs.get("data-buyendtime", "")
        self.match_num = attrs.get("data-matchnum")
        self.league = attrs.get("data-simpleleague", "")
        # data-rangqiu: negative = home gives handicap, positive = away gives handicap
        rq_str = attrs.get("data-rangqiu", "0")
        try:
            self.handicap_line: Optional[float] = float(rq_str)
        except (ValueError, TypeError):
            self.handicap_line = None
        # data-subactive: "nspfdg:1,nspfgg:1,spfdg:0,..." — 1=open, 0=closed
        # We use this to skip odds that are not actually on sale.
        self.subactive: str = attrs.get("data-subactive", "")
        # Collected odds: {value_key: sp_float}
        self.nspf: Dict[str, float] = {}
        self.spf: Dict[str, float] = {}
        self.bf: Dict[str, float] = {}
        self.jqs: Dict[int, float] = {}
        self.bqc: Dict[str, float] = {}

    def subactive_flag(self, key: str) -> bool:
        """Return True if the given subactive component is 1 (=on sale).

        Example keys: "nspfdg", "nspfgg", "bfdg", "bfgg", "jqdg", "jqgg", "hcdg", "hcgg".
        We accept a play if *either* dg (单关) or gg (过关) slot is active.
        """
        # Parse lazily from the raw string
        for token in self.subactive.split(","):
            parts = token.split(":")
            if len(parts) == 2 and parts[0].strip() == key:
                return parts[1].strip() == "1"
        return False

    def any_active(self, *keys: str) -> bool:
        """Return True if any of the given subactive component keys is active."""
        return any(self.subactive_flag(k) for k in keys)


class _500Parser(HTMLParser):
    """Walk the page and collect <tr class="bet-tb-tr"> rows + their bet buttons.

    Two element types matter:
      <tr class="bet-tb-tr" data-fixtureid="..." ...>  — starts a match row
      <p class="betbtn sbetbtn" data-type="..." data-value="..." data-sp="...">
        — odds button inside or attached to the current row

    CRS pages use <p class="sbetbtn"> (inside a bet-more-wrap sub-table) rather
    than <p class="betbtn">.  We track both.
    """

    def __init__(self) -> None:
        super().__init__()
        self._rows: List[_MatchRow] = []
        self._current: Optional[_MatchRow] = None

    def handle_starttag(self, tag: str, attrs_list: list) -> None:
        attrs = dict(attrs_list)
        css = attrs.get("class", "")

        if tag == "tr" and "bet-tb-tr" in css.split():
            # Start a new match row
            self._current = _MatchRow(attrs)
            self._rows.append(self._current)
            return

        if self._current is None:
            return

        # Odds buttons: both betbtn and sbetbtn carry the odds attributes
        if tag == "p" and ("betbtn" in css or "sbetbtn" in css):
            dtype = attrs.get("data-type", "")
            value = attrs.get("data-value", "")
            sp_str = attrs.get("data-sp", "")
            try:
                sp = float(sp_str)
            except (ValueError, TypeError):
                return  # odds not parseable — skip rather than fabricate

            if dtype == "nspf":
                # 胜平负: "3"→home, "1"→draw, "0"→away
                key = {"3": "home", "1": "draw", "0": "away"}.get(value)
                if key:
                    self._current.nspf[key] = sp
            elif dtype == "spf":
                # 让球胜平负: same encoding
                key = {"3": "home", "1": "draw", "0": "away"}.get(value)
                if key:
                    self._current.spf[key] = sp
            elif dtype == "bf":
                # 比分: "H:A" → "H-A"; Chinese catch-alls → canonical bucket keys
                crs_key = _crs_value_to_key(value)
                if crs_key:
                    self._current.bf[crs_key] = sp
            elif dtype == "jqs":
                # 总进球: "0".."7" (7 = 7+) → int key
                try:
                    goal_n = int(value)
                    self._current.jqs[goal_n] = sp
                except (ValueError, TypeError):
                    pass
            elif dtype == "bqc":
                # 半全场: "X-Y" where X/Y ∈ {"3","1","0"} → 2-char key "hh","hd",...
                hafu_key = _hafu_value_to_key(value)
                if hafu_key:
                    self._current.bqc[hafu_key] = sp

    def handle_endtag(self, tag: str) -> None:
        # CRS rows are followed by a bet-more-wrap <tr> that also belongs to the
        # same match; we intentionally do NOT reset _current on </tr> so those
        # nested sbetbtn buttons are collected under the right row.
        # We only reset between successive bet-tb-tr rows via handle_starttag.
        pass

    @property
    def rows(self) -> List[_MatchRow]:
        return self._rows


# ---------------------------------------------------------------------------
# Value-mapping helpers (pure functions — easy to unit-test)
# ---------------------------------------------------------------------------

def _crs_value_to_key(value: str) -> Optional[str]:
    """Map data-value of a bf button to a canonical correct-score key.

    "H:A"      → "H-A"  (e.g. "1:0" → "1-0")
    "胜其它"   → "home_other"
    "平其它"   → "draw_other"
    "负其它"   → "away_other"
    Anything else → None (skip).
    """
    _CATCH_ALL = {
        "胜其它": "home_other",
        "平其它": "draw_other",
        "负其它": "away_other",
    }
    if value in _CATCH_ALL:
        return _CATCH_ALL[value]
    # e.g. "1:0", "2:1", "3:2"
    m = re.match(r"^(\d+):(\d+)$", value)
    if m:
        return f"{m.group(1)}-{m.group(2)}"
    return None


_HAFU_DIGIT = {"3": "h", "1": "d", "0": "a"}


def _hafu_value_to_key(value: str) -> Optional[str]:
    """Map bqc data-value "X-Y" → 2-char key.

    "3-3" → "hh",  "3-1" → "hd",  "3-0" → "ha"
    "1-3" → "dh",  "1-1" → "dd",  "1-0" → "da"
    "0-3" → "ah",  "0-1" → "ad",  "0-0" → "aa"
    """
    parts = value.split("-")
    if len(parts) != 2:
        return None
    ht_char = _HAFU_DIGIT.get(parts[0])
    ft_char = _HAFU_DIGIT.get(parts[1])
    if ht_char and ft_char:
        return ht_char + ft_char
    return None


# ---------------------------------------------------------------------------
# Core parse function
# ---------------------------------------------------------------------------

def parse_html(html: str) -> List[_MatchRow]:
    """Parse a 500.com trade page HTML string and return match rows.

    Raises ValueError if the page contains no match rows — this signals a
    structurally broken page (wrong URL, server error, or empty date) and
    should not be silently swallowed.
    """
    parser = _500Parser()
    parser.feed(html)
    if not parser.rows:
        raise ValueError("No <tr class='bet-tb-tr'> rows found — page may be empty or malformed.")
    return parser.rows


# ---------------------------------------------------------------------------
# Merge rows from multiple pages into TicaiOdds objects
# ---------------------------------------------------------------------------

def _rows_to_ticai(rows_by_page: Dict[str, List[_MatchRow]]) -> List[TicaiOdds]:
    """Merge rows from all play-type pages into one TicaiOdds per fixture-id.

    rows_by_page: {"spf+rqspf": [...], "crs": [...], "ttg": [...], "hafu": [...]}
    Match key is data-fixtureid (a stable numeric string from 500.com).
    """
    # Collect all fixture ids and their data, using spf page as the canonical
    # source for match metadata (home/away/date).
    merged: Dict[str, Dict] = {}

    for label, rows in rows_by_page.items():
        for row in rows:
            fid = row.fid
            if not fid:
                continue
            if fid not in merged:
                merged[fid] = {
                    "home": row.home,
                    "away": row.away,
                    "match_date": row.match_date,
                    "match_time": row.match_time,
                    "buy_end_time": row.buy_end_time,
                    "match_num": row.match_num,
                    "league": row.league,
                    "handicap_line": row.handicap_line,
                    "nspf": {},
                    "spf": {},
                    "bf": {},
                    "jqs": {},
                    "bqc": {},
                    # sale-mode flags default to False; OR-accumulated across pages
                    # so any page that reports a flag open wins.
                    "nspfdg": False, "nspfgg": False,   # → spf dest
                    "spfdg": False,  "spfgg": False,    # → rqspf dest
                    "bfdg": False,   "bfgg": False,     # → correct_score dest
                    "jqdg": False,   "jqgg": False,     # → total_goals dest
                    "hcdg": False,   "hcgg": False,     # → hafu dest (prefix is hc, not bqc)
                }
            m = merged[fid]
            # Later pages may carry better team-name data; prefer non-empty values
            if row.home and not m["home"]:
                m["home"] = row.home
            if row.away and not m["away"]:
                m["away"] = row.away
            if row.match_time and not m["match_time"]:
                m["match_time"] = row.match_time
            if row.buy_end_time and not m["buy_end_time"]:
                m["buy_end_time"] = row.buy_end_time
            # Merge odds maps — do not overwrite existing entries (first page wins)
            for dtype in ("nspf", "spf", "bf", "bqc"):
                src = getattr(row, dtype)
                dst = m[dtype]
                for k, v in src.items():
                    if k not in dst:
                        dst[k] = v
            # jqs has int keys
            for k, v in row.jqs.items():
                if k not in m["jqs"]:
                    m["jqs"][k] = v
            # Handicap line comes from the spf+rqspf page (data-rangqiu attribute)
            if m["handicap_line"] is None and row.handicap_line is not None:
                m["handicap_line"] = row.handicap_line
            # OR-accumulate subactive flags: once any page reports a flag open, it stays open.
            for flag in ("nspfdg", "nspfgg", "spfdg", "spfgg",
                         "bfdg", "bfgg", "jqdg", "jqgg", "hcdg", "hcgg"):
                if row.subactive_flag(flag):
                    m[flag] = True

    result: List[TicaiOdds] = []
    for fid, m in merged.items():
        # Empty the odds dict for any play type where BOTH dg and gg are off —
        # those are not actually on sale and should not carry phantom odds.
        nspf_odds  = m["nspf"] if (m["nspfdg"] or m["nspfgg"]) else {}
        spf_odds   = m["spf"]  if (m["spfdg"]  or m["spfgg"])  else {}
        bf_odds    = m["bf"]   if (m["bfdg"]   or m["bfgg"])   else {}
        jqs_odds   = m["jqs"]  if (m["jqdg"]   or m["jqgg"])   else {}
        bqc_odds   = m["bqc"]  if (m["hcdg"]   or m["hcgg"])   else {}
        # Build ISO-8601 kickoff string ("YYYY-MM-DDTHH:MM" CST) when both date
        # and time are present.  Neither field is fabricated — both come verbatim
        # from the same <tr> row; if either is missing we leave kickoff=None.
        match_date = m["match_date"]
        match_time = m["match_time"]
        kickoff: Optional[str] = (
            f"{match_date}T{match_time}"
            if match_date and match_time
            else None
        )
        # Build ISO-8601 bet_close string ("YYYY-MM-DDTHH:MM:SS" CST) from
        # data-buyendtime ("YYYY-MM-DD HH:MM:SS").  Replace the space with T.
        # This is the 停售 (betting-close) deadline — the trigger fires at
        # bet_close − 70min.  None when the attribute is absent (no fabrication).
        raw_bet_close = m["buy_end_time"]
        bet_close: Optional[str] = (
            raw_bet_close.replace(" ", "T", 1)
            if raw_bet_close
            else None
        )
        result.append(TicaiOdds(
            match_id=fid,
            match_date=match_date,
            league=m["league"],
            home=m["home"],
            away=m["away"],
            match_num=m["match_num"],
            spf=nspf_odds,
            handicap_line=m["handicap_line"],
            rqspf=spf_odds,
            correct_score=bf_odds,
            total_goals=jqs_odds,
            hafu=bqc_odds,
            kickoff=kickoff,
            bet_close=bet_close,
            spf_danjuan=m["nspfdg"],
            spf_guoguan=m["nspfgg"],
            rqspf_danjuan=m["spfdg"],
            rqspf_guoguan=m["spfgg"],
            correct_score_danjuan=m["bfdg"],
            correct_score_guoguan=m["bfgg"],
            total_goals_danjuan=m["jqdg"],
            total_goals_guoguan=m["jqgg"],
            hafu_danjuan=m["hcdg"],
            hafu_guoguan=m["hcgg"],
        ))

    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_odds(
    date: str,
    cache_dir: Optional[Path] = None,
) -> List[TicaiOdds]:
    """Fetch 竞彩 odds from 500.com for the given date string (YYYY-MM-DD).

    Makes 4 HTTP requests (spf+rqspf, crs, ttg, hafu) and merges by fixture-id.

    cache_dir: if provided, each page is saved/replayed as
        <cache_dir>/c500_<label>_<date>.html  (UTF-8 on disk).
    """
    rows_by_page: Dict[str, List[_MatchRow]] = {}

    for qs, _dtype, label in _PLAY_PAGES:
        url = f"{_BASE}?{qs}&date={date}"

        cache_path: Optional[Path] = None
        if cache_dir is not None:
            cache_path = Path(cache_dir) / f"c500_{label.replace('+', '_')}_{date}.html"

        html = _fetch_html(url, cache_path=cache_path)
        rows_by_page[label] = parse_html(html)

    return _rows_to_ticai(rows_by_page)


def load_odds(cache_dir: Path, date: Optional[str] = None) -> List[TicaiOdds]:
    """Load odds from pre-saved cassette files in cache_dir.

    File naming conventions accepted (searched in order):
      1. c500_<label>_<date>.html   — date-stamped (from fetch_odds cache writes)
         e.g. c500_spf+rqspf_2026-06-15.html
      2. c500_<short>.html          — bare label (test cassettes)
         e.g. c500_spf.html, c500_crs.html, c500_ttg.html, c500_hafu.html

    If date is None and no date-stamped files exist, falls back to bare-label files.
    Raises FileNotFoundError if no matching cassettes are found.
    """
    cache_dir = Path(cache_dir)

    # Bare-label short names (for test cassettes stored without a date)
    _SHORT_LABEL = {
        "spf+rqspf": "spf",
        "crs":        "crs",
        "ttg":        "ttg",
        "hafu":       "hafu",
    }

    if date is None:
        # Discover a date that has all four date-stamped cassettes
        candidates = set()
        for f in cache_dir.glob("c500_*.html"):
            # filename: c500_<label>_<date>.html — date is last _-separated segment minus .html
            stem = f.stem  # e.g. "c500_spf+rqspf_2026-06-15"
            parts = stem.split("_")
            if len(parts) >= 3:
                # last segment should look like a date YYYY-MM-DD
                last = parts[-1]
                if len(last) == 10 and last.count("-") == 2:
                    candidates.add(last)
        date = sorted(candidates)[0] if candidates else None

    rows_by_page: Dict[str, List[_MatchRow]] = {}
    for _qs, _dtype, label in _PLAY_PAGES:
        # Try date-stamped name first
        fname: Optional[Path] = None
        if date is not None:
            candidate = cache_dir / f"c500_{label.replace('+', '_')}_{date}.html"
            if candidate.exists():
                fname = candidate
        # Fall back to bare-label name (test cassettes: c500_spf.html etc.)
        if fname is None:
            short = _SHORT_LABEL.get(label, label)
            candidate_bare = cache_dir / f"c500_{short}.html"
            if candidate_bare.exists():
                fname = candidate_bare
        if fname is None:
            raise FileNotFoundError(
                f"Missing cassette for label '{label}' in {cache_dir} "
                f"(tried date-stamped and bare-label forms)"
            )
        html = fname.read_text(encoding="utf-8", errors="replace")
        rows_by_page[label] = parse_html(html)

    return _rows_to_ticai(rows_by_page)


def load_odds_from_fixtures(
    spf_html: str,
    crs_html: str,
    ttg_html: str,
    hafu_html: str,
) -> List[TicaiOdds]:
    """Parse odds directly from HTML strings (for unit tests using cassettes).

    Each argument is the full HTML content of the corresponding page.
    Does not touch the filesystem beyond what parse_html does.
    """
    rows_by_page: Dict[str, List[_MatchRow]] = {
        "spf+rqspf": parse_html(spf_html),
        "crs":        parse_html(crs_html),
        "ttg":        parse_html(ttg_html),
        "hafu":       parse_html(hafu_html),
    }
    return _rows_to_ticai(rows_by_page)
