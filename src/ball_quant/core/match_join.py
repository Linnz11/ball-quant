"""match_join.py — pair a 中国竞彩 (sporttery) TicaiOdds with its Polymarket EventMarketMatrix.

The hard problem is team-name bridging: 体彩 carries Chinese names (阿根廷),
Polymarket carries English (Argentina).  Strategy:
  1. TEAM_ALIASES dict: Chinese → lowercase canonical English.
  2. normalize_team(): Chinese names go through TEAM_ALIASES; all others go
     through models.normalize_key (accent-strip + lowercase + alnum-only).
  3. pair_one(): exact match on (home_key, away_key); swapped fallback flagged.
  4. pair_all(): returns (matched, unmatched) so callers can report gaps — nothing
     is silently dropped.

Unmatched happens when a Chinese name is NOT in TEAM_ALIASES — the known gap is
listed in the comment below.  Extend TEAM_ALIASES to cover new nations/clubs.
"""

from __future__ import annotations

import re
from datetime import date
from typing import Dict, List, Optional, Sequence, Tuple

from ball_quant.models import EventMarketMatrix, TicaiOdds, normalize_key

# ---------------------------------------------------------------------------
# Alias table — Chinese team name → lowercase canonical English.
# This is the known-gap list: any name absent here will not pair.
# Extend by appending entries; prefer the most common English spelling used by
# Polymarket (usually FIFA standard or Wikipedia article title, lowercased).
# ---------------------------------------------------------------------------
TEAM_ALIASES: Dict[str, str] = {
    # South America
    "阿根廷": "argentina",
    "巴西": "brazil",
    "乌拉圭": "uruguay",
    "厄瓜多尔": "ecuador",
    "哥伦比亚": "colombia",
    # Europe
    "西班牙": "spain",
    "比利时": "belgium",
    "荷兰": "netherlands",
    "德国": "germany",
    "法国": "france",
    "英格兰": "england",
    "葡萄牙": "portugal",
    "意大利": "italy",
    "克罗地亚": "croatia",
    "瑞典": "sweden",
    "土耳其": "türkiye",  # Poly uses "Türkiye" (normalize_key strips the diacritic)
    # Africa
    "埃及": "egypt",
    "摩洛哥": "morocco",
    "塞内加尔": "senegal",
    "突尼斯": "tunisia",
    # Ivory Coast has two common Chinese spellings
    "科特迪瓦": "cote d'ivoire",
    "象牙海岸": "cote d'ivoire",
    # Cape Verde has two common Chinese spellings; Polymarket uses "Cabo Verde" (FIFA standard)
    "佛得角": "cabo verde",
    "维德角": "cabo verde",
    # Polymarket alternate spellings (English → canonical Polymarket English)
    "Cape Verde": "cabo verde",
    "cape verde": "cabo verde",
    # Asia-Pacific
    "日本": "japan",
    "韩国": "korea republic",  # Poly WC name is "Korea Republic", not "South Korea"
    "澳大利亚": "australia",
    "新西兰": "new zealand",
    "伊朗": "iran",
    # Polymarket uses "IR Iran" for Iran (FIFA official abbreviation)
    "IR Iran": "iran",
    "ir iran": "iran",
    # Middle East / Gulf
    "沙特": "saudi arabia",
    "沙特阿拉伯": "saudi arabia",
    # North / Central America & Caribbean
    "美国": "united states",
    "墨西哥": "mexico",
    "加拿大": "canada",
    "库拉索": "curacao",
    # --- WC 2026 nations (were MISSING → caused 16/18 unmatched in recommend) ---
    "阿尔及利亚": "algeria",
    "约旦": "jordan",
    "挪威": "norway",
    "伊拉克": "iraq",
    "奥地利": "austria",
    "加纳": "ghana",
    "巴拿马": "panama",
    "乌兹别克斯坦": "uzbekistan",
    "乌兹别克": "uzbekistan",
    "刚果金": "dr congo",
    "刚果(金)": "dr congo",
    "刚果民主共和国": "dr congo",
    "DR Congo": "dr congo",
    "捷克": "czechia",
    "Czechia": "czechia",
    "南非": "south africa",
    "South Africa": "south africa",
    "瑞士": "switzerland",
    "波黑": "bosnia-herzegovina",
    "波斯尼亚和黑塞哥维那": "bosnia-herzegovina",
    "Bosnia-Herzegovina": "bosnia-herzegovina",
    "卡塔尔": "qatar",
    "巴拉圭": "paraguay",
    "苏格兰": "scotland",
    "海地": "haiti",
    "委内瑞拉": "venezuela",
    "秘鲁": "peru",
    "玻利维亚": "bolivia",
    "智利": "chile",
    "波兰": "poland",
    "丹麦": "denmark",
    "斯洛文尼亚": "slovenia",
    "塞尔维亚": "serbia",
    "乌克兰": "ukraine",
    "斯洛伐克": "slovakia",
    "匈牙利": "hungary",
    "阿尔巴尼亚": "albania",
    "牙买加": "jamaica",
    "洪都拉斯": "honduras",
    "哥斯达黎加": "costa rica",
    "印度尼西亚": "indonesia",
    "印尼": "indonesia",
    # Poly uses FIFA name "Korea Republic" for South Korea at the WC
    "Korea Republic": "korea republic",
    # English Premier League clubs (for 竞彩 club fixtures)
    "曼城": "manchester city",
    "曼联": "manchester united",
    "阿森纳": "arsenal",
    "切尔西": "chelsea",
    "利物浦": "liverpool",
    "热刺": "tottenham",
    "纽卡斯尔": "newcastle",
    # Spanish La Liga clubs
    "皇马": "real madrid",
    "巴萨": "barcelona",
    "马竞": "atletico madrid",
    "塞维利亚": "sevilla",
    "皇家社会": "real sociedad",
    # German Bundesliga clubs
    "拜仁": "bayern munich",
    "多特": "borussia dortmund",
    # Italian Serie A clubs
    "尤文": "juventus",
    "国米": "inter milan",
    "米兰": "ac milan",
    # French Ligue 1 clubs
    "大巴黎": "paris sg",
    "马赛": "marseille",
}


def normalize_team(name: str) -> str:
    """Return a canonical comparable key for a team name.

    Chinese names (or any name present in TEAM_ALIASES) are translated to their
    canonical English equivalent first; the result then goes through normalize_key
    for accent-stripping and lowercasing.  Non-Chinese English names go directly
    through normalize_key so they match what Polymarket stores after the same
    normalization.

    normalize_key strips diacritics, lowercases, and keeps only alnum chars —
    so "côte d'ivoire" and "cote d'ivoire" both collapse to "cotedivoire".
    """
    # Translate via alias table first (catches Chinese names and any hand-coded
    # alternate spellings).  Fall through to normalize_key for English names.
    canonical = TEAM_ALIASES.get(name, name)
    return normalize_key(canonical)


# Polymarket event slugs embed the MATCH date: "fifwc-prt-cdr-2026-06-17[...]".
# The raw_event "start_date"/"startDate" field is the market-CREATION timestamp
# (e.g. 2026-04-06), which is useless for pairing — so we read the date off the slug.
_SLUG_DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")


def _parse_date(date_str: str) -> Optional[date]:
    """Parse ISO date string "YYYY-MM-DD"; return None on any failure."""
    try:
        return date.fromisoformat(date_str)
    except (ValueError, TypeError):
        return None


def pair_one(
    ticai: TicaiOdds,
    matrices: Sequence[EventMarketMatrix],
    date_tolerance_days: int = 1,
) -> Optional[EventMarketMatrix]:
    """Find the EventMarketMatrix whose teams match the TicaiOdds teams.

    Matching rules (applied in order; first hit wins):
      1. Exact match: normalize_team(ticai.home) == normalize_team(matrix.home)
                  AND normalize_team(ticai.away) == normalize_team(matrix.away).
      2. Swapped fallback: home/away reversed — signals a data-entry flip; the
         matrix is still returned but this case is flagged via a printed warning
         so the caller can investigate.  (We don't raise because a swap is
         recoverable; the caller decides whether to trust the pairing.)

    Date proximity: if date_tolerance_days > 0 and the TicaiOdds carries a
    match_date AND the matrix carries a raw_event "start_date" key, matches that
    fall outside the tolerance window are skipped.  If either date is absent the
    check is skipped (permissive).
    """
    t_home = normalize_team(ticai.home)
    t_away = normalize_team(ticai.away)
    ticai_date = _parse_date(ticai.match_date)

    exact_hits: List[EventMarketMatrix] = []
    swapped_hits: List[EventMarketMatrix] = []

    for matrix in matrices:
        m_home = normalize_team(matrix.home)
        m_away = normalize_team(matrix.away)

        # Optional date proximity gate — only applied when both dates are present.
        if date_tolerance_days > 0 and ticai_date is not None:
            # Read the MATCH date off the event slug (raw_event "start_date" is the
            # market-CREATION date, useless for pairing). The slug date is ET; the
            # 体彩 date is CST → date_tolerance_days=1 absorbs the off-by-one.
            slug = str(
                getattr(matrix, "event_slug", "")
                or matrix.raw_event.get("slug")
                or ""
            )
            _sm = _SLUG_DATE_RE.search(slug)
            m_date = _parse_date(_sm.group(1)) if _sm else None
            if m_date is not None:
                delta = abs((ticai_date - m_date).days)
                if delta > date_tolerance_days:
                    continue  # Too far apart — skip this candidate

        if t_home == m_home and t_away == m_away:
            exact_hits.append(matrix)
        elif t_home == m_away and t_away == m_home:
            # Swapped home/away — flag it; may still be the right event
            swapped_hits.append(matrix)

    if exact_hits:
        return exact_hits[0]

    if swapped_hits:
        # Warn so the caller can audit — do NOT silently accept without notice
        import warnings

        warnings.warn(
            f"pair_one: home/away swapped for TicaiOdds "
            f"({ticai.home} vs {ticai.away}) — using swapped matrix "
            f"({swapped_hits[0].home} vs {swapped_hits[0].away}). "
            "Verify fixture data.",
            UserWarning,
            stacklevel=2,
        )
        return swapped_hits[0]

    return None


def pair_all(
    ticai_list: Sequence[TicaiOdds],
    matrices: Sequence[EventMarketMatrix],
    date_tolerance_days: int = 1,
) -> Tuple[List[Tuple[TicaiOdds, EventMarketMatrix]], List[TicaiOdds]]:
    """Pair every TicaiOdds against the best-matching EventMarketMatrix.

    Returns:
        (matched, unmatched)
        matched   — list of (TicaiOdds, EventMarketMatrix) pairs
        unmatched — TicaiOdds for which no matrix was found

    NEVER silently drops unmatched entries — callers must inspect the unmatched
    list and report "体彩 match X has no Polymarket counterpart" to the user.
    """
    matched: List[Tuple[TicaiOdds, EventMarketMatrix]] = []
    unmatched: List[TicaiOdds] = []

    for ticai in ticai_list:
        matrix = pair_one(ticai, matrices, date_tolerance_days=date_tolerance_days)
        if matrix is not None:
            matched.append((ticai, matrix))
        else:
            unmatched.append(ticai)

    return matched, unmatched
