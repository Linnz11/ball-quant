from __future__ import annotations

import json
import re
import time
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from ball_quant.adapters.http import HttpError, get_json, get_text
from ball_quant.core.causal import causal_profile_for_category
from ball_quant.models import EventMarketMatrix, MarketQuote, MatchSP, normalize_key


GAMMA_BASE_URL = "https://gamma-api.polymarket.com"
CLOB_BASE_URL = "https://clob.polymarket.com"
SPORTS_BASE_URL = "https://polymarket.com"
WORLD_CUP_TAG_ID = 102232


class PolymarketClient:
    def __init__(
        self,
        gamma_base_url: str = GAMMA_BASE_URL,
        clob_base_url: str = CLOB_BASE_URL,
        sports_base_url: str = SPORTS_BASE_URL,
        cache_dir: Optional[Path] = None,
        offline: bool = False,
        refresh: bool = False,
        cache_ttl_seconds: Optional[int] = None,
        enrich_orderbook: bool = True,
        enrich_sports_payload: bool = True,
    ) -> None:
        self.gamma_base_url = gamma_base_url
        self.clob_base_url = clob_base_url
        self.sports_base_url = sports_base_url
        self.cache_dir = cache_dir
        self.offline = offline
        self.refresh = refresh
        self.cache_ttl_seconds = cache_ttl_seconds
        self.enrich_orderbook = enrich_orderbook
        self.enrich_sports_payload = enrich_sports_payload

    def discover_event(self, match: MatchSP, competition: Optional[str] = None) -> EventMarketMatrix:
        cached = None if self.refresh else self.load_cached_matrix(match)
        if cached and not self.cache_expired(cached):
            return cached
        if self.offline:
            return empty_matrix(match, "offline-cache-miss")

        try:
            events = self.search_events(match)
        except HttpError:
            return empty_matrix(match, "polymarket-network-error")

        event = pick_best_event(as_list(events), match, competition)
        if not event:
            return empty_matrix(match, "event-not-found")
        event = self.prefer_sports_event(event, competition)
        matrix = self.event_to_matrix(match, event)
        if self.enrich_orderbook:
            self.enrich_with_clob(matrix)
        self.write_cached_matrix(matrix, match.date)
        return matrix

    def search_events(self, match: MatchSP) -> List[Dict[str, Any]]:
        query = f"{match.home} {match.away}"
        search_payload = get_json(
            self.gamma_base_url,
            "/public-search",
            params={"q": query, "events_status": "active", "limit_per_type": 20},
        )
        events = as_search_events(search_payload)
        if events:
            return events
        fallback = get_json(
            self.gamma_base_url,
            "/events",
            params={"active": "true", "closed": "false", "limit": 100},
        )
        return as_list(fallback)

    def event_to_matrix(self, match: MatchSP, event: Dict[str, Any]) -> EventMarketMatrix:
        markets = event_to_quotes(event, match.home, match.away)
        return EventMarketMatrix(
            match_id=match.match_id,
            home=match.home,
            away=match.away,
            event_id=str(event.get("id") or ""),
            event_slug=event.get("slug"),
            markets=markets,
            raw_event=event,
        )

    def search_world_cup_events(
        self,
        queries: List[str],
        limit_per_type: int = 20,
        include_closed: bool = False,
    ) -> List[Dict[str, Any]]:
        seen = set()
        events: List[Dict[str, Any]] = []
        status = "all" if include_closed else "active"
        for query in queries:
            payload = get_json(
                self.gamma_base_url,
                "/public-search",
                params={"q": query, "events_status": status, "limit_per_type": limit_per_type},
            )
            for event in as_search_events(payload):
                if not event_relevant_to_query(event, query):
                    continue
                slug = event.get("slug") or event.get("id")
                if not slug or slug in seen:
                    continue
                seen.add(slug)
                events.append(event)
        return events

    def fetch_events_by_tag(
        self,
        tag_id: int,
        max_events: int = 700,
        limit: int = 100,
        include_closed: bool = False,
        related_tags: bool = True,
        include_best_lines: bool = True,
    ) -> List[Dict[str, Any]]:
        events: List[Dict[str, Any]] = []
        cursor = None
        seen = set()
        while len(events) < max_events:
            payload = get_json(
                self.gamma_base_url,
                "/events/keyset",
                params={
                    "tag_id": tag_id,
                    "related_tags": str(related_tags).lower(),
                    "closed": str(include_closed).lower(),
                    "limit": min(limit, max_events - len(events)),
                    "include_best_lines": str(include_best_lines).lower(),
                    "after_cursor": cursor,
                },
                timeout=30,
            )
            page = as_list(payload)
            if not page:
                break
            for event in page:
                key = event.get("slug") or event.get("id")
                if key and key not in seen:
                    seen.add(key)
                    events.append(event)
                    if len(events) >= max_events:
                        break
            cursor = payload.get("next_cursor") if isinstance(payload, dict) else None
            if not cursor:
                break
        return events

    def fetch_world_cup_events(
        self,
        tag_id: int = WORLD_CUP_TAG_ID,
        max_events: int = 700,
        include_closed: bool = False,
    ) -> List[Dict[str, Any]]:
        return self.fetch_events_by_tag(
            tag_id=tag_id,
            max_events=max_events,
            include_closed=include_closed,
            related_tags=True,
            include_best_lines=True,
        )

    def get_event_by_slug(self, slug: str) -> Dict[str, Any]:
        return get_json(self.gamma_base_url, f"/events/slug/{slug}")

    def get_sports_event_by_slug(self, slug: str, league: str = "world-cup") -> Dict[str, Any]:
        html = get_text(
            self.sports_base_url,
            f"/sports/{league}/{slug}",
            headers={"User-Agent": "Mozilla/5.0 (compatible; ball-quant/0.1)"},
            timeout=30,
        )
        event = extract_sports_event_from_next_data(html)
        event["_sports_payload_source"] = f"{self.sports_base_url}/sports/{league}/{slug}"
        return event

    def prefer_sports_event(
        self,
        event: Dict[str, Any],
        competition: Optional[str] = None,
    ) -> Dict[str, Any]:
        if self.offline or not self.enrich_sports_payload:
            return event
        slug = event.get("slug")
        if not slug or not looks_like_sports_match_slug(str(slug)):
            return event
        league = sports_league_from_event(event, competition)
        try:
            sports_event = self.get_sports_event_by_slug(str(slug), league=league)
        except HttpError:
            return event
        if len(sports_event.get("markets") or []) >= len(event.get("markets") or []):
            return merge_event_metadata(event, sports_event)
        return event

    def event_inventory(
        self,
        event: Dict[str, Any],
        enrich_orderbook: Optional[bool] = None,
        orderbook_limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        event = self.prefer_sports_event(event)
        enrich = self.enrich_orderbook if enrich_orderbook is None else enrich_orderbook
        title = event.get("title") or event.get("slug") or ""
        home, away = infer_match_teams(title)
        matrix = EventMarketMatrix(
            match_id=str(event.get("id") or event.get("slug") or ""),
            home=home,
            away=away,
            event_id=str(event.get("id") or ""),
            event_slug=event.get("slug"),
            markets=event_to_quotes(event, home, away),
            raw_event=event,
        )
        if enrich:
            self.enrich_with_clob(matrix, max_quotes=orderbook_limit)
        return matrix_to_inventory(matrix)

    def enrich_with_clob(self, matrix: EventMarketMatrix, max_quotes: Optional[int] = None) -> None:
        if self.offline:
            return
        enriched = 0
        for quote in matrix.markets:
            if not quote.token_id:
                continue
            if max_quotes is not None and enriched >= max_quotes:
                break
            try:
                book = get_json(self.clob_base_url, "/book", params={"token_id": quote.token_id})
            except HttpError:
                continue
            bid, ask = best_bid_ask(book)
            quote.bid = bid
            quote.ask = ask
            enriched += 1
            if bid is not None and ask is not None:
                quote.spread = max(0.0, ask - bid)
                quote.probability = (bid + ask) / 2

    def load_cached_matrix(self, match: MatchSP) -> Optional[EventMarketMatrix]:
        if not self.cache_dir:
            return None
        path = self.cache_dir / f"polymarket_{match.date}_{match.match_id}.json"
        if not path.exists():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        markets = [MarketQuote(**item) for item in payload.get("markets", [])]
        return EventMarketMatrix(
            match_id=payload["match_id"],
            home=payload["home"],
            away=payload["away"],
            event_id=payload.get("event_id"),
            event_slug=payload.get("event_slug"),
            markets=markets,
            raw_event=payload.get("raw_event", {}),
        )

    def cache_expired(self, matrix: EventMarketMatrix) -> bool:
        if self.cache_ttl_seconds is None:
            return False
        cached_at = matrix.raw_event.get("_cached_at")
        if cached_at is None:
            return True
        return time.time() - float(cached_at) > self.cache_ttl_seconds

    def write_cached_matrix(self, matrix: EventMarketMatrix, date: str) -> None:
        if not self.cache_dir:
            return
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path = self.cache_dir / f"polymarket_{date}_{matrix.match_id}.json"
        payload = {
            "match_id": matrix.match_id,
            "home": matrix.home,
            "away": matrix.away,
            "event_id": matrix.event_id,
            "event_slug": matrix.event_slug,
            "markets": [quote.__dict__ for quote in matrix.markets],
            "raw_event": {**matrix.raw_event, "_cached_at": time.time()},
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_matrices_from_file(path: str, matches: Iterable[MatchSP]) -> Dict[str, EventMarketMatrix]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    by_match = payload.get("matches", payload)
    matrices: Dict[str, EventMarketMatrix] = {}
    match_lookup = {match.match_id: match for match in matches}
    for match_id, item in by_match.items():
        match = match_lookup.get(match_id)
        home = item.get("home") or (match.home if match else "")
        away = item.get("away") or (match.away if match else "")
        quotes = [MarketQuote(**quote) for quote in item.get("markets", [])]
        matrices[match_id] = EventMarketMatrix(
            match_id=match_id,
            home=home,
            away=away,
            event_id=item.get("event_id"),
            event_slug=item.get("event_slug"),
            markets=quotes,
            raw_event=item.get("raw_event", {}),
        )
    return matrices


def event_to_quotes(event: Dict[str, Any], home: str = "", away: str = "") -> List[MarketQuote]:
    markets: List[MarketQuote] = []
    event_title = str(event.get("title") or event.get("slug") or "")
    for market in event.get("markets") or []:
        markets.extend(market_to_quotes(market, home, away, event_title=event_title))
    return markets


class NextDataParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.in_next_data = False
        self.chunks: List[str] = []

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        if tag == "script" and dict(attrs).get("id") == "__NEXT_DATA__":
            self.in_next_data = True

    def handle_endtag(self, tag: str) -> None:
        if tag == "script" and self.in_next_data:
            self.in_next_data = False

    def handle_data(self, data: str) -> None:
        if self.in_next_data:
            self.chunks.append(data)


def extract_sports_event_from_next_data(html: str) -> Dict[str, Any]:
    parser = NextDataParser()
    parser.feed(html)
    if not parser.chunks:
        raise HttpError("Polymarket sports page did not contain __NEXT_DATA__")
    try:
        next_data = json.loads("".join(parser.chunks))
    except json.JSONDecodeError as exc:
        raise HttpError("Polymarket sports page contained invalid __NEXT_DATA__") from exc
    page_props = ((next_data.get("props") or {}).get("pageProps") or {})
    sports_event = page_props.get("sportsEvent")
    if not isinstance(sports_event, dict):
        raise HttpError("Polymarket sports page did not contain sportsEvent")
    sports_event["_next_build_id"] = next_data.get("buildId")
    sports_event["_server_date"] = page_props.get("serverDate")
    return sports_event


def looks_like_sports_match_slug(slug: str) -> bool:
    return bool(re.match(r"^[a-z0-9]+-[a-z]{2,4}-[a-z]{2,4}-\d{4}-\d{2}-\d{2}$", slug))


def sports_league_from_event(event: Dict[str, Any], competition: Optional[str] = None) -> str:
    candidates = [
        str(event.get("seriesSlug") or ""),
        str(((event.get("sport") or {}).get("seriesSlug") if isinstance(event.get("sport"), dict) else "") or ""),
        str(competition or ""),
        str(event.get("title") or ""),
        str(event.get("slug") or ""),
    ]
    blob = normalize_key(" ".join(candidates))
    if "worldcup" in blob or "fifwc" in blob:
        return "world-cup"
    return "world-cup"


def merge_event_metadata(source: Dict[str, Any], target: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(target)
    for key in (
        "id",
        "slug",
        "title",
        "eventDate",
        "startTime",
        "startDate",
        "endDate",
        "active",
        "closed",
        "ended",
        "archived",
        "updatedAt",
        "closedTime",
        "automaticallyActive",
    ):
        if source.get(key) is not None and merged.get(key) is None:
            merged[key] = source.get(key)
    return merged


SPORTS_CATEGORY_MAP = {
    "moneyline": "moneyline",
    "money_line": "moneyline",
    "spread": "handicap",
    "spreads": "handicap",
    "handicap": "handicap",
    "total": "total_goals",
    "totals": "total_goals",
    "both_teams_to_score": "btts",
    "btts": "btts",
    "soccer_exact_score": "correct_score",
    "soccer_team_totals": "team_total",
    "soccer_first_half_team_totals": "first_half_team_total",
    "soccer_second_half_team_totals": "second_half_team_total",
    "first_half_totals": "first_half_total_goals",
    "second_half_totals": "second_half_total_goals",
    "soccer_halftime_result": "halftime_result",
    "soccer_second_half_result": "second_half_result",
    "both_teams_to_score_first_half": "btts_first_half",
    "both_teams_to_score_second_half": "btts_second_half",
    "soccer_first_to_score": "first_to_score",
    "total_corners": "total_corners",
    "soccer_team_total_corners": "team_total_corners",
    "soccer_first_half_total_corners": "first_half_total_corners",
    "soccer_second_half_total_corners": "second_half_total_corners",
    "soccer_game_corners_odd_even": "corners_odd_even",
    "soccer_first_corner": "first_corner",
    "soccer_player_goals": "player_goals",
    "soccer_player_assists": "player_assists",
    "soccer_player_shots": "player_shots",
    "soccer_player_shots_on_target": "player_shots_on_target",
    "soccer_player_goals_plus_assists": "player_goal_contributions",
    "soccer_player_goalkeeper_saves": "goalkeeper_saves",
}


PERIOD_BY_CATEGORY = {
    "first_half_team_total": "first_half",
    "first_half_total_goals": "first_half",
    "btts_first_half": "first_half",
    "first_half_total_corners": "first_half",
    "halftime_result": "first_half",
    "second_half_team_total": "second_half",
    "second_half_total_goals": "second_half",
    "btts_second_half": "second_half",
    "second_half_total_corners": "second_half",
    "second_half_result": "second_half",
}


SCOPE_BY_CATEGORY = {
    "player_goals": "player",
    "player_assists": "player",
    "player_shots": "player",
    "player_shots_on_target": "player",
    "player_goal_contributions": "player",
    "goalkeeper_saves": "player",
    "team_total": "team",
    "first_half_team_total": "team",
    "second_half_team_total": "team",
    "team_total_corners": "team",
    "first_to_score": "team",
    "first_corner": "team",
    "moneyline": "match",
    "handicap": "match",
    "total_goals": "match",
    "first_half_total_goals": "match",
    "second_half_total_goals": "match",
    "total_corners": "match",
    "first_half_total_corners": "match",
    "second_half_total_corners": "match",
    "correct_score": "match",
    "btts": "match",
    "btts_first_half": "match",
    "btts_second_half": "match",
    "group_winner": "team",
    "group_advancement": "team",
    "group_position": "team",
    "stage_advancement": "team",
    "stage_elimination": "team",
    "tournament_winner": "team",
    "team_prop": "team",
    "player_award": "player",
    "player_h2h": "player",
    "player_future": "player",
    "starting_lineup": "player",
    "continent_future": "tournament",
    "record_future": "tournament",
    "culture_future": "context",
}


def market_to_quotes(market: Dict[str, Any], home: str, away: str, event_title: str = "") -> List[MarketQuote]:
    question = str(market.get("question") or market.get("title") or "")
    sports_type = market.get("sportsMarketType")
    category = classify_market(
        question,
        event_title=event_title,
        sports_market_type=sports_type,
    )
    outcomes = parse_jsonish_list(market.get("outcomes"))
    prices = parse_jsonish_list(market.get("outcomePrices"))
    token_ids = parse_jsonish_list(market.get("clobTokenIds"))
    liquidity = parse_optional_float(market.get("liquidity") or market.get("liquidityNum"))
    volume = parse_optional_float(market.get("volume") or market.get("volumeNum"))

    quotes: List[MarketQuote] = []
    for idx, outcome in enumerate(outcomes):
        outcome_text = normalize_outcome(question, str(outcome), home, away, category, market)
        price = parse_optional_float(prices[idx]) if idx < len(prices) else None
        token_id = str(token_ids[idx]) if idx < len(token_ids) else None
        bid, ask = market_level_bid_ask(market, idx, len(outcomes))
        if price is None and bid is not None and ask is not None:
            price = (bid + ask) / 2
        metadata = sports_quote_metadata(market, category, outcome_text, str(outcome))
        spread = None
        if bid is not None and ask is not None:
            spread = max(0.0, ask - bid)
        quotes.append(
            MarketQuote(
                market_id=str(market.get("id") or market.get("conditionId") or ""),
                question=question,
                category=category,
                outcome=outcome_text,
                probability=price,
                token_id=token_id,
                bid=bid,
                ask=ask,
                spread=spread,
                liquidity=liquidity,
                volume=volume,
                sports_type=str(sports_type) if sports_type else None,
                line=metadata.get("line"),
                period=metadata.get("period"),
                side=metadata.get("side"),
                entity=metadata.get("entity"),
                scope=metadata.get("scope"),
                horizon=metadata.get("horizon"),
                causal_layer=metadata.get("causal_layer"),
                model_weight=metadata.get("model_weight"),
                is_complement=metadata.get("is_complement", False),
                active=parse_optional_bool(market.get("active")),
                closed=parse_optional_bool(market.get("closed")),
                accepting_orders=parse_optional_bool(market.get("acceptingOrders")),
                raw=market,
            )
        )
    return quotes


def classify_market(question: str, event_title: str = "", sports_market_type: Optional[str] = None) -> str:
    sports_type = (sports_market_type or "").lower()
    if sports_type in SPORTS_CATEGORY_MAP:
        return SPORTS_CATEGORY_MAP[sports_type]
    q = f"{event_title} {question}".lower()
    q_key = normalize_key(q)
    if "trump" in q_key or "culture" in q_key or "cry" in q_key or "shakehands" in q_key:
        return "culture_future"
    if " h2h" in q or "h2h:" in q or (("goals h2h" in q or "goal contributions h2h" in q) and " vs" in q):
        return "player_h2h"
    if "starting 11" in q or "starting xi" in q or "starting lineup" in q:
        return "starting_lineup"
    if (
        "golden boot" in q
        or "silver boot" in q
        or "bronze boot" in q
        or "golden ball" in q
        or "silver ball" in q
        or "bronze ball" in q
        or "golden glove" in q
        or "top goalscorer" in q
        or "top scorer" in q
        or "most assists" in q
    ):
        return "player_award"
    if "fair play award" in q:
        return "team_prop"
    if "world cup winner" in q or "win the world cup" in q or "win the 2026 fifa world cup" in q:
        return "tournament_winner"
    if "stage of elimination" in q or "be eliminated in" in q or "eliminated in the" in q:
        return "stage_elimination"
    if "reach final" in q or "reach the final" in q:
        return "stage_advancement"
    if (
        "reach semifinal" in q
        or "reach semi-final" in q
        or "reach quarterfinal" in q
        or "reach quarter-final" in q
    ):
        return "stage_advancement"
    if "knockout stage" in q or "knockout stages" in q:
        if "advance" in q or "qualify" in q or "reach" in q or "make" in q:
            return "group_advancement"
    if "round of 32" in q or "round of 16" in q or "knockout" in q:
        if "advance" in q or "qualify" in q or "reach" in q or "make" in q:
            return "stage_advancement"
    if "group stage" in q and ("eliminated" in q or "elimination" in q):
        return "stage_elimination"
    if (
        "group position" in q
        or "finish last" in q
        or "last place" in q
        or "second place" in q
        or "third place" in q
        or "fourth place" in q
        or "finish in some other position" in q
    ):
        return "group_position"
    if "continent" in q or "uefa" in q or "conmebol" in q or "concacaf" in q or "afc" in q or "caf" in q:
        return "continent_future"
    if (
        "most goals" in q
        or "most cards" in q
        or "most corners" in q
        or "highest-ranking" in q
        or "record" in q
        or "go unbeaten" in q
        or "unbeaten champion" in q
        or "penalty shootout" in q
        or "extra time" in q
        or "weather protocol" in q
        or "missed penalties" in q
        or "var decisions" in q
    ):
        return "record_future"
    if "will " in q and " play in the world cup" in q:
        return "player_future"
    if "will " in q and ("score" in q or "assist" in q) and "world cup" in q:
        return "player_future"
    if "correct score" in q or "exact score" in q or has_score_pattern(question):
        return "correct_score"
    if "both teams to score" in q or "btts" in q:
        return "btts"
    if "team total" in q or "total goals by" in q:
        return "team_total"
    if ("over" in q and "under" in q) or "o/u" in q:
        return "total_goals"
    if "handicap" in q or has_handicap_pattern(question):
        return "handicap"
    if "group winner" in q or "win group" in q:
        return "group_winner"
    if "qualify" in q or "advance" in q or "make it out" in q:
        return "group_advancement"
    if "world cup" in q and (
        "concede" in q
        or "score" in q
        or "team" in q
        or "nation" in q
        or "country" in q
    ):
        return "team_prop"
    if "draw" in q or "win" in q or "winner" in q:
        return "moneyline"
    return "other"


def normalize_outcome(
    question: str,
    outcome: str,
    home: str,
    away: str,
    category: str,
    market: Optional[Dict[str, Any]] = None,
) -> str:
    raw = outcome.strip()
    key = normalize_key(raw)
    home_key = normalize_key(home)
    away_key = normalize_key(away)
    question_key = normalize_key(question)

    if category in (
        "player_goals",
        "player_assists",
        "player_shots",
        "player_shots_on_target",
        "player_goal_contributions",
        "goalkeeper_saves",
    ):
        prop_label = extract_player_prop_label(question, raw, category, market)
        if prop_label:
            return prop_label
    if category == "moneyline":
        if key in ("yes", "no"):
            proposition = moneyline_proposition(question, home, away)
            if proposition:
                return proposition if key == "yes" else f"not_{proposition}"
            return raw
        if key == "home" or (home_key and (key == home_key or home_key in key)):
            return "home"
        if key in ("draw", "tie"):
            return "draw"
        if key == "away" or (away_key and (key == away_key or away_key in key)):
            return "away"
    if category == "handicap":
        spread_label = extract_spread_label(question, raw, home, away, market)
        if spread_label:
            return spread_label
        if key in ("yes", "no"):
            label = extract_handicap_label(question, home, away)
            if label:
                return label if key == "yes" else f"not:{label}"
        return raw
    if category in (
        "total_goals",
        "team_total",
        "first_half_team_total",
        "second_half_team_total",
        "first_half_total_goals",
        "second_half_total_goals",
        "total_corners",
        "team_total_corners",
        "first_half_total_corners",
        "second_half_total_corners",
    ):
        total_label = extract_total_label_from_market(question, raw, home, away, category, market)
        if total_label:
            return total_label
        if key in ("yes", "no"):
            label = extract_total_label(question, home, away, category)
            if label:
                return label if key == "yes" else complement_total_label(label)
        return raw
    if category == "correct_score":
        if key in ("yes", "no"):
            score = extract_small_score(question)
            if score:
                label = f"{score[0]}-{score[1]}"
                return label if key == "yes" else f"not:{label}"
        return raw
    if category == "btts":
        return "yes" if key in ("yes", "y") else "no"
    if category in ("btts_first_half", "btts_second_half"):
        return "yes" if key in ("yes", "y") else "no"
    if category in ("halftime_result", "second_half_result", "first_to_score"):
        if key in ("yes", "no"):
            label = extract_binary_sports_subject(question, market)
            if label:
                return label if key == "yes" else f"not:{label}"
        return raw
    if category == "first_corner":
        return raw
    if category in (
        "tournament_winner",
        "stage_advancement",
        "stage_elimination",
        "group_advancement",
        "group_winner",
        "group_position",
        "team_prop",
        "player_award",
        "player_h2h",
        "player_future",
        "starting_lineup",
        "continent_future",
        "record_future",
        "culture_future",
    ):
        if key in ("yes", "no"):
            label = extract_world_cup_subject(question) or clean_market_label(question)
            if label:
                return label if key == "yes" else f"not:{label}"
        return raw
    return raw


def moneyline_proposition(question: str, home: str, away: str) -> Optional[str]:
    question_key = normalize_key(question)
    home_key = normalize_key(home)
    away_key = normalize_key(away)
    if "draw" in question_key or "tie" in question_key:
        return "draw"
    if home_key and home_key in question_key and ("win" in question_key or "winner" in question_key):
        return "home"
    if away_key and away_key in question_key and ("win" in question_key or "winner" in question_key):
        return "away"
    return None


def parse_optional_bool(value: Any) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        key = value.strip().lower()
        if key in ("true", "1", "yes"):
            return True
        if key in ("false", "0", "no"):
            return False
    return None


def extract_player_prop_label(
    question: str,
    outcome: str,
    category: str,
    market: Optional[Dict[str, Any]],
) -> Optional[str]:
    player = question.split(":", 1)[0].strip()
    if not player or player == question:
        return None
    line = parse_optional_float(market.get("line") if market else None)
    if line is None:
        plus_match = re.search(r":\s*(\d+)\+\s+", question)
        if plus_match:
            line = float(plus_match.group(1)) - 0.5
    if line is None:
        return None
    stat = {
        "player_goals": "goals",
        "player_assists": "assists",
        "player_shots": "shots",
        "player_shots_on_target": "shots_on_target",
        "player_goal_contributions": "goals_assists",
        "goalkeeper_saves": "saves",
    }[category]
    outcome_key = normalize_key(outcome)
    side = "over" if outcome_key in ("yes", "over") else "under"
    return f"{player} {stat} {side} {line:g}"


def extract_binary_sports_subject(question: str, market: Optional[Dict[str, Any]]) -> Optional[str]:
    title = str((market or {}).get("groupItemTitle") or "").strip()
    if title:
        return title
    cleaned = question.strip().rstrip("?")
    patterns = [
        r"^(.+?)\s+to score first\b",
        r"^(.+?)\s+to win the second half\b",
        r"^(.+?)\s+leading at halftime\b",
        r":\s*(.+?)\s+(?:draw|neither)",
    ]
    for pattern in patterns:
        match = re.search(pattern, cleaned, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip()
    if "draw" in cleaned.lower():
        return "draw"
    if "neither" in cleaned.lower():
        return "neither"
    return None


def market_level_bid_ask(
    market: Dict[str, Any],
    outcome_index: int,
    outcome_count: int,
) -> Tuple[Optional[float], Optional[float]]:
    bid = parse_optional_float(market.get("bestBid"))
    ask = parse_optional_float(market.get("bestAsk"))
    if bid is None or ask is None:
        return None, None
    if outcome_index == 0:
        return bid, ask
    if outcome_count == 2 and outcome_index == 1:
        return max(0.0, 1.0 - ask), min(1.0, 1.0 - bid)
    return None, None


def sports_quote_metadata(
    market: Dict[str, Any],
    category: str,
    outcome_text: str,
    raw_outcome: str,
) -> Dict[str, Any]:
    side = extract_side(outcome_text) or extract_side(raw_outcome)
    entity = extract_quote_entity(market, category, outcome_text)
    line = parse_optional_float(market.get("line"))
    if category == "handicap":
        line = parse_signed_line_from_text(outcome_text) or line
    profile = causal_profile_for_category(category)
    return {
        "line": line,
        "period": PERIOD_BY_CATEGORY.get(category, "full_time"),
        "side": side,
        "entity": entity,
        "scope": SCOPE_BY_CATEGORY.get(category, "market"),
        "horizon": profile.horizon,
        "causal_layer": profile.causal_layer,
        "model_weight": profile.model_weight,
        "is_complement": is_complement_outcome(outcome_text, raw_outcome),
    }


def is_complement_outcome(outcome_text: str, raw_outcome: str) -> bool:
    return (
        outcome_text.startswith("not_")
        or outcome_text.startswith("not:")
        or normalize_key(raw_outcome) in ("no", "n")
    )


def extract_side(*texts: str) -> Optional[str]:
    for text in texts:
        key = normalize_key(text)
        if "over" in key:
            return "over"
        if "under" in key:
            return "under"
        if key in ("yes", "y"):
            return "yes"
        if key in ("no", "n"):
            return "no"
    return None


def extract_quote_entity(
    market: Dict[str, Any],
    category: str,
    outcome_text: str,
) -> Optional[str]:
    question = str(market.get("question") or "")
    if category in (
        "player_goals",
        "player_assists",
        "player_shots",
        "player_shots_on_target",
        "player_goal_contributions",
        "goalkeeper_saves",
    ):
        return question.split(":", 1)[0].strip() or None
    if category == "starting_lineup":
        match = re.search(r"^Will\s+(.+?)\s+be in\b", question, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip()
    if category in (
        "team_total",
        "first_half_team_total",
        "second_half_team_total",
        "team_total_corners",
        "first_to_score",
        "first_corner",
        "halftime_result",
        "second_half_result",
    ):
        title = str(market.get("groupItemTitle") or "").strip()
        if title:
            team = re.split(r"\s+(?:1st|2nd|O/U|\()", title, maxsplit=1)[0].strip()
            return team or title
    if category == "handicap":
        team = re.sub(r"\s+[+-]\d+(?:\.\d+)?$", "", outcome_text).strip()
        return team or None
    return None


def parse_signed_line_from_text(text: str) -> Optional[float]:
    match = re.search(r"([+-]\d+(?:\.\d+)?)$", text.strip())
    return float(match.group(1)) if match else None


def has_score_pattern(text: str) -> bool:
    return extract_small_score(text) is not None


def extract_small_score(text: str) -> Optional[Tuple[int, int]]:
    for match in re.finditer(r"(?<![\d-])(\d{1,2})\s*[-:]\s*(\d{1,2})(?![\d-])", text):
        home_score = int(match.group(1))
        away_score = int(match.group(2))
        if 0 <= home_score <= 15 and 0 <= away_score <= 15:
            return home_score, away_score
    return None


def has_handicap_pattern(text: str) -> bool:
    return re.search(r"(^|[\s(])([+-]\d+(?:\.\d+)?)(?=$|[\s)])", text) is not None


def extract_world_cup_subject(question: str) -> Optional[str]:
    cleaned = question.strip().rstrip("?")
    patterns = [
        r"^Will\s+(.+?)\s+be eliminated in the\s+(.+?)$",
        r"^Will\s+(.+?)\s+be eliminated during the\s+(.+?)$",
        r"^Will\s+(.+?)\s+finish\s+(.+?)\s+in\b",
        r"^Will\s+(.+?)\s+(?:win|advance|qualify|make|reach)\b",
        r"^(.+?)\s+(?:to win|to advance|to qualify|to reach)\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, cleaned, flags=re.IGNORECASE)
        if match:
            if len(match.groups()) >= 2:
                return " ".join(part.strip() for part in match.groups() if part.strip())
            return match.group(1).strip()
    return None


def clean_market_label(question: str) -> Optional[str]:
    cleaned = re.sub(r"\s+", " ", question.strip().rstrip("?"))
    if not cleaned:
        return None
    cleaned = re.sub(r"^Will\s+", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+in the 2026 FIFA World Cup$", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+in the World Cup$", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+at the World Cup$", "", cleaned, flags=re.IGNORECASE)
    return cleaned[:140]


def extract_handicap_label(question: str, home: str, away: str) -> Optional[str]:
    line = re.search(r"(^|[\s(])([+-]\d+(?:\.\d+)?)(?=$|[\s)])", question)
    if not line:
        return None
    question_key = normalize_key(question)
    home_key = normalize_key(home)
    away_key = normalize_key(away)
    if home_key and home_key in question_key:
        return f"{home} {line.group(2)}"
    if away_key and away_key in question_key:
        return f"{away} {line.group(2)}"
    return None


def extract_spread_label(
    question: str,
    outcome: str,
    home: str,
    away: str,
    market: Optional[Dict[str, Any]],
) -> Optional[str]:
    line = parse_spread_line(question, market)
    if line is None:
        return None
    favorite = spread_favorite(question, market, home, away)
    outcome_key = normalize_key(outcome)
    home_key = normalize_key(home)
    away_key = normalize_key(away)
    favorite_key = normalize_key(favorite or "")
    if favorite and favorite_key and favorite_key in outcome_key:
        return f"{favorite} {line:+.1f}"
    if home_key and home_key in outcome_key:
        return f"{home} {line:+.1f}" if normalize_key(home) == favorite_key else f"{home} {-line:+.1f}"
    if away_key and away_key in outcome_key:
        return f"{away} {line:+.1f}" if normalize_key(away) == favorite_key else f"{away} {-line:+.1f}"
    return None


def parse_spread_line(question: str, market: Optional[Dict[str, Any]]) -> Optional[float]:
    if market:
        line = parse_optional_float(market.get("line"))
        if line is not None:
            return line
    candidates = [question]
    if market:
        candidates.append(str(market.get("groupItemTitle") or ""))
    for text in candidates:
        match = re.search(r"\(([+-]?\d+(?:\.\d+)?)\)", text)
        if match:
            return float(match.group(1))
        match = re.search(r"(^|[\s(])([+-]\d+(?:\.\d+)?)(?=$|[\s)])", text)
        if match:
            return float(match.group(2))
    return None


def spread_favorite(question: str, market: Optional[Dict[str, Any]], home: str, away: str) -> Optional[str]:
    title = str(market.get("groupItemTitle") if market else "") or question
    title_key = normalize_key(title)
    if normalize_key(home) and normalize_key(home) in title_key:
        return home
    if normalize_key(away) and normalize_key(away) in title_key:
        return away
    match = re.search(r"Spread:\s*(.+?)\s*\(", question, flags=re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None


def extract_total_label(question: str, home: str, away: str, category: str) -> Optional[str]:
    match = re.search(r"\b(over|under)\s*(\d+(?:\.\d+)?)", question, flags=re.IGNORECASE)
    if not match:
        match = re.search(r"\bO/U\s*(\d+(?:\.\d+)?)", question, flags=re.IGNORECASE)
        if match:
            return f"over {match.group(1)}"
    if not match:
        return None
    side = match.group(1).lower()
    line = match.group(2)
    if category in ("team_total", "first_half_team_total", "second_half_team_total", "team_total_corners"):
        team = extract_team_total_entity(question, home, away, None)
        if team:
            return f"{team} {side} {line}"
        subject_key = normalize_key(team_total_subject_text(question, None))
        home_key = normalize_key(home)
        away_key = normalize_key(away)
        if home_key and home_key in subject_key:
            return f"{home} {side} {line}"
        if away_key and away_key in subject_key:
            return f"{away} {side} {line}"
    return f"{side} {line}"


def extract_total_label_from_market(
    question: str,
    outcome: str,
    home: str,
    away: str,
    category: str,
    market: Optional[Dict[str, Any]],
) -> Optional[str]:
    outcome_key = normalize_key(outcome)
    if outcome_key not in ("over", "under"):
        return None
    line = parse_total_line(question, market)
    if line is None:
        return None
    side = "over" if outcome_key == "over" else "under"
    if category in ("team_total", "first_half_team_total", "second_half_team_total", "team_total_corners"):
        team = extract_team_total_entity(question, home, away, market)
        if team:
            return f"{team} {side} {line:g}"
    return f"{side} {line:g}"


def extract_team_total_entity(
    question: str,
    home: str,
    away: str,
    market: Optional[Dict[str, Any]],
) -> Optional[str]:
    subject_key = normalize_key(team_total_subject_text(question, market))
    home_key = normalize_key(home)
    away_key = normalize_key(away)
    if away_key and away_key in subject_key:
        return away
    if home_key and home_key in subject_key:
        return home
    return None


def team_total_subject_text(question: str, market: Optional[Dict[str, Any]]) -> str:
    title = str((market or {}).get("groupItemTitle") or "").strip()
    if title:
        return title
    after_colon = question.split(":", 1)[1].strip() if ":" in question else question
    return re.split(r"\b(?:O/U|over|under)\b", after_colon, maxsplit=1, flags=re.IGNORECASE)[0]


def parse_total_line(question: str, market: Optional[Dict[str, Any]]) -> Optional[float]:
    if market:
        line = parse_optional_float(market.get("line"))
        if line is not None:
            return line
    candidates = [question]
    if market:
        candidates.append(str(market.get("groupItemTitle") or ""))
    for text in candidates:
        match = re.search(r"\b(?:O/U|over|under)\s*(\d+(?:\.\d+)?)", text, flags=re.IGNORECASE)
        if match:
            return float(match.group(1))
    return None


def complement_total_label(label: str) -> str:
    if " over " in f" {label} ":
        return label.replace(" over ", " under ")
    if " under " in f" {label} ":
        return label.replace(" under ", " over ")
    if label.startswith("over "):
        return label.replace("over ", "under ", 1)
    if label.startswith("under "):
        return label.replace("under ", "over ", 1)
    return f"not:{label}"


def pick_best_event(
    events: List[Dict[str, Any]],
    match: MatchSP,
    competition: Optional[str],
) -> Optional[Dict[str, Any]]:
    best: Optional[Tuple[int, Dict[str, Any]]] = None
    home_key = normalize_key(match.home)
    away_key = normalize_key(match.away)
    comp_key = normalize_key(competition or "")
    for event in events:
        blob = " ".join(str(event.get(field, "")) for field in ("title", "slug", "description"))
        blob_key = normalize_key(blob)
        score = 0
        if home_key in blob_key:
            score += 4
        if away_key in blob_key:
            score += 4
        if comp_key and comp_key in blob_key:
            score += 2
        score += min(len(event.get("markets") or []), 5)
        if best is None or score > best[0]:
            best = (score, event)
    return best[1] if best and best[0] >= 6 else None


def event_relevant_to_query(event: Dict[str, Any], query: str) -> bool:
    blob = " ".join(
        str(event.get(field, ""))
        for field in ("title", "slug", "description", "ticker")
    )
    blob_key = normalize_key(blob)
    if "worldcup" in blob_key or "fifwc" in blob_key:
        return True
    query_tokens = [
        normalize_key(token)
        for token in re.split(r"\s+", query)
        if len(normalize_key(token)) >= 3
    ]
    if query_tokens and all(token in blob_key for token in query_tokens):
        return True
    return False


def parse_jsonish_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, list) else []
        except json.JSONDecodeError:
            return [part.strip() for part in value.split(",") if part.strip()]
    return []


def parse_optional_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def best_bid_ask(book: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    bids = [parse_optional_float(item.get("price")) for item in book.get("bids", [])]
    asks = [parse_optional_float(item.get("price")) for item in book.get("asks", [])]
    bids = [bid for bid in bids if bid is not None]
    asks = [ask for ask in asks if ask is not None]
    return (max(bids) if bids else None, min(asks) if asks else None)


def infer_match_teams(title: str) -> Tuple[str, str]:
    match = re.search(r"(.+?)\s+vs\.?\s+(.+)$", title, flags=re.IGNORECASE)
    if not match:
        return "", ""
    return clean_team_name(match.group(1)), clean_team_name(match.group(2))


def clean_team_name(value: str) -> str:
    cleaned = re.sub(r"\s+", " ", value.strip())
    cleaned = re.sub(
        r"\s+-\s+(?:More Markets|Exact Score|Player Props|Total Corners|Halftime Result|Second Half Result|"
        r"Both Teams To Score|Both Teams to Score|First To Score|First to Score|Team Totals|Corners|"
        r"1st Half|2nd Half|First Half|Second Half).*$",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    return cleaned.strip()


def matrix_to_inventory(matrix: EventMarketMatrix) -> Dict[str, Any]:
    category_counts: Dict[str, int] = {}
    event_meta = inventory_event_metadata(matrix)
    quotes = []
    for quote in matrix.markets:
        category_counts[quote.category] = category_counts.get(quote.category, 0) + 1
        fair_odds = (1.0 / quote.probability) if quote.probability and quote.probability > 0 else None
        quotes.append(
            {
                "event_slug": matrix.event_slug,
                "event_id": matrix.event_id,
                "event_title": matrix.raw_event.get("title"),
                **event_meta,
                "home": matrix.home,
                "away": matrix.away,
                "market_id": quote.market_id,
                "question": quote.question,
                "category": quote.category,
                "sports_type": quote.sports_type,
                "scope": quote.scope,
                "period": quote.period,
                "entity": quote.entity,
                "side": quote.side,
                "line": quote.line,
                "horizon": quote.horizon,
                "causal_layer": quote.causal_layer,
                "model_weight": quote.model_weight,
                "is_complement": quote.is_complement,
                "active": quote.active,
                "closed": quote.closed,
                "accepting_orders": quote.accepting_orders,
                "outcome": quote.outcome,
                "probability": quote.probability,
                "fair_odds": fair_odds,
                "bid": quote.bid,
                "ask": quote.ask,
                "spread": quote.spread,
                "liquidity": quote.liquidity,
                "volume": quote.volume,
                "token_id": quote.token_id,
            }
        )
    return {
        "event_id": matrix.event_id,
        "event_slug": matrix.event_slug,
        "event_title": matrix.raw_event.get("title"),
        **event_meta,
        "home": matrix.home,
        "away": matrix.away,
        "market_count": len(matrix.raw_event.get("markets") or []),
        "quote_count": len(matrix.markets),
        "category_counts": category_counts,
        "quotes": quotes,
    }


def inventory_event_metadata(matrix: EventMarketMatrix) -> Dict[str, Any]:
    return {
        "polymarket_date": matrix.raw_event.get("eventDate"),
        "start_time_utc": matrix.raw_event.get("startTime")
        or matrix.raw_event.get("endDate")
        or matrix.raw_event.get("startDate"),
        "event_active": matrix.raw_event.get("active"),
        "event_closed": matrix.raw_event.get("closed"),
        "event_ended": matrix.raw_event.get("ended"),
        "event_updated_at": matrix.raw_event.get("updatedAt"),
    }


def flatten_inventory(inventories: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for inventory in inventories:
        rows.extend(inventory.get("quotes", []))
    return rows


def as_list(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        if isinstance(payload.get("events"), list):
            return payload["events"]
        if isinstance(payload.get("data"), list):
            return payload["data"]
    return []


def as_search_events(payload: Any) -> List[Dict[str, Any]]:
    if not isinstance(payload, dict):
        return as_list(payload)
    events = payload.get("events")
    if isinstance(events, list):
        return events
    results = payload.get("results")
    if isinstance(results, list):
        return [
            item
            for item in results
            if isinstance(item, dict) and (item.get("type") == "event" or item.get("markets"))
        ]
    data = payload.get("data")
    if isinstance(data, dict):
        return as_search_events(data)
    return []


def empty_matrix(match: MatchSP, reason: str) -> EventMarketMatrix:
    return EventMarketMatrix(
        match_id=match.match_id,
        home=match.home,
        away=match.away,
        raw_event={"warning": reason},
    )
