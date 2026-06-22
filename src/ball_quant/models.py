from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import unicodedata


@dataclass(frozen=True)
class MatchSP:
    match_id: str
    date: str
    home: str
    away: str
    spf_home: float
    spf_draw: float
    spf_away: float
    handicap: int
    rq_home: float
    rq_draw: float
    rq_away: float

    def spf_items(self) -> Sequence[Tuple[str, str, float]]:
        return (
            ("spf", "home", self.spf_home),
            ("spf", "draw", self.spf_draw),
            ("spf", "away", self.spf_away),
        )

    def rq_items(self) -> Sequence[Tuple[str, str, float]]:
        return (
            ("rq", "home", self.rq_home),
            ("rq", "draw", self.rq_draw),
            ("rq", "away", self.rq_away),
        )


@dataclass
class MarketQuote:
    market_id: str
    question: str
    category: str
    outcome: str
    probability: Optional[float]
    token_id: Optional[str] = None
    bid: Optional[float] = None
    ask: Optional[float] = None
    spread: Optional[float] = None
    liquidity: Optional[float] = None
    volume: Optional[float] = None
    sports_type: Optional[str] = None
    line: Optional[float] = None
    period: Optional[str] = None
    side: Optional[str] = None
    entity: Optional[str] = None
    scope: Optional[str] = None
    horizon: Optional[str] = None
    causal_layer: Optional[str] = None
    model_weight: Optional[float] = None
    is_complement: bool = False
    active: Optional[bool] = None
    closed: Optional[bool] = None
    accepting_orders: Optional[bool] = None
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EventMarketMatrix:
    match_id: str
    home: str
    away: str
    event_id: Optional[str] = None
    event_slug: Optional[str] = None
    markets: List[MarketQuote] = field(default_factory=list)
    raw_event: Dict[str, Any] = field(default_factory=dict)

    def quotes(self, category: Optional[str] = None) -> Iterable[MarketQuote]:
        for quote in self.markets:
            if category is None or quote.category == category:
                yield quote

    def best_quote(self, category: str, outcome: str) -> Optional[MarketQuote]:
        outcome_key = normalize_key(outcome)
        candidates = [
            quote
            for quote in self.quotes(category)
            if normalize_key(quote.outcome) == outcome_key
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda q: q.probability or -1.0)

    def implied_probability(self, category: str, outcome: str) -> Optional[float]:
        quote = self.best_quote(category, outcome)
        return quote.probability if quote else None

    def liquidity_snapshot(self) -> Tuple[Optional[float], Optional[float]]:
        spreads = [q.spread for q in self.markets if q.spread is not None]
        liquidities = [q.liquidity for q in self.markets if q.liquidity is not None]
        avg_spread = sum(spreads) / len(spreads) if spreads else None
        total_liquidity = sum(liquidities) if liquidities else None
        return avg_spread, total_liquidity


@dataclass
class TeamFacts:
    match_id: str
    source: str
    home_summary: str
    away_summary: str
    tactical_notes: str = ""
    motivation_notes: str = ""
    injuries: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    confidence_adjustment: float = 0.0
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Branch:
    match_id: str
    play: str
    outcome: str
    condition: str
    probability: Optional[float]
    source: str
    tags: List[str] = field(default_factory=list)


@dataclass
class Selection:
    match_id: str
    home: str
    away: str
    play: str
    outcome: str
    condition: str
    probability: float
    sp: float
    fair_odds: float
    break_even: float
    edge: float
    kelly: float
    confidence: float
    risk_label: str
    tags: List[str] = field(default_factory=list)
    source: str = ""

    @property
    def key(self) -> str:
        return f"{self.match_id}:{self.play}:{self.outcome}"


@dataclass
class Combo:
    name: str
    selections: List[Selection]
    probability: float
    odds: float
    expected_return: float
    combo_type: str
    kelly: float = 0.0
    risk_reward: float = 0.0
    stake: float = 0.0
    payout: float = 0.0
    profit: float = 0.0
    deletion_reason: Optional[str] = None

    @property
    def selection_text(self) -> str:
        return " × ".join(
            f"{s.match_id} {s.home}vs{s.away} {s.play}:{s.outcome}"
            for s in self.selections
        )


@dataclass
class MatchAnalysis:
    match: MatchSP
    matrix: EventMarketMatrix
    facts: TeamFacts
    branches: List[Branch]
    selections: List[Selection]
    deleted_paths: List[str] = field(default_factory=list)


def normalize_key(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value)
    asciiish = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return "".join(ch.lower() for ch in asciiish if ch.isalnum())
