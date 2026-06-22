"""Shared pytest fixtures for ball-quant test suite."""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Generator

import pytest

from ball_quant.models import EventMarketMatrix, MarketQuote, MatchSP


@pytest.fixture
def tmp_store(tmp_path: Path) -> Path:
    """A temporary directory path for store/cache tests."""
    return tmp_path


@pytest.fixture
def cassette_dir() -> Path:
    """Absolute path to the pre-recorded JSON cassettes."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def sample_match() -> MatchSP:
    """Minimal MatchSP fixture reused across test modules."""
    return MatchSP(
        match_id="001",
        date="2026-06-14",
        home="Netherlands",
        away="Japan",
        spf_home=1.55,
        spf_draw=3.9,
        spf_away=5.6,
        handicap=-1,
        rq_home=2.78,
        rq_draw=3.55,
        rq_away=2.05,
    )


@pytest.fixture
def sample_matrix() -> EventMarketMatrix:
    """Minimal EventMarketMatrix fixture reused across test modules."""
    return EventMarketMatrix(
        match_id="001",
        home="Netherlands",
        away="Japan",
        markets=[
            MarketQuote("m1", "winner", "moneyline", "home", 0.62, spread=0.02, liquidity=10000),
            MarketQuote("m1", "winner", "moneyline", "draw", 0.22, spread=0.02, liquidity=10000),
            MarketQuote("m1", "winner", "moneyline", "away", 0.16, spread=0.02, liquidity=10000),
            MarketQuote("m2", "Netherlands -1.5", "handicap", "Netherlands -1.5", 0.34, spread=0.02, liquidity=8000),
            MarketQuote("m3", "Japan +0.5", "handicap", "Japan +0.5", 0.38, spread=0.02, liquidity=8000),
        ],
    )
