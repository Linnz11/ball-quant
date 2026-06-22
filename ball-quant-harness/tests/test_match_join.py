"""Tests for core/match_join.py — team-name bridging and match pairing."""

import warnings

import pytest

from ball_quant.core.match_join import (
    TEAM_ALIASES,
    normalize_team,
    pair_all,
    pair_one,
)
from ball_quant.models import EventMarketMatrix, TicaiOdds


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ticai(home: str, away: str, match_date: str = "2025-07-01") -> TicaiOdds:
    return TicaiOdds(
        match_id="T001",
        match_date=match_date,
        league="INTL",
        home=home,
        away=away,
        match_num=None,
        spf={"home": 2.0, "draw": 3.0, "away": 3.5},
        handicap_line=None,
        rqspf={},
        correct_score={},
        total_goals={},
        hafu={},
    )


def _make_matrix(home: str, away: str, match_id: str = "M001") -> EventMarketMatrix:
    return EventMarketMatrix(match_id=match_id, home=home, away=away)


# ---------------------------------------------------------------------------
# normalize_team
# ---------------------------------------------------------------------------

class TestNormalizeTeam:
    def test_chinese_argentina(self):
        assert normalize_team("阿根廷") == "argentina"

    def test_english_argentina_lowercased(self):
        # English names go through normalize_key → lowercase alnum
        assert normalize_team("Argentina") == "argentina"

    def test_cape_verde_primary_spelling(self):
        assert normalize_team("佛得角") == "caboverde"

    def test_cape_verde_alternate_spelling(self):
        assert normalize_team("维德角") == "caboverde"

    def test_brazil_chinese(self):
        assert normalize_team("巴西") == "brazil"

    def test_spain_chinese(self):
        assert normalize_team("西班牙") == "spain"

    def test_english_with_accents(self):
        # Polymarket may store "Côte d'Ivoire" — normalize_key strips accent
        assert normalize_team("Côte d'Ivoire") == "cotedivoire"

    def test_alias_table_nonempty(self):
        assert len(TEAM_ALIASES) >= 30


# ---------------------------------------------------------------------------
# pair_one — exact match
# ---------------------------------------------------------------------------

class TestPairOneExact:
    def setup_method(self):
        self.matrices = [
            _make_matrix("Argentina", "Brazil", "M001"),
            _make_matrix("Spain", "Germany", "M002"),
        ]

    def test_argentina_vs_brazil_pairs(self):
        ticai = _make_ticai("阿根廷", "巴西")
        result = pair_one(ticai, self.matrices)
        assert result is not None
        assert result.match_id == "M001"

    def test_does_not_pair_wrong_away(self):
        # 阿根廷 vs 西班牙 should NOT match Argentina vs Brazil
        ticai = _make_ticai("阿根廷", "西班牙")
        result = pair_one(ticai, self.matrices)
        assert result is None

    def test_english_names_pair_directly(self):
        ticai = _make_ticai("Spain", "Germany")
        result = pair_one(ticai, self.matrices)
        assert result is not None
        assert result.match_id == "M002"

    def test_no_matrices_returns_none(self):
        ticai = _make_ticai("阿根廷", "巴西")
        assert pair_one(ticai, []) is None


# ---------------------------------------------------------------------------
# pair_one — swapped home/away fallback
# ---------------------------------------------------------------------------

class TestPairOneSwapped:
    def test_swapped_returns_matrix_and_warns(self):
        # Matrix has Brazil as home, Argentina as away — ticai has them reversed
        matrix = _make_matrix("Brazil", "Argentina", "M001")
        ticai = _make_ticai("阿根廷", "巴西")

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = pair_one(ticai, [matrix])

        assert result is not None
        assert result.match_id == "M001"
        # Must have emitted a UserWarning about the swap
        swap_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
        assert len(swap_warnings) == 1
        assert "swapped" in str(swap_warnings[0].message).lower()

    def test_exact_preferred_over_swapped(self):
        exact = _make_matrix("Argentina", "Brazil", "exact")
        swapped = _make_matrix("Brazil", "Argentina", "swapped")
        ticai = _make_ticai("阿根廷", "巴西")

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = pair_one(ticai, [swapped, exact])

        assert result is not None
        assert result.match_id == "exact"
        # No warnings — exact hit was found
        swap_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
        assert len(swap_warnings) == 0


# ---------------------------------------------------------------------------
# pair_one — date proximity
# ---------------------------------------------------------------------------

class TestPairOneDate:
    def test_date_within_tolerance_pairs(self):
        matrix = EventMarketMatrix(
            match_id="M001",
            home="Argentina",
            away="Brazil",
            raw_event={"start_date": "2025-07-01"},
        )
        ticai = _make_ticai("阿根廷", "巴西", match_date="2025-07-02")
        result = pair_one(ticai, [matrix], date_tolerance_days=1)
        assert result is not None

    def test_date_outside_tolerance_skipped(self):
        matrix = EventMarketMatrix(
            match_id="M001",
            home="Argentina",
            away="Brazil",
            raw_event={"start_date": "2025-06-20"},
        )
        ticai = _make_ticai("阿根廷", "巴西", match_date="2025-07-02")
        result = pair_one(ticai, [matrix], date_tolerance_days=1)
        assert result is None

    def test_missing_matrix_date_is_permissive(self):
        # No date in raw_event — should still pair
        matrix = _make_matrix("Argentina", "Brazil")
        ticai = _make_ticai("阿根廷", "巴西", match_date="2025-07-02")
        result = pair_one(ticai, [matrix], date_tolerance_days=1)
        assert result is not None


# ---------------------------------------------------------------------------
# pair_all
# ---------------------------------------------------------------------------

class TestPairAll:
    def test_one_matched_one_unmatched(self):
        matrices = [_make_matrix("Argentina", "Brazil", "M001")]

        ticai_argentina = _make_ticai("阿根廷", "巴西")
        ticai_morocco = _make_ticai("摩洛哥", "塞内加尔")  # no matrix for this

        matched, unmatched = pair_all(
            [ticai_argentina, ticai_morocco], matrices
        )

        assert len(matched) == 1
        assert matched[0][0] is ticai_argentina
        assert matched[0][1].match_id == "M001"

        assert len(unmatched) == 1
        assert unmatched[0] is ticai_morocco

    def test_all_matched(self):
        matrices = [
            _make_matrix("Argentina", "Brazil", "M001"),
            _make_matrix("Spain", "Germany", "M002"),
        ]
        ticai_list = [_make_ticai("阿根廷", "巴西"), _make_ticai("西班牙", "德国")]
        matched, unmatched = pair_all(ticai_list, matrices)
        assert len(matched) == 2
        assert len(unmatched) == 0

    def test_none_matched(self):
        matrices = [_make_matrix("Japan", "Australia", "M001")]
        ticai_list = [_make_ticai("阿根廷", "巴西")]
        matched, unmatched = pair_all(ticai_list, matrices)
        assert len(matched) == 0
        assert len(unmatched) == 1

    def test_unmapped_chinese_name_goes_to_unmatched_not_crash(self):
        # "未知队伍" is not in TEAM_ALIASES — must not raise, just unmatched
        matrices = [_make_matrix("Unknown FC", "Another FC", "M001")]
        ticai = _make_ticai("未知队伍", "另一队伍")
        matched, unmatched = pair_all([ticai], matrices)
        assert len(matched) == 0
        assert len(unmatched) == 1

    def test_empty_inputs(self):
        matched, unmatched = pair_all([], [])
        assert matched == []
        assert unmatched == []
