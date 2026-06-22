"""Tests for core/knowledge_graph.py.

Coverage:
- Round-trip: save → load preserves all fields.
- Upsert: strength fields overwritten; qualitative fields preserved.
- get_team: canonical key; alias-aware lookup; unknown returns None.
- load_kg: missing file returns {}; corrupt JSON returns {}.
"""
import json
import tempfile
from pathlib import Path

import pytest

from ball_quant.core.knowledge_graph import (
    Team,
    load_kg,
    save_kg,
    get_team,
    upsert_team,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_team(name="brazil", **kwargs) -> Team:
    defaults = dict(
        name=name,
        aliases=["Brasil"],
        group="D",
        strength_win=0.18,
        strength_advance=0.85,
        strength_rank=1,
        recent_form="WWWDW",
        key_players=["Vinicius"],
        injuries=[],
        tactical_notes="High press",
        updated_at="2026-06-01T00:00:00Z",
    )
    defaults.update(kwargs)
    return Team(**defaults)


# ---------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------

def test_roundtrip_empty_kg(tmp_path):
    """save_kg then load_kg on an empty store produces empty dict."""
    path = tmp_path / "teams.json"
    save_kg({}, path)
    kg = load_kg(path)
    assert kg == {}


def test_roundtrip_preserves_all_fields(tmp_path):
    """All Team fields survive a save/load cycle."""
    path = tmp_path / "teams.json"
    t = _make_team()
    save_kg({"brazil": t}, path)

    kg = load_kg(path)
    assert "brazil" in kg
    got = kg["brazil"]
    assert got.name == "brazil"
    assert got.aliases == ["Brasil"]
    assert got.group == "D"
    assert abs(got.strength_win - 0.18) < 1e-9
    assert abs(got.strength_advance - 0.85) < 1e-9
    assert got.strength_rank == 1
    assert got.recent_form == "WWWDW"
    assert got.key_players == ["Vinicius"]
    assert got.tactical_notes == "High press"
    assert got.updated_at == "2026-06-01T00:00:00Z"


def test_roundtrip_multiple_teams(tmp_path):
    """Multiple teams survive round-trip with correct keys."""
    path = tmp_path / "teams.json"
    brazil = _make_team("brazil", strength_win=0.18, strength_rank=1)
    spain = _make_team("spain", strength_win=0.14, strength_rank=2, aliases=["España"])
    save_kg({"brazil": brazil, "spain": spain}, path)
    kg = load_kg(path)
    assert len(kg) == 2
    assert kg["spain"].aliases == ["España"]


# ---------------------------------------------------------------------------
# load_kg edge cases
# ---------------------------------------------------------------------------

def test_load_kg_missing_file_returns_empty(tmp_path):
    """load_kg on a nonexistent path returns {} instead of raising."""
    path = tmp_path / "nonexistent.json"
    assert load_kg(path) == {}


def test_load_kg_corrupt_json_returns_empty(tmp_path):
    """Corrupt JSON file returns {} instead of raising."""
    path = tmp_path / "teams.json"
    path.write_text("{not valid json}", encoding="utf-8")
    assert load_kg(path) == {}


def test_load_kg_skips_corrupt_records(tmp_path):
    """load_kg tolerates a bad record and loads good ones."""
    path = tmp_path / "teams.json"
    # Write valid brazil record and one with an unexpected required field issue
    payload = {
        "brazil": {"name": "brazil", "aliases": [], "group": "D",
                   "strength_win": 0.18, "strength_advance": 0.8,
                   "strength_rank": 1, "recent_form": None, "key_players": [],
                   "injuries": [], "tactical_notes": None, "updated_at": None},
        "bad_team": {"this_field_is_wrong": 999},  # will raise TypeError → skip
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    kg = load_kg(path)
    assert "brazil" in kg
    # bad_team is skipped — only brazil is loaded
    assert "bad_team" not in kg


# ---------------------------------------------------------------------------
# get_team
# ---------------------------------------------------------------------------

def test_get_team_by_canonical_key():
    """get_team finds a team by its canonical name."""
    kg = {"brazil": _make_team("brazil")}
    team = get_team("brazil", kg)
    assert team is not None
    assert team.name == "brazil"


def test_get_team_alias_aware_chinese():
    """get_team resolves a Chinese alias through normalize_team."""
    kg = {"brazil": _make_team("brazil")}
    # "巴西" → normalize_team → "brazil"
    team = get_team("巴西", kg)
    assert team is not None
    assert team.name == "brazil"


def test_get_team_alias_field_scan():
    """get_team scans the aliases list for novel spellings not in TEAM_ALIASES."""
    team_obj = _make_team("argentina", aliases=["Argentine", "Argentinia"])
    kg = {"argentina": team_obj}
    found = get_team("Argentine", kg)
    assert found is not None
    assert found.name == "argentina"


def test_get_team_unknown_returns_none():
    """get_team returns None for teams not in the KG."""
    kg = {"brazil": _make_team("brazil")}
    assert get_team("unknownteamxyz", kg) is None


# ---------------------------------------------------------------------------
# upsert_team — merge rules
# ---------------------------------------------------------------------------

def test_upsert_new_team_inserted():
    """upsert_team inserts a brand new team into the KG."""
    kg = {}
    t = _make_team("brazil")
    upsert_team(t, kg)
    assert "brazil" in kg
    assert kg["brazil"].strength_win == 0.18


def test_upsert_strength_overwritten():
    """upsert_team overwrites strength fields even when qualitative fields differ."""
    kg = {"brazil": _make_team("brazil", strength_win=0.18, strength_rank=1)}
    updated = Team(
        name="brazil",
        strength_win=0.22,
        strength_rank=2,
        strength_advance=0.90,
        updated_at="2026-06-10T00:00:00Z",
    )
    upsert_team(updated, kg)
    assert abs(kg["brazil"].strength_win - 0.22) < 1e-9
    assert kg["brazil"].strength_rank == 2
    assert abs(kg["brazil"].strength_advance - 0.90) < 1e-9


def test_upsert_preserves_qualitative_fields():
    """Strength update does NOT clobber analyst-entered qualitative fields."""
    original = _make_team(
        "brazil",
        recent_form="WWWDW",
        key_players=["Vinicius"],
        injuries=["Neymar out"],
        tactical_notes="High press",
        group="D",
    )
    kg = {"brazil": original}

    # Automated refresh — only carries strength, no qualitative data
    strength_only = Team(
        name="brazil",
        strength_win=0.20,
        strength_rank=1,
        strength_advance=0.88,
        updated_at="2026-06-15T00:00:00Z",
    )
    upsert_team(strength_only, kg)

    result = kg["brazil"]
    assert result.recent_form == "WWWDW"        # preserved
    assert result.key_players == ["Vinicius"]   # preserved
    assert result.injuries == ["Neymar out"]    # preserved
    assert result.tactical_notes == "High press"  # preserved
    assert result.group == "D"                  # preserved
    assert abs(result.strength_win - 0.20) < 1e-9  # overwritten


def test_upsert_aliases_merged_deduped():
    """upsert merges alias lists without creating duplicates."""
    kg = {"brazil": _make_team("brazil", aliases=["Brasil"])}
    updated = Team(name="brazil", aliases=["Brasil", "BRA"])
    upsert_team(updated, kg)
    assert "Brasil" in kg["brazil"].aliases
    assert "BRA" in kg["brazil"].aliases
    # "Brasil" should not appear twice
    assert kg["brazil"].aliases.count("Brasil") == 1
