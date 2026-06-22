"""knowledge_graph.py — team knowledge store backed by data/kg/teams.json.

The KG is a thin, stdlib-only JSON store.  Each entry is a Team dataclass
holding both futures-derived strength signals (updated frequently) and
qualitative notes (updated infrequently by analysts).  The merge rule is
intentional: strength fields are overwritten on each futures refresh; every
other field is preserved so analyst notes survive automated re-ingestion.

Relationship model is minimal for now — group membership lives on Team.group.
H2H / schedule edges are a TODO once the schedule store is wired in.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional

from ball_quant.core.match_join import normalize_team

_DEFAULT_KG_PATH = Path("data/kg/teams.json")


@dataclass
class Team:
    """One node in the knowledge graph — canonical team record."""

    # Identity
    name: str                              # canonical key (normalize_team output)
    aliases: List[str] = field(default_factory=list)
    group: Optional[str] = None           # "A" … "H" for WC group stage

    # Futures-derived strength signals (overwritten on every refresh)
    strength_win: Optional[float] = None  # devigged P(win tournament)
    strength_advance: Optional[float] = None  # devigged P(advance from group)
    strength_rank: Optional[int] = None   # rank by strength_win (1 = favourite)

    # Qualitative analyst fields (preserved across automated refreshes)
    recent_form: Optional[str] = None
    key_players: List[str] = field(default_factory=list)
    injuries: List[str] = field(default_factory=list)
    tactical_notes: Optional[str] = None
    updated_at: Optional[str] = None      # ISO-8601 timestamp of last write


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def load_kg(path: Path = _DEFAULT_KG_PATH) -> Dict[str, Team]:
    """Load the KG JSON → dict keyed by canonical team name.

    Returns an empty dict if the file does not exist or is unparseable —
    never crashes so CLI and tests can start cold.
    """
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    kg: Dict[str, Team] = {}
    for name, rec in raw.items():
        try:
            # Unknown future fields are silently ignored — forward-compatible.
            allowed = {f for f in Team.__dataclass_fields__}
            filtered = {k: v for k, v in rec.items() if k in allowed}
            kg[name] = Team(**filtered)
        except (TypeError, ValueError):
            # Corrupt record — skip rather than crash.
            continue
    return kg


def save_kg(teams: Dict[str, Team], path: Path = _DEFAULT_KG_PATH) -> None:
    """Persist the KG dict to JSON.  Creates parent dirs if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {name: asdict(team) for name, team in teams.items()}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Lookup
# ---------------------------------------------------------------------------

def get_team(name: str, kg: Dict[str, Team]) -> Optional[Team]:
    """Alias-aware lookup.  Tries canonical key first, then scans aliases.

    Returns None if the team is not in the KG — never raises.
    """
    key = normalize_team(name)
    if key in kg:
        return kg[key]
    # Linear scan over aliases for names that don't go through TEAM_ALIASES
    # (e.g. a novel romanisation Polymarket uses).
    for team in kg.values():
        for alias in team.aliases:
            if normalize_team(alias) == key:
                return team
    return None


# ---------------------------------------------------------------------------
# Upsert
# ---------------------------------------------------------------------------

def upsert_team(team: Team, kg: Dict[str, Team]) -> None:
    """Merge `team` into `kg` in-place.

    Merge rule — strength fields always overwritten (they come from fresh
    futures data).  Qualitative fields (recent_form / key_players / injuries /
    tactical_notes / aliases / group) are only set when the incoming value is
    non-empty, so automated strength refreshes cannot clobber analyst notes.
    `updated_at` follows the same "keep existing unless incoming is set" rule.
    """
    key = team.name
    if key not in kg:
        kg[key] = team
        return
    existing = kg[key]

    # Overwrite strength (quantitative, refreshed from markets)
    existing.strength_win = team.strength_win
    existing.strength_advance = team.strength_advance
    existing.strength_rank = team.strength_rank

    # Preserve qualitative fields — only update when incoming carries data
    if team.group is not None:
        existing.group = team.group
    if team.aliases:
        # Union of alias lists; avoid duplicates
        merged = list(existing.aliases)
        for a in team.aliases:
            if a not in merged:
                merged.append(a)
        existing.aliases = merged
    if team.recent_form is not None:
        existing.recent_form = team.recent_form
    if team.key_players:
        existing.key_players = team.key_players
    if team.injuries:
        existing.injuries = team.injuries
    if team.tactical_notes is not None:
        existing.tactical_notes = team.tactical_notes
    if team.updated_at is not None:
        existing.updated_at = team.updated_at
