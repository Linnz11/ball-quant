"""
Competition-scoped parameter profiles.

Allows different StrategyParams to be used for different competitions without
touching any code outside the resolver.  The resolver is the single point of
dispatch; callers hold a ParamProfiles and call resolve() per record — they
never need to know which overrides apply.

Design contract:
  - resolve(None)                     -> base (default_overrides on top of DEFAULT_PARAMS)
  - resolve(comp not in profiles)     -> base (same as None)
  - resolve(comp in profiles)         -> base + competition-specific overrides
  - Unknown override key anywhere     -> raises ValueError (via StrategyParams.from_dict)
  - No profiles configured            -> resolve() == DEFAULT_PARAMS (byte-identical path)

JSON schema:
  {"default": {<overrides>}, "by_competition": {"PL": {<overrides>}, ...}}
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, Optional

from ball_quant.core.params import DEFAULT_PARAMS, StrategyParams


def _apply_overrides(base: StrategyParams, overrides: Dict[str, Any]) -> StrategyParams:
    """Apply *overrides* dict onto *base*, validating all keys via from_dict.

    We validate by building a merged dict through StrategyParams.from_dict,
    which raises ValueError on unknown keys.  Then we use dataclasses.replace
    so the frozen dataclass invariants are preserved.
    """
    if not overrides:
        return base
    # Merge: start from current base values, layer overrides on top.
    merged = base.to_dict()
    merged.update(overrides)
    # from_dict raises ValueError on any unknown key — no silent fallback.
    return StrategyParams.from_dict(merged)


@dataclass
class ParamProfiles:
    """Per-competition parameter resolver.

    Attributes
    ----------
    default_overrides:
        Overrides applied on top of DEFAULT_PARAMS for every match unless a
        competition-specific entry also exists (in which case these are still
        the base layer before competition overrides).
    by_competition:
        Map from competition string to its own overrides.  Applied on top of
        the default_overrides-adjusted base.
    """

    default_overrides: Dict[str, Any] = field(default_factory=dict)
    by_competition: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    # Resolution
    # ------------------------------------------------------------------ #

    def resolve(self, competition: Optional[str]) -> StrategyParams:
        """Return the effective StrategyParams for *competition*.

        Resolution order (each layer validated via StrategyParams.from_dict):
          1. DEFAULT_PARAMS  (stdlib frozen dataclass baseline)
          2. self.default_overrides  (global profile adjustments)
          3. self.by_competition[competition]  (only if competition is known)

        When no overrides are configured the result is identical to
        DEFAULT_PARAMS — callers that previously used DEFAULT_PARAMS directly
        see zero behaviour change.
        """
        base = _apply_overrides(DEFAULT_PARAMS, self.default_overrides)
        if competition is not None and competition in self.by_competition:
            return _apply_overrides(base, self.by_competition[competition])
        return base

    # ------------------------------------------------------------------ #
    # Serialisation
    # ------------------------------------------------------------------ #

    def to_json(self, path) -> None:
        """Write profiles to a JSON file.

        Schema: {"default": {...}, "by_competition": {"comp": {...}, ...}}
        """
        payload = {
            "default": self.default_overrides,
            "by_competition": self.by_competition,
        }
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def from_json(cls, path) -> "ParamProfiles":
        """Load profiles from a JSON file written by to_json.

        Raises ValueError on unknown override keys (via resolve validation
        path) so that bad JSON files are caught immediately, not silently
        carried as dead weight.
        """
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        default_overrides: Dict[str, Any] = raw.get("default", {})
        by_competition: Dict[str, Dict[str, Any]] = raw.get("by_competition", {})

        # Validate all override dicts eagerly so callers get a clear error at
        # load time rather than mid-backtest on the first affected record.
        _apply_overrides(DEFAULT_PARAMS, default_overrides)
        for comp, overrides in by_competition.items():
            base = _apply_overrides(DEFAULT_PARAMS, default_overrides)
            _apply_overrides(base, overrides)

        return cls(
            default_overrides=default_overrides,
            by_competition=by_competition,
        )
