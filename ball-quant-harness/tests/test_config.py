"""
Tests for config/settings.py — defaults, JSON override, env override, unknown key.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest

from ball_quant.config.settings import Settings


class TestSettingsDefaults:
    def test_default_store_root(self):
        s = Settings.load()
        assert s.store_root == "data/store"

    def test_default_cache_dir(self):
        s = Settings.load()
        assert s.cache_dir == "data/cache"

    def test_default_reports_dir(self):
        s = Settings.load()
        assert s.reports_dir == "reports"

    def test_default_live_reports_dir(self):
        s = Settings.load()
        assert s.live_reports_dir == "reports/live"

    def test_default_budget(self):
        s = Settings.load()
        assert s.default_budget == 200.0

    def test_default_bankroll(self):
        s = Settings.load()
        assert s.default_bankroll == 1000.0

    def test_default_timezone(self):
        s = Settings.load()
        assert s.timezone == "Asia/Shanghai"

    def test_default_world_cup_tag_id(self):
        s = Settings.load()
        assert s.world_cup_tag_id == 102232

    def test_default_http_timeout(self):
        s = Settings.load()
        assert s.http_timeout == 12

    def test_default_log_level(self):
        s = Settings.load()
        assert s.log_level == "INFO"

    def test_default_cache_ttl(self):
        s = Settings.load()
        assert s.cache_ttl_seconds == 3600

    def test_to_dict_contains_all_fields(self):
        s = Settings.load()
        d = s.to_dict()
        assert "store_root" in d
        assert "default_budget" in d
        assert "log_level" in d
        assert len(d) == 11  # all 11 known fields


class TestSettingsJsonOverride:
    def test_json_overrides_store_root(self, tmp_path):
        cfg = {"store_root": "/tmp/custom_store"}
        p = tmp_path / "settings.json"
        p.write_text(json.dumps(cfg), encoding="utf-8")
        s = Settings.load(path=p)
        assert s.store_root == "/tmp/custom_store"
        # Other fields keep defaults.
        assert s.default_budget == 200.0

    def test_json_overrides_budget_as_float(self, tmp_path):
        cfg = {"default_budget": 500.0}
        p = tmp_path / "settings.json"
        p.write_text(json.dumps(cfg), encoding="utf-8")
        s = Settings.load(path=p)
        assert s.default_budget == 500.0

    def test_json_overrides_int_field(self, tmp_path):
        cfg = {"world_cup_tag_id": 999}
        p = tmp_path / "settings.json"
        p.write_text(json.dumps(cfg), encoding="utf-8")
        s = Settings.load(path=p)
        assert s.world_cup_tag_id == 999

    def test_unknown_json_key_raises(self, tmp_path):
        cfg = {"definitely_not_a_real_key": "oops"}
        p = tmp_path / "settings.json"
        p.write_text(json.dumps(cfg), encoding="utf-8")
        with pytest.raises(ValueError, match="Unknown key"):
            Settings.load(path=p)

    def test_multiple_unknown_keys_listed_in_error(self, tmp_path):
        cfg = {"bad_key_1": 1, "bad_key_2": 2}
        p = tmp_path / "settings.json"
        p.write_text(json.dumps(cfg), encoding="utf-8")
        with pytest.raises(ValueError) as exc_info:
            Settings.load(path=p)
        msg = str(exc_info.value)
        assert "bad_key_1" in msg or "bad_key_2" in msg


class TestSettingsEnvOverride:
    """Env vars must take precedence over both defaults and JSON file."""

    def _patch_env(self, monkeypatch, **kwargs):
        for key, val in kwargs.items():
            monkeypatch.setenv(key, str(val))

    def test_env_overrides_store_root(self, monkeypatch):
        self._patch_env(monkeypatch, BALLQ_STORE_ROOT="/env/store")
        s = Settings.load()
        assert s.store_root == "/env/store"

    def test_env_overrides_budget_as_float(self, monkeypatch):
        self._patch_env(monkeypatch, BALLQ_DEFAULT_BUDGET="999.5")
        s = Settings.load()
        assert s.default_budget == 999.5

    def test_env_overrides_world_cup_tag_id_as_int(self, monkeypatch):
        self._patch_env(monkeypatch, BALLQ_WORLD_CUP_TAG_ID="12345")
        s = Settings.load()
        assert s.world_cup_tag_id == 12345

    def test_env_overrides_log_level(self, monkeypatch):
        self._patch_env(monkeypatch, BALLQ_LOG_LEVEL="DEBUG")
        s = Settings.load()
        assert s.log_level == "DEBUG"

    def test_env_wins_over_json(self, monkeypatch, tmp_path):
        cfg = {"store_root": "/json/store"}
        p = tmp_path / "settings.json"
        p.write_text(json.dumps(cfg), encoding="utf-8")
        self._patch_env(monkeypatch, BALLQ_STORE_ROOT="/env/store")
        s = Settings.load(path=p)
        assert s.store_root == "/env/store"

    def test_env_overrides_cache_ttl_as_int(self, monkeypatch):
        self._patch_env(monkeypatch, BALLQ_CACHE_TTL_SECONDS="7200")
        s = Settings.load()
        assert s.cache_ttl_seconds == 7200
