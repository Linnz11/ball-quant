"""Tests for ball_quant.adapters.http — deterministic, offline."""
from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from ball_quant.adapters.http import HttpError, build_url, get_json, get_text


# ---------------------------------------------------------------------------
# build_url
# ---------------------------------------------------------------------------

class TestBuildUrl:
    def test_simple_path_no_params(self):
        assert build_url("https://example.com", "/fixtures") == "https://example.com/fixtures"

    def test_path_without_leading_slash_gets_one(self):
        assert build_url("https://example.com", "fixtures") == "https://example.com/fixtures"

    def test_base_url_trailing_slash_stripped(self):
        assert build_url("https://example.com/", "/v1") == "https://example.com/v1"

    def test_params_encoded_as_query_string(self):
        url = build_url("https://example.com", "/search", {"q": "hello world", "n": 5})
        assert "q=hello+world" in url or "q=hello%20world" in url
        assert "n=5" in url

    def test_none_params_excluded(self):
        url = build_url("https://example.com", "/x", {"a": 1, "b": None})
        assert "b" not in url
        assert "a=1" in url

    def test_empty_params_dict_no_query_string(self):
        url = build_url("https://example.com", "/x", {})
        assert "?" not in url

    def test_none_params_no_query_string(self):
        url = build_url("https://example.com", "/x", None)
        assert "?" not in url

    def test_list_param_encoded_with_doseq(self):
        url = build_url("https://example.com", "/x", {"ids": ["1", "2"]})
        assert "ids=1" in url
        assert "ids=2" in url


# ---------------------------------------------------------------------------
# get_json — cache hit (no network)
# ---------------------------------------------------------------------------

class TestGetJsonCache:
    def test_returns_cached_content_without_network(self, tmp_path: Path):
        payload = {"result": "cached"}
        cache_file = tmp_path / "cache.json"
        cache_file.write_text(json.dumps(payload), encoding="utf-8")

        # If network were called, urlopen would raise — but it must not be called.
        with patch("ball_quant.adapters.http.request.urlopen") as mock_open:
            result = get_json("https://nowhere.invalid", "/path", cache_path=cache_file)

        mock_open.assert_not_called()
        assert result == payload

    def test_cache_returns_deserialized_object(self, tmp_path: Path):
        payload = [1, 2, {"a": True}]
        cache_file = tmp_path / "data.json"
        cache_file.write_text(json.dumps(payload), encoding="utf-8")

        result = get_json("https://x.invalid", "/y", cache_path=cache_file)
        assert result == payload


# ---------------------------------------------------------------------------
# get_json — cache miss: monkeypatched fetch
# ---------------------------------------------------------------------------

def _make_mock_response(body: str):
    """Return a mock context-manager response that yields body bytes."""
    mock_resp = MagicMock()
    mock_resp.read.return_value = body.encode("utf-8")
    cm = MagicMock()
    cm.__enter__ = MagicMock(return_value=mock_resp)
    cm.__exit__ = MagicMock(return_value=False)
    return cm


class TestGetJsonFetch:
    def test_fetch_and_return_parsed_json(self, tmp_path: Path):
        payload = {"fetched": True}
        with patch("ball_quant.adapters.http.request.urlopen", return_value=_make_mock_response(json.dumps(payload))):
            with patch("ball_quant.adapters.http.time.sleep"):  # skip real sleep
                result = get_json("https://api.test", "/data")
        assert result == payload

    def test_fetch_writes_cache_file_when_cache_path_given(self, tmp_path: Path):
        payload = {"saved": 42}
        cache_file = tmp_path / "sub" / "out.json"

        with patch("ball_quant.adapters.http.request.urlopen", return_value=_make_mock_response(json.dumps(payload))):
            with patch("ball_quant.adapters.http.time.sleep"):
                get_json("https://api.test", "/data", cache_path=cache_file)

        assert cache_file.exists()
        assert json.loads(cache_file.read_text(encoding="utf-8")) == payload

    def test_network_failure_raises_http_error(self):
        with patch("ball_quant.adapters.http.request.urlopen", side_effect=OSError("timeout")):
            with pytest.raises(HttpError):
                get_json("https://bad.invalid", "/x")

    def test_invalid_json_response_raises_http_error(self):
        with patch("ball_quant.adapters.http.request.urlopen", return_value=_make_mock_response("not json {")):
            with pytest.raises(HttpError):
                get_json("https://api.test", "/bad")


# ---------------------------------------------------------------------------
# get_text — monkeypatched urlopen
# ---------------------------------------------------------------------------

class TestGetText:
    def test_returns_body_as_string(self):
        body = "Hello, world!"
        with patch("ball_quant.adapters.http.request.urlopen", return_value=_make_mock_response(body)):
            with patch("ball_quant.adapters.http.time.sleep"):
                result = get_text("https://api.test", "/text")
        assert result == body

    def test_network_failure_raises_http_error(self):
        with patch("ball_quant.adapters.http.request.urlopen", side_effect=ConnectionError("refused")):
            with pytest.raises(HttpError):
                get_text("https://bad.invalid", "/x")

    def test_unicode_body_decoded_correctly(self):
        body = "日本語テスト"
        with patch("ball_quant.adapters.http.request.urlopen", return_value=_make_mock_response(body)):
            with patch("ball_quant.adapters.http.time.sleep"):
                result = get_text("https://api.test", "/unicode")
        assert result == body
