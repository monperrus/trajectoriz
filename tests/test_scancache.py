"""Tests for trajectoriz._scancache — the disk-backed probe memo."""
from __future__ import annotations

import pytest

from trajectoriz import _scancache


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path, monkeypatch):
    """Point the cache at a temp dir and reset all process-level state."""
    monkeypatch.setattr("trajectoriz.Path.home", lambda: tmp_path)
    monkeypatch.setattr(_scancache, "_conn", None)
    monkeypatch.setattr(_scancache, "_conn_failed", False)
    monkeypatch.setattr(_scancache, "_mem", {})
    monkeypatch.setattr(_scancache, "_pending", [])
    monkeypatch.setattr(_scancache, "_pending_static", [])
    monkeypatch.setattr(_scancache, "_preloaded", set())
    yield


def test_probe_runs_once_then_is_memoized(tmp_path):
    target = tmp_path / "a.jsonl"
    target.write_text("{}")
    calls = {"n": 0}

    def probe():
        calls["n"] += 1
        return {"cwd": "/repo"}

    assert _scancache.memo(target, "cwd", probe) == {"cwd": "/repo"}
    assert _scancache.memo(target, "cwd", probe) == {"cwd": "/repo"}
    assert calls["n"] == 1


def test_probe_reruns_when_the_file_changes(tmp_path):
    target = tmp_path / "a.jsonl"
    target.write_text("{}")
    values = iter(["first", "second"])

    assert _scancache.memo(target, "cwd", lambda: next(values)) == "first"
    target.write_text('{"changed": true}')  # different size and mtime
    assert _scancache.memo(target, "cwd", lambda: next(values)) == "second"


def test_cached_value_survives_a_new_process(tmp_path):
    target = tmp_path / "a.jsonl"
    target.write_text("{}")
    _scancache.memo(target, "cwd", lambda: "/repo")
    _scancache.flush()

    _scancache.clear_memory_cache()  # as if a fresh process attached
    def boom():
        pytest.fail("probe should have been served from the database")

    assert _scancache.memo(target, "cwd", boom) == "/repo"


def test_missing_file_is_probed_without_caching(tmp_path):
    absent = tmp_path / "gone.jsonl"
    calls = {"n": 0}

    def probe():
        calls["n"] += 1
        return "x"

    assert _scancache.memo(absent, "cwd", probe) == "x"
    assert _scancache.memo(absent, "cwd", probe) == "x"
    assert calls["n"] == 2, "a file with no stat cannot be cache-validated"


def test_preload_serves_probes_without_further_queries(tmp_path):
    target = tmp_path / "a.jsonl"
    target.write_text("{}")
    _scancache.memo(target, "fmt", lambda: "claude")
    _scancache.flush()
    _scancache.clear_memory_cache()

    _scancache.preload("fmt")
    def boom():
        pytest.fail("probe should have been preloaded")

    assert _scancache.memo(target, "fmt", boom) == "claude"


def test_static_values_round_trip(tmp_path):
    assert _scancache.get_static("sess-1", "opencode-first-prompt") is None

    _scancache.put_static("sess-1", "opencode-first-prompt", "hello")
    assert _scancache.get_static("sess-1", "opencode-first-prompt") == "hello"

    _scancache.flush()
    _scancache.clear_memory_cache()
    assert _scancache.get_static("sess-1", "opencode-first-prompt") == "hello"


def test_unwritable_cache_still_returns_probe_results(tmp_path, monkeypatch):
    monkeypatch.setattr(_scancache, "_conn_failed", True)  # as if sqlite failed
    target = tmp_path / "a.jsonl"
    target.write_text("{}")

    assert _scancache.memo(target, "cwd", lambda: "/repo") == "/repo"
    assert _scancache.get_static("k", "kind") is None
    _scancache.put_static("k", "kind", "v")  # must not raise
    _scancache.flush()


def test_unserializable_probe_result_is_returned_not_cached(tmp_path):
    target = tmp_path / "a.jsonl"
    target.write_text("{}")

    result = _scancache.memo(target, "weird", lambda: {1, 2, 3})

    assert result == {1, 2, 3}
    assert not _scancache._pending
