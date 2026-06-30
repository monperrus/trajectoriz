"""Tests for the SQLite FTS5 backend (_fts.py and SqliteBackend)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import trajectoriz as tz
from trajectoriz._fts import build_index, fts_db_path, search_fts, source_from_json
from trajectoriz._search import SqliteBackend


# ── Fixtures ─────────────────────────────────────────────────────────────────


def _write_claude_traj(path: Path, first_msg: str, deep_content: str) -> None:
    """Write a minimal Claude JSONL trajectory."""
    lines = [
        {"type": "user", "sessionId": "s1", "timestamp": "2024-03-01T10:00:00Z",
         "message": {"content": first_msg}},
        {"type": "assistant", "timestamp": "2024-03-01T10:01:00Z", "message": {
            "model": "claude-sonnet-4-6",
            "content": [
                {"type": "tool_use", "id": "c1", "name": "Bash",
                 "input": {"command": "echo hello"}},
            ],
        }},
        {"type": "user", "timestamp": "2024-03-01T10:02:00Z", "message": {
            "content": [
                {"type": "tool_result", "tool_use_id": "c1", "content": deep_content},
            ],
        }},
        {"type": "assistant", "timestamp": "2024-03-01T10:03:00Z", "message": {
            "model": "claude-sonnet-4-6",
            "content": [{"type": "text", "text": "Done."}],
        }},
    ]
    path.write_text("\n".join(json.dumps(l) for l in lines) + "\n")


@pytest.fixture()
def traj_dir(tmp_path: Path) -> Path:
    d = tmp_path / "trajs"
    d.mkdir()
    return d


@pytest.fixture()
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "fts.db"


# ── fts_db_path ───────────────────────────────────────────────────────────────


def test_fts_db_path_default():
    p = fts_db_path()
    assert p.name == "fts.db"
    assert "trajectoriz" in str(p)


def test_fts_db_path_custom(tmp_path):
    p = fts_db_path(cache_dir=tmp_path)
    assert p == tmp_path / "fts.db"


# ── source_from_json ──────────────────────────────────────────────────────────


def test_source_from_json_absolute_path():
    src = source_from_json("/home/user/.claude/projects/foo/bar.jsonl")
    assert isinstance(src, Path)
    assert str(src) == "/home/user/.claude/projects/foo/bar.jsonl"


def test_source_from_json_dict():
    d = {"type": "opencode", "session_id": "abc"}
    src = source_from_json(json.dumps(d))
    assert isinstance(src, dict)
    assert src["session_id"] == "abc"


# ── build_index ───────────────────────────────────────────────────────────────


def test_build_index_basic(traj_dir, db_path):
    f = traj_dir / "a.jsonl"
    _write_claude_traj(f, "fix the login bug", "ERROR: NullPointerException in auth module")
    rec = tz.TrajectoryRecord("cl-aaa", "claude", "2024-03-01T10:00:00Z", "fix the login bug", f)

    indexed, skipped = build_index([rec], db_path=db_path)

    assert indexed == 1
    assert skipped == 0
    assert db_path.exists()


def test_build_index_incremental(traj_dir, db_path):
    f = traj_dir / "b.jsonl"
    _write_claude_traj(f, "refactor the cache", "CacheManager: hit ratio 0.42")
    rec = tz.TrajectoryRecord("cl-bbb", "claude", "2024-03-01T10:00:00Z", "refactor the cache", f)

    indexed1, skipped1 = build_index([rec], db_path=db_path)
    indexed2, skipped2 = build_index([rec], db_path=db_path)

    assert indexed1 == 1 and skipped1 == 0
    assert indexed2 == 0 and skipped2 == 1  # mtime unchanged → skipped


def test_build_index_force_reindexes(traj_dir, db_path):
    f = traj_dir / "c.jsonl"
    _write_claude_traj(f, "cleanup deps", "removing unused packages")
    rec = tz.TrajectoryRecord("cl-ccc", "claude", "2024-03-01T10:00:00Z", "cleanup deps", f)

    build_index([rec], db_path=db_path)
    indexed, skipped = build_index([rec], db_path=db_path, force=True)

    assert indexed == 1
    assert skipped == 0


def test_build_index_multiple_records(traj_dir, db_path):
    recs = []
    for i in range(3):
        f = traj_dir / f"session_{i}.jsonl"
        _write_claude_traj(f, f"task {i}", f"unique_token_{i} content here")
        recs.append(tz.TrajectoryRecord(f"cl-{i:03d}", "claude", "2024-03-01", f"task {i}", f))

    indexed, skipped = build_index(recs, db_path=db_path)
    assert indexed == 3
    assert skipped == 0


# ── search_fts ────────────────────────────────────────────────────────────────


def test_search_fts_finds_deep_match(traj_dir, db_path):
    f = traj_dir / "deep.jsonl"
    _write_claude_traj(f, "investigate performance", "profiler reveals hotspot in rendering pipeline")
    rec = tz.TrajectoryRecord("cl-deep", "claude", "2024-03-01T10:00:00Z", "investigate performance", f)
    build_index([rec], db_path=db_path)

    matches = search_fts(["hotspot"], db_path=db_path)

    assert len(matches) >= 1
    result_rec, step_id, snippet = matches[0]
    assert result_rec.id == "cl-deep"
    assert result_rec.agent == "claude"
    assert "hotspot" in snippet.lower()


def test_search_fts_no_match(traj_dir, db_path):
    f = traj_dir / "nomatch.jsonl"
    _write_claude_traj(f, "build the widget", "widget compiled successfully")
    rec = tz.TrajectoryRecord("cl-nm", "claude", "2024-03-01T10:00:00Z", "build the widget", f)
    build_index([rec], db_path=db_path)

    matches = search_fts(["xyzzy_nonexistent"], db_path=db_path)
    assert matches == []


def test_search_fts_or_terms(traj_dir, db_path):
    f1 = traj_dir / "s1.jsonl"
    f2 = traj_dir / "s2.jsonl"
    _write_claude_traj(f1, "alpha task", "unique_alpha_word result here")
    _write_claude_traj(f2, "beta task", "unique_beta_word result here")
    recs = [
        tz.TrajectoryRecord("cl-s1", "claude", "2024-03-01", "alpha task", f1),
        tz.TrajectoryRecord("cl-s2", "claude", "2024-03-01", "beta task", f2),
    ]
    build_index(recs, db_path=db_path)

    matches = search_fts(["unique_alpha_word", "unique_beta_word"], db_path=db_path)
    ids = {r.id for r, _, _ in matches}
    assert "cl-s1" in ids
    assert "cl-s2" in ids


def test_search_fts_missing_db(tmp_path):
    with pytest.raises(FileNotFoundError, match="FTS index not found"):
        search_fts(["anything"], db_path=tmp_path / "nonexistent.db")


def test_search_fts_returns_step_id(traj_dir, db_path):
    f = traj_dir / "steps.jsonl"
    _write_claude_traj(f, "run diagnostics", "diagnostic_marker found in step")
    rec = tz.TrajectoryRecord("cl-st", "claude", "2024-03-01T10:00:00Z", "run diagnostics", f)
    build_index([rec], db_path=db_path)

    matches = search_fts(["diagnostic_marker"], db_path=db_path)
    assert len(matches) >= 1
    _, step_id, _ = matches[0]
    assert isinstance(step_id, int)
    assert step_id >= 1


def test_search_fts_reconstructs_path_source(traj_dir, db_path):
    f = traj_dir / "pathsrc.jsonl"
    _write_claude_traj(f, "path source test", "path_source_token content")
    rec = tz.TrajectoryRecord("cl-ps", "claude", "2024-03-01", "path source test", f)
    build_index([rec], db_path=db_path)

    matches = search_fts(["path_source_token"], db_path=db_path)
    assert matches
    result_rec, _, _ = matches[0]
    assert isinstance(result_rec.source, Path)
    assert result_rec.source == f


# ── SqliteBackend ─────────────────────────────────────────────────────────────


def test_sqlite_backend_search(traj_dir, db_path):
    f = traj_dir / "backend.jsonl"
    _write_claude_traj(f, "backend test", "backend_unique_marker in tool result")
    rec = tz.TrajectoryRecord("cl-be", "claude", "2024-03-01", "backend test", f)
    build_index([rec], db_path=db_path)

    backend = SqliteBackend(db_path=db_path)
    matches = backend.search([], ["backend_unique_marker"])

    assert len(matches) >= 1
    result_rec, step_id, snippet = matches[0]
    assert result_rec.id == "cl-be"
    assert "backend_unique_marker" in snippet.lower()


def test_sqlite_backend_missing_db_raises(tmp_path):
    backend = SqliteBackend(db_path=tmp_path / "no.db")
    with pytest.raises(NotImplementedError, match="FTS index not found"):
        backend.search([], ["anything"])
