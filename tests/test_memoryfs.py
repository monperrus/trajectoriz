"""Tests for trajectoriz._memoryfs — the FUSE Operations class (no real mount)."""
from __future__ import annotations

import errno
import json

import pytest

fuse = pytest.importorskip("fuse")

from trajectoriz import claude_project_dir
from trajectoriz._memoryfs import MemoryFS, _filename_for


def _write_claude_session(tmp_path, monkeypatch, repo_root, session_name, first_msg):
    project_dir = claude_project_dir(repo_root, claude_dir=tmp_path / ".claude")
    project_dir.mkdir(parents=True, exist_ok=True)
    f = project_dir / f"{session_name}.jsonl"
    f.write_text(
        json.dumps({
            "sessionId": session_name,
            "type": "user",
            "timestamp": "2024-01-01T00:00:00Z",
            "message": {"content": first_msg},
        }) + "\n"
    )
    monkeypatch.setattr("trajectoriz.Path.home", lambda: tmp_path)
    return f


def test_readdir_lists_one_file_per_trajectory(tmp_path, monkeypatch):
    repo_root = "/tmp/my-repo"
    _write_claude_session(tmp_path, monkeypatch, repo_root, "sess-1", "hello")

    fs = MemoryFS(repo_root)
    entries = fs.readdir("/", None)
    assert entries[:2] == [".", ".."]
    assert len(entries) == 3
    assert entries[2].endswith(".atif.json")
    assert "2024-01-01" in entries[2]
    assert "claude" in entries[2]


def test_read_returns_valid_atif_json(tmp_path, monkeypatch):
    repo_root = "/tmp/my-repo"
    _write_claude_session(tmp_path, monkeypatch, repo_root, "sess-1", "hello world")

    fs = MemoryFS(repo_root)
    name = [e for e in fs.readdir("/", None) if e not in (".", "..")][0]
    path = "/" + name

    assert fs.open(path, 0) == 0
    st = fs.getattr(path)
    data = fs.read(path, st["st_size"], 0, 0)
    envelope = json.loads(data)
    assert envelope["schema_version"] == "ATIF-v1.7"
    assert envelope["steps"][0]["message"] == "hello world"


def test_getattr_missing_file_raises_enoent(tmp_path, monkeypatch):
    repo_root = "/tmp/my-repo"
    monkeypatch.setattr("trajectoriz.Path.home", lambda: tmp_path)
    fs = MemoryFS(repo_root)
    with pytest.raises(fuse.FuseOSError) as exc:
        fs.getattr("/does-not-exist.atif.json")
    assert exc.value.errno == errno.ENOENT


def test_write_operations_are_read_only(tmp_path, monkeypatch):
    repo_root = "/tmp/my-repo"
    monkeypatch.setattr("trajectoriz.Path.home", lambda: tmp_path)
    fs = MemoryFS(repo_root)
    with pytest.raises(fuse.FuseOSError) as exc:
        fs.write("/x", b"data", 0, 0)
    assert exc.value.errno == errno.EROFS
    with pytest.raises(fuse.FuseOSError):
        fs.unlink("/x")
    with pytest.raises(fuse.FuseOSError):
        fs.mkdir("/x", 0o755)


def test_filename_stable_for_same_record(tmp_path, monkeypatch):
    repo_root = "/tmp/my-repo"
    _write_claude_session(tmp_path, monkeypatch, repo_root, "sess-1", "hello")
    fs = MemoryFS(repo_root)
    first = sorted(fs._records())
    second = sorted(fs._records())
    assert first == second
