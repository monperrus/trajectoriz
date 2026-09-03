"""Tests for the caching contract that keeps the memory filesystem usable.

A store scan is expensive (it sniffs every trajectory file on the machine), so
these tests pin down *how often* it happens: once per listing, never per read.
"""
from __future__ import annotations

import errno
import json
import os

import pytest

fuse = pytest.importorskip("fuse")

import trajectoriz as tz
from trajectoriz import claude_project_dir
from trajectoriz._memoryfs import MemoryFS


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A repo root with two Claude sessions, plus a scan counter."""
    repo_root = "/tmp/perf-repo"
    project_dir = claude_project_dir(repo_root, claude_dir=tmp_path / ".claude")
    project_dir.mkdir(parents=True)
    for name in ("one", "two"):
        (project_dir / f"{name}.jsonl").write_text(
            json.dumps({
                "sessionId": name,
                "type": "user",
                "timestamp": "2024-01-01T00:00:00Z",
                "message": {"content": f"hello from {name}"},
            }) + "\n"
        )
    monkeypatch.setattr("trajectoriz.Path.home", lambda: tmp_path)

    scans = {"n": 0}
    real = tz.iter_local_records

    def counting(cwd):
        scans["n"] += 1
        return real(cwd)

    monkeypatch.setattr(tz, "iter_local_records", counting)
    return repo_root, scans


def _sessions(fs: MemoryFS) -> list[str]:
    """The trajectory files of a mount, excluding . .. and README.md."""
    return [
        n for n in fs.readdir("/", None)
        if n not in (".", "..") and n.endswith(".atif.json")
    ]


def _read_all(fs: MemoryFS, name: str, chunk: int = 64) -> bytes:
    """Read a file the way the kernel does: repeated fixed-size requests."""
    path = "/" + name
    size = fs.getattr(path)["st_size"]
    fh = fs.open(path, os.O_RDONLY)
    try:
        out = b""
        while len(out) < size:
            piece = fs.read(path, chunk, len(out), fh)
            if not piece:
                break
            out += piece
        return out
    finally:
        fs.release(path, fh)


def test_browsing_the_whole_tree_scans_the_stores_once(repo):
    repo_root, scans = repo
    fs = MemoryFS(repo_root, listing_ttl=60.0)

    names = _sessions(fs)
    assert len(names) == 2
    for name in names:
        fs.getattr("/" + name)
        _read_all(fs, name)

    assert scans["n"] == 1, "listing should be scanned once, not per operation"


def test_reads_are_served_without_rescanning(repo):
    repo_root, scans = repo
    fs = MemoryFS(repo_root, listing_ttl=60.0)
    name = _sessions(fs)[0]
    before = scans["n"]

    for _ in range(5):
        _read_all(fs, name)

    assert scans["n"] == before


def test_read_returns_the_whole_payload_across_chunks(repo):
    repo_root, _ = repo
    fs = MemoryFS(repo_root, listing_ttl=60.0)
    name = _sessions(fs)[0]

    data = _read_all(fs, name, chunk=7)  # deliberately awkward chunk size

    envelope = json.loads(data)
    assert envelope["schema_version"] == "ATIF-v1.7"
    assert len(data) == fs.getattr("/" + name)["st_size"]


def test_stale_listing_is_refreshed_so_new_sessions_appear(repo, tmp_path):
    repo_root, _ = repo
    fs = MemoryFS(repo_root, listing_ttl=0.0)  # always stale
    assert len(_sessions(fs)) == 2

    project_dir = claude_project_dir(repo_root, claude_dir=tmp_path / ".claude")
    (project_dir / "three.jsonl").write_text(
        json.dumps({
            "sessionId": "three", "type": "user",
            "timestamp": "2024-01-02T00:00:00Z",
            "message": {"content": "a new session"},
        }) + "\n"
    )

    assert len(_sessions(fs)) == 3


def test_content_cache_is_bounded(repo):
    repo_root, _ = repo
    fs = MemoryFS(repo_root, listing_ttl=60.0, cache_bytes=1)  # evict aggressively
    names = _sessions(fs)

    for name in names:
        assert json.loads(_read_all(fs, name))["schema_version"] == "ATIF-v1.7"

    assert len(fs._content) == 1, "cache should hold only the most recent payload"


def test_open_for_writing_is_refused(repo):
    repo_root, _ = repo
    fs = MemoryFS(repo_root, listing_ttl=60.0)
    name = _sessions(fs)[0]

    with pytest.raises(fuse.FuseOSError) as exc:
        fs.open("/" + name, os.O_WRONLY)
    assert exc.value.errno == errno.EROFS


def test_missing_file_does_not_rescan_on_every_probe(repo):
    repo_root, scans = repo
    fs = MemoryFS(repo_root, listing_ttl=60.0)
    fs.readdir("/", None)
    before = scans["n"]

    for _ in range(10):
        with pytest.raises(fuse.FuseOSError):
            fs.getattr("/nope.atif.json")

    assert scans["n"] - before <= 1


def test_readme_explains_the_directory(repo):
    repo_root, _ = repo
    fs = MemoryFS(repo_root, listing_ttl=60.0)

    assert "README.md" in fs.readdir("/", None)
    data = _read_all(fs, "README.md")
    text = data.decode()

    assert repo_root in text
    assert "trajectoriz-cli search" in text, "should point at the fast way to locate a session"
    assert "ATIF" in text
    # st_size is a byte count, and the text is not pure ASCII
    assert len(data) == fs.getattr("/README.md")["st_size"]


def test_readme_is_served_without_scanning_the_stores(repo):
    repo_root, scans = repo
    fs = MemoryFS(repo_root, listing_ttl=60.0)

    fs.getattr("/README.md")
    _read_all(fs, "README.md")

    assert scans["n"] == 0, "the README needs no store scan"
