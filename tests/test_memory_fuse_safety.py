"""Tests for the two ways a memory mount used to wedge the machine.

1. A store scan that walks into a FUSE mount. ``~/.local/share/agent_probe``
   is a symlink into a repo on this machine, and that repo can hold a
   trajectoriz memory mount — so a memory daemon scanning its stores ended up
   sending itself (or a sibling daemon) a request from inside a request.
2. A FUSE handler that waits on that scan without a deadline. The kernel
   holds the mount's inode lock while a handler runs, so anything touching
   the directory afterwards blocks in uninterruptible sleep — D state, not
   killable by SIGKILL.
"""
from __future__ import annotations

import json
import threading
import time

import pytest

import trajectoriz as tz


def _mountinfo(tmp_path, *mountpoints: str) -> str:
    """Write a /proc/self/mountinfo lookalike declaring FUSE mountpoints."""
    lines = [
        "23 1 0:22 / /proc rw,relatime - proc proc rw",
    ]
    for i, point in enumerate(mountpoints):
        escaped = point.replace(" ", "\\040")   # as the kernel writes it
        lines.append(
            f"{100 + i} 30 0:{50 + i} / {escaped} ro,relatime "
            f"- fuse MemoryFS ro,user_id=1000"
        )
    path = tmp_path / "mountinfo"
    path.write_text("\n".join(lines) + "\n")
    return str(path)


def test_fuse_mount_points_reads_only_fuse_rows(tmp_path):
    info = _mountinfo(tmp_path, "/repo/memory", "/mnt/with space")
    assert tz.fuse_mount_points(info) == {"/repo/memory", "/mnt/with space"}


def test_fuse_mount_points_survives_a_missing_mountinfo(tmp_path):
    assert tz.fuse_mount_points(str(tmp_path / "nope")) == frozenset()


def test_walk_does_not_descend_into_a_fuse_mount(tmp_path):
    (tmp_path / "store").mkdir()
    (tmp_path / "store" / "a.jsonl").write_text("{}\n")
    (tmp_path / "store" / "sub").mkdir()
    (tmp_path / "store" / "sub" / "b.jsonl").write_text("{}\n")
    (tmp_path / "store" / "memory").mkdir()
    (tmp_path / "store" / "memory" / "trap.jsonl").write_text("{}\n")

    info = _mountinfo(tmp_path, str(tmp_path / "store" / "memory"))
    found = sorted(
        str(p.relative_to(tmp_path / "store"))
        for p in tz.iter_files_outside_fuse(tmp_path / "store", "*.jsonl", info)
    )
    assert found == ["a.jsonl", "sub/b.jsonl"]


def test_walk_does_not_reach_a_fuse_mount_through_a_symlink(tmp_path):
    real = tmp_path / "real"
    (real / "memory").mkdir(parents=True)
    (real / "keep.jsonl").write_text("{}\n")
    (real / "memory" / "trap.jsonl").write_text("{}\n")
    link = tmp_path / "store"
    link.symlink_to(real)

    # The store itself is reached through a symlink, exactly like agent_probe.
    info = _mountinfo(tmp_path, str(real / "memory"))
    found = [p.name for p in tz.iter_files_outside_fuse(link, "*.jsonl", info)]
    assert found == ["keep.jsonl"]


def test_walk_skips_a_base_that_is_itself_a_fuse_mount(tmp_path):
    base = tmp_path / "memory"
    base.mkdir()
    (base / "trap.jsonl").write_text("{}\n")
    info = _mountinfo(tmp_path, str(base))
    assert list(tz.iter_files_outside_fuse(base, "*.jsonl", info)) == []


def test_walk_terminates_on_a_symlink_cycle(tmp_path):
    store = tmp_path / "store"
    store.mkdir()
    (store / "a.jsonl").write_text("{}\n")
    (store / "loop").symlink_to(store)
    info = _mountinfo(tmp_path)
    found = [p.name for p in tz.iter_files_outside_fuse(store, "*.jsonl", info)]
    assert found == ["a.jsonl"]


def test_agent_probe_store_skips_a_memory_mount_inside_it(tmp_path, monkeypatch):
    """The real shape of the deadlock: the store is a symlink into a repo."""
    repo = tmp_path / "repo"
    (repo / "memory").mkdir(parents=True)
    (repo / "sess_journal.jsonl").write_text("{}\n")
    (repo / "memory" / "2024-01-01_claude_x.atif.json").write_text("{}\n")
    (repo / "memory" / "trap.jsonl").write_text("{}\n")
    store = tmp_path / "agent_probe"
    store.symlink_to(repo)

    monkeypatch.setattr(
        tz, "fuse_mount_points", lambda *a, **k: frozenset({str(repo / "memory")})
    )
    found = [p.name for p in tz.iter_agent_probe_trajectories(store)]
    assert found == ["sess_journal.jsonl"]


# ── The handler deadline ─────────────────────────────────────────────────

@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A repo root with one Claude session recorded against it."""
    repo_root = "/tmp/deadline-repo"
    project_dir = tz.claude_project_dir(repo_root, claude_dir=tmp_path / ".claude")
    project_dir.mkdir(parents=True)
    (project_dir / "one.jsonl").write_text(
        json.dumps({
            "sessionId": "one",
            "type": "user",
            "timestamp": "2024-01-01T00:00:00Z",
            "message": {"content": "hello"},
        }) + "\n"
    )
    monkeypatch.setattr("trajectoriz.Path.home", lambda: tmp_path)
    return repo_root


def _hanging_scan(release: threading.Event):
    def scan(cwd):
        release.wait(30)
        return iter(())
    return scan


def test_a_hung_scan_does_not_hang_the_mount(repo, monkeypatch):
    pytest.importorskip("fuse")
    from trajectoriz._memoryfs import MemoryFS

    release = threading.Event()
    monkeypatch.setattr(tz, "iter_local_records", _hanging_scan(release))
    fs = MemoryFS(repo, scan_timeout=0.2)
    try:
        started = time.monotonic()
        entries = fs.readdir("/", None)
        elapsed = time.monotonic() - started
        assert elapsed < 5.0            # the deadline held
        assert entries == [".", "..", "README.md"]
    finally:
        release.set()


def test_a_hung_rescan_serves_the_previous_listing(repo, monkeypatch):
    pytest.importorskip("fuse")
    from trajectoriz._memoryfs import MemoryFS

    fs = MemoryFS(repo, listing_ttl=0.0, scan_timeout=0.2)  # always stale
    first = [n for n in fs.readdir("/", None) if n.endswith(".atif.json")]
    assert len(first) == 1

    release = threading.Event()
    monkeypatch.setattr(tz, "iter_local_records", _hanging_scan(release))
    try:
        started = time.monotonic()
        again = [n for n in fs.readdir("/", None) if n.endswith(".atif.json")]
        assert time.monotonic() - started < 5.0
        assert again == first           # stale, but served
    finally:
        release.set()


def test_a_broken_store_leaves_the_mount_serving(repo, monkeypatch):
    pytest.importorskip("fuse")
    from trajectoriz._memoryfs import MemoryFS

    fs = MemoryFS(repo, listing_ttl=0.0, scan_timeout=1.0)
    first = [n for n in fs.readdir("/", None) if n.endswith(".atif.json")]

    def exploding(cwd):
        raise OSError("store went away")

    monkeypatch.setattr(tz, "iter_local_records", exploding)
    assert [n for n in fs.readdir("/", None) if n.endswith(".atif.json")] == first
