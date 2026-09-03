"""Tests for trajectoriz.cli._ensure_gitignored."""
from __future__ import annotations

import subprocess

from trajectoriz.cli import _ensure_gitignored


def _git(*args, cwd):
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


def test_adds_entry_to_new_gitignore(tmp_path):
    _git("init", "-q", cwd=tmp_path)
    mountpoint = tmp_path / "memory"
    mountpoint.mkdir()

    _ensure_gitignored(mountpoint)

    gitignore = tmp_path / ".gitignore"
    assert gitignore.exists()
    assert gitignore.read_text().splitlines() == ["/memory/"]


def test_appends_to_existing_gitignore(tmp_path):
    _git("init", "-q", cwd=tmp_path)
    (tmp_path / ".gitignore").write_text("__pycache__/\n")
    mountpoint = tmp_path / "memory"
    mountpoint.mkdir()

    _ensure_gitignored(mountpoint)

    assert (tmp_path / ".gitignore").read_text().splitlines() == ["__pycache__/", "/memory/"]


def test_does_not_duplicate_existing_entry(tmp_path):
    _git("init", "-q", cwd=tmp_path)
    (tmp_path / ".gitignore").write_text("/memory/\n")
    mountpoint = tmp_path / "memory"
    mountpoint.mkdir()

    _ensure_gitignored(mountpoint)

    assert (tmp_path / ".gitignore").read_text().splitlines() == ["/memory/"]


def test_noop_outside_git_repo(tmp_path):
    mountpoint = tmp_path / "memory"
    mountpoint.mkdir()

    _ensure_gitignored(mountpoint)

    assert not (tmp_path / ".gitignore").exists()


def test_noop_when_mountpoint_outside_repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git("init", "-q", cwd=repo)
    outside = tmp_path / "outside-memory"
    outside.mkdir()

    _ensure_gitignored(outside)

    assert not (repo / ".gitignore").exists()
