"""trajectoriz: locate agent trajectory files on the local machine."""

import os
import re
import sqlite3
from pathlib import Path


def iter_claude_trajectories(claude_dir=None):
    """Yield all Claude Code trajectory JSONL paths."""
    d = Path(claude_dir) if claude_dir else Path.home() / ".claude"
    if d.is_dir():
        yield from sorted(d.glob("projects/**/*.jsonl"))


def claude_project_dir(repo_root: str, claude_dir=None) -> Path:
    """Return the Claude Code project directory for a given repo root."""
    d = Path(claude_dir) if claude_dir else Path.home() / ".claude"
    slug = re.sub(r"[^a-zA-Z0-9]", "-", repo_root)
    return d / "projects" / slug


def iter_claude_project_trajectories(repo_root: str, claude_dir=None):
    """Yield Claude Code trajectory JSONL paths for a specific project."""
    d = claude_project_dir(repo_root, claude_dir)
    if d.is_dir():
        yield from sorted(d.glob("*.jsonl"))


def iter_codex_trajectories(codex_dir=None):
    """Yield all Codex CLI session JSONL paths."""
    d = Path(codex_dir) if codex_dir else Path.home() / ".codex"
    base = d / "sessions"
    if base.is_dir():
        yield from sorted(base.rglob("*.jsonl"))


def iter_codex_rollout_files(codex_dir=None):
    """Yield Codex CLI rollout JSONL paths (rollout-*.jsonl files only)."""
    d = Path(codex_dir) if codex_dir else Path.home() / ".codex"
    base = d / "sessions"
    if base.is_dir():
        yield from sorted(base.rglob("rollout-*.jsonl"))


def iter_pi_trajectories(pi_dir=None):
    """Yield all pi coding agent session JSONL paths."""
    if pi_dir:
        d = Path(pi_dir) / "sessions"
    else:
        env = os.environ.get("PI_CODING_AGENT_DIR")
        d = Path(env) / "sessions" if env else Path.home() / ".pi" / "agent" / "sessions"
    if d.is_dir():
        yield from sorted(d.rglob("*.jsonl"))


def iter_cursor_trajectories(cursor_dir=None):
    """Yield all Cursor trajectory JSONL paths."""
    d = Path(cursor_dir) if cursor_dir else Path.home() / ".cursor"
    if not d.is_dir():
        return
    seen = set()
    for pattern in ("sessions/**/*.jsonl", "projects/**/*.jsonl"):
        for p in sorted(d.glob(pattern)):
            if p not in seen:
                seen.add(p)
                yield p


def iter_copilot_sessions(copilot_dir=None):
    """Yield (session_id, created_at) pairs from the Copilot CLI SQLite store."""
    d = Path(copilot_dir) if copilot_dir else Path.home() / ".copilot"
    db = d / "session-store.db"
    if not db.exists():
        return
    try:
        conn = sqlite3.connect(str(db))
        try:
            rows = conn.execute("SELECT id, created_at FROM sessions").fetchall()
            yield from rows
        finally:
            conn.close()
    except Exception:
        return
