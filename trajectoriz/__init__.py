"""trajectoriz: locate agent trajectory files on the local machine."""

__version__ = "0.1.0"

import json
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


def iter_copilot_event_trajectories(copilot_dir=None):
    """Yield Copilot CLI session event JSONL paths (~/.copilot/session-state/*/events.jsonl)."""
    d = Path(copilot_dir) if copilot_dir else Path.home() / ".copilot"
    base = d / "session-state"
    if base.is_dir():
        yield from sorted(base.glob("*/events.jsonl"))


def iter_agent_probe_trajectories(agent_probe_dir=None):
    """Yield all agent_probe session JSONL paths (~/.local/share/agent_probe/*/*/*. jsonl)."""
    d = (
        Path(agent_probe_dir)
        if agent_probe_dir
        else Path.home() / ".local" / "share" / "agent_probe"
    )
    if d.is_dir():
        yield from sorted(d.glob("*/*/*.jsonl"))


def iter_opencode_sessions(opencode_dir=None):
    """Yield (id, updated_at_ms, model_json, directory, first_prompt) from the opencode SQLite store."""
    d = (
        Path(opencode_dir)
        if opencode_dir
        else Path.home() / ".local" / "share" / "opencode"
    )
    db = d / "opencode.db"
    if not db.exists():
        return
    try:
        conn = sqlite3.connect(str(db))
        try:
            rows = conn.execute(
                "SELECT id, time_updated, model, directory FROM session ORDER BY time_updated DESC"
            ).fetchall()
            for session_id, ts_ms, model_json, directory in rows:
                first_prompt = ""
                try:
                    row = conn.execute(
                        """
                        SELECT p.data
                        FROM message m
                        JOIN part p ON m.id = p.message_id
                        WHERE m.session_id = ? AND json_extract(m.data, '$.role') = 'user'
                        ORDER BY m.time_created, p.time_created
                        LIMIT 1
                        """,
                        (session_id,),
                    ).fetchone()
                    if row:
                        first_prompt = json.loads(row[0]).get("text", "").strip()
                except Exception:
                    pass
                yield (session_id, ts_ms, model_json, directory, first_prompt)
        finally:
            conn.close()
    except Exception:
        return


def iter_codex_db_sessions(codex_dir=None):
    """Yield (id, updated_at_ms, first_user_message, model_provider, model, cwd) from ~/.codex/state_5.sqlite."""
    d = Path(codex_dir) if codex_dir else Path.home() / ".codex"
    db = d / "state_5.sqlite"
    if not db.exists():
        return
    try:
        conn = sqlite3.connect(str(db))
        try:
            rows = conn.execute(
                "SELECT id, updated_at_ms, first_user_message, model_provider, model, cwd"
                " FROM threads ORDER BY updated_at_ms DESC"
            ).fetchall()
            yield from rows
        finally:
            conn.close()
    except Exception:
        return


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
