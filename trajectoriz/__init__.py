"""trajectoriz: locate and parse agent trajectory files on the local machine."""

__version__ = "0.1.0"

import json
import hashlib
import math
import pickle
from dataclasses import dataclass, field
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


@dataclass(frozen=True)
class HermesSession:
    """One row from the Hermes Agent sessions table."""
    id: str
    model: str | None
    cwd: str | None
    started_at: float | None
    ended_at: float | None
    message_count: int | None
    tool_call_count: int | None
    input_tokens: int | None
    output_tokens: int | None
    title: str | None
    first_user_message: str | None


def iter_hermes_sessions(hermes_dir=None):
    """Yield HermesSession objects from ~/.hermes/state.db."""
    d = Path(hermes_dir) if hermes_dir else Path.home() / ".hermes"
    db = d / "state.db"
    if not db.exists():
        return
    try:
        conn = sqlite3.connect(str(db))
        try:
            rows = conn.execute(
                "SELECT id, model, cwd, started_at, ended_at, message_count, "
                "tool_call_count, input_tokens, output_tokens, title "
                "FROM sessions ORDER BY started_at DESC"
            ).fetchall()
            for row in rows:
                session_id = row[0]
                first_user_message = None
                try:
                    msg_row = conn.execute(
                        "SELECT content FROM messages WHERE session_id=? AND role='user' "
                        "ORDER BY id LIMIT 1",
                        (session_id,),
                    ).fetchone()
                    if msg_row:
                        first_user_message = (msg_row[0] or "").strip() or None
                except Exception:
                    pass
                yield HermesSession(
                    id=row[0],
                    model=row[1],
                    cwd=row[2],
                    started_at=row[3],
                    ended_at=row[4],
                    message_count=row[5],
                    tool_call_count=row[6],
                    input_tokens=row[7],
                    output_tokens=row[8],
                    title=row[9],
                    first_user_message=first_user_message,
                )
        finally:
            conn.close()
    except Exception:
        return


@dataclass(frozen=True)
class OpencodeSession:
    """One row from the opencode session table, plus a pre-fetched first_prompt."""
    id: str
    time_created: int | None
    time_updated: int | None
    model: str | None
    directory: str | None
    agent: str | None
    cost: float | None
    tokens_input: int | None
    tokens_output: int | None
    tokens_reasoning: int | None
    tokens_cache_read: int | None
    tokens_cache_write: int | None
    first_prompt: str


@dataclass(frozen=True)
class CodexDbSession:
    """One row from the Codex state_5.sqlite threads table."""
    id: str
    rollout_path: str | None
    created_at_ms: int | None
    updated_at_ms: int | None
    model_provider: str | None
    model: str | None
    cwd: str | None
    title: str | None
    tokens_used: int | None
    first_user_message: str | None


def iter_opencode_sessions(opencode_dir=None):
    """Yield OpencodeSession objects from the opencode SQLite store."""
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
                "SELECT id, time_created, time_updated, model, directory, "
                "agent, cost, tokens_input, tokens_output, tokens_reasoning, "
                "tokens_cache_read, tokens_cache_write "
                "FROM session ORDER BY time_updated DESC"
            ).fetchall()
            for row in rows:
                session_id = row[0]
                first_prompt = ""
                try:
                    part_row = conn.execute(
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
                    if part_row:
                        first_prompt = json.loads(part_row[0]).get("text", "").strip()
                except Exception:
                    pass
                yield OpencodeSession(
                    id=row[0],
                    time_created=row[1],
                    time_updated=row[2],
                    model=row[3],
                    directory=row[4],
                    agent=row[5],
                    cost=row[6],
                    tokens_input=row[7],
                    tokens_output=row[8],
                    tokens_reasoning=row[9],
                    tokens_cache_read=row[10],
                    tokens_cache_write=row[11],
                    first_prompt=first_prompt,
                )
        finally:
            conn.close()
    except Exception:
        return


def iter_codex_db_sessions(codex_dir=None):
    """Yield CodexDbSession objects from ~/.codex/state_5.sqlite."""
    d = Path(codex_dir) if codex_dir else Path.home() / ".codex"
    db = d / "state_5.sqlite"
    if not db.exists():
        return
    try:
        conn = sqlite3.connect(str(db))
        try:
            rows = conn.execute(
                "SELECT id, rollout_path, created_at_ms, updated_at_ms, model_provider, "
                "model, cwd, title, tokens_used, first_user_message "
                "FROM threads ORDER BY updated_at_ms DESC"
            ).fetchall()
            for row in rows:
                yield CodexDbSession(
                    id=row[0],
                    rollout_path=row[1],
                    created_at_ms=row[2],
                    updated_at_ms=row[3],
                    model_provider=row[4],
                    model=row[5],
                    cwd=row[6],
                    title=row[7],
                    tokens_used=row[8],
                    first_user_message=row[9],
                )
        finally:
            conn.close()
    except Exception:
        return


def _extract_content_text(content) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return " ".join(
            b.get("text", "")
            for b in content
            if isinstance(b, dict) and b.get("type") == "text"
        ).strip()
    return ""


def get_first_user_message_claude(jsonl_path) -> tuple[str, str]:
    """Return (timestamp, first_user_text) from a Claude Code trajectory JSONL."""
    timestamp = ""
    try:
        with open(Path(jsonl_path), encoding="utf-8") as f:
            meta_prompt_ids: set = set()
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not timestamp:
                    timestamp = d.get("timestamp", "")
                if d.get("isMeta"):
                    meta_prompt_ids.add(d.get("promptId", ""))
                    continue
                if d.get("type") == "user":
                    if d.get("promptId") in meta_prompt_ids:
                        continue
                    text = _extract_content_text(d.get("message", {}).get("content", ""))
                    if text:
                        return timestamp, text
    except OSError:
        pass
    return timestamp, ""


def get_first_user_message_copilot(jsonl_path) -> tuple[str, str]:
    """Return (timestamp, first_user_text) from a Copilot events JSONL."""
    timestamp = ""
    try:
        with open(Path(jsonl_path), encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not timestamp:
                    timestamp = d.get("timestamp", "")
                if d.get("type") == "user.message":
                    text = _extract_content_text(d.get("data", {}).get("content", ""))
                    if text:
                        return timestamp, text
    except OSError:
        pass
    return timestamp, ""


def get_first_user_message_agent_probe(jsonl_path) -> tuple[str, str]:
    """Return (timestamp, first_user_text) from an agent_probe trajectory JSONL."""
    timestamp = ""
    try:
        with open(Path(jsonl_path), encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                event_type = d.get("type")
                if event_type == "user":
                    if not timestamp:
                        timestamp = d.get("ts", "") or d.get("timestamp", "")
                    content = d.get("content", "") or d.get("message", {}).get("content", "")
                elif event_type == "user.message":
                    if not timestamp:
                        timestamp = d.get("ts", "") or d.get("timestamp", "")
                    content = d.get("data", {}).get("content", "")
                else:
                    continue
                text = _extract_content_text(content)
                if text:
                    return timestamp, text
    except OSError:
        pass
    return timestamp, ""


def get_first_user_message(jsonl_path) -> tuple[str, str]:
    """Return (timestamp, first_user_text), dispatching by trajectory source."""
    path = Path(jsonl_path)
    if path.is_relative_to(Path.home() / ".claude"):
        return get_first_user_message_claude(path)
    if path.is_relative_to(Path.home() / ".copilot"):
        return get_first_user_message_copilot(path)
    if path.is_relative_to(Path.home() / ".local" / "share" / "agent_probe"):
        return get_first_user_message_agent_probe(path)
    return "", ""


def get_cwd_from_trajectory(jsonl_path) -> str:
    """Extract the working directory from a JSONL trajectory file."""
    try:
        with open(Path(jsonl_path), encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i > 30:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                for key in ("cwd", "workingDirectory", "working_directory"):
                    val = d.get(key)
                    if val and isinstance(val, str):
                        return val
    except OSError:
        pass
    return ""


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


# ── Trajectory parsing ────────────────────────────────────────────────────────

@dataclass
class ParsedTrajectory:
    steps: list[dict] = field(default_factory=list)
    session_id: str | None = None
    model_name: str | None = None
    agent_version: str | None = None
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_cached_tokens: int = 0
    total_tool_calls: int = 0
    total_tokens: int = 0
    extra_agent: dict = field(default_factory=dict)
    cwd: str = ""


@dataclass(frozen=True)
class TrajectoryRecord:
    """Source-agnostic trajectory metadata for iteration and parsing."""

    id: str
    agent: str
    timestamp: str
    first_msg: str
    source: object


def _short_id(prefix: str, key: str) -> str:
    return f"{prefix}-{hashlib.sha256(key.encode()).hexdigest()[:8]}"


def _hermes_ts(started_at) -> str:
    """Convert a Hermes float Unix timestamp to an ISO-8601 string."""
    if not started_at:
        return ""
    import datetime

    try:
        return datetime.datetime.fromtimestamp(float(started_at)).isoformat()
    except (ValueError, OSError):
        return ""


def _codex_first_user_message(path: Path) -> tuple[str, str]:
    ts = ""
    try:
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not ts:
                    ts = d.get("timestamp", "")
                if d.get("type") == "event_msg":
                    payload = d.get("payload") or {}
                    if payload.get("type") == "user_message":
                        msg = (payload.get("message") or "").strip()
                        if msg:
                            return ts, msg
    except OSError:
        pass
    return ts, ""


def _cwd_matches(cwd_field: str | None, target: str) -> bool:
    """True if cwd_field is target or a subdirectory of target."""
    if not cwd_field:
        return False
    try:
        return Path(cwd_field) == Path(target) or Path(cwd_field).is_relative_to(Path(target))
    except (ValueError, TypeError):
        return False


def _iter_extra_folder_records():
    """Yield TrajectoryRecord objects from directories listed in ~/.config/trajectoriz.yaml."""
    cfg = load_config()
    raw = cfg.get("folders", [])
    if isinstance(raw, str):
        folders: list[str] = [raw]
    elif isinstance(raw, list):
        folders = [str(f) for f in raw]
    else:
        folders = []
    for p, fmt in iter_extra_folder_trajectories(folders):
        if fmt == "claude":
            ts, msg = get_first_user_message_claude(p)
        elif fmt == "codex":
            ts, msg = _codex_first_user_message(p)
        elif fmt == "copilot":
            ts, msg = get_first_user_message_copilot(p)
        elif fmt == "agent_probe":
            ts, msg = get_first_user_message_agent_probe(p)
        else:
            ts, msg = "", ""
        yield TrajectoryRecord(_short_id(fmt[:2], str(p)), fmt, ts, msg, p)


def iter_local_records(cwd: str):
    """Yield trajectory records whose working directory is cwd or a subdirectory."""
    for p in iter_claude_project_trajectories(cwd):
        ts, msg = get_first_user_message_claude(p)
        yield TrajectoryRecord(_short_id("cl", str(p)), "claude", ts, msg, p)

    for p in iter_codex_rollout_files():
        if _cwd_matches(get_cwd_from_trajectory(p), cwd):
            ts, msg = _codex_first_user_message(p)
            yield TrajectoryRecord(_short_id("cx", str(p)), "codex", ts, msg, p)

    for p in iter_copilot_event_trajectories():
        if _cwd_matches(get_cwd_from_trajectory(p), cwd):
            ts, msg = get_first_user_message_copilot(p)
            yield TrajectoryRecord(_short_id("cp", str(p)), "copilot", ts, msg, p)

    for p in iter_agent_probe_trajectories():
        if _cwd_matches(get_cwd_from_trajectory(p), cwd):
            ts, msg = get_first_user_message_agent_probe(p)
            yield TrajectoryRecord(_short_id("ap", str(p)), "agent_probe", ts, msg, p)

    for sess in iter_opencode_sessions():
        if _cwd_matches(sess.directory, cwd):
            yield TrajectoryRecord(
                _short_id("oc", sess.id),
                "opencode",
                str(sess.time_updated or sess.time_created or ""),
                sess.first_prompt,
                {"type": "opencode", "session_id": sess.id, "model": sess.model, "dir": sess.directory},
            )

    for sess in iter_codex_db_sessions():
        if _cwd_matches(sess.cwd, cwd):
            yield TrajectoryRecord(
                _short_id("cd", str(sess.id)),
                "codex_db",
                str(sess.updated_at_ms or ""),
                sess.first_user_message or "",
                {
                    "type": "codex_db",
                    "session_id": sess.id,
                    "model": sess.model,
                    "cwd": sess.cwd,
                    "rollout_path": sess.rollout_path,
                },
            )

    for sess in iter_hermes_sessions():
        if _cwd_matches(sess.cwd, cwd):
            yield TrajectoryRecord(
                _short_id("hm", sess.id),
                "hermes",
                _hermes_ts(sess.started_at),
                sess.first_user_message or "",
                {"type": "hermes", "session_id": sess.id, "model": sess.model, "cwd": sess.cwd},
            )

    for rec in _iter_extra_folder_records():
        if isinstance(rec.source, Path):
            traj_cwd = get_cwd_from_trajectory(rec.source)
            if _cwd_matches(traj_cwd, cwd):
                yield rec


def iter_all_records():
    """Yield trajectory records across all supported local stores."""
    for p in iter_claude_trajectories():
        ts, msg = get_first_user_message_claude(p)
        yield TrajectoryRecord(_short_id("cl", str(p)), "claude", ts, msg, p)

    for p in iter_codex_rollout_files():
        ts, msg = _codex_first_user_message(p)
        yield TrajectoryRecord(_short_id("cx", str(p)), "codex", ts, msg, p)

    for p in iter_copilot_event_trajectories():
        ts, msg = get_first_user_message_copilot(p)
        yield TrajectoryRecord(_short_id("cp", str(p)), "copilot", ts, msg, p)

    for p in iter_agent_probe_trajectories():
        ts, msg = get_first_user_message_agent_probe(p)
        yield TrajectoryRecord(_short_id("ap", str(p)), "agent_probe", ts, msg, p)

    for sess in iter_opencode_sessions():
        yield TrajectoryRecord(
            _short_id("oc", sess.id),
            "opencode",
            str(sess.time_updated or sess.time_created or ""),
            sess.first_prompt,
            {"type": "opencode", "session_id": sess.id, "model": sess.model, "dir": sess.directory},
        )

    for sess in iter_codex_db_sessions():
        yield TrajectoryRecord(
            _short_id("cd", str(sess.id)),
            "codex_db",
            str(sess.updated_at_ms or ""),
            sess.first_user_message or "",
            {
                "type": "codex_db",
                "session_id": sess.id,
                "model": sess.model,
                "cwd": sess.cwd,
                "rollout_path": sess.rollout_path,
            },
        )

    for sess in iter_hermes_sessions():
        yield TrajectoryRecord(
            _short_id("hm", sess.id),
            "hermes",
            _hermes_ts(sess.started_at),
            sess.first_user_message or "",
            {"type": "hermes", "session_id": sess.id, "model": sess.model, "cwd": sess.cwd},
        )

    copilot_db = Path.home() / ".copilot" / "session-store.db"
    if copilot_db.exists():
        for session_id, created_at in iter_copilot_sessions():
            yield TrajectoryRecord(
                _short_id("gh", str(session_id)),
                "copilot_db",
                str(created_at or ""),
                "",
                {"type": "copilot_db", "session_id": session_id, "db_path": str(copilot_db)},
            )

    yield from _iter_extra_folder_records()


def iter_records(cwd: str | None = None):
    """Yield public trajectory records, optionally filtered to a working directory."""
    if cwd is None:
        yield from iter_all_records()
        return
    yield from iter_local_records(cwd)


def _cache_dir(cache_dir=None) -> Path:
    d = Path(cache_dir) if cache_dir else Path.home() / ".cache" / "trajectoriz"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _cached_parse(cache_key: str, mtime: float, parse_fn, cache_dir=None) -> ParsedTrajectory:
    """Return parse_fn(), using an mtime-keyed pickle cache to avoid re-parsing."""
    cpath = _cache_dir(cache_dir) / f"{hashlib.sha256(cache_key.encode()).hexdigest()}.pkl"
    if cpath.exists():
        try:
            with cpath.open("rb") as f:
                cached_mtime, traj = pickle.load(f)
            if cached_mtime == mtime:
                return traj
        except Exception:
            pass
    traj = parse_fn()
    try:
        with cpath.open("wb") as f:
            pickle.dump((mtime, traj), f)
    except OSError:
        pass
    return traj


def parse_record(record: TrajectoryRecord, cache_dir=None) -> ParsedTrajectory | None:
    """Parse a trajectory record into a ParsedTrajectory, or None if unsupported."""
    if isinstance(record.source, Path):
        path = record.source
        try:
            mtime = path.stat().st_mtime
        except OSError:
            mtime = 0.0
        parsers = {
            "claude": parse_claude_trajectory,
            "codex": parse_codex_trajectory,
            "copilot": parse_copilot_event_trajectory,
            "agent_probe": parse_agent_probe_trajectory,
        }
        parse_fn = parsers.get(record.agent)
        if parse_fn is None:
            return None
        return _cached_parse(f"{record.agent}:{path}", mtime, lambda: parse_fn(path), cache_dir)

    if isinstance(record.source, dict) and record.source.get("type") == "codex_db":
        rollout_path = record.source.get("rollout_path")
        if rollout_path:
            path = Path(rollout_path)
            if path.exists():
                try:
                    mtime = path.stat().st_mtime
                except OSError:
                    mtime = 0.0
                return _cached_parse(
                    f"codex_db:{record.source['session_id']}",
                    mtime,
                    lambda p=path: parse_codex_trajectory(p),
                    cache_dir,
                )
        return None

    if isinstance(record.source, dict) and record.source.get("type") == "copilot_db":
        db_path = Path(record.source["db_path"])
        session_id = record.source["session_id"]
        try:
            mtime = db_path.stat().st_mtime
        except OSError:
            mtime = 0.0
        return _cached_parse(
            f"copilot_db:{db_path}:{session_id}",
            mtime,
            lambda: parse_copilot_trajectory(db_path, session_id),
            cache_dir,
        )

    if isinstance(record.source, dict) and record.source.get("type") == "hermes":
        session_id = record.source["session_id"]
        db_path = Path.home() / ".hermes" / "state.db"
        try:
            mtime = db_path.stat().st_mtime
        except OSError:
            mtime = 0.0
        return _cached_parse(
            f"hermes:{session_id}",
            mtime,
            lambda sid=session_id: parse_hermes_trajectory(sid),
            cache_dir,
        )

    return None


def estimate_tokens(value: object) -> int:
    """Roughly estimate the token count of text or JSON-serializable content.

    Uses the ~4 chars/token heuristic, matching agent-reports-codex.
    """
    if value in (None, ""):
        return 0
    if isinstance(value, str):
        text = value
    else:
        try:
            text = json.dumps(value, ensure_ascii=False)
        except TypeError:
            text = str(value)
    return max(1, math.ceil(len(text) / 4)) if text else 0


def estimate_trajectory_tokens(traj: ParsedTrajectory) -> int:
    """Estimate total tokens of a trajectory from its steps' text content.

    Used as a fallback for sources that don't report real token usage
    (e.g. Copilot CLI), summing estimates over messages, reasoning,
    tool-call arguments, and tool results.
    """
    total = 0
    for step in traj.steps:
        total += estimate_tokens(step.get("message"))
        total += estimate_tokens(step.get("reasoning_content"))
        for tc in step.get("tool_calls", []):
            total += estimate_tokens(tc.get("arguments"))
        for res in (step.get("observation") or {}).get("results", []):
            total += estimate_tokens(res.get("content"))
    return total


def _truncate(text: str, limit: int = 4000) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n… [{len(text) - limit} chars truncated]"


def _cc_extract_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                if part.get("type") == "text":
                    parts.append(part.get("text", ""))
                elif part.get("type") == "image":
                    parts.append("[image]")
        return "\n".join(p for p in parts if p)
    return ""


def _cc_tool_result_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            p.get("text", "") for p in content
            if isinstance(p, dict) and p.get("type") == "text"
        )
    return str(content) if content is not None else ""


def _cc_is_pure_tool_result(content: object) -> bool:
    return (
        isinstance(content, list)
        and bool(content)
        and all(isinstance(p, dict) and p.get("type") == "tool_result" for p in content)
    )


def parse_claude_trajectory(jsonl_path: Path, fallback_timestamp: str = "") -> ParsedTrajectory:
    """Parse a Claude Code project JSONL trajectory file."""
    entries: list[dict] = []
    with Path(jsonl_path).open(encoding="utf-8") as fh:
        for raw in fh:
            raw = raw.strip()
            if raw:
                try:
                    entries.append(json.loads(raw))
                except json.JSONDecodeError:
                    continue

    session_id: str | None = None
    model_name: str | None = None
    agent_version: str | None = None
    cwd: str = ""

    for entry in entries:
        if not session_id and "sessionId" in entry:
            session_id = entry["sessionId"]
        if not cwd:
            for _key in ("cwd", "workingDirectory", "working_directory"):
                _val = entry.get(_key)
                if _val and isinstance(_val, str):
                    cwd = _val
                    break
        if entry.get("type") == "assistant":
            msg = entry.get("message") or {}
            if not model_name and msg.get("model"):
                model_name = msg["model"]
            if not agent_version and entry.get("version"):
                agent_version = entry["version"]

    tool_results: dict[str, str] = {}
    for entry in entries:
        if entry.get("type") != "user":
            continue
        msg = entry.get("message") or {}
        content = msg.get("content", [])
        if not isinstance(content, list):
            continue
        for part in content:
            if isinstance(part, dict) and part.get("type") == "tool_result":
                tid = part.get("tool_use_id", "")
                if tid:
                    tool_results[tid] = _cc_tool_result_text(part.get("content", ""))

    steps: list[dict] = []
    step_id = 0
    total_tool_calls = 0
    total_prompt = total_completion = total_cached = 0

    for entry in entries:
        entry_type = entry.get("type")
        timestamp: str = entry.get("timestamp") or fallback_timestamp

        if entry_type == "user":
            msg = entry.get("message") or {}
            content = msg.get("content", [])
            if _cc_is_pure_tool_result(content):
                continue
            text = _cc_extract_text(content)
            if not text.strip():
                continue
            step_id += 1
            steps.append({"step_id": step_id, "timestamp": timestamp,
                           "source": "user", "message": text.strip()})

        elif entry_type == "assistant":
            msg = entry.get("message") or {}
            content = msg.get("content") or []
            if not isinstance(content, list):
                content = []

            text_parts: list[str] = []
            reasoning: str | None = None
            tool_calls: list[dict] = []

            for part in content:
                if not isinstance(part, dict):
                    continue
                ptype = part.get("type")
                if ptype == "text":
                    text_parts.append(part.get("text", ""))
                elif ptype == "thinking":
                    reasoning = part.get("thinking", "")
                elif ptype == "tool_use":
                    tool_calls.append({
                        "tool_call_id": part.get("id", ""),
                        "function_name": part.get("name", ""),
                        "arguments": part.get("input") or {},
                    })
                    total_tool_calls += 1

            usage = msg.get("usage") or {}
            prompt_tokens = (
                (usage.get("input_tokens") or 0)
                + (usage.get("cache_creation_input_tokens") or 0)
                + (usage.get("cache_read_input_tokens") or 0)
            )
            completion_tokens = usage.get("output_tokens") or 0
            cached_tokens = usage.get("cache_read_input_tokens") or 0
            total_prompt += prompt_tokens
            total_completion += completion_tokens
            total_cached += cached_tokens

            observation: dict | None = None
            if tool_calls:
                results = [
                    {"source_call_id": tc["tool_call_id"],
                     "content": _truncate(tool_results[tc["tool_call_id"]])}
                    for tc in tool_calls
                    if tc["tool_call_id"] in tool_results
                ]
                if results:
                    observation = {"results": results}

            step: dict = {
                "step_id": step_id + 1,
                "timestamp": timestamp,
                "source": "agent",
                "message": "\n".join(text_parts).strip(),
            }
            step_id += 1
            if reasoning:
                step["reasoning_content"] = reasoning
            if tool_calls:
                step["tool_calls"] = tool_calls
            if observation:
                step["observation"] = observation
            if prompt_tokens or completion_tokens:
                step["metrics"] = {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "cached_tokens": cached_tokens,
                }
            steps.append(step)

    traj = ParsedTrajectory(
        session_id=session_id,
        model_name=model_name,
        agent_version=agent_version,
        steps=steps,
        total_prompt_tokens=total_prompt,
        total_completion_tokens=total_completion,
        total_cached_tokens=total_cached,
        total_tool_calls=total_tool_calls,
        cwd=cwd,
    )
    traj.total_tokens = (total_prompt + total_completion) or estimate_trajectory_tokens(traj)
    return traj


def parse_codex_trajectory(jsonl_path: Path, fallback_timestamp: str = "") -> ParsedTrajectory:
    """Parse a Codex rollout-*.jsonl trajectory file."""
    entries: list[dict] = []
    with Path(jsonl_path).open(encoding="utf-8") as fh:
        for raw in fh:
            raw = raw.strip()
            if raw:
                try:
                    entries.append(json.loads(raw))
                except json.JSONDecodeError:
                    continue

    session_id: str | None = None
    model_name: str | None = None
    cli_version: str | None = None
    cwd: str = ""

    for entry in entries:
        t = entry.get("type", "")
        p = entry.get("payload") or {}
        if t == "session_meta":
            session_id = p.get("id")
            cli_version = p.get("cli_version")
            if not cwd:
                cwd = p.get("cwd") or ""
        if t == "turn_context" and p.get("model") and not model_name:
            model_name = p["model"]
            if not cwd:
                cwd = p.get("cwd") or ""

    tool_results: dict[str, str] = {}
    for entry in entries:
        if entry.get("type") != "response_item":
            continue
        p = entry.get("payload") or {}
        pt = p.get("type", "")
        if pt in ("function_call_output", "custom_tool_call_output"):
            call_id = p.get("call_id", "")
            if call_id:
                tool_results[call_id] = str(p.get("output", ""))

    steps: list[dict] = []
    step_id = 0
    total_tool_calls = 0
    total_prompt = total_completion = total_cached = 0

    pending: dict | None = None

    def _flush_pending() -> None:
        nonlocal pending
        if pending is None:
            return
        tool_calls: list[dict] = pending.get("tool_calls", [])
        if tool_calls:
            results = [
                {"source_call_id": tc["tool_call_id"],
                 "content": _truncate(tool_results[tc["tool_call_id"]])}
                for tc in tool_calls
                if tc["tool_call_id"] in tool_results
            ]
            if results:
                pending["observation"] = {"results": results}
        if not tool_calls:
            pending.pop("tool_calls", None)
        steps.append(pending)
        pending = None

    for entry in entries:
        t = entry.get("type", "")
        p = entry.get("payload") or {}
        pt = p.get("type", "")
        ts = entry.get("timestamp") or fallback_timestamp

        if t == "event_msg" and pt == "user_message":
            _flush_pending()
            text = (p.get("message") or "").strip()
            if text:
                step_id += 1
                steps.append({"step_id": step_id, "timestamp": ts,
                               "source": "user", "message": text})

        elif t == "response_item" and pt == "message" and p.get("role") == "assistant":
            text = "".join(
                part.get("text", "")
                for part in (p.get("content") or [])
                if isinstance(part, dict) and part.get("type") == "output_text"
            ).strip()
            if pending is None:
                step_id += 1
                pending = {"step_id": step_id, "timestamp": ts,
                           "source": "agent", "message": text, "tool_calls": []}
            elif text:
                existing = pending.get("message", "")
                pending["message"] = (existing + "\n" + text).strip()

        elif t == "response_item" and pt in ("function_call", "custom_tool_call"):
            call_id = p.get("call_id", "")
            name = p.get("name", "")
            if pt == "function_call":
                try:
                    arguments: object = json.loads(p.get("arguments") or "{}")
                except (json.JSONDecodeError, TypeError):
                    arguments = {"raw": p.get("arguments", "")}
            else:
                arguments = {"input": p.get("input", "")}

            if pending is None:
                step_id += 1
                pending = {"step_id": step_id, "timestamp": ts,
                           "source": "agent", "message": "", "tool_calls": []}
            pending["tool_calls"].append({
                "tool_call_id": call_id,
                "function_name": name,
                "arguments": arguments,
            })
            total_tool_calls += 1

        elif t == "event_msg" and pt == "token_count":
            tu = (p.get("info") or {}).get("total_token_usage") or {}
            total_prompt = max(total_prompt, tu.get("input_tokens") or 0)
            total_completion = max(total_completion, tu.get("output_tokens") or 0)
            total_cached = max(total_cached, tu.get("cached_input_tokens") or 0)
            if pending is not None:
                lu = (p.get("info") or {}).get("last_token_usage") or {}
                if lu:
                    pending["metrics"] = {
                        "prompt_tokens": lu.get("input_tokens") or 0,
                        "completion_tokens": lu.get("output_tokens") or 0,
                        "cached_tokens": lu.get("cached_input_tokens") or 0,
                    }

        elif t == "event_msg" and pt in ("task_complete", "turn_aborted"):
            _flush_pending()

    _flush_pending()

    traj = ParsedTrajectory(
        session_id=session_id,
        model_name=model_name,
        agent_version=cli_version,
        steps=steps,
        total_prompt_tokens=total_prompt,
        total_completion_tokens=total_completion,
        total_cached_tokens=total_cached,
        total_tool_calls=total_tool_calls,
        cwd=cwd,
    )
    traj.total_tokens = (total_prompt + total_completion) or estimate_trajectory_tokens(traj)
    return traj


def parse_copilot_event_trajectory(jsonl_path: Path, fallback_timestamp: str = "") -> ParsedTrajectory:
    """Parse a Copilot CLI events.jsonl trajectory file."""
    entries: list[dict] = []
    with Path(jsonl_path).open(encoding="utf-8") as fh:
        for raw in fh:
            raw = raw.strip()
            if raw:
                try:
                    entries.append(json.loads(raw))
                except json.JSONDecodeError:
                    continue

    session_id: str | None = None
    model_name: str | None = None
    copilot_version: str | None = None
    cwd: str = ""

    for entry in entries:
        t = entry.get("type", "")
        data = entry.get("data") or {}
        if t == "session.start":
            session_id = data.get("sessionId")
            copilot_version = data.get("copilotVersion")
            if not cwd:
                cwd = data.get("cwd") or data.get("workingDirectory") or ""
        if t == "session.model_change":
            if not model_name:
                model_name = data.get("newModel")

    # Collect tool results keyed by toolCallId
    tool_results: dict[str, str] = {}
    for entry in entries:
        t = entry.get("type", "")
        data = entry.get("data") or {}
        if t == "tool.execution_complete":
            call_id = data.get("toolCallId", "")
            if call_id:
                result = data.get("result") or {}
                content = result.get("detailedContent") or result.get("content", "")
                tool_results[call_id] = str(content)

    steps: list[dict] = []
    step_id = 0
    total_tool_calls = 0
    total_prompt = total_completion = total_cached = 0

    pending: dict | None = None

    def _flush_pending() -> None:
        nonlocal pending
        if pending is None:
            return
        tool_calls: list[dict] = pending.get("tool_calls", [])
        if tool_calls:
            results = [
                {"source_call_id": tc["tool_call_id"],
                 "content": _truncate(tool_results.get(tc["tool_call_id"], ""))}
                for tc in tool_calls
                if tc["tool_call_id"] in tool_results
            ]
            if results:
                pending["observation"] = {"results": results}
        if not tool_calls:
            pending.pop("tool_calls", None)
        steps.append(pending)
        pending = None

    for entry in entries:
        t = entry.get("type", "")
        data = entry.get("data") or {}
        ts = entry.get("timestamp") or fallback_timestamp

        if t == "user.message":
            _flush_pending()
            text = (data.get("content") or "").strip()
            if text:
                step_id += 1
                steps.append({"step_id": step_id, "timestamp": ts,
                               "source": "user", "message": text})

        elif t == "assistant.message":
            content = (data.get("content") or "").strip()
            tool_requests = data.get("toolRequests") or []

            if pending is None:
                step_id += 1
                pending = {"step_id": step_id, "timestamp": ts,
                           "source": "agent", "message": content, "tool_calls": []}
            elif content:
                existing = pending.get("message", "")
                pending["message"] = (existing + "\n" + content).strip()

            for tr in tool_requests:
                call_id = tr.get("toolCallId", "")
                name = tr.get("name", "")
                arguments = tr.get("arguments") or {}
                pending["tool_calls"].append({
                    "tool_call_id": call_id,
                    "function_name": name,
                    "arguments": arguments,
                })
                total_tool_calls += 1

        elif t == "assistant.turn_end":
            _flush_pending()

        elif t == "session.shutdown":
            _flush_pending()

    _flush_pending()

    traj = ParsedTrajectory(
        session_id=session_id,
        model_name=model_name,
        agent_version=copilot_version,
        steps=steps,
        total_prompt_tokens=total_prompt,
        total_completion_tokens=total_completion,
        total_cached_tokens=total_cached,
        total_tool_calls=total_tool_calls,
        cwd=cwd,
    )
    traj.total_tokens = estimate_trajectory_tokens(traj)
    return traj


def parse_copilot_trajectory(db_path: Path, session_id: str, fallback_timestamp: str = "") -> ParsedTrajectory:
    """Parse a GitHub Copilot CLI session from the SQLite session store."""
    conn = sqlite3.connect(str(db_path))
    try:
        session_row = conn.execute(
            "SELECT cwd, repository, branch, summary, created_at FROM sessions WHERE id=?",
            (session_id,),
        ).fetchone()
        if not session_row:
            return ParsedTrajectory()
        session_cwd, repository, _branch, summary, _created_at = session_row

        turns = conn.execute(
            "SELECT turn_index, user_message, assistant_response, timestamp "
            "FROM turns WHERE session_id=? ORDER BY turn_index",
            (session_id,),
        ).fetchall()
    finally:
        conn.close()

    steps: list[dict] = []
    step_id = 0

    for _turn_idx, user_msg, asst_resp, ts in turns:
        ts = ts or fallback_timestamp
        if user_msg and user_msg.strip():
            step_id += 1
            steps.append({"step_id": step_id, "timestamp": ts,
                           "source": "user", "message": user_msg.strip()})
        if asst_resp and asst_resp.strip():
            step_id += 1
            steps.append({"step_id": step_id, "timestamp": ts,
                           "source": "agent", "message": asst_resp.strip()})

    traj = ParsedTrajectory(
        steps=steps,
        cwd=session_cwd or "",
        extra_agent={
            "copilot_repository": repository or "",
            "copilot_summary": summary or "",
        },
    )
    traj.total_tokens = estimate_trajectory_tokens(traj)
    return traj


def parse_hermes_trajectory(session_id: str, hermes_dir=None, fallback_timestamp: str = "") -> ParsedTrajectory:
    """Parse a Hermes Agent session from ~/.hermes/state.db."""
    d = Path(hermes_dir) if hermes_dir else Path.home() / ".hermes"
    db = d / "state.db"
    conn = sqlite3.connect(str(db))
    try:
        sess_row = conn.execute(
            "SELECT model, cwd, started_at, input_tokens, output_tokens, "
            "cache_read_tokens, cache_write_tokens "
            "FROM sessions WHERE id=?",
            (session_id,),
        ).fetchone()
        if not sess_row:
            return ParsedTrajectory()
        model_name, cwd, _started_at, input_tokens, output_tokens, cache_read, cache_write = sess_row

        msgs = conn.execute(
            "SELECT role, content, tool_calls, tool_name, tool_call_id, timestamp, reasoning_content "
            "FROM messages WHERE session_id=? ORDER BY id",
            (session_id,),
        ).fetchall()
    finally:
        conn.close()

    tool_results: dict[str, str] = {}
    for role, content, _tc, _tool_name, tc_id, _ts, _rc in msgs:
        if role == "tool" and tc_id:
            tool_results[tc_id] = str(content or "")

    steps: list[dict] = []
    step_id = 0
    total_tool_calls = 0
    pending: dict | None = None

    def _flush_pending() -> None:
        nonlocal pending
        if pending is None:
            return
        tool_calls: list[dict] = pending.get("tool_calls", [])
        if tool_calls:
            results = [
                {"source_call_id": tc["tool_call_id"],
                 "content": _truncate(tool_results[tc["tool_call_id"]])}
                for tc in tool_calls
                if tc["tool_call_id"] in tool_results
            ]
            if results:
                pending["observation"] = {"results": results}
        if not tool_calls:
            pending.pop("tool_calls", None)
        steps.append(pending)
        pending = None

    def _ts(raw_ts) -> str:
        if raw_ts:
            import datetime
            try:
                return datetime.datetime.fromtimestamp(float(raw_ts)).isoformat()
            except (ValueError, OSError):
                pass
        return fallback_timestamp

    for role, content, tool_calls_json, tool_name, tc_id, raw_ts, reasoning in msgs:
        ts = _ts(raw_ts)

        if role == "user":
            _flush_pending()
            text = (content or "").strip()
            if text:
                step_id += 1
                steps.append({"step_id": step_id, "timestamp": ts,
                               "source": "user", "message": text})

        elif role == "assistant":
            text = (content or "").strip()
            tool_calls_parsed: list[dict] = []
            if tool_calls_json:
                try:
                    raw_calls = json.loads(tool_calls_json)
                    for tc in raw_calls:
                        call_id = tc.get("id") or tc.get("call_id", "")
                        fn = tc.get("function") or {}
                        name = fn.get("name") or tc.get("name", "")
                        args_raw = fn.get("arguments") or tc.get("arguments") or "{}"
                        try:
                            arguments = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
                        except (json.JSONDecodeError, TypeError):
                            arguments = {"raw": args_raw}
                        tool_calls_parsed.append({
                            "tool_call_id": call_id,
                            "function_name": name,
                            "arguments": arguments,
                        })
                        total_tool_calls += 1
                except (json.JSONDecodeError, TypeError):
                    pass

            if pending is None:
                step_id += 1
                pending = {"step_id": step_id, "timestamp": ts,
                           "source": "agent", "message": text, "tool_calls": []}
            elif text:
                existing = pending.get("message", "")
                pending["message"] = (existing + "\n" + text).strip()

            if reasoning:
                pending["reasoning_content"] = reasoning
            pending["tool_calls"].extend(tool_calls_parsed)

            if not tool_calls_parsed:
                _flush_pending()

        # role == "tool": already collected into tool_results above

    _flush_pending()

    total_prompt = input_tokens or 0
    total_completion = output_tokens or 0
    total_cached = (cache_read or 0) + (cache_write or 0)

    traj = ParsedTrajectory(
        session_id=session_id,
        model_name=model_name,
        cwd=cwd or "",
        steps=steps,
        total_prompt_tokens=total_prompt,
        total_completion_tokens=total_completion,
        total_cached_tokens=total_cached,
        total_tool_calls=total_tool_calls,
    )
    traj.total_tokens = (total_prompt + total_completion) or estimate_trajectory_tokens(traj)
    return traj


def parse_agent_probe_trajectory(jsonl_path: Path, fallback_timestamp: str = "") -> ParsedTrajectory:
    """Parse an agent_probe session JSONL trajectory file.

    Token totals come from the "usage" events logged after each completion
    call (the provider's reported prompt/completion token counts). If a
    trajectory has no such events, totals fall back to
    :func:`estimate_trajectory_tokens`.
    """
    entries: list[dict] = []
    with Path(jsonl_path).open(encoding="utf-8") as fh:
        for raw in fh:
            raw = raw.strip()
            if raw:
                try:
                    entries.append(json.loads(raw))
                except json.JSONDecodeError:
                    continue

    session_id: str | None = None
    model_name: str | None = None
    cwd: str = ""
    steps: list[dict] = []
    step_id = 0
    total_tool_calls = 0
    total_prompt = total_completion = total_cached = 0
    saw_usage = False

    pending: dict | None = None

    def _flush_pending() -> None:
        nonlocal pending
        if pending is None:
            return
        if not pending.get("tool_calls"):
            pending.pop("tool_calls", None)
        if not (pending.get("observation") or {}).get("results"):
            pending.pop("observation", None)
        steps.append(pending)
        pending = None

    for entry in entries:
        t = entry.get("type", "")
        ts = entry.get("ts") or fallback_timestamp

        if t == "session_start":
            session_id = entry.get("session_id")
            model_name = entry.get("model")
            if not cwd:
                cwd = entry.get("cwd") or ""

        elif t == "user":
            _flush_pending()
            text = (entry.get("content") or "").strip()
            if text:
                step_id += 1
                steps.append({"step_id": step_id, "timestamp": ts,
                               "source": "user", "message": text})

        elif t == "tool_call":
            if pending is None:
                step_id += 1
                pending = {"step_id": step_id, "timestamp": ts,
                           "source": "agent", "message": "", "tool_calls": []}
            pending["tool_calls"].append({
                "tool_call_id": "",
                "function_name": entry.get("name", ""),
                "arguments": entry.get("args") or {},
            })
            total_tool_calls += 1

        elif t == "tool_result":
            if pending is not None:
                obs = pending.setdefault("observation", {"results": []})
                obs["results"].append({"content": _truncate(str(entry.get("result", "")))})

        elif t == "assistant":
            text = (entry.get("content") or "").strip()
            if pending is None:
                step_id += 1
                pending = {"step_id": step_id, "timestamp": ts,
                           "source": "agent", "message": text}
            else:
                pending["message"] = text
            _flush_pending()

        elif t == "usage":
            saw_usage = True
            total_prompt += entry.get("prompt_tokens") or 0
            total_completion += entry.get("completion_tokens") or 0
            total_cached += entry.get("cached_tokens") or 0

    _flush_pending()

    traj = ParsedTrajectory(
        session_id=session_id,
        model_name=model_name,
        cwd=cwd,
        steps=steps,
        total_prompt_tokens=total_prompt,
        total_completion_tokens=total_completion,
        total_cached_tokens=total_cached,
        total_tool_calls=total_tool_calls,
    )
    traj.total_tokens = (
        (total_prompt + total_completion) if saw_usage else estimate_trajectory_tokens(traj)
    )
    return traj


# ── Trajectory state tracking (before/after a run) ───────────────────────────

def collect_trajectory_state(kind: str | None, repo_root: str) -> dict[str, float]:
    """Snapshot trajectory file mtimes (or session IDs) before an agent run."""
    if kind == "claude_project_jsonl":
        before: dict[str, float] = {}
        for p in iter_claude_project_trajectories(repo_root):
            try:
                before[str(p)] = p.stat().st_mtime
            except OSError:
                continue
        return before
    if kind == "codex_rollout_jsonl":
        state: dict[str, float] = {}
        for p in iter_codex_rollout_files():
            try:
                state[str(p)] = p.stat().st_mtime
            except OSError:
                pass
        return state
    if kind == "copilot_sqlite":
        return {sid: 0.0 for sid, _created_at in iter_copilot_sessions()}
    return {}


def pick_trajectory_id(kind: str | None, repo_root: str, before: dict[str, float]) -> str | None:
    """Return the trajectory ID for the run that just completed."""
    if kind == "claude_project_jsonl":
        candidates: list[tuple[float, Path]] = []
        for p in iter_claude_project_trajectories(repo_root):
            try:
                mtime = p.stat().st_mtime
            except OSError:
                continue
            old_mtime = before.get(str(p))
            if old_mtime is None or mtime > old_mtime:
                candidates.append((mtime, p))
        if not candidates:
            for p in iter_claude_project_trajectories(repo_root):
                try:
                    candidates.append((p.stat().st_mtime, p))
                except OSError:
                    continue
        if not candidates:
            return None
        candidates.sort(key=lambda item: item[0], reverse=True)
        return candidates[0][1].stem
    if kind == "codex_rollout_jsonl":
        new_files: list[tuple[float, Path]] = []
        for p in iter_codex_rollout_files():
            try:
                mtime = p.stat().st_mtime
            except OSError:
                continue
            if before.get(str(p)) != mtime:
                new_files.append((mtime, p))
        if not new_files:
            return None
        new_files.sort(key=lambda x: x[0], reverse=True)
        return str(new_files[0][1])
    if kind == "copilot_sqlite":
        rows = list(iter_copilot_sessions())
        if not rows:
            return None
        new_sessions = [(created_at, sid) for sid, created_at in rows if sid not in before]
        if new_sessions:
            new_sessions.sort(reverse=True)
            return new_sessions[0][1]
        return sorted(rows, key=lambda r: r[1], reverse=True)[0][0]
    return None


# ── Codex exec --json stdout parsing ─────────────────────────────────────────

def _parse_simple_yaml(text: str) -> dict:
    """Parse a minimal YAML subset: top-level string keys mapping to string lists."""
    result: dict = {}
    current_key: str | None = None
    for line in text.splitlines():
        stripped = line.rstrip()
        if not stripped or stripped.lstrip().startswith("#"):
            continue
        if stripped.startswith("  ") or stripped.startswith("\t"):
            item = stripped.lstrip()
            if item.startswith("- ") and current_key is not None:
                result.setdefault(current_key, []).append(item[2:].strip())
        elif ":" in stripped:
            key, _, value = stripped.partition(":")
            current_key = key.strip()
            v = value.strip()
            if v:
                result[current_key] = v
    return result


def load_config(config_path=None) -> dict:
    """Load ~/.config/trajectoriz.yaml; returns {} if missing or unreadable."""
    path = Path(config_path) if config_path else Path.home() / ".config" / "trajectoriz.yaml"
    if not path.exists():
        return {}
    try:
        return _parse_simple_yaml(path.read_text(encoding="utf-8"))
    except OSError:
        return {}


def _detect_jsonl_format(path: Path) -> str | None:
    """Sniff up to 30 lines of a JSONL file and return the detected format name."""
    scores: dict[str, int] = {"claude": 0, "codex": 0, "copilot": 0, "agent_probe": 0}
    try:
        with path.open(encoding="utf-8") as f:
            for i, raw in enumerate(f):
                if i >= 30:
                    break
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    d = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                t = d.get("type", "")
                if "sessionId" in d or (t in ("user", "assistant") and "message" in d):
                    scores["claude"] += 2
                if t == "session_meta" or (t == "event_msg" and "payload" in d):
                    scores["codex"] += 2
                if t in ("session.start", "user.message", "assistant.message", "assistant.turn_end"):
                    scores["copilot"] += 2
                if t in ("session_start", "tool_call", "tool_result") or (
                    t == "assistant" and "content" in d and "ts" in d
                ):
                    scores["agent_probe"] += 2
    except OSError:
        return None
    best = max(scores, key=lambda k: scores[k])
    return best if scores[best] > 0 else None


def iter_extra_folder_trajectories(folders: list[str]):
    """Yield (path, format_name) for JSONL files in the given extra folders."""
    for folder_str in folders:
        folder = Path(folder_str).expanduser()
        if not folder.is_dir():
            continue
        for p in sorted(folder.rglob("*.jsonl")):
            fmt = _detect_jsonl_format(p)
            if fmt:
                yield p, fmt


def codex_exec_jsonl_final_message(stdout_text: str) -> str | None:
    """Return the last agent_message text from `codex exec --json` stdout."""
    messages: list[str] = []
    for raw in stdout_text.splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            entry = json.loads(raw)
        except json.JSONDecodeError:
            return None
        item = entry.get("item") or {}
        if entry.get("type") == "item.completed" and item.get("type") == "agent_message":
            text = item.get("text")
            if isinstance(text, str) and text:
                messages.append(text)
    if not messages:
        return None
    return messages[-1]


def parse_codex_exec_trajectory(stdout_text: str, prompt: str = "", fallback_ts: str = "") -> ParsedTrajectory:
    """Parse `codex exec --json` stdout JSONL into a ParsedTrajectory."""
    entries: list[dict] = []
    for raw in stdout_text.splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            entries.append(json.loads(raw))
        except json.JSONDecodeError:
            return ParsedTrajectory()

    session_id: str | None = None
    steps: list[dict] = []
    step_id = 0
    total_tool_calls = 0
    total_prompt = total_completion = total_cached = 0
    pending: dict | None = None

    if prompt:
        step_id += 1
        steps.append({"step_id": step_id, "timestamp": fallback_ts,
                       "source": "user", "message": prompt})

    def _ensure_pending() -> dict:
        nonlocal pending, step_id
        if pending is None:
            step_id += 1
            pending = {"step_id": step_id, "timestamp": fallback_ts,
                       "source": "agent", "message": "", "tool_calls": []}
        return pending

    def _flush_pending() -> None:
        nonlocal pending
        if pending is None:
            return
        if not pending.get("tool_calls"):
            pending.pop("tool_calls", None)
        steps.append(pending)
        pending = None

    for entry in entries:
        entry_type = entry.get("type")
        if entry_type == "thread.started" and entry.get("thread_id"):
            session_id = str(entry["thread_id"])
            continue

        if entry_type == "item.completed":
            item = entry.get("item") or {}
            item_type = item.get("type")
            if item_type == "agent_message":
                text = item.get("text")
                if isinstance(text, str) and text:
                    p = _ensure_pending()
                    p["message"] = (p.get("message", "") + "\n" + text).strip()
            elif item_type == "command_execution":
                p = _ensure_pending()
                call_id = str(item.get("id") or f"command_{total_tool_calls + 1}")
                p.setdefault("tool_calls", []).append({
                    "tool_call_id": call_id,
                    "function_name": "command_execution",
                    "arguments": {
                        "command": item.get("command", ""),
                        "exit_code": item.get("exit_code"),
                        "status": item.get("status", ""),
                    },
                })
                output = item.get("aggregated_output")
                if isinstance(output, str) and output:
                    p.setdefault("observation", {"results": []})["results"].append({
                        "source_call_id": call_id,
                        "content": _truncate(output),
                    })
                total_tool_calls += 1

        elif entry_type == "turn.completed":
            usage = entry.get("usage") or {}
            total_prompt = max(total_prompt, usage.get("input_tokens") or 0)
            total_completion = max(total_completion, usage.get("output_tokens") or 0)
            total_cached = max(total_cached, usage.get("cached_input_tokens") or 0)
            if pending is not None:
                pending["metrics"] = {
                    "prompt_tokens": usage.get("input_tokens") or 0,
                    "completion_tokens": usage.get("output_tokens") or 0,
                    "cached_tokens": usage.get("cached_input_tokens") or 0,
                }
            _flush_pending()

    _flush_pending()

    traj = ParsedTrajectory(
        session_id=session_id,
        steps=steps,
        total_prompt_tokens=total_prompt,
        total_completion_tokens=total_completion,
        total_cached_tokens=total_cached,
        total_tool_calls=total_tool_calls,
    )
    traj.total_tokens = (total_prompt + total_completion) or estimate_trajectory_tokens(traj)
    return traj
