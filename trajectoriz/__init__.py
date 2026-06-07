"""trajectoriz: locate and parse agent trajectory files on the local machine."""

__version__ = "0.1.0"

import json
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
    extra_agent: dict = field(default_factory=dict)


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

    for entry in entries:
        if not session_id and "sessionId" in entry:
            session_id = entry["sessionId"]
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

    return ParsedTrajectory(
        session_id=session_id,
        model_name=model_name,
        agent_version=agent_version,
        steps=steps,
        total_prompt_tokens=total_prompt,
        total_completion_tokens=total_completion,
        total_cached_tokens=total_cached,
        total_tool_calls=total_tool_calls,
    )


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

    for entry in entries:
        t = entry.get("type", "")
        p = entry.get("payload") or {}
        if t == "session_meta":
            session_id = p.get("id")
            cli_version = p.get("cli_version")
        if t == "turn_context" and p.get("model") and not model_name:
            model_name = p["model"]

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

    return ParsedTrajectory(
        session_id=session_id,
        model_name=model_name,
        agent_version=cli_version,
        steps=steps,
        total_prompt_tokens=total_prompt,
        total_completion_tokens=total_completion,
        total_cached_tokens=total_cached,
        total_tool_calls=total_tool_calls,
    )


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
        _cwd, repository, _branch, summary, _created_at = session_row

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

    return ParsedTrajectory(
        steps=steps,
        extra_agent={
            "copilot_repository": repository or "",
            "copilot_summary": summary or "",
        },
    )
