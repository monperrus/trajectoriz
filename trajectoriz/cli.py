#!/usr/bin/env python3
"""trajectoriz-cli: search and browse past agent trajectories."""

import argparse
import hashlib
import json
import math
import os
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Optional

import trajectoriz as tz


DEFAULT_SHOW_PAGE_SIZE = 20   # steps per page
DEFAULT_LIST_PAGE_SIZE = 50   # trajectories per page


@dataclass
class TrajRecord:
    id: str
    agent: str
    timestamp: str
    first_msg: str
    source: object  # Path for JSONL files; dict for DB sessions


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
                    p = d.get("payload") or {}
                    if p.get("type") == "user_message":
                        msg = (p.get("message") or "").strip()
                        if msg:
                            return ts, msg
    except OSError:
        pass
    return ts, ""


def _cwd_matches(cwd_field: Optional[str], target: str) -> bool:
    """True if cwd_field is target or a subdirectory of target."""
    if not cwd_field:
        return False
    try:
        return Path(cwd_field) == Path(target) or Path(cwd_field).is_relative_to(Path(target))
    except (ValueError, TypeError):
        return False


def _local_records(cwd: str) -> Iterator[TrajRecord]:
    """Yield only trajectories whose working directory is cwd or a subdirectory."""
    for p in tz.iter_claude_project_trajectories(cwd):
        ts, msg = tz.get_first_user_message_claude(p)
        yield TrajRecord(_short_id("cl", str(p)), "claude", ts, msg, p)

    for p in tz.iter_codex_rollout_files():
        if _cwd_matches(tz.get_cwd_from_trajectory(p), cwd):
            ts, msg = _codex_first_user_message(p)
            yield TrajRecord(_short_id("cx", str(p)), "codex", ts, msg, p)

    for p in tz.iter_copilot_event_trajectories():
        if _cwd_matches(tz.get_cwd_from_trajectory(p), cwd):
            ts, msg = tz.get_first_user_message_copilot(p)
            yield TrajRecord(_short_id("cp", str(p)), "copilot", ts, msg, p)

    for p in tz.iter_agent_probe_trajectories():
        if _cwd_matches(tz.get_cwd_from_trajectory(p), cwd):
            ts, msg = tz.get_first_user_message_agent_probe(p)
            yield TrajRecord(_short_id("ap", str(p)), "agent_probe", ts, msg, p)

    for sess in tz.iter_opencode_sessions():
        if _cwd_matches(sess.directory, cwd):
            yield TrajRecord(
                _short_id("oc", sess.id),
                "opencode",
                str(sess.time_updated or sess.time_created or ""),
                sess.first_prompt,
                {"type": "opencode", "session_id": sess.id, "model": sess.model, "dir": sess.directory},
            )

    for sess in tz.iter_codex_db_sessions():
        if _cwd_matches(sess.cwd, cwd):
            yield TrajRecord(
                _short_id("cd", str(sess.id)),
                "codex_db",
                str(sess.updated_at_ms or ""),
                sess.first_user_message or "",
                {"type": "codex_db", "session_id": sess.id, "model": sess.model, "cwd": sess.cwd,
                 "rollout_path": sess.rollout_path},
            )

    for sess in tz.iter_hermes_sessions():
        if _cwd_matches(sess.cwd, cwd):
            yield TrajRecord(
                _short_id("hm", sess.id),
                "hermes",
                _hermes_ts(sess.started_at),
                sess.first_user_message or "",
                {"type": "hermes", "session_id": sess.id, "model": sess.model, "cwd": sess.cwd},
            )


def _all_records() -> Iterator[TrajRecord]:
    for p in tz.iter_claude_trajectories():
        ts, msg = tz.get_first_user_message_claude(p)
        yield TrajRecord(_short_id("cl", str(p)), "claude", ts, msg, p)

    for p in tz.iter_codex_rollout_files():
        ts, msg = _codex_first_user_message(p)
        yield TrajRecord(_short_id("cx", str(p)), "codex", ts, msg, p)

    for p in tz.iter_copilot_event_trajectories():
        ts, msg = tz.get_first_user_message_copilot(p)
        yield TrajRecord(_short_id("cp", str(p)), "copilot", ts, msg, p)

    for p in tz.iter_agent_probe_trajectories():
        ts, msg = tz.get_first_user_message_agent_probe(p)
        yield TrajRecord(_short_id("ap", str(p)), "agent_probe", ts, msg, p)

    for sess in tz.iter_opencode_sessions():
        yield TrajRecord(
            _short_id("oc", sess.id),
            "opencode",
            str(sess.time_updated or sess.time_created or ""),
            sess.first_prompt,
            {"type": "opencode", "session_id": sess.id, "model": sess.model, "dir": sess.directory},
        )

    for sess in tz.iter_codex_db_sessions():
        yield TrajRecord(
            _short_id("cd", str(sess.id)),
            "codex_db",
            str(sess.updated_at_ms or ""),
            sess.first_user_message or "",
            {"type": "codex_db", "session_id": sess.id, "model": sess.model, "cwd": sess.cwd,
             "rollout_path": sess.rollout_path},
        )

    for sess in tz.iter_hermes_sessions():
        yield TrajRecord(
            _short_id("hm", sess.id),
            "hermes",
            _hermes_ts(sess.started_at),
            sess.first_user_message or "",
            {"type": "hermes", "session_id": sess.id, "model": sess.model, "cwd": sess.cwd},
        )

    copilot_db = Path.home() / ".copilot" / "session-store.db"
    if copilot_db.exists():
        for session_id, created_at in tz.iter_copilot_sessions():
            yield TrajRecord(
                _short_id("gh", str(session_id)),
                "copilot_db",
                str(created_at or ""),
                "",
                {"type": "copilot_db", "session_id": session_id, "db_path": str(copilot_db)},
            )


def _cache_dir(cache_dir=None) -> Path:
    d = Path(cache_dir) if cache_dir else Path.home() / ".cache" / "trajectoriz"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _cached_parse(cache_key: str, mtime: float, parse_fn, cache_dir=None) -> tz.ParsedTrajectory:
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


def _parse_record(record: TrajRecord, cache_dir=None) -> Optional[tz.ParsedTrajectory]:
    """Parse a trajectory record into a ParsedTrajectory, or None if unsupported."""
    if isinstance(record.source, Path):
        path = record.source
        try:
            mtime = path.stat().st_mtime
        except OSError:
            mtime = 0.0
        parsers = {
            "claude": tz.parse_claude_trajectory,
            "codex": tz.parse_codex_trajectory,
            "copilot": tz.parse_copilot_event_trajectory,
            "agent_probe": tz.parse_agent_probe_trajectory,
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
                    f"codex_db:{record.source['session_id']}", mtime,
                    lambda p=path: tz.parse_codex_trajectory(p), cache_dir,
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
            f"copilot_db:{db_path}:{session_id}", mtime,
            lambda: tz.parse_copilot_trajectory(db_path, session_id), cache_dir,
        )

    if isinstance(record.source, dict) and record.source.get("type") == "hermes":
        session_id = record.source["session_id"]
        db_path = Path.home() / ".hermes" / "state.db"
        try:
            mtime = db_path.stat().st_mtime
        except OSError:
            mtime = 0.0
        return _cached_parse(
            f"hermes:{session_id}", mtime,
            lambda sid=session_id: tz.parse_hermes_trajectory(sid), cache_dir,
        )

    return None


def _step_search_blobs(step: dict) -> list[str]:
    """Return the searchable text fields of a step (message, reasoning, tool calls, results)."""
    blobs: list[str] = []
    if step.get("message"):
        blobs.append(step["message"])
    if step.get("reasoning_content"):
        blobs.append(step["reasoning_content"])
    for tc in step.get("tool_calls", []):
        blobs.append(json.dumps(tc.get("arguments", {}), ensure_ascii=False))
    for res in (step.get("observation") or {}).get("results", []):
        blobs.append(res.get("content", ""))
    return blobs


def _parse_terms(query: str) -> list[str]:
    r"""Split a grep-style OR query on \| into individual lowercase terms."""
    return [t.lower() for t in query.split(r"\|") if t]


def _matches_any(text: str, terms: list[str]) -> bool:
    tl = text.lower()
    return any(term in tl for term in terms)


def _make_snippet(text: str, terms: list[str], context: int = 50) -> str:
    """Return a short single-line excerpt around the first match of any term."""
    tl = text.lower()
    best_idx, best_len = -1, 0
    for term in terms:
        idx = tl.find(term)
        if idx != -1 and (best_idx == -1 or idx < best_idx):
            best_idx, best_len = idx, len(term)
    if best_idx == -1:
        return ""
    start = max(0, best_idx - context)
    end = min(len(text), best_idx + best_len + context)
    snippet = text[start:end].replace("\n", " ").strip()
    if start > 0:
        snippet = "…" + snippet
    if end < len(text):
        snippet = snippet + "…"
    return snippet


def _record_source_dir(rec: TrajRecord) -> str:
    """Return the working directory associated with a record, for display."""
    if isinstance(rec.source, Path):
        return str(rec.source.parent)
    if isinstance(rec.source, dict):
        return rec.source.get("dir") or rec.source.get("cwd") or "—"
    return "—"


def _render_step(step: dict, full: bool = False) -> str:
    lines: list[str] = []
    role = "USER" if step["source"] == "user" else "AGENT"
    ts_suffix = f" *{step['timestamp'][:19]}*" if step.get("timestamp") else ""
    lines.append(f"---\n## Step {step['step_id']} — {role}{ts_suffix}\n")
    if step.get("message"):
        lines.append(step["message"])
        lines.append("")
    for tc in step.get("tool_calls", []):
        args_str = json.dumps(tc.get("arguments", {}), indent=2)
        if not full and len(args_str) > 600:
            args_str = args_str[:600] + "\n…"
        lines.append(f"**Tool call:** `{tc['function_name']}`")
        lines.append(f"```json\n{args_str}\n```\n")
    for res in (step.get("observation") or {}).get("results", []):
        content = res.get("content", "")
        if not full and len(content) > 1000:
            content = content[:1000] + "\n…"
        lines.append(f"**Tool result:**\n```\n{content}\n```\n")
    return "\n".join(lines)


def _trajectory_header_and_steps(record: TrajRecord, full: bool = False) -> tuple[str, list[str]]:
    """Return (header_markdown, list_of_rendered_steps)."""
    hlines: list[str] = []
    hlines.append(f"# Trajectory `{record.id}`")
    hlines.append(f"**Agent:** {record.agent}")
    if record.timestamp:
        hlines.append(f"**Date:** {record.timestamp[:19]}")

    if record.agent == "agent_probe" and isinstance(record.source, Path):
        hlines.append(f"**Source:** {record.source}")

    traj = _parse_record(record)
    if traj is None:
        if isinstance(record.source, dict):
            hlines.append(f"**Session ID:** {record.source.get('session_id', '')}")
            if record.source.get("model"):
                hlines.append(f"**Model:** {record.source['model']}")
            d = record.source.get("dir") or record.source.get("cwd") or ""
            if d:
                hlines.append(f"**Directory:** {d}")
            hlines.append("\n*Full trajectory parsing not available for this agent type.*")
        else:
            hlines.append("\n*Full trajectory parsing not supported for this agent type.*")
        return "\n".join(hlines), []

    hlines.append(f"**Steps:** {len(traj.steps)}")
    if traj.model_name:
        hlines.append(f"**Model:** {traj.model_name}")
    if traj.cwd:
        hlines.append(f"**Directory:** {traj.cwd}")
    if traj.total_tool_calls:
        hlines.append(f"**Tool calls:** {traj.total_tool_calls}")
    if traj.total_prompt_tokens or traj.total_completion_tokens:
        hlines.append(
            f"**Tokens:** {traj.total_prompt_tokens} prompt / "
            f"{traj.total_completion_tokens} completion / "
            f"{traj.total_tokens} total"
        )
    elif traj.total_tokens:
        hlines.append(f"**Tokens:** ~{traj.total_tokens} (estimated)")

    return "\n".join(hlines), [_render_step(s, full=full) for s in traj.steps]


def _paginate_items(
    items: list[str], page: int, page_size: int, header: str, unit: str, footer: str = ""
) -> None:
    total = len(items)
    total_pages = max(1, math.ceil(total / page_size))
    if page == -1:
        page = total_pages
    if page < 1 or page > total_pages:
        print(f"Error: page {page} out of range (1–{total_pages}).", file=sys.stderr)
        sys.exit(1)
    start = (page - 1) * page_size
    chunk = items[start : start + page_size]
    showing_end = min(start + page_size, total)
    print(
        f"<!-- trajectoriz | page {page}/{total_pages} | "
        f"{unit} {start + 1}–{showing_end} of {total} -->"
    )
    if header:
        print(header)
    print("\n".join(chunk))
    if footer and page == total_pages:
        print(footer)
    if page < total_pages:
        remaining = total_pages - page
        print(f"\n<!-- {remaining} more page(s) — run with --page {page + 1} to continue -->")


# ── Blame helpers ─────────────────────────────────────────────────────────────


def _count_lines(text: str) -> int:
    return len(text.splitlines()) if text else 0


def _blame_edits_from_tool(
    function_name: str, arguments: dict
) -> list[tuple[str, str, int, int, int, int]]:
    """Return list of (file_path, op, lines_added, lines_removed, chars_added, chars_removed)."""
    fn = function_name.lower()

    def _write(path, content):
        c = str(content)
        return (path, "write", _count_lines(c), 0, len(c), 0)

    def _edit(path, old, new):
        o, n = str(old), str(new)
        return (path, "edit", _count_lines(n), _count_lines(o), len(n), len(o))

    if fn in ("write", "write_file", "create_file", "writefile", "overwrite_file"):
        path = arguments.get("file_path") or arguments.get("path") or ""
        content = arguments.get("content") or arguments.get("file_text") or ""
        if path:
            return [_write(path, content)]

    if fn in ("edit", "editfile", "edit_file"):
        path = arguments.get("file_path") or arguments.get("path") or ""
        old = arguments.get("old_string") or arguments.get("old_str") or ""
        new = arguments.get("new_string") or arguments.get("new_str") or ""
        if path:
            return [_edit(path, old, new)]

    if fn in ("multiedit", "multi_edit"):
        path = arguments.get("file_path") or arguments.get("path") or ""
        edits = arguments.get("edits") or []
        if path and edits:
            la = lr = ca = cr = 0
            for e in edits:
                o = str(e.get("old_string") or e.get("old_str") or "")
                n = str(e.get("new_string") or e.get("new_str") or "")
                la += _count_lines(n); lr += _count_lines(o)
                ca += len(n); cr += len(o)
            return [(path, "edit", la, lr, ca, cr)]

    if fn in ("str_replace_based_edit_tool", "str_replace_editor", "str_replace_tool"):
        path = str(arguments.get("path") or "")
        command = str(arguments.get("command") or "")
        if command in ("create", "write", "create_file"):
            content = str(arguments.get("file_text") or arguments.get("content") or "")
            if path:
                return [_write(path, content)]
        if command == "str_replace":
            old = str(arguments.get("old_str") or "")
            new = str(arguments.get("new_str") or "")
            if path:
                return [_edit(path, old, new)]
        if command == "insert":
            new = str(arguments.get("new_str") or "")
            if path:
                return [(path, "edit", _count_lines(new), 0, len(new), 0)]

    if fn == "apply_patch":
        patch = str(arguments.get("patch") or "")
        if patch:
            return _parse_patch_blame(patch)

    return []


def _parse_patch_blame(patch: str) -> list[tuple[str, str, int, int, int, int]]:
    """Parse a patch string into (path, op, la, lr, ca, cr) per file touched."""
    results: list[tuple[str, str, int, int, int, int]] = []

    # Codex custom format: *** Update File: / *** Add File: / *** Delete File:
    if any(m in patch for m in ("*** Begin Patch", "*** Update File:", "*** Add File:", "*** Delete File:")):
        current_file: str | None = None
        current_op = "edit"
        la = lr = ca = cr = 0
        for line in patch.splitlines():
            for prefix, op in (
                ("*** Update File:", "edit"),
                ("*** Add File:", "write"),
                ("*** Delete File:", "delete"),
            ):
                if line.startswith(prefix):
                    if current_file is not None:
                        results.append((current_file, current_op, la, lr, ca, cr))
                    current_file = line[len(prefix):].strip()
                    current_op = op; la = lr = ca = cr = 0
                    break
            else:
                if current_file and line.startswith("+") and not line.startswith("+++"):
                    la += 1; ca += len(line) - 1
                elif current_file and line.startswith("-") and not line.startswith("---"):
                    lr += 1; cr += len(line) - 1
        if current_file is not None:
            results.append((current_file, current_op, la, lr, ca, cr))
        if results:
            return results

    # Standard unified diff (--- a/path / +++ b/path)
    import re as _re
    current_file = None
    current_op = "edit"
    la = lr = ca = cr = 0
    for line in patch.splitlines():
        if line.startswith("diff --git "):
            if current_file is not None and (la or lr):
                results.append((current_file, current_op, la, lr, ca, cr))
            current_file = None; current_op = "edit"; la = lr = ca = cr = 0
        elif line.startswith("--- "):
            path_part = line[4:].split("\t")[0].strip()
            if path_part != "/dev/null":
                current_file = _re.sub(r"^a/", "", path_part)
                current_op = "edit"
            else:
                current_op = "write"
            la = lr = ca = cr = 0
        elif line.startswith("+++ "):
            path_part = line[4:].split("\t")[0].strip()
            if path_part != "/dev/null":
                current_file = _re.sub(r"^b/", "", path_part)
        elif current_file and line.startswith("+") and not line.startswith("+++"):
            la += 1; ca += len(line) - 1
        elif current_file and line.startswith("-") and not line.startswith("---"):
            lr += 1; cr += len(line) - 1
    if current_file is not None and (la or lr):
        results.append((current_file, current_op, la, lr, ca, cr))
    return results


def _path_matches_target(tool_path: str, target_abs: Path) -> bool:
    tp = (tool_path or "").strip()
    if not tp:
        return False
    p = Path(tp)
    if p.is_absolute():
        return p == target_abs
    # relative: check as suffix of the absolute target
    tp_clean = tp.lstrip("./").lstrip("/")
    target_str = str(target_abs)
    return target_str.endswith("/" + tp_clean)


@dataclass
class BlameEntry:
    timestamp: str
    agent: str
    traj_id: str
    traj_first_msg: str
    op: str
    lines_added: int
    lines_removed: int
    chars_added: int
    chars_removed: int


def _blame_delta(e: BlameEntry) -> str:
    la, lr = e.lines_added, e.lines_removed
    if la or lr:
        parts = ([f"+{la}"] if la else []) + ([f"-{lr}"] if lr else [])
        return "/".join(parts) + " lines"
    ca, cr = e.chars_added, e.chars_removed
    if ca or cr:
        parts = ([f"+{ca}"] if ca else []) + ([f"-{cr}"] if cr else [])
        return "/".join(parts) + " chars"
    return "~"


def cmd_blame(args) -> None:
    target = Path(args.file).resolve()
    source = _all_records() if args.all else _local_records(os.getcwd())
    records = sorted(source, key=lambda r: r.timestamp)  # chronological order

    entries: list[BlameEntry] = []
    for rec in records:
        traj = _parse_record(rec)
        if traj is None:
            continue
        for step in traj.steps:
            if step["source"] != "agent":
                continue
            for tc in step.get("tool_calls", []):
                for file_path, op, la, lr, ca, cr in _blame_edits_from_tool(
                    tc["function_name"], tc.get("arguments") or {}
                ):
                    if _path_matches_target(file_path, target):
                        entries.append(BlameEntry(
                            timestamp=step.get("timestamp") or rec.timestamp,
                            agent=rec.agent,
                            traj_id=rec.id,
                            traj_first_msg=(rec.first_msg or "")[:60],
                            op=op,
                            lines_added=la,
                            lines_removed=lr,
                            chars_added=ca,
                            chars_removed=cr,
                        ))

    try:
        display_path = "~/" + str(target.relative_to(Path.home()))
    except ValueError:
        display_path = str(target)

    if not entries:
        print(f"No agent edits found for `{display_path}`.")
        return

    header = (
        f"## Blame: `{display_path}` — {len(entries)} agent edit(s)\n\n"
        "| Timestamp | Agent | Traj ID | Op | Delta | First message |\n"
        "|---|---|---|---|---|---|"
    )
    rows = []
    for e in entries:
        ts = e.timestamp[:19] if e.timestamp else "—"
        msg = (e.traj_first_msg or "").replace("|", "\\|").replace("\n", " ")
        rows.append(
            f"| {ts} | {e.agent} | `{e.traj_id}` | {e.op} | {_blame_delta(e)} | {msg} |"
        )
    page = -1 if args.last else args.page
    _paginate_items(rows, page, args.page_size, header, "edits",
                    footer="\nUse `trajectoriz-cli show <id>` to view the full trajectory.")


# ── Commands ──────────────────────────────────────────────────────────────────


def _record_row(rec: TrajRecord, show_dir: bool = False) -> str:
    date = rec.timestamp[:10] if rec.timestamp else "—"
    snippet = (rec.first_msg or "")[:80].replace("|", "\\|").replace("\n", " ")
    if show_dir:
        d = _record_source_dir(rec).replace("|", "\\|")
        return f"| `{rec.id}` | {rec.agent} | {date} | {d} | {snippet} |"
    return f"| `{rec.id}` | {rec.agent} | {date} | {snippet} |"


def cmd_list(args) -> None:
    source = _all_records() if args.all else _local_records(os.getcwd())
    records = sorted(source, key=lambda r: r.timestamp, reverse=True)

    if args.since:
        records = [r for r in records if r.timestamp[:10] >= args.since]
    if args.date:
        records = [r for r in records if r.timestamp[:10] == args.date]

    if not records:
        print("No trajectories found.")
        return

    show_dir = args.all
    if show_dir:
        header = (
            f"## All trajectories ({len(records)} total)\n\n"
            "| ID | Agent | Date | Directory | First message |\n"
            "|---|---|---|---|---|"
        )
    else:
        header = (
            f"## Trajectories in {os.getcwd()} ({len(records)} total)\n\n"
            "| ID | Agent | Date | First message |\n"
            "|---|---|---|---|"
        )
    rows = [_record_row(r, show_dir=show_dir) for r in records]
    page = -1 if args.last else args.page
    _paginate_items(rows, page, args.page_size, header, "trajectories",
                    footer="\nUse `trajectoriz-cli show <id>` to view a trajectory.")


def cmd_search(args) -> None:
    terms = _parse_terms(args.query)
    # search defaults to all trajectories; --local restricts to cwd
    source = _local_records(os.getcwd()) if args.local else _all_records()

    if args.fast:
        _cmd_search_fast(args, terms, source)
    else:
        cmd_search_content(args, terms, source)


def _cmd_search_fast(args, terms: list[str], source: Iterable[TrajRecord]) -> None:
    """Search only first message, agent name, and ID (no trajectory parsing)."""
    records = [
        rec
        for rec in source
        if _matches_any(rec.first_msg or "", terms)
        or _matches_any(rec.id, terms)
        or _matches_any(rec.agent, terms)
    ]
    records.sort(key=lambda r: r.timestamp, reverse=True)

    if not records:
        print(f"No trajectories found matching `{args.query}`.")
        return
    header = (
        f"## Search: `{args.query}` — {len(records)} result(s)\n\n"
        "| ID | Agent | Date | First message |\n"
        "|---|---|---|---|"
    )
    rows = [_record_row(r) for r in records]
    page = -1 if args.last else args.page
    _paginate_items(rows, page, args.page_size, header, "trajectories",
                    footer="\nUse `trajectoriz-cli show <id>` to view a trajectory.")


def cmd_search_content(args, terms: list[str], source: Iterable[TrajRecord]) -> None:
    """Search the full content of each trajectory's steps for any of the given terms."""
    records = sorted(source, key=lambda r: r.timestamp, reverse=True)

    matches: list[tuple[TrajRecord, int, str]] = []
    for rec in records:
        # always check first message first (no parse needed)
        if _matches_any(rec.first_msg or "", terms):
            snippet = _make_snippet(rec.first_msg or "", terms)
            matches.append((rec, 1, snippet))
            continue
        traj = _parse_record(rec)
        if traj is None:
            continue
        for step in traj.steps:
            for blob in _step_search_blobs(step):
                if _matches_any(blob, terms):
                    matches.append((rec, step["step_id"], _make_snippet(blob, terms)))
                    break

    if not matches:
        print(f"No trajectories found matching `{args.query}` in their content.")
        return

    header = (
        f"## Search: `{args.query}` — {len(matches)} match(es)\n\n"
        "| ID | Agent | Date | Step | Snippet |\n"
        "|---|---|---|---|---|"
    )
    rows = []
    for rec, step_id, snippet in matches:
        date = rec.timestamp[:10] if rec.timestamp else "—"
        snippet = snippet.replace("|", "\\|").replace("\n", " ")
        rows.append(f"| `{rec.id}` | {rec.agent} | {date} | {step_id} | {snippet} |")
    page = -1 if args.last else args.page
    _paginate_items(rows, page, args.page_size, header, "matches",
                    footer="\nUse `trajectoriz-cli show <id> --step N` to jump to a step.")


def cmd_show(args) -> None:
    target = args.id
    record: Optional[TrajRecord] = None
    for rec in _all_records():
        if rec.id == target:
            record = rec
            break

    if record is None:
        print(f"Error: trajectory `{target}` not found.", file=sys.stderr)
        sys.exit(1)

    header, steps = _trajectory_header_and_steps(record, full=args.full)

    if args.last:
        page = -1
    elif args.step is not None:
        if args.step < 1 or args.step > len(steps):
            print(f"Error: step {args.step} out of range (1–{len(steps)}).", file=sys.stderr)
            sys.exit(1)
        page = math.ceil(args.step / args.page_size)
    else:
        page = args.page

    _paginate_items(steps, page, args.page_size, header, "steps")


def _is_single_message_only(record: TrajRecord, message: str) -> bool:
    """Return True if the trajectory has exactly one user message matching message."""
    if (record.first_msg or "").strip().lower() != message.lower():
        return False
    if isinstance(record.source, Path):
        try:
            if record.agent == "claude":
                traj = tz.parse_claude_trajectory(record.source)
            elif record.agent == "codex":
                traj = tz.parse_codex_trajectory(record.source)
            else:
                return True  # copilot/agent_probe: trust first_msg check above
            return sum(1 for s in traj.steps if s["source"] == "user") == 1
        except Exception:
            return False
    # DB-based sessions: trust the first_prompt field
    return True


def cmd_info(args) -> None:
    target = args.id
    record: Optional[TrajRecord] = None
    for rec in _all_records():
        if rec.id == target:
            record = rec
            break

    if record is None:
        print(f"Error: trajectory `{target}` not found.", file=sys.stderr)
        sys.exit(1)

    header, _ = _trajectory_header_and_steps(record)
    print(header)


def cmd_delete(args) -> None:
    """Delete trajectories that have only one user message matching the given text."""
    source = _all_records() if args.all else _local_records(os.getcwd())
    matching = [r for r in source if _is_single_message_only(r, args.message)]

    if not matching:
        print(f"No trajectories found with only one user message '{args.message}'.")
        return

    print(f"Found {len(matching)} trajectory(ies) with only one user message '{args.message}':")
    for r in matching:
        src_info = str(r.source) if isinstance(r.source, Path) else str(r.source.get("session_id", ""))
        date = r.timestamp[:10] if r.timestamp else "—"
        print(f"  {r.id}  {r.agent:12s}  {date}  {src_info}")

    if not args.yes:
        try:
            reply = input("\nDelete these? [y/N] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nAborted.")
            return
        if reply != "y":
            print("Aborted.")
            return

    deleted = 0
    skipped = 0
    for r in matching:
        if isinstance(r.source, Path):
            try:
                r.source.unlink()
                deleted += 1
            except OSError as e:
                print(f"Error deleting {r.source}: {e}", file=sys.stderr)
        else:
            print(f"Skipping {r.id} ({r.agent}): DB-session deletion not yet supported.",
                  file=sys.stderr)
            skipped += 1

    print(f"Deleted {deleted} trajectory file(s)." + (f" Skipped {skipped}." if skipped else ""))


# ── Entry point ───────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="trajectoriz-cli",
        description="Search and browse past agent trajectories.",
    )

    sub = parser.add_subparsers(dest="command", metavar="<command>")
    sub.required = True

    # list
    p_list = sub.add_parser("list", help="List trajectories (current directory by default).")
    p_list.add_argument("--page", type=int, default=1, metavar="N")
    p_list.add_argument(
        "--page-size", type=int, default=DEFAULT_LIST_PAGE_SIZE, metavar="N",
        help=f"Trajectories per page (default: {DEFAULT_LIST_PAGE_SIZE})",
    )
    p_list.add_argument("--last", action="store_true", help="Jump to the last page.")
    p_list.add_argument("--all", action="store_true", help="Include all agents/directories (adds Directory column).")
    p_list.add_argument(
        "--since", metavar="YYYY-MM-DD",
        help="Only show trajectories on or after this date.",
    )
    p_list.add_argument(
        "--date", metavar="YYYY-MM-DD",
        help="Only show trajectories from exactly this date.",
    )
    p_list.set_defaults(func=cmd_list)

    # search
    p_search = sub.add_parser(
        "search",
        help="Search trajectories by keyword (all directories by default).",
    )
    p_search.add_argument("query", help="Search term (case-insensitive substring).")
    p_search.add_argument("--page", type=int, default=1, metavar="N")
    p_search.add_argument(
        "--page-size", type=int, default=DEFAULT_LIST_PAGE_SIZE, metavar="N",
        help=f"Trajectories per page (default: {DEFAULT_LIST_PAGE_SIZE})",
    )
    p_search.add_argument("--last", action="store_true", help="Jump to the last page.")
    p_search.add_argument(
        "--local", action="store_true",
        help="Restrict search to the current directory (default searches all).",
    )
    p_search.add_argument(
        "--fast", action="store_true",
        help="Search only the first message and metadata (no trajectory parsing). Much faster but misses tool call content.",
    )
    p_search.add_argument(
        "--content", "--grep", action="store_true",
        help="(Deprecated alias — content search is now the default. Use --fast to skip it.)",
    )
    p_search.set_defaults(func=cmd_search)

    # show
    p_show = sub.add_parser(
        "show",
        help="Show a trajectory in agent-readable markdown.",
    )
    p_show.add_argument("id", help="Trajectory ID (from list or search).")
    p_show.add_argument(
        "--page", type=int, default=1, metavar="N",
        help="Page number (default: 1). Increment to scroll.",
    )
    p_show.add_argument(
        "--page-size", type=int, default=DEFAULT_SHOW_PAGE_SIZE, metavar="N",
        help=f"Messages (steps) per page (default: {DEFAULT_SHOW_PAGE_SIZE})",
    )
    p_show.add_argument("--last", action="store_true", help="Jump to the last page.")
    p_show.add_argument(
        "--step", type=int, default=None, metavar="N",
        help="Jump directly to the page containing step N.",
    )
    p_show.add_argument(
        "--full", "--no-truncate", action="store_true",
        help="Show complete tool call arguments and results without truncation.",
    )
    p_show.set_defaults(func=cmd_show)

    # info
    p_info = sub.add_parser(
        "info",
        help="Show compact metadata for a trajectory (no steps).",
    )
    p_info.add_argument("id", help="Trajectory ID (from list or search).")
    p_info.set_defaults(func=cmd_info)

    # blame
    p_blame = sub.add_parser(
        "blame",
        help="Show which agent edits touched a file, in chronological order.",
    )
    p_blame.add_argument("file", help="Path to the file to blame.")
    p_blame.add_argument("--page", type=int, default=1, metavar="N")
    p_blame.add_argument(
        "--page-size", type=int, default=DEFAULT_LIST_PAGE_SIZE, metavar="N",
        help=f"Edits per page (default: {DEFAULT_LIST_PAGE_SIZE})",
    )
    p_blame.add_argument("--last", action="store_true", help="Jump to the last page.")
    p_blame.add_argument(
        "--all", action="store_true",
        help="Search across all agents/directories (default: local project only).",
    )
    p_blame.set_defaults(func=cmd_blame)

    # delete
    p_delete = sub.add_parser(
        "delete",
        help="Delete trajectories whose only user message matches a given text.",
    )
    sub._choices_actions = [
        action for action in sub._choices_actions
        if action.dest != "delete"
    ]
    p_delete.add_argument("message", help="Delete trajectories whose sole user message matches this text (case-insensitive).")
    p_delete.add_argument(
        "--yes", "-y", action="store_true",
        help="Skip confirmation prompt.",
    )
    p_delete.add_argument("--all", action="store_true", help="Search across all agents/directories.")
    p_delete.set_defaults(func=cmd_delete)

    if len(sys.argv) == 1:
        parser.print_help()
        return

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
