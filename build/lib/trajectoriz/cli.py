#!/usr/bin/env python3
"""trajectoriz-cli: search and browse past agent trajectories."""

import argparse
import hashlib
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

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


def _cwd_matches(cwd_field: str, target: str) -> bool:
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
                {"type": "codex_db", "session_id": sess.id, "model": sess.model, "cwd": sess.cwd},
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
            {"type": "codex_db", "session_id": sess.id, "model": sess.model, "cwd": sess.cwd},
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


def _render_step(step: dict) -> str:
    lines: list[str] = []
    role = "USER" if step["source"] == "user" else "AGENT"
    ts_suffix = f" *{step['timestamp'][:19]}*" if step.get("timestamp") else ""
    lines.append(f"---\n## Step {step['step_id']} — {role}{ts_suffix}\n")
    if step.get("message"):
        lines.append(step["message"])
        lines.append("")
    for tc in step.get("tool_calls", []):
        args_str = json.dumps(tc.get("arguments", {}), indent=2)
        if len(args_str) > 600:
            args_str = args_str[:600] + "\n…"
        lines.append(f"**Tool call:** `{tc['function_name']}`")
        lines.append(f"```json\n{args_str}\n```\n")
    for res in (step.get("observation") or {}).get("results", []):
        content = res.get("content", "")
        if len(content) > 1000:
            content = content[:1000] + "\n…"
        lines.append(f"**Tool result:**\n```\n{content}\n```\n")
    return "\n".join(lines)


def _trajectory_header_and_steps(record: TrajRecord) -> tuple[str, list[str]]:
    """Return (header_markdown, list_of_rendered_steps)."""
    hlines: list[str] = []
    hlines.append(f"# Trajectory `{record.id}`")
    hlines.append(f"**Agent:** {record.agent}")
    if record.timestamp:
        hlines.append(f"**Date:** {record.timestamp[:19]}")

    if isinstance(record.source, Path):
        if record.agent == "claude":
            traj = tz.parse_claude_trajectory(record.source)
        elif record.agent == "codex":
            traj = tz.parse_codex_trajectory(record.source)
        elif record.agent == "agent_probe":
            hlines.append(f"**Source:** {record.source}")
            if record.first_msg:
                hlines.append("**Steps:** 1")
                steps = [
                    _render_step(
                        {
                            "step_id": 1,
                            "timestamp": record.timestamp,
                            "source": "user",
                            "message": record.first_msg,
                        }
                    )
                ]
                return "\n".join(hlines), steps
            hlines.append("\n*Full trajectory parsing not supported for this agent type.*")
            return "\n".join(hlines), []
        else:
            hlines.append("\n*Full trajectory parsing not supported for this agent type.*")
            return "\n".join(hlines), []
    elif isinstance(record.source, dict):
        src_type = record.source["type"]
        if src_type == "copilot_db":
            db_path = Path(record.source["db_path"])
            traj = tz.parse_copilot_trajectory(db_path, record.source["session_id"])
        else:
            hlines.append(f"**Session ID:** {record.source.get('session_id', '')}")
            if record.source.get("model"):
                hlines.append(f"**Model:** {record.source['model']}")
            d = record.source.get("dir") or record.source.get("cwd") or ""
            if d:
                hlines.append(f"**Directory:** {d}")
            hlines.append("\n*Full trajectory parsing not available for this agent type.*")
            return "\n".join(hlines), []
    else:
        return "\n".join(hlines), []

    hlines.append(f"**Steps:** {len(traj.steps)}")
    if traj.model_name:
        hlines.append(f"**Model:** {traj.model_name}")
    if traj.total_tool_calls:
        hlines.append(f"**Tool calls:** {traj.total_tool_calls}")
    if traj.total_prompt_tokens:
        hlines.append(
            f"**Tokens:** {traj.total_prompt_tokens} prompt / "
            f"{traj.total_completion_tokens} completion"
        )

    return "\n".join(hlines), [_render_step(s) for s in traj.steps]


def _paginate_items(
    items: list[str], page: int, page_size: int, header: str, unit: str, footer: str = ""
) -> None:
    total = len(items)
    total_pages = max(1, math.ceil(total / page_size))
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


# ── Commands ──────────────────────────────────────────────────────────────────


def _record_row(rec: TrajRecord) -> str:
    date = rec.timestamp[:10] if rec.timestamp else "—"
    snippet = (rec.first_msg or "")[:80].replace("|", "\\|").replace("\n", " ")
    return f"| `{rec.id}` | {rec.agent} | {date} | {snippet} |"


def cmd_list(args) -> None:
    source = _all_records() if args.all else _local_records(os.getcwd())
    records = sorted(source, key=lambda r: r.timestamp, reverse=True)
    if not records:
        print("No trajectories found.")
        return
    header = (
        f"## All trajectories ({len(records)} total)\n\n"
        "| ID | Agent | Date | First message |\n"
        "|---|---|---|---|"
    )
    rows = [_record_row(r) for r in records]
    _paginate_items(rows, args.page, args.page_size, header, "trajectories",
                    footer="\nUse `trajectoriz-cli show <id>` to view a trajectory.")


def cmd_search(args) -> None:
    query = args.query.lower()
    source = _all_records() if args.all else _local_records(os.getcwd())
    records = [
        rec
        for rec in source
        if query in (rec.first_msg or "").lower()
        or query in rec.id.lower()
        or query in rec.agent.lower()
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
    _paginate_items(rows, args.page, args.page_size, header, "trajectories",
                    footer="\nUse `trajectoriz-cli show <id>` to view a trajectory.")


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

    header, steps = _trajectory_header_and_steps(record)
    _paginate_items(steps, args.page, args.page_size, header, "steps")


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
    p_list.add_argument("--all", action="store_true", help="Include all agents/directories.")
    p_list.set_defaults(func=cmd_list)

    # search
    p_search = sub.add_parser(
        "search",
        help="Search trajectories by keyword (current directory by default).",
    )
    p_search.add_argument("query", help="Search term (case-insensitive substring).")
    p_search.add_argument("--page", type=int, default=1, metavar="N")
    p_search.add_argument(
        "--page-size", type=int, default=DEFAULT_LIST_PAGE_SIZE, metavar="N",
        help=f"Trajectories per page (default: {DEFAULT_LIST_PAGE_SIZE})",
    )
    p_search.add_argument("--all", action="store_true", help="Search across all agents/directories.")
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
    p_show.set_defaults(func=cmd_show)

    # delete
    p_delete = sub.add_parser(
        "delete",
        help="Delete trajectories whose only user message matches a given text.",
    )
    p_delete.add_argument("message", help="Delete trajectories whose sole user message matches this text (case-insensitive).")
    p_delete.add_argument(
        "--yes", "-y", action="store_true",
        help="Skip confirmation prompt.",
    )
    p_delete.add_argument("--all", action="store_true", help="Search across all agents/directories.")
    p_delete.set_defaults(func=cmd_delete)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
