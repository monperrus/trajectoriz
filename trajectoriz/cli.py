#!/usr/bin/env python3
"""trajectoriz-cli: search and browse past agent trajectories."""

import argparse
import html
import json
import math
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Optional

import trajectoriz as tz
from trajectoriz._search import (
    GrepBackend,
    _make_snippet,
    _matches_any,
    _parse_terms,
    _step_search_blobs,
    get_backend,
)


DEFAULT_SHOW_PAGE_SIZE = 20   # steps per page
DEFAULT_LIST_PAGE_SIZE = 50   # trajectories per page


TrajRecord = tz.TrajectoryRecord


def _all_records() -> Iterator[TrajRecord]:
    yield from tz.iter_all_records()


def _local_records(cwd: str) -> Iterator[TrajRecord]:
    yield from tz.iter_local_records(cwd)


def _cache_dir(cache_dir=None) -> Path:
    return tz._cache_dir(cache_dir)


def _cached_parse(cache_key: str, mtime: float, parse_fn, cache_dir=None) -> tz.ParsedTrajectory:
    return tz._cached_parse(cache_key, mtime, parse_fn, cache_dir)


def _parse_record(record: TrajRecord, cache_dir=None) -> Optional[tz.ParsedTrajectory]:
    return tz.parse_record(record, cache_dir)


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
    tool_calls = step.get("tool_calls", [])
    results = (step.get("observation") or {}).get("results", [])
    for tc in tool_calls:
        args_str = json.dumps(tc.get("arguments", {}), indent=2)
        if not full and len(args_str) > 600:
            args_str = args_str[:600] + "\n…"
        lines.append(f"**Tool call:** `{tc['function_name']}`")
        lines.append(f"```json\n{args_str}\n```\n")
    for res in results:
        content = res.get("content", "") or "*empty output*"
        if not full and len(content) > 1000:
            content = content[:1000] + "\n…"
        lines.append(f"**Tool result:**\n```\n{content}\n```\n")
    if step.get("message"):
        lines.append(step["message"])
        lines.append("")
    return "\n".join(lines)


_ANSI_COLORS = {
    "30": "#1a1a1a", "31": "#c0392b", "32": "#27ae60", "33": "#f39c12",
    "34": "#2980b9", "35": "#8e44ad", "36": "#16a085", "37": "#bdc3c7",
    "90": "#7f8c8d", "91": "#e74c3c", "92": "#2ecc71", "93": "#f1c40f",
    "94": "#3498db", "95": "#9b59b6", "96": "#1abc9c", "97": "#ecf0f1",
}


def _ansi_to_html(text: str) -> str:
    """Convert ANSI escape sequences to HTML spans."""
    result: list[str] = []
    depth = 0
    for token in re.split(r"(\x1b\[[0-9;]*m)", text):
        if not token.startswith("\x1b"):
            result.append(html.escape(token))
            continue
        codes = token[2:-1].split(";")
        for code in codes:
            if code in ("0", ""):
                result.append("</span>" * depth)
                depth = 0
            elif code == "1":
                result.append('<span style="font-weight:bold">')
                depth += 1
            elif code in _ANSI_COLORS:
                result.append(f'<span style="color:{_ANSI_COLORS[code]}">')
                depth += 1
    result.append("</span>" * depth)
    return "".join(result)


_HTML_CSS = """
body { font-family: system-ui, sans-serif; max-width: 900px; margin: 2rem auto;
       padding: 0 1rem; background: #0d1117; color: #c9d1d9; }
h1 { color: #58a6ff; border-bottom: 1px solid #30363d; padding-bottom: .4rem; }
h2 { color: #79c0ff; margin-top: 2rem; }
.meta { color: #8b949e; font-size: .85rem; margin-bottom: 1.5rem; }
.meta span { margin-right: 1.5rem; }
.step { border: 1px solid #30363d; border-radius: 6px; margin: 1rem 0; overflow: hidden; }
.step-header { padding: .4rem .8rem; font-size: .8rem; font-weight: bold;
               text-transform: uppercase; letter-spacing: .05em; }
.step-user .step-header  { background: #1f3a2d; color: #56d364; }
.step-agent .step-header { background: #1a2332; color: #58a6ff; }
.step-body { padding: .8rem; }
.tool-block { margin: .6rem 0; border-radius: 4px; overflow: hidden;
              border: 1px solid #30363d; }
.tool-call-header { background: #21262d; padding: .3rem .6rem; font-size: .8rem;
                    color: #d2a8ff; font-family: monospace; }
.tool-result-header { background: #161b22; padding: .3rem .6rem; font-size: .8rem;
                      color: #8b949e; }
pre { margin: 0; padding: .6rem; background: #161b22; overflow-x: auto;
      font-size: .82rem; white-space: pre-wrap; word-break: break-all; }
.message { padding: .4rem 0; line-height: 1.6; }
.collapsible summary { cursor: pointer; color: #8b949e; font-size: .8rem;
                       padding: .3rem .6rem; background: #161b22; list-style: none; }
.collapsible summary:hover { color: #c9d1d9; }
.result-collapsible summary { color: #6e7681; }
.result-collapsible[open] summary { color: #8b949e; }
"""


def _render_step_html(step: dict, full: bool = False) -> str:
    role = "user" if step["source"] == "user" else "agent"
    ts = step.get("timestamp", "")[:19]
    label = "User" if role == "user" else "Agent"
    lines = [f'<div class="step step-{role}">']
    lines.append(f'<div class="step-header">{label}  <span style="font-weight:normal;opacity:.7">{ts}</span></div>')
    lines.append('<div class="step-body">')

    tool_calls = step.get("tool_calls", [])
    results = (step.get("observation") or {}).get("results", [])

    for i, tc in enumerate(tool_calls):
        fn = tc["function_name"]
        args = tc.get("arguments", {})
        compact = json.dumps(args, separators=(",", ":"))
        args_str = compact if len(compact) <= 120 else json.dumps(args, indent=2)
        if not full and len(args_str) > 600:
            args_str = args_str[:600] + "\n…"

        lines.append('<div class="tool-block">')
        # collapse create_tool source_code by default
        is_create = fn == "create_tool" and "source_code" in args
        if is_create and not full:
            collapsed_args = {k: v for k, v in args.items() if k != "source_code"}
            collapsed_str = json.dumps(collapsed_args, indent=2)
            lines.append(f'<div class="tool-call-header">&#9654; {html.escape(fn)}</div>')
            lines.append(f"<pre>{html.escape(collapsed_str)}</pre>")
            lines.append('<details class="collapsible"><summary>source_code (click to expand)</summary>')
            lines.append(f"<pre>{html.escape(args.get('source_code', ''))}</pre>")
            lines.append("</details>")
        else:
            lines.append(f'<div class="tool-call-header">&#9654; {html.escape(fn)}</div>')
            lines.append(f"<pre>{html.escape(args_str)}</pre>")

        if i < len(results):
            content = results[i].get("content", "") or ""
            if not full and len(content) > 1000:
                content = content[:1000] + "\n…"
            rendered = _ansi_to_html(content) if content else '<span style="opacity:.5">empty output</span>'
            lines.append('<details class="collapsible result-collapsible"><summary>&#9664; result</summary>')
            lines.append(f"<pre>{rendered}</pre>")
            lines.append("</details>")

        lines.append("</div>")  # tool-block

    # any leftover results (more results than calls — shouldn't happen but be safe)
    for res in results[len(tool_calls):]:
        content = res.get("content", "") or ""
        rendered = _ansi_to_html(content) if content else '<span style="opacity:.5">empty output</span>'
        lines.append('<div class="tool-block">')
        lines.append('<details class="collapsible result-collapsible"><summary>&#9664; result</summary>')
        lines.append(f"<pre>{rendered}</pre>")
        lines.append("</details>")
        lines.append("</div>")

    if step.get("message"):
        lines.append(f'<div class="message">{html.escape(step["message"])}</div>')

    lines.append("</div></div>")  # step-body, step
    return "\n".join(lines)


def _trajectory_to_html(record: TrajRecord, full: bool = False) -> str:
    traj = _parse_record(record)
    meta_parts = [f"<span>Agent: <b>{html.escape(record.agent)}</b></span>"]
    if record.timestamp:
        meta_parts.append(f"<span>Date: {record.timestamp[:19]}</span>")
    if traj:
        if traj.model_name:
            meta_parts.append(f"<span>Model: {html.escape(traj.model_name)}</span>")
        if traj.cwd:
            meta_parts.append(f"<span>Directory: {html.escape(traj.cwd)}</span>")
        if traj.total_tool_calls:
            meta_parts.append(f"<span>Tool calls: {traj.total_tool_calls}</span>")
        if traj.total_tokens:
            meta_parts.append(
                f"<span>Tokens: {traj.total_prompt_tokens}p / "
                f"{traj.total_completion_tokens}c / {traj.total_tokens} total</span>"
            )
    steps_html = "\n".join(_render_step_html(s, full=full) for s in (traj.steps if traj else []))
    title = html.escape(record.id)
    first_msg = html.escape((record.first_msg or "")[:120])
    return f"""<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><title>Trajectory {title}</title>
<style>{_HTML_CSS}</style></head>
<body>
<h1>Trajectory <code>{title}</code></h1>
<p style="color:#8b949e;font-style:italic">{first_msg}</p>
<div class="meta">{"".join(meta_parts)}</div>
{steps_html}
</body></html>"""


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
    if args.dir:
        source = _local_records(args.dir)
    elif args.all:
        source = _all_records()
    else:
        source = _local_records(os.getcwd())
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
    if args.dir:
        source = _local_records(args.dir)
    elif args.all:
        source = _all_records()
    else:
        source = _local_records(os.getcwd())
    records = sorted(source, key=lambda r: r.timestamp, reverse=True)

    if args.since:
        records = [r for r in records if r.timestamp[:10] >= args.since]
    if args.date:
        records = [r for r in records if r.timestamp[:10] == args.date]

    if not records:
        print("No trajectories found.")
        return

    show_dir = args.all and not args.dir
    if args.all and not args.dir:
        header = (
            f"## All trajectories ({len(records)} total)\n\n"
            "| ID | Agent | Date | Directory | First message |\n"
            "|---|---|---|---|---|"
        )
    else:
        search_dir = args.dir if args.dir else os.getcwd()
        header = (
            f"## Trajectories in {search_dir} ({len(records)} total)\n\n"
            "| ID | Agent | Date | First message |\n"
            "|---|---|---|---|"
        )
    rows = [_record_row(r, show_dir=show_dir) for r in records]
    page = -1 if args.last else args.page
    _paginate_items(rows, page, args.page_size, header, "trajectories",
                    footer="\nUse `trajectoriz-cli show <id>` to view a trajectory.")


def cmd_search(args) -> None:
    terms = _parse_terms(args.query)
    if args.dir:
        source = _local_records(args.dir)
    elif args.local:
        source = _local_records(os.getcwd())
    else:
        source = _all_records()

    if args.fast:
        _cmd_search_fast(args, terms, source)
    else:
        backend_name = getattr(args, "backend", "grep")
        try:
            backend = get_backend(backend_name)
        except ValueError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            sys.exit(1)
        cmd_search_content(args, terms, source, backend)


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


def cmd_search_content(args, terms: list[str], source: Iterable[TrajRecord], backend=None) -> None:
    """Search the full content of each trajectory's steps for any of the given terms."""
    if backend is None:
        backend = GrepBackend()
    try:
        matches = backend.search(source, terms)
    except NotImplementedError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

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


def _native_id(rec: TrajRecord) -> Optional[str]:
    """Return the underlying agent's own session/file ID for a record, if any."""
    if isinstance(rec.source, dict):
        return str(rec.source.get("session_id") or "")
    if isinstance(rec.source, Path):
        return rec.source.name
    return None


def _find_record(target: str) -> Optional[TrajRecord]:
    """Look up a record by short ID or native ID."""
    for rec in _all_records():
        if rec.id == target:
            return rec
        native = _native_id(rec)
        if native and native == target:
            return rec
    return None


def cmd_show(args) -> None:
    target = args.id
    record = _find_record(target)

    if record is None:
        print(f"Error: trajectory `{target}` not found.", file=sys.stderr)
        sys.exit(1)

    if getattr(args, "html", False):
        print(_trajectory_to_html(record, full=args.full))
        return

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
    record = _find_record(target)

    if record is None:
        print(json.dumps({"error": f"trajectory `{target}` not found"}, indent=2))
        sys.exit(1)

    traj = _parse_record(record)
    info: dict = {
        "id": record.id,
        "agent": record.agent,
        "timestamp": record.timestamp,
        "first_message": record.first_msg,
    }
    if traj is not None:
        info["steps"] = len(traj.steps)
        info["model"] = traj.model_name
        info["directory"] = traj.cwd
        info["tool_calls"] = traj.total_tool_calls
        info["prompt_tokens"] = traj.total_prompt_tokens
        info["completion_tokens"] = traj.total_completion_tokens
        info["cached_tokens"] = traj.total_cached_tokens
        info["total_tokens"] = traj.total_tokens
        if traj.session_id:
            info["session_id"] = traj.session_id
        if traj.agent_version:
            info["agent_version"] = traj.agent_version
    else:
        if isinstance(record.source, dict):
            info["source"] = record.source
        else:
            info["source"] = str(record.source)

    print(json.dumps(info, indent=2))


def cmd_stats(args) -> None:
    """Print aggregate statistics about trajectories as pretty-printed JSON."""
    source = _all_records() if args.all else _local_records(os.getcwd())
    records = list(source)

    total_trajectories = len(records)
    agent_counts: dict[str, int] = {}
    total_steps = 0
    total_tool_calls = 0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_cached_tokens = 0
    total_tokens_agg = 0
    parsed_count = 0
    unparsed_count = 0

    for rec in records:
        agent_counts[rec.agent] = agent_counts.get(rec.agent, 0) + 1
        traj = _parse_record(rec)
        if traj is not None:
            parsed_count += 1
            total_steps += len(traj.steps)
            total_tool_calls += traj.total_tool_calls
            total_prompt_tokens += traj.total_prompt_tokens
            total_completion_tokens += traj.total_completion_tokens
            total_cached_tokens += traj.total_cached_tokens
            total_tokens_agg += traj.total_tokens
        else:
            unparsed_count += 1

    stats = {
        "total_trajectories": total_trajectories,
        "parsed_trajectories": parsed_count,
        "unparsed_trajectories": unparsed_count,
        "agents": agent_counts,
        "total_steps": total_steps,
        "total_tool_calls": total_tool_calls,
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "total_cached_tokens": total_cached_tokens,
        "total_tokens": total_tokens_agg,
    }

    print(json.dumps(stats, indent=2))


_SHELL_TOOL_NAMES = {
    "bash", "run_bash", "execute_bash", "shell", "run_shell",
    "run_terminal_cmd", "run_command", "execute_command", "terminal",
    "computer", "execute",
}


def _shell_command_from_tool(function_name: str, arguments: dict) -> Optional[str]:
    """Return the shell command string if this tool call executes a shell command."""
    fn = function_name.lower()
    if fn not in _SHELL_TOOL_NAMES:
        return None
    return (
        arguments.get("command")
        or arguments.get("cmd")
        or arguments.get("input")
        or None
    )


def _first_word(cmd: str) -> str:
    """Return the first word (program name) of a shell command."""
    cmd = cmd.strip()
    if not cmd:
        return ""
    # strip leading env var assignments like FOO=bar before the program
    parts = cmd.split()
    for part in parts:
        if "=" not in part:
            return part.split("/")[-1]  # basename only
    return parts[0]


def _count_tools_in_records(records) -> dict[str, int]:
    counts: dict[str, int] = {}
    for rec in records:
        traj = _parse_record(rec)
        if traj is None:
            continue
        for step in traj.steps:
            if step["source"] != "agent":
                continue
            for tc in step.get("tool_calls", []):
                cmd = _shell_command_from_tool(tc["function_name"], tc.get("arguments") or {})
                if cmd:
                    prog = _first_word(cmd)
                    if prog:
                        counts[prog] = counts.get(prog, 0) + 1
    return counts


def _advanced_tools_json(scope: dict[str, str], counts: dict[str, int]) -> dict[str, object]:
    sorted_progs = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return {
        "scope": scope,
        "programs": [
            {"program": prog, "count": count}
            for prog, count in sorted_progs
        ],
    }


def cmd_advanced_tools(args) -> None:
    """Show unique programs called via shell tool calls, by frequency."""
    if args.dir:
        resolved_dir = str(Path(args.dir).resolve())
        records = list(_local_records(resolved_dir))
        counts = _count_tools_in_records(records)
        label = f"all trajectories in `{resolved_dir}`"
        scope = {"type": "dir", "path": resolved_dir}
    else:
        record = _find_record(args.id)
        if record is None:
            print(f"Error: trajectory `{args.id}` not found.", file=sys.stderr)
            sys.exit(1)
        traj = _parse_record(record)
        if traj is None:
            print(f"Error: trajectory `{args.id}` cannot be parsed.", file=sys.stderr)
            sys.exit(1)
        counts = _count_tools_in_records([record])
        label = f"`{args.id}`"
        scope = {"type": "id", "id": args.id}

    if args.json:
        print(json.dumps(_advanced_tools_json(scope, counts), indent=2))
        return

    if not counts:
        print("No shell tool calls found.")
        return

    sorted_progs = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    print(f"## Programs called in {label}\n")
    print("| Program | Count |")
    print("|---|---|")
    for prog, count in sorted_progs:
        print(f"| `{prog}` | {count} |")


def cmd_delete(args) -> None:
    """Delete trajectories that have only one user message matching the given text."""
    source = _all_records() if args.all else _local_records(os.getcwd())
    matching = [r for r in source if _is_single_message_only(r, args.message)]

    if not matching:
        print(f"No trajectories found with only one user message '{args.message}'.")
        return

    print(f"Found {len(matching)} trajectory(ies) with only one user message '{args.message}':")
    for r in matching:
        if isinstance(r.source, Path):
            src_info = str(r.source)
        elif isinstance(r.source, dict):
            src_info = str(r.source.get("session_id", ""))
        else:
            src_info = ""
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


# ── Refresh (recoll index) ────────────────────────────────────────────────────

_RECOLL_CONFDIR = Path.home() / ".recoll-trajectories"
_RECOLL_DATA_DIR = Path(__file__).parent / "_recoll"

_RECOLL_CONF_TEMPLATE = """\
topdirs = {topdirs}

dbdir = {confdir}/xapiandb

# Only index .jsonl trajectory files
skippedNames = *.py *.json *.sh *.md .gitignore *.txt *.yaml *.yml *.toml *.cfg *.ini bin log account.json .git

filtersdir = {confdir}/filters
"""


def _recoll_topdirs() -> list[str]:
    """Return the list of directories that contain JSONL trajectories."""
    candidates = [
        Path.home() / ".claude" / "projects",
        Path.home() / ".local" / "share" / "agent_probe",
        Path.home() / ".codex" / "sessions",
        Path.home() / ".copilot" / "session-state",
    ]
    dirs = [str(p) for p in candidates if p.is_dir()]
    cfg = tz.load_config()
    raw = cfg.get("folders", [])
    extras: list[str] = [raw] if isinstance(raw, str) else [str(f) for f in raw] if isinstance(raw, list) else []
    for f in extras:
        p = Path(f).expanduser()
        if p.is_dir() and str(p) not in dirs:
            dirs.append(str(p))
    return dirs


def cmd_refresh(args) -> None:
    """Install recoll config to ~/.recoll-trajectories/ and run recollindex."""
    import shutil

    confdir = _RECOLL_CONFDIR
    confdir.mkdir(parents=True, exist_ok=True)
    (confdir / "filters").mkdir(exist_ok=True)

    topdirs_list = _recoll_topdirs()
    if not topdirs_list:
        print("Warning: no trajectory directories found — index will be empty.", file=sys.stderr)
    topdirs_str = " ".join(topdirs_list)

    conf_path = confdir / "recoll.conf"
    conf_path.write_text(
        _RECOLL_CONF_TEMPLATE.format(topdirs=topdirs_str, confdir=confdir),
        encoding="utf-8",
    )

    for name in ("mimeconf", "mimemap"):
        src = _RECOLL_DATA_DIR / name
        shutil.copy2(src, confdir / name)

    filter_src = _RECOLL_DATA_DIR / "filters" / "rcltraj.py"
    filter_dst = confdir / "filters" / "rcltraj.py"
    shutil.copy2(filter_src, filter_dst)
    filter_dst.chmod(0o755)

    if not args.config_only:
        env = {**os.environ, "RECOLL_CONFDIR": str(confdir)}
        print(f"Indexing: {topdirs_str or '(none)'}")
        result = subprocess.run(["recollindex"], env=env)
        if result.returncode != 0:
            print("recollindex failed — is recoll installed?", file=sys.stderr)
            sys.exit(result.returncode)
        print("Done. Use `trajectoriz-cli search --backend recoll <query>` to search.")
    else:
        print(f"Config written to {confdir}/ (skipped indexing).")


# ── Entry point ───────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="trajectoriz-cli",
        description="Search and browse past agent trajectories.",
        epilog="Run 'trajectoriz-cli <command> -h' for help on a specific command, e.g. 'trajectoriz-cli list -h' or 'trajectoriz-cli search -h'.",
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
    p_list.add_argument("--dir", metavar="PATH", help="Search in this directory instead of the current one.")
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
    p_search.add_argument("--dir", metavar="PATH", help="Restrict search to this directory.")
    p_search.add_argument(
        "--fast", action="store_true",
        help="Search only the first message and metadata (no trajectory parsing). Much faster but misses tool call content.",
    )
    p_search.add_argument(
        "--content", "--grep", action="store_true",
        help="(Deprecated alias — content search is now the default. Use --fast to skip it.)",
    )
    p_search.add_argument(
        "--backend",
        choices=["grep", "recoll", "sqlite"],
        default="grep",
        help="Search backend: grep (default, in-process), recoll (recoll CLI), sqlite (FTS index).",
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
    p_show.add_argument(
        "--html", action="store_true",
        help="Output a self-contained HTML file instead of markdown.",
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
    p_blame.add_argument("--dir", metavar="PATH", help="Search in this directory instead of the current one.")
    p_blame.set_defaults(func=cmd_blame)

    # stats
    p_stats = sub.add_parser(
        "stats",
        help="Show aggregate statistics as pretty-printed JSON.",
    )
    p_stats.add_argument(
        "--all", action="store_true",
        help="Include all agents/directories (default: current directory only).",
    )
    p_stats.set_defaults(func=cmd_stats)

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

    # advanced
    p_advanced = sub.add_parser(
        "advanced",
        help="Advanced analysis commands.",
    )
    adv_sub = p_advanced.add_subparsers(dest="advanced_command", metavar="<subcommand>")
    adv_sub.required = True

    p_adv_tools = adv_sub.add_parser(
        "tools",
        help="Show programs called via shell tool calls, by frequency.",
    )
    p_adv_tools_group = p_adv_tools.add_mutually_exclusive_group(required=True)
    p_adv_tools_group.add_argument("--id", metavar="ID", help="Single trajectory ID.")
    p_adv_tools_group.add_argument("--dir", metavar="PATH", help="Aggregate over all trajectories in this directory.")
    p_adv_tools.add_argument(
        "--json", action="store_true",
        help="Emit machine-readable JSON instead of a Markdown table.",
    )
    p_adv_tools.set_defaults(func=cmd_advanced_tools)

    # refresh
    p_refresh = sub.add_parser(
        "refresh",
        help="Install recoll config to ~/.recoll-trajectories/ and rebuild the index.",
    )
    p_refresh.add_argument(
        "--config-only", action="store_true",
        help="Write config files without running recollindex.",
    )
    p_refresh.set_defaults(func=cmd_refresh)

    if len(sys.argv) == 1:
        parser.print_help()
        return

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
