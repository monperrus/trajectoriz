#!/usr/bin/env python3
"""Translate parsed agent trajectories to ATIF v1.7.

ATIF = Agent Trajectory Interchange Format
Reference: harborframework.com/docs/agents/trajectory-format
Supported sources:
  - Claude Code  (trajectory_kind="claude_project_jsonl")
  - Codex        (trajectory_kind="codex_rollout_jsonl")
  - Copilot CLI  (trajectory_kind="copilot_sqlite")
  - agentknit    (trajectory_kind="agentknit_jsonl", formerly agent_probe)

These translators are source-agnostic: callers pass their own run/session
identifiers (``run_id``, ``trajectory_id``, ``profile``) and an optional
``extra_agent`` dict merged into ``agent.extra`` for any run-tracking
metadata they own (e.g. repo state, shadow commits). :func:`parsed_record_to_atif`
is a further generic entry point for callers that locate trajectories
through :func:`trajectoriz.iter_records` / :func:`trajectoriz.parse_record`
instead of managing per-store files themselves.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path

from . import (
    ParsedTrajectory,
    TrajectoryRecord,
    estimate_trajectory_tokens,
    parse_agent_probe_trajectory,
    parse_claude_trajectory,
    parse_codex_exec_trajectory,
    parse_codex_trajectory,
    parse_copilot_trajectory,
)


def _atif_envelope(
    *,
    session_id: str,
    run_id: str,
    trajectory_id: str = "",
    agent_name: str,
    agent_version: str,
    model_name: str,
    extra_agent: dict | None = None,
    steps: list | None = None,
    total_prompt: int = 0,
    total_completion: int = 0,
    total_cached: int = 0,
    total_tool_calls: int = 0,
) -> dict:
    agent_extra: dict = {"run_id": run_id}
    if extra_agent:
        agent_extra.update(extra_agent)
    steps = steps or []
    return {
        "schema_version": "ATIF-v1.7",
        "session_id": session_id,
        "trajectory_id": trajectory_id,
        "agent": {
            "name": agent_name,
            "version": agent_version,
            "model_name": model_name,
            "extra": agent_extra,
        },
        "steps": steps,
        "final_metrics": {
            "total_prompt_tokens": total_prompt,
            "total_completion_tokens": total_completion,
            "total_cached_tokens": total_cached,
            "total_cost_usd": 0.0,
            "total_steps": len(steps),
            "total_tool_calls": total_tool_calls,
        },
    }


def translate_claude_jsonl_to_atif(
    jsonl_path: Path,
    *,
    run_id: str,
    profile: str,
    trajectory_id: str = "",
    timestamp_start: str = "",
    extra_agent: dict | None = None,
) -> tuple[dict, int]:
    parsed = parse_claude_trajectory(jsonl_path, timestamp_start)
    return _atif_envelope(
        session_id=parsed.session_id or run_id,
        run_id=run_id,
        trajectory_id=trajectory_id,
        agent_name=profile,
        agent_version=parsed.agent_version or "",
        model_name=parsed.model_name or "",
        extra_agent=extra_agent,
        steps=parsed.steps,
        total_prompt=parsed.total_prompt_tokens,
        total_completion=parsed.total_completion_tokens,
        total_cached=parsed.total_cached_tokens,
        total_tool_calls=parsed.total_tool_calls,
    ), parsed.total_tool_calls


def translate_codex_jsonl_to_atif(
    jsonl_path: Path,
    *,
    run_id: str,
    profile: str,
    trajectory_id: str = "",
    timestamp_start: str = "",
    extra_agent: dict | None = None,
) -> tuple[dict, int]:
    parsed = parse_codex_trajectory(jsonl_path, timestamp_start)
    return _atif_envelope(
        session_id=parsed.session_id or run_id,
        run_id=run_id,
        trajectory_id=trajectory_id,
        agent_name=profile,
        agent_version=parsed.agent_version or "",
        model_name=parsed.model_name or "",
        extra_agent=extra_agent,
        steps=parsed.steps,
        total_prompt=parsed.total_prompt_tokens,
        total_completion=parsed.total_completion_tokens,
        total_cached=parsed.total_cached_tokens,
        total_tool_calls=parsed.total_tool_calls,
    ), parsed.total_tool_calls


def translate_agentknit_jsonl_to_atif(
    jsonl_path: Path,
    *,
    run_id: str,
    profile: str,
    trajectory_id: str = "",
    timestamp_start: str = "",
    extra_agent: dict | None = None,
) -> tuple[dict, int]:
    """Translate an agentknit (formerly agent_probe) session journal to ATIF."""
    parsed = parse_agent_probe_trajectory(jsonl_path, timestamp_start)
    return _atif_envelope(
        session_id=parsed.session_id or run_id,
        run_id=run_id,
        trajectory_id=trajectory_id,
        agent_name=profile,
        agent_version=parsed.agent_version or "",
        model_name=parsed.model_name or "",
        extra_agent=extra_agent,
        steps=parsed.steps,
        total_prompt=parsed.total_prompt_tokens,
        total_completion=parsed.total_completion_tokens,
        total_cached=parsed.total_cached_tokens,
        total_tool_calls=parsed.total_tool_calls,
    ), parsed.total_tool_calls


def codex_exec_prompt_from_command(command: list[str]) -> str:
    """Extract the prompt argument from a `codex exec ...` command line."""
    try:
        exec_index = command.index("exec")
    except ValueError:
        return ""

    args = command[exec_index + 1:]
    i = 0
    while i < len(args):
        arg = args[i]
        if arg == "--":
            return " ".join(args[i + 1:]).strip()
        if arg in {"--json", "--dangerously-bypass-approvals-and-sandbox"}:
            i += 1
            continue
        if arg in {"-c", "--config", "-o", "--output-last-message"}:
            i += 2
            continue
        if arg.startswith("-"):
            i += 1
            continue
        return arg
    return ""


def translate_codex_exec_jsonl_to_atif(
    stdout_text: str,
    *,
    run_id: str,
    profile: str,
    trajectory_id: str = "",
    prompt: str = "",
    timestamp_end: str = "",
    extra_agent: dict | None = None,
) -> tuple[dict, int]:
    """Translate `codex exec --json` stdout JSONL to ATIF."""
    parsed = parse_codex_exec_trajectory(stdout_text, prompt=prompt, fallback_ts=timestamp_end)
    return _atif_envelope(
        session_id=parsed.session_id or run_id,
        run_id=run_id,
        trajectory_id=trajectory_id,
        agent_name=profile,
        agent_version="",
        model_name="",
        extra_agent=extra_agent,
        steps=parsed.steps,
        total_prompt=parsed.total_prompt_tokens,
        total_completion=parsed.total_completion_tokens,
        total_cached=parsed.total_cached_tokens,
        total_tool_calls=parsed.total_tool_calls,
    ), parsed.total_tool_calls


def translate_copilot_sqlite_to_atif(
    db_path: Path,
    session_id: str,
    *,
    run_id: str,
    profile: str,
    trajectory_id: str = "",
    timestamp_start: str = "",
    extra_agent: dict | None = None,
) -> tuple[dict, int]:
    parsed = parse_copilot_trajectory(db_path, session_id, timestamp_start)
    agent_extra = dict(parsed.extra_agent)
    if extra_agent:
        agent_extra.update(extra_agent)
    return _atif_envelope(
        session_id=session_id,
        run_id=run_id,
        trajectory_id=trajectory_id,
        agent_name=profile,
        agent_version="",
        model_name="",
        extra_agent=agent_extra,
        steps=parsed.steps,
        total_tool_calls=0,
    ), 0


# ── Generic "translate any local trajectory record" entry point ───────────────
#
# The store-specific functions above expect the caller to already know which
# store a trajectory came from. Callers that locate trajectories through
# iter_records()/parse_record() instead can use this fully source-agnostic
# translator.


def _opencode_totals(session_id: str) -> tuple[int, int, int]:
    """Sum real per-message token usage for one opencode session.

    trajectoriz does not parse opencode trajectories into steps (it only
    exposes session-level metadata), and opencode's own session row double
    counts (it stores the *last* turn's cumulative numbers in some
    versions).  Summing the per-assistant-message ``tokens`` objects gives
    the provider-reported totals for the whole session.
    """
    db_path = Path.home() / ".local" / "share" / "opencode" / "opencode.db"
    if not db_path.exists():
        return 0, 0, 0
    prompt = completion = cached = 0
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        try:
            rows = conn.execute(
                "SELECT data FROM message WHERE session_id=? ORDER BY time_created",
                (session_id,),
            ).fetchall()
        finally:
            conn.close()
    except sqlite3.Error:
        return 0, 0, 0
    for (data,) in rows:
        try:
            msg = json.loads(data)
        except (json.JSONDecodeError, TypeError):
            continue
        tokens = msg.get("tokens")
        if not isinstance(tokens, dict):
            continue
        prompt += tokens.get("input") or 0
        completion += tokens.get("output") or 0
        cache = tokens.get("cache") or {}
        cached += (cache.get("read") or 0) + (cache.get("write") or 0)
    return prompt, completion, cached


def parsed_record_to_atif(
    record: TrajectoryRecord,
    parsed: ParsedTrajectory | None,
    *,
    run_id: str,
    profile: str,
    timestamp_start: str = "",
    timestamp_end: str = "",
    repo_root: str = "",
    extra_agent: dict | None = None,
) -> dict:
    """Translate any trajectoriz TrajectoryRecord/ParsedTrajectory to ATIF.

    For harnesses that locate the trajectory for a run themselves instead of
    going through a store-specific translator. Token semantics per store
    follow the store-specific translators above: real provider-reported
    usage when the store has it, estimation otherwise (opencode session
    metadata is fetched directly from its SQLite store because trajectoriz
    does not parse it).

    ``parsed`` may be None for stores trajectoriz cannot parse into steps
    (e.g. opencode); the metrics are then taken from the store directly.
    """
    if parsed is None:
        parsed = ParsedTrajectory()
    cwd = parsed.cwd or repo_root
    if not timestamp_start:
        timestamp_start = datetime.now().astimezone().isoformat(timespec="seconds")
    if not timestamp_end:
        timestamp_end = timestamp_start

    total_prompt = parsed.total_prompt_tokens
    total_completion = parsed.total_completion_tokens
    total_cached = parsed.total_cached_tokens
    total_tool_calls = parsed.total_tool_calls
    agent_extra: dict = {
        "repo_root": cwd,
        "trajectory_agent": record.agent,
        "trajectory_source": str(record.source),
        "first_message": record.first_msg,
        "error_count": parsed.error_count,
        "fatal_error_count": parsed.fatal_error_count,
        "token_limit_count": parsed.token_limit_count,
        "compaction_count": parsed.compaction_count,
    }
    if extra_agent:
        agent_extra.update(extra_agent)

    if record.agent == "opencode":
        source = record.source if isinstance(record.source, dict) else {}
        session_id = source.get("session_id", "")
        op_prompt, op_completion, op_cached = _opencode_totals(session_id)
        total_prompt, total_completion, total_cached = op_prompt, op_completion, op_cached
        agent_extra["opencode_model"] = source.get("model", "")
        agent_extra["token_source"] = "opencode_message_tokens"
    elif total_prompt or total_completion:
        agent_extra["token_source"] = "provider_reported"
    else:
        total_prompt = 0
        total_completion = estimate_trajectory_tokens(parsed)
        total_cached = 0
        agent_extra["token_source"] = "estimated"

    return _atif_envelope(
        session_id=parsed.session_id or run_id,
        run_id=run_id,
        trajectory_id=record.id,
        agent_name=profile,
        agent_version=parsed.agent_version or "",
        model_name=parsed.model_name or "",
        extra_agent=agent_extra,
        steps=parsed.steps,
        total_prompt=total_prompt,
        total_completion=total_completion,
        total_cached=total_cached,
        total_tool_calls=total_tool_calls,
    )
