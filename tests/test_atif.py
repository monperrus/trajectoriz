"""Tests for trajectoriz.atif — ATIF v1.7 translation."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

import trajectoriz
from trajectoriz import atif


def _write_jsonl(path: Path, lines: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(x) for x in lines), encoding="utf-8")


def test_atif_envelope_shape():
    envelope = atif._atif_envelope(
        session_id="sess-1",
        run_id="run-1",
        trajectory_id="traj-1",
        agent_name="claude",
        agent_version="1.0",
        model_name="model",
        extra_agent={"provider": "x"},
        steps=[{"kind": "message"}],
        total_prompt=10,
        total_completion=20,
        total_cached=30,
        total_tool_calls=2,
    )
    assert envelope["schema_version"] == "ATIF-v1.7"
    assert envelope["trajectory_id"] == "traj-1"
    assert envelope["agent"]["extra"]["run_id"] == "run-1"
    assert envelope["agent"]["extra"]["provider"] == "x"
    assert envelope["final_metrics"]["total_steps"] == 1
    assert envelope["final_metrics"]["total_tool_calls"] == 2


def test_translate_claude_jsonl_to_atif(tmp_path: Path):
    jsonl = tmp_path / "traj.jsonl"
    _write_jsonl(jsonl, [
        {"sessionId": "sess-2", "type": "user", "timestamp": "2026-01-01T00:00:00",
         "message": {"content": "hi"}},
        {"type": "assistant", "timestamp": "2026-01-01T00:00:01",
         "message": {"model": "claude-x", "content": [{"type": "text", "text": "hello"}],
                      "usage": {"input_tokens": 1, "output_tokens": 2}}},
    ])
    out, tool_calls = atif.translate_claude_jsonl_to_atif(
        jsonl, run_id="run-1", profile="claude", trajectory_id="traj-1",
        extra_agent={"repo_root": "/repo"},
    )
    assert out["schema_version"] == "ATIF-v1.7"
    assert out["session_id"] == "sess-2"
    assert out["trajectory_id"] == "traj-1"
    assert out["agent"]["model_name"] == "claude-x"
    assert out["agent"]["extra"]["repo_root"] == "/repo"
    assert tool_calls == 0


def test_translate_codex_jsonl_to_atif(tmp_path: Path):
    jsonl = tmp_path / "rollout.jsonl"
    _write_jsonl(jsonl, [
        {"type": "session_meta", "payload": {"id": "thread-1", "cli_version": "2.0", "cwd": "/work"}},
        {"type": "event_msg", "payload": {"type": "user_message", "message": "do it"}},
    ])
    out, tool_calls = atif.translate_codex_jsonl_to_atif(
        jsonl, run_id="run-1", profile="codex",
    )
    assert out["session_id"] == "thread-1"
    assert out["agent"]["version"] == "2.0"
    assert tool_calls == 0


def test_translate_agentknit_jsonl_to_atif(tmp_path: Path):
    jsonl = tmp_path / "traj.jsonl"
    _write_jsonl(jsonl, [
        {"type": "session_start", "model": "glm-5.3", "session_id": "s1", "cwd": "/work", "ts": "2026-01-01T00:00:00"},
        {"type": "user", "content": "do the thing", "ts": "2026-01-01T00:00:01"},
        {"type": "usage", "prompt_tokens": 100, "completion_tokens": 20, "cached_tokens": 50, "ts": "2026-01-01T00:00:02"},
    ])
    out, tool_calls = atif.translate_agentknit_jsonl_to_atif(
        jsonl, run_id="run-1", profile="agentknit",
    )
    assert out["schema_version"] == "ATIF-v1.7"
    assert out["session_id"] == "s1"
    assert out["agent"]["name"] == "agentknit"
    assert out["agent"]["model_name"] == "glm-5.3"
    assert out["final_metrics"]["total_prompt_tokens"] == 100
    assert out["final_metrics"]["total_completion_tokens"] == 20
    assert out["final_metrics"]["total_cached_tokens"] == 50
    assert tool_calls == 0


def test_codex_exec_prompt_from_command():
    assert atif.codex_exec_prompt_from_command(
        ["npx", "@openai/codex", "exec", "--json", "list files"]
    ) == "list files"
    assert atif.codex_exec_prompt_from_command(["codex", "exec", "--", "do -x thing"]) == "do -x thing"
    assert atif.codex_exec_prompt_from_command(["codex"]) == ""


def test_translate_codex_exec_jsonl_to_atif():
    stdout = "\n".join([
        json.dumps({"type": "thread.started", "thread_id": "thread-1"}),
        json.dumps({"type": "turn.started"}),
        json.dumps({
            "type": "item.completed",
            "item": {
                "id": "item_0",
                "type": "command_execution",
                "command": "/bin/bash -lc ls",
                "aggregated_output": "README.md\n",
                "exit_code": 0,
                "status": "completed",
            },
        }),
        json.dumps({
            "type": "item.completed",
            "item": {"id": "item_1", "type": "agent_message", "text": "Done."},
        }),
        json.dumps({
            "type": "turn.completed",
            "usage": {"input_tokens": 10, "cached_input_tokens": 3, "output_tokens": 2},
        }),
    ])
    prompt = atif.codex_exec_prompt_from_command(
        ["npx", "@openai/codex", "exec", "--json", "list files"]
    )
    out, tool_calls = atif.translate_codex_exec_jsonl_to_atif(
        stdout, run_id="run-1", profile="codex", prompt=prompt,
    )
    assert out["session_id"] == "thread-1"
    assert out["steps"][0]["message"] == "list files"
    assert out["steps"][1]["message"] == "Done."
    assert out["steps"][1]["tool_calls"][0]["function_name"] == "command_execution"
    assert out["steps"][1]["observation"]["results"][0]["content"] == "README.md\n"
    assert out["final_metrics"]["total_prompt_tokens"] == 10
    assert tool_calls == 1
    assert trajectoriz.codex_exec_jsonl_final_message(stdout) == "Done."


def test_translate_copilot_sqlite_to_atif(tmp_path: Path):
    db = tmp_path / "sessions.db"
    conn = sqlite3.connect(str(db))
    conn.executescript(
        """
        CREATE TABLE sessions (id text PRIMARY KEY, cwd text, repository text, branch text, summary text, created_at text);
        CREATE TABLE turns (session_id text, turn_index integer, user_message text, assistant_response text, timestamp text);
        """
    )
    conn.execute("INSERT INTO sessions VALUES ('session-x', '/work', 'org/repo', 'main', 'a summary', '2026-01-01')")
    conn.execute("INSERT INTO turns VALUES ('session-x', 0, 'hi', 'hello', '2026-01-01T00:00:00')")
    conn.commit()
    conn.close()

    out, tool_calls = atif.translate_copilot_sqlite_to_atif(
        db, "session-x", run_id="run-1", profile="copilot",
    )
    assert out["session_id"] == "session-x"
    assert out["agent"]["extra"]["copilot_repository"] == "org/repo"
    assert tool_calls == 0


def test_parsed_record_to_atif_agent_probe(tmp_path: Path):
    jsonl = tmp_path / "traj.jsonl"
    _write_jsonl(jsonl, [
        {"type": "session_start", "model": "glm-5.3", "session_id": "s1", "cwd": "/work", "ts": "2026-01-01T00:00:00"},
        {"type": "user", "content": "do the thing", "ts": "2026-01-01T00:00:01"},
        {"type": "usage", "prompt_tokens": 100, "completion_tokens": 20, "cached_tokens": 50, "ts": "2026-01-01T00:00:02"},
    ])
    record = trajectoriz.TrajectoryRecord("ap-x", "agent_probe", "2026-01-01T00:00:00", "do the thing", jsonl)
    parsed = trajectoriz.parse_agent_probe_trajectory(jsonl)

    out = atif.parsed_record_to_atif(record, parsed, run_id="run-1", profile="glm")

    assert out["schema_version"] == "ATIF-v1.7"
    assert out["session_id"] == "s1"
    metrics = out["final_metrics"]
    assert metrics["total_prompt_tokens"] == 100
    assert metrics["total_completion_tokens"] == 20
    assert metrics["total_cached_tokens"] == 50
    extra = out["agent"]["extra"]
    assert extra["token_source"] == "provider_reported"
    assert extra["trajectory_agent"] == "agent_probe"
    assert extra["first_message"] == "do the thing"


def test_parsed_record_to_atif_estimates_without_usage(tmp_path: Path):
    jsonl = tmp_path / "traj.jsonl"
    _write_jsonl(jsonl, [
        {"type": "session_start", "model": "m", "session_id": "s2", "cwd": "/work", "ts": "2026-01-01T00:00:00"},
        {"type": "user", "content": "write a very long answer about tokens please", "ts": "2026-01-01T00:00:01"},
        {"type": "assistant", "content": "one two three four five six seven eight nine ten " * 20, "ts": "2026-01-01T00:00:02"},
    ])
    record = trajectoriz.TrajectoryRecord("ap-y", "agent_probe", "2026-01-01T00:00:00", "write a very long answer about tokens please", jsonl)
    parsed = trajectoriz.parse_agent_probe_trajectory(jsonl)

    out = atif.parsed_record_to_atif(record, parsed, run_id="run-2", profile="old-agent")

    assert out["agent"]["extra"]["token_source"] == "estimated"
    assert out["final_metrics"]["total_completion_tokens"] > 0
    assert out["final_metrics"]["total_prompt_tokens"] == 0


def test_parsed_record_to_atif_opencode(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    db = tmp_path / "opencode.db"
    conn = sqlite3.connect(str(db))
    conn.executescript(
        """
        CREATE TABLE message (id text PRIMARY KEY, session_id text, time_created integer, time_updated integer, data text);
        INSERT INTO message VALUES ('m1', 'ses_1', 1, 1, ?);
        INSERT INTO message VALUES ('m2', 'ses_1', 2, 2, ?);
        """
    )
    tokens_1 = json.dumps({"role": "assistant", "tokens": {"input": 1000, "output": 10, "cache": {"read": 500, "write": 0}}})
    tokens_2 = json.dumps({"role": "assistant", "tokens": {"input": 200, "output": 5, "cache": {"read": 1700, "write": 0}}})
    conn.execute("UPDATE message SET data=? WHERE id='m1'", (tokens_1,))
    conn.execute("UPDATE message SET data=? WHERE id='m2'", (tokens_2,))
    conn.commit()
    conn.close()

    monkeypatch.setattr(atif.Path, "home", lambda: tmp_path)
    (tmp_path / ".local/share/opencode").mkdir(parents=True)
    db.rename(tmp_path / ".local" / "share" / "opencode" / "opencode.db")

    source = {"type": "opencode", "session_id": "ses_1", "model": "mimo", "dir": "/tmp/x"}
    record = trajectoriz.TrajectoryRecord("oc-1", "opencode", "1", "prompt", source)

    out = atif.parsed_record_to_atif(record, None, run_id="run-3", profile="opencode-free")

    assert out["agent"]["extra"]["token_source"] == "opencode_message_tokens"
    assert out["agent"]["extra"]["opencode_model"] == "mimo"
    metrics = out["final_metrics"]
    assert metrics["total_prompt_tokens"] == 1200
    assert metrics["total_completion_tokens"] == 15
    assert metrics["total_cached_tokens"] == 2200
    # no steps from trajectoriz for opencode, but the metrics are real
    assert metrics["total_steps"] == 0


def test_parsed_record_to_atif_accepts_none_parsed():
    record = trajectoriz.TrajectoryRecord("x", "unknown", "", "", {"type": "opencode", "session_id": "nope"})
    out = atif.parsed_record_to_atif(record, None, run_id="r", profile="p")
    assert out["schema_version"] == "ATIF-v1.7"
    assert out["final_metrics"]["total_prompt_tokens"] == 0


def test_parsed_record_to_atif_accepts_deprecated_timestamp_end():
    """timestamp_end is accepted but unused, kept for external callers (e.g. agent_benchmark)."""
    record = trajectoriz.TrajectoryRecord("x", "unknown", "", "", {"type": "opencode", "session_id": "nope"})
    out = atif.parsed_record_to_atif(
        record, None, run_id="r", profile="p",
        timestamp_start="2026-01-01T00:00:00", timestamp_end="2026-01-01T00:01:00",
    )
    assert out["schema_version"] == "ATIF-v1.7"
