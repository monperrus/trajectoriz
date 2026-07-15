"""Tests for trajectoriz.cli content search and step-jump features."""

import argparse
import json

import pytest

from trajectoriz import cli
from trajectoriz.cli import _render_step


def _base_step(**kwargs) -> dict:
    base = {"step_id": 1, "source": "agent", "timestamp": None, "tool_calls": [], "observation": None, "message": None}
    base.update(kwargs)
    return base


def test_render_step_tool_call_before_message():
    step = _base_step(
        tool_calls=[{"function_name": "read_file", "arguments": {"path": "/foo"}}],
        message="Here is the result.",
        observation={"results": [{"content": "file contents"}]},
    )
    rendered = _render_step(step)
    tc_pos = rendered.index("**Tool call:**")
    msg_pos = rendered.index("Here is the result.")
    assert tc_pos < msg_pos, "tool call should appear before message"


def test_render_step_empty_content_shows_placeholder():
    step = _base_step(
        tool_calls=[{"function_name": "run", "arguments": {}}],
        observation={"results": [{"content": ""}]},
    )
    rendered = _render_step(step)
    assert "*empty output*" in rendered


def test_render_step_none_content_shows_placeholder():
    step = _base_step(
        tool_calls=[{"function_name": "run", "arguments": {}}],
        observation={"results": [{"content": None}]},
    )
    rendered = _render_step(step)
    assert "*empty output*" in rendered


def test_render_step_none_observation_no_error():
    step = _base_step(observation=None)
    rendered = _render_step(step)
    assert "**Tool result:**" not in rendered


def test_render_step_message_only():
    step = _base_step(source="user", message="Hello agent.")
    rendered = _render_step(step)
    assert "USER" in rendered
    assert "Hello agent." in rendered
    assert "**Tool call:**" not in rendered


def test_main_without_args_shows_help(monkeypatch, capsys):
    monkeypatch.setattr(cli.sys, "argv", ["trajectoriz-cli"])

    cli.main()

    out = capsys.readouterr().out
    assert "usage: trajectoriz-cli" in out
    assert "Search and browse past agent trajectories." in out


def test_main_help_hides_delete_command(monkeypatch, capsys):
    monkeypatch.setattr(cli.sys, "argv", ["trajectoriz-cli", "--help"])

    with pytest.raises(SystemExit) as exc:
        cli.main()

    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "delete" not in out


def _write_claude_trajectory(path):
    """Write a small Claude Code trajectory JSONL with a tool call/result deep in it."""
    lines = [
        {"type": "user", "sessionId": "s1", "timestamp": "2024-01-01T00:00:00Z",
         "message": {"content": "fix the bug"}},
        {"type": "assistant", "timestamp": "2024-01-01T00:01:00Z", "message": {
            "model": "claude-sonnet-4-6",
            "content": [
                {"type": "text", "text": "Let me look into this."},
                {"type": "tool_use", "id": "call-1", "name": "Bash",
                 "input": {"command": "cat config.yaml"}},
            ],
        }},
        {"type": "user", "timestamp": "2024-01-01T00:02:00Z", "message": {
            "content": [
                {"type": "tool_result", "tool_use_id": "call-1",
                 "content": "model: claude-sonnet-4-6 # direct call config"},
            ],
        }},
        {"type": "assistant", "timestamp": "2024-01-01T00:03:00Z", "message": {
            "model": "claude-sonnet-4-6",
            "content": [{"type": "text", "text": "Found it, the config pins sonnet."}],
        }},
    ]
    path.write_text("\n".join(json.dumps(l) for l in lines) + "\n")


# ── _make_snippet ───────────────────────────────────────────────────────────


def test_make_snippet_no_match():
    assert cli._make_snippet("hello world", ["xyz"]) == ""


def test_make_snippet_with_context():
    text = "a" * 60 + "needle" + "b" * 60
    snippet = cli._make_snippet(text, ["needle"])
    assert "needle" in snippet
    assert snippet.startswith("…")
    assert snippet.endswith("…")
    assert len(snippet) < len(text)


def test_make_snippet_or_picks_first():
    text = "alpha is here and beta is further along"
    snippet = cli._make_snippet(text, ["beta", "alpha"])
    assert "alpha" in snippet


# ── _parse_terms ────────────────────────────────────────────────────────────


def test_parse_terms_single():
    assert cli._parse_terms("foo") == [["foo"]]


def test_parse_terms_or():
    assert cli._parse_terms(r"foo\|bar\|baz") == [["foo"], ["bar"], ["baz"]]


def test_parse_terms_lowercases():
    assert cli._parse_terms(r"Foo\|BAR") == [["foo"], ["bar"]]


def test_parse_terms_and_within_clause():
    assert cli._parse_terms("salary KTH overhead") == [["salary", "kth", "overhead"]]


def test_parse_terms_and_or_combined():
    assert cli._parse_terms(r"foo bar\|baz") == [["foo", "bar"], ["baz"]]


# ── _matches_any ─────────────────────────────────────────────────────────────


def test_matches_any_single_hit():
    assert cli._matches_any("hello world", [["world"]])


def test_matches_any_or():
    assert cli._matches_any("hello world", [["zzz"], ["world"]])
    assert not cli._matches_any("hello world", [["zzz"], ["xxx"]])


def test_matches_any_and_all_present():
    assert cli._matches_any("hello world foo", [["hello", "foo"]])
    assert not cli._matches_any("hello world", [["hello", "foo"]])


# ── _step_search_blobs ──────────────────────────────────────────────────────


def test_step_search_blobs_collects_all_fields():
    step = {
        "message": "hello",
        "reasoning_content": "thinking about it",
        "tool_calls": [{"function_name": "Bash", "arguments": {"command": "ls -la sonnet"}}],
        "observation": {"results": [{"content": "result text with sonnet"}]},
    }
    blobs = cli._step_search_blobs(step)
    assert "hello" in blobs
    assert "thinking about it" in blobs
    assert any("sonnet" in b for b in blobs if "command" in b)
    assert "result text with sonnet" in blobs


# ── _cached_parse ────────────────────────────────────────────────────────────


def test_cached_parse_reuses_cache(tmp_path):
    cache_dir = tmp_path / "cache"
    calls = {"n": 0}

    def parse_fn():
        calls["n"] += 1
        return {"value": calls["n"]}

    r1 = cli._cached_parse("key1", 100.0, parse_fn, cache_dir=cache_dir)
    r2 = cli._cached_parse("key1", 100.0, parse_fn, cache_dir=cache_dir)
    assert calls["n"] == 1
    assert r1 == r2 == {"value": 1}


def test_cached_parse_invalidated_by_mtime(tmp_path):
    cache_dir = tmp_path / "cache"
    calls = {"n": 0}

    def parse_fn():
        calls["n"] += 1
        return {"value": calls["n"]}

    cli._cached_parse("key1", 100.0, parse_fn, cache_dir=cache_dir)
    cli._cached_parse("key1", 200.0, parse_fn, cache_dir=cache_dir)
    assert calls["n"] == 2


# ── _parse_record ────────────────────────────────────────────────────────────


def test_parse_record_claude(tmp_path):
    f = tmp_path / "session.jsonl"
    _write_claude_trajectory(f)
    rec = cli.TrajRecord("cl-abc", "claude", "2024-01-01T00:00:00Z", "fix the bug", f)

    traj = cli._parse_record(rec, cache_dir=tmp_path / "cache")
    assert traj is not None
    # The pure tool_result user message is folded into step 2's observation,
    # so 4 entries collapse to 3 steps.
    assert len(traj.steps) == 3


def test_parse_record_unsupported_returns_none(tmp_path):
    rec = cli.TrajRecord("oc-abc", "opencode", "", "hi",
                          {"type": "opencode", "session_id": "x", "model": "m", "dir": "/repo"})
    assert cli._parse_record(rec, cache_dir=tmp_path / "cache") is None


# ── cmd_search_content ───────────────────────────────────────────────────────


def test_cmd_search_content_finds_deep_match(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "_cache_dir", lambda _=None: tmp_path / "cache")

    f = tmp_path / "session.jsonl"
    _write_claude_trajectory(f)
    rec = cli.TrajRecord("cl-abc", "claude", "2024-01-01T00:00:00Z", "fix the bug", f)

    args = argparse.Namespace(query="sonnet", page=1, page_size=50, last=False)
    cli.cmd_search_content(args, [["sonnet"]], [rec])

    out = capsys.readouterr().out
    assert "cl-abc" in out
    assert "## Search:" in out
    # The match is in step 2 (tool result observation) or step 3 (closing message).
    assert "| 2 |" in out or "| 3 |" in out


def test_cmd_search_content_or_query(tmp_path, monkeypatch, capsys):
    """OR query: matching either term finds the trajectory."""
    monkeypatch.setattr(cli, "_cache_dir", lambda _=None: tmp_path / "cache")

    f = tmp_path / "session.jsonl"
    _write_claude_trajectory(f)
    rec = cli.TrajRecord("cl-abc", "claude", "2024-01-01T00:00:00Z", "fix the bug", f)

    # "sonnet" is in the trajectory; "zzz_missing" is not — OR should still match
    args = argparse.Namespace(query=r"zzz_missing\|sonnet", page=1, page_size=50, last=False)
    cli.cmd_search_content(args, cli._parse_terms(args.query), [rec])

    out = capsys.readouterr().out
    assert "cl-abc" in out


def test_cmd_advanced_tools_dir_json(monkeypatch, capsys):
    rec = cli.TrajRecord("cl-abc", "claude", "2024-01-01T00:00:00Z", "fix the bug", object())
    parsed = cli.tz.ParsedTrajectory(
        steps=[
            {
                "source": "agent",
                "tool_calls": [
                    {"function_name": "Bash", "arguments": {"command": "python script.py"}},
                    {"function_name": "Bash", "arguments": {"command": "git status"}},
                ],
            },
            {
                "source": "agent",
                "tool_calls": [
                    {"function_name": "run_command", "arguments": {"cmd": "python -m pytest"}},
                ],
            },
        ]
    )

    monkeypatch.setattr(cli, "_local_records", lambda path: [rec])
    monkeypatch.setattr(cli, "_parse_record", lambda record: parsed)

    args = argparse.Namespace(id=None, dir="/tmp/repo", json=True)
    cli.cmd_advanced_tools(args)

    out = json.loads(capsys.readouterr().out)
    assert out == {
        "scope": {"type": "dir", "path": "/tmp/repo"},
        "programs": [
            {"program": "python", "count": 2},
            {"program": "git", "count": 1},
        ],
    }


def test_cmd_advanced_tools_id_json(monkeypatch, capsys):
    rec = cli.TrajRecord("cl-abc", "claude", "2024-01-01T00:00:00Z", "fix the bug", object())
    parsed = cli.tz.ParsedTrajectory(
        steps=[
            {
                "source": "agent",
                "tool_calls": [
                    {"function_name": "shell", "arguments": {"command": "/usr/bin/make test"}},
                ],
            },
        ]
    )

    monkeypatch.setattr(cli, "_find_record", lambda record_id: rec if record_id == "cl-abc" else None)
    monkeypatch.setattr(cli, "_parse_record", lambda record: parsed)

    args = argparse.Namespace(id="cl-abc", dir=None, json=True)
    cli.cmd_advanced_tools(args)

    out = json.loads(capsys.readouterr().out)
    assert out == {
        "scope": {"type": "id", "id": "cl-abc"},
        "programs": [
            {"program": "make", "count": 1},
        ],
    }


def test_cmd_search_content_no_match(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "_cache_dir", lambda _=None: tmp_path / "cache")

    f = tmp_path / "session.jsonl"
    _write_claude_trajectory(f)
    rec = cli.TrajRecord("cl-abc", "claude", "2024-01-01T00:00:00Z", "fix the bug", f)

    args = argparse.Namespace(query="zzz_not_present", page=1, page_size=50, last=False)
    cli.cmd_search_content(args, [["zzz_not_present"]], [rec])

    out = capsys.readouterr().out
    assert "No trajectories found" in out


# ── cmd_show --step ──────────────────────────────────────────────────────────


def test_cmd_show_step_jumps_to_correct_page(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "_cache_dir", lambda _=None: tmp_path / "cache")

    f = tmp_path / "session.jsonl"
    _write_claude_trajectory(f)
    rec = cli.TrajRecord("cl-abc", "claude", "2024-01-01T00:00:00Z", "fix the bug", f)
    monkeypatch.setattr(cli, "_all_records", lambda: iter([rec]))

    # 3 steps total, page_size=2 -> step 3 should land on page 2.
    args = argparse.Namespace(id="cl-abc", page=1, page_size=2, step=3, full=False, last=False)
    cli.cmd_show(args)

    out = capsys.readouterr().out
    assert "page 2/2" in out
    assert "Step 3" in out


def _write_claude_trajectory_with_cwd(path, cwd="/home/user/myrepo"):
    """Write a Claude trajectory JSONL that includes a cwd in the first entry."""
    lines = [
        {"type": "system", "cwd": cwd},
        {"type": "user", "sessionId": "s1", "timestamp": "2024-06-01T10:00:00Z",
         "message": {"content": "refactor the module"}},
        {"type": "assistant", "timestamp": "2024-06-01T10:00:01Z", "message": {
            "model": "claude-sonnet-4-6",
            "content": [{"type": "text", "text": "Done."}],
            "usage": {"input_tokens": 20, "output_tokens": 5},
        }},
    ]
    path.write_text("\n".join(json.dumps(l) for l in lines) + "\n")


def test_show_header_includes_cwd(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "_cache_dir", lambda _=None: tmp_path / "cache")

    f = tmp_path / "session.jsonl"
    _write_claude_trajectory_with_cwd(f, cwd="/home/user/myrepo")
    rec = cli.TrajRecord("cl-xyz", "claude", "2024-06-01T10:00:00Z", "refactor the module", f)
    monkeypatch.setattr(cli, "_all_records", lambda: iter([rec]))

    args = argparse.Namespace(id="cl-xyz", page=1, page_size=20, step=None, full=False, last=False)
    cli.cmd_show(args)

    out = capsys.readouterr().out
    assert "**Directory:** /home/user/myrepo" in out


def test_cmd_info_prints_json(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "_cache_dir", lambda _=None: tmp_path / "cache")

    f = tmp_path / "session.jsonl"
    _write_claude_trajectory_with_cwd(f, cwd="/home/user/myrepo")
    rec = cli.TrajRecord("cl-xyz", "claude", "2024-06-01T10:00:00Z", "refactor the module", f)
    monkeypatch.setattr(cli, "_all_records", lambda: iter([rec]))

    args = argparse.Namespace(id="cl-xyz")
    cli.cmd_info(args)

    out = capsys.readouterr().out
    data = json.loads(out)
    assert data["id"] == "cl-xyz"
    assert data["agent"] == "claude"
    assert data["directory"] == "/home/user/myrepo"
    assert data["model"] == "claude-sonnet-4-6"
    assert data["timestamp"] == "2024-06-01T10:00:00Z"
    assert data["first_message"] == "refactor the module"
    # info must not include steps
    assert "steps" not in data or isinstance(data["steps"], int)


def test_cmd_info_not_found(monkeypatch, capsys):
    monkeypatch.setattr(cli, "_all_records", lambda: iter([]))

    args = argparse.Namespace(id="cl-missing")
    with pytest.raises(SystemExit):
        cli.cmd_info(args)

    out = capsys.readouterr().out
    data = json.loads(out)
    assert "error" in data


def test_main_info_command(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "_cache_dir", lambda _=None: tmp_path / "cache")

    f = tmp_path / "session.jsonl"
    _write_claude_trajectory_with_cwd(f, cwd="/srv/app")
    rec = cli.TrajRecord("cl-abc", "claude", "2024-06-01T00:00:00Z", "fix it", f)
    monkeypatch.setattr(cli, "_all_records", lambda: iter([rec]))
    monkeypatch.setattr(cli.sys, "argv", ["trajectoriz-cli", "info", "cl-abc"])

    cli.main()

    out = capsys.readouterr().out
    data = json.loads(out)
    assert data["directory"] == "/srv/app"


def test_cmd_show_step_out_of_range(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "_cache_dir", lambda _=None: tmp_path / "cache")

    f = tmp_path / "session.jsonl"
    _write_claude_trajectory(f)
    rec = cli.TrajRecord("cl-abc", "claude", "2024-01-01T00:00:00Z", "fix the bug", f)
    monkeypatch.setattr(cli, "_all_records", lambda: iter([rec]))

    args = argparse.Namespace(id="cl-abc", page=1, page_size=2, step=99, full=False, last=False)
    with pytest.raises(SystemExit):
        cli.cmd_show(args)

    err = capsys.readouterr().err
    assert "out of range" in err


# ── cmd_stats ────────────────────────────────────────────────────────────────


def test_cmd_stats_empty(monkeypatch, capsys):
    monkeypatch.setattr(cli, "_all_records", lambda: iter([]))
    # Also mock _local_records so that --all=False uses the mocked data
    monkeypatch.setattr(cli, "_local_records", lambda _: iter([]))

    args = argparse.Namespace(all=False)
    cli.cmd_stats(args)

    out = capsys.readouterr().out
    data = json.loads(out)
    assert data["total_trajectories"] == 0
    assert data["agents"] == {}


def test_cmd_stats_with_trajectories(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "_cache_dir", lambda _=None: tmp_path / "cache")

    f1 = tmp_path / "session1.jsonl"
    _write_claude_trajectory_with_cwd(f1, cwd="/home/user/repo1")
    rec1 = cli.TrajRecord("cl-abc", "claude", "2024-06-01T00:00:00Z", "fix it", f1)

    f2 = tmp_path / "session2.jsonl"
    _write_claude_trajectory_with_cwd(f2, cwd="/home/user/repo2")
    rec2 = cli.TrajRecord("cl-def", "claude", "2024-06-02T00:00:00Z", "add feature", f2)

    monkeypatch.setattr(cli, "_all_records", lambda: iter([rec1, rec2]))

    args = argparse.Namespace(all=True)
    cli.cmd_stats(args)

    out = capsys.readouterr().out
    data = json.loads(out)
    assert data["total_trajectories"] == 2
    assert data["parsed_trajectories"] == 2
    assert data["agents"] == {"claude": 2}
    assert data["total_tool_calls"] >= 0
    assert data["total_tokens"] >= 0


def test_main_stats_command(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "_cache_dir", lambda _=None: tmp_path / "cache")

    f = tmp_path / "session.jsonl"
    _write_claude_trajectory_with_cwd(f, cwd="/srv/app")
    rec = cli.TrajRecord("cl-abc", "claude", "2024-06-01T00:00:00Z", "fix it", f)
    monkeypatch.setattr(cli, "_all_records", lambda: iter([rec]))
    monkeypatch.setattr(cli, "_local_records", lambda _: iter([rec]))
    monkeypatch.setattr(cli.sys, "argv", ["trajectoriz-cli", "stats"])

    cli.main()

    out = capsys.readouterr().out
    data = json.loads(out)
    assert data["total_trajectories"] == 1
    assert data["agents"] == {"claude": 1}
