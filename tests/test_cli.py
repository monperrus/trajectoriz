"""Tests for trajectoriz.cli content search and step-jump features."""

import argparse
import json

import pytest

from trajectoriz import cli


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
    assert cli._parse_terms("foo") == ["foo"]


def test_parse_terms_or():
    assert cli._parse_terms(r"foo\|bar\|baz") == ["foo", "bar", "baz"]


def test_parse_terms_lowercases():
    assert cli._parse_terms(r"Foo\|BAR") == ["foo", "bar"]


# ── _matches_any ─────────────────────────────────────────────────────────────


def test_matches_any_single_hit():
    assert cli._matches_any("hello world", ["world"])


def test_matches_any_or():
    assert cli._matches_any("hello world", ["zzz", "world"])
    assert not cli._matches_any("hello world", ["zzz", "xxx"])


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
    monkeypatch.setattr(cli, "_cache_dir", lambda _cd=None: tmp_path / "cache")

    f = tmp_path / "session.jsonl"
    _write_claude_trajectory(f)
    rec = cli.TrajRecord("cl-abc", "claude", "2024-01-01T00:00:00Z", "fix the bug", f)

    args = argparse.Namespace(query="sonnet", page=1, page_size=50)
    cli.cmd_search_content(args, ["sonnet"], [rec])

    out = capsys.readouterr().out
    assert "cl-abc" in out
    assert "## Search:" in out
    # The match is in step 2 (tool result observation) or step 3 (closing message).
    assert "| 2 |" in out or "| 3 |" in out


def test_cmd_search_content_or_query(tmp_path, monkeypatch, capsys):
    """OR query: matching either term finds the trajectory."""
    monkeypatch.setattr(cli, "_cache_dir", lambda _cd=None: tmp_path / "cache")

    f = tmp_path / "session.jsonl"
    _write_claude_trajectory(f)
    rec = cli.TrajRecord("cl-abc", "claude", "2024-01-01T00:00:00Z", "fix the bug", f)

    # "sonnet" is in the trajectory; "zzz_missing" is not — OR should still match
    args = argparse.Namespace(query=r"zzz_missing\|sonnet", page=1, page_size=50)
    cli.cmd_search_content(args, cli._parse_terms(args.query), [rec])

    out = capsys.readouterr().out
    assert "cl-abc" in out
    assert "## Search:" in out


def test_cmd_search_content_no_match(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "_cache_dir", lambda _cd=None: tmp_path / "cache")

    f = tmp_path / "session.jsonl"
    _write_claude_trajectory(f)
    rec = cli.TrajRecord("cl-abc", "claude", "2024-01-01T00:00:00Z", "fix the bug", f)

    args = argparse.Namespace(query="zzz_not_present", page=1, page_size=50)
    cli.cmd_search_content(args, ["zzz_not_present"], [rec])

    out = capsys.readouterr().out
    assert "No trajectories found" in out


# ── cmd_show --step ──────────────────────────────────────────────────────────


def test_cmd_show_step_jumps_to_correct_page(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "_cache_dir", lambda _cd=None: tmp_path / "cache")

    f = tmp_path / "session.jsonl"
    _write_claude_trajectory(f)
    rec = cli.TrajRecord("cl-abc", "claude", "2024-01-01T00:00:00Z", "fix the bug", f)
    monkeypatch.setattr(cli, "_all_records", lambda: iter([rec]))

    # 3 steps total, page_size=2 -> step 3 should land on page 2.
    args = argparse.Namespace(id="cl-abc", page=1, page_size=2, step=3, full=False)
    cli.cmd_show(args)

    out = capsys.readouterr().out
    assert "page 2/2" in out
    assert "Step 3" in out


def test_cmd_show_step_out_of_range(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "_cache_dir", lambda _cd=None: tmp_path / "cache")

    f = tmp_path / "session.jsonl"
    _write_claude_trajectory(f)
    rec = cli.TrajRecord("cl-abc", "claude", "2024-01-01T00:00:00Z", "fix the bug", f)
    monkeypatch.setattr(cli, "_all_records", lambda: iter([rec]))

    args = argparse.Namespace(id="cl-abc", page=1, page_size=2, step=99, full=False)
    with pytest.raises(SystemExit):
        cli.cmd_show(args)

    err = capsys.readouterr().err
    assert "out of range" in err
