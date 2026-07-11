"""Tests for trajectoriz."""

import json

from trajectoriz import __version__


def test_version():
    assert __version__ == "0.1.0"


def test_iter_claude_trajectories_empty(tmp_path):
    from trajectoriz import iter_claude_trajectories

    assert list(iter_claude_trajectories(claude_dir=str(tmp_path))) == []


def test_iter_codex_trajectories_empty(tmp_path):
    from trajectoriz import iter_codex_trajectories

    assert list(iter_codex_trajectories(codex_dir=str(tmp_path))) == []


def test_iter_pi_trajectories_empty(tmp_path):
    from trajectoriz import iter_pi_trajectories

    assert list(iter_pi_trajectories(pi_dir=str(tmp_path))) == []


def test_iter_cursor_trajectories_empty(tmp_path):
    from trajectoriz import iter_cursor_trajectories

    assert list(iter_cursor_trajectories(cursor_dir=str(tmp_path))) == []


def test_iter_copilot_sessions_empty(tmp_path):
    from trajectoriz import iter_copilot_sessions

    assert list(iter_copilot_sessions(copilot_dir=str(tmp_path))) == []


def test_iter_claude_trajectories_with_files(tmp_path):
    from trajectoriz import iter_claude_trajectories

    proj = tmp_path / "projects" / "my-repo"
    proj.mkdir(parents=True)
    (proj / "session1.jsonl").write_text("{}")
    (proj / "session2.jsonl").write_text("{}")
    results = list(iter_claude_trajectories(claude_dir=str(tmp_path)))
    assert len(results) == 2
    assert all(str(p).endswith(".jsonl") for p in results)


def test_iter_codex_trajectories_with_files(tmp_path):
    from trajectoriz import iter_codex_trajectories

    sessions = tmp_path / "sessions" / "abc"
    sessions.mkdir(parents=True)
    (sessions / "run.jsonl").write_text("{}")
    results = list(iter_codex_trajectories(codex_dir=str(tmp_path)))
    assert len(results) == 1


def test_iter_codex_rollout_files(tmp_path):
    from trajectoriz import iter_codex_rollout_files

    sessions = tmp_path / "sessions" / "abc"
    sessions.mkdir(parents=True)
    (sessions / "rollout-001.jsonl").write_text("{}")
    (sessions / "other.jsonl").write_text("{}")
    results = list(iter_codex_rollout_files(codex_dir=str(tmp_path)))
    assert len(results) == 1
    assert "rollout-" in str(results[0])


def test_claude_project_dir():
    from trajectoriz import claude_project_dir

    result = claude_project_dir("/home/user/my-repo", claude_dir="/tmp/claude")
    assert str(result) == "/tmp/claude/projects/-home-user-my-repo"


def test_iter_pi_trajectories_with_files(tmp_path):
    from trajectoriz import iter_pi_trajectories

    sessions = tmp_path / "sessions" / "sub"
    sessions.mkdir(parents=True)
    (sessions / "sess1.jsonl").write_text("{}")
    results = list(iter_pi_trajectories(pi_dir=str(tmp_path)))
    assert len(results) == 1


def test_iter_cursor_trajectories_with_files(tmp_path):
    from trajectoriz import iter_cursor_trajectories

    sess = tmp_path / "sessions" / "proj"
    sess.mkdir(parents=True)
    (sess / "s1.jsonl").write_text("{}")
    results = list(iter_cursor_trajectories(cursor_dir=str(tmp_path)))
    assert len(results) == 1


def test_iter_copilot_event_trajectories_empty(tmp_path):
    from trajectoriz import iter_copilot_event_trajectories

    assert list(iter_copilot_event_trajectories(copilot_dir=str(tmp_path))) == []


def test_iter_copilot_event_trajectories_with_files(tmp_path):
    from trajectoriz import iter_copilot_event_trajectories

    sess = tmp_path / "session-state" / "abc123"
    sess.mkdir(parents=True)
    (sess / "events.jsonl").write_text("{}")
    results = list(iter_copilot_event_trajectories(copilot_dir=str(tmp_path)))
    assert len(results) == 1
    assert results[0].name == "events.jsonl"


def test_iter_agent_probe_trajectories_empty(tmp_path):
    from trajectoriz import iter_agent_probe_trajectories

    assert list(iter_agent_probe_trajectories(agent_probe_dir=str(tmp_path))) == []


def test_iter_agent_probe_trajectories_with_files(tmp_path):
    from trajectoriz import iter_agent_probe_trajectories

    sess = tmp_path / "2024" / "01"
    sess.mkdir(parents=True)
    (sess / "session.jsonl").write_text("{}")
    results = list(iter_agent_probe_trajectories(agent_probe_dir=str(tmp_path)))
    assert len(results) == 1


def test_iter_codex_db_sessions_empty(tmp_path):
    from trajectoriz import iter_codex_db_sessions

    assert list(iter_codex_db_sessions(codex_dir=str(tmp_path))) == []


def test_iter_codex_db_sessions_with_data(tmp_path):
    import sqlite3

    from trajectoriz import CodexDbSession, iter_codex_db_sessions

    db = tmp_path / "state_5.sqlite"
    conn = sqlite3.connect(str(db))
    conn.execute(
        "CREATE TABLE threads (id TEXT, rollout_path TEXT, created_at_ms INTEGER, "
        "updated_at_ms INTEGER, model_provider TEXT, model TEXT, cwd TEXT, "
        "title TEXT, tokens_used INTEGER, first_user_message TEXT)"
    )
    conn.execute(
        "INSERT INTO threads VALUES ('id1', '/path/rollout.jsonl', 900, 1000, "
        "'openai', 'gpt-4', '/repo', 'My session', 42, 'hello')"
    )
    conn.commit()
    conn.close()

    results = list(iter_codex_db_sessions(codex_dir=str(tmp_path)))
    assert len(results) == 1
    s = results[0]
    assert isinstance(s, CodexDbSession)
    assert s.id == "id1"
    assert s.rollout_path == "/path/rollout.jsonl"
    assert s.created_at_ms == 900
    assert s.updated_at_ms == 1000
    assert s.model_provider == "openai"
    assert s.model == "gpt-4"
    assert s.cwd == "/repo"
    assert s.title == "My session"
    assert s.tokens_used == 42
    assert s.first_user_message == "hello"


def test_iter_opencode_sessions_empty(tmp_path):
    from trajectoriz import iter_opencode_sessions

    assert list(iter_opencode_sessions(opencode_dir=str(tmp_path))) == []


def test_get_first_user_message_claude(tmp_path):
    import json

    from trajectoriz import get_first_user_message_claude

    f = tmp_path / "session.jsonl"
    f.write_text(
        json.dumps({"type": "system", "timestamp": "2024-01-01T00:00:00Z"}) + "\n"
        + json.dumps({"type": "user", "timestamp": "2024-01-01T00:01:00Z",
                      "message": {"content": "hello claude"}}) + "\n"
    )
    ts, text = get_first_user_message_claude(f)
    assert ts == "2024-01-01T00:00:00Z"
    assert text == "hello claude"


def test_get_first_user_message_claude_skips_meta(tmp_path):
    import json

    from trajectoriz import get_first_user_message_claude

    pid = "pid-1"
    f = tmp_path / "session.jsonl"
    f.write_text(
        json.dumps({"type": "user", "isMeta": True, "promptId": pid,
                    "timestamp": "2024-01-01T00:00:00Z",
                    "message": {"content": "system prompt"}}) + "\n"
        + json.dumps({"type": "user", "promptId": pid,
                      "message": {"content": "should be skipped"}}) + "\n"
        + json.dumps({"type": "user", "message": {"content": "real message"}}) + "\n"
    )
    ts, text = get_first_user_message_claude(f)
    assert text == "real message"


def test_get_first_user_message_copilot(tmp_path):
    import json

    from trajectoriz import get_first_user_message_copilot

    f = tmp_path / "events.jsonl"
    f.write_text(
        json.dumps({"type": "session.start", "timestamp": "2024-01-01T00:00:00Z"}) + "\n"
        + json.dumps({"type": "user.message", "data": {"content": "fix the bug"}}) + "\n"
    )
    ts, text = get_first_user_message_copilot(f)
    assert text == "fix the bug"


def test_get_first_user_message_agent_probe_user_type(tmp_path):
    import json

    from trajectoriz import get_first_user_message_agent_probe

    f = tmp_path / "session.jsonl"
    f.write_text(
        json.dumps({"type": "session_start", "timestamp": "2024-01-01T00:00:00Z"}) + "\n"
        + json.dumps({"type": "user", "message": {"content": "probe task"}}) + "\n"
    )
    ts, text = get_first_user_message_agent_probe(f)
    assert text == "probe task"


def test_get_first_user_message_dispatcher(tmp_path):
    from trajectoriz import get_first_user_message

    claude_dir = tmp_path / ".claude" / "projects" / "repo"
    claude_dir.mkdir(parents=True)
    f = claude_dir / "session.jsonl"
    f.write_text(
        json.dumps({"type": "user", "timestamp": "2024-01-01T00:00:00Z",
                    "message": {"content": "dispatched"}}) + "\n"
    )

    import unittest.mock as mock
    with mock.patch("trajectoriz.Path.home", return_value=tmp_path):
        ts, text = get_first_user_message(f)
    assert text == "dispatched"


def test_iter_records_all_exposes_public_record_api(tmp_path, monkeypatch):
    from trajectoriz import TrajectoryRecord, iter_records

    claude_dir = tmp_path / ".claude" / "projects" / "repo"
    claude_dir.mkdir(parents=True)
    f = claude_dir / "session.jsonl"
    f.write_text(
        json.dumps(
            {
                "type": "user",
                "timestamp": "2024-01-01T00:00:00Z",
                "message": {"content": "public api"},
            }
        )
        + "\n"
    )

    monkeypatch.setattr("trajectoriz.Path.home", lambda: tmp_path)

    records = list(iter_records())
    assert len(records) == 1
    record = records[0]
    assert isinstance(record, TrajectoryRecord)
    assert record.agent == "claude"
    assert record.first_msg == "public api"


def test_iter_records_local_filters_by_cwd(tmp_path, monkeypatch):
    from trajectoriz import claude_project_dir, iter_records

    repo_root = "/tmp/my-repo"
    project_dir = claude_project_dir(repo_root, claude_dir=tmp_path / ".claude")
    project_dir.mkdir(parents=True)
    f = project_dir / "session.jsonl"
    f.write_text(
        json.dumps(
            {
                "type": "user",
                "timestamp": "2024-01-01T00:00:00Z",
                "message": {"content": "repo local"},
            }
        )
        + "\n"
    )

    monkeypatch.setattr("trajectoriz.Path.home", lambda: tmp_path)

    records = list(iter_records(cwd=repo_root))
    assert [record.first_msg for record in records] == ["repo local"]


def test_parse_record_public_api(tmp_path):
    from trajectoriz import TrajectoryRecord, parse_record

    f = tmp_path / "session.jsonl"
    f.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "type": "user",
                        "sessionId": "s1",
                        "timestamp": "2024-01-01T00:00:00Z",
                        "message": {"content": "fix the bug"},
                    }
                ),
                json.dumps(
                    {
                        "type": "assistant",
                        "timestamp": "2024-01-01T00:01:00Z",
                        "message": {
                            "model": "claude-sonnet-4-6",
                            "content": [{"type": "text", "text": "Done."}],
                        },
                    }
                ),
            ]
        )
        + "\n"
    )

    record = TrajectoryRecord("cl-abc", "claude", "2024-01-01T00:00:00Z", "fix the bug", f)
    traj = parse_record(record, cache_dir=tmp_path / "cache")

    assert traj is not None
    assert traj.session_id == "s1"
    assert len(traj.steps) == 2


def test_get_cwd_from_trajectory(tmp_path):
    import json

    from trajectoriz import get_cwd_from_trajectory

    f = tmp_path / "session.jsonl"
    f.write_text(
        json.dumps({"type": "system"}) + "\n"
        + json.dumps({"type": "meta", "cwd": "/home/user/repo"}) + "\n"
    )
    assert get_cwd_from_trajectory(f) == "/home/user/repo"


def test_get_cwd_from_trajectory_missing(tmp_path):
    from trajectoriz import get_cwd_from_trajectory

    f = tmp_path / "session.jsonl"
    f.write_text("{}\n")
    assert get_cwd_from_trajectory(f) == ""


def test_iter_opencode_sessions_with_data(tmp_path):
    import json
    import sqlite3

    from trajectoriz import OpencodeSession, iter_opencode_sessions

    db = tmp_path / "opencode.db"
    conn = sqlite3.connect(str(db))
    conn.execute(
        "CREATE TABLE session (id TEXT, time_created INTEGER, time_updated INTEGER, "
        "model TEXT, directory TEXT, agent TEXT, cost REAL, tokens_input INTEGER, "
        "tokens_output INTEGER, tokens_reasoning INTEGER, tokens_cache_read INTEGER, "
        "tokens_cache_write INTEGER)"
    )
    conn.execute("CREATE TABLE message (id TEXT, session_id TEXT, time_created INTEGER, data TEXT)")
    conn.execute("CREATE TABLE part (id TEXT, message_id TEXT, time_created INTEGER, data TEXT)")
    conn.execute(
        "INSERT INTO session VALUES ('s1', 1000, 2000, '{\"provider\":\"anthropic\"}', "
        "'/repo', 'claude', 0.05, 100, 200, 10, 50, 30)"
    )
    conn.execute("INSERT INTO message VALUES ('m1', 's1', 1, ?)", (json.dumps({"role": "user"}),))
    conn.execute("INSERT INTO part VALUES ('p1', 'm1', 1, ?)", (json.dumps({"text": "fix the bug"}),))
    conn.commit()
    conn.close()

    results = list(iter_opencode_sessions(opencode_dir=str(tmp_path)))
    assert len(results) == 1
    s = results[0]
    assert isinstance(s, OpencodeSession)
    assert s.id == "s1"
    assert s.time_created == 1000
    assert s.time_updated == 2000
    assert s.agent == "claude"
    assert s.cost == 0.05
    assert s.tokens_input == 100
    assert s.tokens_output == 200
    assert s.tokens_reasoning == 10
    assert s.tokens_cache_read == 50
    assert s.tokens_cache_write == 30
    assert s.first_prompt == "fix the bug"


def test_estimate_tokens():
    from trajectoriz import estimate_tokens

    assert estimate_tokens(None) == 0
    assert estimate_tokens("") == 0
    assert estimate_tokens("a" * 4) == 1
    assert estimate_tokens("a" * 5) == 2
    # JSON-serialized: '{"key": "value"}' is 16 chars -> ceil(16/4) == 4
    assert estimate_tokens({"key": "value"}) == 4


def test_parse_claude_trajectory_total_tokens(tmp_path):
    import json

    from trajectoriz import parse_claude_trajectory

    f = tmp_path / "session.jsonl"
    f.write_text(
        json.dumps({"type": "user", "sessionId": "s1",
                     "message": {"content": "hello"}}) + "\n"
        + json.dumps({"type": "assistant", "message": {
            "model": "claude-sonnet-4-6",
            "content": [{"type": "text", "text": "hi there"}],
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }}) + "\n"
    )
    traj = parse_claude_trajectory(f)
    assert traj.total_prompt_tokens == 10
    assert traj.total_completion_tokens == 5
    assert traj.total_tokens == 15
    assert traj.event_types == ["user", "assistant"]
    assert traj.terminal_event == "assistant"


def test_parse_claude_trajectory_total_tokens_estimated_without_usage(tmp_path):
    import json

    from trajectoriz import estimate_tokens, parse_claude_trajectory

    f = tmp_path / "session.jsonl"
    f.write_text(
        json.dumps({"type": "user", "sessionId": "s1",
                     "message": {"content": "hello"}}) + "\n"
        + json.dumps({"type": "assistant", "message": {
            "model": "claude-sonnet-4-6",
            "content": [{"type": "text", "text": "hi there"}],
        }}) + "\n"
    )
    traj = parse_claude_trajectory(f)
    assert traj.total_prompt_tokens == 0
    assert traj.total_completion_tokens == 0
    assert traj.total_tokens == estimate_tokens("hello") + estimate_tokens("hi there")


def test_parse_claude_trajectory_cwd(tmp_path):
    import json

    from trajectoriz import parse_claude_trajectory

    f = tmp_path / "session.jsonl"
    f.write_text(
        json.dumps({"type": "system", "cwd": "/home/user/repo"}) + "\n"
        + json.dumps({"type": "user", "sessionId": "s1",
                      "message": {"content": "hello"}}) + "\n"
    )
    traj = parse_claude_trajectory(f)
    assert traj.cwd == "/home/user/repo"


def test_parse_claude_trajectory_cwd_missing(tmp_path):
    import json

    from trajectoriz import parse_claude_trajectory

    f = tmp_path / "session.jsonl"
    f.write_text(json.dumps({"type": "user", "message": {"content": "hi"}}) + "\n")
    traj = parse_claude_trajectory(f)
    assert traj.cwd == ""


def test_parse_codex_trajectory_cwd(tmp_path):
    import json

    from trajectoriz import parse_codex_trajectory

    f = tmp_path / "rollout-001.jsonl"
    f.write_text(
        json.dumps({"type": "session_meta", "payload": {"id": "s1", "cwd": "/work/project"}}) + "\n"
        + json.dumps({"type": "event_msg", "payload": {"type": "user_message",
                       "message": "hello"}}) + "\n"
    )
    traj = parse_codex_trajectory(f)
    assert traj.cwd == "/work/project"
    assert traj.event_types == ["session_meta", "user_message"]
    assert traj.terminal_event == "user_message"


def test_parse_copilot_event_trajectory_cwd(tmp_path):
    import json

    from trajectoriz import parse_copilot_event_trajectory

    f = tmp_path / "events.jsonl"
    f.write_text(
        json.dumps({"type": "session.start",
                    "data": {"sessionId": "s1", "cwd": "/home/user/project"}}) + "\n"
        + json.dumps({"type": "user.message", "data": {"content": "fix it"}}) + "\n"
    )
    traj = parse_copilot_event_trajectory(f)
    assert traj.cwd == "/home/user/project"
    assert traj.event_types == ["session.start", "user.message"]
    assert traj.terminal_event == "user.message"


def test_parse_agent_probe_trajectory_cwd(tmp_path):
    import json

    from trajectoriz import parse_agent_probe_trajectory

    f = tmp_path / "session.jsonl"
    f.write_text(
        json.dumps({"type": "session_start", "session_id": "s1",
                    "model": "gpt-4", "cwd": "/srv/myapp"}) + "\n"
        + json.dumps({"type": "user", "content": "do the thing"}) + "\n"
    )
    traj = parse_agent_probe_trajectory(f)
    assert traj.cwd == "/srv/myapp"


def test_parse_agent_probe_trajectory_outcome_events(tmp_path):
    from trajectoriz import parse_agent_probe_trajectory

    f = tmp_path / "session.jsonl"
    f.write_text(
        "\n".join(
            json.dumps(event)
            for event in [
                {"type": "session_start", "session_id": "s1"},
                {"type": "user", "content": "do the thing"},
                {"type": "error", "message": "retrying"},
                {"type": "compaction"},
                {"type": "fatal_error", "message": "failed"},
                {"type": "token_limit"},
                {"type": "usage", "prompt_tokens": 2, "completion_tokens": 3},
            ]
        )
        + "\n"
    )

    traj = parse_agent_probe_trajectory(f)

    assert traj.terminal_event == "token_limit"
    assert traj.event_types == [
        "session_start", "user", "error", "compaction", "fatal_error", "token_limit", "usage"
    ]
    assert traj.error_count == 1
    assert traj.fatal_error_count == 1
    assert traj.token_limit_count == 1
    assert traj.compaction_count == 1


def test_parse_copilot_event_trajectory_total_tokens(tmp_path):
    import json

    from trajectoriz import estimate_tokens, parse_copilot_event_trajectory

    f = tmp_path / "events.jsonl"
    f.write_text(
        json.dumps({"type": "session.start", "data": {"sessionId": "s1"}}) + "\n"
        + json.dumps({"type": "user.message", "data": {"content": "fix the bug"}}) + "\n"
        + json.dumps({"type": "assistant.message", "data": {"content": "fixed it"}}) + "\n"
        + json.dumps({"type": "session.shutdown", "data": {}}) + "\n"
    )
    traj = parse_copilot_event_trajectory(f)
    assert traj.total_prompt_tokens == 0
    assert traj.total_tokens == estimate_tokens("fix the bug") + estimate_tokens("fixed it")
