"""Tests for trajectoriz."""

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
    import json

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
