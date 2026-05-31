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

    from trajectoriz import iter_codex_db_sessions

    db = tmp_path / "state_5.sqlite"
    conn = sqlite3.connect(str(db))
    conn.execute(
        "CREATE TABLE threads"
        " (id TEXT, updated_at_ms INTEGER, first_user_message TEXT, model_provider TEXT, model TEXT, cwd TEXT)"
    )
    conn.execute("INSERT INTO threads VALUES ('id1', 1000, 'hello', 'openai', 'gpt-4', '/repo')")
    conn.commit()
    conn.close()

    results = list(iter_codex_db_sessions(codex_dir=str(tmp_path)))
    assert len(results) == 1
    assert results[0][0] == "id1"
    assert results[0][2] == "hello"


def test_iter_opencode_sessions_empty(tmp_path):
    from trajectoriz import iter_opencode_sessions

    assert list(iter_opencode_sessions(opencode_dir=str(tmp_path))) == []


def test_iter_opencode_sessions_with_data(tmp_path):
    import json
    import sqlite3

    from trajectoriz import iter_opencode_sessions

    db = tmp_path / "opencode.db"
    conn = sqlite3.connect(str(db))
    conn.execute("CREATE TABLE session (id TEXT, time_updated INTEGER, model TEXT, directory TEXT)")
    conn.execute("CREATE TABLE message (id TEXT, session_id TEXT, time_created INTEGER, data TEXT)")
    conn.execute("CREATE TABLE part (id TEXT, message_id TEXT, time_created INTEGER, data TEXT)")
    conn.execute("INSERT INTO session VALUES ('s1', 2000, '{}', '/repo')")
    conn.execute("INSERT INTO message VALUES ('m1', 's1', 1, ?)", (json.dumps({"role": "user"}),))
    conn.execute("INSERT INTO part VALUES ('p1', 'm1', 1, ?)", (json.dumps({"text": "fix the bug"}),))
    conn.commit()
    conn.close()

    results = list(iter_opencode_sessions(opencode_dir=str(tmp_path)))
    assert len(results) == 1
    assert results[0][0] == "s1"
    assert results[0][4] == "fix the bug"
