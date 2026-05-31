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
