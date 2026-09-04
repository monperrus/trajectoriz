"""Tests for keyring-secret leak detection (_secrets.py)."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

import trajectoriz as tz
from trajectoriz import _secrets


TOKEN = "ghp_9Zq3kLmN7pR2sT4vW6xY8aB0cD1eF3gH5i"
PASSPHRASE = "correct horse battery staple 42"


def _secret(value: str, label: str = "github token", schema: str = "Secret.Generic"):
    return _secrets.KeyringSecret(label=label, schema=schema, collection="Login", value=value)


def _write_claude_traj(path: Path, first_msg: str, tool_result: str, command: str = "ls") -> None:
    lines = [
        {"type": "user", "sessionId": "s1", "timestamp": "2026-03-01T10:00:00Z",
         "message": {"content": first_msg}},
        {"type": "assistant", "timestamp": "2026-03-01T10:01:00Z", "message": {
            "model": "claude-opus-5",
            "content": [{"type": "tool_use", "id": "c1", "name": "Bash",
                         "input": {"command": command}}],
        }},
        {"type": "user", "timestamp": "2026-03-01T10:02:00Z", "message": {
            "content": [{"type": "tool_result", "tool_use_id": "c1", "content": tool_result}],
        }},
    ]
    path.write_text("\n".join(json.dumps(line) for line in lines) + "\n")


def _record(path: Path) -> tz.TrajectoryRecord:
    return tz.TrajectoryRecord(
        id="cl-test0001", agent="claude", timestamp="2026-03-01T10:00:00Z",
        first_msg="hello", source=path,
    )


@pytest.fixture()
def traj(tmp_path: Path) -> Path:
    return tmp_path / "session.jsonl"


@pytest.fixture(params=[True, False], ids=["grep", "python"])
def use_grep(request) -> bool:
    return request.param


# ── needle / fingerprint ─────────────────────────────────────────────────────


def test_needle_is_the_value_for_single_line_secrets():
    assert _secret(TOKEN).needle == TOKEN


def test_needle_is_longest_line_for_multiline_secrets():
    pem = "-----BEGIN KEY-----\n" + "A" * 64 + "\nshort\n-----END KEY-----"
    assert _secret(pem).needle == "A" * 64


def test_fingerprint_is_stable_and_not_the_value():
    fingerprint = _secret(TOKEN).fingerprint
    assert len(fingerprint) == 12
    assert fingerprint == _secret(TOKEN, label="other").fingerprint
    assert TOKEN not in fingerprint


# ── scanning ─────────────────────────────────────────────────────────────────


def test_finds_secret_in_tool_result_and_attributes_the_step(traj, use_grep):
    _write_claude_traj(traj, "check my token", f"GITHUB_TOKEN={TOKEN}")
    result = _secrets.scan([_secret(TOKEN)], [_record(traj)], use_grep=use_grep, store_dbs=[])

    assert len(result.leaks) == 1
    leak = result.leaks[0]
    assert leak.labels == ("github token",)
    assert leak.trajectory_id == "cl-test0001"
    assert leak.step is not None
    assert leak.occurrences == 1
    assert leak.length == len(TOKEN)
    assert not leak.partial


def test_finds_secret_in_tool_call_arguments(traj, use_grep):
    _write_claude_traj(traj, "deploy", "done", command=f"curl -H 'auth: {TOKEN}' example.com")
    result = _secrets.scan([_secret(TOKEN)], [_record(traj)], use_grep=use_grep, store_dbs=[])
    assert [leak.step for leak in result.leaks] != []


def test_finds_secret_in_the_first_user_message(traj, use_grep):
    _write_claude_traj(traj, f"my password is {PASSPHRASE}", "ok")
    result = _secrets.scan([_secret(PASSPHRASE)], [_record(traj)], use_grep=use_grep, store_dbs=[])
    assert len(result.leaks) == 1


def test_clean_trajectory_yields_nothing(traj, use_grep):
    _write_claude_traj(traj, "nothing to see", "all good")
    result = _secrets.scan([_secret(TOKEN)], [_record(traj)], use_grep=use_grep, store_dbs=[])
    assert result.leaks == []


def test_grep_and_python_paths_agree(traj):
    _write_claude_traj(traj, "check", f"token {TOKEN} and {PASSPHRASE}")
    secrets = [_secret(TOKEN), _secret(PASSPHRASE, label="wifi")]
    with_grep = _secrets.scan(secrets, [_record(traj)], use_grep=True, store_dbs=[])
    without = _secrets.scan(secrets, [_record(traj)], use_grep=False, store_dbs=[])
    assert {leak.fingerprint for leak in with_grep.leaks} == {leak.fingerprint for leak in without.leaks}
    assert len(with_grep.leaks) == 2


def test_short_secrets_are_skipped(traj, use_grep):
    _write_claude_traj(traj, "cat", "the cat sat")
    result = _secrets.scan([_secret("cat")], [_record(traj)], use_grep=use_grep, store_dbs=[])
    assert result.leaks == []
    assert result.skipped_short == 1
    assert result.secrets_scanned == 0


def test_min_length_is_configurable(traj, use_grep):
    _write_claude_traj(traj, "hi", "the cat sat")
    result = _secrets.scan(
        [_secret("cat")], [_record(traj)], min_length=3, use_grep=use_grep, store_dbs=[]
    )
    assert len(result.leaks) == 1


def test_empty_secret_never_matches(traj, use_grep):
    _write_claude_traj(traj, "hi", "ok")
    result = _secrets.scan(
        [_secret("")], [_record(traj)], min_length=0, use_grep=use_grep, store_dbs=[]
    )
    assert result.leaks == []


def test_one_value_under_several_labels_is_reported_once(traj, use_grep):
    _write_claude_traj(traj, "check", f"TOKEN={TOKEN}")
    secrets = [_secret(TOKEN, label="github"), _secret(TOKEN, label="gh-backup")]
    result = _secrets.scan(secrets, [_record(traj)], use_grep=use_grep, store_dbs=[])
    assert len(result.leaks) == 1
    assert result.leaks[0].labels == ("gh-backup", "github")


def test_multiline_secret_reported_as_partial(traj, use_grep):
    body = "B" * 70
    pem = f"-----BEGIN RSA PRIVATE KEY-----\n{body}\n-----END RSA PRIVATE KEY-----"
    _write_claude_traj(traj, "look at this key", f"key body: {body}")
    result = _secrets.scan([_secret(pem, label="ssh key")], [_record(traj)],
                           use_grep=use_grep, store_dbs=[])
    assert len(result.leaks) == 1
    assert result.leaks[0].partial


def test_hit_outside_any_parsed_step_is_still_reported(tmp_path, use_grep):
    """A secret in a file the parser cannot map to a step must not be lost."""
    path = tmp_path / "opaque.jsonl"
    path.write_text(json.dumps({"type": "meta", "env": {"TOKEN": TOKEN}}) + "\n")
    result = _secrets.scan([_secret(TOKEN)], [_record(path)], use_grep=use_grep, store_dbs=[])
    assert len(result.leaks) == 1
    assert result.leaks[0].step is None
    assert result.unattributed == 1


def test_matches_across_a_chunk_boundary(tmp_path, monkeypatch):
    path = tmp_path / "big.jsonl"
    monkeypatch.setattr(_secrets, "_READ_CHUNK", 1024)
    filler = "x" * 1000
    payload = json.dumps({"type": "meta", "note": filler + TOKEN + filler})
    path.write_text(payload + "\n" + payload + "\n")
    result = _secrets.scan(
        [_secret(TOKEN)], [_record(path)], use_grep=False, store_dbs=[]
    )
    assert len(result.leaks) == 1


def test_occurrences_are_counted_once_across_chunk_boundaries(tmp_path, monkeypatch):
    monkeypatch.setattr(_secrets, "_READ_CHUNK", 512)
    path = tmp_path / "chunky.jsonl"
    path.write_text(json.dumps({"note": ("y" * 400 + TOKEN) * 4}) + "\n")
    result = _secrets.scan([_secret(TOKEN)], [_record(path)], use_grep=False, store_dbs=[])
    assert [leak.occurrences for leak in result.leaks] == [4]


def test_common_values_are_quarantined_not_reported_as_leaks(tmp_path, use_grep):
    """A keyring value that is ordinary text must not flood the report."""
    records = []
    for i in range(6):
        path = tmp_path / f"s{i}.jsonl"
        _write_claude_traj(path, "hello", "the password123 is everywhere")
        records.append(
            tz.TrajectoryRecord(
                id=f"cl-{i:08d}", agent="claude", timestamp="2026-03-01T10:00:00Z",
                first_msg="hello", source=path,
            )
        )
    secrets = [_secret("password123", label="weak"), _secret(TOKEN)]
    result = _secrets.scan(secrets, records, max_files=3, use_grep=use_grep, store_dbs=[])

    assert result.leaks == []
    assert len(result.too_common) == 1
    common = result.too_common[0]
    assert common.labels == ("weak",)
    assert common.files == 6
    assert common.occurrences == 6


def test_max_files_zero_disables_the_cap(tmp_path, use_grep):
    records = []
    for i in range(4):
        path = tmp_path / f"s{i}.jsonl"
        _write_claude_traj(path, "hello", f"token {TOKEN}")
        records.append(
            tz.TrajectoryRecord(
                id=f"cl-{i:08d}", agent="claude", timestamp="2026-03-01T10:00:00Z",
                first_msg="hello", source=path,
            )
        )
    result = _secrets.scan(
        [_secret(TOKEN)], records, max_files=0, use_grep=use_grep, store_dbs=[]
    )
    assert len(result.leaks) == 4
    assert result.too_common == []


# ── where the secret was sent ────────────────────────────────────────────────


def _agentknit_session(dir_path: Path, secret: str, endpoint: str | None) -> Path:
    """Write an agentknit journal plus, optionally, the payload it posted."""
    dir_path.mkdir(parents=True, exist_ok=True)
    journal = dir_path / "abcd_journal.jsonl"
    call = "call_1"
    events = [
        {"type": "turn_start", "ts": "2026-09-04T13:18:19", "task": "read the keyring"},
        {"type": "message", "ts": "2026-09-04T13:18:19",
         "msg": {"role": "user", "content": "read the keyring"}},
        {"type": "message", "ts": "2026-09-04T13:18:25", "msg": {
            "role": "assistant", "content": "", "tool_calls": [
                {"id": call, "type": "function",
                 "function": {"name": "exec_shell",
                              "arguments": json.dumps({"command": "secret-tool lookup x y"})}},
            ],
        }},
        {"type": "message", "ts": "2026-09-04T13:18:26",
         "msg": {"role": "tool", "tool_call_id": call, "content": f"secret = {secret}"}},
    ]
    journal.write_text("\n".join(json.dumps(e) for e in events) + "\n")
    if endpoint:
        (dir_path / "abcd_messages.json").write_text(json.dumps({
            "metadata": {"endpoint": endpoint, "model": dir_path.name, "session_id": "abcd"},
            "messages": [
                {"role": "user", "content": "read the keyring"},
                {"role": "tool", "content": f"secret = {secret}"},
            ],
        }))
    return journal


def test_finds_secret_in_an_agentknit_session_and_names_the_endpoint(tmp_path, use_grep):
    journal = _agentknit_session(
        tmp_path / "glm-5.3", TOKEN, "https://api.z.ai/api/coding/paas/v4"
    )
    record = tz.TrajectoryRecord(
        id="ap-abcd1234", agent="agent_probe", timestamp="2026-09-04T13:18:19",
        first_msg="read the keyring", source=journal,
    )
    result = _secrets.scan([_secret(TOKEN)], [record], use_grep=use_grep, store_dbs=[])

    assert result.leaks
    assert all(leak.destination == "glm-5.3 (api.z.ai)" for leak in result.leaks)
    # The journal and the posted payload are both scanned.
    assert {Path(leak.source).name for leak in result.leaks} == {
        "abcd_journal.jsonl", "abcd_messages.json",
    }
    assert any(leak.step is not None for leak in result.leaks)


def test_the_posted_payload_is_scanned_even_when_the_journal_is_clean(tmp_path, use_grep):
    """The payload file is the request as sent, so it is evidence on its own."""
    directory = tmp_path / "glm-5.3"
    directory.mkdir(parents=True)
    journal = directory / "abcd_journal.jsonl"
    journal.write_text(json.dumps({"type": "turn_start", "task": "hello"}) + "\n")
    (directory / "abcd_messages.json").write_text(json.dumps({
        "metadata": {"endpoint": "https://api.z.ai/api/coding/paas/v4", "model": "glm-5.3"},
        "messages": [{"role": "tool", "content": f"secret = {TOKEN}"}],
    }))
    record = tz.TrajectoryRecord(
        id="ap-abcd1234", agent="agent_probe", timestamp="2026-09-04T13:18:19",
        first_msg="hello", source=journal,
    )
    result = _secrets.scan([_secret(TOKEN)], [record], use_grep=use_grep, store_dbs=[])

    assert len(result.leaks) == 1
    assert Path(result.leaks[0].source).name == "abcd_messages.json"
    assert result.leaks[0].destination == "glm-5.3 (api.z.ai)"


def test_destination_falls_back_to_the_model_alone(tmp_path, use_grep):
    journal = _agentknit_session(tmp_path / "glm-5.2", TOKEN, endpoint=None)
    record = tz.TrajectoryRecord(
        id="ap-abcd1234", agent="agent_probe", timestamp="2026-09-04T13:18:19",
        first_msg="read the keyring", source=journal,
    )
    result = _secrets.scan([_secret(TOKEN)], [record], use_grep=use_grep, store_dbs=[])
    assert result.leaks
    assert all(leak.destination == "glm-5.2" for leak in result.leaks)


def test_destinations_are_summarised_in_json(tmp_path, use_grep):
    journal = _agentknit_session(
        tmp_path / "glm-5.3", TOKEN, "https://api.z.ai/api/coding/paas/v4"
    )
    record = tz.TrajectoryRecord(
        id="ap-abcd1234", agent="agent_probe", timestamp="2026-09-04T13:18:19",
        first_msg="read the keyring", source=journal,
    )
    result = _secrets.scan([_secret(TOKEN)], [record], use_grep=use_grep, store_dbs=[])
    dumped = _secrets.leaks_to_json(result)
    assert dumped["summary"]["destinations"] == ["glm-5.3 (api.z.ai)"]
    assert TOKEN not in json.dumps(dumped)


# ── redaction ────────────────────────────────────────────────────────────────


def test_redact_removes_the_secret_and_keeps_context():
    snippet = _secrets.redact(f"export GITHUB_TOKEN={TOKEN} && deploy", TOKEN)
    assert TOKEN not in snippet
    assert "GITHUB_TOKEN=" in snippet
    assert f"REDACTED:{len(TOKEN)} chars" in snippet


def test_no_secret_value_reaches_the_report(traj, use_grep):
    _write_claude_traj(traj, f"token {TOKEN}", f"TOKEN={TOKEN} {PASSPHRASE}")
    secrets = [_secret(TOKEN), _secret(PASSPHRASE, label="wifi")]
    result = _secrets.scan(secrets, [_record(traj)], use_grep=use_grep, store_dbs=[])
    dumped = json.dumps(_secrets.leaks_to_json(result))
    assert result.leaks
    assert TOKEN not in dumped
    assert PASSPHRASE not in dumped


def test_json_summary_counts(traj, use_grep):
    _write_claude_traj(traj, "check", f"TOKEN={TOKEN}")
    result = _secrets.scan(
        [_secret(TOKEN), _secret("shortie", label="tiny")],
        [_record(traj)], use_grep=use_grep, store_dbs=[],
    )
    summary = _secrets.leaks_to_json(result)["summary"]
    assert summary["secrets_total"] == 2
    assert summary["secrets_scanned"] == 1
    assert summary["skipped_short"] == 1
    assert summary["leaked_secrets"] == 1
    assert summary["affected_trajectories"] == 1
    assert summary["files_scanned"] == 1


# ── SQLite session stores ────────────────────────────────────────────────────


def _make_store(path: Path, text: str) -> None:
    conn = sqlite3.connect(str(path))
    conn.execute("CREATE TABLE message (id TEXT, session_id TEXT, content TEXT)")
    conn.execute("INSERT INTO message VALUES ('m1', 'sess-1', ?)", (text,))
    conn.commit()
    conn.close()


def test_finds_secret_inside_a_session_store_db(tmp_path, use_grep):
    store = tmp_path / "opencode.db"
    _make_store(store, f"the token is {TOKEN}")
    record = tz.TrajectoryRecord(
        id="oc-abcd1234", agent="opencode", timestamp="1780000000",
        first_msg="hi", source={"type": "opencode", "session_id": "sess-1"},
    )
    result = _secrets.scan([_secret(TOKEN)], [record], use_grep=use_grep, store_dbs=[store])
    assert len(result.leaks) == 1
    leak = result.leaks[0]
    assert leak.trajectory_id == "oc-abcd1234"
    assert leak.agent == "opencode"
    assert TOKEN not in leak.context


def test_clean_session_store_yields_nothing(tmp_path, use_grep):
    store = tmp_path / "opencode.db"
    _make_store(store, "nothing sensitive here")
    result = _secrets.scan([_secret(TOKEN)], [], use_grep=use_grep, store_dbs=[store])
    assert result.leaks == []


def test_locked_collections_and_binary_items_are_surfaced(traj, use_grep):
    _write_claude_traj(traj, "hi", "ok")
    result = _secrets.scan(
        [_secret(TOKEN)], [_record(traj)], use_grep=use_grep, store_dbs=[],
        locked_collections=["Vault"], skipped_binary=3,
    )
    summary = _secrets.leaks_to_json(result)["summary"]
    assert summary["locked_collections"] == ["Vault"]
    assert summary["skipped_binary"] == 3
