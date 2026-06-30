# trajectoriz

Locate agent trajectory files on the local machine.

## Installation

```bash
pip install trajectoriz
```

## Usage

```python
from trajectoriz import (
    iter_claude_trajectories,
    iter_claude_project_trajectories,
    iter_records,
    iter_codex_trajectories,
    iter_codex_rollout_files,
    iter_codex_db_sessions,
    parse_record,
    iter_pi_trajectories,
    iter_cursor_trajectories,
    iter_copilot_event_trajectories,
    iter_copilot_sessions,
    iter_agent_probe_trajectories,
    iter_opencode_sessions,
)

# List all Claude Code trajectory files
for path in iter_claude_trajectories():
    print(path)

# List Claude trajectories for a specific project
for path in iter_claude_project_trajectories("/path/to/repo"):
    print(path)

# List Codex CLI session files
for path in iter_codex_trajectories():
    print(path)

# Iterate records across all supported sources and parse them uniformly
for record in iter_records():
    trajectory = parse_record(record)
    if trajectory is not None:
        print(record.agent, len(trajectory.steps))

# List Codex CLI rollout files
for path in iter_codex_rollout_files():
    print(path)

# List Codex sessions from SQLite store (~/.codex/state_5.sqlite)
for session_id, updated_at_ms, first_msg, provider, model, cwd in iter_codex_db_sessions():
    print(session_id, first_msg)

# List pi coding agent session files
for path in iter_pi_trajectories():
    print(path)

# List Cursor trajectory files
for path in iter_cursor_trajectories():
    print(path)

# List Copilot CLI session event JSONL files (~/.copilot/session-state/*/events.jsonl)
for path in iter_copilot_event_trajectories():
    print(path)

# List Copilot CLI sessions from SQLite store
for session_id, created_at in iter_copilot_sessions():
    print(session_id, created_at)

# List agent_probe session JSONL files (~/.local/share/agent_probe/*/*/*)
for path in iter_agent_probe_trajectories():
    print(path)

# List opencode sessions from SQLite store (~/.local/share/opencode/opencode.db)
for session_id, updated_at_ms, model_json, directory, first_prompt in iter_opencode_sessions():
    print(session_id, first_prompt)
```

## CLI

```bash
# List trajectories in the current directory
trajectoriz-cli list

# Search all trajectories for a keyword
trajectoriz-cli search raven

# OR search (grep syntax) — matches any of the terms
trajectoriz-cli search "theraven\|raven\|password"

# Full-content search (default) or fast metadata-only search
trajectoriz-cli search foo --fast

# Search backends: grep (default), recoll, sqlite
trajectoriz-cli search foo --backend grep     # in-process substring scan (default)
trajectoriz-cli search foo --backend recoll   # Xapian index via recoll CLI
trajectoriz-cli search foo --backend sqlite   # local SQLite FTS5 index

# Build / update the recoll and SQLite indexes
trajectoriz-cli refresh                       # both
trajectoriz-cli refresh --no-sqlite           # recoll only
trajectoriz-cli refresh --no-recoll           # SQLite only

# Show a trajectory
trajectoriz-cli show cl-1234abcd

# Show the last page of a long trajectory
trajectoriz-cli show cl-1234abcd --last

# Blame a file — show every agent edit in chronological order with line deltas
trajectoriz-cli blame src/main.py

# Aggregate shell-invoked programs for a repo as JSON
trajectoriz-cli advanced tools --dir /path/to/repo --json

# Sample output:
# | Timestamp           | Agent       | Traj ID      | Op    | Delta       | First message         |
# | 2026-05-31T11:31:07 | agent_probe | ap-f5515937  | write | +55 lines   | run checklist ...     |
# | 2026-06-01T14:22:00 | claude      | cl-e20eee97  | edit  | +13/-9 lines| add tests and doc ... |

```

## License

MIT
