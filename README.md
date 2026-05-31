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
    iter_codex_trajectories,
    iter_codex_rollout_files,
    iter_pi_trajectories,
    iter_cursor_trajectories,
    iter_copilot_sessions,
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

# List Codex CLI rollout files
for path in iter_codex_rollout_files():
    print(path)

# List pi coding agent session files
for path in iter_pi_trajectories():
    print(path)

# List Cursor trajectory files
for path in iter_cursor_trajectories():
    print(path)

# List Copilot CLI sessions
for session_id, created_at in iter_copilot_sessions():
    print(session_id, created_at)
```

## License

MIT
