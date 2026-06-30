# trajectoriz

Library to work with agent trajectory files. Support Claude, Codex, OpenCode and others

## Installation

```bash
pip install trajectoriz
```

## Features

- **Unified record API** — iterate and parse sessions from Claude Code, Codex, Copilot, OpenCode, Hermes and more through a single `iter_records()` / `parse_record()` interface
- **Full-content search** — three backends: in-process `grep` (default, no setup), `recoll` (Xapian index), `sqlite` (FTS5 index)
- **Blame** — trace every agent edit to a file across all trajectory sources, with line/char deltas
- **HTML export** — `trajectoriz-cli show <id> --html` renders a trajectory as a self-contained HTML page

## CLI

```bash
# List trajectories in the current directory
trajectoriz-cli list

# Search all trajectories for a keyword (in-process grep, default)
# Show a trajectory (markdown, paginated)
trajectoriz-cli show cl-1234abcd
trajectoriz-cli show cl-1234abcd --last
trajectoriz-cli show cl-1234abcd --html > out.html

# Blame a file — every agent edit in chronological order with line deltas
trajectoriz-cli blame src/main.py

# Aggregate shell-invoked programs across a repo
trajectoriz-cli advanced tools --dir /path/to/repo --json

trajectoriz-cli search raven
# Search backends
trajectoriz-cli search foo --backend grep     # in-process substring scan (default)
trajectoriz-cli search foo --backend recoll   # Xapian index via recoll CLI
trajectoriz-cli search foo --backend sqlite   # local SQLite FTS5 index
# Metadata-only search (no trajectory parsing, much faster)
trajectoriz-cli search foo --fast

# Build / update the recoll and SQLite indexes
trajectoriz-cli refresh           # both
trajectoriz-cli refresh --no-sqlite   # recoll only
trajectoriz-cli refresh --no-recoll   # SQLite only

```

## API

```python
from trajectoriz import iter_records, parse_record

# Iterate sessions across all supported agents (Claude, Codex, Copilot, OpenCode, …)
for record in iter_records():
    print(record.agent, record.timestamp[:10], record.first_msg[:60])

# Iterate sessions for the current project only
for record in iter_records(cwd="/path/to/repo"):
    trajectory = parse_record(record)
    if trajectory is not None:
        print(f"{record.agent}: {len(trajectory.steps)} steps, {trajectory.total_tokens} tokens")
```

## License

MIT
