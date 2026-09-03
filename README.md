# trajectoriz

Library and CLI to search, browse, and analyze past agent trajectory files. Supports Claude Code, Codex, OpenCode, Copilot, Hermes and more.

## Installation

```bash
pip install trajectoriz
```

## Search

**trajectoriz-cli search** lets you find past agent sessions by content — what was discussed, what commands were run, what files were edited.

```bash
# AND semantics: all words must appear in a step
trajectoriz-cli search "salary KTH overhead"
trajectoriz-cli search "telegram send_message bot_token"
trajectoriz-cli search "deploy dokku researchers webapp"

# OR semantics: use \| between alternatives
trajectoriz-cli search "pytest\|unittest"

# Restrict to the current project
trajectoriz-cli search "fix login bug" --local

# Fast metadata-only search (first message only, no parsing)
trajectoriz-cli search "refactor auth" --fast
```

Results are returned as a paginated Markdown table with trajectory ID, agent, date, step number, and a context snippet:

```
## Search: `salary KTH overhead` — 8 match(es)

| ID           | Agent  | Date       | Step | Snippet                                              |
|---|---|---|---|---|
| `cl-4d72f7b5`| claude | 2026-05-15 | 73   | …kth_salary = ws["E8"].value  # KTH  Direkt lön…   |
| `cl-4ef313e6`| claude | 2026-05-04 | 93   | …Organization: KTH Royal Institute of Technology…  |
```

Then inspect any result:

```bash
trajectoriz-cli show cl-4d72f7b5 --step 73
```

### Search backends

| Backend | Setup | Semantics |
|---|---|---|
| `grep` (default) | none | substring, in-process |
| `sqlite` | `trajectoriz-cli refresh --no-recoll` | whole-word FTS5 |
| `recoll` | `trajectoriz-cli refresh --no-sqlite` | full Xapian index |

```bash
trajectoriz-cli search "openssl handshake" --backend sqlite
trajectoriz-cli search "openssl handshake" --backend recoll
```

## CLI

```bash
# List trajectories in the current directory
trajectoriz-cli list

# Show a trajectory (markdown, paginated)
trajectoriz-cli show cl-1234abcd
trajectoriz-cli show cl-1234abcd --last         # jump to the last page
trajectoriz-cli show cl-1234abcd --step 42      # jump to the page containing step 42
trajectoriz-cli show cl-1234abcd --html > out.html   # self-contained HTML export

# Trajectory metadata (JSON)
trajectoriz-cli info cl-1234abcd

# Blame a file — every agent edit in chronological order with line deltas
trajectoriz-cli blame src/main.py

# Aggregate statistics across all trajectories
trajectoriz-cli stats --all

# Aggregate shell-invoked programs across a repo
trajectoriz-cli advanced tools --dir /path/to/repo
```

## Memory filesystem

**trajectoriz-cli memory** mounts a read-only FUSE filesystem where every local trajectory
shows up as an [ATIF v1.7](https://harborframework.com/docs/agents/trajectory-format) JSON
file — so any tool that reads files (grep, an agent's own file tools, etc.) can browse past
sessions directly, without going through this CLI.

```bash
pip install trajectoriz[fuse]   # requires libfuse (Linux) or macFUSE (macOS)

trajectoriz-cli memory                          # mounts ./memory (created if missing), daemonizes
trajectoriz-cli memory ~/mnt/traj-memory        # or mount elsewhere
trajectoriz-cli memory --foreground             # or run attached

ls memory
cat memory/2026-05-15_claude_cl-4d72f7b5.atif.json

trajectoriz-cli memory --unmount                # unmount (or: fusermount -u memory)
```

Use `--dir PATH` to expose a different repo's trajectories instead of the current directory's.
If the mountpoint sits inside a git repo, it's added to that repo's `.gitignore` automatically.

The mount is built for tools that walk and read files: one store scan serves the whole
directory listing (refreshed every couple of seconds, so new sessions still show up), and each
trajectory's ATIF payload is rendered once per open, then served from memory. Reading every
file in a mount of 30 trajectories takes ~0.1s.

If a mount ever stops responding — a killed daemon leaves the mountpoint attached but
unserviced, which hangs anything that walks the tree it sits in — recover it with:

```bash
trajectoriz-cli memory --unmount ./memory       # lazily unmounts if it is wedged
```

## Features

- **Full-content search** — three backends (grep / sqlite / recoll); space-separated words are AND, `\|` is OR; matches at step level across message, tool calls, and results
- **Unified record API** — iterate and parse sessions from Claude Code, Codex, Copilot, OpenCode, Hermes and more through a single `iter_records()` / `parse_record()` interface
- **Blame** — trace every agent edit to a file across all trajectory sources, with line/char deltas
- **HTML export** — `trajectoriz-cli show <id> --html` renders a trajectory as a self-contained HTML page
- **ATIF export** — `trajectoriz.atif` translates parsed trajectories to [ATIF v1.7](https://harborframework.com/docs/agents/trajectory-format) (Claude Code, Codex, Copilot, agentknit, or any `iter_records()`/`parse_record()` result)
- **Memory filesystem** — `trajectoriz-cli memory <mountpoint>` mounts a read-only FUSE view where every local trajectory is one ATIF JSON file

## Python API

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
