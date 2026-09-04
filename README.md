# trajectoriz

Library and CLI to search, browse, and analyze past agent trajectory files. Supports Claude Code, Codex, OpenCode, Copilot, Hermes and more.

## Installation

```bash
pip install trajectoriz
```

## Search

**trajectoriz-cli search** lets you find past agent sessions — by what started them, or with
`--content`, by everything that was discussed, run and edited inside them.

By default it searches first messages, IDs and agent names, which needs no trajectory parsing
and answers the common question ("which session was that?") immediately. `--content` searches
every step instead: slower, exhaustive.

```bash
# Locate a session (default: first messages and metadata)
trajectoriz-cli search "refactor auth"

# Search inside every step: tool calls, their results, all messages
trajectoriz-cli search "salary KTH overhead" --content

# AND semantics: all words must appear
trajectoriz-cli search "telegram send_message bot_token" --content

# OR semantics: use \| between alternatives
trajectoriz-cli search "pytest\|unittest" --content

# Restrict to the current project
trajectoriz-cli search "fix login bug" --local
```

With `--content`, results are a paginated Markdown table with trajectory ID, agent, date, step number, and a context snippet:

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

Backends apply to `--content` searches only.

| Backend | Setup | Semantics |
|---|---|---|
| `sqlite` (default) | builds itself; `trajectoriz-cli refresh --no-recoll` to rebuild | whole-word FTS5 |
| `grep` | none | substring, in-process, always fresh |
| `recoll` | `trajectoriz-cli refresh --no-sqlite` | full Xapian index |

```bash
trajectoriz-cli search "openssl handshake" --content --backend sqlite
trajectoriz-cli search "openssl handshake" --content --backend recoll
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

# Check whether any keyring secret leaked into a trajectory
trajectoriz-cli secrets
```

## Leaked secrets

**trajectoriz-cli secrets** answers one question: has any secret in my OS keyring ever been
written in cleartext into an agent trajectory? Agents print environment variables, `cat`
config files and echo tokens into shell commands; all of it is kept forever in the trajectory
store. The keyring is the ground truth of what a secret is, so there are no entropy or
pattern heuristics here — every keyring value is searched for verbatim.

```bash
trajectoriz-cli secrets                     # scan every local trajectory
trajectoriz-cli secrets --dir /path/to/repo # only this repo's trajectories
trajectoriz-cli secrets --group-by trajectory
trajectoriz-cli secrets --json
```

```
## Leaked secrets: 2 of 297 keyring secrets appear in cleartext in 4 trajectory(ies)

### Sent to

| Model endpoint          | Distinct secrets |
|---|---|
| claude-sonnet-5         | 2 |
| glm-5.3 (api.z.ai)      | 1 |

### `github token` (fp `a1b2c3d4e5f6`, 40 chars, org.freedesktop.Secret.Generic)

| Agent  | Trajectory    | Date       | Step | Sent to         | Context                               |
|---|---|---|---|---|---|
| claude | `cl-4d72f7b5` | 2026-05-15 | 73   | claude-sonnet-5 | …export GH_TOKEN=«REDACTED:40 chars»… |
```

A secret in a trajectory step is a secret that was in the context window, so the report names
the destination: the model, and the endpoint host when the log records it. That is the question
worth answering, since a local file that never left the machine is a smaller problem than a
credential handed to a third party.

Secret values never appear in the output: a finding is identified by its keyring label and a
SHA-256 fingerprint, and the match context is redacted — a report that quoted the secret would
just be the next leak. Exit status is 1 when something leaked, 0 when nothing did, so it works
as a cron or pre-commit check.

Coverage is the same store as `search`: every trajectory file plus the SQLite session stores
(OpenCode, Copilot, Hermes, Codex), scanned as text so nothing hides in a DB page. For
agentknit sessions the `*_messages.json` payload written beside the journal is scanned too:
that file is the request as it was posted to the endpoint, so it is the direct evidence of
what left the machine. Multi-line secrets (PEM keys) are matched on their longest line and
flagged as partial.

Two things keep the report honest rather than noisy. Keyring values shorter than
`--min-length` (default 8) are skipped, and a value that turns up in more than `--max-files`
trajectories (default 25) is listed separately as ordinary text — a dictionary-word password
matches half the corpus and means nothing. Both are reported as counts, so nothing is dropped
silently; `--max-files 0` removes the cap. Locked keyring collections are reported as
unscanned rather than ignored.

## Memory filesystem

**trajectoriz-cli memory** mounts a read-only FUSE filesystem where every local trajectory
shows up as an [ATIF v1.7](https://harborframework.com/docs/agents/trajectory-format) JSON
file — so any tool that reads files (grep, an agent's own file tools, etc.) can browse past
sessions directly, without going through this CLI.

The idea behind it:
[Exposing Past Sessions as Memory for Coding Agents](https://www.monperrus.net/martin/memory-for-coding-agents).

```bash
pip install trajectoriz[fuse]   # requires libfuse (Linux) or macFUSE (macOS)

trajectoriz-cli memory                          # mounts ./memory (created if missing), daemonizes
trajectoriz-cli memory ~/mnt/traj-memory        # or mount elsewhere
trajectoriz-cli memory --foreground             # or run attached

ls memory
cat memory/README.md                            # what the directory is, in the directory
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

- **Search** — metadata by default (no parsing), `--content` for full-content search across messages, tool calls and results with three backends (sqlite / grep / recoll); space-separated words are AND, `\|` is OR
- **Unified record API** — iterate and parse sessions from Claude Code, Codex, Copilot, OpenCode, Hermes and more through a single `iter_records()` / `parse_record()` interface
- **Blame** — trace every agent edit to a file across all trajectory sources, with line/char deltas
- **HTML export** — `trajectoriz-cli show <id> --html` renders a trajectory as a self-contained HTML page
- **ATIF export** — `trajectoriz.atif` translates parsed trajectories to [ATIF v1.7](https://harborframework.com/docs/agents/trajectory-format) (Claude Code, Codex, Copilot, agentknit, or any `iter_records()`/`parse_record()` result)
- **Memory filesystem** — `trajectoriz-cli memory <mountpoint>` mounts a read-only FUSE view where every local trajectory is one ATIF JSON file
- **Leaked-secret scan** — `trajectoriz-cli secrets` searches every trajectory for the verbatim value of every OS keyring secret, and reports hits redacted, by fingerprint

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
