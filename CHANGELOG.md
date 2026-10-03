# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `trajectoriz-cli secrets --since/--until/--date YYYY-MM-DD` restricts the scan
  to trajectories of that period (epoch timestamps and timestamp-less journals
  are dated too).
- `trajectoriz-cli reindex` incrementally updates the SQLite FTS index without
  touching recoll, with `--local` / `--dir PATH` to scope the rebuild to one
  folder — fast enough to run at agent startup.

- Public source-agnostic record parsing API via `TrajectoryRecord`, `iter_records()`,
  `iter_all_records()`, `iter_local_records()`, and `parse_record()`.
- `trajectoriz.atif` module translating parsed trajectories (Claude Code, Codex,
  Copilot, agentknit, or any `parse_record()` result) to ATIF v1.7.
- `trajectoriz-cli memory` mounts a read-only FUSE filesystem exposing each local
  trajectory as an ATIF v1.7 JSON file (`trajectoriz[fuse]` extra), with
  `--unmount` to recover a mountpoint whose daemon died.
- `trajectoriz-cli secrets` reports OS keyring secrets that appear in cleartext in
  local trajectories: every keyring value is searched verbatim (no entropy or
  pattern heuristics) across trajectory files, the agentknit `*_messages.json`
  request payloads and the SQLite session stores. Each finding names the model
  endpoint the conversation was sent to, and is reported by keyring label plus
  SHA-256 fingerprint with redacted context — the value itself is never printed.
  The command exits 1 on any hit.
- `agent_probe_sidecar()` locates the `*_messages.json` payload agentknit writes
  beside a journal, and `parse_agent_probe_trajectory` reads the model, session id
  and endpoint from it (`ParsedTrajectory.extra_agent["endpoint"]`).

### Changed

- **`search` now searches first messages, IDs and agents by default** instead of the
  full content of every step; pass `--content` (or `--grep`) for the exhaustive
  search. Locating a session no longer parses trajectories. `--fast` is still
  accepted and is now a no-op.
- The memory filesystem serves a `README.md` explaining the directory, the ATIF
  payload shape and how to locate a session with `search` instead of grepping.
- Store scans are much faster: the per-file probes (format, working directory,
  first user message) are memoized on disk and invalidated by mtime, and a
  project's scan no longer reads first messages for files it filters out.
  Scanning a machine with ~12k trajectory files went from ~9.5s to ~0.5s,
  speeding up `list`, `blame`, `search --local` and the memory filesystem.

### Fixed

- agent_probe journals without a user message (most of them: sessions that only
  hold `session_start`/`session_end`) had an empty timestamp, so they sorted last
  and were dropped by `list --since/--date`. They are now dated by their first
  event; the cached probe result is invalidated once.
- `list --since/--date`, and the dates shown and sorted on by `list`, `search`,
  `blame` and `secrets`, now handle epoch timestamps (opencode, codex_db) and
  date timestamp-less records by file mtime, via the new
  `normalize_timestamp()`, `record_datetime()` and `record_date()` helpers.
- `list` hides agent_probe sessions without a user message (empty sessions).
- **The memory filesystem could deadlock the whole machine.** Store scans walked
  into FUSE mounts — `~/.local/share/agent_probe` is often a symlink into a repo,
  and that repo can hold a memory mount — so a daemon scanning its stores from
  inside a request waited on a reply it was itself supposed to send. The kernel
  holds the mount's inode lock for the duration, so every process that then
  touched the directory (git, pytest, ripgrep, a language server, the agent's own
  file tools) ended up in uninterruptible `D` state, unkillable even by SIGKILL.
  Three changes: store walks skip FUSE mountpoints (read from `/proc/self/mountinfo`,
  never by stat'ing a mountpoint that may hang) and the daemon's own mountpoint;
  the scan runs off the request path with a deadline, so no handler can block
  indefinitely; and `memory --unmount` now aborts a wedged FUSE connection, which
  is the only thing that releases processes already stuck on it.
- `ls -l`, `find` and git no longer render every trajectory in a mount just to
  learn its size: the ATIF payload size is memoized on disk with the other
  per-file probes. Stat'ing a 108-file mount went from 1.9s to 0.1s.
- **agent_probe sessions were being missed**: the store was walked at a fixed
  depth (`*/*/*.jsonl`), which skipped the 499 sessions filed one level up or in
  the store root, including every agentknit run. The walk is now recursive.
- agentknit's second-generation journal (`turn_start` / `message` / `tool_start` /
  `tool_end`) is parsed instead of yielding an empty trajectory, so its steps,
  tool calls, first user message and model are all available.
- The on-disk parse cache is keyed by a parser revision. A trajectory file does
  not change when a parser learns to read more of it, so stale entries used to
  keep serving the older, poorer parse indefinitely.
- The memory filesystem no longer rescans every trajectory store on each
  `getattr`, `open` and `read`: a listing is shared between operations and each
  file's ATIF payload is rendered once per open. Browsing a mount went from
  minutes to milliseconds.

## [0.1.0] - 2025-05-31

### Added

- Initial release.
- Functions to locate trajectory files for Claude Code, Codex CLI, pi coding agent, Cursor, and Copilot CLI.
