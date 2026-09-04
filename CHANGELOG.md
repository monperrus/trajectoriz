# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Public source-agnostic record parsing API via `TrajectoryRecord`, `iter_records()`,
  `iter_all_records()`, `iter_local_records()`, and `parse_record()`.
- `trajectoriz.atif` module translating parsed trajectories (Claude Code, Codex,
  Copilot, agentknit, or any `parse_record()` result) to ATIF v1.7.
- `trajectoriz-cli memory` mounts a read-only FUSE filesystem exposing each local
  trajectory as an ATIF v1.7 JSON file (`trajectoriz[fuse]` extra), with
  `--unmount` to recover a mountpoint whose daemon died.
- `trajectoriz-cli secrets` reports OS keyring secrets that appear in cleartext in
  local trajectories: every keyring value is searched verbatim (no entropy or
  pattern heuristics) across trajectory files and the SQLite session stores.
  Findings are reported by keyring label plus SHA-256 fingerprint with redacted
  context — the value itself is never printed — and the command exits 1 on any hit.

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

- The memory filesystem no longer rescans every trajectory store on each
  `getattr`, `open` and `read`: a listing is shared between operations and each
  file's ATIF payload is rendered once per open. Browsing a mount went from
  minutes to milliseconds.

## [0.1.0] - 2025-05-31

### Added

- Initial release.
- Functions to locate trajectory files for Claude Code, Codex CLI, pi coding agent, Cursor, and Copilot CLI.
