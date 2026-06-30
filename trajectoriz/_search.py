"""Search backends for trajectoriz full-content trajectory search."""
from __future__ import annotations

import abc
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Iterable

import trajectoriz as tz

TrajRecord = tz.TrajectoryRecord

_HOME = Path.home()
_RECOLL_CONFDIR = _HOME / ".recoll-trajectories"
_RECOLL_MAX_CANDIDATES = 200

# Path prefix → (agent name, short_id prefix)
_PATH_AGENTS: list[tuple[Path, str, str]] = [
    (_HOME / ".claude" / "projects", "claude", "cl"),
    (_HOME / ".local" / "share" / "agent_probe", "agent_probe", "ap"),
    (_HOME / ".codex" / "sessions", "codex", "cx"),
    (_HOME / ".copilot" / "session-state", "copilot", "cp"),
]

_FIRST_MSG_FNS = {
    "claude": tz.get_first_user_message_claude,
    "agent_probe": tz.get_first_user_message_agent_probe,
    "copilot": tz.get_first_user_message_copilot,
}

_PARSERS = {
    "claude": tz.parse_claude_trajectory,
    "agent_probe": tz.parse_agent_probe_trajectory,
    "copilot": tz.parse_copilot_event_trajectory,
    "codex": tz.parse_codex_trajectory,
}


def _parse_terms(query: str) -> list[str]:
    r"""Split a grep-style OR query on \| into individual lowercase terms."""
    return [t.lower() for t in query.split(r"\|") if t]


def _matches_any(text: str, terms: list[str]) -> bool:
    tl = text.lower()
    return any(term in tl for term in terms)


def _make_snippet(text: str, terms: list[str], context: int = 50) -> str:
    """Return a short single-line excerpt around the first match of any term."""
    tl = text.lower()
    best_idx, best_len = -1, 0
    for term in terms:
        idx = tl.find(term)
        if idx != -1 and (best_idx == -1 or idx < best_idx):
            best_idx, best_len = idx, len(term)
    if best_idx == -1:
        return ""
    start = max(0, best_idx - context)
    end = min(len(text), best_idx + best_len + context)
    snippet = text[start:end].replace("\n", " ").strip()
    if start > 0:
        snippet = "…" + snippet
    if end < len(text):
        snippet = snippet + "…"
    return snippet


def _step_search_blobs(step: dict) -> list[str]:
    """Return the searchable text fields of a step."""
    blobs: list[str] = []
    if step.get("message"):
        blobs.append(step["message"])
    if step.get("reasoning_content"):
        blobs.append(step["reasoning_content"])
    for tc in step.get("tool_calls", []):
        blobs.append(json.dumps(tc.get("arguments", {}), ensure_ascii=False))
    for res in (step.get("observation") or {}).get("results", []):
        blobs.append(res.get("content", ""))
    return blobs


SearchMatch = tuple[TrajRecord, int, str]  # (record, step_id, snippet)


class SearchBackend(abc.ABC):
    """Abstract base for full-content trajectory search backends."""

    @abc.abstractmethod
    def search(
        self,
        records: Iterable[TrajRecord],
        terms: list[str],
    ) -> list[SearchMatch]:
        """Return (record, step_id, snippet) for records matching any term."""


class GrepBackend(SearchBackend):
    """In-process search: parse each trajectory file and scan text in memory."""

    def search(
        self,
        records: Iterable[TrajRecord],
        terms: list[str],
    ) -> list[SearchMatch]:
        sorted_records = sorted(records, key=lambda r: r.timestamp, reverse=True)
        matches: list[SearchMatch] = []
        for rec in sorted_records:
            if _matches_any(rec.first_msg or "", terms):
                matches.append((rec, 1, _make_snippet(rec.first_msg or "", terms)))
                continue
            traj = tz.parse_record(rec)
            if traj is None:
                continue
            for step in traj.steps:
                for blob in _step_search_blobs(step):
                    if _matches_any(blob, terms):
                        matches.append((rec, step["step_id"], _make_snippet(blob, terms)))
                        break
        return matches


def _recoll_candidate_paths(query: str, confdir: Path, max_candidates: int) -> list[Path]:
    """Run recoll and return deduplicated JSONL candidate paths."""
    env = {**os.environ, "RECOLL_CONFDIR": str(confdir)}
    result = subprocess.run(
        ["recoll", "-t", "-b", "-n", f"0-{max_candidates}", "-q", query],
        capture_output=True, text=True, env=env,
    )
    seen_stems: set[str] = set()
    paths: list[Path] = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line.startswith("file://"):
            continue
        path = Path(line.removeprefix("file://"))
        if path.suffix != ".jsonl":
            continue
        if path.stem in seen_stems:
            continue
        seen_stems.add(path.stem)
        paths.append(path)
    return paths


def _agent_for_path(path: Path) -> tuple[str, str] | None:
    """Return (agent, short_id_prefix) for a trajectory path, or None."""
    for base, agent, prefix in _PATH_AGENTS:
        if path.is_relative_to(base):
            return agent, prefix
    fmt = tz._detect_jsonl_format(path)
    if not fmt:
        return None
    prefix = {"claude": "cl", "codex": "cx", "copilot": "cp", "agent_probe": "ap"}.get(fmt, "xx")
    return fmt, prefix



def _search_file(path: Path, terms: list[str]) -> list[SearchMatch]:
    """Step-level search within a single JSONL trajectory file."""
    result = _agent_for_path(path)
    if result is None:
        return []
    agent, prefix = result
    parse_fn = _PARSERS.get(agent)
    if parse_fn is None:
        return []
    try:
        traj = parse_fn(path)
    except Exception:
        return []

    traj_id = f"{prefix}-{hashlib.sha256(str(path).encode()).hexdigest()[:8]}"
    ts = traj.steps[0].get("timestamp", "") if traj.steps else ""
    fn = _FIRST_MSG_FNS.get(agent)
    _, first_msg = fn(path) if fn else ("", "")
    rec = TrajRecord(traj_id, agent, ts, first_msg, path)

    matches: list[SearchMatch] = []
    for step in traj.steps:
        for blob in _step_search_blobs(step):
            if _matches_any(blob, terms):
                matches.append((rec, step["step_id"], _make_snippet(blob, terms)))
                break
    return matches


class RecollBackend(SearchBackend):
    """Full-text search via the recoll CLI.

    Uses a dedicated recoll index at ~/.recoll-trajectories/.
    Run `trajectoriz-cli refresh` to install the config and build the index.
    """

    def __init__(
        self,
        confdir: Path | None = None,
        max_candidates: int = _RECOLL_MAX_CANDIDATES,
    ) -> None:
        self._confdir = confdir or _RECOLL_CONFDIR
        self._max_candidates = max_candidates

    def search(
        self,
        records: Iterable[TrajRecord],
        terms: list[str],
    ) -> list[SearchMatch]:
        xapiandb = self._confdir / "xapiandb"
        if not xapiandb.exists():
            raise NotImplementedError(
                "recoll index not found at ~/.recoll-trajectories/xapiandb.\n"
                "Run `trajectoriz-cli refresh` to install the config and build the index."
            )
        # Recoll query: join terms with OR
        query = " OR ".join(terms) if len(terms) > 1 else (terms[0] if terms else "")
        candidates = _recoll_candidate_paths(query, self._confdir, self._max_candidates)
        matches: list[SearchMatch] = []
        for path in candidates:
            if path.exists():
                matches.extend(_search_file(path, terms))
        return matches


class SqliteBackend(SearchBackend):
    """FTS5 search via a local SQLite index at ~/.cache/trajectoriz/fts.db.

    Run `trajectoriz-cli refresh` to build or update the index.
    Note: uses word tokenisation — matches whole words, not substrings.
    """

    def __init__(self, db_path: Path | None = None) -> None:
        self._db_path = db_path

    def search(
        self,
        records: Iterable[TrajRecord],
        terms: list[str],
    ) -> list[SearchMatch]:
        from trajectoriz._fts import fts_db_path, search_fts

        path = self._db_path or fts_db_path()
        if not path.exists():
            raise NotImplementedError(
                f"FTS index not found at {path}.\n"
                "Run `trajectoriz-cli refresh` to build it,\n"
                "then re-run with --backend sqlite."
            )
        try:
            return search_fts(terms, db_path=path)
        except Exception as exc:
            raise NotImplementedError(str(exc)) from exc


_BACKENDS: dict[str, type[SearchBackend]] = {
    "grep": GrepBackend,
    "recoll": RecollBackend,
    "sqlite": SqliteBackend,
}


def get_backend(name: str) -> SearchBackend:
    cls = _BACKENDS.get(name)
    if cls is None:
        choices = ", ".join(_BACKENDS)
        raise ValueError(f"Unknown search backend: {name!r}. Choose from: {choices}")
    return cls()
