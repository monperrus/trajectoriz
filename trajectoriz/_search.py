"""Search backends for trajectoriz full-content trajectory search."""
from __future__ import annotations

import abc
import json
from typing import Iterable

import trajectoriz as tz

TrajRecord = tz.TrajectoryRecord


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


class RecollBackend(SearchBackend):
    """Full-text search via the recoll CLI.

    Not yet implemented. To use this backend:
      1. Install recoll and configure it to index your trajectory directories.
      2. Run `recollindex` to build the index.
      3. Pass `--backend recoll` to `trajectoriz-cli search`.
    """

    def search(
        self,
        records: Iterable[TrajRecord],
        terms: list[str],
    ) -> list[SearchMatch]:
        raise NotImplementedError(
            "recoll backend is not yet implemented.\n"
            "Install recoll, index your trajectory directories with recollindex,\n"
            "then re-run with --backend recoll."
        )


class SqliteBackend(SearchBackend):
    """FTS5 search via a local SQLite index built by `trajectoriz-cli index`.

    Not yet implemented. To use this backend:
      1. Run `trajectoriz-cli index` to build the FTS index.
      2. Pass `--backend sqlite` to `trajectoriz-cli search`.
    """

    def search(
        self,
        records: Iterable[TrajRecord],
        terms: list[str],
    ) -> list[SearchMatch]:
        raise NotImplementedError(
            "sqlite backend is not yet implemented.\n"
            "Run `trajectoriz-cli index` to build the FTS index,\n"
            "then re-run with --backend sqlite."
        )


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
