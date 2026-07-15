"""SQLite FTS5 index for trajectory full-text search."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Iterable

import trajectoriz as tz

TrajRecord = tz.TrajectoryRecord

_FTS_DB_NAME = "fts.db"


def fts_db_path(cache_dir: Path | None = None) -> Path:
    return tz._cache_dir(cache_dir) / _FTS_DB_NAME


def _init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS records (
            traj_id      TEXT PRIMARY KEY,
            agent        TEXT NOT NULL DEFAULT '',
            timestamp    TEXT NOT NULL DEFAULT '',
            first_msg    TEXT NOT NULL DEFAULT '',
            source_json  TEXT NOT NULL DEFAULT '',
            source_mtime REAL NOT NULL DEFAULT 0.0
        );

        CREATE VIRTUAL TABLE IF NOT EXISTS steps_fts USING fts5(
            content,
            traj_id  UNINDEXED,
            step_id  UNINDEXED,
            tokenize = 'unicode61'
        );
    """)


def _source_to_json(source: object) -> str:
    if isinstance(source, Path):
        return str(source)
    return json.dumps(source, ensure_ascii=False)


def _source_mtime(rec: TrajRecord) -> float:
    if isinstance(rec.source, Path):
        try:
            return rec.source.stat().st_mtime
        except OSError:
            return 0.0
    return 0.0


def _step_blobs(step: dict) -> list[str]:
    # Lazy import to avoid circular dependency at module load time
    from trajectoriz._search import _step_search_blobs
    return _step_search_blobs(step)


def _index_one(conn: sqlite3.Connection, rec: TrajRecord) -> int:
    """Parse and index one record into the open connection. Returns steps inserted."""
    traj = tz.parse_record(rec)
    if traj is None:
        return 0

    conn.execute("DELETE FROM steps_fts WHERE traj_id = ?", (rec.id,))

    rows: list[tuple[str, str, int]] = []
    for step in traj.steps:
        text = "\n".join(_step_blobs(step)).strip()
        if text:
            rows.append((text, rec.id, step["step_id"]))

    if rows:
        conn.executemany(
            "INSERT INTO steps_fts (content, traj_id, step_id) VALUES (?, ?, ?)",
            rows,
        )

    conn.execute(
        """INSERT OR REPLACE INTO records
               (traj_id, agent, timestamp, first_msg, source_json, source_mtime)
               VALUES (?, ?, ?, ?, ?, ?)""",
        (
            rec.id,
            rec.agent,
            rec.timestamp or "",
            rec.first_msg or "",
            _source_to_json(rec.source),
            _source_mtime(rec),
        ),
    )
    return len(rows)


def build_index(
    records_iter: Iterable[TrajRecord],
    db_path: Path | None = None,
    force: bool = False,
) -> tuple[int, int]:
    """Build or incrementally update the FTS index.

    Skips records whose source mtime matches the stored value (incremental).
    Pass force=True to re-index everything.

    Returns (indexed_count, skipped_count).
    """
    path = db_path or fts_db_path()
    conn = sqlite3.connect(str(path))
    try:
        _init_schema(conn)
        indexed = skipped = 0
        for rec in records_iter:
            mtime = _source_mtime(rec)
            if not force and mtime != 0.0:
                row = conn.execute(
                    "SELECT source_mtime FROM records WHERE traj_id = ?", (rec.id,)
                ).fetchone()
                if row is not None and row[0] == mtime:
                    skipped += 1
                    continue
            _index_one(conn, rec)
            indexed += 1
        conn.commit()
        return indexed, skipped
    finally:
        conn.close()


def source_from_json(source_json: str) -> object:
    """Reconstruct a record source from its stored JSON representation."""
    if source_json.startswith("/"):
        return Path(source_json)
    try:
        return json.loads(source_json)
    except (json.JSONDecodeError, ValueError):
        return source_json


def _fts_quote(term: str) -> str:
    """Quote a term for FTS5 MATCH syntax."""
    return '"' + term.replace('"', '""') + '"'


def search_fts(
    clauses: list[list[str]],
    db_path: Path | None = None,
) -> list[tuple[TrajRecord, int, str]]:
    """Search the FTS index. Returns (record, step_id, snippet) tuples.

    Each inner list is an AND clause (all words must appear in the step).
    Outer list is OR (any clause suffices).

    Note: FTS5 uses word tokenisation, so only whole-word matches are found
    (unlike the grep backend which does substring matching).
    """
    from trajectoriz._search import _make_snippet

    path = db_path or fts_db_path()
    if not path.exists():
        raise FileNotFoundError(
            f"FTS index not found at {path}. Run `trajectoriz-cli refresh` first."
        )

    # Build FTS5 query: AND within a clause, OR between clauses.
    clause_parts = [
        "(" + " AND ".join(_fts_quote(t) for t in clause) + ")" if len(clause) > 1
        else _fts_quote(clause[0])
        for clause in clauses
    ]
    fts_query = " OR ".join(clause_parts) if clause_parts else '""'
    flat = [t for clause in clauses for t in clause]

    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT f.traj_id, f.step_id, f.content,
                   r.agent, r.timestamp, r.first_msg, r.source_json
            FROM steps_fts f
            JOIN records r ON r.traj_id = f.traj_id
            WHERE steps_fts MATCH ?
            ORDER BY rank
            """,
            (fts_query,),
        ).fetchall()
    finally:
        conn.close()

    matches: list[tuple[TrajRecord, int, str]] = []
    for row in rows:
        source = source_from_json(row["source_json"])
        rec = TrajRecord(
            row["traj_id"],
            row["agent"],
            row["timestamp"],
            row["first_msg"],
            source,
        )
        snippet = _make_snippet(row["content"], flat)
        matches.append((rec, int(row["step_id"]), snippet))
    return matches
