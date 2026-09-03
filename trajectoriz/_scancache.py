"""Disk-backed memo for the per-file probes used when scanning trajectory stores.

Locating trajectories means opening thousands of JSONL files just to sniff
their format, working directory and first user message. Those probes are pure
functions of a file's bytes, so their results are memoized here, keyed by
(path, kind) and validated against the file's mtime and size — a changed file
is re-probed, an unchanged one is not.

The cache is strictly an optimization: on any sqlite or OS error the probe
simply runs, so a missing, locked or corrupt cache only costs speed.
"""
from __future__ import annotations

import atexit
import json
import sqlite3
import threading
from pathlib import Path
from typing import Any, Callable

_DB_NAME = "scan-cache.sqlite3"
_FLUSH_EVERY = 256

_MISS = object()

_lock = threading.Lock()          # guards _mem and _pending
_db_lock = threading.Lock()       # serializes use of the shared connection
_mem: dict[tuple[str, str], tuple[int, int, Any]] = {}
_pending: list[tuple[str, str, int, int, str]] = []
_pending_static: list[tuple[str, str, str]] = []
_conn: sqlite3.Connection | None = None
_conn_failed = False


def _connect() -> sqlite3.Connection | None:
    global _conn, _conn_failed
    if _conn is not None or _conn_failed:
        return _conn
    try:
        from . import _cache_dir

        conn = sqlite3.connect(
            str(_cache_dir() / _DB_NAME), timeout=1.0, check_same_thread=False
        )
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=OFF")
        conn.execute(
            "CREATE TABLE IF NOT EXISTS scan ("
            "path TEXT NOT NULL, kind TEXT NOT NULL, mtime_ns INTEGER NOT NULL, "
            "size INTEGER NOT NULL, value TEXT NOT NULL, PRIMARY KEY (path, kind))"
        )
        conn.execute(
            "CREATE TABLE IF NOT EXISTS static ("
            "key TEXT NOT NULL, kind TEXT NOT NULL, value TEXT NOT NULL, "
            "PRIMARY KEY (key, kind))"
        )
        conn.commit()
        _conn = conn
    except (sqlite3.Error, OSError):
        _conn_failed = True
    return _conn


def _load(path: str, kind: str, mtime_ns: int, size: int) -> Any:
    conn = _connect()
    if conn is None:
        return _MISS
    try:
        with _db_lock:
            row = conn.execute(
                "SELECT mtime_ns, size, value FROM scan WHERE path=? AND kind=?",
                (path, kind),
            ).fetchone()
    except sqlite3.Error:
        return _MISS
    if not row or row[0] != mtime_ns or row[1] != size:
        return _MISS
    try:
        return json.loads(row[2])
    except (json.JSONDecodeError, TypeError):
        return _MISS


def flush() -> None:
    """Write buffered probe results to the cache database."""
    with _lock:
        if not _pending and not _pending_static:
            return
        batch = list(_pending)
        batch_static = list(_pending_static)
        _pending.clear()
        _pending_static.clear()
    conn = _connect()
    if conn is None:
        return
    try:
        with _db_lock:
            if batch:
                conn.executemany(
                    "INSERT OR REPLACE INTO scan (path, kind, mtime_ns, size, value) "
                    "VALUES (?, ?, ?, ?, ?)",
                    batch,
                )
            if batch_static:
                conn.executemany(
                    "INSERT OR REPLACE INTO static (key, kind, value) VALUES (?, ?, ?)",
                    batch_static,
                )
            conn.commit()
    except sqlite3.Error:
        pass


def memo(path, kind: str, compute: Callable[[], Any]) -> Any:
    """Return compute(), reusing a cached result while path is unchanged."""
    try:
        st = Path(path).stat()
    except OSError:
        return compute()
    key = (str(path), kind)
    mtime_ns, size = st.st_mtime_ns, st.st_size

    with _lock:
        hit = _mem.get(key)
    if hit is not None and hit[0] == mtime_ns and hit[1] == size:
        return hit[2]

    stored = _load(key[0], kind, mtime_ns, size)
    if stored is not _MISS:
        with _lock:
            _mem[key] = (mtime_ns, size, stored)
        return stored

    value = compute()
    try:
        encoded = json.dumps(value)
    except (TypeError, ValueError):
        return value  # not cacheable, but still a valid result
    with _lock:
        _mem[key] = (mtime_ns, size, value)
        _pending.append((key[0], kind, mtime_ns, size, encoded))
        due = len(_pending) >= _FLUSH_EVERY
    if due:
        flush()
    return value


_preloaded: set[str] = set()


def preload(*kinds: str) -> None:
    """Load every cached row of the given kinds into the in-process memo.

    A scan probes tens of thousands of files, and querying the database once
    per file costs more than reading the whole (small) kind in one statement.
    Only the compact probes are worth preloading this way — a file's first
    user message can be arbitrarily long, so those stay lazy.
    """
    wanted = [kind for kind in kinds if kind not in _preloaded]
    if not wanted:
        return
    conn = _connect()
    if conn is None:
        return
    placeholders = ",".join("?" * len(wanted))
    try:
        with _db_lock:
            rows = conn.execute(
                f"SELECT path, kind, mtime_ns, size, value FROM scan "
                f"WHERE kind IN ({placeholders})",
                wanted,
            ).fetchall()
    except sqlite3.Error:
        return
    with _lock:
        for path, kind, mtime_ns, size, value in rows:
            try:
                decoded = json.loads(value)
            except (json.JSONDecodeError, TypeError):
                continue
            _mem.setdefault((path, kind), (mtime_ns, size, decoded))
        _preloaded.update(wanted)


def get_static(key: str, kind: str) -> Any:
    """Return a cached value that is immutable for its key, or None if absent.

    For facts that never change once known (a session's first user prompt, say)
    and so need no mtime validation.
    """
    mem_key = (f"static:{key}", kind)
    with _lock:
        hit = _mem.get(mem_key)
    if hit is not None:
        return hit[2]
    conn = _connect()
    if conn is None:
        return None
    try:
        with _db_lock:
            row = conn.execute(
                "SELECT value FROM static WHERE key=? AND kind=?", (key, kind)
            ).fetchone()
    except sqlite3.Error:
        return None
    if not row:
        return None
    try:
        value = json.loads(row[0])
    except (json.JSONDecodeError, TypeError):
        return None
    with _lock:
        _mem[mem_key] = (0, 0, value)
    return value


def put_static(key: str, kind: str, value: Any) -> None:
    """Cache a value that is immutable for its key."""
    try:
        encoded = json.dumps(value)
    except (TypeError, ValueError):
        return
    with _lock:
        _mem[(f"static:{key}", kind)] = (0, 0, value)
        _pending_static.append((key, kind, encoded))
        due = len(_pending_static) >= _FLUSH_EVERY
    if due:
        flush()


def clear_memory_cache() -> None:
    """Drop the in-process memo (the database is left alone)."""
    with _lock:
        _mem.clear()


atexit.register(flush)
