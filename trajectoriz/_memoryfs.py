"""Read-only FUSE filesystem exposing local trajectories as ATIF JSON files.

Each file in the mounted directory is one trajectory, rendered to ATIF on
first access. The filesystem is built to survive a coding agent browsing it:

* The directory listing comes from one scan of the trajectory stores, shared
  by every operation and refreshed at most once per ``listing_ttl`` seconds,
  so a new session still appears without remounting.
* A file's ATIF payload is rendered once per ``open`` and served from an
  open-file handle, so the kernel's many read requests for one file cost a
  slice each instead of a re-render.
* Rendered payloads are kept in a byte-bounded LRU cache, so re-reading the
  same trajectory is free while a walk over thousands of them stays bounded.
"""
from __future__ import annotations

import errno
import json
import os
import re
import stat
import threading
import time
from collections import OrderedDict
from pathlib import Path

import fuse  # pyright: ignore[reportMissingImports]  # optional 'fuse' extra

import trajectoriz as tz
from . import atif as atif_mod

DEFAULT_LISTING_TTL = 2.0                       # seconds
DEFAULT_CACHE_BYTES = 256 * 1024 * 1024         # rendered payloads held in memory
_MISS_REFRESH_AGE = 1.0                         # rescan on ENOENT at most this often

README_NAME = "README.md"

_README_TEMPLATE = """\
# Trajectory memory

Every file in this directory is one past agent session for `{repo_root}`, in
[ATIF v1.7](https://harborframework.com/docs/agents/trajectory-format) — one
JSON envelope holding the agent and model, the ordered `steps` (messages, tool
calls, observations) and `final_metrics`. The same shape whatever agent
recorded it: Claude Code, Codex, Copilot CLI, opencode and friends.

Files are named `<date>_<agent>_<id>.atif.json` and are generated when read,
so this directory is a view, not a copy. It is read-only.

## Finding a session

Grepping works, but it parses every session it touches. To locate one, ask
trajectoriz instead — it searches first messages, IDs and agents without
parsing anything:

    trajectoriz-cli search "the thing you remember" --local

Then read the file whose name carries the ID it reports:

    ls | grep <id>

Add `--content` to that search to look inside every step (tool calls, their
results, all messages) rather than only first messages.

## Reading a session

    jq -r '.steps[0].message' <file>              # the instruction that started it
    jq '.final_metrics' <file>                    # steps, tool calls, tokens
    jq -r '.steps[].tool_calls[]?.function_name' <file>   # what it actually ran
"""


def _slugify(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-") or "x"


def _filename_for(rec: tz.TrajectoryRecord) -> str:
    date = (rec.timestamp or "")[:10] or "0000-00-00"
    return f"{date}_{_slugify(rec.agent)}_{_slugify(rec.id)}.atif.json"


def _source_mtime(rec: tz.TrajectoryRecord) -> float:
    if isinstance(rec.source, Path):
        try:
            return rec.source.stat().st_mtime
        except OSError:
            return 0.0
    return 0.0


class MemoryFS(fuse.Operations):
    """Exposes trajectoriz.iter_local_records(repo_root) as ATIF files."""

    use_ns = True

    def __init__(
        self,
        repo_root: str,
        listing_ttl: float = DEFAULT_LISTING_TTL,
        cache_bytes: int = DEFAULT_CACHE_BYTES,
    ):
        self.repo_root = repo_root
        self.listing_ttl = listing_ttl
        self.cache_bytes = cache_bytes
        self._mount_time = time.time()
        self._lock = threading.Lock()       # guards the caches below
        self._scan_lock = threading.Lock()  # lets one thread scan at a time
        self._listing: dict[str, tz.TrajectoryRecord] | None = None
        self._listing_at = 0.0
        self._content: OrderedDict[str, tuple[float, bytes]] = OrderedDict()
        self._content_bytes = 0
        self._handles: dict[int, bytes] = {}
        self._next_fh = 1
        # A short explainer for whoever opens the directory — including an
        # agent that found it and has no idea what these files are.
        self._readme = _README_TEMPLATE.format(repo_root=repo_root).encode("utf-8")

    # ── Caches ───────────────────────────────────────────────────────────

    def _cached_listing(self, max_age: float) -> dict[str, tz.TrajectoryRecord] | None:
        with self._lock:
            if self._listing is None:
                return None
            if time.monotonic() - self._listing_at > max_age:
                return None
            return self._listing

    def _records(self, max_age: float | None = None) -> dict[str, tz.TrajectoryRecord]:
        """Return {filename: record}, scanning the stores only when stale."""
        if max_age is None:
            max_age = self.listing_ttl
        listing = self._cached_listing(max_age)
        if listing is not None:
            return listing
        with self._scan_lock:
            # Another thread may have refreshed the listing while we queued.
            listing = self._cached_listing(max_age)
            if listing is not None:
                return listing
            scanned = {
                _filename_for(rec): rec
                for rec in tz.iter_local_records(self.repo_root)
            }
            with self._lock:
                self._listing = scanned
                self._listing_at = time.monotonic()
            return scanned

    def _content_for(self, name: str, rec: tz.TrajectoryRecord) -> bytes:
        """Return the ATIF payload for a record, rendering it at most once."""
        mtime = _source_mtime(rec)
        with self._lock:
            hit = self._content.get(name)
            if hit is not None and hit[0] == mtime:
                self._content.move_to_end(name)
                return hit[1]

        parsed = tz.parse_record(rec)
        envelope = atif_mod.parsed_record_to_atif(
            rec, parsed, run_id=rec.id, profile=rec.agent, repo_root=self.repo_root,
        )
        data = json.dumps(envelope, indent=2).encode("utf-8")

        with self._lock:
            stale = self._content.pop(name, None)
            if stale is not None:
                self._content_bytes -= len(stale[1])
            self._content[name] = (mtime, data)
            self._content_bytes += len(data)
            while self._content_bytes > self.cache_bytes and len(self._content) > 1:
                _, (_, evicted) = self._content.popitem(last=False)
                self._content_bytes -= len(evicted)
        return data

    def _lookup(self, path: str) -> tuple[str, tz.TrajectoryRecord]:
        name = path.lstrip("/")
        rec = self._records().get(name)
        if rec is None:
            # Could be a session recorded since the last scan — but don't
            # rescan for every probe of a name that simply does not exist.
            rec = self._records(_MISS_REFRESH_AGE).get(name)
        if rec is None:
            raise fuse.FuseOSError(errno.ENOENT)
        return name, rec

    # ── Operations ───────────────────────────────────────────────────────

    def getattr(self, path, fh=None):
        now = self._mount_time
        uid, gid = os.getuid(), os.getgid()
        if path == "/":
            return {
                "st_mode": stat.S_IFDIR | 0o555,
                "st_nlink": 2,
                "st_size": 0,
                "st_uid": uid, "st_gid": gid,
                "st_ctime": now, "st_mtime": now, "st_atime": now,
            }
        if path == "/" + README_NAME:
            data = self._readme
        else:
            name, rec = self._lookup(path)
            data = self._content_for(name, rec)
        return {
            "st_mode": stat.S_IFREG | 0o444,
            "st_nlink": 1,
            "st_size": len(data),
            "st_uid": uid, "st_gid": gid,
            "st_ctime": now, "st_mtime": now, "st_atime": now,
        }

    def readdir(self, path, fh):
        return [".", "..", README_NAME] + sorted(self._records())

    def open(self, path, flags):  # pyright: ignore[reportIncompatibleMethodOverride]
        if flags & (os.O_WRONLY | os.O_RDWR):
            raise fuse.FuseOSError(errno.EROFS)
        if path == "/" + README_NAME:
            data = self._readme
        else:
            name, rec = self._lookup(path)
            data = self._content_for(name, rec)
        with self._lock:
            fh = self._next_fh
            self._next_fh += 1
            self._handles[fh] = data
        return fh

    def read(self, path, size, offset, fh):  # pyright: ignore[reportIncompatibleMethodOverride]
        with self._lock:
            data = self._handles.get(fh)
        if data is None:  # read without a handle of ours
            if path == "/" + README_NAME:
                data = self._readme
            else:
                name, rec = self._lookup(path)
                data = self._content_for(name, rec)
        return data[offset : offset + size]

    def release(self, path, fh):
        with self._lock:
            self._handles.pop(fh, None)
        return 0

    def write(self, path, data, offset, fh):
        raise fuse.FuseOSError(errno.EROFS)

    def unlink(self, path):
        raise fuse.FuseOSError(errno.EROFS)

    def mkdir(self, path, mode):
        raise fuse.FuseOSError(errno.EROFS)

    def rmdir(self, path):
        raise fuse.FuseOSError(errno.EROFS)

    def create(self, path, mode, fi=None):
        raise fuse.FuseOSError(errno.EROFS)

    def truncate(self, path, length, fh=None):
        raise fuse.FuseOSError(errno.EROFS)
