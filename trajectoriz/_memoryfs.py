"""Read-only FUSE filesystem exposing local trajectories as ATIF JSON files.

Each file in the mounted directory is one trajectory, rendered on first
access via :func:`trajectoriz.atif.parsed_record_to_atif`. Directory
listings are recomputed on every ``readdir`` (a cheap glob, same cost the
``list``/``search`` commands already pay), so newly recorded sessions show
up without remounting.
"""
from __future__ import annotations

import errno
import json
import os
import re
import stat
import time
from pathlib import Path

import fuse

import trajectoriz as tz
from . import atif as atif_mod


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

    def __init__(self, repo_root: str):
        self.repo_root = repo_root
        self._mount_time = time.time()
        # filename -> (source_mtime, encoded ATIF bytes)
        self._content_cache: dict[str, tuple[float, bytes]] = {}

    def _records(self) -> dict[str, tz.TrajectoryRecord]:
        by_name: dict[str, tz.TrajectoryRecord] = {}
        for rec in tz.iter_local_records(self.repo_root):
            by_name[_filename_for(rec)] = rec
        return by_name

    def _content(self, name: str, rec: tz.TrajectoryRecord) -> bytes:
        mtime = _source_mtime(rec)
        cached = self._content_cache.get(name)
        if cached is not None and cached[0] == mtime:
            return cached[1]
        parsed = tz.parse_record(rec)
        envelope = atif_mod.parsed_record_to_atif(
            rec, parsed, run_id=rec.id, profile=rec.agent, repo_root=self.repo_root,
        )
        data = json.dumps(envelope, indent=2).encode("utf-8")
        self._content_cache[name] = (mtime, data)
        return data

    def _lookup(self, path: str) -> tuple[str, tz.TrajectoryRecord]:
        name = path.lstrip("/")
        records = self._records()
        rec = records.get(name)
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
        name, rec = self._lookup(path)
        data = self._content(name, rec)
        return {
            "st_mode": stat.S_IFREG | 0o444,
            "st_nlink": 1,
            "st_size": len(data),
            "st_uid": uid, "st_gid": gid,
            "st_ctime": now, "st_mtime": now, "st_atime": now,
        }

    def readdir(self, path, fh):
        return [".", ".."] + sorted(self._records())

    def open(self, path, flags):
        if flags & (os.O_WRONLY | os.O_RDWR):
            raise fuse.FuseOSError(errno.EROFS)
        self._lookup(path)  # raises ENOENT if missing
        return 0

    def read(self, path, size, offset, fh):
        name, rec = self._lookup(path)
        data = self._content(name, rec)
        return data[offset : offset + size]

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
