#!/usr/bin/env python3
"""Recoll filter for AI trajectory JSONL files.

Converts a JSONL trajectory file to HTML using the trajectoriz library.
No content is duplicated on disk — the JSONL is the sole source of truth.
"""
from __future__ import annotations

import hashlib
import html
import sys
from pathlib import Path

sys.path.insert(0, "/usr/share/recoll/filters")
import rclexecm
from rclbasehandler import RclBaseHandler

import trajectoriz as tz
from trajectoriz.cli import _trajectory_to_html, TrajRecord

_HOME = Path.home()

# Path-prefix → (agent, short_id_prefix)
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


def _short_id(prefix: str, key: str) -> str:
    return f"{prefix}-{hashlib.sha256(key.encode()).hexdigest()[:8]}"


def _traj_record_for(path: Path) -> TrajRecord | None:
    agent = prefix = None
    for base, a, p in _PATH_AGENTS:
        if path.is_relative_to(base):
            agent, prefix = a, p
            break

    if agent is None:
        fmt = tz._detect_jsonl_format(path)
        if not fmt:
            return None
        agent = fmt
        prefix = {"claude": "cl", "codex": "cx", "copilot": "cp", "agent_probe": "ap"}.get(fmt, "xx")

    fn = _FIRST_MSG_FNS.get(agent)
    ts, msg = fn(path) if fn else ("", "")
    return TrajRecord(_short_id(prefix, str(path)), agent, ts, msg, path)


class TrajHandler(RclBaseHandler):
    def __init__(self, em):
        super().__init__(em)

    def html_text(self, filename: bytes) -> bytes:
        path = Path(filename.decode("utf-8", errors="replace"))
        record = _traj_record_for(path)
        if record is None:
            self.em.rclog(f"rcltraj: unknown format for {path}")
            return b"<html><body>Unknown trajectory format.</body></html>"
        try:
            out = _trajectory_to_html(record, full=True)
        except Exception as exc:
            self.em.rclog(f"rcltraj: error rendering {path}: {exc}")
            return (
                f"<html><body>Error: {html.escape(str(exc))}</body></html>"
            ).encode()
        return out.encode("utf-8", errors="replace")


if __name__ == "__main__":
    proto = rclexecm.RclExecM()
    extract = TrajHandler(proto)
    rclexecm.main(proto, extract)
