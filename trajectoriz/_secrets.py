"""Find keyring secrets that appear in cleartext inside agent trajectories.

The scan is deliberately literal: every secret stored in the OS keyring is
searched for, verbatim, in every trajectory the machine keeps. No entropy or
dictionary heuristics decide what counts as a secret -- the keyring already did.

Secret values never leave this module: findings are reported by keyring label
and a truncated SHA-256 fingerprint, and match context is redacted. Printing a
leaked secret into a scan report would just create the next leak.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import subprocess
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import trajectoriz as tz
from trajectoriz._search import _step_search_blobs

DEFAULT_MIN_LENGTH = 8
DEFAULT_MAX_FILES = 25
_GREP_CHUNK = 1000
_READ_CHUNK = 8 << 20

# SQLite stores that back sessions with no trajectory file of their own.
_STORE_DBS = (
    Path.home() / ".local" / "share" / "opencode" / "opencode.db",
    Path.home() / ".copilot" / "session-store.db",
    Path.home() / ".hermes" / "state.db",
    Path.home() / ".codex" / "state_5.sqlite",
)


def _fingerprint(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", "replace")).hexdigest()[:12]


@dataclass(frozen=True)
class KeyringSecret:
    """A secret held in the OS keyring, and where it came from."""

    label: str
    schema: str
    collection: str
    value: str

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.value)

    @property
    def needle(self) -> str:
        """The literal text to search for: the longest single line of the value.

        Multi-line secrets (PEM keys, JSON blobs) are commonly reflowed when
        they leak, so the longest line is the reliable marker.
        """
        return max(self.value.splitlines() or [""], key=len)


@dataclass(frozen=True)
class Leak:
    """One trajectory in which one keyring secret appears in cleartext."""

    fingerprint: str
    labels: tuple[str, ...]
    schema: str
    length: int
    agent: str
    trajectory_id: str
    timestamp: str
    step: int | None
    source: str
    model: str
    endpoint: str
    occurrences: int
    context: str
    partial: bool

    @property
    def destination(self) -> str:
        return format_destination(self.model, self.endpoint)


@dataclass(frozen=True)
class CommonMatch:
    """A keyring value that occurs so widely it is ordinary text, not a leak."""

    fingerprint: str
    labels: tuple[str, ...]
    schema: str
    length: int
    files: int
    occurrences: int


@dataclass
class ScanResult:
    leaks: list[Leak] = field(default_factory=list)
    too_common: list[CommonMatch] = field(default_factory=list)
    secrets_total: int = 0
    secrets_scanned: int = 0
    skipped_short: int = 0
    skipped_binary: int = 0
    locked_collections: list[str] = field(default_factory=list)
    files_scanned: int = 0
    bytes_scanned: int = 0
    unattributed: int = 0


# --------------------------------------------------------------------------- #
# keyring
# --------------------------------------------------------------------------- #


def iter_keyring_secrets() -> tuple[list[KeyringSecret], list[str], int]:
    """Read every unlocked keyring item.

    Returns (secrets, locked collection labels, count of undecodable items).
    """
    try:
        import secretstorage  # pyright: ignore[reportMissingImports]  # optional 'keyring' extra
    except ImportError as exc:  # pragma: no cover - depends on the host
        raise RuntimeError(
            "reading the keyring needs the 'secretstorage' package "
            "(pip install secretstorage)"
        ) from exc

    secrets: list[KeyringSecret] = []
    locked: list[str] = []
    binary = 0
    connection = secretstorage.dbus_init()
    for collection in secretstorage.get_all_collections(connection):
        try:
            label = collection.get_label()
        except Exception:  # pragma: no cover - flaky D-Bus items
            label = "?"
        if collection.is_locked():
            locked.append(label or "(unnamed)")
            continue
        for item in collection.get_all_items():
            try:
                raw = item.get_secret()
                attributes = item.get_attributes()
                item_label = item.get_label()
            except Exception:  # pragma: no cover - flaky D-Bus items
                continue
            try:
                value = raw.decode("utf-8")
            except UnicodeDecodeError:
                binary += 1
                continue
            secrets.append(
                KeyringSecret(
                    label=item_label or "(unnamed)",
                    schema=attributes.get("xdg:schema", ""),
                    collection=label or "(unnamed)",
                    value=value,
                )
            )
    return secrets, locked, binary


def dedupe_secrets(secrets: Iterable[KeyringSecret]) -> dict[str, list[KeyringSecret]]:
    """Group keyring items by secret value: one value can have several labels."""
    groups: dict[str, list[KeyringSecret]] = {}
    for secret in secrets:
        groups.setdefault(secret.value, []).append(secret)
    return groups


# --------------------------------------------------------------------------- #
# targets
# --------------------------------------------------------------------------- #


def _record_path(record) -> Path | None:
    source = record.source
    if isinstance(source, Path):
        return source
    if isinstance(source, dict):
        rollout = source.get("rollout_path")
        if rollout:
            return Path(rollout)
    return None


def collect_targets(
    records: Iterable,
    store_dbs: Sequence[Path] | None = None,
) -> tuple[dict[Path, list], list[Path]]:
    """Map each scannable file to the records it holds, plus the store DBs.

    A session can be more than its journal: agentknit writes the message array
    it posted to the model endpoint in a sidecar file. That payload is the
    strongest evidence a secret left the machine, so it is scanned alongside
    the journal and attributed to the same session.
    """
    by_path: dict[Path, list] = {}
    for record in records:
        path = _record_path(record)
        if path is None or not path.is_file():
            continue
        by_path.setdefault(path, []).append(record)
        sidecar = tz.agent_probe_sidecar(path)
        if sidecar is not None:
            by_path.setdefault(sidecar, []).append(record)
    candidates = _STORE_DBS if store_dbs is None else store_dbs
    stores = [Path(db) for db in candidates if Path(db).is_file()]
    return by_path, stores


# --------------------------------------------------------------------------- #
# stage A: locate files containing any secret
# --------------------------------------------------------------------------- #


Matches = dict[str, dict[Path, int]]  # needle -> file -> occurrences


def _grep_matches(paths: Sequence[Path], needles: Sequence[str]) -> Matches | None:
    """Return, per needle, how often it occurs in each file it occurs in.

    One pass over the corpus for all needles at once: `grep -o` reports the
    matched text itself, which is what makes a single pass attributable to a
    specific secret. Needles go in on stdin, so no secret ever reaches argv,
    a temp file or the process table; matched text stays in this process.

    Returns None when grep cannot be used, so the caller can fall back.
    """
    patterns = "\n".join(needles) + "\n"
    wanted = set(needles)
    matches: Matches = {}
    for start in range(0, len(paths), _GREP_CHUNK):
        chunk = paths[start : start + _GREP_CHUNK]
        cmd = ["grep", "-o", "-H", "-Z", "-F", "-f", "-", "--binary-files=text", "--"]
        cmd += [str(p) for p in chunk]
        try:
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                errors="replace",
            )
        except (OSError, ValueError):
            return None
        assert proc.stdin is not None and proc.stdout is not None
        try:
            proc.stdin.write(patterns)
            proc.stdin.close()
        except OSError:
            proc.kill()
            proc.wait()
            return None
        # Streamed, not buffered: a needle that is common text can match
        # millions of times, and none of that has to be held in memory.
        for line in proc.stdout:
            path_text, _, matched = line.rstrip("\n").partition("\0")
            if not matched or matched not in wanted:
                continue
            per_file = matches.setdefault(matched, {})
            path = Path(path_text)
            per_file[path] = per_file.get(path, 0) + 1
        proc.stdout.close()
        if proc.wait() not in (0, 1, 2):
            return None
    return matches


def _iter_file_text(path: Path, overlap: int) -> Iterator[tuple[str, int]]:
    """Yield a file's text in overlapping chunks so matches can't straddle.

    Each chunk comes with the length of its replayed prefix, so a caller
    counting occurrences can skip what the previous chunk already counted.
    """
    try:
        size = path.stat().st_size
    except OSError:
        return
    try:
        with path.open("rb") as handle:
            if size <= _READ_CHUNK:
                yield handle.read().decode("utf-8", "replace"), 0
                return
            tail = b""
            while True:
                block = handle.read(_READ_CHUNK)
                if not block:
                    break
                prefix = tail.decode("utf-8", "replace")
                yield prefix + block.decode("utf-8", "replace"), len(prefix)
                tail = block[-overlap:] if overlap else b""
    except OSError:
        return


def _count_in_chunk(text: str, needle: str, skip: int) -> int:
    """Count occurrences in a chunk, ignoring ones the previous chunk counted."""
    return text.count(needle, max(0, skip - len(needle) + 1))


def _python_matches(paths: Sequence[Path], needles: Sequence[str]) -> Matches:
    """grep-free fallback: scan every file in process."""
    overlap = max((len(n) for n in needles), default=0)
    matches: Matches = {}
    for path in paths:
        for text, skip in _iter_file_text(path, overlap):
            for needle in needles:
                count = _count_in_chunk(text, needle, skip)
                if count:
                    per_file = matches.setdefault(needle, {})
                    per_file[path] = per_file.get(path, 0) + count
    return matches


# --------------------------------------------------------------------------- #
# stage B: attribute a hit file to secrets, steps and sessions
# --------------------------------------------------------------------------- #


def redact(text: str, needle: str, width: int = 30) -> str:
    """Return the context around a match with the secret itself removed."""
    index = text.find(needle)
    if index < 0:
        return ""
    before = text[max(0, index - width) : index]
    after = text[index + len(needle) : index + len(needle) + width]
    placeholder = f"«REDACTED:{len(needle)} chars»"
    snippet = f"{before}{placeholder}{after}"
    snippet = " ".join(snippet.split())
    prefix = "…" if index > width else ""
    suffix = "…" if index + len(needle) + width < len(text) else ""
    return f"{prefix}{snippet}{suffix}"


def _parse(record):
    """Parse a record, or return None when the format is unsupported."""
    try:
        return tz.parse_record(record)
    except Exception:
        return None


def _model_label(model: str) -> str:
    """Normalize a recorded model name to something readable.

    Some agents store the whole model selector rather than its name (OpenCode
    writes a JSON object), so unwrap that to the model id.
    """
    model = (model or "").strip()
    if model.startswith("{"):
        try:
            parsed = json.loads(model)
        except json.JSONDecodeError:
            return model
        if isinstance(parsed, dict):
            return str(parsed.get("id") or parsed.get("model") or model)
    return model


def destination_of(traj) -> tuple[str, str]:
    """Return (model, endpoint host) a trajectory's conversation was sent to.

    A secret in a conversation is a secret handed to whoever served that
    conversation, so the model and the host it was posted to are the finding
    that matters most. Most agents record only the model; agentknit also
    records the endpoint it posted to.
    """
    if traj is None:
        return "", ""
    model = _model_label(traj.model_name or "")
    endpoint = (traj.extra_agent or {}).get("endpoint") or ""
    host = ""
    if endpoint:
        match = re.match(r"[a-z]+://([^/]+)", endpoint)
        host = match.group(1) if match else endpoint
    return model, host


def format_destination(model: str, endpoint: str) -> str:
    """Render a destination for display: the model, and the host when known."""
    if model and endpoint:
        return f"{model} ({endpoint})"
    return model or endpoint


def _steps_containing(traj, needle: str) -> list[tuple[int, str]]:
    """Return (step_id, blob) pairs whose text contains the needle."""
    if traj is None:
        return []
    found: list[tuple[int, str]] = []
    for step in traj.steps:
        for blob in _step_search_blobs(step):
            if blob and needle in blob:
                found.append((step["step_id"], blob))
                break
    return found


def _file_context(path: Path, needle: str) -> str:
    """Read a file only to build a redacted snippet around the first match."""
    for text, _ in _iter_file_text(path, len(needle)):
        if needle in text:
            return redact(text, needle)
    return ""


def _leaks_for_file(
    path: Path,
    records: Sequence,
    secret: KeyringSecret,
    labels: tuple[str, ...],
    occurrences: int,
) -> tuple[list[Leak], int]:
    """Locate a known-present secret inside one file, per trajectory step."""
    needle = secret.needle
    leaks: list[Leak] = []
    parsed = [(record, _parse(record)) for record in records]
    for record, traj in parsed:
        model, endpoint = destination_of(traj)
        for step_id, blob in _steps_containing(traj, needle):
            leaks.append(
                Leak(
                    fingerprint=secret.fingerprint,
                    labels=labels,
                    schema=secret.schema,
                    length=len(needle),
                    agent=record.agent,
                    trajectory_id=record.id,
                    timestamp=record.timestamp,
                    step=step_id,
                    source=str(path),
                    model=model,
                    endpoint=endpoint,
                    occurrences=blob.count(needle),
                    context=redact(blob, needle),
                    partial=needle != secret.value,
                )
            )
    if leaks:
        return leaks, 0

    # Present in the file but not attributable to a parsed step (metadata
    # lines, an unparseable format, sidecar fields): still a leak.
    record, traj = parsed[0] if parsed else (None, None)
    model, endpoint = destination_of(traj)
    return [
        Leak(
            fingerprint=secret.fingerprint,
            labels=labels,
            schema=secret.schema,
            length=len(needle),
            agent=record.agent if record else "-",
            trajectory_id=record.id if record else "-",
            timestamp=record.timestamp if record else "",
            step=None,
            source=str(path),
            model=model,
            endpoint=endpoint,
            occurrences=occurrences,
            context=_file_context(path, needle),
            partial=needle != secret.value,
        )
    ], 1


_SESSION_COLUMNS = ("session_id", "sessionid", "session", "thread_id", "id")


def _iter_sqlite_text(path: Path) -> Iterator[tuple[str, tuple[str, ...], str]]:
    """Yield (table, session hints, text) for every text cell in a SQLite file.

    Hints are the row's id-like column values, most session-specific first, so
    a leak found in an arbitrary store schema can still be tied to a session.
    """
    try:
        uri = f"file:{path}?mode=ro"
        conn = sqlite3.connect(uri, uri=True)
    except sqlite3.Error:
        return
    try:
        conn.text_factory = lambda b: b.decode("utf-8", "replace")
        tables = [
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        ]
        for table in tables:
            try:
                cursor = conn.execute(f'SELECT * FROM "{table}"')
            except sqlite3.Error:
                continue
            columns = [str(d[0]).lower() for d in cursor.description or []]
            hint_idxs = [
                columns.index(name) for name in _SESSION_COLUMNS if name in columns
            ]
            while True:
                try:
                    rows = cursor.fetchmany(200)
                except sqlite3.Error:
                    break
                if not rows:
                    break
                for row in rows:
                    hints = tuple(
                        str(row[i]) for i in hint_idxs if row[i] is not None
                    )
                    for cell in row:
                        if isinstance(cell, str):
                            yield table, hints, cell
                        elif isinstance(cell, (bytes, bytearray)):
                            yield table, hints, bytes(cell).decode("utf-8", "replace")
    finally:
        conn.close()


def _leaks_for_store(
    path: Path,
    present: dict[str, tuple[KeyringSecret, tuple[str, ...]]],
    sessions: dict[str, tuple[str, str, str]],
    models: dict[str, str],
) -> list[Leak]:
    """Identify secrets inside a session store DB, attributed by session id."""
    found: dict[tuple[str, str], Leak] = {}
    for table, hints, text in _iter_sqlite_text(path):
        for needle, (secret, labels) in present.items():
            count = text.count(needle)
            if not count:
                continue
            known = next((h for h in hints if h in sessions), None)
            if known is not None:
                agent, traj_id, timestamp = sessions[known]
            else:
                agent, traj_id, timestamp = "-", (hints[0] if hints else "-"), ""
            key = (secret.fingerprint, traj_id)
            existing = found.get(key)
            if existing is not None:
                found[key] = replace(existing, occurrences=existing.occurrences + count)
                continue
            found[key] = Leak(
                fingerprint=secret.fingerprint,
                labels=labels,
                schema=secret.schema,
                length=len(needle),
                agent=agent,
                trajectory_id=traj_id,
                timestamp=timestamp,
                step=None,
                source=f"{path}:{table}",
                model=_model_label(models.get(traj_id, "")),
                endpoint="",
                occurrences=count,
                context=redact(text, needle),
                partial=needle != secret.value,
            )
    return list(found.values())


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #


def scan(
    secrets: Iterable[KeyringSecret],
    records: Iterable,
    min_length: int = DEFAULT_MIN_LENGTH,
    max_files: int = DEFAULT_MAX_FILES,
    use_grep: bool = True,
    locked_collections: Sequence[str] = (),
    skipped_binary: int = 0,
    store_dbs: Sequence[Path] | None = None,
) -> ScanResult:
    """Search every trajectory for every keyring secret."""
    secrets = list(secrets)
    result = ScanResult(
        secrets_total=len(secrets),
        locked_collections=list(locked_collections),
        skipped_binary=skipped_binary,
    )

    groups = dedupe_secrets(secrets)
    usable: dict[str, list[KeyringSecret]] = {}
    for value, group in groups.items():
        if len(group[0].needle) < max(min_length, 1):
            result.skipped_short += len(group)
            continue
        usable[value] = group
    result.secrets_scanned = sum(len(g) for g in usable.values())
    if not usable:
        return result

    # One keyring value can be stored under several labels, and two values can
    # even share a needle (same longest line); key the work by needle.
    by_needle: dict[str, tuple[KeyringSecret, tuple[str, ...]]] = {}
    for value, group in usable.items():
        needle = group[0].needle
        secret, labels = by_needle.get(needle, (group[0], ()))
        by_needle[needle] = (
            secret,
            tuple(sorted(set(labels) | {item.label for item in group})),
        )

    records = list(records)
    by_path, stores = collect_targets(records, store_dbs)
    sessions = {
        str(r.source.get("session_id")): (r.agent, r.id, r.timestamp)
        for r in records
        if isinstance(r.source, dict) and r.source.get("session_id") is not None
    }
    db_models = {
        r.id: str(r.source.get("model") or "")
        for r in records
        if isinstance(r.source, dict) and r.source.get("model")
    }

    all_paths = sorted(by_path) + [db for db in stores if db not in by_path]
    result.files_scanned = len(all_paths)
    result.bytes_scanned = sum(p.stat().st_size for p in all_paths if p.exists())

    needles = list(by_needle)
    matches = _grep_matches(all_paths, needles) if use_grep else None
    if matches is None:
        matches = _python_matches(all_paths, needles)

    store_set = set(stores)
    stores_to_inspect: dict[Path, dict[str, tuple[KeyringSecret, tuple[str, ...]]]] = {}

    for needle, (secret, labels) in by_needle.items():
        per_file = matches.get(needle) or {}
        if not per_file:
            continue
        if max_files and len(per_file) > max_files:
            # A keyring value that occurs in this many distinct trajectories is
            # ordinary text (a dictionary-word password, a common path), not a
            # leaked credential. Report it once instead of drowning the report.
            result.too_common.append(
                CommonMatch(
                    fingerprint=secret.fingerprint,
                    labels=labels,
                    schema=secret.schema,
                    length=len(needle),
                    files=len(per_file),
                    occurrences=sum(per_file.values()),
                )
            )
            continue
        for path, occurrences in sorted(per_file.items()):
            if path in store_set:
                stores_to_inspect.setdefault(path, {})[needle] = (secret, labels)
                continue
            leaks, unattributed = _leaks_for_file(
                path, by_path.get(path, []), secret, labels, occurrences
            )
            result.leaks += leaks
            result.unattributed += unattributed

    for path, present in stores_to_inspect.items():
        result.leaks += _leaks_for_store(path, present, sessions, db_models)

    result.leaks.sort(key=lambda leak: (leak.labels, leak.timestamp, leak.trajectory_id))
    result.too_common.sort(key=lambda common: -common.files)
    return result


def leaks_to_json(result: ScanResult) -> dict:
    """Serialize a scan, by fingerprint only -- never a secret value."""
    return {
        "summary": {
            "secrets_total": result.secrets_total,
            "secrets_scanned": result.secrets_scanned,
            "skipped_short": result.skipped_short,
            "skipped_binary": result.skipped_binary,
            "locked_collections": result.locked_collections,
            "files_scanned": result.files_scanned,
            "bytes_scanned": result.bytes_scanned,
            "leaked_secrets": len({leak.fingerprint for leak in result.leaks}),
            "affected_trajectories": len({leak.trajectory_id for leak in result.leaks}),
            "destinations": sorted(
                {leak.destination for leak in result.leaks if leak.destination}
            ),
            "too_common": len(result.too_common),
            "unattributed_to_a_step": result.unattributed,
        },
        "too_common": [
            {
                "fingerprint": common.fingerprint,
                "keyring_labels": list(common.labels),
                "schema": common.schema,
                "length": common.length,
                "files": common.files,
                "occurrences": common.occurrences,
            }
            for common in result.too_common
        ],
        "leaks": [
            {
                "fingerprint": leak.fingerprint,
                "keyring_labels": list(leak.labels),
                "schema": leak.schema,
                "length": leak.length,
                "agent": leak.agent,
                "trajectory_id": leak.trajectory_id,
                "timestamp": leak.timestamp,
                "step": leak.step,
                "source": leak.source,
                "model": leak.model,
                "endpoint": leak.endpoint,
                "occurrences": leak.occurrences,
                "context": leak.context,
                "partial_match": leak.partial,
            }
            for leak in result.leaks
        ],
    }
