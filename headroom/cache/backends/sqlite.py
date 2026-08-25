"""SQLite storage backend for CompressionStore.

Default backend for the CCR store. Two properties the in-memory backend
cannot provide, both load-bearing for the no-accuracy-loss guarantee:

- **Restart survival.** A proxy restart no longer destroys every
  retrievable original mid-session. With the session-scale 30-minute
  TTL, entries are expected to outlive any single process.
- **Multi-worker sharing.** The database file (WAL mode) is shared
  across worker processes, so a `headroom_retrieve` call served by a
  different worker than the one that compressed still finds the entry.
  This closes the largest of the documented multi-worker gaps.

Set ``HEADROOM_CCR_BACKEND=memory`` to opt back into the in-memory
backend, or ``HEADROOM_CCR_SQLITE_PATH`` to relocate the database file
(default ``workspace_dir()/ccr_store.db``).
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import sqlite3
import threading
import time
from dataclasses import asdict, fields, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..compression_store import CompressionEntry

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS ccr_entries (
    hash TEXT PRIMARY KEY,
    entry_json TEXT NOT NULL,
    created_at REAL NOT NULL,
    ttl INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_ccr_expiry ON ccr_entries (created_at);
CREATE TABLE IF NOT EXISTS ccr_discarded_events (
    compression_event_id TEXT PRIMARY KEY,
    expires_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_ccr_discarded_expiry
    ON ccr_discarded_events (expires_at);
"""

# A tombstone outlives the default CCR entry long enough to reject a stale
# post-discard write. Entries with a longer configured TTL extend it.
_DISCARD_TOMBSTONE_TTL_SECONDS = 1800

# Purge expired rows at most this often (seconds). Purging is hygiene,
# not correctness — CompressionStore checks TTL on every get().
_PURGE_INTERVAL = 60.0


def default_db_path() -> Path:
    """Resolve the database path (env override, else workspace root)."""
    env = os.environ.get("HEADROOM_CCR_SQLITE_PATH", "").strip()
    if env:
        return Path(env).expanduser()
    from ...paths import workspace_dir

    return workspace_dir() / "ccr_store.db"


class SQLiteBackend:
    """Thread-safe SQLite storage backend (WAL mode).

    Entries are serialized as one JSON blob per row; ``created_at`` and
    ``ttl`` are duplicated into columns so expired rows can be purged
    with one DELETE. TTL *enforcement* on reads stays in
    CompressionStore, matching the backend protocol contract.

    Deserialization is field-filtered: unknown keys in stored JSON are
    dropped (forward-compatible with newer versions that add fields).
    Missing keys load cleanly only when the corresponding
    ``CompressionEntry`` field has a default; a blob missing a required
    field (one without a default) raises ``TypeError`` on construction.
    """

    # CompressionStore must not pre-merge a stale read: ``set`` performs the
    # merge under BEGIN IMMEDIATE and is the cross-worker source of truth.
    merges_attribution_candidates_atomically = True

    def __init__(self, db_path: str | Path | None = None) -> None:
        self._path = Path(db_path).expanduser() if db_path else default_db_path()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._last_purge = 0.0
        self._conn = self._open()

    def _open(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._path, check_same_thread=False)
        # Wait for competing writers instead of failing with SQLITE_BUSY —
        # multiple proxy workers share this file, and writes are frequent
        # but tiny, so contention resolves in milliseconds.
        conn.execute("PRAGMA busy_timeout=5000")
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.executescript(_SCHEMA)
        # Startup hygiene: expired rows are only purged opportunistically
        # on writes, so a quiet store could otherwise hold expired
        # originals (which may contain sensitive tool output) on disk
        # indefinitely. Sweep them on every open.
        now = time.time()
        conn.execute(
            "DELETE FROM ccr_entries WHERE created_at + ttl < ?",
            (now,),
        )
        conn.execute(
            "DELETE FROM ccr_discarded_events WHERE expires_at < ?",
            (now,),
        )
        conn.commit()
        # Originals can contain sensitive tool output (file contents,
        # command output) — keep the database private to the user.
        for suffix in ("", "-wal", "-shm"):
            p = Path(str(self._path) + suffix)
            if p.exists():
                try:
                    p.chmod(0o600)
                except OSError:
                    pass
        return conn

    @staticmethod
    def _is_corruption(error: Exception) -> bool:
        """Only genuine file corruption justifies recreating the database.

        ``sqlite3.OperationalError`` (a DatabaseError subclass) also covers
        transient conditions like ``database is locked`` under multi-worker
        write contention — misclassifying those as corruption would delete
        live data while sibling workers still hold handles to the unlinked
        inode (split-brain). Match the corruption messages explicitly.
        """
        msg = str(error).lower()
        return "malformed" in msg or "not a database" in msg

    def _handle_db_error(self, error: sqlite3.DatabaseError, op: str) -> None:
        """Corruption → recreate (loud). Anything else (busy/locked/io) →
        log and treat the operation as a miss; never destroy data over a
        transient error."""
        if not self._is_corruption(error):
            logger.warning("CCR SQLite %s failed (transient, no reset): %s", op, error)
            return
        logger.warning(
            "CCR SQLite store at %s is corrupt (%s); recreating. "
            "Previously stored originals are lost — affected retrieval "
            "markers will miss until their content is re-compressed.",
            self._path,
            error,
        )
        try:
            self._conn.close()
        except Exception:  # noqa: BLE001 - best-effort close on corrupt handle
            pass
        self._path.unlink(missing_ok=True)
        self._conn = self._open()

    def _entry_from_json(self, raw: str) -> CompressionEntry | None:
        from ..compression_store import CompressionEntry

        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return None
        known = {f.name for f in fields(CompressionEntry)}
        return CompressionEntry(**{k: v for k, v in data.items() if k in known})

    def _maybe_purge(self) -> None:
        """Delete expired rows; called opportunistically under the lock."""
        now = time.time()
        if now - self._last_purge < _PURGE_INTERVAL:
            return
        self._last_purge = now
        self._conn.execute(
            "DELETE FROM ccr_entries WHERE created_at + ttl < ?",
            (now,),
        )
        self._conn.execute(
            "DELETE FROM ccr_discarded_events WHERE expires_at < ?",
            (now,),
        )
        self._conn.commit()

    def get(self, hash_key: str) -> CompressionEntry | None:
        with self._lock:
            try:
                row = self._conn.execute(
                    "SELECT entry_json FROM ccr_entries WHERE hash = ?",
                    (hash_key,),
                ).fetchone()
            except sqlite3.DatabaseError as e:
                self._handle_db_error(e, "get")
                return None
        if row is None:
            return None
        return self._entry_from_json(row[0])

    def set(self, hash_key: str, entry: CompressionEntry) -> None:
        with self._lock:
            try:
                # Serialize same-hash attribution merging across workers. A
                # process-local CompressionStore lock cannot protect two live
                # SQLite connections from a stale read followed by overwrite.
                self._conn.execute("BEGIN IMMEDIATE")
                from ..compression_store import (
                    MAX_ATTRIBUTION_CANDIDATES,
                    _attribution_candidates,
                    _candidate_identity,
                    _entry_attribution_candidate,
                    _legacy_unattributed_candidate,
                )

                # Persisted tombstones distinguish a genuinely new delayed
                # compression event from a stale snapshot written after that
                # event was rejected by another process. Filter every carried
                # candidate before considering the same-hash merge.
                incoming_candidates = _attribution_candidates(entry)
                now = time.time()
                filtered_candidates: list[dict[str, str]] = []
                for candidate in incoming_candidates:
                    event_id = candidate.get("compression_event_id", "")
                    tombstoned = False
                    if event_id:
                        tombstoned = (
                            self._conn.execute(
                                "SELECT 1 FROM ccr_discarded_events "
                                "WHERE compression_event_id = ? AND expires_at >= ?",
                                (event_id, now),
                            ).fetchone()
                            is not None
                        )
                    if not tombstoned:
                        filtered_candidates.append(candidate)
                if incoming_candidates and not filtered_candidates:
                    self._conn.commit()
                    return
                if len(filtered_candidates) != len(incoming_candidates):
                    preserved = filtered_candidates[-1]
                    entry = replace(
                        entry,
                        attribution_candidates=[dict(item) for item in filtered_candidates],
                        compression_event_id=preserved["compression_event_id"] or None,
                        retrieval_handle=preserved["retrieval_handle"] or None,
                        session_id=preserved["session_id"] or None,
                        request_id=preserved["request_id"] or None,
                        tool_call_id=preserved["tool_call_id"] or None,
                        provider_slot=preserved["provider_slot"] or None,
                        tool_name=preserved["tool_name"] or None,
                        compression_strategy=preserved["compression_strategy"] or None,
                        tool_signature_hash=preserved["tool_signature_hash"] or None,
                        attribution_kind=preserved["attribution_kind"] or None,
                    )

                row = self._conn.execute(
                    "SELECT entry_json FROM ccr_entries WHERE hash = ?",
                    (hash_key,),
                ).fetchone()
                if row is not None:
                    existing = self._entry_from_json(row[0])
                    if existing is not None and existing.original_content == entry.original_content:
                        # A store operation contributes only its top-level
                        # candidate. Never union every candidate carried by the
                        # incoming object: it may be a stale pre-discard read
                        # from another worker.
                        merged = _attribution_candidates(existing)
                        if not merged:
                            merged = [_legacy_unattributed_candidate()]
                        incoming_candidates = _attribution_candidates(entry)
                        current_candidate = _entry_attribution_candidate(entry)
                        accept_current_candidate = len(incoming_candidates) <= 1
                        if accept_current_candidate and any(current_candidate.values()):
                            identity = _candidate_identity(current_candidate)
                            merged = [
                                item for item in merged if _candidate_identity(item) != identity
                            ]
                            merged.append(current_candidate)
                        entry = replace(
                            entry,
                            attribution_candidates=merged[-MAX_ATTRIBUTION_CANDIDATES:],
                            retrieval_count=existing.retrieval_count,
                            search_queries=list(existing.search_queries),
                            last_accessed=existing.last_accessed,
                        )
                        if (
                            not accept_current_candidate or not any(current_candidate.values())
                        ) and merged:
                            preserved = merged[-1]
                            entry = replace(
                                entry,
                                compression_event_id=preserved["compression_event_id"] or None,
                                retrieval_handle=preserved["retrieval_handle"] or None,
                                session_id=preserved["session_id"] or None,
                                request_id=preserved["request_id"] or None,
                                tool_call_id=preserved["tool_call_id"] or None,
                                provider_slot=preserved["provider_slot"] or None,
                                tool_name=preserved["tool_name"] or None,
                                compression_strategy=preserved["compression_strategy"] or None,
                                tool_signature_hash=preserved["tool_signature_hash"] or None,
                                attribution_kind=preserved["attribution_kind"] or None,
                            )
                payload = json.dumps(asdict(entry), ensure_ascii=False)
                self._conn.execute(
                    "INSERT OR REPLACE INTO ccr_entries "
                    "(hash, entry_json, created_at, ttl) VALUES (?, ?, ?, ?)",
                    (hash_key, payload, entry.created_at, entry.ttl),
                )
                self._conn.commit()
                self._maybe_purge()
            except sqlite3.DatabaseError as e:
                with contextlib.suppress(sqlite3.DatabaseError):
                    self._conn.rollback()
                self._handle_db_error(e, "set")
            except Exception:
                # A malformed legacy entry can fail while it is being decoded
                # after BEGIN IMMEDIATE. Never leave that write transaction
                # open if the non-database exception must propagate.
                with contextlib.suppress(sqlite3.DatabaseError):
                    self._conn.rollback()
                raise

    def retrieve_and_record_access(
        self,
        hash_key: str,
        query: str | None = None,
        *,
        compression_event_id: str | None = None,
        retrieval_handle: str | None = None,
        session_id: str | None = None,
        request_id: str | None = None,
        tool_call_id: str | None = None,
        provider_slot: str | None = None,
    ) -> tuple[str, CompressionEntry | None]:
        """Resolve attribution and increment feedback in one transaction."""
        with self._lock:
            try:
                self._conn.execute("BEGIN IMMEDIATE")
                row = self._conn.execute(
                    "SELECT entry_json FROM ccr_entries WHERE hash = ?",
                    (hash_key,),
                ).fetchone()
                if row is None:
                    self._conn.commit()
                    return "missing", None
                entry = self._entry_from_json(row[0])
                if entry is None:
                    self._conn.commit()
                    return "unresolved", None
                if entry.is_expired():
                    self._conn.execute("DELETE FROM ccr_entries WHERE hash = ?", (hash_key,))
                    self._conn.commit()
                    return "expired", None

                from ..compression_store import (
                    _entry_attribution_resolution_status,
                    _entry_with_resolved_attribution,
                )

                status = _entry_attribution_resolution_status(
                    entry,
                    compression_event_id=compression_event_id,
                    retrieval_handle=retrieval_handle,
                    session_id=session_id,
                    request_id=request_id,
                    tool_call_id=tool_call_id,
                    provider_slot=provider_slot,
                )
                if status != "available":
                    self._conn.commit()
                    return status, None

                entry.record_access(query)
                payload = json.dumps(asdict(entry), ensure_ascii=False)
                self._conn.execute(
                    "UPDATE ccr_entries SET entry_json = ? WHERE hash = ?",
                    (payload, hash_key),
                )
                self._conn.commit()
                resolved = _entry_with_resolved_attribution(
                    entry,
                    compression_event_id=compression_event_id,
                    retrieval_handle=retrieval_handle,
                    session_id=session_id,
                    request_id=request_id,
                    tool_call_id=tool_call_id,
                    provider_slot=provider_slot,
                )
                return "available", resolved
            except sqlite3.DatabaseError as e:
                with contextlib.suppress(sqlite3.DatabaseError):
                    self._conn.rollback()
                self._handle_db_error(e, "retrieve and record access")
                return "missing", None
            except Exception:
                with contextlib.suppress(sqlite3.DatabaseError):
                    self._conn.rollback()
                raise

    def discard_attribution_candidate(self, hash_key: str, compression_event_id: str) -> bool:
        """Atomically remove one never-emitted producer from a shared row."""
        with self._lock:
            try:
                self._conn.execute("BEGIN IMMEDIATE")
                row = self._conn.execute(
                    "SELECT entry_json FROM ccr_entries WHERE hash = ?",
                    (hash_key,),
                ).fetchone()
                entry = self._entry_from_json(row[0]) if row is not None else None
                retention = max(
                    _DISCARD_TOMBSTONE_TTL_SECONDS,
                    int(entry.ttl) if entry is not None else 0,
                )
                self._conn.execute(
                    "INSERT INTO ccr_discarded_events (compression_event_id, expires_at) "
                    "VALUES (?, ?) ON CONFLICT(compression_event_id) DO UPDATE SET "
                    "expires_at = MAX(expires_at, excluded.expires_at)",
                    (compression_event_id, time.time() + retention),
                )
                if entry is None:
                    self._conn.commit()
                    return False
                from ..compression_store import _entry_without_compression_event

                replacement, changed = _entry_without_compression_event(entry, compression_event_id)
                if not changed:
                    self._conn.commit()
                    return False
                if replacement is None:
                    self._conn.execute("DELETE FROM ccr_entries WHERE hash = ?", (hash_key,))
                else:
                    payload = json.dumps(asdict(replacement), ensure_ascii=False)
                    self._conn.execute(
                        "UPDATE ccr_entries SET entry_json = ?, created_at = ?, ttl = ? "
                        "WHERE hash = ?",
                        (payload, replacement.created_at, replacement.ttl, hash_key),
                    )
                self._conn.commit()
                return True
            except sqlite3.DatabaseError as e:
                with contextlib.suppress(sqlite3.DatabaseError):
                    self._conn.rollback()
                self._handle_db_error(e, "discard attribution candidate")
                return False
            except Exception:
                with contextlib.suppress(sqlite3.DatabaseError):
                    self._conn.rollback()
                raise

    def delete(self, hash_key: str) -> bool:
        with self._lock:
            try:
                cur = self._conn.execute(
                    "DELETE FROM ccr_entries WHERE hash = ?",
                    (hash_key,),
                )
                self._conn.commit()
                return cur.rowcount > 0
            except sqlite3.DatabaseError as e:
                self._handle_db_error(e, "op")
                return False

    def exists(self, hash_key: str) -> bool:
        with self._lock:
            try:
                row = self._conn.execute(
                    "SELECT 1 FROM ccr_entries WHERE hash = ?",
                    (hash_key,),
                ).fetchone()
            except sqlite3.DatabaseError as e:
                self._handle_db_error(e, "op")
                return False
        return row is not None

    def clear(self) -> None:
        with self._lock:
            try:
                self._conn.execute("DELETE FROM ccr_entries")
                self._conn.execute("DELETE FROM ccr_discarded_events")
                self._conn.commit()
            except sqlite3.DatabaseError as e:
                self._handle_db_error(e, "op")

    def count(self) -> int:
        with self._lock:
            try:
                row = self._conn.execute("SELECT COUNT(*) FROM ccr_entries").fetchone()
            except sqlite3.DatabaseError as e:
                self._handle_db_error(e, "op")
                return 0
        return int(row[0])

    def keys(self) -> list[str]:
        with self._lock:
            try:
                rows = self._conn.execute("SELECT hash FROM ccr_entries").fetchall()
            except sqlite3.DatabaseError as e:
                self._handle_db_error(e, "op")
                return []
        return [r[0] for r in rows]

    def items(self) -> list[tuple[str, CompressionEntry]]:
        with self._lock:
            try:
                rows = self._conn.execute("SELECT hash, entry_json FROM ccr_entries").fetchall()
            except sqlite3.DatabaseError as e:
                self._handle_db_error(e, "op")
                return []
        out: list[tuple[str, CompressionEntry]] = []
        for hash_key, raw in rows:
            entry = self._entry_from_json(raw)
            if entry is not None:
                out.append((hash_key, entry))
        return out

    def get_stats(self) -> dict[str, Any]:
        with self._lock:
            try:
                count_row = self._conn.execute("SELECT COUNT(*) FROM ccr_entries").fetchone()
                tombstone_row = self._conn.execute(
                    "SELECT COUNT(*) FROM ccr_discarded_events WHERE expires_at >= ?",
                    (time.time(),),
                ).fetchone()
            except sqlite3.DatabaseError as e:
                self._handle_db_error(e, "op")
                count_row = (0,)
                tombstone_row = (0,)
        try:
            bytes_used = self._path.stat().st_size
        except OSError:
            bytes_used = 0
        return {
            "backend_type": "sqlite",
            "entry_count": int(count_row[0]),
            "discard_tombstone_count": int(tombstone_row[0]),
            "bytes_used": bytes_used,
            "db_path": str(self._path),
        }
