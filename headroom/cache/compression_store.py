"""Compression Store for CCR (Compress-Cache-Retrieve) architecture.

This module implements reversible compression: when SmartCrusher compresses
tool outputs, the original data is cached here for on-demand retrieval.

Key insight from research: REVERSIBLE compression beats irreversible compression.
If the LLM needs data that was compressed away, it can retrieve it instantly.

Features:
- Thread-safe in-memory storage with TTL expiration
- BM25-based search within cached content
- Retrieval event tracking for feedback loop
- Automatic eviction when capacity is reached

Usage:
    store = get_compression_store()

    # Store compressed content
    hash_key = store.store(
        original=original_json,
        compressed=compressed_json,
        original_tokens=1000,
        compressed_tokens=100,
        tool_name="search_api",
    )

    # Retrieve later (by hash; always returns the full original content)
    entry = store.retrieve(hash_key)
"""

from __future__ import annotations

import hashlib
import heapq
import json
import logging
import os
import re
import threading
import time
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..memory.tracker import ComponentStats
    from .backends import CompressionStoreBackend

logger = logging.getLogger(__name__)

DEFAULT_CCR_TTL_SECONDS = 1800  # session-scale; override via HEADROOM_CCR_TTL_SECONDS
CCR_TTL_SECONDS_ENV = "HEADROOM_CCR_TTL_SECONDS"
MAX_ATTRIBUTION_CANDIDATES = 32

_ATTRIBUTION_FIELDS = (
    "compression_event_id",
    "retrieval_handle",
    "session_id",
    "request_id",
    "tool_call_id",
    "provider_slot",
    "tool_name",
    "compression_strategy",
    "tool_signature_hash",
    "attribution_kind",
)

_RETRIEVAL_LOG_PREVIEW_CHARS = 4096
# Previews carry verbatim tool-result content (post-redaction), which makes
# proxy.log too sensitive for users to share in bug reports. Set to
# 0/false/no/off to log byte counts only.
PAYLOAD_PREVIEW_ENV = "HEADROOM_LOG_PAYLOAD_PREVIEW"
_SECRET_KEY_VALUE_RE = re.compile(
    r"(?i)\b([A-Z0-9_-]*(?:API[_-]?KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL|AUTH)[A-Z0-9_-]*)"
    r"(\s*[:=]\s*)([\"']?)([^\"'\s,}]+)"
)
_AUTH_VALUE_RE = re.compile(r"(?i)\b(Bearer|Basic)\s+[A-Za-z0-9._~+/=-]{12,}")
_API_KEY_VALUE_RE = re.compile(r"\bsk-[A-Za-z0-9_-]{12,}\b")


def _get_env_default_ttl_seconds() -> int:
    raw_value = os.environ.get(CCR_TTL_SECONDS_ENV)
    if raw_value is None or not raw_value.strip():
        return DEFAULT_CCR_TTL_SECONDS

    try:
        ttl_seconds = int(raw_value)
    except ValueError:
        logger.warning(
            "%s must be a positive integer number of seconds, got %r; using %s",
            CCR_TTL_SECONDS_ENV,
            raw_value,
            DEFAULT_CCR_TTL_SECONDS,
        )
        return DEFAULT_CCR_TTL_SECONDS

    if ttl_seconds <= 0:
        logger.warning(
            "%s must be greater than 0, got %s; using %s",
            CCR_TTL_SECONDS_ENV,
            ttl_seconds,
            DEFAULT_CCR_TTL_SECONDS,
        )
        return DEFAULT_CCR_TTL_SECONDS

    return ttl_seconds


def format_retrieval_miss_detail(status: dict[str, Any]) -> str:
    """Return an operator-facing miss reason for CCR retrieval failures."""
    default_ttl = status.get("default_ttl_seconds", DEFAULT_CCR_TTL_SECONDS)
    ttl_seconds = status.get("ttl_seconds", default_ttl)

    if status.get("status") == "expired":
        age_seconds = status.get("age_seconds")
        if isinstance(age_seconds, (int, float)):
            return f"Entry expired (CCR TTL: {ttl_seconds} seconds; age: {age_seconds:.0f} seconds)"
        return f"Entry expired (CCR TTL: {ttl_seconds} seconds)"

    return f"Entry not found (CCR TTL: {default_ttl} seconds)"


def _redact_retrieval_log_payload(payload: str) -> str:
    redacted = _SECRET_KEY_VALUE_RE.sub(r"\1\2\3[REDACTED]", payload)
    redacted = _AUTH_VALUE_RE.sub(r"\1 [REDACTED]", redacted)
    return _API_KEY_VALUE_RE.sub("sk-[REDACTED]", redacted)


def _payload_preview_enabled() -> bool:
    raw = os.environ.get(PAYLOAD_PREVIEW_ENV)
    if raw is None:
        return True
    return raw.strip().lower() not in ("0", "false", "no", "off")


def _payload_for_retrieval_log(payload: str) -> dict[str, Any]:
    if not _payload_preview_enabled():
        return {
            "payload_chars": len(payload),
            "payload_preview_chars": 0,
            "payload_truncated": len(payload) > 0,
            "payload_preview": "",
        }
    redacted = _redact_retrieval_log_payload(payload)
    preview = redacted[:_RETRIEVAL_LOG_PREVIEW_CHARS]
    truncated = len(redacted) > len(preview)
    return {
        "payload_chars": len(payload),
        "payload_preview_chars": len(preview),
        "payload_truncated": truncated,
        "payload_preview": preview,
    }


# Single source of truth for the retrieval-miss message. Actionable by
# design: the model still has the marker in context (Read markers carry
# the file path), so tell it how to recover instead of just reporting
# the miss.
CCR_MISS_MESSAGE = (
    "Entry not found or expired. To recover: if the compression marker "
    "references a file Read, re-read that file (the path is in the "
    "marker; disk is the source of truth). If it was command output, "
    "re-run the command. Entries expire after the store TTL "
    "(default 30 minutes; configurable via HEADROOM_CCR_TTL_SECONDS)."
)


@dataclass
class CompressionEntry:
    """A cached compression entry with metadata for retrieval and feedback."""

    hash: str
    original_content: str
    compressed_content: str
    original_tokens: int
    compressed_tokens: int
    original_item_count: int
    compressed_item_count: int
    tool_name: str | None
    tool_call_id: str | None
    query_context: str | None
    created_at: float
    session_id: str | None = None
    request_id: str | None = None
    compression_event_id: str | None = None
    retrieval_handle: str | None = None
    provider_slot: str | None = None
    attribution_kind: str | None = None
    ttl: int = DEFAULT_CCR_TTL_SECONDS

    # One content-derived hash can be emitted by multiple compression events.
    # Persist their attribution so duplicate stores cannot silently redirect a
    # later recovery report to whichever event happened to write last.
    attribution_candidates: list[dict[str, str]] = field(default_factory=list)

    # TOIN integration: Store the tool signature hash for retrieval correlation
    # This MUST match the hash used by SmartCrusher when recording compression
    tool_signature_hash: str | None = None
    compression_strategy: str | None = None  # Strategy used for compression

    # Feedback tracking
    retrieval_count: int = 0
    search_queries: list[str] = field(default_factory=list)
    last_accessed: float | None = None

    def is_expired(self) -> bool:
        """Check if this entry has expired."""
        return time.time() - self.created_at > self.ttl

    def record_access(self, query: str | None = None) -> None:
        """Record an access to this entry for feedback tracking."""
        self.retrieval_count += 1
        self.last_accessed = time.time()
        if query and query not in self.search_queries:
            self.search_queries.append(query)
            # Keep only last 10 queries
            if len(self.search_queries) > 10:
                self.search_queries = self.search_queries[-10:]


def _entry_attribution_candidate(entry: CompressionEntry) -> dict[str, str]:
    return {
        field_name: str(getattr(entry, field_name, "") or "") for field_name in _ATTRIBUTION_FIELDS
    }


def _normalize_attribution_candidate(candidate: Any) -> dict[str, str]:
    if not isinstance(candidate, dict):
        return {}
    normalized = {
        field_name: str(candidate.get(field_name, "") or "") for field_name in _ATTRIBUTION_FIELDS
    }
    return normalized if any(normalized.values()) else {}


def _legacy_unattributed_candidate() -> dict[str, str]:
    candidate = dict.fromkeys(_ATTRIBUTION_FIELDS, "")
    candidate["attribution_kind"] = "legacy_unattributed"
    return candidate


def _candidate_identity(candidate: dict[str, str]) -> tuple[str, ...]:
    event_id = candidate.get("compression_event_id", "")
    if event_id:
        return ("event", event_id)
    return ("metadata", *(candidate.get(field_name, "") for field_name in _ATTRIBUTION_FIELDS))


def _attribution_candidates(entry: CompressionEntry) -> list[dict[str, str]]:
    candidates: list[dict[str, str]] = []
    seen: set[tuple[str, ...]] = set()
    raw_candidates = getattr(entry, "attribution_candidates", None) or []
    for raw_candidate in [*raw_candidates, _entry_attribution_candidate(entry)]:
        candidate = _normalize_attribution_candidate(raw_candidate)
        if not candidate:
            continue
        identity = _candidate_identity(candidate)
        if identity in seen:
            continue
        seen.add(identity)
        candidates.append(candidate)
    return candidates[-MAX_ATTRIBUTION_CANDIDATES:]


def resolve_entry_attribution(
    entry: CompressionEntry,
    *,
    compression_event_id: str | None = None,
    retrieval_handle: str | None = None,
    session_id: str | None = None,
    request_id: str | None = None,
    tool_call_id: str | None = None,
    provider_slot: str | None = None,
) -> dict[str, str]:
    """Resolve one persisted attribution candidate or return blank metadata.

    Hash-only retrieval cannot distinguish two compression events that emitted
    identical content. Blank fields let the ledger reject that ambiguity
    explicitly instead of charging whichever event stored the hash last.
    """

    filters = {
        "compression_event_id": str(compression_event_id or ""),
        "retrieval_handle": str(retrieval_handle or ""),
        "session_id": str(session_id or ""),
        "request_id": str(request_id or ""),
        "tool_call_id": str(tool_call_id or ""),
        "provider_slot": str(provider_slot or ""),
    }
    candidates = _attribution_candidates(entry)
    for field_name, expected in filters.items():
        if expected:
            candidates = [item for item in candidates if item.get(field_name) == expected]
    if len(candidates) == 1:
        return candidates[0]
    return dict.fromkeys(_ATTRIBUTION_FIELDS, "")


def _entry_attribution_resolution_status(
    entry: CompressionEntry,
    *,
    compression_event_id: str | None = None,
    retrieval_handle: str | None = None,
    session_id: str | None = None,
    request_id: str | None = None,
    tool_call_id: str | None = None,
    provider_slot: str | None = None,
) -> str:
    """Classify whether a retrieval reference identifies one producer.

    Old entries without event attribution remain retrievable. Once attribution
    candidates exist, however, content must not leave the store unless the
    supplied selectors resolve exactly one candidate. In particular, returning
    content for an ambiguous legacy hash would make model-visible recovery
    impossible to charge to the correct compression event.
    """
    candidates = _attribution_candidates(entry)
    if not candidates:
        return "available"
    filters = {
        "compression_event_id": str(compression_event_id or ""),
        "retrieval_handle": str(retrieval_handle or ""),
        "session_id": str(session_id or ""),
        "request_id": str(request_id or ""),
        "tool_call_id": str(tool_call_id or ""),
        "provider_slot": str(provider_slot or ""),
    }
    for field_name, expected in filters.items():
        if expected:
            candidates = [item for item in candidates if item.get(field_name) == expected]
    if len(candidates) == 1:
        return "available"
    return "ambiguous" if len(candidates) > 1 else "unresolved"


def _entry_with_resolved_attribution(
    entry: CompressionEntry,
    *,
    compression_event_id: str | None = None,
    retrieval_handle: str | None = None,
    session_id: str | None = None,
    request_id: str | None = None,
    tool_call_id: str | None = None,
    provider_slot: str | None = None,
) -> CompressionEntry:
    attribution = resolve_entry_attribution(
        entry,
        compression_event_id=compression_event_id,
        retrieval_handle=retrieval_handle,
        session_id=session_id,
        request_id=request_id,
        tool_call_id=tool_call_id,
        provider_slot=provider_slot,
    )
    resolved_candidates = (
        [dict(attribution)] if any(attribution.values()) else _attribution_candidates(entry)
    )
    return replace(
        entry,
        compression_event_id=attribution["compression_event_id"] or None,
        retrieval_handle=attribution["retrieval_handle"] or None,
        session_id=attribution["session_id"] or None,
        request_id=attribution["request_id"] or None,
        tool_call_id=attribution["tool_call_id"] or None,
        provider_slot=attribution["provider_slot"] or None,
        tool_name=attribution["tool_name"] or None,
        compression_strategy=attribution["compression_strategy"] or None,
        tool_signature_hash=attribution["tool_signature_hash"] or None,
        attribution_kind=attribution["attribution_kind"] or None,
        search_queries=list(entry.search_queries),
        attribution_candidates=[dict(item) for item in resolved_candidates],
    )


def _entry_without_compression_event(
    entry: CompressionEntry, compression_event_id: str
) -> tuple[CompressionEntry | None, bool]:
    """Remove one producer from a same-content entry.

    Returns ``(replacement, changed)``. ``replacement`` is ``None`` when the
    removed producer was the only reason the content existed.
    """
    candidates = _attribution_candidates(entry)
    remaining = [
        candidate
        for candidate in candidates
        if candidate.get("compression_event_id") != compression_event_id
    ]
    if len(remaining) == len(candidates):
        return entry, False
    if not remaining:
        return None, True
    attribution = remaining[-1]
    return (
        replace(
            entry,
            compression_event_id=attribution["compression_event_id"] or None,
            retrieval_handle=attribution["retrieval_handle"] or None,
            session_id=attribution["session_id"] or None,
            request_id=attribution["request_id"] or None,
            tool_call_id=attribution["tool_call_id"] or None,
            provider_slot=attribution["provider_slot"] or None,
            tool_name=attribution["tool_name"] or None,
            compression_strategy=attribution["compression_strategy"] or None,
            tool_signature_hash=attribution["tool_signature_hash"] or None,
            attribution_kind=attribution["attribution_kind"] or None,
            attribution_candidates=[dict(item) for item in remaining],
        ),
        True,
    )


@dataclass
class RetrievalEvent:
    """Event logged when content is retrieved from cache."""

    hash: str
    query: str | None
    items_retrieved: int
    total_items: int
    tool_name: str | None
    timestamp: float
    retrieval_type: str  # always "full" (retrieval is by hash)
    tool_signature_hash: str | None = None  # For TOIN correlation


class CompressionStore:
    """Thread-safe store for compressed content with retrieval support.

    This is the core of the CCR architecture. When SmartCrusher compresses
    an array, the original content is stored here. If the LLM needs more
    data, it can retrieve from this cache instantly.

    Design principles:
    - Zero external dependencies (pure Python)
    - Thread-safe for concurrent access
    - TTL-based expiration (default 300 seconds, env-configurable)
    - LRU-style eviction when capacity is reached
    - Hash-keyed retrieval that always returns the full original content
    """

    def __init__(
        self,
        max_entries: int = 1000,
        default_ttl: int = DEFAULT_CCR_TTL_SECONDS,
        enable_feedback: bool = True,
        backend: CompressionStoreBackend | None = None,
    ):
        """Initialize the compression store.

        Args:
            max_entries: Maximum number of entries to store.
            default_ttl: Default TTL in seconds (default 30 minutes — session scale).
            enable_feedback: Whether to track retrieval events.
            backend: Storage backend to use. Defaults to InMemoryBackend
                     when constructed directly; `get_compression_store()`
                     defaults to SQLiteBackend for restart/multi-worker
                     safety. Custom backends can be passed for
                     persistence (MongoDB, Redis).
        """
        # Import here to avoid circular imports
        from .backends import InMemoryBackend

        self._backend: CompressionStoreBackend = backend or InMemoryBackend()
        self._lock = threading.Lock()
        self._max_entries = max_entries
        self._default_ttl = default_ttl
        self._enable_feedback = enable_feedback

        # Feedback tracking
        self._retrieval_events: list[RetrievalEvent] = []
        self._max_events = 1000  # Keep last 1000 events
        self._pending_feedback_events: list[RetrievalEvent] = []
        self._ambiguous_retrieval_attempts = 0
        self._rejected_retrieval_references = 0

        # MEDIUM FIX #16: Use a min-heap for O(log n) eviction instead of O(n)
        # Heap entries are (created_at, hash_key) tuples
        self._eviction_heap: list[tuple[float, str]] = []
        # CRITICAL FIX: Track stale entries count to know when heap cleanup is needed
        self._stale_heap_entries = 0
        # Threshold for triggering heap rebuild (when 50% are stale)
        self._heap_rebuild_threshold = 0.5

    @property
    def default_ttl_seconds(self) -> int:
        """Default TTL applied to new entries when callers do not override it."""
        return self._default_ttl

    def store(
        self,
        original: str,
        compressed: str,
        *,
        original_tokens: int = 0,
        compressed_tokens: int = 0,
        original_item_count: int = 0,
        compressed_item_count: int = 0,
        tool_name: str | None = None,
        tool_call_id: str | None = None,
        session_id: str | None = None,
        request_id: str | None = None,
        compression_event_id: str | None = None,
        retrieval_handle: str | None = None,
        provider_slot: str | None = None,
        query_context: str | None = None,
        tool_signature_hash: str | None = None,
        compression_strategy: str | None = None,
        ttl: int | None = None,
        explicit_hash: str | None = None,
    ) -> str:
        """Store compressed content and return hash for retrieval.

        Args:
            original: Original JSON content before compression.
            compressed: Compressed JSON content.
            original_tokens: Token count of original content.
            compressed_tokens: Token count of compressed content.
            original_item_count: Number of items in original array.
            compressed_item_count: Number of items after compression.
            tool_name: Name of the tool that produced this output.
            tool_call_id: ID of the tool call.
            session_id: Originating session, when available.
            request_id: Originating provider/proxy request, when available.
            compression_event_id: Unique adopted compression occurrence.
            retrieval_handle: Opaque event-scoped marker selector.
            provider_slot: Provider-owned message/block slot identifier.
            query_context: User query context for relevance matching.
            tool_signature_hash: Hash from ToolSignature for TOIN correlation.
            compression_strategy: Strategy used for compression.
            ttl: Custom TTL in seconds (uses default if not specified).
            explicit_hash: Use this exact hex hash as the storage key
                instead of computing SHA-256(original)[:24]. Required when
                the marker that points at this entry was emitted by a
                producer with its own hash function (e.g. SmartCrusher's
                Rust row-drop path uses SHA-256[:12]). If not a hex
                string, raises ``ValueError``. The marker hash and the
                store key MUST match — otherwise ``/v1/retrieve/{hash}``
                returns 404 even though the data is present.

        Returns:
            Hash key for retrieving this content.
        """
        # Generate hash from original content. Default: SHA-256[:24] of the
        # original. When the caller provides `explicit_hash`, use it
        # verbatim — required when the hash that ends up in the prompt
        # marker is produced by another component (e.g. the Rust
        # SmartCrusher row-drop path emits SHA-256[:12], which the
        # Python store has to mirror so /v1/retrieve resolves it).
        # 24 chars (96 bits) was chosen for collision resistance under the
        # birthday bound: 50% collision probability at ~280 trillion entries
        # (2^48), versus ~4 billion (2^32) for the previous 16-char default.
        if explicit_hash is not None:
            # Validate as hex. Bail loudly per `feedback_no_silent_fallbacks`
            # — silently falling back to the default hash when the caller
            # asked for a specific key would defeat the marker/store
            # consistency we're trying to preserve.
            if not explicit_hash or not all(c in "0123456789abcdefABCDEF" for c in explicit_hash):
                raise ValueError(
                    f"explicit_hash must be a non-empty hex string, got {explicit_hash!r}"
                )
            hash_key = explicit_hash.lower()
        else:
            # SHA-256 truncated to 24 hex chars (96 bits) — same collision
            # space as the MD5[:24] this replaced. Switched from MD5 in
            # PR #395 to silence CodeQL's `py/weak-sensitive-data-hashing`
            # rule (the `usedforsecurity=False` parameter and the `lgtm`
            # comment marker both failed to suppress it). The cache is
            # in-memory, so changing the hash function on upgrade has no
            # persistence-side effect — the same content always hashes
            # deterministically under whichever function is in use.
            hash_key = hashlib.sha256(original.encode()).hexdigest()[:24]

        # Refuse to persist a bare CCR marker as an entry's "original"
        # (#2694). A marker is a *pointer* to content, never content: an
        # entry like `hash=abc123 -> "<<ccr:abc123,base64,2.0KB>>"` answers a
        # retrieve with the very placeholder the caller is trying to resolve,
        # and (worse) can overwrite a good entry with a useless one. Any
        # producer that gets here has lost the source bytes upstream, so fail
        # loudly rather than silently converting "retrievable" into "gone".
        # Narrow by design: only a *bare* marker is rejected. Legitimate
        # originals may legally CONTAIN markers (nested offloads, a tool that
        # echoed one), and refusing those would drop recoverable data.
        #
        # The rejected value is never echoed into the log. It is provably a
        # bare marker here, but `original` is the store's credential-bearing
        # payload in the general case (this issue was reported against an
        # OAuth token), and an error path is exactly where that sort of leak
        # survives review. `hash_key` already identifies the entry.
        stripped = original.strip()
        if stripped.startswith("<<ccr:") and stripped.endswith(">>") and "\n" not in stripped:
            logger.error(
                "CCR store: refusing to persist a bare retrieval marker as "
                "original_content (hash=%s tool=%s strategy=%s len=%d) — the "
                "producer lost the source bytes; retrieval for this hash will "
                "miss instead of returning a placeholder",
                hash_key,
                tool_name,
                compression_strategy,
                len(stripped),
            )
            return hash_key

        entry = CompressionEntry(
            hash=hash_key,
            original_content=original,
            compressed_content=compressed,
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            original_item_count=original_item_count,
            compressed_item_count=compressed_item_count,
            tool_name=tool_name,
            tool_call_id=tool_call_id,
            session_id=session_id,
            request_id=request_id,
            compression_event_id=compression_event_id,
            retrieval_handle=retrieval_handle,
            provider_slot=provider_slot,
            query_context=query_context,
            created_at=time.time(),
            ttl=ttl if ttl is not None else self._default_ttl,
            tool_signature_hash=tool_signature_hash,
            compression_strategy=compression_strategy,
        )

        # Process pending feedback BEFORE acquiring lock for eviction.
        # This ensures feedback from entries about to be evicted is captured.
        if self._enable_feedback:
            self.process_pending_feedback()

        with self._lock:
            # Decide whether this is a NEW key before evicting. Evicting to make
            # room only applies to a genuinely new entry; a re-store of an
            # existing key overwrites in place (no room needed). Evicting first
            # for a duplicate would needlessly destroy a live, unrelated entry
            # and drop the store below capacity, making that entry's <<ccr:...>>
            # marker (still sitting in the conversation) unredeemable — a 404.
            # The CCR mirror bridge re-stores the same explicit_hash on every
            # turn a marker is re-encountered, so duplicate stores are common.
            existing = self._backend.get(hash_key)
            if existing is None:
                self._evict_if_needed()
            else:
                # Hash already present. Different content means a true (extremely
                # rare with SHA256[:24]) collision; same content is a duplicate
                # re-store. Either way we overwrite in place.
                if existing.original_content != original:
                    logger.warning(
                        "Hash collision detected: hash=%s tool=%s (existing_len=%d, new_len=%d)",
                        hash_key,
                        tool_name,
                        len(existing.original_content),
                        len(original),
                    )
                else:
                    logger.debug(
                        "Duplicate store for hash=%s, updating entry",
                        hash_key,
                    )
                    # SQLite performs this merge atomically under BEGIN
                    # IMMEDIATE. Pre-merging its earlier read could resurrect a
                    # candidate another worker discarded between get() and set().
                    if not getattr(
                        self._backend, "merges_attribution_candidates_atomically", False
                    ):
                        candidates = _attribution_candidates(existing)
                        if not candidates:
                            candidates = [_legacy_unattributed_candidate()]
                        current_candidate = _entry_attribution_candidate(entry)
                        if any(current_candidate.values()):
                            current_identity = _candidate_identity(current_candidate)
                            candidates = [
                                item
                                for item in candidates
                                if _candidate_identity(item) != current_identity
                            ]
                            candidates.append(current_candidate)
                        entry.attribution_candidates = candidates[-MAX_ATTRIBUTION_CANDIDATES:]
                        if not any(current_candidate.values()) and candidates:
                            preserved = candidates[-1]
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
                # Mark old heap entry as stale since we're replacing it.
                self._stale_heap_entries += 1

            if not entry.attribution_candidates:
                candidate = _entry_attribution_candidate(entry)
                if any(candidate.values()):
                    entry.attribution_candidates = [candidate]
            self._backend.set(hash_key, entry)
            # MEDIUM FIX #16: Add to eviction heap for O(log n) eviction
            heapq.heappush(self._eviction_heap, (entry.created_at, hash_key))

        return hash_key

    def discard_attribution_candidate(self, hash_key: str, *, compression_event_id: str) -> bool:
        """Remove storage created for a policy event that never reached the wire."""
        if not hash_key or not compression_event_id:
            return False
        atomic_discard = getattr(self._backend, "discard_attribution_candidate", None)
        if callable(atomic_discard):
            changed = bool(atomic_discard(hash_key, compression_event_id))
            if changed and not self._backend.exists(hash_key):
                with self._lock:
                    self._stale_heap_entries += 1
            return changed
        with self._lock:
            entry = self._backend.get(hash_key)
            if entry is None:
                return False
            replacement, changed = _entry_without_compression_event(entry, compression_event_id)
            if not changed:
                return False
            if replacement is None:
                self._backend.delete(hash_key)
                self._stale_heap_entries += 1
            else:
                self._backend.set(hash_key, replacement)
            return True

    def retrieve(
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
    ) -> CompressionEntry | None:
        """Retrieve original content by hash.

        Args:
            hash_key: Hash key returned by store().
            query: Optional query for feedback tracking.
            compression_event_id: Optional event-level attribution selector.
            retrieval_handle: Optional opaque event-scoped selector.
            session_id: Optional session-level attribution selector.
            request_id: Optional request-level attribution selector.
            tool_call_id: Optional originating tool-call selector.

        Returns:
            CompressionEntry if found and not expired, None otherwise.
        """
        with self._lock:
            atomic_retrieve = getattr(self._backend, "retrieve_and_record_access", None)
            if callable(atomic_retrieve):
                resolution_status, resolved_entry = atomic_retrieve(
                    hash_key,
                    query,
                    compression_event_id=compression_event_id,
                    retrieval_handle=retrieval_handle,
                    session_id=session_id,
                    request_id=request_id,
                    tool_call_id=tool_call_id,
                    provider_slot=provider_slot,
                )
                if resolution_status != "available" or resolved_entry is None:
                    if resolution_status == "expired":
                        self._stale_heap_entries += 1
                    if resolution_status in {"ambiguous", "unresolved"}:
                        self._rejected_retrieval_references += 1
                        if resolution_status == "ambiguous":
                            self._ambiguous_retrieval_attempts += 1
                    return None
                entry = resolved_entry
            else:
                entry = self._backend.get(hash_key)
                if entry is None:
                    return None
                if entry.is_expired():
                    self._backend.delete(hash_key)
                    self._stale_heap_entries += 1
                    return None

                resolution_status = _entry_attribution_resolution_status(
                    entry,
                    compression_event_id=compression_event_id,
                    retrieval_handle=retrieval_handle,
                    session_id=session_id,
                    request_id=request_id,
                    tool_call_id=tool_call_id,
                    provider_slot=provider_slot,
                )
                if resolution_status != "available":
                    self._rejected_retrieval_references += 1
                    if resolution_status == "ambiguous":
                        self._ambiguous_retrieval_attempts += 1
                    return None

                entry.record_access(query)
                self._backend.set(hash_key, entry)
                resolved_entry = _entry_with_resolved_attribution(
                    entry,
                    compression_event_id=compression_event_id,
                    retrieval_handle=retrieval_handle,
                    session_id=session_id,
                    request_id=request_id,
                    tool_call_id=tool_call_id,
                    provider_slot=provider_slot,
                )

            if retrieval_handle and resolved_entry.retrieval_handle != retrieval_handle:
                return None
            if compression_event_id and resolved_entry.compression_event_id != compression_event_id:
                return None
            # Log retrieval feedback only against a uniquely resolved producer.
            if self._enable_feedback:
                self._log_retrieval(
                    hash_key=hash_key,
                    query=query,
                    items_retrieved=entry.original_item_count,
                    total_items=entry.original_item_count,
                    tool_name=resolved_entry.tool_name,
                    retrieval_type="full",
                    tool_signature_hash=resolved_entry.tool_signature_hash,
                )
            self._log_retrieval_payload(
                hash_key=hash_key,
                query=query,
                retrieval_type="full",
                payload=entry.original_content,
                items_retrieved=entry.original_item_count,
                total_items=entry.original_item_count,
                entry=resolved_entry,
            )

            # Return an independent copy with only unambiguous attribution.
            result_entry = resolved_entry

        # Process feedback immediately to ensure TOIN learns in real-time
        if self._enable_feedback:
            self.process_pending_feedback()

        return result_entry

    def retrieve_for_internal_use(
        self,
        hash_key: str,
        *,
        retrieval_handle: str | None = None,
        compression_event_id: str | None = None,
        session_id: str | None = None,
        request_id: str | None = None,
        tool_call_id: str | None = None,
        provider_slot: str | None = None,
    ) -> CompressionEntry | None:
        """Read an entry without claiming that its payload reached a model."""
        with self._lock:
            entry = self._backend.get(hash_key)
            if entry is None:
                return None
            if entry.is_expired():
                self._backend.delete(hash_key)
                self._stale_heap_entries += 1
                return None
            resolution_status = _entry_attribution_resolution_status(
                entry,
                compression_event_id=compression_event_id,
                retrieval_handle=retrieval_handle,
                session_id=session_id,
                request_id=request_id,
                tool_call_id=tool_call_id,
                provider_slot=provider_slot,
            )
            if resolution_status != "available":
                self._rejected_retrieval_references += 1
                if resolution_status == "ambiguous":
                    self._ambiguous_retrieval_attempts += 1
                return None
            result_entry = _entry_with_resolved_attribution(
                entry,
                retrieval_handle=retrieval_handle,
                compression_event_id=compression_event_id,
                session_id=session_id,
                request_id=request_id,
                tool_call_id=tool_call_id,
                provider_slot=provider_slot,
            )
            if retrieval_handle and result_entry.retrieval_handle != retrieval_handle:
                return None
            if compression_event_id and result_entry.compression_event_id != compression_event_id:
                return None
        try:
            from ..transforms.retrieval_aware_policy import (
                compression_cost_tracking_enabled,
                get_compression_cost_ledger,
            )

            if compression_cost_tracking_enabled():
                get_compression_cost_ledger().record_internal_store_access()
        except Exception:
            logger.debug("Internal store-access accounting failed", exc_info=True)
        return result_entry

    def get_retrieval_reference_status(
        self,
        hash_key: str,
        *,
        retrieval_handle: str | None = None,
        compression_event_id: str | None = None,
        session_id: str | None = None,
        request_id: str | None = None,
        tool_call_id: str | None = None,
        provider_slot: str | None = None,
    ) -> dict[str, Any]:
        """Describe reference resolution without returning or charging content."""
        with self._lock:
            entry = self._backend.get(hash_key)
            if entry is None:
                return {"hash": hash_key, "status": "missing", "candidate_count": 0}
            if entry.is_expired():
                return {
                    "hash": hash_key,
                    "status": "expired",
                    "candidate_count": len(_attribution_candidates(entry)),
                    "ttl_seconds": entry.ttl,
                }
            return {
                "hash": hash_key,
                "status": _entry_attribution_resolution_status(
                    entry,
                    compression_event_id=compression_event_id,
                    retrieval_handle=retrieval_handle,
                    session_id=session_id,
                    request_id=request_id,
                    tool_call_id=tool_call_id,
                    provider_slot=provider_slot,
                ),
                "candidate_count": len(_attribution_candidates(entry)),
                "ttl_seconds": entry.ttl,
            }

    def observe_retrieval_call(
        self,
        hash_key: str,
        query: str | None = None,
        *,
        retrieval_handle: str | None = None,
    ) -> bool:
        """Record a model-generated retrieval call without returning payload.

        Legacy mode preserves the existing feedback/TOIN signal. With
        retrieval-aware tracking enabled the observation is diagnostic only:
        it does not train feedback, increment ``CompressionEntry.retrieval_count``,
        or enter payload accounting. A later model-injected recovery is the
        only learning and charging boundary.
        """
        retrieval_aware_tracking = False
        try:
            from ..transforms.retrieval_aware_policy import (
                compression_cost_tracking_enabled,
                get_compression_cost_ledger,
            )

            retrieval_aware_tracking = compression_cost_tracking_enabled()
        except Exception:
            logger.debug("Retrieval-call observation accounting failed", exc_info=True)

        with self._lock:
            entry = self._backend.get(hash_key)
            if entry is None or entry.is_expired():
                return False
            resolution_status = _entry_attribution_resolution_status(
                entry, retrieval_handle=retrieval_handle
            )
            if resolution_status != "available":
                self._rejected_retrieval_references += 1
                if resolution_status == "ambiguous":
                    self._ambiguous_retrieval_attempts += 1
                return False
            resolved_entry = _entry_with_resolved_attribution(
                entry, retrieval_handle=retrieval_handle
            )
            if retrieval_aware_tracking:
                # Demand observation is diagnostic only. Learning waits for the
                # provider result to be injected and acknowledged.
                get_compression_cost_ledger().record_retrieval_call_observation()
            if self._enable_feedback and not retrieval_aware_tracking:
                self._log_retrieval(
                    hash_key=hash_key,
                    query=query,
                    items_retrieved=0,
                    total_items=entry.original_item_count,
                    tool_name=resolved_entry.tool_name,
                    retrieval_type="observed_tool_call",
                    tool_signature_hash=resolved_entry.tool_signature_hash,
                )
        if self._enable_feedback and not retrieval_aware_tracking:
            self.process_pending_feedback()
        return True

    def get_metadata(
        self,
        hash_key: str,
        *,
        retrieval_handle: str | None = None,
    ) -> dict[str, Any] | None:
        """Get metadata about a stored entry without retrieving full content.

        Useful for context tracking to know what was compressed without
        fetching the entire original content.

        Args:
            hash_key: Hash key returned by store().

        Returns:
            Dict with metadata if found and not expired, None otherwise.
        """
        with self._lock:
            entry = self._backend.get(hash_key)

            if entry is None:
                return None

            if entry.is_expired():
                self._backend.delete(hash_key)
                self._stale_heap_entries += 1
                return None

            attribution = resolve_entry_attribution(entry, retrieval_handle=retrieval_handle)
            return {
                "hash": entry.hash,
                "tool_name": attribution["tool_name"],
                "tool_call_id": attribution["tool_call_id"],
                "session_id": attribution["session_id"],
                "request_id": attribution["request_id"],
                "compression_event_id": attribution["compression_event_id"],
                "retrieval_handle": attribution["retrieval_handle"],
                "provider_slot": attribution["provider_slot"],
                "attribution_candidate_count": len(_attribution_candidates(entry)),
                "original_item_count": entry.original_item_count,
                "compressed_item_count": entry.compressed_item_count,
                "query_context": entry.query_context,
                "compressed_content": entry.compressed_content,
                "original_content_preview": entry.original_content[:2000],
                "created_at": entry.created_at,
                "ttl": entry.ttl,
            }

    def _log_retrieval_payload(
        self,
        *,
        hash_key: str,
        query: str | None,
        retrieval_type: str,
        payload: str,
        items_retrieved: int,
        total_items: int,
        entry: CompressionEntry,
    ) -> None:
        event = {
            "event": "headroom_retrieve",
            "hash": hash_key,
            "retrieval_type": retrieval_type,
            "query": query,
            "items_retrieved": items_retrieved,
            "total_items": total_items,
            "tool_name": entry.tool_name,
            "tool_call_id": entry.tool_call_id,
            "compression_strategy": entry.compression_strategy,
            "tool_signature_hash": entry.tool_signature_hash,
            "original_tokens": entry.original_tokens,
            "compressed_tokens": entry.compressed_tokens,
            "original_item_count": entry.original_item_count,
            "compressed_item_count": entry.compressed_item_count,
            **_payload_for_retrieval_log(payload),
        }
        logger.info(
            "event=headroom_retrieve %s",
            json.dumps(event, ensure_ascii=False, separators=(",", ":")),
        )

    def exists(self, hash_key: str, clean_expired: bool = False) -> bool:
        """Check if a hash key exists and is not expired.

        Args:
            hash_key: The hash key to check.
            clean_expired: If True, delete the entry if expired.
                          LOW FIX #20: Default False to make this a pure check.

        Returns:
            True if the entry exists and is not expired.
        """
        with self._lock:
            entry = self._backend.get(hash_key)
            if entry is None:
                return False
            if entry.is_expired():
                # LOW FIX #20: Only delete if explicitly requested
                # This makes exists() a pure check by default
                if clean_expired:
                    self._backend.delete(hash_key)
                    # CRITICAL FIX: Track stale heap entry
                    self._stale_heap_entries += 1
                return False
            return True

    def get_entry_status(
        self,
        hash_key: str,
        *,
        clean_expired: bool = False,
    ) -> dict[str, Any]:
        """Return availability and TTL metadata for a stored entry."""
        now = time.time()
        with self._lock:
            entry = self._backend.get(hash_key)
            if entry is None:
                return {
                    "hash": hash_key,
                    "status": "missing",
                    "default_ttl_seconds": self._default_ttl,
                }

            age_seconds = now - entry.created_at
            expires_at = entry.created_at + entry.ttl
            expired = age_seconds > entry.ttl
            status = {
                "hash": hash_key,
                "status": "expired" if expired else "available",
                "ttl_seconds": entry.ttl,
                "default_ttl_seconds": self._default_ttl,
                "created_at": entry.created_at,
                "expires_at": expires_at,
                "age_seconds": age_seconds,
            }

            if expired and clean_expired:
                self._backend.delete(hash_key)
                self._stale_heap_entries += 1

            return status

    def get_stats(self) -> dict[str, Any]:
        """Get store statistics for monitoring."""
        with self._lock:
            # Clean expired entries
            self._clean_expired()

            # Get all entries for statistics
            entries = [entry for _, entry in self._backend.items()]
            total_original_tokens = sum(e.original_tokens for e in entries)
            total_compressed_tokens = sum(e.compressed_tokens for e in entries)
            total_retrievals = sum(e.retrieval_count for e in entries)

            # Include backend stats
            backend_stats = self._backend.get_stats()

            return {
                "entry_count": self._backend.count(),
                "max_entries": self._max_entries,
                "default_ttl_seconds": self._default_ttl,
                "total_original_tokens": total_original_tokens,
                "total_compressed_tokens": total_compressed_tokens,
                "total_retrievals": total_retrievals,
                "ambiguous_retrieval_attempts": self._ambiguous_retrieval_attempts,
                "rejected_retrieval_references": self._rejected_retrieval_references,
                "event_count": len(self._retrieval_events),
                "backend": backend_stats,
            }

    def get_memory_stats(self) -> ComponentStats:
        """Get memory statistics for the MemoryTracker.

        Returns:
            ComponentStats with current memory usage.
        """
        from ..memory.tracker import ComponentStats

        with self._lock:
            # Get backend stats which include bytes_used
            backend_stats = self._backend.get_stats()
            bytes_used = backend_stats.get("bytes_used", 0)

            # Add retrieval events memory
            import sys

            bytes_used += sys.getsizeof(self._retrieval_events)
            for event in self._retrieval_events:
                bytes_used += sys.getsizeof(event)

            # Add eviction heap memory
            bytes_used += sys.getsizeof(self._eviction_heap)

            return ComponentStats(
                name="compression_store",
                entry_count=self._backend.count(),
                size_bytes=bytes_used,
                budget_bytes=None,  # No budget set yet
                hits=sum(1 for _, e in self._backend.items() if e.retrieval_count > 0),
                misses=0,  # CompressionStore doesn't track misses directly
                evictions=0,  # Would need to track this separately
            )

    def get_retrieval_events(
        self,
        limit: int = 100,
        tool_name: str | None = None,
    ) -> list[RetrievalEvent]:
        """Get recent retrieval events for feedback analysis.

        Args:
            limit: Maximum number of events to return.
            tool_name: Filter by tool name if specified.

        Returns:
            List of recent retrieval events (copies to prevent mutation).
        """
        with self._lock:
            # MEDIUM FIX #17: Take a slice copy immediately to avoid race conditions
            # if another thread modifies _retrieval_events after we release the lock
            events_copy = list(self._retrieval_events)

        # Filter and slice outside lock (safe since we have a copy)
        if tool_name:
            events_copy = [e for e in events_copy if e.tool_name == tool_name]

        return list(reversed(events_copy[-limit:]))

    def clear(self) -> None:
        """Clear all entries. Mainly for testing."""
        with self._lock:
            self._backend.clear()
            self._retrieval_events.clear()
            self._pending_feedback_events.clear()
            self._ambiguous_retrieval_attempts = 0
            self._rejected_retrieval_references = 0
            self._eviction_heap.clear()  # MEDIUM FIX #16: Clear heap too
            self._stale_heap_entries = 0  # CRITICAL FIX: Reset stale counter

    def _evict_if_needed(self) -> None:
        """Evict old entries if at capacity. Must be called with lock held.

        MEDIUM FIX #16: Use heap for O(log n) eviction instead of O(n) scan.
        CRITICAL FIX: Track and clean stale heap entries to prevent memory leak.
        """
        # First, remove expired entries
        self._clean_expired()

        # CRITICAL FIX: Rebuild heap if too many stale entries
        # This prevents unbounded heap growth when entries are deleted/replaced
        heap_size = len(self._eviction_heap)
        if heap_size > 0:
            stale_ratio = self._stale_heap_entries / heap_size
            if stale_ratio >= self._heap_rebuild_threshold:
                self._rebuild_heap()

        # If still at capacity, remove oldest entries using heap
        while self._backend.count() >= self._max_entries and self._eviction_heap:
            # Pop oldest from heap (O(log n))
            created_at, hash_key = heapq.heappop(self._eviction_heap)

            # Check if entry still exists and matches timestamp
            # (entry might have been deleted or replaced)
            entry = self._backend.get(hash_key)
            if entry is not None and entry.created_at == created_at:
                # HIGH FIX: Track eviction as "successful compression" if never retrieved
                # This prevents state divergence between store and feedback loop
                if self._enable_feedback and entry.retrieval_count == 0:
                    # Entry was never retrieved = compression was successful
                    # Notify feedback system so it knows this strategy worked
                    self._record_eviction_success(entry)
                self._backend.delete(hash_key)
            else:
                # CRITICAL FIX: This was a stale entry, decrement counter
                # (we already popped it, so the stale entry is now gone)
                if self._stale_heap_entries > 0:
                    self._stale_heap_entries -= 1

    def _clean_expired(self) -> None:
        """Remove expired entries. Must be called with lock held.

        CRITICAL FIX: Track stale heap entries when deleting to prevent memory leak.
        """
        expired_keys = [key for key, entry in self._backend.items() if entry.is_expired()]
        for key in expired_keys:
            self._backend.delete(key)
            # CRITICAL FIX: Increment stale counter - the heap still has an entry
            # for this key that will be stale when we try to evict
            self._stale_heap_entries += 1

    def _rebuild_heap(self) -> None:
        """Rebuild heap from current store entries. Must be called with lock held.

        CRITICAL FIX: This removes stale heap entries that accumulate when entries
        are deleted or replaced. Without this, the heap grows unboundedly.
        """
        # Build new heap from current store entries only
        self._eviction_heap = [
            (entry.created_at, hash_key) for hash_key, entry in self._backend.items()
        ]
        heapq.heapify(self._eviction_heap)
        # Reset stale counter - heap is now clean
        self._stale_heap_entries = 0
        logger.debug(
            "Rebuilt eviction heap: %d entries",
            len(self._eviction_heap),
        )

    def _record_eviction_success(self, entry: CompressionEntry) -> None:
        """Record successful compression when an entry is evicted without retrieval.

        HIGH FIX: State divergence on eviction
        When an entry is evicted and was NEVER retrieved, this indicates the
        compression was fully successful - the LLM never needed the original data.
        We notify the feedback system so it can learn from this success.

        Must be called with lock held (entry data access).
        Actual feedback notification happens outside lock.

        Args:
            entry: The entry being evicted.
        """
        # Capture entry data while we have the lock
        tool_name = entry.tool_name
        sig_hash = entry.tool_signature_hash
        strategy = entry.compression_strategy

        # We can't call feedback while holding the lock (would cause deadlock)
        # Instead, queue this for deferred processing
        if sig_hash is not None and strategy is not None:
            # Create a synthetic "success" event that we'll process later
            # Use a special retrieval type to indicate this was an eviction success
            success_event = RetrievalEvent(
                hash=entry.hash,
                query=None,
                items_retrieved=0,  # No retrieval happened
                total_items=entry.original_item_count,
                tool_name=tool_name,
                timestamp=time.time(),
                retrieval_type="eviction_success",  # Special marker
                tool_signature_hash=sig_hash,
            )
            self._pending_feedback_events.append(success_event)
            logger.debug(
                "Recorded eviction success: hash=%s strategy=%s",
                entry.hash[:8],
                strategy,
            )

    def _log_retrieval(
        self,
        hash_key: str,
        query: str | None,
        items_retrieved: int,
        total_items: int,
        tool_name: str | None,
        retrieval_type: str,
        tool_signature_hash: str | None = None,
    ) -> None:
        """Log a retrieval event. Must be called with lock held."""
        event = RetrievalEvent(
            hash=hash_key,
            query=query,
            items_retrieved=items_retrieved,
            total_items=total_items,
            tool_name=tool_name,
            timestamp=time.time(),
            retrieval_type=retrieval_type,
            tool_signature_hash=tool_signature_hash,
        )

        self._retrieval_events.append(event)

        # Keep only recent events
        if len(self._retrieval_events) > self._max_events:
            self._retrieval_events = self._retrieval_events[-self._max_events :]

        # Queue event for feedback processing (will be processed after lock release)
        # This is safe because process_pending_feedback() uses the lock to atomically
        # swap out the pending list before processing
        self._pending_feedback_events.append(event)

    def process_pending_feedback(self) -> None:
        """Process pending feedback events.

        Forwards events to:
        1. CompressionFeedback - for learning compression hints
        2. TelemetryCollector - for the data flywheel
        3. TOIN - for cross-user intelligence network

        This is called automatically on each retrieval to ensure the
        feedback loop operates in real-time.
        """
        from ..telemetry import get_telemetry_collector
        from ..telemetry.toin import get_toin
        from .compression_feedback import get_compression_feedback

        # Get pending events and related entry data atomically
        with self._lock:
            events = self._pending_feedback_events
            self._pending_feedback_events = []

            # Gather entry data while holding lock to avoid race conditions
            # Tuple: (event, tool_name, sig_hash, strategy, compressed_content)
            event_data: list[
                tuple[RetrievalEvent, str | None, str | None, str | None, str | None]
            ] = []
            for event in events:
                entry = self._backend.get(event.hash)
                if entry:
                    # Use the ACTUAL tool_signature_hash stored during compression
                    # This MUST match the hash used by SmartCrusher
                    event_data.append(
                        (
                            event,
                            entry.tool_name,
                            entry.tool_signature_hash,  # The correct hash!
                            entry.compression_strategy,
                            entry.compressed_content,  # For TOIN field-level learning
                        )
                    )
                else:
                    event_data.append((event, None, None, None, None))

        # Process outside lock
        if event_data:
            feedback = get_compression_feedback()
            telemetry = get_telemetry_collector()
            toin = get_toin()

            for event, _tool_name, sig_hash, strategy, compressed_content in event_data:
                # Notify feedback system (pass strategy for success rate tracking)
                feedback.record_retrieval(event, strategy=strategy)

                # Extract query fields if present
                query_fields = None
                if event.query:
                    # Extract field:value patterns
                    query_fields = re.findall(r"(\w+)[=:]", event.query)

                # Notify telemetry for data flywheel
                try:
                    if sig_hash is not None:
                        telemetry.record_retrieval(
                            tool_signature_hash=sig_hash,
                            retrieval_type=event.retrieval_type,
                            query_fields=query_fields,
                        )
                except Exception:
                    # Telemetry should never break the feedback loop
                    logger.debug("Telemetry record_retrieval failed", exc_info=True)

                # Parse compressed content to extract items for TOIN field-level learning
                retrieved_items: list[dict[str, Any]] | None = None
                if compressed_content:
                    try:
                        parsed = json.loads(compressed_content)
                        # Handle both direct arrays and wrapped arrays
                        if isinstance(parsed, list):
                            # Filter to dicts only (field learning needs dict items)
                            retrieved_items = [item for item in parsed if isinstance(item, dict)]
                        elif isinstance(parsed, dict):
                            # Check for common wrapper patterns: {"items": [...], "results": [...]}
                            for key in ("items", "results", "data", "records"):
                                if key in parsed and isinstance(parsed[key], list):
                                    retrieved_items = [
                                        item for item in parsed[key] if isinstance(item, dict)
                                    ]
                                    break
                    except (json.JSONDecodeError, TypeError):
                        # Invalid JSON - skip field learning for this retrieval
                        pass

                # Notify TOIN for cross-user learning
                try:
                    if sig_hash is not None:
                        toin.record_retrieval(
                            tool_signature_hash=sig_hash,
                            retrieval_type=event.retrieval_type,
                            query=event.query,
                            query_fields=query_fields,
                            strategy=strategy,  # Pass strategy for success rate tracking
                            retrieved_items=retrieved_items,  # For field-level learning
                        )
                except Exception:
                    # TOIN should never break the feedback loop
                    logger.debug("TOIN record_retrieval failed", exc_info=True)


# Request-scoped store (for multi-tenant SaaS: one store per request/tenant)
_request_ccr_store: ContextVar[CompressionStore | None] = ContextVar(
    "headroom_request_ccr_store", default=None
)

# Global store instance (lazy initialization)
_compression_store: CompressionStore | None = None
_store_lock = threading.Lock()


def set_request_compression_store(store: CompressionStore | None) -> None:
    """Set the compression store for the current request context.

    Used by middleware (e.g. SaaS) to provide a tenant-scoped store.
    When set, get_compression_store() returns this store instead of the global one.

    Args:
        store: CompressionStore to use for this request, or None to clear.
    """
    _request_ccr_store.set(store)


def clear_request_compression_store() -> None:
    """Clear the request-scoped compression store."""
    _request_ccr_store.set(None)


def _create_default_ccr_backend() -> CompressionStoreBackend | None:
    """Create a CCR backend from env (e.g. HEADROOM_CCR_BACKEND=redis).

    Default (env unset or "sqlite"): SQLiteBackend at workspace_dir()/ccr_store.db
    — restart-safe and shared across worker processes, which the
    session-scale 30-minute TTL assumes.
    "memory" opts back into the in-process dict. Other values load
    adapters via setuptools entry point 'headroom.ccr_backend'.
    Returns None to use InMemoryBackend.
    """
    backend_type = (os.environ.get("HEADROOM_CCR_BACKEND") or "").strip().lower()
    if backend_type == "memory":
        return None
    if not backend_type or backend_type == "sqlite":
        try:
            from .backends.sqlite import SQLiteBackend

            return SQLiteBackend()
        except Exception as e:
            logger.warning(
                "Failed to initialize SQLite CCR backend (%s); "
                "falling back to in-memory store. Retrieval will not "
                "survive proxy restarts.",
                e,
            )
            return None
    try:
        from importlib.metadata import entry_points

        all_eps = entry_points(group="headroom.ccr_backend")
        ep = next((e for e in all_eps if e.name == backend_type), None)
        if ep is None:
            logger.warning(
                "HEADROOM_CCR_BACKEND=%s but no entry point headroom.ccr_backend[%s]",
                backend_type,
                backend_type,
            )
            return None
        fn = ep.load()
        kwargs = {
            "url": os.environ.get("HEADROOM_REDIS_URL", ""),
            "tenant_prefix": os.environ.get("HEADROOM_CCR_TENANT_PREFIX", ""),
        }
        backend: CompressionStoreBackend = fn(**kwargs)
        return backend
    except Exception as e:
        logger.warning("Failed to load CCR backend %s: %s", backend_type, e)
        return None


def get_compression_store(
    max_entries: int = 1000,
    default_ttl: int | None = None,
    backend: CompressionStoreBackend | None = None,
) -> CompressionStore:
    """Get the compression store instance.

    If a request-scoped store was set (e.g. by SaaS middleware), returns it.
    Otherwise uses lazy-initialized global singleton. Backend can be supplied
    explicitly or created from env (HEADROOM_CCR_BACKEND) when building the global.

    Args:
        max_entries: Maximum entries (only used on first call for global store).
        default_ttl: Default TTL (only used on first call for global store).
            When omitted, HEADROOM_CCR_TTL_SECONDS overrides the 1800-second default.
        backend: Custom storage backend (only used on first call for global store).
                 Defaults to InMemoryBackend if not provided; env backend used if backend is None.

    Returns:
        Request-scoped CompressionStore if set, else global CompressionStore instance.
    """
    request_store = _request_ccr_store.get()
    if request_store is not None:
        return request_store

    global _compression_store
    if _compression_store is None:
        with _store_lock:
            if _compression_store is None:
                if backend is None:
                    backend = _create_default_ccr_backend()
                effective_default_ttl = (
                    default_ttl if default_ttl is not None else _get_env_default_ttl_seconds()
                )
                _compression_store = CompressionStore(
                    max_entries=max_entries,
                    default_ttl=effective_default_ttl,
                    backend=backend,
                )
    return _compression_store


def reset_compression_store() -> None:
    """Reset the global compression store. Mainly for testing."""
    global _compression_store

    with _store_lock:
        if _compression_store is not None:
            _compression_store.clear()
        _compression_store = None
