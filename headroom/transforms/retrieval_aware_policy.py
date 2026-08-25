"""Retrieval-aware policy and payload-boundary compression accounting.

Compression-event metrics stop at the recovery payload boundary; provider usage,
conversation replay, API requests, and wall time are trajectory-level benchmark
metrics. The deterministic controller compares input-token costs as::

    passthrough = original
    lossless = verified_lossless
    lossy = predicted_compressed
            + E[retrieval_count] * predicted_recovery_payload
            + P(any retrieval) * predicted_extra_turn

The configurable extra-turn term covers non-payload arguments, wrappers, and
history replay. It is an estimate, not an observed end-to-end round-trip cost.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import threading
import time
import uuid
from collections import OrderedDict, deque
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any


class CompressionAction(str, Enum):
    PASSTHROUGH = "passthrough"
    LOSSLESS = "lossless"
    LOSSY = "lossy"


class RecoveryPayloadPath(str, Enum):
    """Known model-input shapes for a recovered CCR payload."""

    MCP = "mcp"
    ANTHROPIC = "anthropic"
    OPENAI_CHAT = "openai_chat"
    OPENAI_RESPONSES = "openai_responses"
    GOOGLE = "google"
    GENERIC = "generic"
    UNKNOWN = "unknown"


class RetrievalReportStatus(str, Enum):
    ATTRIBUTED = "attributed"
    DUPLICATE = "duplicate"
    UNATTRIBUTED = "unattributed"
    REJECTED = "rejected"
    DROPPED = "dropped"


def stable_retrieval_event_id(namespace: str, *parts: object) -> str:
    """Derive an opaque retry-stable ID from a provider/MCP call identity."""
    material = "\x1f".join([str(namespace), *(str(part or "") for part in parts)])
    return f"re-{hashlib.sha256(material.encode('utf-8')).hexdigest()[:32]}"


def _env_int(name: str, default: int) -> int:
    try:
        return max(int(os.environ.get(name, str(default))), 0)
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        value = float(os.environ.get(name, str(default)))
        return max(value, 0.0) if math.isfinite(value) else default
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class RetrievalAwarePolicyConfig:
    """Expected-cost and outcome-maturation settings.

    The fallback terms are operator-configurable with
    ``HEADROOM_RETRIEVAL_EXTRA_TURN_TOKENS``,
    ``HEADROOM_RETRIEVAL_PAYLOAD_ENVELOPE_TOKENS``, and
    ``HEADROOM_RETRIEVAL_OBSERVATION_WINDOW_SECONDS``.
    """

    min_original_tokens: int = 200
    min_history: int = 3
    min_expected_savings_tokens: int = 16
    predicted_lossy_ratio: float = 0.30
    default_retrieval_probability: float = 0.25
    disposable_tool_probability: float = 0.10
    read_tool_probability: float = 0.70
    shell_tool_probability: float = 0.70
    exact_intent_probability: float = 0.80
    analysis_intent_probability: float = 0.45
    history_prior_strength: float = 2.0
    cold_start_retrievals_given_retrieval: float = 1.0
    predicted_recovery_payload_envelope_tokens: int = 64
    predicted_extra_turn_tokens: int = 1024
    observation_window_seconds: float = 300.0

    @classmethod
    def from_env(cls) -> RetrievalAwarePolicyConfig:
        return cls(
            predicted_recovery_payload_envelope_tokens=_env_int(
                "HEADROOM_RETRIEVAL_PAYLOAD_ENVELOPE_TOKENS", 64
            ),
            predicted_extra_turn_tokens=_env_int("HEADROOM_RETRIEVAL_EXTRA_TURN_TOKENS", 1024),
            observation_window_seconds=_env_float(
                "HEADROOM_RETRIEVAL_OBSERVATION_WINDOW_SECONDS", 300.0
            ),
        )


@dataclass(frozen=True)
class RetrievalAwareDecision:
    action: CompressionAction
    original_tokens: int
    lossless_tokens: int
    predicted_lossy_tokens: int
    probability_of_any_retrieval: float
    expected_retrieval_count: float
    predicted_recovery_payload_tokens: int
    recovery_payload_path: str
    predicted_extra_turn_tokens: int
    passthrough_cost: float
    lossless_cost: float
    lossy_expected_cost: float
    probability_source: str
    reason: str

    @property
    def retrieval_probability(self) -> float:
        """Deprecated in-process alias."""
        return self.probability_of_any_retrieval

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["action"] = self.action.value
        return result


@dataclass
class CompressionCostEvent:
    compression_event_id: str
    session_id: str
    request_id: str
    tool_call_id: str
    provider_slot: str
    tool_name: str
    strategy: str
    action: CompressionAction
    predicted_action: CompressionAction
    original_tokens: int
    initially_emitted_tokens: int
    ccr_hashes: tuple[str, ...]
    created_at: float
    maturity_deadline: float
    eligible_for_learning: bool = True
    explicitly_mature: bool = False
    retrieval_count: int = 0
    recovery_payload_tokens: int = 0

    @property
    def event_id(self) -> str:
        return self.compression_event_id

    @property
    def compressed_tokens(self) -> int:
        return self.initially_emitted_tokens

    @property
    def gross_savings_tokens(self) -> int:
        return self.original_tokens - self.initially_emitted_tokens

    @property
    def payload_net_savings_tokens(self) -> int:
        return self.gross_savings_tokens - self.recovery_payload_tokens

    @property
    def was_retrieved(self) -> bool:
        return self.retrieval_count > 0

    def is_mature(self, now: float) -> bool:
        return self.explicitly_mature or now >= self.maturity_deadline

    def to_dict(self, now: float) -> dict[str, Any]:
        return {
            "compression_event_id": self.compression_event_id,
            "session_id": self.session_id,
            "request_id": self.request_id,
            "tool_call_id": self.tool_call_id,
            "provider_slot": self.provider_slot,
            "hashes": list(self.ccr_hashes),
            "tool_name": self.tool_name,
            "strategy": self.strategy,
            "action": self.action.value,
            "predicted_action": self.predicted_action.value,
            "original_tokens": self.original_tokens,
            "initially_emitted_tokens": self.initially_emitted_tokens,
            "gross_savings_tokens": self.gross_savings_tokens,
            "retrieval_count": self.retrieval_count,
            "recovery_payload_tokens": self.recovery_payload_tokens,
            "payload_net_savings_tokens": self.payload_net_savings_tokens,
            "eligible_for_learning": self.eligible_for_learning,
            "mature": self.is_mature(now),
            "created_at": self.created_at,
            "maturity_deadline": self.maturity_deadline,
        }


# Deprecated type alias retained for in-process compatibility.
RealizedCompressionEvent = CompressionCostEvent


@dataclass(frozen=True)
class ToolCompressionStats:
    mature_compressions: int = 0
    pending_compressions: int = 0
    retrieved_compressions: int = 0
    retrieval_count: int = 0
    original_tokens: int = 0
    initially_emitted_tokens: int = 0
    gross_savings_tokens: int = 0
    recovery_payload_tokens: int = 0

    @property
    def compressions(self) -> int:
        """Deprecated alias for mature compression count."""
        return self.mature_compressions

    @property
    def retrieval_rate(self) -> float:
        if not self.mature_compressions:
            return 0.0
        return min(self.retrieved_compressions / self.mature_compressions, 1.0)

    @property
    def average_retrieval_count(self) -> float:
        if not self.mature_compressions:
            return 0.0
        return self.retrieval_count / self.mature_compressions

    @property
    def average_compressed_ratio(self) -> float:
        if not self.original_tokens:
            return 1.0
        return min(max(self.initially_emitted_tokens / self.original_tokens, 0.0), 1.0)

    @property
    def payload_net_savings_tokens(self) -> int:
        return self.gross_savings_tokens - self.recovery_payload_tokens

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "retrieval_rate": self.retrieval_rate,
            "average_retrieval_count": self.average_retrieval_count,
            "average_compressed_ratio": self.average_compressed_ratio,
            "payload_net_savings_tokens": self.payload_net_savings_tokens,
        }


# Deprecated type alias retained for in-process compatibility.
ToolRealizedStats = ToolCompressionStats


class CompressionCostLedger:
    """Bounded event ledger with session-aware hash attribution and idempotency."""

    def __init__(
        self,
        max_events: int = 10_000,
        *,
        max_hash_candidates: int = 16,
        max_retrieval_reports: int = 50_000,
        observation_window_seconds: float = 300.0,
        clock: Any = time.time,
    ):
        if min(max_events, max_hash_candidates, max_retrieval_reports) <= 0:
            raise ValueError("ledger bounds must be positive")
        self._max_events = max_events
        self._max_hash_candidates = max_hash_candidates
        self._max_retrieval_reports = max_retrieval_reports
        self._observation_window_seconds = max(float(observation_window_seconds), 0.0)
        self._clock = clock
        self._events: OrderedDict[str, CompressionCostEvent] = OrderedDict()
        self._hash_to_events: dict[str, deque[str]] = {}
        # Bounded LRU idempotency horizon. New reports displace the least
        # recently seen ID instead of permanently saturating the ledger.
        self._retrieval_reports: OrderedDict[str, tuple[str, str]] = OrderedDict()
        # Hash-scoped fail-closed horizon for evicted metadata-free report IDs.
        # It is independently bounded by ``max_retrieval_reports``.
        self._legacy_hash_report_hazards: OrderedDict[str, None] = OrderedDict()
        self._counters = {
            "attributed_retrieval_reports": 0,
            "unattributed_retrieval_reports": 0,
            "duplicate_retrieval_reports": 0,
            "rejected_retrieval_reports": 0,
            # Deprecated capacity-drop counter: retained for API compatibility.
            "dropped_retrieval_reports": 0,
            "idempotency_report_evictions": 0,
            "dropped_attribution_candidates": 0,
            "legacy_hash_hazard_evictions": 0,
            "ambiguous_attributions": 0,
            "retrieval_call_observations": 0,
            "internal_store_accesses": 0,
            "client_inline_expansion_events": 0,
            "client_inline_expansion_tokens": 0,
            "http_transport_retrieval_events": 0,
            "http_transport_payload_tokens": 0,
        }
        self._lock = threading.RLock()

    @staticmethod
    def new_compression_event_id() -> str:
        return f"ce-{uuid.uuid4()}"

    @staticmethod
    def new_retrieval_event_id() -> str:
        return f"re-{uuid.uuid4()}"

    @staticmethod
    def new_retrieval_handle() -> str:
        """Return an opaque marker handle that reveals no request metadata."""
        return f"rh-{uuid.uuid4().hex}"

    def claim_feedback_report(self, retrieval_event_id: str) -> bool:
        """Claim one report ID when only legacy store feedback is active.

        The proxy uses this while retrieval-aware accounting is disabled so a
        retry cannot increment CompressionStore/TOIN feedback twice. Claims
        share the ledger's bounded LRU horizon with accounting reports; a later
        accounting attempt carrying the same ID is therefore also a duplicate.
        """
        retrieval_id = str(retrieval_event_id or "")
        with self._lock:
            if not retrieval_id:
                self._counters["rejected_retrieval_reports"] += 1
                return False
            if retrieval_id in self._retrieval_reports:
                self._retrieval_reports.move_to_end(retrieval_id)
                self._counters["duplicate_retrieval_reports"] += 1
                return False
            self._remember_retrieval_report_locked(retrieval_id, "feedback_only", "")
            return True

    def record_outcome(
        self,
        *,
        tool_name: str | None,
        strategy: str,
        action: CompressionAction,
        predicted_action: CompressionAction,
        original_tokens: int,
        compressed_tokens: int | None = None,
        initially_emitted_tokens: int | None = None,
        ccr_hashes: Iterable[str] = (),
        session_id: str | None = None,
        request_id: str | None = None,
        tool_call_id: str | None = None,
        provider_slot: str | None = None,
        compression_event_id: str | None = None,
        eligible_for_learning: bool = True,
        created_at: float | None = None,
    ) -> str:
        original = max(int(original_tokens), 0)
        value = (
            initially_emitted_tokens if initially_emitted_tokens is not None else compressed_tokens
        )
        emitted = max(int(value or 0), 0)
        hashes = tuple(dict.fromkeys(str(item).lower() for item in ccr_hashes if item))
        event_id = compression_event_id or self.new_compression_event_id()
        timestamp = self._clock() if created_at is None else float(created_at)
        with self._lock:
            if event_id in self._events:
                raise ValueError(f"duplicate compression_event_id: {event_id}")
            event = CompressionCostEvent(
                compression_event_id=event_id,
                session_id=str(session_id or ""),
                request_id=str(request_id or ""),
                tool_call_id=str(tool_call_id or ""),
                provider_slot=str(provider_slot or ""),
                tool_name=(tool_name or "unknown").strip() or "unknown",
                strategy=str(strategy or "unknown"),
                action=action,
                predicted_action=predicted_action,
                original_tokens=original,
                initially_emitted_tokens=emitted,
                ccr_hashes=hashes,
                created_at=timestamp,
                maturity_deadline=timestamp + self._observation_window_seconds,
                eligible_for_learning=bool(eligible_for_learning),
            )
            self._events[event_id] = event
            for ccr_hash in hashes:
                candidates = self._hash_to_events.setdefault(
                    ccr_hash, deque(maxlen=self._max_hash_candidates)
                )
                if len(candidates) == candidates.maxlen:
                    self._counters["dropped_attribution_candidates"] += 1
                candidates.append(event_id)
            self._trim_locked()
        return event_id

    def discard_outcome(self, compression_event_id: str) -> bool:
        with self._lock:
            event = self._events.pop(compression_event_id, None)
            if event is None:
                return False
            self._remove_event_indexes_locked(event)
            return True

    def event_hashes(self, compression_event_id: str) -> tuple[str, ...]:
        """Return hashes owned by an event before an outer gate reconciles it."""
        with self._lock:
            event = self._events.get(compression_event_id)
            return event.ccr_hashes if event is not None else ()

    def event_action(self, compression_event_id: str) -> CompressionAction | None:
        """Return the current adopted action for deferred side-effect commits."""
        with self._lock:
            event = self._events.get(compression_event_id)
            return event.action if event is not None else None

    def adopt_passthrough(
        self,
        compression_event_id: str,
        *,
        strategy: str = "outer_gate_passthrough",
    ) -> bool:
        """Convert a rejected compression candidate into the wire action.

        Tokenizer, batching, and reversibility gates run after the controller.
        When one rejects a candidate, passthrough is the action actually emitted;
        retaining that action is more accurate than dropping the decision event.
        """
        with self._lock:
            event = self._events.get(compression_event_id)
            if event is None or event.retrieval_count or event.recovery_payload_tokens:
                return False
            self._remove_event_indexes_locked(event)
            event.action = CompressionAction.PASSTHROUGH
            event.strategy = strategy
            event.initially_emitted_tokens = event.original_tokens
            event.ccr_hashes = ()
            event.explicitly_mature = True
            return True

    def mark_event_mature(self, compression_event_id: str) -> bool:
        with self._lock:
            event = self._events.get(compression_event_id)
            if event is None:
                return False
            event.explicitly_mature = True
            return True

    def mark_session_complete(self, session_id: str) -> int:
        if not session_id:
            return 0
        with self._lock:
            events = [item for item in self._events.values() if item.session_id == session_id]
            for event in events:
                event.explicitly_mature = True
            return len(events)

    def record_recovery(
        self,
        ccr_hash: str,
        *,
        retrieval_event_id: str,
        recovery_payload_tokens: int,
        compression_event_id: str | None = None,
        session_id: str | None = None,
        request_id: str | None = None,
        tool_call_id: str | None = None,
        provider_slot: str | None = None,
    ) -> RetrievalReportStatus:
        normalized_hash = str(ccr_hash or "").lower()
        retrieval_id = str(retrieval_event_id or "")
        try:
            payload_tokens = int(recovery_payload_tokens)
        except (TypeError, ValueError):
            payload_tokens = -1
        with self._lock:
            if not normalized_hash or not retrieval_id or payload_tokens < 0:
                self._counters["rejected_retrieval_reports"] += 1
                return RetrievalReportStatus.REJECTED
            if retrieval_id in self._retrieval_reports:
                self._retrieval_reports.move_to_end(retrieval_id)
                self._counters["duplicate_retrieval_reports"] += 1
                return RetrievalReportStatus.DUPLICATE
            has_immutable_event_scope = bool(compression_event_id)
            # Only the unique compression-event ID remains safe after a report
            # leaves the bounded idempotency horizon. Session/request/tool-call
            # selectors can be reused by a newer same-content event and must
            # therefore retain the hash hazard just like a hash-only report.
            legacy_hazard_hash = "" if has_immutable_event_scope else normalized_hash
            # horizon fail closed. Fresh unrelated legacy hashes remain usable.
            if (
                not has_immutable_event_scope
                and normalized_hash in self._legacy_hash_report_hazards
            ):
                self._legacy_hash_report_hazards.move_to_end(normalized_hash)
                self._remember_retrieval_report_locked(
                    retrieval_id, RetrievalReportStatus.REJECTED.value, legacy_hazard_hash
                )
                self._counters["rejected_retrieval_reports"] += 1
                return RetrievalReportStatus.REJECTED
            event, rejected = self._resolve_event_locked(
                normalized_hash,
                compression_event_id=compression_event_id,
                session_id=session_id,
                request_id=request_id,
                tool_call_id=tool_call_id,
                provider_slot=provider_slot,
            )
            if rejected:
                self._remember_retrieval_report_locked(
                    retrieval_id, RetrievalReportStatus.REJECTED.value, legacy_hazard_hash
                )
                self._counters["rejected_retrieval_reports"] += 1
                return RetrievalReportStatus.REJECTED
            if event is None:
                self._remember_retrieval_report_locked(
                    retrieval_id, RetrievalReportStatus.UNATTRIBUTED.value, legacy_hazard_hash
                )
                self._counters["unattributed_retrieval_reports"] += 1
                return RetrievalReportStatus.UNATTRIBUTED
            event.retrieval_count += 1
            event.recovery_payload_tokens += payload_tokens
            # A positive outcome is final as soon as recovery occurs. Only
            # negative outcomes need the observation window/session boundary.
            event.explicitly_mature = True
            self._remember_retrieval_report_locked(
                retrieval_id, event.compression_event_id, legacy_hazard_hash
            )
            self._counters["attributed_retrieval_reports"] += 1
            return RetrievalReportStatus.ATTRIBUTED

    def record_rejected_retrieval_attempt(
        self,
        ccr_hash: str,
        *,
        retrieval_event_id: str,
        ambiguous: bool = False,
    ) -> RetrievalReportStatus:
        """Record a fail-closed public retrieval without maturing any event."""
        normalized_hash = str(ccr_hash or "").lower()
        retrieval_id = str(retrieval_event_id or "")
        with self._lock:
            if not normalized_hash or not retrieval_id:
                self._counters["rejected_retrieval_reports"] += 1
                return RetrievalReportStatus.REJECTED
            if retrieval_id in self._retrieval_reports:
                self._retrieval_reports.move_to_end(retrieval_id)
                self._counters["duplicate_retrieval_reports"] += 1
                return RetrievalReportStatus.DUPLICATE
            self._remember_retrieval_report_locked(
                retrieval_id, RetrievalReportStatus.REJECTED.value, normalized_hash
            )
            if ambiguous:
                self._counters["ambiguous_attributions"] += 1
            self._counters["rejected_retrieval_reports"] += 1
            return RetrievalReportStatus.REJECTED

    def record_retrieval(
        self,
        ccr_hash: str,
        *,
        retrieved_tokens: int,
        round_trip_tokens: int = 0,
    ) -> bool:
        """Deprecated wrapper; ``round_trip_tokens`` is payload envelope only."""
        status = self.record_recovery(
            ccr_hash,
            retrieval_event_id=self.new_retrieval_event_id(),
            recovery_payload_tokens=max(int(retrieved_tokens), 0) + max(int(round_trip_tokens), 0),
        )
        return status is RetrievalReportStatus.ATTRIBUTED

    def record_retrieval_call_observation(self) -> None:
        with self._lock:
            self._counters["retrieval_call_observations"] += 1

    def record_internal_store_access(self) -> None:
        with self._lock:
            self._counters["internal_store_accesses"] += 1

    def record_client_inline_expansion(self, payload_tokens: int) -> None:
        """Record client-visible outbound expansion, never model recovery."""
        with self._lock:
            self._counters["client_inline_expansion_events"] += 1
            self._counters["client_inline_expansion_tokens"] += max(int(payload_tokens), 0)

    def record_http_transport_retrieval(self, payload_tokens: int) -> None:
        """Record raw HTTP retrieval separately from model input."""
        with self._lock:
            self._counters["http_transport_retrieval_events"] += 1
            self._counters["http_transport_payload_tokens"] += max(int(payload_tokens), 0)

    def tool_stats(
        self, tool_name: str | None, *, now: float | None = None
    ) -> ToolCompressionStats:
        label = (tool_name or "unknown").strip() or "unknown"
        current = self._clock() if now is None else float(now)
        with self._lock:
            relevant = [
                event
                for event in self._events.values()
                if event.tool_name == label
                and event.action is CompressionAction.LOSSY
                and event.eligible_for_learning
            ]
        mature = [event for event in relevant if event.is_mature(current)]
        return self._stats_for_events(mature, len(relevant) - len(mature))

    def snapshot(self, recent_limit: int = 20) -> dict[str, Any]:
        now = self._clock()
        with self._lock:
            events = list(self._events.values())
            counters = dict(self._counters)
        by_tool: dict[str, list[CompressionCostEvent]] = {}
        for event in events:
            by_tool.setdefault(event.tool_name, []).append(event)
        gross = sum(event.gross_savings_tokens for event in events)
        recovery = sum(event.recovery_payload_tokens for event in events)
        action_counts = {action.value: 0 for action in CompressionAction}
        for event in events:
            action_counts[event.action.value] += 1
        return {
            "events": len(events),
            "action_counts": action_counts,
            "gross_savings_tokens": gross,
            "recovery_payload_tokens": recovery,
            "payload_net_savings_tokens": gross - recovery,
            "actual_recovery_events": sum(event.retrieval_count for event in events),
            "mature_lossy_events": sum(
                event.action is CompressionAction.LOSSY
                and event.eligible_for_learning
                and event.is_mature(now)
                for event in events
            ),
            "pending_lossy_events": sum(
                event.action is CompressionAction.LOSSY
                and event.eligible_for_learning
                and not event.is_mature(now)
                for event in events
            ),
            "attribution": counters,
            "non_model_retrievals": {
                "client_inline_expansion_events": counters["client_inline_expansion_events"],
                "client_inline_expansion_tokens": counters["client_inline_expansion_tokens"],
                "http_transport_retrieval_events": counters["http_transport_retrieval_events"],
                "http_transport_payload_tokens": counters["http_transport_payload_tokens"],
            },
            "idempotency": {
                "retained_report_ids": len(self._retrieval_reports),
                "retention_limit": self._max_retrieval_reports,
                "evictions": counters["idempotency_report_evictions"],
                "legacy_hash_hazards": len(self._legacy_hash_report_hazards),
                "legacy_hash_hazard_limit": self._max_retrieval_reports,
                "legacy_hash_hazard_evictions": counters["legacy_hash_hazard_evictions"],
            },
            "by_tool": {
                tool: self._stats_for_snapshot(tool_events, now).to_dict()
                for tool, tool_events in sorted(by_tool.items())
            },
            "recent": (
                [event.to_dict(now) for event in events[-recent_limit:]] if recent_limit > 0 else []
            ),
        }

    def clear(self) -> None:
        with self._lock:
            self._events.clear()
            self._hash_to_events.clear()
            self._retrieval_reports.clear()
            self._legacy_hash_report_hazards.clear()
            for name in self._counters:
                self._counters[name] = 0

    def _resolve_event_locked(
        self,
        ccr_hash: str,
        *,
        compression_event_id: str | None,
        session_id: str | None,
        request_id: str | None,
        tool_call_id: str | None,
        provider_slot: str | None,
    ) -> tuple[CompressionCostEvent | None, bool]:
        candidates = [
            self._events[event_id]
            for event_id in self._hash_to_events.get(ccr_hash, ())
            if event_id in self._events
        ]
        if compression_event_id:
            exact = self._events.get(compression_event_id)
            if exact is None or ccr_hash not in exact.ccr_hashes:
                return None, True
            return exact, False
        for field, value in (
            ("session_id", session_id),
            ("request_id", request_id),
            ("tool_call_id", tool_call_id),
            ("provider_slot", provider_slot),
        ):
            if value:
                candidates = [event for event in candidates if getattr(event, field) == value]
        if not candidates:
            return None, False
        if len(candidates) > 1:
            self._counters["ambiguous_attributions"] += 1
            # Charging the newest same-hash event would be deterministic but
            # not trustworthy. Require event/session/request/tool-call metadata.
            return None, True
        return candidates[0], False

    def _remember_retrieval_report_locked(
        self, retrieval_id: str, value: str, ccr_hash: str
    ) -> None:
        if retrieval_id in self._retrieval_reports:
            self._retrieval_reports.move_to_end(retrieval_id)
            self._retrieval_reports[retrieval_id] = (value, ccr_hash)
            return
        if len(self._retrieval_reports) >= self._max_retrieval_reports:
            _old_id, (old_value, old_hash) = self._retrieval_reports.popitem(last=False)
            hazard_hashes = {old_hash} if old_hash else set()
            for hazard_hash in hazard_hashes:
                self._remember_legacy_hash_hazard_locked(hazard_hash)
            self._counters["idempotency_report_evictions"] += 1
        self._retrieval_reports[retrieval_id] = (value, ccr_hash)

    def _remember_legacy_hash_hazard_locked(self, ccr_hash: str) -> None:
        """Retain a bounded fail-closed horizon for evicted hash-only retries."""
        if not ccr_hash:
            return
        if ccr_hash in self._legacy_hash_report_hazards:
            self._legacy_hash_report_hazards.move_to_end(ccr_hash)
            return
        if len(self._legacy_hash_report_hazards) >= self._max_retrieval_reports:
            self._legacy_hash_report_hazards.popitem(last=False)
            self._counters["legacy_hash_hazard_evictions"] += 1
        self._legacy_hash_report_hazards[ccr_hash] = None

    def _trim_locked(self) -> None:
        while len(self._events) > self._max_events:
            _, event = self._events.popitem(last=False)
            self._remove_event_indexes_locked(event)

    def _remove_event_indexes_locked(self, event: CompressionCostEvent) -> None:
        for ccr_hash in event.ccr_hashes:
            candidates = self._hash_to_events.get(ccr_hash)
            if candidates is None:
                continue
            remaining = deque(
                (value for value in candidates if value != event.compression_event_id),
                maxlen=self._max_hash_candidates,
            )
            if remaining:
                self._hash_to_events[ccr_hash] = remaining
            else:
                self._hash_to_events.pop(ccr_hash, None)
        # Bounded report IDs and hash hazards outlive event eviction, so delayed
        # legacy retries fail closed for the documented retention horizon.

    @staticmethod
    def _stats_for_events(events: list[CompressionCostEvent], pending: int) -> ToolCompressionStats:
        return ToolCompressionStats(
            mature_compressions=len(events),
            pending_compressions=pending,
            retrieved_compressions=sum(event.was_retrieved for event in events),
            retrieval_count=sum(event.retrieval_count for event in events),
            original_tokens=sum(event.original_tokens for event in events),
            initially_emitted_tokens=sum(event.initially_emitted_tokens for event in events),
            gross_savings_tokens=sum(event.gross_savings_tokens for event in events),
            recovery_payload_tokens=sum(event.recovery_payload_tokens for event in events),
        )

    @classmethod
    def _stats_for_snapshot(
        cls, events: list[CompressionCostEvent], now: float
    ) -> ToolCompressionStats:
        eligible = [
            event
            for event in events
            if event.action is CompressionAction.LOSSY and event.eligible_for_learning
        ]
        mature = [event for event in eligible if event.is_mature(now)]
        return cls._stats_for_events(mature, len(eligible) - len(mature))


# Deprecated type alias retained for in-process compatibility.
RealizedCostLedger = CompressionCostLedger

_EXACT_INTENT_RE = re.compile(
    r"\b(?:implement|edit|modify|patch|refactor|fix|debug|update|rewrite|write|change|repair)\b",
    re.IGNORECASE,
)
_ANALYSIS_INTENT_RE = re.compile(
    r"\b(?:analy[sz]e|review|audit|inspect|explain|investigate|understand)\b",
    re.IGNORECASE,
)
_DISPOSABLE_TOOL_RE = re.compile(
    r"(?:search|grep|\brg\b|glob|find|list|directory|index|schema)", re.IGNORECASE
)
_READ_TOOL_RE = re.compile(r"(?:^|[_-])(?:read|cat|view|open)(?:$|[_-])", re.IGNORECASE)
_SHELL_TOOL_RE = re.compile(
    r"(?:^|[_-])(?:exec_command|execute|shell|bash|terminal)(?:$|[_-])", re.IGNORECASE
)
_CCR_HASH_RE = re.compile(r"(?:<<ccr:|Retrieve (?:more|original): hash=)([0-9a-fA-F]{6,64})")


def extract_ccr_hashes(content: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys(match.group(1).lower() for match in _CCR_HASH_RE.finditer(content)))


def model_visible_mcp_payload(result: Mapping[str, Any]) -> dict[str, Any]:
    """Build the MCP envelope whose serialized content enters the model."""
    return {
        "content": [
            {
                "type": "text",
                "text": json.dumps(dict(result), ensure_ascii=False, indent=2, default=str),
            }
        ]
    }


def _predicted_mcp_recovery(original_content: str) -> dict[str, Any]:
    """Build the logical result returned by the MCP retrieval tool."""
    return {
        # Production CCR keys are 24 hexadecimal characters. Keeping the
        # placeholder at the real width makes the known-path envelope estimate
        # structurally faithful even though live provider IDs remain unknown.
        "hash": "0" * 24,
        "source": "local",
        "original_content": original_content,
        "original_item_count": 0,
        "compressed_item_count": 0,
        "retrieval_count": 1,
    }


def _predicted_automatic_recovery(original_content: str) -> dict[str, Any]:
    """Build the logical result used by automatic provider continuations."""
    return {
        "hash": "0" * 24,
        "original_content": original_content,
        "original_item_count": 0,
    }


def estimate_recovery_payload_tokens(
    original_content: str,
    *,
    path: str | RecoveryPayloadPath = RecoveryPayloadPath.UNKNOWN,
    fallback_envelope_tokens: int = 64,
) -> int:
    """Estimate model-injected recovery bytes in the ledger's byte/4 units.

    Known paths serialize the actual content into their provider shape, so JSON
    escaping and UTF-8 size affect the result. Unknown paths retain an explicit
    configurable envelope fallback. This is not a provider tokenizer estimate.
    """
    try:
        normalized = RecoveryPayloadPath(str(getattr(path, "value", path)).lower())
    except ValueError:
        normalized = RecoveryPayloadPath.UNKNOWN
    mcp_logical = _predicted_mcp_recovery(original_content)
    automatic_logical = _predicted_automatic_recovery(original_content)
    rendered = json.dumps(automatic_logical, ensure_ascii=False, indent=2)
    if normalized is RecoveryPayloadPath.MCP:
        payload: Any = model_visible_mcp_payload(mcp_logical)
    elif normalized is RecoveryPayloadPath.ANTHROPIC:
        payload = {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "call", "content": rendered}],
        }
    elif normalized is RecoveryPayloadPath.OPENAI_CHAT:
        payload = {"role": "tool", "tool_call_id": "call", "content": rendered}
    elif normalized is RecoveryPayloadPath.OPENAI_RESPONSES:
        payload = {"type": "function_call_output", "call_id": "call", "output": rendered}
    elif normalized is RecoveryPayloadPath.GOOGLE:
        payload = {
            "role": "user",
            "parts": [
                {
                    "functionResponse": {
                        "name": "headroom_retrieve",
                        "id": "call",
                        "response": automatic_logical,
                    }
                }
            ],
        }
    elif normalized is RecoveryPayloadPath.GENERIC:
        generic_items = [{"tool_call_id": "call", "result": rendered}]
        payload = {"role": "tool", "content": json.dumps(generic_items)}
    else:
        return estimate_payload_tokens(original_content) + max(int(fallback_envelope_tokens), 0)
    return estimate_payload_tokens(payload)


class RetrievalAwarePolicy:
    """Deterministic expected-input-token controller for structured outputs."""

    def __init__(
        self,
        ledger: CompressionCostLedger,
        config: RetrievalAwarePolicyConfig | None = None,
    ):
        self.ledger = ledger
        self.config = config or RetrievalAwarePolicyConfig()

    def decide(
        self,
        *,
        original_tokens: int,
        lossless_tokens: int,
        tool_name: str | None,
        query_context: str,
        tool_context: str = "",
        extra_turn_tokens: int | None = None,
        original_content: str | None = None,
        recovery_payload_path: str | RecoveryPayloadPath = RecoveryPayloadPath.UNKNOWN,
    ) -> RetrievalAwareDecision:
        original = max(int(original_tokens), 0)
        lossless = min(max(int(lossless_tokens), 0), original)
        probability, expected_count, source = self._retrieval_estimates(
            tool_name, query_context, tool_context
        )
        ratio = self._predicted_lossy_ratio(tool_name)
        predicted_lossy = min(original, max(1, round(original * ratio))) if original else 0
        path_value = str(getattr(recovery_payload_path, "value", recovery_payload_path)).lower()
        predicted_payload = (
            estimate_recovery_payload_tokens(
                original_content,
                path=path_value,
                fallback_envelope_tokens=(self.config.predicted_recovery_payload_envelope_tokens),
            )
            if original_content is not None
            else original + self.config.predicted_recovery_payload_envelope_tokens
        )
        extra_turn = (
            self.config.predicted_extra_turn_tokens
            if extra_turn_tokens is None
            else max(int(extra_turn_tokens), 0)
        )
        passthrough_cost = float(original)
        lossless_cost = float(lossless if lossless < original else original)
        lossy_cost = predicted_lossy + expected_count * predicted_payload + probability * extra_turn
        candidates = [(CompressionAction.PASSTHROUGH, passthrough_cost)]
        if lossless < original:
            candidates.append((CompressionAction.LOSSLESS, lossless_cost))
        candidates.append((CompressionAction.LOSSY, lossy_cost))
        action, best_cost = min(candidates, key=lambda item: (item[1], item[0].value))
        if original < self.config.min_original_tokens:
            action, reason = CompressionAction.PASSTHROUGH, "below_minimum_size"
        elif passthrough_cost - best_cost < self.config.min_expected_savings_tokens:
            action, reason = CompressionAction.PASSTHROUGH, "expected_saving_below_floor"
        else:
            reason = f"lowest_expected_cost:{action.value}"
        return RetrievalAwareDecision(
            action=action,
            original_tokens=original,
            lossless_tokens=lossless,
            predicted_lossy_tokens=predicted_lossy,
            probability_of_any_retrieval=probability,
            expected_retrieval_count=expected_count,
            predicted_recovery_payload_tokens=predicted_payload,
            recovery_payload_path=path_value,
            predicted_extra_turn_tokens=extra_turn,
            passthrough_cost=passthrough_cost,
            lossless_cost=lossless_cost,
            lossy_expected_cost=float(lossy_cost),
            probability_source=source,
            reason=reason,
        )

    def _cold_start_probability(
        self, tool_name: str | None, tool_context: str
    ) -> tuple[float, str]:
        label = tool_name or ""
        if _DISPOSABLE_TOOL_RE.search(f"{label} {tool_context}"):
            return self.config.disposable_tool_probability, "tool_class:disposable"
        if _READ_TOOL_RE.search(label):
            return self.config.read_tool_probability, "tool_class:read"
        if _SHELL_TOOL_RE.search(label):
            return self.config.shell_tool_probability, "tool_class:shell"
        return self.config.default_retrieval_probability, "cold_start_default"

    def _retrieval_estimates(
        self, tool_name: str | None, query_context: str, tool_context: str
    ) -> tuple[float, float, str]:
        prior, prior_source = self._cold_start_probability(tool_name, tool_context)
        label = (tool_name or "").strip()
        # Unknown events are diagnostic only and never become a global prior.
        stats = self.ledger.tool_stats(label) if label else ToolCompressionStats()
        if label and stats.mature_compressions >= self.config.min_history:
            strength = self.config.history_prior_strength
            denominator = stats.mature_compressions + strength
            probability = (stats.retrieved_compressions + strength * prior) / denominator
            expected_count = (
                stats.retrieval_count
                + strength * prior * self.config.cold_start_retrievals_given_retrieval
            ) / denominator
            source = f"history_smoothed:{prior_source}"
        else:
            probability = prior
            expected_count = prior * self.config.cold_start_retrievals_given_retrieval
            source = prior_source
        if _EXACT_INTENT_RE.search(query_context or ""):
            probability = max(probability, self.config.exact_intent_probability)
            expected_count = max(expected_count, probability)
            source += "+exact_intent"
        elif _ANALYSIS_INTENT_RE.search(query_context or ""):
            probability = max(probability, self.config.analysis_intent_probability)
            expected_count = max(expected_count, probability)
            source += "+analysis_intent"
        return min(max(probability, 0.0), 1.0), max(expected_count, 0.0), source

    def _predicted_lossy_ratio(self, tool_name: str | None) -> float:
        label = (tool_name or "").strip()
        stats = self.ledger.tool_stats(label) if label else ToolCompressionStats()
        ratio = (
            stats.average_compressed_ratio
            if label and stats.mature_compressions >= self.config.min_history
            else self.config.predicted_lossy_ratio
        )
        return min(max(ratio, 0.01), 1.0)


def estimate_payload_tokens(payload: Any) -> int:
    """Estimate serialized recovery-payload tokens in the ledger's units."""
    rendered = (
        payload
        if isinstance(payload, str)
        else json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=str)
    )
    return max(1, (len(rendered.encode("utf-8", errors="replace")) + 3) // 4) if rendered else 0


def entry_attribution(entry: Any) -> dict[str, str]:
    candidates = getattr(entry, "attribution_candidates", None) or []
    if candidates:
        from ..cache.compression_store import resolve_entry_attribution

        resolved = resolve_entry_attribution(entry)
        return {
            "compression_event_id": resolved["compression_event_id"],
            "session_id": resolved["session_id"],
            "request_id": resolved["request_id"],
            "tool_call_id": resolved["tool_call_id"],
            "provider_slot": resolved["provider_slot"],
        }
    return {
        "compression_event_id": str(getattr(entry, "compression_event_id", "") or ""),
        "session_id": str(getattr(entry, "session_id", "") or ""),
        "request_id": str(getattr(entry, "request_id", "") or ""),
        "tool_call_id": str(getattr(entry, "tool_call_id", "") or ""),
        "provider_slot": str(getattr(entry, "provider_slot", "") or ""),
    }


def account_recovery_payload_tokens(
    entry: Any,
    recovery_payload_tokens: int,
    *,
    retrieval_event_id: str | None = None,
    ledger: CompressionCostLedger | None = None,
) -> RetrievalReportStatus:
    """Charge an already-estimated model-input payload exactly once."""
    if ledger is None and not compression_cost_tracking_enabled():
        return RetrievalReportStatus.REJECTED
    target = ledger or get_compression_cost_ledger()
    return target.record_recovery(
        str(getattr(entry, "hash", "") or ""),
        retrieval_event_id=retrieval_event_id or target.new_retrieval_event_id(),
        recovery_payload_tokens=max(int(recovery_payload_tokens), 0),
        **entry_attribution(entry),
    )


def account_recovery_payload(
    entry: Any,
    payload: Any,
    *,
    retrieval_event_id: str | None = None,
    ledger: CompressionCostLedger | None = None,
) -> RetrievalReportStatus:
    """Charge content that was actually returned or injected into a model."""
    if ledger is None and not compression_cost_tracking_enabled():
        return RetrievalReportStatus.REJECTED
    return account_recovery_payload_tokens(
        entry,
        estimate_payload_tokens(payload),
        retrieval_event_id=retrieval_event_id,
        ledger=ledger,
    )


_GLOBAL_LEDGER = CompressionCostLedger(
    observation_window_seconds=RetrievalAwarePolicyConfig.from_env().observation_window_seconds
)
_TRACKING_ENABLED = False
_TRACKING_LOCK = threading.Lock()


def enable_compression_cost_tracking(enabled: bool = True) -> None:
    global _TRACKING_ENABLED
    with _TRACKING_LOCK:
        _TRACKING_ENABLED = bool(enabled)


def compression_cost_tracking_enabled() -> bool:
    with _TRACKING_LOCK:
        return _TRACKING_ENABLED


def get_compression_cost_ledger() -> CompressionCostLedger:
    return _GLOBAL_LEDGER


def enable_realized_cost_tracking(enabled: bool = True) -> None:
    """Deprecated alias for :func:`enable_compression_cost_tracking`."""
    enable_compression_cost_tracking(enabled)


def realized_cost_tracking_enabled() -> bool:
    """Deprecated alias for :func:`compression_cost_tracking_enabled`."""
    return compression_cost_tracking_enabled()


def get_realized_cost_ledger() -> CompressionCostLedger:
    """Deprecated alias for :func:`get_compression_cost_ledger`."""
    return _GLOBAL_LEDGER


def attribution_from_mapping(value: Mapping[str, Any] | None) -> dict[str, str]:
    source = value or {}
    return {
        key: str(source.get(key, "") or "")[:512]
        for key in ("session_id", "request_id", "tool_call_id", "provider_slot")
    }
