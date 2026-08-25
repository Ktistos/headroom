from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from headroom.cache.compression_store import CompressionStore
from headroom.transforms.compressor_registry import CompressInput
from headroom.transforms.content_router import (
    _ACTIVE_POLICY_ATTRIBUTION,
    _ACTIVE_POLICY_EVENT_IDS,
    ContentRouter,
    ContentRouterConfig,
    PolicySideEffectTransaction,
    RequestPolicySideEffectHolder,
    _adopt_policy_passthrough,
    _invoke_smart_crusher,
    activate_policy_side_effect_transaction,
    activate_request_policy_side_effect_holder,
)
from headroom.transforms.retrieval_aware_policy import (
    CompressionAction,
    CompressionCostLedger,
    RecoveryPayloadPath,
    RetrievalAwarePolicy,
    RetrievalAwarePolicyConfig,
    RetrievalReportStatus,
    account_recovery_payload,
    compression_cost_tracking_enabled,
    enable_compression_cost_tracking,
    estimate_payload_tokens,
    estimate_recovery_payload_tokens,
    extract_ccr_hashes,
    get_compression_cost_ledger,
    model_visible_mcp_payload,
    stable_retrieval_event_id,
)


@pytest.fixture(autouse=True)
def _reset_global_ledger():
    ledger = get_compression_cost_ledger()
    ledger.clear()
    enable_compression_cost_tracking(False)
    yield
    ledger.clear()
    enable_compression_cost_tracking(False)


def _lossy_event(
    ledger: CompressionCostLedger,
    *,
    tool: str = "catalog_search",
    ccr_hash: str = "abcdef123456",
    session: str = "session-a",
    eligible: bool = True,
) -> str:
    return ledger.record_outcome(
        tool_name=tool,
        strategy="row_drop",
        action=CompressionAction.LOSSY,
        predicted_action=CompressionAction.LOSSY,
        original_tokens=1000,
        initially_emitted_tokens=300,
        ccr_hashes=[ccr_hash],
        session_id=session,
        request_id=f"request-{session}",
        tool_call_id=f"call-{session}",
        eligible_for_learning=eligible,
    )


def _recover(
    ledger: CompressionCostLedger,
    ccr_hash: str,
    event_id: str,
    retrieval_id: str,
    tokens: int = 1064,
) -> RetrievalReportStatus:
    return ledger.record_recovery(
        ccr_hash,
        compression_event_id=event_id,
        retrieval_event_id=retrieval_id,
        recovery_payload_tokens=tokens,
    )


def test_deprecated_tracking_aliases_delegate_to_primary_api():
    from headroom.transforms.retrieval_aware_policy import (
        enable_realized_cost_tracking,
        get_realized_cost_ledger,
        realized_cost_tracking_enabled,
    )

    enable_realized_cost_tracking(True)
    assert realized_cost_tracking_enabled()
    assert get_realized_cost_ledger() is get_compression_cost_ledger()
    enable_compression_cost_tracking(False)
    assert not realized_cost_tracking_enabled()


@pytest.mark.parametrize("value", ["nan", "inf", "-inf"])
def test_nonfinite_observation_window_env_falls_back(monkeypatch, value: str):
    monkeypatch.setenv("HEADROOM_RETRIEVAL_OBSERVATION_WINDOW_SECONDS", value)

    config = RetrievalAwarePolicyConfig.from_env()

    assert config.observation_window_seconds == 300.0


def test_retrieval_aware_router_does_not_mutate_process_tracking_switch():
    assert not compression_cost_tracking_enabled()


def test_request_owned_transaction_rolls_back_synchronous_smart_crusher_effects():
    ledger = get_compression_cost_ledger()
    rows = [{"id": index, "value": "payload"} for index in range(80)]
    original = json.dumps(rows, indent=2)
    compact = json.dumps(rows, separators=(",", ":"))
    toin_calls: list[str] = []

    class FakeCrusher:
        def crush(self, content, **kwargs):
            if kwargs.get("lossless_only"):
                return SimpleNamespace(
                    original=content,
                    compressed=compact,
                    was_modified=True,
                    strategy="lossless_json",
                )
            return SimpleNamespace(
                original=content,
                compressed='[{"id":0},"<<ccr:abcdef123456 79_rows_offloaded>>"]',
                was_modified=True,
                strategy="row_drop",
            )

        def _record_to_toin(self, **_kwargs):
            toin_calls.append("recorded")

    router = ContentRouter(
        ContentRouterConfig(
            retrieval_aware_enabled=True,
            retrieval_aware_forced_action="lossy",
        )
    )
    router._smart_crusher = FakeCrusher()
    holder = RequestPolicySideEffectHolder()

    with activate_request_policy_side_effect_holder(holder):
        result = router.compress(original, tool_name="catalog_search")
        event_id = result.compression_event_ids[0]
        assert ledger.event_action(event_id) is CompressionAction.LOSSY
        assert toin_calls == []

    holder.finalize(commit=False)

    assert ledger.event_action(event_id) is None
    assert ledger.snapshot()["events"] == 0
    assert toin_calls == []

    router = ContentRouter(ContentRouterConfig(retrieval_aware_enabled=True))

    assert router._retrieval_aware_policy is not None
    assert not compression_cost_tracking_enabled()


def test_payload_accounting_uses_explicit_terms():
    ledger = CompressionCostLedger(observation_window_seconds=0)
    event_id = _lossy_event(ledger)
    assert _recover(ledger, "abcdef123456", event_id, "re-1") is RetrievalReportStatus.ATTRIBUTED

    snapshot = ledger.snapshot()
    assert snapshot["gross_savings_tokens"] == 700
    assert snapshot["recovery_payload_tokens"] == 1064
    assert snapshot["payload_net_savings_tokens"] == -364
    assert snapshot["actual_recovery_events"] == 1
    assert "realized_net_savings" not in snapshot


def test_gross_savings_is_exact_signed_subtraction():
    ledger = CompressionCostLedger(observation_window_seconds=0)
    ledger.record_outcome(
        tool_name="custom",
        strategy="expanded",
        action=CompressionAction.LOSSLESS,
        predicted_action=CompressionAction.LOSSLESS,
        original_tokens=100,
        initially_emitted_tokens=120,
    )
    snapshot = ledger.snapshot()
    assert snapshot["gross_savings_tokens"] == -20
    assert snapshot["payload_net_savings_tokens"] == -20


def test_probability_is_unique_events_while_count_tracks_repeats():
    ledger = CompressionCostLedger(observation_window_seconds=0)
    event_ids = [_lossy_event(ledger, ccr_hash=f"abcdef12345{i}") for i in range(3)]
    for index, event_id in enumerate(event_ids):
        assert (
            _recover(ledger, f"abcdef12345{index}", event_id, f"first-{index}").value
            == "attributed"
        )
    for repeat in range(4):
        assert (
            _recover(ledger, "abcdef123450", event_ids[0], f"repeat-{repeat}").value == "attributed"
        )

    stats = ledger.tool_stats("catalog_search")
    assert stats.retrieval_rate == 1.0
    assert stats.retrieved_compressions == 3
    assert stats.retrieval_count == 7
    assert stats.average_retrieval_count == pytest.approx(7 / 3)


def test_outcomes_mature_only_after_window_or_session_completion():
    now = [100.0]
    ledger = CompressionCostLedger(observation_window_seconds=10, clock=lambda: now[0])
    first = _lossy_event(ledger, session="open-session")
    assert ledger.tool_stats("catalog_search").mature_compressions == 0
    assert ledger.tool_stats("catalog_search").pending_compressions == 1

    assert ledger.mark_session_complete("open-session") == 1
    assert ledger.tool_stats("catalog_search").mature_compressions == 1

    _lossy_event(ledger, ccr_hash="abcdef654321", session="ttl-session")
    now[0] = 111.0
    assert ledger.tool_stats("catalog_search").mature_compressions == 2
    assert ledger.mark_event_mature(first)


def test_positive_retrieval_matures_immediately_but_negative_waits():
    now = [100.0]
    ledger = CompressionCostLedger(observation_window_seconds=100, clock=lambda: now[0])
    retrieved = _lossy_event(ledger, ccr_hash="abcdea123456")
    _lossy_event(ledger, ccr_hash="abcdea654321", session="still-open")
    assert ledger.tool_stats("catalog_search").mature_compressions == 0
    assert _recover(ledger, "abcdea123456", retrieved, "known-positive").value == "attributed"
    stats = ledger.tool_stats("catalog_search")
    assert stats.mature_compressions == 1
    assert stats.pending_compressions == 1
    assert stats.retrieval_rate == 1.0


def test_cold_start_prior_stays_separate_and_unknown_does_not_contaminate_named_tool():
    ledger = CompressionCostLedger(observation_window_seconds=0)
    for index in range(5):
        _lossy_event(ledger, tool="unknown", ccr_hash=f"ffffef12345{index}")
    policy = RetrievalAwarePolicy(ledger)

    decision = policy.decide(
        original_tokens=1000,
        lossless_tokens=650,
        tool_name="catalog_search",
        query_context="List matching records",
    )
    assert decision.probability_of_any_retrieval == pytest.approx(0.1)
    assert decision.probability_source == "tool_class:disposable"


def test_shadow_events_never_enter_empirical_history():
    ledger = CompressionCostLedger(observation_window_seconds=0)
    for index in range(3):
        event_id = _lossy_event(
            ledger,
            ccr_hash=f"abcdee12345{index}",
            eligible=False,
        )
        _recover(ledger, f"abcdee12345{index}", event_id, f"shadow-{index}")
    assert ledger.tool_stats("catalog_search").mature_compressions == 0
    assert ledger.snapshot()["mature_lossy_events"] == 0


def test_identical_hashes_are_attributed_by_event_or_session():
    ledger = CompressionCostLedger(observation_window_seconds=0)
    event_a = _lossy_event(ledger, session="a")
    event_b = _lossy_event(ledger, session="b")

    assert _recover(ledger, "abcdef123456", event_a, "retrieval-a").value == "attributed"
    assert (
        ledger.record_recovery(
            "abcdef123456",
            session_id="b",
            retrieval_event_id="retrieval-b",
            recovery_payload_tokens=20,
        ).value
        == "attributed"
    )
    recent = {event["compression_event_id"]: event for event in ledger.snapshot()["recent"]}
    assert recent[event_a]["recovery_payload_tokens"] == 1064
    assert recent[event_b]["recovery_payload_tokens"] == 20


def test_ambiguous_same_hash_report_is_rejected_without_metadata():
    ledger = CompressionCostLedger(observation_window_seconds=0)
    _lossy_event(ledger, session="a")
    _lossy_event(ledger, session="b")
    status = ledger.record_recovery(
        "abcdef123456",
        retrieval_event_id="ambiguous",
        recovery_payload_tokens=20,
    )
    assert status is RetrievalReportStatus.REJECTED
    snapshot = ledger.snapshot()
    assert snapshot["recovery_payload_tokens"] == 0
    assert snapshot["attribution"]["ambiguous_attributions"] == 1
    assert snapshot["attribution"]["rejected_retrieval_reports"] == 1


def test_legacy_unattributed_entry_is_not_overwritten_by_modern_same_content():
    store = CompressionStore(enable_feedback=False)
    hash_key = "abcdef654321"
    store.store(
        original="same original",
        compressed=f"<<ccr:{hash_key}>>",
        explicit_hash=hash_key,
    )
    handle = CompressionCostLedger.new_retrieval_handle()
    store.store(
        original="same original",
        compressed=f"<<ccr:{hash_key}@{handle}>>",
        explicit_hash=hash_key,
        compression_event_id="event-modern",
        retrieval_handle=handle,
    )

    assert store.get_retrieval_reference_status(hash_key)["status"] == "ambiguous"
    modern = store.retrieve_for_internal_use(hash_key, retrieval_handle=handle)
    assert modern is not None
    assert modern.compression_event_id == "event-modern"

    assert store.discard_attribution_candidate(hash_key, compression_event_id="event-modern")
    legacy = store.retrieve_for_internal_use(hash_key)
    assert legacy is not None
    assert legacy.compression_event_id is None
    assert legacy.retrieval_handle is None


def test_store_rejects_ambiguous_same_hash_and_resolves_session_candidate():
    ledger = CompressionCostLedger(observation_window_seconds=0)
    event_a = _lossy_event(ledger, session="a")
    event_b = _lossy_event(ledger, session="b")
    store = CompressionStore(enable_feedback=False)
    for event_id, session in ((event_a, "a"), (event_b, "b")):
        store.store(
            original="identical original",
            compressed="<<ccr:abcdef123456>>",
            explicit_hash="abcdef123456",
            compression_event_id=event_id,
            session_id=session,
            request_id=f"request-{session}",
            tool_call_id=f"call-{session}",
        )

    assert store.retrieve("abcdef123456") is None
    assert store.retrieve_for_internal_use("abcdef123456") is None
    assert store.get_retrieval_reference_status("abcdef123456")["status"] == "ambiguous"
    store_stats = store.get_stats()
    assert store_stats["ambiguous_retrieval_attempts"] == 2
    assert store_stats["rejected_retrieval_references"] == 2

    resolved = store.retrieve("abcdef123456", session_id="a")
    assert resolved is not None
    assert resolved.compression_event_id == event_a
    assert len(resolved.attribution_candidates) == 1
    assert (
        account_recovery_payload(
            resolved, {"original_content": "identical original"}, ledger=ledger
        )
        is RetrievalReportStatus.ATTRIBUTED
    )
    recent = {event["compression_event_id"]: event for event in ledger.snapshot()["recent"]}
    assert recent[event_a]["recovery_payload_tokens"] > 0
    assert recent[event_b]["recovery_payload_tokens"] == 0


def test_outer_gate_removes_never_emitted_store_candidate(monkeypatch):
    ledger = CompressionCostLedger(observation_window_seconds=0)
    event_id = _lossy_event(ledger)
    handle = ledger.new_retrieval_handle()
    store = CompressionStore(enable_feedback=False)
    store.store(
        original="never emitted",
        compressed=f"<<ccr:abcdef123456@{handle}>>",
        explicit_hash="abcdef123456",
        compression_event_id=event_id,
        retrieval_handle=handle,
    )
    monkeypatch.setattr("headroom.cache.compression_store.get_compression_store", lambda: store)
    router = ContentRouter(ContentRouterConfig())
    router._retrieval_aware_policy = RetrievalAwarePolicy(ledger)

    _adopt_policy_passthrough(router, [event_id], strategy="test_gate")

    assert store.get_entry_status("abcdef123456")["status"] == "missing"
    assert ledger.snapshot()["recent"][0]["action"] == "passthrough"


def test_retrieval_aware_netcost_rejection_does_not_cache_dead_event(
    monkeypatch,
):
    from headroom.transforms.content_detector import ContentType
    from headroom.transforms.content_router import (
        CompressionStrategy,
        RouterCompressionResult,
        RoutingDecision,
    )

    monkeypatch.setenv("HEADROOM_NET_COST_POLICY", "1")
    ledger = CompressionCostLedger(observation_window_seconds=0)
    router = ContentRouter(
        ContentRouterConfig(
            exclude_tools=set(),
            protect_recent_code=0,
            protect_analysis_context=False,
        )
    )
    router._retrieval_aware_policy = RetrievalAwarePolicy(ledger)
    gate_results = iter((False, True))
    monkeypatch.setattr(router, "_net_cost_allows", lambda **_kwargs: next(gate_results))
    calls = 0
    original = "ground truth " * 300
    compressed = "summary <<ccr:abcdef123456>>"

    def compress(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        event_id = ledger.record_outcome(
            tool_name="catalog",
            strategy="row_drop",
            action=CompressionAction.LOSSY,
            predicted_action=CompressionAction.LOSSY,
            original_tokens=1000,
            initially_emitted_tokens=100,
            ccr_hashes=["abcdef123456"],
        )
        return RouterCompressionResult(
            compressed=compressed,
            original=original,
            strategy_used=CompressionStrategy.SMART_CRUSHER,
            routing_log=[
                RoutingDecision(
                    content_type=ContentType.PLAIN_TEXT,
                    strategy=CompressionStrategy.SMART_CRUSHER,
                    original_tokens=1000,
                    compressed_tokens=100,
                )
            ],
            compression_event_ids=[event_id],
        )

    monkeypatch.setattr(router, "compress", compress)
    messages = [{"role": "tool", "tool_call_id": "call", "content": original}]
    first = router.apply(
        messages,
        _ProviderShapeCounter(),
        min_tokens_to_compress=1,
        frozen_message_count=0,
        request_id="same-request",
        session_id="same-session",
    )
    second = router.apply(
        messages,
        _ProviderShapeCounter(),
        min_tokens_to_compress=1,
        frozen_message_count=0,
        request_id="same-request",
        session_id="same-session",
    )

    assert first.messages[0]["content"] == original
    assert second.messages[0]["content"] == compressed
    assert calls == 2
    assert [event["action"] for event in ledger.snapshot()["recent"]] == [
        "passthrough",
        "lossy",
    ]


def test_outer_gate_adopts_passthrough_and_removes_recovery_attribution():
    ledger = CompressionCostLedger(observation_window_seconds=0)
    event_id = _lossy_event(ledger)

    assert ledger.adopt_passthrough(event_id)

    snapshot = ledger.snapshot()
    assert snapshot["action_counts"] == {"passthrough": 1, "lossless": 0, "lossy": 0}
    assert snapshot["gross_savings_tokens"] == 0
    assert snapshot["recent"][0]["strategy"] == "outer_gate_passthrough"
    assert (
        ledger.record_recovery(
            "abcdef123456",
            retrieval_event_id="no-longer-attributable",
            recovery_payload_tokens=20,
        )
        is RetrievalReportStatus.UNATTRIBUTED
    )


def test_cross_process_report_is_idempotent_even_after_event_discard():
    ledger = CompressionCostLedger(observation_window_seconds=0)
    event_id = _lossy_event(ledger)
    assert _recover(ledger, "abcdef123456", event_id, "stable-id", 50).value == "attributed"
    assert _recover(ledger, "abcdef123456", event_id, "stable-id", 50).value == "duplicate"
    assert ledger.discard_outcome(event_id)
    assert _recover(ledger, "abcdef123456", event_id, "stable-id", 50).value == "duplicate"
    assert ledger.snapshot()["attribution"]["duplicate_retrieval_reports"] == 2


def test_report_id_lru_evicts_and_event_scoped_accounting_continues():
    ledger = CompressionCostLedger(max_retrieval_reports=2, observation_window_seconds=0)
    event_ids = [
        _lossy_event(ledger, ccr_hash=f"abcde000000{index}", session=f"s-{index}")
        for index in range(4)
    ]
    for index, event_id in enumerate(event_ids):
        assert (
            _recover(
                ledger,
                f"abcde000000{index}",
                event_id,
                f"report-{index}",
                10 + index,
            )
            is RetrievalReportStatus.ATTRIBUTED
        )

    snapshot = ledger.snapshot()
    assert snapshot["actual_recovery_events"] == 4
    assert snapshot["recovery_payload_tokens"] == 46
    assert snapshot["idempotency"] == {
        "retained_report_ids": 2,
        "retention_limit": 2,
        "evictions": 2,
        "legacy_hash_hazards": 0,
        "legacy_hash_hazard_limit": 2,
        "legacy_hash_hazard_evictions": 0,
    }
    assert snapshot["attribution"]["dropped_retrieval_reports"] == 0

    # Event-scoped reports remain safe after eviction and do not contaminate
    # unrelated legacy hashes.
    assert (
        _recover(ledger, "abcde0000000", event_ids[0], "post-eviction", 7)
        is RetrievalReportStatus.ATTRIBUTED
    )
    fresh_event = _lossy_event(ledger, ccr_hash="fedcba654321", session="fresh")
    assert (
        ledger.record_recovery(
            "fedcba654321",
            retrieval_event_id="fresh-legacy-after-eviction",
            recovery_payload_tokens=7,
        )
        is RetrievalReportStatus.ATTRIBUTED
    )
    assert ledger.snapshot()["recent"][-1]["compression_event_id"] == fresh_event


def test_evicted_partial_scope_retry_cannot_charge_new_same_content_event():
    ledger = CompressionCostLedger(
        max_events=1,
        max_retrieval_reports=1,
        observation_window_seconds=0,
    )
    ccr_hash = "abcdef123456"
    _lossy_event(ledger, ccr_hash=ccr_hash, session="reused-session")
    assert (
        ledger.record_recovery(
            ccr_hash,
            retrieval_event_id="old-session-report",
            recovery_payload_tokens=11,
            session_id="reused-session",
        )
        is RetrievalReportStatus.ATTRIBUTED
    )

    new_event = _lossy_event(ledger, ccr_hash=ccr_hash, session="reused-session")
    ledger.record_rejected_retrieval_attempt(
        "feedface0000",
        retrieval_event_id="evict-old-report",
    )

    assert (
        ledger.record_recovery(
            ccr_hash,
            retrieval_event_id="old-session-report",
            recovery_payload_tokens=11,
            session_id="reused-session",
        )
        is RetrievalReportStatus.REJECTED
    )
    event = ledger.snapshot()["recent"][0]
    assert event["compression_event_id"] == new_event
    assert event["retrieval_count"] == 0
    assert event["recovery_payload_tokens"] == 0


def test_hash_only_retry_hazards_are_bounded_and_scoped_to_evicted_hashes():
    ledger = CompressionCostLedger(max_retrieval_reports=2, observation_window_seconds=0)
    hashes = [f"abcde100000{index}" for index in range(6)]
    for index, ccr_hash in enumerate(hashes):
        _lossy_event(ledger, ccr_hash=ccr_hash, session=f"legacy-{index}")
        assert (
            ledger.record_recovery(
                ccr_hash,
                retrieval_event_id=f"legacy-report-{index}",
                recovery_payload_tokens=10,
            )
            is RetrievalReportStatus.ATTRIBUTED
        )

    snapshot = ledger.snapshot()
    assert snapshot["actual_recovery_events"] == 6
    assert snapshot["idempotency"]["retained_report_ids"] == 2
    assert snapshot["idempotency"]["legacy_hash_hazards"] == 2
    assert snapshot["idempotency"]["legacy_hash_hazard_evictions"] == 2
    assert not hasattr(ledger, "_event_to_retrieval_reports")

    # The most recently evicted hash remains protected from a delayed retry.
    assert (
        ledger.record_recovery(
            hashes[-3],
            retrieval_event_id="delayed-legacy-retry",
            recovery_payload_tokens=10,
        )
        is RetrievalReportStatus.REJECTED
    )

    # A new distinct legacy event continues to account after both bounds roll.
    fresh_hash = "fedcba999999"
    _lossy_event(ledger, ccr_hash=fresh_hash, session="post-rollover")
    assert (
        ledger.record_recovery(
            fresh_hash,
            retrieval_event_id="post-rollover-fresh",
            recovery_payload_tokens=11,
        )
        is RetrievalReportStatus.ATTRIBUTED
    )


def test_rejected_and_unattributed_report_counters_are_explicit():
    ledger = CompressionCostLedger()
    assert (
        ledger.record_recovery(
            "deadbeef1234", retrieval_event_id="unknown", recovery_payload_tokens=1
        )
        is RetrievalReportStatus.UNATTRIBUTED
    )
    assert (
        ledger.record_recovery("", retrieval_event_id="bad", recovery_payload_tokens=-1)
        is RetrievalReportStatus.REJECTED
    )
    counters = ledger.snapshot()["attribution"]
    assert counters["unattributed_retrieval_reports"] == 1
    assert counters["rejected_retrieval_reports"] == 1


def test_expected_cost_formula_has_distinct_retrieval_and_extra_turn_terms():
    config = RetrievalAwarePolicyConfig(
        predicted_lossy_ratio=0.3,
        default_retrieval_probability=0.25,
        cold_start_retrievals_given_retrieval=2.0,
        predicted_recovery_payload_envelope_tokens=64,
        predicted_extra_turn_tokens=800,
    )
    decision = RetrievalAwarePolicy(CompressionCostLedger(), config).decide(
        original_tokens=1000,
        lossless_tokens=700,
        tool_name="custom_tool",
        query_context="",
    )
    assert decision.expected_retrieval_count == pytest.approx(0.5)
    assert decision.predicted_recovery_payload_tokens == 1064
    assert decision.lossy_expected_cost == pytest.approx(300 + 0.5 * 1064 + 0.25 * 800)


@pytest.mark.parametrize(
    ("content", "path"),
    [
        ('{"quoted":"' + '\\"' * 100 + '"}', RecoveryPayloadPath.MCP),
        ("plain text " * 100, RecoveryPayloadPath.OPENAI_CHAT),
        ("κόσμος 🌍" * 100, RecoveryPayloadPath.ANTHROPIC),
        (json.dumps({"nested": [{"value": "x" * 80}]}), RecoveryPayloadPath.OPENAI_RESPONSES),
    ],
)
def test_recovery_prediction_serializes_content_sensitive_provider_shapes(content, path):
    predicted = estimate_recovery_payload_tokens(content, path=path)
    raw_without_envelope = estimate_recovery_payload_tokens(
        content, path=RecoveryPayloadPath.UNKNOWN, fallback_envelope_tokens=0
    )
    assert predicted > raw_without_envelope


def test_recovery_prediction_is_sensitive_to_escaping_unicode_and_nesting():
    plain = "x" * 256
    escaped = '\\"' * 128
    unicode_content = "λ" * 256
    nested = json.dumps({"outer": [{"quoted": '"' * 64}]})

    assert estimate_recovery_payload_tokens(
        escaped, path=RecoveryPayloadPath.MCP
    ) > estimate_recovery_payload_tokens(plain, path=RecoveryPayloadPath.MCP)
    assert estimate_recovery_payload_tokens(
        unicode_content, path=RecoveryPayloadPath.MCP
    ) > estimate_recovery_payload_tokens(plain, path=RecoveryPayloadPath.MCP)
    assert estimate_recovery_payload_tokens(
        nested, path=RecoveryPayloadPath.MCP
    ) > estimate_recovery_payload_tokens("x" * len(nested), path=RecoveryPayloadPath.MCP)


def test_mcp_prediction_uses_production_hash_width_at_action_boundary():
    content = "x" * 1000
    legacy_logical = {
        "hash": "0" * 12,
        "source": "local",
        "original_content": content,
        "original_item_count": 0,
        "compressed_item_count": 0,
        "retrieval_count": 1,
    }
    legacy_payload = estimate_payload_tokens(model_visible_mcp_payload(legacy_logical))
    production_payload = estimate_recovery_payload_tokens(content, path=RecoveryPayloadPath.MCP)
    assert production_payload == legacy_payload + 3

    policy = RetrievalAwarePolicy(
        CompressionCostLedger(),
        RetrievalAwarePolicyConfig(
            min_original_tokens=1,
            min_expected_savings_tokens=0,
            predicted_lossy_ratio=0.3,
            default_retrieval_probability=1.0,
            cold_start_retrievals_given_retrieval=1.0,
            predicted_extra_turn_tokens=0,
        ),
    )
    decision = policy.decide(
        original_tokens=1000,
        lossless_tokens=608,
        tool_name="custom_tool",
        query_context="",
        original_content=content,
        recovery_payload_path=RecoveryPayloadPath.MCP,
    )
    assert 300 + legacy_payload < decision.lossless_cost < decision.lossy_expected_cost
    assert decision.action is CompressionAction.LOSSLESS


def test_recovery_prediction_is_path_aware():
    content = '{"value": "x"}' * 20
    estimates = {
        estimate_recovery_payload_tokens(content, path=path)
        for path in (
            RecoveryPayloadPath.MCP,
            RecoveryPayloadPath.ANTHROPIC,
            RecoveryPayloadPath.OPENAI_CHAT,
            RecoveryPayloadPath.OPENAI_RESPONSES,
            RecoveryPayloadPath.GOOGLE,
        )
    }
    assert len(estimates) >= 3


def test_mcp_envelope_prediction_can_change_the_action_boundary():
    content = json.dumps({"escaped": '\\"' * 900})
    config = RetrievalAwarePolicyConfig(
        min_original_tokens=1,
        min_expected_savings_tokens=0,
        predicted_lossy_ratio=0.3,
        default_retrieval_probability=0.25,
        cold_start_retrievals_given_retrieval=1.0,
        predicted_recovery_payload_envelope_tokens=0,
        predicted_extra_turn_tokens=0,
    )
    policy = RetrievalAwarePolicy(CompressionCostLedger(), config)
    fallback = policy.decide(
        original_tokens=1000,
        lossless_tokens=760,
        tool_name="custom_tool",
        query_context="",
    )
    shaped = policy.decide(
        original_tokens=1000,
        lossless_tokens=760,
        tool_name="custom_tool",
        query_context="",
        original_content=content,
        recovery_payload_path=RecoveryPayloadPath.MCP,
    )
    assert shaped.predicted_recovery_payload_tokens > fallback.predicted_recovery_payload_tokens
    assert fallback.action is CompressionAction.LOSSY
    assert shaped.action is CompressionAction.LOSSLESS


def test_request_completion_does_not_mature_cross_request_continuation():
    now = [10.0]
    ledger = CompressionCostLedger(observation_window_seconds=30, clock=lambda: now[0])
    event_id = _lossy_event(ledger, ccr_hash="abcde0999999", session="conversation")

    # No request-end hook marks this negative: retrieval can happen in a later
    # continuation that shares the session.
    now[0] = 20.0
    assert ledger.tool_stats("catalog_search").pending_compressions == 1
    assert (
        _recover(ledger, "abcde0999999", event_id, "delayed-continuation", 25)
        is RetrievalReportStatus.ATTRIBUTED
    )
    stats = ledger.tool_stats("catalog_search")
    assert stats.mature_compressions == 1
    assert stats.retrieved_compressions == 1


def test_minimum_savings_floor_is_relative_to_passthrough():
    config = RetrievalAwarePolicyConfig(
        min_original_tokens=1,
        min_expected_savings_tokens=50,
        default_retrieval_probability=0,
        predicted_lossy_ratio=0.97,
        predicted_extra_turn_tokens=0,
        predicted_recovery_payload_envelope_tokens=0,
    )
    decision = RetrievalAwarePolicy(CompressionCostLedger(), config).decide(
        original_tokens=1000,
        lossless_tokens=1000,
        tool_name="custom",
        query_context="",
    )
    assert decision.action is CompressionAction.PASSTHROUGH
    assert decision.reason == "expected_saving_below_floor"


def test_warm_three_outcome_history_changes_estimate_and_action():
    ledger = CompressionCostLedger(observation_window_seconds=0)
    policy = RetrievalAwarePolicy(ledger)
    cold = policy.decide(
        original_tokens=1000,
        lossless_tokens=650,
        tool_name="catalog_search",
        query_context="List matching records",
    )
    assert cold.action is CompressionAction.LOSSY
    for index in range(3):
        ccr_hash = f"aaaadef1234{index}"
        event_id = _lossy_event(ledger, ccr_hash=ccr_hash)
        _recover(ledger, ccr_hash, event_id, f"warm-{index}")
    warm = policy.decide(
        original_tokens=1000,
        lossless_tokens=650,
        tool_name="catalog_search",
        query_context="List matching records",
    )
    assert warm.probability_source.startswith("history_smoothed:")
    assert warm.probability_of_any_retrieval > cold.probability_of_any_retrieval
    assert warm.action is CompressionAction.LOSSLESS


def test_exact_intent_and_disposable_tool_priors_choose_different_actions():
    policy = RetrievalAwarePolicy(CompressionCostLedger())
    exact = policy.decide(
        original_tokens=1000,
        lossless_tokens=650,
        tool_name="Read",
        query_context="Implement the fix using the exact records",
    )
    disposable = policy.decide(
        original_tokens=1000,
        lossless_tokens=650,
        tool_name="shell",
        tool_context="rg --files | head",
        query_context="List matches",
    )
    assert exact.action is CompressionAction.LOSSLESS
    assert exact.probability_of_any_retrieval >= 0.8
    assert disposable.action is CompressionAction.LOSSY
    assert disposable.probability_source == "tool_class:disposable"


def test_default_policy_selects_passthrough_below_size_threshold():
    decision = RetrievalAwarePolicy(CompressionCostLedger()).decide(
        original_tokens=180,
        lossless_tokens=130,
        tool_name="shell",
        query_context="List records",
        tool_context="rg schema",
    )
    assert decision.action is CompressionAction.PASSTHROUGH
    assert decision.reason == "below_minimum_size"


def test_content_router_reuses_validated_lossless_preview_and_threads_tool_name():
    ledger = get_compression_cost_ledger()
    rows = [{"id": index, "value": "payload"} for index in range(60)]
    original = json.dumps(rows, indent=2)
    compact = json.dumps(rows, separators=(",", ":"))
    calls: list[bool | None] = []

    class FakeCrusher:
        def crush(self, content, **kwargs):
            calls.append(kwargs.get("lossless_only"))
            if kwargs.get("lossless_only"):
                return SimpleNamespace(compressed=compact, strategy="lossless_json")
            raise AssertionError("lossy execution must not run")

    router = ContentRouter(ContentRouterConfig(retrieval_aware_enabled=True, enable_kompress=False))
    router._smart_crusher = FakeCrusher()
    result = router.compress(
        original,
        context="Implement the exact catalog fix",
        tool_name="Read",
        attribution={"session_id": "s", "request_id": "r", "tool_call_id": "c"},
    )
    assert result.compressed == compact
    assert calls == [True]
    event = ledger.snapshot()["recent"][0]
    assert event["tool_name"] == "Read"
    assert event["session_id"] == "s"
    assert event["request_id"] == "r"
    assert event["tool_call_id"] == "c"


def test_content_router_restores_policy_context_after_exception(monkeypatch):
    router = ContentRouter(ContentRouterConfig(enable_kompress=False))
    event_token = _ACTIVE_POLICY_EVENT_IDS.set(["outer-event"])
    attribution_token = _ACTIVE_POLICY_ATTRIBUTION.set({"session_id": "outer-session"})

    def fail(*_args, **_kwargs):
        raise RuntimeError("compression failed")

    monkeypatch.setattr(router, "_compress_with_active_policy_context", fail)
    try:
        with pytest.raises(RuntimeError, match="compression failed"):
            router.compress(
                "payload",
                attribution={"session_id": "inner-session"},
            )
        assert _ACTIVE_POLICY_EVENT_IDS.get() == ["outer-event"]
        assert _ACTIVE_POLICY_ATTRIBUTION.get() == {"session_id": "outer-session"}
    finally:
        _ACTIVE_POLICY_ATTRIBUTION.reset(attribution_token)
        _ACTIVE_POLICY_EVENT_IDS.reset(event_token)


def test_store_internal_or_plain_retrieve_does_not_charge_payload_twice():
    ledger = CompressionCostLedger(observation_window_seconds=0)
    event_id = _lossy_event(ledger)
    store = CompressionStore(enable_feedback=False)
    store.store(
        original="x" * 4000,
        compressed="<<ccr:abcdef123456>>",
        explicit_hash="abcdef123456",
        compression_event_id=event_id,
        session_id="session-a",
        request_id="request-session-a",
        tool_call_id="call-session-a",
    )
    assert store.retrieve_for_internal_use("abcdef123456") is not None
    entry = store.retrieve("abcdef123456")
    assert entry is not None
    assert ledger.snapshot()["recovery_payload_tokens"] == 0
    account_recovery_payload(entry, {"original_content": entry.original_content}, ledger=ledger)
    charged = ledger.snapshot()["recovery_payload_tokens"]
    assert charged > 0
    assert store.retrieve_for_internal_use("abcdef123456") is not None
    assert ledger.snapshot()["recovery_payload_tokens"] == charged


def test_marker_extraction_supports_all_ccr_shapes():
    content = (
        "<<ccr:abcdef123456 90_rows_offloaded>> "
        "Retrieve more: hash=ABCDEF999999 "
        "Retrieve original: hash=abcdef123456"
    )
    assert extract_ccr_hashes(content) == ("abcdef123456", "abcdef999999")


def test_observe_mode_records_prediction_but_never_learns_from_shadow():
    ledger = get_compression_cost_ledger()
    original = json.dumps([{"id": i, "value": "x"} for i in range(80)], indent=2)
    compact = json.dumps(json.loads(original), separators=(",", ":"))

    class FakeCrusher:
        def crush(self, content, **kwargs):
            if kwargs.get("lossless_only"):
                return SimpleNamespace(compressed=compact, strategy="lossless_json")
            return SimpleNamespace(
                compressed=json.dumps([{"_ccr": "<<ccr:abcdef123456 70_rows_offloaded>>"}]),
                strategy="row_drop",
            )

    router = SimpleNamespace(
        _get_smart_crusher=lambda: FakeCrusher(),
        _retrieval_aware_policy=RetrievalAwarePolicy(ledger),
        config=SimpleNamespace(retrieval_aware_observe_only=True),
    )
    output = _invoke_smart_crusher(
        router,
        CompressInput(
            content=original,
            content_type="application/json",
            query="Implement the exact fix",
            config={"tool_name": "Read"},
        ),
    )
    assert output is not None and "<<ccr:" in output
    event = ledger.snapshot()["recent"][0]
    assert event["predicted_action"] == "lossless"
    assert event["action"] == "lossy"
    assert event["eligible_for_learning"] is False
    assert ledger.tool_stats("Read").mature_compressions == 0


class _ProviderShapeCounter:
    def count_text(self, text: str) -> int:
        return max(len(text) // 4, 1)

    def count_messages(self, messages) -> int:
        total = 0
        for message in messages:
            content = message.get("content", "")
            if isinstance(content, str):
                total += self.count_text(content)
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        value = block.get("content") or block.get("text") or ""
                        if isinstance(value, str):
                            total += self.count_text(value)
        return total


@pytest.mark.parametrize("provider_shape", ["openai_chat", "anthropic"])
def test_chat_and_anthropic_tool_results_propagate_tool_attribution(provider_shape):
    from headroom.transforms.content_detector import ContentType
    from headroom.transforms.content_router import (
        CompressionStrategy,
        RouterCompressionResult,
        RoutingDecision,
    )

    router = ContentRouter(
        ContentRouterConfig(
            exclude_tools=set(),
            protect_recent_code=0,
            protect_analysis_context=False,
            min_chars_for_block_compression=1,
        )
    )
    seen: list[tuple[str | None, dict[str, str] | None]] = []
    original = json.dumps([{"id": i, "value": "payload"} for i in range(100)])

    def compress(content, **kwargs):
        seen.append((kwargs.get("tool_name"), kwargs.get("attribution")))
        compressed = '[{"kept":true,"marker":"<<ccr:abcdef123456 90_rows_offloaded>>"}]'
        return RouterCompressionResult(
            compressed=compressed,
            original=content,
            strategy_used=CompressionStrategy.SMART_CRUSHER,
            routing_log=[
                RoutingDecision(
                    content_type=ContentType.JSON_ARRAY,
                    strategy=router._strategy_from_detection_type(ContentType.JSON_ARRAY),
                    original_tokens=1000,
                    compressed_tokens=100,
                )
            ],
        )

    # Bind a plain callable because ContentRouter invokes the instance attribute.
    router.compress = compress
    if provider_shape == "openai_chat":
        messages = [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_chat",
                        "type": "function",
                        "function": {
                            "name": "catalog_reader",
                            "arguments": '{"path":"chat.json"}',
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_chat", "content": original},
        ]
        expected_call_id = "call_chat"
    else:
        messages = [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "call_anthropic",
                        "name": "catalog_reader",
                        "input": {"path": "anthropic.json"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "call_anthropic",
                        "content": original,
                    }
                ],
            },
        ]
        expected_call_id = "call_anthropic"

    router.apply(
        messages,
        _ProviderShapeCounter(),
        context="list catalog records",
        min_tokens_to_compress=1,
        frozen_message_count=0,
        request_id="provider-request",
        session_id="provider-session",
    )

    assert seen
    tool_name, attribution = seen[-1]
    assert tool_name == "catalog_reader"
    assert attribution is not None
    assert attribution["tool_call_id"] == expected_call_id
    assert attribution["request_id"] == "provider-request"
    assert attribution["session_id"] == "provider-session"


def test_streaming_observation_is_diagnostic_only_in_retrieval_aware_mode(monkeypatch):
    store = CompressionStore(enable_feedback=True)
    hash_key = "abcdef777777"
    handle = "rh-" + "7" * 32
    store.store(
        original="payload",
        compressed=f"<<ccr:{hash_key}@{handle}>>",
        explicit_hash=hash_key,
        compression_event_id="event-stream-observation",
        retrieval_handle=handle,
        tool_name="catalog_reader",
    )
    feedback_flushes: list[bool] = []
    monkeypatch.setattr(
        store,
        "process_pending_feedback",
        lambda: feedback_flushes.append(True),
    )

    enable_compression_cost_tracking(True)
    assert store.observe_retrieval_call(hash_key, retrieval_handle=handle)
    assert feedback_flushes == []
    assert store.get_retrieval_events() == []
    assert (
        get_compression_cost_ledger().snapshot()["attribution"]["retrieval_call_observations"] == 1
    )

    # Feature-off behavior remains backward compatible: the existing Headroom
    # feedback path still receives its model-call demand signal.
    enable_compression_cost_tracking(False)
    assert store.observe_retrieval_call(hash_key, retrieval_handle=handle)
    assert feedback_flushes == [True]
    events = store.get_retrieval_events()
    assert len(events) == 1
    assert events[0].retrieval_type == "observed_tool_call"


def test_ambiguous_streaming_observation_fails_closed_without_learning():
    store = CompressionStore(enable_feedback=False)
    hash_key = "abcdef123456"
    handles = [f"rh-{index:032x}" for index in range(2)]
    for index, handle in enumerate(handles):
        store.store(
            original="identical",
            compressed=f"<<ccr:{hash_key}@{handle}>>",
            explicit_hash=hash_key,
            compression_event_id=f"event-{index}",
            retrieval_handle=handle,
            tool_name=f"tool-{index}",
        )

    assert store.observe_retrieval_call(hash_key) is False
    stats = store.get_stats()
    assert stats["ambiguous_retrieval_attempts"] == 1
    assert stats["rejected_retrieval_references"] == 1

    assert store.observe_retrieval_call(hash_key, retrieval_handle=handles[0]) is True
    entry = store.retrieve_for_internal_use(hash_key, retrieval_handle=handles[0])
    assert entry is not None
    assert entry.compression_event_id == "event-0"
    assert entry.retrieval_count == 0


def test_rejected_attempt_ids_are_retry_idempotent_without_maturing_events():
    ledger = CompressionCostLedger(observation_window_seconds=0)
    _lossy_event(ledger, session="a")
    _lossy_event(ledger, session="b")
    retrieval_id = stable_retrieval_event_id("openai", "call-1", "abcdef123456", "legacy")

    first = ledger.record_rejected_retrieval_attempt(
        "abcdef123456", retrieval_event_id=retrieval_id, ambiguous=True
    )
    second = ledger.record_rejected_retrieval_attempt(
        "abcdef123456", retrieval_event_id=retrieval_id, ambiguous=True
    )

    assert first is RetrievalReportStatus.REJECTED
    assert second is RetrievalReportStatus.DUPLICATE
    snapshot = ledger.snapshot()
    assert snapshot["actual_recovery_events"] == 0
    assert snapshot["recovery_payload_tokens"] == 0
    assert snapshot["attribution"]["ambiguous_attributions"] == 1
    assert snapshot["attribution"]["rejected_retrieval_reports"] == 1
    assert snapshot["attribution"]["duplicate_retrieval_reports"] == 1


def test_known_recovery_predictions_match_production_provider_shapes():
    original = '{"quoted":"\\"λ"}'
    automatic = {
        "hash": "0" * 24,
        "original_content": original,
        "original_item_count": 0,
    }
    rendered = json.dumps(automatic, ensure_ascii=False, indent=2)
    mcp_logical = {
        "hash": "0" * 24,
        "source": "local",
        "original_content": original,
        "original_item_count": 0,
        "compressed_item_count": 0,
        "retrieval_count": 1,
    }
    expected = {
        RecoveryPayloadPath.MCP: model_visible_mcp_payload(mcp_logical),
        RecoveryPayloadPath.ANTHROPIC: {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "call", "content": rendered}],
        },
        RecoveryPayloadPath.OPENAI_CHAT: {
            "role": "tool",
            "tool_call_id": "call",
            "content": rendered,
        },
        RecoveryPayloadPath.OPENAI_RESPONSES: {
            "type": "function_call_output",
            "call_id": "call",
            "output": rendered,
        },
        RecoveryPayloadPath.GOOGLE: {
            "role": "user",
            "parts": [
                {
                    "functionResponse": {
                        "name": "headroom_retrieve",
                        "id": "call",
                        "response": automatic,
                    }
                }
            ],
        },
        RecoveryPayloadPath.GENERIC: {
            "role": "tool",
            "content": json.dumps([{"tool_call_id": "call", "result": rendered}]),
        },
    }

    for path, payload in expected.items():
        assert estimate_recovery_payload_tokens(original, path=path) == estimate_payload_tokens(
            payload
        )


def test_content_router_ratio_gate_reconciles_unemitted_policy_event(monkeypatch):
    from headroom.transforms.content_detector import ContentType
    from headroom.transforms.content_router import (
        CompressionStrategy,
        RouterCompressionResult,
        RoutingDecision,
    )

    ledger = CompressionCostLedger(observation_window_seconds=0)
    event_id = _lossy_event(ledger)
    original = "ground truth " * 200
    compressed = "summary <<ccr:abcdef123456>>"
    router = ContentRouter(
        ContentRouterConfig(
            exclude_tools=set(),
            protect_recent_code=0,
            protect_analysis_context=False,
            min_ratio_relaxed=0.5,
            min_ratio_aggressive=0.5,
        )
    )
    router._retrieval_aware_policy = RetrievalAwarePolicy(ledger)

    monkeypatch.setattr(
        router,
        "compress",
        lambda *_args, **_kwargs: RouterCompressionResult(
            compressed=compressed,
            original=original,
            strategy_used=CompressionStrategy.SMART_CRUSHER,
            routing_log=[
                RoutingDecision(
                    content_type=ContentType.PLAIN_TEXT,
                    strategy=CompressionStrategy.SMART_CRUSHER,
                    original_tokens=1000,
                    compressed_tokens=900,
                )
            ],
            compression_event_ids=[event_id],
        ),
    )

    result = router.apply(
        [{"role": "tool", "tool_call_id": "call", "content": original}],
        _ProviderShapeCounter(),
        min_tokens_to_compress=1,
        frozen_message_count=0,
    )

    assert result.messages[0]["content"] == original
    event = ledger.snapshot()["recent"][0]
    assert event["action"] == "passthrough"
    assert event["strategy"] == "ratio_gate_passthrough"
    assert event["gross_savings_tokens"] == 0
    assert event["hashes"] == []


def test_snapshot_zero_recent_limit_returns_no_events():
    ledger = CompressionCostLedger()
    _lossy_event(ledger)
    assert ledger.snapshot(recent_limit=0)["recent"] == []


def test_retrieval_aware_toin_waits_for_transaction_adoption():
    ledger = CompressionCostLedger()
    original = json.dumps([{"id": i, "value": "x"} for i in range(80)])
    compact = json.dumps(json.loads(original), separators=(",", ":"))
    toin_calls: list[dict] = []

    class FakeCrusher:
        def crush(self, content, **kwargs):
            if kwargs.get("lossless_only"):
                return SimpleNamespace(
                    compressed=compact,
                    original=content,
                    strategy="lossless_json",
                    was_modified=True,
                )
            return SimpleNamespace(
                compressed=json.dumps([{"_ccr": "<<ccr:abcdef123456 70_rows_offloaded>>"}]),
                original=content,
                strategy="row_drop",
                was_modified=True,
            )

        def _record_to_toin(self, **kwargs):
            toin_calls.append(kwargs)

    router = SimpleNamespace(
        _get_smart_crusher=lambda: FakeCrusher(),
        _retrieval_aware_policy=RetrievalAwarePolicy(ledger),
        config=SimpleNamespace(
            retrieval_aware_observe_only=False,
            retrieval_aware_forced_action="lossy",
        ),
    )
    transaction = PolicySideEffectTransaction()
    with activate_policy_side_effect_transaction(transaction):
        output = _invoke_smart_crusher(
            router,
            CompressInput(
                content=original,
                content_type="application/json",
                query="find a row",
                config={"tool_name": "catalog_search"},
            ),
        )

    assert output is not None and "<<ccr:" in output
    assert toin_calls == []
    transaction.commit()
    assert len(toin_calls) == 1
    assert ledger.snapshot()["action_counts"]["lossy"] == 1


@pytest.mark.parametrize("evict_before_discard", [False, True])
def test_cancelled_transaction_discards_event_store_candidate_and_toin(
    monkeypatch, evict_before_discard: bool
):
    from headroom.cache.compression_store import CompressionStore

    ledger = (
        CompressionCostLedger(max_events=1) if evict_before_discard else CompressionCostLedger()
    )
    store = CompressionStore()
    monkeypatch.setattr("headroom.cache.compression_store.get_compression_store", lambda: store)
    original = json.dumps([{"id": i, "value": "x"} for i in range(80)])
    toin_calls: list[dict] = []

    class FakeCrusher:
        def crush(self, content, **kwargs):
            if kwargs.get("lossless_only"):
                return SimpleNamespace(
                    compressed=content,
                    original=content,
                    strategy="passthrough",
                    was_modified=False,
                )
            compressed = json.dumps([{"_ccr": "<<ccr:abcdef123456 70_rows_offloaded>>"}])
            store.store(
                content,
                compressed,
                compression_event_id=kwargs["compression_event_id"],
                retrieval_handle=kwargs["retrieval_handle"],
                explicit_hash="abcdef123456",
            )
            return SimpleNamespace(
                compressed=compressed,
                original=content,
                strategy="row_drop",
                was_modified=True,
            )

        def _record_to_toin(self, **kwargs):
            toin_calls.append(kwargs)

    router = SimpleNamespace(
        _get_smart_crusher=lambda: FakeCrusher(),
        _retrieval_aware_policy=RetrievalAwarePolicy(ledger),
        config=SimpleNamespace(
            retrieval_aware_observe_only=False,
            retrieval_aware_forced_action="lossy",
        ),
    )
    transaction = PolicySideEffectTransaction()
    with activate_policy_side_effect_transaction(transaction):
        _invoke_smart_crusher(
            router,
            CompressInput(
                content=original,
                content_type="application/json",
                query="find a row",
                config={"tool_name": "catalog_search"},
            ),
        )
        assert store.retrieve_for_internal_use("abcdef123456") is not None
    if evict_before_discard:
        ledger.record_outcome(
            tool_name="newer",
            strategy="passthrough",
            action=CompressionAction.PASSTHROUGH,
            predicted_action=CompressionAction.PASSTHROUGH,
            original_tokens=1,
            initially_emitted_tokens=1,
            compression_event_id="newer-event",
        )
    transaction.discard()

    snapshot = ledger.snapshot()
    assert snapshot["action_counts"] == {
        "passthrough": int(evict_before_discard),
        "lossless": 0,
        "lossy": 0,
    }
    assert snapshot["events"] == int(evict_before_discard)
    assert snapshot["gross_savings_tokens"] == 0
    assert store.retrieve_for_internal_use("abcdef123456") is None
    assert toin_calls == []


def test_retrieval_aware_observer_metrics_follow_transaction_adoption():
    from headroom.transforms.content_router import (
        CompressionStrategy,
        ContentType,
        RouterCompressionResult,
        RoutingDecision,
    )

    state: dict[str, list[object]] = {"compressions": [], "size_gates": []}

    class Observer:
        def record_compression(self, **kwargs):
            state["compressions"].append(kwargs)

        def record_kompress_size_gate(self, outcome):
            state["size_gates"].append(outcome)

    router = ContentRouter.__new__(ContentRouter)
    router._observer = Observer()
    router._retrieval_aware_policy = object()
    result = RouterCompressionResult(
        compressed="short",
        original="original content",
        strategy_used=CompressionStrategy.TEXT,
        routing_log=[
            RoutingDecision(
                content_type=ContentType.PLAIN_TEXT,
                strategy=CompressionStrategy.TEXT,
                original_tokens=10,
                compressed_tokens=3,
            )
        ],
    )

    discarded = PolicySideEffectTransaction()
    with activate_policy_side_effect_transaction(discarded):
        router._observe(result)
        router._observe_kompress_size_gate("within")
        assert state == {"compressions": [], "size_gates": []}
    discarded.discard()
    assert state == {"compressions": [], "size_gates": []}

    adopted = PolicySideEffectTransaction()
    with activate_policy_side_effect_transaction(adopted):
        router._observe(result)
        router._observe_kompress_size_gate("exceeded")
        assert state == {"compressions": [], "size_gates": []}
    adopted.commit()

    assert state["compressions"] == [
        {
            "strategy": "text",
            "original_tokens": 10,
            "compressed_tokens": 3,
        }
    ]
    assert state["size_gates"] == ["exceeded"]


def test_completed_deadline_child_loses_passthrough_discard_disposition():
    parent = PolicySideEffectTransaction()
    child = PolicySideEffectTransaction(adopt_policy_passthrough_on_discard=True)
    observed_dispositions: list[bool] = []
    child.register(
        "completed-deadline-child",
        commit=lambda: None,
        discard=lambda: observed_dispositions.append(child.adopt_policy_passthrough_on_discard),
    )

    parent.merge_from(child)
    parent.discard()

    assert observed_dispositions == [False]
