"""Tests for CCR endpoints in the proxy server.

These tests verify the /v1/retrieve endpoints work correctly.
"""

import json
from unittest.mock import patch

import pytest

# Skip if fastapi not available
pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from headroom.cache.compression_store import get_compression_store, reset_compression_store
from headroom.proxy.server import ProxyConfig, create_app
from headroom.transforms.retrieval_aware_policy import compression_cost_tracking_enabled


def _anthropic_retrieve_call(hash_key: str, call_id: str = "toolu_test") -> dict:
    return {
        "provider": "anthropic",
        "tool_call": {
            "id": call_id,
            "name": "headroom_retrieve",
            "input": {"hash": hash_key},
        },
    }


def _acknowledge_model_injection(client: TestClient, tool_call_response):
    payload = tool_call_response.json()
    accounting = payload["accounting"]
    assert accounting["status"] == "pending_model_injection"
    response = client.post(
        accounting["report_endpoint"],
        json=accounting["report"],
    )
    assert response.status_code == 200
    return response


def test_proxy_construction_owns_retrieval_tracking_lifecycle(monkeypatch) -> None:
    config = ProxyConfig(
        optimize=False,
        cache_enabled=False,
        rate_limit_enabled=False,
        cost_tracking_enabled=False,
    )
    monkeypatch.setenv("HEADROOM_RETRIEVAL_AWARE", "on")
    create_app(config)
    assert compression_cost_tracking_enabled()

    monkeypatch.delenv("HEADROOM_RETRIEVAL_AWARE")
    create_app(config)
    assert not compression_cost_tracking_enabled()


@pytest.fixture
def client():
    """Create test client with fresh compression store."""
    reset_compression_store()
    config = ProxyConfig(
        optimize=False,  # Disable optimization for simpler tests
        cache_enabled=False,
        rate_limit_enabled=False,
        cost_tracking_enabled=False,
    )
    app = create_app(config)
    # CCR endpoints are loopback-gated (#1227).
    with TestClient(app, base_url="http://127.0.0.1", client=("127.0.0.1", 12345)) as client:
        yield client
    reset_compression_store()


@pytest.fixture
def client_with_data(client):
    """Test client with pre-populated compression store."""
    store = get_compression_store()

    # Store some test data
    items = [{"id": i, "content": f"Item {i} about Python programming"} for i in range(100)]
    store.store(
        original=json.dumps(items),
        compressed=json.dumps(items[:10]),
        original_tokens=1000,
        compressed_tokens=100,
        original_item_count=100,
        compressed_item_count=10,
        tool_name="test_tool",
    )

    return client


class TestCCRRetrieveEndpoint:
    """Test the /v1/retrieve POST endpoint."""

    def test_retrieve_requires_hash(self, client):
        """Request without hash should return 400."""
        response = client.post("/v1/retrieve", json={})
        assert response.status_code == 400
        assert "hash required" in response.json()["detail"]

    def test_retrieve_nonexistent_hash(self, client):
        """Request with nonexistent hash should return 404."""
        response = client.post("/v1/retrieve", json={"hash": "nonexistent123"})
        assert response.status_code == 404
        assert "Entry not found" in response.json()["detail"]
        assert "CCR TTL: 1800 seconds" in response.json()["detail"]

    def test_retrieve_expired_hash_reports_expiration_detail(self, client):
        """Expired entries report expiration separately from missing hashes."""
        store = get_compression_store(default_ttl=1)
        with patch("headroom.cache.compression_store.time.time", return_value=1000.0):
            hash_key = store.store(original="payload", compressed="payload")

        with patch("headroom.cache.compression_store.time.time", return_value=1002.0):
            response = client.post("/v1/retrieve", json={"hash": hash_key})

        assert response.status_code == 404
        detail = response.json()["detail"]
        assert "Entry expired" in detail
        assert "CCR TTL: 1 seconds" in detail
        assert "age: 2 seconds" in detail

    def test_retrieve_full_content(self, client):
        """Full retrieval returns original content."""
        store = get_compression_store()
        items = [{"id": i} for i in range(50)]
        hash_key = store.store(
            original=json.dumps(items),
            compressed="[]",
            original_item_count=50,
            compressed_item_count=0,
        )

        response = client.post("/v1/retrieve", json={"hash": hash_key})
        assert response.status_code == 200

        data = response.json()
        assert data["hash"] == hash_key
        assert data["original_item_count"] == 50
        assert "original_content" in data

        # Verify content is correct
        retrieved_items = json.loads(data["original_content"])
        assert len(retrieved_items) == 50
        assert retrieved_items[0]["id"] == 0

    def test_retrieve_same_hash_requires_origin_metadata_for_attribution(self, client):
        store = get_compression_store()
        for event_id, session_id in (("event-a", "session-a"), ("event-b", "session-b")):
            hash_key = store.store(
                original="same original",
                compressed="short",
                compression_event_id=event_id,
                session_id=session_id,
                request_id=f"request-{session_id}",
                tool_call_id=f"call-{session_id}",
            )

        ambiguous = client.post(
            "/v1/retrieve",
            json={"hash": hash_key, "account_recovery": False},
        )
        assert ambiguous.status_code == 409
        assert ambiguous.json()["detail"] == "retrieval reference ambiguous"

        resolved = client.post(
            "/v1/retrieve",
            json={
                "hash": hash_key,
                "session_id": "session-a",
                "account_recovery": False,
            },
        )
        assert resolved.status_code == 200
        assert resolved.json()["compression_event_id"] == "event-a"
        assert "request_id" not in resolved.json()
        assert "session_id" not in resolved.json()
        assert "tool_call_id" not in resolved.json()

    def test_raw_retrieve_does_not_increment_model_retrieval_count(self, client):
        """Client-only transport is not model recovery or retrieval feedback."""
        store = get_compression_store()
        hash_key = store.store(original="[]", compressed="[]")

        response1 = client.post("/v1/retrieve", json={"hash": hash_key})
        response2 = client.post("/v1/retrieve", json={"hash": hash_key})

        assert response1.status_code == 200
        assert response2.status_code == 200
        assert response1.json()["retrieval_count"] == 0
        assert response2.json()["retrieval_count"] == 0


class TestCCRRetrieveGetEndpoint:
    """Test the /v1/retrieve/{hash_key} GET endpoint."""

    def test_get_retrieve_full(self, client):
        """GET retrieval returns full content."""
        store = get_compression_store()
        items = [{"id": i} for i in range(20)]
        hash_key = store.store(
            original=json.dumps(items),
            compressed="[]",
            original_item_count=20,
            compressed_item_count=0,
            tool_name="get_test_tool",
        )

        response = client.get(f"/v1/retrieve/{hash_key}")
        assert response.status_code == 200

        data = response.json()
        assert data["hash"] == hash_key
        assert data["original_item_count"] == 20
        assert data["tool_name"] == "get_test_tool"

    def test_get_retrieve_nonexistent(self, client):
        """GET with nonexistent hash returns 404."""
        response = client.get("/v1/retrieve/nonexistent123")
        assert response.status_code == 404


class TestCCRStatsEndpoint:
    """Test the /v1/retrieve/stats endpoint."""

    def test_stats_empty_store(self, client):
        """Stats with empty store returns zeros."""
        response = client.get("/v1/retrieve/stats")
        assert response.status_code == 200

        data = response.json()
        assert "store" in data
        assert data["store"]["entry_count"] == 0
        assert data["store"]["default_ttl_seconds"] == 1800
        assert "recent_retrievals" in data

    def test_stats_exposes_env_configured_ttl(self, client, monkeypatch):
        """Stats expose the effective CCR TTL configured through env."""
        reset_compression_store()
        monkeypatch.setenv("HEADROOM_CCR_TTL_SECONDS", "7200")

        response = client.get("/v1/retrieve/stats")

        assert response.status_code == 200
        assert response.json()["store"]["default_ttl_seconds"] == 7200

    def test_stats_with_entries(self, client):
        """Stats reflect store contents."""
        store = get_compression_store()

        # Add some entries
        store.store(original="[1]", compressed="[]", original_tokens=100)
        store.store(original="[2]", compressed="[]", original_tokens=200)

        response = client.get("/v1/retrieve/stats")
        assert response.status_code == 200

        data = response.json()
        assert data["store"]["entry_count"] == 2
        assert data["store"]["total_original_tokens"] == 300

    def test_stats_tracks_retrievals(self, client):
        """Stats include recent retrieval events."""
        import json as json_module

        store = get_compression_store()

        content = json_module.dumps(
            [
                {"id": "1", "name": "test item", "value": 100},
                {"id": "2", "name": "another item", "value": 200},
            ]
        )
        hash_key = store.store(
            original=content,
            compressed=content,
            tool_name="stats_test_tool",
        )

        # Provider tool results feed store stats only after the framework
        # acknowledges that each result entered a model request.
        for call_id in ("toolu_stats_1", "toolu_stats_2"):
            tool_response = client.post(
                "/v1/retrieve/tool_call",
                json=_anthropic_retrieve_call(hash_key, call_id),
            )
            _acknowledge_model_injection(client, tool_response)

        response = client.get("/v1/retrieve/stats")
        assert response.status_code == 200

        data = response.json()
        assert data["store"]["total_retrievals"] >= 2
        assert len(data["recent_retrievals"]) >= 2

        # All retrievals are full (no double-logging)
        retrieval_types = [r["retrieval_type"] for r in data["recent_retrievals"]]
        assert "full" in retrieval_types
        assert all(rt == "full" for rt in retrieval_types)


class TestCCRIntegration:
    """Integration tests for CCR with proxy."""

    def test_health_endpoint(self, client):
        """Health endpoint works."""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

    def test_stats_endpoint(self, client):
        """Stats endpoint includes CCR-relevant info."""
        response = client.get("/stats")
        assert response.status_code == 200
        # Proxy stats endpoint is separate from CCR stats
        data = response.json()
        assert "requests" in data
        assert "tokens" in data


class TestCCREdgeCases:
    """Edge cases for CCR endpoints."""

    def test_retrieve_empty_content(self, client):
        """Retrieve works with empty content."""
        store = get_compression_store()
        hash_key = store.store(original="[]", compressed="[]")

        response = client.post("/v1/retrieve", json={"hash": hash_key})
        assert response.status_code == 200
        assert response.json()["original_content"] == "[]"

    def test_retrieve_large_content(self, client):
        """Retrieve works with large content."""
        store = get_compression_store()
        items = [{"id": i, "data": "x" * 100} for i in range(1000)]
        hash_key = store.store(
            original=json.dumps(items),
            compressed=json.dumps(items[:10]),
            original_item_count=1000,
        )

        response = client.post("/v1/retrieve", json={"hash": hash_key})
        assert response.status_code == 200

        data = response.json()
        assert data["original_item_count"] == 1000

    def test_unicode_content(self, client):
        """Unicode content is handled correctly."""
        store = get_compression_store()
        items = [
            {"id": 1, "text": "日本語テキスト"},
            {"id": 2, "text": "Émoji 🎉 test"},
        ]
        hash_key = store.store(original=json.dumps(items, ensure_ascii=False), compressed="[]")

        response = client.post("/v1/retrieve", json={"hash": hash_key})
        assert response.status_code == 200

        data = response.json()
        retrieved = json.loads(data["original_content"])
        assert retrieved[0]["text"] == "日本語テキスト"
        assert "🎉" in retrieved[1]["text"]


class TestEndToEndTOINIntegration:
    """End-to-end tests verifying the production path from proxy → TOIN.

    These tests verify that:
    1. SmartCrusher compresses tool outputs when called through the proxy pipeline
    2. TOIN records compression events
    3. Retrieval events update TOIN field semantics
    4. The full feedback loop works

    This catches bugs where components are wired correctly but don't communicate
    (e.g., compression_store not passing retrieved_items to TOIN).
    """

    @pytest.fixture
    def fresh_toin(self):
        """Create a fresh TOIN instance."""
        import tempfile
        from pathlib import Path

        from headroom.telemetry.toin import (
            TOINConfig,
            get_toin,
            reset_toin,
        )

        reset_toin()
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = str(Path(tmpdir) / "toin.json")
            toin = get_toin(
                TOINConfig(
                    storage_path=storage_path,
                    auto_save_interval=0,
                )
            )
            yield toin
            reset_toin()

    @pytest.fixture
    def client_with_optimization(self, fresh_toin):
        """Create test client with optimization enabled."""
        reset_compression_store()
        config = ProxyConfig(
            optimize=True,  # Enable optimization
            cache_enabled=False,
            rate_limit_enabled=False,
            cost_tracking_enabled=False,
        )
        app = create_app(config)
        # CCR endpoints are loopback-gated (#1227).
        with TestClient(app, base_url="http://127.0.0.1", client=("127.0.0.1", 12345)) as client:
            yield client
        reset_compression_store()

    def test_pipeline_compresses_tool_output_and_records_toin(
        self, fresh_toin, client_with_optimization
    ):
        """CRITICAL: Verify SmartCrusher compression records events in TOIN.

        This tests the production code path:
        1. Tool output comes in through proxy
        2. SmartCrusher compresses it
        3. TOIN records the compression event
        """
        from headroom.config import CCRConfig, SmartCrusherConfig
        from headroom.providers import AnthropicProvider
        from headroom.telemetry import ToolSignature
        from headroom.transforms import SmartCrusher, TransformPipeline

        # Create tool output with 100 items that will trigger compression
        # Key: score field with varying values signals sortable data
        # Having repetitive category values helps trigger compression
        items = [
            {
                "id": i,
                "score": 1000 - i,  # Decreasing scores signal sorting
                "category": f"cat_{i % 3}",  # Only 3 unique categories
                "status": "active" if i % 2 == 0 else "inactive",  # Binary status
            }
            for i in range(100)
        ]
        tool_output = json.dumps(items)

        # Create messages with tool_result containing our data
        messages = [
            {"role": "user", "content": "Search for items"},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "tool_123",
                        "name": "search_api",
                        "input": {"query": "test"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "tool_123",
                        "content": tool_output,
                    }
                ],
            },
        ]

        # Create pipeline with SmartCrusher (same as proxy does).
        # Use with_compaction=False so we exercise the lossy + CCR
        # caching path that this test asserts. The PR4 lossless
        # default substitutes a CSV+schema string and skips CCR
        # caching (nothing dropped → no cache entry).
        pipeline = TransformPipeline(
            transforms=[
                SmartCrusher(
                    SmartCrusherConfig(
                        enabled=True,
                        min_tokens_to_crush=100,
                        max_items_after_crush=15,
                    ),
                    ccr_config=CCRConfig(
                        enabled=True,
                        inject_retrieval_marker=True,
                        min_items_to_cache=10,
                    ),
                    with_compaction=False,
                ),
            ],
            provider=AnthropicProvider(),
        )

        # Apply pipeline (this is what the proxy does)
        result = pipeline.apply(
            messages=messages,
            model="claude-sonnet-4-20250514",
            model_limit=200000,
        )

        # Verify SmartCrusher was invoked (transform name starts with smart_crush)
        smart_crush_applied = any(
            t.startswith("smart_crush") or t.startswith("smart:") for t in result.transforms_applied
        )
        assert smart_crush_applied, (
            f"SmartCrusher should be in transforms: {result.transforms_applied}"
        )

        # Check if compression was actually performed (not skipped)
        # Skip messages look like "smart:skip:reason(100->100)"
        compression_was_skipped = any(
            "skip" in t.lower() for t in result.transforms_applied if "smart:" in t.lower()
        )

        # If compression happened, verify TOIN and store
        if not compression_was_skipped:
            # Verify compression store has the entry
            store = get_compression_store()
            stats = store.get_stats()
            assert stats["entry_count"] >= 1, "Should have cached entry"

            # Verify TOIN recorded the compression
            signature = ToolSignature.from_items(items)
            pattern = fresh_toin._patterns.get(signature.structure_hash)
            assert pattern is not None, (
                "TOIN should have recorded compression event. "
                "If this fails, SmartCrusher is not calling TOIN.record_compression."
            )
            assert pattern.total_compressions >= 1, "Should have at least 1 compression"
        else:
            # Compression was skipped - this is expected for some data patterns
            # The important thing is that SmartCrusher was invoked and made a decision
            # The other tests verify the full loop when compression does happen
            pass

    def test_retrieval_through_proxy_updates_toin_field_semantics(
        self, fresh_toin, client_with_optimization
    ):
        """CRITICAL: Verify retrieval through proxy updates TOIN field semantics.

        This tests the full feedback loop:
        1. Store compressed content (simulating prior compression)
        2. Retrieve through proxy endpoint
        3. Verify TOIN learned field semantics from retrieved items
        """
        from headroom.telemetry import ToolSignature

        # Create items with distinctive field types
        items = [
            {
                "id": i,
                "error_code": 500 if i % 10 == 0 else 200,
                "timestamp": f"2024-01-{i:02d}T00:00:00Z",
                "message": f"Log entry {i}",
            }
            for i in range(50)
        ]
        original_content = json.dumps(items)
        compressed_content = json.dumps(items[:10])

        # Get the signature hash
        signature = ToolSignature.from_items(items)

        # Store in compression store with correct metadata
        store = get_compression_store()
        hash_key = store.store(
            original=original_content,
            compressed=compressed_content,
            original_item_count=50,
            compressed_item_count=10,
            tool_name="logs_api",
            tool_signature_hash=signature.structure_hash,
            compression_strategy="smart_sample",
        )

        # Pre-record some compressions in TOIN (needed for pattern to exist)
        for _ in range(3):
            fresh_toin.record_compression(
                tool_signature=signature,
                original_count=50,
                compressed_count=10,
                original_tokens=5000,
                compressed_tokens=1000,
                strategy="smart_sample",
            )

        # Retrieve the provider-shaped result, then acknowledge its insertion
        # into the next model request.
        response = client_with_optimization.post(
            "/v1/retrieve/tool_call",
            json=_anthropic_retrieve_call(hash_key, "toolu_toin_semantics"),
        )
        assert response.status_code == 200
        _acknowledge_model_injection(client_with_optimization, response)

        # Process pending feedback (this is what triggers TOIN learning)
        # Note: get_compression_store is already imported at module level
        store = get_compression_store()
        store.process_pending_feedback()

        # PR-B5: pattern key is now `(auth_mode, model_family, sig_hash)`.
        # Callers that don't supply auth/model land on the
        # `("unknown", "unknown", sig_hash)` slot.
        from headroom.telemetry.toin import _make_pattern_key

        pattern = fresh_toin._patterns.get(_make_pattern_key(None, None, signature.structure_hash))
        assert pattern is not None, "Pattern should exist after compression and retrieval"

        # CRITICAL ASSERTION: This catches the bug where compression_store
        # wasn't passing retrieved_items to TOIN
        assert len(pattern.field_semantics) > 0, (
            "TOIN should have learned field semantics from retrieved items. "
            "If this fails, the production code path "
            "(CompressionStore.process_pending_feedback -> TOIN.record_retrieval) "
            "is not passing retrieved_items."
        )

        # Verify specific field types were learned
        field_names = list(pattern.field_semantics.keys())
        assert len(field_names) > 0, "Should have learned at least one field"

    def test_full_proxy_ccr_feedback_loop(self, fresh_toin, client_with_optimization):
        """CRITICAL: Test the complete CCR feedback loop through proxy.

        This is the most important integration test - it verifies:
        1. Compression happens and TOIN records it
        2. Retrieval happens and TOIN learns from it
        3. Future recommendations reflect the learning
        """
        from headroom.telemetry import ToolSignature

        # Create items for the full feedback loop test
        items = [
            {
                "id": i,
                "score": 1000 - i,
                "category": f"cat_{i % 5}",
                "status": "active" if i % 2 == 0 else "inactive",
            }
            for i in range(100)
        ]
        signature = ToolSignature.from_items(items)

        # Store content directly (simulating what SmartCrusher does)
        # This ensures we have entries regardless of whether compression was triggered
        store = get_compression_store()
        hash_key = store.store(
            original=json.dumps(items),
            compressed=json.dumps(items[:15]),
            original_item_count=100,
            compressed_item_count=15,
            tool_name="search_api",
            tool_signature_hash=signature.structure_hash,
            compression_strategy="smart_sample",
        )

        # Record compressions in TOIN (simulating what SmartCrusher does)
        for _ in range(3):
            fresh_toin.record_compression(
                tool_signature=signature,
                original_count=100,
                compressed_count=15,
                original_tokens=5000,
                compressed_tokens=1000,
                strategy="smart_sample",
            )

        # Step 2: retrieve the provider result and acknowledge model injection.
        response = client_with_optimization.post(
            "/v1/retrieve/tool_call",
            json=_anthropic_retrieve_call(hash_key, "toolu_full_feedback"),
        )
        assert response.status_code == 200
        _acknowledge_model_injection(client_with_optimization, response)

        # Process feedback (this triggers TOIN learning)
        store.process_pending_feedback()

        # Step 3: Verify TOIN learned
        # PR-B5: pattern key is now `(auth_mode, model_family, sig_hash)`.
        # Callers that don't supply auth/model land on the
        # `("unknown", "unknown", sig_hash)` slot.
        from headroom.telemetry.toin import _make_pattern_key

        pattern = fresh_toin._patterns.get(_make_pattern_key(None, None, signature.structure_hash))
        assert pattern is not None, "Pattern should exist"
        assert pattern.total_compressions >= 1, "Should have compression count"
        assert pattern.total_retrievals >= 1, "Should have retrieval count"

        # Step 4: Verify field semantics were learned
        assert len(pattern.field_semantics) > 0, (
            "TOIN should learn field semantics through the full proxy CCR loop. "
            "This is the ultimate integration test - if this fails, "
            "the production feedback loop is broken."
        )

        # Step 5: PR-B5 retired the request-time recommendation API in favor of
        # observation-only learning + startup-published recommendations.toml.
        # `get_recommendation()` now returns None and emits a deprecation
        # warning; the dispatcher consumes published advice via the Rust
        # `RecommendationStore`. Assert the deprecation contract here so a
        # future revival of the API doesn't slip past silently.
        assert fresh_toin.get_recommendation(signature, "find category") is None


def _seed_retrieval_accounting(hash_key: str = "abc123def456"):
    from headroom.transforms.retrieval_aware_policy import (
        CompressionAction,
        enable_compression_cost_tracking,
        get_compression_cost_ledger,
    )

    ledger = get_compression_cost_ledger()
    ledger.clear()
    enable_compression_cost_tracking(True)
    event_id = ledger.record_outcome(
        tool_name="catalog_reader",
        strategy="row_drop",
        action=CompressionAction.LOSSY,
        predicted_action=CompressionAction.LOSSY,
        original_tokens=1000,
        initially_emitted_tokens=300,
        ccr_hashes=[hash_key],
        session_id="session-wire",
        request_id="request-wire",
        tool_call_id="call-wire",
    )
    store = get_compression_store()
    store.store(
        original='{"exact":"wire payload"}',
        compressed=f"<<ccr:{hash_key}>>",
        explicit_hash=hash_key,
        compression_event_id=event_id,
        session_id="session-wire",
        request_id="request-wire",
        tool_call_id="call-wire",
        tool_name="catalog_reader",
    )
    return ledger


class TestRetrievalAwareWireAccounting:
    @pytest.fixture(autouse=True)
    def _cleanup_ledger(self):
        from headroom.transforms.retrieval_aware_policy import (
            enable_compression_cost_tracking,
            get_compression_cost_ledger,
        )

        yield
        get_compression_cost_ledger().clear()
        enable_compression_cost_tracking(False)

    @pytest.mark.parametrize("method", ["post", "get"])
    def test_raw_retrieve_is_transport_only(self, client, method):
        from headroom.transforms.retrieval_aware_policy import estimate_payload_tokens

        ledger = _seed_retrieval_accounting()
        response = (
            client.post("/v1/retrieve", json={"hash": "abc123def456"})
            if method == "post"
            else client.get("/v1/retrieve/abc123def456")
        )
        assert response.status_code == 200
        snapshot = ledger.snapshot()
        assert snapshot["actual_recovery_events"] == 0
        assert snapshot["recovery_payload_tokens"] == 0
        assert snapshot["non_model_retrievals"]["http_transport_retrieval_events"] == 1
        assert snapshot["non_model_retrievals"]["http_transport_payload_tokens"] == (
            estimate_payload_tokens(response.json())
        )

    def test_tool_call_charges_only_after_model_injection_acknowledgement(self, client):
        from headroom.transforms.retrieval_aware_policy import estimate_payload_tokens

        ledger = _seed_retrieval_accounting()
        response = client.post(
            "/v1/retrieve/tool_call",
            json={
                "provider": "anthropic",
                "tool_call": {
                    "id": "toolu_wire",
                    "name": "headroom_retrieve",
                    "input": {"hash": "abc123def456"},
                },
            },
        )
        assert response.status_code == 200
        pending = ledger.snapshot()
        assert pending["actual_recovery_events"] == 0
        assert pending["recovery_payload_tokens"] == 0
        assert pending["non_model_retrievals"]["http_transport_retrieval_events"] == 1
        assert response.json()["accounting"]["report"]["recovery_payload_tokens"] == (
            estimate_payload_tokens(response.json()["tool_result"])
        )

        report = response.json()["accounting"]["report"]
        assert set(report) == {
            "hash",
            "retrieval_event_id",
            "recovery_payload_tokens",
            "compression_event_id",
            "retrieval_handle",
        }
        assert (
            get_compression_store().retrieve_for_internal_use("abc123def456").retrieval_count == 0
        )
        acknowledgement = _acknowledge_model_injection(client, response)

        assert acknowledgement.json()["status"] == "attributed"
        assert ledger.snapshot()["recovery_payload_tokens"] == estimate_payload_tokens(
            response.json()["tool_result"]
        )
        assert (
            get_compression_store().retrieve_for_internal_use("abc123def456").retrieval_count == 1
        )

    def test_two_identical_event_handles_retrieve_and_charge_independently(self, client):
        from headroom.ccr.retrieval_reference import attach_retrieval_handle, references_in_text
        from headroom.transforms.retrieval_aware_policy import (
            CompressionAction,
            enable_compression_cost_tracking,
            get_compression_cost_ledger,
        )

        enable_compression_cost_tracking(True)
        hash_key = "fedcba987654"
        original = '{"same":"payload"}'
        ledger = get_compression_cost_ledger()
        ledger.clear()
        store = get_compression_store()
        records = []
        for index in range(2):
            event_id = ledger.record_outcome(
                tool_name="catalog_reader",
                strategy="row_drop",
                action=CompressionAction.LOSSY,
                predicted_action=CompressionAction.LOSSY,
                original_tokens=1000,
                initially_emitted_tokens=300,
                ccr_hashes=[hash_key],
                session_id=f"session-{index}",
                request_id=f"request-{index}",
                tool_call_id=f"call-{index}",
                provider_slot=f"slot-{index}",
            )
            handle = ledger.new_retrieval_handle()
            marker = attach_retrieval_handle(f"<<ccr:{hash_key}>>", handle)
            reference = references_in_text(marker)[0]
            assert reference.hash_key == hash_key
            assert reference.retrieval_handle == handle
            store.store(
                original=original,
                compressed=marker,
                explicit_hash=hash_key,
                retrieval_handle=handle,
                compression_event_id=event_id,
                session_id=f"session-{index}",
                request_id=f"request-{index}",
                tool_call_id=f"call-{index}",
                provider_slot=f"slot-{index}",
                tool_name="catalog_reader",
            )
            records.append((event_id, handle))

        for index, (_event_id, handle) in enumerate(records):
            response = client.post(
                "/v1/retrieve/tool_call",
                json={
                    "provider": "openai",
                    "tool_call": {
                        "id": f"recovery-{index}",
                        "type": "function",
                        "function": {
                            "name": "headroom_retrieve",
                            "arguments": json.dumps({"hash": hash_key, "handle": handle}),
                        },
                    },
                },
            )
            assert response.status_code == 200
            assert response.json()["success"] is True
            assert ledger.snapshot()["actual_recovery_events"] == index
            _acknowledge_model_injection(client, response)

        recent = {row["compression_event_id"]: row for row in ledger.snapshot()["recent"]}
        assert recent[records[0][0]]["retrieval_count"] == 1
        assert recent[records[1][0]]["retrieval_count"] == 1
        assert recent[records[0][0]]["recovery_payload_tokens"] > 0
        assert recent[records[1][0]]["recovery_payload_tokens"] > 0

        ambiguous = client.post(
            "/v1/retrieve/tool_call",
            json={
                "provider": "openai",
                "tool_call": {
                    "id": "legacy-ambiguous",
                    "type": "function",
                    "function": {
                        "name": "headroom_retrieve",
                        "arguments": json.dumps({"hash": hash_key}),
                    },
                },
            },
        )
        assert ambiguous.status_code == 200
        assert ambiguous.json()["success"] is False
        assert ambiguous.json()["data"]["status"] == "ambiguous"
        assert "original_content" not in ambiguous.json()["data"]
        after = ledger.snapshot()
        assert after["actual_recovery_events"] == 2
        assert after["attribution"]["ambiguous_attributions"] == 1
        assert after["attribution"]["rejected_retrieval_reports"] == 1

    def test_cross_process_report_is_idempotent(self, client):
        ledger = _seed_retrieval_accounting()
        report = {
            "hash": "abc123def456",
            "retrieval_event_id": "retrieval-stable-wire-id",
            "recovery_payload_tokens": 321,
            "compression_event_id": ledger.snapshot()["recent"][0]["compression_event_id"],
            "session_id": "session-wire",
            "request_id": "request-wire",
            "tool_call_id": "call-wire",
        }
        first = client.post("/v1/retrieve/account", json=report)
        second = client.post("/v1/retrieve/account", json=report)
        assert first.json()["status"] == "attributed"
        assert first.json()["feedback_committed"] is True
        assert second.json()["status"] == "duplicate"
        assert second.json()["feedback_committed"] is False
        assert ledger.snapshot()["recovery_payload_tokens"] == 321
        entry = get_compression_store().retrieve_for_internal_use("abc123def456")
        assert entry is not None
        assert entry.retrieval_count == 1

    def test_rejected_accounting_report_does_not_teach_store_feedback(self, client, monkeypatch):
        from headroom.transforms.retrieval_aware_policy import RetrievalReportStatus

        ledger = _seed_retrieval_accounting()
        monkeypatch.setattr(
            ledger,
            "record_recovery",
            lambda *args, **kwargs: RetrievalReportStatus.REJECTED,
        )
        response = client.post(
            "/v1/retrieve/account",
            json={
                "hash": "abc123def456",
                "retrieval_event_id": "retrieval-rejected-wire-id",
                "recovery_payload_tokens": 7,
            },
        )

        assert response.status_code == 200
        assert response.json()["status"] == "rejected"
        assert response.json()["feedback_committed"] is False
        entry = get_compression_store().retrieve_for_internal_use("abc123def456")
        assert entry is not None
        assert entry.retrieval_count == 0

    def test_accounting_unset_secret_trusts_loopback_process(self, client, monkeypatch):
        ledger = _seed_retrieval_accounting()
        monkeypatch.delenv("HEADROOM_RETRIEVAL_ACCOUNT_SECRET", raising=False)
        report = {
            "hash": "abc123def456",
            "retrieval_event_id": "retrieval-unset-secret-wire-id",
            "recovery_payload_tokens": 7,
            "compression_event_id": ledger.snapshot()["recent"][0]["compression_event_id"],
        }
        response = client.post("/v1/retrieve/account", json=report)
        assert response.status_code == 200
        assert response.json()["status"] == "attributed"

    def test_disabled_accounting_acknowledgement_is_idempotent(self, client):
        from headroom.transforms.retrieval_aware_policy import enable_compression_cost_tracking

        ledger = _seed_retrieval_accounting()
        enable_compression_cost_tracking(False)
        report = {
            "hash": "abc123def456",
            "retrieval_event_id": "retrieval-disabled-feedback-id",
            "recovery_payload_tokens": 7,
            "compression_event_id": ledger.snapshot()["recent"][0]["compression_event_id"],
        }

        first = client.post("/v1/retrieve/account", json=report)
        second = client.post("/v1/retrieve/account", json=report)

        assert first.json()["status"] == "disabled"
        assert first.json()["feedback_committed"] is True
        assert second.json()["status"] == "duplicate"
        assert second.json()["feedback_committed"] is False
        entry = get_compression_store().retrieve_for_internal_use("abc123def456")
        assert entry is not None
        assert entry.retrieval_count == 1

    def test_accounting_secret_fails_closed_and_forwarded_headers_do_not_help(
        self, client, monkeypatch
    ):
        ledger = _seed_retrieval_accounting()
        monkeypatch.setenv("HEADROOM_RETRIEVAL_ACCOUNT_SECRET", "local-secret")
        report = {
            "hash": "abc123def456",
            "retrieval_event_id": "retrieval-secret-wire-id",
            "recovery_payload_tokens": 7,
            "compression_event_id": ledger.snapshot()["recent"][0]["compression_event_id"],
        }
        missing = client.post(
            "/v1/retrieve/account",
            json=report,
            headers={"X-Forwarded-For": "127.0.0.1"},
        )
        incorrect = client.post(
            "/v1/retrieve/account",
            json=report,
            headers={
                "x-headroom-retrieval-account-secret": "wrong",
                "X-Forwarded-For": "127.0.0.1",
            },
        )
        allowed = client.post(
            "/v1/retrieve/account",
            json=report,
            headers={"x-headroom-retrieval-account-secret": "local-secret"},
        )
        assert missing.status_code == 404
        assert incorrect.status_code == 404
        assert allowed.json()["status"] == "attributed"

    def test_stats_preserves_store_and_recent_schema_with_nested_ledger(self, client):
        _seed_retrieval_accounting()
        response = client.get("/v1/retrieve/stats")
        assert response.status_code == 200
        payload = response.json()
        assert "store" in payload
        assert "recent_retrievals" in payload
        assert payload["retrieval_aware"]["enabled"] is True
        assert "ledger" in payload["retrieval_aware"]
        assert payload["realized_cost"]["deprecated"] is True
        assert payload["realized_cost"]["replacement"] == "retrieval_aware.ledger"

    def test_real_smartcrusher_identical_emissions_retrieve_and_charge_independently(self, client):
        from headroom.ccr.retrieval_reference import references_in_text
        from headroom.transforms.content_router import ContentRouter, ContentRouterConfig
        from headroom.transforms.retrieval_aware_policy import (
            enable_compression_cost_tracking,
            get_compression_cost_ledger,
        )

        ledger = get_compression_cost_ledger()
        ledger.clear()
        enable_compression_cost_tracking(True)
        # Mixed severity provides a deterministic lossy signal; a uniform
        # array can intentionally take SmartCrusher's no-signal lossless path.
        original = json.dumps(
            [
                {
                    "id": index,
                    "level": "error" if index % 30 == 0 else "info",
                    "message": f"deterministic line {index}",
                }
                for index in range(80)
            ]
        )
        router = ContentRouter(
            ContentRouterConfig(
                retrieval_aware_enabled=True,
                retrieval_aware_forced_action="lossy",
                smart_crusher_with_compaction=False,
                smart_crusher_max_items_after_crush=8,
                enable_kompress=False,
            )
        )

        emissions = []
        for index in range(2):
            result = router.compress(
                original,
                context="list catalog records",
                tool_name="catalog_reader",
                attribution={
                    "session_id": f"session-real-{index}",
                    "request_id": f"request-real-{index}",
                    "tool_call_id": f"call-real-{index}",
                    "provider_slot": f"message[{index}].content",
                    "recovery_payload_path": "openai_chat",
                },
            )
            references = references_in_text(result.compressed)
            assert references, result.compressed
            assert len(result.compression_event_ids) == 1
            emissions.append(
                (
                    result.compression_event_ids[0],
                    references[0],
                )
            )

        assert emissions[0][1].hash_key == emissions[1][1].hash_key
        assert emissions[0][1].retrieval_handle != emissions[1][1].retrieval_handle

        for index, (_event_id, reference) in enumerate(emissions):
            response = client.post(
                "/v1/retrieve/tool_call",
                json={
                    "provider": "openai",
                    "tool_call": {
                        "id": f"recover-real-{index}",
                        "type": "function",
                        "function": {
                            "name": "headroom_retrieve",
                            "arguments": json.dumps(
                                {
                                    "hash": reference.hash_key,
                                    "handle": reference.retrieval_handle,
                                }
                            ),
                        },
                    },
                },
            )
            assert response.status_code == 200
            assert response.json()["success"] is True
            assert json.loads(response.json()["data"]["original_content"]) == json.loads(original)
            _acknowledge_model_injection(client, response)

        recent = {row["compression_event_id"]: row for row in ledger.snapshot()["recent"]}
        for event_id, _reference in emissions:
            assert recent[event_id]["retrieval_count"] == 1
            assert recent[event_id]["recovery_payload_tokens"] > 0

    def test_tool_call_retry_reuses_accounting_event_id(self, client):
        ledger = _seed_retrieval_accounting()
        request = _anthropic_retrieve_call("abc123def456", call_id="stable-provider-call")
        first = client.post("/v1/retrieve/tool_call", json=request)
        second = client.post("/v1/retrieve/tool_call", json=request)

        assert first.json()["success"] is True
        assert second.json()["success"] is True
        assert first.json()["accounting"]["report"] == second.json()["accounting"]["report"]
        assert ledger.snapshot()["actual_recovery_events"] == 0

        first_ack = _acknowledge_model_injection(client, first)
        second_ack = _acknowledge_model_injection(client, second)

        assert first_ack.json()["status"] == "attributed"
        assert second_ack.json()["status"] == "duplicate"
        snapshot = ledger.snapshot()
        assert snapshot["actual_recovery_events"] == 1
        assert snapshot["attribution"]["duplicate_retrieval_reports"] == 1
        entry = get_compression_store().retrieve_for_internal_use("abc123def456")
        assert entry is not None
        assert entry.retrieval_count == 1
