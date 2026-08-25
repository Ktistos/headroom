from __future__ import annotations

from types import MethodType

import pytest

from headroom.cache.compression_store import CompressionStore
from headroom.transforms.compression_units import (
    CompressionUnit,
    RoutedCompressionUnit,
    compress_unit_with_router,
    compress_units_with_router,
)
from headroom.transforms.content_router import (
    CompressionStrategy,
    ContentRouter,
    RouterCompressionResult,
)
from headroom.transforms.retrieval_aware_policy import (
    CompressionAction,
    CompressionCostLedger,
    RetrievalAwarePolicy,
)


class TokenCounter:
    def count_text(self, text: str) -> int:
        return len(text.split())


class Router:
    def __init__(self, compressed: str):
        self.compressed = compressed

    def compress(self, content: str, **_kwargs):
        return RouterCompressionResult(
            compressed=self.compressed,
            original=content,
            strategy_used=CompressionStrategy.KOMPRESS,
        )


class CharacterCounter:
    def count_text(self, text: str) -> int:
        return len(text)


def test_compression_unit_uses_utf8_bytes_for_floor():
    result = compress_unit_with_router(
        CompressionUnit(
            text="你" * 256,
            provider="openai",
            endpoint="responses",
            role="tool",
            item_type="function_call_output",
            min_bytes=512,
        ),
        router=Router("短"),
        tokenizer=CharacterCounter(),
    )

    assert result.modified is True
    assert result.reason is None


def test_compression_unit_accepts_token_shrinking_replacement():
    result = compress_unit_with_router(
        CompressionUnit(
            text="alpha beta gamma delta epsilon",
            provider="openai",
            endpoint="responses",
            role="assistant",
            item_type="message",
            metadata={"compress_assistant": "true"},
            min_bytes=1,
        ),
        router=Router("alpha beta"),
        tokenizer=TokenCounter(),
    )

    assert result.modified is True
    assert result.tokens_saved == 3
    assert result.compressed == "alpha beta"
    assert "router:openai:responses:message:kompress" in result.transforms_applied


def test_compression_unit_keeps_lossy_unmarked_tool_output_verbatim():
    original = (
        "src/app.py:12 render shell status panel\n"
        "src/ui.py:44 draw health badge\n"
        "src/theme.py:9 set accent color"
    )
    result = compress_unit_with_router(
        CompressionUnit(
            text=original,
            provider="openai",
            endpoint="responses",
            role="tool",
            item_type="local_shell_call_output",
            min_bytes=1,
        ),
        router=Router("shell output looks organized and green"),
        tokenizer=TokenCounter(),
    )

    assert result.modified is False
    assert result.reason == "lossy_unrecoverable_tool_output"
    assert result.original == original
    assert result.compressed == original


def test_compression_unit_accepts_lossy_tool_output_when_recoverable():
    original = "alpha beta gamma delta epsilon zeta eta theta"
    result = compress_unit_with_router(
        CompressionUnit(
            text=original,
            provider="openai",
            endpoint="responses",
            role="tool",
            item_type="local_shell_call_output",
            min_bytes=1,
        ),
        router=Router("summary <<ccr:abc123>>"),
        tokenizer=TokenCounter(),
    )

    assert result.modified is True
    assert result.reason is None
    assert result.compressed == "summary <<ccr:abc123>>"


def test_compression_unit_still_compresses_non_shell_tool_output():
    result = compress_unit_with_router(
        CompressionUnit(
            text="alpha beta gamma delta epsilon zeta eta theta",
            provider="openai",
            endpoint="responses",
            role="tool",
            item_type="function_call_output",
            min_bytes=1,
        ),
        router=Router("summary for tool=0"),
        tokenizer=TokenCounter(),
    )

    assert result.modified is True
    assert result.reason is None
    assert result.compressed == "summary for tool=0"


def test_compression_unit_still_compresses_assistant_text():
    result = compress_unit_with_router(
        CompressionUnit(
            text="alpha beta gamma delta epsilon",
            provider="openai",
            endpoint="responses",
            role="assistant",
            item_type="message",
            min_bytes=1,
            metadata={"compress_assistant": "true"},
        ),
        router=Router("alpha beta"),
        tokenizer=TokenCounter(),
    )

    assert result.modified is True
    assert result.reason is None
    assert result.compressed == "alpha beta"


def test_compression_unit_rejects_non_shrinking_replacement():
    result = compress_unit_with_router(
        CompressionUnit(
            text="alpha beta",
            provider="anthropic",
            endpoint="messages",
            role="tool",
            item_type="tool_result",
            min_bytes=1,
        ),
        router=Router("alpha beta gamma"),
        tokenizer=TokenCounter(),
    )

    assert result.modified is False
    assert result.reason == "rejected_not_smaller"
    assert result.original == "alpha beta"


def test_compression_unit_respects_cache_zone_and_floor():
    frozen = compress_unit_with_router(
        CompressionUnit(
            text="alpha beta gamma delta",
            provider="anthropic",
            endpoint="messages",
            role="tool",
            item_type="tool_result",
            cache_zone="frozen",
            min_bytes=1,
        ),
        router=Router("alpha"),
        tokenizer=TokenCounter(),
    )
    small = compress_unit_with_router(
        CompressionUnit(
            text="small text",
            provider="openai",
            endpoint="responses",
            role="tool",
            item_type="function_call_output",
            min_bytes=500,
        ),
        router=Router("small"),
        tokenizer=TokenCounter(),
    )

    assert frozen.modified is False
    assert frozen.reason == "cache_zone_frozen"
    assert small.modified is False
    assert small.reason == "below_unit_floor"


def test_batch_compression_preserves_provider_slot_references():
    routed = [
        RoutedCompressionUnit(
            unit=CompressionUnit(
                text="alpha beta gamma",
                provider="openai",
                endpoint="responses",
                role="assistant",
                item_type="message",
                metadata={"compress_assistant": "true"},
                min_bytes=1,
            ),
            slot=("input", 3, "output"),
        ),
        RoutedCompressionUnit(
            unit=CompressionUnit(
                text="one two three",
                provider="gemini",
                endpoint="generateContent",
                role="user",
                item_type="part.text",
                min_bytes=1,
            ),
            slot={"path": ["contents", 0, "parts", 0, "text"]},
        ),
    ]

    results = compress_units_with_router(
        routed,
        router=Router("short"),
        tokenizer=TokenCounter(),
    )

    assert results[0][0] == ("input", 3, "output")
    assert results[1][0] == {"path": ["contents", 0, "parts", 0, "text"]}
    assert [result.modified for _slot, result in results] == [True, False]


def test_compress_unit_protects_prompt_roles() -> None:
    for role, reason in [
        ("user", "protected_user_message"),
        ("developer", "protected_system_message"),
        ("system", "protected_system_message"),
        ("assistant", "protected_assistant_message"),
    ]:
        unit = CompressionUnit(
            text="alpha beta gamma delta",
            provider="openai",
            endpoint="responses",
            role=role,
            item_type="message",
            min_bytes=1,
        )

        result = compress_unit_with_router(unit, router=Router("alpha"), tokenizer=TokenCounter())

        assert result.modified is False
        assert result.reason == reason


def test_live_unit_with_retrieval_marker_compresses_surrounding_text() -> None:
    marker = "[100 items compressed to 10. Retrieve more: hash=abc123]"
    text = f"alpha beta gamma delta epsilon\n{marker}\nzeta eta theta iota kappa"

    result = compress_unit_with_router(
        CompressionUnit(
            text=text,
            provider="openai",
            endpoint="responses",
            role="tool",
            item_type="function_call_output",
            min_bytes=1,
        ),
        router=Router("short"),
        tokenizer=TokenCounter(),
    )

    assert result.modified is True
    assert result.reason is None
    assert result.strategy == "ccr_marker_preserving"
    assert result.compressed == f"short\n{marker}\nshort"
    assert marker in result.compressed
    assert result.tokens_saved > 0
    assert "ccr_marker_preserving" in result.transforms_applied


def test_non_live_unit_with_retrieval_marker_preserves_prefix_cache() -> None:
    marker = "[100 items compressed to 10. Retrieve more: hash=abc123]"
    text = f"alpha beta gamma delta epsilon\n{marker}\nzeta eta theta"

    result = compress_unit_with_router(
        CompressionUnit(
            text=text,
            provider="openai",
            endpoint="responses",
            role="tool",
            item_type="function_call_output",
            cache_zone="prefix",
            min_bytes=1,
        ),
        router=Router("short"),
        tokenizer=TokenCounter(),
    )

    assert result.modified is False
    assert result.reason == "cache_zone_prefix"
    assert result.compressed == text


class _RejectReconstructedUnitCounter:
    def __init__(self, original: str):
        self.original = original

    def count_text(self, text: str) -> int:
        return 1 if text == self.original else 2


def _marker_span_fixture(
    monkeypatch: pytest.MonkeyPatch,
    *,
    fail_on_call: int | None = None,
    max_events: int = 10_000,
):
    ledger = CompressionCostLedger(
        observation_window_seconds=300,
        max_events=max_events,
    )
    store = CompressionStore(enable_feedback=False)
    monkeypatch.setattr("headroom.cache.compression_store.get_compression_store", lambda: store)
    router = ContentRouter()
    router._retrieval_aware_policy = RetrievalAwarePolicy(ledger)
    event_ids: list[str] = []
    hashes: list[str] = []
    calls = 0

    def compress(self, content: str, **_kwargs):
        nonlocal calls
        index = calls
        calls += 1
        if fail_on_call == index:
            raise RuntimeError("later span failed")
        hash_key = f"{index + 1:012x}"
        event_id = ledger.record_outcome(
            tool_name="catalog_reader",
            strategy="row_drop",
            action=CompressionAction.LOSSY,
            predicted_action=CompressionAction.LOSSY,
            original_tokens=100,
            initially_emitted_tokens=20,
            ccr_hashes=[hash_key],
            compression_event_id=f"span-event-{index}",
        )
        handle = ledger.new_retrieval_handle()
        marker = f"<<ccr:{hash_key}@{handle}>>"
        store.store(
            original=content,
            compressed=marker,
            explicit_hash=hash_key,
            compression_event_id=event_id,
            retrieval_handle=handle,
            tool_name="catalog_reader",
        )
        event_ids.append(event_id)
        hashes.append(hash_key)
        return RouterCompressionResult(
            compressed=marker,
            original=content,
            strategy_used=CompressionStrategy.SMART_CRUSHER,
            strategy_chain=["smart_crusher"],
            compression_event_ids=[event_id],
            compression_event_hashes={event_id: (hash_key,)},
        )

    router.compress = MethodType(compress, router)
    return router, ledger, store, event_ids, hashes


def _multi_span_unit() -> CompressionUnit:
    existing_marker = "[100 items compressed to 10. Retrieve more: hash=abc123]"
    text = f"{'alpha ' * 40}\n{existing_marker}\n{'beta ' * 40}"
    return CompressionUnit(
        text=text,
        provider="openai",
        endpoint="responses",
        role="tool",
        item_type="function_call_output",
        min_bytes=1,
    )


def test_accepted_marker_spans_retain_every_event_and_store_candidate(monkeypatch) -> None:
    router, ledger, store, event_ids, hashes = _marker_span_fixture(monkeypatch)

    result = compress_unit_with_router(
        _multi_span_unit(),
        router=router,
        tokenizer=TokenCounter(),
    )

    assert result.modified is True
    assert result.router_result is not None
    assert result.router_result.compression_event_ids == event_ids
    assert len(event_ids) == 2
    assert all(store.get_entry_status(hash_key)["status"] == "available" for hash_key in hashes)
    recent = {row["compression_event_id"]: row for row in ledger.snapshot()["recent"]}
    assert all(recent[event_id]["action"] == "lossy" for event_id in event_ids)


def test_rejected_marker_spans_reconcile_every_event_and_candidate(monkeypatch) -> None:
    unit = _multi_span_unit()
    router, ledger, store, event_ids, hashes = _marker_span_fixture(monkeypatch)

    result = compress_unit_with_router(
        unit,
        router=router,
        tokenizer=_RejectReconstructedUnitCounter(unit.text),
    )

    assert result.reason == "rejected_not_smaller"
    assert result.router_result is not None
    assert result.router_result.compression_event_ids == event_ids
    assert len(event_ids) == 2
    assert all(store.get_entry_status(hash_key)["status"] == "missing" for hash_key in hashes)
    recent = {row["compression_event_id"]: row for row in ledger.snapshot()["recent"]}
    assert all(recent[event_id]["action"] == "passthrough" for event_id in event_ids)


def test_rejected_marker_spans_clean_candidates_after_ledger_eviction(monkeypatch) -> None:
    unit = _multi_span_unit()
    router, ledger, store, event_ids, hashes = _marker_span_fixture(
        monkeypatch,
        max_events=1,
    )

    result = compress_unit_with_router(
        unit,
        router=router,
        tokenizer=_RejectReconstructedUnitCounter(unit.text),
    )

    assert result.reason == "rejected_not_smaller"
    assert len(event_ids) == 2
    assert ledger.snapshot()["recent"][0]["compression_event_id"] == event_ids[-1]
    assert all(store.get_entry_status(hash_key)["status"] == "missing" for hash_key in hashes)


def test_later_marker_span_exception_reconciles_earlier_events(monkeypatch) -> None:
    router, ledger, store, event_ids, hashes = _marker_span_fixture(
        monkeypatch,
        fail_on_call=1,
    )

    with pytest.raises(RuntimeError, match="later span failed"):
        compress_unit_with_router(
            _multi_span_unit(),
            router=router,
            tokenizer=TokenCounter(),
        )

    assert event_ids == ["span-event-0"]
    assert hashes == ["000000000001"]
    assert store.get_entry_status(hashes[0])["status"] == "missing"
    event = ledger.snapshot()["recent"][0]
    assert event["action"] == "passthrough"
    assert event["strategy"] == "provider_unit_gate_passthrough"
