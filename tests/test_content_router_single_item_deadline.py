from __future__ import annotations

import json
import threading
import time

import headroom.transforms.kompress_compressor as kc
from headroom.transforms.content_detector import ContentType
from headroom.transforms.content_router import (
    CompressionStrategy,
    ContentRouter,
    ContentRouterConfig,
    RouterCompressionResult,
    RoutingDecision,
)
from headroom.transforms.kompress_compressor import KompressCompressor, KompressConfig


class _Tokenizer:
    def count_text(self, content: str) -> int:
        return len(content.split())


def _compression_result(content: str, compressed: str) -> RouterCompressionResult:
    return RouterCompressionResult(
        compressed=compressed,
        original=content,
        strategy_used=CompressionStrategy.TEXT,
        routing_log=[
            RoutingDecision(
                content_type=ContentType.PLAIN_TEXT,
                strategy=CompressionStrategy.TEXT,
                original_tokens=len(content.split()),
                compressed_tokens=len(compressed.split()),
            )
        ],
    )


def _router() -> ContentRouter:
    return ContentRouter(
        ContentRouterConfig(
            protect_recent_code=0,
            protect_analysis_context=False,
            skip_user_messages=False,
        )
    )


def _messages() -> list[dict[str, str]]:
    return [
        {"role": "assistant", "content": "frozen prefix content remains unchanged"},
        {
            "role": "assistant",
            "content": "pending cache miss content takes the inline compression branch today",
        },
    ]


def test_single_cache_miss_fails_open_at_deadline(monkeypatch, caplog):
    router = _router()

    def slow_compress(content, *, context="", bias=1.0):
        time.sleep(0.2)
        return _compression_result(content, "compressed output")

    monkeypatch.setattr(router, "compress", slow_compress)
    monkeypatch.setenv("HEADROOM_COMPRESSION_DEADLINE_MS", "10")

    started = time.perf_counter()
    result = router.apply(
        _messages(),
        _Tokenizer(),
        frozen_message_count=1,
        min_tokens_to_compress=1,
    )

    assert time.perf_counter() - started < 0.12
    assert result.messages[1]["content"] == _messages()[1]["content"]
    assert "failing open via PASSTHROUGH" in caplog.text


def test_single_cache_miss_preserves_under_deadline_output(monkeypatch):
    router = _router()
    monkeypatch.setattr(
        router,
        "compress",
        lambda content, *, context="", bias=1.0: _compression_result(content, "compressed output"),
    )
    monkeypatch.setenv("HEADROOM_COMPRESSION_DEADLINE_MS", "1000")

    result = router.apply(
        _messages(),
        _Tokenizer(),
        frozen_message_count=1,
        min_tokens_to_compress=1,
    )

    assert result.messages[1]["content"] == "compressed output"


def test_single_cache_miss_effects_wait_for_request_holder(monkeypatch):
    from headroom.transforms.content_router import (
        _ACTIVE_POLICY_SIDE_EFFECT_TRANSACTION,
        RequestPolicySideEffectHolder,
        activate_request_policy_side_effect_holder,
    )

    router = _router()
    state = {"discarded": 0, "committed": 0}

    def compress(content, *, context="", bias=1.0):
        transaction = _ACTIVE_POLICY_SIDE_EFFECT_TRANSACTION.get()
        assert transaction is not None
        transaction.register(
            "request-owned-deadline-effect",
            commit=lambda: state.__setitem__("committed", state["committed"] + 1),
            discard=lambda: state.__setitem__("discarded", state["discarded"] + 1),
        )
        return _compression_result(content, "compressed output")

    monkeypatch.setattr(router, "compress", compress)
    monkeypatch.setenv("HEADROOM_COMPRESSION_DEADLINE_MS", "1000")
    holder = RequestPolicySideEffectHolder()

    with activate_request_policy_side_effect_holder(holder):
        result = router.apply(
            _messages(),
            _Tokenizer(),
            frozen_message_count=1,
            min_tokens_to_compress=1,
        )
        assert result.messages[1]["content"] == "compressed output"
        assert state == {"discarded": 0, "committed": 0}

    holder.finalize(commit=False)
    assert state == {"discarded": 1, "committed": 0}


def test_single_cache_miss_preserves_disabled_deadline(monkeypatch):
    router = _router()
    monkeypatch.setattr(
        router,
        "compress",
        lambda content, *, context="", bias=1.0: _compression_result(content, "compressed output"),
    )
    monkeypatch.setenv("HEADROOM_COMPRESSION_DEADLINE_MS", "0")

    result = router.apply(
        _messages(),
        _Tokenizer(),
        frozen_message_count=1,
        min_tokens_to_compress=1,
    )

    assert result.messages[1]["content"] == "compressed output"


def test_single_cache_miss_deadline_starts_before_kompress_load(monkeypatch, caplog):
    router = _router()

    class _Encoding(dict):
        def __init__(self, rows: list[list[str]]):
            super().__init__(
                input_ids=[[0] * len(row) for row in rows],
                attention_mask=[[1] * len(row) for row in rows],
            )
            self._rows = rows

        def word_ids(self, batch_index: int = 0):
            return list(range(len(self._rows[batch_index])))

    class _Tokenizer:
        def count_text(self, content: str) -> int:
            return len(content.split())

        def __call__(self, words, **_kwargs):
            rows = words if words and isinstance(words[0], list) else [words]
            return _Encoding(rows)

    class _Model:
        def __init__(self):
            self.calls = 0

        def get_keep_mask(self, input_ids, attention_mask):
            self.calls += 1
            return [[i % 2 == 0 for i in range(len(row))] for row in input_ids]

    model = _Model()
    compressor = KompressCompressor(config=KompressConfig(enable_ccr=False))
    monkeypatch.setattr(compressor, "_should_batch_single_content", lambda *a, **k: False)
    load_state = {"calls": 0}

    def _slow_load(*_args, **_kwargs):
        load_state["calls"] += 1
        time.sleep(0.05)
        return model, _Tokenizer(), "onnx"

    monkeypatch.setattr(kc, "_load_kompress", _slow_load)
    monkeypatch.setattr(
        router,
        "compress",
        lambda content, *, context="", bias=1.0: _compression_result(
            content,
            compressor.compress(content).compressed,
        ),
    )
    monkeypatch.setenv("HEADROOM_COMPRESSION_DEADLINE_MS", "10")

    started = time.perf_counter()
    result = router.apply(
        _messages(),
        _Tokenizer(),
        frozen_message_count=1,
        min_tokens_to_compress=1,
    )
    elapsed = time.perf_counter() - started
    time.sleep(0.1)

    assert elapsed < 0.12
    assert result.messages[1]["content"] == _messages()[1]["content"]
    assert "failing open via PASSTHROUGH" in caplog.text
    assert load_state["calls"] == 1
    assert model.calls == 0


def test_single_cache_miss_deadline_discards_late_policy_effects(monkeypatch):
    from headroom.transforms.content_router import _ACTIVE_POLICY_SIDE_EFFECT_TRANSACTION

    router = _router()
    state = {"discarded": 0, "committed": 0}
    finished = threading.Event()

    def slow_compress(content, *, context="", bias=1.0):
        time.sleep(0.05)
        transaction = _ACTIVE_POLICY_SIDE_EFFECT_TRANSACTION.get()
        assert transaction is not None
        transaction.register(
            "late-event",
            commit=lambda: state.__setitem__("committed", state["committed"] + 1),
            discard=lambda: state.__setitem__("discarded", state["discarded"] + 1),
        )
        finished.set()
        return _compression_result(content, "compressed output")

    monkeypatch.setattr(router, "compress", slow_compress)
    monkeypatch.setenv("HEADROOM_COMPRESSION_DEADLINE_MS", "5")

    result = router.apply(
        _messages(),
        _Tokenizer(),
        frozen_message_count=1,
        min_tokens_to_compress=1,
    )

    assert result.messages[1]["content"] == _messages()[1]["content"]
    assert finished.wait(timeout=1.0)
    assert state == {"discarded": 1, "committed": 0}


def test_deadline_reconciles_real_policy_ledger_store_and_toin(monkeypatch):
    from headroom.cache.compression_store import (
        get_compression_store,
        reset_compression_store,
    )
    from headroom.transforms.retrieval_aware_policy import get_compression_cost_ledger
    from headroom.transforms.smart_crusher import CrushResult

    monkeypatch.setenv("HEADROOM_CCR_BACKEND", "memory")
    monkeypatch.setenv("HEADROOM_COMPRESSION_DEADLINE_MS", "5")
    reset_compression_store()
    store = get_compression_store()
    ledger = get_compression_cost_ledger()
    ledger.clear()
    router = ContentRouter(
        ContentRouterConfig(
            retrieval_aware_enabled=True,
            retrieval_aware_forced_action="lossy",
            protect_recent_code=0,
            protect_analysis_context=False,
            exclude_tools=set(),
        )
    )
    toin_calls: list[dict] = []
    hash_key = "abcdef123456abcdef123456"

    class _Crusher:
        def crush(self, content, **kwargs):
            if kwargs.get("lossless_only"):
                return CrushResult(
                    json.dumps(json.loads(content), separators=(",", ":")),
                    content,
                    True,
                    "lossless_json",
                )
            handle = kwargs["retrieval_handle"]
            marker = json.dumps([{"_ccr": f"<<ccr:{hash_key}@{handle}>>"}])
            store.store(
                content,
                marker,
                explicit_hash=hash_key,
                compression_event_id=kwargs["compression_event_id"],
                retrieval_handle=handle,
            )
            return CrushResult(marker, content, True, "row_drop")

        def _record_to_toin(self, **kwargs):
            toin_calls.append(kwargs)

    router._get_smart_crusher = lambda: _Crusher()
    original_compress = router.compress

    def slow_compress(content, **kwargs):
        time.sleep(0.05)
        return original_compress(content, **kwargs)

    monkeypatch.setattr(router, "compress", slow_compress)
    content = json.dumps([{"id": i, "value": "x" * 20} for i in range(80)])
    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "catalog", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call-1", "content": content},
    ]

    try:
        result = router.apply(
            messages,
            _Tokenizer(),
            min_tokens_to_compress=1,
            request_id="request-1",
            session_id="session-1",
        )
        time.sleep(0.12)
        snapshot = ledger.snapshot()

        assert result.messages[1]["content"] == content
        assert snapshot["action_counts"] == {
            "passthrough": 1,
            "lossless": 0,
            "lossy": 0,
        }
        assert store.retrieve_for_internal_use(hash_key) is None
        assert toin_calls == []
    finally:
        ledger.clear()
        reset_compression_store()


def test_single_cache_miss_propagates_request_scoped_store(monkeypatch):
    from headroom.cache.compression_store import (
        CompressionStore,
        clear_request_compression_store,
        get_compression_store,
        set_request_compression_store,
    )

    router = _router()
    scoped_store = CompressionStore()
    observed: list[bool] = []

    def compress(content, *, context="", bias=1.0):
        observed.append(get_compression_store() is scoped_store)
        return _compression_result(content, "compressed output")

    monkeypatch.setattr(router, "compress", compress)
    monkeypatch.setenv("HEADROOM_COMPRESSION_DEADLINE_MS", "1000")
    set_request_compression_store(scoped_store)
    try:
        result = router.apply(
            _messages(),
            _Tokenizer(),
            frozen_message_count=1,
            min_tokens_to_compress=1,
        )
    finally:
        clear_request_compression_store()

    assert result.messages[1]["content"] == "compressed output"
    assert observed == [True]


def test_deadline_passthrough_does_not_poison_skip_cache(monkeypatch):
    router = _router()
    calls = {"count": 0}

    def compress(content, *, context="", bias=1.0):
        calls["count"] += 1
        if calls["count"] == 1:
            time.sleep(0.05)
        return _compression_result(content, "compressed output")

    monkeypatch.setattr(router, "compress", compress)
    monkeypatch.setenv("HEADROOM_COMPRESSION_DEADLINE_MS", "5")
    first = router.apply(
        _messages(),
        _Tokenizer(),
        frozen_message_count=1,
        min_tokens_to_compress=1,
    )
    assert first.messages[1]["content"] == _messages()[1]["content"]
    time.sleep(0.08)

    monkeypatch.setenv("HEADROOM_COMPRESSION_DEADLINE_MS", "1000")
    second = router.apply(
        _messages(),
        _Tokenizer(),
        frozen_message_count=1,
        min_tokens_to_compress=1,
    )

    assert second.messages[1]["content"] == "compressed output"
    assert calls["count"] == 2
