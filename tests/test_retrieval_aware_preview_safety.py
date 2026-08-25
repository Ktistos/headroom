from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from headroom.config import SmartCrusherConfig
from headroom.transforms.compressor_registry import CompressInput
from headroom.transforms.content_router import (
    ContentRouter,
    ContentRouterConfig,
    _invoke_smart_crusher,
    _lossless_preview_is_safe,
)
from headroom.transforms.retrieval_aware_policy import get_compression_cost_ledger
from headroom.transforms.smart_crusher import SmartCrusher


def test_marker_free_lossy_preview_is_rejected(monkeypatch):
    ledger = get_compression_cost_ledger()
    ledger.clear()
    calls: list[bool | None] = []

    class FakeCrusher:
        def crush(self, content, **kwargs):
            calls.append(kwargs.get("lossless_only"))
            if kwargs.get("lossless_only"):
                # Mirrors the native scalar-array edge case: strict mode still
                # samples entries but emits no CCR marker.
                return SimpleNamespace(
                    compressed='["row-000"]',
                    was_modified=True,
                    strategy="string:adaptive(160->15)",
                )
            return SimpleNamespace(
                compressed='["row-000", "<<ccr:abcdef123456 145_rows_offloaded>>"]',
                was_modified=True,
                strategy="string:adaptive(160->15)",
            )

    router = ContentRouter(ContentRouterConfig(retrieval_aware_enabled=True))
    router._smart_crusher = FakeCrusher()
    monkeypatch.setattr(
        "headroom.transforms.content_router._estimate_tokens", lambda text: len(text)
    )
    content = json.dumps([f"row-{index:03d}" for index in range(160)], indent=2)

    output = _invoke_smart_crusher(
        router,
        CompressInput(
            content=content,
            content_type="application/json",
            query="Implement the exact catalog fix",
            config={"tool_name": "Read"},
        ),
    )

    assert output == content
    assert calls == [True]
    assert ledger.snapshot()["action_counts"]["passthrough"] == 1
    ledger.clear()


@pytest.mark.parametrize(
    ("original", "preview"),
    [
        ('{"value":1}', '{"value":1.0}'),
        ('{"value":1.0}', '{"value":1}'),
        ('{"value":1,"value":1}', '{"value":1}'),
        ('{"value":NaN}', '{"value":NaN}'),
        ('{"value":Infinity}', '{"value":Infinity}'),
        ('{"value":[1]}', '{"value":{"0":1}}'),
        ('{"value":true}', '{"value":1}'),
        ("[1,2]", "[2,1]"),
        ("01", "01"),
        ("+1", "+1"),
        ("1.", "1."),
        ("1e", "1e"),
        (".5", ".5"),
        ('"unterminated', '"unterminated'),
    ],
)
def test_type_sensitive_json_preview_rejects_unsafe_changes(original: str, preview: str):
    assert not _lossless_preview_is_safe(
        original, SimpleNamespace(compressed=preview, strategy="lossless_json")
    )


def test_type_sensitive_json_preview_ignores_only_whitespace_and_object_key_order():
    original = '{\n  "b": [1, 2.0],\n  "a": {"ok": true}\n}'
    preview = '{"a":{"ok":true},"b":[1,2.0]}'
    assert _lossless_preview_is_safe(
        original, SimpleNamespace(compressed=preview, strategy="lossless_json")
    )


def test_non_json_byte_identical_passthrough_is_safe_but_not_a_lossless_win():
    text = "ordinary plain text"
    assert _lossless_preview_is_safe(text, SimpleNamespace(compressed=text, strategy="passthrough"))


def test_preview_does_not_mutate_crusher_cache_or_emit_side_effects(monkeypatch):
    crusher = SmartCrusher(
        config=SmartCrusherConfig(min_tokens_to_crush=0, max_items_after_crush=10)
    )
    cached_before = dict(crusher._rust_by_lossless_only)
    monkeypatch.setattr(
        crusher,
        "_record_to_toin",
        lambda **kwargs: pytest.fail("preview wrote TOIN feedback"),
    )
    monkeypatch.setattr(
        crusher,
        "_mirror_ccr_to_python_store",
        lambda **kwargs: pytest.fail("preview wrote CCR storage"),
    )
    content = json.dumps([{"id": index, "value": "payload"} for index in range(80)])

    crusher.crush(content, lossless_only=True, record_outcome=False)

    assert crusher._rust_by_lossless_only == cached_before
