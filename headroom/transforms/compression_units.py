"""Provider-neutral compression units.

Provider adapters own request-envelope details and cache/live-zone decisions.
They should extract only safe, mutable text ranges into ``CompressionUnit``
objects, ask ContentRouter to compress each unit, then splice accepted
replacements back into their native request shape.
"""

from __future__ import annotations

import inspect
import re
from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from typing import Protocol

from .content_router import (
    CompressionStrategy,
    ContentRouter,
    RouterCompressionResult,
    _adopt_policy_passthrough,
    policy_side_effect_transaction,
)


class TokenCounterLike(Protocol):
    def count_text(self, text: str) -> int: ...


@dataclass(frozen=True)
class CompressionUnit:
    """One provider-extracted, cache-safe text slot."""

    text: str
    provider: str
    endpoint: str
    role: str
    item_type: str
    cache_zone: str = "live"
    mutable: bool = True
    context: str = ""
    question: str | None = None
    bias: float = 1.0
    min_bytes: int = 512
    metadata: dict[str, str] = field(default_factory=dict)


# Categorical buckets for unit-level outcomes. Lets log readers filter
# by "what kind of decision" without parsing per-event reason strings.
# - applied:           the compressor ran and produced shorter bytes
# - protected_role:    role guard (user/system/assistant) refused compression
# - cache_zone:        unit lived in a non-live cache zone (e.g. prefix)
# - size_floor:        text_bytes < min_bytes — too small to be worth it
# - immutable:         caller marked the unit unmutable
# - compressor_noop:   router returned identical bytes (no compression possible)
# - already_compressed: input already carried a CCR retrieval marker
# - rejected_not_smaller: compressor produced output >= input tokens
# - cache_hit:         reserved; cached unit reuse is caller-level (the OpenAI
#                      Responses handler's unit-result cache) and is surfaced
#                      via RouterCompressionResult.cache_hit, not as a reason
#                      here — no code path produces this category today
UNIT_REASON_CATEGORIES = {
    None: "applied",
    "protected_user_message": "protected_role",
    "protected_system_message": "protected_role",
    "protected_assistant_message": "protected_role",
    "immutable": "immutable",
    "below_unit_floor": "size_floor",
    "router_no_change": "compressor_noop",
    "already_compressed": "already_compressed",
    "rejected_not_smaller": "rejected_not_smaller",
}


def _categorize_reason(reason: str | None) -> str:
    if reason is None:
        return "applied"
    if reason in UNIT_REASON_CATEGORIES:
        return UNIT_REASON_CATEGORIES[reason] or "applied"
    # cache_zone_* uses a dynamic suffix (cache_zone_frozen, cache_zone_prefix, …)
    if reason.startswith("cache_zone_"):
        return "cache_zone"
    return "other"


@dataclass(frozen=True)
class UnitCompressionResult:
    original: str
    compressed: str
    modified: bool
    tokens_before: int
    tokens_after: int
    tokens_saved: int
    transforms_applied: list[str]
    strategy: str
    reason: str | None = None
    router_result: RouterCompressionResult | None = None
    # Context for log readers: why the outcome looked the way it did.
    # `text_bytes` + `min_bytes` together explain size_floor decisions;
    # `reason_category` is the high-level bucket for dashboard grouping.
    text_bytes: int = 0
    min_bytes: int = 0
    reason_category: str = "applied"


@dataclass(frozen=True)
class RoutedCompressionUnit:
    """A unit paired with its provider-owned slot reference."""

    unit: CompressionUnit
    slot: object


_CCR_MARKER_RE = re.compile(
    r"(?m)^.*(?:Retrieve more: hash=|Retrieve original: hash=|<<ccr:[^>]+>>).*$"
)

_LOSSY_UNMARKED_STRATEGIES = {
    CompressionStrategy.KOMPRESS.value,
    CompressionStrategy.TEXT.value,
    CompressionStrategy.CODE_AWARE.value,
}


def _is_structured_shell_output(text: str) -> bool:
    nonempty_lines = [line for line in text.splitlines() if line.strip()]
    return len(nonempty_lines) >= 3


def find_content_router(transforms: object) -> ContentRouter | None:
    """Return the first ContentRouter in a pipeline or iterable."""

    candidates = getattr(transforms, "transforms", transforms)
    if not isinstance(candidates, Iterable):
        return None
    for transform in candidates:
        if isinstance(transform, ContentRouter):
            return transform
    return None


def _compress_live_text_with_markers(
    unit: CompressionUnit,
    *,
    router: ContentRouter,
) -> tuple[str, list[str], RouterCompressionResult | None]:
    """Compress text around CCR markers while preserving marker bytes."""

    parts: list[str] = []
    transforms: list[str] = []
    router_results: list[RouterCompressionResult] = []
    last_end = 0

    try:
        for match in _CCR_MARKER_RE.finditer(unit.text):
            prefix = unit.text[last_end : match.start()]
            if prefix:
                compressed_prefix, prefix_transforms, prefix_result = _compress_marker_free_text(
                    prefix,
                    unit=unit,
                    router=router,
                )
                if prefix_result is not None:
                    router_results.append(prefix_result)
                parts.append(compressed_prefix)
                transforms.extend(prefix_transforms)
            parts.append(match.group(0))
            last_end = match.end()

        suffix = unit.text[last_end:]
        if suffix:
            compressed_suffix, suffix_transforms, suffix_result = _compress_marker_free_text(
                suffix,
                unit=unit,
                router=router,
            )
            if suffix_result is not None:
                router_results.append(suffix_result)
            parts.append(compressed_suffix)
            transforms.extend(suffix_transforms)
    except Exception:
        # The provider unit is emitted atomically. If a later span fails, none
        # of the earlier span results can reach the provider, so reconcile all
        # policy/store candidates created so far before propagating the error.
        _discard_unemitted_policy_events(_combine_router_results(router_results), router)
        raise

    if transforms:
        transforms.insert(0, "ccr_marker_preserving")

    replacement = "".join(parts)
    return (
        replacement,
        transforms,
        _combine_router_results(
            router_results,
            original=unit.text,
            compressed=replacement,
        ),
    )


def _combine_router_results(
    results: list[RouterCompressionResult],
    *,
    original: str | None = None,
    compressed: str | None = None,
) -> RouterCompressionResult | None:
    """Combine independently compressed spans into one attributable result."""

    if not results:
        return None
    event_ids = list(
        dict.fromkeys(
            event_id for result in results for event_id in result.compression_event_ids if event_id
        )
    )
    event_hashes = {
        event_id: tuple(hashes)
        for result in results
        for event_id, hashes in result.compression_event_hashes.items()
    }
    return replace(
        results[-1],
        original=results[-1].original if original is None else original,
        compressed=results[-1].compressed if compressed is None else compressed,
        routing_log=[decision for result in results for decision in result.routing_log],
        sections_processed=sum(result.sections_processed for result in results),
        strategy_chain=[strategy for result in results for strategy in result.strategy_chain],
        cache_hit=all(result.cache_hit for result in results),
        compression_event_ids=event_ids,
        compression_event_hashes=event_hashes,
    )


def _router_attribution_kwargs(
    router: ContentRouter, metadata: dict[str, str]
) -> dict[str, dict[str, str]]:
    try:
        supported = "attribution" in inspect.signature(router.compress).parameters
    except (TypeError, ValueError):
        supported = False
    return {"attribution": metadata} if supported else {}


def _discard_unemitted_policy_events(
    router_result: RouterCompressionResult | None, router: ContentRouter
) -> None:
    if router_result is None or not router_result.compression_event_ids:
        return
    # Use the same reconciliation path as ContentRouter's own outer gates.
    # Merely changing the ledger action would leave a never-emitted producer
    # in the shared CCR store, making later same-content hash-only retrievals
    # ambiguous.
    _adopt_policy_passthrough(
        router,
        router_result.compression_event_ids,
        strategy="provider_unit_gate_passthrough",
        event_hashes=router_result.compression_event_hashes,
    )


def _compress_marker_free_text(
    text: str,
    *,
    unit: CompressionUnit,
    router: ContentRouter,
) -> tuple[str, list[str], RouterCompressionResult | None]:
    boundary = re.match(r"^(\s*)(.*?)(\s*)$", text, flags=re.DOTALL)
    if boundary is None:
        return text, [], None

    leading, core, trailing = boundary.groups()
    if len(core.encode("utf-8", errors="replace")) < unit.min_bytes:
        return text, [], None

    router_result = router.compress(
        core,
        context=unit.context,
        question=unit.question,
        bias=unit.bias,
        tool_name=unit.metadata.get("tool_name") or None,
        **_router_attribution_kwargs(router, unit.metadata),
    )
    if router_result.compressed == core:
        return text, [], router_result

    strategy = router_result.strategy_used.value
    return (
        f"{leading}{router_result.compressed}{trailing}",
        [
            f"router:{unit.provider}:{unit.endpoint}:{unit.item_type}:{strategy}",
            strategy,
        ],
        router_result,
    )


def compress_unit_with_router(
    unit: CompressionUnit,
    *,
    router: ContentRouter,
    tokenizer: TokenCounterLike,
    target_ratio: float | None = None,
) -> UnitCompressionResult:
    with policy_side_effect_transaction():
        return _compress_unit_with_router_in_transaction(
            unit,
            router=router,
            tokenizer=tokenizer,
            target_ratio=target_ratio,
        )


def _compress_unit_with_router_in_transaction(
    unit: CompressionUnit,
    *,
    router: ContentRouter,
    tokenizer: TokenCounterLike,
    target_ratio: float | None = None,
) -> UnitCompressionResult:
    """Compress one safe text unit through ContentRouter.

    The final accept/reject gate uses the provider/model tokenizer, not the
    router's internal word-count estimates.
    """

    tokens_before = tokenizer.count_text(unit.text)
    text_bytes = len(unit.text.encode("utf-8", errors="replace"))
    base = UnitCompressionResult(
        original=unit.text,
        compressed=unit.text,
        modified=False,
        tokens_before=tokens_before,
        tokens_after=tokens_before,
        tokens_saved=0,
        transforms_applied=[],
        strategy=CompressionStrategy.PASSTHROUGH.value,
        router_result=None,
        text_bytes=text_bytes,
        min_bytes=unit.min_bytes,
        reason_category="applied",
    )

    def _with_reason(**kw: object) -> UnitCompressionResult:
        # Every early-return path goes through here so reason_category
        # stays in sync with reason. Log readers grep by category for
        # quick "how many units were size-floored this hour" answers.
        reason_val = kw.get("reason")
        if isinstance(reason_val, str) or reason_val is None:
            kw["reason_category"] = _categorize_reason(reason_val)
        return replace(base, **kw)  # type: ignore[arg-type]

    if not unit.mutable:
        return _with_reason(reason="immutable")
    if unit.role == "user":
        return _with_reason(reason="protected_user_message")
    if unit.role in {"system", "developer"}:
        return _with_reason(reason="protected_system_message")
    if unit.role == "assistant" and unit.metadata.get("compress_assistant") != "true":
        return _with_reason(reason="protected_assistant_message")
    if unit.cache_zone != "live":
        return _with_reason(reason=f"cache_zone_{unit.cache_zone}")
    if text_bytes < unit.min_bytes:
        return _with_reason(reason="below_unit_floor")

    prior_target_ratio = getattr(router, "_runtime_target_ratio", None)
    if target_ratio is not None:
        router._runtime_target_ratio = target_ratio
    if _CCR_MARKER_RE.search(unit.text):
        try:
            replacement, marker_transforms, router_result = _compress_live_text_with_markers(
                unit,
                router=router,
            )
        finally:
            if target_ratio is not None:
                router._runtime_target_ratio = prior_target_ratio
        if replacement == unit.text:
            return _with_reason(
                router_result=router_result,
                reason="already_compressed",
            )

        tokens_after = tokenizer.count_text(replacement)
        if tokens_after >= tokens_before:
            _discard_unemitted_policy_events(router_result, router)
            return _with_reason(
                compressed=replacement,
                tokens_after=tokens_after,
                router_result=router_result,
                reason="rejected_not_smaller",
            )

        return UnitCompressionResult(
            original=unit.text,
            compressed=replacement,
            modified=True,
            tokens_before=tokens_before,
            tokens_after=tokens_after,
            tokens_saved=tokens_before - tokens_after,
            transforms_applied=marker_transforms,
            strategy="ccr_marker_preserving",
            reason=None,
            router_result=router_result,
            text_bytes=text_bytes,
            min_bytes=unit.min_bytes,
            reason_category="applied",
        )

    try:
        router_result = router.compress(
            unit.text,
            context=unit.context,
            question=unit.question,
            bias=unit.bias,
            tool_name=unit.metadata.get("tool_name") or None,
            **_router_attribution_kwargs(router, unit.metadata),
        )
    finally:
        if target_ratio is not None:
            router._runtime_target_ratio = prior_target_ratio
    replacement = router_result.compressed
    strategy = router_result.strategy_used.value
    if replacement == unit.text:
        return _with_reason(
            strategy=strategy,
            router_result=router_result,
            reason="router_no_change",
        )

    tokens_after = tokenizer.count_text(replacement)
    if tokens_after >= tokens_before:
        _discard_unemitted_policy_events(router_result, router)
        return _with_reason(
            compressed=replacement,
            tokens_after=tokens_after,
            strategy=strategy,
            router_result=router_result,
            reason="rejected_not_smaller",
        )

    if (
        unit.role == "tool"
        and unit.item_type == "local_shell_call_output"
        and _is_structured_shell_output(unit.text)
        and strategy in _LOSSY_UNMARKED_STRATEGIES
    ):
        if not _CCR_MARKER_RE.search(replacement):
            _discard_unemitted_policy_events(router_result, router)
            return _with_reason(
                strategy=strategy,
                router_result=router_result,
                reason="lossy_unrecoverable_tool_output",
            )

    return UnitCompressionResult(
        original=unit.text,
        compressed=replacement,
        modified=True,
        tokens_before=tokens_before,
        tokens_after=tokens_after,
        tokens_saved=tokens_before - tokens_after,
        transforms_applied=[
            f"router:{unit.provider}:{unit.endpoint}:{unit.item_type}:{strategy}",
            strategy,
        ],
        strategy=strategy,
        reason=None,
        router_result=router_result,
        text_bytes=text_bytes,
        min_bytes=unit.min_bytes,
        reason_category="applied",
    )


def compress_units_with_router(
    units: Iterable[RoutedCompressionUnit],
    *,
    router: ContentRouter,
    tokenizer: TokenCounterLike,
) -> list[tuple[object, UnitCompressionResult]]:
    """Compress provider-extracted units and preserve provider slot refs.

    Provider adapters use this when they have many candidate text slots in one
    request envelope. The slot object is intentionally opaque here; only the
    provider adapter knows how to splice the result back into its native shape.
    """

    return [
        (routed.slot, compress_unit_with_router(routed.unit, router=router, tokenizer=tokenizer))
        for routed in units
    ]
