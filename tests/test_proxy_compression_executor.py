"""Audit follow-up C3: bounded compression executor + cancel-aware metrics.

Replaces ``asyncio.to_thread`` for ``pipeline.apply()`` calls with a dedicated
``ThreadPoolExecutor`` that's bounded by ``ProxyConfig.compression_max_workers``.

Locks the following invariants:

1. The pool exists and respects ``compression_max_workers`` (auto and explicit).
2. ``compression_in_flight`` increments while a compression is running and
   decrements after it completes — under load, the high-water mark moves up
   as expected.
3. When a compression call exceeds its timeout, the awaiter unblocks with
   ``TimeoutError`` — but the worker thread keeps running (Python cannot
   preempt running CPython bytecode or in-flight Rust calls), and when the
   work eventually completes, ``compression_leaked_threads`` increments.
4. Jobs that time out while still queued do not leak the running gauge.
5. ``/stats runtime.compression_executor`` surfaces the gauges + counters so
   operators can see leaked-thread rate and queue pressure.
6. Once a timed-out worker is known to still be running, new compression work
   raises an asyncio timeout immediately until that worker exits instead of
   multiplying the timeout debt across the executor.

These tests also serve as documentation: anyone reading them sees that
"timeout fired" does not mean "compression was cancelled" — it means "we
stopped waiting; the worker is still going". A bounded pool plus the
leaked-thread counter is how we make that visible.
"""

from __future__ import annotations

import asyncio
import contextvars
import threading
import time

import pytest

pytest.importorskip("fastapi")

from headroom.proxy.helpers import COMPRESSION_TIMEOUT_SECONDS  # noqa: F401
from headroom.proxy.server import ProxyConfig, create_app


def _make_proxy(compression_max_workers: int | None = None):
    """Construct a HeadroomProxy with a no-op pipeline. Returns the proxy."""
    config = ProxyConfig(
        optimize=False,
        cache_enabled=False,
        rate_limit_enabled=False,
        cost_tracking_enabled=False,
        log_requests=False,
        ccr_inject_tool=False,
        ccr_handle_responses=False,
        ccr_context_tracking=False,
        image_optimize=False,
        compression_max_workers=compression_max_workers,
    )
    app = create_app(config)
    return app.state.proxy


def test_compression_executor_default_size_matches_cpu_count() -> None:
    """When ``compression_max_workers`` is None, the resolved size should
    match the host CPU count.
    """
    import os

    proxy = _make_proxy(compression_max_workers=None)
    expected = max(1, os.cpu_count() or 1)
    assert proxy.compression_max_workers == expected
    assert proxy._compression_executor._max_workers == expected


def test_compression_executor_explicit_override() -> None:
    """``ProxyConfig.compression_max_workers=N`` is honored verbatim."""
    proxy = _make_proxy(compression_max_workers=3)
    assert proxy.compression_max_workers == 3
    assert proxy._compression_executor._max_workers == 3


def test_compression_executor_minimum_one_worker() -> None:
    """A non-positive override clamps to 1 (zero workers would deadlock)."""
    proxy = _make_proxy(compression_max_workers=0)
    assert proxy.compression_max_workers == 1


def test_in_flight_gauge_tracks_running_compressions() -> None:
    """While a compression is running, ``_compression_in_flight`` reads ≥ 1.
    After it completes, it returns to 0. The high-water mark records the
    peak observed.
    """
    proxy = _make_proxy(compression_max_workers=4)

    enter_event = threading.Event()
    release_event = threading.Event()
    observed: dict[str, int] = {}

    def _slow_compression():
        enter_event.set()
        # Block until the test thread reads in_flight from the gauge.
        release_event.wait(timeout=5.0)
        return "done"

    async def _drive():
        task = asyncio.create_task(
            proxy._run_compression_in_executor(_slow_compression, timeout=10.0)
        )
        # Wait for the worker to actually start.
        for _ in range(50):
            if enter_event.is_set():
                break
            await asyncio.sleep(0.01)
        with proxy._compression_metrics_lock:
            observed["mid_flight"] = proxy._compression_in_flight
            observed["mid_flight_max"] = proxy._compression_in_flight_max
        release_event.set()
        result = await task
        return result

    result = asyncio.run(_drive())
    assert result == "done"
    assert observed["mid_flight"] == 1, (
        f"in_flight should be 1 mid-call, got {observed['mid_flight']}"
    )
    assert observed["mid_flight_max"] >= 1
    # Decremented after task completes.
    with proxy._compression_metrics_lock:
        assert proxy._compression_in_flight == 0


def test_high_water_mark_persists_after_completion() -> None:
    """``_compression_in_flight_max`` is monotonic — never decreases."""
    proxy = _make_proxy(compression_max_workers=8)

    enter_events = [threading.Event() for _ in range(3)]
    release_events = [threading.Event() for _ in range(3)]

    def _make_slow(idx: int):
        def _slow():
            enter_events[idx].set()
            release_events[idx].wait(timeout=5.0)
            return idx

        return _slow

    async def _drive():
        tasks = [
            asyncio.create_task(proxy._run_compression_in_executor(_make_slow(i), timeout=10.0))
            for i in range(3)
        ]
        # Wait for all 3 to enter.
        for ev in enter_events:
            for _ in range(50):
                if ev.is_set():
                    break
                await asyncio.sleep(0.01)
        peak = proxy._compression_in_flight
        for ev in release_events:
            ev.set()
        for t in tasks:
            await t
        return peak

    peak = asyncio.run(_drive())
    assert peak == 3, f"Should have observed 3 concurrent compressions, got {peak}"
    # After all complete, in_flight is back to 0 but max remains 3.
    with proxy._compression_metrics_lock:
        assert proxy._compression_in_flight == 0
        assert proxy._compression_in_flight_max >= 3


def test_timeout_fires_and_leaked_thread_is_counted() -> None:
    """When the compression exceeds ``timeout``, the awaiter sees
    ``TimeoutError`` immediately. The worker keeps running; when it finishes,
    ``_compression_leaked_threads`` increments by 1.
    """
    proxy = _make_proxy(compression_max_workers=2)
    finished_event = threading.Event()
    timeout_seconds = 0.10

    def _slow_compression():
        # Sleep well past the timeout so the asyncio side cancels first.
        time.sleep(timeout_seconds * 5)
        finished_event.set()
        return "completed-after-deadline"

    async def _drive():
        with pytest.raises(asyncio.TimeoutError):
            await proxy._run_compression_in_executor(_slow_compression, timeout=timeout_seconds)

    asyncio.run(_drive())

    # Wait for the worker to actually finish (it ran past the deadline).
    finished_event.wait(timeout=2.0)
    # Give the worker thread a moment to update the counter under the lock.
    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline:
        with proxy._compression_metrics_lock:
            if proxy._compression_leaked_threads >= 1:
                break
        time.sleep(0.01)

    with proxy._compression_metrics_lock:
        assert proxy._compression_leaked_threads >= 1, (
            f"leaked_threads should be ≥ 1; got {proxy._compression_leaked_threads}. "
            f"The worker either didn't finish past the deadline, or the wrapper "
            f"didn't increment the counter."
        )
        # In-flight gauge restored.
        assert proxy._compression_in_flight == 0


def test_timeout_quarantines_new_work_until_timed_out_worker_finishes() -> None:
    """One post-timeout worker must not admit more compression work.

    This is the production failure mode behind the executor cascade: the
    asyncio waiter times out, but its thread continues running. Without a
    quarantine, every subsequent request can occupy another worker and repeat
    the same timeout until the pool is exhausted.
    """
    proxy = _make_proxy(compression_max_workers=2)
    first_started = threading.Event()
    release_first = threading.Event()
    second_started = threading.Event()

    def _timed_out_compression():
        first_started.set()
        release_first.wait(timeout=5.0)
        return "late"

    def _second_compression():
        second_started.set()
        return "second"

    async def _drive():
        with pytest.raises(asyncio.TimeoutError):
            await proxy._run_compression_in_executor(_timed_out_compression, timeout=0.05)
        assert first_started.is_set()

        bypass_started = time.monotonic()
        try:
            # asyncio.TimeoutError is distinct from builtin TimeoutError on
            # Python 3.10. The quarantine signal must follow the former so the
            # existing handler failure policy classifies it as a timeout.
            with pytest.raises(asyncio.TimeoutError, match="quarantin"):
                await proxy._run_compression_in_executor(_second_compression, timeout=1.0)
            bypass_elapsed = time.monotonic() - bypass_started
            assert bypass_elapsed < 0.2
            assert not second_started.is_set()

            with proxy._compression_metrics_lock:
                assert proxy._compression_timed_out_in_flight == 1
                assert proxy._compression_quarantine_skips == 1
                assert proxy._compression_quarantine_activations == 1
        finally:
            release_first.set()

        for _ in range(100):
            with proxy._compression_metrics_lock:
                if proxy._compression_timed_out_in_flight == 0:
                    break
            await asyncio.sleep(0.01)

        with proxy._compression_metrics_lock:
            assert proxy._compression_timed_out_in_flight == 0
            assert proxy._compression_leaked_threads == 1

        # Quarantine is self-clearing: normal compression resumes after the
        # timed-out worker has genuinely left the executor.
        assert (
            await proxy._run_compression_in_executor(_second_compression, timeout=1.0) == "second"
        )
        return await proxy.metrics.export()

    prometheus_text = asyncio.run(_drive())
    assert 'headroom_compression_quarantine_total{event="activated"} 1' in prometheus_text
    assert 'headroom_compression_quarantine_total{event="skipped"} 1' in prometheus_text


def test_timeout_before_worker_start_does_not_leak_in_flight() -> None:
    """If a queued job times out before a worker starts, queued accounting
    is cleaned up without touching the running gauge.
    """
    proxy = _make_proxy(compression_max_workers=1)
    first_started = threading.Event()
    release_first = threading.Event()
    second_started = threading.Event()

    def _blocking_compression():
        first_started.set()
        release_first.wait(timeout=5.0)
        return "first"

    def _queued_compression():
        second_started.set()
        return "second"

    async def _drive():
        first_task = asyncio.create_task(
            proxy._run_compression_in_executor(_blocking_compression, timeout=10.0)
        )
        for _ in range(50):
            if first_started.is_set():
                break
            await asyncio.sleep(0.01)
        assert first_started.is_set()

        with pytest.raises(asyncio.TimeoutError):
            await proxy._run_compression_in_executor(_queued_compression, timeout=0.05)

        with proxy._compression_metrics_lock:
            mid_queued = proxy._compression_queued
            mid_in_flight = proxy._compression_in_flight
            queue_timeouts = proxy._compression_queue_timeouts

        release_first.set()
        assert await first_task == "first"
        return mid_queued, mid_in_flight, queue_timeouts

    mid_queued, mid_in_flight, queue_timeouts = asyncio.run(_drive())

    assert not second_started.is_set()
    assert mid_queued == 0
    assert mid_in_flight == 1
    assert queue_timeouts == 1
    with proxy._compression_metrics_lock:
        assert proxy._compression_queued == 0
        assert proxy._compression_in_flight == 0
        assert proxy._compression_leaked_threads == 0
        assert proxy._compression_timed_out_in_flight == 0
        assert proxy._compression_quarantine_activations == 0
        assert proxy._compression_quarantine_skips == 0


def test_compression_executor_skip_signal_remains_visible() -> None:
    """A compression executor queue timeout increments visible runtime counters."""
    from fastapi.testclient import TestClient

    config = ProxyConfig(
        optimize=False,
        cache_enabled=False,
        rate_limit_enabled=False,
        cost_tracking_enabled=False,
        log_requests=False,
        ccr_inject_tool=False,
        ccr_handle_responses=False,
        ccr_context_tracking=False,
        image_optimize=False,
        compression_max_workers=1,
    )
    app = create_app(config)
    proxy = app.state.proxy

    with TestClient(app) as client:
        baseline = client.get("/health").json()["runtime"]["compression_executor"][
            "queue_timeouts_total"
        ]

    first_started = threading.Event()
    release_first = threading.Event()

    def _blocking_compression():
        first_started.set()
        release_first.wait(timeout=5.0)
        return "first"

    def _queued_compression():
        return "second"

    async def _drive():
        first_task = asyncio.create_task(
            proxy._run_compression_in_executor(_blocking_compression, timeout=10.0)
        )
        for _ in range(50):
            if first_started.is_set():
                break
            await asyncio.sleep(0.01)
        assert first_started.is_set()

        with pytest.raises(asyncio.TimeoutError):
            await proxy._run_compression_in_executor(_queued_compression, timeout=0.05)

        with proxy._compression_metrics_lock:
            assert proxy._compression_queued == 0

        release_first.set()
        return await first_task

    asyncio.run(_drive())

    with TestClient(app) as client:
        after = client.get("/health").json()["runtime"]["compression_executor"]
        assert after["queue_timeouts_total"] == baseline + 1


def test_compression_executor_metrics_appear_in_runtime_payload() -> None:
    """``/stats runtime.compression_executor`` surfaces the new gauges."""
    from fastapi.testclient import TestClient

    config = ProxyConfig(
        optimize=False,
        cache_enabled=False,
        rate_limit_enabled=False,
        cost_tracking_enabled=False,
        log_requests=False,
        ccr_inject_tool=False,
        ccr_handle_responses=False,
        ccr_context_tracking=False,
        image_optimize=False,
        compression_max_workers=5,
    )
    app = create_app(config)

    with TestClient(app) as client:
        # The compression_executor metrics are published from the runtime
        # payload (also surfaced in /health). Hit /health and look there.
        r = client.get("/health")
        assert r.status_code == 200
        runtime = r.json()["runtime"]
        assert "compression_executor" in runtime
        ce = runtime["compression_executor"]
        assert ce["max_workers"] == 5
        assert ce["queued"] == 0
        assert ce["running"] == 0
        assert ce["in_flight"] == 0
        assert ce["queue_timeouts_total"] == 0
        assert ce["queue_wait_seconds_total"] == 0.0
        assert ce["run_seconds_total"] == 0.0
        assert ce["leaked_threads_total"] == 0
        assert ce["quarantine_active"] is False
        assert ce["timed_out_workers"] == 0
        assert ce["timed_out_workers_max"] == 0
        assert ce["quarantine_activations_total"] == 0
        assert ce["quarantine_skips_total"] == 0
        assert ce["source"] == "explicit"


def test_explicit_None_resolves_to_auto_source() -> None:
    """When max_workers is None (default), the runtime payload reports
    ``source: auto``."""
    from fastapi.testclient import TestClient

    config = ProxyConfig(
        optimize=False,
        cache_enabled=False,
        rate_limit_enabled=False,
        cost_tracking_enabled=False,
        log_requests=False,
        ccr_inject_tool=False,
        ccr_handle_responses=False,
        ccr_context_tracking=False,
        image_optimize=False,
    )
    app = create_app(config)
    with TestClient(app) as client:
        r = client.get("/health")
        assert r.json()["runtime"]["compression_executor"]["source"] == "auto"


def test_timeout_discards_effects_registered_by_late_executor_worker() -> None:
    from headroom.transforms.content_router import _ACTIVE_POLICY_SIDE_EFFECT_TRANSACTION

    proxy = _make_proxy(compression_max_workers=1)
    finished = threading.Event()
    state = {"discarded": 0, "committed": 0}

    def _late_compression():
        time.sleep(0.08)
        transaction = _ACTIVE_POLICY_SIDE_EFFECT_TRANSACTION.get()
        assert transaction is not None
        transaction.register(
            "late-provider-event",
            commit=lambda: state.__setitem__("committed", state["committed"] + 1),
            discard=lambda: state.__setitem__("discarded", state["discarded"] + 1),
        )
        finished.set()
        return "late"

    async def _drive():
        with pytest.raises(asyncio.TimeoutError):
            await proxy._run_compression_in_executor(_late_compression, timeout=0.01)

    asyncio.run(_drive())
    assert finished.wait(timeout=1.0)
    assert state == {"discarded": 1, "committed": 0}


def test_compression_executor_propagates_request_context() -> None:
    proxy = _make_proxy(compression_max_workers=1)
    request_scope = contextvars.ContextVar("test_request_scope", default="missing")

    async def _drive():
        token = request_scope.set("request-store")
        try:
            return await proxy._run_compression_in_executor(request_scope.get, timeout=1.0)
        finally:
            request_scope.reset(token)

    assert asyncio.run(_drive()) == "request-store"


def test_async_cancellation_discards_late_executor_effects() -> None:
    from headroom.transforms.content_router import _ACTIVE_POLICY_SIDE_EFFECT_TRANSACTION

    proxy = _make_proxy(compression_max_workers=1)
    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    state = {"discarded": 0, "committed": 0}

    def _compression():
        started.set()
        release.wait(timeout=1.0)
        transaction = _ACTIVE_POLICY_SIDE_EFFECT_TRANSACTION.get()
        assert transaction is not None
        transaction.register(
            "cancelled-provider-event",
            commit=lambda: state.__setitem__("committed", state["committed"] + 1),
            discard=lambda: state.__setitem__("discarded", state["discarded"] + 1),
        )
        finished.set()
        return "late"

    async def _drive():
        task = asyncio.create_task(proxy._run_compression_in_executor(_compression, timeout=10.0))
        while not started.is_set():
            await asyncio.sleep(0.001)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(_drive())
    release.set()
    assert finished.wait(timeout=1.0)
    assert state == {"discarded": 1, "committed": 0}


@pytest.mark.parametrize("adopted", [True, False])
def test_executor_effects_wait_for_request_provider_adoption(adopted: bool) -> None:
    from headroom.transforms.content_router import (
        RequestPolicySideEffectHolder,
        activate_request_policy_side_effect_holder,
        current_policy_side_effect_transaction,
        finalize_request_policy_side_effects,
    )

    proxy = _make_proxy(compression_max_workers=1)
    holder = RequestPolicySideEffectHolder()
    state = {"committed": 0, "discarded": 0}

    def _compression():
        transaction = current_policy_side_effect_transaction()
        assert transaction is not None
        transaction.register(
            "provider-adoption",
            commit=lambda: state.__setitem__("committed", state["committed"] + 1),
            discard=lambda: state.__setitem__("discarded", state["discarded"] + 1),
        )
        return "compressed"

    async def _drive():
        with activate_request_policy_side_effect_holder(holder):
            result = await proxy._run_compression_in_executor(
                _compression,
                timeout=1.0,
            )
            assert state == {"committed": 0, "discarded": 0}
            finalize_request_policy_side_effects(commit=adopted)
            return result

    assert asyncio.run(_drive()) == "compressed"
    assert state == {
        "committed": int(adopted),
        "discarded": int(not adopted),
    }


def test_http_request_transaction_uses_endpoint_status_as_adoption_boundary(monkeypatch) -> None:
    from fastapi.responses import JSONResponse
    from fastapi.testclient import TestClient

    from headroom.transforms.content_router import current_policy_side_effect_transaction

    monkeypatch.setenv("HEADROOM_RETRIEVAL_AWARE", "control")
    config = ProxyConfig(
        optimize=False,
        cache_enabled=False,
        rate_limit_enabled=False,
        cost_tracking_enabled=False,
        log_requests=False,
        ccr_inject_tool=False,
        ccr_handle_responses=False,
        ccr_context_tracking=False,
        image_optimize=False,
    )
    app = create_app(config)
    state = {"committed": 0, "discarded": 0}

    @app.get("/test-policy-adoption/{status_code}")
    async def _policy_adoption(status_code: int):
        transaction = current_policy_side_effect_transaction()
        assert transaction is not None
        transaction.register(
            f"http-status-{status_code}",
            commit=lambda: state.__setitem__("committed", state["committed"] + 1),
            discard=lambda: state.__setitem__("discarded", state["discarded"] + 1),
        )
        return JSONResponse({"status": status_code}, status_code=status_code)

    registered_route = app.router.routes.pop()
    catch_all_index = next(
        index
        for index, route in enumerate(app.router.routes)
        if getattr(route, "path", "") == "/{path:path}"
    )
    app.router.routes.insert(catch_all_index, registered_route)
    with TestClient(app) as client:
        accepted = client.get("/test-policy-adoption/200")
        rejected = client.get("/test-policy-adoption/502")

    assert accepted.status_code == 200
    assert rejected.status_code == 502
    assert state == {
        "committed": 1,
        "discarded": 1,
    }
