# Retrieval-aware compression for Headroom

## 1. Problem and motivating failure mode

Headroom can replace a large structured tool result with a smaller sample and a CCR marker. Emission-time savings alone can reward the wrong decision: if an omitted row is later needed, the agent retrieves the original in a serialized tool-result envelope and may make another request with history replay. Lossy compression can therefore look cheaper than a lossless alternative while costing more over the trajectory.

The motivating stress task asks for one exact middle record from a 180-row catalog. Historical behavior (observe) normally samples away that row. CCR keeps the task recoverable, but recovery can erase the apparent savings. This extension makes the trade-off an explicit deterministic action choice.

## 2. Relationship to existing Headroom feedback and TOIN

Headroom already had retrieval feedback. `CompressionStore` records retrievals, streaming handlers observe model-generated retrieval calls, and TOIN uses tool/query outcomes to adjust later compression aggressiveness. This extension does not claim otherwise. Its contribution is selection among passthrough, verified lossless, and lossy compression using predicted recovery cost, plus counterfactual observe mode and compression-event attribution.

The new cost ledger remains separate from upstream feedback. In legacy modes, streaming observations can still drive the pre-existing feedback path. With retrieval-aware accounting enabled they are diagnostic only: neither streaming observations nor internal reads train retrieval outcomes or count as model-visible recovery. Automatic and proactive paths learn only after the provider accepts the injected continuation; MCP learns after its result is successfully delivered to the host, subject to the host-injection limitation below.

## 3. Controller objective and equations

All actions use the same estimated input-token units:

```text
passthrough_cost = original_tokens
lossless_cost = verified_lossless_tokens
lossy_expected_cost =
    predicted_compressed_tokens
    + expected_retrieval_count * predicted_recovery_payload_tokens
    + probability_of_any_retrieval * predicted_extra_turn_tokens
```

Known-path payload prediction serializes the original into the expected MCP, Anthropic, OpenAI Chat, OpenAI Responses, Google, or generic recovery shape. Escaping, Unicode, nesting, and the production 24-character hash width therefore affect the estimate. Unknown paths retain a configurable 64-token envelope fallback. The estimator is `ceil(UTF-8 bytes / 4)`, not a provider tokenizer. Payload cost stays separate from the configurable 1,024-token extra-turn fallback for arguments, wrappers, and replay; no safe request-context estimate is available at the early router boundary.

The savings floor is relative to passthrough. Retrieval probability uses mature unique compression events retrieved at least once, while retrieval count estimates `E[number of retrievals]`; repeats cannot push probability above one. Positive outcomes mature immediately. Because request completion does not prove a conversation is finished, production negative outcomes use a configurable 300-second window. Cold-start priors remain separate from smoothed per-tool history, which activates after three mature outcomes. Unknown-tool and preview/shadow events do not contaminate named-tool learning.

## 4. Verified-lossless safety finding

`lossless_only=True` was not a sufficient safety certificate: native output could be marker-free while changing a scalar array. The controller now validates the actual preview. JSON equality is type-sensitive, so `1` differs from `1.0`; duplicate keys, `NaN`, infinities, structural changes, and scalar-type changes are rejected. Object-key order and insignificant whitespace are ignored.

The selected lossless action reuses the validated preview rather than rerunning a potentially randomized crusher. Preview calls suppress storage, cache mutation, TOIN/feedback, policy observations, telemetry, and retrieval/savings statistics.

## 5. Architecture and integration

The feature is disabled by default and adds no LLM calls. Its optional background warmer is disabled in retrieval-aware mode because speculative results lack trustworthy emission and tenant-store boundaries. A request-level transaction spans nested worker pools. Worker completion transfers effects to the request owner; multi-row provider batches first isolate each row in a child transaction. Rejecting an inflated or failed row therefore cannot erase earlier accepted rows. HTTP effects commit only for a successful endpoint response. Responses WebSocket frames are staged before transport and commit only on the matching upstream `response.created`; pre-acceptance errors and disconnects discard them. Provider failure, request cancellation, and measurement-only preview remove the un-emitted event and store candidate while rolling back deferred caches, TOIN, policy observers, and metrics, including effects registered later by an unpreemptable worker. A tokenizer, batching, reversibility, multi-span, net-cost, or inner-deadline gate that actually sends the original instead records passthrough. Transient inner deadlines do not poison the skip cache.

Attribution carries session, request, tool-call, provider-slot, compression-event, hash, tool, and strategy identifiers through string/list Chat Completions output, Anthropic blocks, Responses units, batching, HTTP, and WebSocket routing. OpenAI Batch creates a unique row scope. Legacy caching is unchanged when the feature is off; retrieval-aware cache entries publish only after transaction commit.

Each lossy event gets an opaque `rh-...` handle embedded as `<<ccr:HASH@HANDLE...>>`. Legacy hash markers still work when unique, while ambiguous hash-only calls fail closed. SQLite adds, resolves, and discards candidates atomically. Retrieval feedback increments in the same transaction, same-content upserts preserve prior feedback, and bounded discard tombstones prevent stale snapshots from resurrecting rejected events.

Recovery reports have unique IDs. A thread-safe 50,000-entry LRU keeps accepting reports after rollover and exposes evictions. Event-scoped retries cannot be charged to a newer same-content event. When a metadata-free report ID rolls out, its hash enters a separate bounded fail-closed hazard LRU; unrelated legacy hashes remain usable instead of global accounting shutting down. MCP JSON-RPC IDs retain their numeric/string distinction and are scoped to the transport session; numeric ID zero is stable. Because the ledger and LRUs are process-local, retrieval-aware startup rejects `--workers > 1`.

`/v1/retrieve/stats` preserves store/recent fields and adds `retrieval_aware.ledger`; `realized_cost` is only a deprecated alias to that same object. Accounting endpoints require both peer-socket and Host loopback checks. Forwarded headers grant no authority. A configured accounting secret fails closed when missing or wrong; if unset, every loopback process is trusted. Default CORS remains restricted.

## 6. Accounting definitions

```text
gross_savings_tokens = original_tokens - initially_emitted_tokens
recovery_payload_tokens = estimated tokens actually injected into a model turn
payload_net_savings_tokens = gross_savings_tokens - recovery_payload_tokens
```

Local/shared SQLite MCP and proxy fallback charge the same model-shaped MCP `content` envelope only after the MCP SDK successfully sends the result to its host; proxy fallback's raw fetch is non-charging. The server cannot observe whether a nonconforming host later suppresses that delivered result instead of injecting it into the model, so standard MCP host behavior is an explicit accounting assumption. POST `/v1/retrieve`, GET `/v1/retrieve/{hash}`, and `/v1/retrieve/tool_call` are client transport only. `/tool_call` returns a provider result and pending idempotent report; `/v1/retrieve/account` charges only after injection, covering the provider `tool_result`, `function_call_output`, or `functionResponse`, never duplicated response `data`. Automatic provider recovery charges the accepted continuation payload once, including shared-envelope allocation across multiple results.

Outbound inline resolution is client-visible expansion and has separate counters. Proactive expansion registers its formatted model-input block transactionally and charges it only after the provider accepts the outbound request; a failed or reverted request discards the charge and retrieval feedback. Streaming observation and internal reads are non-charging. Thus an observation read followed by real MCP recovery is not charged twice.

Two retained MCP rows originally used a pre-envelope logical dictionary. The validated artifact reconstructs corrected model-visible values from sanitized traces with the production estimator, preserves historical values, and leaves raw sources unchanged.

These metrics are not full trajectory cost. `trajectory_input_tokens`, cached input, output, API requests, and wall time come from Codex/provider usage and include replay, caching, wrappers, and behavioral variation. Current API output uses only the explicit compression-event names above.

## 7. Experimental protocol

The validated v2 matrix has two delayed-target high-retrieval tasks, two low-retrieval schema tasks, and one small passthrough task. Each runs Historical behavior (observe), Always lossless, and Retrieval-aware on identical paired input and prompt, with rotating condition order and fresh Codex process, agent home, proxy, ledger, SQLite store, source, and workspace. Observe executes the lossless preview while emitting historical behavior, so it is a behavioral baseline, not a clean historical runtime-overhead baseline.

High tasks reveal the required position only after the catalog command. Predeclared validity rules reject prohibited source/database access, hidden-path inspection, duplicate or reversed source commands, missing primary events, nonzero exit, or missing required recovery attribution. Failing to call MCP, reasoning incorrectly, or answering wrongly without bypassing isolation remains a behavioral failure.

There were 18 attempts: 17 valid, one invalid contaminated, and 15 included. The excluded beta observe trace read both prohibited tool sources and searched the hidden source directory. Those isolation violations—not its wrong answer—invalidated measurement. Its replacement used fresh seed `2927455522424617784`; all three policies were rerun. Two valid original companion rows were superseded, not marked invalid. Accepted-run reporting may therefore overstate real-world compliance.

A machine manifest maps every included row, records invalid/superseded attempts, and declares two trace-derived accounting corrections. The offline rebuild parses those traces, validates the complete 5-by-3 matrix, excludes invalid rows, and recomputes all aggregates.

## 8. Per-regime and aggregate results

| policy | actions | gross | recovery payload | payload-net | recoveries | requests | input | cached | uncached | output | wall s |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Historical behavior (observe) | 1 lossless, 4 lossy | 20,968 | 14,539 | 6,429 | 2 | 35 | 541,576 | 473,216 | 68,360 | 5,867 | 211.2 |
| Always lossless | 5 lossless | 7,615 | 0 | 7,615 | 0 | 27 | 391,273 | 340,096 | 51,177 | 3,702 | 152.9 |
| Retrieval-aware | 2 lossless, 2 lossy, 1 passthrough | 13,980 | 0 | 13,980 | 0 | 27 | 369,447 | 323,712 | 45,735 | 4,200 | 160.8 |

Across five validated tasks under three policies, all protocol-valid included runs passed; this is not statistical equivalence and is qualified by the contaminated-run accounting above. Compared with Historical behavior (observe), Retrieval-aware improved payload-net savings by 7,551 tokens (2.17x), avoided two recoveries, made 27 rather than 35 requests, and used 172,129 fewer input tokens (31.8%). Compared with Always lossless, it gained 6,365 payload-net tokens (1.84x) and used 21,826 fewer input tokens (5.6%) with the same 27 requests.

In high retrieval it gained 7,604 payload-net tokens over observe, `3,781 - (-3,823)`, and avoided two recoveries. In low retrieval it gained 6,418 over Always lossless, `10,199 - 3,781`, or 2.70x. It selected passthrough on the passthrough task as intended.

## 9. Ablations and metric reconciliation

Historical observe tests recoverable lossy behavior; Always lossless tests whether the controller merely avoids loss. The mixed Retrieval-aware actions and its 1.84x payload-net result over Always lossless show that its benefit is not simply choosing lossless.

Payload-net and provider input differences are not interchangeable. The former covers compression and injected recovery payload; the latter also includes request count, replay, caching, wrappers, and behavior. Cached and uncached input have different economic significance, but no unsupported dollar estimate is made.

Retrieval-aware was faster than observe in this matrix but 7.9 seconds slower than Always lossless. One accepted trajectory per condition is insufficient for a general latency claim.

The legacy v1 stress comparison mixed recovered payload with fixed overhead. Its +2,876 / 2.45x result is retained only as historical evidence and is not comparable with corrected payload-net accounting.

## 10. Threats to validity and limitations

The evaluation uses five synthetic tasks, one model, one reasoning setting, and one accepted trajectory per condition. The tasks deliberately separate high-, low-, and passthrough regimes but do not calibrate close policy thresholds. Repeated identical content, delayed continuations, handle attribution, and estimator boundaries are deterministic tests, not live-matrix cases. Order rotates but is not fully randomized.

Provider token differences include caching and behavioral variation. Observe's preview prevents it from isolating historical runtime overhead. The byte estimator is not a provider tokenizer, provider shapes may change, and the extra-turn/context fallback remains approximate.

Learning, the event ledger, and idempotency LRUs are process-local; SQLite persists content/attribution but not priors. Retrieval-aware mode therefore supports one proxy worker. No provider supplies a trustworthy session-final signal, so negative outcomes use the observation window. Idempotency and legacy-hazard retention are deliberately bounded: after both horizons expire, only an event-scoped handle can safely identify a very old retry, and an MCP host that hides invocation identity cannot deduplicate whole-invocation retries. MCP delivery is observable only through the successful SDK response boundary, not the host's subsequent model injection. Broader natural tasks, repeated trials, persistent learning, and a safe request-context estimate remain future work.

## 11. Reproduction commands and artifacts

Primary files are under `evaluation/retrieval_aware_agent_v2_validated/`: `results.json`, `SUMMARY.md`, `manifest.json`, `MANIFEST.md`, and `VALIDATION.md`. Source rows and invalid evidence are referenced there.

```bash
python benchmarks/rebuild_retrieval_aware_validated.py --check

pytest -q \
  tests/test_retrieval_aware_policy.py \
  tests/test_retrieval_aware_preview_safety.py \
  tests/test_retrieval_aware_benchmark.py \
  tests/test_retrieval_aware_artifact_rebuild.py \
  tests/test_compression_units.py tests/test_compression_batches.py \
  tests/test_ccr_mcp_server.py tests/test_ccr_response_handler.py \
  tests/test_proxy_ccr.py

ruff check headroom tests benchmarks
ruff format --check headroom tests benchmarks
```

A fresh live matrix can be generated with `benchmarks/retrieval_aware_agent_benchmark.py`, but it creates new seeds and is not the retained-artifact reproduction command.
