# Retrieval-aware Codex benchmark

> Raw source artifact: the recovery columns below are the run-time pre-envelope ledger values. The primary validated matrix corrects MCP envelope accounting from the retained trace and preserves both values in `evaluation/retrieval_aware_agent_v2_validated/results.json`.

## Individual runs

| run | task | regime | condition | grader | valid | selected | predicted | retrieval calls | actual recoveries | gross savings tokens | recovery payload tokens | payload-net savings tokens | API requests | input tokens | cached input tokens | output tokens | wall s |
|---:|---|---|---|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | high_middle_beta | high_retrieval | Historical behavior (observe) | 1 | 1 | lossy | lossless | 1 | 1 | 5,358 | 5,434 | -76 | 10 | 179,938 | 156,416 | 2,061 | 68.824 |
| 2 | high_middle_beta | high_retrieval | Always lossless | 1 | 1 | lossless | lossless | 0 | 0 | 1,890 | 0 | 1,890 | 6 | 92,101 | 80,640 | 804 | 33.136 |
| 3 | high_middle_beta | high_retrieval | Retrieval-aware | 1 | 1 | lossless | lossless | 0 | 0 | 1,890 | 0 | 1,890 | 6 | 92,738 | 80,640 | 940 | 38.621 |

## Aggregates

| regime | condition | passed | actions | actual recoveries | payload-net savings tokens | API requests | input tokens | cached input tokens | output tokens | wall s |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| high_retrieval | Historical behavior (observe) | 1/1 | {"lossy": 1} | 1 | -76 | 10 | 179,938 | 156,416 | 2,061 | 68.8 |
| high_retrieval | Always lossless | 1/1 | {"lossless": 1} | 0 | 1,890 | 6 | 92,101 | 80,640 | 804 | 33.1 |
| high_retrieval | Retrieval-aware | 1/1 | {"lossless": 1} | 0 | 1,890 | 6 | 92,738 | 80,640 | 940 | 38.6 |
| all | Historical behavior (observe) | 1/1 | {"lossy": 1} | 1 | -76 | 10 | 179,938 | 156,416 | 2,061 | 68.8 |
| all | Always lossless | 1/1 | {"lossless": 1} | 0 | 1,890 | 6 | 92,101 | 80,640 | 804 | 33.1 |
| all | Retrieval-aware | 1/1 | {"lossless": 1} | 0 | 1,890 | 6 | 92,738 | 80,640 | 940 | 38.6 |

Agent: `codex-cli 0.147.0`; model `gpt-5.4`; reasoning effort `low`; seed mode `fresh_random`.

High-retrieval tasks reveal an exact middle-record target only after catalog emission. Low-retrieval tasks infer schema from fields present in every retained sample row. The passthrough task is below the controller size threshold.
