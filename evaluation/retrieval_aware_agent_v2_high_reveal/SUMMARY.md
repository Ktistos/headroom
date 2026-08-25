# Retrieval-aware Codex benchmark

> Raw source artifact: the recovery columns below are the run-time pre-envelope ledger values. The primary validated matrix corrects MCP envelope accounting from the retained trace and preserves both values in `evaluation/retrieval_aware_agent_v2_validated/results.json`.

## Individual runs

| run | task | regime | condition | grader | valid | selected | predicted | retrieval calls | actual recoveries | gross savings tokens | recovery payload tokens | payload-net savings tokens | API requests | input tokens | cached input tokens | output tokens | wall s |
|---:|---|---|---|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | high_middle_alpha | high_retrieval | Historical behavior (observe) | 1 | 1 | lossy | lossless | 1 | 1 | 5,358 | 5,434 | -76 | 10 | 178,057 | 154,880 | 1,650 | 58.234 |
| 2 | high_middle_alpha | high_retrieval | Always lossless | 1 | 1 | lossless | lossless | 0 | 0 | 1,891 | 0 | 1,891 | 6 | 92,185 | 81,152 | 822 | 34.76 |
| 3 | high_middle_alpha | high_retrieval | Retrieval-aware | 1 | 1 | lossless | lossless | 0 | 0 | 1,891 | 0 | 1,891 | 6 | 92,244 | 80,640 | 901 | 33.654 |
| 4 | high_middle_beta | high_retrieval | Always lossless | 1 | 1 | lossless | lossless | 0 | 0 | 1,891 | 0 | 1,891 | 6 | 92,184 | 80,640 | 901 | 41.83 |
| 5 | high_middle_beta | high_retrieval | Retrieval-aware | 1 | 1 | lossless | lossless | 0 | 0 | 1,891 | 0 | 1,891 | 6 | 92,840 | 80,640 | 969 | 37.813 |
| 6 | high_middle_beta | high_retrieval | Historical behavior (observe) | 0 | 0 | None | None | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 143.31 |

## Aggregates

| regime | condition | passed | actions | actual recoveries | payload-net savings tokens | API requests | input tokens | cached input tokens | output tokens | wall s |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| high_retrieval | Historical behavior (observe) | 1/2 | {"lossy": 1, "none": 1} | 1 | -76 | 10 | 178,057 | 154,880 | 1,650 | 201.5 |
| high_retrieval | Always lossless | 2/2 | {"lossless": 2} | 0 | 3,782 | 12 | 184,369 | 161,792 | 1,723 | 76.6 |
| high_retrieval | Retrieval-aware | 2/2 | {"lossless": 2} | 0 | 3,782 | 12 | 185,084 | 161,280 | 1,870 | 71.5 |
| all | Historical behavior (observe) | 1/2 | {"lossy": 1, "none": 1} | 1 | -76 | 10 | 178,057 | 154,880 | 1,650 | 201.5 |
| all | Always lossless | 2/2 | {"lossless": 2} | 0 | 3,782 | 12 | 184,369 | 161,792 | 1,723 | 76.6 |
| all | Retrieval-aware | 2/2 | {"lossless": 2} | 0 | 3,782 | 12 | 185,084 | 161,280 | 1,870 | 71.5 |

Agent: `codex-cli 0.147.0`; model `gpt-5.4`; reasoning effort `low`; seed mode `fresh_random`.

High-retrieval tasks reveal an exact middle-record target only after catalog emission. Low-retrieval tasks infer schema from fields present in every retained sample row. The passthrough task is below the controller size threshold.
