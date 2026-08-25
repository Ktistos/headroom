# Retrieval-aware Codex benchmark

This is the primary, validated v2 evaluation. Across five tasks under three policies, all 15 included protocol-valid runs passed. The final protocol had 18 attempts: 17 valid, one invalid contaminated, and 15 included; two valid companion rows were superseded when the complete beta trio was rerun symmetrically with a fresh seed. See `VALIDATION.md` for the predeclared rule and retained evidence. Accepted-run reporting may overstate real-world agent compliance.

The raw run-time ledger charged the compact logical retrieval dictionary, not the serialized MCP content envelope. The validated artifact therefore applies a deterministic correction from two sanitized, commit-eligible retained MCP traces: 5,434 recorded tokens become 7,269 and 7,270 envelope tokens. Raw source artifacts remain unchanged, and `results.json` preserves both recorded and corrected values. No live run was repeated.

## Aggregate comparisons

- Retrieval-aware versus Historical behavior (observe): 13,980 versus 6,429 `payload_net_savings_tokens`, +7,551 tokens (2.17x); 0 versus 2 recoveries; 27 versus 35 API requests; and 172,129 fewer input tokens (31.8%).
- Retrieval-aware versus Always lossless: 13,980 versus 7,615 `payload_net_savings_tokens`, +6,365 tokens (1.84x); 27 API requests each; and 21,826 fewer input tokens (5.6%).
- Uncached input (`input - cached input`) was 68,360 for Historical behavior (observe), 51,177 for Always lossless, and 45,735 for Retrieval-aware. Cached and uncached input have different economic significance; no dollar-cost estimate is inferred.
- High retrieval: Retrieval-aware gained 7,604 payload-net tokens over Historical behavior (observe) and avoided two recoveries. Low retrieval: it gained 6,418 tokens, or 2.70x, over Always lossless. Passthrough: it selected passthrough as intended.

Retrieval-aware execution was observably faster than Historical behavior (observe) in this matrix, but 7.9 seconds slower than Always lossless. Because the experiment used one accepted run per condition and agent trajectories varied, no general latency improvement is claimed.

## Individual runs

| run | task | regime | condition | grader | valid | selected | predicted | retrieval calls | actual recoveries | gross savings tokens | recovery payload tokens | payload-net savings tokens | API requests | input tokens | cached input tokens | output tokens | wall s |
|---:|---|---|---|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | high_middle_alpha | high_retrieval | Historical behavior (observe) | 1 | 1 | lossy | lossless | 1 | 1 | 5,358 | 7,269 | -1,911 | 10 | 178,057 | 154,880 | 1,650 | 58.234 |
| 2 | high_middle_alpha | high_retrieval | Always lossless | 1 | 1 | lossless | lossless | 0 | 0 | 1,891 | 0 | 1,891 | 6 | 92,185 | 81,152 | 822 | 34.76 |
| 3 | high_middle_alpha | high_retrieval | Retrieval-aware | 1 | 1 | lossless | lossless | 0 | 0 | 1,891 | 0 | 1,891 | 6 | 92,244 | 80,640 | 901 | 33.654 |
| 4 | high_middle_beta | high_retrieval | Historical behavior (observe) | 1 | 1 | lossy | lossless | 1 | 1 | 5,358 | 7,270 | -1,912 | 10 | 179,938 | 156,416 | 2,061 | 68.824 |
| 5 | high_middle_beta | high_retrieval | Always lossless | 1 | 1 | lossless | lossless | 0 | 0 | 1,890 | 0 | 1,890 | 6 | 92,101 | 80,640 | 804 | 33.136 |
| 6 | high_middle_beta | high_retrieval | Retrieval-aware | 1 | 1 | lossless | lossless | 0 | 0 | 1,890 | 0 | 1,890 | 6 | 92,738 | 80,640 | 940 | 38.621 |
| 7 | low_search_alpha | low_retrieval | Retrieval-aware | 1 | 1 | lossy | lossy | 0 | 0 | 5,174 | 0 | 5,174 | 5 | 62,376 | 55,680 | 740 | 28.423 |
| 8 | low_search_alpha | low_retrieval | Historical behavior (observe) | 1 | 1 | lossy | lossy | 0 | 0 | 5,174 | 0 | 5,174 | 5 | 62,298 | 55,680 | 729 | 27.419 |
| 9 | low_search_alpha | low_retrieval | Always lossless | 1 | 1 | lossless | lossy | 0 | 0 | 1,891 | 0 | 1,891 | 5 | 74,237 | 61,312 | 673 | 27.568 |
| 10 | low_search_beta | low_retrieval | Historical behavior (observe) | 1 | 1 | lossy | lossy | 0 | 0 | 5,025 | 0 | 5,025 | 5 | 62,822 | 53,120 | 728 | 28.844 |
| 11 | low_search_beta | low_retrieval | Always lossless | 1 | 1 | lossless | lossy | 0 | 0 | 1,890 | 0 | 1,890 | 5 | 74,403 | 63,872 | 728 | 28.914 |
| 12 | low_search_beta | low_retrieval | Retrieval-aware | 1 | 1 | lossy | lossy | 0 | 0 | 5,025 | 0 | 5,025 | 5 | 63,118 | 56,192 | 851 | 30.983 |
| 13 | small_passthrough | passthrough | Always lossless | 1 | 1 | lossless | passthrough | 0 | 0 | 53 | 0 | 53 | 5 | 58,347 | 53,120 | 675 | 28.55 |
| 14 | small_passthrough | passthrough | Retrieval-aware | 1 | 1 | passthrough | passthrough | 0 | 0 | 0 | 0 | 0 | 5 | 58,971 | 50,560 | 768 | 29.167 |
| 15 | small_passthrough | passthrough | Historical behavior (observe) | 1 | 1 | lossless | passthrough | 0 | 0 | 53 | 0 | 53 | 5 | 58,461 | 53,120 | 699 | 27.842 |

## Aggregates

| regime | condition | passed | actions | actual recoveries | payload-net savings tokens | API requests | input tokens | cached input tokens | output tokens | wall s |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| high_retrieval | Historical behavior (observe) | 2/2 | {"lossy": 2} | 2 | -3,823 | 20 | 357,995 | 311,296 | 3,711 | 127.1 |
| high_retrieval | Always lossless | 2/2 | {"lossless": 2} | 0 | 3,781 | 12 | 184,286 | 161,792 | 1,626 | 67.9 |
| high_retrieval | Retrieval-aware | 2/2 | {"lossless": 2} | 0 | 3,781 | 12 | 184,982 | 161,280 | 1,841 | 72.3 |
| low_retrieval | Historical behavior (observe) | 2/2 | {"lossy": 2} | 0 | 10,199 | 10 | 125,120 | 108,800 | 1,457 | 56.3 |
| low_retrieval | Always lossless | 2/2 | {"lossless": 2} | 0 | 3,781 | 10 | 148,640 | 125,184 | 1,401 | 56.5 |
| low_retrieval | Retrieval-aware | 2/2 | {"lossy": 2} | 0 | 10,199 | 10 | 125,494 | 111,872 | 1,591 | 59.4 |
| passthrough | Historical behavior (observe) | 1/1 | {"lossless": 1} | 0 | 53 | 5 | 58,461 | 53,120 | 699 | 27.8 |
| passthrough | Always lossless | 1/1 | {"lossless": 1} | 0 | 53 | 5 | 58,347 | 53,120 | 675 | 28.6 |
| passthrough | Retrieval-aware | 1/1 | {"passthrough": 1} | 0 | 0 | 5 | 58,971 | 50,560 | 768 | 29.2 |
| all | Historical behavior (observe) | 5/5 | {"lossless": 1, "lossy": 4} | 2 | 6,429 | 35 | 541,576 | 473,216 | 5,867 | 211.2 |
| all | Always lossless | 5/5 | {"lossless": 5} | 0 | 7,615 | 27 | 391,273 | 340,096 | 3,702 | 152.9 |
| all | Retrieval-aware | 5/5 | {"lossless": 2, "lossy": 2, "passthrough": 1} | 0 | 13,980 | 27 | 369,447 | 323,712 | 4,200 | 160.8 |

Agent: `codex-cli 0.147.0`; model `gpt-5.4`; reasoning effort `low`; seed mode `fresh_random`.

High-retrieval tasks reveal an exact middle-record target only after catalog emission. Low-retrieval tasks infer schema from fields present in every retained sample row. The passthrough task is below the controller size threshold.
