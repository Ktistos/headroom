# Retrieval-aware Codex benchmark

## Individual runs

| run | task | regime | condition | grader | valid | selected | predicted | retrieval calls | actual recoveries | gross savings tokens | recovery payload tokens | payload-net savings tokens | API requests | input tokens | cached input tokens | output tokens | wall s |
|---:|---|---|---|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | high_middle_alpha | high_retrieval | Historical behavior (observe) | 1 | 0 | lossy | lossless | 0 | 0 | 5,053 | 0 | 5,053 | 5 | 62,747 | 56,192 | 689 | 30.702 |
| 2 | high_middle_alpha | high_retrieval | Always lossless | 1 | 1 | lossless | lossless | 0 | 0 | 1,890 | 0 | 1,890 | 5 | 74,194 | 55,168 | 696 | 29.838 |
| 3 | high_middle_alpha | high_retrieval | Retrieval-aware | 1 | 1 | lossless | lossless | 0 | 0 | 1,890 | 0 | 1,890 | 5 | 74,330 | 63,872 | 720 | 30.353 |
| 4 | high_middle_beta | high_retrieval | Always lossless | 1 | 1 | lossless | lossless | 0 | 0 | 1,890 | 0 | 1,890 | 5 | 74,249 | 63,872 | 716 | 30.927 |
| 5 | high_middle_beta | high_retrieval | Retrieval-aware | 1 | 1 | lossless | lossless | 0 | 0 | 1,890 | 0 | 1,890 | 5 | 74,282 | 61,312 | 725 | 32.146 |
| 6 | high_middle_beta | high_retrieval | Historical behavior (observe) | 1 | 0 | lossy | lossless | 0 | 0 | 5,003 | 0 | 5,003 | 5 | 62,976 | 56,192 | 864 | 37.661 |
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
| high_retrieval | Historical behavior (observe) | 2/2 | {"lossy": 2} | 0 | 10,056 | 10 | 125,723 | 112,384 | 1,553 | 68.4 |
| high_retrieval | Always lossless | 2/2 | {"lossless": 2} | 0 | 3,780 | 10 | 148,443 | 119,040 | 1,412 | 60.8 |
| high_retrieval | Retrieval-aware | 2/2 | {"lossless": 2} | 0 | 3,780 | 10 | 148,612 | 125,184 | 1,445 | 62.5 |
| low_retrieval | Historical behavior (observe) | 2/2 | {"lossy": 2} | 0 | 10,199 | 10 | 125,120 | 108,800 | 1,457 | 56.3 |
| low_retrieval | Always lossless | 2/2 | {"lossless": 2} | 0 | 3,781 | 10 | 148,640 | 125,184 | 1,401 | 56.5 |
| low_retrieval | Retrieval-aware | 2/2 | {"lossy": 2} | 0 | 10,199 | 10 | 125,494 | 111,872 | 1,591 | 59.4 |
| passthrough | Historical behavior (observe) | 1/1 | {"lossless": 1} | 0 | 53 | 5 | 58,461 | 53,120 | 699 | 27.8 |
| passthrough | Always lossless | 1/1 | {"lossless": 1} | 0 | 53 | 5 | 58,347 | 53,120 | 675 | 28.6 |
| passthrough | Retrieval-aware | 1/1 | {"passthrough": 1} | 0 | 0 | 5 | 58,971 | 50,560 | 768 | 29.2 |
| all | Historical behavior (observe) | 5/5 | {"lossless": 1, "lossy": 4} | 0 | 20,308 | 25 | 309,304 | 274,304 | 3,709 | 152.5 |
| all | Always lossless | 5/5 | {"lossless": 5} | 0 | 7,614 | 25 | 355,430 | 297,344 | 3,488 | 145.8 |
| all | Retrieval-aware | 5/5 | {"lossless": 2, "lossy": 2, "passthrough": 1} | 0 | 13,979 | 25 | 333,077 | 287,616 | 3,804 | 151.1 |

Agent: `codex-cli 0.147.0`; model `gpt-5.4`; reasoning effort `low`; seed mode `fresh_random`.

High-retrieval tasks reveal an exact middle-record target only after catalog emission. Low-retrieval tasks infer schema from fields present in every retained sample row. The passthrough task is below the controller size threshold.
