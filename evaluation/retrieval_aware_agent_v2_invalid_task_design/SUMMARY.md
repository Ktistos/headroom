# Retrieval-aware Codex benchmark

## Individual runs

| run | task | regime | condition | grader | valid | selected | predicted | retrieval calls | actual recoveries | gross savings tokens | recovery payload tokens | payload-net savings tokens | API requests | input tokens | cached input tokens | output tokens | wall s |
|---:|---|---|---|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | high_middle_alpha | high_retrieval | Historical behavior (observe) | 0 | 0 | lossy | lossless | 0 | 0 | 5,198 | 0 | 5,198 | 13 | 0 | 0 | 0 | 191.544 |
| 2 | high_middle_alpha | high_retrieval | Always lossless | 1 | 1 | lossless | lossless | 0 | 0 | 1,891 | 0 | 1,891 | 5 | 74,322 | 57,728 | 729 | 31.144 |
| 3 | high_middle_alpha | high_retrieval | Retrieval-aware | 1 | 1 | lossless | lossless | 0 | 0 | 1,891 | 0 | 1,891 | 5 | 74,200 | 63,872 | 704 | 27.634 |
| 4 | high_middle_beta | high_retrieval | Always lossless | 1 | 1 | lossless | lossless | 0 | 0 | 1,890 | 0 | 1,890 | 5 | 74,047 | 63,872 | 712 | 29.117 |
| 5 | high_middle_beta | high_retrieval | Retrieval-aware | 1 | 1 | lossless | lossless | 0 | 0 | 1,890 | 0 | 1,890 | 6 | 91,808 | 80,640 | 765 | 31.319 |
| 6 | high_middle_beta | high_retrieval | Historical behavior (observe) | 1 | 1 | lossy | lossless | 1 | 1 | 4,975 | 5,431 | -456 | 7 | 135,355 | 113,792 | 1,062 | 39.6 |
| 7 | low_search_alpha | low_retrieval | Retrieval-aware | 0 | 1 | lossy | lossy | 0 | 0 | 4,953 | 0 | 4,953 | 5 | 63,253 | 56,192 | 778 | 30.988 |
| 8 | low_search_alpha | low_retrieval | Historical behavior (observe) | 1 | 1 | lossy | lossy | 1 | 1 | 4,953 | 5,433 | -480 | 7 | 134,887 | 113,280 | 882 | 39.584 |
| 9 | low_search_alpha | low_retrieval | Always lossless | 1 | 1 | lossless | lossy | 0 | 0 | 1,890 | 0 | 1,890 | 5 | 74,024 | 55,168 | 709 | 32.936 |
| 10 | low_search_beta | low_retrieval | Historical behavior (observe) | 0 | 1 | lossy | lossy | 0 | 0 | 5,076 | 0 | 5,076 | 6 | 76,540 | 68,864 | 804 | 31.16 |
| 11 | low_search_beta | low_retrieval | Always lossless | 1 | 1 | lossless | lossy | 0 | 0 | 1,890 | 0 | 1,890 | 5 | 74,256 | 63,872 | 702 | 28.391 |
| 12 | low_search_beta | low_retrieval | Retrieval-aware | 0 | 1 | lossy | lossy | 0 | 0 | 5,076 | 0 | 5,076 | 5 | 62,900 | 56,192 | 826 | 32.409 |
| 13 | small_passthrough | passthrough | Always lossless | 1 | 1 | lossless | passthrough | 0 | 0 | 52 | 0 | 52 | 5 | 58,499 | 53,120 | 726 | 39.071 |
| 14 | small_passthrough | passthrough | Retrieval-aware | 1 | 0 | None | None | 0 | 0 | 0 | 0 | 0 | 5 | 58,925 | 53,632 | 746 | 32.438 |
| 15 | small_passthrough | passthrough | Historical behavior (observe) | 1 | 1 | lossless | passthrough | 0 | 0 | 52 | 0 | 52 | 5 | 58,562 | 53,120 | 744 | 30.464 |

## Aggregates

| regime | condition | passed | actions | actual recoveries | payload-net savings tokens | API requests | input tokens | cached input tokens | output tokens | wall s |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| high_retrieval | Historical behavior (observe) | 1/2 | {"lossy": 2} | 1 | 4,742 | 20 | 135,355 | 113,792 | 1,062 | 231.1 |
| high_retrieval | Always lossless | 2/2 | {"lossless": 2} | 0 | 3,781 | 10 | 148,369 | 121,600 | 1,441 | 60.3 |
| high_retrieval | Retrieval-aware | 2/2 | {"lossless": 2} | 0 | 3,781 | 11 | 166,008 | 144,512 | 1,469 | 59.0 |
| low_retrieval | Historical behavior (observe) | 1/2 | {"lossy": 2} | 1 | 4,596 | 13 | 211,427 | 182,144 | 1,686 | 70.7 |
| low_retrieval | Always lossless | 2/2 | {"lossless": 2} | 0 | 3,780 | 10 | 148,280 | 119,040 | 1,411 | 61.3 |
| low_retrieval | Retrieval-aware | 0/2 | {"lossy": 2} | 0 | 10,029 | 10 | 126,153 | 112,384 | 1,604 | 63.4 |
| passthrough | Historical behavior (observe) | 1/1 | {"lossless": 1} | 0 | 52 | 5 | 58,562 | 53,120 | 744 | 30.5 |
| passthrough | Always lossless | 1/1 | {"lossless": 1} | 0 | 52 | 5 | 58,499 | 53,120 | 726 | 39.1 |
| passthrough | Retrieval-aware | 1/1 | {"none": 1} | 0 | 0 | 5 | 58,925 | 53,632 | 746 | 32.4 |
| all | Historical behavior (observe) | 3/5 | {"lossless": 1, "lossy": 4} | 2 | 9,390 | 38 | 405,344 | 349,056 | 3,492 | 332.4 |
| all | Always lossless | 5/5 | {"lossless": 5} | 0 | 7,613 | 25 | 355,148 | 293,760 | 3,578 | 160.7 |
| all | Retrieval-aware | 3/5 | {"lossless": 2, "lossy": 2, "none": 1} | 0 | 13,810 | 26 | 351,086 | 310,528 | 3,819 | 154.8 |

Agent: `codex-cli 0.147.0`; model `gpt-5.4`; reasoning effort `low`; seed mode `fresh_random`.

High-retrieval tasks reveal an exact middle-record target only after catalog emission. Low-retrieval tasks infer schema from fields present in every retained sample row. The passthrough task is below the controller size threshold.
