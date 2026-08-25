# Validation note

Rows 7-15—the corrected schema and passthrough regimes—are valid and included in the validated matrix.

Rows 1-6 are a pre-reveal high-task pilot and are excluded as a complete task-design block, not selectively by outcome. Historical behavior (observe) rows 1 and 6 have `benchmark_valid: false` because lossy output did not exercise the required recovery path. Their Always lossless and Retrieval-aware companion rows are technically marked valid in `results.json`, contrary to an earlier version of this note, but all six are excluded symmetrically because they do not use the final delayed-target protocol. They are outside the 18 final-protocol attempts counted in `evaluation/retrieval_aware_agent_v2_validated/VALIDATION.md`.
