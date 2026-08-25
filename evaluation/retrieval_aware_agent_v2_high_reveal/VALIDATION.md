# Validation note

Rows 1-3 form the valid alpha delayed-reveal trio and are included in the validated matrix. Rows 4-5 are valid original-seed beta companion runs, but they are superseded by the complete fresh-seed beta retry trio.

Row 6 is an invalid contaminated run under the predeclared isolation rule. The agent made no MCP call and produced a wrong/non-literal answer; those are behavioral failures, not reasons to exclude a row. It also inspected both prohibited tool sources and searched the hidden temporary source directory. Those actions bypassed the experimental interface and invalidated measurement. `INVALID_RUN_06_TRACE.md` records the exact evidence and identifies the untouched raw trace.

The original beta seed was `3181966428966728432`. The replacement used fresh seed `2927455522424617784` and reran Historical behavior (observe), Always lossless, and Retrieval-aware symmetrically. Final-protocol counts and the accepted-run caveat are in `evaluation/retrieval_aware_agent_v2_validated/VALIDATION.md`.
