# Validated matrix manifest

Machine-readable condition identifiers remain unchanged in `results.json` and source paths. Human display labels are Historical behavior (observe), Always lossless, and Retrieval-aware.

The executable composition manifest is `manifest.json`. Rebuild and verify the primary `results.json` and `SUMMARY.md` entirely offline with:

```bash
python benchmarks/rebuild_retrieval_aware_validated.py --check
```

The command verifies every selected source row against its retained per-run `result.json`, excludes declared invalid and superseded rows, parses both sanitized MCP traces with the production byte estimator, reapplies only the declared derived corrections, and recomputes all aggregates. It does not contact Codex or any external model.

## Trace-derived accounting correction

Raw source rows retain the run-time ledger values. That ledger measured the compact logical retrieval dictionary before MCP serialized it into a `TextContent` envelope. The primary validated `results.json` corrects included runs 1 and 4 from sanitized, commit-eligible copies of their retained MCP results using the implementation's `ceil(UTF-8 bytes / 4)` estimator: 5,434 recorded tokens become 7,269 and 7,270 envelope tokens. Recorded values remain alongside corrected values, and the top-level `accounting_correction` block identifies both traces (`evaluation/retrieval_aware_agent_v2_high_reveal/RECOVERY_RUN_01_CODEX_TRACE.txt` and `evaluation/retrieval_aware_agent_v2_high_reveal_retry/RECOVERY_RUN_01_CODEX_TRACE.txt`). No agent, proxy, input, or live benchmark run was repeated.

## Seeds and condition order

| task | seed | source order |
|---|---:|---|
| high_middle_alpha | 7292468702409724928 | Historical behavior (observe), Always lossless, Retrieval-aware |
| high_middle_beta | 2927455522424617784 | Historical behavior (observe), Always lossless, Retrieval-aware |
| low_search_alpha | 1053749515337374634 | Retrieval-aware, Historical behavior (observe), Always lossless |
| low_search_beta | 6933354447170879506 | Historical behavior (observe), Always lossless, Retrieval-aware |
| small_passthrough | 4925836408471854222 | Always lossless, Retrieval-aware, Historical behavior (observe) |

## Source rows

- merged run 1: `evaluation/retrieval_aware_agent_v2_high_reveal/results.json` source run 1 (high_middle_alpha / Historical behavior (observe))
- merged run 2: `evaluation/retrieval_aware_agent_v2_high_reveal/results.json` source run 2 (high_middle_alpha / Always lossless)
- merged run 3: `evaluation/retrieval_aware_agent_v2_high_reveal/results.json` source run 3 (high_middle_alpha / Retrieval-aware)
- merged run 4: `evaluation/retrieval_aware_agent_v2_high_reveal_retry/results.json` source run 1 (high_middle_beta / Historical behavior (observe))
- merged run 5: `evaluation/retrieval_aware_agent_v2_high_reveal_retry/results.json` source run 2 (high_middle_beta / Always lossless)
- merged run 6: `evaluation/retrieval_aware_agent_v2_high_reveal_retry/results.json` source run 3 (high_middle_beta / Retrieval-aware)
- merged run 7: `evaluation/retrieval_aware_agent_v2/results.json` source run 7 (low_search_alpha / Retrieval-aware)
- merged run 8: `evaluation/retrieval_aware_agent_v2/results.json` source run 8 (low_search_alpha / Historical behavior (observe))
- merged run 9: `evaluation/retrieval_aware_agent_v2/results.json` source run 9 (low_search_alpha / Always lossless)
- merged run 10: `evaluation/retrieval_aware_agent_v2/results.json` source run 10 (low_search_beta / Historical behavior (observe))
- merged run 11: `evaluation/retrieval_aware_agent_v2/results.json` source run 11 (low_search_beta / Always lossless)
- merged run 12: `evaluation/retrieval_aware_agent_v2/results.json` source run 12 (low_search_beta / Retrieval-aware)
- merged run 13: `evaluation/retrieval_aware_agent_v2/results.json` source run 13 (small_passthrough / Always lossless)
- merged run 14: `evaluation/retrieval_aware_agent_v2/results.json` source run 14 (small_passthrough / Retrieval-aware)
- merged run 15: `evaluation/retrieval_aware_agent_v2/results.json` source run 15 (small_passthrough / Historical behavior (observe))
