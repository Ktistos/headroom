# Retrieval-aware Codex benchmark

| condition | tasks passed | input tokens | cached input | CCR retrievals | legacy net-proxy tokens |
|---|---:|---:|---:|---:|---:|
| observe | 3/3 | 378,557 | 330,624 | 3 | 1,985 |
| control | 3/3 | 283,513 | 244,480 | 0 | 4,861 |

This legacy v1 debit mixed recovered payload with fixed overhead. Its net-proxy values are not directly comparable with corrected `payload_net_savings_tokens` and are retained only as historical evidence.

Agent: Codex CLI `codex-cli 0.147.0`; model `gpt-5.4`; reasoning effort `low`.

`observe` executes the historical action; `control` executes the policy decision. Task workspaces, raw Codex JSONL, proxy logs, and per-run ledgers are under `runs/`.
