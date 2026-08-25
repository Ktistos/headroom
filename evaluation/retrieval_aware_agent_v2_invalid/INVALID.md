# Invalid interrupted matrix

These runs are excluded from analysis and aggregates. The first benchmark version exposed the per-run SQLite path to the evaluated agent, and Historical behavior (observe) agents recovered missing content by querying SQLite directly rather than using the MCP retrieval contract. That bypass produced no attributed recovery event. The matrix was interrupted after the flaw was confirmed.

The benchmark now uses a one-shot data source, fresh per-invocation task seeds shared only across paired conditions, a randomly named temporary SQLite store visible only to the proxy, MCP proxy fallback, delayed artifact persistence, and explicit checks for direct-store access and external artifact searches. A rerun of an intermediate corrected matrix was attempted, but Codex CLI returned its usage-limit error before the agent turn. Raw invalid run artifacts were removed because they could leak deterministic answers into later agent searches; this audit note is retained.
