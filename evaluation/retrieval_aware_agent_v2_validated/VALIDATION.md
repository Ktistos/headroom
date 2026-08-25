# Final protocol validation

## Scope and counts

The final delayed-reveal/schema/passthrough protocol produced **18 attempted runs, 17 valid runs, one invalid contaminated run, and 15 included runs**. The included matrix contains three policies for each of five tasks. Two valid original beta companion rows were superseded—not invalidated—when all three beta policies were rerun symmetrically. Earlier smoke, pre-reveal pilot, and invalid-task-design artifacts are retained as development evidence but are outside these final-protocol counts.

## Predeclared validity rule

`TASK.md` and `benchmarks/retrieval_aware_agent_benchmark.py` require exactly one catalog command; for high-retrieval tasks, exactly one later target command; no direct database access, external artifact search, or inspection of `data_tool.py`/`target_tool.py`; a primary compression event; a zero agent exit; and, when lossy output requires recovery, an MCP retrieval plus an attributed actual recovery. Hidden-grader correctness is separate from protocol validity. A wrong answer, failure to call MCP, ordinary extra model requests, or a reasoning error remains a behavioral failure unless an isolation rule is also violated.

## Contaminated beta attempt

The excluded attempt is source run 6 in `evaluation/retrieval_aware_agent_v2_high_reveal/results.json`, Historical behavior (observe), seed `3181966428966728432`. Its retained trace records:

- one permitted catalog command and one later target command;
- no MCP retrieval;
- prohibited reads of both `data_tool.py` and `target_tool.py`;
- prohibited searches under the hidden temporary catalog directory;
- a malformed/non-literal answer that failed the hidden grader.

The absent MCP retrieval and wrong answer are behavioral failures and would not alone justify exclusion. The source inspection and external hidden-source search contaminated isolation, so the run could no longer measure behavior through the experimental interface. The concise note and sanitized, commit-eligible trace are retained at the paths identified in `evaluation/retrieval_aware_agent_v2_high_reveal/INVALID_RUN_06_TRACE.md`.

## Symmetric replacement

The replacement beta trio used fresh seed `2927455522424617784`. Historical behavior (observe), Always lossless, and Retrieval-aware were all rerun against identical generated input and prompt, in that order. All three passed the hidden grader and every validity check. The two valid original-seed beta companion rows were excluded from the composed matrix so policies were not mixed across seeds. Because a noncompliant trajectory was replaced, the protocol-valid view may overstate real-world agent compliance even though exclusion followed isolation rules rather than the pass/fail outcome.

Every included row maps through `MANIFEST.md` to a source JSON row and per-run `result.json`. No invalid row contributes to `results.json` aggregates.
