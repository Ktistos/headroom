#!/usr/bin/env python3
"""Rebuild the validated retrieval-aware artifact from retained evidence only."""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_production_estimator():
    """Load the production estimator without requiring an installed package."""
    policy_path = REPO_ROOT / "headroom/transforms/retrieval_aware_policy.py"
    spec = importlib.util.spec_from_file_location("_headroom_retrieval_aware_policy", policy_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load production estimator from {policy_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.estimate_payload_tokens


estimate_payload_tokens = _load_production_estimator()


DEFAULT_MANIFEST = REPO_ROOT / "evaluation/retrieval_aware_agent_v2_validated/manifest.json"
DEFAULT_OUTPUT = REPO_ROOT / "evaluation/retrieval_aware_agent_v2_validated"
CONDITION_DISPLAY = {
    "historical_lossy": "Historical behavior (observe)",
    "always_lossless": "Always lossless",
    "retrieval_aware": "Retrieval-aware",
}
AGGREGATE_FIELDS = (
    "retrieval_calls",
    "actual_recovery_events",
    "gross_savings_tokens",
    "recovery_payload_tokens",
    "payload_net_savings_tokens",
    "api_requests",
    "trajectory_input_tokens",
    "trajectory_cached_input_tokens",
    "trajectory_output_tokens",
    "wall_time",
)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _source_row(source_path: Path, source_run_index: int) -> dict[str, Any]:
    source = _load_json(source_path)
    matches = [
        row
        for row in source.get("runs", [])
        if isinstance(row, dict) and row.get("run_index") == source_run_index
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected one source run {source_run_index} in {source_path}, got {len(matches)}"
        )
    row = matches[0]
    task = row.get("task") or {}
    result_path = (
        source_path.parent
        / "runs"
        / f"{source_run_index:02d}_{task.get('name')}_{row.get('condition')}"
        / "result.json"
    )
    if not result_path.is_file():
        raise FileNotFoundError(f"missing per-run source artifact: {result_path}")
    if _load_json(result_path) != row:
        raise ValueError(f"source aggregate row differs from {result_path}")
    return row


def _trace_mcp_payload(trace_path: Path) -> dict[str, Any]:
    matches: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        trace_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        item = event.get("item", {}) if isinstance(event, dict) else {}
        if not isinstance(item, dict):
            continue
        if (
            item.get("type") == "mcp_tool_call"
            and item.get("status") == "completed"
            and "headroom_retrieve" in str(item.get("tool") or "")
            and isinstance(item.get("result"), dict)
        ):
            result = item["result"]
            content = result.get("content")
            if not isinstance(content, list) or not content:
                raise ValueError(f"completed recovery has no content at {trace_path}:{line_number}")
            # structured_content is trace metadata. Only content is the MCP
            # result envelope injected into the model.
            matches.append({"content": content})
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one completed MCP recovery in {trace_path}, got {len(matches)}"
        )
    return matches[0]


def _apply_accounting_correction(
    row: dict[str, Any], correction: dict[str, Any], trace_path: Path
) -> dict[str, Any]:
    recorded = int(correction["recorded_logical_payload_tokens"])
    if int(row.get("recovery_payload_tokens", -1)) != recorded:
        raise ValueError(
            f"run {row.get('run_index')} recorded recovery payload changed: "
            f"{row.get('recovery_payload_tokens')} != {recorded}"
        )
    model_payload = _trace_mcp_payload(trace_path)
    corrected = estimate_payload_tokens(model_payload)
    expected = int(correction["expected_model_visible_mcp_tokens"])
    if corrected != expected:
        raise ValueError(
            f"production estimator returned {corrected} for {trace_path}, expected {expected}"
        )

    result = copy.deepcopy(row)
    result["accounting_source"] = "retained_mcp_trace_envelope"
    result["recorded_recovery_payload_tokens"] = recorded
    result["recorded_payload_net_savings_tokens"] = int(result["payload_net_savings_tokens"])
    result["recovery_payload_tokens"] = corrected
    result["payload_net_savings_tokens"] = int(result["gross_savings_tokens"]) - corrected

    ledger = result.get("ledger")
    if not isinstance(ledger, dict):
        raise ValueError(f"corrected run {result.get('run_index')} has no ledger")
    ledger_recorded = int(ledger.get("recovery_payload_tokens", -1))
    if ledger_recorded != recorded:
        raise ValueError(
            f"run {result.get('run_index')} ledger recovery changed: "
            f"{ledger_recorded} != {recorded}"
        )
    ledger["recorded_recovery_payload_tokens"] = ledger_recorded
    ledger["recorded_payload_net_savings_tokens"] = int(ledger["payload_net_savings_tokens"])
    ledger["recovery_payload_tokens"] = corrected
    ledger["payload_net_savings_tokens"] = int(ledger["gross_savings_tokens"]) - corrected

    recovered_events = [
        event
        for event in ledger.get("recent", [])
        if isinstance(event, dict) and int(event.get("retrieval_count", 0)) > 0
    ]
    if len(recovered_events) != 1:
        raise ValueError(f"corrected run {result.get('run_index')} must have one recovered event")
    event = recovered_events[0]
    if int(event.get("recovery_payload_tokens", -1)) != recorded:
        raise ValueError("recovered ledger event does not match recorded correction")
    event["recorded_recovery_payload_tokens"] = recorded
    event["recorded_payload_net_savings_tokens"] = int(event["payload_net_savings_tokens"])
    event["recovery_payload_tokens"] = corrected
    event["payload_net_savings_tokens"] = int(event["gross_savings_tokens"]) - corrected
    return result


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "runs": len(rows),
        "passed": sum(bool(row.get("success")) for row in rows),
        "selected_actions": {},
    }
    for field in AGGREGATE_FIELDS:
        result[field] = round(sum(float(row.get(field, 0)) for row in rows), 3)
    for row in rows:
        action = str(row.get("selected_action") or "none")
        actions = result["selected_actions"]
        actions[action] = actions.get(action, 0) + 1
    return result


def _aggregates(rows: list[dict[str, Any]], conditions: list[str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for regime in ("high_retrieval", "low_retrieval", "passthrough"):
        for condition in conditions:
            selected = [
                row
                for row in rows
                if row.get("regime") == regime and row.get("condition") == condition
            ]
            if selected:
                result[f"{regime}:{condition}"] = _aggregate(selected)
    for condition in conditions:
        result[f"all:{condition}"] = _aggregate(
            [row for row in rows if row.get("condition") == condition]
        )
    return result


def _validate_attempts(manifest: dict[str, Any], included: list[dict[str, Any]]) -> None:
    attempts: list[tuple[str, int, dict[str, Any]]] = []
    for source in manifest["attempts"]:
        path = REPO_ROOT / source["source_artifact"]
        for source_run_index in source["source_run_indexes"]:
            attempts.append(
                (
                    source["source_artifact"],
                    source_run_index,
                    _source_row(path, source_run_index),
                )
            )
    valid = sum(bool(row.get("benchmark_valid")) for _, _, row in attempts)
    invalid = len(attempts) - valid
    expected = manifest["expected_counts"]
    actual = {
        "attempted": len(attempts),
        "valid": valid,
        "invalid": invalid,
        "included": len(included),
        "superseded_valid": len(manifest["excluded"]["superseded_valid"]),
    }
    if actual != expected:
        raise ValueError(f"attempt accounting mismatch: {actual} != {expected}")

    attempt_keys = [(path, int(index)) for path, index, _row in attempts]
    if len(set(attempt_keys)) != len(attempt_keys):
        raise ValueError("validated manifest contains duplicate attempt mappings")

    included_keys = {(row["source_artifact"], int(row["source_run_index"])) for row in included}
    invalid_keys = {
        (row["source_artifact"], int(row["source_run_index"]))
        for row in manifest["excluded"]["invalid"]
    }
    superseded_keys = {
        (row["source_artifact"], int(row["source_run_index"]))
        for row in manifest["excluded"]["superseded_valid"]
    }
    classifications = (included_keys, invalid_keys, superseded_keys)
    for index, left in enumerate(classifications):
        for right in classifications[index + 1 :]:
            overlap = left & right
            if overlap:
                raise ValueError(f"attempt classifications overlap: {sorted(overlap)}")
    classified_keys = included_keys | invalid_keys | superseded_keys
    attempted_key_set = set(attempt_keys)
    if classified_keys != attempted_key_set:
        missing = sorted(attempted_key_set - classified_keys)
        unexpected = sorted(classified_keys - attempted_key_set)
        raise ValueError(
            "attempt classifications do not exactly partition attempts: "
            f"unclassified={missing}, not_attempted={unexpected}"
        )

    for excluded in manifest["excluded"]["invalid"]:
        key = (excluded["source_artifact"], int(excluded["source_run_index"]))
        row = next(row for path, index, row in attempts if (path, index) == key)
        if row.get("benchmark_valid"):
            raise ValueError(f"declared invalid row is protocol-valid: {key}")
        if key in included_keys:
            raise ValueError(f"invalid row entered validated output: {key}")
        evidence = REPO_ROOT / excluded["evidence"]
        if not evidence.is_file() or not evidence.read_text(encoding="utf-8").strip():
            raise ValueError(f"missing invalidation evidence: {evidence}")
    for excluded in manifest["excluded"]["superseded_valid"]:
        key = (excluded["source_artifact"], int(excluded["source_run_index"]))
        row = next(row for path, index, row in attempts if (path, index) == key)
        if not row.get("benchmark_valid"):
            raise ValueError(f"superseded row is not valid: {key}")
        if key in included_keys:
            raise ValueError(f"superseded row entered validated output: {key}")


def _validate_source_matrix(rows: list[dict[str, Any]], conditions: list[str]) -> None:
    """Require one unique source row for every task/condition cell."""
    if not conditions or len(set(conditions)) != len(conditions):
        raise ValueError(f"conditions must be non-empty and unique: {conditions}")
    if len(rows) % len(conditions):
        raise ValueError(
            f"included row count {len(rows)} is not divisible by {len(conditions)} conditions"
        )

    expected_conditions = set(conditions)
    seen_pairs: set[tuple[str, str]] = set()
    by_task: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        task = row.get("task")
        if not isinstance(task, dict) or not isinstance(task.get("name"), str):
            raise ValueError(f"included row has invalid task metadata: {row.get('run_index')}")
        task_name = str(task["name"])
        condition = str(row.get("condition") or "")
        if condition not in expected_conditions:
            raise ValueError(f"unexpected condition for {task_name}: {condition}")
        pair = (task_name, condition)
        if pair in seen_pairs:
            raise ValueError(f"duplicate task/condition row: {task_name}/{condition}")
        seen_pairs.add(pair)
        by_task.setdefault(task_name, []).append(row)

    expected_task_count = len(rows) // len(conditions)
    if len(by_task) != expected_task_count:
        raise ValueError(
            f"expected {expected_task_count} tasks, got {len(by_task)}: {sorted(by_task)}"
        )
    for task_name, task_rows in sorted(by_task.items()):
        actual_conditions = {str(row.get("condition") or "") for row in task_rows}
        if actual_conditions != expected_conditions:
            raise ValueError(
                f"task {task_name} condition matrix mismatch: "
                f"{sorted(actual_conditions)} != {sorted(expected_conditions)}"
            )
        task_metadata = {
            json.dumps(row["task"], sort_keys=True, separators=(",", ":")) for row in task_rows
        }
        if len(task_metadata) != 1:
            raise ValueError(f"task metadata differs across conditions: {task_name}")
        regimes = {str(row.get("regime") or "") for row in task_rows}
        task_regime = str(task_rows[0]["task"].get("regime") or "")
        if len(regimes) != 1 or regimes != {task_regime}:
            raise ValueError(f"task regime differs across conditions: {task_name}")


def _summary_markdown(payload: dict[str, Any]) -> str:
    aggs = payload["aggregates"]
    historical = aggs["all:historical_lossy"]
    lossless = aggs["all:always_lossless"]
    aware = aggs["all:retrieval_aware"]
    high_historical = aggs["high_retrieval:historical_lossy"]
    high_aware = aggs["high_retrieval:retrieval_aware"]
    low_lossless = aggs["low_retrieval:always_lossless"]
    low_aware = aggs["low_retrieval:retrieval_aware"]
    correction_rows = payload["accounting_correction"]["corrected_runs"]
    recorded = correction_rows[0]["recorded_logical_payload_tokens"]
    corrected = [row["corrected_mcp_envelope_tokens"] for row in correction_rows]
    lines = [
        "# Retrieval-aware Codex benchmark",
        "",
        "This is the primary, validated v2 evaluation. Across five tasks under three policies, all 15 included protocol-valid runs passed. The final protocol had 18 attempts: 17 valid, one invalid contaminated, and 15 included; two valid companion rows were superseded when the complete beta trio was rerun symmetrically with a fresh seed. See `VALIDATION.md` for the predeclared rule and retained evidence. Accepted-run reporting may overstate real-world agent compliance.",
        "",
        f"The raw run-time ledger charged the compact logical retrieval dictionary, not the serialized MCP content envelope. The validated artifact therefore applies a deterministic correction from two sanitized, commit-eligible retained MCP traces: {recorded:,} recorded tokens become {corrected[0]:,} and {corrected[1]:,} envelope tokens. Raw source artifacts remain unchanged, and `results.json` preserves both recorded and corrected values. No live run was repeated.",
        "",
        "## Aggregate comparisons",
        "",
        f"- Retrieval-aware versus Historical behavior (observe): {aware['payload_net_savings_tokens']:,.0f} versus {historical['payload_net_savings_tokens']:,.0f} `payload_net_savings_tokens`, +{aware['payload_net_savings_tokens'] - historical['payload_net_savings_tokens']:,.0f} tokens ({aware['payload_net_savings_tokens'] / historical['payload_net_savings_tokens']:.2f}x); {aware['actual_recovery_events']:g} versus {historical['actual_recovery_events']:g} recoveries; {aware['api_requests']:g} versus {historical['api_requests']:g} API requests; and {historical['trajectory_input_tokens'] - aware['trajectory_input_tokens']:,.0f} fewer input tokens ({(historical['trajectory_input_tokens'] - aware['trajectory_input_tokens']) / historical['trajectory_input_tokens']:.1%}).",
        f"- Retrieval-aware versus Always lossless: {aware['payload_net_savings_tokens']:,.0f} versus {lossless['payload_net_savings_tokens']:,.0f} `payload_net_savings_tokens`, +{aware['payload_net_savings_tokens'] - lossless['payload_net_savings_tokens']:,.0f} tokens ({aware['payload_net_savings_tokens'] / lossless['payload_net_savings_tokens']:.2f}x); {aware['api_requests']:g} API requests each; and {lossless['trajectory_input_tokens'] - aware['trajectory_input_tokens']:,.0f} fewer input tokens ({(lossless['trajectory_input_tokens'] - aware['trajectory_input_tokens']) / lossless['trajectory_input_tokens']:.1%}).",
        f"- Uncached input (`input - cached input`) was {historical['trajectory_input_tokens'] - historical['trajectory_cached_input_tokens']:,.0f} for Historical behavior (observe), {lossless['trajectory_input_tokens'] - lossless['trajectory_cached_input_tokens']:,.0f} for Always lossless, and {aware['trajectory_input_tokens'] - aware['trajectory_cached_input_tokens']:,.0f} for Retrieval-aware. Cached and uncached input have different economic significance; no dollar-cost estimate is inferred.",
        f"- High retrieval: Retrieval-aware gained {high_aware['payload_net_savings_tokens'] - high_historical['payload_net_savings_tokens']:,.0f} payload-net tokens over Historical behavior (observe) and avoided two recoveries. Low retrieval: it gained {low_aware['payload_net_savings_tokens'] - low_lossless['payload_net_savings_tokens']:,.0f} tokens, or {low_aware['payload_net_savings_tokens'] / low_lossless['payload_net_savings_tokens']:.2f}x, over Always lossless. Passthrough: it selected passthrough as intended.",
        "",
        f"Retrieval-aware execution was observably faster than Historical behavior (observe) in this matrix, but {aware['wall_time'] - lossless['wall_time']:.1f} seconds slower than Always lossless. Because the experiment used one accepted run per condition and agent trajectories varied, no general latency improvement is claimed.",
        "",
        "## Individual runs",
        "",
        "| run | task | regime | condition | grader | valid | selected | predicted | retrieval calls | actual recoveries | gross savings tokens | recovery payload tokens | payload-net savings tokens | API requests | input tokens | cached input tokens | output tokens | wall s |",
        "|---:|---|---|---|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["runs"]:
        condition = CONDITION_DISPLAY.get(row["condition"], row["condition"])
        lines.append(
            f"| {row.get('run_index')} | {row['task']['name']} | {row['regime']} "
            f"| {condition} | {int(bool(row.get('hidden_grader_pass')))} "
            f"| {int(bool(row.get('benchmark_valid')))} | {row.get('selected_action')} "
            f"| {row.get('predicted_action')} | {row.get('retrieval_calls', 0)} "
            f"| {row.get('actual_recovery_events', 0)} | {row.get('gross_savings_tokens', 0):,} "
            f"| {row.get('recovery_payload_tokens', 0):,} "
            f"| {row.get('payload_net_savings_tokens', 0):,} | {row.get('api_requests', 0)} "
            f"| {row.get('trajectory_input_tokens', 0):,} "
            f"| {row.get('trajectory_cached_input_tokens', 0):,} "
            f"| {row.get('trajectory_output_tokens', 0):,} | {row.get('wall_time', 0)} |"
        )
    lines.extend(
        [
            "",
            "## Aggregates",
            "",
            "| regime | condition | passed | actions | actual recoveries | payload-net savings tokens | API requests | input tokens | cached input tokens | output tokens | wall s |",
            "|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for regime in ("high_retrieval", "low_retrieval", "passthrough", "all"):
        for condition in payload["conditions"]:
            key = f"{regime}:{condition}"
            if key not in aggs:
                continue
            aggregate = aggs[key]
            lines.append(
                f"| {regime} | {CONDITION_DISPLAY.get(condition, condition)} "
                f"| {aggregate['passed']}/{aggregate['runs']} "
                f"| {json.dumps(aggregate['selected_actions'], sort_keys=True)} "
                f"| {aggregate['actual_recovery_events']:g} "
                f"| {aggregate['payload_net_savings_tokens']:,.0f} "
                f"| {aggregate['api_requests']:g} "
                f"| {aggregate['trajectory_input_tokens']:,.0f} "
                f"| {aggregate['trajectory_cached_input_tokens']:,.0f} "
                f"| {aggregate['trajectory_output_tokens']:,.0f} "
                f"| {aggregate['wall_time']:.1f} |"
            )
    lines.extend(
        [
            "",
            f"Agent: `{payload['codex_version']}`; model `{payload['model']}`; "
            f"reasoning effort `{payload['reasoning_effort']}`; seed mode "
            f"`{payload.get('seed_mode', 'unknown')}`.",
            "",
            "High-retrieval tasks reveal an exact middle-record target only after "
            "catalog emission. Low-retrieval tasks infer schema from fields present "
            "in every retained sample row. The passthrough task is below the "
            "controller size threshold.",
        ]
    )
    return "\n".join(lines) + "\n"


def build_validated_artifact(manifest_path: Path, output_dir: Path) -> dict[str, Any]:
    manifest = _load_json(manifest_path)
    mappings = manifest.get("source_rows", [])
    if len(mappings) != 15:
        raise ValueError(f"validated manifest must map 15 rows, got {len(mappings)}")
    output_indexes = [int(item["output_run_index"]) for item in mappings]
    if output_indexes != list(range(1, 16)):
        raise ValueError(f"output run indexes are not exactly 1..15: {output_indexes}")
    source_keys = [
        (str(item["source_artifact"]), int(item["source_run_index"])) for item in mappings
    ]
    if len(set(source_keys)) != len(source_keys):
        raise ValueError("validated manifest contains duplicate source mappings")

    correction_items = list(manifest.get("accounting_corrections", []))
    correction_indexes = [int(item["output_run_index"]) for item in correction_items]
    if len(set(correction_indexes)) != len(correction_indexes):
        raise ValueError("validated manifest contains duplicate accounting corrections")
    if not set(correction_indexes).issubset(output_indexes):
        raise ValueError("accounting correction references an unknown output run")
    corrections = {int(item["output_run_index"]): item for item in correction_items}
    rows: list[dict[str, Any]] = []
    for mapping in mappings:
        source_artifact = str(mapping["source_artifact"])
        source_run_index = int(mapping["source_run_index"])
        output_run_index = int(mapping["output_run_index"])
        row = copy.deepcopy(_source_row(REPO_ROOT / source_artifact, source_run_index))
        if not row.get("benchmark_valid"):
            raise ValueError(f"included row is invalid: {source_artifact}#{source_run_index}")
        if row.get("actual") is None:
            raise ValueError(f"included row has no output: {source_artifact}#{source_run_index}")
        if output_run_index in corrections:
            correction = corrections[output_run_index]
            trace_path = REPO_ROOT / correction["source_trace"]
            row = _apply_accounting_correction(row, correction, trace_path)
        row["source_artifact"] = source_artifact
        row["source_run_index"] = source_run_index
        row["run_index"] = output_run_index
        rows.append(row)

    _validate_attempts(manifest, rows)
    output_meta = manifest["output"]
    conditions = list(output_meta["conditions"])
    _validate_source_matrix(rows, conditions)
    source_artifacts = list(dict.fromkeys(row["source_artifact"] for row in rows))
    correction_rows = []
    for output_run_index in sorted(corrections):
        correction = corrections[output_run_index]
        row = rows[output_run_index - 1]
        correction_rows.append(
            {
                "corrected_mcp_envelope_tokens": row["recovery_payload_tokens"],
                "corrected_payload_net_savings_tokens": row["payload_net_savings_tokens"],
                "recorded_logical_payload_tokens": correction["recorded_logical_payload_tokens"],
                "run_index": output_run_index,
                "source_trace": correction["source_trace"],
            }
        )
    payload = {
        "schema_version": output_meta["schema_version"],
        "codex_version": output_meta["codex_version"],
        "model": output_meta["model"],
        "reasoning_effort": output_meta["reasoning_effort"],
        "seed_mode": output_meta["seed_mode"],
        "conditions": conditions,
        "composition": {
            "kind": "validated_rows_only",
            "note": (
                "Raw measurements are preserved; only declared trace-derived "
                "model-visible recovery fields are corrected."
            ),
            "source_artifacts": source_artifacts,
        },
        "accounting_correction": {
            "applied": bool(correction_rows),
            "corrected_runs": correction_rows,
            "estimator": "ceil(UTF-8 bytes / 4), matching estimate_payload_tokens",
            "metric_boundary": (
                "Compact JSON serialization of the model-visible MCP content envelope "
                "containing the indented retrieval JSON text."
            ),
            "reason": (
                "The run-time ledger charged the compact logical retrieval dictionary "
                "before the MCP TextContent envelope was serialized."
            ),
            "source": "retained Codex MCP result traces; no live run was repeated",
        },
        "runs": rows,
        "aggregates": _aggregates(rows, conditions),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "SUMMARY.md").write_text(_summary_markdown(payload), encoding="utf-8")
    return payload


def _check(manifest_path: Path, checked_dir: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="retrieval-aware-rebuild-") as temp_dir:
        generated_dir = Path(temp_dir)
        generated = build_validated_artifact(manifest_path, generated_dir)
        checked = _load_json(checked_dir / "results.json")
        if generated != checked:
            raise ValueError("rebuilt results.json differs from checked-in validated artifact")
        generated_summary = (generated_dir / "SUMMARY.md").read_text(encoding="utf-8")
        checked_summary = (checked_dir / "SUMMARY.md").read_text(encoding="utf-8")
        if generated_summary != checked_summary:
            raise ValueError("rebuilt SUMMARY.md differs from checked-in validated artifact")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--check",
        action="store_true",
        help="rebuild in a temporary directory and compare with --output-dir",
    )
    args = parser.parse_args()
    manifest_path = args.manifest.resolve()
    output_dir = args.output_dir.resolve()
    if args.check:
        _check(manifest_path, output_dir)
        print("validated retrieval-aware artifact rebuild check passed")
    else:
        build_validated_artifact(manifest_path, output_dir)
        print(f"rebuilt validated retrieval-aware artifact in {output_dir}")


if __name__ == "__main__":
    main()
