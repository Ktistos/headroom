from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


def _load_benchmark():
    path = Path(__file__).parents[1] / "benchmarks" / "retrieval_aware_agent_benchmark.py"
    spec = importlib.util.spec_from_file_location("retrieval_aware_agent_benchmark", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_one_shot_catalog_is_identical_and_not_replayable(tmp_path):
    benchmark = _load_benchmark()
    task = benchmark.TASKS[0]
    workspace = tmp_path / "workspace"
    expected, socket_path, payload = benchmark._write_task(workspace, task)
    listener, stop, thread = benchmark._start_one_shot_catalog(socket_path, payload)
    try:
        first = subprocess.run(
            [sys.executable, "data_tool.py", task.command_label],
            cwd=workspace,
            capture_output=True,
            text=True,
            timeout=5,
        )
        second = subprocess.run(
            [sys.executable, "data_tool.py", task.command_label],
            cwd=workspace,
            capture_output=True,
            text=True,
            timeout=5,
        )
    finally:
        stop.set()
        listener.close()
        thread.join(timeout=1)
        socket_path.unlink(missing_ok=True)

    assert first.returncode == 0
    rows = json.loads(first.stdout)
    assert rows == benchmark._rows(task.seed, task.row_count)
    assert rows[task.target_index] == expected
    assert second.returncode != 0


def test_high_target_is_revealed_only_after_catalog_output(tmp_path):
    benchmark = _load_benchmark()
    task = benchmark.TASKS[0]
    workspace = tmp_path / "workspace"
    expected, socket_path, payload = benchmark._write_task(workspace, task)
    listener, stop, thread = benchmark._start_task_source(
        socket_path, payload, f"{task.target_index}\n".encode()
    )
    try:
        early = subprocess.run(
            [sys.executable, "target_tool.py"],
            cwd=workspace,
            capture_output=True,
            text=True,
            timeout=5,
        )
        data = subprocess.run(
            [sys.executable, "data_tool.py", task.command_label],
            cwd=workspace,
            capture_output=True,
            text=True,
            timeout=5,
        )
        target = subprocess.run(
            [sys.executable, "target_tool.py"],
            cwd=workspace,
            capture_output=True,
            text=True,
            timeout=5,
        )
    finally:
        stop.set()
        listener.close()
        thread.join(timeout=1)
        socket_path.unlink(missing_ok=True)

    assert "ERROR" in early.stdout
    assert json.loads(data.stdout)[task.target_index] == expected
    assert target.stdout == f"{task.target_index}\n"
    assert str(task.target_index) not in (workspace / "TASK.md").read_text()
    assert str(task.target_index) not in (workspace / "target_tool.py").read_text()


def test_low_schema_answer_is_present_in_every_possible_sample_row(tmp_path):
    benchmark = _load_benchmark()
    task = benchmark.TASKS[2]
    expected, _socket_path, payload = benchmark._write_task(tmp_path, task)
    rows = json.loads(payload)

    assert expected == {"fields": ["index", "bucket", "status", "value", "token", "note"]}
    assert all(list(row) == expected["fields"] for row in rows)
    assert "do not recover omitted individual records" in (tmp_path / "TASK.md").read_text()
    assert '{"fields"}' in (tmp_path / "test_solution.py").read_text()


def test_task_can_use_short_socket_outside_long_workspace(tmp_path):
    benchmark = _load_benchmark()
    workspace = tmp_path / ("long-workspace-name-" * 8)
    socket_path = tmp_path / "catalog.sock"

    _expected, returned_path, _payload = benchmark._write_task(
        workspace, benchmark.TASKS[0], socket_path=socket_path
    )

    assert returned_path == socket_path
    assert repr(str(socket_path)) in (workspace / "data_tool.py").read_text()


def test_trace_parser_rejects_direct_store_access(tmp_path):
    benchmark = _load_benchmark()
    trace = tmp_path / "trace.jsonl"
    items = [
        {
            "type": "item.completed",
            "item": {
                "type": "command_execution",
                "command": "python data_tool.py exact",
            },
        },
        {
            "type": "item.completed",
            "item": {
                "type": "command_execution",
                "command": "sqlite3 /tmp/store 'select * from ccr_entries'",
            },
        },
        {
            "type": "item.completed",
            "item": {
                "type": "mcp_tool_call",
                "tool": "headroom_retrieve",
            },
        },
    ]
    trace.write_text("\n".join(json.dumps(item) for item in items) + "\n")

    parsed = benchmark._parse_codex_events(trace, "exact")
    assert parsed["catalog_command_events"] == 1
    assert parsed["retrieval_calls"] == 1
    assert parsed["direct_store_access"] is True
    assert parsed["external_artifact_search"] is True


def test_catalog_command_count_requires_an_exact_invocation(tmp_path):
    benchmark = _load_benchmark()
    trace = tmp_path / "trace.jsonl"
    commands = (
        '/bin/bash -lc "python data_tool.py exact"',
        'rg -n "python data_tool.py exact" /tmp/old-traces',
        "python data_tool.py exact | sed -n 1p",
    )
    trace.write_text(
        "\n".join(
            json.dumps(
                {
                    "type": "item.completed",
                    "item": {"type": "command_execution", "command": command},
                }
            )
            for command in commands
        )
        + "\n"
    )

    parsed = benchmark._parse_codex_events(trace, "exact")
    assert parsed["catalog_command_events"] == 1
    assert parsed["target_command_events"] == 0
    assert benchmark._is_target_command('/bin/bash -lc "python target_tool.py"')
    assert benchmark.ENV_MODE["historical_lossy"] == "observe"


def test_trace_parser_requires_target_after_catalog(tmp_path):
    benchmark = _load_benchmark()
    trace = tmp_path / "trace.jsonl"
    commands = ("python target_tool.py", "python data_tool.py exact")
    trace.write_text(
        "\n".join(
            json.dumps(
                {
                    "type": "item.completed",
                    "item": {"type": "command_execution", "command": command},
                }
            )
            for command in commands
        )
        + "\n"
    )

    parsed = benchmark._parse_codex_events(trace, "exact")
    assert parsed["catalog_command_events"] == 1
    assert parsed["target_command_events"] == 1
    assert parsed["target_after_catalog"] is False


def test_malformed_solution_is_a_clean_grader_failure(tmp_path):
    benchmark = _load_benchmark()
    solution = tmp_path / "solution.py"
    solution.write_text("EXPECTED_RECORD = dict(index=1)\n")

    assert benchmark._literal_from_solution(solution) is None


def test_trace_parser_rejects_tool_source_inspection(tmp_path):
    benchmark = _load_benchmark()
    trace = tmp_path / "trace.jsonl"
    item = {
        "type": "item.completed",
        "item": {
            "type": "command_execution",
            "command": "sed -n 1,200p target_tool.py",
        },
    }
    trace.write_text(json.dumps(item) + "\n")

    parsed = benchmark._parse_codex_events(trace, "exact")
    assert parsed["source_inspection"] is True


def test_trace_parser_rejects_external_artifact_search(tmp_path):
    benchmark = _load_benchmark()
    trace = tmp_path / "trace.jsonl"
    item = {
        "type": "item.completed",
        "item": {
            "type": "command_execution",
            "command": "rg -n ccr /tmp/retrieval-aware-old evaluation/",
        },
    }
    trace.write_text(json.dumps(item) + "\n")

    parsed = benchmark._parse_codex_events(trace, "exact")
    assert parsed["external_artifact_search"] is True


@pytest.mark.parametrize(
    "command",
    (
        "cat /tmp/hidden-generator.json",
        "sed -n 1,20p /home/user/private-results.json",
        "python -c \"open('/tmp/ccr_store.db', 'rb').read()\"",
        "ls evaluation/retrieval_aware_agent_v2",
    ),
)
def test_trace_parser_rejects_non_search_external_artifact_access(tmp_path, command):
    benchmark = _load_benchmark()
    trace = tmp_path / "trace.jsonl"
    trace.write_text(
        json.dumps(
            {
                "type": "item.completed",
                "item": {"type": "command_execution", "command": command},
            }
        )
        + "\n"
    )

    parsed = benchmark._parse_codex_events(trace, "exact")
    assert parsed["external_artifact_search"] is True


def test_materialized_tasks_use_fresh_unique_seeds(monkeypatch):
    benchmark = _load_benchmark()
    values = iter((11, 11, 12))
    monkeypatch.setattr(benchmark.secrets, "randbits", lambda _bits: next(values))

    generated = benchmark._materialize_tasks(benchmark.TASKS[:2], fixed_seeds=False)

    assert [task.seed for task in generated] == [11, 12]
    assert generated[0].name == benchmark.TASKS[0].name
    assert (
        benchmark._materialize_tasks(benchmark.TASKS[:2], fixed_seeds=True) == benchmark.TASKS[:2]
    )


def test_run_artifacts_are_buffered_until_explicit_persist(tmp_path):
    benchmark = _load_benchmark()
    run_dir = tmp_path / "staging" / "01_example"
    (run_dir / "workspace").mkdir(parents=True)
    (run_dir / "workspace" / "solution.py").write_text("answer = 1\n")
    (run_dir / "result.json").write_text("{}\n")

    artifacts = benchmark._capture_run_artifacts(run_dir)
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    assert list(output_dir.iterdir()) == []

    benchmark._persist_run_artifacts(output_dir, artifacts)

    persisted = output_dir / "runs" / "01_example"
    assert (persisted / "workspace" / "solution.py").read_text() == "answer = 1\n"
    assert (persisted / "result.json").read_text() == "{}\n"


def test_prepare_output_dir_rejects_existing_artifacts(tmp_path):
    benchmark = _load_benchmark()
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    (output_dir / "old-result.json").write_text("{}")

    with pytest.raises(FileExistsError, match="not empty"):
        benchmark._prepare_output_dir(output_dir)


def test_codex_runner_stops_at_api_request_budget(tmp_path, monkeypatch):
    benchmark = _load_benchmark()
    monkeypatch.setattr(
        benchmark,
        "_get_json",
        lambda *_args, **_kwargs: {"latency": {"total_requests": 13}},
    )

    returncode, reason = benchmark._run_codex(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        cwd=tmp_path,
        env=dict(os.environ),
        output_path=tmp_path / "trace.jsonl",
        base_url="http://127.0.0.1:1",
        timeout=10,
        max_api_requests=12,
    )

    assert returncode != 0
    assert reason == "api_request_budget_exceeded:13"


def test_primary_event_prefers_catalog_command_over_larger_mcp_metadata():
    benchmark = _load_benchmark()
    ledger = {
        "recent": [
            {
                "tool_name": "exec_command",
                "original_tokens": 6000,
                "action": "lossless",
            },
            {
                "tool_name": "list_mcp_resources",
                "original_tokens": 7000,
                "action": "lossy",
            },
        ]
    }
    assert benchmark._primary_event(ledger)["action"] == "lossless"


def test_summary_uses_human_policy_labels_without_renaming_machine_condition():
    benchmark = _load_benchmark()
    payload = {
        "runs": [
            {
                "run_index": 1,
                "task": {"name": "example"},
                "regime": "passthrough",
                "condition": "historical_lossy",
                "hidden_grader_pass": True,
                "benchmark_valid": True,
                "selected_action": "lossless",
                "predicted_action": "passthrough",
                "wall_time": 1.0,
            }
        ],
        "conditions": ["historical_lossy"],
        "aggregates": {
            "passthrough:historical_lossy": {
                "passed": 1,
                "runs": 1,
                "selected_actions": {"lossless": 1},
                "actual_recovery_events": 0,
                "payload_net_savings_tokens": 1,
                "api_requests": 1,
                "trajectory_input_tokens": 1,
                "trajectory_cached_input_tokens": 0,
                "trajectory_output_tokens": 1,
                "wall_time": 1.0,
            }
        },
        "codex_version": "test",
        "model": "test",
        "reasoning_effort": "test",
    }

    summary = benchmark._summary_markdown(payload)

    assert "Historical behavior (observe)" in summary
    assert "historical_lossy" not in summary
    assert payload["runs"][0]["condition"] == "historical_lossy"
