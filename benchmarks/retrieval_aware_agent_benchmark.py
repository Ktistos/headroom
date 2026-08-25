#!/usr/bin/env python3
"""Multi-regime Codex benchmark for retrieval-aware compression.

The matrix compares Historical behavior (observe), always-lossless behavior, and
retrieval-aware action selection. Every run gets a fresh agent process, proxy,
ledger, SQLite CCR store, and workspace. Paired conditions use identical task
input and prompts, while condition order rotates by task.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import random
import secrets
import shlex
import shutil
import signal
import socket
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TaskSpec:
    name: str
    regime: str
    seed: int
    target_index: int
    row_count: int
    command_label: str


TASKS = (
    TaskSpec("high_middle_alpha", "high_retrieval", 104729, 73, 180, "exact"),
    TaskSpec("high_middle_beta", "high_retrieval", 130363, 121, 180, "exact"),
    TaskSpec("low_search_alpha", "low_retrieval", 155921, 0, 180, "search"),
    TaskSpec("low_search_beta", "low_retrieval", 181081, 1, 180, "search"),
    TaskSpec("small_passthrough", "passthrough", 196613, 2, 5, "small"),
)
CONDITIONS = ("historical_lossy", "always_lossless", "retrieval_aware")
CONDITION_DISPLAY = {
    "historical_lossy": "Historical behavior (observe)",
    "always_lossless": "Always lossless",
    "retrieval_aware": "Retrieval-aware",
}
ENV_MODE = {
    "historical_lossy": "observe",
    "always_lossless": "always_lossless",
    "retrieval_aware": "control",
}


@dataclass(frozen=True)
class RunArtifacts:
    run_name: str
    files: dict[str, bytes]


def _materialize_tasks(
    templates: tuple[TaskSpec, ...], *, fixed_seeds: bool
) -> tuple[TaskSpec, ...]:
    if fixed_seeds:
        return templates
    used: set[int] = set()
    tasks: list[TaskSpec] = []
    for template in templates:
        seed = secrets.randbits(63)
        while seed in used:
            seed = secrets.randbits(63)
        used.add(seed)
        tasks.append(replace(template, seed=seed))
    return tuple(tasks)


def _rows(seed: int, count: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    statuses = ["ready"] * 8 + ["paused", "warming"]
    return [
        {
            "index": index,
            "bucket": f"b{rng.randrange(6)}",
            "status": rng.choice(statuses),
            "value": rng.randrange(11),
            "token": f"code-{rng.randrange(14):02d}",
            "note": f"catalog-note-{rng.randrange(1000):03d}",
        }
        for index in range(count)
    ]


def _write_task(
    root: Path, spec: TaskSpec, *, socket_path: Path | None = None
) -> tuple[dict[str, Any], Path, bytes]:
    root.mkdir(parents=True, exist_ok=True)
    rows = _rows(spec.seed, spec.row_count)
    if spec.regime == "low_retrieval":
        expected = {"fields": list(rows[0])}
        answer_instruction = (
            "Infer the ordered field names shared by the displayed catalog records. "
            "Set EXPECTED_RECORD in solution.py to a dictionary with the single key "
            "fields and the ordered list of field names. The lossy sample or summary "
            "is sufficient; do not recover omitted individual records. "
        )
        visible_shape = '            {"fields"},\n'
    elif spec.regime == "high_retrieval":
        expected = rows[spec.target_index]
        answer_instruction = (
            'Only after the catalog command finishes, run "python target_tool.py" '
            "exactly once as a plain command to reveal the required zero-based "
            "position. Do not inspect target_tool.py. Copy the complete record at "
            "that revealed position into EXPECTED_RECORD in solution.py, preserving "
            "every key and value. "
        )
        visible_shape = '            {"index", "bucket", "status", "value", "token", "note"},\n'
    else:
        expected = rows[spec.target_index]
        answer_instruction = (
            f"Copy the complete zero-based record at position {spec.target_index} "
            "into EXPECTED_RECORD in solution.py. Preserve every key and value. "
        )
        visible_shape = '            {"index", "bucket", "status", "value", "token", "note"},\n'
    command = f"python data_tool.py {spec.command_label}"
    (root / "TASK.md").write_text(
        "# Catalog task\n\n"
        f'Run "{command}" exactly once as a plain command. '
        "Do not inspect data_tool.py, and do not pipe or redirect its output. "
        "Treat the displayed output and its normal tool contract as authoritative. "
        "Do not search outside this workspace or inspect environment, runtime/client "
        "state, databases, logs, caches, histories, or prior artifacts. If required "
        "content is omitted behind a recovery marker, its configured recovery tool "
        "is the only permitted recovery path and does not rerun the data command.\n\n"
        + answer_instruction
        + "Then run python -m unittest -q. The visible test checks shape; a hidden "
        "grader checks the complete value.\n",
        encoding="utf-8",
    )
    (root / "solution.py").write_text(
        '"""Catalog answer populated from the runtime tool output."""\n\nEXPECTED_RECORD = None\n',
        encoding="utf-8",
    )
    (root / "test_solution.py").write_text(
        "import unittest\n\n"
        "from solution import EXPECTED_RECORD\n\n\n"
        "class ShapeTest(unittest.TestCase):\n"
        "    def test_record_shape(self):\n"
        "        self.assertIsInstance(EXPECTED_RECORD, dict)\n"
        "        self.assertEqual(\n"
        "            set(EXPECTED_RECORD),\n" + visible_shape + "        )\n\n\n"
        'if __name__ == "__main__":\n'
        "    unittest.main()\n",
        encoding="utf-8",
    )
    socket_path = socket_path or root / ".catalog.sock"
    client_program = (
        "import socket\n\n"
        "client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)\n"
        f"client.connect({str(socket_path)!r})\n"
        'client.sendall(REQUEST + b"\\n")\n'
        "chunks = []\n"
        "while True:\n"
        "    chunk = client.recv(65536)\n"
        "    if not chunk:\n"
        "        break\n"
        "    chunks.append(chunk)\n"
        "client.close()\n"
        'print(b"".join(chunks).decode("utf-8"), end="")\n'
    )
    (root / "data_tool.py").write_text(
        'REQUEST = b"data"\n' + client_program,
        encoding="utf-8",
    )
    if spec.regime == "high_retrieval":
        (root / "target_tool.py").write_text(
            'REQUEST = b"target"\n' + client_program,
            encoding="utf-8",
        )
    return expected, socket_path, json.dumps(rows, indent=2).encode("utf-8") + b"\n"


def _start_task_source(
    socket_path: Path,
    data_payload: bytes,
    target_payload: bytes | None = None,
) -> tuple[socket.socket, threading.Event, threading.Thread]:
    socket_path.unlink(missing_ok=True)
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(str(socket_path))
    listener.listen(2)
    listener.settimeout(0.2)
    stop = threading.Event()
    served_data = False
    served_target = target_payload is None

    def serve() -> None:
        nonlocal served_data, served_target
        try:
            while not stop.is_set() and not (served_data and served_target):
                try:
                    connection, _ = listener.accept()
                except TimeoutError:
                    continue
                except OSError:
                    return
                with connection:
                    request = connection.recv(32).strip()
                    if request == b"data" and not served_data:
                        connection.sendall(data_payload)
                        served_data = True
                    elif (
                        request == b"target"
                        and served_data
                        and not served_target
                        and target_payload is not None
                    ):
                        connection.sendall(target_payload)
                        served_target = True
                    else:
                        connection.sendall(b"ERROR: request unavailable or already consumed\n")
        finally:
            try:
                listener.close()
            finally:
                socket_path.unlink(missing_ok=True)

    thread = threading.Thread(target=serve, name="one-shot-catalog", daemon=True)
    thread.start()
    return listener, stop, thread


def _start_one_shot_catalog(
    socket_path: Path, payload: bytes
) -> tuple[socket.socket, threading.Event, threading.Thread]:
    return _start_task_source(socket_path, payload)


def _get_json(url: str, timeout: float = 5.0) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"Host": url.split("//", 1)[-1].split("/", 1)[0]})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _wait_ready(base_url: str, process: subprocess.Popen[Any], timeout: float = 30.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"proxy exited early with status {process.returncode}")
        try:
            if _get_json(f"{base_url}/readyz").get("status") in {"ready", "healthy"}:
                return
        except (OSError, urllib.error.URLError, json.JSONDecodeError):
            pass
        time.sleep(0.2)
    raise TimeoutError("Headroom proxy did not become ready")


def _literal_from_solution(path: Path) -> Any:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "EXPECTED_RECORD"
            for target in node.targets
        ):
            try:
                return ast.literal_eval(node.value)
            except (ValueError, TypeError, SyntaxError):
                return None
    return None


def _is_catalog_command(command: str, command_label: str) -> bool:
    try:
        parts = shlex.split(command)
        if len(parts) >= 3 and parts[0].endswith(("bash", "sh")) and parts[1] == "-lc":
            parts = shlex.split(parts[2])
    except ValueError:
        return False
    return parts == ["python", "data_tool.py", command_label]


def _is_target_command(command: str) -> bool:
    try:
        parts = shlex.split(command)
        if len(parts) >= 3 and parts[0].endswith(("bash", "sh")) and parts[1] == "-lc":
            parts = shlex.split(parts[2])
    except ValueError:
        return False
    return parts == ["python", "target_tool.py"]


def _searches_external_artifacts(command: str) -> bool:
    """Conservatively flag any command that names a prohibited external target.

    The compatibility field remains ``external_artifact_search``, but direct
    reads via cat/sed/Python and directory inspection are just as contaminating
    as rg/find searches. Valid benchmark commands operate in the isolated
    workspace and do not need to mention these paths.
    """
    normalized = command.lower()
    return any(
        fragment in normalized
        for fragment in (
            "/tmp",
            "/home/",
            "evaluation/",
            "retrieval_aware_agent",
        )
    )


def _parse_codex_events(path: Path, command_label: str) -> dict[str, Any]:
    events: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            events.append(event)
    usage: dict[str, Any] = {}
    for event in events:
        if event.get("type") == "turn.completed" and isinstance(event.get("usage"), dict):
            usage = event["usage"]
    items = [
        event["item"]
        for event in events
        if event.get("type") == "item.completed" and isinstance(event.get("item"), dict)
    ]
    commands = [
        str(item.get("command") or "") for item in items if item.get("type") == "command_execution"
    ]
    mcp_calls = [item for item in items if item.get("type") == "mcp_tool_call"]
    catalog_positions = [
        index
        for index, command in enumerate(commands)
        if _is_catalog_command(command, command_label)
    ]
    target_positions = [
        index for index, command in enumerate(commands) if _is_target_command(command)
    ]
    return {
        "usage": usage,
        "tool_events": len(commands) + len(mcp_calls),
        "catalog_command_events": len(catalog_positions),
        "target_command_events": len(target_positions),
        "target_after_catalog": (
            not target_positions
            or (bool(catalog_positions) and target_positions[0] > catalog_positions[0])
        ),
        "retrieval_calls": sum(
            "headroom_retrieve" in str(item.get("tool") or item.get("name") or "").lower()
            for item in mcp_calls
        ),
        "direct_store_access": any(
            "ccr_store" in command.lower()
            or "ccr_entries" in command.lower()
            or "sqlite3" in command.lower()
            for command in commands
        ),
        "external_artifact_search": any(
            _searches_external_artifacts(command) for command in commands
        ),
        "source_inspection": any(
            ("data_tool.py" in command or "target_tool.py" in command)
            and not _is_catalog_command(command, command_label)
            and not _is_target_command(command)
            for command in commands
        ),
    }


def _toml_string(value: str) -> str:
    return json.dumps(value)


def _primary_event(ledger: dict[str, Any]) -> dict[str, Any]:
    events = ledger.get("recent", [])
    if not isinstance(events, list) or not events:
        return {}
    candidates = [
        event
        for event in events
        if isinstance(event, dict) and int(event.get("original_tokens", 0)) > 0
    ]
    command_candidates = [
        event
        for event in candidates
        if str(event.get("tool_name") or "").lower()
        in {"exec_command", "execute", "shell", "bash", "terminal"}
    ]
    pool = command_candidates or candidates
    return max(pool, key=lambda event: int(event.get("original_tokens", 0)), default={})


def _capture_run_artifacts(run_dir: Path) -> RunArtifacts:
    files: dict[str, bytes] = {}
    for path in sorted(run_dir.rglob("*")):
        if path.is_symlink():
            raise RuntimeError(f"refusing to buffer symlinked artifact: {path}")
        if path.is_file():
            files[str(path.relative_to(run_dir))] = path.read_bytes()
    return RunArtifacts(run_name=run_dir.name, files=files)


def _persist_run_artifacts(output_dir: Path, artifacts: RunArtifacts) -> None:
    run_dir = output_dir / "runs" / artifacts.run_name
    if run_dir.exists():
        raise FileExistsError(f"run artifact directory already exists: {run_dir}")
    for relative, payload in artifacts.files.items():
        destination = run_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)


def _prepare_output_dir(output_dir: Path) -> None:
    if output_dir.exists():
        if not output_dir.is_dir():
            raise NotADirectoryError(output_dir)
        if next(output_dir.iterdir(), None) is not None:
            raise FileExistsError(f"output directory is not empty: {output_dir}")
    else:
        output_dir.mkdir(parents=True)


def _stop_process_group(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=5)


def _run_codex(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    output_path: Path,
    base_url: str,
    timeout: float,
    max_api_requests: int,
) -> tuple[int, str | None]:
    deadline = time.monotonic() + timeout
    termination_reason: str | None = None
    with output_path.open("w", encoding="utf-8") as output:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdout=output,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        while process.poll() is None:
            if time.monotonic() >= deadline:
                termination_reason = "timeout"
                break
            if max_api_requests > 0:
                try:
                    stats = _get_json(f"{base_url}/stats", timeout=2.0)
                    request_count = int(stats.get("latency", {}).get("total_requests", 0))
                except (
                    OSError,
                    urllib.error.URLError,
                    json.JSONDecodeError,
                    TypeError,
                    ValueError,
                ):
                    request_count = 0
                if request_count > max_api_requests:
                    termination_reason = f"api_request_budget_exceeded:{request_count}"
                    break
            time.sleep(0.5)
        if termination_reason is not None:
            _stop_process_group(process)
        returncode = process.wait()
    return returncode, termination_reason


def _run_one(
    args: argparse.Namespace, condition: str, spec: TaskSpec, run_index: int
) -> tuple[dict[str, Any], RunArtifacts]:
    run_dir = args.output_dir / "runs" / f"{run_index:02d}_{spec.name}_{condition}"
    task_dir = run_dir / "workspace"
    catalog_socket_tempdir = tempfile.TemporaryDirectory(prefix=".hr-cat-")
    catalog_socket_path = Path(catalog_socket_tempdir.name) / "catalog.sock"
    expected, catalog_socket_path, catalog_payload = _write_task(
        task_dir, spec, socket_path=catalog_socket_path
    )
    target_payload = f"{spec.target_index}\n".encode() if spec.regime == "high_retrieval" else None
    catalog_listener, catalog_stop, catalog_thread = _start_task_source(
        catalog_socket_path, catalog_payload, target_payload
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    proxy_log_path = run_dir / "proxy.log"
    proxy_log = proxy_log_path.open("w", encoding="utf-8")
    base_url = f"http://127.0.0.1:{args.port}"
    mode = ENV_MODE[condition]
    sqlite_tempdir = tempfile.TemporaryDirectory(prefix=".hr-ra-")
    sqlite_path = Path(sqlite_tempdir.name) / f"store-{random.getrandbits(96):024x}"
    agent_home_tempdir = tempfile.TemporaryDirectory(prefix=".hr-agent-")
    agent_home = Path(agent_home_tempdir.name)
    agent_codex_home = agent_home / ".codex"
    agent_codex_home.mkdir()
    auth_source = Path.home() / ".codex" / "auth.json"
    if auth_source.is_file():
        shutil.copy2(auth_source, agent_codex_home / "auth.json")
    env = os.environ.copy()
    # Proxy and its MCP child share a fresh process-local accounting secret.
    # It is passed only through process configuration and is never emitted to
    # benchmark artifacts or the evaluated agent's general environment.
    accounting_secret = secrets.token_urlsafe(32)
    env.update(
        {
            "HEADROOM_RETRIEVAL_AWARE": mode,
            "HEADROOM_RETRIEVAL_OBSERVATION_WINDOW_SECONDS": "0",
            "HEADROOM_CCR_SQLITE_PATH": str(sqlite_path),
            "HEADROOM_SMART_CRUSHER_COMPACTION": "0",
            "HEADROOM_DISABLE_KOMPRESS": "1",
            "HEADROOM_TELEMETRY": "off",
            "HEADROOM_RETRIEVAL_ACCOUNT_SECRET": accounting_secret,
        }
    )
    proxy_cmd = [
        args.headroom_cli,
        "proxy",
        "--port",
        str(args.port),
        "--mode",
        "token",
        "--stateless",
        "--no-subscription-tracking",
        "--disable-kompress",
    ]
    proxy = subprocess.Popen(
        proxy_cmd,
        cwd=args.repo_root,
        env=env,
        stdout=proxy_log,
        stderr=subprocess.STDOUT,
        text=True,
    )
    codex_output = run_dir / "codex.jsonl"
    started = time.monotonic()
    try:
        _wait_ready(base_url, proxy)
        provider = (
            '{name="Headroom",base_url="'
            f"{base_url}/v1"
            '",wire_api="responses",requires_openai_auth=true,'
            "supports_websockets=false}"
        )
        mcp_args = json.dumps(["mcp", "serve", "--proxy-url", base_url], separators=(",", ":"))
        codex_cmd = [
            args.codex,
            "exec",
            "--ephemeral",
            "--ignore-user-config",
            "--ignore-rules",
            "--skip-git-repo-check",
            "--json",
            "--sandbox",
            "danger-full-access",
            "-C",
            str(task_dir),
            "-m",
            args.model,
            "-c",
            'model_provider="headroom"',
            "-c",
            f"model_providers.headroom={provider}",
            "-c",
            f'model_reasoning_effort="{args.reasoning_effort}"',
            "-c",
            'approval_policy="never"',
            "-c",
            "agents.enabled=false",
            "-c",
            f"mcp_servers.headroom.command={_toml_string(args.headroom_cli)}",
            "-c",
            f"mcp_servers.headroom.args={mcp_args}",
            "-c",
            f"mcp_servers.headroom.env.PYTHONPATH={_toml_string(str(args.repo_root))}",
            (
                "mcp_servers.headroom.env.HEADROOM_RETRIEVAL_ACCOUNT_SECRET="
                f"{_toml_string(accounting_secret)}"
            ),
            (
                "Complete the task in TASK.md. Inspect TASK.md and solution.py, run "
                "the required data command once, and follow any tool contract exposed "
                "with its output. Stay inside the task workspace; do not inspect "
                "environment, runtime/client state, databases, logs, caches, histories, "
                "or prior artifacts. Populate the answer and stop after the local "
                "unittest passes."
            ),
        ]
        agent_env = env.copy()
        # Only the proxy receives store/accounting credentials. The MCP server
        # uses proxy fallback; the evaluated agent cannot open SQLite directly.
        agent_env.pop("HEADROOM_CCR_SQLITE_PATH", None)
        agent_env.pop("HEADROOM_RETRIEVAL_ACCOUNT_SECRET", None)
        agent_env["HOME"] = str(agent_home)
        agent_env["CODEX_HOME"] = str(agent_codex_home)
        codex_exit_code, termination_reason = _run_codex(
            codex_cmd,
            cwd=task_dir,
            env=agent_env,
            output_path=codex_output,
            base_url=base_url,
            timeout=args.timeout,
            max_api_requests=args.max_api_requests,
        )
        parsed = _parse_codex_events(codex_output, spec.command_label)
        actual = _literal_from_solution(task_dir / "solution.py")
        try:
            retrieval_stats = _get_json(f"{base_url}/v1/retrieve/stats")
            ledger = retrieval_stats["retrieval_aware"]["ledger"]
        except (KeyError, OSError, urllib.error.URLError, json.JSONDecodeError):
            ledger = {}
        try:
            proxy_stats = _get_json(f"{base_url}/stats")
            api_requests = int(proxy_stats.get("latency", {}).get("total_requests", 0))
        except (OSError, urllib.error.URLError, json.JSONDecodeError, TypeError, ValueError):
            api_requests = 0
        primary = _primary_event(ledger)
        usage = parsed.get("usage", {})
        hidden_grader_pass = actual == expected
        expected_recovery = spec.regime == "high_retrieval" and primary.get("action") == "lossy"
        valid_attribution_path = (
            not parsed["direct_store_access"]
            and not parsed["external_artifact_search"]
            and not parsed["source_inspection"]
            and (
                not expected_recovery
                or (
                    parsed["retrieval_calls"] > 0
                    and int(ledger.get("actual_recovery_events", 0)) > 0
                )
            )
        )
        expected_target_commands = 1 if spec.regime == "high_retrieval" else 0
        benchmark_valid = (
            codex_exit_code == 0
            and parsed["catalog_command_events"] == 1
            and parsed["target_command_events"] == expected_target_commands
            and parsed["target_after_catalog"]
            and bool(primary)
            and valid_attribution_path
        )
        result = {
            "run_index": run_index,
            "task": asdict(spec),
            "regime": spec.regime,
            "condition": condition,
            "hidden_grader_pass": hidden_grader_pass,
            "success": hidden_grader_pass and codex_exit_code == 0,
            "expected": expected,
            "actual": actual,
            "codex_exit_code": codex_exit_code,
            "termination_reason": termination_reason,
            "wall_time": round(time.monotonic() - started, 3),
            "selected_action": primary.get("action"),
            "predicted_action": primary.get("predicted_action"),
            "retrieval_calls": parsed["retrieval_calls"],
            "actual_recovery_events": int(ledger.get("actual_recovery_events", 0)),
            "gross_savings_tokens": int(ledger.get("gross_savings_tokens", 0)),
            "recovery_payload_tokens": int(ledger.get("recovery_payload_tokens", 0)),
            "payload_net_savings_tokens": int(ledger.get("payload_net_savings_tokens", 0)),
            "api_requests": api_requests,
            "trajectory_input_tokens": int(usage.get("input_tokens", 0)),
            "trajectory_cached_input_tokens": int(usage.get("cached_input_tokens", 0)),
            "trajectory_output_tokens": int(usage.get("output_tokens", 0)),
            "catalog_command_events": parsed["catalog_command_events"],
            "target_command_events": parsed["target_command_events"],
            "target_after_catalog": parsed["target_after_catalog"],
            "direct_store_access": parsed["direct_store_access"],
            "external_artifact_search": parsed["external_artifact_search"],
            "source_inspection": parsed["source_inspection"],
            "tool_events": parsed["tool_events"],
            "ledger": ledger,
            "policy_exercised": bool(primary),
            "valid_attribution_path": valid_attribution_path,
            "benchmark_valid": benchmark_valid,
        }
    except Exception as exc:
        result = {
            "run_index": run_index,
            "task": asdict(spec),
            "regime": spec.regime,
            "condition": condition,
            "hidden_grader_pass": False,
            "success": False,
            "benchmark_valid": False,
            "error": f"{type(exc).__name__}: {exc}",
            "wall_time": round(time.monotonic() - started, 3),
        }
    finally:
        proxy.terminate()
        try:
            proxy.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proxy.kill()
            proxy.wait(timeout=5)
        proxy_log.close()
        if sqlite_path.exists():
            shutil.copy2(sqlite_path, run_dir / "ccr_store.db")
        sqlite_tempdir.cleanup()
        agent_home_tempdir.cleanup()
        catalog_stop.set()
        catalog_listener.close()
        catalog_thread.join(timeout=1)
        catalog_socket_path.unlink(missing_ok=True)
        catalog_socket_tempdir.cleanup()
    (run_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    artifacts = _capture_run_artifacts(run_dir)
    shutil.rmtree(run_dir)
    return result, artifacts


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    fields = (
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
    result = {
        "runs": len(rows),
        "passed": sum(bool(row.get("success")) for row in rows),
        "selected_actions": {},
    }
    for field in fields:
        result[field] = round(sum(float(row.get(field, 0)) for row in rows), 3)
    for row in rows:
        action = str(row.get("selected_action") or "none")
        result["selected_actions"][action] = result["selected_actions"].get(action, 0) + 1
    return result


def _summary_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Retrieval-aware Codex benchmark",
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
            if key not in payload["aggregates"]:
                continue
            agg = payload["aggregates"][key]
            display_condition = CONDITION_DISPLAY.get(condition, condition)
            lines.append(
                f"| {regime} | {display_condition} | {agg['passed']}/{agg['runs']} "
                f"| {json.dumps(agg['selected_actions'], sort_keys=True)} "
                f"| {agg['actual_recovery_events']:g} "
                f"| {agg['payload_net_savings_tokens']:,.0f} | {agg['api_requests']:g} "
                f"| {agg['trajectory_input_tokens']:,.0f} "
                f"| {agg['trajectory_cached_input_tokens']:,.0f} "
                f"| {agg['trajectory_output_tokens']:,.0f} | {agg['wall_time']:.1f} |"
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("evaluation/retrieval_aware_agent_v2")
    )
    parser.add_argument("--headroom-cli", default="headroom")
    parser.add_argument("--codex", default="codex")
    parser.add_argument("--model", default="gpt-5.4")
    parser.add_argument("--reasoning-effort", default="low")
    parser.add_argument("--port", type=int, default=8891)
    parser.add_argument("--timeout", type=int, default=420)
    parser.add_argument(
        "--max-api-requests",
        type=int,
        default=12,
        help="Terminate a run after this many proxy model requests (0 disables the guard)",
    )
    parser.add_argument("--tasks", type=int, default=len(TASKS))
    parser.add_argument(
        "--task-names",
        default="",
        help="Comma-separated task-name subset, applied before --tasks",
    )
    parser.add_argument(
        "--fixed-seeds",
        action="store_true",
        help="Use checked-in deterministic seeds (unsafe when old artifacts are accessible)",
    )
    parser.add_argument(
        "--conditions",
        default=",".join(CONDITIONS),
        help="Comma-separated subset of historical_lossy,always_lossless,retrieval_aware",
    )
    args = parser.parse_args()
    args.repo_root = Path(__file__).resolve().parents[1]
    args.output_dir = args.output_dir.resolve()
    requested = tuple(item.strip() for item in args.conditions.split(",") if item.strip())
    invalid = [item for item in requested if item not in CONDITIONS]
    if invalid:
        parser.error(f"unknown conditions: {', '.join(invalid)}")
    args.conditions = requested
    task_names = tuple(item.strip() for item in args.task_names.split(",") if item.strip())
    invalid_tasks = [item for item in task_names if item not in {task.name for task in TASKS}]
    if invalid_tasks:
        parser.error("unknown tasks: " + ", ".join(invalid_tasks))
    args.task_names = task_names
    return args


def main() -> int:
    args = parse_args()
    _prepare_output_dir(args.output_dir)
    available_templates = tuple(
        task for task in TASKS if not args.task_names or task.name in args.task_names
    )
    templates = available_templates[: max(0, min(args.tasks, len(available_templates)))]
    selected = _materialize_tasks(templates, fixed_seeds=args.fixed_seeds)
    runs: list[dict[str, Any]] = []
    total = len(selected) * len(args.conditions)
    run_index = 0
    for task_index, spec in enumerate(selected):
        shift = task_index % max(len(args.conditions), 1)
        order = args.conditions[shift:] + args.conditions[:shift]
        task_artifacts: list[RunArtifacts] = []
        for condition in order:
            run_index += 1
            print(f"[{run_index}/{total}] {spec.name} / {condition}", flush=True)
            result, artifacts = _run_one(args, condition, spec, run_index)
            runs.append(result)
            task_artifacts.append(artifacts)
        for artifacts in task_artifacts:
            _persist_run_artifacts(args.output_dir, artifacts)
    aggregates: dict[str, Any] = {}
    for regime in {spec.regime for spec in selected}:
        for condition in args.conditions:
            rows = [
                row for row in runs if row["regime"] == regime and row["condition"] == condition
            ]
            if rows:
                aggregates[f"{regime}:{condition}"] = _aggregate(rows)
    for condition in args.conditions:
        aggregates[f"all:{condition}"] = _aggregate(
            [row for row in runs if row["condition"] == condition]
        )
    try:
        codex_version = subprocess.check_output(
            [args.codex, "--version"], text=True, stderr=subprocess.STDOUT
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        codex_version = "unknown"
    payload = {
        "schema_version": 2,
        "model": args.model,
        "reasoning_effort": args.reasoning_effort,
        "codex_version": codex_version,
        "seed_mode": "fixed" if args.fixed_seeds else "fresh_random",
        "conditions": list(args.conditions),
        "runs": runs,
        "aggregates": aggregates,
    }
    (args.output_dir / "results.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    summary = _summary_markdown(payload)
    (args.output_dir / "SUMMARY.md").write_text(summary, encoding="utf-8")
    print(summary)
    valid = all(run.get("benchmark_valid") for run in runs)
    passed = all(run.get("success") for run in runs)
    return 0 if valid and passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
