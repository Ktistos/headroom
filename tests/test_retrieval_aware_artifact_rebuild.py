from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.rebuild_retrieval_aware_validated import (
    _validate_source_matrix,
    build_validated_artifact,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
VALIDATED = REPO_ROOT / "evaluation/retrieval_aware_agent_v2_validated"


def test_validated_artifact_rebuilds_exactly_offline(tmp_path):
    generated = build_validated_artifact(VALIDATED / "manifest.json", tmp_path)
    checked = json.loads((VALIDATED / "results.json").read_text(encoding="utf-8"))

    assert generated == checked
    assert (tmp_path / "SUMMARY.md").read_text(encoding="utf-8") == (
        VALIDATED / "SUMMARY.md"
    ).read_text(encoding="utf-8")
    assert len(generated["runs"]) == 15
    assert all(row["benchmark_valid"] for row in generated["runs"])
    assert {row["run_index"] for row in generated["runs"]} == set(range(1, 16))


def test_validated_artifact_rejects_duplicate_source_mapping(tmp_path):
    manifest = json.loads((VALIDATED / "manifest.json").read_text(encoding="utf-8"))
    first = manifest["source_rows"][0]
    manifest["source_rows"][1]["source_artifact"] = first["source_artifact"]
    manifest["source_rows"][1]["source_run_index"] = first["source_run_index"]
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate source mappings"):
        build_validated_artifact(manifest_path, tmp_path / "output")


def test_validated_artifact_rejects_incomplete_task_condition_matrix():
    rows = [
        {
            "run_index": 1,
            "condition": "observe",
            "regime": "low",
            "task": {"name": "alpha", "regime": "low", "seed": 1},
        },
        {
            "run_index": 2,
            "condition": "lossless",
            "regime": "low",
            "task": {"name": "alpha", "regime": "low", "seed": 1},
        },
        {
            "run_index": 3,
            "condition": "observe",
            "regime": "high",
            "task": {"name": "beta", "regime": "high", "seed": 2},
        },
        {
            "run_index": 4,
            "condition": "observe",
            "regime": "high",
            "task": {"name": "beta", "regime": "high", "seed": 2},
        },
    ]

    with pytest.raises(ValueError, match="duplicate task/condition row"):
        _validate_source_matrix(rows, ["observe", "lossless"])


def test_validated_artifact_rejects_duplicate_attempt_mapping(tmp_path):
    manifest = json.loads((VALIDATED / "manifest.json").read_text(encoding="utf-8"))
    first_attempt = manifest["attempts"][0]
    duplicated_index = first_attempt["source_run_indexes"][0]
    first_attempt["source_run_indexes"].append(duplicated_index)
    manifest["expected_counts"]["attempted"] += 1
    manifest["expected_counts"]["valid"] += 1
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate attempt mappings"):
        build_validated_artifact(manifest_path, tmp_path / "output")


def test_validated_artifact_rejects_unclassified_attempt(tmp_path):
    manifest = json.loads((VALIDATED / "manifest.json").read_text(encoding="utf-8"))
    manifest["excluded"]["superseded_valid"].pop()
    manifest["expected_counts"]["superseded_valid"] -= 1
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="do not exactly partition attempts"):
        build_validated_artifact(manifest_path, tmp_path / "output")


def test_validated_artifact_rejects_unattempted_classification(tmp_path):
    manifest = json.loads((VALIDATED / "manifest.json").read_text(encoding="utf-8"))
    unexpected = dict(manifest["excluded"]["superseded_valid"][0])
    unexpected["source_run_index"] = 999
    manifest["excluded"]["superseded_valid"].append(unexpected)
