"""Data-only public CLI integration: raw observations through persisted reports.

The explicit real-evidence flags are invented scalar fixtures for exercising
eligibility, not measurements from photographs, models or production runs.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from transformation_portal.metrics.aggregator import compute_bucket_stats, log_aggregated_stats_to_ledger
from transformation_portal.metrics.ledger import PerformanceLedger
from transformation_portal.metrics.performance_capsule import PerformanceBucket, PerformanceCapsule

pytestmark = pytest.mark.integration

REPOSITORY = Path(__file__).resolve().parents[1]
RUN_ID = "integration-current"
COMMIT = "a" * 40
TIMESTAMP = "2026-10-16T00:00:00+00:00"


def _capsules(workflow, *, total=5.0):
    return [
        PerformanceCapsule(
            image_id=f"invented-{index}",
            image_path=f"/invented/{index}.jpg",
            input_hash=hashlib.sha256(str(index).encode()).hexdigest(),
            original_shape=(2000, 3000),
            enforced_shape=(2000, 3000),
            pixel_count=6_000_000,
            dimension_adjustment="exact",
            backend_id="da3",
            model_variant="metric",
            device="cpu",
            dtype="float32",
            config_hash="frozen-recipe",
            pipeline_version="frozen-source",
            timings={"total": total},
            workflow_version=workflow,
            zone="local",
            is_synthetic=False,
            schema_version="2.1.0",
            measurement_id=f"{workflow}-call-{index}",
            measurement_unit="image_wall_seconds",
            workload_id="fixed-fixture-workload",
            captured_at=TIMESTAMP,
        )
        for index in range(20)
    ]


def _write_observations(directory, mode="eligible"):
    directory.mkdir()
    for workflow in ("v1", "v2"):
        capsules = _capsules(workflow, total=20.0 if workflow == "v2" else 5.0)
        if mode == "repeated":
            capsules = [capsules[0]] * 20
        elif mode == "mixed-context":
            capsules[-1] = replace(capsules[-1], backend_id="depth_pro")
        elif mode == "legacy":
            capsules = [
                replace(item, schema_version="2.0.0", measurement_id=None, measurement_unit=None, workload_id=None)
                for item in capsules
            ]
        payload = {
            "run_spec": {"run_id": RUN_ID, "commit_sha": COMMIT, "workflow_version": workflow},
            "zone": "local",
            "capsules": [item.to_dict() for item in capsules],
        }
        (directory / f"observation_{workflow}_local.json").write_text(json.dumps(payload), encoding="utf-8")


def _cli(wrapper, *arguments, expected=0):
    environment = os.environ.copy()
    environment.update(
        PYTHONPATH=os.pathsep.join((str(REPOSITORY / "src"), str(REPOSITORY))),
        PYTHONDONTWRITEBYTECODE="1",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
    )
    command = [sys.executable, "-B", str(REPOSITORY / "scripts" / wrapper), *map(str, arguments)]
    result = subprocess.run(command, cwd=REPOSITORY, env=environment, capture_output=True, text=True, check=False, timeout=30)
    assert result.returncode == expected, f"{command!r}\n{result.stdout}\n{result.stderr}"
    return result


def _rebuild(directory, database):
    _cli("apex_rebuild_ledger.py", "--input-dir", directory, "--ledger-db", database, "--clean")


def _aggregate(database):
    _cli("apex_aggregate_ledger.py", "--ledger-db", database, "--run-id", RUN_ID, "--commit-sha", COMMIT)


def _report(database, output, *, expected=0):
    _cli(
        "apex_pr_comment.py",
        "--ledger-db",
        database,
        "--run-id",
        RUN_ID,
        "--commit-sha",
        COMMIT,
        "--output",
        output,
        expected=expected,
    )
    return output.read_text(encoding="utf-8")


def _rows(database, where="1", parameters=()):
    with sqlite3.connect(database) as connection:
        connection.row_factory = sqlite3.Row
        return [dict(row) for row in connection.execute(f"SELECT * FROM apex_runs WHERE {where}", parameters)]


def _writer_stats(bucket_name, *, total=5.0, workflow="v2"):
    budget = PerformanceBucket(
        bucket_name, {}, 6.0, 10.0, measurement_unit="image_wall_seconds", workload_id="fixed-fixture-workload"
    )
    return compute_bucket_stats(_capsules(workflow, total=total), budget)


def _write_stats(database, stats, *, run_id=RUN_ID, workflow="v2", zone=None, timestamp=TIMESTAMP):
    log_aggregated_stats_to_ledger(run_id, COMMIT, workflow, timestamp, {zone: {stats.bucket_name: stats}}, str(database))


def test_cli_rebuild_aggregate_report_keeps_eligible_failure_and_compatible_comparison(tmp_path):
    observations, database = tmp_path / "observations", tmp_path / "ledger.db"
    _write_observations(observations)
    _rebuild(observations, database)
    _aggregate(database)
    report = _report(database, tmp_path / "report.md")

    assert "APEX Performance Verdict: **FAIL**" in report
    assert "**V1 Gate:**" in report
    assert "**V2 Gate:** SHADOW ⚠️ (would block) (mode: shadow)" in report
    assert "**V2 Gate:** PASSED" not in report
    assert "V1 Workflow Performance" in report and "V2 Workflow Performance" in report
    assert "| generic_medium | 5.00s | 20.00s | +300.0% | ❌ Slower |" in report
    assert "20 independent / 20 capsules" in report
    globals_ = _rows(database, "run_id = ? AND zone IS NULL AND bucket_name = 'generic_medium'", (RUN_ID,))
    assert {row["workflow_version"] for row in globals_} == {"v1", "v2"}
    assert len(globals_) == 2
    assert all(row["count"] == 20 for row in globals_)


def test_cli_reaggregation_has_one_global_per_key_and_preserves_another_run(tmp_path):
    observations, database = tmp_path / "observations", tmp_path / "ledger.db"
    _write_observations(observations)
    _rebuild(observations, database)
    _write_stats(database, _writer_stats("unrelated-budget"), run_id="unrelated-run")
    unrelated_before = _rows(database, "run_id = ?", ("unrelated-run",))

    _aggregate(database)
    _aggregate(database)
    current = _rows(database, "run_id = ?", (RUN_ID,))
    global_keys = [(row["workflow_version"], row["bucket_name"]) for row in current if row["zone"] is None]
    assert set(global_keys) == {("v1", "generic_medium"), ("v1", "unknown"), ("v2", "generic_medium"), ("v2", "unknown")}
    assert len(global_keys) == len(set(global_keys))
    assert len(current) == 8  # Two versions, two buckets, local + global.
    assert _rows(database, "run_id = ?", ("unrelated-run",)) == unrelated_before
    assert "APEX Performance Verdict: **FAIL**" in _report(database, tmp_path / "reaggregated.md")


@pytest.mark.parametrize("mode", ["repeated", "mixed-context", "legacy"])
def test_cli_ineligible_evidence_cannot_gain_comparison_or_pass(tmp_path, mode):
    observations, database = tmp_path / "observations", tmp_path / "ledger.db"
    _write_observations(observations, mode)
    _rebuild(observations, database)
    _aggregate(database)
    report = _report(database, tmp_path / "insufficient.md")
    assert "APEX Performance Verdict: **INSUFFICIENT_DATA**" in report
    assert "V1 vs V2 Comparison" in report
    assert "**V1 Gate:** INSUFFICIENT DATA" in report
    assert "**V2 Gate:** INSUFFICIENT DATA" in report
    assert "Faster" not in report and "Slower" not in report
    assert "+300.0%" not in report
    assert "**V2 Gate:** PASSED" not in report


def test_public_writer_replaces_only_requested_nullable_global_key(tmp_path):
    database = tmp_path / "ledger.db"
    PerformanceLedger(database)
    first = _writer_stats("writer-a")
    second = _writer_stats("writer-b")
    _write_stats(database, first)
    _write_stats(database, second)
    _write_stats(database, first, run_id="another-run")
    _write_stats(database, _writer_stats("writer-a", workflow="v1"), workflow="v1")
    _write_stats(database, first, zone="local")
    unchanged_where = "NOT (run_id = ? AND workflow_version = 'v2' AND zone IS NULL AND bucket_name = 'writer-a')"
    unaffected = _rows(database, unchanged_where, (RUN_ID,))

    _write_stats(database, _writer_stats("writer-a", total=9.0), timestamp="2026-10-16T01:00:00+00:00")
    target = _rows(database, "run_id = ? AND workflow_version = 'v2' AND zone IS NULL AND bucket_name = 'writer-a'", (RUN_ID,))
    assert len(target) == 1
    assert target[0]["p95"] == 9.0
    assert _rows(database, unchanged_where, (RUN_ID,)) == unaffected


def test_public_writer_global_replacement_rolls_back_whole_call_on_insert_failure(tmp_path):
    database = tmp_path / "ledger.db"
    PerformanceLedger(database)
    first, second = _writer_stats("writer-a"), _writer_stats("writer-b")
    _write_stats(database, first)
    _write_stats(database, second)
    before = _rows(database)
    with sqlite3.connect(database) as connection:
        connection.execute(
            "CREATE TRIGGER reject_second BEFORE INSERT ON apex_runs "
            "WHEN NEW.bucket_name = 'writer-b' BEGIN SELECT RAISE(ABORT, 'invented insertion refusal'); END"
        )
    with pytest.raises(sqlite3.IntegrityError, match="invented insertion refusal"):
        log_aggregated_stats_to_ledger(
            RUN_ID,
            COMMIT,
            "v2",
            "2026-10-16T01:00:00+00:00",
            {None: {"writer-a": _writer_stats("writer-a", total=9.0), "writer-b": second}},
            str(database),
        )
    assert _rows(database) == before


def test_report_cli_still_rejects_manually_duplicated_global_rows(tmp_path):
    database = tmp_path / "ledger.db"
    PerformanceLedger(database)
    _write_stats(database, _writer_stats("writer-a"))
    with sqlite3.connect(database) as connection:
        connection.execute("INSERT INTO apex_runs SELECT * FROM apex_runs")
    report = _report(database, tmp_path / "duplicate.md", expected=1)
    assert "APEX Performance Report [ERROR]" in report
    assert "Duplicate global bucket in requested run" in report
    assert "APEX Performance Verdict: **PASS**" not in report
