"""Human-readable APEX reports must not promote absent evidence to a pass."""

from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

import scripts.apex_pr_comment as apex_pr_comment
from transformation_portal.metrics.aggregator import compute_bucket_stats
from transformation_portal.metrics.contracts import BucketStats
from transformation_portal.metrics.ledger import PerformanceLedger
from transformation_portal.metrics.performance_capsule import PerformanceBucket, PerformanceCapsule

pytestmark = pytest.mark.unit


def _stats(*, insufficient: bool, status: str = "pass") -> list[dict]:
    return [
        {
            "bucket_name": "unknown",
            "zone": None,
            "pass_fail": status,
            "is_insufficient_data": insufficient,
            "p95": 1.0,
            "threshold_p95": 2.0,
            "count": 3 if insufficient else 20,
        }
    ]


@pytest.mark.parametrize("stats", [[], _stats(insufficient=True)])
def test_empty_or_insufficient_report_does_not_claim_performance_pass(stats):
    report = apex_pr_comment.generate_pr_comment(
        run_id="real-run",
        commit_sha="a" * 40,
        v1_stats=stats,
        v2_stats=stats,
        gate_result_v1={"should_block": False, "mode": "enforce"},
        gate_result_v2={"should_block": False, "mode": "shadow"},
    )

    assert "APEX Performance Verdict: **INSUFFICIENT_DATA**" in report
    assert "**V1 Gate:** INSUFFICIENT DATA" in report
    assert "**V2 Gate:** INSUFFICIENT DATA" in report
    assert "PASSED" not in report


@pytest.mark.parametrize("status", ["pass", "warn", "fail"])
def test_assessed_report_retains_existing_verdict(status):
    report = apex_pr_comment.generate_pr_comment(
        run_id="real-run",
        commit_sha="a" * 40,
        v1_stats=[],
        v2_stats=_stats(insufficient=False, status=status),
    )

    assert f"APEX Performance Verdict: **{status.upper()}**" in report


def test_blocked_gate_is_not_hidden_by_absent_statistics():
    report = apex_pr_comment.generate_pr_comment(
        run_id="real-run",
        commit_sha="a" * 40,
        v1_stats=[],
        v2_stats=[],
        gate_result_v1={"should_block": True, "mode": "enforce"},
    )

    assert "**V1 Gate:** BLOCKED" in report


@pytest.mark.parametrize("assessed_status", ["pass", "warn", "fail"])
def test_partial_sample_support_does_not_promote_a_pass_or_hide_a_regression(assessed_status):
    stats = _stats(insufficient=False, status=assessed_status) + _stats(insufficient=True)
    report = apex_pr_comment.generate_pr_comment(
        run_id="partial-run",
        commit_sha="a" * 40,
        v1_stats=[],
        v2_stats=stats,
        gate_result_v2={"should_block": assessed_status == "fail", "mode": "enforce"},
    )

    expected = "INSUFFICIENT_DATA" if assessed_status == "pass" else assessed_status.upper()
    assert f"APEX Performance Verdict: **{expected}**" in report
    assert "**V2 Gate:** PASSED" not in report
    if assessed_status == "fail":
        assert "**V2 Gate:** BLOCKED" in report


def test_ledger_rows_derive_insufficient_data_from_real_sample_count(tmp_path):
    ledger = tmp_path / "ledger.db"
    with sqlite3.connect(ledger) as connection:
        connection.execute(
            "CREATE TABLE apex_runs (run_id TEXT, bucket_name TEXT, zone TEXT, "
            "p50 REAL, p95 REAL, p99 REAL, count INTEGER, pass_fail TEXT, "
            "threshold_p95 REAL, workflow_version TEXT)"
        )
        connection.execute(
            "INSERT INTO apex_runs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("real-run", "unknown", "local", 55.0, 55.0, None, 3, "pass", 120.0, "v2"),
        )

    stats = apex_pr_comment.fetch_run_stats(str(ledger), "real-run", "v2")
    report = apex_pr_comment.generate_pr_comment(run_id="real-run", commit_sha="a" * 40, v1_stats=[], v2_stats=stats)

    assert stats[0]["is_insufficient_data"] is True
    assert "APEX Performance Verdict: **INSUFFICIENT_DATA**" in report


def test_small_comparison_does_not_label_an_observed_delta_as_faster():
    def bucket(p95):
        return BucketStats(
            bucket_name="unknown",
            count=3,
            p50=p95,
            p95=p95,
            p99=None,
            mean=p95,
            min=p95,
            max=p95,
            threshold_p50=90.0,
            threshold_p95=120.0,
            pass_fail="pass",
        )

    baseline, candidate = bucket(142.0), bucket(55.0)
    comparison = apex_pr_comment.generate_v1_v2_comparison({"unknown": baseline}, {"unknown": candidate})
    table = apex_pr_comment.generate_bucket_table({"unknown": candidate}, "Candidate")

    assert "Insufficient data" in comparison
    assert "Faster" not in comparison
    assert "📊" in table


def test_measurement_reader_does_not_promote_legacy_twenty_capsules(tmp_path):
    ledger = tmp_path / "legacy-twenty.db"
    PerformanceLedger(ledger)
    with sqlite3.connect(ledger) as connection:
        connection.execute(
            "INSERT INTO apex_runs (run_id, commit_sha, timestamp, workflow_version, zone, "
            "bucket_name, p50, p95, p99, count, threshold_p50, threshold_p95, pass_fail) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "real-run",
                "a" * 40,
                "2026-10-05T00:00:00+00:00",
                "v2",
                "local",
                "unknown",
                142.0,
                142.0,
                142.0,
                20,
                60.0,
                120.0,
                "fail",
            ),
        )

    stats = apex_pr_comment.fetch_run_stats(str(ledger), "real-run", "v2")
    report = apex_pr_comment.generate_pr_comment(run_id="real-run", commit_sha="a" * 40, v1_stats=[], v2_stats=stats)

    assert stats[0]["is_insufficient_data"] is True
    assert stats[0]["independent_count"] == 0
    assert stats[0]["capsule_count"] == 20
    assert stats[0]["measurement_unit"] is None
    assert stats[0]["insufficiency_reason"] == "unattributed_legacy"
    assert "APEX Performance Verdict: **INSUFFICIENT_DATA**" in report
    assert "Worst Offenders" not in report


def test_measurement_reader_rejects_unknown_summary(tmp_path):
    ledger = tmp_path / "unknown-summary.db"
    PerformanceLedger(ledger)
    with sqlite3.connect(ledger) as connection:
        connection.execute(
            "INSERT INTO apex_runs (run_id, commit_sha, timestamp, workflow_version, zone, "
            "bucket_name, p50, p95, count, threshold_p50, threshold_p95, pass_fail, raw_capsules_json) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "real-run",
                "a" * 40,
                "2026-10-05T00:00:00+00:00",
                "v2",
                "local",
                "unknown",
                1.0,
                1.0,
                20,
                60.0,
                120.0,
                "pass",
                '{"schema":"untrusted"}',
            ),
        )

    with pytest.raises(ValueError):
        apex_pr_comment.fetch_run_stats(str(ledger), "real-run", "v2")


def _store_reader_measurements(
    tmp_path,
    *,
    unit="image_wall_seconds",
    repeated_identity=False,
    workload="a" * 64,
    workflow_version="v2",
    run_id="real-run",
    total=1.0,
    timestamp="2026-10-05T00:00:00+00:00",
):
    capsules = [
        PerformanceCapsule(
            image_id=f"image-{index}",
            image_path=f"image-{index}.jpg",
            input_hash=f"input-{index}",
            original_shape=(16, 16),
            enforced_shape=(16, 16),
            pixel_count=256,
            dimension_adjustment="exact",
            timings={"total": total},
            workflow_version=workflow_version,
            zone="local",
            backend_id="fixture-backend",
            model_variant="fixture-model",
            config_hash="fixture-config",
            pipeline_version="fixture-pipeline",
            schema_version="2.1.0",
            measurement_id="batch-0" if repeated_identity else f"{run_id}:{workflow_version}:{unit}:{workload}:{index}",
            measurement_unit=unit,
            workload_id=workload,
        )
        for index in range(20)
    ]
    bucket = PerformanceBucket("reader-fixture", {}, 2.0, 3.0, measurement_unit=unit, workload_id=workload)
    stats = compute_bucket_stats(capsules, bucket)
    assert stats is not None
    ledger = tmp_path / "explicit-measurements.db"
    PerformanceLedger(ledger)
    with sqlite3.connect(ledger) as connection:
        connection.execute(
            "INSERT INTO apex_runs (run_id, commit_sha, timestamp, workflow_version, zone, bucket_name, "
            "p50, p95, p99, count, threshold_p50, threshold_p95, pass_fail, raw_capsules_json) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                run_id,
                "a" * 40,
                timestamp,
                workflow_version,
                "local",
                bucket.name,
                stats.p50,
                stats.p95,
                stats.p99,
                stats.count,
                stats.threshold_p50,
                stats.threshold_p95,
                stats.pass_fail,
                json.dumps(stats.measurement_evidence),
            ),
        )
    return ledger, stats


def test_measurement_reader_retains_explicit_twenty_image_measurements(tmp_path):
    ledger, _ = _store_reader_measurements(tmp_path)
    stats = apex_pr_comment.fetch_run_stats(str(ledger), "real-run", "v2")
    assert stats[0]["independent_count"] == 20
    assert stats[0]["capsule_count"] == 20
    assert stats[0]["measurement_unit"] == "image_wall_seconds"
    assert stats[0]["workload_id"] == "a" * 64
    assert stats[0]["is_insufficient_data"] is False


def test_measurement_reader_reports_one_batch_for_twenty_capsules(tmp_path):
    ledger, _ = _store_reader_measurements(tmp_path, unit="batch_wall_seconds", repeated_identity=True)
    stats = apex_pr_comment.fetch_run_stats(str(ledger), "real-run", "v2")
    report = apex_pr_comment.generate_pr_comment(run_id="real-run", commit_sha="a" * 40, v1_stats=[], v2_stats=stats)
    assert stats[0]["independent_count"] == 1
    assert stats[0]["capsule_count"] == 20
    assert stats[0]["is_insufficient_data"] is True
    assert "1 independent / 20 capsules" in report
    assert "batch_wall_seconds" in report


def test_measurement_reader_comparison_does_not_pool_units_or_workloads(tmp_path):
    first = tmp_path / "image"
    first.mkdir()
    second = tmp_path / "batch"
    second.mkdir()
    third = tmp_path / "different-workload"
    third.mkdir()
    _, image = _store_reader_measurements(first)
    _, batch = _store_reader_measurements(second, unit="batch_wall_seconds")
    _, other_workload = _store_reader_measurements(third, workload="b" * 64)
    for incompatible in (batch, other_workload):
        comparison = apex_pr_comment.generate_v1_v2_comparison({"fixture": image}, {"fixture": incompatible})
        assert "Insufficient data" in comparison
        assert "Faster" not in comparison
        assert "Similar" not in comparison
        assert "+0.0%" not in comparison


def test_measurement_reader_comparison_preserves_compatible_v1_v2_axis(tmp_path):
    first = tmp_path / "v1"
    first.mkdir()
    second = tmp_path / "v2"
    second.mkdir()
    _, v1 = _store_reader_measurements(first, workflow_version="v1")
    _, v2 = _store_reader_measurements(second, workflow_version="v2")
    comparison = apex_pr_comment.generate_v1_v2_comparison({"fixture": v1}, {"fixture": v2})
    assert "Similar" in comparison
    assert "+0.0%" in comparison
    assert "Insufficient data" not in comparison


def test_measurement_reader_mixed_legacy_and_eligible_does_not_promote_partial_report(tmp_path):
    ledger, _ = _store_reader_measurements(tmp_path)
    with sqlite3.connect(ledger) as connection:
        connection.execute(
            "INSERT INTO apex_runs (run_id, commit_sha, timestamp, workflow_version, zone, bucket_name, "
            "p50, p95, p99, count, threshold_p50, threshold_p95, pass_fail) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "real-run",
                "a" * 40,
                "2026-10-05T00:00:00+00:00",
                "v2",
                "local",
                "legacy",
                142.0,
                142.0,
                142.0,
                20,
                60.0,
                120.0,
                "fail",
            ),
        )
    stats = apex_pr_comment.fetch_run_stats(str(ledger), "real-run", "v2")
    report = apex_pr_comment.generate_pr_comment(
        run_id="real-run",
        commit_sha="a" * 40,
        v1_stats=[],
        v2_stats=stats,
        gate_result_v2={"should_block": False, "mode": "enforce"},
    )
    assert "APEX Performance Verdict: **INSUFFICIENT_DATA**" in report
    assert "**V2 Gate:** INSUFFICIENT DATA" in report
    assert "Worst Offenders" not in report
    assert "0 independent / 20 capsules" in report


def test_measurement_reader_direct_legacy_table_does_not_promote_count_twenty():
    legacy = BucketStats(
        bucket_name="legacy",
        count=20,
        p50=1.0,
        p95=1.0,
        p99=1.0,
        mean=1.0,
        min=1.0,
        max=1.0,
        threshold_p50=2.0,
        threshold_p95=3.0,
        pass_fail="pass",
    )
    table = apex_pr_comment.generate_bucket_table({"legacy": legacy}, "Legacy")
    assert "0 independent / 20 capsules" in table
    assert "📊" in table
    assert "✅" not in table


def test_measurement_reader_cli_reports_eligible_shadow_failure(tmp_path):
    ledger, stats = _store_reader_measurements(tmp_path, total=4.0)
    assert stats.count == 20 and not stats.is_insufficient_data
    assert stats.pass_fail == "fail"
    with sqlite3.connect(ledger) as connection:
        connection.execute(
            "INSERT INTO apex_runs (run_id, commit_sha, timestamp, workflow_version, zone, bucket_name, "
            "p50, p95, p99, count, threshold_p50, threshold_p95, pass_fail, raw_capsules_json) "
            "SELECT run_id, commit_sha, timestamp, workflow_version, NULL, bucket_name, "
            "p50, p95, p99, count, threshold_p50, threshold_p95, pass_fail, raw_capsules_json FROM apex_runs"
        )
    output = tmp_path / "shadow-report.md"
    repository = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            str(repository / "scripts/apex_pr_comment.py"),
            "--run-id",
            "real-run",
            "--commit-sha",
            "a" * 40,
            "--ledger-db",
            str(ledger),
            "--output",
            str(output),
        ],
        cwd=repository,
        capture_output=True,
        text=True,
        check=False,
    )
    report = output.read_text()
    assert result.returncode == 0, result.stdout + result.stderr
    assert "APEX Performance Verdict: **FAIL**" in report
    assert "**V2 Gate:** SHADOW ⚠️ (would block) (mode: shadow)" in report
    assert "**V2 Gate:** PASSED" not in report


@pytest.mark.parametrize("duplicate_global", [False, True])
def test_measurement_reader_cli_binds_judgement_to_requested_run_and_global_context(tmp_path, duplicate_global):
    ledger, _ = _store_reader_measurements(tmp_path)
    _store_reader_measurements(tmp_path, run_id="later-run-same-commit", total=142.0, timestamp="2026-10-05T01:00:00+00:00")
    with sqlite3.connect(ledger) as connection:
        # The matrix persists each run's local and global summaries for the same
        # bucket. A later invocation of the same commit must not replace either.
        connection.execute(
            "INSERT INTO apex_runs (run_id, commit_sha, timestamp, workflow_version, zone, bucket_name, "
            "p50, p95, p99, count, threshold_p50, threshold_p95, pass_fail, raw_capsules_json) "
            "SELECT run_id, commit_sha, timestamp, workflow_version, NULL, bucket_name, "
            "p50, p95, p99, count, threshold_p50, threshold_p95, pass_fail, raw_capsules_json "
            "FROM apex_runs WHERE zone = 'local'"
        )
        if duplicate_global:
            connection.execute(
                "INSERT INTO apex_runs (run_id, commit_sha, timestamp, workflow_version, zone, bucket_name, "
                "p50, p95, p99, count, threshold_p50, threshold_p95, pass_fail, raw_capsules_json) "
                "SELECT run_id, commit_sha, timestamp, workflow_version, zone, bucket_name, "
                "p50, p95, p99, count, threshold_p50, threshold_p95, pass_fail, raw_capsules_json "
                "FROM apex_runs WHERE run_id = 'real-run' AND zone IS NULL"
            )
    output = tmp_path / "requested-run-comment.md"
    repository = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            str(repository / "scripts/apex_pr_comment.py"),
            "--run-id",
            "real-run",
            "--commit-sha",
            "a" * 40,
            "--ledger-db",
            str(ledger),
            "--output",
            str(output),
        ],
        cwd=repository,
        capture_output=True,
        text=True,
        check=False,
    )
    report = output.read_text()
    if duplicate_global:
        assert result.returncode == 1
        assert "APEX Performance Report [ERROR]" in report
        assert "Duplicate global bucket in requested run" in report
        assert "APEX Performance Verdict: **PASS**" not in report
        return
    assert result.returncode == 0, result.stdout + result.stderr
    assert "APEX Performance Verdict: **PASS**" in report
    assert "**V2 Gate:** PASSED" in report
    assert "1.00s" in report
    assert "142.00s" not in report
    assert "later-run-same-commit" not in report
