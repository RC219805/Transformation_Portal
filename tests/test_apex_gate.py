"""Tests for APEX gate logic.

Tests validate:
- Gate rule evaluation
- Mode enforcement (enforce/shadow/disabled)
- Worst-zone p95 gating
- Regression gating
- Bucket threshold gating
"""

import json
import logging
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.apex_enforce_gate import enforce_gate
from transformation_portal.metrics.aggregator import compute_bucket_stats
from transformation_portal.metrics.comparator import compare_to_baseline
from transformation_portal.metrics.contracts import BucketStats, Judgement, RegressionReport
from transformation_portal.metrics.gate import evaluate_gate, should_block
from transformation_portal.metrics.ledger import PerformanceLedger
from transformation_portal.metrics.performance_capsule import PerformanceBucket, PerformanceCapsule

pytestmark = pytest.mark.unit


@pytest.fixture
def legacy_gate_cli_ledger(tmp_path):
    ledger = tmp_path / "legacy-cli.db"
    PerformanceLedger(ledger)
    with sqlite3.connect(ledger) as connection:
        connection.execute(
            "INSERT INTO apex_runs (run_id, commit_sha, timestamp, workflow_version, zone, bucket_name, "
            "p50, p95, count, threshold_p50, threshold_p95, pass_fail) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "real-run",
                "a" * 40,
                "2026-10-05T00:00:00+00:00",
                "v2",
                "local",
                "unknown",
                142.0,
                142.0,
                3,
                60.0,
                120.0,
                "pass",
            ),
        )
    return ledger


@pytest.mark.parametrize("mode", ["shadow", "enforce"])
def test_measurement_reader_cli_logging_explains_insufficient_after_import_warning(legacy_gate_cli_ledger, mode):
    repository = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import logging,runpy,sys; logging.warning('package import warning fixture'); "
            "sys.argv=sys.argv[1:]; runpy.run_path(sys.argv[0],run_name='__main__')",
            str(repository / "scripts/apex_enforce_gate.py"),
            "--ledger-db",
            str(legacy_gate_cli_ledger),
            "--run-id",
            "real-run",
            "--commit-sha",
            "a" * 40,
            "--mode",
            mode,
        ],
        cwd=repository,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    output = result.stdout + result.stderr
    assert "package import warning fixture" in output
    assert "insufficient measurement evidence" in output.lower()
    assert "non-blocking" in output
    assert "gate passed" not in output.lower()


def test_measurement_reader_library_preserves_caller_logging_control(legacy_gate_cli_ledger, caplog):
    logger = logging.getLogger("apex_gate")
    with caplog.at_level(logging.ERROR, logger="apex_gate"):
        level = logger.level
        assert enforce_gate(legacy_gate_cli_ledger, "real-run", "a" * 40) == 0
        assert logger.level == level
        assert "insufficient" not in caplog.text.lower()


def test_measurement_reader_gate_abstains_on_legacy_twenty_capsules(tmp_path, caplog):
    ledger = tmp_path / "legacy-twenty.db"
    PerformanceLedger(ledger)
    with sqlite3.connect(ledger) as connection:
        connection.execute(
            "INSERT INTO apex_runs (run_id, commit_sha, timestamp, workflow_version, zone, "
            "bucket_name, p50, p95, count, threshold_p50, threshold_p95, pass_fail) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "real-run",
                "a" * 40,
                "2026-10-05T00:00:00+00:00",
                "v2",
                "local",
                "unknown",
                142.0,
                142.0,
                20,
                60.0,
                120.0,
                "fail",
            ),
        )

    with caplog.at_level(logging.INFO):
        result = enforce_gate(ledger, "real-run", "a" * 40)

    assert result == 0
    assert "insufficient" in caplog.text.lower()
    assert "gate passed" not in caplog.text.lower()


def test_measurement_reader_gate_rejects_malformed_passing_row(tmp_path, caplog):
    ledger = tmp_path / "malformed-summary.db"
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

    with caplog.at_level(logging.INFO):
        result = enforce_gate(ledger, "real-run", "a" * 40)

    assert result == 2
    assert "gate passed" not in caplog.text.lower()


def test_measurement_reader_direct_legacy_count_twenty_cannot_authorize_gate():
    legacy = BucketStats(
        bucket_name="legacy",
        count=20,
        p50=142.0,
        p95=142.0,
        p99=142.0,
        mean=142.0,
        min=142.0,
        max=142.0,
        threshold_p50=60.0,
        threshold_p95=120.0,
        pass_fail="fail",
    )
    result = evaluate_gate(
        bucket_stats={"legacy": legacy},
        worst_zone_p95_threshold=1.0,
        mode="enforce",
    )
    assert not result.should_block
    assert "insufficient" in result.explanation.lower()


def test_measurement_reader_direct_unbound_worst_scalar_abstains():
    stats = TestGate().create_bucket_stats("gate-fixture", 12.0)
    judgement = Judgement(
        run_id="current",
        workflow_version="v1",
        zone=None,
        bucket_stats={"gate-fixture": stats},
        regression_report=None,
        pass_fail="pass",
        explanation="fixture",
        worst_zone_p95=400.0,
    )
    result = evaluate_gate(judgement, worst_zone_p95_threshold=15.0)
    assert not result.should_block
    assert result.worst_zone_p95 == 12.0


def test_measurement_reader_direct_unbound_regression_scalar_abstains():
    stats = TestGate().create_bucket_stats("gate-fixture", 12.0)
    legacy_report = RegressionReport(
        baseline_run_id="baseline",
        baseline_commit_sha="abc",
        current_run_id="current",
        current_commit_sha="def",
        bucket_regressions={"gate-fixture": 1.0},
        max_regression=1.0,
        max_regression_bucket="gate-fixture",
        status="fail",
        explanation="legacy scalar",
    )
    result = evaluate_gate(bucket_stats={"gate-fixture": stats}, regression_report=legacy_report)
    assert not result.should_block


def _insert_gate_measurements(connection, stats, *, run_id="real-run"):
    connection.execute(
        "INSERT INTO apex_runs (run_id, commit_sha, timestamp, workflow_version, zone, bucket_name, "
        "p50, p95, p99, count, threshold_p50, threshold_p95, pass_fail, raw_capsules_json) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            run_id,
            "a" * 40,
            "2026-10-05T00:00:00+00:00",
            "v2",
            "local",
            stats.bucket_name,
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


def test_measurement_reader_gate_retains_explicit_threshold_failure(tmp_path):
    ledger = tmp_path / "explicit-failure.db"
    PerformanceLedger(ledger)
    stats = TestGate().create_bucket_stats("gate-fixture", 16.0, pass_fail="fail", workflow_version="v2")
    with sqlite3.connect(ledger) as connection:
        _insert_gate_measurements(connection, stats)
    assert enforce_gate(ledger, "real-run", "a" * 40) == 1
    assert enforce_gate(ledger, "real-run", "a" * 40, mode="shadow") == 0


def test_measurement_reader_gate_mixed_legacy_and_eligible_does_not_claim_overall_pass(tmp_path, caplog):
    ledger = tmp_path / "partial-evidence.db"
    PerformanceLedger(ledger)
    stats = TestGate().create_bucket_stats("gate-fixture", 12.0, workflow_version="v2")
    with sqlite3.connect(ledger) as connection:
        _insert_gate_measurements(connection, stats)
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
    with caplog.at_level(logging.INFO):
        assert enforce_gate(ledger, "real-run", "a" * 40) == 0
    assert "insufficient" in caplog.text.lower()
    assert "gate passed" not in caplog.text.lower()


class TestGate:
    """Test APEX gate functionality."""

    def create_bucket_stats(
        self,
        name: str,
        p95: float,
        threshold_p95: float = 15.0,
        pass_fail: str = "pass",
        count: int = 20,
        workflow_version: str = "v1",
        zone: str = "local",
    ) -> BucketStats:
        """Helper to create test BucketStats."""
        capsules = [
            PerformanceCapsule(
                image_id=f"image-{index}",
                image_path=f"image-{index}.jpg",
                input_hash=f"input-{index}",
                original_shape=(16, 16),
                enforced_shape=(16, 16),
                pixel_count=256,
                dimension_adjustment="exact",
                timings={"total": p95 if index == count - 1 else p95 * 0.7},
                workflow_version=workflow_version,
                zone=zone,
                backend_id="fixture-backend",
                model_variant="fixture-model",
                config_hash="fixture-config",
                pipeline_version="fixture-pipeline",
                schema_version="2.1.0",
                measurement_id=f"gate:{name}:{p95}:{index}",
                measurement_unit="image_wall_seconds",
                workload_id="a" * 64,
            )
            for index in range(count)
        ]
        bucket = PerformanceBucket(
            name,
            {},
            threshold_p95 * 0.7,
            threshold_p95,
            measurement_unit="image_wall_seconds",
            workload_id="a" * 64,
        )
        stats = compute_bucket_stats(capsules, bucket)
        assert stats is not None
        assert stats.pass_fail == pass_fail
        return stats

    def test_gate_disabled_always_passes(self):
        """Test that disabled mode always passes."""
        judgement = Judgement(
            run_id="test123",
            workflow_version="v1",
            zone=None,
            bucket_stats={},
            regression_report=None,
            pass_fail="fail",  # Even with fail verdict
            explanation="Threshold exceeded",
        )

        result = evaluate_gate(judgement, mode="disabled")

        assert not result.should_block
        assert result.mode == "disabled"
        assert "disabled" in result.explanation.lower()

    def test_gate_bucket_threshold_violation(self):
        """Test gate blocks on bucket threshold violation."""
        bucket_stats = {"pool_medium": self.create_bucket_stats("pool_medium", p95=16.0, threshold_p95=15.0, pass_fail="fail")}

        judgement = Judgement(
            run_id="test123",
            workflow_version="v1",
            zone=None,
            bucket_stats=bucket_stats,
            regression_report=None,
            pass_fail="fail",
            explanation="Bucket threshold exceeded",
        )

        result = evaluate_gate(judgement, mode="enforce")

        assert result.should_block
        assert "threshold violation" in result.explanation.lower()
        assert len(result.reasons) > 0

    def test_gate_worst_zone_p95_violation(self):
        """Test gate blocks on worst-zone p95 violation."""
        judgement = Judgement(
            run_id="test123",
            workflow_version="v1",
            zone=None,
            bucket_stats={
                "worst-fixture": self.create_bucket_stats("worst-fixture", 18.0, threshold_p95=30.0, zone="us-west-2b")
            },
            regression_report=None,
            pass_fail="pass",
            explanation="Within thresholds",
            worst_zone_p95=18.0,
            worst_zone_name="us-west-2b",
        )

        result = evaluate_gate(
            judgement,
            worst_zone_p95_threshold=15.0,
            mode="enforce",
        )

        assert result.should_block
        assert "worst-zone p95" in result.explanation.lower()
        assert "us-west-2b" in result.explanation

    def test_gate_regression_violation(self):
        """Test gate blocks on regression violation."""
        current = {"pool_medium": self.create_bucket_stats("pool_medium", 12.0)}
        regression_report = compare_to_baseline(
            current,
            {"pool_medium": self.create_bucket_stats("pool_medium", 10.0)},
            "current",
            "baseline",
            "def",
            "abc",
        )

        judgement = Judgement(
            run_id="test123",
            workflow_version="v1",
            zone=None,
            bucket_stats=current,
            regression_report=regression_report,
            pass_fail="pass",
            explanation="Regression detected",
        )

        result = evaluate_gate(
            judgement,
            max_regression_threshold=0.15,
            mode="enforce",
        )

        assert result.should_block
        assert "regression" in result.explanation.lower()
        assert "pool_medium" in result.explanation

    def test_gate_shadow_mode_warns_but_not_blocks(self):
        """Test shadow mode logs warnings but doesn't block."""
        judgement = Judgement(
            run_id="test123",
            workflow_version="v2",
            zone=None,
            bucket_stats={"shadow-fixture": self.create_bucket_stats("shadow-fixture", 18.0, threshold_p95=30.0)},
            regression_report=None,
            pass_fail="fail",
            explanation="Threshold exceeded",
            worst_zone_p95=18.0,
        )

        result = evaluate_gate(
            judgement,
            worst_zone_p95_threshold=15.0,
            mode="shadow",
        )

        assert not result.should_block  # Shadow mode doesn't block
        assert result.mode == "shadow"
        assert len(result.reasons) > 0  # But reasons are still recorded
        assert "shadow" in result.explanation.lower()

    def test_gate_passes_when_all_rules_pass(self):
        """Test gate passes when all rules pass."""
        bucket_stats = {"pool_medium": self.create_bucket_stats("pool_medium", p95=12.0, threshold_p95=15.0, pass_fail="pass")}

        judgement = Judgement(
            run_id="test123",
            workflow_version="v1",
            zone=None,
            bucket_stats=bucket_stats,
            regression_report=None,
            pass_fail="pass",
            explanation="All thresholds met",
            worst_zone_p95=12.0,
        )

        result = evaluate_gate(
            judgement,
            worst_zone_p95_threshold=15.0,
            max_regression_threshold=0.15,
            mode="enforce",
        )

        assert not result.should_block
        assert result.mode == "enforce"
        assert len(result.reasons) == 0
        assert "passed" in result.explanation.lower()

    def test_should_block_simple_api(self):
        """Test simple should_block API."""
        bucket_stats = {"pool_medium": self.create_bucket_stats("pool_medium", p95=16.0, threshold_p95=15.0, pass_fail="fail")}

        judgement = Judgement(
            run_id="test123",
            workflow_version="v1",
            zone=None,
            bucket_stats=bucket_stats,
            regression_report=None,
            pass_fail="fail",
            explanation="Threshold exceeded",
        )

        block, reason = should_block(judgement, mode="enforce")

        assert block
        assert "threshold" in reason.lower() or "bucket" in reason.lower()

    def test_multiple_violations_all_reported(self):
        """Test that multiple violations are all reported."""
        bucket_stats = {"pool_medium": self.create_bucket_stats("pool_medium", p95=18.0, threshold_p95=15.0, pass_fail="fail")}

        regression_report = compare_to_baseline(
            bucket_stats,
            {"pool_medium": self.create_bucket_stats("pool_medium", 15.0)},
            "current",
            "baseline",
            "def",
            "abc",
        )

        judgement = Judgement(
            run_id="test123",
            workflow_version="v1",
            zone=None,
            bucket_stats=bucket_stats,
            regression_report=regression_report,
            pass_fail="fail",
            explanation="Multiple violations",
            worst_zone_p95=18.0,
        )

        result = evaluate_gate(
            judgement,
            worst_zone_p95_threshold=15.0,
            max_regression_threshold=0.15,
            mode="enforce",
        )

        assert result.should_block
        # Should have 3 reasons: bucket violation, worst-zone, regression
        assert len(result.reasons) == 3
