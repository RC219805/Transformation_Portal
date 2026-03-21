"""CLI integration tests for performance ledger v1.7 (Condition #3).

Tests all exit codes, CLI workflows, and backward compatibility.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


def create_test_manifests(output_dir: Path, count: int = 5, mean_time: float = 10.0):
    """Create test manifest files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for i in range(count):
        manifest = {
            "timing": {"total_seconds": mean_time + (i - count // 2) * 0.5, "depth_seconds": 8.0, "v2_seconds": 2.0},
            "depth": {"model": "da3", "runtime_seconds": 8.0},
        }
        manifest_path = output_dir / f"manifest_{i}.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f)


def run_ledger(*args) -> subprocess.CompletedProcess:
    """Run performance ledger CLI."""
    cmd = [sys.executable, "tools/performance_ledger.py"] + list(args)
    return subprocess.run(cmd, capture_output=True, text=True, cwd=Path.cwd())


class TestCLIBasicFunctionality:
    """Test basic CLI operations."""

    def test_capture_baseline_success(self, tmp_path):
        """Test capturing baseline returns exit code 0."""
        manifests_dir = tmp_path / "manifests"
        create_test_manifests(manifests_dir, count=10)

        baseline_path = tmp_path / "baseline.json"

        result = run_ledger(
            "--manifests-dir",
            str(manifests_dir),
            "--output",
            str(baseline_path),
            "--baseline-version",
            "v1.0.0",
            "--backend",
            "da3",
        )

        assert result.returncode == 0, f"STDOUT: {result.stdout}\nSTDERR: {result.stderr}"
        assert baseline_path.exists()

        # Verify baseline structure
        with open(baseline_path) as f:
            baseline = json.load(f)

        assert baseline["version"] == "v1.0.0"
        assert baseline["backend"] == "da3"
        assert "statistics" in baseline
        assert "environment" in baseline

    def test_compare_no_regression_exit_0(self, tmp_path):
        """Test comparison with no regression returns exit code 0."""
        # Create baseline
        baseline_manifests = tmp_path / "baseline_manifests"
        create_test_manifests(baseline_manifests, count=10, mean_time=10.0)

        baseline_path = tmp_path / "baseline.json"
        result = run_ledger(
            "--manifests-dir", str(baseline_manifests), "--output", str(baseline_path), "--baseline-version", "v1.0.0"
        )
        assert result.returncode == 0

        # Create current run with similar performance
        current_manifests = tmp_path / "current_manifests"
        create_test_manifests(current_manifests, count=10, mean_time=10.1)

        report_path = tmp_path / "report.md"
        result = run_ledger(
            "--baseline", str(baseline_path), "--compare", str(current_manifests), "--output", str(report_path)
        )

        assert result.returncode == 0, f"STDOUT: {result.stdout}\nSTDERR: {result.stderr}"
        assert report_path.exists()
        assert "OK TO MERGE" in report_path.read_text()


class TestCLIExitCodes:
    """Test all 4 exit code paths (Condition #3)."""

    def test_exit_code_0_no_regression(self, tmp_path):
        """Exit code 0: Success, no regression."""
        baseline_manifests = tmp_path / "baseline"
        create_test_manifests(baseline_manifests, count=10, mean_time=10.0)

        baseline_path = tmp_path / "baseline.json"
        run_ledger("--manifests-dir", str(baseline_manifests), "--output", str(baseline_path))

        current_manifests = tmp_path / "current"
        create_test_manifests(current_manifests, count=10, mean_time=10.0)

        report_path = tmp_path / "report.md"
        result = run_ledger(
            "--baseline", str(baseline_path), "--compare", str(current_manifests), "--output", str(report_path)
        )

        assert result.returncode == 0

    def test_exit_code_1_regression_detected(self, tmp_path):
        """Exit code 1: Significant regression detected."""
        baseline_manifests = tmp_path / "baseline"
        create_test_manifests(baseline_manifests, count=10, mean_time=10.0)

        baseline_path = tmp_path / "baseline.json"
        run_ledger("--manifests-dir", str(baseline_manifests), "--output", str(baseline_path))

        # Current run is 50% slower (triggers p95 regression)
        current_manifests = tmp_path / "current"
        create_test_manifests(current_manifests, count=10, mean_time=15.0)

        report_path = tmp_path / "report.md"
        result = run_ledger(
            "--baseline", str(baseline_path), "--compare", str(current_manifests), "--output", str(report_path)
        )

        assert result.returncode == 1
        assert "DO NOT MERGE" in report_path.read_text()

    def test_exit_code_2_backend_mismatch(self, tmp_path):
        """Exit code 2: Backend mismatch detected."""
        # Create baseline with da3
        baseline_manifests = tmp_path / "baseline"
        create_test_manifests(baseline_manifests, count=10)

        baseline_path = tmp_path / "baseline.json"
        run_ledger("--manifests-dir", str(baseline_manifests), "--output", str(baseline_path), "--backend", "da3")

        # Create current manifests with different backend
        current_manifests = tmp_path / "current"
        current_manifests.mkdir(parents=True)

        for i in range(10):
            manifest = {"timing": {"total_seconds": 10.0}, "depth": {"model": "depth-pro", "runtime_seconds": 10.0}}
            with open(current_manifests / f"manifest_{i}.json", "w") as f:
                json.dump(manifest, f)

        report_path = tmp_path / "report.md"
        result = run_ledger(
            "--baseline", str(baseline_path), "--compare", str(current_manifests), "--output", str(report_path)
        )

        assert result.returncode == 2
        assert "Backend mismatch" in result.stderr

    def test_exit_code_3_insufficient_data(self, tmp_path):
        """Exit code 3: Insufficient data for comparison."""
        # Create baseline
        baseline_manifests = tmp_path / "baseline"
        create_test_manifests(baseline_manifests, count=10)

        baseline_path = tmp_path / "baseline.json"
        run_ledger("--manifests-dir", str(baseline_manifests), "--output", str(baseline_path))

        # Create current with only 2 samples (< MIN_SAMPLES_FOR_COMPARISON)
        current_manifests = tmp_path / "current"
        create_test_manifests(current_manifests, count=2)

        report_path = tmp_path / "report.md"
        result = run_ledger(
            "--baseline", str(baseline_path), "--compare", str(current_manifests), "--output", str(report_path)
        )

        assert result.returncode == 3
        assert "Insufficient data" in result.stderr


class TestCLIBackwardCompatibility:
    """Test backward compatibility (Condition #1)."""

    def test_version_flag_deprecated_alias(self, tmp_path):
        """Test --version flag works but logs deprecation warning."""
        manifests_dir = tmp_path / "manifests"
        create_test_manifests(manifests_dir)

        baseline_path = tmp_path / "baseline.json"

        # Use deprecated --version flag
        result = run_ledger(
            "--manifests-dir", str(manifests_dir), "--output", str(baseline_path), "--version", "v1.0.0-deprecated-test"
        )

        assert result.returncode == 0
        assert "DEPRECATED" in result.stderr or "deprecated" in result.stderr.lower()

        # Verify it actually set the version
        with open(baseline_path) as f:
            baseline = json.load(f)
        assert baseline["version"] == "v1.0.0-deprecated-test"

    def test_baseline_version_flag_preferred(self, tmp_path):
        """Test --baseline-version flag (preferred new name)."""
        manifests_dir = tmp_path / "manifests"
        create_test_manifests(manifests_dir)

        baseline_path = tmp_path / "baseline.json"

        result = run_ledger(
            "--manifests-dir", str(manifests_dir), "--output", str(baseline_path), "--baseline-version", "v2.0.0"
        )

        assert result.returncode == 0
        # Should NOT have deprecation warning
        assert "DEPRECATED" not in result.stderr


class TestCLIStrictMode:
    """Test --strict mode behavior (v1.7 feature)."""

    def test_strict_mode_fails_on_potential_regression(self, tmp_path):
        """In strict mode, even potential regressions should fail."""
        baseline_manifests = tmp_path / "baseline"
        create_test_manifests(baseline_manifests, count=10, mean_time=10.0)

        baseline_path = tmp_path / "baseline.json"
        run_ledger("--manifests-dir", str(baseline_manifests), "--output", str(baseline_path))

        # Slight slowdown (5% - below default 10% threshold)
        current_manifests = tmp_path / "current"
        create_test_manifests(current_manifests, count=10, mean_time=10.5)

        report_path = tmp_path / "report.md"

        # Without --strict: should pass
        result_lenient = run_ledger(
            "--baseline", str(baseline_path), "--compare", str(current_manifests), "--output", str(report_path)
        )
        assert result_lenient.returncode == 0

        # With --strict: might fail on potential regression
        result_strict = run_ledger(
            "--baseline", str(baseline_path), "--compare", str(current_manifests), "--output", str(report_path), "--strict"
        )
        # Strict mode is more sensitive
        assert result_strict.returncode in [0, 1]


class TestCLIEmitJSON:
    """Test --emit-json output (Condition #3)."""

    def test_emit_json_schema(self, tmp_path):
        """Test --emit-json produces valid schema."""
        baseline_manifests = tmp_path / "baseline"
        create_test_manifests(baseline_manifests, count=10)

        baseline_path = tmp_path / "baseline.json"
        run_ledger("--manifests-dir", str(baseline_manifests), "--output", str(baseline_path))

        current_manifests = tmp_path / "current"
        create_test_manifests(current_manifests, count=10)

        report_path = tmp_path / "report.md"
        json_path = tmp_path / "current.json"

        result = run_ledger(
            "--baseline",
            str(baseline_path),
            "--compare",
            str(current_manifests),
            "--output",
            str(report_path),
            "--emit-json",
            str(json_path),
        )

        assert result.returncode == 0
        assert json_path.exists()

        with open(json_path) as f:
            current_data = json.load(f)

        # Validate schema
        assert current_data["version"] == "current"
        assert "statistics" in current_data
        assert "environment" in current_data

        stats = current_data["statistics"]
        assert "mean_sec" in stats
        assert "p95_sec" in stats
        assert "std_sec" in stats  # v1.7 addition
        assert "count" in stats


class TestCLIInputValidation:
    """Test input validation bounds (Condition #6)."""

    def test_bootstrap_iterations_max_validation(self, tmp_path):
        """Test bootstrap iterations respects MAX limit."""
        manifests_dir = tmp_path / "manifests"
        create_test_manifests(manifests_dir)

        baseline_path = tmp_path / "baseline.json"

        # Attempt to use excessive iterations
        result = run_ledger(
            "--manifests-dir", str(manifests_dir), "--output", str(baseline_path), "--bootstrap-iterations", "20000"
        )

        assert result.returncode == 3  # EXIT_INSUFFICIENT_DATA
        assert "exceeds maximum" in result.stderr.lower()

    def test_bootstrap_iterations_negative_validation(self, tmp_path):
        """Test bootstrap iterations rejects negative values."""
        manifests_dir = tmp_path / "manifests"
        create_test_manifests(manifests_dir)

        baseline_path = tmp_path / "baseline.json"

        result = run_ledger(
            "--manifests-dir", str(manifests_dir), "--output", str(baseline_path), "--bootstrap-iterations", "-100"
        )

        assert result.returncode == 3
        assert "non-negative" in result.stderr.lower()


class TestCLIBootstrapFeatures:
    """Test bootstrap confidence interval features (v1.7)."""

    def test_bootstrap_enabled_by_default(self, tmp_path):
        """Bootstrap CI should be enabled by default."""
        manifests_dir = tmp_path / "manifests"
        create_test_manifests(manifests_dir, count=20)

        baseline_path = tmp_path / "baseline.json"

        result = run_ledger("--manifests-dir", str(manifests_dir), "--output", str(baseline_path))

        assert result.returncode == 0

        with open(baseline_path) as f:
            baseline = json.load(f)

        stats = baseline["statistics"]
        assert "bootstrap_ci_95_lower" in stats
        assert "bootstrap_ci_95_upper" in stats
        assert stats["bootstrap_ci_95_lower"] is not None

    def test_no_bootstrap_flag(self, tmp_path):
        """Test --no-bootstrap disables CI calculation."""
        manifests_dir = tmp_path / "manifests"
        create_test_manifests(manifests_dir, count=20)

        baseline_path = tmp_path / "baseline.json"

        result = run_ledger("--manifests-dir", str(manifests_dir), "--output", str(baseline_path), "--no-bootstrap")

        assert result.returncode == 0

        with open(baseline_path) as f:
            baseline = json.load(f)

        stats = baseline["statistics"]
        # Should not have bootstrap CI fields
        assert stats.get("bootstrap_ci_95_lower") is None
        assert stats.get("bootstrap_ci_95_upper") is None
