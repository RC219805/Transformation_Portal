"""Tests for scripts/ci/check_performance_evidence.py (repair 1.6-a, #2062).

Acceptance criteria from the issue:
1. A run with zero executed tests fails.
2. A run missing any of the four artifacts fails.
3. A run whose baseline JSON is absent or unparsable fails.
4. A run where the comparison did not execute fails (a required
   baseline-writer test that skipped instead of passing).
5. Green requires: >=1 executed test AND four parsed artifacts AND the
   required tests passed against parseable committed baselines.

Exit-code classification (review rounds on PR #2071): invalid evidence
exits 1 and must never be reported as a regression; a FAILED required
baseline-writer test with otherwise-valid evidence exits 2 (the only
regression-classifiable outcome, since only the writers perform the
committed-baseline comparison); a failed non-writer benchmark test with
valid evidence exits 3 (suite failure, never a regression claim);
invalidity takes precedence over both failure classes.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "ci"))

from check_performance_evidence import (  # noqa: E402
    DEFAULT_REQUIRED_ARTIFACTS,
    DEFAULT_REQUIRED_TESTS,
    check_evidence,
    main,
)

pytestmark = [pytest.mark.unit]

_MODULE = "tests/benchmarks/test_lux_depth_v3_perf_smoke.py"


def _report(outcomes: dict) -> dict:
    return {
        "summary": {"total": len(outcomes)},
        "tests": [{"nodeid": f"{_MODULE}::{name}", "outcome": outcome} for name, outcome in outcomes.items()],
    }


def _passing_outcomes() -> dict:
    return {name: "passed" for name in DEFAULT_REQUIRED_TESTS}


def _write_environment(
    tmp_path: Path, outcomes: dict, artifacts=DEFAULT_REQUIRED_ARTIFACTS, baselines=DEFAULT_REQUIRED_ARTIFACTS
):
    report_path = tmp_path / "perf-test-report.json"
    report_path.write_text(json.dumps(_report(outcomes)), encoding="utf-8")
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)
    for name in artifacts:
        (artifacts_dir / name).write_text(json.dumps({"p95_ms": 12.5, "test": name}), encoding="utf-8")
    baselines_dir = tmp_path / "baselines"
    baselines_dir.mkdir(exist_ok=True)
    for name in baselines:
        (baselines_dir / name).write_text(json.dumps({"p95_ms": 10.0}), encoding="utf-8")
    return report_path, artifacts_dir, baselines_dir


def _run(report_path: Path, artifacts_dir: Path, baselines_dir: Path):
    return check_evidence(
        report_path=report_path,
        artifacts_dir=artifacts_dir,
        baselines_dir=baselines_dir,
        required_tests=list(DEFAULT_REQUIRED_TESTS),
        required_artifacts=list(DEFAULT_REQUIRED_ARTIFACTS),
    )


def _cli_args(report_path: Path, artifacts_dir: Path, baselines_dir: Path) -> list:
    return [
        "--report",
        str(report_path),
        "--artifacts-dir",
        str(artifacts_dir),
        "--baselines-dir",
        str(baselines_dir),
    ]


class TestEvidenceGate:
    def test_valid_evidence_passes(self, tmp_path: Path) -> None:
        invalid, regressions, other_failures = _run(*_write_environment(tmp_path, _passing_outcomes()))
        assert invalid == []
        assert regressions == []
        assert other_failures == []

    def test_all_skipped_run_fails_as_historic_false_green(self, tmp_path: Path) -> None:
        """Criterion 1: the exact --benchmark-only failure mode — every
        selected test skipped, pytest exit 0 — must fail loudly."""
        outcomes = {name: "skipped" for name in DEFAULT_REQUIRED_TESTS}
        invalid, _, _ = _run(*_write_environment(tmp_path, outcomes))
        assert any("zero benchmark tests executed" in e for e in invalid)
        assert any("false-green" in e for e in invalid)

    def test_missing_report_fails(self, tmp_path: Path) -> None:
        _, artifacts_dir, baselines_dir = _write_environment(tmp_path, _passing_outcomes())
        invalid, _, _ = _run(tmp_path / "absent-report.json", artifacts_dir, baselines_dir)
        assert any("pytest json report missing" in e for e in invalid)

    def test_missing_artifact_fails(self, tmp_path: Path) -> None:
        """Criterion 2: any of the four artifacts missing fails."""
        report_path, artifacts_dir, baselines_dir = _write_environment(tmp_path, _passing_outcomes())
        (artifacts_dir / "baseline_memory.json").unlink()
        invalid, _, _ = _run(report_path, artifacts_dir, baselines_dir)
        assert any("baseline_memory.json" in e and "missing" in e for e in invalid)

    def test_unparsable_artifact_fails(self, tmp_path: Path) -> None:
        report_path, artifacts_dir, baselines_dir = _write_environment(tmp_path, _passing_outcomes())
        (artifacts_dir / "baseline_batch.json").write_text("{not json", encoding="utf-8")
        invalid, _, _ = _run(report_path, artifacts_dir, baselines_dir)
        assert any("baseline_batch.json" in e and "invalid JSON" in e for e in invalid)

    def test_missing_or_invalid_committed_baseline_fails(self, tmp_path: Path) -> None:
        """Criterion 3: absent or unparsable committed baselines fail."""
        report_path, artifacts_dir, baselines_dir = _write_environment(tmp_path, _passing_outcomes())
        (baselines_dir / "baseline_cold_start.json").unlink()
        (baselines_dir / "baseline_steady_state.json").write_text("[]", encoding="utf-8")
        invalid, _, _ = _run(report_path, artifacts_dir, baselines_dir)
        assert any("committed baseline for 'baseline_cold_start.json' missing" in e for e in invalid)
        assert any("baseline_steady_state.json" in e and "non-empty JSON object" in e for e in invalid)

    def test_required_test_skipped_fails_comparison_evidence(self, tmp_path: Path) -> None:
        """Criterion 4: the in-test committed-baseline comparison only runs
        when the writer test executes — a skipped required test (e.g. psutil
        missing starving the memory baseline) is invalid evidence, not a
        regression."""
        outcomes = _passing_outcomes()
        outcomes["test_memory_peak_rss_baseline"] = "skipped"
        report_path, artifacts_dir, baselines_dir = _write_environment(tmp_path, outcomes)
        invalid, regressions, other_failures = _run(report_path, artifacts_dir, baselines_dir)
        assert any("test_memory_peak_rss_baseline" in e and "did not pass" in e for e in invalid)
        assert regressions == []
        assert other_failures == []

    def test_failed_writer_is_regression_classifiable(self, tmp_path: Path) -> None:
        """A failed writer with all artifacts and baselines intact is the
        one outcome allowed to claim a committed-baseline regression."""
        outcomes = _passing_outcomes()
        outcomes["test_single_image_cold_start_p95"] = "failed"
        report_path, artifacts_dir, baselines_dir = _write_environment(tmp_path, outcomes)
        invalid, regressions, other_failures = _run(report_path, artifacts_dir, baselines_dir)
        assert invalid == []
        assert any("regression or broken harness" in e for e in regressions)
        assert other_failures == []

    def test_failed_non_writer_is_suite_failure_not_regression(self, tmp_path: Path) -> None:
        """A failed non-writer benchmark test proves neither a latency nor a
        memory regression — it must never be regression-classified."""
        outcomes = _passing_outcomes()
        outcomes["test_output_invariants_smoke"] = "failed"
        report_path, artifacts_dir, baselines_dir = _write_environment(tmp_path, outcomes)
        invalid, regressions, other_failures = _run(report_path, artifacts_dir, baselines_dir)
        assert invalid == []
        assert regressions == []
        assert any("suite failure, not a baseline comparison" in e for e in other_failures)

    def test_required_test_not_selected_fails(self, tmp_path: Path) -> None:
        outcomes = _passing_outcomes()
        del outcomes["test_batch_throughput_baseline"]
        report_path, artifacts_dir, baselines_dir = _write_environment(tmp_path, outcomes)
        invalid, _, _ = _run(report_path, artifacts_dir, baselines_dir)
        assert any("test_batch_throughput_baseline" in e and "not selected" in e for e in invalid)

    def test_cli_exit_codes(self, tmp_path: Path) -> None:
        """Criterion 5: exit 0 only with full evidence; 1 otherwise."""
        report_path, artifacts_dir, baselines_dir = _write_environment(tmp_path, _passing_outcomes())
        args = _cli_args(report_path, artifacts_dir, baselines_dir)
        assert main(args) == 0
        (artifacts_dir / "baseline_cold_start.json").unlink()
        assert main(args) == 1

    def test_cli_exit_2_for_regression_with_valid_evidence(self, tmp_path: Path) -> None:
        """A tolerance failure whose evidence is otherwise complete exits 2
        so the workflow may report status=regression."""
        outcomes = _passing_outcomes()
        outcomes["test_batch_throughput_baseline"] = "failed"
        report_path, artifacts_dir, baselines_dir = _write_environment(tmp_path, outcomes)
        assert main(_cli_args(report_path, artifacts_dir, baselines_dir)) == 2

    def test_cli_exit_3_for_non_writer_failure_with_valid_evidence(self, tmp_path: Path) -> None:
        """A non-writer failure with complete evidence exits 3: the workflow
        reports status=failed and must not open a regression issue."""
        outcomes = _passing_outcomes()
        outcomes["test_output_invariants_smoke"] = "failed"
        report_path, artifacts_dir, baselines_dir = _write_environment(tmp_path, outcomes)
        assert main(_cli_args(report_path, artifacts_dir, baselines_dir)) == 3

    def test_cli_invalid_evidence_outranks_test_failure(self, tmp_path: Path) -> None:
        """A writer that failed BEFORE producing its artifact is a broken
        harness (exit 1), never a regression claim (exit 2)."""
        outcomes = _passing_outcomes()
        outcomes["test_memory_peak_rss_baseline"] = "failed"
        report_path, artifacts_dir, baselines_dir = _write_environment(tmp_path, outcomes)
        (artifacts_dir / "baseline_memory.json").unlink()
        assert main(_cli_args(report_path, artifacts_dir, baselines_dir)) == 1
