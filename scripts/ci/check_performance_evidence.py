#!/usr/bin/env python3
"""Fail-closed evidence gate for the nightly performance workflow (repair 1.6-a, issue #2062).

The historic failure mode: ``pytest -m benchmark --benchmark-only`` selected the
Lux perf smoke tests but skipped every one of them (none use the
pytest-benchmark fixture), reported exit 0, and the workflow published
``status=passed`` — green with zero executed benchmarks, on every nightly run.

This gate makes the run's own evidence a hard requirement. It validates, from a
pytest-json-report file and the harness's artifact directory:

1. the report exists and parses;
2. at least one selected benchmark test actually EXECUTED (passed or failed) —
   an all-skipped run is the false-green and fails loudly;
3. no executed test failed (a failure is a regression or broken harness, not
   valid green evidence);
4. every REQUIRED test (the four baseline writers, whose bodies perform the
   committed-baseline comparison unconditionally) has outcome ``passed`` — a
   skipped required test (e.g. psutil missing starving the memory baseline)
   fails instead of silently thinning the evidence;
5. every required artifact exists in the artifacts directory and parses as a
   non-empty JSON object;
6. every required artifact's same-named committed baseline exists and parses —
   an invalid or missing baseline means the in-test comparison could not have
   run against real data.

Because each required test's body writes its artifact and then compares it
against the committed baseline with no conditional, "all required tests passed
AND all artifacts present AND all baselines valid" is proof the comparison
executed — the property the old workflow never checked.

Exit codes (one line per violation on stderr):

* 0 — every check holds: valid, green evidence.
* 1 — the evidence itself is INVALID (missing/unparsable report, zero
  executed tests, a required test not selected or skipped, a missing or
  unparsable artifact or committed baseline). Such a run must never be
  reported as a performance regression — it is a broken harness.
* 2 — the evidence is structurally valid (report parsed, tests executed,
  all artifacts and baselines present and parsable) and at least one
  REQUIRED baseline-writer test FAILED. Only the writers perform the
  committed-baseline comparison (assert_regression_within_tolerance), so
  this is the sole regression-classifiable outcome.
* 3 — the evidence is structurally valid and every required writer
  passed, but some OTHER selected benchmark test failed. That proves a
  suite failure, not a baseline-tolerance breach, and must not open a
  regression issue.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

DEFAULT_REQUIRED_ARTIFACTS = (
    "baseline_cold_start.json",
    "baseline_steady_state.json",
    "baseline_batch.json",
    "baseline_memory.json",
)

DEFAULT_REQUIRED_TESTS = (
    "test_single_image_cold_start_p95",
    "test_single_image_steady_state_p95",
    "test_batch_throughput_baseline",
    "test_memory_peak_rss_baseline",
)

EXECUTED_OUTCOMES = {"passed", "failed", "error"}


def _load_json_object(path: Path, description: str, errors: List[str]) -> Dict[str, Any] | None:
    if not path.is_file():
        errors.append(f"{description} missing: {path}")
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        errors.append(f"{description} unreadable/invalid JSON: {path} ({exc})")
        return None
    if not isinstance(payload, dict) or not payload:
        errors.append(f"{description} is not a non-empty JSON object: {path}")
        return None
    return payload


def check_evidence(
    report_path: Path,
    artifacts_dir: Path,
    baselines_dir: Path,
    required_tests: List[str],
    required_artifacts: List[str],
) -> Tuple[List[str], List[str], List[str]]:
    """Return ``(invalid, regressions, other_failures)`` violation lists.

    ``invalid`` holds structural-evidence violations (the run cannot be
    classified as a regression); ``regressions`` holds failures of REQUIRED
    baseline-writer tests — the only tests whose bodies perform the
    committed-baseline comparison — and is regression-classifiable only when
    ``invalid`` is empty; ``other_failures`` holds failures of any other
    selected benchmark test, which prove a suite failure rather than a
    tolerance breach. All three empty means the evidence is valid and green.
    """
    invalid: List[str] = []
    regressions: List[str] = []
    other_failures: List[str] = []

    report = _load_json_object(report_path, "pytest json report", invalid)
    if report is not None:
        tests = report.get("tests")
        if not isinstance(tests, list):
            invalid.append(f"pytest json report has no 'tests' array: {report_path}")
            tests = []
        outcomes = {str(t.get("nodeid", "")): str(t.get("outcome", "")) for t in tests if isinstance(t, dict)}
        executed = [nodeid for nodeid, outcome in outcomes.items() if outcome in EXECUTED_OUTCOMES]
        skipped = [nodeid for nodeid, outcome in outcomes.items() if outcome == "skipped"]
        failed = [nodeid for nodeid, outcome in outcomes.items() if outcome in {"failed", "error"}]

        if not executed:
            invalid.append(
                "zero benchmark tests executed"
                + (f" ({len(skipped)} selected tests were all skipped — the historic false-green)" if skipped else "")
            )
        for nodeid in failed:
            if any(required in nodeid for required in required_tests):
                regressions.append(f"required baseline-writer test failed (regression or broken harness): {nodeid}")
            else:
                other_failures.append(f"benchmark test failed (suite failure, not a baseline comparison): {nodeid}")
        for required in required_tests:
            matches = {nodeid: outcome for nodeid, outcome in outcomes.items() if required in nodeid}
            if not matches:
                invalid.append(f"required benchmark test not selected: {required}")
            elif not any(outcome in EXECUTED_OUTCOMES for outcome in matches.values()):
                # A required test that FAILED is already recorded above as a
                # test failure; only a required test that never executed
                # (e.g. skipped) is missing evidence outright.
                observed = ", ".join(sorted(set(matches.values())))
                invalid.append(f"required benchmark test did not pass: {required} (outcome: {observed})")

    for artifact_name in required_artifacts:
        _load_json_object(
            artifacts_dir / artifact_name,
            f"required benchmark artifact '{artifact_name}'",
            invalid,
        )
        _load_json_object(
            baselines_dir / artifact_name,
            f"committed baseline for '{artifact_name}'",
            invalid,
        )

    return invalid, regressions, other_failures


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--report", type=Path, required=True, help="pytest-json-report output file")
    parser.add_argument("--artifacts-dir", type=Path, required=True, help="BENCHMARK_ARTIFACTS_DIR used by the run")
    parser.add_argument("--baselines-dir", type=Path, required=True, help="committed baselines directory")
    parser.add_argument(
        "--require-test",
        action="append",
        dest="required_tests",
        default=None,
        help="required test name substring (repeatable); defaults to the four baseline writers",
    )
    parser.add_argument(
        "--require-artifact",
        action="append",
        dest="required_artifacts",
        default=None,
        help="required artifact filename (repeatable); defaults to the four baseline JSONs",
    )
    args = parser.parse_args(argv)

    required_tests = args.required_tests or list(DEFAULT_REQUIRED_TESTS)
    required_artifacts = args.required_artifacts or list(DEFAULT_REQUIRED_ARTIFACTS)

    invalid, regressions, other_failures = check_evidence(
        report_path=args.report,
        artifacts_dir=args.artifacts_dir,
        baselines_dir=args.baselines_dir,
        required_tests=required_tests,
        required_artifacts=required_artifacts,
    )

    if invalid:
        print("Performance evidence gate FAILED — run is NOT valid evidence:", file=sys.stderr)
        for error in invalid + regressions + other_failures:
            print(f"  - {error}", file=sys.stderr)
        return 1

    if regressions:
        print(
            "Performance evidence gate: evidence is valid but a committed-baseline comparison FAILED "
            "(regression candidate):",
            file=sys.stderr,
        )
        for error in regressions + other_failures:
            print(f"  - {error}", file=sys.stderr)
        return 2

    if other_failures:
        print(
            "Performance evidence gate: benchmark-suite failure with complete evidence — "
            "not a committed-baseline regression:",
            file=sys.stderr,
        )
        for error in other_failures:
            print(f"  - {error}", file=sys.stderr)
        return 3

    print(
        "Performance evidence gate passed: "
        f"{len(required_tests)} required tests passed, "
        f"{len(required_artifacts)} artifacts and committed baselines validated."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
