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

Exit code 0 only when every check holds; 1 otherwise, with one line per
violation on stderr.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

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
) -> List[str]:
    """Return a list of violations; empty means the evidence is valid."""
    errors: List[str] = []

    report = _load_json_object(report_path, "pytest json report", errors)
    if report is not None:
        tests = report.get("tests")
        if not isinstance(tests, list):
            errors.append(f"pytest json report has no 'tests' array: {report_path}")
            tests = []
        outcomes = {str(t.get("nodeid", "")): str(t.get("outcome", "")) for t in tests if isinstance(t, dict)}
        executed = [nodeid for nodeid, outcome in outcomes.items() if outcome in EXECUTED_OUTCOMES]
        skipped = [nodeid for nodeid, outcome in outcomes.items() if outcome == "skipped"]
        failed = [nodeid for nodeid, outcome in outcomes.items() if outcome in {"failed", "error"}]

        if not executed:
            errors.append(
                "zero benchmark tests executed"
                + (f" ({len(skipped)} selected tests were all skipped — the historic false-green)" if skipped else "")
            )
        for nodeid in failed:
            errors.append(f"benchmark test failed (regression or broken harness): {nodeid}")
        for required in required_tests:
            matches = {nodeid: outcome for nodeid, outcome in outcomes.items() if required in nodeid}
            if not matches:
                errors.append(f"required benchmark test not selected: {required}")
            elif not any(outcome == "passed" for outcome in matches.values()):
                observed = ", ".join(sorted(set(matches.values())))
                errors.append(f"required benchmark test did not pass: {required} (outcome: {observed})")

    for artifact_name in required_artifacts:
        _load_json_object(
            artifacts_dir / artifact_name,
            f"required benchmark artifact '{artifact_name}'",
            errors,
        )
        _load_json_object(
            baselines_dir / artifact_name,
            f"committed baseline for '{artifact_name}'",
            errors,
        )

    return errors


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

    errors = check_evidence(
        report_path=args.report,
        artifacts_dir=args.artifacts_dir,
        baselines_dir=args.baselines_dir,
        required_tests=required_tests,
        required_artifacts=required_artifacts,
    )

    if errors:
        print("Performance evidence gate FAILED:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1

    print(
        "Performance evidence gate passed: "
        f"{len(required_tests)} required tests passed, "
        f"{len(required_artifacts)} artifacts and committed baselines validated."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
