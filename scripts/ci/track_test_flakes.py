#!/usr/bin/env python3
"""
Track test flake rates over time.

This script parses pytest JSON output and updates the flake ledger with
test pass/fail history. It identifies flaky tests (tests that fail
intermittently) and calculates flake rates.

Usage:
    python scripts/track_test_flakes.py <pytest_json_report>

Example:
    pytest --json-report --json-report-file=report.json
    python scripts/track_test_flakes.py report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
LEDGER_PATH = REPO_ROOT / "tests" / "flake_ledger.json"


def load_ledger() -> dict[str, Any]:
    """Load the flake tracking ledger."""
    if not LEDGER_PATH.exists():
        return {
            "version": "1.0.0",
            "last_updated": datetime.now(timezone.utc).isoformat(),
            "config": {
                "flake_threshold": 0.01,  # 1%
                "min_runs_for_analysis": 10,
                "quarantine_threshold": 0.03,  # 3%
                "auto_quarantine_enabled": False,
            },
            "tests": {},
        }

    with open(LEDGER_PATH) as f:
        return json.load(f)


def save_ledger(ledger: dict[str, Any]) -> None:
    """Save the flake tracking ledger with atomic writes."""
    ledger["last_updated"] = datetime.now(timezone.utc).isoformat()
    LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Atomic write: write to temp file, then replace
    temp_path = LEDGER_PATH.with_suffix(".tmp")
    try:
        with open(temp_path, "w") as f:
            json.dump(ledger, f, indent=2, sort_keys=False)
            f.write("\n")  # Trailing newline
            f.flush()  # Ensure data written to disk
        # Atomic rename (POSIX guarantees atomicity)
        temp_path.replace(LEDGER_PATH)
    except Exception as e:
        # Clean up temp file on error
        if temp_path.exists():
            temp_path.unlink()
        raise e


def parse_pytest_json(report_path: Path) -> dict[str, Any]:
    """Parse pytest JSON report with error handling."""
    try:
        with open(report_path) as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        print(
            f"⚠️  Error: Malformed JSON in {report_path}: {e}",
            file=sys.stderr,
        )
        return {"tests": []}  # Return empty but valid structure
    except IOError as e:
        print(
            f"⚠️  Error: Cannot read {report_path}: {e}",
            file=sys.stderr,
        )
        return {"tests": []}


def update_test_record(ledger: dict[str, Any], test_id: str, outcome: str, duration: float) -> None:
    """Update a single test's record in the ledger."""
    tests = ledger["tests"]

    if test_id not in tests:
        tests[test_id] = {
            "test_id": test_id,
            "total_runs": 0,
            "passes": 0,
            "failures": 0,
            "flake_count": 0,
            "flake_rate": 0.0,
            "last_run": None,
            "last_outcome": None,
            "last_failure": None,
            "status": "stable",  # stable | monitored | quarantined
            "history": [],  # Last 20 runs
        }

    record = tests[test_id]
    now = datetime.now(timezone.utc).isoformat()

    # Update counts
    record["total_runs"] += 1
    if outcome == "passed":
        record["passes"] += 1
    elif outcome == "failed":
        record["failures"] += 1
        record["last_failure"] = now

    # Detect flake: if outcome differs from last outcome
    if record["last_outcome"] is not None and record["last_outcome"] != outcome:
        record["flake_count"] += 1

    # Update metadata
    record["last_run"] = now
    record["last_outcome"] = outcome

    # Update history (keep last 20)
    record["history"].append({"timestamp": now, "outcome": outcome, "duration": duration})
    if len(record["history"]) > 20:
        record["history"] = record["history"][-20:]

    # Calculate flake rate
    if record["total_runs"] >= ledger["config"]["min_runs_for_analysis"]:
        record["flake_rate"] = record["flake_count"] / record["total_runs"]

        # Update status
        if record["flake_rate"] >= ledger["config"]["quarantine_threshold"]:
            record["status"] = "quarantined"
        elif record["flake_rate"] >= ledger["config"]["flake_threshold"]:
            record["status"] = "monitored"
        else:
            record["status"] = "stable"


def process_report(report_path: Path, ledger: dict[str, Any]) -> dict[str, Any]:
    """Process pytest JSON report and update ledger."""
    report = parse_pytest_json(report_path)

    # Parse test results
    for test in report.get("tests", []):
        test_id = test["nodeid"]
        outcome = test["outcome"]  # passed, failed, skipped
        duration = test.get("call", {}).get("duration", 0.0)

        # Only track passed/failed (skip skipped tests)
        if outcome in ("passed", "failed"):
            update_test_record(ledger, test_id, outcome, duration)

    return ledger


def print_summary(ledger: dict[str, Any]) -> None:
    """Print flake rate summary."""
    tests = ledger["tests"]
    config = ledger["config"]

    total_tests = len(tests)
    stable = sum(1 for t in tests.values() if t["status"] == "stable")
    monitored = sum(1 for t in tests.values() if t["status"] == "monitored")
    quarantined = sum(1 for t in tests.values() if t["status"] == "quarantined")

    print("\n" + "=" * 60)
    print("FLAKE RATE SUMMARY")
    print("=" * 60)
    print(f"Total tests tracked: {total_tests}")
    print(f"  Stable:        {stable} ({stable/total_tests*100 if total_tests else 0:.1f}%)")
    print(f"  Monitored:     {monitored} ({monitored/total_tests*100 if total_tests else 0:.1f}%)")
    print(f"  Quarantined:   {quarantined} ({quarantined/total_tests*100 if total_tests else 0:.1f}%)")
    print()
    print(f"Flake threshold:       {config['flake_threshold']*100:.1f}%")
    print(f"Quarantine threshold:  {config['quarantine_threshold']*100:.1f}%")
    print()

    if monitored > 0 or quarantined > 0:
        print("⚠️  FLAKY TESTS DETECTED:")
        print()
        for test_id, record in sorted(tests.items()):
            if record["status"] in ("monitored", "quarantined"):
                status_icon = "🔴" if record["status"] == "quarantined" else "🟡"
                print(f"{status_icon} {test_id}")
                print(
                    f"   Flake rate: {record['flake_rate']*100:.2f}% "
                    f"({record['flake_count']}/{record['total_runs']} runs)"
                )
                print(f"   Last outcome: {record['last_outcome']}")
                print()

    if quarantined == 0 and monitored == 0:
        print("✅ No flaky tests detected!")

    print("=" * 60)


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Track test flake rates from pytest JSON reports")
    parser.add_argument(
        "report",
        type=Path,
        help="Path to pytest JSON report file",
    )
    parser.add_argument(
        "--ledger",
        type=Path,
        default=LEDGER_PATH,
        help=f"Path to flake ledger (default: {LEDGER_PATH})",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress summary output",
    )

    args = parser.parse_args()

    if not args.report.exists():
        print(f"❌ Error: Report file not found: {args.report}", file=sys.stderr)
        return 1

    # Load ledger
    ledger = load_ledger()

    # Process report
    ledger = process_report(args.report, ledger)

    # Save ledger
    save_ledger(ledger)

    # Print summary
    if not args.quiet:
        print_summary(ledger)

    # Exit with error if quarantined tests exist
    quarantined = sum(1 for t in ledger["tests"].values() if t["status"] == "quarantined")
    if quarantined > 0:
        print(
            f"\n⚠️  WARNING: {quarantined} test(s) exceed quarantine threshold",
            file=sys.stderr,
        )
        # Don't fail CI yet - just warn
        # return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
