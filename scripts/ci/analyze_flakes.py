#!/usr/bin/env python3
"""
Analyze flake ledger and generate reports.

Usage:
    .venv/bin/python scripts/analyze_flakes.py [--format=text|json|markdown]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
LEDGER_PATH = REPO_ROOT / "tests" / "flake_ledger.json"


def load_ledger() -> dict[str, Any]:
    """Load the flake tracking ledger."""
    if not LEDGER_PATH.exists():
        return {"tests": {}, "config": {}}

    with open(LEDGER_PATH) as f:
        return json.load(f)


def analyze_ledger(ledger: dict[str, Any]) -> dict[str, Any]:
    """Analyze ledger and compute statistics."""
    tests = ledger.get("tests", {})
    config = ledger.get("config", {})

    total = len(tests)
    stable = sum(1 for t in tests.values() if t["status"] == "stable")
    monitored = sum(1 for t in tests.values() if t["status"] == "monitored")
    quarantined = sum(1 for t in tests.values() if t["status"] == "quarantined")

    # Repo-wide flake rate
    total_runs = sum(t["total_runs"] for t in tests.values())
    total_flakes = sum(t["flake_count"] for t in tests.values())
    repo_flake_rate = total_flakes / total_runs if total_runs > 0 else 0.0

    # Worst offenders
    flaky_tests = [t for t in tests.values() if t["status"] in ("monitored", "quarantined")]
    flaky_tests.sort(key=lambda t: t["flake_rate"], reverse=True)

    return {
        "total_tests": total,
        "stable": stable,
        "monitored": monitored,
        "quarantined": quarantined,
        "repo_flake_rate": repo_flake_rate,
        "total_runs": total_runs,
        "total_flakes": total_flakes,
        "flake_threshold": config.get("flake_threshold", 0.01),
        "quarantine_threshold": config.get("quarantine_threshold", 0.03),
        "flaky_tests": flaky_tests[:10],  # Top 10
    }


def format_text(stats: dict[str, Any]) -> str:
    """Format as plain text."""
    lines = []
    total_tests = stats["total_tests"]
    stable_pct = stats["stable"] / total_tests * 100 if total_tests else 0
    monitored_pct = stats["monitored"] / total_tests * 100 if total_tests else 0
    quarantined_pct = stats["quarantined"] / total_tests * 100 if total_tests else 0
    lines.append("=" * 60)
    lines.append("FLAKE ANALYSIS REPORT")
    lines.append("=" * 60)
    lines.append(f"Total tests tracked: {total_tests}")
    lines.append(f"  Stable:        {stats['stable']} ({stable_pct:.1f}%)")
    lines.append(f"  Monitored:     {stats['monitored']} ({monitored_pct:.1f}%)")
    lines.append(f"  Quarantined:   {stats['quarantined']} ({quarantined_pct:.1f}%)")
    lines.append("")
    lines.append(f"Repository-wide flake rate: {stats['repo_flake_rate']*100:.2f}%")
    lines.append(f"  ({stats['total_flakes']} flakes / {stats['total_runs']} runs)")
    lines.append("")
    lines.append(f"Thresholds:")
    lines.append(f"  Flake:       {stats['flake_threshold']*100:.1f}%")
    lines.append(f"  Quarantine:  {stats['quarantine_threshold']*100:.1f}%")
    lines.append("")

    if stats["flaky_tests"]:
        lines.append("TOP FLAKY TESTS:")
        lines.append("")
        for i, test in enumerate(stats["flaky_tests"], 1):
            status_icon = "🔴" if test["status"] == "quarantined" else "🟡"
            lines.append(f"{i}. {status_icon} {test['test_id']}")
            lines.append(
                f"   Flake rate: {test['flake_rate']*100:.2f}% "
                f"({test['flake_count']}/{test['total_runs']} runs)"
            )
            lines.append(f"   Last: {test['last_outcome']}")
            lines.append("")
    else:
        lines.append("✅ No flaky tests detected!")
        lines.append("")

    lines.append("=" * 60)
    return "\n".join(lines)


def format_markdown(stats: dict[str, Any]) -> str:
    """Format as GitHub-flavored markdown."""
    lines = []
    total_tests = stats["total_tests"]
    stable_pct = stats["stable"] / total_tests * 100 if total_tests else 0
    monitored_pct = stats["monitored"] / total_tests * 100 if total_tests else 0
    quarantined_pct = stats["quarantined"] / total_tests * 100 if total_tests else 0
    lines.append("## 📊 Flake Analysis Report")
    lines.append("")
    lines.append(f"**Total tests tracked:** {total_tests}")
    lines.append("")
    lines.append("| Status | Count | Percentage |")
    lines.append("|--------|-------|------------|")
    lines.append(f"| ✅ Stable | {stats['stable']} | {stable_pct:.1f}% |")
    lines.append(f"| 🟡 Monitored | {stats['monitored']} | {monitored_pct:.1f}% |")
    lines.append(f"| 🔴 Quarantined | {stats['quarantined']} | {quarantined_pct:.1f}% |")
    lines.append("")
    lines.append(
        f"**Repository-wide flake rate:** {stats['repo_flake_rate']*100:.2f}% "
        f"({stats['total_flakes']} flakes / {stats['total_runs']} runs)"
    )
    lines.append("")
    lines.append(
        f"**Thresholds:** Flake = {stats['flake_threshold']*100:.1f}%, "
        f"Quarantine = {stats['quarantine_threshold']*100:.1f}%"
    )
    lines.append("")

    if stats["flaky_tests"]:
        lines.append("### ⚠️ Top Flaky Tests")
        lines.append("")
        lines.append("| Test | Status | Flake Rate | Runs |")
        lines.append("|------|--------|------------|------|")
        for test in stats["flaky_tests"]:
            status_icon = "🔴" if test["status"] == "quarantined" else "🟡"
            test_name = test["test_id"].split("::")[-1] if "::" in test["test_id"] else test["test_id"]
            lines.append(
                f"| `{test_name}` | {status_icon} | {test['flake_rate']*100:.2f}% | "
                f"{test['flake_count']}/{test['total_runs']} |"
            )
        lines.append("")
    else:
        lines.append("✅ **No flaky tests detected!**")
        lines.append("")

    return "\n".join(lines)


def format_json(stats: dict[str, Any]) -> str:
    """Format as JSON."""
    return json.dumps(stats, indent=2)


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Analyze flake ledger")
    parser.add_argument(
        "--format",
        choices=["text", "json", "markdown"],
        default="text",
        help="Output format",
    )
    parser.add_argument(
        "--ledger",
        type=Path,
        default=LEDGER_PATH,
        help=f"Path to flake ledger (default: {LEDGER_PATH})",
    )

    args = parser.parse_args()

    ledger = load_ledger()
    stats = analyze_ledger(ledger)

    if args.format == "text":
        print(format_text(stats))
    elif args.format == "markdown":
        print(format_markdown(stats))
    elif args.format == "json":
        print(format_json(stats))

    return 0


if __name__ == "__main__":
    sys.exit(main())
