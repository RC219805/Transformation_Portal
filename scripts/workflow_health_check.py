#!/usr/bin/env python3
"""
Workflow Health Check - Monitor CI/CD pipeline status
=======================================================

Programmatically checks the health of GitHub Actions workflows and provides
actionable insights for maintaining CI/CD reliability.

Features:
- Recent workflow run status analysis
- Failure rate calculations per workflow
- Identification of flaky workflows
- Performance metrics (duration trends)
- Actionable recommendations

Usage:
    python scripts/workflow_health_check.py
    python scripts/workflow_health_check.py --json
    python scripts/workflow_health_check.py --workflow ci-consolidated.yml
"""

import json
import subprocess
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict


@dataclass
class WorkflowRun:
    """Represents a single workflow run."""

    status: str
    conclusion: Optional[str]
    name: str
    workflow: str
    branch: str
    event: str
    run_id: str
    duration: str
    created_at: str


@dataclass
class WorkflowHealth:
    """Health metrics for a workflow."""

    workflow_name: str
    total_runs: int
    success_count: int
    failure_count: int
    cancelled_count: int
    success_rate: float
    avg_duration_seconds: float
    is_flaky: bool
    last_failure: Optional[str]
    recommendations: List[str]


def run_gh_command(cmd: List[str]) -> str:
    """Run a GitHub CLI command and return output."""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {' '.join(cmd)}", file=sys.stderr)
        print(f"Error: {e.stderr}", file=sys.stderr)
        sys.exit(1)


def parse_duration(duration_str: str) -> int:
    """Parse duration string (e.g., '10m54s') to seconds."""
    try:
        parts = duration_str.replace("m", " ").replace("s", "").split()
        if len(parts) == 2:
            return int(parts[0]) * 60 + int(parts[1])
        elif len(parts) == 1 and "s" in duration_str:
            return int(parts[0])
        elif len(parts) == 1 and "m" in duration_str:
            return int(parts[0]) * 60
        return 0
    except (ValueError, IndexError):
        return 0


def get_workflow_runs(workflow: Optional[str] = None, limit: int = 50) -> List[WorkflowRun]:
    """Fetch recent workflow runs."""
    cmd = ["gh", "run", "list", "--limit", str(limit)]
    if workflow:
        cmd.extend(["--workflow", workflow])

    output = run_gh_command(cmd)
    runs = []

    for line in output.split("\n"):
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) >= 8:
            runs.append(
                WorkflowRun(
                    status=parts[0],
                    conclusion=parts[1] if parts[1] else None,
                    name=parts[2],
                    workflow=parts[3],
                    branch=parts[4],
                    event=parts[5],
                    run_id=parts[6],
                    duration=parts[7],
                    created_at=parts[8] if len(parts) > 8 else "",
                )
            )

    return runs


def analyze_workflow_health(runs: List[WorkflowRun]) -> Dict[str, WorkflowHealth]:
    """Analyze workflow health from runs."""
    workflows = {}

    for run in runs:
        if run.workflow not in workflows:
            workflows[run.workflow] = {
                "runs": [],
                "success": 0,
                "failure": 0,
                "cancelled": 0,
                "durations": [],
            }

        wf = workflows[run.workflow]
        wf["runs"].append(run)

        if run.status == "completed":
            if run.conclusion == "success":
                wf["success"] += 1
            elif run.conclusion == "failure":
                wf["failure"] += 1
            elif run.conclusion == "cancelled":
                wf["cancelled"] += 1

        duration = parse_duration(run.duration)
        if duration > 0:
            wf["durations"].append(duration)

    # Calculate health metrics
    health_report = {}
    for workflow_name, data in workflows.items():
        total = len(data["runs"])
        success_rate = (data["success"] / total * 100) if total > 0 else 0
        avg_duration = sum(data["durations"]) / len(data["durations"]) if data["durations"] else 0

        # Determine if workflow is flaky (success rate < 90%)
        is_flaky = success_rate < 90 and total >= 5

        # Find last failure
        last_failure = None
        for run in data["runs"]:
            if run.conclusion == "failure":
                last_failure = run.created_at
                break

        # Generate recommendations
        recommendations = []
        if is_flaky:
            recommendations.append("⚠️ FLAKY: Success rate below 90% - investigate root cause")
        if data["failure"] > 0:
            recommendations.append(f"💡 Review {data['failure']} recent failure(s)")
        if avg_duration > 600:  # > 10 minutes
            recommendations.append("🐢 Consider optimization - avg duration > 10 minutes")
        if data["cancelled"] > data["failure"]:
            recommendations.append("🔄 High cancellation rate - check concurrency settings")

        health_report[workflow_name] = WorkflowHealth(
            workflow_name=workflow_name,
            total_runs=total,
            success_count=data["success"],
            failure_count=data["failure"],
            cancelled_count=data["cancelled"],
            success_rate=round(success_rate, 2),
            avg_duration_seconds=round(avg_duration, 2),
            is_flaky=is_flaky,
            last_failure=last_failure,
            recommendations=recommendations,
        )

    return health_report


def print_health_report(health: Dict[str, WorkflowHealth], json_output: bool = False):
    """Print health report in human-readable or JSON format."""
    if json_output:
        data = {name: asdict(h) for name, h in health.items()}
        print(json.dumps(data, indent=2))
        return

    print("\n" + "=" * 80)
    print("GitHub Actions Workflow Health Report")
    print("=" * 80 + "\n")

    # Sort by success rate (worst first)
    sorted_workflows = sorted(health.items(), key=lambda x: x[1].success_rate)

    for workflow_name, h in sorted_workflows:
        status_icon = "✅" if h.success_rate >= 90 else "⚠️" if h.success_rate >= 70 else "❌"

        print(f"{status_icon} {workflow_name}")
        print(
            f"   Runs: {h.total_runs} | Success: {h.success_count} | Failure: {h.failure_count} | Cancelled: {h.cancelled_count}"
        )
        print(f"   Success Rate: {h.success_rate}% | Avg Duration: {h.avg_duration_seconds}s")

        if h.last_failure:
            print(f"   Last Failure: {h.last_failure}")

        if h.recommendations:
            print("   Recommendations:")
            for rec in h.recommendations:
                print(f"     • {rec}")

        print()

    # Summary statistics
    total_runs = sum(h.total_runs for h in health.values())
    total_success = sum(h.success_count for h in health.values())
    overall_success_rate = (total_success / total_runs * 100) if total_runs > 0 else 0

    print("=" * 80)
    print(f"Overall Statistics:")
    print(f"  Total Runs: {total_runs}")
    print(f"  Total Success: {total_success}")
    print(f"  Overall Success Rate: {overall_success_rate:.2f}%")
    print(f"  Workflows Analyzed: {len(health)}")
    print(f"  Flaky Workflows: {sum(1 for h in health.values() if h.is_flaky)}")
    print("=" * 80 + "\n")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Check GitHub Actions workflow health")
    parser.add_argument("--workflow", help="Specific workflow to analyze")
    parser.add_argument("--limit", type=int, default=50, help="Number of runs to analyze (default: 50)")
    parser.add_argument("--json", action="store_true", help="Output in JSON format")

    args = parser.parse_args()

    # Check if gh CLI is available
    try:
        subprocess.run(["gh", "--version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: GitHub CLI (gh) not found or not authenticated.", file=sys.stderr)
        print("Install: https://cli.github.com/", file=sys.stderr)
        print("Authenticate: gh auth login", file=sys.stderr)
        sys.exit(1)

    # Fetch and analyze runs
    runs = get_workflow_runs(workflow=args.workflow, limit=args.limit)

    if not runs:
        print("No workflow runs found.", file=sys.stderr)
        sys.exit(1)

    health = analyze_workflow_health(runs)
    print_health_report(health, json_output=args.json)

    # Exit with error code if any workflow is unhealthy
    unhealthy = [h for h in health.values() if h.is_flaky or h.success_rate < 80]
    if unhealthy:
        sys.exit(1)


if __name__ == "__main__":
    main()
