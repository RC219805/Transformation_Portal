#!/usr/bin/env python3
"""
Phase 2 RAG System - Trend Analysis Script
==========================================

Standalone script for analyzing quality trends from Knowledge Engine data.
Can be run locally or as part of CI workflows.

Usage:
    python run_trend_analysis.py --days 30
    python run_trend_analysis.py --output trend_report.json --markdown
"""

import argparse
import json
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

# Constants for display and retention limits
MAX_TEST_ID_LENGTH = 50
MAX_FLAKY_TESTS_DISPLAY = 10
MAX_METRICS_SAMPLE_SIZE = 10


def load_knowledge_base(cache_dir: Path) -> Dict:
    """Load knowledge base from cache directory."""
    knowledge = {
        "test_results": [],
        "quality_metrics": []
    }
    
    knowledge_dir = cache_dir / "knowledge"
    
    if (knowledge_dir / "ci_test_results.json").exists():
        with open(knowledge_dir / "ci_test_results.json") as f:
            knowledge["test_results"] = json.load(f)
    
    if (knowledge_dir / "quality_metrics.json").exists():
        with open(knowledge_dir / "quality_metrics.json") as f:
            knowledge["quality_metrics"] = json.load(f)
    
    return knowledge


def analyze_pass_rate_trend(metrics: List[Dict]) -> Optional[Dict]:
    """Analyze pass rate trend from metrics."""
    pass_rate_metrics = [
        m for m in metrics 
        if m.get("metric_id") == "test_pass_rate"
    ]
    
    if len(pass_rate_metrics) < 3:
        return None
    
    values = [m.get("value", 0) for m in pass_rate_metrics[-10:]]
    
    if len(values) < 2:
        return None
    
    first_half = values[:len(values)//2]
    second_half = values[len(values)//2:]
    
    avg_first = statistics.mean(first_half) if first_half else 0
    avg_second = statistics.mean(second_half) if second_half else 0
    trend = avg_second - avg_first
    
    return {
        "current": values[-1],
        "average": statistics.mean(values),
        "trend": round(trend, 2),
        "direction": "improving" if trend > 0 else "declining" if trend < 0 else "stable",
        "samples": len(values)
    }


def analyze_execution_time_trend(metrics: List[Dict]) -> Optional[Dict]:
    """Analyze execution time trend from metrics."""
    duration_metrics = [
        m for m in metrics 
        if m.get("metric_id") == "test_execution_time"
    ]
    
    if len(duration_metrics) < 3:
        return None
    
    values = [m.get("value", 0) for m in duration_metrics[-10:]]
    
    if len(values) < 2:
        return None
    
    avg = statistics.mean(values)
    first_val = values[0]
    last_val = values[-1]
    change_pct = ((last_val - first_val) / first_val * 100) if first_val > 0 else 0
    
    return {
        "current_seconds": round(last_val, 2),
        "average_seconds": round(avg, 2),
        "change_percent": round(change_pct, 1),
        "direction": "slower" if change_pct > 5 else "faster" if change_pct < -5 else "stable"
    }


def detect_flaky_tests(test_results: List[Dict]) -> List[Dict]:
    """Detect flaky tests from test results."""
    test_outcomes = defaultdict(list)
    
    for result in test_results:
        test_id = result.get("test_id", "")
        status = result.get("status", "")
        if test_id and status:
            test_outcomes[test_id].append(status)
    
    flaky_tests = []
    
    for test_id, outcomes in test_outcomes.items():
        if len(outcomes) >= 3:
            unique_outcomes = set(outcomes)
            if "passed" in unique_outcomes and "failed" in unique_outcomes:
                pass_count = outcomes.count("passed")
                fail_count = outcomes.count("failed")
                total = pass_count + fail_count
                
                if 0.2 < (fail_count / total) < 0.8:
                    flaky_tests.append({
                        "test_id": test_id,
                        "pass_count": pass_count,
                        "fail_count": fail_count,
                        "flakiness_score": round(min(pass_count, fail_count) / total, 2)
                    })
    
    flaky_tests.sort(key=lambda x: x["flakiness_score"], reverse=True)
    return flaky_tests[:10]


def detect_regressions(
    pass_rate_trend: Optional[Dict],
    execution_time_trend: Optional[Dict]
) -> List[Dict]:
    """Detect quality regressions."""
    regressions = []
    
    if pass_rate_trend and pass_rate_trend["trend"] < -5:
        regressions.append({
            "type": "pass_rate_decline",
            "severity": "high" if pass_rate_trend["trend"] < -10 else "medium",
            "change": pass_rate_trend["trend"],
            "message": f"Pass rate declined by {abs(pass_rate_trend['trend']):.1f}% over analysis period"
        })
    
    if execution_time_trend and execution_time_trend["change_percent"] > 20:
        regressions.append({
            "type": "performance_regression",
            "severity": "high" if execution_time_trend["change_percent"] > 50 else "medium",
            "change_percent": execution_time_trend["change_percent"],
            "message": f"Test execution time increased by {execution_time_trend['change_percent']:.1f}%"
        })
    
    return regressions


def generate_insights(
    pass_rate_trend: Optional[Dict],
    flaky_tests: List[Dict],
    test_results_count: int
) -> List[Dict]:
    """Generate insights from analysis."""
    insights = []
    
    if pass_rate_trend and pass_rate_trend["direction"] == "improving":
        insights.append({
            "type": "positive",
            "message": "Test pass rate is trending upward - quality improvements are working"
        })
    
    if flaky_tests:
        insights.append({
            "type": "warning",
            "message": f"Identified {len(flaky_tests)} flaky tests requiring attention"
        })
    
    if test_results_count < 100:
        insights.append({
            "type": "info",
            "message": "Limited historical data - trends will become more reliable over time"
        })
    
    return insights


def generate_recommendations(
    regressions: List[Dict],
    flaky_tests: List[Dict],
    execution_time_trend: Optional[Dict]
) -> List[str]:
    """Generate recommendations based on analysis."""
    recommendations = []
    
    if regressions:
        recommendations.append(
            "Investigate recent changes that may have caused quality regressions"
        )
    
    if flaky_tests:
        recommendations.append(
            "Prioritize stabilizing flaky tests to improve CI reliability"
        )
    
    if execution_time_trend and execution_time_trend["direction"] == "slower":
        recommendations.append(
            "Review test parallelization and consider test splitting strategies"
        )
    
    return recommendations


def generate_markdown_report(report: Dict) -> str:
    """Generate markdown report from analysis."""
    lines = [
        "# 📊 Quality Trend Dashboard",
        "",
        f"**Generated:** {report['generated_at']}",
        f"**Analysis Period:** Last {report['analysis_period_days']} days",
        "",
        "---",
        "",
        "## 📈 Quality Trends",
        ""
    ]
    
    # Pass rate trend
    if "pass_rate" in report.get("trends", {}):
        pr = report["trends"]["pass_rate"]
        emoji = "✅" if pr["direction"] == "improving" else "⚠️" if pr["direction"] == "declining" else "➡️"
        lines.extend([
            f"### Test Pass Rate {emoji}",
            f"- **Current:** {pr['current']:.1f}%",
            f"- **Average:** {pr['average']:.1f}%",
            f"- **Trend:** {pr['direction']} ({pr['trend']:+.1f}%)",
            ""
        ])
    
    # Execution time trend
    if "execution_time" in report.get("trends", {}):
        et = report["trends"]["execution_time"]
        emoji = "⚡" if et["direction"] == "faster" else "🐌" if et["direction"] == "slower" else "➡️"
        lines.extend([
            f"### Execution Time {emoji}",
            f"- **Current:** {et['current_seconds']:.1f}s",
            f"- **Average:** {et['average_seconds']:.1f}s",
            f"- **Change:** {et['change_percent']:+.1f}%",
            ""
        ])
    
    # Regressions
    if report.get("regressions"):
        lines.extend([
            "## 🚨 Detected Regressions",
            ""
        ])
        for reg in report["regressions"]:
            severity_emoji = "🔴" if reg["severity"] == "high" else "🟡"
            lines.append(f"- {severity_emoji} **{reg['type']}**: {reg['message']}")
        lines.append("")
    
    # Flaky tests
    if report.get("flaky_tests"):
        lines.extend([
            "## 🎲 Flaky Tests",
            "",
            "| Test | Pass | Fail | Flakiness |",
            "|------|------|------|-----------|"
        ])
        for ft in report["flaky_tests"][:5]:
            lines.append(
                f"| `{ft['test_id'][:MAX_TEST_ID_LENGTH]}` | {ft['pass_count']} | {ft['fail_count']} | {ft['flakiness_score']:.0%} |"
            )
        lines.append("")
    
    # Insights
    if report.get("insights"):
        lines.extend([
            "## 💡 Insights",
            ""
        ])
        for insight in report["insights"]:
            emoji = "✅" if insight["type"] == "positive" else "⚠️" if insight["type"] == "warning" else "ℹ️"
            lines.append(f"- {emoji} {insight['message']}")
        lines.append("")
    
    # Recommendations
    if report.get("recommendations"):
        lines.extend([
            "## 📋 Recommendations",
            ""
        ])
        for i, rec in enumerate(report["recommendations"], 1):
            lines.append(f"{i}. {rec}")
        lines.append("")
    
    lines.extend([
        "---",
        "*Generated by Phase 2 RAG System Knowledge Engine*"
    ])
    
    return "\n".join(lines)


def run_analysis(
    cache_dir: Path,
    analysis_days: int
) -> Dict[str, Any]:
    """Run full trend analysis."""
    knowledge = load_knowledge_base(cache_dir)
    
    test_results = knowledge["test_results"]
    quality_metrics = knowledge["quality_metrics"]
    
    # Analyze trends
    pass_rate_trend = analyze_pass_rate_trend(quality_metrics)
    execution_time_trend = analyze_execution_time_trend(quality_metrics)
    flaky_tests = detect_flaky_tests(test_results)
    regressions = detect_regressions(pass_rate_trend, execution_time_trend)
    insights = generate_insights(pass_rate_trend, flaky_tests, len(test_results))
    recommendations = generate_recommendations(regressions, flaky_tests, execution_time_trend)
    
    # Build report
    report = {
        "generated_at": datetime.now().isoformat(),
        "analysis_period_days": analysis_days,
        "total_test_results": len(test_results),
        "total_metrics": len(quality_metrics),
        "trends": {},
        "regressions": regressions,
        "flaky_tests": flaky_tests,
        "insights": insights,
        "recommendations": recommendations
    }
    
    if pass_rate_trend:
        report["trends"]["pass_rate"] = pass_rate_trend
    
    if execution_time_trend:
        report["trends"]["execution_time"] = execution_time_trend
    
    return report


def main():
    parser = argparse.ArgumentParser(
        description="Run quality trend analysis"
    )
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Analysis period in days"
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(".rag_cache"),
        help="RAG cache directory"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output file for JSON report"
    )
    parser.add_argument(
        "--markdown",
        action="store_true",
        help="Also generate markdown report"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        print(f"Analyzing trends for the last {args.days} days", file=sys.stderr)
        print(f"Cache directory: {args.cache_dir}", file=sys.stderr)
    
    # Run analysis
    report = run_analysis(args.cache_dir, args.days)
    
    # Output JSON
    if args.output:
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Report written to {args.output}", file=sys.stderr)
    else:
        print(json.dumps(report, indent=2))
    
    # Generate markdown if requested
    if args.markdown:
        md_report = generate_markdown_report(report)
        md_path = args.output.with_suffix(".md") if args.output else Path("TREND_DASHBOARD.md")
        with open(md_path, "w") as f:
            f.write(md_report)
        print(f"Markdown report written to {md_path}", file=sys.stderr)
    
    # Exit with error if regressions detected
    if report["regressions"]:
        print(f"Warning: {len(report['regressions'])} regression(s) detected", file=sys.stderr)
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
