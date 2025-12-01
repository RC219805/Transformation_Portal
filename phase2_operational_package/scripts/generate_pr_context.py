#!/usr/bin/env python3
"""
Phase 2 RAG System - PR Context Generation Script
==================================================

Standalone script for generating PR context from Knowledge Engine data.
Can be run locally or as part of CI workflows.

Usage:
    python generate_pr_context.py --changed-files file1.py file2.py
    python generate_pr_context.py --pr-number 123
    python generate_pr_context.py --auto  # Auto-detect from git diff
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set

# Constants for display limits
MAX_ERROR_MESSAGE_LENGTH = 100
MAX_AFFECTED_TESTS_DISPLAY = 10
MAX_FAILURE_HISTORY_DISPLAY = 5


def get_changed_files_from_git() -> List[str]:
    """Get changed files from git diff against main branch."""
    try:
        result = subprocess.run(
            ["git", "diff", "--name-only", "origin/main...HEAD"],
            capture_output=True,
            text=True,
            check=True
        )
        files = [f.strip() for f in result.stdout.strip().split("\n") if f.strip()]
        return [f for f in files if f.endswith(".py")]
    except subprocess.CalledProcessError:
        return []


def load_knowledge_base(cache_dir: Path) -> Dict:
    """Load knowledge base from cache directory."""
    knowledge = {
        "test_results": [],
        "quality_metrics": [],
        "detected_patterns": {}
    }
    
    knowledge_dir = cache_dir / "knowledge"
    
    if (knowledge_dir / "ci_test_results.json").exists():
        with open(knowledge_dir / "ci_test_results.json") as f:
            knowledge["test_results"] = json.load(f)
    
    if (knowledge_dir / "quality_metrics.json").exists():
        with open(knowledge_dir / "quality_metrics.json") as f:
            knowledge["quality_metrics"] = json.load(f)
    
    if (knowledge_dir / "detected_patterns.json").exists():
        with open(knowledge_dir / "detected_patterns.json") as f:
            knowledge["detected_patterns"] = json.load(f)
    
    return knowledge


def analyze_changed_files(
    changed_files: List[str],
    knowledge: Dict
) -> Dict:
    """Analyze changed files against knowledge base."""
    analysis = {
        "affected_tests": set(),
        "failure_history": {},
        "quality_trend": None,
        "recommendations": []
    }
    
    test_results = knowledge.get("test_results", [])
    quality_metrics = knowledge.get("quality_metrics", [])
    
    # Map changed files to tests
    for changed_file in changed_files:
        file_stem = Path(changed_file).stem
        
        for result in test_results:
            test_file = result.get("test_file", "")
            if file_stem in test_file or f"test_{file_stem}" in test_file:
                test_id = result.get("test_id", "")
                analysis["affected_tests"].add(test_id)
                
                if result.get("status") == "failed":
                    if test_id not in analysis["failure_history"]:
                        analysis["failure_history"][test_id] = []
                    analysis["failure_history"][test_id].append({
                        "timestamp": result.get("timestamp", ""),
                        "message": result.get("error_message", "")
                    })
    
    # Analyze quality trends
    pass_rate_metrics = [
        m for m in quality_metrics 
        if m.get("metric_id") == "test_pass_rate"
    ][-5:]
    
    if len(pass_rate_metrics) >= 2:
        first_rate = pass_rate_metrics[0].get("value", 0)
        last_rate = pass_rate_metrics[-1].get("value", 0)
        trend = last_rate - first_rate
        
        analysis["quality_trend"] = {
            "first": first_rate,
            "last": last_rate,
            "change": trend,
            "direction": "improving" if trend > 0 else "declining" if trend < 0 else "stable"
        }
    
    # Generate recommendations
    if analysis["failure_history"]:
        analysis["recommendations"].append(
            "Pay extra attention to tests with historical failures"
        )
    
    if analysis["quality_trend"] and analysis["quality_trend"]["direction"] == "declining":
        analysis["recommendations"].append(
            "Quality trend is declining - consider adding more tests"
        )
    
    return analysis


def generate_context_markdown(
    changed_files: List[str],
    analysis: Dict
) -> str:
    """Generate markdown context for PR."""
    lines = [
        "## 🔍 PR Context (Knowledge Engine)",
        "",
        "*Automated analysis based on historical CI data*",
        "",
    ]
    
    # Test impact section
    if analysis["affected_tests"]:
        lines.append("### 🧪 Test Impact Analysis")
        lines.append("")
        lines.append(f"**{len(analysis['affected_tests'])} tests** are historically associated with the changed files:")
        lines.append("")
        
        for test_id in list(analysis["affected_tests"])[:10]:
            lines.append(f"- `{test_id}`")
        
        if len(analysis["affected_tests"]) > 10:
            lines.append(f"- ... and {len(analysis['affected_tests']) - 10} more")
        
        lines.append("")
    
    # Failure warnings
    if analysis["failure_history"]:
        lines.append("### ⚠️ Historical Failure Patterns")
        lines.append("")
        lines.append("The following tests have failed previously when similar files were changed:")
        lines.append("")
        
        for test_id, failures in list(analysis["failure_history"].items())[:MAX_FAILURE_HISTORY_DISPLAY]:
            lines.append(f"- **`{test_id}`**: {len(failures)} historical failure(s)")
            if failures and failures[0].get("message"):
                msg = failures[0]["message"][:MAX_ERROR_MESSAGE_LENGTH]
                lines.append(f"  - Last failure: `{msg}...`")
        
        lines.append("")
    
    # Quality trends
    if analysis["quality_trend"]:
        trend = analysis["quality_trend"]
        direction = "📈 improving" if trend["direction"] == "improving" else \
                   "📉 declining" if trend["direction"] == "declining" else "➡️ stable"
        
        lines.append("### 📊 Quality Trends")
        lines.append("")
        lines.append(f"Test pass rate is {direction}: {trend['first']:.1f}% → {trend['last']:.1f}%")
        lines.append("")
    
    # Recommendations
    if analysis["recommendations"]:
        lines.append("### 💡 Recommendations")
        lines.append("")
        for rec in analysis["recommendations"]:
            lines.append(f"- {rec}")
        lines.append("")
    
    # No data fallback
    if not (analysis["affected_tests"] or analysis["failure_history"] or analysis["quality_trend"]):
        lines.append("No specific historical patterns found for the changed files.")
        lines.append("This may be new code or code with limited test history.")
        lines.append("")
    
    lines.extend([
        "---",
        "*Generated by Phase 2 RAG System Knowledge Engine*"
    ])
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Generate PR context from Knowledge Engine"
    )
    parser.add_argument(
        "--changed-files",
        nargs="*",
        help="List of changed files"
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Auto-detect changed files from git"
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
        help="Output file for markdown (default: stdout)"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output analysis as JSON instead of markdown"
    )
    
    args = parser.parse_args()
    
    # Get changed files
    if args.auto:
        changed_files = get_changed_files_from_git()
        print(f"Auto-detected {len(changed_files)} changed Python files", file=sys.stderr)
    elif args.changed_files:
        changed_files = args.changed_files
    else:
        parser.error("Either --changed-files or --auto is required")
    
    if not changed_files:
        print("No Python files to analyze", file=sys.stderr)
        sys.exit(0)
    
    # Load knowledge base
    knowledge = load_knowledge_base(args.cache_dir)
    
    # Analyze
    analysis = analyze_changed_files(changed_files, knowledge)
    
    # Convert sets to lists for JSON serialization
    analysis["affected_tests"] = list(analysis["affected_tests"])
    
    # Output
    if args.json:
        output = json.dumps(analysis, indent=2, default=str)
    else:
        output = generate_context_markdown(changed_files, analysis)
    
    if args.output:
        args.output.write_text(output)
        print(f"Output written to {args.output}", file=sys.stderr)
    else:
        print(output)


if __name__ == "__main__":
    main()
