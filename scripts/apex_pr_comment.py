#!/usr/bin/env python3
"""APEX PR comment generator for human-readable performance summaries.

This script generates markdown comments for GitHub PRs with:
- V1 vs V2 comparison
- Per-zone performance heatmap
- Worst offenders list
- Gate verdict (pass/warn/fail)

Design:
- Markdown output (GitHub-compatible)
- Color-coded status indicators (✅ ⚠️ ❌)
- Concise but actionable
- Links to detailed artifacts

Usage:
    python scripts/apex_pr_comment.py \\
        --run-id abc123 \\
        --ledger-db ./apex_performance.db \\
        --output comment.md

Version: 1.0.0
"""

from __future__ import annotations

import argparse
import logging
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from transformation_portal.metrics.comparator import query_baseline_stats
from transformation_portal.metrics.contracts import BucketStats, Judgement
from transformation_portal.metrics.gate import evaluate_gate

__version__ = "1.0.0"

# GitHub comment size limit (leave buffer for safety)
MAX_GITHUB_COMMENT_SIZE = 65_000

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def format_time(seconds: float) -> str:
    """Format time in seconds to human-readable string."""
    if seconds < 1.0:
        return f"{seconds * 1000:.0f}ms"
    return f"{seconds:.2f}s"


def get_status_icon(pass_fail: str, is_insufficient_data: bool = False) -> str:
    """Get emoji icon for pass/fail status.

    Args:
        pass_fail: Verdict ("pass", "warn", "fail")
        is_insufficient_data: Whether sample count is below minimum

    Returns:
        Status icon emoji
    """
    if is_insufficient_data:
        return "📊"  # Data icon for insufficient samples
    elif pass_fail == "pass":
        return "✅"
    elif pass_fail == "warn":
        return "⚠️"
    else:
        return "❌"


def worst_status(stats: List[Dict[str, Any]]) -> str:
    """Return worst pass_fail status across all rows (excluding insufficient data).

    Ordering: fail > warn > pass (worst-of logic)
    Insufficient data buckets are excluded from verdict (never block)

    Args:
        stats: List of performance stats dicts

    Returns:
        Worst status string ("pass", "warn", or "fail")
    """
    if not stats:
        return "pass"

    order = {"pass": 0, "warn": 1, "fail": 2}
    inv = {0: "pass", 1: "warn", 2: "fail"}

    worst = 0
    for row in stats:
        # Skip insufficient data buckets per contract
        if row.get("is_insufficient_data", False):
            continue
        status = row.get("pass_fail", "pass")
        worst = max(worst, order.get(status, 0))

    return inv[worst]


def table_has_column(conn: sqlite3.Connection, table: str, column: str) -> bool:
    """Check if table has column (works across schema versions).

    Args:
        conn: SQLite connection
        table: Table name
        column: Column name

    Returns:
        True if column exists, False otherwise
    """
    try:
        cursor = conn.execute(f"PRAGMA table_info({table})")
        columns = [row[1] for row in cursor.fetchall()]
        return column in columns
    except sqlite3.Error:
        return False


def fetch_run_stats(
    db_path: str,
    run_id: str,
    workflow_version: str,
    commit_sha: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Fetch performance stats for a run, optionally filtered by commit SHA.

    Args:
        db_path: Path to ledger database
        run_id: Run identifier
        workflow_version: Workflow version ("v1" or "v2")
        commit_sha: Optional commit SHA to filter by

    Returns:
        List of performance stats dicts
    """
    query_base = """
        SELECT bucket_name, zone, p50, p95, p99, count,
               pass_fail, threshold_p95, workflow_version
        FROM apex_runs
        WHERE run_id = ? AND workflow_version = ?
    """
    params: List[Any] = [run_id, workflow_version]

    try:
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row

            # Only filter by commit_sha if column exists
            if commit_sha and table_has_column(conn, "apex_runs", "commit_sha"):
                query_base += " AND commit_sha = ?"
                params.append(commit_sha)

            query = query_base + " ORDER BY bucket_name, zone"
            cursor = conn.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]

    except sqlite3.Error as e:
        logger.error(f"Database error: {e}")
        return []


def generate_bucket_table(
    bucket_stats: Dict[str, BucketStats],
    title: str,
) -> str:
    """Generate markdown table for bucket statistics.

    Args:
        bucket_stats: Dict mapping bucket_name -> BucketStats
        title: Table title

    Returns:
        Markdown table string
    """
    if not bucket_stats:
        return f"### {title}\n\nNo data available.\n"

    lines = [
        f"### {title}",
        "",
        "| Bucket | Count | p50 | p95 | Threshold p95 | Status |",
        "|--------|-------|-----|-----|---------------|--------|",
    ]

    for bucket_name, stats in sorted(bucket_stats.items()):
        status = get_status_icon(stats.pass_fail, stats.is_insufficient_data)
        p50_str = format_time(stats.p50)
        p95_str = format_time(stats.p95)
        threshold_str = format_time(stats.threshold_p95)

        lines.append(f"| {bucket_name} | {stats.count} | {p50_str} | {p95_str} | {threshold_str} | {status} |")

    return "\n".join(lines)


def generate_zone_heatmap(
    per_zone_stats: Dict[str, Dict[str, BucketStats]],
    title: str,
) -> str:
    """Generate zone × bucket heatmap.

    Args:
        per_zone_stats: Dict mapping zone -> bucket_name -> BucketStats
        title: Heatmap title

    Returns:
        Markdown heatmap string
    """
    if not per_zone_stats:
        return f"### {title}\n\nNo multi-zone data available.\n"

    # Collect all unique buckets
    all_buckets = set()
    for zone_stats in per_zone_stats.values():
        all_buckets.update(zone_stats.keys())

    lines = [
        f"### {title}",
        "",
        "| Zone | " + " | ".join(sorted(all_buckets)) + " |",
        "|------|" + "|".join(["---"] * len(all_buckets)) + "|",
    ]

    for zone, zone_stats in sorted(per_zone_stats.items()):
        cells = [zone]
        for bucket_name in sorted(all_buckets):
            if bucket_name in zone_stats:
                stats = zone_stats[bucket_name]
                p95_str = format_time(stats.p95)
                status = get_status_icon(stats.pass_fail, stats.is_insufficient_data)
                cells.append(f"{p95_str} {status}")
            else:
                cells.append("—")

        lines.append("| " + " | ".join(cells) + " |")

    return "\n".join(lines)


def generate_zone_heatmap_from_stats(
    stats: List[Dict[str, Any]],
    title: str,
    max_zones: int = 8,
) -> str:
    """Render bucket × zone status icon matrix in Markdown table.

    Args:
        stats: List of performance stats dicts
        title: Section title
        max_zones: Maximum zones to show (truncate to keep PR compact)

    Returns:
        Markdown table with collapsed section
    """
    if not stats:
        return ""

    # Extract unique buckets and zones
    buckets = sorted({r["bucket_name"] for r in stats})
    zones = sorted({r["zone"] for r in stats if r["zone"] is not None})

    # Truncate zones to keep PR compact
    shown_zones = zones[:max_zones]
    hidden_count = max(0, len(zones) - len(shown_zones))

    # Index for O(1) lookup
    index: Dict[Tuple[str, Optional[str]], Dict[str, Any]] = {(r["bucket_name"], r["zone"]): r for r in stats}

    # Table header
    header = "| Bucket | " + " | ".join(shown_zones) + " | Global |"
    sep = "|" + "---|" * (len(shown_zones) + 2)

    lines = [header, sep]

    # Table rows (one per bucket)
    for bucket in buckets:
        row_cells = []

        # Per-zone cells
        for zone in shown_zones:
            row = index.get((bucket, zone))
            if row:
                icon = get_status_icon(row["pass_fail"], row.get("is_insufficient_data", False))
                row_cells.append(icon)
            else:
                row_cells.append("—")

        # Global cell
        global_row = index.get((bucket, None))
        global_cell = (
            get_status_icon(global_row["pass_fail"], global_row.get("is_insufficient_data", False)) if global_row else "—"
        )

        lines.append(f"| {bucket} | " + " | ".join(row_cells) + f" | {global_cell} |")

    if hidden_count:
        lines.append("")
        lines.append(f"_Note: {hidden_count} additional zones not shown " f"to keep PR comment compact._")

    # Wrap in collapsed section
    output = f"<details>\n<summary>{title}</summary>\n\n"
    output += "\n".join(lines)
    output += "\n\n</details>\n"

    return output


def generate_worst_offenders(
    stats: List[Dict[str, Any]],
    title: str,
    top_n: int = 10,
) -> str:
    """Generate top-N worst performers by p95/limit ratio.

    Prioritizes:
    1. warn/fail status
    2. p95 > limit (ratio > 1.0)
    3. Highest ratio first

    Args:
        stats: List of performance stats dicts
        title: Section title
        top_n: Number of offenders to show

    Returns:
        Markdown table with collapsed section
    """
    if not stats:
        return ""

    rows = []
    for r in stats:
        p95 = r.get("p95")
        limit = r.get("threshold_p95")

        # Skip if missing data
        if p95 is None or limit is None or limit <= 0:
            continue

        ratio = p95 / limit
        delta = p95 - limit

        # Only include warnings/fails or cases exceeding limit
        if r.get("pass_fail") in ("warn", "fail") or ratio > 1.0:
            rows.append((ratio, delta, r))

    if not rows:
        return ""

    # Sort by ratio (worst first), take top N
    rows.sort(key=lambda x: x[0], reverse=True)
    rows = rows[:top_n]

    lines = [
        "| Rank | Bucket | Zone | Workflow | p95 | Limit | Over | Status |",
        "|------|--------|------|----------|-----|-------|------|--------|",
    ]

    for i, (ratio, delta, r) in enumerate(rows, start=1):
        zone_str = r["zone"] or "Global"
        workflow = r.get("workflow_version", "?")
        p95_str = format_time(r["p95"])
        limit_str = format_time(r["threshold_p95"])
        delta_str = format_time(delta)
        icon = get_status_icon(r["pass_fail"], r.get("is_insufficient_data", False))

        lines.append(
            f"| {i} | {r['bucket_name']} | {zone_str} | {workflow} | "
            f"{p95_str} | {limit_str} | {delta_str} ({ratio:.2f}×) | {icon} |"
        )

    # Wrap in collapsed section
    output = f"<details>\n<summary>{title}</summary>\n\n"
    output += "\n".join(lines)
    output += "\n\n</details>\n"

    return output


def truncate_comment(lines: List[str], max_chars: int = MAX_GITHUB_COMMENT_SIZE) -> List[str]:
    """Truncate comment to fit GitHub's character limit.

    Keeps header and verdict, truncates details sections.

    Args:
        lines: Comment lines
        max_chars: Maximum characters allowed

    Returns:
        Truncated lines if needed
    """
    full_text = "\n".join(lines)

    if len(full_text) <= max_chars:
        return lines  # No truncation needed

    # Find first collapsed section
    truncate_index = next(
        (i for i, line in enumerate(lines) if line.strip().startswith("<details>")),
        len(lines),
    )

    # Keep header + verdict, add truncation notice
    truncated = lines[:truncate_index]
    truncated.append("\n---\n")
    truncated.append(
        f"⚠️ **Comment truncated** (exceeded {max_chars:,} characters). " f"See job summary or ledger DB for full details.\n"
    )

    return truncated


def generate_v1_v2_comparison(
    v1_stats: Dict[str, BucketStats],
    v2_stats: Dict[str, BucketStats],
) -> str:
    """Generate V1 vs V2 comparison table.

    Args:
        v1_stats: V1 bucket statistics
        v2_stats: V2 bucket statistics

    Returns:
        Markdown comparison table
    """
    if not v1_stats or not v2_stats:
        return "### V1 vs V2 Comparison\n\nInsufficient data for comparison.\n"

    lines = [
        "### V1 vs V2 Comparison",
        "",
        "| Bucket | V1 p95 | V2 p95 | Delta | Status |",
        "|--------|--------|--------|-------|--------|",
    ]

    all_buckets = set(v1_stats.keys()) | set(v2_stats.keys())

    for bucket_name in sorted(all_buckets):
        if bucket_name in v1_stats and bucket_name in v2_stats:
            v1_p95 = v1_stats[bucket_name].p95
            v2_p95 = v2_stats[bucket_name].p95
            delta = ((v2_p95 - v1_p95) / v1_p95) * 100

            if delta < -5:
                status = "✅ Faster"
            elif delta < 10:
                status = "⚠️ Similar"
            else:
                status = "❌ Slower"

            lines.append(f"| {bucket_name} | {format_time(v1_p95)} | {format_time(v2_p95)} | " f"{delta:+.1f}% | {status} |")
        elif bucket_name in v1_stats:
            lines.append(f"| {bucket_name} | {format_time(v1_stats[bucket_name].p95)} | — | — | New in V2 |")
        else:
            lines.append(f"| {bucket_name} | — | {format_time(v2_stats[bucket_name].p95)} | — | Only in V2 |")

    return "\n".join(lines)


def generate_pr_comment(
    run_id: str,
    commit_sha: str,
    v1_stats: List[Dict[str, Any]],
    v2_stats: List[Dict[str, Any]],
    v1_judgement: Optional[Judgement] = None,
    v2_judgement: Optional[Judgement] = None,
    gate_result_v1: Optional[Dict] = None,
    gate_result_v2: Optional[Dict] = None,
    is_synthetic: bool = False,
) -> str:
    """Generate full PR comment markdown.

    Args:
        run_id: Run identifier
        commit_sha: Git commit SHA
        v1_stats: V1 raw stats from database
        v2_stats: V2 raw stats from database
        v1_judgement: V1 workflow judgement (optional)
        v2_judgement: V2 workflow judgement (optional)
        gate_result_v1: V1 gate result (optional)
        gate_result_v2: V2 gate result (optional)
        is_synthetic: Whether data is from --dry-run mode (default: False)

    Returns:
        Markdown comment string
    """
    lines = []

    # Conditionally show synthetic banner
    if is_synthetic:
        lines.append("# 🎯 APEX Performance Report [SYNTHETIC DATA]\n")
        lines.append("> ⚠️ **This report uses mock data (dry-run mode)**  \n")
        lines.append("> Real pipeline integration tracked in `docs/apex/phase2/REAL_PIPELINE_INTEGRATION.md`\n")
    else:
        lines.append("# 🎯 APEX Performance Report\n")

    # Overall gate verdict at top
    overall_status = worst_status(v2_stats or v1_stats)
    overall_icon = get_status_icon(overall_status)

    lines.append(f"## {overall_icon} APEX Performance Verdict: **{overall_status.upper()}**\n")
    lines.append(f"**Run ID:** `{run_id}` | **Commit:** `{commit_sha[:8]}`\n")

    # Gate verdict details
    if gate_result_v1:
        v1_verdict = "PASSED ✅" if not gate_result_v1["should_block"] else "BLOCKED ❌"
        lines.append(f"**V1 Gate:** {v1_verdict}")

    if gate_result_v2:
        v2_verdict = "PASSED ✅" if not gate_result_v2["should_block"] else "BLOCKED ❌"
        lines.append(f"**V2 Gate:** {v2_verdict} (mode: {gate_result_v2['mode']})")

    lines.append("")

    # V1 vs V2 comparison (if both exist and using judgements)
    if v1_judgement and v2_judgement:
        comparison = generate_v1_v2_comparison(
            v1_judgement.bucket_stats,
            v2_judgement.bucket_stats,
        )
        lines.append(comparison)
        lines.append("")

    # V1 bucket stats
    if v1_judgement:
        v1_table = generate_bucket_table(v1_judgement.bucket_stats, "V1 Workflow Performance")
        lines.append(v1_table)
        lines.append("")

    # V2 bucket stats
    if v2_judgement:
        v2_table = generate_bucket_table(v2_judgement.bucket_stats, "V2 Workflow Performance")
        lines.append(v2_table)
        lines.append("")

    # Per-zone heatmap (collapsed sections)
    if v1_stats and len(v1_stats) > 0:
        heatmap = generate_zone_heatmap_from_stats(v1_stats, "📊 V1 Zone × Bucket Heatmap", max_zones=8)
        if heatmap:
            lines.append(heatmap)
            lines.append("")

    if v2_stats and len(v2_stats) > 0:
        heatmap = generate_zone_heatmap_from_stats(v2_stats, "📊 V2 Zone × Bucket Heatmap", max_zones=8)
        if heatmap:
            lines.append(heatmap)
            lines.append("")

    # Worst offenders (collapsed sections)
    if v1_stats and len(v1_stats) > 0:
        offenders = generate_worst_offenders(v1_stats, "⚠️ V1 Worst Offenders (Top 10)", top_n=10)
        if offenders:
            lines.append(offenders)
            lines.append("")

    if v2_stats and len(v2_stats) > 0:
        offenders = generate_worst_offenders(v2_stats, "⚠️ V2 Worst Offenders (Top 10)", top_n=10)
        if offenders:
            lines.append(offenders)
            lines.append("")

    # Worst-zone summary (if using judgements)
    if v1_judgement and v1_judgement.worst_zone_p95:
        lines.append(
            f"**V1 Worst-Zone p95:** {format_time(v1_judgement.worst_zone_p95)} "
            f"(zone: {v1_judgement.worst_zone_name or 'unknown'})"
        )

    if v2_judgement and v2_judgement.worst_zone_p95:
        lines.append(
            f"**V2 Worst-Zone p95:** {format_time(v2_judgement.worst_zone_p95)} "
            f"(zone: {v2_judgement.worst_zone_name or 'unknown'})"
        )

    lines.extend(
        [
            "",
            "---",
            f"*Generated by APEX PR Comment Generator v{__version__}*",
        ]
    )

    # Apply size guardrails
    lines = truncate_comment(lines)

    return "\n".join(lines)


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Generate APEX PR comment")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    parser.add_argument("--run-id", required=True, help="Run identifier")
    parser.add_argument("--commit-sha", required=True, help="Git commit SHA")
    parser.add_argument("--ledger-db", type=Path, required=True, help="Ledger database")
    parser.add_argument("--output", type=Path, required=True, help="Output markdown file")
    parser.add_argument("--summary-file", type=Path, help="Optional summary.json from matrix runner")
    parser.add_argument("--synthetic", action="store_true", help="Mark as synthetic data (dry-run mode)")

    args = parser.parse_args()

    try:
        # Fetch V1 stats directly from database
        v1_stats = fetch_run_stats(
            db_path=str(args.ledger_db),
            run_id=args.run_id,
            workflow_version="v1",
            commit_sha=args.commit_sha,
        )

        # Fetch V2 stats directly from database
        v2_stats = fetch_run_stats(
            db_path=str(args.ledger_db),
            run_id=args.run_id,
            workflow_version="v2",
            commit_sha=args.commit_sha,
        )

        # Determine if data is synthetic from explicit flag only
        # Schema does not currently have is_synthetic column in apex_runs,
        # so we rely on caller to pass --synthetic when using --dry-run
        is_synthetic = args.synthetic

        # Also query using existing baseline stats API (for judgements)
        v1_bucket_stats = query_baseline_stats(
            ledger_db_path=str(args.ledger_db),
            workflow_version="v1",
            commit_sha=args.commit_sha,
        )

        v2_bucket_stats = query_baseline_stats(
            ledger_db_path=str(args.ledger_db),
            workflow_version="v2",
            commit_sha=args.commit_sha,
        )

        # Create judgements if we have bucket stats
        v1_judgement = None
        v2_judgement = None

        if v1_bucket_stats:
            v1_judgement = Judgement(
                run_id=args.run_id,
                workflow_version="v1",
                zone=None,
                bucket_stats=v1_bucket_stats,
                regression_report=None,
                pass_fail=worst_status(v1_stats) if v1_stats else "pass",
                explanation="V1 workflow performance",
            )

        if v2_bucket_stats:
            v2_judgement = Judgement(
                run_id=args.run_id,
                workflow_version="v2",
                zone=None,
                bucket_stats=v2_bucket_stats,
                regression_report=None,
                pass_fail=worst_status(v2_stats) if v2_stats else "pass",
                explanation="V2 workflow performance",
            )

        # Evaluate gates (returns tuple: verdict, explanation)
        gate_result_v1 = None
        gate_result_v2 = None

        if v1_judgement:
            gate_obj = evaluate_gate(v1_judgement, mode="enforce")
            gate_result_v1 = {
                "verdict": "fail" if gate_obj.should_block else ("warn" if gate_obj.reasons else "pass"),
                "explanation": gate_obj.explanation,
                "mode": gate_obj.mode,
                "should_block": gate_obj.should_block,
            }

        if v2_judgement:
            gate_obj = evaluate_gate(v2_judgement, mode="shadow")
            gate_result_v2 = {
                "verdict": "fail" if gate_obj.should_block else ("warn" if gate_obj.reasons else "pass"),
                "explanation": gate_obj.explanation,
                "mode": gate_obj.mode,
                "should_block": gate_obj.should_block,
            }

        # Generate comment with new signature
        comment = generate_pr_comment(
            run_id=args.run_id,
            commit_sha=args.commit_sha,
            v1_stats=v1_stats,
            v2_stats=v2_stats,
            v1_judgement=v1_judgement,
            v2_judgement=v2_judgement,
            gate_result_v1=gate_result_v1,
            gate_result_v2=gate_result_v2,
            is_synthetic=is_synthetic,
        )

        # Write to file
        args.output.write_text(comment)
        logger.info(f"Wrote PR comment to {args.output} ({len(comment):,} characters)")

        # Also print to stdout for CI
        print(comment)

        return 0

    except Exception as e:
        logger.error(f"Failed to generate PR comment: {e}")

        # Write fallback error comment so CI doesn't fail on missing file
        error_comment = f"""# 🎯 APEX Performance Report [ERROR]

⚠️ **Failed to generate performance report**

**Error:** `{e}`

**Run ID:** {args.run_id}
**Commit:** {args.commit_sha}

This likely means:
- No performance data was aggregated
- Database query failed
- Schema mismatch

Please check the job logs for details.
"""
        try:
            args.output.write_text(error_comment)
            logger.info(f"Wrote fallback error comment to {args.output}")
        except Exception as write_err:
            logger.error(f"Could not write fallback comment: {write_err}")

        return 1


if __name__ == "__main__":
    sys.exit(main())
