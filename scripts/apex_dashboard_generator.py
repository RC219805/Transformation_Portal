#!/usr/bin/env python3
"""Generate static APEX performance dashboard from ledger database.

This tool generates a static HTML dashboard with interactive charts for
visualizing APEX performance trends, regressions, and worst offenders.

Design:
- Static HTML generation (no backend required)
- Chart.js for interactive visualizations
- GitHub Pages compatible
- Mobile-responsive design
- Time-series trend analysis
- Zone-based heatmaps

Usage:
    # Generate dashboard from ledger
    python scripts/apex_dashboard_generator.py \\
        --ledger-db apex_performance.db \\
        --output-dir docs/apex/ \\
        --days 90

    # Generate with custom retention
    python scripts/apex_dashboard_generator.py \\
        --ledger-db apex_performance.db \\
        --output-dir docs/apex/ \\
        --days 365

Architecture:
- Queries apex_runs table (not individual capsules)
- Generates index.html (main dashboard)
- Generates latest.html (latest run summary)
- Exports data.json for external tools
- All visualization in client-side JavaScript

Version: 1.0.0
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict

__version__ = "1.0.0"


def generate_dashboard_data(db_path: Path, days: int = 90) -> Dict[str, Any]:
    """Extract time-series data for dashboard visualization.

    Args:
        db_path: Path to APEX ledger database
        days: Number of days of history to include

    Returns:
        Dict with trends, regressions, worst offenders, and metadata
    """
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row

        # Query trends using pre-aggregated view (Phase 3 optimization)
        trends_query = """
            SELECT
                date,
                bucket_name,
                zone,
                workflow_version,
                avg_p50,
                avg_p95,
                avg_p99,
                run_count,
                fail_count,
                warn_count
            FROM apex_trends
            WHERE date >= DATE(?)
            ORDER BY date DESC
        """

        cursor = conn.execute(trends_query, (cutoff.isoformat(),))
        trends = [dict(row) for row in cursor.fetchall()]

        # Query regressions (warnings and failures)
        regressions_query = """
            SELECT
                timestamp,
                commit_sha,
                bucket_name,
                zone,
                workflow_version,
                p95,
                threshold_p95,
                pass_fail
            FROM apex_runs
            WHERE pass_fail IN ('warn', 'fail')
            AND timestamp >= ?
            ORDER BY timestamp DESC
            LIMIT 100
        """

        cursor = conn.execute(regressions_query, (cutoff.isoformat(),))
        regressions = [dict(row) for row in cursor.fetchall()]

        # Query worst offenders (all time, by max ratio)
        # Using optimized composite index on bucket_name, zone, timestamp
        worst_query = """
            SELECT
                bucket_name,
                zone,
                workflow_version,
                MAX(p95 / NULLIF(threshold_p95, 0)) as max_ratio,
                COUNT(*) as total_runs,
                SUM(CASE WHEN pass_fail = 'fail' THEN 1 ELSE 0 END) as fail_count
            FROM apex_runs
            WHERE threshold_p95 > 0
            GROUP BY bucket_name, zone, workflow_version
            HAVING max_ratio > 1.0
            ORDER BY max_ratio DESC
            LIMIT 20
        """

        cursor = conn.execute(worst_query)
        worst_offenders = [dict(row) for row in cursor.fetchall()]

        # Latest run summary (using timestamp DESC index)
        latest_query = """
            SELECT
                run_id,
                commit_sha,
                timestamp,
                workflow_version,
                zone,
                bucket_name,
                p50,
                p95,
                p99,
                count,
                threshold_p50,
                threshold_p95,
                pass_fail
            FROM apex_runs
            ORDER BY timestamp DESC
            LIMIT 100
        """

        cursor = conn.execute(latest_query)
        latest_runs = [dict(row) for row in cursor.fetchall()]

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "days": days,
        "trends": trends,
        "regressions": regressions,
        "worst_offenders": worst_offenders,
        "latest_runs": latest_runs,
    }


def generate_index_html(data: Dict[str, Any], output_dir: Path) -> None:
    """Generate main dashboard HTML with Chart.js visualizations.

    Args:
        data: Dashboard data from generate_dashboard_data()
        output_dir: Output directory for HTML files
    """

    # Compute summary metrics
    total_runs = len(data["latest_runs"])
    total_regressions = len(data["regressions"])
    total_problem_buckets = len(data["worst_offenders"])

    # Prepare data for charts
    trends_json = json.dumps(data["trends"])
    worst_json = json.dumps(data["worst_offenders"])

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>APEX Performance Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Roboto", "Helvetica Neue", sans-serif;
            line-height: 1.6;
            color: #24292e;
            background: #f6f8fa;
            padding: 20px;
        }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        header {{
            background: white;
            border-radius: 6px;
            padding: 30px;
            margin-bottom: 20px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.12);
        }}
        h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
            color: #0366d6;
        }}
        .subtitle {{
            color: #586069;
            font-size: 0.95em;
        }}
        .metrics {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .metric-card {{
            background: white;
            border-radius: 6px;
            padding: 25px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.12);
            border-left: 4px solid #0366d6;
        }}
        .metric-value {{
            font-size: 2.5em;
            font-weight: bold;
            color: #0366d6;
            margin-bottom: 5px;
        }}
        .metric-label {{
            color: #586069;
            font-size: 0.9em;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        .chart-section {{
            background: white;
            border-radius: 6px;
            padding: 30px;
            margin: 20px 0;
            box-shadow: 0 1px 3px rgba(0,0,0,0.12);
        }}
        .chart-section h2 {{
            margin-bottom: 20px;
            font-size: 1.5em;
            color: #24292e;
        }}
        canvas {{
            max-height: 400px;
            margin-top: 20px;
        }}
        .footer {{
            text-align: center;
            color: #586069;
            padding: 30px 0;
            font-size: 0.9em;
        }}
        .footer a {{
            color: #0366d6;
            text-decoration: none;
        }}
        .footer a:hover {{
            text-decoration: underline;
        }}
        @media (max-width: 768px) {{
            h1 {{ font-size: 1.8em; }}
            .metrics {{ grid-template-columns: 1fr; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🎯 APEX Performance Dashboard</h1>
            <p class="subtitle">
                Generated: {data['generated_at']} UTC |
                Showing last {data['days']} days |
                <a href="latest.html">Latest Run →</a>
            </p>
        </header>

        <div class="metrics">
            <div class="metric-card">
                <div class="metric-value">{total_runs}</div>
                <div class="metric-label">Recent Runs</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{total_regressions}</div>
                <div class="metric-label">Warnings/Failures</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{total_problem_buckets}</div>
                <div class="metric-label">Problem Buckets</div>
            </div>
        </div>

        <div class="chart-section">
            <h2>📈 Performance Trends (p95 Latency)</h2>
            <canvas id="trendsChart"></canvas>
        </div>

        <div class="chart-section">
            <h2>🔥 Worst Offenders (Max p95/Threshold Ratio)</h2>
            <canvas id="worstChart"></canvas>
        </div>

        <div class="chart-section">
            <h2>⚠️ Recent Regressions</h2>
            <canvas id="regressionsChart"></canvas>
        </div>

        <div class="footer">
            <p>
                APEX Performance Observability Platform v1.0 |
                <a href="data.json">Download Raw Data</a>
            </p>
        </div>
    </div>

    <script>
        const trendsData = {trends_json};
        const worstData = {worst_json};
        const regressionsData = {json.dumps(data['regressions'])};

        // ========================================
        // Trends Chart (Time Series)
        // ========================================
        const trendsCtx = document.getElementById('trendsChart').getContext('2d');

        // Group by bucket and prepare datasets
        const buckets = [...new Set(trendsData.map(d => d.bucket_name))];
        const dates = [...new Set(trendsData.map(d => d.date))].sort().reverse();

        const datasets = buckets.map((bucket, idx) => {{
            const bucketData = trendsData.filter(d => d.bucket_name === bucket);
            const colors = [
                'rgb(75, 192, 192)',
                'rgb(255, 99, 132)',
                'rgb(54, 162, 235)',
                'rgb(255, 206, 86)',
                'rgb(153, 102, 255)',
            ];

            return {{
                label: bucket,
                data: dates.map(date => {{
                    const match = bucketData.find(d => d.date === date);
                    return match ? match.avg_p95 : null;
                }}),
                borderColor: colors[idx % colors.length],
                backgroundColor: colors[idx % colors.length] + '33',
                tension: 0.3,
                fill: false,
            }};
        }});

        new Chart(trendsCtx, {{
            type: 'line',
            data: {{
                labels: dates,
                datasets: datasets
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: true,
                interaction: {{
                    mode: 'index',
                    intersect: false,
                }},
                plugins: {{
                    legend: {{
                        position: 'top',
                    }},
                    title: {{
                        display: false,
                    }}
                }},
                scales: {{
                    y: {{
                        beginAtZero: true,
                        title: {{
                            display: true,
                            text: 'p95 Latency (seconds)'
                        }}
                    }},
                    x: {{
                        title: {{
                            display: true,
                            text: 'Date'
                        }}
                    }}
                }}
            }}
        }});

        // ========================================
        // Worst Offenders Chart (Horizontal Bar)
        // ========================================
        const worstCtx = document.getElementById('worstChart').getContext('2d');

        new Chart(worstCtx, {{
            type: 'bar',
            data: {{
                labels: worstData.map(d => {{
                    const zone_label = d.zone || 'global';
                    return `${{d.bucket_name}} [${{zone_label}}]`;
                }}),
                datasets: [{{
                    label: 'Max p95/threshold ratio',
                    data: worstData.map(d => d.max_ratio),
                    backgroundColor: worstData.map(d =>
                        d.max_ratio > 1.5 ? 'rgb(220, 38, 38)' : 'rgb(251, 146, 60)'
                    ),
                }}]
            }},
            options: {{
                responsive: true,
                indexAxis: 'y',
                plugins: {{
                    legend: {{
                        display: false
                    }},
                    tooltip: {{
                        callbacks: {{
                            afterLabel: function(context) {{
                                const item = worstData[context.dataIndex];
                                return [
                                    `Failures: ${{item.fail_count}}/${{item.total_runs}}`,
                                    `Workflow: ${{item.workflow_version}}`
                                ];
                            }}
                        }}
                    }}
                }},
                scales: {{
                    x: {{
                        beginAtZero: true,
                        title: {{
                            display: true,
                            text: 'Ratio (higher = worse)'
                        }}
                    }}
                }}
            }}
        }});

        // ========================================
        // Regressions Timeline (Scatter)
        // ========================================
        const regressionsCtx = document.getElementById('regressionsChart').getContext('2d');

        const regressionPoints = regressionsData.map(d => ({{
            x: d.timestamp,
            y: d.p95 / d.threshold_p95,
            bucket: d.bucket_name,
            zone: d.zone || 'global',
            status: d.pass_fail
        }}));

        new Chart(regressionsCtx, {{
            type: 'scatter',
            data: {{
                datasets: [{{
                    label: 'Failures',
                    data: regressionPoints.filter(p => p.status === 'fail'),
                    backgroundColor: 'rgb(220, 38, 38)',
                    pointRadius: 6,
                }}, {{
                    label: 'Warnings',
                    data: regressionPoints.filter(p => p.status === 'warn'),
                    backgroundColor: 'rgb(251, 146, 60)',
                    pointRadius: 5,
                }}]
            }},
            options: {{
                responsive: true,
                plugins: {{
                    tooltip: {{
                        callbacks: {{
                            label: function(context) {{
                                const point = context.raw;
                                return [
                                    `Bucket: ${{point.bucket}}`,
                                    `Zone: ${{point.zone}}`,
                                    `Ratio: ${{point.y.toFixed(2)}}x`,
                                ];
                            }}
                        }}
                    }}
                }},
                scales: {{
                    x: {{
                        type: 'time',
                        time: {{
                            unit: 'day'
                        }},
                        title: {{
                            display: true,
                            text: 'Timestamp'
                        }}
                    }},
                    y: {{
                        beginAtZero: true,
                        title: {{
                            display: true,
                            text: 'p95/threshold ratio'
                        }}
                    }}
                }}
            }}
        }});
    </script>
</body>
</html>
"""

    output_file = output_dir / "index.html"
    output_file.write_text(html)
    print(f"✓ Generated {output_file}")


def generate_latest_html(data: Dict[str, Any], output_dir: Path) -> None:
    """Generate latest run summary page.

    Args:
        data: Dashboard data from generate_dashboard_data()
        output_dir: Output directory for HTML files
    """

    latest_runs = data["latest_runs"][:20]  # Show top 20 most recent

    rows_html = ""
    for run in latest_runs:
        status_emoji = "✅" if run["pass_fail"] == "pass" else ("⚠️" if run["pass_fail"] == "warn" else "❌")
        zone_display = run["zone"] or "global"

        rows_html += f"""
        <tr>
            <td>{status_emoji}</td>
            <td><code>{run["commit_sha"][:8]}</code></td>
            <td>{run["workflow_version"]}</td>
            <td>{zone_display}</td>
            <td>{run["bucket_name"]}</td>
            <td>{run["p50"]:.2f}s</td>
            <td>{run["p95"]:.2f}s</td>
            <td>{run["threshold_p95"]:.2f}s</td>
        </tr>
        """

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>APEX Latest Run Summary</title>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            background: #f6f8fa;
            padding: 20px;
        }}
        .container {{ max-width: 1400px; margin: 0 auto; }}
        header {{
            background: white;
            border-radius: 6px;
            padding: 30px;
            margin-bottom: 20px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.12);
        }}
        h1 {{
            color: #0366d6;
            margin-bottom: 10px;
        }}
        .subtitle {{
            color: #586069;
            font-size: 0.95em;
        }}
        table {{
            width: 100%;
            background: white;
            border-radius: 6px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.12);
            border-collapse: collapse;
            overflow: hidden;
        }}
        th, td {{
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid #e1e4e8;
        }}
        th {{
            background: #f6f8fa;
            font-weight: 600;
            color: #24292e;
        }}
        code {{
            background: #f6f8fa;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: 'Monaco', 'Courier New', monospace;
            font-size: 0.9em;
        }}
        .footer {{
            text-align: center;
            color: #586069;
            padding: 30px 0;
            font-size: 0.9em;
        }}
        .footer a {{
            color: #0366d6;
            text-decoration: none;
        }}
        .footer a:hover {{
            text-decoration: underline;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>📊 Latest APEX Runs</h1>
            <p class="subtitle">
                Generated: {data['generated_at']} UTC |
                <a href="index.html">← Back to Dashboard</a>
            </p>
        </header>

        <table>
            <thead>
                <tr>
                    <th>Status</th>
                    <th>Commit</th>
                    <th>Version</th>
                    <th>Zone</th>
                    <th>Bucket</th>
                    <th>p50</th>
                    <th>p95</th>
                    <th>Threshold</th>
                </tr>
            </thead>
            <tbody>
                {rows_html}
            </tbody>
        </table>

        <div class="footer">
            <p>
                APEX Performance Observability Platform v1.0 |
                <a href="data.json">Download Raw Data</a>
            </p>
        </div>
    </div>
</body>
</html>
"""

    output_file = output_dir / "latest.html"
    output_file.write_text(html)
    print(f"✓ Generated {output_file}")


def main() -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Generate APEX dashboard from performance ledger")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("--ledger-db", type=Path, required=True, help="Path to APEX ledger database")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory for dashboard files")
    parser.add_argument("--days", type=int, default=90, help="Number of days of history to include (default: 90)")

    args = parser.parse_args()

    # Validate inputs
    if not args.ledger_db.exists():
        print(f"❌ Error: Ledger database not found: {args.ledger_db}")
        return 1

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    try:
        print(f"Generating dashboard data (last {args.days} days)...")
        data = generate_dashboard_data(args.ledger_db, args.days)

        print("Generating index.html...")
        generate_index_html(data, args.output_dir)

        print("Generating latest.html...")
        generate_latest_html(data, args.output_dir)

        # Export raw data for external tools
        data_file = args.output_dir / "data.json"
        data_file.write_text(json.dumps(data, indent=2))
        print(f"✓ Exported raw data to {data_file}")

        print("\n✅ Dashboard generation complete!")
        print(f"📁 Output: {args.output_dir}")

        return 0

    except Exception as e:
        print(f"❌ Error generating dashboard: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
