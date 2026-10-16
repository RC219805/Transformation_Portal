#!/usr/bin/env python3
"""Tests for APEX dashboard generation and data extraction.

Validates:
- Data extraction from ledger using apex_trends view
- JSON payload structure and validation
- Dashboard HTML generation
- Query performance with indexes
- Chart.js data structure compliance
"""

# pylint: disable=redefined-outer-name

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

from transformation_portal.metrics.aggregator import compute_bucket_stats
from transformation_portal.metrics.ledger import PerformanceLedger
from transformation_portal.metrics.performance_capsule import PerformanceBucket, PerformanceCapsule


def _insert_measured_dashboard_row(
    connection,
    *,
    run_id,
    timestamp,
    version="v2",
    bucket_name="dashboard-fixture",
    total=1.0,
    unit="image_wall_seconds",
    workload="a" * 64,
    config="fixture-config",
):
    capsules = [
        PerformanceCapsule(
            image_id=f"image-{index}",
            image_path=f"/private/fixture/image-{index}.jpg",
            input_hash=f"input-{index}",
            original_shape=(16, 16),
            enforced_shape=(16, 16),
            pixel_count=256,
            dimension_adjustment="exact",
            timings={"total": total},
            workflow_version=version,
            zone="local",
            backend_id="fixture-backend",
            model_variant="fixture-model",
            config_hash=config,
            pipeline_version="fixture-pipeline",
            schema_version="2.1.0",
            measurement_id=f"{run_id}:{index}",
            measurement_unit=unit,
            workload_id=workload,
        )
        for index in range(20)
    ]
    bucket = PerformanceBucket(bucket_name, {}, 1.4, 2.0, measurement_unit=unit, workload_id=workload)
    stats = compute_bucket_stats(capsules, bucket)
    assert stats is not None
    connection.execute(
        "INSERT INTO apex_runs (run_id, commit_sha, timestamp, workflow_version, zone, bucket_name, "
        "p50, p95, p99, count, threshold_p50, threshold_p95, pass_fail, raw_capsules_json) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            run_id,
            "a" * 40,
            timestamp,
            version,
            "local",
            bucket_name,
            stats.p50,
            stats.p95,
            stats.p99,
            stats.count,
            stats.threshold_p50,
            stats.threshold_p95,
            stats.pass_fail,
            json.dumps(stats.measurement_evidence),
        ),
    )


def test_measurement_reader_dashboard_separates_unit_workload_and_context(tmp_path: Path) -> None:
    from scripts.apex_dashboard_generator import generate_dashboard_data, generate_index_html

    ledger = tmp_path / "mixed-context.db"
    PerformanceLedger(ledger)
    timestamp = datetime.now(timezone.utc).isoformat()
    with sqlite3.connect(ledger) as connection:
        _insert_measured_dashboard_row(connection, run_id="image", timestamp=timestamp, total=1.0)
        _insert_measured_dashboard_row(connection, run_id="batch", timestamp=timestamp, unit="batch_wall_seconds", total=2.5)
        _insert_measured_dashboard_row(connection, run_id="other-workload", timestamp=timestamp, workload="b" * 64, total=1.8)
        _insert_measured_dashboard_row(
            connection, run_id="other-context", timestamp=timestamp, config="other-config", total=1.2
        )

    data = generate_dashboard_data(ledger)
    assert len(data["trends"]) == 4
    assert len({row["series_id"] for row in data["trends"]}) == 4
    assert sorted(row["avg_p95"] for row in data["trends"]) == [1.0, 1.2, 1.8, 2.5]
    assert {row["measurement_unit"] for row in data["trends"]} == {"image_wall_seconds", "batch_wall_seconds"}
    assert data["measurement_evidence"]["eligible_rows"] == 4
    serialized = json.dumps(data)
    assert "image_path" not in serialized
    assert "raw_capsules_json" not in serialized
    assert '"capsules"' not in serialized
    output = tmp_path / "dashboard"
    output.mkdir()
    generate_index_html(data, output)
    html = (output / "index.html").read_text()
    assert "d.series_id === seriesId" in html
    assert "/private/fixture" not in html


def test_measurement_reader_dashboard_keeps_legacy_visible_but_out_of_eligible_series(tmp_path: Path) -> None:
    from scripts.apex_dashboard_generator import generate_dashboard_data

    ledger = tmp_path / "partial-evidence.db"
    PerformanceLedger(ledger)
    timestamp = datetime.now(timezone.utc).isoformat()
    with sqlite3.connect(ledger) as connection:
        _insert_measured_dashboard_row(connection, run_id="eligible", timestamp=timestamp)
        connection.execute(
            "INSERT INTO apex_runs (run_id, commit_sha, timestamp, workflow_version, zone, bucket_name, "
            "p50, p95, p99, count, threshold_p50, threshold_p95, pass_fail) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("legacy", "a" * 40, timestamp, "v2", "local", "dashboard-fixture", 142.0, 142.0, 142.0, 20, 1.4, 2.0, "fail"),
        )

    data = generate_dashboard_data(ledger)
    assert len(data["latest_runs"]) == 2
    assert len(data["trends"]) == 1
    assert data["trends"][0]["avg_p95"] == 1.0
    assert data["trends"][0]["run_count"] == 1
    assert data["regressions"] == []
    assert data["worst_offenders"] == []
    assert data["measurement_evidence"] == {
        "eligible_rows": 1,
        "insufficient_rows": 1,
        "legacy_or_insufficient_rows_excluded_from_trends": 1,
    }


def test_measurement_reader_dashboard_excludes_legacy_twenty_capsules(tmp_path: Path) -> None:
    from scripts.apex_dashboard_generator import generate_dashboard_data, generate_index_html, generate_latest_html

    ledger = tmp_path / "legacy-twenty.db"
    PerformanceLedger(ledger)
    with sqlite3.connect(ledger) as connection:
        connection.execute(
            "INSERT INTO apex_runs (run_id, commit_sha, timestamp, workflow_version, zone, "
            "bucket_name, p50, p95, p99, count, threshold_p50, threshold_p95, pass_fail) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "real-run",
                "a" * 40,
                datetime.now(timezone.utc).isoformat(),
                "v2",
                "local",
                "unknown",
                142.0,
                142.0,
                142.0,
                20,
                60.0,
                120.0,
                "fail",
            ),
        )

    data = generate_dashboard_data(ledger)
    assert data["trends"] == []
    assert data["regressions"] == []
    assert data["worst_offenders"] == []
    assert data["latest_runs"][0]["independent_count"] == 0
    assert data["latest_runs"][0]["capsule_count"] == 20
    assert data["latest_runs"][0]["is_insufficient_data"] is True
    assert data["measurement_evidence"]["insufficient_rows"] == 1
    output = tmp_path / "dashboard"
    output.mkdir()
    generate_latest_html(data, output)
    generate_index_html(data, output)
    html = (output / "latest.html").read_text()
    assert "Insufficient data" in html
    assert "0 independent / 20 capsules" in html
    assert "✅" not in html
    assert "0 eligible aggregate rows; 1 insufficient rows excluded" in (output / "index.html").read_text()


def test_measurement_reader_dashboard_rejects_unknown_summary(tmp_path: Path) -> None:
    from scripts.apex_dashboard_generator import generate_dashboard_data

    ledger = tmp_path / "unknown-summary.db"
    PerformanceLedger(ledger)
    with sqlite3.connect(ledger) as connection:
        connection.execute(
            "INSERT INTO apex_runs (run_id, commit_sha, timestamp, workflow_version, zone, "
            "bucket_name, p50, p95, count, threshold_p50, threshold_p95, pass_fail, raw_capsules_json) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "real-run",
                "a" * 40,
                datetime.now(timezone.utc).isoformat(),
                "v2",
                "local",
                "unknown",
                1.0,
                1.0,
                20,
                60.0,
                120.0,
                "pass",
                '{"schema":"untrusted"}',
            ),
        )

    with pytest.raises(ValueError):
        generate_dashboard_data(ledger)


@pytest.fixture
def sample_ledger(tmp_path: Path) -> Path:
    """Create a ledger with sample APEX run data."""
    db_path = tmp_path / "test_apex.db"
    PerformanceLedger(db_path)

    # Verify apex_runs table and apex_trends view exist
    with sqlite3.connect(db_path) as conn:
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='apex_runs'")
        assert cursor.fetchone() is not None, "apex_runs table should exist"

        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='view' AND name='apex_trends'")
        assert cursor.fetchone() is not None, "apex_trends view should exist"

    # Insert sample APEX run data
    with sqlite3.connect(db_path) as conn:
        for i in range(30):  # 30 days of data
            timestamp = (datetime.now(timezone.utc) - timedelta(days=i)).isoformat()

            # V1 and V2 runs for each day
            for version in ["v1", "v2"]:
                for bucket in ["small_depth_v1", "medium_depth_v1", "large_depth_v1"]:
                    # Some runs pass, some warn, some fail
                    if i % 7 == 0 and version == "v2":
                        p95 = 2.5  # Over threshold
                    elif i % 5 == 0:
                        p95 = 1.8
                    else:
                        p95 = 1.2

                    _insert_measured_dashboard_row(
                        conn,
                        run_id=f"run-{i}-{version}-{bucket}",
                        timestamp=timestamp,
                        version=version,
                        bucket_name=bucket,
                        total=p95,
                    )
        conn.commit()

    return db_path


def test_ledger_schema_version(sample_ledger: Path) -> None:
    """Verify ledger schema is version 3."""
    with sqlite3.connect(sample_ledger) as conn:
        cursor = conn.execute("SELECT version FROM schema_version")
        version = cursor.fetchone()[0]
        assert version == 3, "Schema should be v3 (Phase 3)"


def test_apex_trends_view_exists(sample_ledger: Path) -> None:
    """Verify apex_trends view exists and returns data."""
    with sqlite3.connect(sample_ledger) as conn:
        cursor = conn.execute("SELECT COUNT(*) FROM apex_trends")
        count = cursor.fetchone()[0]
        assert count > 0, "apex_trends view should return aggregated data"


def test_apex_trends_view_structure(sample_ledger: Path) -> None:
    """Verify apex_trends view has correct columns."""
    with sqlite3.connect(sample_ledger) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.execute("SELECT * FROM apex_trends LIMIT 1")
        row = cursor.fetchone()

        expected_columns = {
            "bucket_name",
            "zone",
            "workflow_version",
            "date",
            "avg_p50",
            "avg_p95",
            "avg_p99",
            "run_count",
            "fail_count",
            "warn_count",
        }

        actual_columns = set(row.keys())
        assert expected_columns.issubset(actual_columns), f"Missing columns: {expected_columns - actual_columns}"


def test_optimized_indexes_exist(sample_ledger: Path) -> None:
    """Verify Phase 3 optimized indexes exist."""
    with sqlite3.connect(sample_ledger) as conn:
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='apex_runs'")
        indexes = {row[0] for row in cursor.fetchall()}

        required_indexes = {
            "idx_apex_runs_timestamp",
            "idx_apex_runs_bucket_zone_time",
            "idx_apex_runs_pass_fail",
        }

        assert required_indexes.issubset(indexes), f"Missing indexes: {required_indexes - indexes}"


def test_dashboard_data_extraction(sample_ledger: Path) -> None:
    """Test data extraction for dashboard generation."""
    from scripts.apex_dashboard_generator import generate_dashboard_data

    data = generate_dashboard_data(sample_ledger, days=30)

    # Verify structure
    assert "generated_at" in data
    assert "days" in data
    assert "trends" in data
    assert "regressions" in data
    assert "worst_offenders" in data
    assert "latest_runs" in data

    # Verify data types
    assert isinstance(data["trends"], list)
    assert isinstance(data["regressions"], list)
    assert isinstance(data["worst_offenders"], list)
    assert isinstance(data["latest_runs"], list)

    # Verify non-empty (sample data should produce results)
    assert len(data["trends"]) > 0, "Should have trend data"
    assert len(data["latest_runs"]) > 0, "Should have latest runs"


def test_dashboard_data_json_serializable(sample_ledger: Path) -> None:
    """Verify dashboard data is JSON serializable."""
    from scripts.apex_dashboard_generator import generate_dashboard_data

    data = generate_dashboard_data(sample_ledger, days=30)

    # Should not raise
    json_str = json.dumps(data, indent=2)

    # Should round-trip
    reloaded = json.loads(json_str)
    assert reloaded["days"] == data["days"]


def test_dashboard_html_generation(sample_ledger: Path, tmp_path: Path) -> None:
    """Test HTML dashboard generation."""
    from scripts.apex_dashboard_generator import generate_dashboard_data, generate_index_html, generate_latest_html

    data = generate_dashboard_data(sample_ledger, days=30)
    output_dir = tmp_path / "dashboard"
    output_dir.mkdir()

    # Generate HTML files
    generate_index_html(data, output_dir)
    generate_latest_html(data, output_dir)

    # Verify files exist
    assert (output_dir / "index.html").exists()
    assert (output_dir / "latest.html").exists()

    # Verify HTML structure
    index_html = (output_dir / "index.html").read_text()
    assert "APEX Performance Dashboard" in index_html
    assert "chart.js" in index_html.lower()  # CDN reference (case-insensitive)
    assert "trendsChart" in index_html
    assert "worstChart" in index_html
    assert "regressionsChart" in index_html

    latest_html = (output_dir / "latest.html").read_text()
    assert "Latest APEX Runs" in latest_html
    assert "<table>" in latest_html


def test_chart_js_data_structure(sample_ledger: Path) -> None:
    """Verify Chart.js data structure is valid."""
    from scripts.apex_dashboard_generator import generate_dashboard_data

    data = generate_dashboard_data(sample_ledger, days=30)

    # Trends data should have required fields for line chart
    for trend in data["trends"]:
        assert "date" in trend
        assert "bucket_name" in trend
        assert "avg_p95" in trend
        assert isinstance(trend["avg_p95"], (int, float, type(None)))

    # Worst offenders should have max_ratio for bar chart
    for offender in data["worst_offenders"]:
        assert "bucket_name" in offender
        assert "max_ratio" in offender
        assert isinstance(offender["max_ratio"], (int, float))


def test_query_performance_with_indexes(sample_ledger: Path) -> None:
    """Verify query performance is acceptable with indexes."""
    import time

    with sqlite3.connect(sample_ledger) as conn:
        # Query using timestamp index
        start = time.perf_counter()
        cursor = conn.execute("SELECT * FROM apex_runs ORDER BY timestamp DESC LIMIT 100")
        cursor.fetchall()
        elapsed_timestamp = time.perf_counter() - start

        # Query using composite index
        start = time.perf_counter()
        cursor = conn.execute("""
            SELECT * FROM apex_runs
            WHERE bucket_name = 'small_depth_v1'
              AND zone = 'local'
            ORDER BY timestamp DESC
            LIMIT 100
            """)
        cursor.fetchall()
        elapsed_composite = time.perf_counter() - start

        # Query using apex_trends view
        start = time.perf_counter()
        cursor = conn.execute("SELECT * FROM apex_trends LIMIT 100")
        cursor.fetchall()
        elapsed_view = time.perf_counter() - start

        # All queries should complete quickly (< 100ms for small dataset)
        assert elapsed_timestamp < 0.1, f"Timestamp query too slow: {elapsed_timestamp:.3f}s"
        assert elapsed_composite < 0.1, f"Composite query too slow: {elapsed_composite:.3f}s"
        assert elapsed_view < 0.1, f"View query too slow: {elapsed_view:.3f}s"


def test_regression_detection_in_data(sample_ledger: Path) -> None:
    """Verify regressions are correctly identified."""
    from scripts.apex_dashboard_generator import generate_dashboard_data

    data = generate_dashboard_data(sample_ledger, days=30)

    # Should have some regressions (sample data includes fails/warns)
    assert len(data["regressions"]) > 0, "Should detect regressions"

    # Regressions should have required fields
    for regression in data["regressions"]:
        assert "timestamp" in regression
        assert "commit_sha" in regression
        assert "bucket_name" in regression
        assert "p95" in regression
        assert "threshold_p95" in regression
        assert "pass_fail" in regression
        assert regression["pass_fail"] in ["warn", "fail"]


def test_worst_offenders_ranking(sample_ledger: Path) -> None:
    """Verify worst offenders are ranked by max ratio."""
    from scripts.apex_dashboard_generator import generate_dashboard_data

    data = generate_dashboard_data(sample_ledger, days=30)

    if len(data["worst_offenders"]) > 1:
        # Should be sorted descending by max_ratio
        ratios = [w["max_ratio"] for w in data["worst_offenders"]]
        assert ratios == sorted(ratios, reverse=True), "Should be sorted by max_ratio DESC"


def test_data_json_export(sample_ledger: Path, tmp_path: Path) -> None:
    """Test data.json export for external tools."""
    from scripts.apex_dashboard_generator import generate_dashboard_data

    data = generate_dashboard_data(sample_ledger, days=30)
    output_file = tmp_path / "data.json"

    # Write data.json
    output_file.write_text(json.dumps(data, indent=2))

    # Verify file is valid JSON
    reloaded = json.loads(output_file.read_text())
    assert reloaded["days"] == 30

    # Verify structure for external tools
    assert "trends" in reloaded
    assert "latest_runs" in reloaded


def test_empty_ledger_handling(tmp_path: Path) -> None:
    """Verify graceful handling of empty ledger."""
    from scripts.apex_dashboard_generator import generate_dashboard_data

    db_path = tmp_path / "empty.db"
    PerformanceLedger(db_path)

    # Should not crash
    data = generate_dashboard_data(db_path, days=30)

    # Should return empty lists
    assert data["trends"] == []
    assert data["latest_runs"] == []
    assert data["regressions"] == []
    assert data["worst_offenders"] == []


def test_ledger_migration_v2_to_v3(tmp_path: Path) -> None:
    """Verify schema migration from v2 to v3."""
    db_path = tmp_path / "migration.db"

    # Create v2 schema manually
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE schema_version (version INTEGER PRIMARY KEY)")
        conn.execute("INSERT INTO schema_version (version) VALUES (2)")
        conn.execute("""
            CREATE TABLE apex_runs (
                run_id TEXT NOT NULL,
                commit_sha TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                workflow_version TEXT NOT NULL,
                zone TEXT,
                bucket_name TEXT NOT NULL,
                p50 REAL NOT NULL,
                p95 REAL NOT NULL,
                p99 REAL,
                count INTEGER NOT NULL,
                threshold_p50 REAL NOT NULL,
                threshold_p95 REAL NOT NULL,
                pass_fail TEXT NOT NULL,
                raw_capsules_json TEXT,
                PRIMARY KEY (run_id, workflow_version, zone, bucket_name)
            )
            """)
        conn.commit()

    # Initialize ledger (should trigger migration)
    PerformanceLedger(db_path)

    # Verify v3 schema elements exist
    with sqlite3.connect(db_path) as conn:
        # Check version
        cursor = conn.execute("SELECT version FROM schema_version")
        assert cursor.fetchone()[0] == 3

        # Check indexes exist
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='index' AND name='idx_apex_runs_timestamp'")
        assert cursor.fetchone() is not None

        # Check view exists
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='view' AND name='apex_trends'")
        assert cursor.fetchone() is not None


def test_dashboard_responsive_design(sample_ledger: Path, tmp_path: Path) -> None:
    """Verify dashboard HTML includes responsive design elements."""
    from scripts.apex_dashboard_generator import generate_dashboard_data, generate_index_html

    data = generate_dashboard_data(sample_ledger, days=30)
    output_dir = tmp_path / "dashboard"
    output_dir.mkdir()

    generate_index_html(data, output_dir)
    html = (output_dir / "index.html").read_text()

    # Check for responsive viewport meta tag
    assert 'name="viewport"' in html
    assert "width=device-width" in html

    # Check for mobile-responsive CSS
    assert "@media" in html or "max-width: 768px" in html


@pytest.mark.parametrize("days", [7, 30, 90, 365])
def test_dashboard_data_retention_windows(sample_ledger: Path, days: int) -> None:
    """Test dashboard generation with different retention windows."""
    from scripts.apex_dashboard_generator import generate_dashboard_data

    data = generate_dashboard_data(sample_ledger, days=days)

    assert data["days"] == days
    assert isinstance(data["trends"], list)

    # Trends should respect retention window (sample has 30 days)
    if days <= 30:
        assert len(data["trends"]) > 0
