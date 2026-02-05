"""Tests for performance ledger tool (ADR-023 Phase 2)."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from tools.performance_ledger import (
    DEFAULT_REGRESSION_THRESHOLDS,
    Baseline,
    EnvironmentMetadata,
    Regression,
    Statistics,
    capture_environment,
    compute_statistics,
    detect_regressions,
    extract_timings,
    format_markdown,
    load_baseline,
    parse_manifests,
    save_baseline,
)


def test_parse_manifests_valid(tmp_path):
    """Test parsing valid manifests from directory."""
    manifests_dir = tmp_path / "manifests"
    manifests_dir.mkdir()

    # Create test manifests
    for i in range(3):
        manifest_path = manifests_dir / f"manifest_{i}.json"
        manifest_data = {
            "timing": {"total_seconds": 10.0 + i, "depth_seconds": 8.0, "v2_seconds": 2.0},
            "depth": {"model": "da3", "runtime_seconds": 8.0},
        }
        with open(manifest_path, "w") as f:
            json.dump(manifest_data, f)

    manifests = parse_manifests(manifests_dir)
    assert len(manifests) == 3
    assert all("timing" in m for m in manifests)


def test_parse_manifests_empty_directory(tmp_path):
    """Test parsing from empty directory raises ValueError."""
    manifests_dir = tmp_path / "empty"
    manifests_dir.mkdir()

    with pytest.raises(ValueError, match="No JSON manifests found"):
        parse_manifests(manifests_dir)


def test_parse_manifests_malformed_json(tmp_path):
    """Test that malformed JSON is skipped gracefully."""
    manifests_dir = tmp_path / "manifests"
    manifests_dir.mkdir()

    # Create valid manifest
    valid_path = manifests_dir / "valid.json"
    with open(valid_path, "w") as f:
        json.dump({"timing": {"total_seconds": 10.0}}, f)

    # Create malformed manifest
    invalid_path = manifests_dir / "invalid.json"
    with open(invalid_path, "w") as f:
        f.write("{invalid json")

    manifests = parse_manifests(manifests_dir)
    assert len(manifests) == 1


def test_extract_timings():
    """Test extracting timings from manifests."""
    manifests = [
        {"timing": {"total_seconds": 10.5}, "depth": {"model": "da3"}},
        {"timing": {"total_seconds": 12.3}, "depth": {"model": "da3"}},
        {"timing": {"total_seconds": 8.9}, "depth": {"model": "da3"}},
    ]

    timings, success_count, failure_count = extract_timings(manifests)

    assert len(timings) == 3
    assert timings == [10.5, 12.3, 8.9]
    assert success_count == 3
    assert failure_count == 0


def test_extract_timings_handles_missing_fields():
    """Test extracting timings handles missing fields gracefully."""
    manifests = [
        {"timing": {"total_seconds": 10.5}, "depth": {"model": "da3"}},
        {"timing": {}, "depth": {"model": "da3"}},  # Missing total_seconds
        {},  # Missing timing
    ]

    timings, success_count, failure_count = extract_timings(manifests)

    assert len(timings) == 1
    assert timings == [10.5]


def test_compute_statistics():
    """Test statistics computation correctness."""
    timings = [10.0, 12.0, 15.0, 18.0, 20.0, 25.0, 30.0, 35.0, 40.0, 50.0]

    stats = compute_statistics(timings)

    assert stats.count == 10
    assert stats.mean_sec == pytest.approx(25.5, abs=0.01)
    assert stats.median_sec == pytest.approx(22.5, abs=0.01)
    assert stats.min_sec == 10.0
    assert stats.max_sec == 50.0
    # NumPy percentile uses linear interpolation
    assert stats.p90_sec == pytest.approx(41.0, abs=1.0)
    assert stats.p95_sec == pytest.approx(45.5, abs=1.0)
    assert stats.total_sec == pytest.approx(255.0, abs=0.01)


def test_compute_statistics_empty_list_raises():
    """Test that empty timings list raises ValueError."""
    with pytest.raises(ValueError, match="No timings provided"):
        compute_statistics([])


def test_capture_environment():
    """Test environment metadata capture."""
    env = capture_environment()

    assert env.python is not None
    assert env.os is not None
    assert env.device is not None
    assert "." in env.python  # Version format


def test_detect_regressions_p95_threshold():
    """Test p95 regression detection."""
    baseline_stats = Statistics(
        count=10,
        mean_sec=10.0,
        median_sec=10.0,
        p90_sec=12.0,
        p95_sec=13.0,
        min_sec=8.0,
        max_sec=15.0,
        success_rate=1.0,
    )
    baseline = Baseline(
        version="v1.0",
        backend="da3",
        quality_tier="standard",
        environment=EnvironmentMetadata(python="3.11", torch="2.0", device="cpu", os="Linux"),
        statistics=baseline_stats,
        captured_at="2026-01-01T00:00:00Z",
    )

    # Current run is 15% slower on p95
    current_stats = Statistics(
        count=10,
        mean_sec=10.0,
        median_sec=10.0,
        p90_sec=12.0,
        p95_sec=14.95,  # 15% worse than baseline
        min_sec=8.0,
        max_sec=17.0,
        success_rate=1.0,
    )

    regressions = detect_regressions(baseline, current_stats, DEFAULT_REGRESSION_THRESHOLDS)

    assert len(regressions) == 1
    assert regressions[0].metric == "p95_sec"
    assert regressions[0].status == "regression"


def test_detect_regressions_mean_threshold():
    """Test mean regression detection."""
    baseline_stats = Statistics(
        count=10,
        mean_sec=10.0,
        median_sec=10.0,
        p90_sec=12.0,
        p95_sec=13.0,
        min_sec=8.0,
        max_sec=15.0,
        success_rate=1.0,
    )
    baseline = Baseline(
        version="v1.0",
        backend="da3",
        quality_tier="standard",
        environment=EnvironmentMetadata(python="3.11", torch="2.0", device="cpu", os="Linux"),
        statistics=baseline_stats,
        captured_at="2026-01-01T00:00:00Z",
    )

    # Current run is 20% slower on mean
    current_stats = Statistics(
        count=10,
        mean_sec=12.0,  # 20% worse
        median_sec=10.0,
        p90_sec=12.0,
        p95_sec=13.0,
        min_sec=8.0,
        max_sec=15.0,
        success_rate=1.0,
    )

    regressions = detect_regressions(baseline, current_stats, DEFAULT_REGRESSION_THRESHOLDS)

    assert len(regressions) == 1
    assert regressions[0].metric == "mean_sec"


def test_detect_regressions_failure_rate():
    """Test failure rate regression detection."""
    baseline_stats = Statistics(
        count=10,
        mean_sec=10.0,
        median_sec=10.0,
        p90_sec=12.0,
        p95_sec=13.0,
        min_sec=8.0,
        max_sec=15.0,
        success_rate=1.0,
    )
    baseline = Baseline(
        version="v1.0",
        backend="da3",
        quality_tier="standard",
        environment=EnvironmentMetadata(python="3.11", torch="2.0", device="cpu", os="Linux"),
        statistics=baseline_stats,
        captured_at="2026-01-01T00:00:00Z",
    )

    # Current run has failures
    current_stats = Statistics(
        count=10,
        mean_sec=10.0,
        median_sec=10.0,
        p90_sec=12.0,
        p95_sec=13.0,
        min_sec=8.0,
        max_sec=15.0,
        success_rate=0.9,  # 10% failure rate
    )

    regressions = detect_regressions(baseline, current_stats, DEFAULT_REGRESSION_THRESHOLDS)

    assert len(regressions) == 1
    assert regressions[0].metric == "success_rate"


def test_detect_regressions_no_regressions():
    """Test no regressions when performance is same or better."""
    baseline_stats = Statistics(
        count=10,
        mean_sec=10.0,
        median_sec=10.0,
        p90_sec=12.0,
        p95_sec=13.0,
        min_sec=8.0,
        max_sec=15.0,
        success_rate=1.0,
    )
    baseline = Baseline(
        version="v1.0",
        backend="da3",
        quality_tier="standard",
        environment=EnvironmentMetadata(python="3.11", torch="2.0", device="cpu", os="Linux"),
        statistics=baseline_stats,
        captured_at="2026-01-01T00:00:00Z",
    )

    # Current run is faster
    current_stats = Statistics(
        count=10,
        mean_sec=9.0,
        median_sec=9.0,
        p90_sec=11.0,
        p95_sec=12.0,
        min_sec=7.0,
        max_sec=14.0,
        success_rate=1.0,
    )

    regressions = detect_regressions(baseline, current_stats, DEFAULT_REGRESSION_THRESHOLDS)

    assert len(regressions) == 0


def test_format_markdown_with_regressions():
    """Test markdown report generation with regressions."""
    baseline_stats = Statistics(
        count=10,
        mean_sec=10.0,
        median_sec=10.0,
        p90_sec=12.0,
        p95_sec=13.0,
        min_sec=8.0,
        max_sec=15.0,
        success_rate=1.0,
    )
    baseline = Baseline(
        version="v1.0",
        backend="da3",
        quality_tier="standard",
        environment=EnvironmentMetadata(python="3.11", torch="2.0", device="cpu", os="Linux"),
        statistics=baseline_stats,
        captured_at="2026-01-01T00:00:00Z",
    )

    current_stats = Statistics(
        count=10,
        mean_sec=12.0,
        median_sec=11.0,
        p90_sec=14.0,
        p95_sec=15.0,
        min_sec=9.0,
        max_sec=18.0,
        success_rate=1.0,
    )

    regressions = [
        Regression(
            metric="mean_sec",
            baseline=10.0,
            current=12.0,
            change_pct=20.0,
            threshold_pct=15.0,
            status="regression",
        )
    ]

    env = EnvironmentMetadata(python="3.11", torch="2.0", device="cpu", os="Linux")
    report = format_markdown(baseline, current_stats, regressions, env)

    assert "Performance Comparison Report" in report
    assert "DO NOT MERGE" in report
    assert "mean_sec regression" in report
    assert "⚠️" in report


def test_format_markdown_without_regressions():
    """Test markdown report generation without regressions."""
    baseline_stats = Statistics(
        count=10,
        mean_sec=10.0,
        median_sec=10.0,
        p90_sec=12.0,
        p95_sec=13.0,
        min_sec=8.0,
        max_sec=15.0,
        success_rate=1.0,
    )
    baseline = Baseline(
        version="v1.0",
        backend="da3",
        quality_tier="standard",
        environment=EnvironmentMetadata(python="3.11", torch="2.0", device="cpu", os="Linux"),
        statistics=baseline_stats,
        captured_at="2026-01-01T00:00:00Z",
    )

    current_stats = Statistics(
        count=10,
        mean_sec=9.0,
        median_sec=9.0,
        p90_sec=11.0,
        p95_sec=12.0,
        min_sec=7.0,
        max_sec=14.0,
        success_rate=1.0,
    )

    env = EnvironmentMetadata(python="3.11", torch="2.0", device="cpu", os="Linux")
    report = format_markdown(baseline, current_stats, [], env)

    assert "Performance Comparison Report" in report
    assert "OK TO MERGE" in report
    assert "✅" in report


def test_baseline_serialization_roundtrip(tmp_path):
    """Test baseline save/load roundtrip."""
    baseline_path = tmp_path / "baseline.json"

    stats = Statistics(
        count=10,
        mean_sec=10.0,
        median_sec=10.0,
        p90_sec=12.0,
        p95_sec=13.0,
        min_sec=8.0,
        max_sec=15.0,
        success_rate=1.0,
        total_sec=100.0,
    )
    env = EnvironmentMetadata(python="3.11", torch="2.0", device="cpu", os="Linux")
    baseline = Baseline(
        version="v1.0",
        backend="da3",
        quality_tier="standard",
        environment=env,
        statistics=stats,
        captured_at="2026-01-01T00:00:00Z",
        notes="Test baseline",
    )

    save_baseline(baseline, baseline_path)
    loaded = load_baseline(baseline_path)

    assert loaded.version == baseline.version
    assert loaded.backend == baseline.backend
    assert loaded.statistics.mean_sec == baseline.statistics.mean_sec
    assert loaded.environment.python == baseline.environment.python


def test_load_baseline_not_found():
    """Test loading non-existent baseline raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="Baseline not found"):
        load_baseline(Path("/nonexistent/baseline.json"))
