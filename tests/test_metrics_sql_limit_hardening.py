"""Security hardening tests for SQL LIMIT handling."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from transformation_portal.metrics.comparator import query_baseline_stats
from transformation_portal.metrics.ledger import PerformanceLedger
from transformation_portal.metrics.performance_capsule import PerformanceCapsule
from transformation_portal.metrics.sql_safety import SQL_LIMIT_MAX, normalize_query_limit


def _make_capsule(image_id: str, captured_at: str) -> PerformanceCapsule:
    return PerformanceCapsule(
        image_id=image_id,
        image_path=f"/tmp/{image_id}.jpg",
        input_hash=f"hash-{image_id}",
        original_shape=(3000, 4000),
        enforced_shape=(3000, 4000),
        pixel_count=12_000_000,
        dimension_adjustment="exact",
        timings={"total": 1.5},
        scene_type="pool",
        backend_id="da3",
        device="mps",
        firewall_status="pass",
        captured_at=captured_at,
    )


def _seed_capsules(db_path: Path, count: int) -> None:
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    rows = []
    for idx in range(count):
        capsule = _make_capsule(
            image_id=f"image-{idx}",
            captured_at=(base + timedelta(seconds=idx)).isoformat(),
        )
        rows.append(
            (
                capsule.image_id,
                capsule.captured_at,
                capsule.scene_type,
                capsule.device,
                capsule.backend_id,
                capsule.pixel_count,
                capsule.timings["total"],
                capsule.firewall_status,
                capsule.workflow_version,
                capsule.zone,
                json.dumps(capsule.to_dict()),
                capsule.schema_version,
            )
        )

    with sqlite3.connect(db_path) as conn:
        conn.executemany(
            """
            INSERT INTO performance_capsules (
                image_id, captured_at, scene_type, device, backend_id,
                pixel_count, total_sec, firewall_status, workflow_version, zone,
                capsule_json, schema_version
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        conn.commit()


def _seed_apex_runs(db_path: Path) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO apex_runs (
                run_id, commit_sha, timestamp, workflow_version,
                zone, bucket_name, p50, p95, p99, count,
                threshold_p50, threshold_p95, pass_fail
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "run-old",
                "abc123",
                "2026-01-01T00:00:00+00:00",
                "v1",
                "local",
                "pool_medium_mps",
                1.0,
                1.4,
                1.6,
                10,
                1.5,
                2.0,
                "pass",
            ),
        )
        conn.execute(
            """
            INSERT INTO apex_runs (
                run_id, commit_sha, timestamp, workflow_version,
                zone, bucket_name, p50, p95, p99, count,
                threshold_p50, threshold_p95, pass_fail
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "run-new",
                "def456",
                "2026-01-02T00:00:00+00:00",
                "v1",
                "local",
                "pool_medium_mps",
                1.1,
                1.5,
                1.7,
                10,
                1.5,
                2.0,
                "pass",
            ),
        )
        conn.commit()


def test_normalize_query_limit_clamps_to_max() -> None:
    assert normalize_query_limit(None) is None
    assert normalize_query_limit(5) == 5
    assert normalize_query_limit(SQL_LIMIT_MAX + 1) == SQL_LIMIT_MAX


@pytest.mark.parametrize("invalid_limit", [0, -1, 1.5, "10", True])
def test_normalize_query_limit_rejects_invalid_values(invalid_limit: object) -> None:
    with pytest.raises(ValueError, match="limit"):
        normalize_query_limit(invalid_limit)  # type: ignore[arg-type]


def test_query_capsules_clamps_limit_to_bounded_range(tmp_path: Path) -> None:
    db_path = tmp_path / "ledger.db"
    ledger = PerformanceLedger(db_path)
    _seed_capsules(db_path, SQL_LIMIT_MAX + 25)

    results = ledger.query_capsules(limit=SQL_LIMIT_MAX + 25)

    assert len(results) == SQL_LIMIT_MAX


@pytest.mark.parametrize("invalid_limit", [0, -1, 1.5, "1", True])
def test_query_capsules_rejects_non_integer_limits(tmp_path: Path, invalid_limit: object) -> None:
    db_path = tmp_path / "ledger.db"
    ledger = PerformanceLedger(db_path)
    _seed_capsules(db_path, 1)

    with pytest.raises(ValueError, match="limit"):
        ledger.query_capsules(limit=invalid_limit)  # type: ignore[arg-type]


def test_query_baseline_stats_accepts_large_limit_with_bounding(tmp_path: Path) -> None:
    db_path = tmp_path / "baseline.db"
    PerformanceLedger(db_path)
    _seed_apex_runs(db_path)

    stats = query_baseline_stats(
        ledger_db_path=str(db_path),
        workflow_version="v1",
        zone="local",
        limit=SQL_LIMIT_MAX + 500,
    )

    assert "pool_medium_mps" in stats
    assert stats["pool_medium_mps"].p95 == 1.5


@pytest.mark.parametrize("invalid_limit", [0, -1, 1.5, "1", True, None])
def test_query_baseline_stats_rejects_invalid_limits(tmp_path: Path, invalid_limit: object) -> None:
    db_path = tmp_path / "baseline.db"
    PerformanceLedger(db_path)
    _seed_apex_runs(db_path)

    with pytest.raises(ValueError, match="limit"):
        query_baseline_stats(
            ledger_db_path=str(db_path),
            workflow_version="v1",
            zone="local",
            limit=invalid_limit,  # type: ignore[arg-type]
        )
