"""Behavioral coverage for the performance ledger (``metrics/ledger.py``).

These tests are deterministic and dependency-light: a temporary SQLite file,
synthetic :class:`PerformanceCapsule` rows, and direct invocation of the
``main()`` CLI via ``sys.argv``. No ML runtimes, no network, no clock skew
(timestamps are explicit). They exercise the ledger lifecycle (schema +
migration), the query filter matrix, statistics/pruning, regression detection
across bucket thresholds, report generation, and every CLI subcommand
(``log``/``query``/``regression``/``report``/``prune``) plus the error path.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import pytest

pytestmark = pytest.mark.unit

from transformation_portal.metrics import ledger as ledger_module
from transformation_portal.metrics.ledger import (
    PerformanceLedger,
    SCHEMA_VERSION,
    detect_regression,
    generate_performance_report,
    main,
)
from transformation_portal.metrics.performance_capsule import (
    PerformanceBucket,
    PerformanceCapsule,
)

_BASE_TS = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _make_capsule(
    image_id: str = "img",
    *,
    total_sec: float = 1.5,
    scene_type: Optional[str] = "pool",
    device: str = "mps",
    backend_id: str = "da3",
    firewall_status: str = "pass",
    workflow_version: str = "v1",
    zone: Optional[str] = None,
    pixel_count: int = 12_000_000,
    captured_at: Optional[str] = None,
) -> PerformanceCapsule:
    """Build a valid capsule with explicit, deterministic fields."""
    return PerformanceCapsule(
        image_id=image_id,
        image_path=f"/tmp/{image_id}.jpg",
        input_hash=f"hash-{image_id}",
        original_shape=(3000, 4000),
        enforced_shape=(3000, 4000),
        pixel_count=pixel_count,
        dimension_adjustment="exact",
        timings={"total": total_sec},
        scene_type=scene_type,
        backend_id=backend_id,
        device=device,
        firewall_status=firewall_status,
        workflow_version=workflow_version,
        zone=zone,
        captured_at=captured_at or _BASE_TS.isoformat(),
    )


def _write_capsule_json(path: Path, capsule: PerformanceCapsule) -> Path:
    path.write_text(json.dumps(capsule.to_dict()))
    return path


# --------------------------------------------------------------------------- #
# Schema lifecycle
# --------------------------------------------------------------------------- #


def test_new_ledger_creates_schema_and_reports_ready(tmp_path: Path) -> None:
    db = tmp_path / "perf.db"
    led = PerformanceLedger(db)

    assert db.is_file()
    assert led.ensure_ready() is True

    with sqlite3.connect(db) as conn:
        (version,) = conn.execute("SELECT version FROM schema_version").fetchone()
    assert version == SCHEMA_VERSION


def test_reopening_existing_ledger_is_idempotent(tmp_path: Path) -> None:
    db = tmp_path / "perf.db"
    PerformanceLedger(db).log_capsule(_make_capsule("a"))

    # Re-open: schema_version table already exists and is current → no migration,
    # existing rows are preserved.
    reopened = PerformanceLedger(db)
    assert len(reopened.query_capsules()) == 1


def test_schema_migration_from_v1_is_idempotent(tmp_path: Path) -> None:
    db = tmp_path / "perf.db"
    PerformanceLedger(db)  # establish current schema

    # Pretend this database was written by an older v1 build.
    with sqlite3.connect(db) as conn:
        conn.execute("UPDATE schema_version SET version = 1")
        conn.commit()

    # Re-opening must run the v1->current migration branch without raising
    # (the ALTER TABLE adds columns that already exist; the error is swallowed).
    migrated = PerformanceLedger(db)
    assert migrated.ensure_ready() is True
    with sqlite3.connect(db) as conn:
        (version,) = conn.execute("SELECT version FROM schema_version").fetchone()
    assert version == SCHEMA_VERSION


def test_ensure_ready_returns_false_when_table_missing(tmp_path: Path) -> None:
    db = tmp_path / "perf.db"
    led = PerformanceLedger(db)

    # Drop the table ensure_ready() probes; it should fail closed, not raise.
    with sqlite3.connect(db) as conn:
        conn.execute("DROP TABLE apex_runs")
        conn.commit()

    assert led.ensure_ready() is False


# --------------------------------------------------------------------------- #
# Logging + query filter matrix
# --------------------------------------------------------------------------- #


def test_log_and_query_roundtrips_capsule(tmp_path: Path) -> None:
    led = PerformanceLedger(tmp_path / "perf.db")
    led.log_capsule(_make_capsule("img-1", total_sec=2.0))

    results = led.query_capsules()
    assert len(results) == 1
    assert results[0].image_id == "img-1"
    assert results[0].timings["total"] == 2.0


def test_query_orders_by_captured_at_descending(tmp_path: Path) -> None:
    led = PerformanceLedger(tmp_path / "perf.db")
    for idx in range(3):
        led.log_capsule(
            _make_capsule(f"img-{idx}", captured_at=(_BASE_TS + timedelta(hours=idx)).isoformat())
        )

    ordered = [c.image_id for c in led.query_capsules()]
    assert ordered == ["img-2", "img-1", "img-0"]


def test_query_filters_match_every_supported_column(tmp_path: Path) -> None:
    led = PerformanceLedger(tmp_path / "perf.db")
    wanted = _make_capsule(
        "wanted",
        scene_type="interior",
        device="cuda",
        backend_id="depth_pro",
        firewall_status="warn",
        workflow_version="v2",
        zone="us-east",
        captured_at=(_BASE_TS + timedelta(days=1)).isoformat(),
    )
    led.log_capsule(wanted)
    # A row that differs in every filterable dimension.
    led.log_capsule(
        _make_capsule(
            "other",
            scene_type="aerial",
            device="mps",
            backend_id="da3",
            firewall_status="pass",
            workflow_version="v1",
            zone="eu-west",
            captured_at=(_BASE_TS + timedelta(days=5)).isoformat(),
        )
    )

    hits = led.query_capsules(
        scene_type="interior",
        device="cuda",
        backend_id="depth_pro",
        firewall_status="warn",
        workflow_version="v2",
        zone="us-east",
        min_captured_at=(_BASE_TS + timedelta(hours=12)).isoformat(),
        max_captured_at=(_BASE_TS + timedelta(days=2)).isoformat(),
    )
    assert [c.image_id for c in hits] == ["wanted"]


def test_query_limit_is_respected(tmp_path: Path) -> None:
    led = PerformanceLedger(tmp_path / "perf.db")
    for idx in range(5):
        led.log_capsule(
            _make_capsule(f"img-{idx}", captured_at=(_BASE_TS + timedelta(minutes=idx)).isoformat())
        )

    assert len(led.query_capsules(limit=2)) == 2


# --------------------------------------------------------------------------- #
# Statistics + pruning
# --------------------------------------------------------------------------- #


def test_statistics_empty_returns_zero_count(tmp_path: Path) -> None:
    led = PerformanceLedger(tmp_path / "perf.db")
    assert led.get_statistics() == {"count": 0}


def test_statistics_computes_percentiles(tmp_path: Path) -> None:
    led = PerformanceLedger(tmp_path / "perf.db")
    for idx, total in enumerate([1.0, 2.0, 3.0, 4.0]):
        led.log_capsule(
            _make_capsule(f"img-{idx}", total_sec=total, captured_at=(_BASE_TS + timedelta(minutes=idx)).isoformat())
        )

    stats = led.get_statistics()
    assert stats["count"] == 4
    assert stats["min_sec"] == 1.0
    assert stats["max_sec"] == 4.0
    assert stats["mean_sec"] == pytest.approx(2.5)


def test_statistics_single_capsule_uses_p95_fallback(tmp_path: Path) -> None:
    led = PerformanceLedger(tmp_path / "perf.db")
    led.log_capsule(_make_capsule("only", total_sec=3.0))

    stats = led.get_statistics()
    # With n == 1 the p95 branch falls back to the single sample.
    assert stats["count"] == 1
    assert stats["p95_sec"] == 3.0


def test_prune_old_entries_deletes_only_stale_rows(tmp_path: Path) -> None:
    led = PerformanceLedger(tmp_path / "perf.db")
    old_ts = (datetime.now(timezone.utc) - timedelta(days=200)).isoformat()
    fresh_ts = datetime.now(timezone.utc).isoformat()
    led.log_capsule(_make_capsule("old", captured_at=old_ts))
    led.log_capsule(_make_capsule("fresh", captured_at=fresh_ts))

    deleted = led.prune_old_entries(days_to_keep=90)
    assert deleted == 1
    assert [c.image_id for c in led.query_capsules()] == ["fresh"]


# --------------------------------------------------------------------------- #
# Regression detection
# --------------------------------------------------------------------------- #


def _bucket(p50: float, p95: float, name: str = "test_bucket") -> PerformanceBucket:
    return PerformanceBucket(name=name, filters={}, p50_threshold_sec=p50, p95_threshold_sec=p95)


def test_detect_regression_insufficient_data() -> None:
    result = detect_regression(_make_capsule("cur"), [])
    assert result["status"] == "insufficient_data"


def test_detect_regression_pass_within_thresholds() -> None:
    bucket = _bucket(p50=10.0, p95=15.0)
    result = detect_regression(
        _make_capsule("cur", total_sec=8.0),
        [_make_capsule("h1", total_sec=7.0), _make_capsule("h2", total_sec=9.0)],
        bucket=bucket,
    )
    assert result["status"] == "pass"
    assert result["bucket"] == "test_bucket"
    assert result["current_total_sec"] == 8.0


def test_detect_regression_flags_p95_breach() -> None:
    bucket = _bucket(p50=10.0, p95=15.0)
    result = detect_regression(
        _make_capsule("cur", total_sec=20.0),
        [_make_capsule("h1", total_sec=9.0)],
        bucket=bucket,
    )
    assert result["status"] == "regression_p95"
    assert "p95" in result["message"]


def test_detect_regression_warns_above_p50_margin() -> None:
    # Bucket where p50*1.5 < p95 so the warning branch is reachable.
    bucket = _bucket(p50=10.0, p95=100.0)
    result = detect_regression(
        _make_capsule("cur", total_sec=20.0),  # > 15 (p50*1.5) but < 100 (p95)
        [_make_capsule("h1", total_sec=9.0)],
        bucket=bucket,
    )
    assert result["status"] == "warning_p50"


def test_detect_regression_infers_bucket_when_omitted() -> None:
    # scene_type/device/pixel_count chosen so only the lenient catch-all matches.
    cur = _make_capsule("cur", total_sec=1.0, scene_type="garage", device="cpu", pixel_count=1_000_000)
    result = detect_regression(cur, [_make_capsule("h1", total_sec=1.0)])
    assert result["status"] == "pass"
    assert result["bucket"] == "unknown"


# --------------------------------------------------------------------------- #
# Report generation
# --------------------------------------------------------------------------- #


def test_report_handles_empty_ledger(tmp_path: Path) -> None:
    led = PerformanceLedger(tmp_path / "perf.db")
    out = tmp_path / "report.md"
    generate_performance_report(led, out)
    assert "No data available" in out.read_text()


def test_report_summarizes_scenes_and_buckets(tmp_path: Path) -> None:
    led = PerformanceLedger(tmp_path / "perf.db")
    # Two scenes; pool rows land in the pool_medium_mps bucket (>=10MP, mps).
    led.log_capsule(_make_capsule("pool-1", total_sec=9.0, scene_type="pool"))
    led.log_capsule(_make_capsule("pool-2", total_sec=20.0, scene_type="pool"))
    led.log_capsule(_make_capsule("int-1", total_sec=5.0, scene_type="interior", pixel_count=8_000_000))

    out = tmp_path / "report.md"
    generate_performance_report(led, out)
    text = out.read_text()

    assert "Performance Analysis Report" in text
    assert "Pool Scenes" in text
    assert "Interior Scenes" in text
    assert "Performance Bucket Analysis" in text


# --------------------------------------------------------------------------- #
# CLI surface
# --------------------------------------------------------------------------- #


def _run_cli(monkeypatch: pytest.MonkeyPatch, *args: str) -> int:
    monkeypatch.setattr("sys.argv", ["ledger", *args])
    return main()


def test_cli_log_then_query_writes_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db = tmp_path / "perf.db"
    capsule_json = _write_capsule_json(tmp_path / "cap.json", _make_capsule("cli-img", total_sec=4.0))

    assert _run_cli(monkeypatch, "log", "--capsule", str(capsule_json), "--ledger-db", str(db)) == 0

    out = tmp_path / "out.json"
    rc = _run_cli(
        monkeypatch,
        "query",
        "--ledger-db", str(db),
        "--scene-type", "pool",
        "--device", "mps",
        "--limit", "10",
        "--output", str(out),
    )
    assert rc == 0
    payload = json.loads(out.read_text())
    assert [row["image_id"] for row in payload] == ["cli-img"]


def test_cli_query_with_min_days_filter(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db = tmp_path / "perf.db"
    led = PerformanceLedger(db)
    led.log_capsule(_make_capsule("recent", captured_at=datetime.now(timezone.utc).isoformat()))

    # --min-days exercises the cutoff-timestamp branch; just assert it succeeds.
    assert _run_cli(monkeypatch, "query", "--ledger-db", str(db), "--min-days", "7") == 0


def test_cli_regression_pass_returns_zero(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db = tmp_path / "perf.db"
    led = PerformanceLedger(db)
    # Historical row must fall inside the baseline window the CLI queries.
    recent = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
    led.log_capsule(_make_capsule("hist", total_sec=9.0, captured_at=recent))
    cap = _write_capsule_json(tmp_path / "cur.json", _make_capsule("cur", total_sec=9.5))

    rc = _run_cli(monkeypatch, "regression", "--ledger-db", str(db), "--capsule", str(cap), "--baseline-days", "30")
    assert rc == 0


def test_cli_regression_detected_returns_one(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db = tmp_path / "perf.db"
    led = PerformanceLedger(db)
    recent = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
    led.log_capsule(_make_capsule("hist", total_sec=9.0, captured_at=recent))
    # total well past the pool bucket p95 (15s) → regression → exit 1.
    cap = _write_capsule_json(tmp_path / "cur.json", _make_capsule("cur", total_sec=99.0))

    rc = _run_cli(monkeypatch, "regression", "--ledger-db", str(db), "--capsule", str(cap))
    assert rc == 1


def test_cli_report_and_prune(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db = tmp_path / "perf.db"
    led = PerformanceLedger(db)
    recent = datetime.now(timezone.utc).isoformat()
    led.log_capsule(_make_capsule("r1", total_sec=8.0, captured_at=recent))
    led.log_capsule(
        _make_capsule("old", captured_at=(datetime.now(timezone.utc) - timedelta(days=400)).isoformat())
    )

    report = tmp_path / "report.md"
    # --min-days exercises the cutoff branch; keep the recent row in-window.
    assert _run_cli(monkeypatch, "report", "--ledger-db", str(db), "--output", str(report), "--min-days", "7") == 0
    assert report.is_file()

    assert _run_cli(monkeypatch, "prune", "--ledger-db", str(db), "--days-to-keep", "90") == 0


def test_cli_returns_one_on_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Missing capsule file → the top-level try/except returns 1, not a traceback.
    rc = _run_cli(
        monkeypatch,
        "log",
        "--capsule", str(tmp_path / "does-not-exist.json"),
        "--ledger-db", str(tmp_path / "perf.db"),
    )
    assert rc == 1


def test_cli_requires_subcommand(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.argv", ["ledger"])
    with pytest.raises(SystemExit):
        main()


def test_module_exposes_main_entrypoint() -> None:
    assert callable(ledger_module.main)
