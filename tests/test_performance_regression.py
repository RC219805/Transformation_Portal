"""Performance regression benchmarks for the performance ledger tool.

These are *benchmarked* regression checks (not unit tests):
- They execute the performance ledger end-to-end via subprocess.
- They generate synthetic manifests so they do not require ML models.
- CI should skip these by default (marker: benchmark).

Run manually:
    pytest -m benchmark -v tests/test_performance_regression.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any, List

import pytest


pytestmark = pytest.mark.benchmark


def _repo_root() -> Path:
    # tests/ -> repo root
    return Path(__file__).resolve().parents[1]


def _ledger_script() -> Path:
    return _repo_root() / "tools" / "performance_ledger.py"


def _write_manifest(
    path: Path,
    *,
    identifier: str,
    duration: float | None,
    ok: bool,
    backend: str,
    error: str | None = None,
) -> None:
    payload: Dict[str, Any] = {
        "identifier": identifier,
        "backend_selection": {
            "requested_backend": backend,
            "resolved_backend": backend,
            "resolution_status": "ok",
        },
        "v2": {"status": "ok" if ok else "failed"},
    }
    if duration is not None:
        payload["timing"] = {"total_seconds": float(duration)}
    # Tool checks for "depth" presence to determine success (line 158 in performance_ledger.py)
    if ok:
        payload["depth"] = {"status": "ok", "backend": backend}
    if not ok and error:
        payload["v2"]["error_message"] = error
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _make_manifests(
    dirpath: Path,
    *,
    durations: List[float],
    backend: str,
    fail_count: int = 0,
    fail_duration: float = 1.0,
    error: str = "TimeoutError: timed out",
) -> None:
    dirpath.mkdir(parents=True, exist_ok=True)

    # successes
    for i, d in enumerate(durations):
        _write_manifest(dirpath / f"ok_{i:04d}.json", identifier=f"img_{i:04d}.jpg", duration=d, ok=True, backend=backend)

    # failures (with timings - failed runs still have durations)
    for j in range(fail_count):
        _write_manifest(
            dirpath / f"fail_{j:04d}.json",
            identifier=f"bad_{j:04d}.jpg",
            duration=fail_duration,
            ok=False,
            backend=backend,
            error=error,
        )


def _run_ledger(args: List[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    cmd = [sys.executable, str(_ledger_script()), *args]
    return subprocess.run(cmd, cwd=str(cwd), text=True, capture_output=True, timeout=300)


def test_capture_and_compare_ok(tmp_path: Path) -> None:
    baseline_dir = tmp_path / "baseline_manifests"
    current_dir = tmp_path / "current_manifests"
    baseline_json = tmp_path / "baseline.json"
    report_md = tmp_path / "report.md"
    report_json = tmp_path / "report.json"

    _make_manifests(baseline_dir, durations=[1.0] * 30, backend="da3", fail_count=0)
    _make_manifests(current_dir, durations=[0.95] * 30, backend="da3", fail_count=0)

    # Capture baseline
    r1 = _run_ledger(
        [
            "--manifests-dir",
            str(baseline_dir),
            "--output",
            str(baseline_json),
            "--version",
            "test",
            "--quality-tier",
            "unit",
        ],
        cwd=_repo_root(),
    )
    assert r1.returncode == 0, f"capture failed: {r1.stderr}\n{r1.stdout}"
    assert baseline_json.exists()

    # Compare
    r2 = _run_ledger(
        [
            "--baseline",
            str(baseline_json),
            "--compare",
            str(current_dir),
            "--output",
            str(report_md),
            "--emit-json",
            str(report_json),
        ],
        cwd=_repo_root(),
    )
    assert r2.returncode == 0, f"compare failed: {r2.stderr}\n{r2.stdout}"
    assert report_md.exists()
    assert report_json.exists()

    # Emit-json contains Baseline schema
    out = json.loads(report_json.read_text(encoding="utf-8"))
    assert "statistics" in out
    assert "environment" in out


def test_detects_significant_regression(tmp_path: Path) -> None:
    baseline_dir = tmp_path / "baseline_manifests"
    current_dir = tmp_path / "current_manifests"
    baseline_json = tmp_path / "baseline.json"
    report_md = tmp_path / "report.md"
    report_json = tmp_path / "report.json"

    # Baseline: 1.0s, Current: 1.35s => +35% (mean + p95 should exceed thresholds)
    _make_manifests(baseline_dir, durations=[1.0] * 40, backend="da3", fail_count=0)
    _make_manifests(current_dir, durations=[1.35] * 40, backend="da3", fail_count=0)

    r1 = _run_ledger(
        ["--manifests-dir", str(baseline_dir), "--output", str(baseline_json)],
        cwd=_repo_root(),
    )
    assert r1.returncode == 0, f"capture failed: {r1.stderr}\n{r1.stdout}"

    r2 = _run_ledger(
        [
            "--baseline",
            str(baseline_json),
            "--compare",
            str(current_dir),
            "--output",
            str(report_md),
            "--emit-json",
            str(report_json),
        ],
        cwd=_repo_root(),
    )

    # Should fail with exit code 1 due to significant regression
    assert r2.returncode == 1, (
        f"expected regression exit=1, got={r2.returncode}\n"
        f"{r2.stderr}\n{r2.stdout}"
    )
    assert report_md.exists()

    # Emit-json contains current stats in Baseline format
    assert report_json.exists()
    out = json.loads(report_json.read_text(encoding="utf-8"))
    assert "statistics" in out
    assert out["statistics"]["mean_sec"] > 1.3


def test_backend_mismatch_warning_in_report(tmp_path: Path) -> None:
    """Test that tool can handle different backends."""
    baseline_dir = tmp_path / "baseline_manifests"
    current_dir = tmp_path / "current_manifests"
    baseline_json = tmp_path / "baseline.json"
    report_md = tmp_path / "report.md"
    report_json = tmp_path / "report.json"

    _make_manifests(baseline_dir, durations=[1.0] * 10, backend="da3")
    _make_manifests(current_dir, durations=[1.0] * 10, backend="depth_pro")

    r1 = _run_ledger(
        [
            "--manifests-dir",
            str(baseline_dir),
            "--output",
            str(baseline_json),
            "--backend",
            "da3",
        ],
        cwd=_repo_root(),
    )
    assert r1.returncode == 0

    # Backend mismatch is not enforced by the tool (current contract)
    r2 = _run_ledger(
        [
            "--baseline",
            str(baseline_json),
            "--compare",
            str(current_dir),
            "--output",
            str(report_md),
            "--emit-json",
            str(report_json),
            "--backend",
            "depth_pro",
        ],
        cwd=_repo_root(),
    )
    assert r2.returncode == 0
    assert report_md.exists()


def test_failure_rate_regression(tmp_path: Path) -> None:
    """Test that increased failure rate triggers regression detection."""
    baseline_dir = tmp_path / "baseline_manifests"
    current_dir = tmp_path / "current_manifests"
    baseline_json = tmp_path / "baseline.json"
    report_md = tmp_path / "report.md"
    report_json = tmp_path / "report.json"

    # Create a baseline with no failures, current with high failure rate
    _make_manifests(baseline_dir, durations=[1.0] * 20, backend="da3", fail_count=0)
    _make_manifests(
        current_dir,
        durations=[1.0] * 10,
        backend="da3",
        fail_count=10,
        error="TimeoutError: timed out",
    )

    r1 = _run_ledger(
        ["--manifests-dir", str(baseline_dir), "--output", str(baseline_json)],
        cwd=_repo_root(),
    )
    assert r1.returncode == 0

    r2 = _run_ledger(
        [
            "--baseline",
            str(baseline_json),
            "--compare",
            str(current_dir),
            "--output",
            str(report_md),
            "--emit-json",
            str(report_json),
        ],
        cwd=_repo_root(),
    )

    # Failure rate regression detected
    assert r2.returncode == 1
    assert report_md.exists()

    # Verify degraded success rate
    out = json.loads(report_json.read_text(encoding="utf-8"))
    assert "statistics" in out
    assert out["statistics"]["success_rate"] < 0.6
