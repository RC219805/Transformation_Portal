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


def _write_manifest(path: Path, *, identifier: str, duration: float | None, ok: bool, backend: str, error: str | None = None) -> None:
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
    if not ok and error:
        payload["v2"]["error_message"] = error
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _make_manifests(dirpath: Path, *, durations: List[float], backend: str, fail_count: int = 0, error: str = "TimeoutError: timed out") -> None:
    dirpath.mkdir(parents=True, exist_ok=True)

    # successes
    for i, d in enumerate(durations):
        _write_manifest(dirpath / f"ok_{i:04d}.json", identifier=f"img_{i:04d}.jpg", duration=d, ok=True, backend=backend)

    # failures (no timings needed)
    for j in range(fail_count):
        _write_manifest(dirpath / f"fail_{j:04d}.json", identifier=f"bad_{j:04d}.jpg", duration=None, ok=False, backend=backend, error=error)


def _run_ledger(args: List[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    cmd = [sys.executable, str(_ledger_script()), *args]
    return subprocess.run(cmd, cwd=str(cwd), text=True, capture_output=True)


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
        ["--manifests-dir", str(baseline_dir), "--output", str(baseline_json), "--baseline-version", "test", "--quality-tier", "unit"],
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
            "--bootstrap-iterations",
            "300",
        ],
        cwd=_repo_root(),
    )
    assert r2.returncode == 0, f"compare failed: {r2.stderr}\n{r2.stdout}"
    assert report_md.exists()
    assert report_json.exists()

    out = json.loads(report_json.read_text(encoding="utf-8"))
    assert out["status"] in {"ok", "potential_regression"}, out
    assert out["exit_code"] == 0, out


def test_detects_significant_regression(tmp_path: Path) -> None:
    baseline_dir = tmp_path / "baseline_manifests"
    current_dir = tmp_path / "current_manifests"
    baseline_json = tmp_path / "baseline.json"
    report_md = tmp_path / "report.md"
    report_json = tmp_path / "report.json"

    # Baseline: 1.0s, Current: 1.35s => +35% (mean + p95 should exceed thresholds)
    _make_manifests(baseline_dir, durations=[1.0] * 40, backend="da3", fail_count=0)
    _make_manifests(current_dir, durations=[1.35] * 40, backend="da3", fail_count=0)

    r1 = _run_ledger(["--manifests-dir", str(baseline_dir), "--output", str(baseline_json)], cwd=_repo_root())
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
            "--bootstrap-iterations",
            "400",
            "--confidence-level",
            "0.95",
        ],
        cwd=_repo_root(),
    )

    # Should fail with exit code 1 due to significant regression
    assert r2.returncode == 1, f"expected regression exit=1, got={r2.returncode}\n{r2.stderr}\n{r2.stdout}"

    out = json.loads(report_json.read_text(encoding="utf-8"))
    assert out["exit_code"] == 1
    assert out["status"] == "regression"
    assert len(out["significant_regressions"]) >= 1


def test_backend_mismatch_exit_2(tmp_path: Path) -> None:
    baseline_dir = tmp_path / "baseline_manifests"
    current_dir = tmp_path / "current_manifests"
    baseline_json = tmp_path / "baseline.json"
    report_md = tmp_path / "report.md"
    report_json = tmp_path / "report.json"

    _make_manifests(baseline_dir, durations=[1.0] * 10, backend="da3")
    _make_manifests(current_dir, durations=[1.0] * 10, backend="depth_pro")

    r1 = _run_ledger(["--manifests-dir", str(baseline_dir), "--output", str(baseline_json)], cwd=_repo_root())
    assert r1.returncode == 0

    r2 = _run_ledger(
        ["--baseline", str(baseline_json), "--compare", str(current_dir), "--output", str(report_md), "--emit-json", str(report_json)],
        cwd=_repo_root(),
    )
    assert r2.returncode == 2, f"expected backend mismatch exit=2, got={r2.returncode}\n{r2.stderr}\n{r2.stdout}"

    out = json.loads(report_json.read_text(encoding="utf-8"))
    assert out["exit_code"] == 2
    assert out["status"] == "backend_mismatch"


def test_failure_rate_regression_takes_precedence_over_insufficient_latency(tmp_path: Path) -> None:
    baseline_dir = tmp_path / "baseline_manifests"
    current_dir = tmp_path / "current_manifests"
    baseline_json = tmp_path / "baseline.json"
    report_md = tmp_path / "report.md"
    report_json = tmp_path / "report.json"

    # Create a baseline with timings, but current with mostly failures and very few timings.
    _make_manifests(baseline_dir, durations=[1.0] * 20, backend="da3", fail_count=0)
    _make_manifests(current_dir, durations=[1.0] * 2, backend="da3", fail_count=10, error="TimeoutError: timed out")

    r1 = _run_ledger(["--manifests-dir", str(baseline_dir), "--output", str(baseline_json)], cwd=_repo_root())
    assert r1.returncode == 0

    # Require min samples=5 for latency significance; we only have 2 current timings.
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
            "--min-samples",
            "5",
        ],
        cwd=_repo_root(),
    )

    # Even though latency significance is insufficient, failure-rate regression is significant -> exit 1.
    assert r2.returncode == 1, f"expected regression exit=1, got={r2.returncode}\n{r2.stderr}\n{r2.stdout}"
    out = json.loads(report_json.read_text(encoding="utf-8"))
    assert out["exit_code"] == 1
    assert out["status"] == "regression"
