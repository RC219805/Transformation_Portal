"""Regression tests for ``scripts/ci/track_test_flakes.py`` and ``scripts/ci/analyze_flakes.py``.

Both CLIs document a ``--ledger`` flag; it must select the ledger that is
read and written instead of silently falling back to ``tests/flake_ledger.json``.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.regression]

REPO_ROOT = Path(__file__).resolve().parents[1]
TRACK = REPO_ROOT / "scripts" / "ci" / "track_test_flakes.py"
ANALYZE = REPO_ROOT / "scripts" / "ci" / "analyze_flakes.py"
DEFAULT_LEDGER = REPO_ROOT / "tests" / "flake_ledger.json"


def _run(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, *args], capture_output=True, text=True, timeout=60, check=False)


def test_track_writes_the_ledger_named_by_flag_and_leaves_default_untouched(tmp_path: Path) -> None:
    report = tmp_path / "report.json"
    report.write_text(
        json.dumps({"tests": [{"nodeid": "tests/test_x.py::test_a", "outcome": "failed", "call": {"duration": 0.1}}]}),
        encoding="utf-8",
    )
    custom = tmp_path / "nested" / "custom_ledger.json"
    default_before = DEFAULT_LEDGER.read_bytes() if DEFAULT_LEDGER.exists() else None

    proc = _run([str(TRACK), str(report), "--ledger", str(custom), "--quiet"])

    assert proc.returncode == 0, proc.stderr
    assert custom.exists()
    ledger = json.loads(custom.read_text(encoding="utf-8"))
    assert ledger["tests"]["tests/test_x.py::test_a"]["total_runs"] == 1
    default_after = DEFAULT_LEDGER.read_bytes() if DEFAULT_LEDGER.exists() else None
    assert default_after == default_before

    # A second run must accumulate into the same custom ledger.
    proc = _run([str(TRACK), str(report), "--ledger", str(custom), "--quiet"])
    assert proc.returncode == 0, proc.stderr
    ledger = json.loads(custom.read_text(encoding="utf-8"))
    assert ledger["tests"]["tests/test_x.py::test_a"]["total_runs"] == 2


def test_analyze_reads_the_ledger_named_by_flag(tmp_path: Path) -> None:
    custom = tmp_path / "custom_ledger.json"
    custom.write_text(
        json.dumps(
            {
                "config": {},
                "tests": {
                    "t": {
                        "status": "quarantined",
                        "total_runs": 10,
                        "flake_count": 5,
                        "flake_rate": 0.5,
                        "test_id": "t",
                        "last_outcome": "failed",
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    proc = _run([str(ANALYZE), "--format", "json", "--ledger", str(custom)])

    assert proc.returncode == 0, proc.stderr
    stats = json.loads(proc.stdout)
    assert stats["total_tests"] == 1
    assert stats["quarantined"] == 1
    assert stats["total_runs"] == 10
