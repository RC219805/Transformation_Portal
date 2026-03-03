"""Enforce fast-ML collection boundary for reconstruction tests.

This contract prevents slow/integration creep into the PR fast-ML lane.

Baseline rationale (March 3, 2026):
- Pre-marker baseline: 72 selected tests for
  "ml and not slow and not integration and not benchmark"
- Post-marker baseline: 67 selected tests after marking
  tests/spatial_ai/reconstruction/test_integration_phase23.py as integration

Contract:
- fast-ML selected count must remain <= 67
- integration-marked Phase 2.3 tests must stay out of fast-ML selection

Layer-1 note:
- Enforcement jobs may not have torch installed.
- We create a tiny local torch stub on PYTHONPATH for collection-only subprocesses.
  This keeps marker-selection accounting meaningful without adding ML dependencies.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
FAST_ML_MARKEXPR = "ml and not slow and not integration and not benchmark"
INTEGRATION_MARKEXPR = "ml and integration and not slow and not benchmark"
FAST_ML_SELECTED_CEILING = 67

_RATIO_SUMMARY = re.compile(r"(?P<selected>\d+)/(?P<collected>\d+) tests collected")
_TOTAL_ONLY_SUMMARY = re.compile(r"(?P<selected>\d+) tests collected")
_NO_TESTS = re.compile(r"no tests collected")


def _write_torch_stub(stub_root: Path) -> None:
    """Write a minimal torch stub sufficient for pytest collection."""
    torch_dir = stub_root / "torch"
    nn_dir = torch_dir / "nn"
    nn_dir.mkdir(parents=True, exist_ok=True)

    (torch_dir / "__init__.py").write_text(
        '"""Minimal torch stub for collection-only contract tests."""\n'
        "\n"
        "class _Cuda:\n"
        "    @staticmethod\n"
        "    def is_available():\n"
        "        return False\n"
        "\n"
        "    @staticmethod\n"
        "    def device_count():\n"
        "        return 0\n"
        "\n"
        "    @staticmethod\n"
        "    def manual_seed_all(_seed):\n"
        "        return None\n"
        "\n"
        "\n"
        "class _MPS:\n"
        "    @staticmethod\n"
        "    def is_available():\n"
        "        return False\n"
        "\n"
        "\n"
        "class _Backends:\n"
        "    mps = _MPS()\n"
        "\n"
        "\n"
        "cuda = _Cuda()\n"
        "backends = _Backends()\n"
        "\n"
        "\n"
        "def manual_seed(_seed):\n"
        "    return None\n"
        "\n"
        "\n"
        "def set_num_threads(_n):\n"
        "    return None\n"
        "\n"
        "\n"
        "def set_num_interop_threads(_n):\n"
        "    return None\n",
        encoding="utf-8",
    )
    (nn_dir / "__init__.py").write_text('"""torch.nn stub."""\n', encoding="utf-8")
    (nn_dir / "functional.py").write_text('"""torch.nn.functional stub."""\n', encoding="utf-8")


def _run_collect(markexpr: str, target: str) -> subprocess.CompletedProcess[str]:
    """Run pytest --collect-only in a subprocess with torch stubbed."""
    stub_root = Path.cwd() / ".pytest_tmp_torch_stub"
    _write_torch_stub(stub_root)
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{stub_root}{os.pathsep}{existing_pythonpath}" if existing_pythonpath else str(stub_root)
    try:
        return subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "--collect-only",
                "-q",
                "-m",
                markexpr,
                target,
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=60,
            env=env,
            check=False,
        )
    finally:
        for path in sorted(stub_root.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink(missing_ok=True)
            else:
                path.rmdir()
        stub_root.rmdir()


def _parse_selected_count(output: str) -> int:
    """Parse selected test count from pytest collect output."""
    match = _RATIO_SUMMARY.search(output)
    if match:
        return int(match.group("selected"))
    total_match = _TOTAL_ONLY_SUMMARY.search(output)
    if total_match:
        return int(total_match.group("selected"))
    if _NO_TESTS.search(output):
        return 0
    raise AssertionError(f"Unable to parse selected count from output:\n{output}")


def test_fast_ml_collection_contract():
    """Fast-ML lane must remain at or under the strict 67-test ceiling."""
    result = _run_collect(FAST_ML_MARKEXPR, "tests/spatial_ai/reconstruction")
    assert result.returncode == 0, (
        "Fast-ML collection failed unexpectedly.\n" f"stdout:\n{result.stdout}\n" f"stderr:\n{result.stderr}"
    )
    selected = _parse_selected_count(result.stdout)
    assert selected <= FAST_ML_SELECTED_CEILING, (
        "Fast-ML collection contract violated: selected test count exceeded ceiling.\n"
        f"markexpr={FAST_ML_MARKEXPR!r}\n"
        f"selected={selected}, ceiling={FAST_ML_SELECTED_CEILING}\n"
        "Mark integration/slow tests explicitly to keep PR fast-ML lane bounded."
    )


def test_phase23_integration_marker_contract():
    """Phase 2.3 integration tests must be in integration lane, not fast-ML."""
    integration_result = _run_collect(
        INTEGRATION_MARKEXPR,
        "tests/spatial_ai/reconstruction/test_integration_phase23.py",
    )
    assert integration_result.returncode == 0, (
        "Integration-marked Phase 2.3 tests were not collectable.\n"
        f"stdout:\n{integration_result.stdout}\n"
        f"stderr:\n{integration_result.stderr}"
    )
    integration_selected = _parse_selected_count(integration_result.stdout)
    assert integration_selected > 0, (
        "Expected integration tests to be selected for test_integration_phase23.py.\n"
        f"markexpr={INTEGRATION_MARKEXPR!r}\n"
        f"stdout:\n{integration_result.stdout}"
    )

    fast_result = _run_collect(
        FAST_ML_MARKEXPR,
        "tests/spatial_ai/reconstruction/test_integration_phase23.py",
    )
    assert fast_result.returncode in {0, 5}, (
        "Fast-ML probe for Phase 2.3 integration tests failed unexpectedly.\n"
        f"stdout:\n{fast_result.stdout}\n"
        f"stderr:\n{fast_result.stderr}"
    )
    fast_selected = _parse_selected_count(fast_result.stdout)
    assert fast_selected == 0, (
        "Phase 2.3 integration tests leaked into fast-ML selection.\n"
        f"markexpr={FAST_ML_MARKEXPR!r}\n"
        f"selected={fast_selected}\n"
        f"stdout:\n{fast_result.stdout}"
    )
