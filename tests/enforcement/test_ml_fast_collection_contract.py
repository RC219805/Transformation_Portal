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

Design notes:
- Selection-count checks and torch-boundary checks are intentionally split.
- Count checks run with a temporary torch shim so collection accounting remains stable
  in Layer-1 environments without torch installed.
- Boundary checks run with explicit torch import blocking and fail with a focused
  boundary-violation message when unguarded import-time torch usage is introduced.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
FAST_ML_MARKEXPR = "ml and not slow and not integration and not benchmark"
INTEGRATION_MARKEXPR = "ml and integration and not slow and not benchmark"
FAST_ML_SELECTED_CEILING = 67
TORCH_BLOCK_MESSAGE = "Torch import blocked during fast-ML contract collect"

_RATIO_SUMMARY = re.compile(r"(?P<selected>\d+)/(?P<collected>\d+) tests collected")
_TOTAL_ONLY_SUMMARY = re.compile(r"(?P<selected>\d+) tests collected")
_NO_TESTS = re.compile(r"no tests collected")


def _write_torch_stub(stub_root: Path) -> None:
    """Write a torch shim for collection-count accounting in no-torch lanes."""
    torch_dir = stub_root / "torch"
    nn_dir = torch_dir / "nn"
    nn_dir.mkdir(parents=True, exist_ok=True)

    (torch_dir / "__init__.py").write_text(
        '"""Temporary torch shim for collection-only count contract tests."""\n'
        "\n"
        "__version__ = '0.0-contract-shim'\n"
        "\n"
        "class _TorchProxy:\n"
        "    def __getattr__(self, _name):\n"
        "        return self\n"
        "\n"
        "    def __call__(self, *_args, **_kwargs):\n"
        "        return self\n"
        "\n"
        "    def __bool__(self):\n"
        "        return False\n"
        "\n"
        "_PROXY = _TorchProxy()\n"
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
        "class _MPS:\n"
        "    @staticmethod\n"
        "    def is_available():\n"
        "        return False\n"
        "\n"
        "class _Backends:\n"
        "    mps = _MPS()\n"
        "\n"
        "cuda = _Cuda()\n"
        "backends = _Backends()\n"
        "\n"
        "def manual_seed(_seed):\n"
        "    return None\n"
        "\n"
        "def set_num_threads(_n):\n"
        "    return None\n"
        "\n"
        "def set_num_interop_threads(_n):\n"
        "    return None\n"
        "\n"
        "def __getattr__(_name):\n"
        "    return _PROXY\n",
        encoding="utf-8",
    )
    (nn_dir / "__init__.py").write_text('"""torch.nn shim."""\n', encoding="utf-8")
    (nn_dir / "functional.py").write_text('"""torch.nn.functional shim."""\n', encoding="utf-8")


def _write_torch_blocker(sitecustomize_path: Path) -> None:
    """Write sitecustomize that blocks torch imports with an explicit message."""
    sitecustomize_path.write_text(
        "import importlib.abc\n"
        "import sys\n"
        "\n"
        "class TorchBlocker(importlib.abc.MetaPathFinder):\n"
        "    def find_spec(self, fullname, path=None, target=None):\n"
        "        if fullname == 'torch' or fullname.startswith('torch.'):\n"
        f"            raise ImportError('{TORCH_BLOCK_MESSAGE}: ' + fullname)\n"
        "        return None\n"
        "\n"
        "sys.meta_path.insert(0, TorchBlocker())\n",
        encoding="utf-8",
    )


def _run_collect(markexpr: str, target: str, pythonpath_prefix: list[str]) -> subprocess.CompletedProcess[str]:
    """Run pytest --collect-only in a subprocess with an adjusted PYTHONPATH."""
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    merged = list(pythonpath_prefix)
    if existing_pythonpath:
        merged.append(existing_pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(merged)

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
        timeout=90,
        env=env,
        check=False,
    )


def _run_collect_with_torch_stub(markexpr: str, target: str) -> subprocess.CompletedProcess[str]:
    """Run collection with temporary torch shim for stable count accounting."""
    with tempfile.TemporaryDirectory(prefix="ml_fast_collect_stub_") as tmpdir:
        stub_root = Path(tmpdir)
        _write_torch_stub(stub_root)
        return _run_collect(markexpr=markexpr, target=target, pythonpath_prefix=[str(stub_root)])


def _run_collect_with_torch_blocker(markexpr: str, target: str) -> subprocess.CompletedProcess[str]:
    """Run collection with explicit torch import blocking for boundary checks."""
    with tempfile.TemporaryDirectory(prefix="ml_fast_collect_blocker_") as tmpdir:
        blocker_root = Path(tmpdir)
        _write_torch_blocker(blocker_root / "sitecustomize.py")
        return _run_collect(markexpr=markexpr, target=target, pythonpath_prefix=[str(blocker_root)])


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


def _format_subprocess_failure(result: subprocess.CompletedProcess[str], markexpr: str, target: str, context: str) -> str:
    """Format subprocess failure details for diagnostics."""
    return (
        f"{context}\n"
        f"markexpr={markexpr!r}\n"
        f"target={target!r}\n"
        f"returncode={result.returncode}\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )


def _assert_no_torch_boundary_violation(result: subprocess.CompletedProcess[str], markexpr: str, target: str) -> None:
    """Assert collect run did not fail due to unguarded import-time torch usage."""
    if result.returncode in {0, 5}:
        return

    combined_output = f"{result.stdout}\n{result.stderr}"
    if TORCH_BLOCK_MESSAGE in combined_output:
        raise AssertionError(
            "Torch boundary violation during collect-only contract check.\n"
            "A module imported torch at import time without a proper guard (for example, "
            "pytest.importorskip('torch')).\n"
            f"markexpr={markexpr!r}\n"
            f"target={target!r}\n"
            f"returncode={result.returncode}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )

    raise AssertionError(
        _format_subprocess_failure(
            result=result,
            markexpr=markexpr,
            target=target,
            context="Unexpected subprocess failure during torch-boundary collect check.",
        )
    )


def test_fast_ml_collection_contract():
    """Fast-ML lane must remain at or under the strict 67-test ceiling."""
    result = _run_collect_with_torch_stub(FAST_ML_MARKEXPR, "tests/spatial_ai/reconstruction")
    assert result.returncode == 0, _format_subprocess_failure(
        result=result,
        markexpr=FAST_ML_MARKEXPR,
        target="tests/spatial_ai/reconstruction",
        context="Fast-ML collection-count check failed unexpectedly.",
    )

    selected = _parse_selected_count(result.stdout)
    assert selected <= FAST_ML_SELECTED_CEILING, (
        "Fast-ML collection contract violated: selected test count exceeded ceiling.\n"
        f"markexpr={FAST_ML_MARKEXPR!r}\n"
        f"selected={selected}, ceiling={FAST_ML_SELECTED_CEILING}\n"
        "Mark integration/slow tests explicitly to keep PR fast-ML lane bounded."
    )


def test_fast_ml_collection_torch_boundary_contract():
    """Collection should not rely on unguarded import-time torch imports."""
    result = _run_collect_with_torch_blocker(FAST_ML_MARKEXPR, "tests/spatial_ai/reconstruction")
    _assert_no_torch_boundary_violation(
        result=result,
        markexpr=FAST_ML_MARKEXPR,
        target="tests/spatial_ai/reconstruction",
    )


def test_phase23_integration_marker_contract():
    """Phase 2.3 integration tests must be in integration lane, not fast-ML."""
    integration_result = _run_collect_with_torch_stub(
        INTEGRATION_MARKEXPR,
        "tests/spatial_ai/reconstruction/test_integration_phase23.py",
    )
    assert integration_result.returncode == 0, _format_subprocess_failure(
        result=integration_result,
        markexpr=INTEGRATION_MARKEXPR,
        target="tests/spatial_ai/reconstruction/test_integration_phase23.py",
        context="Integration selection check failed unexpectedly.",
    )
    integration_selected = _parse_selected_count(integration_result.stdout)
    assert integration_selected > 0, (
        "Expected integration tests to be selected for test_integration_phase23.py.\n"
        f"markexpr={INTEGRATION_MARKEXPR!r}\n"
        f"stdout:\n{integration_result.stdout}"
    )

    fast_result = _run_collect_with_torch_stub(
        FAST_ML_MARKEXPR,
        "tests/spatial_ai/reconstruction/test_integration_phase23.py",
    )
    assert fast_result.returncode in {0, 5}, _format_subprocess_failure(
        result=fast_result,
        markexpr=FAST_ML_MARKEXPR,
        target="tests/spatial_ai/reconstruction/test_integration_phase23.py",
        context="Fast-ML probe for Phase 2.3 integration tests failed unexpectedly.",
    )
    fast_selected = _parse_selected_count(fast_result.stdout)
    assert fast_selected == 0, (
        "Phase 2.3 integration tests leaked into fast-ML selection.\n"
        f"markexpr={FAST_ML_MARKEXPR!r}\n"
        f"selected={fast_selected}\n"
        f"stdout:\n{fast_result.stdout}"
    )
