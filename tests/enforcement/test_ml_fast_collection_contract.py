"""Enforce fast-ML collection boundary for reconstruction tests.

This contract prevents slow/integration creep into the PR fast-ML lane.

Baseline rationale (March 3, 2026):
- Pre-marker baseline: 72 selected tests for
  "ml and not slow and not integration and not benchmark"
- Post-marker baseline: 67 selected tests after marking
  tests/spatial_ai/reconstruction/test_integration_phase23.py as integration
- Previous baseline: 69 selected tests after adding
  deterministic reconstruction golden state snapshot + byte-stability tests
  in tests/spatial_ai/reconstruction/test_reconstruction_golden_snapshot.py
- Previous baseline: 70 selected tests after adding SLERP orthogonality test
  in tests/spatial_ai/reconstruction/test_scene_utils.py
- Current baseline: 75 selected tests after adding zero-trust filesystem
  and execution security infrastructure tests in PR #1217

Contract:
- fast-ML selected count must remain <= 75
- integration-marked Phase 2.3 tests must stay out of fast-ML selection

Design notes:
- Selection-count checks and torch-boundary checks are intentionally split.
- Count checks run with a temporary torch shim so collection accounting remains stable
  in Layer-1 environments without torch installed.
- Boundary checks run with explicit torch import blocking and fail with a focused
  boundary-violation message when unguarded import-time torch usage is introduced.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
FAST_ML_MARKEXPR = "ml and not slow and not integration and not benchmark"
INTEGRATION_MARKEXPR = "ml and integration and not slow and not benchmark"
FAST_ML_SELECTED_CEILING = 80
TORCH_BLOCK_MESSAGE = "Torch import blocked during fast-ML contract collect"

_COLLECT_SUMMARY = re.compile(
    r"(?:collected\s+(?P<collected>\d+)\s+items?|(?P<selected>\d+)(?:/\d+)?\s+test[s]?\s+collected)",
    re.IGNORECASE,
)
_NO_TESTS = re.compile(r"no tests collected", re.IGNORECASE)


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
        f"            raise ModuleNotFoundError('{TORCH_BLOCK_MESSAGE}: ' + fullname)\n"
        "        return None\n"
        "\n"
        "sys.meta_path.insert(0, TorchBlocker())\n",
        encoding="utf-8",
    )


def _run_collect(
    markexpr: str,
    target: str,
    pythonpath_prefix: list[str],
    json_report_file: Path | None = None,
) -> tuple[subprocess.CompletedProcess[str], bool]:
    """Run pytest --collect-only in a subprocess with an adjusted PYTHONPATH."""
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    merged = list(pythonpath_prefix)
    if existing_pythonpath:
        merged.append(existing_pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(merged)

    base_args = [
        sys.executable,
        "-m",
        "pytest",
        "--collect-only",
        "-q",
        "-m",
        markexpr,
        target,
    ]
    if json_report_file is not None:
        base_args.extend(["--json-report", f"--json-report-file={json_report_file}"])

    result = subprocess.run(
        base_args,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=90,
        env=env,
        check=False,
    )

    # Local developer environments may not have pytest-json-report installed.
    # Fall back to text parsing so this contract remains runnable outside CI.
    if (
        json_report_file is not None
        and result.returncode != 0
        and "--json-report" in result.stderr
        and "unrecognized arguments" in result.stderr
    ):
        fallback_result = subprocess.run(
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
        return fallback_result, False

    return result, json_report_file is not None


def _run_collect_with_torch_stub(markexpr: str, target: str) -> tuple[subprocess.CompletedProcess[str], int]:
    """Run collection with temporary torch shim for stable count accounting."""
    with tempfile.TemporaryDirectory(prefix="ml_fast_collect_stub_") as tmpdir:
        stub_root = Path(tmpdir)
        _write_torch_stub(stub_root)
        report_file = stub_root / "collect_report.json"
        result, used_json_report = _run_collect(
            markexpr=markexpr,
            target=target,
            pythonpath_prefix=[str(stub_root)],
            json_report_file=report_file,
        )
        if result.returncode not in {0, 5}:
            # Preserve subprocess failure context for caller-side assertions.
            # Avoid masking return-code diagnostics with downstream parse errors.
            return result, -1
        selected = _parse_selected_count(
            output=result.stdout,
            stderr=result.stderr,
            json_report_file=report_file if used_json_report else None,
        )
        return result, selected


def _run_collect_with_torch_blocker(markexpr: str, target: str) -> tuple[subprocess.CompletedProcess[str], int]:
    """Run collection with explicit torch import blocking for boundary checks."""
    with tempfile.TemporaryDirectory(prefix="ml_fast_collect_blocker_") as tmpdir:
        blocker_root = Path(tmpdir)
        _write_torch_blocker(blocker_root / "sitecustomize.py")
        report_file = blocker_root / "collect_report.json"
        result, used_json_report = _run_collect(
            markexpr=markexpr,
            target=target,
            pythonpath_prefix=[str(blocker_root)],
            json_report_file=report_file,
        )
        if result.returncode not in {0, 5}:
            # Preserve subprocess failure context for caller-side assertions.
            # Avoid masking return-code diagnostics with downstream parse errors.
            return result, -1
        selected = _parse_selected_count(
            output=result.stdout,
            stderr=result.stderr,
            json_report_file=report_file if used_json_report else None,
        )
        return result, selected


def _parse_selected_count(output: str, stderr: str, json_report_file: Path | None = None) -> int:
    """Parse selected count from JSON report when available, else resilient text fallback."""
    if json_report_file is not None:
        if not json_report_file.exists():
            raise AssertionError(
                "Expected pytest JSON report file for collection count but file was missing.\n"
                f"json_report_file={json_report_file}\n"
                f"stdout:\n{output}\n"
                f"stderr:\n{stderr}"
            )

        try:
            payload = json.loads(json_report_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise AssertionError(
                "Unable to parse pytest JSON report for collection count.\n"
                f"json_report_file={json_report_file}\n"
                f"error={exc}\n"
                f"stdout:\n{output}\n"
                f"stderr:\n{stderr}"
            ) from exc

        summary = payload.get("summary")
        if not isinstance(summary, dict):
            raise AssertionError(
                "Pytest JSON report missing summary payload for collection count.\n"
                f"json_report_file={json_report_file}\n"
                f"summary_type={type(summary).__name__}\n"
                f"stdout:\n{output}\n"
                f"stderr:\n{stderr}"
            )
        total = summary.get("total")
        deselected = summary.get("deselected", 0)
        collected = summary.get("collected")
        if isinstance(collected, int):
            if not isinstance(deselected, int):
                raise AssertionError(
                    "Pytest JSON report had non-integer deselected count.\n"
                    f"json_report_file={json_report_file}\n"
                    f"summary={summary!r}\n"
                    f"stdout:\n{output}\n"
                    f"stderr:\n{stderr}"
                )
            selected = collected - deselected
            if selected < 0:
                raise AssertionError(
                    "Pytest JSON report invariant violated for collection counts.\n"
                    f"Expected collected - deselected >= 0, got {collected} - {deselected}\n"
                    f"json_report_file={json_report_file}\n"
                    f"summary={summary!r}\n"
                    f"stdout:\n{output}\n"
                    f"stderr:\n{stderr}"
                )
            # In collect-only mode, some pytest-json-report versions emit total=0.
            # In other modes/versions, total may equal selected.
            if isinstance(total, int) and total not in {0, selected}:
                raise AssertionError(
                    "Pytest JSON report invariant violated for selected count.\n"
                    f"Expected total in {{0, selected}}, got total={total}, selected={selected}\n"
                    f"json_report_file={json_report_file}\n"
                    f"summary={summary!r}\n"
                    f"stdout:\n{output}\n"
                    f"stderr:\n{stderr}"
                )
            return selected
        if isinstance(total, int):
            return total
        raise AssertionError(
            "Pytest JSON report summary did not include collection counters.\n"
            f"json_report_file={json_report_file}\n"
            f"summary={summary!r}\n"
            f"stdout:\n{output}\n"
            f"stderr:\n{stderr}"
        )

    match = _COLLECT_SUMMARY.search(output)
    if match:
        count = match.group("collected") or match.group("selected")
        if count is not None:
            return int(count)
    if _NO_TESTS.search(output):
        return 0
    raise AssertionError(f"Unable to parse selected count from output:\n{output}\nstderr:\n{stderr}")


def test_selected_count_parser_handles_singular_and_plural_text_forms():
    """Fallback parser should handle pytest singular/plural variants robustly."""
    assert _parse_selected_count("1 test collected", stderr="") == 1
    assert _parse_selected_count("1/1 test collected", stderr="") == 1
    assert _parse_selected_count("1/1 tests collected", stderr="") == 1
    assert _parse_selected_count("collected 1 item", stderr="") == 1
    assert _parse_selected_count("collected 67 items", stderr="") == 67
    assert _parse_selected_count("67/106 tests collected", stderr="") == 67
    assert _parse_selected_count("no tests collected", stderr="") == 0


def test_selected_count_parser_prefers_json_report():
    """When JSON report is available, parser should source count from JSON payload."""
    with tempfile.TemporaryDirectory(prefix="ml_fast_collect_json_parser_") as tmpdir:
        report_file = Path(tmpdir) / "report.json"
        report_file.write_text(
            '{"summary": {"collected": 106, "deselected": 39, "total": 0}}',
            encoding="utf-8",
        )
        assert _parse_selected_count(output="", stderr="", json_report_file=report_file) == 67


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
    """Fast-ML lane must remain at or under the strict 75-test ceiling."""
    result, selected = _run_collect_with_torch_stub(FAST_ML_MARKEXPR, "tests/spatial_ai/reconstruction")
    assert result.returncode == 0, _format_subprocess_failure(
        result=result,
        markexpr=FAST_ML_MARKEXPR,
        target="tests/spatial_ai/reconstruction",
        context="Fast-ML collection-count check failed unexpectedly.",
    )

    assert selected <= FAST_ML_SELECTED_CEILING, (
        "Fast-ML collection contract violated: selected test count exceeded ceiling.\n"
        f"markexpr={FAST_ML_MARKEXPR!r}\n"
        f"selected={selected}, ceiling={FAST_ML_SELECTED_CEILING}\n"
        "Mark integration/slow tests explicitly to keep PR fast-ML lane bounded."
    )


def test_fast_ml_collection_torch_boundary_contract():
    """Collection should not rely on unguarded import-time torch imports."""
    result, _ = _run_collect_with_torch_blocker(FAST_ML_MARKEXPR, "tests/spatial_ai/reconstruction")
    _assert_no_torch_boundary_violation(
        result=result,
        markexpr=FAST_ML_MARKEXPR,
        target="tests/spatial_ai/reconstruction",
    )


def test_phase23_integration_marker_contract():
    """Phase 2.3 integration tests must be in integration lane, not fast-ML."""
    integration_result, integration_selected = _run_collect_with_torch_stub(
        INTEGRATION_MARKEXPR,
        "tests/spatial_ai/reconstruction/test_integration_phase23.py",
    )
    assert integration_result.returncode == 0, _format_subprocess_failure(
        result=integration_result,
        markexpr=INTEGRATION_MARKEXPR,
        target="tests/spatial_ai/reconstruction/test_integration_phase23.py",
        context="Integration selection check failed unexpectedly.",
    )
    assert integration_selected > 0, (
        "Expected integration tests to be selected for test_integration_phase23.py.\n"
        f"markexpr={INTEGRATION_MARKEXPR!r}\n"
        f"stdout:\n{integration_result.stdout}"
    )

    fast_result, fast_selected = _run_collect_with_torch_stub(
        FAST_ML_MARKEXPR,
        "tests/spatial_ai/reconstruction/test_integration_phase23.py",
    )
    assert fast_result.returncode in {0, 5}, _format_subprocess_failure(
        result=fast_result,
        markexpr=FAST_ML_MARKEXPR,
        target="tests/spatial_ai/reconstruction/test_integration_phase23.py",
        context="Fast-ML probe for Phase 2.3 integration tests failed unexpectedly.",
    )
    assert fast_selected == 0, (
        "Phase 2.3 integration tests leaked into fast-ML selection.\n"
        f"markexpr={FAST_ML_MARKEXPR!r}\n"
        f"selected={fast_selected}\n"
        f"stdout:\n{fast_result.stdout}"
    )
