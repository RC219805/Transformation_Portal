"""Shared RAW runtime contract helpers.

This module keeps the repo-local RAW subprocess contract lightweight so CLI
preflight, config resolution, and ingest adapters can all resolve the same
runtime path without importing heavier pipeline modules.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np

from ..ingest.canonical_json import dump_json
from .da3_runtime import find_repo_root

_REPO_LOCAL_RAW_PYTHON_PARTS = (".venv-raw", "bin", "python")
REPO_LOCAL_RAW_PYTHON = f"./{'/'.join(_REPO_LOCAL_RAW_PYTHON_PARTS)}"
RAW_RUNTIME_ENV_VAR = "TRANSFORMATION_PORTAL_RAW_PYTHON"
RAW_WORKER_MODULE = "transformation_portal.spatial_ai.ingest.raw_worker"
RAW_RUNTIME_CHECK_TIMEOUT_SECONDS = 30
RAW_WORKER_TIMEOUT_SECONDS = 300

# Demosaic algorithm names accepted by the orchestrator and CLI surfaces.
# Membership here is necessary but not sufficient — `resolve_demosaic_algorithm()`
# still verifies the name is exposed by the installed rawpy/LibRaw build at
# runtime. Defined here (rather than in lux_depth_v3) so both the rendering
# path (lux_depth_v3.raw_loader) and the research/training path
# (spatial_ai.ingest.linear_decoder) can share a single source of truth
# without violating ADR-023's isolation between those surfaces.
SUPPORTED_DEMOSAIC_ALGORITHMS = frozenset(
    {
        "AHD",
        "AAHD",
        "AMAZE",
        "DCB",
        "DHT",
        "LINEAR",
        "LMMSE",
        "MODIFIED_AHD",
        "PPG",
        "VNG",
    }
)


def resolve_demosaic_algorithm(name: str):
    """Resolve a demosaic algorithm name to a rawpy.DemosaicAlgorithm enum value.

    Raises ValueError with a clear message if the name is unknown to the
    installed rawpy/LibRaw build. Imports rawpy lazily so callers without the
    raw extra installed don't pay the import cost.
    """
    import rawpy  # type: ignore

    n = name.strip().upper()
    try:
        return getattr(rawpy.DemosaicAlgorithm, n)
    except AttributeError as e:
        raise ValueError(
            f"Unknown demosaic algorithm: {name!r}. "
            f"Available in this rawpy build: "
            f"{sorted(a for a in dir(rawpy.DemosaicAlgorithm) if not a.startswith('_'))}"
        ) from e


def repo_local_raw_python_path(start: Path) -> Optional[Path]:
    """Return the canonical repo-local RAW interpreter path when in a checkout."""
    repo_root = find_repo_root(start)
    if repo_root is None:
        return None
    return repo_root.joinpath(*_REPO_LOCAL_RAW_PYTHON_PARTS)


def normalize_python_executable(value: Any) -> Optional[str]:
    """Normalize Python executable configuration values."""
    if value is None:
        return None
    try:
        normalized = os.fspath(value).strip()
    except TypeError:
        normalized = str(value).strip()
    return normalized or None


def _worker_cwd(start: Path) -> Path:
    """Choose a stable subprocess working directory."""
    repo_root = find_repo_root(start)
    if repo_root is not None:
        return repo_root
    return Path.cwd()


def resolve_raw_python_for_execution(
    python_executable: str,
    *,
    start: Path,
) -> str:
    """Resolve a configured RAW Python executable to an executable path."""
    candidate = normalize_python_executable(python_executable)
    if not candidate:
        raise ValueError("RAW Python executable must be a non-empty string.")

    has_separator = os.sep in candidate or (os.altsep is not None and os.altsep in candidate)
    if candidate.startswith(".") or has_separator:
        path = Path(candidate).expanduser()
        if not path.is_absolute():
            path = _worker_cwd(start) / path
        if not path.exists():
            raise FileNotFoundError(f"RAW Python executable not found: {path}")
        return os.path.abspath(os.fspath(path))

    resolved = shutil.which(candidate)
    if resolved is None:
        raise FileNotFoundError(f"RAW Python executable not found on PATH: {candidate}")
    return resolved


def build_raw_worker_env(start: Path) -> dict[str, str]:
    """Build environment for the RAW subprocess worker."""
    env = os.environ.copy()
    repo_root = find_repo_root(start)
    repo_src = repo_root / "src" if repo_root is not None else None
    if repo_src is not None and repo_src.exists():
        existing = env.get("PYTHONPATH")
        env["PYTHONPATH"] = f"{repo_src}{os.pathsep}{existing}" if existing else str(repo_src)
    return env


def _format_subprocess_output(stdout: str, stderr: str) -> str:
    """Format subprocess output for actionable error messages."""
    stdout_clean = stdout.strip()
    stderr_clean = stderr.strip()
    if stdout_clean and stderr_clean:
        return f"stdout:\n{stdout_clean}\n\nstderr:\n{stderr_clean}"
    if stderr_clean:
        return stderr_clean
    if stdout_clean:
        return stdout_clean
    return "<no output>"


def _build_subprocess_failure_message(
    *,
    title: str,
    python_executable: str,
    command: list[str],
    stdout: str,
    stderr: str,
    timeout_seconds: int | None = None,
) -> str:
    """Build a deterministic subprocess failure message."""
    timeout_line = f"Timeout: {timeout_seconds}s\n" if timeout_seconds is not None else ""
    return (
        f"{title}\n\n"
        f"Python: {python_executable}\n"
        f"Command: {' '.join(command)}\n"
        f"{timeout_line}"
        f"Output:\n{_format_subprocess_output(stdout, stderr)}"
    )


def check_raw_runtime(
    python_executable: str,
    *,
    start: Path,
) -> None:
    """Validate that the dedicated RAW subprocess environment is usable."""
    worker_python = resolve_raw_python_for_execution(python_executable, start=start)
    command = [
        worker_python,
        "-m",
        RAW_WORKER_MODULE,
        "--check",
    ]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            cwd=_worker_cwd(start),
            env=build_raw_worker_env(start),
            check=False,
            timeout=RAW_RUNTIME_CHECK_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            _build_subprocess_failure_message(
                title="RAW subprocess environment is not ready.",
                python_executable=python_executable,
                command=command,
                stdout=exc.stdout or "",
                stderr=exc.stderr or "",
                timeout_seconds=RAW_RUNTIME_CHECK_TIMEOUT_SECONDS,
            )
        ) from exc
    if result.returncode != 0:
        raise RuntimeError(
            _build_subprocess_failure_message(
                title="RAW subprocess environment is not ready.",
                python_executable=python_executable,
                command=command,
                stdout=result.stdout,
                stderr=result.stderr,
            )
        )


def run_raw_worker(
    *,
    python_executable: str,
    command_name: str,
    input_path: Path,
    payload: Optional[Mapping[str, Any]] = None,
    start: Path,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Execute the RAW subprocess worker and return the array + metadata."""
    worker_python = resolve_raw_python_for_execution(python_executable, start=start)
    resolved_input_path = Path(input_path).expanduser().resolve()

    with tempfile.TemporaryDirectory(prefix="tp-raw-worker-") as tmpdir:
        temp_root = Path(tmpdir)
        payload_path = temp_root / "payload.json"
        output_array_path = temp_root / "output.npy"
        output_json_path = temp_root / "output.json"

        with payload_path.open("w", encoding="utf-8") as handle:
            dump_json(
                dict(payload or {}),
                handle,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )

        command = [
            worker_python,
            "-m",
            RAW_WORKER_MODULE,
            "--command",
            command_name,
            "--input-path",
            str(resolved_input_path),
            "--payload-json",
            str(payload_path),
            "--output-array",
            str(output_array_path),
            "--output-json",
            str(output_json_path),
        ]
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                cwd=_worker_cwd(start),
                env=build_raw_worker_env(start),
                check=False,
                timeout=RAW_WORKER_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                _build_subprocess_failure_message(
                    title="RAW subprocess worker failed.",
                    python_executable=python_executable,
                    command=command,
                    stdout=exc.stdout or "",
                    stderr=exc.stderr or "",
                    timeout_seconds=RAW_WORKER_TIMEOUT_SECONDS,
                )
            ) from exc
        if result.returncode != 0:
            raise RuntimeError(
                _build_subprocess_failure_message(
                    title="RAW subprocess worker failed.",
                    python_executable=python_executable,
                    command=command,
                    stdout=result.stdout,
                    stderr=result.stderr,
                )
            )

        array = np.load(output_array_path, allow_pickle=False)
        with output_json_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        return array, metadata
