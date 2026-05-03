"""Shared RAW runtime contract helpers.

This module keeps the repo-local RAW subprocess contract lightweight so CLI
preflight, config resolution, and ingest adapters can all resolve the same
runtime path without importing heavier pipeline modules.
"""

from __future__ import annotations

import json
import os
import re
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

# Demosaic name validation has two layers:
#
#   * Upstream (CLI / orchestrator): a *syntactic* check via
#     ``is_valid_demosaic_name`` — the orchestrator may not have rawpy
#     installed locally (it dispatches RAW decode to the .venv-raw
#     subprocess), so it cannot enumerate enum members. The syntactic gate
#     just guards against typos / shell-quoting bugs and rejects values that
#     could not possibly be a ``rawpy.DemosaicAlgorithm`` member.
#
#   * Decode-time: ``resolve_demosaic_algorithm`` reflects the actual
#     installed ``rawpy.DemosaicAlgorithm`` enum and fails closed if the
#     name is not present in this LibRaw build. This is the authoritative
#     gate. The set of valid members varies by rawpy/LibRaw version (e.g.
#     ``AFD``, ``VCD``, ``VCD_MODIFIED_AHD`` are present in some builds and
#     not others) — that variability is the reason a curated upstream
#     allowlist is the wrong abstraction.
_DEMOSAIC_NAME_RE = re.compile(r"^[A-Z][A-Z0-9_]{0,31}$")


def is_valid_demosaic_name(name: object) -> bool:
    """Return True if ``name`` is syntactically plausible as a rawpy enum member.

    This is a *format* check only — it does not verify that the name exists
    in the installed rawpy/LibRaw build. Use ``resolve_demosaic_algorithm``
    for the semantic check, which fails closed at decode time and surfaces
    the actual available enum members.

    Strips and uppercases the input first, matching the normalization used
    by the CLI/orchestrator before they emit ``--raw-demosaic`` into argv.
    """
    if not isinstance(name, str):
        return False
    return bool(_DEMOSAIC_NAME_RE.fullmatch(name.strip().upper()))


def _available_demosaic_names(demosaic_algorithm: object) -> list[str]:
    """Return the public member names of ``rawpy.DemosaicAlgorithm``.

    ``rawpy.DemosaicAlgorithm`` is normally a ``Flag``/``IntEnum`` subclass,
    so prefer ``__members__`` which lists exactly the algorithm members
    (and excludes the magic methods/inherited attributes that ``dir()``
    would otherwise mix in). Fall back to filtered ``dir()`` for builds
    where the enum protocol differs.
    """
    members = getattr(demosaic_algorithm, "__members__", None)
    if members is not None:
        try:
            return sorted(members.keys())
        except Exception:  # pragma: no cover - defensive: unusual mapping
            pass
    return sorted(a for a in dir(demosaic_algorithm) if a.isupper() and not a.startswith("_"))


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
            f"Available in this rawpy build: {_available_demosaic_names(rawpy.DemosaicAlgorithm)}"
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
