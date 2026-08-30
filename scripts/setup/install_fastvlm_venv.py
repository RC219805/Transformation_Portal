#!/usr/bin/env python3
"""Build and promote a clean FastVLM venv with durable dependency evidence."""

from __future__ import annotations

import argparse
import errno
import importlib.util
import json
import os
import re
import secrets
import shutil
import stat
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
VALIDATION_DIR = SCRIPT_DIR.parent / "validation"
VENV_BUILD_TIMEOUT_SECONDS = 300
PIP_INSTALL_TIMEOUT_SECONDS = 3600
PIP_BOOTSTRAP_CLEANUP_TIMEOUT_SECONDS = 120
PIP_CHECK_TIMEOUT_SECONDS = 120
PIP_FREEZE_TIMEOUT_SECONDS = 120
FREEZE_EVIDENCE_NAME = "fastvlm-pip-freeze.txt"
_ALLOWED_BOOTSTRAP_PTH_PAYLOADS = frozenset(
    {
        b"import os; var = 'SETUPTOOLS_USE_DISTUTILS'; enabled = os.environ.get(var, 'local') == 'local'; "
        b"enabled and __import__('_distutils_hack').add_shim(); \n",
    }
)


def _load_manifest_helpers() -> Any:
    helper_path = VALIDATION_DIR / "fastvlm_runtime_manifest.py"
    spec = importlib.util.spec_from_file_location("fastvlm_runtime_manifest_venv_install", helper_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load FastVLM manifest helpers from {helper_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_manifest_helpers = _load_manifest_helpers()
ManifestError = _manifest_helpers.ManifestError
RuntimeVerificationError = _manifest_helpers.RuntimeVerificationError
load_manifest = _manifest_helpers.load_manifest
require_valid_manifest = _manifest_helpers.require_valid_manifest
runtime_root = _manifest_helpers.runtime_root
safe_child = _manifest_helpers.safe_child


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--runtime-root", required=True)
    parser.add_argument("--base-python", default="")
    parser.add_argument("--requirements", default="")
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args(argv)
    if not args.audit_only and (not args.base_python or not args.requirements):
        parser.error("--base-python and --requirements are required unless --audit-only is selected")
    return args


def _lexical_absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.path.expanduser(str(path))))


def _ensure_no_symlink_components(path: Path, *, allow_missing_leaf: bool = False) -> None:
    target = _lexical_absolute(path)
    current = Path(target.anchor)
    for index, part in enumerate(target.parts[1:], start=1):
        current /= part
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            if allow_missing_leaf and index == len(target.parts) - 1:
                return
            raise RuntimeVerificationError(f"FastVLM install path is missing: {current}")
        if stat.S_ISLNK(metadata.st_mode):
            raise RuntimeVerificationError(f"FastVLM install path must not contain symlinks: {current}")


def _ensure_safe_runtime_root(root: Path, *, create: bool = True) -> Path:
    target = _lexical_absolute(root)
    existing = target
    while not existing.exists() and existing != existing.parent:
        existing = existing.parent
    _ensure_no_symlink_components(existing)
    if create:
        target.mkdir(parents=True, exist_ok=True)
    elif not target.exists():
        raise RuntimeVerificationError(f"FastVLM runtime root is missing: {target}")
    metadata = target.lstat()
    if not stat.S_ISDIR(metadata.st_mode):
        raise RuntimeVerificationError(f"FastVLM runtime root must be a real directory: {target}")
    _ensure_no_symlink_components(target)
    return target


def _require_regular_file(path: Path, *, description: str, executable: bool = False) -> Path:
    target = Path(os.path.realpath(_lexical_absolute(path)))
    try:
        metadata = target.lstat()
    except FileNotFoundError as exc:
        raise RuntimeVerificationError(f"{description} is missing: {target}") from exc
    if not stat.S_ISREG(metadata.st_mode):
        raise RuntimeVerificationError(f"{description} must be a regular file: {target}")
    if executable and not os.access(target, os.X_OK):
        raise RuntimeVerificationError(f"{description} is not executable: {target}")
    return target


def _validate_existing_target(path: Path, *, directory: bool, description: str) -> None:
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return
    expected = stat.S_ISDIR(metadata.st_mode) if directory else stat.S_ISREG(metadata.st_mode)
    if stat.S_ISLNK(metadata.st_mode) or not expected:
        kind = "directory" if directory else "regular file"
        raise RuntimeVerificationError(f"{description} must be a real {kind}: {path}")


def _isolated_python_environment() -> dict[str, str]:
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.upper().startswith("PYTHON")
        and not key.upper().startswith("PIP_")
        and key.upper()
        not in {
            "PYTHONHOME",
            "PYTHONPATH",
            "PYTHONSTARTUP",
            "PYTHONUSERBASE",
            "VIRTUAL_ENV",
            "__PYVENV_LAUNCHER__",
        }
    }
    environment.update(
        {
            "LC_ALL": "C",
            "PIP_CONFIG_FILE": os.devnull,
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
            "PIP_NO_INPUT": "1",
            "PIP_REQUIRE_VIRTUALENV": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
            "PYTHONSAFEPATH": "1",
        }
    )
    return environment


def _run_checked(command: Sequence[str], *, timeout_seconds: int, description: str) -> subprocess.CompletedProcess[str]:
    try:
        completed = subprocess.run(
            list(command),
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=_isolated_python_environment(),
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeVerificationError(f"{description} timed out after {timeout_seconds}s") from exc
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or f"exit code {completed.returncode}"
        raise RuntimeVerificationError(f"{description} failed: {detail}")
    return completed


def _read_regular_json(path: Path) -> Any:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise RuntimeVerificationError(f"FastVLM venv metadata must be regular: {path}")
        with os.fdopen(os.dup(descriptor), "r", encoding="utf-8") as stream:
            return json.load(stream)
    finally:
        os.close(descriptor)


def _read_regular_bytes(path: Path, *, description: str) -> bytes:
    try:
        before = path.lstat()
    except FileNotFoundError as exc:
        raise RuntimeVerificationError(f"{description} is missing: {path}") from exc
    if not stat.S_ISREG(before.st_mode):
        raise RuntimeVerificationError(f"{description} must be a regular file: {path}")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise RuntimeVerificationError(f"{description} could not be opened safely: {path}") from exc
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
            raise RuntimeVerificationError(f"{description} changed while being inspected: {path}")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 64 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after_open = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    after_path = path.lstat()
    stable_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns")
    if any(getattr(opened, field) != getattr(after_open, field) for field in stable_fields) or any(
        getattr(opened, field) != getattr(after_path, field) for field in stable_fields
    ):
        raise RuntimeVerificationError(f"{description} changed while being inspected: {path}")
    return b"".join(chunks)


def _remove_allowlisted_bootstrap_pth(stage: Path) -> None:
    """Remove only setuptools' exact bootstrap hook before staged Python runs."""

    candidates = list(stage.glob("lib/python*/site-packages/distutils-precedence.pth"))
    candidates.extend(stage.glob("lib/python*/dist-packages/distutils-precedence.pth"))
    candidates.append(stage / "Lib" / "site-packages" / "distutils-precedence.pth")
    for candidate in sorted(set(candidates)):
        if not candidate.exists() and not candidate.is_symlink():
            continue
        _ensure_no_symlink_components(candidate)
        payload = _read_regular_bytes(candidate, description="FastVLM bootstrap setuptools .pth")
        if payload not in _ALLOWED_BOOTSTRAP_PTH_PAYLOADS:
            raise RuntimeVerificationError(f"FastVLM bootstrap setuptools .pth content is not allowlisted: {candidate}")
        candidate.unlink()


def _uninstall_bootstrap_setuptools(stage_python: Path) -> None:
    """Remove setuptools after its executable bootstrap hook is gone."""

    _run_checked(
        [str(stage_python), "-I", "-m", "pip", "--isolated", "uninstall", "--yes", "setuptools"],
        timeout_seconds=PIP_BOOTSTRAP_CLEANUP_TIMEOUT_SECONDS,
        description="FastVLM bootstrap setuptools removal",
    )


def _is_python_launcher(name: str) -> bool:
    return re.fullmatch(r"pythonw?(?:\d+(?:\.\d+)?)?(?:\.exe)?", name.lower()) is not None


def _is_startup_module_entry(name: str) -> bool:
    lower_name = name.lower()
    return any(lower_name == module or lower_name.startswith(f"{module}.") for module in ("sitecustomize", "usercustomize"))


def _parse_pyvenv_config(path: Path) -> dict[str, str]:
    payload = _read_regular_bytes(path, description="FastVLM pyvenv.cfg")
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RuntimeVerificationError(f"FastVLM pyvenv.cfg must be UTF-8: {path}") from exc
    values: dict[str, str] = {}
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        key, separator, value = line.partition("=")
        normalized_key = key.strip().lower()
        if not separator or not normalized_key or normalized_key in values:
            raise RuntimeVerificationError(f"FastVLM pyvenv.cfg contains invalid metadata at line {line_number}: {path}")
        values[normalized_key] = value.strip()
    return values


def _validate_venv_controls(venv_dir: Path, *, expected_base_python: Path) -> None:
    expected_python = _require_regular_file(
        expected_base_python,
        description="FastVLM expected base Python",
        executable=True,
    )
    config_path = venv_dir / "pyvenv.cfg"
    _ensure_no_symlink_components(config_path)
    values = _parse_pyvenv_config(config_path)
    if values.get("include-system-site-packages", "").lower() != "false":
        raise RuntimeVerificationError("FastVLM pyvenv.cfg must set include-system-site-packages = false")

    base_metadata = _run_checked(
        [
            str(expected_python),
            "-I",
            "-S",
            "-c",
            (
                "import json, os, sys; "
                "print(json.dumps({'executable': os.path.realpath(sys._base_executable), "
                "'home': os.path.realpath(os.path.dirname(sys._base_executable))}))"
            ),
        ],
        timeout_seconds=PIP_BOOTSTRAP_CLEANUP_TIMEOUT_SECONDS,
        description="FastVLM trusted base Python metadata",
    )
    try:
        expected_metadata = json.loads(base_metadata.stdout)
    except (json.JSONDecodeError, TypeError) as exc:
        raise RuntimeVerificationError("FastVLM trusted base Python returned invalid metadata") from exc
    expected_executable = Path(str(expected_metadata.get("executable") or ""))
    expected_home = Path(str(expected_metadata.get("home") or ""))
    configured_home = Path(values.get("home", ""))
    configured_executable = Path(values.get("executable", ""))
    if not configured_home.is_absolute() or not configured_executable.is_absolute():
        raise RuntimeVerificationError("FastVLM pyvenv.cfg must bind absolute home and executable paths")
    if Path(os.path.realpath(configured_home)) != expected_home:
        raise RuntimeVerificationError("FastVLM pyvenv.cfg home does not match the trusted base Python")
    if Path(os.path.realpath(configured_executable)) != expected_executable:
        raise RuntimeVerificationError("FastVLM pyvenv.cfg executable does not match the trusted base Python")

    launcher_dir = venv_dir / ("Scripts" if os.name == "nt" else "bin")
    canonical_launcher = launcher_dir / ("python.exe" if os.name == "nt" else "python")
    base_payload = _read_regular_bytes(expected_python, description="FastVLM expected base Python")
    launchers = [path for path in launcher_dir.iterdir() if _is_python_launcher(path.name)]
    if canonical_launcher not in launchers:
        raise RuntimeVerificationError(f"FastVLM venv Python launcher is missing: {canonical_launcher}")
    for launcher in launchers:
        if _read_regular_bytes(launcher, description="FastVLM venv Python launcher") != base_payload:
            raise RuntimeVerificationError(f"FastVLM venv Python launcher does not match the trusted base: {launcher}")


def audit_runtime_venv(venv_dir: Path, *, expected_base_python: Path | None = None) -> None:
    """Reject startup code, editable installs, symlinks, and special site entries."""

    target = _lexical_absolute(venv_dir)
    _validate_existing_target(target, directory=True, description="FastVLM venv")
    found_target = target.exists()
    if not found_target:
        raise RuntimeVerificationError(f"FastVLM venv is missing: {target}")

    def visit(directory: Path, *, in_site_packages: bool) -> None:
        try:
            entries = os.scandir(directory)
        except OSError as exc:
            raise RuntimeVerificationError(f"FastVLM venv directory could not be read: {directory}") from exc
        with entries:
            for entry in entries:
                path = Path(entry.path)
                metadata = entry.stat(follow_symlinks=False)
                child_in_site = in_site_packages or entry.name in {"site-packages", "dist-packages"}
                if stat.S_ISLNK(metadata.st_mode):
                    raise RuntimeVerificationError(f"FastVLM venv must not contain symlinks: {path}")
                if stat.S_ISDIR(metadata.st_mode):
                    if child_in_site and _is_startup_module_entry(entry.name):
                        raise RuntimeVerificationError(f"FastVLM venv contains prohibited startup module: {path}")
                    visit(path, in_site_packages=child_in_site)
                    continue
                if not stat.S_ISREG(metadata.st_mode):
                    raise RuntimeVerificationError(f"FastVLM venv contains unsupported filesystem entry: {path}")
                lower_name = entry.name.lower()
                if lower_name.endswith("._pth"):
                    raise RuntimeVerificationError(f"FastVLM venv contains prohibited path-control artifact: {path}")
                if not child_in_site:
                    continue
                if lower_name.endswith((".pth", ".egg-link")) or _is_startup_module_entry(lower_name):
                    raise RuntimeVerificationError(f"FastVLM venv contains prohibited startup/editable artifact: {path}")
                if lower_name == "direct_url.json":
                    payload = _read_regular_json(path)
                    if isinstance(payload, Mapping):
                        dir_info = payload.get("dir_info")
                        if isinstance(dir_info, Mapping) and dir_info.get("editable") is True:
                            raise RuntimeVerificationError(f"FastVLM venv contains an editable install: {path}")

    visit(target, in_site_packages=False)
    trusted_base = expected_base_python or Path(getattr(sys, "_base_executable", sys.executable))
    _validate_venv_controls(target, expected_base_python=trusted_base)


def _validate_freeze_output(output: str) -> bytes:
    for line in output.splitlines():
        normalized = line.strip().lower()
        if normalized.startswith(("-e ", "--editable ")) or " @ file://" in normalized:
            raise RuntimeVerificationError("FastVLM pip freeze contains an editable or local-path install")
    canonical = output.rstrip("\n") + "\n"
    return canonical.encode("utf-8")


def _build_staged_venv(base_python: Path, stage: Path, requirements: Path) -> bytes:
    _run_checked(
        [str(base_python), "-I", "-S", "-m", "venv", "--copies", str(stage)],
        timeout_seconds=VENV_BUILD_TIMEOUT_SECONDS,
        description="FastVLM venv creation",
    )
    stage_python = stage / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    _remove_allowlisted_bootstrap_pth(stage)
    audit_runtime_venv(stage, expected_base_python=base_python)
    _uninstall_bootstrap_setuptools(stage_python)
    audit_runtime_venv(stage, expected_base_python=base_python)
    _run_checked(
        [
            str(stage_python),
            "-I",
            "-m",
            "pip",
            "--isolated",
            "install",
            "--no-input",
            "--disable-pip-version-check",
            "--no-deps",
            "--only-binary=:all:",
            "--requirement",
            str(requirements),
        ],
        timeout_seconds=PIP_INSTALL_TIMEOUT_SECONDS,
        description="FastVLM dependency installation",
    )
    _remove_allowlisted_bootstrap_pth(stage)
    audit_runtime_venv(stage, expected_base_python=base_python)
    _run_checked(
        [str(stage_python), "-I", "-m", "pip", "--isolated", "check"],
        timeout_seconds=PIP_CHECK_TIMEOUT_SECONDS,
        description="FastVLM dependency consistency check",
    )
    audit_runtime_venv(stage, expected_base_python=base_python)
    freeze = _run_checked(
        [str(stage_python), "-I", "-m", "pip", "--isolated", "freeze", "--all"],
        timeout_seconds=PIP_FREEZE_TIMEOUT_SECONDS,
        description="FastVLM dependency evidence capture",
    )
    audit_runtime_venv(stage, expected_base_python=base_python)
    return _validate_freeze_output(freeze.stdout)


def _write_evidence_temp(runtime: Path, payload: bytes) -> Path:
    path = runtime / f".{FREEZE_EVIDENCE_NAME}.tmp-{os.getpid()}-{secrets.token_hex(8)}"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise RuntimeVerificationError(f"FastVLM dependency evidence temp must be regular: {path}")
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short write while recording FastVLM dependency evidence")
            view = view[written:]
        os.fsync(descriptor)
    except Exception:
        os.close(descriptor)
        path.unlink(missing_ok=True)
        raise
    os.close(descriptor)
    return path


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(path, flags)
    try:
        try:
            os.fsync(descriptor)
        except OSError as exc:
            if exc.errno not in {errno.EINVAL, getattr(errno, "ENOTSUP", errno.EINVAL)}:
                raise
    finally:
        os.close(descriptor)


def _remove_path(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)


def _replace_path(source: Path, destination: Path) -> None:
    os.replace(source, destination)


def _promote_venv_and_evidence(
    *,
    runtime: Path,
    staged_venv: Path,
    target_venv: Path,
    staged_evidence: Path,
    evidence_path: Path,
    expected_base_python: Path,
) -> None:
    token = f"{os.getpid()}-{secrets.token_hex(8)}"
    venv_backup = runtime / f".{target_venv.name}.backup-{token}"
    evidence_backup = runtime / f".{evidence_path.name}.backup-{token}"
    moved_venv = False
    promoted_venv = False
    moved_evidence = False
    promoted_evidence = False
    try:
        _validate_existing_target(target_venv, directory=True, description="FastVLM venv")
        _validate_existing_target(evidence_path, directory=False, description="FastVLM dependency evidence")
        if target_venv.exists():
            _replace_path(target_venv, venv_backup)
            moved_venv = True
        _replace_path(staged_venv, target_venv)
        promoted_venv = True
        if evidence_path.exists():
            _replace_path(evidence_path, evidence_backup)
            moved_evidence = True
        _replace_path(staged_evidence, evidence_path)
        promoted_evidence = True
        audit_runtime_venv(target_venv, expected_base_python=expected_base_python)
        _fsync_directory(runtime)
    except Exception as exc:
        rollback_errors: list[str] = []
        if promoted_evidence:
            try:
                _remove_path(evidence_path)
            except OSError as rollback_exc:
                rollback_errors.append(f"could not withdraw dependency evidence: {rollback_exc}")
        if moved_evidence:
            try:
                _replace_path(evidence_backup, evidence_path)
            except OSError as rollback_exc:
                rollback_errors.append(f"could not restore dependency evidence: {rollback_exc}")
        if promoted_venv:
            try:
                _remove_path(target_venv)
            except OSError as rollback_exc:
                rollback_errors.append(f"could not withdraw venv: {rollback_exc}")
        if moved_venv:
            try:
                _replace_path(venv_backup, target_venv)
            except OSError as rollback_exc:
                rollback_errors.append(f"could not restore venv: {rollback_exc}")
        if rollback_errors:
            raise RuntimeVerificationError(
                f"FastVLM venv/evidence promotion failed ({exc}); rollback incomplete: " + "; ".join(rollback_errors)
            ) from exc
        raise
    _remove_path(venv_backup)
    _remove_path(evidence_backup)


def prepare_runtime_venv(
    manifest: Mapping[str, Any],
    *,
    root: Path,
    base_python: Path,
    requirements: Path,
) -> str:
    require_valid_manifest(manifest)
    runtime = _ensure_safe_runtime_root(root)
    python_config = manifest.get("python")
    if not isinstance(python_config, Mapping):
        raise ManifestError("FastVLM manifest python section must be an object")
    target_venv = safe_child(runtime, python_config.get("venv_dir") or ".venv-fastvlm")
    evidence_path = runtime / FREEZE_EVIDENCE_NAME
    _validate_existing_target(target_venv, directory=True, description="FastVLM venv")
    _validate_existing_target(evidence_path, directory=False, description="FastVLM dependency evidence")
    trusted_python = _require_regular_file(base_python, description="FastVLM base Python", executable=True)
    trusted_requirements = _require_regular_file(requirements, description="FastVLM requirements")

    staged_venv = Path(tempfile.mkdtemp(prefix=".venv-fastvlm-stage-", dir=runtime))
    staged_evidence: Path | None = None
    try:
        freeze_payload = _build_staged_venv(trusted_python, staged_venv, trusted_requirements)
        staged_evidence = _write_evidence_temp(runtime, freeze_payload)
        _promote_venv_and_evidence(
            runtime=runtime,
            staged_venv=staged_venv,
            target_venv=target_venv,
            staged_evidence=staged_evidence,
            evidence_path=evidence_path,
            expected_base_python=trusted_python,
        )
    finally:
        _remove_path(staged_venv)
        if staged_evidence is not None:
            staged_evidence.unlink(missing_ok=True)
    return "installed"


def audit_installed_runtime_venv(manifest: Mapping[str, Any], *, root: Path) -> str:
    """Audit the promoted venv and evidence without executing the venv."""

    require_valid_manifest(manifest)
    runtime = _ensure_safe_runtime_root(root, create=False)
    python_config = manifest.get("python")
    if not isinstance(python_config, Mapping):
        raise ManifestError("FastVLM manifest python section must be an object")
    target_venv = safe_child(runtime, python_config.get("venv_dir") or ".venv-fastvlm")
    evidence_path = runtime / FREEZE_EVIDENCE_NAME
    _validate_existing_target(target_venv, directory=True, description="FastVLM venv")
    _validate_existing_target(evidence_path, directory=False, description="FastVLM dependency evidence")
    if not target_venv.exists():
        raise RuntimeVerificationError(f"FastVLM venv is missing: {target_venv}")
    if not evidence_path.exists():
        raise RuntimeVerificationError(f"FastVLM dependency evidence is missing: {evidence_path}")
    audit_runtime_venv(target_venv)
    return "verified"


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        manifest = load_manifest(Path(args.manifest))
        root = runtime_root(manifest, override=args.runtime_root)
        if args.audit_only:
            status = audit_installed_runtime_venv(manifest, root=root)
        else:
            status = prepare_runtime_venv(
                manifest,
                root=root,
                base_python=Path(args.base_python),
                requirements=Path(args.requirements),
            )
    except (ManifestError, OSError, RuntimeVerificationError, UnicodeError, ValueError) as exc:
        print(f"FastVLM venv installation failed: {exc}", file=sys.stderr)
        return 1
    print(f"FastVLM governed venv: {status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
