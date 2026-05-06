"""Portal path resolution and filesystem trust helpers."""

from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path
from typing import Any, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[3]
_UNSAFE_PATH_SEGMENT_RE = re.compile(r"[\x00/\\]")


class PathSecurityValidationError(ValueError):
    """Validation error raised by app-independent path security helpers."""

    def __init__(self, message: str, *, reason: str = "invalid_path_value") -> None:
        cleaned_message = str(message or "").strip() or "invalid request"
        self.reason = str(reason or "invalid_path_value").strip() or "invalid_path_value"
        super().__init__(cleaned_message)


def _repo_root(repo_root: Path | None = None) -> Path:
    return Path(os.path.realpath(REPO_ROOT if repo_root is None else repo_root))


def _normalize_root_path(value: str | Path, *, repo_root: Path | None = None) -> Path:
    raw = str(value or "").strip()
    if not raw or raw.startswith("~") or "\x00" in raw:
        raise ValueError("Invalid path root")
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = _repo_root(repo_root) / candidate
    return Path(os.path.realpath(candidate))


def _default_allowed_path_roots(*, repo_root: Path | None = None) -> List[Path]:
    root = _repo_root(repo_root)
    roots: List[Path] = [root]
    candidate_paths: List[Path] = [Path(tempfile.gettempdir()).resolve()]
    if os.name != "nt":
        # Accept the common POSIX temp aliases used by operators and local tooling.
        candidate_paths.extend([Path("/tmp"), Path("/private/tmp")])

    for candidate in candidate_paths:
        try:
            normalized = _normalize_root_path(candidate, repo_root=root)
        except (OSError, RuntimeError, ValueError):
            continue
        if normalized not in roots:
            roots.append(normalized)
    return roots


def _env_path_roots(
    name: str,
    default: List[Path],
    *,
    repo_root: Path | None = None,
    logger: Any | None = None,
) -> List[Path]:
    raw = os.getenv(name)
    if raw is None:
        values = [str(path) for path in default]
    else:
        values = [item.strip() for item in raw.split(",") if item.strip()]

    roots: List[Path] = []
    invalid_values: List[str] = []
    for value in values:
        try:
            root = _normalize_root_path(value, repo_root=repo_root)
        except (OSError, RuntimeError, ValueError):
            invalid_values.append(value)
            continue
        if root not in roots:
            roots.append(root)
    if invalid_values and logger is not None:
        logger.warning(
            "%s ignored invalid roots: %s",
            name,
            ", ".join(sorted(set(invalid_values))),
        )
    if raw is not None and not roots:
        raise RuntimeError(f"{name} is set but contains no valid roots")
    if not roots:
        raise RuntimeError(f"{name} resolved to an empty root allowlist")
    return roots


def _resolve_untrusted_request_path(path_value: str, *, repo_root: Path | None = None) -> Path:
    raw = str(path_value or "").strip()
    if not raw or raw.startswith("~") or "\x00" in raw:
        raise ValueError("Invalid path value")
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = _repo_root(repo_root) / candidate
    return Path(os.path.realpath(candidate))


def _validate_path_against_roots(
    path_value: str,
    allowed_roots: List[Path],
    *,
    repo_root: Path | None = None,
) -> str:
    return str(_resolve_allowed_request_path(path_value, allowed_roots, repo_root=repo_root))


def _resolve_allowed_request_path(
    path_value: str,
    allowed_roots: List[Path],
    *,
    repo_root: Path | None = None,
) -> Path:
    if not allowed_roots:
        raise PathSecurityValidationError("No allowed roots configured", reason="invalid_path_value")

    try:
        resolved = _resolve_untrusted_request_path(path_value, repo_root=repo_root)
    except (OSError, RuntimeError, ValueError):
        raise PathSecurityValidationError("Invalid path value", reason="invalid_path_value") from None

    for root in allowed_roots:
        try:
            root_real = Path(os.path.realpath(root))
        except (OSError, RuntimeError, ValueError):
            continue
        try:
            resolved.relative_to(root_real)
        except ValueError:
            continue
        else:
            return resolved

    raise PathSecurityValidationError("Path outside allowed roots", reason="path_outside_allowed_roots")


def _path_is_within_root(resolved_path: Path, root: Path) -> bool:
    try:
        resolved_path.relative_to(Path(os.path.realpath(root)))
    except ValueError:
        return False
    return True


def _trusted_allowed_entry(
    resolved_path: Path,
    allowed_roots: List[Path],
) -> Optional[Path]:
    for root in allowed_roots:
        try:
            root_real = Path(os.path.realpath(root))
            relative_parts = resolved_path.relative_to(root_real).parts
        except (OSError, RuntimeError, ValueError):
            continue
        current = root_real
        if not relative_parts:
            return current
        for part in relative_parts:
            if part in {"", ".", ".."}:
                return None
            try:
                next_path = next((child for child in current.iterdir() if child.name == part), None)
            except (NotADirectoryError, FileNotFoundError, OSError, RuntimeError, ValueError):
                return None
            if next_path is None:
                return None
            current = next_path
        return current
    return None


def _trusted_existing_dir(value: str, allowed_roots: List[Path], *, repo_root: Path | None = None) -> Optional[Path]:
    """Return ``value`` as a trusted existing directory inside an allowed root."""

    try:
        resolved = _resolve_allowed_request_path(value, allowed_roots, repo_root=repo_root)
    except PathSecurityValidationError:
        return None
    trusted = _trusted_allowed_entry(resolved, allowed_roots)
    if trusted is None:
        return None
    try:
        if not trusted.is_dir():
            return None
    except OSError:
        return None
    return trusted


def _trusted_creatable_dir(value: str, allowed_roots: List[Path], *, repo_root: Path | None = None) -> Optional[Path]:
    """Return ``value`` as a trusted directory path safe to ``mkdir`` later."""

    try:
        resolved = _resolve_allowed_request_path(value, allowed_roots, repo_root=repo_root)
    except PathSecurityValidationError:
        return None
    for root in allowed_roots:
        try:
            root_real = Path(os.path.realpath(root))
            relative_parts = resolved.relative_to(root_real).parts
        except (OSError, RuntimeError, ValueError):
            continue
        current = root_real
        creating = False
        for part in relative_parts:
            if part in {"", ".", ".."} or _UNSAFE_PATH_SEGMENT_RE.search(part):
                return None
            if creating:
                current = current / part
                continue
            try:
                next_path = next((child for child in current.iterdir() if child.name == part), None)
            except (NotADirectoryError, FileNotFoundError, OSError, RuntimeError, ValueError):
                return None
            if next_path is None:
                creating = True
                current = current / part
            else:
                current = next_path
        return current
    return None


def _ensure_safe_regular_file_path(
    path_value: Path,
    allowed_roots: List[Path],
    *,
    repo_root: Path | None = None,
) -> Path:
    try:
        candidate_real = _resolve_allowed_request_path(str(path_value), allowed_roots, repo_root=repo_root)
    except PathSecurityValidationError:
        raise
    except (OSError, RuntimeError, ValueError) as exc:
        raise PathSecurityValidationError("Invalid path value", reason="invalid_path_value") from exc
    trusted_entry = _trusted_allowed_entry(candidate_real, allowed_roots)
    if trusted_entry is None:
        raise PathSecurityValidationError("Invalid path value", reason="invalid_path_value")
    try:
        if not trusted_entry.is_file():
            raise PathSecurityValidationError("Invalid path value", reason="invalid_path_value")
    except OSError as exc:
        raise PathSecurityValidationError("Invalid path value", reason="invalid_path_value") from exc
    return trusted_entry
