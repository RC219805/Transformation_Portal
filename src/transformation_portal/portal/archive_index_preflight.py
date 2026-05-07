"""Archive index preflight helpers for portal archive-gate readiness."""

from __future__ import annotations

import copy
import csv
import gzip
import os
import re
import threading
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, MutableMapping, Optional, Tuple

from transformation_portal.portal import path_security

REPO_ROOT = Path(__file__).resolve().parents[3]

ARCHIVE_INDEX_REQUIRED_COLUMNS = {"origin_drive", "partition", "relpath"}
ARCHIVE_INDEX_PREFLIGHT_EXAMPLE_LIMIT = 5
ARCHIVE_INDEX_PREFLIGHT_PREVIEW_ROW_LIMIT = 256
ARCHIVE_INDEX_PREFLIGHT_CACHE_MAX = 64
ARCHIVE_INDEX_PREFLIGHT_SCAN_MODES = {"preview", "full"}
_ARCHIVE_INDEX_PREFLIGHT_CACHE_LOCK = threading.Lock()
_ARCHIVE_INDEX_PREFLIGHT_CACHE: Dict[Tuple[str, int, int, str, str], Dict[str, Any]] = {}


def _repo_root(repo_root: Path | None = None) -> Path:
    return Path(os.path.realpath(REPO_ROOT if repo_root is None else repo_root))


def _archive_index_preflight_example(
    *,
    row: int,
    relpath: str,
    reason: str,
) -> Dict[str, Any]:
    return {
        "row": row,
        "relpath": relpath,
        "reason": reason,
    }


def _archive_index_preflight_result(
    *,
    rows_total: int,
    blocked_rows: int,
    examples: List[Dict[str, Any]],
    scan_mode: str = "full",
    truncated: bool = False,
) -> Dict[str, Any]:
    return {
        "ok": blocked_rows == 0 and rows_total > 0,
        "rows_total": rows_total,
        "blocked_rows": blocked_rows,
        "examples": examples[:ARCHIVE_INDEX_PREFLIGHT_EXAMPLE_LIMIT],
        "scan_mode": scan_mode,
        "truncated": bool(truncated),
    }


def _archive_index_preflight_message(result: Mapping[str, Any]) -> str:
    rows_total = int(result.get("rows_total") or 0)
    blocked_rows = int(result.get("blocked_rows") or 0)
    examples = result.get("examples") if isinstance(result.get("examples"), list) else []
    example_text = "; ".join(
        f"row {item.get('row')}: {item.get('relpath')!r} ({item.get('reason')})"
        for item in examples[:ARCHIVE_INDEX_PREFLIGHT_EXAMPLE_LIMIT]
        if isinstance(item, Mapping)
    )
    if not example_text:
        example_text = "no valid archive index rows found"
    if rows_total == 0:
        return (
            "Archive index does not match the selected archive root: "
            f"{blocked_rows} blocking issue before row validation. Examples: {example_text}."
        )
    return (
        "Archive index does not match the selected archive root: "
        f"{blocked_rows}/{rows_total} rows blocked. Examples: {example_text}."
    )


def _archive_index_preflight_primary_reason(result: Mapping[str, Any]) -> str:
    examples = result.get("examples") if isinstance(result.get("examples"), list) else []
    first = examples[0] if examples and isinstance(examples[0], Mapping) else {}
    return str(first.get("reason") or "").strip()


def _archive_index_preflight_root_reason(result: Mapping[str, Any]) -> Optional[str]:
    reason = _archive_index_preflight_primary_reason(result)
    if reason.startswith("archive_root_"):
        return reason
    return None


def _is_drive_prefixed_relpath(raw_relpath: str) -> bool:
    return bool(re.match(r"^[A-Za-z]:", raw_relpath))


def _archive_index_preflight_scan_mode(scan_mode: str) -> str:
    normalized = str(scan_mode or "full").strip().lower()
    return normalized if normalized in ARCHIVE_INDEX_PREFLIGHT_SCAN_MODES else "full"


def _archive_index_preflight_row_limit(scan_mode: str) -> Optional[int]:
    if _archive_index_preflight_scan_mode(scan_mode) == "preview":
        return ARCHIVE_INDEX_PREFLIGHT_PREVIEW_ROW_LIMIT
    return None


def _copy_archive_index_preflight_result(result: Mapping[str, Any]) -> Dict[str, Any]:
    return copy.deepcopy(dict(result))


def _archive_index_preflight_cache_key(
    archive_index: Path,
    archive_root: Path,
    *,
    scan_mode: str,
) -> Tuple[str, int, int, str, str]:
    stat_result = archive_index.stat()
    return (
        str(archive_index),
        int(stat_result.st_mtime_ns),
        int(stat_result.st_size),
        str(archive_root),
        _archive_index_preflight_scan_mode(scan_mode),
    )


def _trusted_existing_entry_without_realpath(
    path_value: Any,
    allowed_roots: List[Path],
    *,
    repo_root: Path | None = None,
) -> Optional[Path]:
    raw = str(path_value or "").strip()
    if not raw or raw.startswith("~") or "\x00" in raw:
        return None
    try:
        candidate = Path(raw)
    except (OSError, RuntimeError, ValueError):
        return None
    if not candidate.is_absolute():
        candidate = _repo_root(repo_root) / candidate
    try:
        candidate_absolute = Path(os.path.abspath(candidate))
    except (OSError, RuntimeError, ValueError):
        return None

    for root in allowed_roots:
        try:
            root_real = Path(os.path.realpath(root))
            relative_parts = candidate_absolute.relative_to(root_real).parts
        except (OSError, RuntimeError, ValueError):
            continue
        current = root_real
        if not relative_parts:
            return current
        for index, part in enumerate(relative_parts):
            if part in {"", ".", ".."} or path_security._UNSAFE_PATH_SEGMENT_RE.search(part):
                return None
            try:
                next_path = next((child for child in current.iterdir() if child.name == part), None)
            except (NotADirectoryError, FileNotFoundError, OSError, RuntimeError, ValueError):
                return None
            if next_path is None:
                return None
            try:
                if index < len(relative_parts) - 1 and next_path.is_symlink():
                    return None
            except (OSError, RuntimeError):
                return None
            current = next_path
        return current
    return None


def _validate_archive_index_relpath(
    raw_relpath: Any,
    *,
    archive_root: Path,
) -> Tuple[bool, str, str]:
    relpath = str(raw_relpath or "").strip()
    if not relpath:
        return False, relpath, "empty_relpath"
    if "\x00" in relpath:
        return False, relpath, "nul_relpath"
    if relpath.startswith(("/", "\\")):
        return False, relpath, "absolute_relpath"
    if _is_drive_prefixed_relpath(relpath):
        return False, relpath, "drive_prefixed_relpath"

    normalized = relpath.replace("\\", "/")
    parts = tuple(part for part in normalized.split("/") if part and part != ".")
    if not parts:
        return False, relpath, "empty_relpath"
    if any(part == ".." for part in parts):
        return False, relpath, "parent_traversal"

    current = archive_root
    for part in parts:
        try:
            next_path = next((child for child in current.iterdir() if child.name == part), None)
        except FileNotFoundError:
            return False, relpath, "missing"
        except (NotADirectoryError, OSError, RuntimeError, ValueError):
            return False, relpath, "unreadable"
        if next_path is None:
            return False, relpath, "missing"
        try:
            if next_path.is_symlink():
                return False, relpath, "symlink_traversal"
        except (OSError, RuntimeError):
            return False, relpath, "unreadable"
        current = next_path

    try:
        if current.is_dir():
            return False, relpath, "directory"
        if not current.is_file():
            return False, relpath, "not_regular_file"
    except (OSError, RuntimeError):
        return False, relpath, "unreadable"

    return True, relpath, "ok"


def _default_allowed_roots(repo_root: Path | None) -> List[Path]:
    return path_security._default_allowed_path_roots(repo_root=_repo_root(repo_root))


def _validate_archive_index_against_root(
    archive_index: Path,
    archive_root: Path,
    *,
    scan_mode: str = "full",
    allowed_path_roots: Optional[List[Path]] = None,
    allowed_input_roots: Optional[List[Path]] = None,
    repo_root: Path | None = None,
    cache: MutableMapping[Tuple[str, int, int, str, str], Dict[str, Any]] | None = None,
    cache_lock: Any | None = None,
    cache_max_entries: int = ARCHIVE_INDEX_PREFLIGHT_CACHE_MAX,
    relpath_validator: Callable[..., Tuple[bool, str, str]] = _validate_archive_index_relpath,
) -> Dict[str, Any]:
    scan_mode = _archive_index_preflight_scan_mode(scan_mode)
    if cache_max_entries < 1:
        raise ValueError("cache_max_entries must be at least 1")
    resolved_repo_root = _repo_root(repo_root)
    path_roots = allowed_path_roots if allowed_path_roots is not None else _default_allowed_roots(resolved_repo_root)
    input_roots = allowed_input_roots if allowed_input_roots is not None else _default_allowed_roots(resolved_repo_root)
    preflight_cache = cache if cache is not None else _ARCHIVE_INDEX_PREFLIGHT_CACHE
    preflight_cache_lock = cache_lock if cache_lock is not None else _ARCHIVE_INDEX_PREFLIGHT_CACHE_LOCK

    try:
        trusted_archive_index = path_security._ensure_safe_regular_file_path(
            archive_index,
            path_roots,
            repo_root=resolved_repo_root,
        )
    except (OSError, RuntimeError, ValueError, path_security.PathSecurityValidationError):
        return _archive_index_preflight_result(
            rows_total=0,
            blocked_rows=1,
            examples=[
                _archive_index_preflight_example(
                    row=0,
                    relpath=str(archive_index),
                    reason="archive_index_unreadable",
                )
            ],
            scan_mode=scan_mode,
        )

    trusted_archive_root = _trusted_existing_entry_without_realpath(
        archive_root,
        input_roots,
        repo_root=resolved_repo_root,
    )
    if trusted_archive_root is None:
        return _archive_index_preflight_result(
            rows_total=0,
            blocked_rows=1,
            examples=[
                _archive_index_preflight_example(
                    row=0,
                    relpath=str(archive_root),
                    reason="archive_root_not_directory",
                )
            ],
            scan_mode=scan_mode,
        )
    try:
        if trusted_archive_root.is_symlink():
            return _archive_index_preflight_result(
                rows_total=0,
                blocked_rows=1,
                examples=[
                    _archive_index_preflight_example(
                        row=0,
                        relpath=str(archive_root),
                        reason="archive_root_symlink",
                    )
                ],
                scan_mode=scan_mode,
            )
        if not trusted_archive_root.is_dir():
            return _archive_index_preflight_result(
                rows_total=0,
                blocked_rows=1,
                examples=[
                    _archive_index_preflight_example(
                        row=0,
                        relpath=str(archive_root),
                        reason="archive_root_not_directory",
                    )
                ],
                scan_mode=scan_mode,
            )
        archive_root_real = Path(os.path.realpath(trusted_archive_root))
    except (OSError, RuntimeError):
        return _archive_index_preflight_result(
            rows_total=0,
            blocked_rows=1,
            examples=[
                _archive_index_preflight_example(
                    row=0,
                    relpath=str(archive_root),
                    reason="archive_root_unreadable",
                )
            ],
            scan_mode=scan_mode,
        )

    try:
        cache_key = _archive_index_preflight_cache_key(
            trusted_archive_index,
            archive_root_real,
            scan_mode=scan_mode,
        )
    except OSError:
        return _archive_index_preflight_result(
            rows_total=0,
            blocked_rows=1,
            examples=[
                _archive_index_preflight_example(
                    row=0,
                    relpath=str(archive_index),
                    reason="archive_index_unreadable",
                )
            ],
            scan_mode=scan_mode,
        )

    with preflight_cache_lock:
        cached = preflight_cache.get(cache_key)
    if cached is not None:
        return _copy_archive_index_preflight_result(cached)

    rows_total = 0
    blocked_rows = 0
    examples: List[Dict[str, Any]] = []
    row_limit = _archive_index_preflight_row_limit(scan_mode)
    truncated = False

    try:
        opener: Callable[..., Any] = gzip.open if trusted_archive_index.name.endswith(".gz") else Path.open
        with opener(trusted_archive_index, "rt", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = set(reader.fieldnames or [])
            missing_columns = sorted(ARCHIVE_INDEX_REQUIRED_COLUMNS - fieldnames)
            if missing_columns:
                return _archive_index_preflight_result(
                    rows_total=0,
                    blocked_rows=1,
                    examples=[
                        _archive_index_preflight_example(
                            row=1,
                            relpath="",
                            reason=f"missing_columns:{','.join(missing_columns)}",
                        )
                    ],
                    scan_mode=scan_mode,
                )

            for row_number, row in enumerate(reader, start=2):
                rows_total += 1
                ok, relpath, reason = relpath_validator(
                    row.get("relpath"),
                    archive_root=archive_root_real,
                )
                if row_limit is not None and rows_total >= row_limit:
                    truncated = True
                if ok:
                    if truncated:
                        break
                    continue
                blocked_rows += 1
                if len(examples) < ARCHIVE_INDEX_PREFLIGHT_EXAMPLE_LIMIT:
                    examples.append(
                        _archive_index_preflight_example(
                            row=row_number,
                            relpath=relpath,
                            reason=reason,
                        )
                    )
                if truncated:
                    break
    except (csv.Error, gzip.BadGzipFile, OSError, RuntimeError, UnicodeDecodeError) as exc:
        return _archive_index_preflight_result(
            rows_total=rows_total,
            blocked_rows=max(1, blocked_rows),
            examples=[
                _archive_index_preflight_example(
                    row=max(1, rows_total + 1),
                    relpath=str(trusted_archive_index),
                    reason=f"archive_index_unreadable:{type(exc).__name__}",
                )
            ],
            scan_mode=scan_mode,
        )

    if rows_total == 0:
        return _archive_index_preflight_result(
            rows_total=0,
            blocked_rows=1,
            examples=[
                _archive_index_preflight_example(
                    row=1,
                    relpath="",
                    reason="empty_archive_index",
                )
            ],
            scan_mode=scan_mode,
        )

    result = _archive_index_preflight_result(
        rows_total=rows_total,
        blocked_rows=blocked_rows,
        examples=examples,
        scan_mode=scan_mode,
        truncated=truncated,
    )
    with preflight_cache_lock:
        if len(preflight_cache) >= cache_max_entries:
            preflight_cache.pop(next(iter(preflight_cache)), None)
        preflight_cache[cache_key] = _copy_archive_index_preflight_result(result)
    return result
