"""JSON file I/O helpers for Spatial AI orchestration artifacts."""

from __future__ import annotations

import contextlib
import os
import stat
import uuid
from pathlib import Path
from typing import Any, Optional

from transformation_portal.ingest.canonical_json import dump_json


def _open_atomic_temp(path: Path) -> tuple[Path, int]:
    """Open a unique same-directory temp file using normal umask semantics."""
    for _ in range(100):
        temp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            fd = os.open(temp_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
            return temp_path, fd
        except FileExistsError:
            continue
    raise FileExistsError(f"Could not allocate temporary file for {path}")


def write_json_atomic(
    path: Path,
    payload: Any,
    *,
    indent: Optional[int] = 2,
    sort_keys: bool = False,
    ensure_ascii: bool = True,
    allow_nan: bool = True,
    trailing_newline: bool = False,
) -> None:
    """Write JSON via same-directory temp file, fsync, and atomic replace.

    New files are created with the same mode that ``open(path, "w")`` would use
    under the active process umask. Existing files keep their previous mode.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    existing_mode: Optional[int] = None
    with contextlib.suppress(FileNotFoundError):
        existing_mode = stat.S_IMODE(path.stat().st_mode)

    temp_path, fd = _open_atomic_temp(path)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            dump_json(
                payload,
                handle,
                indent=indent,
                sort_keys=sort_keys,
                ensure_ascii=ensure_ascii,
                allow_nan=allow_nan,
            )
            if trailing_newline:
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())

        if existing_mode is not None:
            temp_path.chmod(existing_mode)
        temp_path.replace(path)
    except Exception:
        with contextlib.suppress(FileNotFoundError):
            temp_path.unlink()
        raise


__all__ = ["write_json_atomic"]
