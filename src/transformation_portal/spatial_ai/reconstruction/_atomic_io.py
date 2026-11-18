"""Same-directory temp-file publication for reconstruction exports.

Exporters write through :func:`atomic_open` so that a write which fails part
way (invalid splat arrays, a full disk, an interrupted process) never replaces
or truncates a previously exported file at the destination.
"""

from __future__ import annotations

import contextlib
import os
import stat
import uuid
from pathlib import Path
from typing import IO, Any, Iterator

_ALLOWED_MODES = frozenset({"w", "wb"})


@contextlib.contextmanager
def atomic_open(path: Path, mode: str, **open_kwargs: Any) -> Iterator[IO[Any]]:
    """Open a temp file beside ``path`` and publish it there on a clean exit.

    The temp file is flushed, fsynced and moved over ``path`` with
    :func:`os.replace` only when the ``with`` block finishes without an
    exception; otherwise it is removed and ``path`` keeps its previous content.
    New files get the mode ``open(path, mode)`` would give under the process
    umask; an existing destination keeps its mode.
    """
    if mode not in _ALLOWED_MODES:
        raise ValueError(f"atomic_open supports modes {sorted(_ALLOWED_MODES)}, got {mode!r}")
    path = Path(path)
    existing_mode = None
    with contextlib.suppress(FileNotFoundError):
        existing_mode = stat.S_IMODE(path.stat().st_mode)

    temp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    fd = os.open(temp_path, flags, 0o666)
    try:
        handle = os.fdopen(fd, mode, **open_kwargs)
    except BaseException:
        os.close(fd)
        temp_path.unlink(missing_ok=True)
        raise
    try:
        with handle:
            yield handle
            handle.flush()
            os.fsync(handle.fileno())
        if existing_mode is not None:
            os.chmod(temp_path, existing_mode)
        os.replace(temp_path, path)
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise


__all__ = ["atomic_open"]
