"""Bounded, read-only photographic header checks inside a pinned input namespace."""

from __future__ import annotations

import io
import os
import stat
from pathlib import Path
from typing import Any, BinaryIO, cast

from transformation_portal.lux_depth_v3.execution_evidence import (
    _open_confined_artifact,
    _pin_output_root,
    _validate_confined_entry_identity,
    _validate_pinned_root_namespace,
)
from transformation_portal.lux_depth_v4.io import PHOTOGRAPHIC_INPUT_SUFFIXES
from transformation_portal.lux_depth_v4.photography import validate_input_color_metadata

# Header inspection must not decode pixels or scan an entire photographic batch.
_MAX_FILE_METADATA_BYTES = 4 * 1024**2
_MAX_BATCH_METADATA_BYTES = 64 * 1024**2
_MAX_FILES = 1024
_MAX_ENTRIES = 8192


class _MetadataReader:
    """A seekable parser stream with per-file and aggregate read limits."""

    def __init__(self, stream: BinaryIO, remaining: list[int]) -> None:
        self._stream = stream
        self._remaining = remaining
        self._file_remaining = _MAX_FILE_METADATA_BYTES

    def read(self, size: int = -1) -> bytes:
        if size < 0 or size > min(self._file_remaining, self._remaining[0]):
            raise ValueError("Photographic metadata exceeds the preflight read budget")
        data = self._stream.read(size)
        self._file_remaining -= len(data)
        self._remaining[0] -= len(data)
        return data

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        return self._stream.seek(offset, whence)

    def tell(self) -> int:
        return self._stream.tell()


def validate_input_directory_colors(
    root: Path, *, input_color: str, max_input_bytes: int, max_pixels: int
) -> list[dict[str, Any]]:
    """Reject unsupported input colors without model loads, pixel decode, or writes.

    Paths must already be authorized for the caller. Pinning and confined opens
    reject linked or concurrently replaced namespaces while header parsing runs.
    Execution remains responsible for full immutable input and pixel validation.
    """
    selected: list[Path] = []
    entries = 0
    remaining = [_MAX_BATCH_METADATA_BYTES]
    preparations: list[dict[str, Any]] = []

    with _pin_output_root(root) as pinned:
        pending = [root]
        while pending:
            current = pending.pop()
            descriptor = os.dup(pinned.descriptor) if current == root else _open_confined_artifact(pinned, current)[0]
            try:
                if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
                    raise ValueError("Input directory changed during preflight")
                with os.scandir(descriptor) as directory:
                    for entry in directory:
                        entries += 1
                        if entries > _MAX_ENTRIES:
                            raise ValueError("Input directory exceeds the preflight inventory budget")
                        path = current / entry.name
                        if entry.is_dir(follow_symlinks=False):
                            pending.append(path)
                        elif entry.is_symlink() and entry.is_dir():
                            raise ValueError("Input tree must not contain linked directories")
                        elif path.suffix.lower() in PHOTOGRAPHIC_INPUT_SUFFIXES:
                            selected.append(path)
                            if len(selected) > _MAX_FILES:
                                raise ValueError("Input count exceeds 1024")
            finally:
                os.close(descriptor)
        for path in sorted(selected, key=lambda item: item.relative_to(root).as_posix()):
            descriptor, relative = _open_confined_artifact(pinned, path)
            try:
                before = os.fstat(descriptor)
                if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or not 0 < before.st_size <= max_input_bytes:
                    raise ValueError("Input must be a bounded regular file without link aliases")
                with os.fdopen(os.dup(descriptor), "rb") as stream:
                    preparation = validate_input_color_metadata(
                        cast(BinaryIO, _MetadataReader(stream, remaining)),
                        source_name=relative,
                        input_color=input_color,
                        max_pixels=max_pixels,
                    )
                    if preparation is not None:
                        preparations.append(preparation)
                _validate_confined_entry_identity(pinned, relative, before, context="photographic input header")
            finally:
                os.close(descriptor)
        _validate_pinned_root_namespace(pinned)
    return preparations
