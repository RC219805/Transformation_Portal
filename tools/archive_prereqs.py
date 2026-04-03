#!/usr/bin/env python3
"""Filesystem prerequisite helpers for archive tooling."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

KIND_FILE = "file"
KIND_DIR = "dir"
REASON_NOT_FOUND = "not_found"
REASON_WRONG_TYPE = "wrong_type"


@dataclass(frozen=True)
class ArchivePrereqError(Exception):
    """Represents a missing or wrong-type archive prerequisite path."""

    arg_name: str
    label: str
    path: Path
    expected_kind: str
    reason: str

    def message_text(self) -> str:
        path_text = str(self.path)
        if self.reason == REASON_NOT_FOUND:
            return f"{self.label} not found: {path_text}"
        if self.expected_kind == KIND_FILE:
            return f"{self.label} must be a regular file: {path_text}"
        return f"{self.label} must be a directory: {path_text}"

    def cli_message(self) -> str:
        return f"Error: {self.message_text()}"


def ensure_regular_file(path: str | Path, *, arg_name: str, label: str) -> Path:
    candidate = Path(path)
    if not candidate.exists():
        raise ArchivePrereqError(
            arg_name=arg_name,
            label=label,
            path=candidate,
            expected_kind=KIND_FILE,
            reason=REASON_NOT_FOUND,
        )
    if not candidate.is_file():
        raise ArchivePrereqError(
            arg_name=arg_name,
            label=label,
            path=candidate,
            expected_kind=KIND_FILE,
            reason=REASON_WRONG_TYPE,
        )
    return candidate


def ensure_directory(path: str | Path, *, arg_name: str, label: str) -> Path:
    candidate = Path(path)
    if not candidate.exists():
        raise ArchivePrereqError(
            arg_name=arg_name,
            label=label,
            path=candidate,
            expected_kind=KIND_DIR,
            reason=REASON_NOT_FOUND,
        )
    if not candidate.is_dir():
        raise ArchivePrereqError(
            arg_name=arg_name,
            label=label,
            path=candidate,
            expected_kind=KIND_DIR,
            reason=REASON_WRONG_TYPE,
        )
    return candidate
