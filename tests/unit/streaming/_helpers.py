"""Shared helpers for streaming stage unit tests."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

T = TypeVar("T")


class RecordingIOPool:
    """Small async worker-pool stand-in for deterministic process() tests."""

    def __init__(self, *, record_result: bool = False) -> None:
        self._record_result = record_result
        self.calls: list[Path] = []

    async def run_io(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        result = func(*args, **kwargs)
        recorded = result if self._record_result else args[0]
        if not isinstance(recorded, Path):
            raise TypeError("RecordingIOPool only records pathlib.Path values")
        self.calls.append(recorded)
        return result
