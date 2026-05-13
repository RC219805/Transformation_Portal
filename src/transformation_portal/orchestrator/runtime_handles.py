"""In-process runtime handles for live orchestrator jobs.

These objects must NEVER be persisted: ``asyncio.subprocess.Process`` and
``asyncio.Task`` are local to the running event loop. They live here so the
persistent ``JobRepository`` can stay focused on data that survives restart.

The legacy ``app.py:Job`` dataclass carried ``proc`` and ``terminate_task``
alongside persistent fields; Phase 1 splits them.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class RuntimeHandles:
    """Live, non-persistent state for one job within this process."""

    proc: Optional[asyncio.subprocess.Process] = None
    terminate_task: Optional[asyncio.Task[None]] = None
    subscribers: Dict[str, "asyncio.Queue[Dict[str, Any]]"] = field(default_factory=dict)


class RuntimeRegistry:
    """Process-wide registry of live runtime handles, keyed by job id."""

    def __init__(self) -> None:
        self._handles: Dict[str, RuntimeHandles] = {}

    def ensure(self, job_id: str) -> RuntimeHandles:
        existing = self._handles.get(job_id)
        if existing is None:
            existing = RuntimeHandles()
            self._handles[job_id] = existing
        return existing

    def get(self, job_id: str) -> Optional[RuntimeHandles]:
        return self._handles.get(job_id)

    def pop(self, job_id: str) -> Optional[RuntimeHandles]:
        return self._handles.pop(job_id, None)

    def has(self, job_id: str) -> bool:
        return job_id in self._handles

    def live_job_ids(self) -> list[str]:
        """Return ids of jobs that currently have a non-finished process attached."""
        live: list[str] = []
        for jid, h in self._handles.items():
            proc = h.proc
            if proc is not None and proc.returncode is None:
                live.append(jid)
        return live

    def clear(self) -> None:
        """Test-only: drop everything. Production callers must not invoke."""
        self._handles.clear()


_registry: Optional[RuntimeRegistry] = None


def get_runtime_registry() -> RuntimeRegistry:
    global _registry
    if _registry is None:
        _registry = RuntimeRegistry()
    return _registry


def reset_runtime_registry() -> None:
    """Test-only: drop the singleton registry."""
    global _registry
    _registry = None


__all__ = [
    "RuntimeHandles",
    "RuntimeRegistry",
    "get_runtime_registry",
    "reset_runtime_registry",
]
