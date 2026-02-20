from __future__ import annotations

import secrets
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class TraceContext:
    trace_id: str
    parent_id: str
    trace_flags: str = "01"  # sampled by default
    version: str = "00"

    @property
    def traceparent(self) -> str:
        return f"{self.version}-{self.trace_id}-{self.parent_id}-{self.trace_flags}"


def _is_hex(s: str, n: int) -> bool:
    return len(s) == n and all(c in "0123456789abcdef" for c in s)


def parse_traceparent(tp: str) -> TraceContext:
    parts = tp.strip().lower().split("-")
    if len(parts) != 4:
        raise ValueError("Invalid traceparent format")
    version, trace_id, parent_id, flags = parts
    if not _is_hex(version, 2):
        raise ValueError("Invalid traceparent version")
    if not _is_hex(trace_id, 32) or trace_id == "0" * 32:
        raise ValueError("Invalid trace_id")
    if not _is_hex(parent_id, 16) or parent_id == "0" * 16:
        raise ValueError("Invalid parent_id")
    if not _is_hex(flags, 2):
        raise ValueError("Invalid trace_flags")
    return TraceContext(trace_id=trace_id, parent_id=parent_id, trace_flags=flags, version=version)


def new_trace_context(sampled: bool = True) -> TraceContext:
    trace_id = secrets.token_hex(16)
    parent_id = secrets.token_hex(8)
    flags = "01" if sampled else "00"
    return TraceContext(trace_id=trace_id, parent_id=parent_id, trace_flags=flags)


def get_or_create_trace_context(traceparent: Optional[str]) -> TraceContext:
    if traceparent:
        return parse_traceparent(traceparent)
    return new_trace_context()
