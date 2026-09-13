"""Closed locator envelope and process-local claim context for ADR-051."""

from __future__ import annotations

import json
import re
from contextvars import ContextVar
from dataclasses import asdict, dataclass
from typing import Any, Optional

from transformation_portal.ingest.canonical_json import canonicalize_json

LOCATOR_SCHEMA = "tp.dispatch.locator.v1"
_SAFE_ID = re.compile(r"[A-Za-z0-9_-]{1,64}\Z")
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")


@dataclass(frozen=True)
class DispatchLocator:
    """A queue message carries immutable identifiers, never execution code."""

    job_id: str
    attempt_id: str
    dispatch_id: str
    plan_digest: str
    tenant_id: str
    api_version: str = "v1"
    schema: str = LOCATOR_SCHEMA

    def __post_init__(self) -> None:
        for name in ("job_id", "attempt_id", "dispatch_id", "tenant_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not _SAFE_ID.fullmatch(value):
                raise ValueError(f"invalid locator {name}")
        if not isinstance(self.plan_digest, str) or not _DIGEST.fullmatch(self.plan_digest):
            raise ValueError("invalid locator plan_digest")
        if self.api_version != "v1" or self.schema != LOCATOR_SCHEMA:
            raise ValueError("unsupported dispatch locator version")

    def to_payload(self) -> dict[str, str]:
        return asdict(self)

    def to_json(self) -> str:
        return canonicalize_json(self.to_payload()).decode("utf-8")

    @classmethod
    def from_json(cls, raw: str) -> DispatchLocator:
        if not isinstance(raw, str) or len(raw.encode("utf-8")) > 2048:
            raise ValueError("dispatch locator exceeds byte limit")

        def unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
            result: dict[str, Any] = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError("duplicate dispatch locator field")
                result[key] = value
            return result

        payload = json.loads(raw, object_pairs_hook=unique_pairs)
        if not isinstance(payload, dict) or set(payload) != set(cls.__dataclass_fields__):
            raise ValueError("dispatch locator fields do not match the closed schema")
        return cls(**payload)


@dataclass(frozen=True)
class DispatchFence:
    locator: DispatchLocator
    holder: str
    lease_epoch: int
    lease_valid_until: float
    output_root: str
    requested_output_root: str


# Lease authority is created by the database claim inside a worker. It is
# never serialized into a broker message or supplied by the HTTP caller.
_dispatch_fence: ContextVar[Optional[DispatchFence]] = ContextVar("dispatch_fence", default=None)


def current_dispatch_fence() -> Optional[DispatchFence]:
    return _dispatch_fence.get()
