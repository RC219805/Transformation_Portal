"""Closed runtime evidence required before identity-v3 depth-cache access.

Backends may execute in the host process or in isolated interpreters.  This
module gives both paths one narrow hand-off contract: evidence is prepared
before a lookup and the same digest must be echoed by a cache-miss execution.
The execution identity remains the authority; this object only carries the
runtime facts needed to materialize it.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from transformation_portal.core.execution_identity_v3 import BackendRuntimeIdentity
from transformation_portal.ingest.canonical_json import canonicalize_json

DEPTH_CACHE_RUNTIME_EVIDENCE_SCHEMA = "tp.lux.depth_cache.runtime-evidence.v1"
_RUNTIME_EVIDENCE_DOMAIN = b"tp.lux.depth-cache.runtime-evidence.v1\0"
_RUNTIME_AGGREGATE_DOMAIN = b"tp.execution.runtime-aggregate.v1\0"
_HEX_DIGITS = frozenset("0123456789abcdef")


class DepthCacheRuntimeEvidenceError(ValueError):
    """Runtime evidence is incomplete, malformed, or non-authorizing."""


def _require_sha256(name: str, value: object) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(character not in _HEX_DIGITS for character in value):
        raise DepthCacheRuntimeEvidenceError(f"{name} must be a lowercase SHA-256 digest")
    if value == "0" * 64:
        raise DepthCacheRuntimeEvidenceError(f"{name} cannot be a placeholder digest")
    return value


def _aggregate_runtime_field(
    identities: tuple[BackendRuntimeIdentity, ...],
    field_name: str,
) -> str:
    if len(identities) == 1:
        return getattr(identities[0], field_name)
    projection = [
        {
            "constituent_ordinal": identity.constituent_ordinal,
            "backend_id": identity.backend_id,
            "model_canonical_key": identity.model_canonical_key,
            "model_lock_revision": identity.model_lock_revision,
            field_name: getattr(identity, field_name),
        }
        for identity in identities
    ]
    return hashlib.sha256(
        _RUNTIME_AGGREGATE_DOMAIN + field_name.encode("ascii") + b"\0" + canonicalize_json(projection)
    ).hexdigest()


@dataclass(frozen=True)
class PreparedDepthCacheRuntimeEvidence:
    """Verified backend evidence prepared before one cache lookup.

    ``runtime_identity_sha256`` is intentionally distinct from the complete
    execution-identity digest.  It lets an isolated inference worker prove it
    executed with the exact runtime prepared before the lookup, while the
    execution identity additionally binds the input and canonical plan.
    """

    backend_runtime_identities: tuple[BackendRuntimeIdentity, ...]
    dependency_lock_sha256: str
    interpreter_identity_sha256: str
    platform_identity_sha256: str
    accelerator_identity_sha256: str
    source_identity_sha256: str
    schema: str = field(default=DEPTH_CACHE_RUNTIME_EVIDENCE_SCHEMA, init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.backend_runtime_identities, tuple):
            raise DepthCacheRuntimeEvidenceError("backend_runtime_identities must be an immutable tuple")
        if not self.backend_runtime_identities or len(self.backend_runtime_identities) > 8:
            raise DepthCacheRuntimeEvidenceError("backend_runtime_identities must contain between one and eight constituents")
        if not all(isinstance(item, BackendRuntimeIdentity) for item in self.backend_runtime_identities):
            raise DepthCacheRuntimeEvidenceError("backend_runtime_identities must contain only BackendRuntimeIdentity values")
        ordinals = tuple(item.constituent_ordinal for item in self.backend_runtime_identities)
        if ordinals != tuple(sorted(set(ordinals))):
            raise DepthCacheRuntimeEvidenceError("backend_runtime_identities must use unique ascending constituent ordinals")
        for name in (
            "dependency_lock_sha256",
            "interpreter_identity_sha256",
            "platform_identity_sha256",
            "accelerator_identity_sha256",
            "source_identity_sha256",
        ):
            _require_sha256(name, getattr(self, name))
            if getattr(self, name) != _aggregate_runtime_field(self.backend_runtime_identities, name):
                raise DepthCacheRuntimeEvidenceError(f"{name} does not match the ordered backend runtime identities")

    @classmethod
    def create(
        cls,
        *,
        backend_runtime_identities: Iterable[BackendRuntimeIdentity],
        dependency_lock_sha256: str,
        interpreter_identity_sha256: str,
        platform_identity_sha256: str,
        accelerator_identity_sha256: str,
        source_identity_sha256: str,
    ) -> "PreparedDepthCacheRuntimeEvidence":
        return cls(
            backend_runtime_identities=tuple(backend_runtime_identities),
            dependency_lock_sha256=dependency_lock_sha256,
            interpreter_identity_sha256=interpreter_identity_sha256,
            platform_identity_sha256=platform_identity_sha256,
            accelerator_identity_sha256=accelerator_identity_sha256,
            source_identity_sha256=source_identity_sha256,
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "backend_runtime_identities": [item.to_payload() for item in self.backend_runtime_identities],
            "dependency_lock_sha256": self.dependency_lock_sha256,
            "interpreter_identity_sha256": self.interpreter_identity_sha256,
            "platform_identity_sha256": self.platform_identity_sha256,
            "accelerator_identity_sha256": self.accelerator_identity_sha256,
            "source_identity_sha256": self.source_identity_sha256,
        }

    @property
    def runtime_identity_sha256(self) -> str:
        return hashlib.sha256(_RUNTIME_EVIDENCE_DOMAIN + canonicalize_json(self.to_payload())).hexdigest()

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "PreparedDepthCacheRuntimeEvidence":
        """Parse a trusted-boundary payload with an exact, closed key set."""

        expected = {
            "schema",
            "backend_runtime_identities",
            "dependency_lock_sha256",
            "interpreter_identity_sha256",
            "platform_identity_sha256",
            "accelerator_identity_sha256",
            "source_identity_sha256",
        }
        if not isinstance(payload, Mapping) or set(payload) != expected:
            raise DepthCacheRuntimeEvidenceError("Runtime evidence must use the exact supported field set")
        if payload["schema"] != DEPTH_CACHE_RUNTIME_EVIDENCE_SCHEMA:
            raise DepthCacheRuntimeEvidenceError("Unsupported runtime-evidence schema")
        raw_identities = payload["backend_runtime_identities"]
        if not isinstance(raw_identities, list):
            raise DepthCacheRuntimeEvidenceError("backend_runtime_identities must be an array")
        try:
            identities = tuple(BackendRuntimeIdentity.from_payload(item) for item in raw_identities)
        except (TypeError, ValueError) as exc:
            raise DepthCacheRuntimeEvidenceError("Invalid backend runtime identity") from exc
        return cls.create(
            backend_runtime_identities=identities,
            dependency_lock_sha256=payload["dependency_lock_sha256"],
            interpreter_identity_sha256=payload["interpreter_identity_sha256"],
            platform_identity_sha256=payload["platform_identity_sha256"],
            accelerator_identity_sha256=payload["accelerator_identity_sha256"],
            source_identity_sha256=payload["source_identity_sha256"],
        )


__all__ = [
    "DEPTH_CACHE_RUNTIME_EVIDENCE_SCHEMA",
    "DepthCacheRuntimeEvidenceError",
    "PreparedDepthCacheRuntimeEvidence",
]
