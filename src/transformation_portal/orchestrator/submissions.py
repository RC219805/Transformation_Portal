"""Pure, versioned identities for optional durable job submissions.

The key is scoped to the verified tenant and composite portal/Access owner.
Intent identity never resolves paths, runs previews, or substitutes for the
canonical execution plan frozen by admission. API version and transport data
are deliberately absent so v1/v2 requests share the same admission identity.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any, Mapping

from transformation_portal.ingest.canonical_json import canonicalize_json

KEY_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,63}")
MAX_INTENT_BYTES = 1024 * 1024


def validate_submission_key(value: str) -> str:
    if not isinstance(value, str) or KEY_PATTERN.fullmatch(value) is None:
        raise ValueError("Submission key requires 1..64 portable opaque characters")
    return value


def _hash(payload: Any) -> str:
    return hashlib.sha256(canonicalize_json(payload)).hexdigest()


def verified_actor_owner(tenant_id: str, verified_actor: Mapping[str, str]) -> str:
    values = [verified_actor.get(name) for name in ("accessEmail", "username")]
    if (
        not isinstance(tenant_id, str)
        or not tenant_id
        or any(not isinstance(value, str) or not value.strip() or len(value) > 320 for value in values)
    ):
        raise ValueError("Submission ownership requires a verified tenant and composite actor")
    return _hash(
        {
            "schema": "tp.portal.actor_scope.v1",
            "tenant_id": tenant_id,
            "access_email": str(values[0]).strip().lower(),
            "username": str(values[1]).strip().lower(),
        }
    )


def submission_owner(tenant_id: str, verified_actor: Mapping[str, str]) -> str:
    return _hash(
        {
            "schema": "tp.job_submission.owner.v1",
            "operation": "job_submit",
            "actor_scope": verified_actor_owner(tenant_id, verified_actor),
        }
    )


def submission_intent(payload: dict[str, Any]) -> str:
    if not isinstance(payload, dict):
        raise ValueError("Submission intent must be a JSON object")
    try:

        def validate(value: Any, depth: int = 0) -> None:
            if depth > 32:
                raise ValueError("Submission JSON nesting exceeds its bound")
            if isinstance(value, dict):
                if any(not isinstance(key, str) for key in value):
                    raise ValueError("Submission JSON keys must be strings")
                for item in value.values():
                    validate(item, depth + 1)
            elif isinstance(value, list):
                for item in value:
                    validate(item, depth + 1)
            elif value is not None and type(value) not in {str, int, float, bool}:
                raise ValueError("Submission intent must contain JSON primitives")

        validate(payload)
        raw = canonicalize_json({"schema": "tp.job_submission.intent.v1", "request": payload})
    except (TypeError, ValueError, UnicodeError, RecursionError) as error:
        raise ValueError("Submission intent must contain bounded finite JSON") from error
    if len(raw) > MAX_INTENT_BYTES:
        raise ValueError("Submission intent exceeds its byte bound")
    return hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True)
class SubmissionIdentity:
    key: str
    owner_sha256: str
    intent_sha256: str

    def __post_init__(self) -> None:
        validate_submission_key(self.key)
        for value in (self.owner_sha256, self.intent_sha256):
            if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
                raise ValueError("Submission identity requires lowercase SHA-256 bindings")


@dataclass(frozen=True)
class SubmissionReceipt:
    key: str
    intent_sha256: str
    job_id: str
    pipeline: str
    state: str
    retired: bool


class SubmissionConflict(ValueError):
    """A consumed key cannot authorize a different intent."""
