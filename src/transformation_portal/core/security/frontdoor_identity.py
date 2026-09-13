"""Bounded, request-bound identity assertions from the authenticated frontdoor.

The dedicated shared secret authenticates the frontdoor, never browser headers.
Tenant membership remains backend configuration; assertions cannot select it.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import re
import time
from typing import Any

from transformation_portal.core.security.path_safety import PathSafetyError, validate_safe_name

ACTOR_ASSERTION_HEADER = "x-tp-actor-assertion"
_ASSERTION_DOMAIN = b"tp.frontdoor.actor.v1\n"
_ASSERTION_MAX_AGE_SECONDS = 60
_ASSERTION_FUTURE_SKEW_SECONDS = 5


class IdentityConfigurationError(ValueError):
    """The server has no usable identity verification configuration."""


class IdentityAssertionError(ValueError):
    """An assertion does not authenticate this request."""


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate object key")
        result[key] = value
    return result


def parse_actor_tenants(raw: str) -> dict[str, str]:
    """Parse explicit Access-email membership, rejecting ambiguous/invalid maps."""
    try:
        if len(raw.encode("utf-8")) > 65536:
            raise ValueError("oversized mapping")
        parsed = json.loads(raw, object_pairs_hook=_unique_object)
        if not isinstance(parsed, dict) or not parsed:
            raise ValueError("empty mapping")
        result: dict[str, str] = {}
        for email, tenant_id in parsed.items():
            normalized = email.strip().lower()
            if not normalized or len(normalized) > 320 or normalized in result:
                raise ValueError("invalid or duplicate actor")
            if not isinstance(tenant_id, str):
                raise ValueError("invalid tenant")
            validate_safe_name(tenant_id)
            result[normalized] = tenant_id
        return result
    except (ValueError, PathSafetyError, RecursionError) as exc:
        raise IdentityConfigurationError("invalid TP_PILOT_ACTOR_TENANTS_JSON") from exc


def verify_actor_assertion(
    assertion: str,
    *,
    secret: str,
    method: str,
    target: str,
    now: float | None = None,
) -> dict[str, str]:
    """Verify exact request target/method, timestamp, signature, and closed claims."""
    key = secret.encode("utf-8")
    if len(key) < 32:
        raise IdentityConfigurationError("TP_FRONTDOOR_IDENTITY_SECRET requires at least 32 UTF-8 bytes")
    if not assertion or len(assertion) > 8192:
        raise IdentityAssertionError("missing or oversized actor assertion")
    try:
        encoded, signature = assertion.split(".")
        if not re.fullmatch(r"[A-Za-z0-9_-]+", encoded) or not re.fullmatch(r"[0-9a-f]{64}", signature):
            raise ValueError("invalid assertion encoding")
        expected = hmac.new(key, _ASSERTION_DOMAIN + encoded.encode("ascii"), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(signature, expected):
            raise ValueError("invalid assertion signature")
        payload = json.loads(
            base64.b64decode(encoded + "=" * (-len(encoded) % 4), altchars=b"-_", validate=True).decode("utf-8"),
            object_pairs_hook=_unique_object,
        )
        if not isinstance(payload, dict) or set(payload) != {"v", "iat", "method", "target", "actor"}:
            raise ValueError("invalid assertion claims")
        if type(payload["v"]) is not int or payload["v"] != 1:
            raise ValueError("invalid assertion version")
        issued = payload["iat"]
        timestamp = time.time() if now is None else now
        if (
            type(issued) is not int
            or not timestamp - _ASSERTION_MAX_AGE_SECONDS <= issued <= timestamp + _ASSERTION_FUTURE_SKEW_SECONDS
        ):
            raise ValueError("expired or future actor assertion")
        if payload["method"] != method.upper() or payload["target"] != target:
            raise ValueError("actor assertion request mismatch")
        actor = payload["actor"]
        if not isinstance(actor, dict) or set(actor) != {"username", "accessEmail", "role"}:
            raise ValueError("invalid actor claims")
        for value in actor.values():
            if not isinstance(value, str) or not value or len(value) > 320 or value != value.strip().lower():
                raise ValueError("invalid actor value")
        return {key: str(value) for key, value in actor.items()}
    except (ValueError, UnicodeError, TypeError, RecursionError) as exc:
        raise IdentityAssertionError("invalid actor assertion") from exc
