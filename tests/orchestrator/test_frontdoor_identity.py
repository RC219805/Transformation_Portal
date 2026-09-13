"""Cross-language signed identity and fail-closed verification contracts."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
from pathlib import Path

import pytest

from transformation_portal.core.security.frontdoor_identity import (
    IdentityAssertionError,
    IdentityConfigurationError,
    parse_actor_tenants,
    verify_actor_assertion,
)

pytestmark = pytest.mark.unit
FIXTURE = json.loads((Path(__file__).parents[1] / "fixtures/frontdoor_identity_v1.json").read_text())


def _verify(assertion=None, **changes):
    kwargs = {
        "secret": FIXTURE["secret"],
        "method": FIXTURE["method"],
        "target": FIXTURE["target"],
        "now": FIXTURE["now"] / 1000,
    }
    kwargs.update(changes)
    return verify_actor_assertion(FIXTURE["assertion"] if assertion is None else assertion, **kwargs)


def test_node_signature_verifies_exact_identity_and_request():
    assert _verify() == FIXTURE["actor"]


@pytest.mark.parametrize(
    "changes",
    [
        {"method": "POST"},
        {"target": "/v1/jobs"},
        {"target": FIXTURE["target"].replace("%20", " ")},
        {"target": FIXTURE["target"].replace("download=1", "download=0")},
        {"now": FIXTURE["now"] / 1000 + 61},
        {"now": FIXTURE["now"] / 1000 - 6},
        {"secret": "different-secret-with-at-least-32-bytes"},
    ],
)
def test_assertion_rejects_different_request_time_or_signer(changes):
    with pytest.raises(IdentityAssertionError):
        _verify(**changes)


@pytest.mark.parametrize(
    "assertion", ["", "raw@example.com", "x" * 8193, FIXTURE["assertion"] + "x", FIXTURE["assertion"].replace(".", ".00.")]
)
def test_malformed_assertion_fails_closed(assertion):
    with pytest.raises(IdentityAssertionError):
        _verify(assertion)


@pytest.mark.parametrize(
    "claims", [{"v": True}, {"iat": True}, {"tenant_id": "tenant_b"}, {"actor": {"accessEmail": "admin@example.com"}}]
)
def test_signed_but_invalid_claims_fail_closed(claims):
    payload = {
        "v": 1,
        "iat": FIXTURE["now"] // 1000,
        "method": FIXTURE["method"],
        "target": FIXTURE["target"],
        "actor": FIXTURE["actor"],
    }
    payload.update(claims)
    encoded = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("=")
    signature = hmac.new(FIXTURE["secret"].encode(), b"tp.frontdoor.actor.v1\n" + encoded.encode(), hashlib.sha256).hexdigest()
    with pytest.raises(IdentityAssertionError):
        _verify(encoded + "." + signature)


@pytest.mark.parametrize(
    "mapping",
    [
        "{}",
        "[]",
        "null",
        "not json",
        '{"a":"../escape"}',
        '{"A@example.com":"tenant_a","a@example.com":"tenant_b"}',
        '{"a":"tenant_a","a":"tenant_b"}',
        '{"a":false}',
        " " * 65537,
        "[" * 2000 + "0" + "]" * 2000,
    ],
)
def test_membership_config_rejects_invalid_ambiguous_or_oversized_mapping(mapping):
    with pytest.raises(IdentityConfigurationError):
        parse_actor_tenants(mapping)


def test_membership_config_normalizes_access_identity_only():
    assert parse_actor_tenants('{" Admin@Example.com ":"tenant_a"}') == {"admin@example.com": "tenant_a"}
