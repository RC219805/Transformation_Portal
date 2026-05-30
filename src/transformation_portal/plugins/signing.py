"""Signed manifest verification for opt-in external plugins."""

from __future__ import annotations

import hashlib
import hmac
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from transformation_portal.ingest.canonical_json import dumps_json

PLUGIN_SIGNATURE_ALGORITHM = "hmac-sha256"
PLUGIN_SIGNATURE_FIELDS = frozenset(
    {
        "signature",
        "signature_algorithm",
        "signature_key_id",
    }
)


class PluginSignatureError(ValueError):
    """Raised when a plugin manifest does not match the configured trust set."""


def canonical_manifest_payload(manifest_data: Mapping[str, Any]) -> bytes:
    """Return canonical JSON bytes for the signed portion of a manifest."""
    signed_payload = {key: value for key, value in manifest_data.items() if key not in PLUGIN_SIGNATURE_FIELDS}
    return dumps_json(
        signed_payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def load_plugin_trust_store(path: Path) -> dict[str, str]:
    """Load a key-id to shared-secret trust map from JSON."""
    with open(path, encoding="utf-8") as f:
        raw_store = json.load(f)

    keys = raw_store.get("keys", raw_store) if isinstance(raw_store, dict) else None
    if not isinstance(keys, dict):
        raise PluginSignatureError("Plugin trust store must be a JSON object or contain a 'keys' object")

    trust_store: dict[str, str] = {}
    for key_id, secret in keys.items():
        if not isinstance(key_id, str) or not key_id:
            raise PluginSignatureError("Plugin trust store key ids must be non-empty strings")
        if not isinstance(secret, str) or not secret:
            raise PluginSignatureError(f"Plugin trust store secret for {key_id!r} must be a non-empty string")
        trust_store[key_id] = secret
    return trust_store


def sign_manifest(manifest_data: Mapping[str, Any], *, secret: str) -> str:
    """Return the hex HMAC-SHA256 signature for a manifest payload."""
    return hmac.new(
        secret.encode("utf-8"),
        canonical_manifest_payload(manifest_data),
        hashlib.sha256,
    ).hexdigest()


def verify_manifest_signature(manifest_data: Mapping[str, Any], *, trust_store_path: Path) -> None:
    """Verify a plugin manifest against the configured trust store."""
    algorithm = manifest_data.get("signature_algorithm")
    if algorithm != PLUGIN_SIGNATURE_ALGORITHM:
        raise PluginSignatureError("Plugin manifest signature_algorithm must be 'hmac-sha256'")

    key_id = manifest_data.get("signature_key_id")
    if not isinstance(key_id, str) or not key_id:
        raise PluginSignatureError("Plugin manifest signature_key_id is required")

    signature = manifest_data.get("signature")
    if not isinstance(signature, str) or not signature:
        raise PluginSignatureError("Plugin manifest signature is required")

    trust_store = load_plugin_trust_store(trust_store_path)
    secret = trust_store.get(key_id)
    if secret is None:
        raise PluginSignatureError(f"Plugin manifest signature_key_id {key_id!r} is not trusted")

    expected = sign_manifest(manifest_data, secret=secret)
    if not hmac.compare_digest(signature.strip().lower(), expected):
        raise PluginSignatureError("Plugin manifest signature does not match trusted key")


__all__ = [
    "PLUGIN_SIGNATURE_ALGORITHM",
    "PluginSignatureError",
    "canonical_manifest_payload",
    "load_plugin_trust_store",
    "sign_manifest",
    "verify_manifest_signature",
]
