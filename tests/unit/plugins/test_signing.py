"""Security-boundary tests for external plugin manifest signing."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from transformation_portal.plugins.signing import (
    PLUGIN_SIGNATURE_ALGORITHM,
    PluginSignatureError,
    canonical_manifest_payload,
    load_plugin_trust_store,
    sign_manifest,
    verify_manifest_signature,
)

pytestmark = [pytest.mark.unit, pytest.mark.security]


def _write_json(path: Path, payload: object) -> Path:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def test_canonical_manifest_payload_excludes_signature_metadata_and_sorts_keys() -> None:
    unsigned = {
        "version": "1.0.0",
        "entry_point": "demo:Plugin",
        "name": "demo",
        "config": {"z": 2, "a": 1},
    }
    signed = {
        "signature": "ignored",
        "signature_key_id": "ignored-key",
        "signature_algorithm": PLUGIN_SIGNATURE_ALGORITHM,
        "name": "demo",
        "config": {"a": 1, "z": 2},
        "entry_point": "demo:Plugin",
        "version": "1.0.0",
    }

    assert canonical_manifest_payload(signed) == canonical_manifest_payload(unsigned)
    assert json.loads(canonical_manifest_payload(signed).decode("utf-8")) == unsigned


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf")])
def test_canonical_manifest_payload_rejects_non_finite_numbers(bad_value: float) -> None:
    manifest = {"name": "demo", "version": "1.0.0", "quality_score": bad_value}

    with pytest.raises(ValueError, match="Out of range"):
        canonical_manifest_payload(manifest)


def test_sign_manifest_ignores_existing_signature_metadata() -> None:
    unsigned = {"name": "demo", "version": "1.0.0", "entry_point": "demo:Plugin"}
    signed_shape = {
        **unsigned,
        "signature_algorithm": PLUGIN_SIGNATURE_ALGORITHM,
        "signature_key_id": "local-dev",
        "signature": "stale-signature-value",
    }

    assert sign_manifest(signed_shape, secret="test-secret") == sign_manifest(unsigned, secret="test-secret")


def test_verify_manifest_signature_accepts_nested_trust_store_and_normalized_signature(tmp_path: Path) -> None:
    trust_store_path = _write_json(tmp_path / "trust.json", {"keys": {"local-dev": "test-secret"}})
    manifest = {
        "name": "demo",
        "version": "1.0.0",
        "entry_point": "demo:Plugin",
        "signature_algorithm": PLUGIN_SIGNATURE_ALGORITHM,
        "signature_key_id": "local-dev",
    }
    manifest["signature"] = f"  {sign_manifest(manifest, secret='test-secret').upper()}  "

    verify_manifest_signature(manifest, trust_store_path=trust_store_path)


def test_load_plugin_trust_store_accepts_flat_key_map(tmp_path: Path) -> None:
    trust_store_path = _write_json(tmp_path / "flat-trust.json", {"local-dev": "test-secret"})

    assert load_plugin_trust_store(trust_store_path) == {"local-dev": "test-secret"}


def test_load_plugin_trust_store_uses_nested_keys_without_trusting_metadata(tmp_path: Path) -> None:
    trust_store_path = _write_json(
        tmp_path / "nested-trust.json",
        {
            "schema_version": "1.0",
            "owner": "security",
            "keys": {"local-dev": "test-secret"},
        },
    )

    assert load_plugin_trust_store(trust_store_path) == {"local-dev": "test-secret"}


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ([], "must be a JSON object"),
        ({"keys": []}, "must be a JSON object"),
        ({"keys": {"": "secret"}}, "key ids must be non-empty strings"),
        ({"keys": {"   ": "secret"}}, "key ids must be non-empty strings"),
        ({"keys": {" local-dev ": "secret"}}, "key ids must not contain leading or trailing whitespace"),
        ({"keys": {"local-dev": 123}}, "secret for 'local-dev' must be a non-empty string"),
        ({"keys": {"local-dev": ""}}, "secret for 'local-dev' must be a non-empty string"),
        ({"keys": {"local-dev": "   "}}, "secret for 'local-dev' must be a non-empty string"),
    ],
)
def test_load_plugin_trust_store_rejects_malformed_trust_sets(
    tmp_path: Path,
    payload: object,
    message: str,
) -> None:
    trust_store_path = _write_json(tmp_path / "bad-trust.json", payload)

    with pytest.raises(PluginSignatureError, match=message):
        load_plugin_trust_store(trust_store_path)


def test_verify_manifest_signature_rejects_unknown_key_id(tmp_path: Path) -> None:
    trust_store_path = _write_json(tmp_path / "trust.json", {"keys": {"trusted": "test-secret"}})
    manifest = {
        "name": "demo",
        "version": "1.0.0",
        "entry_point": "demo:Plugin",
        "signature_algorithm": PLUGIN_SIGNATURE_ALGORITHM,
        "signature_key_id": "untrusted",
    }
    manifest["signature"] = sign_manifest(manifest, secret="test-secret")

    with pytest.raises(PluginSignatureError, match="signature_key_id 'untrusted' is not trusted"):
        verify_manifest_signature(manifest, trust_store_path=trust_store_path)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"signature_key_id": 123}, "signature_key_id is required"),
        ({"signature_key_id": ""}, "signature_key_id is required"),
        ({"signature_key_id": "   "}, "signature_key_id is required"),
        (
            {"signature_key_id": " local-dev "},
            "signature_key_id must not contain leading or trailing whitespace",
        ),
        ({"signature": 123}, "signature is required"),
        ({"signature": ""}, "signature is required"),
        ({"signature": "   "}, "signature is required"),
        ({"signature": "0" * 64}, "signature does not match trusted key"),
    ],
)
def test_verify_manifest_signature_rejects_missing_or_mismatched_signature_fields(
    tmp_path: Path,
    overrides: dict[str, object],
    message: str,
) -> None:
    trust_store_path = _write_json(tmp_path / "trust.json", {"keys": {"local-dev": "test-secret"}})
    manifest = {
        "name": "demo",
        "version": "1.0.0",
        "entry_point": "demo:Plugin",
        "signature_algorithm": PLUGIN_SIGNATURE_ALGORITHM,
        "signature_key_id": "local-dev",
    }
    manifest["signature"] = sign_manifest(manifest, secret="test-secret")
    manifest.update(overrides)

    with pytest.raises(PluginSignatureError, match=message):
        verify_manifest_signature(manifest, trust_store_path=trust_store_path)


@pytest.mark.parametrize(
    "overrides",
    [
        {},
        {"signature_algorithm": None},
        {"signature_algorithm": 123},
        {"signature_algorithm": ""},
        {"signature_algorithm": "   "},
        {"signature_algorithm": "sha256"},
    ],
)
def test_verify_manifest_signature_rejects_signature_algorithm_drift(
    tmp_path: Path,
    overrides: dict[str, object],
) -> None:
    trust_store_path = _write_json(tmp_path / "trust.json", {"keys": {"local-dev": "test-secret"}})
    manifest = {
        "name": "demo",
        "version": "1.0.0",
        "entry_point": "demo:Plugin",
        "signature_key_id": "local-dev",
        "signature": "not-a-valid-hmac",
    }
    manifest.update(overrides)

    with pytest.raises(PluginSignatureError, match="signature_algorithm must be 'hmac-sha256'"):
        verify_manifest_signature(manifest, trust_store_path=trust_store_path)
