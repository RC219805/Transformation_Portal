"""
Common helpers for Phase 3.4 bundle root computation and validation.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Mapping

BUNDLE_VERSION = "1"
HASH_ALGORITHM = "sha256"
BUNDLE_ROOT_ALGORITHM = "sha256"
BUNDLE_ROOT_PREIMAGE_VERSION = "1"
EXPECTED_MANIFEST_FILENAME = "evidence_bundle_manifest.json"
EXPECTED_ROOTS_FILENAME = "merkle_roots.json"
EXPECTED_HASH_MANIFEST_FILENAME = "hash_manifest.csv.gz"
EXPECTED_HASH_SUMMARY_FILENAME = "hash_summary.json"
EXPECTED_SIGNATURE_FILENAME = "merkle_roots.sig.json"
TIMESTAMP_FILENAME_BY_TARGET = {
    "roots": "merkle_roots.tsr",
    "signature": "merkle_roots.sig.tsr",
}
HEX64_RE = re.compile(r"^[a-f0-9]{64}$")

REQUIRED_FIELDS = {
    "bundle_version",
    "hash_algorithm",
    "roots_path",
    "roots_sha256",
    "hash_manifest_path",
    "hash_manifest_sha256",
    "hash_summary_path",
    "hash_summary_sha256",
    "signature_path",
    "signature_sha256",
    "timestamp_target",
    "timestamp_path",
    "timestamp_sha256",
    "merkle_leaf_count",
    "phase3_version",
    "phase3_1_version",
    "phase3_2_version",
    "bundle_tool_name",
    "bundle_tool_version",
}

OPTIONAL_ROOT_FIELDS = {
    "bundle_root_algorithm",
    "bundle_root_preimage_version",
    "bundle_root_sha256",
}
OPTIONAL_TOP_LEVEL_FIELDS = OPTIONAL_ROOT_FIELDS | {"notarization"}
KNOWN_FIELDS = REQUIRED_FIELDS | OPTIONAL_TOP_LEVEL_FIELDS

ROOT_PROJECTION_FIELDS = tuple(
    sorted(
        (
            "bundle_version",
            "hash_algorithm",
            "roots_sha256",
            "hash_manifest_sha256",
            "hash_summary_sha256",
            "signature_sha256",
            "timestamp_target",
            "timestamp_sha256",
            "merkle_leaf_count",
            "phase3_version",
            "phase3_1_version",
            "phase3_2_version",
            "bundle_tool_name",
            "bundle_tool_version",
        )
    )
)

_RFC3161_FIELDS = {"timestamp_path", "timestamp_sha256"}
_SIGSTORE_FIELDS = {"bundle_path", "bundle_sha256"}


def sha256_hexdigest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_merkle_leaf_count(roots_path: Path) -> int:
    roots_payload = json.loads(roots_path.read_text(encoding="utf-8"))
    if not isinstance(roots_payload, dict):
        raise ValueError(f"{EXPECTED_ROOTS_FILENAME} must be a JSON object")

    global_block = roots_payload.get("global")
    if not isinstance(global_block, dict):
        raise ValueError(f"{EXPECTED_ROOTS_FILENAME} missing object field: global")

    leaf_count = global_block.get("leaf_count")
    if type(leaf_count) is not int or leaf_count < 0:
        raise ValueError(f"{EXPECTED_ROOTS_FILENAME}.global.leaf_count must be a non-negative integer")
    return leaf_count


def require_string_field(manifest: Mapping[str, object], field: str) -> str:
    value = manifest.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value


def require_hex_digest(manifest: Mapping[str, object], field: str) -> str:
    value = require_string_field(manifest, field)
    if HEX64_RE.fullmatch(value) is None:
        raise ValueError(f"{field} must be 64 lowercase hex characters")
    return value


def _validate_notarization_subobject(
    manifest: Mapping[str, object],
    object_name: str,
    required_fields: set[str],
    digest_field: str,
    strict: bool,
) -> None:
    block = manifest.get(object_name)
    if block is None:
        return
    if not isinstance(block, dict):
        raise ValueError(f"notarization.{object_name} must be an object")
    if strict:
        unexpected = sorted(set(block) - required_fields)
        if unexpected:
            raise ValueError(f"notarization.{object_name} has unexpected field(s): {', '.join(unexpected)}")
    missing = sorted(required_fields - set(block))
    if missing:
        raise ValueError(f"notarization.{object_name} missing required field(s): {', '.join(missing)}")
    for field in sorted(required_fields - {digest_field}):
        value = block.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"notarization.{object_name}.{field} must be a non-empty string")
    digest_value = block.get(digest_field)
    if not isinstance(digest_value, str) or HEX64_RE.fullmatch(digest_value) is None:
        raise ValueError(f"notarization.{object_name}.{digest_field} must be 64 lowercase hex characters")


def validate_manifest_structure(manifest: dict[str, object], *, strict: bool = True) -> None:
    keys = set(manifest)
    missing = sorted(REQUIRED_FIELDS - keys)
    if missing:
        raise ValueError(f"missing required field(s): {', '.join(missing)}")
    if strict:
        unexpected = sorted(keys - KNOWN_FIELDS)
        if unexpected:
            raise ValueError(f"unexpected field(s): {', '.join(unexpected)}")

    if manifest["bundle_version"] != BUNDLE_VERSION:
        raise ValueError(f"bundle_version must be {BUNDLE_VERSION!r}")
    if manifest["hash_algorithm"] != HASH_ALGORITHM:
        raise ValueError(f"hash_algorithm must be {HASH_ALGORITHM!r}")

    if manifest["roots_path"] != EXPECTED_ROOTS_FILENAME:
        raise ValueError(f"roots_path must be {EXPECTED_ROOTS_FILENAME!r}")
    if manifest["hash_manifest_path"] != EXPECTED_HASH_MANIFEST_FILENAME:
        raise ValueError(f"hash_manifest_path must be {EXPECTED_HASH_MANIFEST_FILENAME!r}")
    if manifest["hash_summary_path"] != EXPECTED_HASH_SUMMARY_FILENAME:
        raise ValueError(f"hash_summary_path must be {EXPECTED_HASH_SUMMARY_FILENAME!r}")
    if manifest["signature_path"] != EXPECTED_SIGNATURE_FILENAME:
        raise ValueError(f"signature_path must be {EXPECTED_SIGNATURE_FILENAME!r}")

    timestamp_target = manifest["timestamp_target"]
    if timestamp_target not in TIMESTAMP_FILENAME_BY_TARGET:
        raise ValueError("timestamp_target must be 'roots' or 'signature'")
    expected_timestamp_path = TIMESTAMP_FILENAME_BY_TARGET[str(timestamp_target)]
    if manifest["timestamp_path"] != expected_timestamp_path:
        raise ValueError(f"timestamp_path must be {expected_timestamp_path!r} for timestamp_target={timestamp_target!r}")

    require_hex_digest(manifest, "roots_sha256")
    require_hex_digest(manifest, "hash_manifest_sha256")
    require_hex_digest(manifest, "hash_summary_sha256")
    require_hex_digest(manifest, "signature_sha256")
    require_hex_digest(manifest, "timestamp_sha256")

    merkle_leaf_count = manifest["merkle_leaf_count"]
    if type(merkle_leaf_count) is not int or merkle_leaf_count < 0:
        raise ValueError("merkle_leaf_count must be a non-negative integer")

    require_string_field(manifest, "phase3_version")
    require_string_field(manifest, "phase3_1_version")
    require_string_field(manifest, "phase3_2_version")
    require_string_field(manifest, "bundle_tool_name")
    require_string_field(manifest, "bundle_tool_version")

    root_field_count = len(OPTIONAL_ROOT_FIELDS & keys)
    if root_field_count not in (0, len(OPTIONAL_ROOT_FIELDS)):
        raise ValueError("bundle_root fields must be provided together")
    if root_field_count:
        if manifest["bundle_root_algorithm"] != BUNDLE_ROOT_ALGORITHM:
            raise ValueError(f"bundle_root_algorithm must be {BUNDLE_ROOT_ALGORITHM!r}")
        if manifest["bundle_root_preimage_version"] != BUNDLE_ROOT_PREIMAGE_VERSION:
            raise ValueError(f"bundle_root_preimage_version must be {BUNDLE_ROOT_PREIMAGE_VERSION!r}")
        require_hex_digest(manifest, "bundle_root_sha256")

    notarization_value = manifest.get("notarization")
    if notarization_value is None:
        return
    if not isinstance(notarization_value, dict):
        raise ValueError("notarization must be an object")

    if strict:
        unexpected_notarization = sorted(set(notarization_value) - {"rfc3161", "sigstore"})
        if unexpected_notarization:
            raise ValueError(f"notarization has unexpected field(s): {', '.join(unexpected_notarization)}")

    if "rfc3161" not in notarization_value and "sigstore" not in notarization_value:
        raise ValueError("notarization must include at least one provider: rfc3161 or sigstore")

    _validate_notarization_subobject(
        notarization_value,
        "rfc3161",
        _RFC3161_FIELDS,
        "timestamp_sha256",
        strict,
    )
    _validate_notarization_subobject(
        notarization_value,
        "sigstore",
        _SIGSTORE_FIELDS,
        "bundle_sha256",
        strict,
    )


def build_bundle_root_projection(manifest: Mapping[str, object]) -> dict[str, object]:
    projection: dict[str, object] = {}
    for field in ROOT_PROJECTION_FIELDS:
        if field not in manifest:
            raise ValueError(f"missing required root projection field: {field}")
        projection[field] = manifest[field]
    return projection


def canonical_root_preimage_bytes(manifest: Mapping[str, object]) -> bytes:
    projection = build_bundle_root_projection(manifest)
    payload = json.dumps(
        projection,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return payload + b"\n"


def compute_bundle_root_sha256(manifest: Mapping[str, object]) -> str:
    return hashlib.sha256(canonical_root_preimage_bytes(manifest)).hexdigest()
