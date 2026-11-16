"""Offline derivative inspection identities, without approval or access grants.

Callers supply immutable snapshots and retain the identities they inspected.
Managed authorization, current source approvals, human proof decisions and
revocation remain the caller's responsibility; this module implements none of
those states. Existing generic exports and packages keep their source-only
approval scope.

Use ``prepare_derivative_proofs(inventory_snapshot)`` after an export to obtain
the candidate records and their SHA-256 identities. Present the actual retained
products for inspection and store the selected identities separately. Later,
call ``package_proof_delivery`` with those previously retained identities,
the original immutable inventory/package-spec snapshots, the retained export
directory, and a fresh archive path. Its returned proof map is additional caller
evidence; the stdlib packager's ZIP/sidecar formats remain unchanged. Persist
that returned evidence in the caller's own authorized storage if needed.

The entire inventory snapshot participates in every proof identity. Reformatting
or changing any inventory evidence therefore requires new identities. This
conservative first version deliberately has no per-destination approval state,
provider profiles, crop controls, ICC conversion, or automatic acceptance.
"""

from __future__ import annotations

import hashlib
import json
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from transformation_portal.core import verified_delivery
from transformation_portal.ingest.canonical_json import canonicalize_json

from .models import ExportPreset

MAX_SNAPSHOT_BYTES = 8 * 1024 * 1024
PROOF_SCHEMA = "tp.photo.derivative_proof.v1"
PROOF_SCOPE = "inspection_identity_only"


def _digest(value: Any) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("Proof bindings require lowercase SHA-256 digests")
    return value


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key in proof snapshot")
        result[key] = value
    return result


def _invalid_constant(value: str) -> None:
    raise ValueError(f"Non-finite JSON number in proof snapshot: {value}")


def _snapshot(raw: bytes, schema: str) -> dict[str, Any]:
    if not isinstance(raw, bytes) or not 0 < len(raw) <= MAX_SNAPSHOT_BYTES:
        raise ValueError("Proof snapshots require immutable JSON bytes of at most 8 MiB")
    try:
        value = json.loads(raw, object_pairs_hook=_unique_object, parse_constant=_invalid_constant)
        # Reject unsupported Unicode and numeric values before constructing any identity.
        canonicalize_json(value)
    except (UnicodeError, ValueError, TypeError, RecursionError) as error:
        raise ValueError(f"Invalid proof snapshot: {error}") from error
    if not isinstance(value, dict) or value.get("schema") != schema:
        raise ValueError(f"Proof snapshot requires schema {schema}")
    return value


def _path(value: Any) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 1024
        or re.fullmatch(r"[A-Za-z0-9_./-]+", value) is None
        or any(part in {"", ".", ".."} for part in value.split("/"))
        or PurePosixPath(value).is_absolute()
    ):
        raise ValueError("Proof files require safe relative export paths")
    return value


def _file(row: Any) -> dict[str, Any]:
    if not isinstance(row, dict):
        raise ValueError("Proof file records must be objects")
    _path(row.get("path"))
    _digest(row.get("sha256"))
    size = row.get("size_bytes")
    if type(size) is not int or not 0 < size <= 2 * 1024**3:
        raise ValueError("Proof file size must be a positive integer of at most 2 GiB")
    return row


@dataclass(frozen=True)
class DerivativeProof:
    """A deeply immutable inspection identity; a hash is not a human decision."""

    canonical_payload: bytes

    def __post_init__(self) -> None:
        payload = _snapshot(self.canonical_payload, PROOF_SCHEMA)
        if (
            set(payload) != {"schema", "scope", "inventory_sha256", "derivative_record"}
            or payload["scope"] != PROOF_SCOPE
            or canonicalize_json(payload) != self.canonical_payload
        ):
            raise ValueError("Derivative proof requires canonical inspection-identity bytes")
        _digest(payload["inventory_sha256"])
        _file(payload["derivative_record"])

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_payload).hexdigest()

    @property
    def path(self) -> str:
        return self.to_payload()["derivative_record"]["path"]

    def to_payload(self) -> dict[str, Any]:
        """Return a detached copy; edits cannot change this proof's identity."""
        return json.loads(self.canonical_payload)


def prepare_derivative_proofs(inventory_snapshot: bytes) -> tuple[DerivativeProof, ...]:
    """Bind every exported product to the complete inventory and applied record.

    This prepares inspection candidates from recorded export evidence. It does
    not open or authenticate files, validate photographic quality, or assert
    that anyone has inspected or approved them. Packaging revalidates bytes.
    Any inventory edit intentionally requires fresh inspection identities.
    """
    inventory = _snapshot(inventory_snapshot, "tp.photo.delivery_inventory.v1")
    sources, records = inventory.get("sources"), inventory.get("files")
    if not isinstance(sources, list) or not 1 <= len(sources) <= 100:
        raise ValueError("Proof inventory requires one to one hundred sources")
    if not isinstance(records, list) or not 1 <= len(records) <= 700:
        raise ValueError("Proof inventory requires bounded exported products")
    indexed: dict[str, dict[str, Any]] = {}
    for source in sources:
        if not isinstance(source, dict):
            raise ValueError("Proof sources must be objects")
        identity = _digest(source.get("source_identity_sha256"))
        master = _digest(source.get("source_master_sha256"))
        bundle = source.get("source_bundle")
        if not isinstance(bundle, dict) or set(bundle) != {
            "schema",
            "master_sha256",
            "descriptor_sha256",
            "alpha_sha256",
            "source_icc_sha256",
        }:
            raise ValueError("Proof source requires its complete source bundle")
        if bundle["schema"] != "tp.photo.source_identity.v1" or bundle["master_sha256"] != master:
            raise ValueError("Proof source bundle differs from its master")
        for name in ("descriptor_sha256", "alpha_sha256", "source_icc_sha256"):
            if bundle[name] is not None:
                _digest(bundle[name])
        if bundle["descriptor_sha256"] is None and any(
            bundle[name] is not None for name in ("alpha_sha256", "source_icc_sha256")
        ):
            raise ValueError("Proof source companions require a native descriptor")
        expected_identity = hashlib.sha256(canonicalize_json(bundle)).hexdigest() if bundle["descriptor_sha256"] else master
        if identity != expected_identity or identity in indexed:
            raise ValueError("Proof source identity is inconsistent or duplicated")
        _digest(source.get("master_content_sha256"))
        if "presentation_policy_sha256" not in source:
            raise ValueError("Proof source requires an explicit presentation binding")
        if source.get("presentation_policy_sha256") is not None:
            _digest(source["presentation_policy_sha256"])
        indexed[identity] = source
    proofs: dict[str, DerivativeProof] = {}
    for record in records:
        _file(record)
        if (
            not {
                "source_master_sha256",
                "source_identity_sha256",
                "presentation_policy_sha256",
                "source_bundle",
                "source_master_content_sha256",
                "source_bit_depth",
                "precision_recovery",
                "container_promotion",
                "preset",
                "recipe",
                "recipe_sha256",
                "width",
                "height",
                "bit_depth",
                "color_profile",
                "icc_sha256",
                "color_domain",
                "intended_use",
                "render",
                "approval_transferred",
            }
            <= record.keys()
        ):
            raise ValueError("Proof product requires a complete applied export record")
        source = indexed.get(_digest(record.get("source_identity_sha256")))
        if source is None or any(
            record.get(key) != source.get(key)
            for key in ("source_master_sha256", "source_bundle", "presentation_policy_sha256")
        ):
            raise ValueError("Proof product differs from its source binding")
        if record.get("source_master_content_sha256") != source["master_content_sha256"]:
            raise ValueError("Proof product differs from its source content binding")
        recipe = record.get("recipe")
        recipe_keys = {
            "schema",
            "name",
            "format",
            "intended_use",
            "long_edge",
            "jpeg_quality",
            "sharpen_amount",
            "render",
            "shoulder",
        }
        if (
            not isinstance(recipe, dict)
            or not recipe_keys <= recipe.keys()
            or recipe.keys() - recipe_keys - {"companion"}
            or recipe.get("schema") != "tp.photo.export_preset.v1"
            or recipe.get("name") != record.get("preset")
            or hashlib.sha256(canonicalize_json(recipe)).hexdigest() != record.get("recipe_sha256")
            or record.get("approval_transferred") is not False
            or "render" not in record
        ):
            raise ValueError("Proof product requires its applied recipe and source-only approval scope")
        ExportPreset(**{key: recipe[key] for key in recipe_keys - {"schema"}})
        if "companion" in recipe and (
            recipe["name"] != "photographic_master" or recipe["companion"] != "analytic_extended_srgb_transfer_no_clipping"
        ):
            raise ValueError("Unsupported proof export companion recipe")
        path = record["path"]
        if path in proofs:
            raise ValueError("Duplicate proof product path")
        proofs[path] = DerivativeProof(
            canonicalize_json(
                {
                    "schema": PROOF_SCHEMA,
                    "scope": PROOF_SCOPE,
                    "inventory_sha256": hashlib.sha256(inventory_snapshot).hexdigest(),
                    "derivative_record": record,
                }
            )
        )
    return tuple(proofs[path] for path in sorted(proofs))


def package_proof_delivery(
    inventory_snapshot: bytes,
    package_spec_snapshot: bytes,
    source_root: Path,
    archive: Path,
    *,
    expected_proofs: Mapping[str, str],
    copy_to: Path | None = None,
) -> dict[str, Any]:
    """Package retained bytes matching caller-pinned inspection identities.

    Expected hashes must come from the caller's earlier inspection context,
    never from recomputing and automatically accepting the current inventory.
    No export/encoder runs. This is an offline integrity gate, not a managed
    approval/revocation service or destination-compliance claim.
    """
    actual = {proof.path: proof.sha256 for proof in prepare_derivative_proofs(inventory_snapshot)}
    if not isinstance(expected_proofs, Mapping) or dict(expected_proofs) != actual:
        raise ValueError("Expected derivative proof identities must match every exported product exactly")
    inventory = _snapshot(inventory_snapshot, "tp.photo.delivery_inventory.v1")
    spec = _snapshot(package_spec_snapshot, "tp.delivery.package_spec.v1")
    components = inventory.get("native_components")
    if not isinstance(components, list) or len(components) > 300:
        raise ValueError("Proof inventory requires bounded native components")
    selected = [*inventory["files"], *components, inventory.get("contact_sheet")]
    required: dict[str, tuple[str, int]] = {}
    for row in selected:
        _file(row)
        if row["path"] in required or row["path"] in {"inventory.json", "inventory.txt"}:
            raise ValueError("Duplicate or reserved proof package path")
        required[row["path"]] = row["sha256"], row["size_bytes"]
    required["inventory.json"] = hashlib.sha256(inventory_snapshot).hexdigest(), len(inventory_snapshot)
    rows = spec.get("files")
    if not isinstance(rows, list) or len(rows) != len(required) + 1:
        raise ValueError("Proof package must retain exactly the inspected inventory members")
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"source", "archive_path", "sha256", "size_bytes"}:
            raise ValueError("Invalid proof package member")
        path = _path(row["source"])
        if path in seen or row["archive_path"] != path:
            raise ValueError("Proof package members require unique unchanged export paths")
        seen.add(path)
        _file({"path": path, "sha256": row["sha256"], "size_bytes": row["size_bytes"]})
        if path != "inventory.txt" and required.get(path) != (row["sha256"], row["size_bytes"]):
            raise ValueError("Proof package member differs from the inspected inventory")
    if seen != set(required) | {"inventory.txt"}:
        raise ValueError("Proof package is missing inspected inventory members")
    # Freeze the caller's exact selection: mutable package-spec.json is never
    # reopened. The existing packager pins the source root, streams/verifies
    # each retained member and verifies the ZIP before no-replace publication.
    with tempfile.TemporaryDirectory(prefix="tp-photo-proof-") as private:
        manifest = Path(private) / "package-spec.json"
        manifest.write_bytes(package_spec_snapshot)
        try:
            receipt = verified_delivery.build_package(manifest, source_root, archive, copy_to)
        except verified_delivery.PackageError as error:
            raise ValueError(f"Verified proof packaging failed: {error}") from error
    return {**receipt, "derivative_proof_sha256": actual, "proof_scope": PROOF_SCOPE}
