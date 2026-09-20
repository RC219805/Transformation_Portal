"""Freeze and rebind opt-in Materials V4 bundles before Lux execution.

This manifest deliberately has a separate version from legacy companions. Its
digests bind both the serialized manifest and the complete numeric evidence.
Loading evidence never grants classifier or calibration trust.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from transformation_portal.core.execution_plan import decode_bounded_json_object
from transformation_portal.core.execution_plan_v2 import require_digest
from transformation_portal.lux_depth_v4.companions import _object, _portable_path
from transformation_portal.lux_depth_v4.io import directory_path, pinned_directory, snapshot
from transformation_portal.materials_v4.artifacts import load_evidence
from transformation_portal.materials_v4.contracts import MaterialEvidence, MaterialLimits

MATERIALS_MANIFEST_SCHEMA = "tp.lux.materials_manifest.v1"
_MAX_MANIFEST_BYTES = 1024 * 1024
_FROZEN_KEYS = {"path", "sha256", "size_bytes", "content_sha256", "source_sha256", "shape"}


def evidence_limits(resources: Mapping[str, Any]) -> MaterialLimits:
    """Use the stricter execution and artifact admission budgets."""
    return MaterialLimits(
        max_pixels=min(resources["max_pixels"], MaterialLimits().max_pixels),
        max_mask_bytes=min(resources["max_input_bytes"], MaterialLimits().max_mask_bytes),
        max_manifest_bytes=min(resources["max_input_bytes"], _MAX_MANIFEST_BYTES),
        max_bundle_bytes=min(resources["max_input_bytes"], MaterialLimits().max_bundle_bytes),
    )


def validate_frozen_record(record: Mapping[str, Any], resources: Mapping[str, Any]) -> None:
    """Validate a portable plan binding without reading any local path."""
    _object(record, _FROZEN_KEYS)
    _portable_path(record["path"])
    for field in ("sha256", "source_sha256", "content_sha256"):
        require_digest(record[field])
    if type(record["size_bytes"]) is not int or not 0 < record["size_bytes"] <= min(
        resources["max_input_bytes"], _MAX_MANIFEST_BYTES
    ):
        raise ValueError("Material evidence manifest exceeds the input byte budget")
    shape = record["shape"]
    if (
        not isinstance(shape, list)
        or len(shape) != 2
        or any(type(size) is not int or size <= 0 for size in shape)
        or math.prod(shape) > evidence_limits(resources).max_pixels
    ):
        raise ValueError("Material evidence geometry exceeds the canonical-master pixel budget")


def load_bound_evidence(root: Path, record: Mapping[str, Any], resources: Mapping[str, Any]) -> MaterialEvidence:
    """Verify the manifest and every numeric constituent against frozen identities."""
    validate_frozen_record(record, resources)
    with pinned_directory(root):
        _, observed = snapshot(root, root / record["path"], maximum_bytes=record["size_bytes"], retain_bytes=False)
        if any(observed[key] != record[key] for key in ("path", "sha256", "size_bytes")):
            raise ValueError("Material evidence manifest changed after preparation")
        evidence = load_evidence(
            root / record["path"],
            expected_source_sha256=record["source_sha256"],
            expected_shape=tuple(record["shape"]),
            limits=evidence_limits(resources),
            expected_content_hash=record["content_sha256"],
        )
        if evidence.content_hash() != record["content_sha256"]:
            raise ValueError("Material evidence content changed after preparation")
        # Close the interval across the separate bundle reader's pinned handle.
        _, after = snapshot(root, root / record["path"], maximum_bytes=record["size_bytes"], retain_bytes=False)
        if after != observed:
            raise ValueError("Material evidence manifest changed during admission")
    return evidence


def freeze_materials(
    manifest_path: Path,
    inputs: Sequence[Mapping[str, Any]],
    resources: Mapping[str, Any],
) -> tuple[dict[str, dict[str, Any]], Path, dict[str, Any]]:
    """Freeze a bounded source-to-evidence map without loading any model."""
    manifest = Path(manifest_path).expanduser().absolute()
    root = directory_path(manifest.parent)
    selected = {item["path"]: item["sha256"] for item in inputs}
    result: dict[str, dict[str, Any]] = {}
    with pinned_directory(root):
        data, receipt = snapshot(
            root, root / manifest.name, maximum_bytes=min(resources["max_input_bytes"], _MAX_MANIFEST_BYTES)
        )
        payload = decode_bounded_json_object(data)
        _object(payload, {"schema", "inputs"})
        if payload["schema"] != MATERIALS_MANIFEST_SCHEMA or not isinstance(payload["inputs"], list):
            raise ValueError("Unsupported Materials V4 manifest schema")
        if not 1 <= len(payload["inputs"]) <= 1024:
            raise ValueError("Materials manifest requires between one and 1024 source bindings")
        for supplied in payload["inputs"]:
            _object(supplied, {"path", "source_sha256", "evidence_path", "evidence_sha256", "shape"})
            path = _portable_path(supplied["path"])
            if path in result or path not in selected or supplied["source_sha256"] != selected[path]:
                raise ValueError("Material source binding is duplicate, absent, or differs from selected bytes")
            evidence_path = _portable_path(supplied["evidence_path"])
            require_digest(supplied["evidence_sha256"])
            evidence_data, evidence_receipt = snapshot(
                root, root / evidence_path, maximum_bytes=min(resources["max_input_bytes"], _MAX_MANIFEST_BYTES)
            )
            if evidence_receipt["sha256"] != supplied["evidence_sha256"]:
                raise ValueError("Material evidence manifest digest differs from the source binding")
            # Validate shape and bounds before the bundle reader allocates masks.
            record = {
                **evidence_receipt,
                "source_sha256": supplied["source_sha256"],
                "shape": supplied["shape"],
                "content_sha256": decode_bounded_json_object(evidence_data).get("content_sha256"),
            }
            validate_frozen_record(record, resources)
            evidence = load_evidence(
                root / evidence_path,
                expected_source_sha256=supplied["source_sha256"],
                expected_shape=tuple(supplied["shape"]),
                limits=evidence_limits(resources),
                expected_content_hash=record["content_sha256"],
            )
            if evidence.content_hash() != record["content_sha256"]:
                raise ValueError("Material evidence differs from the captured manifest content")
            _, after = snapshot(root, root / evidence_path, maximum_bytes=evidence_receipt["size_bytes"], retain_bytes=False)
            if after != evidence_receipt:
                raise ValueError("Material evidence changed during preparation")
            result[path] = record
    return result, root, receipt
