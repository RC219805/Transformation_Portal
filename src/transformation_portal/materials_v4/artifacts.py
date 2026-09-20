"""Bounded, non-executable Materials V4 evidence bundles.

The manifest is published last, after content-addressed numeric files. Admission
snapshots nonlinked regular files under a pinned directory, validates NPY headers
before array construction, and verifies both file and semantic content digests.
"""

from __future__ import annotations

import hashlib
import io
import math
import struct
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

import numpy as np

from transformation_portal.core.execution_plan import decode_bounded_json_object
from transformation_portal.ingest.canonical_json import canonicalize_json
from transformation_portal.lux_depth_v3.execution_evidence import ArtifactEvidenceError
from transformation_portal.lux_depth_v4.io import directory_path, pinned_directory, snapshot
from transformation_portal.lux_depth_v4.io import write_evidence as _write_bytes

from .contracts import (
    CALIBRATION_SCHEMA,
    EVIDENCE_SCHEMA,
    CalibrationReceipt,
    MaterialEvidence,
    MaterialLimits,
    MaterialsError,
    RegionEvidence,
    _shape,
    validate_digest,
)
from .taxonomy import TAXONOMY_VERSION

_REGION_KEYS = {
    "region_id",
    "label",
    "mask",
    "semantic_confidence",
    "geometric_quality",
    "provenance",
    "score_type",
    "calibration_sha256",
}
_MASK_KEYS = {"shape", "dtype", "sha256", "path", "file_sha256", "size_bytes"}
_MANIFEST_KEYS = {
    "schema",
    "taxonomy",
    "coordinate_space",
    "source_sha256",
    "shape",
    "regions",
    "producer",
    "calibration",
    "status",
    "reason",
    "content_sha256",
}


def _object(value: Any, keys: set[str]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise MaterialsError("Evidence object contains missing or unsupported fields")
    return value


def _relative_path(value: Any) -> str:
    if not isinstance(value, str) or not value or len(value) > 256 or not value.isascii():
        raise MaterialsError("Mask path must be a bounded portable relative path")
    path = PurePosixPath(value)
    if (
        value != path.as_posix()
        or path.is_absolute()
        or value == "."
        or ".." in path.parts
        or "\\" in value
        or ":" in value
        or any(ord(char) < 32 or ord(char) == 127 for char in value)
        or any(part.endswith((".", " ")) for part in path.parts)
    ):
        raise MaterialsError("Mask path must be a canonical relative path without traversal")
    return value


def _calibration(value: Any) -> CalibrationReceipt | None:
    if value is None:
        return None
    record = _object(value, {"schema", *CalibrationReceipt.__dataclass_fields__})
    if record["schema"] != CALIBRATION_SCHEMA:
        raise MaterialsError("Unsupported calibration receipt schema")
    return CalibrationReceipt(**{key: item for key, item in record.items() if key != "schema"})


def _mask_array(data: bytes, expected_shape: tuple[int, int], limits: MaterialLimits) -> np.ndarray:
    """Parse a bounded numeric NPY header before making any array view."""
    if len(data) < 10 or data[:6] != b"\x93NUMPY":
        raise MaterialsError("Material masks require NPY numeric arrays")
    version = tuple(data[6:8])
    if version == (1, 0):
        prefix, header_size = 10, struct.unpack("<H", data[8:10])[0]
    elif version == (2, 0) and len(data) >= 12:
        prefix, header_size = 12, struct.unpack("<I", data[8:12])[0]
    else:
        raise MaterialsError("Material masks require NPY format 1 or 2")
    if header_size > limits.max_header_bytes or header_size <= 0 or prefix + header_size > len(data):
        raise MaterialsError("Material mask NPY header exceeds budget or is truncated")
    stream = io.BytesIO(data)
    np.lib.format.read_magic(stream)
    reader = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
    shape, fortran, dtype = reader(stream, max_header_size=limits.max_header_bytes)
    if _shape(shape) != expected_shape or fortran or dtype != np.dtype("<f4"):
        raise MaterialsError("Material mask header must describe contiguous float32 canonical geometry")
    expected_bytes = math.prod(expected_shape) * 4
    if len(data) - stream.tell() != expected_bytes:
        raise MaterialsError("Material mask payload length differs from declared geometry")
    array = np.frombuffer(data, dtype=np.dtype("<f4"), offset=stream.tell()).reshape(expected_shape)
    if not np.isfinite(array).all() or np.any((array < 0) | (array > 1)):
        raise MaterialsError("Material mask samples must be finite in [0,1]")
    return array


def load_evidence(
    path: Path | str,
    expected_source_sha256: str,
    expected_shape: tuple[int, int],
    limits: MaterialLimits = MaterialLimits(),
    *,
    expected_content_hash: str | None = None,
) -> MaterialEvidence:
    """Admit a source-bound bundle; this function never grants calibration trust."""
    validate_digest(expected_source_sha256, "expected_source_sha256")
    if expected_content_hash is not None:
        validate_digest(expected_content_hash, "expected_content_hash")
    shape = _shape(expected_shape)
    if not isinstance(limits, MaterialLimits) or math.prod(shape) > limits.max_pixels:
        raise MaterialsError("Expected geometry exceeds the admission budget")
    manifest = Path(path).expanduser().absolute()
    try:
        root = directory_path(manifest.parent)
        with pinned_directory(root):
            data, _ = snapshot(root, root / manifest.name, maximum_bytes=limits.max_manifest_bytes)
            record = _object(decode_bounded_json_object(data), _MANIFEST_KEYS)
            if (
                record["schema"] != EVIDENCE_SCHEMA
                or record["taxonomy"] != TAXONOMY_VERSION
                or record["coordinate_space"] != "canonical_master"
            ):
                raise MaterialsError("Unsupported evidence schema, taxonomy, or coordinates")
            if record["source_sha256"] != expected_source_sha256 or _shape(record["shape"]) != shape:
                raise MaterialsError("Evidence source or geometry differs from expected photograph")
            validate_digest(record["content_sha256"], "content_sha256")
            if expected_content_hash is not None and record["content_sha256"] != expected_content_hash:
                raise MaterialsError("Evidence content differs from frozen execution identity")
            regions = record["regions"]
            if not isinstance(regions, list) or len(regions) > limits.max_regions:
                raise MaterialsError("Evidence region count exceeds budget")
            raw_bytes = math.prod(shape) * 4
            if raw_bytes * len(regions) > limits.max_mask_bytes:
                raise MaterialsError("Evidence aggregate mask bytes exceed budget")
            # Preflight every descriptor before reading or constructing mask arrays.
            descriptors = []
            ids: set[str] = set()
            bundle_bytes = len(data)
            for region in regions:
                _object(region, _REGION_KEYS)
                region_id = region["region_id"]
                if not isinstance(region_id, str) or region_id in ids:
                    raise MaterialsError("Evidence region IDs must be unique strings")
                ids.add(region_id)
                mask = _object(region["mask"], _MASK_KEYS)
                relative = _relative_path(mask["path"])
                validate_digest(mask["file_sha256"], "mask file_sha256")
                validate_digest(mask["sha256"], "mask sha256")
                size = mask["size_bytes"]
                if (
                    type(size) is not int
                    or not raw_bytes < size <= raw_bytes + limits.max_header_bytes + 12
                    or _shape(mask["shape"]) != shape
                    or mask["dtype"] != "float32"
                ):
                    raise MaterialsError("Mask descriptor has invalid geometry, dtype, or byte budget")
                descriptors.append((region, mask, relative))
                bundle_bytes += size
            if bundle_bytes > limits.max_bundle_bytes:
                raise MaterialsError("Evidence encoded bundle exceeds aggregate byte budget")
            calibration = _calibration(record["calibration"])
            admitted = []
            for region, descriptor, relative in descriptors:
                mask_data, receipt = snapshot(root, root / relative, maximum_bytes=descriptor["size_bytes"])
                if receipt["sha256"] != descriptor["file_sha256"] or receipt["size_bytes"] != descriptor["size_bytes"]:
                    raise MaterialsError("Mask file changed or differs from declared digest")
                array = _mask_array(mask_data, shape, limits)
                if hashlib.sha256(array.tobytes()).hexdigest() != descriptor["sha256"]:
                    raise MaterialsError("Mask numeric content differs from declared digest")
                admitted.append(RegionEvidence(**{key: value for key, value in region.items() if key != "mask"}, mask=array))
            evidence = MaterialEvidence(
                source_sha256=expected_source_sha256,
                shape=shape,
                regions=tuple(admitted),
                producer=record["producer"],
                calibration=calibration,
                status=record["status"],
                reason=record["reason"],
            )
            evidence.validate_limits(limits)
            if evidence.content_hash() != record["content_sha256"]:
                raise MaterialsError("Evidence semantic content differs from declared digest")
            return evidence
    except MaterialsError:
        raise
    except (ArtifactEvidenceError, OSError, ValueError, TypeError, OverflowError, KeyError) as exc:
        raise MaterialsError(f"Unable to admit material evidence: {exc}") from exc


def write_evidence(evidence: MaterialEvidence, path: Path | str) -> None:
    """Publish numeric content first and its canonical manifest last.

    The destination directory must already exist. Content hashes exclude bundle
    locations, so bundles remain portable without changing semantic identity.
    """
    if not isinstance(evidence, MaterialEvidence):
        raise MaterialsError("write_evidence requires MaterialEvidence")
    limits = MaterialLimits()
    evidence.validate_limits(limits)
    manifest = Path(path).expanduser().absolute()
    try:
        root = directory_path(manifest.parent)
        record = evidence.to_payload()
        record["content_sha256"] = evidence.content_hash()
        # Calculate bounded manifest descriptors without retaining every NPY payload.
        with pinned_directory(root):
            for region, entry in zip(evidence.regions, record["regions"]):
                stream = io.BytesIO()
                np.save(stream, region.mask, allow_pickle=False)
                data = stream.getvalue()
                digest = hashlib.sha256(data).hexdigest()
                relative = f"mask-{digest}.npy"
                if relative == manifest.name:
                    raise MaterialsError("Manifest destination collides with numeric artifact")
                entry["mask"].update({"path": relative, "file_sha256": digest, "size_bytes": len(data)})
                _write_bytes(root, relative, data)
            data = canonicalize_json(record)
            if len(data) > limits.max_manifest_bytes:
                raise MaterialsError("Evidence manifest exceeds byte budget")
            if len(data) + sum(entry["mask"]["size_bytes"] for entry in record["regions"]) > limits.max_bundle_bytes:
                raise MaterialsError("Evidence encoded bundle exceeds aggregate byte budget")
            _write_bytes(root, manifest.name, data)
    except MaterialsError:
        raise
    except (ArtifactEvidenceError, OSError, ValueError, TypeError, OverflowError) as exc:
        raise MaterialsError(f"Unable to publish material evidence: {exc}") from exc
