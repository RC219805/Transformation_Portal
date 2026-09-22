"""Immutable, bounded admission of independently verified V5 photographs.

V5 verification proves retained-output consistency. It does not authenticate the
retained source master against original camera bytes or prove scene accuracy.
Material responses requiring masks absent from the retained generation are not
accepted as a replayable V6 source.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from transformation_portal.core.depth_evidence import DepthEvidence, build_depth_evidence
from transformation_portal.core.execution_plan import decode_bounded_json_object
from transformation_portal.core.execution_plan_v2 import require_digest
from transformation_portal.core.execution_plan_v4 import parse_plan_v4
from transformation_portal.core.image_artifact import ImageMaster, ImageProxy
from transformation_portal.ingest.canonical_json import canonicalize_json
from transformation_portal.lux_depth_v4.evidence import MAX_EVIDENCE_BYTES, _decode_evidence, _inventory
from transformation_portal.lux_depth_v4.io import directory_path, pinned_directory, snapshot
from transformation_portal.lux_depth_v4.photography import create_proxy
from transformation_portal.lux_depth_v5.evidence import MAX_ICC_BYTES, _array, _image, verify_execution_evidence_v3

_MAX_FILES = 32769


def _check(cancellation: Callable[[], bool] | None) -> None:
    if cancellation is not None and cancellation():
        raise RuntimeError("V6 source verification cancelled")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


@dataclass(frozen=True)
class SourceLimits:
    """Total retained bytes and per-image geometry/working-memory admission."""

    max_input_bytes: int = 64 * 1024**3
    max_pixels: int = 100_000_000
    memory_mib: int = 16384

    def __post_init__(self) -> None:
        bounds = {"max_input_bytes": 1024**4, "max_pixels": 200_000_000, "memory_mib": 262144}
        for name, maximum in bounds.items():
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= maximum:
                raise ValueError(f"V6 {name} requires a positive bounded integer")

    def to_payload(self) -> dict[str, int]:
        return {"max_input_bytes": self.max_input_bytes, "max_pixels": self.max_pixels, "memory_mib": self.memory_mib}


@dataclass(frozen=True)
class SourceArtifact:
    path: str
    size_bytes: int
    sha256: str

    def __post_init__(self) -> None:
        if type(self.path) is not str or type(self.size_bytes) is not int or self.size_bytes <= 0:
            raise ValueError("V6 source artifact requires an immutable path and positive byte count")
        require_digest(self.sha256)

    def to_payload(self) -> dict[str, Any]:
        return {"path": self.path, "size_bytes": self.size_bytes, "sha256": self.sha256}


@dataclass(frozen=True)
class SourceImage:
    input_id: str
    shape: tuple[int, int]
    master_sha256: str
    descriptor_bytes: bytes

    def __post_init__(self) -> None:
        if (
            type(self.input_id) is not str
            or type(self.shape) is not tuple
            or len(self.shape) != 2
            or any(type(size) is not int or size <= 0 for size in self.shape)
            or type(self.descriptor_bytes) is not bytes
        ):
            raise ValueError("V6 source image requires immutable typed geometry and descriptor bytes")
        require_digest(self.master_sha256)

    @property
    def descriptor(self) -> dict[str, Any]:
        return decode_bounded_json_object(self.descriptor_bytes)

    def to_payload(self) -> dict[str, Any]:
        return {
            "input_id": self.input_id,
            "shape": list(self.shape),
            "master_sha256": self.master_sha256,
            "descriptor_sha256": _sha256(self.descriptor_bytes),
        }


@dataclass(frozen=True)
class VerifiedV5Source:
    root: Path
    canonical_plan_bytes: bytes
    canonical_evidence_bytes: bytes
    source_digest: str
    images: tuple[SourceImage, ...]
    inventory: tuple[SourceArtifact, ...]
    limits: SourceLimits

    def __post_init__(self) -> None:
        if (
            not isinstance(self.root, Path)
            or type(self.canonical_plan_bytes) is not bytes
            or type(self.canonical_evidence_bytes) is not bytes
            or type(self.images) is not tuple
            or any(type(image) is not SourceImage for image in self.images)
            or type(self.inventory) is not tuple
            or any(type(item) is not SourceArtifact for item in self.inventory)
            or type(self.limits) is not SourceLimits
        ):
            raise ValueError("V6 verified source requires exact immutable data carriers")
        require_digest(self.source_digest)

    @property
    def plan_sha256(self) -> str:
        return _sha256(self.canonical_plan_bytes)

    @property
    def evidence_sha256(self) -> str:
        return _sha256(self.canonical_evidence_bytes)


def _source_digest(plan_bytes: bytes, evidence_bytes: bytes, inventory: tuple[SourceArtifact, ...]) -> str:
    return _sha256(
        canonicalize_json(
            {
                "schema": "tp.lux.grade.source.v1",
                "plan_sha256": _sha256(plan_bytes),
                "evidence_sha256": _sha256(evidence_bytes),
                "artifacts": [item.to_payload() for item in inventory],
            }
        )
    )


def _preflight_inventory(evidence: dict, completion: dict, limits: SourceLimits) -> dict[str, dict]:
    records = evidence.get("artifacts")
    if not isinstance(records, list) or not 1 <= len(records) < _MAX_FILES:
        raise ValueError("V6 source has an invalid or oversized artifact inventory")
    declared = {}
    total = completion["size_bytes"]
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("V6 source artifact must be a complete record")
        relative, size, digest = record.get("path"), record.get("size_bytes"), record.get("sha256")
        if (
            type(relative) is not str
            or not relative
            or len(relative) > 4096
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or Path(relative).as_posix() != relative
            or any(character in relative for character in ("\\", "\x00", ":"))
            or relative in declared
            or relative == "execution-evidence.json"
            or type(size) is not int
            or size <= 0
        ):
            raise ValueError("V6 source artifact inventory is not canonical")
        require_digest(digest)
        total += size
        if total > limits.max_input_bytes:
            raise ValueError("V6 source exceeds its total input-byte budget")
        declared[relative] = record
    return declared


def _read_bound(root: Path, relative: str, record: dict, maximum: int) -> bytes:
    raw, observed = snapshot(root, root / relative, maximum_bytes=maximum)
    if any(observed[key] != record[key] for key in ("path", "sha256", "size_bytes")):
        raise ValueError("V6 source artifact changed after verification")
    return raw


def _inspect_image(root: Path, input_id: str, declared: dict[str, dict], target: int, limits: SourceLimits) -> SourceImage:
    relative = f"{input_id}/photograph.json"
    if relative not in declared or declared[relative].get("kind") != "descriptor":
        raise ValueError("V6 source omits its photograph descriptor")
    raw = _read_bound(root, relative, declared[relative], min(MAX_EVIDENCE_BYTES, limits.max_input_bytes))
    descriptor = decode_bounded_json_object(raw)
    if canonicalize_json(descriptor) != raw:
        raise ValueError("V6 source photograph descriptor must be canonical")
    master = descriptor.get("master")
    shape = master.get("shape") if isinstance(master, dict) else None
    if not isinstance(shape, list) or len(shape) != 2 or any(type(value) is not int or value <= 0 for value in shape):
        raise ValueError("V6 source has invalid master geometry")
    original = descriptor.get("source")
    if not isinstance(original, dict) or original.get("shape") != shape:
        raise ValueError("V6 source and retained master geometry must agree before allocation")
    pixels = math.prod(shape)
    if pixels > limits.max_pixels:
        raise ValueError("V6 source exceeds its per-image pixel budget")
    # Include the independent V5 semantic replay's retained arrays and working
    # copies before invoking its numeric verifier, not after allocating them.
    proxy_edge = ((target + 13) // 14) * 14
    if pixels * 256 + proxy_edge**2 * 128 + MAX_ICC_BYTES > limits.memory_mib * 1024**2:
        raise ValueError("V6 source verification exceeds its working-memory budget")
    materials = descriptor.get("materials")
    if not isinstance(materials, dict) or materials.get("status") != "abstained":
        raise ValueError("V6 cannot replay applied Materials responses because V5 does not retain their masks")
    # A claimed abstention is subsequently checked by the full V5 verifier;
    # explicit zero deltas additionally keep this boundary self-describing.
    if materials.get("changed_pixels", 0) != 0 or materials.get("max_abs_delta", 0) != 0:
        raise ValueError("V6 cannot replay a nonzero Materials response")
    master_record = declared.get(f"{input_id}/master.npy")
    if not isinstance(master_record, dict) or master_record.get("kind") != "array":
        raise ValueError("V6 source omits its retained float master")
    return SourceImage(input_id, tuple(shape), master_record["sha256"], raw)


def prepare_source(
    root: Path, *, limits: SourceLimits = SourceLimits(), cancellation: Callable[[], bool] | None = None
) -> VerifiedV5Source:
    """Bound inventory/geometry first, then verify every V5 semantic product."""
    if type(limits) is not SourceLimits:
        raise TypeError("V6 source requires exact SourceLimits")
    _check(cancellation)
    root = directory_path(root)
    with pinned_directory(root):
        maximum = min(MAX_EVIDENCE_BYTES, limits.max_input_bytes)
        plan_bytes, _ = snapshot(root, root / "execution-plan.json", maximum_bytes=maximum)
        plan = parse_plan_v4(plan_bytes)
        if plan.canonical_bytes != plan_bytes:
            raise ValueError("V6 requires exact canonical V5 plan bytes")
        evidence_bytes, completion = snapshot(root, root / "execution-evidence.json", maximum_bytes=maximum)
        evidence = _decode_evidence(evidence_bytes)
        declared = _preflight_inventory(evidence, completion, limits)
        payload = plan.to_payload()
        images = []
        for item in payload["inputs"]:
            _check(cancellation)
            images.append(_inspect_image(root, item["id"], declared, payload["configuration"]["target_size"], limits))
        _check(cancellation)
        verified = verify_execution_evidence_v3(root, expected_plan_sha256=plan.plan_fingerprint_sha256)
        if verified.canonical_bytes != evidence_bytes:
            raise ValueError("V6 source completion changed during verification")
        inventory = tuple(
            SourceArtifact(item.path, item.size_bytes, item.sha256)
            for item in sorted(verified.artifacts, key=lambda a: a.path)
        )
        by_path = {item.path: item for item in inventory}
        if by_path["execution-plan.json"].sha256 != _sha256(plan_bytes):
            raise ValueError("V6 source plan changed during verification")
        source = VerifiedV5Source(
            root,
            plan_bytes,
            evidence_bytes,
            _source_digest(plan_bytes, evidence_bytes, inventory),
            tuple(images),
            inventory,
            limits,
        )
        validate_source(source, cancellation=cancellation)
        return source


def validate_source(source: VerifiedV5Source, *, cancellation: Callable[[], bool] | None = None) -> None:
    """Rebind the exact admitted source bytes without rediscovery or regrading."""
    if type(source) is not VerifiedV5Source or type(source.limits) is not SourceLimits:
        raise TypeError("V6 requires an exact verified source carrier")
    _check(cancellation)
    if directory_path(source.root) != source.root:
        raise ValueError("V6 source root is no longer canonical")
    SourceLimits(**source.limits.to_payload())
    if (
        type(source.inventory) is not tuple
        or any(type(item) is not SourceArtifact for item in source.inventory)
        or type(source.images) is not tuple
        or any(type(image) is not SourceImage for image in source.images)
    ):
        raise ValueError("V6 source requires immutable typed inventory and images")
    plan = parse_plan_v4(source.canonical_plan_bytes)
    evidence = _decode_evidence(source.canonical_evidence_bytes)
    if plan.canonical_bytes != source.canonical_plan_bytes or canonicalize_json(evidence) != source.canonical_evidence_bytes:
        raise ValueError("V6 source requires canonical frozen plan and evidence bytes")
    completion = {
        "path": "execution-evidence.json",
        "size_bytes": len(source.canonical_evidence_bytes),
        "sha256": source.evidence_sha256,
    }
    declared = _preflight_inventory(evidence, completion, source.limits)
    expected_inventory = tuple(
        SourceArtifact(row["path"], row["size_bytes"], row["sha256"])
        for row in sorted([*declared.values(), completion], key=lambda row: row["path"])
    )
    if source.inventory != expected_inventory:
        raise ValueError("V6 source inventory differs from its frozen completion")
    if declared.get("execution-plan.json", {}).get("sha256") != source.plan_sha256:
        raise ValueError("V6 source plan differs from its frozen completion")
    if _source_digest(source.canonical_plan_bytes, source.canonical_evidence_bytes, source.inventory) != source.source_digest:
        raise ValueError("V6 source digest differs from its immutable inventory")
    with pinned_directory(source.root):
        payload = plan.to_payload()
        images = []
        for item in payload["inputs"]:
            _check(cancellation)
            images.append(
                _inspect_image(source.root, item["id"], declared, payload["configuration"]["target_size"], source.limits)
            )
        if tuple(images) != source.images:
            raise ValueError("V6 source images differ from their frozen descriptors")
        for artifact in source.inventory:
            _check(cancellation)
            _, observed = snapshot(
                source.root, source.root / artifact.path, maximum_bytes=artifact.size_bytes, retain_bytes=False
            )
            if observed != artifact.to_payload():
                raise ValueError("V6 source artifact changed after verification")
        if _inventory(source.root) != {item.path for item in source.inventory}:
            raise ValueError("V6 source inventory changed after verification")
        _check(cancellation)


def _selected(source: VerifiedV5Source, input_id: str) -> tuple[SourceImage, dict, dict, dict]:
    if type(source) is not VerifiedV5Source:
        raise TypeError("V6 requires an exact verified source carrier")
    images = [image for image in source.images if image.input_id == input_id]
    if len(images) != 1:
        raise ValueError("V6 source does not contain the requested image")
    payload = parse_plan_v4(source.canonical_plan_bytes).to_payload()
    inputs = [item for item in payload["inputs"] if item["id"] == input_id]
    if len(inputs) != 1:
        raise ValueError("V6 image differs from its frozen V5 plan")
    declared = _preflight_inventory(
        _decode_evidence(source.canonical_evidence_bytes),
        {"size_bytes": len(source.canonical_evidence_bytes)},
        source.limits,
    )
    image = _inspect_image(source.root, input_id, declared, payload["configuration"]["target_size"], source.limits)
    if image != images[0]:
        raise ValueError("V6 image descriptor changed after verification")
    return image, image.descriptor, inputs[0], declared


def _load_image(
    source: VerifiedV5Source, image: SourceImage, descriptor: dict, input_record: dict, declared: dict, *, original: bool
) -> ImageMaster:
    prefix, shape = image.input_id, image.shape
    image_descriptor = descriptor["source" if original else "master"]
    alpha = None
    if image_descriptor["alpha_mode"] is not None:
        alpha = _array(source.root, f"{prefix}/alpha.npy", shape, np.dtype("float32"), declared)
    icc = None
    if image_descriptor["source_icc_sha256"] is not None:
        icc = _array(source.root, f"{prefix}/source-icc.npy", None, np.dtype("uint8"), declared).tobytes()
    name = "source-master.npy" if original else "master.npy"
    pixels = _array(source.root, f"{prefix}/{name}", (*shape, 3), np.dtype("float32"), declared)
    return _image(image_descriptor, pixels, input_record, alpha, icc)


def load_master(source: VerifiedV5Source, input_id: str) -> ImageMaster:
    """Load the final V5 master using its immutable descriptors and file hashes."""
    with pinned_directory(source.root):
        image, descriptor, input_record, declared = _selected(source, input_id)
        return _load_image(source, image, descriptor, input_record, declared, original=False)


def load_depth_inputs(source: VerifiedV5Source, input_id: str) -> tuple[ImageMaster, DepthEvidence, ImageProxy]:
    """Load bound source pixels and reconstruct retained native depth evidence."""
    with pinned_directory(source.root):
        image, descriptor, input_record, declared = _selected(source, input_id)
        original = _load_image(source, image, descriptor, input_record, declared, original=True)
        configuration = parse_plan_v4(source.canonical_plan_bytes).to_payload()["configuration"]
        proxy = create_proxy(original, configuration["target_size"])
        native = _array(
            source.root,
            f"{input_id}/native-depth.npy",
            proxy.transform.padded_shape,
            np.dtype("float32"),
            declared,
            allow_nonfinite=True,
        )
        sky = _array(source.root, f"{input_id}/native-sky.npy", proxy.transform.padded_shape, np.dtype("bool"), declared)
        depth = build_depth_evidence(
            native,
            sky if descriptor["depth"]["sky_status"] == "model_mask" else None,
            proxy,
            input_record["sha256"],
            companion=input_record.get("companions"),
            precision=configuration["depth"]["precision"],
        )
        if depth.content_hash() != descriptor["depth_content_sha256"] or depth.to_payload() != descriptor["depth"]:
            raise ValueError("V6 native depth differs from the verified V5 evidence")
        return original, depth, proxy
