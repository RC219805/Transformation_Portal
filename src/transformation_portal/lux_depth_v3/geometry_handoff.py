"""Closed, byte-bound native geometry inputs for corrected V3 to V2 execution.

File names are supplied separately by the execution owner. Receipts never
authorize a path, and generated array identities are execution evidence rather
than inputs to their own producing plan.
"""

from __future__ import annotations

import hashlib
import io
import math
import os
import re
import stat
import zipfile
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Mapping

import numpy as np

from transformation_portal.core.execution_plan import MAX_DECODED_PIXELS_PER_INPUT, decode_bounded_json_object
from transformation_portal.core.image_artifact import (
    freeze_metadata,
    immutable_array,
    metadata_payload,
    validate_source_sha256,
)
from transformation_portal.ingest.canonical_json import canonicalize_json

if TYPE_CHECKING:
    from .geometry import GeometryTransform

HANDOFF_SCHEMA = "tp.lux.geometry_handoff.v1"
LEGACY_GEOMETRY_RECIPE = "legacy_encoded_crop_stretch_v1"
CORRECTED_GEOMETRY_RECIPE = "normalized_crop_support_v1"
MAX_HANDOFF_BYTES = 64 * 1024
MAX_MASK_BYTES = 100 * 1024**2
_BODY_KEYS = {
    "schema",
    "recipe",
    "source_sha256",
    "v2_input_sha256",
    "plan_fingerprint_sha256",
    "transform",
    "depth",
    "support",
    "masks",
}


def _snapshot(path: Path, maximum: int) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    descriptor = os.open(path, flags)
    with os.fdopen(descriptor, "rb") as handle:
        before = os.fstat(handle.fileno())
        if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= maximum:
            raise ValueError("Geometry artifact is not a bounded regular file")
        data = handle.read(before.st_size + 1)
        after = os.fstat(handle.fileno())
    if (
        len(data) != before.st_size
        or not os.path.samestat(before, after)
        or (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (after.st_size, after.st_mtime_ns, after.st_ctime_ns)
    ):
        raise ValueError("Geometry artifact changed while its bytes were snapshotted")
    return data


def snapshot_geometry_artifact(path: Path, maximum: int) -> bytes:
    """Read one bounded regular artifact through a stable, non-symlink inode."""
    if type(maximum) is not int or maximum <= 0:
        raise ValueError("Geometry artifact byte budget must be a positive integer")
    return _snapshot(Path(path), maximum)


def array_identity(array: np.ndarray) -> dict[str, Any]:
    if not isinstance(array, np.ndarray):
        raise ValueError("Geometry identity requires an explicit NumPy array")
    return {
        "shape": list(array.shape),
        "dtype": array.dtype.str,
        "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
    }


def _npy(data: bytes, shape: tuple[int, int], dtype: np.dtype) -> np.ndarray:
    stream = io.BytesIO(data)
    version = np.lib.format.read_magic(stream)
    if version not in {(1, 0), (2, 0)}:
        raise ValueError("Geometry arrays require bounded NPY version 1 or 2")
    length = 2 if version == (1, 0) else 4
    encoded_length = stream.read(length)
    header_length = int.from_bytes(encoded_length, "little")
    if len(encoded_length) != length or not 0 < header_length <= 4096:
        raise ValueError("Geometry NPY header exceeds its limit")
    header = stream.read(header_length)
    reader = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
    actual_shape, fortran, actual_dtype = reader(io.BytesIO(encoded_length + header), max_header_size=4096)
    if (
        actual_shape != shape
        or fortran
        or actual_dtype != dtype
        or len(data) - stream.tell() != math.prod(shape) * dtype.itemsize
    ):
        raise ValueError("Geometry array differs from its native shape, dtype or byte budget")
    array = np.frombuffer(data, dtype=dtype, offset=stream.tell()).reshape(shape)
    if not np.isfinite(array).all():
        raise ValueError("Geometry arrays must contain finite values")
    return array


def _masks(data: bytes | None, shape: tuple[int, int]) -> dict[str, np.ndarray]:
    if data is None:
        return {}
    result: dict[str, np.ndarray] = {}
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        members = archive.infolist()
        if not 1 <= len(members) <= 32 or sum(member.file_size for member in members) > MAX_MASK_BYTES:
            raise ValueError("Geometry masks exceed their bounded inventory")
        for member in members:
            name = member.filename.removesuffix(".npy")
            if (
                member.filename != f"{name}.npy"
                or re.fullmatch(r"[a-z][a-z0-9_]{0,63}", name) is None
                or name in result
                or member.compress_type not in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}
                or member.file_size > math.prod(shape) * 8 + 4096
            ):
                raise ValueError("Geometry mask inventory is malformed")
            encoded = archive.read(member)
            try:
                mask = _npy(encoded, shape, np.dtype("float32"))
            except ValueError:
                mask = _npy(encoded, shape, np.dtype("float64"))
            if np.any((mask < 0) | (mask > 1)):
                raise ValueError("Geometry masks must be bounded in [0,1]")
            result[name] = mask
    return result


def _transform(payload: Mapping[str, Any]) -> GeometryTransform:
    from .geometry import GeometryTransform

    transform = GeometryTransform.from_payload(payload)
    if math.prod(transform.normalized_shape) > MAX_DECODED_PIXELS_PER_INPUT:
        raise ValueError("Geometry handoff exceeds its native pixel budget")
    return transform


def _identity(body: Mapping[str, Any]) -> str:
    return hashlib.sha256(HANDOFF_SCHEMA.encode("ascii") + b"\0" + canonicalize_json(body)).hexdigest()


@dataclass(frozen=True)
class GeometryBoundInputs:
    """Immutable verified arrays; the stage rechecks its exact image binding."""

    identity_sha256: str
    source_sha256: str
    v2_input_sha256: str
    native_shape: tuple[int, int]
    depth_map: np.ndarray
    support: np.ndarray
    masks: Mapping[str, np.ndarray] = field(default_factory=dict)
    image: np.ndarray | None = None
    alpha: np.ndarray | None = None
    handoff: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.native_shape, tuple)
            or len(self.native_shape) != 2
            or any(type(side) is not int or side <= 0 for side in self.native_shape)
            or math.prod(self.native_shape) > MAX_DECODED_PIXELS_PER_INPUT
        ):
            raise ValueError("Bound geometry requires a bounded positive native shape")
        if not isinstance(self.depth_map, np.ndarray) or not isinstance(self.support, np.ndarray):
            raise ValueError("Bound geometry requires explicit NumPy depth and support arrays")
        for digest in (self.identity_sha256, self.source_sha256, self.v2_input_sha256):
            validate_source_sha256(digest)
        if self.handoff is not None:
            payload = metadata_payload(freeze_metadata(self.handoff))
            if set(payload) != _BODY_KEYS | {"identity_sha256"}:
                raise ValueError("Bound handoff has missing or unsupported fields")
            body = {key: payload[key] for key in _BODY_KEYS}
            if payload["identity_sha256"] != self.identity_sha256 or _identity(body) != self.identity_sha256:
                raise ValueError("Bound handoff differs from its exact execution identity")
            if payload["source_sha256"] != self.source_sha256 or payload["v2_input_sha256"] != self.v2_input_sha256:
                raise ValueError("Bound handoff differs from its source identities")
            object.__setattr__(self, "handoff", freeze_metadata(payload))
        if (
            self.depth_map.dtype != np.float32
            or self.depth_map.shape != self.native_shape
            or not np.isfinite(self.depth_map).all()
        ):
            raise ValueError("Bound depth must be finite float32 on the exact native grid")
        if np.any((self.depth_map < 0) | (self.depth_map > 1)):
            raise ValueError("Bound V2 depth must retain normalized-relative [0,1] semantics")
        if self.support.dtype != np.bool_ or self.support.shape != self.native_shape:
            raise ValueError("Bound support must be boolean on the exact native grid")
        if np.any(self.depth_map[~self.support] != 0):
            raise ValueError("Unsupported depth must retain its explicit zero sentinel")
        object.__setattr__(self, "depth_map", immutable_array(self.depth_map, np.float32))
        object.__setattr__(self, "support", immutable_array(self.support, np.bool_))
        masks = {}
        if not isinstance(self.masks, Mapping) or len(self.masks) > 32:
            raise ValueError("Bound masks require a bounded explicit mapping")
        for name, mask in self.masks.items():
            if (
                not isinstance(name, str)
                or not isinstance(mask, np.ndarray)
                or re.fullmatch(r"[a-z][a-z0-9_]{0,63}", name) is None
                or mask.dtype not in {np.dtype("float32"), np.dtype("float64")}
                or mask.shape != self.native_shape
                or not np.isfinite(mask).all()
                or np.any((mask < 0) | (mask > 1))
            ):
                raise ValueError("Every supplied mask must match the bound native geometry and domain")
            masks[name] = immutable_array(mask, mask.dtype)
        object.__setattr__(self, "masks", MappingProxyType(masks))
        if self.handoff is not None:
            payload = metadata_payload(self.handoff)
            if (
                payload["depth"]["array"] != array_identity(self.depth_map)
                or payload["support"] != array_identity(self.support)
                or payload["masks"]["arrays"] != {name: array_identity(mask) for name, mask in sorted(masks.items())}
                or _transform(payload["transform"]).normalized_shape != self.native_shape
            ):
                raise ValueError("Bound arrays differ from the exact validated handoff")
        if self.image is not None:
            if (
                self.image.dtype != np.float32
                or self.image.shape != (*self.native_shape, 3)
                or not np.isfinite(self.image).all()
                or np.any((self.image < 0) | (self.image > 1))
            ):
                raise ValueError("Bound photographic image must match the native RGB grid")
            object.__setattr__(self, "image", immutable_array(self.image, self.image.dtype))
        if self.alpha is not None:
            if (
                self.alpha.shape != self.native_shape
                or not np.isfinite(self.alpha).all()
                or np.any((self.alpha < 0) | (self.alpha > 1))
            ):
                raise ValueError("Bound alpha must match native geometry and [0,1] domain")
            object.__setattr__(self, "alpha", immutable_array(self.alpha, np.float32))

    def with_image(self, image: np.ndarray, alpha: np.ndarray | None = None) -> GeometryBoundInputs:
        return replace(self, image=image, alpha=alpha)

    def validate_stage(self, image: np.ndarray, depth: np.ndarray | None, masks: object) -> None:
        # Immutable buffers still permit NumPy shape metadata to be reassigned.
        # Recheck the frame before cache lookup as well as before execution.
        if (
            not isinstance(image, np.ndarray)
            or image.shape != (*self.native_shape, 3)
            or depth is None
            or depth.shape != self.native_shape
            or self.support.shape != self.native_shape
            or self.support.dtype != np.bool_
            or (self.alpha is not None and self.alpha.shape != self.native_shape)
            or any(mask.shape != self.native_shape for mask in self.masks.values())
        ):
            raise ValueError("Stage arrays differ from their verified native geometry")
        if self.handoff is not None:
            payload = metadata_payload(self.handoff)
            if (
                payload["depth"]["array"] != array_identity(self.depth_map)
                or payload["support"] != array_identity(self.support)
                or payload["masks"]["arrays"] != {name: array_identity(mask) for name, mask in sorted(self.masks.items())}
            ):
                raise ValueError("Stage arrays differ from the exact validated handoff")
        if self.image is None or array_identity(image) != array_identity(self.image):
            raise ValueError("Stage image differs from its verified geometry binding")
        if depth is None or array_identity(depth) != array_identity(self.depth_map):
            raise ValueError("Stage depth differs from its verified geometry binding")
        if not isinstance(masks, Mapping) or set(masks) != set(self.masks):
            raise ValueError("Stage masks differ from their verified geometry inventory")
        if any(array_identity(masks[name]) != array_identity(mask) for name, mask in self.masks.items()):
            raise ValueError("Stage mask differs from its verified geometry binding")

    def allowed_pixels(self) -> np.ndarray:
        return self.support if self.alpha is None else self.support & (self.alpha == 1)


def build_geometry_handoff(
    *,
    source_sha256: str,
    v2_input_path: Path,
    plan_fingerprint_sha256: str | None,
    transform: Any,
    depth_map_path: Path,
    masks_file: Path | None = None,
    recipe: str = CORRECTED_GEOMETRY_RECIPE,
    depth_normalization: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Freeze exact generated artifacts after their owning operation succeeds."""
    if recipe != CORRECTED_GEOMETRY_RECIPE:
        raise ValueError("A geometry handoff requires the corrected recipe")
    validate_source_sha256(source_sha256)
    if plan_fingerprint_sha256 is not None:
        validate_source_sha256(plan_fingerprint_sha256)
    checked = _transform(transform.to_payload())
    if checked.source_sha256 != source_sha256:
        raise ValueError("Geometry transform belongs to another source")
    shape = checked.normalized_shape
    depth_data = _snapshot(depth_map_path, math.prod(shape) * 4 + 4096)
    depth = _npy(depth_data, shape, np.dtype("float32"))
    mask_data = None if masks_file is None else _snapshot(masks_file, MAX_MASK_BYTES)
    masks = _masks(mask_data, shape)
    support = checked.support()
    body: dict[str, Any] = {
        "schema": HANDOFF_SCHEMA,
        "recipe": recipe,
        "source_sha256": source_sha256,
        "v2_input_sha256": hashlib.sha256(_snapshot(v2_input_path, 2 * 1024**3)).hexdigest(),
        "plan_fingerprint_sha256": plan_fingerprint_sha256,
        "transform": checked.to_payload(),
        "depth": {
            "file_sha256": hashlib.sha256(depth_data).hexdigest(),
            "array": array_identity(depth),
            "semantics": "normalized_relative_v1",
            "normalization": None if depth_normalization is None else metadata_payload(freeze_metadata(depth_normalization)),
        },
        "support": array_identity(support),
        "masks": {
            "file_sha256": None if mask_data is None else hashlib.sha256(mask_data).hexdigest(),
            "arrays": {name: array_identity(mask) for name, mask in sorted(masks.items())},
        },
    }
    GeometryBoundInputs(_identity(body), source_sha256, body["v2_input_sha256"], shape, depth, support, masks)
    return {**body, "identity_sha256": _identity(body)}


def write_geometry_handoff(path: Path, **kwargs: Any) -> dict[str, Any]:
    from .io_atomic import atomic_write_bytes

    payload = build_geometry_handoff(**kwargs)
    encoded = canonicalize_json(payload)
    if len(encoded) > MAX_HANDOFF_BYTES:
        raise ValueError("Geometry handoff exceeds its byte budget")
    atomic_write_bytes(path, encoded)
    return payload


def load_geometry_handoff(
    path: Path,
    *,
    input_bytes: bytes,
    depth_map_path: Path,
    masks_file: Path | None = None,
    recipe: str = CORRECTED_GEOMETRY_RECIPE,
) -> GeometryBoundInputs:
    """Verify explicit paths against receipt bytes before materializing edits."""
    payload = decode_bounded_json_object(_snapshot(path, MAX_HANDOFF_BYTES))
    if set(payload) != _BODY_KEYS | {"identity_sha256"}:
        raise ValueError("Geometry handoff has missing or unsupported fields")
    body = {key: payload[key] for key in _BODY_KEYS}
    if body["schema"] != HANDOFF_SCHEMA or body["recipe"] != recipe or recipe != CORRECTED_GEOMETRY_RECIPE:
        raise ValueError("Geometry handoff recipe differs from the selected operation")
    if payload["identity_sha256"] != _identity(body):
        raise ValueError("Geometry handoff identity differs from its exact content")
    for digest in (body["source_sha256"], body["v2_input_sha256"]):
        validate_source_sha256(digest)
    if body["plan_fingerprint_sha256"] is not None:
        validate_source_sha256(body["plan_fingerprint_sha256"])
    if hashlib.sha256(input_bytes).hexdigest() != body["v2_input_sha256"]:
        raise ValueError("V2 source bytes differ from the bound geometry input")
    transform = _transform(body["transform"])
    if transform.source_sha256 != body["source_sha256"]:
        raise ValueError("Geometry transform belongs to another source")
    shape = transform.normalized_shape
    depth_data = _snapshot(depth_map_path, math.prod(shape) * 4 + 4096)
    depth = _npy(depth_data, shape, np.dtype("float32"))
    depth_record = body["depth"]
    if not isinstance(depth_record, dict) or set(depth_record) != {"file_sha256", "array", "semantics", "normalization"}:
        raise ValueError("Depth record has missing or unsupported fields")
    if depth_record["normalization"] is not None and not isinstance(depth_record["normalization"], dict):
        raise ValueError("Depth normalization evidence must be an explicit object or null")
    if depth_record != {
        "file_sha256": hashlib.sha256(depth_data).hexdigest(),
        "array": array_identity(depth),
        "semantics": "normalized_relative_v1",
        "normalization": depth_record["normalization"],
    }:
        raise ValueError("Depth bytes or native geometry differ from the handoff")
    support = transform.support()
    if body["support"] != array_identity(support):
        raise ValueError("Photographic support differs from the bound transform")
    mask_data = None if masks_file is None else _snapshot(masks_file, MAX_MASK_BYTES)
    masks = _masks(mask_data, shape)
    expected_masks = {
        "file_sha256": None if mask_data is None else hashlib.sha256(mask_data).hexdigest(),
        "arrays": {name: array_identity(mask) for name, mask in sorted(masks.items())},
    }
    if body["masks"] != expected_masks:
        raise ValueError("Mask inventory, bytes or native geometry differ from the handoff")
    return GeometryBoundInputs(
        payload["identity_sha256"],
        body["source_sha256"],
        body["v2_input_sha256"],
        shape,
        depth,
        support,
        masks,
        handoff=payload,
    )
