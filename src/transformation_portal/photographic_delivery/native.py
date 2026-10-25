"""Hash-bound retained V5/V6 working arrays, independent of delivery TIFFs.

The caller authorizes each component through its current artifact inventory.
This module verifies the selected bytes and the photographic array contract;
it does not replay a generation or authenticate the original camera profile.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from transformation_portal.core.image_artifact import (
    ImageMaster,
    artifact_content_hash,
    freeze_metadata,
    validate_source_sha256,
)
from transformation_portal.ingest.canonical_json import canonicalize_json

MAX_SOURCE_BYTES = 2 * 1024**3
MAX_IMAGE_PIXELS = 100_000_000
MAX_DESCRIPTOR_BYTES = 4 * 1024**2
MAX_ICC_BYTES = 16 * 1024**2


def read_snapshot(data: bytes | None, path: Path | None, digest: str, maximum: int = MAX_SOURCE_BYTES) -> bytes:
    """Read immutable bytes or a regular file through pinned, no-follow FDs."""
    validate_source_sha256(digest)
    if (data is None) == (path is None):
        raise ValueError("Provide exactly one component bytes snapshot or caller-authorized path")
    if data is None:
        if path is None:
            raise ValueError("A source snapshot requires a caller-authorized path")
        path = Path(path)
        if not path.is_absolute() or ".." in path.parts:
            raise ValueError("Source path must be absolute and contain no symlinks")
        directory_fd = os.open(path.anchor, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            for component in path.parts[1:-1]:
                next_fd = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory_fd)
                os.close(directory_fd)
                directory_fd = next_fd
            source_fd = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory_fd)
        except OSError as error:
            raise ValueError("Source must be a regular file under directories without symlinks") from error
        finally:
            os.close(directory_fd)
        with os.fdopen(source_fd, "rb") as stream:
            info = os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode) or not 0 < info.st_size <= maximum:
                raise ValueError("Source must be a bounded regular file")
            data = stream.read(maximum + 1)
    if not isinstance(data, bytes) or not 0 < len(data) <= maximum or hashlib.sha256(data).hexdigest() != digest:
        raise ValueError("Source component SHA-256 changed; review and select the current artifact")
    return data


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("Native descriptor contains duplicate JSON keys")
        value[key] = item
    return value


def _array(raw: bytes, shape: tuple[int, ...]) -> np.ndarray:
    stream = io.BytesIO(raw)
    try:
        version = np.lib.format.read_magic(stream)
        if version not in {(1, 0), (2, 0)}:
            raise ValueError("Native master requires NPY v1/v2")
        length_bytes = 2 if version == (1, 0) else 4
        encoded = stream.read(length_bytes)
        header_size = int.from_bytes(encoded, "little")
        if len(encoded) != length_bytes or not 0 < header_size <= 4096 or stream.tell() + header_size > len(raw):
            raise ValueError("Native NPY header exceeds its byte bound or is truncated")
        stream.seek(8)
        reader = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
        actual_shape, fortran, dtype = reader(stream, max_header_size=4096)
    except (ValueError, TypeError, EOFError, OverflowError) as error:
        raise ValueError("Native master has an invalid bounded NPY header") from error
    if actual_shape != shape or fortran or dtype != np.dtype("float32"):
        raise ValueError("Native master requires C-layout float32 samples matching descriptor geometry")
    if len(raw) - stream.tell() != math.prod(shape) * 4:
        raise ValueError("Native master payload size differs from descriptor geometry")
    pixels = np.frombuffer(raw, dtype=dtype, offset=stream.tell()).reshape(shape)
    if not np.isfinite(pixels).all():
        raise ValueError("Native master requires finite float32 samples")
    return pixels


@dataclass(frozen=True)
class NativeWorkingMaster:
    """Verified photograph.json and optional straight alpha/source ICC components.

    ``source_icc_*`` accepts actual raw ICC bytes, not the NPY container used by
    V5. Omission preserves the descriptor's declared original-profile digest as
    provenance only. The current working pixels remain extended linear sRGB.
    """

    descriptor_sha256: str
    descriptor_bytes: bytes | None = None
    descriptor_path: Path | None = None
    alpha_sha256: str | None = None
    alpha_bytes: bytes | None = None
    alpha_path: Path | None = None
    source_icc_sha256: str | None = None
    source_icc_bytes: bytes | None = None
    source_icc_path: Path | None = None

    def __post_init__(self) -> None:
        for name, maximum in (
            ("descriptor", MAX_DESCRIPTOR_BYTES),
            ("alpha", MAX_SOURCE_BYTES),
            ("source_icc", MAX_ICC_BYTES),
        ):
            digest, data, path = (getattr(self, name + suffix) for suffix in ("_sha256", "_bytes", "_path"))
            if digest is None:
                if name == "descriptor" or data is not None or path is not None:
                    raise ValueError("Native components require their exact SHA-256")
                continue
            validate_source_sha256(digest)
            if (data is None) == (path is None):
                raise ValueError("Native components require exactly one bytes snapshot or authorized path")
            if data is not None and (not isinstance(data, bytes) or not 0 < len(data) <= maximum):
                raise ValueError("Native component exceeds its immutable byte bound")

    def source_bundle(self, master_sha256: str) -> dict[str, Any]:
        validate_source_sha256(master_sha256)
        return {
            "schema": "tp.photo.source_identity.v1",
            "master_sha256": master_sha256,
            "descriptor_sha256": self.descriptor_sha256,
            "alpha_sha256": self.alpha_sha256,
            "source_icc_sha256": self.source_icc_sha256,
        }

    def identity(self, master_sha256: str) -> str:
        return hashlib.sha256(canonicalize_json(self.source_bundle(master_sha256))).hexdigest()

    def component_snapshots(self) -> dict[str, bytes]:
        result = {}
        for name, maximum in (
            ("descriptor", MAX_DESCRIPTOR_BYTES),
            ("alpha", MAX_SOURCE_BYTES),
            ("source_icc", MAX_ICC_BYTES),
        ):
            digest = getattr(self, name + "_sha256")
            if digest is not None:
                result[name] = read_snapshot(getattr(self, name + "_bytes"), getattr(self, name + "_path"), digest, maximum)
        return result

    def decode(self, raw: bytes, master_sha256: str) -> tuple[ImageMaster, dict[str, Any]]:
        if hashlib.sha256(raw).hexdigest() != master_sha256:
            raise ValueError("Native master physical SHA-256 differs from selected artifact")
        components = self.component_snapshots()
        try:
            photograph = json.loads(components["descriptor"], object_pairs_hook=_unique_object)
        except (ValueError, UnicodeError, RecursionError) as error:
            raise ValueError("Invalid native photograph descriptor") from error
        if (
            not isinstance(photograph, dict)
            or not isinstance(photograph.get("schema"), str)
            or photograph.get("schema")
            not in {
                "tp.lux.photograph.v2",
                "tp.lux.photograph.v3",
                "tp.lux.graded_photograph.v1",
                "tp.lux.graded_photograph.v2",
            }
        ):
            raise ValueError("Native master requires a retained V5/V6 photographic descriptor")
        payload = photograph.get("master")
        expected = {
            "schema",
            "color_space",
            "alpha_mode",
            "shape",
            "source_sha256",
            "source_bit_depth",
            "source_icc_sha256",
            "metadata",
        }
        if (
            not isinstance(payload, dict)
            or set(payload) != expected
            or payload["schema"] != "tp.image.master.v1"
            or payload["color_space"] != "linear_srgb"
        ):
            raise ValueError("Native master descriptor must explicitly declare the linear-sRGB master contract")
        shape = payload["shape"]
        if (
            not isinstance(shape, list)
            or len(shape) != 2
            or any(type(value) is not int or value <= 0 for value in shape)
            or math.prod(shape) > MAX_IMAGE_PIXELS
        ):
            raise ValueError("Native master descriptor exceeds its photographic geometry bound")
        validate_source_sha256(payload["source_sha256"])
        if (
            type(payload["source_bit_depth"]) is not int
            or payload["source_bit_depth"] not in {8, 16, 32}
            or not isinstance(payload["metadata"], dict)
        ):
            raise ValueError("Native descriptor requires typed source precision and metadata")
        try:
            freeze_metadata(payload["metadata"])
        except RecursionError as error:
            raise ValueError("Native descriptor metadata exceeds its nesting bound") from error
        if payload["alpha_mode"] != "straight" and payload["alpha_mode"] is not None:
            raise ValueError("Native photographic alpha must explicitly be straight or absent")
        if (payload["alpha_mode"] == "straight") != ("alpha" in components):
            raise ValueError("Native straight-alpha descriptor and selected alpha component must match")
        declared_icc = payload["source_icc_sha256"]
        if declared_icc is not None:
            validate_source_sha256(declared_icc)
        if "source_icc" in components and self.source_icc_sha256 != declared_icc:
            raise ValueError("Actual source ICC bytes differ from the retained descriptor's original-profile digest")
        pixels = _array(raw, (*shape, 3))
        alpha = _array(components["alpha"], tuple(shape)) if "alpha" in components else None
        if alpha is not None and np.any((alpha < 0) | (alpha > 1)):
            raise ValueError("Native straight alpha must be in [0,1]")
        provenance = {
            "source_identity_sha256": self.identity(master_sha256),
            "source_bundle": self.source_bundle(master_sha256),
            "descriptor_schema": photograph["schema"],
            "retained_descriptor_content_sha256": artifact_content_hash(payload, {"pixels": pixels, "alpha": alpha}),
            "pixel_sha256": hashlib.sha256(pixels.tobytes(order="C")).hexdigest(),
            "original_source_sha256": payload["source_sha256"],
            "original_source_bit_depth": payload["source_bit_depth"],
            "original_source_icc_sha256": declared_icc,
            "original_source_icc_verified": "source_icc" in components,
            "verification_scope": "Selected component integrity and working encoding; generation replay and camera-profile authenticity remain separate",
        }
        metadata = {
            "source_format": "NPY",
            "input_color": "linear_srgb",
            "color_preparation": {
                "schema": "tp.photo.native_working_color.v1",
                "working_color": "extended_linear_srgb",
                "assumption": None,
            },
            "native_working_master": provenance,
        }
        # Original ICC bytes describe source provenance, never the current linear
        # working encoding. No fabricated ICC carrier recreates an old hash.
        master = ImageMaster(pixels, master_sha256, 32, alpha, metadata)
        return master, provenance
