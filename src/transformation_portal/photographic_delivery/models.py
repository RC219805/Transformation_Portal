"""Explicit source snapshots and bounded destination recipes."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from transformation_portal.core.image_artifact import ImageMaster, metadata_payload, validate_source_sha256
from transformation_portal.core.photographic_export import export_metadata
from transformation_portal.lux_depth_v4.photography import decode_master

from .native import NativeWorkingMaster, read_snapshot

MAX_SOURCE_BYTES = 2 * 1024**3
MAX_IMAGE_PIXELS = 100_000_000


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_bytes(value: Any) -> bytes:
    from transformation_portal.ingest.canonical_json import canonicalize_json

    return canonicalize_json(value)


def bounded_number(value: Any, name: str, lower: float, upper: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number in [{lower},{upper}]")
    try:
        result = float(value)
    except OverflowError as error:
        raise ValueError(f"{name} must be a finite number") from error
    if not np.isfinite(result) or not lower <= result <= upper:
        raise ValueError(f"{name} must be a finite number in [{lower},{upper}]")
    return result


def slug(value: str, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) > 256
        or not value.strip()
        or any(unicodedata.category(char) in {"Cc", "Cf", "Cs"} for char in value)
    ):
        raise ValueError(f"{name} must be a nonempty label of at most 256 characters")
    result = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")[:48].rstrip("-")
    if not result:
        raise ValueError(f"{name} must contain ASCII letters or digits")
    return result


@dataclass(frozen=True)
class DeliveryImage:
    """One exact, caller-authorized source; path inputs load one image at a time.

    ``source_sha256`` is the artifact's byte hash, not an inferred filename or
    pixel hash. For native arrays, ``source_identity_sha256`` additionally binds
    descriptor, alpha, and selected original ICC components. The managed
    boundary validates authorization and human approval independently; a new
    derivative never inherits that approval.
    """

    source_name: str
    source_sha256: str
    scene: str
    version: str = "v1"
    source_bytes: bytes | None = None
    source_path: Path | None = None
    input_color: str = "auto"
    metadata: Mapping[str, str] = field(default_factory=dict)
    native: NativeWorkingMaster | None = None

    def __post_init__(self) -> None:
        validate_source_sha256(self.source_sha256)
        slug(self.scene, "scene")
        slug(self.version, "version")
        if not isinstance(self.source_name, str) or Path(self.source_name).name != self.source_name:
            raise ValueError("source_name must be a filename")
        if self.native is not None and not isinstance(self.native, NativeWorkingMaster):
            raise ValueError("Native photographic arrays require an explicit NativeWorkingMaster")
        suffixes = {".npy"} if self.native is not None else {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
        if Path(self.source_name).suffix.lower() not in suffixes:
            raise ValueError("Delivery requires an explicit photographic image or a descriptor-bound native NPY master")
        if (self.source_bytes is None) == (self.source_path is None):
            raise ValueError("Provide exactly one source_bytes or caller-authorized source_path")
        if self.source_bytes is not None and (
            not isinstance(self.source_bytes, bytes) or not 0 < len(self.source_bytes) <= MAX_SOURCE_BYTES
        ):
            raise ValueError("Source snapshot must contain bounded immutable bytes")
        if not isinstance(self.input_color, str) or self.input_color not in {
            "auto",
            "auto_assume_srgb",
            "srgb",
            "linear_srgb",
        }:
            raise ValueError("Unsupported explicit input color policy")
        object.__setattr__(self, "metadata", export_metadata(self.metadata))

    @property
    def source_identity_sha256(self) -> str:
        return self.native.identity(self.source_sha256) if self.native is not None else self.source_sha256

    @property
    def source_bundle(self) -> dict[str, Any]:
        return (
            self.native.source_bundle(self.source_sha256)
            if self.native is not None
            else {
                "schema": "tp.photo.source_identity.v1",
                "master_sha256": self.source_sha256,
                "descriptor_sha256": None,
                "alpha_sha256": None,
                "source_icc_sha256": None,
            }
        )

    def snapshot(self) -> bytes:
        return read_snapshot(self.source_bytes, self.source_path, self.source_sha256)

    def decode(self, data: bytes | None = None) -> ImageMaster:
        snapshot = self.snapshot() if data is None else data
        if sha256(snapshot) != self.source_sha256:
            raise ValueError("Decoded snapshot does not match the source master SHA-256")
        if self.native is not None:
            if self.input_color not in {"auto", "linear_srgb"}:
                raise ValueError("Native working masters explicitly use extended linear sRGB")
            return self.native.decode(snapshot, self.source_sha256)[0]
        return decode_master(snapshot, source_name=self.source_name, input_color=self.input_color, max_pixels=MAX_IMAGE_PIXELS)


@dataclass(frozen=True)
class ExportPreset:
    """Destination constraints, not a claim of universal MLS/provider compliance."""

    name: str
    format: str
    intended_use: str
    long_edge: int | None = None
    jpeg_quality: int = 96
    sharpen_amount: float = 0.0
    render: str = "perceptual_srgb"
    shoulder: float = 0.8

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or self.name not in {
            "photographic_master",
            "website",
            "mls",
            "social",
            "print",
            "retouch_print",
        }:
            raise ValueError("Unsupported named photographic destination")
        required_format = {
            "photographic_master": "source",
            "website": "jpeg",
            "mls": "jpeg",
            "social": "jpeg",
            "print": "tiff16",
            "retouch_print": "tiff32",
        }[self.name]
        if self.format != required_format:
            raise ValueError("Destination format must preserve its named profile contract")
        if not isinstance(self.intended_use, str) or not 1 <= len(self.intended_use) <= 200:
            raise ValueError("intended_use must be a bounded description")
        if self.long_edge is not None and (type(self.long_edge) is not int or not 1 <= self.long_edge <= 16384):
            raise ValueError("long_edge must be null or an integer in [1,16384]")
        if self.format in {"source", "tiff32"} and (self.long_edge is not None or self.sharpen_amount != 0):
            raise ValueError("Master and retouch profiles retain native precision/geometry without output sharpening")
        if type(self.jpeg_quality) is not int or not 50 <= self.jpeg_quality <= 100:
            raise ValueError("jpeg_quality must be an integer in [50,100]")
        object.__setattr__(self, "sharpen_amount", bounded_number(self.sharpen_amount, "sharpen_amount", 0, 1))
        if not isinstance(self.render, str) or self.render not in {"perceptual_srgb", "soft_srgb", "clip_srgb"}:
            raise ValueError("Unsupported explicit SDR display render")
        object.__setattr__(self, "shoulder", bounded_number(self.shoulder, "shoulder", 0.1, 0.95))

    def to_payload(self) -> dict[str, Any]:
        return dict(schema="tp.photo.export_preset.v1", **self.__dict__)

    @classmethod
    def from_payload(cls, payload: str | Mapping[str, Any]) -> ExportPreset:
        defaults = {preset.name: preset for preset in destination_presets()}
        if isinstance(payload, str):
            if payload not in defaults:
                raise ValueError("Unsupported photographic destination preset")
            return defaults[payload]
        if (
            not isinstance(payload, Mapping)
            or set(payload) - {"name", "long_edge", "jpeg_quality", "sharpen_amount", "render", "shoulder"}
            or not isinstance(payload.get("name"), str)
            or payload.get("name") not in defaults
        ):
            raise ValueError("Preset overrides require a known name and bounded explicit controls")
        values = dict(defaults[payload["name"]].__dict__)
        values.update(payload)
        return cls(**values)


def destination_presets() -> tuple[ExportPreset, ...]:
    return (
        ExportPreset("photographic_master", "source", "Native source precision and original color; byte-preserved master"),
        ExportPreset("website", "jpeg", "Website encoded-sRGB delivery", 2560, 92, 0.2),
        ExportPreset(
            "mls", "jpeg", "Configurable MLS encoded-sRGB export; confirm your provider's requirements", 3000, 95, 0.15
        ),
        ExportPreset("social", "jpeg", "Aspect-preserving social export; no automatic platform crop", 2048, 92, 0.2),
        ExportPreset("print", "tiff16", "16-bit SDR sRGB proof; printer-specific profile conversion remains external"),
        ExportPreset("retouch_print", "tiff32", "32-bit extended encoded-sRGB TIFF for retouch; no tone mapping or clipping"),
    )


def inspect_image(image: DeliveryImage) -> dict[str, Any]:
    master = image.decode()
    return {
        "width": master.shape[1],
        "height": master.shape[0],
        "bit_depth": master.source_bit_depth,
        "color_profile": master.metadata["input_color"],
        "source_icc_sha256": sha256(master.source_icc) if master.source_icc else None,
        "working_color": "extended_linear_srgb",
        "source_sha256": image.source_sha256,
        "source_identity_sha256": image.source_identity_sha256,
        "source_bundle": image.source_bundle,
        "master_content_sha256": master.content_hash(),
        "color_preparation": metadata_payload(master.metadata["color_preparation"]),
        "intended_use": "Photographic master",
        "alpha": master.alpha is not None,
        **metadata_payload(master.metadata.get("native_working_master", {})),
    }
