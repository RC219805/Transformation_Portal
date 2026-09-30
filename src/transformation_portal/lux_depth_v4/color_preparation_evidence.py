"""Replay bounded color-resolution receipts against retained profile bytes.

This validates the declared preparation recipe, not the original photographic
pixels. V4/V5 retain the prepared master and source digest, not a source copy.
"""

from __future__ import annotations

import hashlib
import io
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from transformation_portal.core.image_artifact import metadata_payload
from transformation_portal.ingest.canonical_json import canonicalize_json

from .color_preparation import MAX_ICC_BYTES, prepare_input_color
from .io import snapshot


def validate_color_preparation_evidence(metadata: Any, *, input_color: str, source_icc: bytes | None) -> None:
    """Bind the receipt to the frozen request, master metadata, and original ICC."""
    if not isinstance(metadata, Mapping):
        raise ValueError("Color preparation requires master metadata")
    if "color_preparation" not in metadata:
        if input_color == "auto_assume_srgb":
            raise ValueError("The sRGB assumption policy requires color preparation evidence")
        return  # Historical masters predate color preparation receipts.
    try:
        receipt = metadata_payload(metadata["color_preparation"])
        bounded = isinstance(receipt, dict) and len(canonicalize_json(receipt)) <= 8192
    except (TypeError, OverflowError, RecursionError) as exc:
        raise ValueError("Color preparation evidence exceeds its bounded object contract") from exc
    if not bounded:
        raise ValueError("Color preparation evidence exceeds its bounded object contract")
    source = receipt.get("source_metadata")
    if (
        not isinstance(source, dict)
        or set(source) != {"image_format", "declared_color", "exif_color", "png"}
        or not isinstance(source["image_format"], str)
        or source["image_format"] not in {"JPEG", "PNG", "TIFF", "RAW"}
        or not (
            source["declared_color"] is None
            or (
                isinstance(source["declared_color"], str)
                and source["declared_color"] in {"srgb", "linear_srgb", "unsupported"}
            )
        )
        or not (
            type(source["exif_color"]) in {int, type(None)}
            or (isinstance(source["exif_color"], str) and source["exif_color"] == "unsupported")
        )
        or not isinstance(source["png"], dict)
        or not set(source["png"]).issubset({"srgb", "gamma", "chromaticity", "cicp"})
    ):
        raise ValueError("Color preparation has invalid source metadata")
    if (
        receipt.get("requested_input_color") != input_color
        or metadata.get("source_format") != source["image_format"]
        or metadata.get("input_color") != receipt.get("source_color")
        or metadata.get("color_resolution") != receipt.get("resolution")
        or receipt.get("source_icc_sha256") != (hashlib.sha256(source_icc).hexdigest() if source_icc is not None else None)
    ):
        raise ValueError("Color preparation differs from its request, master, or retained source ICC")
    try:
        expected = prepare_input_color(
            input_color=input_color,
            image_format=source["image_format"],
            profile=source_icc,
            declared_color=source["declared_color"],
            exif_color=source["exif_color"],
            png_metadata=source["png"],
        ).evidence
    except (TypeError, OverflowError, RecursionError) as exc:
        raise ValueError("Color preparation has invalid source metadata") from exc
    if canonicalize_json(receipt) != canonicalize_json(expected):
        raise ValueError("Color preparation evidence differs from independently resolved policy")


def retained_source_icc(root: Path, relative: str, declared: Mapping[str, Any]) -> bytes:
    """Read one inventory-bound byte array without trusting its allocation header."""
    record = declared.get(relative)
    maximum = MAX_ICC_BYTES + 4096
    if (
        not isinstance(record, dict)
        or record.get("kind") != "array"
        or type(record.get("size_bytes")) is not int
        or not 0 < record["size_bytes"] <= maximum
    ):
        raise ValueError("Color preparation requires a bounded retained source ICC array")
    raw, observed = snapshot(root, root / relative, maximum_bytes=maximum)
    if any(observed[key] != record[key] for key in ("path", "size_bytes", "sha256")):
        raise ValueError("Retained source ICC changed after inventory verification")
    stream = io.BytesIO(raw)
    try:
        version = np.lib.format.read_magic(stream)
        if version not in {(1, 0), (2, 0)}:
            raise ValueError("Source ICC requires NPY v1/v2")
        length_bytes = 2 if version == (1, 0) else 4
        encoded = stream.read(length_bytes)
        header_size = int.from_bytes(encoded, "little")
        if len(encoded) != length_bytes or not 0 < header_size <= 4096 or stream.tell() + header_size > len(raw):
            raise ValueError("Source ICC array header exceeds its byte bound")
        stream.seek(8)
        reader = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
        shape, fortran, dtype = reader(stream, max_header_size=4096)
        if (
            len(shape) != 1
            or type(shape[0]) is not int
            or not 1 <= shape[0] <= MAX_ICC_BYTES
            or dtype != np.dtype("uint8")
            or fortran
        ):
            raise ValueError("Source ICC requires a bounded one-dimensional byte array")
        if len(raw) - stream.tell() != shape[0]:
            raise ValueError("Source ICC array payload differs from its header")
    except (TypeError, EOFError, OverflowError) as exc:
        raise ValueError("Invalid retained source ICC array") from exc
    return raw[stream.tell() :]
