"""Replayable V6 photographs and explicitly estimated Depth Pro meter products.

Depth Pro does not supply sky evidence. Its finite positive samples support
numeric visualization only; they never authorize depth-driven photographic edits
or claims of calibrated surface geometry.
"""

from __future__ import annotations

import hashlib
import io
import re
from typing import Any, Callable, Iterator

import numpy as np
from PIL import Image

from transformation_portal.core.image_artifact import ImageMaster, freeze_metadata, metadata_payload
from transformation_portal.ingest.canonical_json import canonicalize_json
from transformation_portal.lux_depth_v4.photography import decode_master, linear_to_srgb
from transformation_portal.lux_depth_v5.preview import MAX_PREVIEW_BYTES, MAX_PREVIEW_EDGE, encode_preview

from .color import GradeRecipe, RenderRecipe, grade_master, render_master
from .products import _delivery

MODEL_INPUT_RECIPE = "tp.lux.depth_pro.original_grid_srgb_uint8.v1"


def model_input_bytes(master: ImageMaster) -> bytes:
    """Encode the oriented original grid as explicit, deterministic RGB sRGB.

    The inference-only derivative clips extended values and quantizes to uint8.
    Alpha is excluded from model input, but retained in photographic products.
    Neither operation modifies the high-precision source master.
    """
    if type(master) is not ImageMaster:
        raise ValueError("Depth Pro input requires an ImageMaster")
    # Clip in linear space before the monotonic transfer to avoid overflow in
    # its negative branch for finite extended-range float32 source samples.
    samples = np.rint(np.clip(linear_to_srgb(np.clip(master.pixels, 0, 1)), 0, 1) * 255).astype(np.uint8)
    buffer = io.BytesIO()
    with Image.fromarray(samples) as image:
        image.save(buffer, format="PNG", optimize=False, compress_level=6)
    return buffer.getvalue()


def _worker_payload(worker_metadata: dict[str, Any], shape: tuple[int, int]) -> dict[str, Any]:
    if type(worker_metadata) is not dict:
        raise ValueError("Depth Pro worker metadata must be a JSON object")
    payload = metadata_payload(freeze_metadata(worker_metadata))
    if (
        payload.get("depth_units") != "meters"
        or payload.get("dtype") != "float32"
        or type(payload.get("input_size")) is not list
        or any(type(size) is not int for size in payload["input_size"])
        or payload["input_size"] != list(shape)
        or not isinstance(payload.get("provenance"), dict)
        or payload["provenance"].get("engine") != "apple_depth_pro"
    ):
        raise ValueError("Depth Pro worker units, dtype, engine, or original-grid geometry disagree")
    return payload


def _depth_preview(native: np.ndarray, numeric: np.ndarray) -> tuple[bytes, bytes, dict[str, Any]]:
    height, width = native.shape
    longest = max(height, width)
    shape = tuple(max(1, (size * MAX_PREVIEW_EDGE + longest // 2) // longest) for size in native.shape)
    if longest <= MAX_PREVIEW_EDGE:
        shape = native.shape
    rows = ((2 * np.arange(shape[0]) + 1) * height) // (2 * shape[0])
    columns = ((2 * np.arange(shape[1]) + 1) * width) // (2 * shape[1])
    samples = native[rows[:, None], columns[None, :]].astype(np.float64)
    mask = numeric[rows[:, None], columns[None, :]]
    limits = None
    relative = np.zeros(shape, np.float64)
    if numeric.any():
        lower, upper = np.percentile(native[numeric].astype(np.float64), [1, 99], method="linear")
        limits = [float(lower), float(upper)]
        if upper > lower:
            relative[mask] = np.clip((samples[mask] - lower) / (upper - lower), 0, 1)
    encoded = np.rint(relative * 65535).astype(np.uint16)
    preview, validity = io.BytesIO(), io.BytesIO()
    with Image.fromarray(encoded) as image:
        image.save(preview, format="PNG", compress_level=6, optimize=False)
    with Image.fromarray(mask.astype(np.uint8) * 255) as image:
        image.save(validity, format="PNG", compress_level=6, optimize=False)
    if preview.tell() + validity.tell() > MAX_PREVIEW_BYTES:
        raise ValueError("Depth Pro previews exceed their combined encoded-byte budget")
    return (
        preview.getvalue(),
        validity.getvalue(),
        {
            "shape": list(shape),
            "maximum_edge": MAX_PREVIEW_EDGE,
            "sampling": "nearest_pixel_center_integer_floor_v1",
            "normalization": "finite_positive_percentile_1_99_linear_clip_v1",
            "normalization_limits_m": limits,
            "constant_or_empty_depth_value": 0,
            "relative_semantics": "near_0_far_1",
            "units": "normalized_relative_not_meters",
            "bit_depth": 16,
            "invalid_value": 0,
            "validity_semantics": "finite_positive_numeric_only_not_surface_validity",
            "validity_encoding": "grayscale_uint8_255_numeric_0_invalid",
            "transfer": "linear_scalar_no_color_profile",
            "quantization": "round_to_nearest_even(relative_depth*65535)",
        },
    )


def native_image_products(
    source_bytes: bytes,
    source_name: str,
    input_color: str,
    native_depth: np.ndarray,
    worker_metadata: dict[str, Any],
    input_id: str,
    grade: GradeRecipe,
    render: RenderRecipe,
    *,
    max_pixels: int,
    checkpoint: Callable[[], None],
) -> Iterator[tuple[str, bytes]]:
    """Emit deterministic derivatives from a frozen original and native estimate.

    The caller separately retains source.bin, native-depth.npy and worker JSON.
    This stream includes the estimated-meter array/TIFF, numeric-only mask and
    previews, depth descriptor, and independently graded photographic products.
    Replay reproduces these bytes without rerunning nondeterministic inference.
    """
    if type(input_id) is not str or re.fullmatch(r"[a-z0-9_-]{1,128}", input_id) is None:
        raise ValueError("Depth Pro products require a bounded portable input identifier")
    if type(grade) is not GradeRecipe or type(render) is not RenderRecipe:
        raise ValueError("Depth Pro products require explicit V6 grade and render recipes")
    checkpoint()
    baseline = decode_master(source_bytes, source_name=source_name, input_color=input_color, max_pixels=max_pixels)
    if (
        type(native_depth) is not np.ndarray
        or native_depth.dtype != np.dtype("float32")
        or native_depth.shape != baseline.shape
    ):
        raise ValueError("Depth Pro native depth must be exact float32 on the oriented original grid")
    native = np.array(native_depth, dtype=np.float32, copy=True, order="C")
    native.setflags(write=False)
    numeric = np.isfinite(native) & (native > 0)
    metadata = _worker_payload(worker_metadata, baseline.shape)
    input_bytes = model_input_bytes(baseline)
    model_input = {
        "recipe": MODEL_INPUT_RECIPE,
        "sha256": hashlib.sha256(input_bytes).hexdigest(),
        "shape": list(baseline.shape),
        "color_space": "srgb",
        "orientation": "canonical_master",
        "quantization": "round_to_nearest_even(clip(srgb,0,1)*255)",
        "alpha": "ignored_for_inference_preserved_in_photograph",
        "resizing": "none",
    }
    del input_bytes
    records: dict[str, Any] = {}

    def record(filename: str, data: bytes, **details: Any) -> tuple[str, bytes]:
        relative = f"{input_id}/{filename}"
        records[filename] = {"path": relative, "sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data), **details}
        return relative, data

    for filename, values in (("estimated-depth-meters.npy", native), ("numeric-valid.npy", numeric)):
        checkpoint()
        buffer = io.BytesIO()
        np.save(buffer, values, allow_pickle=False)
        yield record(filename, buffer.getvalue(), shape=list(values.shape), dtype=values.dtype.name)
    import tifffile

    checkpoint()
    buffer = io.BytesIO()
    tifffile.imwrite(
        buffer,
        native,
        photometric="minisblack",
        metadata=None,
        description=(
            "Depth Pro model-estimated meters, not calibrated or measured geometry. "
            "Raw invalid/nonfinite samples are retained; numeric-valid.npy is numeric validity only."
        ),
        extratags=[(274, "H", 1, 1, False)],
    )
    yield record("estimated-depth-meters.tif", buffer.getvalue(), shape=list(native.shape), dtype="float32")
    checkpoint()
    preview, validity, preview_descriptor = _depth_preview(native, numeric)
    yield record("depth-preview.png", preview, shape=preview_descriptor["shape"], bit_depth=16)
    yield record("depth-preview-numeric-valid.png", validity, shape=preview_descriptor["shape"], bit_depth=8)
    depth_descriptor = {
        "schema": "tp.lux.depth_pro.depth.v1",
        "input_id": input_id,
        "source_sha256": baseline.source_sha256,
        "original_content_sha256": baseline.content_hash(),
        "model_input": model_input,
        "shape": list(native.shape),
        "native_semantics": "model_estimated_meters",
        "native_invalid_policy": "retain_raw_values_consult_numeric-valid.npy",
        "numeric_valid_pixels": int(numeric.sum()),
        "numeric_valid_semantics": "finite_and_positive_not_surface_validity",
        "sky_status": "unavailable",
        "surface_status": "unavailable",
        "calibration": None,
        "measured_scene_accuracy": "not_established",
        "worker_metadata": metadata,
        "worker_metadata_sha256": hashlib.sha256(canonicalize_json(metadata)).hexdigest(),
        "preview": {
            **preview_descriptor,
            "path": f"{input_id}/depth-preview.png",
            "numeric_validity_path": f"{input_id}/depth-preview-numeric-valid.png",
        },
        "artifacts": records,
        "production_acceptance": "not_established",
    }
    yield f"{input_id}/depth.json", canonicalize_json(depth_descriptor)
    del native, numeric, buffer, preview, validity
    checkpoint()
    graded, grade_receipt = grade_master(baseline, grade)
    checkpoint()
    display, render_receipt = render_master(graded, render)
    descriptor: dict[str, Any] = {
        "schema": "tp.lux.depth_pro.photograph.v1",
        "input_id": input_id,
        "source_sha256": baseline.source_sha256,
        "reconstruction": {
            "recipe": "source_master_identity_v1",
            "depth_edits": "abstained",
            "reason": "sky_evidence_unavailable",
            "changed_pixels": 0,
            "alpha_preserved": baseline.alpha is not None,
        },
        "baseline": baseline.to_payload(),
        "master": graded.to_payload(),
        "display": display.to_payload(),
        "grade": grade_receipt,
        "render": render_receipt,
        "depth": {"path": f"{input_id}/depth.json", "semantics": "model_estimated_meters"},
        "production_acceptance": "not_established",
    }
    arrays = [("baseline.npy", baseline.pixels), ("master.npy", graded.pixels), ("display.npy", display.pixels)]
    if graded.alpha is not None:
        arrays.append(("alpha.npy", graded.alpha))
    for filename, values in arrays:
        checkpoint()
        buffer = io.BytesIO()
        np.save(buffer, values, allow_pickle=False)
        yield f"{input_id}/{filename}", buffer.getvalue()
    checkpoint()
    data, descriptor["delivery"] = _delivery(display, f"{input_id}/delivery.tif")
    yield f"{input_id}/delivery.tif", data
    del data
    checkpoint()
    data, descriptor["preview"] = encode_preview(display, relative_path=f"{input_id}/preview.png")
    yield f"{input_id}/preview.png", data
    del data
    checkpoint()
    yield f"{input_id}/photograph.json", canonicalize_json(descriptor)
