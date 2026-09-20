"""Snapshot optional photographic masks and calibrated camera intrinsics.

Companions refer to orientation-normalized master coordinates. Their complete
semantic records and mask byte digests are carried by the execution plan;
mutable filesystem paths never stand in for authorizing content.
"""

from __future__ import annotations

import io
import math
import re
import unicodedata
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

import numpy as np

from transformation_portal.core.depth_artifact import CameraIntrinsics, DepthArtifact
from transformation_portal.core.execution_plan import decode_bounded_json_object
from transformation_portal.core.image_artifact import ImageProxy, metadata_payload, validate_source_sha256
from transformation_portal.lux_depth_v4.io import directory_path, pinned_directory, snapshot

COMPANIONS_SCHEMA = "tp.lux.companions.v1"
_MAX_MANIFEST_BYTES = 1024 * 1024
_MAX_MATERIALS = 32


def _object(value: Any, required: set[str], optional: set[str] | None = None) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not required <= set(value) or set(value) - required - (optional or set()):
        raise ValueError("Companion object contains missing or unsupported members")
    return value


def _portable_path(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError("Companion paths must be portable relative paths")
    path = PurePosixPath(value)
    if (
        not value
        or value == "."
        or value != path.as_posix()
        or path.is_absolute()
        or ".." in path.parts
        or "\\" in value
        or ":" in value
        or any(unicodedata.category(char) in {"Cc", "Cf", "Cs"} for char in value)
        or any(part.endswith((".", " ")) for part in path.parts)
    ):
        raise ValueError("Companion paths must be portable relative paths")
    return value


def _positive_int(value: Any) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError("Companion dimensions and budgets must be positive integers")
    return value


def _number(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("Companion numeric values must be finite numbers")
    try:
        result = float(value)
    except OverflowError as exc:
        raise ValueError("Companion numeric values must be finite numbers") from exc
    if not math.isfinite(result):
        raise ValueError("Companion numeric values must be finite numbers")
    return result


def _calibration(value: Any) -> None:
    calibration = _object(value, {"width", "height", "fx", "fy", "cx", "cy", "source", "coordinate_space"})
    _positive_int(calibration["width"])
    _positive_int(calibration["height"])
    if _number(calibration["fx"]) <= 0 or _number(calibration["fy"]) <= 0:
        raise ValueError("Calibration focal lengths must be positive pixels")
    if (
        not 0 <= _number(calibration["cx"]) < calibration["width"]
        or not 0 <= _number(calibration["cy"]) < calibration["height"]
    ):
        raise ValueError("Calibration principal point must be on the declared canonical master grid")
    source = calibration["source"]
    if (
        not isinstance(source, str)
        or not source.strip()
        or source != source.strip()
        or len(source) > 512
        or source.casefold() in {"estimated", "unknown", "unavailable"}
        or any(unicodedata.category(char) in {"Cc", "Cf", "Cs"} for char in source)
    ):
        raise ValueError("Calibration requires explicit measured intrinsics provenance")
    if calibration["coordinate_space"] != "canonical_master":
        raise ValueError("Calibration must use orientation-normalized canonical_master coordinates")


def _materials(value: Any, *, frozen: bool) -> None:
    materials = _object(value, {"masks", "confidences", "coordinate_space"})
    if materials["coordinate_space"] != "canonical_master":
        raise ValueError("Material masks must use orientation-normalized canonical_master coordinates")
    masks = materials["masks"]
    confidences = materials["confidences"]
    if not isinstance(masks, Mapping) or not 1 <= len(masks) <= _MAX_MATERIALS:
        raise ValueError("Materials require a bounded nonempty mask mapping")
    if not isinstance(confidences, Mapping) or set(confidences) - set(masks):
        raise ValueError("Confidence cannot authorize an absent material mask")
    paths: set[str] = set()
    geometry = None
    for name, mask in masks.items():
        if not isinstance(name, str) or re.fullmatch(r"[a-z][a-z0-9_]{0,63}", name) is None:
            raise ValueError("Material names must be bounded lowercase taxonomy keys")
        keys = {"path", "sha256", "size_bytes", "shape", "dtype"} if frozen else {"path", "sha256"}
        mask = _object(mask, keys)
        path = _portable_path(mask["path"])
        key = unicodedata.normalize("NFC", path).casefold()
        if key in paths:
            raise ValueError("Material mask paths must be unique")
        paths.add(key)
        validate_source_sha256(mask["sha256"])
        if frozen:
            _positive_int(mask["size_bytes"])
            shape = mask["shape"]
            if not isinstance(shape, (list, tuple)) or len(shape) != 2:
                raise ValueError("Material mask geometry must be HW")
            shape = tuple(_positive_int(size) for size in shape)
            if geometry is not None and shape != geometry:
                raise ValueError("Material masks must share canonical master geometry")
            geometry = shape
            if mask["dtype"] not in {"bool", "float32"}:
                raise ValueError("Material masks must be bool or float32")
            if mask["size_bytes"] <= math.prod(shape) * (1 if mask["dtype"] == "bool" else 4):
                raise ValueError("Material mask size must include the bounded NPY header")
        if name in confidences and not 0 <= _number(confidences[name]) <= 1:
            raise ValueError("Material confidence must be in [0,1]")


def validate_record(record: Mapping[str, Any]) -> None:
    """Validate a complete frozen plan record without touching the filesystem."""
    record = _object(record, {"path", "source_sha256"}, {"calibration", "materials"})
    _portable_path(record["path"])
    validate_source_sha256(record["source_sha256"])
    if "calibration" not in record and "materials" not in record:
        raise ValueError("A companion record must contain calibration or materials")
    if "calibration" in record:
        _calibration(record["calibration"])
    if "materials" in record:
        _materials(record["materials"], frozen=True)
        if "calibration" in record:
            calibration = record["calibration"]
            expected = (calibration["height"], calibration["width"])
            if any(tuple(mask["shape"]) != expected for mask in record["materials"]["masks"].values()):
                raise ValueError("Calibration and masks must describe the same canonical master geometry")


def _mask_array(data: bytes, max_pixels: int) -> np.ndarray:
    """Reject oversized/unsafe headers before constructing an array view."""
    stream = io.BytesIO(data)
    version = np.lib.format.read_magic(stream)
    if version == (1, 0):
        shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream, max_header_size=4096)
    elif version == (2, 0):
        shape, fortran, dtype = np.lib.format.read_array_header_2_0(stream, max_header_size=4096)
    else:
        raise ValueError("Material masks require NPY format 1 or 2")
    if (
        len(shape) != 2
        or any(type(size) is not int or size <= 0 for size in shape)
        or math.prod(shape) > max_pixels
        or fortran
        or dtype not in {np.dtype("bool"), np.dtype("float32")}
    ):
        raise ValueError("Material masks require bounded contiguous bool or float32 HW arrays")
    if len(data) - stream.tell() != math.prod(shape) * dtype.itemsize:
        raise ValueError("Material mask payload length differs from its declared shape")
    array = np.frombuffer(data, dtype=dtype, offset=stream.tell()).reshape(shape)
    if not np.isfinite(array).all() or np.any((array < 0) | (array > 1)):
        raise ValueError("Material masks must be finite in [0,1]")
    return array


def freeze_companions(
    manifest_path: Path,
    input_records: Sequence[Mapping[str, Any]],
    max_input_bytes: int,
    max_pixels: int,
) -> tuple[dict[str, dict[str, Any]], Path, dict[str, Any]]:
    """Read one strict manifest and freeze complete mask/input authorization records."""
    _positive_int(max_input_bytes)
    _positive_int(max_pixels)
    manifest = Path(manifest_path).expanduser().absolute()
    root = directory_path(manifest.parent)
    result: dict[str, dict[str, Any]] = {}
    selected = {item["path"]: item["sha256"] for item in input_records}
    with pinned_directory(root):
        data, manifest_receipt = snapshot(root, root / manifest.name, maximum_bytes=min(max_input_bytes, _MAX_MANIFEST_BYTES))
        payload = decode_bounded_json_object(data)
        _object(payload, {"schema", "inputs"})
        if payload["schema"] != COMPANIONS_SCHEMA or not isinstance(payload["inputs"], list):
            raise ValueError("Unsupported companion manifest schema")
        if not 1 <= len(payload["inputs"]) <= 1024:
            raise ValueError("Companion manifests require between 1 and 1024 input records")
        for supplied in payload["inputs"]:
            _object(supplied, {"path", "source_sha256"}, {"calibration", "materials"})
            path = _portable_path(supplied["path"])
            if path in result or path not in selected or supplied["source_sha256"] != selected[path]:
                raise ValueError("Companion source binding is duplicate, absent, or differs from selected input bytes")
            record = metadata_payload(supplied)
            if "calibration" in record:
                _calibration(record["calibration"])
                calibration = record["calibration"]
                if calibration["width"] * calibration["height"] > max_pixels:
                    raise ValueError("Calibration geometry exceeds the pixel budget")
            if "materials" in record:
                _materials(record["materials"], frozen=False)
                total = 0
                for name, mask in sorted(record["materials"]["masks"].items()):
                    mask_data, receipt = snapshot(root, root / mask["path"], maximum_bytes=max_input_bytes)
                    if receipt["sha256"] != mask["sha256"]:
                        raise ValueError("Material mask digest differs from the manifest")
                    total += receipt["size_bytes"]
                    if total > max_input_bytes:
                        raise ValueError("Combined material masks exceed the input byte budget")
                    array = _mask_array(mask_data, max_pixels)
                    record["materials"]["masks"][name] = {
                        **receipt,
                        "shape": list(array.shape),
                        "dtype": array.dtype.name,
                    }
            validate_record(record)
            result[path] = record
    return result, root, manifest_receipt


def load_materials(
    root: Path,
    record: Mapping[str, Any] | None,
    *,
    max_input_bytes: int,
    max_pixels: int,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Recheck frozen byte identities and return immutable canonical-master masks."""
    _positive_int(max_input_bytes)
    _positive_int(max_pixels)
    if record is None:
        return {}, {}
    validate_record(record)
    materials = record.get("materials")
    if materials is None:
        return {}, {}
    result: dict[str, np.ndarray] = {}
    total = 0
    with pinned_directory(root):
        for name, receipt in sorted(materials["masks"].items()):
            total += receipt["size_bytes"]
            if total > max_input_bytes:
                raise ValueError("Combined material masks exceed the input byte budget")
            data, observed = snapshot(root, root / receipt["path"], maximum_bytes=max_input_bytes)
            if any(observed[key] != receipt[key] for key in ("path", "sha256", "size_bytes")):
                raise ValueError("Material mask changed after preparation")
            array = _mask_array(data, max_pixels)
            if list(array.shape) != list(receipt["shape"]) or array.dtype.name != receipt["dtype"]:
                raise ValueError("Material mask array differs from the frozen descriptor")
            result[name] = array
    return result, {name: float(value) for name, value in materials["confidences"].items()}


def calibrated_depth(
    native: np.ndarray,
    proxy: ImageProxy,
    record: Mapping[str, Any] | None,
    source_sha256: str,
    confidence: np.ndarray | None = None,
) -> DepthArtifact:
    """Preserve native DA3 values and derive meters only with explicit calibration.

    The pinned DA3 README specifies ``focal_pixels * net_output / 300`` for
    DA3METRIC-LARGE. Focal pixels are transformed onto the actual model grid;
    bottom/right padding translates neither focal lengths nor principal point.
    """
    validate_source_sha256(source_sha256)
    if record is not None:
        validate_record(record)
        if record["source_sha256"] != source_sha256:
            raise ValueError("Calibration source digest differs from photographic input")
    native = np.asarray(native)
    if native.shape != proxy.transform.padded_shape:
        raise ValueError("Native depth geometry differs from the admitted proxy grid")
    metadata: dict[str, Any] = {
        "alignment": proxy.transform.to_payload(),
        "native_grid": "model_proxy",
        "confidence_status": "unavailable" if confidence is None else "bounded_model_score_uncalibrated",
        "metric_status": "unavailable_without_calibration",
    }
    calibration = record.get("calibration") if record is not None else None
    if calibration is None:
        return DepthArtifact(
            native,
            "da3_metric_uncalibrated",
            np.isfinite(native),
            source_sha256,
            confidence=confidence,
            metadata=metadata,
        )
    height, width = proxy.transform.original_shape
    if (calibration["height"], calibration["width"]) != (height, width):
        raise ValueError("Calibration geometry differs from the canonical photographic master")
    resized_height, resized_width = proxy.transform.resized_shape
    scale_x, scale_y = resized_width / width, resized_height / height
    intrinsics = CameraIntrinsics(
        fx=float(calibration["fx"]) * scale_x,
        fy=float(calibration["fy"]) * scale_y,
        cx=(float(calibration["cx"]) + 0.5) * scale_x - 0.5,
        cy=(float(calibration["cy"]) + 0.5) * scale_y - 0.5,
        width=proxy.transform.padded_shape[1],
        height=proxy.transform.padded_shape[0],
        source=calibration["source"],
    )
    factor = (intrinsics.fx + intrinsics.fy) / 600.0
    metric = np.asarray(native, dtype=np.float32) * factor
    metadata["metric_status"] = "calibrated_from_supplied_intrinsics"
    return DepthArtifact(
        native,
        "da3_metric_uncalibrated",
        np.isfinite(native) & (native > 0),
        source_sha256,
        metric_map_m=metric,
        confidence=confidence,
        intrinsics=intrinsics,
        calibration={
            "method": "da3_metric_focal_pixels_div_300",
            "input_intrinsics": dict(calibration),
            "focal_pixels": (intrinsics.fx + intrinsics.fy) / 2.0,
            "meters_per_native_unit": factor,
        },
        metadata=metadata,
    )
