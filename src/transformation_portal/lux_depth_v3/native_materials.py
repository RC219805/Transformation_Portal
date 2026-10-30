"""Closed native SDR Materials bindings for the opt-in V3/V2 finishing path.

The private evidence-root locator is not semantic identity. Every admitted
source, manifest, mask, policy and numerical operation remains content-bound.
Native straight-alpha samples follow MaterialsV4's existing protection policy;
this adapter does not resample photographic pixels or quantize a carrier.
"""

from __future__ import annotations

import hashlib
import io
import math
from dataclasses import replace
from numbers import Integral
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image

from transformation_portal.core.execution_plan import MAX_PLAN_BODY_BYTES, decode_bounded_json_object
from transformation_portal.core.image_artifact import ImageMaster, immutable_array
from transformation_portal.ingest.canonical_json import canonicalize_json
from transformation_portal.lux_depth_v4.companions import _portable_path as _validate_portable_path
from transformation_portal.lux_depth_v4.io import directory_path, pinned_directory, snapshot
from transformation_portal.lux_depth_v4.materials import freeze_materials, load_bound_evidence, validate_frozen_record
from transformation_portal.lux_depth_v4.photography import decode_master, validate_input_color_metadata
from transformation_portal.materials_v4.contracts import MaterialEvidence, MaterialLimits
from transformation_portal.materials_v4.engine import ResponsePolicy, apply_response, plan_response
from transformation_portal.materials_v4.operations import operation_contract_hash

from .geometry_handoff import snapshot_geometry_artifact

NATIVE_MATERIALS_SCHEMA = "tp.lux.native_materials.v1"
NATIVE_MATERIALS_RECIPE = "native_sdr_v1"
MAX_NATIVE_MATERIALS_BINDING_BYTES = MAX_PLAN_BODY_BYTES
NATIVE_MATERIALS_LIMITS = {
    "max_pixels": 67_108_864,
    "max_input_bytes": 2 * 1024**3,
    "max_working_bytes": 12 * 1024**3,
}
_BINDING_KEYS = {
    "schema",
    "recipe",
    "input_color",
    "evidence_root",
    "manifest",
    "policy",
    "operations_sha256",
    "limits",
    "sources",
    "binding_sha256",
}
_SOURCE_KEYS = {
    "path",
    "sha256",
    "master_sha256",
    "size_bytes",
    "shape",
    "source_bit_depth",
    "alpha",
    "mask_bytes",
    "evidence",
}
_SOURCE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def _digest(value: Any, name: str) -> None:
    if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"Native Materials {name} must be a lowercase SHA-256 digest")


def _portable_path(value: Any) -> str:
    if not isinstance(value, str) or not value or len(value.encode("utf-8")) > 4096:
        raise ValueError("Native Materials requires a bounded relative source/artifact path")
    return _validate_portable_path(value)


def native_materials_binding_hash(binding: Mapping[str, Any]) -> str:
    """Bind portable execution meaning while excluding its local root locator."""
    return hashlib.sha256(
        canonicalize_json({key: value for key, value in binding.items() if key not in {"binding_sha256", "evidence_root"}})
    ).hexdigest()


def _allocation_bytes(source: Mapping[str, Any], tile_scratch_bytes: int) -> int:
    # Covers simultaneous native master/output/encoding/geometry and renderer
    # arrays at 256 bytes/pixel, four mask copies and two source-byte snapshots.
    # The frozen tile-scratch ceiling is part of this complete reservation.
    return math.prod(source["shape"]) * 256 + source["mask_bytes"] * 4 + source["size_bytes"] * 2 + tile_scratch_bytes


def _admit_allocation(source: Mapping[str, Any], limits: Mapping[str, int], policy: ResponsePolicy) -> None:
    height = min(source["shape"][0], policy.tile_size + 2 * policy.halo)
    width = min(source["shape"][1], policy.tile_size + 2 * policy.halo)
    if height * width * 256 > policy.max_working_bytes:
        raise ValueError("Native Materials tile scratch exceeds its frozen policy budget")
    if _allocation_bytes(source, policy.max_working_bytes) > limits["max_working_bytes"]:
        raise ValueError("Native Materials conservative allocation exceeds max_working_bytes")


def _receipt(value: Any) -> None:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256", "size_bytes"}:
        raise ValueError("Native Materials manifest requires its exact file receipt")
    _portable_path(value["path"])
    _digest(value["sha256"], "manifest digest")
    if type(value["size_bytes"]) is not int or not 0 < value["size_bytes"] <= 1024**2:
        raise ValueError("Native Materials manifest exceeds its byte budget")


def validate_native_materials(binding: Mapping[str, Any]) -> None:
    """Validate a closed frozen binding without reading files or decoding pixels."""
    if not isinstance(binding, Mapping) or set(binding) != _BINDING_KEYS:
        raise ValueError("Native Materials binding requires its exact closed schema")
    if binding["schema"] != NATIVE_MATERIALS_SCHEMA or binding["recipe"] != NATIVE_MATERIALS_RECIPE:
        raise ValueError("Unsupported native Materials schema or recipe")
    if not isinstance(binding["input_color"], str) or binding["input_color"] not in {
        "auto",
        "auto_assume_srgb",
        "srgb",
        "linear_srgb",
    }:
        raise ValueError("Unsupported native Materials input color")
    root = binding["evidence_root"]
    if not isinstance(root, str) or not root or len(root.encode("utf-8")) > 4096 or "\x00" in root:
        raise ValueError("Native Materials evidence root must be a bounded absolute path")
    if not Path(root).is_absolute() or str(Path(root)) != root or ".." in Path(root).parts:
        raise ValueError("Native Materials evidence root must be canonical and absolute")
    _receipt(binding["manifest"])
    policy = ResponsePolicy.from_payload(binding["policy"])
    if policy.to_payload() != binding["policy"]:
        raise ValueError("Native Materials policy must be complete and canonical")
    _digest(binding["operations_sha256"], "operation digest")
    if binding["operations_sha256"] != operation_contract_hash():
        raise ValueError("Native Materials operation contract differs from the running implementation")
    limits = binding["limits"]
    if (
        not isinstance(limits, Mapping)
        or set(limits) != set(NATIVE_MATERIALS_LIMITS)
        or any(type(limits[key]) is not int or limits[key] != value for key, value in NATIVE_MATERIALS_LIMITS.items())
    ):
        raise ValueError("Native Materials requires its exact conservative allocation limits")
    sources = binding["sources"]
    if not isinstance(sources, (list, tuple)) or not 1 <= len(sources) <= 1024:
        raise ValueError("Native Materials requires a bounded nonempty selected source list")
    paths: set[str] = set()
    for source in sources:
        if not isinstance(source, Mapping) or set(source) != _SOURCE_KEYS:
            raise ValueError("Native Materials source requires its exact closed schema")
        path = _portable_path(source["path"])
        if path in paths or PurePosixPath(path).suffix.lower() not in _SOURCE_SUFFIXES:
            raise ValueError("Native Materials source path is duplicate or is not a supported SDR raster")
        paths.add(path)
        _digest(source["sha256"], "source digest")
        _digest(source["master_sha256"], "native master digest")
        if type(source["size_bytes"]) is not int or not 0 < source["size_bytes"] <= limits["max_input_bytes"]:
            raise ValueError("Native Materials source exceeds its input byte budget")
        shape = source["shape"]
        if (
            not isinstance(shape, (list, tuple))
            or len(shape) != 2
            or any(type(side) is not int or side <= 0 for side in shape)
            or math.prod(shape) > limits["max_pixels"]
        ):
            raise ValueError("Native Materials source geometry exceeds its pixel budget")
        if type(source["source_bit_depth"]) is not int or source["source_bit_depth"] not in {8, 16}:
            raise ValueError("Native Materials supports 8/16-bit SDR sources only")
        if type(source["alpha"]) is not bool:
            raise ValueError("Native Materials source alpha presence must be boolean")
        mask_bytes = source["mask_bytes"]
        if type(mask_bytes) is not int or not 0 <= mask_bytes <= MaterialLimits().max_mask_bytes:
            raise ValueError("Native Materials decoded mask bytes exceed their budget")
        if mask_bytes % (math.prod(shape) * 4):
            raise ValueError("Native Materials mask bytes must bind complete native float32 masks")
        if source["evidence"] is not None:
            validate_frozen_record(source["evidence"], limits)
            if source["evidence"]["source_sha256"] != source["sha256"] or tuple(source["evidence"]["shape"]) != tuple(shape):
                raise ValueError("Native Materials evidence differs from its selected source/frame")
        elif mask_bytes:
            raise ValueError("Native Materials absent evidence cannot reserve actionable masks")
        _admit_allocation(source, limits, policy)
    _digest(binding["binding_sha256"], "binding digest")
    if native_materials_binding_hash(binding) != binding["binding_sha256"]:
        raise ValueError("Native Materials binding digest differs from its frozen contents")
    if len(canonicalize_json(dict(binding))) > MAX_NATIVE_MATERIALS_BINDING_BYTES:
        raise ValueError("Native Materials binding exceeds its serialized byte budget")


def _header(source: bytes, name: str) -> tuple[tuple[int, int], int, bool]:
    """Inspect native precision and normalized geometry before allocating arrays."""
    if source[:4] in {b"II*\x00", b"MM\x00*", b"II+\x00", b"MM\x00+"}:
        import tifffile

        with tifffile.TiffFile(io.BytesIO(source)) as image:
            page = image.pages[0]
            if not isinstance(page, tifffile.TiffPage) or page.dtype is None:
                raise ValueError("Native Materials TIFF requires a complete image page")
            shape = (int(page.imagelength), int(page.imagewidth))
            bits = page.dtype.itemsize * 8
            if page.dtype not in {np.dtype("uint8"), np.dtype("uint16")}:
                raise ValueError("Native Materials requires integer 8/16-bit SDR source samples")
            orientation = page.tags[274].value if 274 in page.tags else 1
            alpha = bool(page.extrasamples)
    else:
        with Image.open(io.BytesIO(source)) as image:
            if image.format not in {"JPEG", "PNG"}:
                raise ValueError("Native Materials supports JPEG/PNG/TIFF SDR sources only")
            shape = (image.height, image.width)
            bits = source[24] if image.format == "PNG" and len(source) >= 26 else 8
            if bits not in {8, 16}:
                raise ValueError("Native Materials requires integer 8/16-bit SDR source samples")
            orientation = image.getexif().get(274, 1)
            alpha = "A" in image.mode or "transparency" in image.info
    if isinstance(orientation, bool) or not isinstance(orientation, Integral):
        raise ValueError("Native Materials source orientation must be an exact EXIF orientation")
    orientation = int(orientation)
    if orientation not in range(1, 9):
        raise ValueError("Native Materials source orientation must be an exact EXIF orientation")
    return (shape[::-1] if orientation >= 5 else shape), bits, alpha


def _manifest_unchanged(binding: Mapping[str, Any]) -> Path:
    root = directory_path(Path(binding["evidence_root"]))
    expected = binding["manifest"]
    data, observed = snapshot(root, root / expected["path"], maximum_bytes=expected["size_bytes"])
    if observed != expected:
        raise ValueError("Native Materials manifest changed after preparation")
    payload = decode_bounded_json_object(data)
    if set(payload) != {"schema", "inputs"} or payload["schema"] != "tp.lux.materials_manifest.v1":
        raise ValueError("Native Materials manifest has an unsupported closed schema")
    supplied = payload["inputs"]
    if not isinstance(supplied, list) or not 1 <= len(supplied) <= 1024:
        raise ValueError("Native Materials manifest requires a bounded nonempty source map")
    selected = {source["path"]: source for source in binding["sources"]}
    seen = set()
    for record in supplied:
        if not isinstance(record, Mapping) or set(record) != {
            "path",
            "source_sha256",
            "evidence_path",
            "evidence_sha256",
            "shape",
        }:
            raise ValueError("Native Materials manifest source requires its exact closed schema")
        path = _portable_path(record["path"])
        source = selected.get(path)
        if source is None or path in seen or source["evidence"] is None:
            raise ValueError("Native Materials manifest source map differs from its frozen selection")
        seen.add(path)
        expected_record = source["evidence"]
        if (
            record["source_sha256"] != source["sha256"]
            or record["evidence_path"] != expected_record["path"]
            or record["evidence_sha256"] != expected_record["sha256"]
            or record["shape"] != source["shape"]
        ):
            raise ValueError("Native Materials manifest evidence differs from its frozen binding")
    if seen != {path for path, source in selected.items() if source["evidence"] is not None}:
        raise ValueError("Native Materials frozen evidence is absent from its manifest")
    return root


def freeze_native_materials(
    manifest_path: Path,
    policy_path: Path,
    input_root: Path,
    input_files: Sequence[Path],
    input_color: str,
) -> dict[str, Any]:
    """Freeze supplied evidence and native source frames before any execution."""
    if input_color not in {"auto", "auto_assume_srgb", "srgb", "linear_srgb"}:
        raise ValueError("Unsupported native Materials input color")
    root = directory_path(input_root)
    selected = tuple(Path(path) for path in input_files)
    if not 1 <= len(selected) <= 1024:
        raise ValueError("Native Materials requires a bounded nonempty source selection")
    policy_file = Path(policy_path).expanduser().absolute()
    raw_policy, _ = snapshot(directory_path(policy_file.parent), policy_file, maximum_bytes=1024**2)
    policy = ResponsePolicy.from_payload(decode_bounded_json_object(raw_policy))
    sources = []
    with pinned_directory(root):
        for supplied in selected:
            path = supplied if supplied.is_absolute() else root / supplied
            data, source = snapshot(root, path, maximum_bytes=NATIVE_MATERIALS_LIMITS["max_input_bytes"])
            if Path(source["path"]).suffix.lower() not in _SOURCE_SUFFIXES:
                raise ValueError("Native Materials supports JPEG/PNG/TIFF SDR sources only")
            shape, bits, alpha = _header(data, source["path"])
            record = {
                **source,
                "shape": list(shape),
                "source_bit_depth": bits,
                "alpha": alpha,
                "mask_bytes": 0,
                "evidence": None,
            }
            if math.prod(shape) > NATIVE_MATERIALS_LIMITS["max_pixels"]:
                raise ValueError("Native Materials source geometry exceeds its pixel budget")
            _admit_allocation(record, NATIVE_MATERIALS_LIMITS, policy)
            validate_input_color_metadata(
                io.BytesIO(data),
                source_name=source["path"],
                input_color=input_color,
                max_pixels=NATIVE_MATERIALS_LIMITS["max_pixels"],
            )
            master = decode_master(
                data,
                source_name=source["path"],
                input_color=input_color,
                max_pixels=NATIVE_MATERIALS_LIMITS["max_pixels"],
            )
            if master.shape != shape or master.source_bit_depth != bits or (master.alpha is not None) != alpha:
                raise ValueError("Native Materials decoded samples differ from admitted native headers")
            _bounded_master(master)
            record["master_sha256"] = master.content_hash()
            sources.append(record)
            del data, master
    records, evidence_root, manifest = freeze_materials(Path(manifest_path), sources, NATIVE_MATERIALS_LIMITS)
    for source in sources:
        bound_evidence = records.get(source["path"])
        source["evidence"] = bound_evidence
        if bound_evidence is not None:
            evidence = load_bound_evidence(evidence_root, bound_evidence, NATIVE_MATERIALS_LIMITS)
            source["mask_bytes"] = sum(region.mask.nbytes for region in evidence.regions)
            del evidence
        _admit_allocation(source, NATIVE_MATERIALS_LIMITS, policy)
    binding = {
        "schema": NATIVE_MATERIALS_SCHEMA,
        "recipe": NATIVE_MATERIALS_RECIPE,
        "input_color": input_color,
        "evidence_root": str(evidence_root),
        "manifest": manifest,
        "policy": policy.to_payload(),
        "operations_sha256": operation_contract_hash(),
        "limits": dict(NATIVE_MATERIALS_LIMITS),
        "sources": sources,
    }
    binding["binding_sha256"] = native_materials_binding_hash(binding)
    binding = _binding_snapshot(binding)
    revalidate_native_materials(binding, input_root=root)
    return binding


def revalidate_native_materials(
    binding: Mapping[str, Any], input_root: Path | None = None, *, source_path: str | None = None
) -> None:
    """Rebind all sources at admission, or one selected source before its work.

    Every call validates the complete closed binding and manifest source map.
    Per-image checks decode only that image's evidence; initial whole-batch
    admission never substitutes for revalidating the selected numeric bytes.
    """
    binding = _binding_snapshot(binding)
    sources = binding["sources"]
    if source_path is not None:
        selected_path = _portable_path(source_path)
        sources = [source for source in sources if source["path"] == selected_path]
        if len(sources) != 1:
            raise ValueError("Native Materials revalidation requires exactly one frozen source path")
    root = _manifest_unchanged(binding)
    input_directory = directory_path(input_root) if input_root is not None else None
    for source in sources:
        if input_directory is not None:
            _, observed = snapshot(
                input_directory,
                input_directory / source["path"],
                maximum_bytes=source["size_bytes"],
                retain_bytes=False,
            )
            if observed != {key: source[key] for key in ("path", "sha256", "size_bytes")}:
                raise ValueError("Native Materials source changed after preparation")
        if source["evidence"] is not None:
            evidence = load_bound_evidence(root, source["evidence"], binding["limits"])
            if sum(region.mask.nbytes for region in evidence.regions) != source["mask_bytes"]:
                raise ValueError("Native Materials decoded mask allocation differs from its frozen binding")
            del evidence
    _manifest_unchanged(binding)


def _bounded_master(master: ImageMaster) -> None:
    if not isinstance(master, ImageMaster) or master.source_bit_depth not in {8, 16}:
        raise ValueError("Native Materials requires a native 8/16-bit ImageMaster")
    if np.any(master.pixels < 0) or np.any(master.pixels > 1):
        raise ValueError("Native Materials requires bounded SDR linear-sRGB samples")


def _binding_snapshot(binding: Mapping[str, Any]) -> dict[str, Any]:
    """Detach caller-owned containers before they can change an admitted operation."""
    validate_native_materials(binding)
    detached = decode_bounded_json_object(canonicalize_json(dict(binding)))
    validate_native_materials(detached)
    return detached


def selected_native_materials_source(binding: Mapping[str, Any], *, source_path: str | None = None) -> dict[str, Any]:
    """Select detached frozen source authority without opening or decoding files."""
    frozen = _binding_snapshot(binding)
    sources = frozen["sources"]
    if source_path is not None:
        selected_path = _portable_path(source_path)
        sources = [source for source in sources if source["path"] == selected_path]
    if len(sources) != 1:
        raise ValueError("Native Materials requires exactly one frozen source; select an explicit original relative path")
    return sources[0]


def snapshot_native_materials_source(
    path: Path, binding: Mapping[str, Any], *, source_path: str | None = None
) -> tuple[bytes, dict[str, Any]]:
    """Verify bounded source bytes before any photographic pixel allocation.

    The caller must decode these same immutable bytes under the selected native
    pixel bound. A changed file never acquires allocation authority merely by
    retaining a familiar path, byte count or photographic header.
    """
    source = selected_native_materials_source(binding, source_path=source_path)
    try:
        data = snapshot_geometry_artifact(Path(path), source["size_bytes"])
    except (OSError, ValueError) as exc:
        raise ValueError("Native Materials source snapshot is not its bounded frozen regular file") from exc
    if len(data) != source["size_bytes"] or hashlib.sha256(data).hexdigest() != source["sha256"]:
        raise ValueError("Native Materials source bytes changed from the frozen selection")
    return data, source


def apply_native_materials(
    master: ImageMaster,
    binding: Mapping[str, Any],
    source_path: str | Path | None = None,
    support: np.ndarray | None = None,
) -> tuple[ImageMaster, dict[str, Any]]:
    """Apply native evidence without promotion, proxy edits or carrier quantization."""
    binding = _binding_snapshot(binding)
    _bounded_master(master)
    master_sha256 = master.content_hash()
    selected_path = None if source_path is None else _portable_path(str(source_path))
    selected = [
        source
        for source in binding["sources"]
        if source["sha256"] == master.source_sha256
        and tuple(source["shape"]) == master.shape
        and source["source_bit_depth"] == master.source_bit_depth
        and source["alpha"] == (master.alpha is not None)
        and source["master_sha256"] == master_sha256
        and (selected_path is None or source["path"] == selected_path)
    ]
    if len(selected) != 1:
        raise ValueError("Native Materials requires exactly one matching frozen source/frame")
    source = selected[0]
    policy = ResponsePolicy.from_payload(binding["policy"])
    _admit_allocation(source, binding["limits"], policy)
    if not isinstance(support, np.ndarray) or support.dtype != np.bool_ or support.shape != master.shape:
        raise ValueError("Native Materials requires exact native boolean geometry support")
    support = immutable_array(support, np.bool_)
    root = _manifest_unchanged(binding)
    evidence = (
        load_bound_evidence(root, source["evidence"], binding["limits"])
        if source["evidence"] is not None
        else MaterialEvidence(master.source_sha256, master.shape, (), status="unavailable", reason="no_evidence_for_source")
    )
    if sum(region.mask.nbytes for region in evidence.regions) != source["mask_bytes"]:
        raise ValueError("Native Materials decoded mask allocation differs from its frozen binding")
    original_evidence_sha256 = evidence.content_hash()
    evidence = replace(
        evidence,
        regions=tuple(replace(region, mask=np.where(support, region.mask, np.float32(0))) for region in evidence.regions),
    )
    plan = plan_response(master, evidence, policy)
    result, execution = apply_response(master, evidence, plan)
    _bounded_master(result)
    _manifest_unchanged(binding)
    receipt = {
        "schema": "tp.lux.native_materials.execution.v1",
        "recipe": NATIVE_MATERIALS_RECIPE,
        "binding_sha256": binding["binding_sha256"],
        "selected_source_path": source["path"],
        "source_sha256": master.source_sha256,
        "input_master_sha256": master_sha256,
        "output_master_sha256": result.content_hash(),
        "support_sha256": hashlib.sha256(support.tobytes(order="C")).hexdigest(),
        "source_evidence_sha256": original_evidence_sha256,
        "effective_evidence_sha256": evidence.content_hash(),
        "working_color": "linear_srgb",
        "working_precision": "float32",
        "source_bit_depth": master.source_bit_depth,
        "alpha_mode": "straight" if master.alpha is not None else None,
        "precision_recovery": False,
        "allocation_upper_bound_bytes": _allocation_bytes(source, policy.max_working_bytes),
        "execution": execution,
    }
    return result, receipt
