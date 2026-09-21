"""Independently rederive V5 depth and response before fenced publication.

This proves internal semantic consistency and retained output integrity, not
measured scene accuracy or independent authentication of source-camera pixels.
"""

from __future__ import annotations

import hashlib
import io
import math
from pathlib import Path
from typing import Any

import numpy as np

from transformation_portal.core.depth_evidence import build_depth_evidence
from transformation_portal.core.execution_plan import decode_bounded_json_object
from transformation_portal.core.execution_plan_v4 import parse_plan_v4
from transformation_portal.core.image_artifact import ImageMaster
from transformation_portal.lux_depth_v4.evidence import (
    MAX_EVIDENCE_BYTES,
    VerifiedExecutionEvidenceV2,
    _verify_execution_evidence,
    _verify_materials_receipt,
)
from transformation_portal.lux_depth_v4.io import snapshot
from transformation_portal.lux_depth_v4.photography import create_proxy, linear_to_srgb, output_srgb_icc

from .photography import align_depth, enhance_master_v5, generate_preview_maps_v5
from .preview import MAX_PREVIEW_BYTES, preview_descriptor, preview_samples, preview_shape

MAX_ICC_BYTES = 16 * 1024 * 1024


def _array(
    root: Path,
    relative: str,
    shape: tuple[int, ...] | None,
    dtype: np.dtype,
    declared: dict,
    *,
    allow_nonfinite: bool = False,
) -> np.ndarray:
    record = declared.get(relative)
    if not isinstance(record, dict) or record.get("kind") != "array":
        raise ValueError(f"V5 completion omits required array: {relative}")
    maximum = MAX_ICC_BYTES + 4096 if shape is None else math.prod(shape) * dtype.itemsize + 4096
    if record["size_bytes"] > maximum:
        raise ValueError("V5 array exceeds its geometry-derived byte bound")
    raw, observed = snapshot(root, root / relative, maximum_bytes=maximum)
    if any(observed[key] != record[key] for key in ("path", "size_bytes", "sha256")):
        raise ValueError("V5 array changed after inventory verification")
    stream = io.BytesIO(raw)
    try:
        version = np.lib.format.read_magic(stream)
        if version not in {(1, 0), (2, 0)}:
            raise ValueError("V5 evidence requires NPY v1/v2")
        length_bytes = 2 if version == (1, 0) else 4
        encoded = stream.read(length_bytes)
        header_size = int.from_bytes(encoded, "little")
        if len(encoded) != length_bytes or not 0 < header_size <= 4096 or stream.tell() + header_size > len(raw):
            raise ValueError("V5 array header exceeds byte bound or is truncated")
        stream.seek(8)
        reader = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
        actual_shape, fortran, actual_dtype = reader(stream, max_header_size=4096)
    except (ValueError, TypeError, EOFError, OverflowError) as exc:
        raise ValueError("Invalid bounded V5 array header") from exc
    if shape is None:
        if len(actual_shape) != 1 or not 1 <= actual_shape[0] <= MAX_ICC_BYTES:
            raise ValueError("Source ICC must be a bounded one-dimensional byte array")
        shape = actual_shape
    if actual_shape != shape or actual_dtype != dtype or fortran:
        raise ValueError("V5 array geometry, dtype, or layout differs from its semantic contract")
    if len(raw) - stream.tell() != math.prod(shape) * dtype.itemsize:
        raise ValueError("V5 array payload size differs from declared geometry")
    array = np.frombuffer(raw, dtype=dtype, offset=stream.tell()).reshape(shape)
    if not allow_nonfinite and not np.isfinite(array).all():
        raise ValueError("V5 derivative array contains nonfinite samples")
    return array


def _same(actual: np.ndarray, expected: np.ndarray, name: str) -> None:
    if (
        actual.shape != expected.shape
        or actual.dtype != expected.dtype
        or not np.array_equal(actual.view(np.uint8), expected.view(np.uint8))
    ):
        raise ValueError(f"V5 {name} differs from independently reconstructed evidence")


def _image(payload: Any, pixels: np.ndarray, source: dict, alpha: np.ndarray | None, icc: bytes | None) -> ImageMaster:
    if not isinstance(payload, dict) or payload.get("source_sha256") != source["sha256"]:
        raise ValueError("V5 photographic master has invalid source binding")
    if type(payload.get("source_bit_depth")) is not int or not isinstance(payload.get("metadata"), dict):
        raise ValueError("V5 photographic master requires typed precision and metadata")
    master = ImageMaster(pixels, source["sha256"], payload["source_bit_depth"], alpha, payload["metadata"], icc)
    if master.to_payload() != payload:
        raise ValueError("V5 master descriptor differs from its reconstructed photographic contract")
    return master


def _verify_delivery(root: Path, relative: str, master: ImageMaster, descriptor: dict, declared: dict) -> None:
    import tifffile

    record = declared.get(relative)
    if not isinstance(record, dict) or record.get("kind") != "image":
        raise ValueError("V5 completion omits photographic delivery")
    raw, observed = snapshot(root, root / relative, maximum_bytes=math.prod(master.shape) * 8 + 1024 * 1024)
    if any(observed[key] != record[key] for key in ("path", "size_bytes", "sha256")):
        raise ValueError("V5 delivery changed after inventory verification")
    encoded = linear_to_srgb(master.pixels)
    samples = np.rint(np.clip(encoded, 0, 1) * 65535).astype(np.uint16)
    if master.alpha is not None:
        samples = np.concatenate([samples, np.rint(master.alpha * 65535).astype(np.uint16)[..., None]], axis=2)
    with tifffile.TiffFile(io.BytesIO(raw)) as image:
        if len(image.pages) != 1 or len(image.series) != 1:
            raise ValueError("V5 delivery requires one photographic TIFF image")
        page = image.pages[0]
        if not isinstance(page, tifffile.TiffPage) or tuple(page.shape) != samples.shape or page.dtype != np.dtype("uint16"):
            raise ValueError("V5 delivery geometry or sample precision differs from master")
        profile, orientation = page.tags.get(34675), page.tags.get(274)
        if (
            page.photometric != tifffile.PHOTOMETRIC.RGB
            or page.planarconfig != 1
            or page.bitspersample != 16
            or profile is None
            or bytes(profile.value) != output_srgb_icc()
            or orientation is None
            or orientation.value != 1
            or tuple(int(value) for value in page.extrasamples) != ((2,) if master.alpha is not None else ())
        ):
            raise ValueError("V5 delivery color, orientation, alpha, or precision contract differs")
        _same(page.asarray(), samples, "delivery pixels")
    expected = {
        "path": relative,
        "output_bit_depth": 16,
        "source_bit_depth": master.source_bit_depth,
        "color_space": "srgb",
        "alpha_preserved": master.alpha is not None,
        "clipped_low_sample_fraction": float(np.mean(encoded < 0)),
        "clipped_high_sample_fraction": float(np.mean(encoded > 1)),
        "master_content_hash": master.content_hash(),
        "output_icc_sha256": hashlib.sha256(output_srgb_icc()).hexdigest(),
    }
    if descriptor != expected:
        raise ValueError("V5 delivery descriptor differs from independently measured encoding")


def _verify_photograph(root: Path, input_id: str, source: dict, payload: dict, declared: dict) -> None:
    relative = f"{input_id}/photograph.json"
    if declared.get(relative, {}).get("kind") != "descriptor":
        raise ValueError("V5 completion omits photographic descriptor")
    raw, observed = snapshot(root, root / relative, maximum_bytes=MAX_EVIDENCE_BYTES)
    if any(observed[key] != declared[relative][key] for key in ("path", "size_bytes", "sha256")):
        raise ValueError("V5 photograph descriptor changed after inventory verification")
    descriptor = decode_bounded_json_object(raw)
    keys = {
        "schema",
        "input_id",
        "source",
        "master",
        "depth",
        "depth_content_sha256",
        "aligned_depth",
        "delivery",
        "materials",
        "preview_maps",
        "depth_baseline",
        "depth_response",
    }
    materials = "materials_v4" in payload["configuration"]
    browser_preview = "browser_preview" in payload["configuration"]
    if materials:
        keys.add("materials_baseline")
    if browser_preview:
        keys.add("browser_preview")
    schema = "tp.lux.photograph.v3" if browser_preview else "tp.lux.photograph.v2"
    if set(descriptor) != keys or descriptor["schema"] != schema or descriptor["input_id"] != input_id:
        raise ValueError("V5 photograph descriptor has an invalid closed schema")
    shape_value = descriptor["source"].get("shape") if isinstance(descriptor["source"], dict) else None
    if (
        not isinstance(shape_value, list)
        or len(shape_value) != 2
        or any(type(size) is not int or size <= 0 for size in shape_value)
    ):
        raise ValueError("V5 master geometry is invalid")
    shape = tuple(shape_value)
    if math.prod(shape) > payload["resources"]["max_pixels"]:
        raise ValueError("V5 master exceeds the admitted pixel budget")
    target = payload["configuration"]["target_size"]
    proxy_edge = ((target + 13) // 14) * 14
    if math.prod(shape) * 128 + proxy_edge**2 * 80 + MAX_ICC_BYTES > payload["resources"]["memory_mib"] * 1024**2:
        raise ValueError("V5 semantic verification exceeds the admitted memory budget")
    expected_files = {
        f"{input_id}/{name}"
        for name in (
            "source-master.npy",
            "master.npy",
            "native-depth.npy",
            "depth-valid.npy",
            "native-numeric-valid.npy",
            "native-support.npy",
            "native-sky.npy",
            "relative-depth.npy",
            "aligned-depth-valid.npy",
            "aligned-depth-support.npy",
            "depth-support-score.npy",
            "depth-baseline.npy",
            "delivery.tif",
            "photograph.json",
        )
    }

    def load(
        name: str, dimensions: tuple[int, ...], dtype: np.dtype = np.dtype("float32"), *, nonfinite: bool = False
    ) -> np.ndarray:
        return _array(root, f"{input_id}/{name}", dimensions, dtype, declared, allow_nonfinite=nonfinite)

    alpha = None
    if descriptor["source"].get("alpha_mode") is not None:
        expected_files.add(f"{input_id}/alpha.npy")
        alpha = load("alpha.npy", shape)
    icc = None
    if descriptor["source"].get("source_icc_sha256") is not None:
        expected_files.add(f"{input_id}/source-icc.npy")
        icc = _array(root, f"{input_id}/source-icc.npy", None, np.dtype("uint8"), declared).tobytes()
    original = _image(descriptor["source"], load("source-master.npy", (*shape, 3)), source, alpha, icc)
    proxy = create_proxy(original, target)
    native_shape = proxy.transform.padded_shape
    native = load("native-depth.npy", native_shape, nonfinite=True)
    sky_carrier = load("native-sky.npy", native_shape, np.dtype("bool"))
    depth_descriptor = descriptor["depth"]
    if not isinstance(depth_descriptor, dict) or depth_descriptor.get("sky_status") not in {"model_mask", "unavailable"}:
        raise ValueError("V5 native sky status is invalid")
    sky = sky_carrier if depth_descriptor["sky_status"] == "model_mask" else None
    if sky is None and sky_carrier.any():
        raise ValueError("Unavailable sky evidence must use the declared empty carrier")
    companion = source.get("companions")
    if companion is not None and "materials" in companion:
        raise ValueError("V5 completion requires MaterialsV4 evidence instead of legacy companion materials")
    depth = build_depth_evidence(
        native, sky, proxy, source["sha256"], companion=companion, precision=payload["configuration"]["depth"]["precision"]
    )
    if depth.to_payload() != depth_descriptor or depth.content_hash() != descriptor["depth_content_sha256"]:
        raise ValueError("V5 native depth descriptor or identity differs from reconstructed semantics")
    for name, expected in (
        ("native-numeric-valid.npy", depth.numeric_valid),
        ("native-support.npy", depth.support_mask),
        ("depth-valid.npy", depth.valid_mask),
    ):
        _same(load(name, native_shape, np.dtype("bool")), expected, name)
    aligned = align_depth(depth, original, proxy, refinement=payload["configuration"]["depth"]["refinement"])
    for name, expected in (
        ("relative-depth.npy", aligned.relative_depth),
        ("aligned-depth-valid.npy", aligned.valid_mask),
        ("aligned-depth-support.npy", aligned.support_mask),
        ("depth-support-score.npy", aligned.support_confidence),
    ):
        _same(load(name, shape, expected.dtype), expected, name)
    if depth.metric_map_m is not None:
        if aligned.metric_map_m is None:
            raise ValueError("V5 aligned metric evidence is missing")
        expected_files.update(f"{input_id}/{name}" for name in ("metric-depth-m.npy", "aligned-metric-depth-m.npy"))
        _same(load("metric-depth-m.npy", native_shape), depth.metric_map_m, "metric native derivative")
        _same(load("aligned-metric-depth-m.npy", shape), aligned.metric_map_m, "aligned metric derivative")
    alignment_descriptor = {
        "evidence": aligned.to_payload(),
        "content_sha256": aligned.content_hash(),
        "relative_path": f"{input_id}/relative-depth.npy",
        "validity_path": f"{input_id}/aligned-depth-valid.npy",
        "support_path": f"{input_id}/aligned-depth-support.npy",
        "support_score_path": f"{input_id}/depth-support-score.npy",
        "metric_path": f"{input_id}/aligned-metric-depth-m.npy" if aligned.metric_map_m is not None else None,
    }
    if descriptor["aligned_depth"] != alignment_descriptor:
        raise ValueError("V5 aligned depth descriptor differs from reconstructed semantics")
    baseline, response = enhance_master_v5(
        original, aligned, strength=payload["configuration"]["strength"], clarity=payload["configuration"]["clarity"]
    )
    _same(load("depth-baseline.npy", (*shape, 3)), baseline.pixels, "depth response baseline")
    if descriptor["depth_baseline"] != baseline.to_payload() or descriptor["depth_response"] != response:
        raise ValueError("V5 depth response receipt differs from independent reconstruction")
    final = _image(descriptor["master"], load("master.npy", (*shape, 3)), source, alpha, icc)
    if final.to_payload() != baseline.to_payload():
        raise ValueError("V5 finishing must preserve the source/response photographic metadata")
    if materials:
        expected_files.add(f"{input_id}/materials-baseline.npy")
        _same(load("materials-baseline.npy", (*shape, 3)), baseline.pixels, "Materials baseline")
        _verify_materials_receipt(root, input_id, descriptor, source, payload, declared)
    else:
        _same(final.pixels, baseline.pixels, "final master without Materials")
        if descriptor["materials"] != {"status": "abstained", "reason": "no_authoritative_masks"}:
            raise ValueError("V5 absent Materials must retain its explicit abstention")
    if payload["configuration"]["preview_maps"]:
        maps, preview = generate_preview_maps_v5(depth)
        for name, expected in maps.items():
            filename = f"preview-{name}.npy"
            expected_files.add(f"{input_id}/{filename}")
            _same(load(filename, expected.shape, np.dtype("uint8")), expected, filename)
    else:
        preview = {"status": "disabled", "classification": "depth_derived_preview"}
    if descriptor["preview_maps"] != preview:
        raise ValueError("V5 preview receipt differs from its declared recipe")
    if browser_preview:
        expected_files.add(f"{input_id}/preview.png")
        _verify_browser_preview(root, f"{input_id}/preview.png", final, descriptor["browser_preview"], declared)
    if {name for name in declared if name.startswith(f"{input_id}/")} != expected_files:
        raise ValueError("V5 inventory differs from the exact admitted photographic products")
    _verify_delivery(root, f"{input_id}/delivery.tif", final, descriptor["delivery"], declared)


def _verify_browser_preview(root: Path, relative: str, master: ImageMaster, descriptor: dict, declared: dict) -> None:
    """Decode only a bounded admitted PNG and independently reconstruct its samples."""
    from PIL import Image

    record = declared.get(relative)
    if not isinstance(record, dict) or record.get("kind") != "image":
        raise ValueError("V5 completion omits photographic browser preview")
    if descriptor != preview_descriptor(master, relative_path=relative):
        raise ValueError("V5 browser preview receipt differs from the admitted photographic recipe")
    raw, observed = snapshot(root, root / relative, maximum_bytes=MAX_PREVIEW_BYTES)
    if any(observed[key] != record[key] for key in ("path", "size_bytes", "sha256")):
        raise ValueError("V5 browser preview changed after inventory verification")
    height, width = preview_shape(master.shape)
    # Pillow downconverts RGB/RGBA16 to uint8 and does not expose the encoded
    # bit depth through image.mode. Bind the actual IHDR recipe before decode.
    if (
        len(raw) < 33
        or raw[:16] != b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
        or raw[16:24] != width.to_bytes(4, "big") + height.to_bytes(4, "big")
        or raw[24:29] != bytes((8, 6 if master.alpha is not None else 2, 0, 0, 0))
    ):
        raise ValueError("V5 browser preview geometry, encoding, or color contract differs")
    try:
        with Image.open(io.BytesIO(raw)) as image:
            if (
                image.format != "PNG"
                or image.size != (width, height)
                or image.mode != ("RGBA" if master.alpha is not None else "RGB")
            ):
                raise ValueError("V5 browser preview geometry, encoding, or color contract differs")
            # PNG metadata can follow IDAT. Decode only after the geometry is
            # bounded, then validate the complete metadata rather than the
            # partial header state exposed by Image.open().
            image.load()
            if getattr(image, "n_frames", 1) != 1 or image.info != {"icc_profile": output_srgb_icc()}:
                raise ValueError("V5 browser preview geometry, encoding, or color contract differs")
            _same(np.asarray(image), preview_samples(master), "browser preview pixels")
    except (OSError, SyntaxError) as exc:
        raise ValueError("V5 browser preview is not a valid bounded PNG") from exc


class _EvidenceProfile:
    schema_file = "evidence.v3.schema.json"
    parse_plan = staticmethod(parse_plan_v4)
    verify_photograph = staticmethod(_verify_photograph)


def verify_execution_evidence_v3(output_root: Path, *, expected_plan_sha256: str) -> VerifiedExecutionEvidenceV2:
    """Rehash and independently reconstruct the complete V5 evidence contract."""
    try:
        return _verify_execution_evidence(output_root, expected_plan_sha256=expected_plan_sha256, profile=_EvidenceProfile)
    except (KeyError, TypeError, OverflowError, IndexError) as exc:
        raise ValueError("Malformed V5 completion semantics") from exc
