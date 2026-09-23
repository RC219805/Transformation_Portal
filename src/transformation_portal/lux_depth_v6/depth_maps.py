"""Replayable depth derivatives from retained, independently verified V5 evidence.

Native model samples remain unchanged, including invalid numeric sentinels.
Reconstruction changes the sampling grid, not the model's inference resolution
or its physical accuracy. The caller must verify the retained V5 source first.
"""

from __future__ import annotations

import hashlib
import io
import re
from dataclasses import dataclass
from typing import Any, Iterator, Mapping

import numpy as np
from PIL import Image

from transformation_portal.core.depth_evidence import DepthEvidence
from transformation_portal.core.image_artifact import ImageMaster, ImageProxy, artifact_content_hash, metadata_payload
from transformation_portal.ingest.canonical_json import canonicalize_json
from transformation_portal.lux_depth_v5.photography import AlignedDepth, align_depth

DEPTH_MAP_SCHEMA = "tp.lux.depth_maps.v1"
MAX_PREVIEW_EDGE = 1600
MAX_PREVIEW_BYTES = 8 * 1024**2
_REFINEMENTS = frozenset({"bilinear", "guided_bilinear_v3", "guided_bilinear_v4"})
_RECEIPT_SCHEMA = "tp.lux.depth_map_reconstruction.v1"


@dataclass(frozen=True)
class DepthMapRecipe:
    """Explicit successor reconstruction; existing V5 recipes are unchanged."""

    refinement: str = "guided_bilinear_v4"

    def __post_init__(self) -> None:
        if type(self.refinement) is not str or self.refinement not in _REFINEMENTS:
            raise ValueError("Unsupported V6 depth map refinement")

    def to_payload(self) -> dict[str, str]:
        return {"schema": DEPTH_MAP_SCHEMA, "refinement": self.refinement}

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> DepthMapRecipe:
        if (
            not isinstance(payload, Mapping)
            or set(payload) != {"schema", "refinement"}
            or payload["schema"] != DEPTH_MAP_SCHEMA
        ):
            raise ValueError("Unsupported closed V6 depth map recipe")
        return cls(payload["refinement"])


def reconstruct_depth(
    original: ImageMaster, evidence: DepthEvidence, proxy: ImageProxy, recipe: DepthMapRecipe
) -> tuple[AlignedDepth, dict[str, Any]]:
    """Bind original/proxy/native evidence, then reconstruct or explicitly abstain."""
    if (
        type(original) is not ImageMaster
        or type(evidence) is not DepthEvidence
        or type(proxy) is not ImageProxy
        or type(recipe) is not DepthMapRecipe
    ):
        raise ValueError("Depth reconstruction requires exact typed source, evidence, proxy, and recipe")
    master_hash = original.content_hash()
    proxy_hash = artifact_content_hash(
        {"transform": proxy.transform.to_payload(), "master_content_hash": proxy.master_content_hash},
        {"pixels": proxy.pixels},
    )
    if (
        evidence.source_sha256 != original.source_sha256
        or evidence.transform != proxy.transform
        or proxy.transform.original_shape != original.shape
        or proxy.master_content_hash != master_hash
        or evidence.metadata.get("proxy_master_content_hash") != master_hash
        or evidence.metadata.get("proxy_content_hash") != proxy_hash
    ):
        raise ValueError("Depth reconstruction inputs must bind the same source, master, and proxy geometry")
    evidence_hash = evidence.content_hash()
    abstention = original.alpha is not None and bool(np.any(original.alpha != 1))
    if abstention:
        aligned = AlignedDepth(
            relative_depth=np.zeros(original.shape, np.float32),
            valid_mask=np.zeros(original.shape, bool),
            support_mask=np.ones(original.shape, bool),
            support_confidence=np.zeros(original.shape, np.float32),
            source_sha256=original.source_sha256,
            evidence_content_hash=evidence_hash,
            master_content_hash=master_hash,
            metric_map_m=None if evidence.metric_map_m is None else np.zeros(original.shape, np.float32),
            metadata={
                "recipe": "alpha_abstention_v1",
                "refinement": recipe.refinement,
                "geometry": proxy.transform.to_payload(),
                "native_preserved": True,
                "reason": "alpha_unaware_upstream_proxy",
            },
        )
    else:
        aligned = align_depth(evidence, original, proxy, refinement=recipe.refinement)
    receipt = {
        "schema": _RECEIPT_SCHEMA,
        "recipe": recipe.to_payload(),
        "source_sha256": original.source_sha256,
        "original_content_sha256": master_hash,
        "depth_evidence_content_sha256": evidence_hash,
        "proxy_content_sha256": proxy_hash,
        "aligned_content_sha256": aligned.content_hash(),
        "alpha_abstention": abstention,
        "reason": "alpha_unaware_upstream_proxy" if abstention else None,
        "quality_acceptance": "unestablished",
    }
    return aligned, receipt


def _validate_bindings(evidence: DepthEvidence, aligned: AlignedDepth, receipt: Mapping[str, Any]) -> None:
    if type(evidence) is not DepthEvidence or type(aligned) is not AlignedDepth:
        raise ValueError("Depth products require exact typed native and aligned evidence")
    keys = {
        "schema",
        "recipe",
        "source_sha256",
        "original_content_sha256",
        "depth_evidence_content_sha256",
        "proxy_content_sha256",
        "aligned_content_sha256",
        "alpha_abstention",
        "reason",
        "quality_acceptance",
    }
    if not isinstance(receipt, Mapping) or set(receipt) != keys or receipt["schema"] != _RECEIPT_SCHEMA:
        raise ValueError("Depth products require a closed reconstruction receipt")
    recipe = DepthMapRecipe.from_payload(receipt["recipe"])
    if (
        aligned.source_sha256 != evidence.source_sha256
        or aligned.shape != evidence.transform.original_shape
        or aligned.evidence_content_hash != evidence.content_hash()
        or aligned.master_content_hash != evidence.metadata.get("proxy_master_content_hash")
        or aligned.metadata.get("refinement") != recipe.refinement
        or receipt["source_sha256"] != evidence.source_sha256
        or receipt["original_content_sha256"] != aligned.master_content_hash
        or receipt["depth_evidence_content_sha256"] != evidence.content_hash()
        or receipt["proxy_content_sha256"] != evidence.metadata.get("proxy_content_hash")
        or receipt["aligned_content_sha256"] != aligned.content_hash()
        or (aligned.metric_map_m is None) != (evidence.metric_map_m is None)
        or receipt["quality_acceptance"] != "unestablished"
        or type(receipt["alpha_abstention"]) is not bool
    ):
        raise ValueError("Depth products differ from their bound source or reconstruction")
    if receipt["alpha_abstention"]:
        if receipt["reason"] != "alpha_unaware_upstream_proxy" or aligned.valid_mask.any():
            raise ValueError("Alpha abstention must prohibit all reconstructed surface use")
    elif receipt["reason"] is not None:
        raise ValueError("Unexpected depth reconstruction abstention reason")
    if evidence.sky_mask is None and aligned.valid_mask.any():
        raise ValueError("Unknown native sky cannot authorize reconstructed surfaces")


def _preview(aligned: AlignedDepth) -> tuple[bytes, bytes, dict[str, Any]]:
    """Sample depth and validity together without inventing interpolation support."""
    height, width = aligned.shape
    longest = max(height, width)
    if longest > MAX_PREVIEW_EDGE:
        shape = tuple(max(1, (size * MAX_PREVIEW_EDGE + longest // 2) // longest) for size in (height, width))
    else:
        shape = (height, width)
    rows = ((2 * np.arange(shape[0]) + 1) * height) // (2 * shape[0])
    columns = ((2 * np.arange(shape[1]) + 1) * width) // (2 * shape[1])
    samples = aligned.relative_depth[rows[:, None], columns[None, :]]
    valid = aligned.valid_mask[rows[:, None], columns[None, :]]
    encoded = np.rint(samples.astype(np.float64) * 65535).astype(np.uint16)
    encoded[~valid] = 0
    preview, validity = io.BytesIO(), io.BytesIO()
    Image.fromarray(encoded).save(preview, format="PNG", compress_level=6, optimize=False)
    Image.fromarray(valid.astype(np.uint8) * 255).save(validity, format="PNG", compress_level=6, optimize=False)
    if preview.tell() + validity.tell() > MAX_PREVIEW_BYTES:
        raise ValueError("Depth previews exceed their combined encoded-byte budget")
    return (
        preview.getvalue(),
        validity.getvalue(),
        {
            "shape": list(shape),
            "maximum_edge": MAX_PREVIEW_EDGE,
            "sampling": "nearest_pixel_center_integer_floor_v1",
            "sampling_formula": "source_index=floor((2*target_index+1)*source_size/(2*target_size))",
            "upsampling": False,
            "bit_depth": 16,
            "color_type": "grayscale",
            "transfer": "linear_scalar_no_color_profile",
            "units": "normalized_relative_not_meters",
            "invalid_value": 0,
            "validity_encoding": "grayscale_uint8_255_valid_0_invalid",
            "quantization": "round_to_nearest_even(relative_depth*65535)",
            "accuracy_claim": False,
        },
    )


def depth_map_products(
    evidence: DepthEvidence, aligned: AlignedDepth, receipt: Mapping[str, Any], input_id: str
) -> Iterator[tuple[str, bytes]]:
    """Encode frozen native evidence and explicit master-grid scalar derivatives.

    At most 13 products are emitted, including the optional calibrated map and
    the bounded preview validity mask. Float arrays and the float TIFF retain
    precision; the 16-bit PNG is a visualization and never metric authority.
    """
    _validate_bindings(evidence, aligned, receipt)
    if type(input_id) is not str or re.fullmatch(r"[a-z0-9_-]{1,128}", input_id) is None:
        raise ValueError("Depth products require a bounded portable input identifier")
    records: dict[str, dict[str, Any]] = {}

    def record(filename: str, data: bytes, **details: Any) -> tuple[str, bytes]:
        path = f"{input_id}/{filename}"
        records[filename] = {"path": path, "sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data), **details}
        return path, data

    arrays = [
        ("native-depth.npy", evidence.native_depth, "native_api_output_unchanged"),
        ("native-numeric-valid.npy", evidence.numeric_valid, "finite_positive_native_samples"),
        ("native-support.npy", evidence.support_mask, "unpadded_image_support"),
        (
            "native-sky.npy",
            np.zeros(evidence.shape, bool) if evidence.sky_mask is None else evidence.sky_mask,
            "unknown_placeholder_not_non_sky_evidence" if evidence.sky_mask is None else "model_sky_mask",
        ),
        ("relative-depth.npy", aligned.relative_depth, "master_grid_near_0_far_1"),
        ("depth-valid.npy", aligned.valid_mask, "usable_surface_validity"),
        ("depth-support.npy", aligned.support_mask, "master_image_support"),
        ("depth-support-score.npy", aligned.support_confidence, "interpolation_support_not_accuracy_probability"),
    ]
    if aligned.metric_map_m is not None:
        arrays.append(("metric-depth-m.npy", aligned.metric_map_m, "inferred_with_supplied_camera_calibration"))
    for filename, array, semantics in arrays:
        buffer = io.BytesIO()
        np.save(buffer, array, allow_pickle=False)
        yield record(filename, buffer.getvalue(), shape=list(array.shape), dtype=array.dtype.name, semantics=semantics)

    import tifffile

    buffer = io.BytesIO()
    tifffile.imwrite(
        buffer,
        aligned.relative_depth,
        photometric="minisblack",
        metadata=None,
        description="V6 normalized relative depth; near=0, far=1; depth-valid.npy defines usable samples; not meters.",
        extratags=[(274, "H", 1, 1, False)],
    )
    yield record("depth-relative.tif", buffer.getvalue(), shape=list(aligned.shape), dtype="float32", units="relative")
    preview, validity, preview_descriptor = _preview(aligned)
    yield record("depth-preview.png", preview, shape=preview_descriptor["shape"], bit_depth=16)
    yield record("depth-preview-valid.png", validity, shape=preview_descriptor["shape"], bit_depth=8)
    descriptor = {
        "schema": "tp.lux.depth_map_products.v1",
        "input_id": input_id,
        "source_sha256": evidence.source_sha256,
        "depth_evidence_content_sha256": evidence.content_hash(),
        "aligned_content_sha256": aligned.content_hash(),
        "original_content_sha256": aligned.master_content_hash,
        "reconstruction": dict(receipt),
        "dimensions": {
            "native_padded": list(evidence.shape),
            "native_unpadded": list(evidence.transform.resized_shape),
            "reconstructed_master": list(aligned.shape),
        },
        "geometry": evidence.transform.to_payload(),
        "native_evidence": evidence.to_payload(),
        "aligned_evidence": aligned.to_payload(),
        "sky_status": "unavailable" if evidence.sky_mask is None else "model_mask",
        "surface_status": "available" if aligned.valid_mask.any() else "unavailable",
        "relative_semantics": "near_0_far_1",
        "invalid_value": 0,
        "validity_required": "zero_may_be_valid_near_depth_consult_depth-valid.npy",
        "metric": {
            "status": "unavailable" if aligned.metric_map_m is None else "inferred_with_supplied_camera_calibration",
            "usable_pixels": 0 if aligned.metric_map_m is None else int(aligned.valid_mask.sum()),
            "calibration": metadata_payload(evidence.calibration),
            "measured_scene_accuracy": "not_established",
        },
        "preview": {
            **preview_descriptor,
            "path": f"{input_id}/depth-preview.png",
            "validity_path": f"{input_id}/depth-preview-valid.png",
        },
        "artifacts": records,
        "resolution_authority": "reconstructed_grid_not_new_model_inference_or_recovered_native_detail",
        "production_acceptance": "not_established",
    }
    yield f"{input_id}/depth.json", canonicalize_json(descriptor)
