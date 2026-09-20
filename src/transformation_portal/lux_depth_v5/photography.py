"""Bounded depth alignment and photographic response for LuxDepthV5.

RGB may select between existing native surface depths only when a complete
native neighborhood supports two locally stable depth groups. Affine and
gradual ramps and ambiguous color matches retain the bilinear baseline. This
optional reconstruction does not prove physical discontinuities or recover
missing geometry.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from transformation_portal.core.depth_evidence import DepthEvidence
from transformation_portal.core.image_artifact import (
    ImageMaster,
    ImageProxy,
    artifact_content_hash,
    freeze_metadata,
    immutable_array,
    metadata_payload,
    validate_source_sha256,
)
from transformation_portal.lux_depth_v4.photography import (
    linear_to_srgb,
    restore_depth,
    restore_depth_with_validity,
    srgb_to_linear,
)

_ALIGNMENT_SCHEMA = "tp.depth.aligned.v1"
_MIN_DEPTH_SPAN = 0.10
_MIN_COLOR_SEPARATION = 0.08
_MAX_COLOR_DISTANCE = 0.12
_MIN_COLOR_MARGIN = 0.03
_NATIVE_CLUSTER_TOLERANCE = 0.20
_MAX_CHUNK_PIXELS = 65536


@dataclass(frozen=True)
class AlignedDepth:
    """Immutable master-grid derivatives; support score is not error probability."""

    relative_depth: np.ndarray
    valid_mask: np.ndarray
    support_mask: np.ndarray
    support_confidence: np.ndarray
    source_sha256: str
    evidence_content_hash: str
    master_content_hash: str
    metric_map_m: np.ndarray | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for digest in (self.source_sha256, self.evidence_content_hash, self.master_content_hash):
            validate_source_sha256(digest)
        relative = np.asarray(self.relative_depth)
        if relative.dtype != np.float32 or relative.ndim != 2 or min(relative.shape) <= 0:
            raise ValueError("Aligned relative depth must be nonempty float32 HW")
        for name in ("valid_mask", "support_mask"):
            mask = np.asarray(getattr(self, name))
            if mask.dtype != np.bool_ or mask.shape != relative.shape:
                raise ValueError("Aligned validity and support must be matching boolean rasters")
            object.__setattr__(self, name, immutable_array(mask, np.bool_))
        if np.any(self.valid_mask & ~self.support_mask):
            raise ValueError("Aligned usable samples cannot lie outside image support")
        for name in ("relative_depth", "support_confidence"):
            values = np.asarray(getattr(self, name))
            if values.dtype != np.float32 or values.shape != relative.shape or not np.isfinite(values).all():
                raise ValueError("Aligned derivatives require matching finite float32 rasters")
            if np.any((values < 0) | (values > 1)) or np.any(values[~self.valid_mask] != 0):
                raise ValueError("Aligned relative/support values must be bounded with zero invalid sentinels")
            object.__setattr__(self, name, immutable_array(values, np.float32))
        if self.metric_map_m is not None:
            metric = np.asarray(self.metric_map_m)
            if (
                metric.dtype != np.float32
                or metric.shape != relative.shape
                or not np.isfinite(metric).all()
                or np.any(metric[self.valid_mask] <= 0)
                or np.any(metric[~self.valid_mask] != 0)
            ):
                raise ValueError("Aligned metric depth requires positive valid values and zero invalid sentinels")
            object.__setattr__(self, "metric_map_m", immutable_array(metric, np.float32))
        if not isinstance(self.metadata, Mapping):
            raise ValueError("Aligned metadata must be a mapping")
        object.__setattr__(self, "metadata", freeze_metadata(self.metadata))

    @property
    def shape(self) -> tuple[int, int]:
        return self.relative_depth.shape

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema": _ALIGNMENT_SCHEMA,
            "shape": list(self.shape),
            "source_sha256": self.source_sha256,
            "evidence_content_hash": self.evidence_content_hash,
            "master_content_hash": self.master_content_hash,
            "relative_semantics": "near_0_far_1",
            "has_metric_depth": self.metric_map_m is not None,
            "support_score_semantics": "interpolation_valid_support_fraction_not_accuracy_probability",
            "usable_pixels": int(self.valid_mask.sum()),
            "invalid_value": 0,
            "metadata": metadata_payload(self.metadata),
        }

    def content_hash(self) -> str:
        return artifact_content_hash(
            self.to_payload(),
            {
                "relative_depth": self.relative_depth,
                "valid_mask": self.valid_mask,
                "support_mask": self.support_mask,
                "support_confidence": self.support_confidence,
                "metric_map_m": self.metric_map_m,
            },
        )


def _native_discontinuity_support(evidence: DepthEvidence, y0: np.ndarray, x0: np.ndarray) -> np.ndarray:
    """Require two stable groups in a complete 4x4 native neighborhood.

    A 2x2 footprint cannot distinguish a one-dimensional affine slope from two
    surfaces. Both groups must persist in the surrounding native samples. Use
    raw float64 values for this check so clipped percentile derivatives cannot
    disguise slopes or overflow the local range calculation.
    """
    height, width = evidence.transform.resized_shape
    usable = evidence.valid_mask
    native = evidence.native_depth
    central = np.stack(
        [
            native[np.clip(y0 + dy, 0, height - 1), np.clip(x0 + dx, 0, width - 1)]
            for dy, dx in ((0, 0), (0, 1), (1, 0), (1, 1))
        ]
    )
    central = np.where(np.isfinite(central), central, 0).astype(np.float64)
    low, high = central.min(axis=0), central.max(axis=0)
    tolerance = (high - low) * _NATIVE_CLUSTER_TOLERANCE
    supported = (y0 >= 1) & (y0 + 2 < height) & (x0 >= 1) & (x0 + 2 < width) & (high > low)
    for dy in (-1, 0, 1, 2):
        y = np.clip(y0 + dy, 0, height - 1)
        for dx in (-1, 0, 1, 2):
            x = np.clip(x0 + dx, 0, width - 1)
            values = native[y, x].astype(np.float64)
            supported &= usable[y, x]
            supported &= (np.abs(values - low) <= tolerance) | (np.abs(values - high) <= tolerance)
    return supported


def _guided_selection(
    evidence: DepthEvidence,
    master: ImageMaster,
    proxy: ImageProxy,
    relative: np.ndarray,
    valid: np.ndarray,
    metric: np.ndarray | None,
) -> dict[str, Any]:
    """Select existing surface samples only at supported two-cluster boundaries.

    Scratch allocations are bounded by a 65,536-pixel tile, independent of the
    master dimensions. The output and photographic master are the only full-grid
    carriers owned by this operation.
    """
    native = evidence.relative_depth()
    source_valid = evidence.valid_mask
    height, width = master.shape
    small_height, small_width = proxy.transform.resized_shape
    refined = candidates = ambiguous = 0
    tile_width = min(width, _MAX_CHUNK_PIXELS)
    tile_height = max(1, _MAX_CHUNK_PIXELS // tile_width)
    for top in range(0, height, tile_height):
        bottom = min(height, top + tile_height)
        ys = np.clip((np.arange(top, bottom) + 0.5) * small_height / height - 0.5, 0, small_height - 1)
        y0 = np.floor(ys).astype(np.intp)[:, None]
        y1 = np.minimum(y0 + 1, small_height - 1)
        for left in range(0, width, tile_width):
            right = min(width, left + tile_width)
            xs = np.clip((np.arange(left, right) + 0.5) * small_width / width - 0.5, 0, small_width - 1)
            x0 = np.floor(xs).astype(np.intp)[None, :]
            x1 = np.minimum(x0 + 1, small_width - 1)
            indices = ((y0, x0), (y0, x1), (y1, x0), (y1, x1))
            samples = np.stack([native[y, x] for y, x in indices])
            available = np.stack([source_valid[y, x] for y, x in indices]).all(axis=0)
            low, high = samples.min(axis=0), samples.max(axis=0)
            span = high - low
            lower = samples <= low + span * 0.20
            upper = samples >= high - span * 0.20
            edge = available & valid[top:bottom, left:right] & (span >= _MIN_DEPTH_SPAN) & (lower | upper).all(axis=0)
            if edge.any():
                edge &= _native_discontinuity_support(evidence, y0, x0)
            count = int(edge.sum())
            candidates += count
            if not count:
                continue
            colors = np.stack([proxy.pixels[y, x].astype(np.float32) / 255 for y, x in indices])
            guide = np.clip(linear_to_srgb(master.pixels[top:bottom, left:right]), 0, 1)
            costs = np.sqrt(np.mean((colors - guide) ** 2, axis=3))
            lower_cost = np.where(lower, costs, np.inf)
            upper_cost = np.where(upper, costs, np.inf)
            best_lower = lower_cost.argmin(axis=0)
            best_upper = upper_cost.argmin(axis=0)
            min_lower, min_upper = lower_cost.min(axis=0), upper_cost.min(axis=0)
            chosen = np.where(min_lower <= min_upper, best_lower, best_upper)
            low_color = np.take_along_axis(colors, best_lower[None, ..., None], axis=0)[0]
            high_color = np.take_along_axis(colors, best_upper[None, ..., None], axis=0)[0]
            separation = np.sqrt(np.mean((low_color - high_color) ** 2, axis=2))
            accepted = (
                edge
                & (separation >= _MIN_COLOR_SEPARATION)
                & (np.minimum(min_lower, min_upper) <= _MAX_COLOR_DISTANCE)
                & (np.abs(min_lower - min_upper) >= _MIN_COLOR_MARGIN)
            )
            selected = np.take_along_axis(samples, chosen[None], axis=0)[0]
            destination = relative[top:bottom, left:right]
            refined += int(np.count_nonzero(accepted & (destination != selected)))
            ambiguous += int(np.count_nonzero(edge & ~accepted))
            np.copyto(destination, selected, where=accepted)
            if metric is not None:
                assert evidence.metric_map_m is not None
                metric_samples = np.stack([evidence.metric_map_m[y, x] for y, x in indices])
                selected_metric = np.take_along_axis(metric_samples, chosen[None], axis=0)[0]
                np.copyto(metric[top:bottom, left:right], selected_metric, where=accepted)
    return {
        "candidate_pixels": candidates,
        "refined_pixels": refined,
        "ambiguous_pixels": ambiguous,
        "min_native_relative_span": _MIN_DEPTH_SPAN,
        "native_support": "complete_4x4_two_stable_depth_groups",
        "max_native_group_deviation_fraction": _NATIVE_CLUSTER_TOLERANCE,
        "min_rgb_separation": _MIN_COLOR_SEPARATION,
        "max_rgb_match_distance": _MAX_COLOR_DISTANCE,
        "min_rgb_match_margin": _MIN_COLOR_MARGIN,
        "max_scratch_tile_pixels": _MAX_CHUNK_PIXELS,
        "fallback": "valid_weighted_bilinear",
        "quality_acceptance": "unestablished",
    }


def align_depth(
    evidence: DepthEvidence,
    master: ImageMaster,
    proxy: ImageProxy,
    *,
    refinement: str = "guided_bilinear",
) -> AlignedDepth:
    """Reconstruct master-grid derivatives without changing native evidence."""
    if not isinstance(evidence, DepthEvidence) or not isinstance(master, ImageMaster) or not isinstance(proxy, ImageProxy):
        raise ValueError("Alignment requires typed depth, master, and proxy evidence")
    if refinement not in {"guided_bilinear", "bilinear"}:
        raise ValueError("Unsupported depth refinement recipe")
    master_hash = master.content_hash()
    if (
        evidence.source_sha256 != master.source_sha256
        or evidence.transform != proxy.transform
        or proxy.transform.original_shape != master.shape
        or proxy.master_content_hash != master_hash
        or evidence.metadata.get("proxy_master_content_hash") != master_hash
        or evidence.metadata.get("proxy_content_hash")
        != artifact_content_hash(
            {"transform": proxy.transform.to_payload(), "master_content_hash": proxy.master_content_hash},
            {"pixels": proxy.pixels},
        )
    ):
        raise ValueError("Depth alignment inputs do not bind the same source, master, and geometry")
    relative, valid = restore_depth_with_validity(evidence.relative_depth(), evidence.valid_mask, proxy.transform)
    # Pillow-backed bilinear output can be a read-only NumPy view.
    relative = relative.copy()
    support_score = restore_depth(evidence.valid_mask.astype(np.float32), proxy.transform)
    support_score = np.where(valid, np.clip(support_score, 0, 1), 0).astype(np.float32)
    metric = None
    if evidence.metric_map_m is not None:
        metric, metric_valid = restore_depth_with_validity(evidence.metric_map_m, evidence.valid_mask, proxy.transform)
        metric = metric.copy()
        if not np.array_equal(metric_valid, valid):
            raise ValueError("Metric and relative alignment support disagree")
    if refinement == "guided_bilinear" and valid.any():
        recipe = _guided_selection(evidence, master, proxy, relative, valid, metric)
    else:
        recipe = {"candidate_pixels": 0, "refined_pixels": 0, "ambiguous_pixels": 0}
    return AlignedDepth(
        relative_depth=relative,
        valid_mask=valid,
        support_mask=np.ones(master.shape, bool),
        support_confidence=support_score,
        source_sha256=master.source_sha256,
        evidence_content_hash=evidence.content_hash(),
        master_content_hash=master_hash,
        metric_map_m=metric,
        metadata={
            "recipe": "bounded_two_surface_rgb_selection_v2" if refinement == "guided_bilinear" else "valid_bilinear_v1",
            "refinement": refinement,
            "geometry": proxy.transform.to_payload(),
            "native_preserved": True,
            **recipe,
        },
    )


def enhance_master_v5(
    master: ImageMaster,
    aligned: AlignedDepth,
    *,
    strength: float = 0.25,
    clarity: float = 0.0,
    protected_mask: np.ndarray | None = None,
) -> tuple[ImageMaster, dict[str, Any]]:
    """Apply bounded exposure/clarity only where surface use is authorized.

    Explicit protected regions, sky/invalid samples, and nonopaque pixels are
    copied exactly. Interpolation support attenuates the edit, never certifies
    depth correctness. Clarity adds at most 0.05 * clarity linear units/channel.
    """
    if not isinstance(master, ImageMaster) or not isinstance(aligned, AlignedDepth):
        raise ValueError("V5 enhancement requires typed photographic and aligned evidence")
    if (
        master.shape != aligned.shape
        or master.source_sha256 != aligned.source_sha256
        or master.content_hash() != aligned.master_content_hash
    ):
        raise ValueError("Aligned depth does not authorize this photographic master")
    if (
        isinstance(strength, bool)
        or isinstance(clarity, bool)
        or not np.isfinite([strength, clarity]).all()
        or not 0 <= strength <= 1
        or not 0 <= clarity <= 1
    ):
        raise ValueError("Photographic strength and clarity must be finite in [0,1]")
    allowed = aligned.valid_mask.copy()
    if protected_mask is not None:
        protected = np.asarray(protected_mask)
        if protected.dtype != np.bool_ or protected.shape != master.shape:
            raise ValueError("Protected regions must be a boolean mask on the master grid")
        allowed &= ~protected
    if master.alpha is not None:
        allowed &= master.alpha == 1
    pixels = master.pixels.copy()
    depth_applied = clarity_applied = False
    if allowed.any():
        guide = aligned.relative_depth
        if strength and np.ptp(guide[allowed]) > 1e-6:
            center = float(np.percentile(guide[allowed], 75))
            exposure = np.clip(center - guide, -1, 1) * float(strength) * aligned.support_confidence
            gain = np.where(allowed, np.exp2(exposure), 1.0)
            pixels *= gain[..., None]
            depth_applied = True
        if clarity:
            from scipy.ndimage import gaussian_filter

            encoded = linear_to_srgb(pixels)
            weight = gaussian_filter(allowed.astype(np.float32), sigma=1.0)
            smooth = gaussian_filter(np.where(allowed[..., None], encoded, 0), sigma=(1, 1, 0))
            smooth /= np.maximum(weight[..., None], 1e-6)
            candidate = srgb_to_linear(encoded + float(clarity) * 0.2 * (encoded - smooth))
            delta = np.clip(candidate - pixels, -0.05 * float(clarity), 0.05 * float(clarity))
            pixels += np.where(allowed[..., None], delta * aligned.support_confidence[..., None], 0)
            clarity_applied = True
    # Preserve protected samples bit-for-bit, including signed zero.
    pixels[~allowed] = master.pixels[~allowed]
    if not np.isfinite(pixels).all():
        raise ValueError("Photographic response exceeded finite float32 representation")
    changed = np.any(pixels != master.pixels, axis=2)
    report = {
        "schema": "tp.depth.response.v1",
        "depth_applied": depth_applied,
        "clarity_applied": clarity_applied,
        "strength": float(strength),
        "clarity": float(clarity),
        "max_exposure_stops": float(strength),
        "max_clarity_linear_delta": 0.05 * float(clarity),
        "authorized_pixels": int(allowed.sum()),
        "protected_pixels": int((~allowed).sum()),
        "changed_pixels": int(changed.sum()),
        "max_absolute_linear_delta": float(np.max(np.abs(pixels - master.pixels))),
        "aligned_content_hash": aligned.content_hash(),
        "quality_acceptance": "unestablished",
    }
    metadata = metadata_payload(master.metadata)
    metadata["finishing"] = {"version": "v5.0", **report}
    return (
        ImageMaster(pixels, master.source_sha256, master.source_bit_depth, master.alpha, metadata, master.source_icc),
        report,
    )


def generate_preview_maps_v5(evidence: DepthEvidence) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Produce explicitly nonphysical previews with complete valid filter support.

    Fixed derivative scales avoid letting an invalid hole alter global preview
    normalization. A map pixel is unknown unless its complete derivative and
    smoothing neighborhood lies on usable in-frame surface evidence.
    """
    from scipy import ndimage

    if not isinstance(evidence, DepthEvidence):
        raise ValueError("Preview maps require typed native depth evidence")
    values = evidence.relative_depth()
    usable = evidence.valid_mask
    normal_valid = ndimage.minimum_filter(usable, size=3, mode="constant", cval=0)
    roughness_valid = ndimage.minimum_filter(usable, size=9, mode="constant", cval=0)
    ao_valid = ndimage.minimum_filter(usable, size=13, mode="constant", cval=0)
    dx = ndimage.sobel(values, axis=1, mode="nearest") / 8.0
    dy = ndimage.sobel(values, axis=0, mode="nearest") / 8.0
    normals = np.stack((-dx, -dy, np.ones_like(values)), axis=2)
    normals /= np.linalg.norm(normals, axis=2, keepdims=True)
    normal = np.rint((normals + 1) * 127.5).astype(np.uint8)
    detail = np.abs(ndimage.laplace(values, mode="nearest"))
    roughness = np.rint(np.clip(ndimage.uniform_filter(detail, size=7, mode="nearest"), 0, 1) * 255).astype(np.uint8)
    occlusion = ndimage.uniform_filter(np.hypot(dx, dy), size=11, mode="nearest")
    ao = np.rint((1 - 0.5 * np.clip(occlusion, 0, 1)) * 255).astype(np.uint8)
    normal[~normal_valid] = [128, 128, 255]
    roughness[~roughness_valid] = 0
    ao[~ao_valid] = 255
    return {"normal": normal, "roughness": roughness, "ao": ao}, {
        "kind": "depth_derived_preview",
        "physical_material_estimate": False,
        "recipe": "valid_support_fixed_derivatives_v1",
        "depth_content_hash": evidence.content_hash(),
        "shape": list(evidence.shape),
        "support_policy": "complete_usable_surface_filter_neighborhood",
        "support_radius_pixels": {"normal": 1, "roughness": 4, "ao": 6},
        "valid_fraction": {
            "normal": float(normal_valid.mean()),
            "roughness": float(roughness_valid.mean()),
            "ao": float(ao_valid.mean()),
        },
        "invalid_values": {"normal": [128, 128, 255], "roughness": 0, "ao": 255},
    }
