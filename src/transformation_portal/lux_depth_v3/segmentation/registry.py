"""Segmentation backend registry and public execution API."""

from __future__ import annotations

import logging
import time
from contextvars import ContextVar
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from ..config import EnhanceConfig
from ..protocols.segmentation_backend import SegmentationBackend
from ._cache import (
    _build_segmentation_cache_key,
    _material_confidence_evidence_from_metadata,
    _material_confidence_metadata,
    _normalise_cache_policy,
    _read_cached_material_masks,
    _split_material_results,
    _write_cached_material_masks,
)
from .efficient_sam import EfficientSAMBackend
from .sam2 import SAM2SegmentationBackend
from .sam_vit_h import SAMVitHBackend
from .stub import StubBackend

logger = logging.getLogger(__name__)

_LAST_SEGMENTATION_RUNTIME_METADATA: ContextVar[Optional[Dict[str, Any]]] = ContextVar(
    "_LAST_SEGMENTATION_RUNTIME_METADATA",
    default=None,
)


@lru_cache(maxsize=1)  # Enforce a single loaded SAM ViT-H instance per process (~2.4 GB each).
def _get_sam_vit_h_instance(
    checkpoint_path: Optional[str] = None,
    points_per_side: int = 32,
    pred_iou_thresh: float = 0.88,
    confidence_threshold: float = 0.85,
    expected_sha256: Optional[str] = None,
    device: str = "auto",
    strict: bool = False,
) -> SegmentationBackend:
    """Load and cache exactly one SAMVitHBackend per process.

    Separated from _get_backend_instance so its maxsize=1 constraint is
    independent of the broader backend cache, preventing two SAM ViT-H
    models from coexisting in memory when callers vary parameters.
    """
    backend = SAMVitHBackend(
        checkpoint_path=checkpoint_path,
        points_per_side=points_per_side,
        pred_iou_thresh=pred_iou_thresh,
        confidence_threshold=confidence_threshold,
    )
    try:
        backend.load(device=device, expected_sha256=expected_sha256)
    except (FileNotFoundError, RuntimeError) as e:
        if strict:
            raise RuntimeError(f"Failed to load sam_vit_h backend: {e}") from e
        logger.warning(
            "Failed to load SAM ViT-H backend: %s\n"
            "Checkpoint missing or dependencies unavailable. "
            "Falling back to stub backend.",
            e,
        )
        return _get_backend_instance("stub", device="cpu", strict=False)
    return backend


# Keep this small: SAM2 instances are heavyweight and multiple cached variants can exhaust memory.
@lru_cache(maxsize=2)  # Cache backend instances by backend + device + model options
def _get_backend_instance(
    backend_name: str,
    device: str = "auto",
    strict: bool = False,
    sam2_model_size: str = "base",
    sam2_checkpoint_path: Optional[str] = None,
    sam2_tiling_enabled: bool = False,
    sam2_tile_size_px: int = 1536,
    sam2_overlap_px: int = 256,
    sam2_global_pass_longest_side: int = 1280,
    sam2_max_concurrency: int = 1,
    sam2_points_per_side: int = 32,
    sam2_points_per_batch: int = 64,
    sam2_pred_iou_thresh: float = 0.88,
    sam2_stability_score_thresh: float = 0.85,
    sam2_crop_n_layers: int = 1,
    sam_vit_h_checkpoint_path: Optional[str] = None,
    sam_vit_h_points_per_side: int = 32,
    sam_vit_h_pred_iou_thresh: float = 0.88,
    sam_vit_h_confidence_threshold: float = 0.85,
    sam_vit_h_expected_sha256: Optional[str] = None,
    sky_top_region_fraction: float = 0.5,
    sky_gradient_threshold: float = 0.05,
    sky_brightness_threshold: float = 0.4,
) -> SegmentationBackend:
    """Get or create a cached backend instance.

    Args:
        backend_name: "stub", "efficientsam", "sam2", or "sam_vit_h"
        device: Device for backend (used by model backends)
        strict: If True, raise on errors instead of falling back
        sam2_model_size: SAM2 checkpoint family ("base" or "large")
        sam2_checkpoint_path: Optional SAM2 checkpoint override
        sam_vit_h_checkpoint_path: Optional SAM ViT-H checkpoint path override
        sam_vit_h_points_per_side: Grid density for SAM ViT-H mask generation
        sam_vit_h_pred_iou_thresh: IoU quality threshold for SAM ViT-H masks
        sam_vit_h_confidence_threshold: Minimum predicted_iou to include a mask

    Returns:
        SegmentationBackend instance

    Raises:
        ValueError: If backend_name is unknown
        RuntimeError: If strict=True and backend fails to load
    """
    if backend_name == "stub":
        stub_backend: SegmentationBackend = StubBackend()
        stub_backend.load()  # No-op for stub
        return stub_backend

    if backend_name == "efficientsam":
        esam_backend: SegmentationBackend = EfficientSAMBackend(
            sky_top_region_fraction=sky_top_region_fraction,
            sky_gradient_threshold=sky_gradient_threshold,
            sky_brightness_threshold=sky_brightness_threshold,
        )
        # Lazy load will happen on first segment() call if needed
        # But we can pre-load here for better error handling
        try:
            esam_backend.load(device=device)
        except RuntimeError as e:
            if strict:
                # In strict mode, propagate the error
                raise RuntimeError(f"Failed to load {backend_name} backend: {e}") from e

            # Non-strict mode: log warning and fall back to stub
            logger.warning(
                f"Failed to load EfficientSAM backend: {e}\n"
                f"This is expected if torch is not installed or weights are missing.\n"
                f"Falling back to stub backend."
            )
            # Return stub instead
            return _get_backend_instance("stub", device="cpu", strict=False)
        return esam_backend

    if backend_name == "sam2":
        sam2_backend: SegmentationBackend = SAM2SegmentationBackend(
            model_size=sam2_model_size,
            checkpoint_path=sam2_checkpoint_path,
            tiling_enabled=sam2_tiling_enabled,
            tile_size_px=sam2_tile_size_px,
            overlap_px=sam2_overlap_px,
            global_pass_longest_side=sam2_global_pass_longest_side,
            max_concurrency=sam2_max_concurrency,
            points_per_side=sam2_points_per_side,
            points_per_batch=sam2_points_per_batch,
            pred_iou_thresh=sam2_pred_iou_thresh,
            stability_score_thresh=sam2_stability_score_thresh,
            crop_n_layers=sam2_crop_n_layers,
            sky_top_region_fraction=sky_top_region_fraction,
            sky_gradient_threshold=sky_gradient_threshold,
            sky_brightness_threshold=sky_brightness_threshold,
        )
        try:
            sam2_backend.load(device=device)
        except RuntimeError as e:
            if strict:
                raise RuntimeError(f"Failed to load {backend_name} backend: {e}") from e
            logger.warning(
                "Failed to load SAM2 backend: %s\n"
                "This is expected if checkpoint/dependencies are missing.\n"
                "Falling back to stub backend.",
                e,
            )
            return _get_backend_instance("stub", device="cpu", strict=False)
        return sam2_backend

    if backend_name == "sam_vit_h":
        return _get_sam_vit_h_instance(
            checkpoint_path=sam_vit_h_checkpoint_path,
            points_per_side=sam_vit_h_points_per_side,
            pred_iou_thresh=sam_vit_h_pred_iou_thresh,
            confidence_threshold=sam_vit_h_confidence_threshold,
            expected_sha256=sam_vit_h_expected_sha256,
            device=device,
            strict=strict,
        )

    raise ValueError(
        f"Unknown segmentation backend: {backend_name}\n" f"Valid options: 'stub', 'efficientsam', 'sam2', 'sam_vit_h'"
    )


def segment_materials(
    image: np.ndarray,
    config: EnhanceConfig,
    cache_dir: Optional[Path] = None,
) -> Dict[str, np.ndarray]:
    """Segment image into material masks.

    This is the main entry point for material segmentation in Materials V3.

    Backends:
    - stub (default): Returns empty masks, production-safe
    - efficientsam (opt-in): ML-powered segmentation
    - sam2 (opt-in): SAM2 mask proposals + CLIP/heuristic material labeling
    - sam_vit_h (opt-in, research-only): SAM ViT-H mask proposals + heuristic labeling

    Args:
        image: Input image as numpy array (H, W, 3) in RGB, uint8 [0-255]
        config: EnhanceConfig instance with segmentation settings
            - enable_material_segmentation: Enable/disable segmentation
            - material_segmentation_backend: Backend to use ("stub", "efficientsam", "sam2", or "sam_vit_h")
            - strict_backend: If True, raise on errors instead of falling back

    Returns:
        Dict mapping material names to binary masks (H, W) with values 0.0-1.0
        Example: {"glass": mask1, "water": mask2, ...}

        For stub backend, returns empty dict.
        For real backends, returns detected materials only.

    Raises:
        RuntimeError: If strict_backend=True and backend fails to load
        ValueError: If image format is invalid
    """
    _LAST_SEGMENTATION_RUNTIME_METADATA.set(None)
    t_total = time.perf_counter()
    timing_ms: Dict[str, float] = {}

    # Check if segmentation is enabled
    enable_segmentation = getattr(config, "enable_material_segmentation", False)

    if not enable_segmentation:
        logger.debug("Material segmentation disabled in config")
        return {}

    # Get backend selection
    backend_name = getattr(config, "material_segmentation_backend", "stub")
    strict_backend = getattr(config, "strict_backend", False)
    sam2_model_size = str(getattr(config, "sam2_model_size", "base")).lower()
    sam2_checkpoint_path = getattr(config, "sam2_checkpoint_path", None)
    sam2_tiling_enabled = bool(getattr(config, "sam2_tiling_enabled", False))
    sam2_tile_size_px = int(getattr(config, "sam2_tile_size_px", 1536))
    sam2_overlap_px = int(getattr(config, "sam2_overlap_px", 256))
    sam2_global_pass_longest_side = int(getattr(config, "sam2_global_pass_longest_side", 1280))
    sam2_max_concurrency = int(getattr(config, "sam2_max_concurrency", 1))
    sam2_points_per_side = int(getattr(config, "sam2_points_per_side", 32))
    sam2_points_per_batch = int(getattr(config, "sam2_points_per_batch", 64))
    sam2_pred_iou_thresh = float(getattr(config, "sam2_pred_iou_thresh", 0.88))
    sam2_stability_score_thresh = float(getattr(config, "sam2_stability_score_thresh", 0.85))
    sam2_crop_n_layers = int(getattr(config, "sam2_crop_n_layers", 1))
    sam_vit_h_checkpoint_path = getattr(config, "sam_vit_h_checkpoint_path", None)
    sam_vit_h_points_per_side = int(getattr(config, "sam_vit_h_points_per_side", 32))
    sam_vit_h_pred_iou_thresh = float(getattr(config, "sam_vit_h_pred_iou_thresh", 0.88))
    sam_vit_h_confidence_threshold = float(getattr(config, "sam_vit_h_confidence_threshold", 0.85))
    sam_vit_h_expected_sha256 = getattr(config, "sam_vit_h_expected_sha256", None)
    sky_top_region_fraction = float(getattr(config, "sky_top_region_fraction", 0.5))
    sky_gradient_threshold = float(getattr(config, "sky_gradient_threshold", 0.05))
    sky_brightness_threshold = float(getattr(config, "sky_brightness_threshold", 0.4))
    cache_policy = _normalise_cache_policy(getattr(config, "material_segmentation_cache_policy", "read_write"))

    # Get device for backend (if applicable)
    device = getattr(config, "depth_device", "cpu")  # Reuse depth_device setting
    cache_key: Optional[str] = None
    cache_payload: Optional[Dict[str, Any]] = None
    cache_enabled = bool(cache_dir) and cache_policy == "read_write" and backend_name != "stub"

    if cache_enabled and cache_dir is not None:
        t_cache = time.perf_counter()
        try:
            # When the runtime would fall through SAMVitHBackend.EXPECTED_SHA256
            # (because EnhanceConfig.sam_vit_h_expected_sha256 is unset), record
            # that effective hash in the cache key. Otherwise cache entries
            # written before the backend default became fail-closed would still
            # match — segment_materials() returns cached masks before the
            # backend is loaded, so a replayed hit would silently bypass the
            # newly pinned integrity check.
            effective_sam_vit_h_expected_sha256 = sam_vit_h_expected_sha256
            if backend_name == "sam_vit_h":
                effective_sam_vit_h_expected_sha256 = (
                    sam_vit_h_expected_sha256 or SAMVitHBackend.EXPECTED_SHA256
                )
            cache_key, cache_payload = _build_segmentation_cache_key(
                image=image,
                backend_name=backend_name,
                device=device,
                strict_backend=strict_backend,
                sam2_model_size=sam2_model_size,
                sam2_checkpoint_path=sam2_checkpoint_path,
                sam2_tiling_enabled=sam2_tiling_enabled,
                sam2_tile_size_px=sam2_tile_size_px,
                sam2_overlap_px=sam2_overlap_px,
                sam2_global_pass_longest_side=sam2_global_pass_longest_side,
                sam2_max_concurrency=sam2_max_concurrency,
                sam2_points_per_side=sam2_points_per_side,
                sam2_points_per_batch=sam2_points_per_batch,
                sam2_pred_iou_thresh=sam2_pred_iou_thresh,
                sam2_stability_score_thresh=sam2_stability_score_thresh,
                sam2_crop_n_layers=sam2_crop_n_layers,
                sam_vit_h_checkpoint_path=sam_vit_h_checkpoint_path,
                sam_vit_h_points_per_side=sam_vit_h_points_per_side,
                sam_vit_h_pred_iou_thresh=sam_vit_h_pred_iou_thresh,
                sam_vit_h_confidence_threshold=sam_vit_h_confidence_threshold,
                sam_vit_h_expected_sha256=effective_sam_vit_h_expected_sha256,
                sky_top_region_fraction=sky_top_region_fraction,
                sky_gradient_threshold=sky_gradient_threshold,
                sky_brightness_threshold=sky_brightness_threshold,
            )
            cached = _read_cached_material_masks(
                cache_dir=cache_dir,
                cache_key=cache_key,
                expected_payload=cache_payload,
            )
            timing_ms["cache_lookup"] = round((time.perf_counter() - t_cache) * 1000.0, 3)
            if cached is not None:
                cached_results, cached_metadata = cached
                masks, material_confidences = _split_material_results(cached_results)
                cached_runtime = cached_metadata.get("runtime_metadata")
                confidence_metadata = _material_confidence_metadata(
                    material_confidences,
                    evidence=_material_confidence_evidence_from_metadata(cached_runtime),
                )
                timing_ms["total"] = round((time.perf_counter() - t_total) * 1000.0, 3)
                metadata: Dict[str, Any] = {
                    "cache_hit": True,
                    "cache_key": cache_key,
                    "cache_policy": cache_policy,
                    "timing_ms": timing_ms,
                    "mask_count": len(masks),
                    "backend": backend_name,
                    "device": device,
                    "model_size": sam2_model_size if backend_name == "sam2" else None,
                }
                if isinstance(cached_runtime, dict):
                    metadata.update(cached_runtime)
                metadata.update(confidence_metadata)
                _LAST_SEGMENTATION_RUNTIME_METADATA.set(metadata)
                logger.debug("Material segmentation cache hit: %s", cache_key)
                return masks
        except Exception as exc:
            timing_ms["cache_lookup"] = round((time.perf_counter() - t_cache) * 1000.0, 3)
            logger.debug("Material segmentation cache lookup skipped after error: %s", exc)

    try:
        # Get or create backend instance (cached)
        t_backend = time.perf_counter()
        backend = _get_backend_instance(
            backend_name,
            device=device,
            strict=strict_backend,
            sam2_model_size=sam2_model_size,
            sam2_checkpoint_path=sam2_checkpoint_path,
            sam2_tiling_enabled=sam2_tiling_enabled,
            sam2_tile_size_px=sam2_tile_size_px,
            sam2_overlap_px=sam2_overlap_px,
            sam2_global_pass_longest_side=sam2_global_pass_longest_side,
            sam2_max_concurrency=sam2_max_concurrency,
            sam2_points_per_side=sam2_points_per_side,
            sam2_points_per_batch=sam2_points_per_batch,
            sam2_pred_iou_thresh=sam2_pred_iou_thresh,
            sam2_stability_score_thresh=sam2_stability_score_thresh,
            sam2_crop_n_layers=sam2_crop_n_layers,
            sam_vit_h_checkpoint_path=sam_vit_h_checkpoint_path,
            sam_vit_h_points_per_side=sam_vit_h_points_per_side,
            sam_vit_h_pred_iou_thresh=sam_vit_h_pred_iou_thresh,
            sam_vit_h_confidence_threshold=sam_vit_h_confidence_threshold,
            sam_vit_h_expected_sha256=sam_vit_h_expected_sha256,
            sky_top_region_fraction=sky_top_region_fraction,
            sky_gradient_threshold=sky_gradient_threshold,
            sky_brightness_threshold=sky_brightness_threshold,
        )
        timing_ms["backend_load"] = round((time.perf_counter() - t_backend) * 1000.0, 3)

        # Run segmentation
        t_segment = time.perf_counter()
        results = backend.segment(image)
        timing_ms["backend_segment"] = round((time.perf_counter() - t_segment) * 1000.0, 3)
        runtime_metadata: Optional[Dict[str, Any]] = None
        if hasattr(backend, "get_runtime_metadata"):
            try:
                runtime_metadata = backend.get_runtime_metadata()
            except Exception as exc:
                logger.debug("Failed to query segmentation runtime metadata: %s", exc)
                runtime_metadata = None

        # Extract masks from (mask, confidence) tuples for backward compatibility
        # while preserving real classifier confidence for downstream Materials V3
        # decisions through runtime metadata.
        masks, material_confidences = _split_material_results(results)
        confidence_metadata = _material_confidence_metadata(
            material_confidences,
            evidence=_material_confidence_evidence_from_metadata(runtime_metadata),
        )
        runtime_metadata_for_cache = dict(runtime_metadata) if isinstance(runtime_metadata, dict) else {}
        runtime_metadata_for_cache.update(confidence_metadata)

        if cache_enabled and cache_dir is not None and cache_key and cache_payload and results:
            t_cache_write = time.perf_counter()
            try:
                _write_cached_material_masks(
                    cache_dir=cache_dir,
                    cache_key=cache_key,
                    key_payload=cache_payload,
                    results=results,
                    runtime_metadata=runtime_metadata_for_cache,
                )
            except Exception as exc:
                logger.debug("Material segmentation cache write failed: %s", exc)
            timing_ms["cache_write"] = round((time.perf_counter() - t_cache_write) * 1000.0, 3)

        timing_ms["total"] = round((time.perf_counter() - t_total) * 1000.0, 3)
        metadata = {
            "cache_hit": False,
            "cache_key": cache_key,
            "cache_policy": cache_policy,
            "timing_ms": timing_ms,
            "mask_count": len(masks),
            "backend": backend_name,
            "executed_backend": getattr(getattr(backend, "info", None), "model_id", None),
            "device": getattr(backend, "_device", device),
            "model_size": sam2_model_size if backend_name == "sam2" else None,
        }
        if isinstance(runtime_metadata, dict):
            metadata.update(runtime_metadata)
        metadata.update(confidence_metadata)
        _LAST_SEGMENTATION_RUNTIME_METADATA.set(metadata)

        logger.debug(
            f"Segmentation completed using {backend.info.name}: " f"{len(masks)} materials detected: {list(masks.keys())}"
        )

        return masks

    except Exception as e:
        if strict_backend:
            logger.error(f"Segmentation failed with strict_backend=True: {e}")
            raise RuntimeError(f"Material segmentation failed: {e}") from e

        # Fail-safe: Return empty masks on error
        logger.warning(
            f"Material segmentation failed, returning empty masks: {e}\n"
            f"This is safe - Materials V3 will continue without segmentation.\n"
            f"To debug, set strict_backend=True in config."
        )
        return {}


def get_last_segmentation_runtime_metadata() -> Optional[Dict[str, Any]]:
    """Return last segmentation runtime metadata captured by segment_materials()."""
    metadata = _LAST_SEGMENTATION_RUNTIME_METADATA.get()
    if metadata is None:
        return None
    return dict(metadata)
