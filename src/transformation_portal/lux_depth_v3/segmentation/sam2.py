"""SAM2 material segmentation backend adapter."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, cast

import numpy as np

from ..material_confidence_contract import MATERIAL_CLASSIFIER_SCORE_TYPE, MATERIALS_V3_CALIBRATION_VERSION
from ..protocols.segmentation_backend import SegmentationBackendInfo
from ._cache import (
    SAM2_AUTO_TILING_MAX_AREA_PX,
    SAM2_AUTO_TILING_MAX_DIM_PX,
    _build_sam2_generator_kwargs,
    _build_sam2_tiling_config,
    _serialize_sam2_tiling_config,
)
from .efficient_sam import EfficientSAMBackend

logger = logging.getLogger(__name__)

try:
    from transformation_portal.spatial_ai.segmentation.contracts import SegmentationInput as SpatialSegmentationInput
    from transformation_portal.spatial_ai.segmentation.sam2_backend import SAM2Backend as SpatialSAM2Backend
    from transformation_portal.spatial_ai.segmentation.sam2_backend import (
        SAM2CheckpointIntegrityError,
    )
    from transformation_portal.spatial_ai.segmentation.tiling.config import GlobalPassConfig, SegmentationTilingConfig

    SPATIAL_SAM2_AVAILABLE = True
except ImportError:
    SPATIAL_SAM2_AVAILABLE = False
    GlobalPassConfig = None  # type: ignore
    SAM2CheckpointIntegrityError = None  # type: ignore
    SpatialSAM2Backend = None  # type: ignore
    SpatialSegmentationInput = None  # type: ignore
    SegmentationTilingConfig = None  # type: ignore


class SAM2SegmentationBackend(EfficientSAMBackend):
    """SAM2-based material segmentation backend.

    This backend reuses the existing CLIP/heuristic material classification flow
    from EfficientSAMBackend, but sources instance masks from the spatial_ai SAM2
    backend.
    """

    _LABEL_ALIASES = {
        "sky": "sky",
        "cloud": "sky",
        "glass": "glass",
        "window": "glass",
        "water": "water",
        "pool": "water",
        "ocean": "water",
        "sea": "water",
        "foliage": "foliage",
        "plant": "foliage",
        "tree": "foliage",
        "leaf": "foliage",
        "stone": "stone",
        "marble": "stone",
        "granite": "stone",
        "limestone": "stone",
        "travertine": "stone",
        "concrete": "stone",
        "wood": "wood",
        "metal": "metal",
        "fabric": "fabric",
        "stucco": "stucco",
        "plaster": "stucco",
    }

    def __init__(
        self,
        model_size: str = "base",
        checkpoint_path: Optional[str] = None,
        model_config: Optional[str] = None,
        expected_sha256: Optional[str] = None,
        enable_material_classification: bool = False,
        material_confidence_threshold: float = 0.3,
        tiling_enabled: bool = False,
        tile_size_px: int = 1536,
        overlap_px: int = 256,
        global_pass_longest_side: int = 1280,
        max_concurrency: int = 1,
        points_per_side: int = 32,
        points_per_batch: int = 64,
        pred_iou_thresh: float = 0.88,
        stability_score_thresh: float = 0.85,
        crop_n_layers: int = 1,
        sky_top_region_fraction: float = 0.5,
        sky_gradient_threshold: float = 0.05,
        sky_brightness_threshold: float = 0.4,
    ) -> None:
        super().__init__(
            sky_top_region_fraction=sky_top_region_fraction,
            sky_gradient_threshold=sky_gradient_threshold,
            sky_brightness_threshold=sky_brightness_threshold,
        )
        self._model_size = model_size
        self._checkpoint_path = checkpoint_path
        self._model_config = model_config
        self._expected_sha256 = expected_sha256
        self._enable_material_classification = enable_material_classification
        self._material_confidence_threshold = material_confidence_threshold
        self._generator_kwargs = _build_sam2_generator_kwargs(
            points_per_side=points_per_side,
            points_per_batch=points_per_batch,
            pred_iou_thresh=pred_iou_thresh,
            stability_score_thresh=stability_score_thresh,
            crop_n_layers=crop_n_layers,
        )
        self._configured_tiling = _build_sam2_tiling_config(
            enabled=tiling_enabled,
            tile_size_px=tile_size_px,
            overlap_px=overlap_px,
            global_pass_longest_side=global_pass_longest_side,
            max_concurrency=max_concurrency,
        )
        self._last_runtime_metadata: Optional[Dict[str, Any]] = None
        self._sam2_backend: Any = None

    @property
    def info(self) -> SegmentationBackendInfo:
        model_id = f"facebook/sam2-hiera-{self._model_size}"
        if self._model_size == "large":
            model_id = "facebook/sam2.1-hiera-large"
        return SegmentationBackendInfo(
            name="SAM2",
            model_id=model_id,
            requires_gpu=False,
            requires_weights=True,
            approximate_memory_mb=850 if self._model_size == "large" else 400,
            description="SAM2 segmentation backend via spatial_ai wrapper",
        )

    @classmethod
    def _canonicalize_material_label(cls, label: Optional[str]) -> Optional[str]:
        """Map free-form labels to the Materials V3 taxonomy keys."""
        if not label:
            return None

        norm = label.strip().lower()
        for token, canonical in cls._LABEL_ALIASES.items():
            if token in norm:
                return canonical
        return None

    @staticmethod
    def _merge_material_result(
        accumulator: Dict[str, Tuple[np.ndarray, float, int]],
        material: str,
        mask: np.ndarray,
        confidence: float,
    ) -> None:
        """Merge another (mask, confidence) contribution into a material bucket."""
        mask_f32 = mask.astype(np.float32, copy=False)
        area = int(np.count_nonzero(mask_f32 > 0.5))
        if area <= 0:
            return

        previous = accumulator.get(material)
        if previous is None:
            accumulator[material] = (mask_f32, float(np.clip(confidence, 0.0, 1.0)), area)
            return

        prev_mask, prev_conf, prev_area = previous
        merged_mask = np.maximum(prev_mask, mask_f32)
        total_area = prev_area + area
        if total_area <= 0:
            merged_conf = 0.0
        else:
            merged_conf = (prev_conf * prev_area + float(np.clip(confidence, 0.0, 1.0)) * area) / total_area
        accumulator[material] = (merged_mask, merged_conf, total_area)

    def _resolve_effective_tiling(self, image: np.ndarray) -> tuple[Any, bool]:
        """Return the effective SAM2 tiling config and whether it was auto-enabled."""
        configured_tiling = self._configured_tiling
        if configured_tiling is None:
            return None, False
        if bool(getattr(configured_tiling, "enabled", False)):
            return configured_tiling, False

        height, width = image.shape[:2]
        if (height * width) <= SAM2_AUTO_TILING_MAX_AREA_PX and max(height, width) <= SAM2_AUTO_TILING_MAX_DIM_PX:
            return configured_tiling, False

        return (
            _build_sam2_tiling_config(
                enabled=True,
                tile_size_px=int(getattr(configured_tiling, "tile_size_px", 1536)),
                overlap_px=int(getattr(configured_tiling, "overlap_px", 256)),
                global_pass_longest_side=int(getattr(getattr(configured_tiling, "global_pass", None), "longest_side", 1280)),
                max_concurrency=int(getattr(configured_tiling, "max_concurrency", 1)),
            ),
            True,
        )

    def _record_runtime_metadata(self, image: np.ndarray, *, effective_tiling: Any, auto_enabled: bool) -> None:
        """Capture the effective SAM2 execution surface for manifests and run cards."""
        height, width = image.shape[:2]
        self._last_runtime_metadata = {
            "sam2_runtime": {
                "model_size": self._model_size,
                "device": self._device,
                "checkpoint_path": self._checkpoint_path,
                "model_config": getattr(self._sam2_backend, "model_config", self._model_config),
                "expected_sha256": getattr(self._sam2_backend, "expected_sha256", self._expected_sha256),
                "generator_kwargs": dict(self._generator_kwargs),
                "tiling": {
                    "configured": _serialize_sam2_tiling_config(self._configured_tiling),
                    "effective": _serialize_sam2_tiling_config(effective_tiling),
                    "auto_enabled": bool(auto_enabled),
                    "decision": "auto_large_image" if auto_enabled else "configured_or_disabled",
                    "image_shape": [int(height), int(width), int(image.shape[2])],
                    "image_area_px": int(height * width),
                },
            }
        }

    def get_runtime_metadata(self) -> Optional[Dict[str, Any]]:
        metadata: Dict[str, Any] = {}
        parent_metadata = super().get_runtime_metadata()
        if isinstance(parent_metadata, dict):
            metadata.update(parent_metadata)
        if isinstance(self._last_runtime_metadata, dict):
            metadata.update(self._last_runtime_metadata)
        return metadata or None

    def load(self, device: str = "auto", weights_path: Optional[Path] = None) -> None:
        """Load SAM2 backend from spatial_ai module."""
        if self._model_loaded:
            logger.debug("SAM2 backend already loaded, skipping")
            return

        if not SPATIAL_SAM2_AVAILABLE:
            raise RuntimeError(
                "SAM2 backend unavailable. Install spatial AI segmentation deps "
                "(sam2 + torch + torchvision), or choose --segmentation-backend efficientsam."
            )

        if self._model_size not in {"base", "large"}:
            raise RuntimeError(f"Invalid sam2 model size '{self._model_size}'. Expected 'base' or 'large'.")

        resolved_device = self._resolve_device(device)
        # Validate device for Literal compatibility
        if resolved_device not in {"auto", "cuda", "cpu", "mps"}:
            raise RuntimeError(f"Invalid device '{resolved_device}'. Expected auto/cuda/cpu/mps.")

        checkpoint_override: Optional[str] = None
        if weights_path is not None:
            checkpoint_override = str(weights_path)
        elif self._checkpoint_path:
            checkpoint_override = str(self._checkpoint_path)

        try:
            sam2_kwargs = {
                "model_size": cast(Literal["base", "large"], self._model_size),
                "device": cast(Literal["auto", "cuda", "cpu", "mps"], resolved_device),
                "checkpoint_path": checkpoint_override,
                "model_config": self._model_config,
                "expected_sha256": self._expected_sha256,
                "generator_kwargs": dict(self._generator_kwargs),
                "enable_material_classification": self._enable_material_classification,
                "material_confidence_threshold": self._material_confidence_threshold,
                "tiling": self._configured_tiling,
            }
            # Cast to Literal types after validation above
            try:
                self._sam2_backend = SpatialSAM2Backend(**sam2_kwargs)
            except TypeError as exc:
                if "generator_kwargs" not in str(exc) and "tiling" not in str(exc):
                    raise
                legacy_kwargs = dict(sam2_kwargs)
                legacy_kwargs.pop("generator_kwargs", None)
                legacy_kwargs.pop("tiling", None)
                legacy_kwargs.pop("model_config", None)
                legacy_kwargs.pop("expected_sha256", None)
                self._sam2_backend = SpatialSAM2Backend(**legacy_kwargs)
        except SAM2CheckpointIntegrityError:
            raise
        except Exception as exc:
            raise RuntimeError(f"SAM2 backend loading failed: {exc}") from exc

        self._device = getattr(self._sam2_backend, "device", resolved_device)
        self._model = self._sam2_backend
        self._model_loaded = True
        self._use_real_model = True
        logger.info("SAM2 backend loaded successfully (model=%s, device=%s)", self._model_size, self._device)

    def segment(self, image: np.ndarray) -> Dict[str, Tuple[np.ndarray, float]]:
        """Run SAM2 segmentation and map masks to material outputs."""
        if not self._model_loaded or self._sam2_backend is None:
            raise RuntimeError("SAM2 model not loaded. Call .load() first.")

        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Expected RGB image (H, W, 3), got shape {image.shape}")
        if image.dtype != np.uint8:
            raise ValueError(f"Expected uint8 image, got dtype {image.dtype}")

        image_linear = image.astype(np.float32) / 255.0

        try:
            effective_tiling, auto_enabled = self._resolve_effective_tiling(image)
            if effective_tiling is not None:
                self._sam2_backend.tiling = effective_tiling
            seg_input = SpatialSegmentationInput(
                image=image_linear,
                gamma=1.0,
                mode="auto",
            )
            seg_result = self._sam2_backend.segment(seg_input)
        except SAM2CheckpointIntegrityError:
            raise
        except Exception as exc:
            raise RuntimeError(f"SAM2 inference failed: {exc}") from exc
        self._record_runtime_metadata(
            image,
            effective_tiling=effective_tiling,
            auto_enabled=auto_enabled,
        )

        if seg_result.masks.shape[0] == 0:
            logger.debug("SAM2 produced no masks; falling back to heuristic material segmentation")
            return self._heuristic_segmentation(image)

        # First preference: use SAM2/CLIP labels when available.
        material_buckets: Dict[str, Tuple[np.ndarray, float, int]] = {}
        segments: List[Dict[str, Any]] = []
        masks = np.asarray(seg_result.masks)
        scores = np.asarray(seg_result.scores, dtype=np.float32)

        for idx in range(masks.shape[0]):
            raw_mask = masks[idx]
            mask_2d = np.asarray(raw_mask).squeeze()
            if mask_2d.ndim != 2:
                logger.debug("Skipping SAM2 mask with unexpected shape: %s", raw_mask.shape)
                continue

            mask_bool = mask_2d.astype(bool, copy=False)
            area = int(mask_bool.sum())
            if area <= 0:
                continue

            metadata = seg_result.metadata[idx] if idx < len(seg_result.metadata) else None
            if metadata is not None:
                x, y, w_box, h_box = metadata.bbox
                bbox = [int(x), int(y), int(w_box), int(h_box)]
            else:
                rows, cols = np.where(mask_bool)
                bbox = [
                    int(cols.min()),
                    int(rows.min()),
                    int(cols.max() - cols.min() + 1),
                    int(rows.max() - rows.min() + 1),
                ]

            segments.append(
                {
                    "segmentation": mask_bool,
                    "bbox": bbox,
                    "area": area,
                    "predicted_iou": float(scores[idx]) if idx < len(scores) else 0.5,
                }
            )

            label = self._canonicalize_material_label(getattr(metadata, "material_label", None))
            if label:
                confidence = getattr(metadata, "material_confidence", None)
                if confidence is None:
                    confidence = float(scores[idx]) if idx < len(scores) else 0.5
                self._merge_material_result(
                    material_buckets,
                    label,
                    mask_bool.astype(np.float32, copy=False),
                    float(confidence),
                )

        if material_buckets:
            logger.debug(
                "SAM2 classified %d masks via metadata labels: %s",
                len(segments),
                list(material_buckets.keys()),
            )
            self._material_confidence_evidence = {
                material: {
                    "material_confidence": float(bucket[1]),
                    "confidence_score_type": MATERIAL_CLASSIFIER_SCORE_TYPE,
                    "raw_clip_similarity": None,
                    "clip_softmax_probability": None,
                    "clip_top2_margin": None,
                    "calibration_version": MATERIALS_V3_CALIBRATION_VERSION,
                }
                for material, bucket in material_buckets.items()
            }
            return {k: (v[0], float(v[1])) for k, v in material_buckets.items()}

        # No explicit labels: reuse existing CLIP/heuristic material labeling.
        classified = self._classify_segments_with_clip(image, segments)
        if classified:
            return classified
        return self._heuristic_segmentation(image)
