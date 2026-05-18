"""Material segmentation backend implementations for Lux Depth V3."""

from __future__ import annotations

from ._cache import (
    SAM2_AUTO_TILING_MAX_AREA_PX,
    SAM2_AUTO_TILING_MAX_DIM_PX,
    SEGMENTATION_CACHE_SCHEMA_VERSION,
    _build_sam2_generator_kwargs,
    _build_sam2_tiling_config,
    _build_segmentation_cache_key,
    _cached_file_sha256,
    _coerce_material_result,
    _coerce_unit_confidence,
    _file_identity,
    _mask_checksum,
    _material_confidence_evidence_from_metadata,
    _material_confidence_metadata,
    _normalise_cache_policy,
    _read_cached_material_masks,
    _segmentation_cache_paths,
    _serialize_sam2_tiling_config,
    _softmax_probabilities,
    _split_material_results,
    _stable_array_hash,
    _tensor_values_1d,
    _write_cached_material_masks,
)
from .efficient_sam import EfficientSAMBackend
from .registry import _get_backend_instance, _get_sam_vit_h_instance, get_last_segmentation_runtime_metadata, segment_materials
from .sam2 import SAM2CheckpointIntegrityError, SAM2SegmentationBackend
from .sam_vit_h import SAMCheckpointIntegrityError, SAMVitHBackend
from .stub import StubBackend

__all__ = [
    "EfficientSAMBackend",
    "SAM2SegmentationBackend",
    "SAM2CheckpointIntegrityError",
    "SAM2_AUTO_TILING_MAX_AREA_PX",
    "SAM2_AUTO_TILING_MAX_DIM_PX",
    "SAMCheckpointIntegrityError",
    "SAMVitHBackend",
    "SEGMENTATION_CACHE_SCHEMA_VERSION",
    "StubBackend",
    "_build_sam2_generator_kwargs",
    "_build_sam2_tiling_config",
    "_build_segmentation_cache_key",
    "_cached_file_sha256",
    "_coerce_material_result",
    "_coerce_unit_confidence",
    "_file_identity",
    "_get_backend_instance",
    "_get_sam_vit_h_instance",
    "_mask_checksum",
    "_material_confidence_evidence_from_metadata",
    "_material_confidence_metadata",
    "_normalise_cache_policy",
    "_read_cached_material_masks",
    "_segmentation_cache_paths",
    "_serialize_sam2_tiling_config",
    "_softmax_probabilities",
    "_split_material_results",
    "_stable_array_hash",
    "_tensor_values_1d",
    "_write_cached_material_masks",
    "get_last_segmentation_runtime_metadata",
    "segment_materials",
]
