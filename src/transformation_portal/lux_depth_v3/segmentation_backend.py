"""Backward-compatible segmentation backend shim.

The concrete Materials V3 segmentation backends live under
``transformation_portal.lux_depth_v3.segmentation``. This module keeps the
legacy private import surface stable for tests and downstream callers while
Phase 3 decomposition migrates implementation code into focused modules.
"""

from __future__ import annotations

import logging
import sys
import types
from typing import Any

from .segmentation import (
    SAM2_AUTO_TILING_MAX_AREA_PX,
    SAM2_AUTO_TILING_MAX_DIM_PX,
    SEGMENTATION_CACHE_SCHEMA_VERSION,
    EfficientSAMBackend,
    SAM2SegmentationBackend,
    SAMCheckpointIntegrityError,
    SAMVitHBackend,
    StubBackend,
    _build_sam2_generator_kwargs,
    _build_sam2_tiling_config,
    _build_segmentation_cache_key,
)
from .segmentation import _cache as _segmentation_cache
from .segmentation import (
    _cached_file_sha256,
    _coerce_material_result,
    _coerce_unit_confidence,
    _file_identity,
    _get_backend_instance,
    _get_sam_vit_h_instance,
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
from .segmentation import efficient_sam as _efficient_sam
from .segmentation import (
    get_last_segmentation_runtime_metadata,
)
from .segmentation import registry as _segmentation_registry
from .segmentation import sam2 as _sam2
from .segmentation import sam_vit_h as _sam_vit_h
from .segmentation import (
    segment_materials,
)
from .segmentation import stub as _stub
from .segmentation._cache import _CACHE_MASK_CHECKSUM_CHUNK_SIZE
from .segmentation.efficient_sam import (
    EFFICIENTVIT_AVAILABLE,
    OPEN_CLIP_AVAILABLE,
    TORCH_AVAILABLE,
    TORCHVISION_AVAILABLE,
    CachedSamModel,
    EfficientViTSamAutomaticMaskGenerator,
    create_efficientvit_sam_model,
    open_clip,
    torch,
    torchvision,
)
from .segmentation.registry import _LAST_SEGMENTATION_RUNTIME_METADATA
from .segmentation.sam2 import (
    SPATIAL_SAM2_AVAILABLE,
    GlobalPassConfig,
    SegmentationTilingConfig,
    SpatialSAM2Backend,
    SpatialSegmentationInput,
)
from .segmentation.sam_vit_h import _SAM_IMPORT_ERROR, SAM_AVAILABLE, SamAutomaticMaskGenerator, sam_model_registry

logger = logging.getLogger(__name__)

__all__ = [
    "CachedSamModel",
    "EFFICIENTVIT_AVAILABLE",
    "EfficientSAMBackend",
    "EfficientViTSamAutomaticMaskGenerator",
    "GlobalPassConfig",
    "OPEN_CLIP_AVAILABLE",
    "SAM2SegmentationBackend",
    "SAM2_AUTO_TILING_MAX_AREA_PX",
    "SAM2_AUTO_TILING_MAX_DIM_PX",
    "SAMCheckpointIntegrityError",
    "SAMVitHBackend",
    "SAM_AVAILABLE",
    "SEGMENTATION_CACHE_SCHEMA_VERSION",
    "SPATIAL_SAM2_AVAILABLE",
    "SamAutomaticMaskGenerator",
    "SegmentationTilingConfig",
    "SpatialSAM2Backend",
    "SpatialSegmentationInput",
    "StubBackend",
    "TORCHVISION_AVAILABLE",
    "TORCH_AVAILABLE",
    "_CACHE_MASK_CHECKSUM_CHUNK_SIZE",
    "_LAST_SEGMENTATION_RUNTIME_METADATA",
    "_SAM_IMPORT_ERROR",
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
    "create_efficientvit_sam_model",
    "get_last_segmentation_runtime_metadata",
    "logger",
    "open_clip",
    "sam_model_registry",
    "segment_materials",
    "torch",
    "torchvision",
]

_COMPAT_PROPAGATION = {
    "CachedSamModel": (_efficient_sam,),
    "EFFICIENTVIT_AVAILABLE": (_efficient_sam,),
    "EfficientSAMBackend": (_segmentation_registry, _sam2),
    "EfficientViTSamAutomaticMaskGenerator": (_efficient_sam,),
    "GlobalPassConfig": (_segmentation_cache, _sam2),
    "OPEN_CLIP_AVAILABLE": (_efficient_sam,),
    "SAM2SegmentationBackend": (_segmentation_registry,),
    "SAM2_AUTO_TILING_MAX_AREA_PX": (_segmentation_cache, _sam2),
    "SAM2_AUTO_TILING_MAX_DIM_PX": (_segmentation_cache, _sam2),
    "SAMCheckpointIntegrityError": (_sam_vit_h,),
    "SAMVitHBackend": (_segmentation_registry,),
    "SAM_AVAILABLE": (_sam_vit_h,),
    "SEGMENTATION_CACHE_SCHEMA_VERSION": (_segmentation_cache,),
    "SPATIAL_SAM2_AVAILABLE": (_sam2,),
    "SamAutomaticMaskGenerator": (_sam_vit_h,),
    "SegmentationTilingConfig": (_segmentation_cache, _sam2),
    "SpatialSAM2Backend": (_sam2,),
    "SpatialSegmentationInput": (_sam2,),
    "StubBackend": (_segmentation_registry,),
    "TORCHVISION_AVAILABLE": (_efficient_sam,),
    "TORCH_AVAILABLE": (_efficient_sam, _sam_vit_h),
    "_CACHE_MASK_CHECKSUM_CHUNK_SIZE": (_segmentation_cache,),
    "_LAST_SEGMENTATION_RUNTIME_METADATA": (_segmentation_registry,),
    "_SAM_IMPORT_ERROR": (_sam_vit_h,),
    "_build_sam2_generator_kwargs": (_segmentation_cache, _sam2),
    "_build_sam2_tiling_config": (_segmentation_cache, _sam2),
    "_build_segmentation_cache_key": (_segmentation_cache, _segmentation_registry),
    "_cached_file_sha256": (_segmentation_cache,),
    "_coerce_material_result": (_segmentation_cache,),
    "_coerce_unit_confidence": (_segmentation_cache,),
    "_file_identity": (_segmentation_cache,),
    "_get_backend_instance": (_segmentation_registry,),
    "_get_sam_vit_h_instance": (_segmentation_registry,),
    "_mask_checksum": (_segmentation_cache,),
    "_material_confidence_evidence_from_metadata": (_segmentation_cache, _segmentation_registry),
    "_material_confidence_metadata": (_segmentation_cache, _segmentation_registry),
    "_normalise_cache_policy": (_segmentation_cache, _segmentation_registry),
    "_read_cached_material_masks": (_segmentation_cache, _segmentation_registry),
    "_segmentation_cache_paths": (_segmentation_cache,),
    "_serialize_sam2_tiling_config": (_segmentation_cache, _sam2),
    "_softmax_probabilities": (_segmentation_cache, _efficient_sam),
    "_split_material_results": (_segmentation_cache, _segmentation_registry),
    "_stable_array_hash": (_segmentation_cache,),
    "_tensor_values_1d": (_segmentation_cache, _efficient_sam),
    "_write_cached_material_masks": (_segmentation_cache, _segmentation_registry),
    "create_efficientvit_sam_model": (_efficient_sam,),
    "get_last_segmentation_runtime_metadata": (_segmentation_registry,),
    "logger": (
        _segmentation_cache,
        _efficient_sam,
        _segmentation_registry,
        _sam2,
        _sam_vit_h,
        _stub,
    ),
    "open_clip": (_efficient_sam,),
    "sam_model_registry": (_sam_vit_h,),
    "segment_materials": (_segmentation_registry,),
    "torch": (_efficient_sam, _sam_vit_h),
    "torchvision": (_efficient_sam,),
}


class _SegmentationBackendCompatModule(types.ModuleType):
    def __setattr__(self, name: str, value: Any) -> None:
        super().__setattr__(name, value)
        for module in _COMPAT_PROPAGATION.get(name, ()):
            setattr(module, name, value)


sys.modules[__name__].__class__ = _SegmentationBackendCompatModule
