"""Shared segmentation cache, confidence, and utility helpers."""

from __future__ import annotations

import hashlib
import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple, cast

import numpy as np

from transformation_portal.ingest.canonical_json import canonicalize_json, dump_json

try:
    from transformation_portal.spatial_ai.segmentation.tiling.config import GlobalPassConfig, SegmentationTilingConfig
except ImportError:
    GlobalPassConfig = None  # type: ignore
    SegmentationTilingConfig = None  # type: ignore

logger = logging.getLogger(__name__)


def _coerce_unit_confidence(value: Any) -> Optional[float]:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return None
    if 0.0 <= confidence <= 1.0 and np.isfinite(confidence):
        return confidence
    return None


def _coerce_material_result(value: Any) -> Tuple[np.ndarray, Optional[float]]:
    if isinstance(value, tuple) and len(value) == 2:
        mask, confidence = value
        return mask, _coerce_unit_confidence(confidence)
    return value, None


def _split_material_results(results: Mapping[str, Any]) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    masks: Dict[str, np.ndarray] = {}
    material_confidences: Dict[str, float] = {}
    for material, value in results.items():
        mask, confidence = _coerce_material_result(value)
        masks[material] = mask
        if confidence is not None:
            material_confidences[material] = confidence
    return masks, material_confidences


def _material_confidence_metadata(
    material_confidences: Dict[str, float],
    evidence: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    if not material_confidences:
        return {}

    scores = list(material_confidences.values())
    metadata: Dict[str, Any] = {
        "material_confidences": dict(material_confidences),
        "confidence_summary": {
            "count": len(scores),
            "min": float(min(scores)),
            "mean": float(np.mean(scores)),
            "max": float(max(scores)),
        },
    }
    if evidence:
        metadata["material_confidence_evidence"] = {
            str(material): dict(values)
            for material, values in evidence.items()
            if material in material_confidences and isinstance(values, dict)
        }
    return metadata


def _material_confidence_evidence_from_metadata(
    metadata: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Dict[str, Any]]]:
    if not isinstance(metadata, dict):
        return None
    evidence = metadata.get("material_confidence_evidence")
    if not isinstance(evidence, dict):
        return None
    return {str(material): dict(values) for material, values in evidence.items() if isinstance(values, dict)}


def _tensor_values_1d(value: Any) -> np.ndarray:
    """Convert torch/fake tensor rows into a 1D float32 array."""
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        try:
            value = value.detach().cpu()
            if hasattr(value, "float"):
                value = value.float()
            if hasattr(value, "numpy"):
                try:
                    return np.asarray(value.numpy(), dtype=np.float32).reshape(-1)
                except RuntimeError as exc:
                    if "Numpy is not available" not in str(exc):
                        raise
            if hasattr(value, "tolist"):
                return np.asarray(value.tolist(), dtype=np.float32).reshape(-1)
        except (AttributeError, TypeError, ValueError):
            logger.debug("Tensor-like value conversion failed; falling back to array coercion", exc_info=True)
    values = getattr(value, "values", None)
    if values is not None and not callable(values):
        return np.asarray(values, dtype=np.float32).reshape(-1)
    return np.asarray(value, dtype=np.float32).reshape(-1)


def _softmax_probabilities(values: np.ndarray, logit_scale: float = 20.0) -> np.ndarray:
    if values.size == 0:
        return values.astype(np.float32)
    logits = values.astype(np.float32) * float(logit_scale)
    shifted = logits - float(np.max(logits))
    exp_values = np.exp(shifted)
    total = float(exp_values.sum())
    if total <= 0.0 or not np.isfinite(total):
        return np.zeros_like(values, dtype=np.float32)
    return (exp_values / total).astype(np.float32)


SAM2_AUTO_TILING_MAX_AREA_PX = 8_000_000
SAM2_AUTO_TILING_MAX_DIM_PX = 4096
SEGMENTATION_CACHE_SCHEMA_VERSION = "materials-segmentation-cache.v1"
_CACHE_MASK_CHECKSUM_CHUNK_SIZE = 1024 * 1024


def _build_sam2_generator_kwargs(
    *,
    points_per_side: int,
    points_per_batch: int,
    pred_iou_thresh: float,
    stability_score_thresh: float,
    crop_n_layers: int,
) -> Dict[str, Any]:
    return {
        "points_per_side": int(points_per_side),
        "points_per_batch": int(points_per_batch),
        "pred_iou_thresh": float(pred_iou_thresh),
        "stability_score_thresh": float(stability_score_thresh),
        "crop_n_layers": int(crop_n_layers),
    }


def _build_sam2_tiling_config(
    *,
    enabled: bool,
    tile_size_px: int,
    overlap_px: int,
    global_pass_longest_side: int,
    max_concurrency: int,
) -> Any:
    if SegmentationTilingConfig is None:
        return None
    return SegmentationTilingConfig(
        enabled=bool(enabled),
        tile_size_px=int(tile_size_px),
        overlap_px=int(overlap_px),
        global_pass=GlobalPassConfig(longest_side=int(global_pass_longest_side)),
        max_concurrency=int(max_concurrency),
    )


def _serialize_sam2_tiling_config(tiling: Any) -> Optional[Dict[str, Any]]:
    if tiling is None:
        return None
    return {
        "enabled": bool(getattr(tiling, "enabled", False)),
        "policy": getattr(tiling, "policy", None),
        "tile_size_px": getattr(tiling, "tile_size_px", None),
        "overlap_px": getattr(tiling, "overlap_px", None),
        "global_pass": {
            "enabled": bool(getattr(getattr(tiling, "global_pass", None), "enabled", False)),
            "longest_side": getattr(getattr(tiling, "global_pass", None), "longest_side", None),
        },
        "max_concurrency": getattr(tiling, "max_concurrency", None),
    }


def _stable_array_hash(array: np.ndarray) -> str:
    arr = array if array.flags.c_contiguous else np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(arr.shape).encode("utf-8"))
    digest.update(str(arr.dtype).encode("utf-8"))
    view = memoryview(cast(Any, arr.view(np.uint8).reshape(-1)))
    digest.update(cast(Any, view))
    return digest.hexdigest()


def _mask_checksum(mask: np.ndarray) -> str:
    arr = mask if mask.flags.c_contiguous else np.ascontiguousarray(mask)
    digest = hashlib.sha256()
    digest.update(str(arr.shape).encode("utf-8"))
    digest.update(str(arr.dtype).encode("utf-8"))
    view = memoryview(cast(Any, arr.view(np.uint8).reshape(-1)))
    for offset in range(0, len(view), _CACHE_MASK_CHECKSUM_CHUNK_SIZE):
        digest.update(cast(Any, view[offset : offset + _CACHE_MASK_CHECKSUM_CHUNK_SIZE]))
    return digest.hexdigest()


@lru_cache(maxsize=8)
def _cached_file_sha256(path: str, size: int, mtime_ns: int) -> str:
    del size, mtime_ns
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(_CACHE_MASK_CHECKSUM_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_identity(path_value: Optional[str]) -> Optional[Dict[str, Any]]:
    if not path_value:
        return None
    path = Path(path_value)
    if not path.is_file():
        return {"path": str(path), "exists": False, "sha256": None, "size": None, "mtime_ns": None}
    try:
        stat = path.stat()
        return {
            "path": str(path),
            "exists": True,
            "sha256": _cached_file_sha256(str(path), int(stat.st_size), int(stat.st_mtime_ns)),
            "size": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
        }
    except OSError:
        return {"path": str(path), "exists": False, "sha256": None, "size": None, "mtime_ns": None}


def _normalise_cache_policy(value: Any) -> str:
    policy = str(value or "read_write").strip().lower()
    return policy if policy in {"off", "read_write"} else "off"


def _segmentation_cache_paths(cache_dir: Path, cache_key: str) -> tuple[Path, Path]:
    shard = cache_key[:2]
    root = cache_dir / shard
    return root / f"{cache_key}.npz", root / f"{cache_key}.json"


def _build_segmentation_cache_key(
    *,
    image: np.ndarray,
    backend_name: str,
    device: str,
    strict_backend: bool,
    sam2_model_size: str,
    sam2_checkpoint_path: Optional[str],
    sam2_tiling_enabled: bool,
    sam2_tile_size_px: int,
    sam2_overlap_px: int,
    sam2_global_pass_longest_side: int,
    sam2_max_concurrency: int,
    sam2_points_per_side: int,
    sam2_points_per_batch: int,
    sam2_pred_iou_thresh: float,
    sam2_stability_score_thresh: float,
    sam2_crop_n_layers: int,
    sam_vit_h_checkpoint_path: Optional[str],
    sam_vit_h_points_per_side: int,
    sam_vit_h_pred_iou_thresh: float,
    sam_vit_h_confidence_threshold: float,
    sam_vit_h_expected_sha256: Optional[str],
    sky_top_region_fraction: float,
    sky_gradient_threshold: float,
    sky_brightness_threshold: float,
) -> tuple[str, Dict[str, Any]]:
    payload: Dict[str, Any] = {
        "schema_version": SEGMENTATION_CACHE_SCHEMA_VERSION,
        "image_hash": _stable_array_hash(image),
        "image_shape": list(image.shape),
        "image_dtype": str(image.dtype),
        "backend": backend_name,
        "device": device,
        "strict_backend": bool(strict_backend),
        "sam2_model_size": sam2_model_size,
        "sam2_checkpoint": _file_identity(sam2_checkpoint_path),
        "sam2_generator": {
            "points_per_side": int(sam2_points_per_side),
            "points_per_batch": int(sam2_points_per_batch),
            "pred_iou_thresh": float(sam2_pred_iou_thresh),
            "stability_score_thresh": float(sam2_stability_score_thresh),
            "crop_n_layers": int(sam2_crop_n_layers),
        },
        "sam2_tiling": {
            "enabled": bool(sam2_tiling_enabled),
            "tile_size_px": int(sam2_tile_size_px),
            "overlap_px": int(sam2_overlap_px),
            "global_pass_longest_side": int(sam2_global_pass_longest_side),
            "max_concurrency": int(sam2_max_concurrency),
        },
        "sam_vit_h": {
            "checkpoint": _file_identity(sam_vit_h_checkpoint_path),
            "points_per_side": int(sam_vit_h_points_per_side),
            "pred_iou_thresh": float(sam_vit_h_pred_iou_thresh),
            "confidence_threshold": float(sam_vit_h_confidence_threshold),
            "expected_sha256": sam_vit_h_expected_sha256,
        },
        "sky_bootstrap": {
            "top_region_fraction": float(sky_top_region_fraction),
            "gradient_threshold": float(sky_gradient_threshold),
            "brightness_threshold": float(sky_brightness_threshold),
        },
    }
    cache_key = hashlib.sha256(canonicalize_json(payload)).hexdigest()
    return cache_key, payload


def _read_cached_material_masks(
    *,
    cache_dir: Path,
    cache_key: str,
    expected_payload: Dict[str, Any],
) -> Optional[tuple[Dict[str, Tuple[np.ndarray, float]], Dict[str, Any]]]:
    masks_path, metadata_path = _segmentation_cache_paths(cache_dir, cache_key)
    if not masks_path.is_file() or not metadata_path.is_file():
        return None

    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("schema_version") != SEGMENTATION_CACHE_SCHEMA_VERSION:
            return None
        if metadata.get("cache_key") != cache_key:
            return None
        if metadata.get("key_payload") != expected_payload:
            return None
        mask_entries = metadata.get("masks")
        if not isinstance(mask_entries, dict):
            return None

        results: Dict[str, Tuple[np.ndarray, float]] = {}
        with np.load(masks_path, allow_pickle=False) as data:
            for material, entry in mask_entries.items():
                if not isinstance(entry, dict) or material not in data.files:
                    return None
                mask = np.asarray(data[material])
                if list(mask.shape) != list(entry.get("shape") or []):
                    return None
                if str(mask.dtype) != str(entry.get("dtype")):
                    return None
                if _mask_checksum(mask) != entry.get("sha256"):
                    return None
                confidence = _coerce_unit_confidence(entry.get("confidence"))
                if confidence is None:
                    return None
                results[str(material)] = (mask.astype(np.float32, copy=False), confidence)
        return results, metadata
    except Exception as exc:
        logger.debug("Ignoring invalid material segmentation cache entry %s: %s", cache_key, exc)
        return None


def _write_cached_material_masks(
    *,
    cache_dir: Path,
    cache_key: str,
    key_payload: Dict[str, Any],
    results: Dict[str, Tuple[np.ndarray, float]],
    runtime_metadata: Optional[Dict[str, Any]],
) -> None:
    if not results:
        return

    masks_path, metadata_path = _segmentation_cache_paths(cache_dir, cache_key)
    masks_path.parent.mkdir(parents=True, exist_ok=True)

    arrays: Dict[str, np.ndarray] = {}
    mask_entries: Dict[str, Dict[str, Any]] = {}
    for material, (mask, confidence) in sorted(results.items()):
        arr = np.asarray(mask, dtype=np.float32)
        arrays[material] = arr
        mask_entries[material] = {
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
            "sha256": _mask_checksum(arr),
            "confidence": float(confidence),
        }

    temp_npz = masks_path.with_suffix(".npz.tmp")
    temp_json = metadata_path.with_suffix(".json.tmp")
    try:
        with temp_npz.open("wb") as handle:
            np.savez_compressed(handle, **cast(Any, arrays))
        metadata = {
            "schema_version": SEGMENTATION_CACHE_SCHEMA_VERSION,
            "cache_key": cache_key,
            "key_payload": key_payload,
            "masks": mask_entries,
            "runtime_metadata": runtime_metadata or {},
        }
        with temp_json.open("w", encoding="utf-8") as handle:
            dump_json(metadata, handle, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
        temp_npz.replace(masks_path)
        temp_json.replace(metadata_path)
    finally:
        for temp_path in (temp_npz, temp_json):
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except OSError:
                    logger.debug("Failed to remove temporary segmentation cache file: %s", temp_path)
