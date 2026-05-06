"""Segmentation cache helpers for Spatial AI orchestration."""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np

from transformation_portal.ingest.canonical_json import canonicalize_json, dump_json
from transformation_portal.spatial_ai.segmentation.contracts import MaskMetadata, SegmentationResult

from .artifact_utils import _sanitize_json_value, _sha256_array

logger = logging.getLogger("transformation_portal.spatial_ai.orchestration.pipeline")

_SEGMENTATION_CACHE_SCHEMA_VERSION = "spatial-ai-segmentation-cache.v1"


def _segmentation_cache_paths(cache_dir: Path, cache_key: str) -> tuple[Path, Path]:
    root = cache_dir / cache_key[:2]
    return root / f"{cache_key}.npz", root / f"{cache_key}.json"


def _file_identity(path_value: Any) -> Optional[dict[str, Any]]:
    if not path_value:
        return None
    path = Path(str(path_value))
    if not path.is_file():
        return {
            "path": str(path),
            "exists": False,
            "sha256": None,
            "size": None,
            "mtime_ns": None,
        }
    try:
        stat = path.stat()
        return {
            "path": str(path),
            "exists": True,
            "sha256": _sha256_file_cached(str(path), int(stat.st_size), int(stat.st_mtime_ns)),
            "size": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
        }
    except OSError:
        return {
            "path": str(path),
            "exists": False,
            "sha256": None,
            "size": None,
            "mtime_ns": None,
        }


@lru_cache(maxsize=8)
def _sha256_file_cached(path: str, size: int, mtime_ns: int) -> str:
    del size, mtime_ns
    return _sha256_file(Path(path))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metadata_to_cache_dict(metadata: MaskMetadata) -> dict[str, Any]:
    return {
        "area": int(metadata.area),
        "bbox": list(metadata.bbox),
        "stability_score": float(metadata.stability_score),
        "material_label": metadata.material_label,
        "material_confidence": metadata.material_confidence,
        "is_empty": bool(metadata.is_empty),
    }


def _metadata_from_cache_dict(data: Mapping[str, Any]) -> MaskMetadata:
    return MaskMetadata(
        area=int(data["area"]),
        bbox=tuple(int(value) for value in data["bbox"]),
        stability_score=float(data["stability_score"]),
        material_label=data.get("material_label"),
        material_confidence=(float(data["material_confidence"]) if data.get("material_confidence") is not None else None),
        is_empty=bool(data.get("is_empty", False)),
    )


def _segmentation_result_checksum(result: SegmentationResult) -> str:
    digest = hashlib.sha256()
    for array in (result.masks, result.scores):
        contiguous = array if array.flags["C_CONTIGUOUS"] else np.ascontiguousarray(array)
        digest.update(str(contiguous.shape).encode("utf-8"))
        digest.update(str(contiguous.dtype).encode("utf-8"))
        digest.update(memoryview(contiguous.view(np.uint8).ravel()))
    digest.update(canonicalize_json([_metadata_to_cache_dict(item) for item in result.metadata]))
    return digest.hexdigest()


def _segmentation_mask_count(result: Any) -> int:
    masks = getattr(result, "masks", None)
    if masks is None:
        return 0
    shape = getattr(masks, "shape", None)
    if shape:
        return int(shape[0])
    try:
        return len(masks)
    except TypeError:
        return 0


def _build_segmentation_cache_key(
    *,
    image: np.ndarray,
    segmentation_cfg: Mapping[str, Any],
    device: str,
) -> tuple[str, dict[str, Any]]:
    model_cfg = segmentation_cfg.get("model", {}) if isinstance(segmentation_cfg.get("model"), Mapping) else {}
    sanitized_model_cfg = dict(model_cfg)
    checkpoint_path = sanitized_model_cfg.get("checkpoint_path")
    sanitized_model_cfg["checkpoint"] = _file_identity(checkpoint_path)
    sanitized_model_cfg.pop("checkpoint_path", None)
    payload = {
        "schema_version": _SEGMENTATION_CACHE_SCHEMA_VERSION,
        "image_hash": _sha256_array(image),
        "image_shape": list(image.shape),
        "image_dtype": str(image.dtype),
        "backend": segmentation_cfg.get("backend", "sam2"),
        "device": device,
        "model": _sanitize_json_value(sanitized_model_cfg),
        "generator": dict(segmentation_cfg.get("generator", {}) or {}),
        "material_classification": bool(segmentation_cfg.get("material_classification", False)),
        "material_confidence_threshold": float(segmentation_cfg.get("material_confidence_threshold", 0.3)),
        "tiling": _sanitize_json_value(segmentation_cfg.get("tiling", {}) or {}),
    }
    cache_key = hashlib.sha256(canonicalize_json(payload)).hexdigest()
    return cache_key, payload


def _read_segmentation_cache(
    *,
    cache_dir: Path,
    cache_key: str,
    key_payload: Mapping[str, Any],
) -> Optional[SegmentationResult]:
    masks_path, metadata_path = _segmentation_cache_paths(cache_dir, cache_key)
    if not masks_path.is_file() or not metadata_path.is_file():
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("schema_version") != _SEGMENTATION_CACHE_SCHEMA_VERSION:
            return None
        if metadata.get("cache_key") != cache_key or metadata.get("key_payload") != dict(key_payload):
            return None
        with np.load(masks_path, allow_pickle=False) as data:
            masks = np.asarray(data["masks"])
            scores = np.asarray(data["scores"])
        mask_metadata = [_metadata_from_cache_dict(item) for item in metadata.get("metadata", [])]
        result = SegmentationResult(
            masks=masks.astype(bool, copy=False), scores=scores.astype(np.float32), metadata=mask_metadata
        )
        if _segmentation_result_checksum(result) != metadata.get("result_sha256"):
            return None
        return result
    except Exception as exc:
        logger.debug("Ignoring invalid spatial segmentation cache entry %s: %s", cache_key, exc)
        return None


def _write_segmentation_cache(
    *,
    cache_dir: Path,
    cache_key: str,
    key_payload: Mapping[str, Any],
    result: SegmentationResult,
) -> None:
    if _segmentation_mask_count(result) == 0:
        return
    if not isinstance(result.masks, np.ndarray) or not isinstance(result.scores, np.ndarray):
        return
    masks_path, metadata_path = _segmentation_cache_paths(cache_dir, cache_key)
    masks_path.parent.mkdir(parents=True, exist_ok=True)
    temp_npz = masks_path.with_suffix(".npz.tmp")
    temp_json = metadata_path.with_suffix(".json.tmp")
    try:
        with temp_npz.open("wb") as handle:
            np.savez_compressed(handle, masks=result.masks, scores=result.scores)
        metadata = {
            "schema_version": _SEGMENTATION_CACHE_SCHEMA_VERSION,
            "cache_key": cache_key,
            "key_payload": dict(key_payload),
            "metadata": [_metadata_to_cache_dict(item) for item in result.metadata],
            "result_sha256": _segmentation_result_checksum(result),
        }
        with temp_json.open("w", encoding="utf-8") as handle:
            dump_json(metadata, handle, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
        temp_npz.replace(masks_path)
        temp_json.replace(metadata_path)
    finally:
        for temp_path in (temp_npz, temp_json):
            if temp_path.exists():
                with contextlib.suppress(OSError):
                    temp_path.unlink()


__all__ = [
    "_SEGMENTATION_CACHE_SCHEMA_VERSION",
    "_build_segmentation_cache_key",
    "_file_identity",
    "_metadata_from_cache_dict",
    "_metadata_to_cache_dict",
    "_read_segmentation_cache",
    "_segmentation_cache_paths",
    "_segmentation_mask_count",
    "_segmentation_result_checksum",
    "_sha256_file",
    "_sha256_file_cached",
    "_write_segmentation_cache",
]
