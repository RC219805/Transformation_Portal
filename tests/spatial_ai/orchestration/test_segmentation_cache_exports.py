from __future__ import annotations

import importlib
import json
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

import numpy as np
import pytest

from transformation_portal.spatial_ai.segmentation.contracts import MaskMetadata, SegmentationResult

pytestmark = pytest.mark.unit


def _import_modules() -> tuple[ModuleType, ModuleType, ModuleType]:
    pipeline_mod = importlib.import_module("transformation_portal.spatial_ai.orchestration.pipeline")
    cache_mod = importlib.import_module("transformation_portal.spatial_ai.orchestration.segmentation_cache")
    artifact_utils_mod = importlib.import_module("transformation_portal.spatial_ai.orchestration.artifact_utils")
    return pipeline_mod, cache_mod, artifact_utils_mod


def _make_image() -> np.ndarray:
    return np.arange(12, dtype=np.float32).reshape(2, 2, 3)


def _make_config(checkpoint_path: Path | None = None) -> dict[str, Any]:
    model: dict[str, Any] = {
        "size": "large",
        "repo_id": "facebook/sam2.1-hiera-large",
        "revision": "a" * 40,
        "prefer_hf_pipeline": True,
    }
    if checkpoint_path is not None:
        model["checkpoint_path"] = str(checkpoint_path)
    return {
        "backend": "sam2",
        "model": model,
        "generator": {"points_per_side": np.int64(8)},
        "material_classification": True,
        "material_confidence_threshold": np.float32(0.35),
        "tiling": {"enabled": True, "tile_size": np.int64(512)},
    }


def _make_result() -> SegmentationResult:
    masks = np.zeros((2, 2, 2), dtype=bool)
    masks[0, 0, 0] = True
    masks[1, :, 1] = True
    return SegmentationResult(
        masks=masks,
        scores=np.array([0.95, 0.55], dtype=np.float32),
        metadata=[
            MaskMetadata(
                area=1,
                bbox=(0, 0, 1, 1),
                stability_score=0.91,
                material_label="glass",
                material_confidence=0.82,
            ),
            MaskMetadata(
                area=2,
                bbox=(1, 0, 1, 2),
                stability_score=0.73,
                material_label="stone",
                material_confidence=0.64,
                is_empty=False,
            ),
        ],
    )


def _write_cache_entry(cache_mod: ModuleType, tmp_path: Path) -> tuple[Path, str, dict[str, Any], SegmentationResult]:
    checkpoint_path = tmp_path / "sam2.pt"
    checkpoint_path.write_bytes(b"checkpoint")
    cache_key, key_payload = cache_mod._build_segmentation_cache_key(
        image=_make_image(),
        segmentation_cfg=_make_config(checkpoint_path),
        device="cpu",
    )
    result = _make_result()
    cache_dir = tmp_path / ".cache" / "spatial_ai" / "segmentation"
    cache_mod._write_segmentation_cache(
        cache_dir=cache_dir,
        cache_key=cache_key,
        key_payload=key_payload,
        result=result,
    )
    return cache_dir, cache_key, key_payload, result


def test_segmentation_cache_helper_identity_is_preserved_across_import_surfaces() -> None:
    pipeline_mod, cache_mod, artifact_utils_mod = _import_modules()

    for helper_name in (
        "_build_segmentation_cache_key",
        "_read_segmentation_cache",
        "_write_segmentation_cache",
        "_segmentation_mask_count",
        "_segmentation_cache_paths",
        "_segmentation_result_checksum",
        "_metadata_to_cache_dict",
        "_metadata_from_cache_dict",
        "_file_identity",
        "_sha256_file",
        "_sha256_file_cached",
    ):
        assert getattr(pipeline_mod, helper_name) is getattr(cache_mod, helper_name)

    assert pipeline_mod._sha256_array is artifact_utils_mod._sha256_array
    assert pipeline_mod._sanitize_json_value is artifact_utils_mod._sanitize_json_value
    assert pipeline_mod._SEGMENTATION_CACHE_SCHEMA_VERSION == cache_mod._SEGMENTATION_CACHE_SCHEMA_VERSION


def test_direct_segmentation_cache_roundtrip_preserves_payload(tmp_path: Path) -> None:
    _, cache_mod, _ = _import_modules()
    cache_dir, cache_key, key_payload, result = _write_cache_entry(cache_mod, tmp_path)
    masks_path, metadata_path = cache_mod._segmentation_cache_paths(cache_dir, cache_key)

    assert masks_path == cache_dir / cache_key[:2] / f"{cache_key}.npz"
    assert metadata_path == cache_dir / cache_key[:2] / f"{cache_key}.json"
    assert masks_path.is_file()
    assert metadata_path.is_file()

    cached = cache_mod._read_segmentation_cache(
        cache_dir=cache_dir,
        cache_key=cache_key,
        key_payload=key_payload,
    )

    assert cached is not None
    assert np.array_equal(cached.masks, result.masks)
    assert np.allclose(cached.scores, result.scores)
    assert [cache_mod._metadata_to_cache_dict(item) for item in cached.metadata] == [
        cache_mod._metadata_to_cache_dict(item) for item in result.metadata
    ]
    assert cache_mod._segmentation_result_checksum(cached) == cache_mod._segmentation_result_checksum(result)

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["schema_version"] == cache_mod._SEGMENTATION_CACHE_SCHEMA_VERSION
    assert metadata["cache_key"] == cache_key
    assert metadata["key_payload"] == key_payload
    assert metadata["result_sha256"] == cache_mod._segmentation_result_checksum(result)


@pytest.mark.parametrize(
    "mutate_metadata",
    [
        lambda metadata, cache_key, key_payload: metadata.update({"schema_version": "invalid"}),
        lambda metadata, cache_key, key_payload: metadata.update({"cache_key": f"{cache_key}-stale"}),
        lambda metadata, cache_key, key_payload: metadata.update({"result_sha256": "0" * 64}),
    ],
)
def test_direct_segmentation_cache_invalid_entries_return_none_without_raising(
    mutate_metadata: Callable[[dict[str, Any], str, dict[str, Any]], None],
    tmp_path: Path,
) -> None:
    _, cache_mod, _ = _import_modules()
    cache_dir, cache_key, key_payload, _ = _write_cache_entry(cache_mod, tmp_path)
    _, metadata_path = cache_mod._segmentation_cache_paths(cache_dir, cache_key)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    mutate_metadata(metadata, cache_key, key_payload)
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    assert (
        cache_mod._read_segmentation_cache(
            cache_dir=cache_dir,
            cache_key=cache_key,
            key_payload=key_payload,
        )
        is None
    )


def test_artifact_utils_preserve_array_hash_and_sanitizer_behavior() -> None:
    _, _, artifact_utils_mod = _import_modules()
    array = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    non_contiguous = array[:, ::-1, :]

    assert artifact_utils_mod._sha256_array(non_contiguous) == artifact_utils_mod._sha256_array(
        np.ascontiguousarray(non_contiguous)
    )

    @dataclass
    class Payload:
        values: Any
        nested: Any
        non_finite: float

    assert artifact_utils_mod._sanitize_json_value(
        Payload(
            values=np.array([np.float32(1.5), np.float32(2.5)]),
            nested={1: (np.int64(7),)},
            non_finite=float("nan"),
        )
    ) == {
        "values": [1.5, 2.5],
        "nested": {"1": [7]},
        "non_finite": None,
    }
