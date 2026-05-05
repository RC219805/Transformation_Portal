"""Regression tests for the Phase 3 segmentation package split."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


def test_segmentation_backend_shim_reexports_extracted_modules() -> None:
    import transformation_portal.lux_depth_v3.segmentation_backend as legacy
    from transformation_portal.lux_depth_v3.segmentation import _cache, efficient_sam, registry, sam2, sam_vit_h, stub

    assert legacy.StubBackend is stub.StubBackend
    assert legacy.EfficientSAMBackend is efficient_sam.EfficientSAMBackend
    assert legacy.SAM2SegmentationBackend is sam2.SAM2SegmentationBackend
    assert legacy.SAMVitHBackend is sam_vit_h.SAMVitHBackend
    assert legacy.segment_materials is registry.segment_materials
    assert legacy._CACHE_MASK_CHECKSUM_CHUNK_SIZE == _cache._CACHE_MASK_CHECKSUM_CHUNK_SIZE
    assert legacy._LAST_SEGMENTATION_RUNTIME_METADATA is registry._LAST_SEGMENTATION_RUNTIME_METADATA


def test_segmentation_backend_shim_propagates_legacy_monkeypatches(monkeypatch: pytest.MonkeyPatch) -> None:
    import transformation_portal.lux_depth_v3.segmentation_backend as legacy
    from transformation_portal.lux_depth_v3.segmentation import _cache, efficient_sam, registry, sam2, sam_vit_h, stub

    def fake_backend(*_args, **_kwargs):
        return legacy.StubBackend()

    fake_open_clip = object()
    fake_spatial_backend = object()
    fake_logger = object()

    monkeypatch.setattr(legacy, "_get_backend_instance", fake_backend)
    monkeypatch.setattr(legacy, "open_clip", fake_open_clip)
    monkeypatch.setattr(legacy, "SpatialSAM2Backend", fake_spatial_backend)
    monkeypatch.setattr(legacy, "logger", fake_logger)

    assert registry._get_backend_instance is fake_backend
    assert efficient_sam.open_clip is fake_open_clip
    assert sam2.SpatialSAM2Backend is fake_spatial_backend
    assert _cache.logger is fake_logger
    assert efficient_sam.logger is fake_logger
    assert registry.logger is fake_logger
    assert sam2.logger is fake_logger
    assert sam_vit_h.logger is fake_logger
    assert stub.logger is fake_logger
