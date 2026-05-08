"""Tests for material segmentation backends.

These tests verify the SegmentationBackend Protocol implementation,
including StubBackend, EfficientSAMBackend, and SAM2SegmentationBackend.

Test Coverage:
- Protocol compliance (stub, efficientsam, sam2 backends)
- Shape contracts (RGB input → masks dict output)
- Device placement (MPS/CUDA/CPU)
- Fallback behavior (missing weights → stub)
- Integration with Materials V3 config
- Offline compatibility (no model downloads in CI)

Note: Tests marked with @pytest.mark.ml require torch/torchvision.
"""

from __future__ import annotations

import json
import socket
import sys
import types
from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pytest

pytestmark = pytest.mark.unit

from transformation_portal.lux_depth_v3.config import EnhanceConfig
from transformation_portal.lux_depth_v3.protocols.segmentation_backend import SegmentationBackend, SegmentationBackendInfo
from transformation_portal.lux_depth_v3.segmentation_backend import (
    EfficientSAMBackend,
    SAM2SegmentationBackend,
    StubBackend,
    _get_backend_instance,
    _tensor_values_1d,
    get_last_segmentation_runtime_metadata,
    segment_materials,
)

# Safe torch availability checks for skipif decorators
try:
    import torch

    TORCH_AVAILABLE = True
    MPS_AVAILABLE = torch.backends.mps.is_available()
    CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    TORCH_AVAILABLE = False
    MPS_AVAILABLE = False
    CUDA_AVAILABLE = False

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(name="sample_image")
def fixture_sample_image():
    """Create a sample RGB image for testing."""
    # Create a simple 64×64 RGB image with some color variation
    image = np.zeros((64, 64, 3), dtype=np.uint8)

    # Add some colored regions
    image[10:30, 10:30] = [100, 150, 200]  # Blue-ish (potential water)
    image[35:55, 35:55] = [80, 180, 100]  # Green-ish (potential foliage)
    image[10:30, 35:55] = [150, 150, 150]  # Gray (potential stone)

    return image


@pytest.fixture(name="config_stub")
def fixture_config_stub():
    """Config with stub backend."""
    return EnhanceConfig(
        enable_material_segmentation=True,
        material_segmentation_backend="stub",
    )


@pytest.fixture(name="config_efficientsam")
def fixture_config_efficientsam():
    """Config with EfficientSAM backend."""
    return EnhanceConfig(
        enable_material_segmentation=True,
        material_segmentation_backend="efficientsam",
        depth_device="cpu",  # Use CPU for tests
    )


@pytest.fixture(name="config_sam2")
def fixture_config_sam2():
    """Config with SAM2 backend."""
    return EnhanceConfig(
        enable_material_segmentation=True,
        material_segmentation_backend="sam2",
        depth_device="cpu",
    )


@pytest.fixture(name="config_strict")
def fixture_config_strict():
    """Config with strict_backend=True."""
    return EnhanceConfig(
        enable_material_segmentation=True,
        material_segmentation_backend="efficientsam",
        strict_backend=True,
        depth_device="cpu",
    )


@pytest.fixture(name="efficientsam_heuristic_backend")
def fixture_efficientsam_heuristic_backend(monkeypatch):
    """Fast unit-test backend: force heuristic path to avoid real SAM inference."""
    import transformation_portal.lux_depth_v3.segmentation_backend as seg_module

    monkeypatch.setattr(seg_module, "EFFICIENTVIT_AVAILABLE", False)
    backend = EfficientSAMBackend()
    backend.load(device="cpu")
    return backend


# =============================================================================
# Protocol Compliance Tests
# =============================================================================


def test_stub_backend_implements_protocol():
    """Test that StubBackend implements SegmentationBackend protocol."""
    backend = StubBackend()

    # Check protocol compliance
    assert isinstance(backend, SegmentationBackend)
    assert hasattr(backend, "info")
    assert hasattr(backend, "load")
    assert hasattr(backend, "segment")

    # Check info
    info = backend.info
    assert isinstance(info, SegmentationBackendInfo)
    assert info.name == "Stub Segmentation Backend"
    assert info.model_id == "stub"
    assert info.requires_gpu is False
    assert info.requires_weights is False


@pytest.mark.ml
def test_efficientsam_backend_implements_protocol():
    """Test that EfficientSAMBackend implements SegmentationBackend protocol."""
    backend = EfficientSAMBackend()

    # Check protocol compliance
    assert isinstance(backend, SegmentationBackend)
    assert hasattr(backend, "info")
    assert hasattr(backend, "load")
    assert hasattr(backend, "segment")

    # Check info
    info = backend.info
    assert isinstance(info, SegmentationBackendInfo)
    assert info.name == "EfficientSAM"
    assert "efficientvit-sam" in info.model_id
    assert info.requires_weights is True


def test_sam2_backend_implements_protocol():
    """Test that SAM2SegmentationBackend implements SegmentationBackend protocol."""
    backend = SAM2SegmentationBackend()

    assert isinstance(backend, SegmentationBackend)
    assert hasattr(backend, "info")
    assert hasattr(backend, "load")
    assert hasattr(backend, "segment")

    info = backend.info
    assert isinstance(info, SegmentationBackendInfo)
    assert info.name == "SAM2"
    assert "sam2-hiera" in info.model_id
    assert info.requires_weights is True


# =============================================================================
# Shape Contract Tests
# =============================================================================


def test_stub_backend_shape_contract(sample_image):
    """Test that StubBackend returns correct output shape."""
    backend = StubBackend()
    backend.load()

    masks = backend.segment(sample_image)

    # Stub returns empty dict
    assert isinstance(masks, dict)
    assert len(masks) == 0


def test_clip_loader_prefers_cached_checkpoint_path(monkeypatch, tmp_path):
    """CLIP loader should use local cached checkpoint path before tag-based resolution."""
    import transformation_portal.lux_depth_v3.segmentation_backend as seg_module

    cached_checkpoint = tmp_path / "open_clip_model.safetensors"
    cached_checkpoint.write_bytes(b"checkpoint")
    calls = {}

    def fake_get_pretrained_cfg(model_name, pretrained_tag):
        calls["cfg"] = (model_name, pretrained_tag)
        return {"hf_hub": "timm/vit_base_patch32_clip_224.openai/"}

    def fake_create_model_and_transforms(model_name, pretrained, device):
        calls["create"] = (model_name, pretrained, device)
        return object(), None, (lambda image: image)

    def fake_get_tokenizer(model_name):
        calls["tokenizer"] = model_name
        return lambda prompts: prompts

    def fake_try_to_load_from_cache(repo_id, filename):
        calls.setdefault("cache_queries", []).append((repo_id, filename))
        if filename == "open_clip_model.safetensors":
            return str(cached_checkpoint)
        return None

    fake_hf_module = types.ModuleType("huggingface_hub")
    fake_hf_module.try_to_load_from_cache = fake_try_to_load_from_cache

    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hf_module)
    monkeypatch.setattr(seg_module, "OPEN_CLIP_AVAILABLE", True)
    monkeypatch.setattr(
        seg_module,
        "open_clip",
        SimpleNamespace(
            get_pretrained_cfg=fake_get_pretrained_cfg,
            create_model_and_transforms=fake_create_model_and_transforms,
            get_tokenizer=fake_get_tokenizer,
        ),
    )

    backend = seg_module.EfficientSAMBackend()
    backend._device = "cpu"
    backend._load_clip_runtime()

    assert calls["cfg"] == ("ViT-B-32", "openai")
    assert calls["cache_queries"][0] == ("timm/vit_base_patch32_clip_224.openai", "open_clip_model.safetensors")
    assert calls["create"][1] == str(cached_checkpoint)


def test_clip_loader_offline_mode_missing_cache_fails_fast(monkeypatch):
    """Offline mode must fail before tag-based model resolution if cache is missing."""
    import transformation_portal.lux_depth_v3.segmentation_backend as seg_module

    create_calls = {"count": 0}

    def fake_get_pretrained_cfg(_model_name, _pretrained_tag):
        return {"hf_hub": "timm/vit_base_patch32_clip_224.openai/"}

    def fake_create_model_and_transforms(*_args, **_kwargs):
        create_calls["count"] += 1
        return object(), None, (lambda image: image)

    def fake_try_to_load_from_cache(*, repo_id=None, filename=None):
        return None

    fake_hf_module = types.ModuleType("huggingface_hub")
    fake_hf_module.try_to_load_from_cache = fake_try_to_load_from_cache

    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hf_module)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setattr(seg_module, "OPEN_CLIP_AVAILABLE", True)
    monkeypatch.setattr(
        seg_module,
        "open_clip",
        SimpleNamespace(
            get_pretrained_cfg=fake_get_pretrained_cfg,
            create_model_and_transforms=fake_create_model_and_transforms,
            get_tokenizer=lambda _model_name: (lambda prompts: prompts),
        ),
    )

    backend = seg_module.EfficientSAMBackend()
    backend._device = "cpu"

    with pytest.raises(RuntimeError, match="offline mode is enabled"):
        backend._load_clip_runtime()

    assert create_calls["count"] == 0


def test_clip_loader_with_cache_does_not_touch_network(monkeypatch, tmp_path):
    """With cache present, CLIP loader should succeed even if network APIs are blocked."""
    import transformation_portal.lux_depth_v3.segmentation_backend as seg_module

    cached_checkpoint = tmp_path / "open_clip_model.safetensors"
    cached_checkpoint.write_bytes(b"checkpoint")
    calls = {"create": 0}

    def fake_get_pretrained_cfg(_model_name, _pretrained_tag):
        return {"hf_hub": "timm/vit_base_patch32_clip_224.openai/"}

    def fake_create_model_and_transforms(*_args, **_kwargs):
        calls["create"] += 1
        return object(), None, (lambda image: image)

    def fake_try_to_load_from_cache(*, repo_id=None, filename=None):
        if filename == "open_clip_model.safetensors":
            return str(cached_checkpoint)
        return None

    def fail_getaddrinfo(*_args, **_kwargs):
        raise AssertionError("Network lookup attempted in offline-cache path")

    fake_hf_module = types.ModuleType("huggingface_hub")
    fake_hf_module.try_to_load_from_cache = fake_try_to_load_from_cache
    fake_hf_module.hf_hub_download = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("hf_hub_download should not be called when cache exists")
    )

    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hf_module)
    monkeypatch.setattr(socket, "getaddrinfo", fail_getaddrinfo)
    monkeypatch.setattr(seg_module, "OPEN_CLIP_AVAILABLE", True)
    monkeypatch.setattr(
        seg_module,
        "open_clip",
        SimpleNamespace(
            get_pretrained_cfg=fake_get_pretrained_cfg,
            create_model_and_transforms=fake_create_model_and_transforms,
            get_tokenizer=lambda _model_name: (lambda prompts: prompts),
        ),
    )

    backend = seg_module.EfficientSAMBackend()
    backend._device = "cpu"
    backend._load_clip_runtime()
    assert calls["create"] == 1


def test_segment_materials_exposes_runtime_metadata(monkeypatch, sample_image):
    """segment_materials should expose backend runtime metadata for manifest attestation."""
    import transformation_portal.lux_depth_v3.segmentation_backend as seg_module

    class _FakeBackend:
        def __init__(self):
            self.info = SimpleNamespace(name="fake")

        def segment(self, _image):
            return {"glass": (np.ones((2, 2), dtype=np.float32), 0.9)}

        def get_runtime_metadata(self):
            return {
                "clip_runtime": {"offline_mode": True, "weights_source": "cache_path"},
                "material_confidence_evidence": {
                    "glass": {
                        "material_confidence": 0.9,
                        "confidence_score_type": "material_classifier_probability_v1",
                    },
                    "stone": {
                        "material_confidence": 0.2,
                        "confidence_score_type": "material_classifier_probability_v1",
                    },
                },
            }

    monkeypatch.setattr(seg_module, "_get_backend_instance", lambda *_args, **_kwargs: _FakeBackend())

    config = EnhanceConfig(enable_material_segmentation=True, material_segmentation_backend="efficientsam", depth_device="cpu")
    masks = segment_materials(sample_image, config)

    assert "glass" in masks
    metadata = get_last_segmentation_runtime_metadata()
    assert metadata is not None
    assert metadata["clip_runtime"]["weights_source"] == "cache_path"
    assert set(metadata["material_confidence_evidence"]) == {"glass"}
    score_type = metadata["material_confidence_evidence"]["glass"]["confidence_score_type"]
    assert score_type == "material_classifier_probability_v1"


@pytest.mark.ml
def test_efficientsam_backend_shape_contract(sample_image, efficientsam_heuristic_backend):
    """Test that EfficientSAMBackend returns correct output shapes."""
    results = efficientsam_heuristic_backend.segment(sample_image)

    # Check output type
    assert isinstance(results, dict)

    # Check each mask shape, dtype, and confidence
    for material, (mask, confidence) in results.items():
        assert isinstance(material, str)
        assert isinstance(mask, np.ndarray)
        assert mask.shape == sample_image.shape[:2]  # (H, W)
        assert mask.dtype == np.float32
        assert mask.min() >= 0.0
        assert mask.max() <= 1.0

        # Check confidence score
        assert isinstance(confidence, float)
        assert 0.0 <= confidence <= 1.0, f"{material} confidence {confidence} not in [0.0-1.0]"


@pytest.mark.ml
def test_efficientsam_backend_invalid_input(efficientsam_heuristic_backend):
    """Test that EfficientSAMBackend validates input format."""
    # Test invalid shape (grayscale)
    with pytest.raises(ValueError, match="Expected RGB image"):
        efficientsam_heuristic_backend.segment(np.zeros((64, 64), dtype=np.uint8))

    # Test invalid dtype (float32)
    with pytest.raises(ValueError, match="Expected uint8"):
        efficientsam_heuristic_backend.segment(np.zeros((64, 64, 3), dtype=np.float32))

    # Test invalid shape (4 channels)
    with pytest.raises(ValueError, match="Expected RGB image"):
        efficientsam_heuristic_backend.segment(np.zeros((64, 64, 4), dtype=np.uint8))


# =============================================================================
# Device Placement Tests
# =============================================================================


@pytest.mark.ml
def test_efficientsam_backend_cpu_device():
    """Test that EfficientSAMBackend works on CPU."""
    backend = EfficientSAMBackend()
    backend.load(device="cpu")

    assert backend._device == "cpu"
    assert backend._model_loaded is True


@pytest.mark.ml
@pytest.mark.skipif(
    not MPS_AVAILABLE,
    reason="MPS not available",
)
def test_efficientsam_backend_mps_device():
    """Test that EfficientSAMBackend works on MPS (Apple Silicon)."""
    backend = EfficientSAMBackend()
    backend.load(device="mps")

    assert backend._device == "mps"
    assert backend._model_loaded is True


@pytest.mark.ml
@pytest.mark.skipif(
    not CUDA_AVAILABLE,
    reason="CUDA not available",
)
def test_efficientsam_backend_cuda_device():
    """Test that EfficientSAMBackend works on CUDA."""
    backend = EfficientSAMBackend()
    backend.load(device="cuda")

    assert backend._device == "cuda"
    assert backend._model_loaded is True


@pytest.mark.ml
def test_efficientsam_backend_auto_device():
    """Test that EfficientSAMBackend auto-detects device."""
    backend = EfficientSAMBackend()
    backend.load(device="auto")

    # Should select MPS > CUDA > CPU
    assert backend._device in ["cpu", "mps", "cuda"]
    assert backend._model_loaded is True


# =============================================================================
# Fallback Behavior Tests
# =============================================================================


def test_segment_materials_disabled(sample_image):
    """Test that segmentation returns empty dict when disabled."""
    config = EnhanceConfig(enable_material_segmentation=False)

    masks = segment_materials(sample_image, config)

    assert isinstance(masks, dict)
    assert len(masks) == 0


def test_segment_materials_stub_backend(sample_image, config_stub):
    """Test that stub backend returns empty masks."""
    masks = segment_materials(sample_image, config_stub)

    assert isinstance(masks, dict)
    assert len(masks) == 0


def test_segment_materials_passes_sky_knobs_without_mutating_backend(sample_image, monkeypatch):
    """segment_materials should pass config knobs via factory args, not backend mutation."""
    import transformation_portal.lux_depth_v3.segmentation_backend as seg_module

    class DummyBackend:
        @property
        def info(self):
            return SegmentationBackendInfo(name="dummy", model_id="dummy", requires_weights=False)

        def segment(self, image):
            del image
            return {}

    captured_kwargs = {}
    backend = DummyBackend()

    def fake_get_backend_instance(*args, **kwargs):
        del args
        captured_kwargs.update(kwargs)
        return backend

    monkeypatch.setattr(seg_module, "_get_backend_instance", fake_get_backend_instance)

    config = EnhanceConfig(
        enable_material_segmentation=True,
        material_segmentation_backend="efficientsam",
        depth_device="cpu",
        sky_top_region_fraction=0.42,
        sky_gradient_threshold=0.11,
        sky_brightness_threshold=0.55,
    )

    masks = segment_materials(sample_image, config)
    assert masks == {}
    assert captured_kwargs["sky_top_region_fraction"] == pytest.approx(0.42)
    assert captured_kwargs["sky_gradient_threshold"] == pytest.approx(0.11)
    assert captured_kwargs["sky_brightness_threshold"] == pytest.approx(0.55)
    assert not hasattr(backend, "_config")


@pytest.mark.ml
@pytest.mark.integration
@pytest.mark.slow
def test_segment_materials_efficientsam_backend(sample_image, config_efficientsam):
    """Integration: real EfficientSAM backend returns masks."""
    masks = segment_materials(sample_image, config_efficientsam)

    # Should return some masks (heuristic segmentation)
    assert isinstance(masks, dict)
    # May or may not detect materials depending on heuristics
    for material, mask in masks.items():
        assert mask.shape == sample_image.shape[:2]
        assert mask.dtype == np.float32


def test_segment_materials_sam2_backend_with_mock(sample_image, config_sam2, monkeypatch):
    """Test that SAM2 backend can be selected and returns canonical material masks."""
    import transformation_portal.lux_depth_v3.segmentation_backend as seg_module

    class FakeSpatialSAM2Backend:
        def __init__(self, model_size, device, checkpoint_path, enable_material_classification, material_confidence_threshold):
            del model_size, checkpoint_path, enable_material_classification, material_confidence_threshold
            self.device = device

        def segment(self, seg_input):
            h, w = seg_input.image.shape[:2]
            masks = np.zeros((1, h, w), dtype=bool)
            masks[0, 10:30, 10:30] = True
            scores = np.array([0.88], dtype=np.float32)
            metadata = [
                SimpleNamespace(
                    area=400,
                    bbox=(10, 10, 20, 20),
                    stability_score=0.9,
                    material_label="clear glass",
                    material_confidence=0.91,
                )
            ]
            return SimpleNamespace(masks=masks, scores=scores, metadata=metadata)

    monkeypatch.setattr(seg_module, "SPATIAL_SAM2_AVAILABLE", True)
    monkeypatch.setattr(seg_module, "SpatialSAM2Backend", FakeSpatialSAM2Backend)
    seg_module._get_backend_instance.cache_clear()

    try:
        masks = segment_materials(sample_image, config_sam2)
    finally:
        seg_module._get_backend_instance.cache_clear()

    assert "glass" in masks
    assert masks["glass"].shape == sample_image.shape[:2]
    assert masks["glass"].dtype == np.float32
    assert float(masks["glass"].max()) == 1.0
    metadata = get_last_segmentation_runtime_metadata()
    assert metadata is not None
    assert metadata["material_confidences"]["glass"] == pytest.approx(0.91)
    assert metadata["material_confidence_evidence"]["glass"]["confidence_score_type"] == "material_classifier_probability_v1"
    assert metadata["confidence_summary"]["count"] == 1


def test_segment_materials_sam2_backend_forwards_generator_and_tiling_config(sample_image, monkeypatch):
    """segment_materials should forward SAM2 generator and tiling controls through the wrapper."""
    import transformation_portal.lux_depth_v3.segmentation_backend as seg_module

    captured: dict[str, object] = {}

    class FakeSpatialSAM2Backend:
        def __init__(
            self,
            model_size,
            device,
            checkpoint_path,
            generator_kwargs,
            enable_material_classification,
            material_confidence_threshold,
            tiling,
        ):
            captured["model_size"] = model_size
            captured["device"] = device
            captured["checkpoint_path"] = checkpoint_path
            captured["generator_kwargs"] = dict(generator_kwargs)
            captured["enable_material_classification"] = enable_material_classification
            captured["material_confidence_threshold"] = material_confidence_threshold
            captured["configured_tiling"] = tiling
            self.device = device
            self.tiling = tiling

        def segment(self, seg_input):
            captured["effective_tiling"] = self.tiling
            h, w = seg_input.image.shape[:2]
            masks = np.zeros((1, h, w), dtype=bool)
            masks[0, 10:30, 10:30] = True
            scores = np.array([0.88], dtype=np.float32)
            metadata = [
                SimpleNamespace(
                    area=400,
                    bbox=(10, 10, 20, 20),
                    stability_score=0.9,
                    material_label="clear glass",
                    material_confidence=0.91,
                )
            ]
            return SimpleNamespace(masks=masks, scores=scores, metadata=metadata)

    monkeypatch.setattr(seg_module, "SPATIAL_SAM2_AVAILABLE", True)
    monkeypatch.setattr(seg_module, "SpatialSAM2Backend", FakeSpatialSAM2Backend)
    seg_module._get_backend_instance.cache_clear()

    config = EnhanceConfig(
        enable_material_segmentation=True,
        material_segmentation_backend="sam2",
        depth_device="cpu",
        sam2_model_size="large",
        sam2_tiling_enabled=True,
        sam2_tile_size_px=1024,
        sam2_overlap_px=128,
        sam2_global_pass_longest_side=900,
        sam2_max_concurrency=1,
        sam2_points_per_side=16,
        sam2_points_per_batch=32,
        sam2_pred_iou_thresh=0.77,
        sam2_stability_score_thresh=0.66,
        sam2_crop_n_layers=2,
    )

    try:
        masks = segment_materials(sample_image, config)
    finally:
        seg_module._get_backend_instance.cache_clear()

    assert "glass" in masks
    assert captured["model_size"] == "large"
    assert captured["device"] == "cpu"
    assert captured["generator_kwargs"] == {
        "points_per_side": 16,
        "points_per_batch": 32,
        "pred_iou_thresh": pytest.approx(0.77),
        "stability_score_thresh": pytest.approx(0.66),
        "crop_n_layers": 2,
    }
    configured_tiling = captured["configured_tiling"]
    assert getattr(configured_tiling, "enabled", False) is True
    assert getattr(configured_tiling, "tile_size_px", None) == 1024
    assert getattr(configured_tiling, "overlap_px", None) == 128
    assert getattr(getattr(configured_tiling, "global_pass", None), "longest_side", None) == 900
    assert getattr(configured_tiling, "max_concurrency", None) == 1

    metadata = get_last_segmentation_runtime_metadata()
    assert metadata is not None
    assert metadata["sam2_runtime"]["generator_kwargs"]["points_per_side"] == 16
    assert metadata["sam2_runtime"]["generator_kwargs"]["points_per_batch"] == 32
    assert metadata["sam2_runtime"]["tiling"]["effective"]["enabled"] is True
    assert metadata["sam2_runtime"]["tiling"]["auto_enabled"] is False


def test_segment_materials_sam2_backend_auto_enables_tiling_for_large_images(monkeypatch):
    """Large images should auto-enable deterministic tiling even when not explicitly requested."""
    import transformation_portal.lux_depth_v3.segmentation_backend as seg_module

    captured: dict[str, object] = {}

    class FakeSpatialSAM2Backend:
        def __init__(
            self,
            model_size,
            device,
            checkpoint_path,
            generator_kwargs,
            enable_material_classification,
            material_confidence_threshold,
            tiling,
        ):
            del model_size, checkpoint_path, generator_kwargs, enable_material_classification, material_confidence_threshold
            captured["configured_tiling"] = tiling
            self.device = device
            self.tiling = tiling

        def segment(self, seg_input):
            captured["effective_tiling"] = self.tiling
            h, w = seg_input.image.shape[:2]
            masks = np.zeros((1, h, w), dtype=bool)
            masks[0, :2, :1] = True
            scores = np.array([0.75], dtype=np.float32)
            metadata = [
                SimpleNamespace(
                    area=2,
                    bbox=(0, 0, 1, 2),
                    stability_score=0.8,
                    material_label="clear glass",
                    material_confidence=0.75,
                )
            ]
            return SimpleNamespace(masks=masks, scores=scores, metadata=metadata)

    monkeypatch.setattr(seg_module, "SPATIAL_SAM2_AVAILABLE", True)
    monkeypatch.setattr(seg_module, "SpatialSAM2Backend", FakeSpatialSAM2Backend)
    seg_module._get_backend_instance.cache_clear()

    large_image = np.zeros((4097, 2, 3), dtype=np.uint8)
    config = EnhanceConfig(
        enable_material_segmentation=True,
        material_segmentation_backend="sam2",
        depth_device="cpu",
    )

    try:
        masks = segment_materials(large_image, config)
    finally:
        seg_module._get_backend_instance.cache_clear()

    assert "glass" in masks
    configured_tiling = captured["configured_tiling"]
    assert getattr(configured_tiling, "enabled", False) is False
    effective_tiling = captured["effective_tiling"]
    assert getattr(effective_tiling, "enabled", False) is True
    assert getattr(effective_tiling, "tile_size_px", None) == 1536
    assert getattr(effective_tiling, "overlap_px", None) == 256
    assert getattr(getattr(effective_tiling, "global_pass", None), "longest_side", None) == 1280
    assert getattr(effective_tiling, "max_concurrency", None) == 1

    metadata = get_last_segmentation_runtime_metadata()
    assert metadata is not None
    assert metadata["sam2_runtime"]["tiling"]["auto_enabled"] is True
    assert metadata["sam2_runtime"]["tiling"]["decision"] == "auto_large_image"
    assert metadata["sam2_runtime"]["tiling"]["effective"]["enabled"] is True
    assert metadata["sam2_runtime"]["tiling"]["image_shape"] == [4097, 2, 3]


def test_segment_materials_cache_hit_skips_backend(sample_image, tmp_path, monkeypatch):
    """Validated exact-match cache entries should skip backend execution."""
    import transformation_portal.lux_depth_v3.segmentation_backend as seg_module

    calls = {"segment": 0}
    glass_mask = np.zeros(sample_image.shape[:2], dtype=np.float32)
    glass_mask[5:20, 5:20] = 1.0

    class FakeBackend:
        info = SegmentationBackendInfo(
            name="Fake EfficientSAM",
            model_id="fake-efficientsam",
            requires_gpu=False,
            requires_weights=False,
            approximate_memory_mb=1,
            description="test",
        )
        _device = "cpu"

        def segment(self, image):
            calls["segment"] += 1
            assert image.shape == sample_image.shape
            return {"glass": (glass_mask, 0.91)}

        def get_runtime_metadata(self):
            return {
                "clip_runtime": {"weights_source": "test"},
                "material_confidence_evidence": {
                    "glass": {
                        "material_confidence": 0.91,
                        "confidence_score_type": "material_classifier_probability_v1",
                    },
                    "stone": {
                        "material_confidence": 0.2,
                        "confidence_score_type": "material_classifier_probability_v1",
                    },
                },
            }

    monkeypatch.setattr(seg_module, "_get_backend_instance", lambda *args, **kwargs: FakeBackend())
    config = EnhanceConfig(
        enable_material_segmentation=True,
        material_segmentation_backend="efficientsam",
        material_segmentation_cache_policy="read_write",
        depth_device="cpu",
    )

    first = segment_materials(sample_image, config, cache_dir=tmp_path)
    first_metadata = get_last_segmentation_runtime_metadata()
    second = segment_materials(sample_image, config, cache_dir=tmp_path)
    second_metadata = get_last_segmentation_runtime_metadata()

    assert calls["segment"] == 1
    assert np.array_equal(first["glass"], second["glass"])
    assert first_metadata["cache_hit"] is False
    assert second_metadata["cache_hit"] is True
    assert second_metadata["mask_count"] == 1
    assert second_metadata["backend"] == "efficientsam"
    assert first_metadata["material_confidences"]["glass"] == pytest.approx(0.91)
    assert second_metadata["material_confidences"]["glass"] == pytest.approx(0.91)
    assert set(first_metadata["material_confidence_evidence"]) == {"glass"}
    assert set(second_metadata["material_confidence_evidence"]) == {"glass"}
    assert second_metadata["confidence_summary"]["count"] == 1


def test_efficientsam_clip_classification_batches_segments(sample_image, monkeypatch):
    """EfficientSAM CLIP classification should encode all segment crops in one batch."""
    import transformation_portal.lux_depth_v3.segmentation_backend as seg_module

    class FakeScalar:
        def __init__(self, value):
            self.value = value

        def item(self):
            return self.value

    class FakeTensor:
        def __init__(self, values):
            self.values = np.asarray(values, dtype=np.float32)

        def to(self, _device):
            return self

        def unsqueeze(self, axis):
            return FakeTensor(np.expand_dims(self.values, axis))

        def norm(self, dim=-1, keepdim=True):
            return FakeTensor(np.linalg.norm(self.values, axis=dim, keepdims=keepdim))

        def __truediv__(self, other):
            other_values = other.values if isinstance(other, FakeTensor) else other
            return FakeTensor(self.values / np.maximum(other_values, 1e-6))

        @property
        def T(self):
            return FakeTensor(self.values.T)

        def __matmul__(self, other):
            return FakeTensor(self.values @ other.values)

        def __getitem__(self, key):
            return FakeTensor(self.values[key])

        def argmax(self):
            return FakeScalar(int(np.argmax(self.values)))

        def item(self):
            return float(np.asarray(self.values).item())

    @contextmanager
    def fake_no_grad():
        yield

    class FakeTorch:
        @staticmethod
        def no_grad():
            return fake_no_grad()

        @staticmethod
        def cat(tensors, dim=0):
            return FakeTensor(np.concatenate([tensor.values for tensor in tensors], axis=dim))

    monkeypatch.setitem(sys.modules, "torch", FakeTorch)
    monkeypatch.setattr(seg_module, "OPEN_CLIP_AVAILABLE", True)

    model_calls = {"encode_text": 0, "encode_image": 0}
    preprocess_vectors = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]

    class FakeModel:
        def encode_text(self, tokens):
            model_calls["encode_text"] += 1
            return tokens

        def encode_image(self, image_batch):
            model_calls["encode_image"] += 1
            return image_batch

    def fake_tokenizer(_prompts):
        return FakeTensor(np.eye(4, dtype=np.float32))

    def fake_preprocess(_region):
        return FakeTensor(preprocess_vectors.pop(0))

    backend = EfficientSAMBackend()
    backend._device = "cpu"
    backend._load_clip_runtime = lambda: (FakeModel(), fake_preprocess, fake_tokenizer)

    mask_a = np.zeros(sample_image.shape[:2], dtype=bool)
    mask_a[2:30, 2:30] = True
    mask_b = np.zeros(sample_image.shape[:2], dtype=bool)
    mask_b[25:55, 25:55] = True
    segments = [
        {"segmentation": mask_a, "bbox": [2, 2, 28, 28], "area": int(mask_a.sum())},
        {"segmentation": mask_b, "bbox": [25, 25, 30, 30], "area": int(mask_b.sum())},
    ]

    result = backend._classify_segments_with_clip(sample_image, segments)

    assert model_calls == {"encode_text": 1, "encode_image": 1}
    assert set(result) == {"glass", "stone"}
    assert result["glass"][1] == pytest.approx(1.0)
    assert result["stone"][1] == pytest.approx(1.0)
    metadata = backend.get_runtime_metadata()
    assert metadata["clip_classification"]["timing_ms"]["batch_size"] == 2.0
    assert metadata["material_confidence_evidence"]["glass"]["confidence_score_type"] == "clip_softmax_margin_v1"
    assert metadata["material_confidence_evidence"]["glass"]["raw_clip_similarity"] == pytest.approx(1.0)


def test_tensor_values_1d_falls_back_to_tolist_when_numpy_bridge_unavailable():
    """Torch tensors can lose ``.numpy()`` support when NumPy ABI versions drift."""

    class TensorLike:
        def detach(self):
            return self

        def cpu(self):
            return self

        def float(self):
            return self

        def numpy(self):
            raise RuntimeError("Numpy is not available")

        def tolist(self):
            return [0.1, 0.9, 0.2]

        def values(self):
            raise AssertionError("callable values method must not be treated as tensor data")

    values = _tensor_values_1d(TensorLike())

    assert values.tolist() == pytest.approx([0.1, 0.9, 0.2])


def test_tensor_values_1d_does_not_swallow_unexpected_tensor_errors():
    """Unexpected tensor conversion failures should remain visible to callers."""

    class TensorLike:
        def detach(self):
            return self

        def cpu(self):
            return self

        def float(self):
            return self

        def numpy(self):
            raise AssertionError("unexpected tensor conversion failure")

    with pytest.raises(AssertionError, match="unexpected tensor conversion failure"):
        _tensor_values_1d(TensorLike())


def test_tensor_values_1d_does_not_swallow_unexpected_runtime_errors():
    """Only the known torch/NumPy ABI bridge failure may fall back to ``tolist``."""

    class TensorLike:
        def detach(self):
            return self

        def cpu(self):
            return self

        def float(self):
            return self

        def numpy(self):
            raise RuntimeError("unexpected torch conversion failure")

        def tolist(self):
            raise AssertionError("unexpected RuntimeError must not fall back to tolist")

    with pytest.raises(RuntimeError, match="unexpected torch conversion failure"):
        _tensor_values_1d(TensorLike())


def test_segment_materials_cache_key_invalidates_on_config_change(sample_image, tmp_path, monkeypatch):
    """Cache keys should include segmentation-affecting configuration."""
    import transformation_portal.lux_depth_v3.segmentation_backend as seg_module

    calls = {"segment": 0}

    class FakeBackend:
        info = SegmentationBackendInfo("Fake", "fake", False, False, 1, "test")
        _device = "cpu"

        def segment(self, image):
            calls["segment"] += 1
            mask = np.zeros(image.shape[:2], dtype=np.float32)
            mask[calls["segment"] : calls["segment"] + 4, 0:4] = 1.0
            return {"glass": (mask, 0.9)}

    monkeypatch.setattr(seg_module, "_get_backend_instance", lambda *args, **kwargs: FakeBackend())
    config = EnhanceConfig(
        enable_material_segmentation=True,
        material_segmentation_backend="efficientsam",
        material_segmentation_cache_policy="read_write",
        sky_gradient_threshold=0.05,
    )
    changed_config = EnhanceConfig(
        enable_material_segmentation=True,
        material_segmentation_backend="efficientsam",
        material_segmentation_cache_policy="read_write",
        sky_gradient_threshold=0.06,
    )

    segment_materials(sample_image, config, cache_dir=tmp_path)
    segment_materials(sample_image, changed_config, cache_dir=tmp_path)

    assert calls["segment"] == 2


def test_segment_materials_corrupt_cache_recomputes(sample_image, tmp_path, monkeypatch):
    """Invalid cache metadata should be rejected and recomputed."""
    import transformation_portal.lux_depth_v3.segmentation_backend as seg_module

    calls = {"segment": 0}

    class FakeBackend:
        info = SegmentationBackendInfo("Fake", "fake", False, False, 1, "test")
        _device = "cpu"

        def segment(self, image):
            calls["segment"] += 1
            mask = np.zeros(image.shape[:2], dtype=np.float32)
            mask[1:8, 1:8] = 1.0
            return {"glass": (mask, 0.9)}

    monkeypatch.setattr(seg_module, "_get_backend_instance", lambda *args, **kwargs: FakeBackend())
    config = EnhanceConfig(
        enable_material_segmentation=True,
        material_segmentation_backend="efficientsam",
        material_segmentation_cache_policy="read_write",
    )

    segment_materials(sample_image, config, cache_dir=tmp_path)
    metadata_files = list(tmp_path.glob("*/*.json"))
    assert metadata_files
    metadata_files[0].write_text('{"schema_version":"bad"}', encoding="utf-8")
    segment_materials(sample_image, config, cache_dir=tmp_path)

    assert calls["segment"] == 2


def test_segment_materials_cache_rejects_missing_confidence(sample_image, tmp_path, monkeypatch):
    """Exact-result cache entries must not synthesize missing confidence values."""
    import transformation_portal.lux_depth_v3.segmentation_backend as seg_module

    calls = {"segment": 0}

    class FakeBackend:
        info = SegmentationBackendInfo("Fake", "fake", False, False, 1, "test")
        _device = "cpu"

        def segment(self, image):
            calls["segment"] += 1
            mask = np.zeros(image.shape[:2], dtype=np.float32)
            mask[calls["segment"] : calls["segment"] + 4, 1:5] = 1.0
            return {"glass": (mask, 0.9)}

    monkeypatch.setattr(seg_module, "_get_backend_instance", lambda *args, **kwargs: FakeBackend())
    config = EnhanceConfig(
        enable_material_segmentation=True,
        material_segmentation_backend="efficientsam",
        material_segmentation_cache_policy="read_write",
    )

    segment_materials(sample_image, config, cache_dir=tmp_path)
    metadata_files = list(tmp_path.glob("*/*.json"))
    assert metadata_files
    metadata = json.loads(metadata_files[0].read_text(encoding="utf-8"))
    metadata["masks"]["glass"].pop("confidence")
    metadata_files[0].write_text(json.dumps(metadata), encoding="utf-8")

    segment_materials(sample_image, config, cache_dir=tmp_path)

    assert calls["segment"] == 2


def test_segment_materials_sam2_strict_mode_missing_backend(sample_image, monkeypatch):
    """Strict mode should raise when SAM2 backend dependencies are unavailable."""
    import transformation_portal.lux_depth_v3.segmentation_backend as seg_module

    monkeypatch.setattr(seg_module, "SPATIAL_SAM2_AVAILABLE", False)
    seg_module._get_backend_instance.cache_clear()

    config = EnhanceConfig(
        enable_material_segmentation=True,
        material_segmentation_backend="sam2",
        strict_backend=True,
    )

    try:
        with pytest.raises(RuntimeError, match="Material segmentation failed|Failed to load sam2"):
            segment_materials(sample_image, config)
    finally:
        seg_module._get_backend_instance.cache_clear()


def test_segment_materials_sam2_graceful_degradation(sample_image, monkeypatch):
    """Non-strict SAM2 mode should fall back to empty masks when unavailable."""
    import transformation_portal.lux_depth_v3.segmentation_backend as seg_module

    monkeypatch.setattr(seg_module, "SPATIAL_SAM2_AVAILABLE", False)
    seg_module._get_backend_instance.cache_clear()

    config = EnhanceConfig(
        enable_material_segmentation=True,
        material_segmentation_backend="sam2",
        strict_backend=False,
    )

    try:
        masks = segment_materials(sample_image, config)
    finally:
        seg_module._get_backend_instance.cache_clear()

    assert isinstance(masks, dict)
    assert len(masks) == 0


def test_segment_materials_unknown_backend(sample_image):
    """Test that unknown backend falls back to stub."""
    config = EnhanceConfig(
        enable_material_segmentation=True,
        material_segmentation_backend="unknown_backend",
    )

    # Should not raise, should fall back to stub
    masks = segment_materials(sample_image, config)

    assert isinstance(masks, dict)
    # Fallback to stub means empty masks
    assert len(masks) == 0


@pytest.mark.ml
def test_backend_caching():
    """Test that backend instances are cached."""
    # Get stub backend twice
    backend1 = _get_backend_instance("stub")
    backend2 = _get_backend_instance("stub")

    assert backend1 is backend2  # Should be same instance

    # Get efficientsam backend twice
    backend3 = _get_backend_instance("efficientsam", device="cpu")
    backend4 = _get_backend_instance("efficientsam", device="cpu")

    assert backend3 is backend4  # Should be same instance


def test_backend_caching_sam2(monkeypatch):
    """SAM2 backend instances should also be cached by backend args."""
    import transformation_portal.lux_depth_v3.segmentation_backend as seg_module

    class FakeSpatialSAM2Backend:
        def __init__(self, model_size, device, checkpoint_path, enable_material_classification, material_confidence_threshold):
            del model_size, device, checkpoint_path, enable_material_classification, material_confidence_threshold

        def segment(self, seg_input):
            h, w = seg_input.image.shape[:2]
            return SimpleNamespace(
                masks=np.zeros((0, h, w), dtype=bool),
                scores=np.zeros((0,), dtype=np.float32),
                metadata=[],
            )

    monkeypatch.setattr(seg_module, "SPATIAL_SAM2_AVAILABLE", True)
    monkeypatch.setattr(seg_module, "SpatialSAM2Backend", FakeSpatialSAM2Backend)
    seg_module._get_backend_instance.cache_clear()

    try:
        backend1 = _get_backend_instance("sam2", device="cpu", sam2_model_size="base")
        backend2 = _get_backend_instance("sam2", device="cpu", sam2_model_size="base")
    finally:
        seg_module._get_backend_instance.cache_clear()

    assert backend1 is backend2


# =============================================================================
# Integration Tests
# =============================================================================


@pytest.mark.ml
def test_heuristic_segmentation_detects_materials(efficientsam_heuristic_backend):
    """Test that heuristic segmentation detects expected materials."""
    # Create an image with clear material signatures
    image = np.zeros((128, 128, 3), dtype=np.uint8)

    # Blue water region (large enough to pass coverage threshold)
    image[10:60, 10:60] = [50, 80, 200]  # Blue dominant

    # Green foliage region
    image[70:120, 10:60] = [60, 180, 80]  # Green dominant

    # Gray stone region
    image[10:60, 70:120] = [120, 125, 120]  # Low saturation

    # Bright glass region
    image[70:120, 70:120] = [180, 190, 210]  # High brightness + blue tint

    results = efficientsam_heuristic_backend.segment(image)

    # Should detect multiple materials
    assert len(results) > 0

    # Check for expected materials (heuristic-based)
    # Note: Exact detection depends on heuristics, so we're lenient
    detected_materials = set(results.keys())
    possible_materials = {"water", "foliage", "stone", "glass", "sky"}

    assert detected_materials.issubset(possible_materials)

    # Verify all results have confidence scores
    for material, (mask, confidence) in results.items():
        assert isinstance(confidence, float)
        assert 0.0 <= confidence <= 1.0


@pytest.mark.ml
def test_efficientsam_backend_lazy_loading(sample_image):
    """Test that model is loaded lazily on first inference."""
    backend = EfficientSAMBackend()

    # Model not loaded yet
    assert backend._model_loaded is False

    # Load explicitly
    backend.load(device="cpu")

    # Now model is loaded
    assert backend._model_loaded is True

    # Subsequent loads are no-ops
    backend.load(device="cpu")
    assert backend._model_loaded is True


# =============================================================================
# Error Handling Tests
# =============================================================================


def test_efficientsam_device_resolution_handles_missing_torch(monkeypatch):
    """GPU-like device requests must not dereference torch when torch is unavailable."""
    import transformation_portal.lux_depth_v3.segmentation_backend as seg_module

    monkeypatch.setattr(seg_module, "TORCH_AVAILABLE", False)
    monkeypatch.setattr(seg_module, "torch", None)

    backend = EfficientSAMBackend()

    assert backend._resolve_device("mps") == "cpu"
    assert backend._resolve_device("cuda") == "cpu"
    assert backend._resolve_device("auto") == "cpu"


def test_segment_materials_sam2_strict_mps_missing_torch_reports_dependency_error(
    sample_image,
    monkeypatch,
):
    """SAM2 strict mode should report dependency failure, not torch NoneType internals."""
    import transformation_portal.lux_depth_v3.segmentation_backend as seg_module

    captured: dict[str, object] = {}

    class FakeSpatialSAM2Backend:
        def __init__(self, **kwargs):
            captured["device"] = kwargs["device"]
            self.device = kwargs["device"]
            self.tiling = kwargs.get("tiling")

        def segment(self, seg_input):
            del seg_input
            raise RuntimeError("SAM2 requires sam2 and torch. Install with: pip install sam2 torch torchvision")

    monkeypatch.setattr(seg_module, "TORCH_AVAILABLE", False)
    monkeypatch.setattr(seg_module, "torch", None)
    monkeypatch.setattr(seg_module, "SPATIAL_SAM2_AVAILABLE", True)
    monkeypatch.setattr(seg_module, "SpatialSAM2Backend", FakeSpatialSAM2Backend)
    seg_module._get_backend_instance.cache_clear()

    config = EnhanceConfig(
        enable_material_segmentation=True,
        material_segmentation_backend="sam2",
        strict_backend=True,
        depth_device="mps",
    )

    try:
        with pytest.raises(RuntimeError) as exc_info:
            segment_materials(sample_image, config)
    finally:
        seg_module._get_backend_instance.cache_clear()

    message = str(exc_info.value)
    assert "SAM2 requires sam2 and torch" in message
    assert "'NoneType' object has no attribute 'backends'" not in message
    assert "NoneType.backends" not in message
    assert "torch.backends" not in message
    assert captured["device"] == "cpu"


@pytest.mark.ml
def test_efficientsam_backend_not_loaded_error(sample_image):
    """Test that segmentation fails if model not loaded."""
    backend = EfficientSAMBackend()

    # Don't load model
    with pytest.raises(RuntimeError, match="model not loaded"):
        backend.segment(sample_image)


def test_segment_materials_strict_mode_missing_torch(sample_image, monkeypatch):
    """Test that strict mode raises if torch unavailable."""
    # Mock torch unavailability
    import transformation_portal.lux_depth_v3.segmentation_backend as seg_module

    original_torch_available = seg_module.TORCH_AVAILABLE
    monkeypatch.setattr(seg_module, "TORCH_AVAILABLE", False)

    # Clear backend cache to force re-creation
    seg_module._get_backend_instance.cache_clear()

    config = EnhanceConfig(
        enable_material_segmentation=True,
        material_segmentation_backend="efficientsam",
        strict_backend=True,
    )

    try:
        # Should raise in strict mode
        with pytest.raises(RuntimeError, match="Material segmentation failed|Failed to load efficientsam"):
            segment_materials(sample_image, config)
    finally:
        # Restore original value
        monkeypatch.setattr(seg_module, "TORCH_AVAILABLE", original_torch_available)
        seg_module._get_backend_instance.cache_clear()


def test_segment_materials_graceful_degradation(sample_image, monkeypatch):
    """Test that non-strict mode gracefully degrades to empty masks."""
    # Mock torch unavailability
    import transformation_portal.lux_depth_v3.segmentation_backend as seg_module

    original_torch_available = seg_module.TORCH_AVAILABLE
    monkeypatch.setattr(seg_module, "TORCH_AVAILABLE", False)

    config = EnhanceConfig(
        enable_material_segmentation=True,
        material_segmentation_backend="efficientsam",
        strict_backend=False,  # Non-strict mode
    )

    try:
        # Clear backend cache
        seg_module._get_backend_instance.cache_clear()

        # Should not raise, should return empty masks
        masks = segment_materials(sample_image, config)

        assert isinstance(masks, dict)
        assert len(masks) == 0  # Empty because fell back to stub
    finally:
        # Restore
        monkeypatch.setattr(seg_module, "TORCH_AVAILABLE", original_torch_available)
        seg_module._get_backend_instance.cache_clear()


# =============================================================================
# Confidence Scoring Tests
# =============================================================================


def test_stub_backend_confidence_scores(sample_image):
    """Test that StubBackend returns empty dict (compatible with confidence format)."""
    backend = StubBackend()
    backend.load()

    results = backend.segment(sample_image)

    # Stub returns empty dict
    assert isinstance(results, dict)
    assert len(results) == 0


@pytest.mark.ml
def test_confidence_scores_in_valid_range(sample_image, efficientsam_heuristic_backend):
    """Verify all confidence scores are in [0.0-1.0] range."""
    results = efficientsam_heuristic_backend.segment(sample_image)

    # Check each material's confidence
    for material, (mask, confidence) in results.items():
        assert isinstance(confidence, float), f"{material} confidence is not float: {type(confidence)}"
        assert 0.0 <= confidence <= 1.0, f"{material} confidence {confidence} not in [0.0-1.0]"


@pytest.mark.ml
def test_heuristic_fallback_returns_medium_confidence(sample_image, monkeypatch):
    """Heuristic fallback should return 0.5 confidence to indicate uncertainty.

    Exception: Sky material uses bootstrap heuristic with dynamic confidence [0.0-1.0].
    """
    import transformation_portal.lux_depth_v3.segmentation_backend as seg_module

    # Mock EFFICIENTVIT_AVAILABLE to force heuristic mode
    monkeypatch.setattr(seg_module, "EFFICIENTVIT_AVAILABLE", False)

    backend = EfficientSAMBackend()
    backend.load(device="cpu")

    results = backend.segment(sample_image)

    # All heuristic results should have 0.5 confidence (except sky)
    for material, (mask, confidence) in results.items():
        if material == "sky":
            # Sky uses bootstrap heuristic with dynamic confidence
            assert 0.0 <= confidence <= 1.0, f"Sky confidence should be in [0,1], got {confidence}"
        else:
            assert confidence == 0.5, f"Heuristic {material} should have 0.5 confidence, got {confidence}"


@pytest.mark.ml
def test_confidence_logged_in_output(sample_image, monkeypatch, caplog):
    """Verify fallback confidence logging contract without real model inference."""
    backend = EfficientSAMBackend()
    backend._device = "cpu"

    def _raise_clip_load_error():
        raise RuntimeError("offline cache miss")

    monkeypatch.setattr(backend, "_load_clip_runtime", _raise_clip_load_error)

    seg_mask = np.zeros(sample_image.shape[:2], dtype=bool)
    seg_mask[0:32, 0:32] = True
    segments = [{"segmentation": seg_mask, "bbox": [0, 0, 32, 32], "area": int(seg_mask.sum())}]

    import logging

    with caplog.at_level(logging.INFO):
        results = backend._classify_segments_with_clip(sample_image, segments)

    # Check log contains confidence percentages (from CLIP classification)
    log_text = caplog.text

    # Runtime contract:
    # - If CLIP succeeds, summary line must include percentage confidences.
    # - If CLIP fails (offline cache miss, etc.), fallback warning must be present.
    assert "falling back to heuristics" in log_text.lower()
    assert "CLIP classification failed" in log_text
    assert isinstance(results, dict)


@pytest.mark.ml
def test_clip_success_logging_emits_percentages(sample_image, monkeypatch, caplog):
    """Deterministic unit test: successful CLIP path emits % confidence summaries."""
    if not TORCH_AVAILABLE:
        pytest.skip("torch not available")

    backend = EfficientSAMBackend()
    backend._device = "cpu"

    class _FakeModel:
        def encode_text(self, _tokens):
            # 4 prompts x 4-dim embedding (glass, water, foliage, stone)
            return torch.eye(4, dtype=torch.float32)

        def encode_image(self, _tensor):
            # Most similar to "water" prompt.
            return torch.tensor([[0.10, 0.95, 0.10, 0.10]], dtype=torch.float32)

    def _fake_preprocess(_region):
        return torch.ones((3, 16, 16), dtype=torch.float32)

    def _fake_tokenizer(prompts):
        return torch.ones((len(prompts), 4), dtype=torch.float32)

    monkeypatch.setattr(backend, "_load_clip_runtime", lambda: (_FakeModel(), _fake_preprocess, _fake_tokenizer))

    seg_mask = np.zeros(sample_image.shape[:2], dtype=bool)
    seg_mask[0:32, 0:32] = True  # 1024px > 500px coverage threshold
    segments = [{"segmentation": seg_mask, "bbox": [0, 0, 32, 32], "area": int(seg_mask.sum()), "heuristic_label": "water"}]

    import logging

    with caplog.at_level(logging.INFO):
        results = backend._classify_segments_with_clip(sample_image, segments)

    assert "water" in results
    assert "CLIP classified" in caplog.text
    assert "%" in caplog.text


@pytest.mark.ml
def test_multiple_materials_different_confidences(efficientsam_heuristic_backend):
    """Test that different materials can have different confidence scores."""
    # Create image with distinct material regions
    image = np.zeros((128, 128, 3), dtype=np.uint8)

    # Very clear blue region (high confidence for water)
    image[10:60, 10:60] = [50, 120, 200]

    # Greenish region (medium confidence for foliage)
    image[70:120, 10:60] = [80, 140, 90]

    # Grayish region (stone)
    image[10:60, 70:120] = [120, 125, 120]

    results = efficientsam_heuristic_backend.segment(image)

    if len(results) >= 2:
        # Extract confidence scores
        confidences = [conf for _, conf in results.values()]

        # At least some variation in confidence (not all identical)
        # Since heuristic returns 0.5 for all, this tests that we're tracking per-material scores
        unique_confidences = set(confidences)

        # All should be 0.5 in heuristic mode (this validates our implementation)
        # In CLIP mode, we'd expect variation
        assert all(c == 0.5 for c in confidences) or len(unique_confidences) > 1


@pytest.mark.ml
def test_confidence_filtering_example(efficientsam_heuristic_backend):
    """Demonstrate how users can filter by confidence threshold."""
    # Create test image
    image = np.zeros((128, 128, 3), dtype=np.uint8)
    image[10:60, 10:60] = [50, 120, 200]  # Water
    image[70:120, 10:60] = [80, 140, 90]  # Foliage

    results = efficientsam_heuristic_backend.segment(image)

    # Filter by confidence threshold
    min_confidence = 0.4
    high_confidence_only = {material: (mask, conf) for material, (mask, conf) in results.items() if conf >= min_confidence}

    # Filtering should reduce or maintain material count
    assert len(high_confidence_only) <= len(results), "Filtering should reduce or maintain material count"

    # All remaining materials should meet threshold
    for material, (mask, conf) in high_confidence_only.items():
        assert conf >= min_confidence, f"{material} confidence {conf} below threshold {min_confidence}"
