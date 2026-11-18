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
    SAM2CheckpointIntegrityError,
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

# EfficientSAMBackend.load() needs a real torch even on the heuristic path, so
# tests that load it must skip (not error) in the torch-free core environment.
requires_torch = pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch required to load the EfficientSAM backend")

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
    pytest.importorskip("torch", reason="PyTorch required to load the EfficientSAM backend")
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
    assert "sam2" in info.model_id
    assert "hiera" in info.model_id
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
@requires_torch
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
@requires_torch
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
        def __init__(self, **kwargs):
            self.device = kwargs["device"]

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
        def __init__(self, **kwargs):
            captured["model_size"] = kwargs["model_size"]
            captured["device"] = kwargs["device"]
            captured["checkpoint_path"] = kwargs["checkpoint_path"]
            captured["model_config"] = kwargs["model_config"]
            captured["expected_sha256"] = kwargs["expected_sha256"]
            captured["generator_kwargs"] = dict(kwargs["generator_kwargs"])
            captured["enable_material_classification"] = kwargs["enable_material_classification"]
            captured["material_confidence_threshold"] = kwargs["material_confidence_threshold"]
            captured["configured_tiling"] = kwargs["tiling"]
            self.device = kwargs["device"]
            self.tiling = kwargs["tiling"]

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
    assert captured["model_config"] == "configs/sam2.1/sam2.1_hiera_l.yaml"
    assert captured["expected_sha256"] == "2647878d5dfa5098f2f8649825738a9345572bae2d4350a2468587ece47dd318"
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
        def __init__(self, **kwargs):
            del kwargs["model_size"], kwargs["checkpoint_path"], kwargs["generator_kwargs"]
            del kwargs["enable_material_classification"], kwargs["material_confidence_threshold"]
            del kwargs["model_config"], kwargs["expected_sha256"]
            captured["configured_tiling"] = kwargs["tiling"]
            self.device = kwargs["device"]
            self.tiling = kwargs["tiling"]

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


def test_segment_materials_does_not_cache_degraded_heuristic_fallback(tmp_path, monkeypatch):
    """LC-02: heuristic fallback masks must not be replayed as the requested backend's results."""
    import transformation_portal.lux_depth_v3.segmentation_backend as seg_module

    image = np.full((120, 160, 3), 90, dtype=np.uint8)
    image[60:, :80] = (40, 160, 50)  # foliage-like block
    image[:40, 80:] = (60, 80, 220)  # blue block

    # A loaded EfficientSAM backend running in heuristic mode (efficientvit unavailable).
    heuristic_backend = EfficientSAMBackend()
    heuristic_backend._model_loaded = True
    heuristic_backend._use_real_model = False
    heuristic_backend._device = "cpu"

    calls = {"healthy": 0}
    healthy_mask = np.zeros(image.shape[:2], dtype=np.float32)
    healthy_mask[10:60, 10:60] = 1.0

    class HealthyBackend:
        info = SegmentationBackendInfo("Fake", "fake", False, False, 1, "test")
        _device = "cpu"

        def segment(self, _image):
            calls["healthy"] += 1
            return {"glass": (healthy_mask, 0.93)}

        def segmentation_degraded(self):
            return False

    config = EnhanceConfig(
        enable_material_segmentation=True,
        material_segmentation_backend="efficientsam",
        material_segmentation_cache_policy="read_write",
        depth_device="cpu",
    )

    monkeypatch.setattr(seg_module, "_get_backend_instance", lambda *args, **kwargs: heuristic_backend)
    degraded = segment_materials(image, config, cache_dir=tmp_path)
    degraded_metadata = get_last_segmentation_runtime_metadata()
    assert degraded, "heuristic fallback should still serve masks for this run"
    assert heuristic_backend.segmentation_degraded() is True
    assert {ev["confidence_score_type"] for ev in degraded_metadata["material_confidence_evidence"].values()} == {
        "heuristic_material_confidence_v1"
    }
    assert list(tmp_path.rglob("*.json")) == []

    monkeypatch.setattr(seg_module, "_get_backend_instance", lambda *args, **kwargs: HealthyBackend())
    healthy = segment_materials(image, config, cache_dir=tmp_path)
    healthy_metadata = get_last_segmentation_runtime_metadata()

    assert calls["healthy"] == 1
    assert healthy_metadata["cache_hit"] is False
    assert set(healthy) == {"glass"}
    # The healthy result is cached and replayed normally.
    segment_materials(image, config, cache_dir=tmp_path)
    assert get_last_segmentation_runtime_metadata()["cache_hit"] is True
    assert calls["healthy"] == 1


def test_efficientsam_segmentation_degraded_tracks_clip_evidence():
    """A CLIP-classified result is not degraded; a heuristic one is."""
    backend = EfficientSAMBackend()
    backend._material_confidence_evidence = {"glass": {"confidence_score_type": "clip_softmax_margin_v1"}}
    assert backend.segmentation_degraded() is False
    backend._material_confidence_evidence = {"glass": {"confidence_score_type": "heuristic_material_confidence_v1"}}
    assert backend.segmentation_degraded() is True


def test_segment_materials_does_not_cache_when_sam_proposals_fail(tmp_path, monkeypatch):
    """LC-02: CLIP labels on heuristic regions (SAM proposals failed) are still a degraded result."""
    import transformation_portal.lux_depth_v3.segmentation_backend as seg_module

    image = np.full((120, 160, 3), 90, dtype=np.uint8)
    proposal = np.zeros(image.shape[:2], dtype=bool)
    proposal[10:60, 10:60] = True

    class FlakyMaskGenerator:
        fail = True

        def generate(self, _image):
            if self.fail:
                raise RuntimeError("transient SAM failure")
            return [{"segmentation": proposal, "bbox": [10, 10, 50, 50], "area": int(proposal.sum())}]

    generator = FlakyMaskGenerator()
    backend = EfficientSAMBackend()
    backend._model_loaded = True
    backend._use_real_model = True
    backend._device = "cpu"
    backend._mask_generator = generator

    def clip_labels_any_regions(_image, segments):
        # Stand-in for a working CLIP stage: it labels whatever regions it is handed
        # (SAM proposals, or heuristic regions when SAM returned nothing).
        mask = segments[0]["segmentation"].astype(np.float32) if segments else np.ones(image.shape[:2], np.float32)
        backend._material_confidence_evidence = {
            "glass": {"material_confidence": 0.9, "confidence_score_type": "clip_softmax_margin_v1"}
        }
        return {"glass": (mask, 0.9)}

    monkeypatch.setattr(seg_module, "EFFICIENTVIT_AVAILABLE", True)
    monkeypatch.setattr(seg_module, "OPEN_CLIP_AVAILABLE", True)
    monkeypatch.setattr(backend, "_classify_segments_with_clip", clip_labels_any_regions)
    monkeypatch.setattr(seg_module, "_get_backend_instance", lambda *args, **kwargs: backend)
    config = EnhanceConfig(
        enable_material_segmentation=True,
        material_segmentation_backend="efficientsam",
        material_segmentation_cache_policy="read_write",
        depth_device="cpu",
    )

    served = segment_materials(image, config, cache_dir=tmp_path)
    assert set(served) == {"glass"}, "the degraded result is still served for this run"
    assert backend.segmentation_degraded() is True
    assert list(tmp_path.rglob("*.json")) == []

    # The generator recovers: the next run executes the backend and its result is cached.
    generator.fail = False
    recovered = segment_materials(image, config, cache_dir=tmp_path)
    assert get_last_segmentation_runtime_metadata()["cache_hit"] is False
    assert np.array_equal(recovered["glass"], proposal.astype(np.float32))
    assert backend.segmentation_degraded() is False
    assert len(list(tmp_path.rglob("*.json"))) == 1

    # A missing generator is the same degraded condition.
    backend._mask_generator = None
    backend.segment(image)
    assert backend.segmentation_degraded() is True


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
        def __init__(self, **kwargs):
            del kwargs

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
@requires_torch
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
    assert "torch.backends" not in message
    assert "'NoneType' object has no attribute 'backends'" not in message
    assert captured["device"] == "cpu"


def test_segment_materials_sam2_integrity_error_fails_closed(sample_image, monkeypatch):
    """SAM2 checksum mismatches must propagate instead of degrading to empty masks."""
    import transformation_portal.lux_depth_v3.segmentation_backend as seg_module

    class FakeSpatialSAM2Backend:
        def __init__(self, **kwargs):
            self.device = kwargs["device"]

        def segment(self, seg_input):
            del seg_input
            raise AssertionError("segment() should not run after load() integrity failure")

    original_load = SAM2SegmentationBackend.load

    def fake_load(self, device="auto", weights_path=None):
        del self, device, weights_path
        raise SAM2CheckpointIntegrityError("SHA-256 mismatch for SAM2 checkpoint")

    monkeypatch.setattr(seg_module, "SPATIAL_SAM2_AVAILABLE", True)
    monkeypatch.setattr(seg_module, "SpatialSAM2Backend", FakeSpatialSAM2Backend)
    monkeypatch.setattr(SAM2SegmentationBackend, "load", fake_load)
    seg_module._get_backend_instance.cache_clear()

    config = EnhanceConfig(
        enable_material_segmentation=True,
        material_segmentation_backend="sam2",
        strict_backend=False,
        depth_device="cpu",
    )

    try:
        with pytest.raises(SAM2CheckpointIntegrityError, match="SHA-256 mismatch"):
            segment_materials(sample_image, config)
    finally:
        monkeypatch.setattr(SAM2SegmentationBackend, "load", original_load)
        seg_module._get_backend_instance.cache_clear()


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
@requires_torch
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


def _legacy_heuristic_component_segments(heuristic_results):
    """Reference: the original per-component full-image loop (pre audit LC-12)."""
    from scipy import ndimage

    reference = []
    for material_name, (mask, _) in heuristic_results.items():
        labeled, num_features = ndimage.label(mask > 0.5)
        for region_id in range(1, num_features + 1):
            region_mask = labeled == region_id
            area = region_mask.sum()
            if area < 500:
                continue
            rows, cols = np.where(region_mask)
            x1, x2 = cols.min(), cols.max()
            y1, y2 = rows.min(), rows.max()
            reference.append(
                {
                    "segmentation": region_mask,
                    "bbox": [int(x1), int(y1), int(x2 - x1 + 1), int(y2 - y1 + 1)],
                    "area": int(area),
                    "heuristic_label": material_name,
                }
            )
    return reference


def test_heuristic_component_segments_match_legacy_loop(monkeypatch):
    """Regression (audit LC-12): vectorised component extraction stays bit-identical.

    The heuristic fallback used a full-image pass per connected component, which is
    quadratic on speckled masks; the bincount/find_objects path must produce the same
    segments (order, masks, bboxes, areas, labels).
    """
    pytest.importorskip("scipy")
    rng = np.random.default_rng(1234)
    h, w = 96, 128

    stone = rng.random((h, w)) < 0.35  # speckle: many tiny components
    stone[0:30, 0:40] = True  # touches the top-left border
    stone[60:96, 90:128] = True  # touches the bottom-right border
    ring = np.zeros((h, w), dtype=bool)
    ring[20:80, 50:85] = True
    ring[30:70, 60:75] = False  # non-convex component with a hole
    heuristic_results = {
        "stone": (stone.astype(np.float32), 0.5),
        "glass": (ring.astype(np.float32), 0.5),
        "water": (np.zeros((h, w), dtype=np.float32), 0.5),
    }

    actual = _capture_heuristic_component_segments(monkeypatch, heuristic_results, (h, w))

    expected = _legacy_heuristic_component_segments(heuristic_results)
    assert len(expected) >= 3
    assert len(actual) == len(expected)
    for got, want in zip(actual, expected):
        assert got["heuristic_label"] == want["heuristic_label"]
        assert got["bbox"] == want["bbox"]
        assert got["area"] == want["area"]
        assert type(got["area"]) is int
        assert got["segmentation"].dtype == want["segmentation"].dtype
        assert np.array_equal(got["segmentation"], want["segmentation"])


def test_heuristic_component_segments_scale_linearly_on_speckled_masks(monkeypatch):
    """Regression (audit LC-12): tens of thousands of speckle components stay cheap.

    The legacy loop compared the whole label image once per component (~15 s for a
    500x750 speckled mask); the vectorised path needs well under a second.
    """
    pytest.importorskip("scipy")
    import time

    rng = np.random.default_rng(7)
    h, w = 500, 750
    speckle = rng.random((h, w)) < 0.4
    speckle[100:200, 100:300] = True  # one component above the 500-px area floor
    heuristic_results = {"stone": (speckle.astype(np.float32), 0.5)}

    t0 = time.perf_counter()
    segments = _capture_heuristic_component_segments(monkeypatch, heuristic_results, (h, w))
    elapsed = time.perf_counter() - t0

    assert segments, "the large block must survive the area filter"
    assert elapsed < 3.0, f"component extraction took {elapsed:.2f}s on a speckled 500x750 mask"


def _capture_heuristic_component_segments(monkeypatch, heuristic_results, shape):
    """Run the no-SAM-segments branch and return the segments handed to CLIP."""
    h, w = shape
    backend = EfficientSAMBackend()
    backend._device = "cpu"
    monkeypatch.setattr(backend, "_heuristic_segmentation", lambda _image: heuristic_results)
    monkeypatch.delenv("SKIP_CLIP_INFERENCE", raising=False)
    if not TORCH_AVAILABLE:
        # The CLIP branch imports torch before loading the runtime; a bare stand-in
        # lets the capture hook below run in the core (torch-free) lane.
        monkeypatch.setitem(sys.modules, "torch", types.ModuleType("torch"))

    captured = {}

    def _capture_segments_then_fail():
        captured["segments"] = list(sys._getframe(1).f_locals["segments"])
        raise RuntimeError("offline: CLIP not loaded in this test")

    monkeypatch.setattr(backend, "_load_clip_runtime", _capture_segments_then_fail)

    backend._classify_segments_with_clip(np.zeros((h, w, 3), dtype=np.uint8), [])
    return captured["segments"]
