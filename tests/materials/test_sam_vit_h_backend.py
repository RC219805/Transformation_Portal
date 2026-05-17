"""Tests for SAMVitHBackend (Phase 2, APEX Research segmentation).

Test Coverage:
1. Protocol compliance (no model load needed)
2. Checkpoint validation (SHA256 pass/fail, missing checkpoint)
3. Mock inference shape/dtype contract
4. Device resolution (cpu/auto)
5. Factory fallback (non-strict → stub, strict → RuntimeError)
6. segment_materials integration with sam_vit_h backend
7. EnhanceConfig sam_vit_h_* field defaults

All tests run without downloading the 2.4GB checkpoint.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pytest

pytestmark = pytest.mark.unit

from transformation_portal.lux_depth_v3.config import EnhanceConfig
from transformation_portal.lux_depth_v3.protocols.segmentation_backend import (
    SegmentationBackend,
    SegmentationBackendInfo,
)
from transformation_portal.lux_depth_v3.segmentation_backend import (
    SAMVitHBackend,
    StubBackend,
    _get_backend_instance,
    _get_sam_vit_h_instance,
    get_last_segmentation_runtime_metadata,
    segment_materials,
)

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(name="sample_image")
def fixture_sample_image() -> np.ndarray:
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    image[10:30, 10:30] = [100, 150, 200]
    image[35:55, 35:55] = [80, 180, 100]
    image[10:30, 35:55] = [150, 150, 150]
    return image


# =============================================================================
# 1. Protocol compliance
# =============================================================================


def test_sam_vit_h_backend_implements_protocol():
    """SAMVitHBackend satisfies the SegmentationBackend protocol before loading."""
    backend = SAMVitHBackend()
    assert isinstance(backend, SegmentationBackend)
    assert hasattr(backend, "info")
    assert hasattr(backend, "load")
    assert hasattr(backend, "segment")


def test_sam_vit_h_backend_info_contract():
    """info property returns correctly-typed SegmentationBackendInfo."""
    backend = SAMVitHBackend()
    info = backend.info
    assert isinstance(info, SegmentationBackendInfo)
    assert info.name == "SAM ViT-H"
    assert info.model_id == "facebook/sam-vit-huge"
    assert info.requires_weights is True
    assert info.approximate_memory_mb == 2400


def test_sam_vit_h_segment_before_load_raises():
    """segment() before load() must raise RuntimeError."""
    backend = SAMVitHBackend()
    with pytest.raises(RuntimeError, match="not loaded"):
        backend.segment(np.zeros((64, 64, 3), dtype=np.uint8))


def test_sam_vit_h_segment_wrong_ndim_raises():
    """segment() raises ValueError for non-3D input."""
    backend = SAMVitHBackend()
    backend._model_loaded = True
    backend._mask_generator = object()
    with pytest.raises(ValueError, match="Expected RGB image"):
        backend.segment(np.zeros((64, 64), dtype=np.uint8))


def test_sam_vit_h_segment_wrong_dtype_raises():
    """segment() raises ValueError for non-uint8 input."""
    backend = SAMVitHBackend()
    backend._model_loaded = True
    backend._mask_generator = object()
    with pytest.raises(ValueError, match="uint8"):
        backend.segment(np.zeros((64, 64, 3), dtype=np.float32))


# =============================================================================
# 2. Checkpoint validation
# =============================================================================


def test_checkpoint_sha256_validation_pass(tmp_path: Path):
    """_validate_checkpoint_sha256 does not raise when hash matches."""
    checkpoint = tmp_path / "sam_vit_h_4b8939.pth"
    data = b"fake checkpoint data for testing"
    checkpoint.write_bytes(data)
    expected = hashlib.sha256(data).hexdigest()
    SAMVitHBackend._validate_checkpoint_sha256(checkpoint, expected)  # no raise


def test_checkpoint_sha256_validation_fail(tmp_path: Path):
    """_validate_checkpoint_sha256 raises RuntimeError on hash mismatch."""
    checkpoint = tmp_path / "sam_vit_h_4b8939.pth"
    checkpoint.write_bytes(b"corrupted checkpoint data")
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        SAMVitHBackend._validate_checkpoint_sha256(checkpoint, "a" * 64)


def test_checkpoint_not_found_raises_file_not_found(tmp_path: Path, monkeypatch):
    """load() raises FileNotFoundError when the configured checkpoint path is absent."""
    import transformation_portal.lux_depth_v3.segmentation_backend as seg_module

    monkeypatch.setattr(seg_module, "SAM_AVAILABLE", True)
    monkeypatch.setattr(seg_module, "TORCH_AVAILABLE", True)
    backend = SAMVitHBackend(checkpoint_path=str(tmp_path / "nonexistent.pth"))
    with pytest.raises(FileNotFoundError):
        backend.load(device="cpu")


def test_resolve_checkpoint_search_order(tmp_path: Path, monkeypatch):
    """_resolve_checkpoint finds a checkpoint placed in the checkpoints/ directory."""
    checkpoints_dir = tmp_path / "checkpoints"
    checkpoints_dir.mkdir()
    expected = checkpoints_dir / SAMVitHBackend.CHECKPOINT_FILENAME
    expected.write_bytes(b"dummy")
    monkeypatch.chdir(tmp_path)
    backend = SAMVitHBackend()
    resolved = backend._resolve_checkpoint(None)
    assert resolved.resolve() == expected.resolve()


def test_resolve_checkpoint_explicit_arg(tmp_path: Path):
    """_resolve_checkpoint honours an explicit weights_path argument."""
    checkpoint = tmp_path / "my_sam.pth"
    checkpoint.write_bytes(b"dummy")
    backend = SAMVitHBackend()
    assert backend._resolve_checkpoint(checkpoint) == checkpoint


def test_resolve_checkpoint_explicit_arg_missing_raises(tmp_path: Path):
    """_resolve_checkpoint raises FileNotFoundError for a missing explicit path."""
    backend = SAMVitHBackend()
    with pytest.raises(FileNotFoundError):
        backend._resolve_checkpoint(tmp_path / "ghost.pth")


# =============================================================================
# 3. Mock inference shape/dtype contract
# =============================================================================


def test_sam_vit_h_segment_shape_contract(sample_image: np.ndarray):
    """segment() returns Dict[str, Tuple[ndarray(H,W), float]] with float32 masks."""
    h, w = sample_image.shape[:2]
    fake_mask = np.zeros((h, w), dtype=bool)
    fake_mask[10:30, 10:30] = True

    class FakeMaskGenerator:
        def generate(self, image: np.ndarray):
            return [
                {
                    "segmentation": fake_mask,
                    "predicted_iou": 0.92,
                    "area": int(fake_mask.sum()),
                    "bbox": [10, 10, 20, 20],
                }
            ]

    backend = SAMVitHBackend(confidence_threshold=0.5)
    backend._model_loaded = True
    backend._device = "cpu"
    backend._mask_generator = FakeMaskGenerator()

    results = backend.segment(sample_image)
    assert isinstance(results, dict)
    for material, (mask, confidence) in results.items():
        assert isinstance(material, str)
        assert mask.shape == (h, w)
        assert mask.dtype == np.float32
        assert 0.0 <= float(mask.min()) and float(mask.max()) <= 1.0
        assert 0.0 <= confidence <= 1.0


def test_sam_vit_h_segment_returns_empty_dict_on_no_masks(sample_image: np.ndarray):
    """segment() returns empty dict when mask generator returns no masks."""

    class EmptyGenerator:
        def generate(self, image: np.ndarray):
            return []

    backend = SAMVitHBackend()
    backend._model_loaded = True
    backend._device = "cpu"
    backend._mask_generator = EmptyGenerator()
    assert backend.segment(sample_image) == {}


def test_sam_vit_h_segment_filters_low_iou(sample_image: np.ndarray):
    """Masks with predicted_iou below confidence_threshold are excluded."""
    h, w = sample_image.shape[:2]
    fake_mask = np.ones((h, w), dtype=bool)

    class LowIoUGenerator:
        def generate(self, image: np.ndarray):
            return [{"segmentation": fake_mask, "predicted_iou": 0.1, "area": h * w, "bbox": [0, 0, w, h]}]

    backend = SAMVitHBackend(confidence_threshold=0.85)
    backend._model_loaded = True
    backend._device = "cpu"
    backend._mask_generator = LowIoUGenerator()
    # Low IoU mask should be filtered out
    assert backend.segment(sample_image) == {}


def test_masks_to_material_dict_merges_same_material():
    """Multiple masks with the same label are merged via np.maximum.

    Uses a 100×100 image so each mask covers 625 px — above the 500 px
    minimum area filter in _masks_to_material_dict.
    """
    h, w = 100, 100
    mask_a = np.zeros((h, w), dtype=bool)
    mask_a[0:25, 0:25] = True  # 625 px
    mask_b = np.zeros((h, w), dtype=bool)
    mask_b[30:55, 30:55] = True  # 625 px

    # Both masks are stone-colored (gray, low saturation, medium brightness)
    stone_image = np.full((h, w, 3), 120, dtype=np.uint8)

    raw = [
        {"segmentation": mask_a, "predicted_iou": 0.88, "area": 625, "bbox": [0, 0, 25, 25]},
        {"segmentation": mask_b, "predicted_iou": 0.91, "area": 625, "bbox": [30, 30, 25, 25]},
    ]
    backend = SAMVitHBackend(confidence_threshold=0.50)
    result = backend._masks_to_material_dict(stone_image, raw)

    assert "stone" in result, f"Expected 'stone' label from gray stone_image; got keys: {list(result.keys())}"
    merged_mask, conf = result["stone"]
    # Both regions should be present in merged mask
    assert merged_mask[12, 12] == pytest.approx(1.0)
    assert merged_mask[42, 42] == pytest.approx(1.0)
    assert conf == pytest.approx(0.91)


# =============================================================================
# 4. Device resolution
# =============================================================================


def test_sam_vit_h_resolve_device_cpu():
    """_resolve_device('cpu') always returns 'cpu'."""
    backend = SAMVitHBackend()
    assert backend._resolve_device("cpu") == "cpu"


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not available")
def test_sam_vit_h_resolve_device_auto_returns_valid():
    """_resolve_device('auto') returns a valid device string."""
    backend = SAMVitHBackend()
    resolved = backend._resolve_device("auto")
    assert resolved in {"cpu", "mps", "cuda"}


# =============================================================================
# 5. Factory fallback
# =============================================================================


def test_get_backend_instance_sam_vit_h_missing_checkpoint_fallback_to_stub(monkeypatch):
    """Non-strict mode: missing checkpoint falls back to StubBackend."""
    import transformation_portal.lux_depth_v3.segmentation_backend as seg_module

    monkeypatch.setattr(seg_module, "SAM_AVAILABLE", True)
    monkeypatch.setattr(seg_module, "TORCH_AVAILABLE", True)
    seg_module._get_backend_instance.cache_clear()
    seg_module._get_sam_vit_h_instance.cache_clear()
    try:
        backend = _get_backend_instance(
            "sam_vit_h",
            device="cpu",
            strict=False,
            sam_vit_h_checkpoint_path="/nonexistent/path/sam_vit_h.pth",
        )
        assert isinstance(backend, StubBackend)
    finally:
        seg_module._get_backend_instance.cache_clear()
        seg_module._get_sam_vit_h_instance.cache_clear()


def test_get_backend_instance_sam_vit_h_missing_checkpoint_strict_raises(monkeypatch):
    """Strict mode: missing checkpoint raises RuntimeError."""
    import transformation_portal.lux_depth_v3.segmentation_backend as seg_module

    monkeypatch.setattr(seg_module, "SAM_AVAILABLE", True)
    monkeypatch.setattr(seg_module, "TORCH_AVAILABLE", True)
    seg_module._get_backend_instance.cache_clear()
    seg_module._get_sam_vit_h_instance.cache_clear()
    try:
        with pytest.raises(RuntimeError, match="Failed to load sam_vit_h"):
            _get_backend_instance(
                "sam_vit_h",
                device="cpu",
                strict=True,
                sam_vit_h_checkpoint_path="/nonexistent/path/sam_vit_h.pth",
            )
    finally:
        seg_module._get_backend_instance.cache_clear()
        seg_module._get_sam_vit_h_instance.cache_clear()


def test_get_backend_instance_sam_vit_h_not_installed_fallback(monkeypatch):
    """Non-strict mode: segment_anything not installed falls back to stub."""
    import transformation_portal.lux_depth_v3.segmentation_backend as seg_module

    monkeypatch.setattr(seg_module, "SAM_AVAILABLE", False)
    seg_module._get_backend_instance.cache_clear()
    seg_module._get_sam_vit_h_instance.cache_clear()
    try:
        backend = _get_backend_instance("sam_vit_h", device="cpu", strict=False)
        assert isinstance(backend, StubBackend)
    finally:
        seg_module._get_backend_instance.cache_clear()
        seg_module._get_sam_vit_h_instance.cache_clear()


def test_get_backend_instance_unknown_backend_raises():
    """Unknown backend name raises ValueError listing valid options."""
    import transformation_portal.lux_depth_v3.segmentation_backend as seg_module

    seg_module._get_backend_instance.cache_clear()
    try:
        with pytest.raises(ValueError, match="sam_vit_h"):
            _get_backend_instance("completely_unknown_backend", device="cpu", strict=False)
    finally:
        seg_module._get_backend_instance.cache_clear()


# =============================================================================
# 6. segment_materials integration
# =============================================================================


def test_segment_materials_sam_vit_h_routes_to_backend(sample_image: np.ndarray, monkeypatch):
    """segment_materials with sam_vit_h backend routes to SAMVitHBackend and returns masks."""
    import transformation_portal.lux_depth_v3.segmentation_backend as seg_module

    class FakeSAMVitHBackend:
        info = SegmentationBackendInfo("SAM ViT-H", "facebook/sam-vit-huge", requires_weights=True)
        _device = "cpu"

        def segment(self, img: np.ndarray) -> Dict[str, Tuple[np.ndarray, float]]:
            mask = np.zeros(img.shape[:2], dtype=np.float32)
            mask[5:20, 5:20] = 1.0
            return {"glass": (mask, 0.91)}

        def get_runtime_metadata(self):
            return {"backend": "sam_vit_h"}

    monkeypatch.setattr(seg_module, "_get_backend_instance", lambda *a, **kw: FakeSAMVitHBackend())

    config = EnhanceConfig(
        enable_material_segmentation=True,
        material_segmentation_backend="sam_vit_h",
        depth_device="cpu",
    )
    masks = segment_materials(sample_image, config)
    assert "glass" in masks
    assert masks["glass"].shape == sample_image.shape[:2]
    assert masks["glass"].dtype == np.float32


def test_segment_materials_sam_vit_h_runtime_metadata(sample_image: np.ndarray, monkeypatch):
    """segment_materials records runtime metadata for sam_vit_h backend."""
    import transformation_portal.lux_depth_v3.segmentation_backend as seg_module

    class FakeSAMVitHBackend:
        info = SegmentationBackendInfo("SAM ViT-H", "facebook/sam-vit-huge", requires_weights=True)
        _device = "cpu"

        def segment(self, img: np.ndarray) -> Dict[str, Tuple[np.ndarray, float]]:
            mask = np.zeros(img.shape[:2], dtype=np.float32)
            mask[10:20, 10:20] = 1.0
            return {"stone": (mask, 0.88)}

        def get_runtime_metadata(self):
            return {"backend": "sam_vit_h", "device": "cpu"}

    monkeypatch.setattr(seg_module, "_get_backend_instance", lambda *a, **kw: FakeSAMVitHBackend())

    config = EnhanceConfig(
        enable_material_segmentation=True,
        material_segmentation_backend="sam_vit_h",
        depth_device="cpu",
    )
    segment_materials(sample_image, config)
    meta = get_last_segmentation_runtime_metadata()
    assert meta is not None
    assert meta["backend"] == "sam_vit_h"


# =============================================================================
# 7. EnhanceConfig sam_vit_h_* field defaults
# =============================================================================


def test_enhance_config_sam_vit_h_defaults():
    """EnhanceConfig has correct sam_vit_h_* field defaults."""
    config = EnhanceConfig()
    assert config.sam_vit_h_checkpoint_path is None
    assert config.sam_vit_h_points_per_side == 32
    assert config.sam_vit_h_pred_iou_thresh == pytest.approx(0.88)
    assert config.sam_vit_h_confidence_threshold == pytest.approx(0.85)
    assert config.sam_vit_h_expected_sha256 is None


def test_enhance_config_sam_vit_h_overrides():
    """EnhanceConfig accepts non-default sam_vit_h_* values."""
    config = EnhanceConfig(
        sam_vit_h_checkpoint_path="/opt/checkpoints/sam_vit_h.pth",
        sam_vit_h_points_per_side=64,
        sam_vit_h_pred_iou_thresh=0.9,
        sam_vit_h_confidence_threshold=0.7,
    )
    assert config.sam_vit_h_checkpoint_path == "/opt/checkpoints/sam_vit_h.pth"
    assert config.sam_vit_h_points_per_side == 64
    assert config.sam_vit_h_pred_iou_thresh == pytest.approx(0.9)
    assert config.sam_vit_h_confidence_threshold == pytest.approx(0.7)


# =============================================================================
# 8. Heuristic label tests
# =============================================================================


def test_heuristic_label_glass():
    """High brightness blue-tinted region → glass (brightness strictly > 0.6)."""
    label = SAMVitHBackend._heuristic_label(r=0.5, g=0.55, b=0.8, brightness=0.617)
    assert label == "glass"


def test_heuristic_label_water():
    """Blue-dominant medium-brightness region → water."""
    label = SAMVitHBackend._heuristic_label(r=0.2, g=0.25, b=0.6, brightness=0.35)
    assert label == "water"


def test_heuristic_label_foliage():
    """Green-dominant region → foliage."""
    label = SAMVitHBackend._heuristic_label(r=0.2, g=0.5, b=0.2, brightness=0.3)
    assert label == "foliage"


def test_heuristic_label_stone():
    """Low-saturation medium-brightness region → stone."""
    label = SAMVitHBackend._heuristic_label(r=0.45, g=0.47, b=0.46, brightness=0.46)
    assert label == "stone"


def test_heuristic_label_none_for_ambiguous():
    """Very bright white region without clear blue tint → None."""
    label = SAMVitHBackend._heuristic_label(r=0.95, g=0.93, b=0.92, brightness=0.93)
    assert label is None


# =============================================================================
# 9. get_runtime_metadata
# =============================================================================


def test_get_runtime_metadata_before_load_returns_none():
    """get_runtime_metadata() returns None before load()."""
    backend = SAMVitHBackend()
    assert backend.get_runtime_metadata() is None


def test_get_runtime_metadata_after_mock_load():
    """get_runtime_metadata() returns a dict after load completes (mocked)."""
    backend = SAMVitHBackend(checkpoint_path="/fake/path.pth")
    # Simulate post-load state
    backend._model_loaded = True
    backend._device = "cpu"
    backend._runtime_metadata = {
        "backend": "sam_vit_h",
        "model_id": "facebook/sam-vit-huge",
        "device": "cpu",
        "checkpoint": "/fake/path.pth",
        "points_per_side": 32,
        "pred_iou_thresh": 0.88,
        "confidence_threshold": 0.85,
    }
    meta = backend.get_runtime_metadata()
    assert meta is not None
    assert meta["backend"] == "sam_vit_h"
    assert meta["device"] == "cpu"


# =============================================================================
# 10. Pinned-checkpoint enforcement (apex_research preset cross-reference)
# =============================================================================

_PINNED_SAM_VIT_H_SHA256 = "a7bf3b02f3ebf1267aba913ff637d9a2d5c33d3173bb679e46d9f338c26f262e"


def test_sam_vit_h_expected_sha256_is_pinned():
    """SAMVitHBackend.EXPECTED_SHA256 must be the pinned canonical Meta release hash.

    The runtime constant is the fail-closed default: callers that do not
    populate EnhanceConfig.sam_vit_h_expected_sha256 still get checksum
    validation against the official checkpoint bytes served by CHECKPOINT_URL.
    """
    assert SAMVitHBackend.EXPECTED_SHA256 == _PINNED_SAM_VIT_H_SHA256


def test_sam_vit_h_expected_sha256_matches_apex_research_preset():
    """The backend default hash must match config/presets/apex_research.yaml.

    Drift between the runtime constant and the shipped preset would give
    operators conflicting checksum guidance and silently weaken the
    fail-closed posture. Parse the preset YAML directly rather than the
    inheritance-merged form so this test stays decoupled from preset loaders.
    """
    import yaml

    repo_root = Path(__file__).resolve().parents[2]
    preset_path = repo_root / "config" / "presets" / "apex_research.yaml"
    with preset_path.open(encoding="utf-8") as fp:
        # YAML_GOVERNANCE_EXEMPT: contract test reads a single preset for cross-reference.
        preset = yaml.safe_load(fp)
    preset_hash = preset["segmentation"]["expected_sha256"]
    assert preset_hash == SAMVitHBackend.EXPECTED_SHA256, (
        f"apex_research.yaml segmentation.expected_sha256 ({preset_hash!r}) "
        f"drifted from SAMVitHBackend.EXPECTED_SHA256 ({SAMVitHBackend.EXPECTED_SHA256!r}). "
        "Update both together."
    )


def test_sam_vit_h_load_invokes_validation_against_class_default(tmp_path: Path, monkeypatch):
    """load() must invoke _validate_checkpoint_sha256 with the class default
    when no explicit expected_sha256 is supplied, so tampered checkpoint bytes
    fail closed even when callers leave EnhanceConfig.sam_vit_h_expected_sha256 unset.
    """
    import transformation_portal.lux_depth_v3.segmentation.sam_vit_h as sam_module

    monkeypatch.setattr(sam_module, "SAM_AVAILABLE", True)
    monkeypatch.setattr(sam_module, "TORCH_AVAILABLE", True)

    checkpoint = tmp_path / SAMVitHBackend.CHECKPOINT_FILENAME
    checkpoint.write_bytes(b"tampered checkpoint bytes - not the real SAM release")

    captured: Dict[str, object] = {}

    def fake_validate(path, expected):
        captured["path"] = Path(path)
        captured["expected"] = expected
        raise RuntimeError(f"SHA-256 mismatch: expected {expected}, got tampered")

    monkeypatch.setattr(SAMVitHBackend, "_validate_checkpoint_sha256", staticmethod(fake_validate))

    backend = SAMVitHBackend(checkpoint_path=str(checkpoint))
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        backend.load(device="cpu")

    assert captured["path"].resolve() == checkpoint.resolve()
    assert captured["expected"] == _PINNED_SAM_VIT_H_SHA256


def test_sam_vit_h_load_explicit_expected_sha256_overrides_class_default(tmp_path: Path, monkeypatch):
    """An explicit expected_sha256 argument must take precedence over the
    class-level EXPECTED_SHA256 (preserves the override path documented on
    the backend so fine-tuned checkpoints remain usable)."""
    import transformation_portal.lux_depth_v3.segmentation.sam_vit_h as sam_module

    monkeypatch.setattr(sam_module, "SAM_AVAILABLE", True)
    monkeypatch.setattr(sam_module, "TORCH_AVAILABLE", True)

    checkpoint = tmp_path / SAMVitHBackend.CHECKPOINT_FILENAME
    checkpoint.write_bytes(b"fine-tuned checkpoint bytes")

    captured: Dict[str, object] = {}
    override_hash = "b" * 64

    def fake_validate(path, expected):
        captured["expected"] = expected
        raise RuntimeError("stop after validation call")

    monkeypatch.setattr(SAMVitHBackend, "_validate_checkpoint_sha256", staticmethod(fake_validate))

    backend = SAMVitHBackend(checkpoint_path=str(checkpoint))
    with pytest.raises(RuntimeError, match="stop after validation call"):
        backend.load(device="cpu", expected_sha256=override_hash)

    assert captured["expected"] == override_hash
