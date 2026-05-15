"""Unit tests for the material-response engine.

Phase 5 coverage. Exercises MaterialResponseEngine -- the pure numpy/PIL
physics-based surface enhancer that drives the luxury-interior profiles.
No ML / GPU dependencies; scipy is required at runtime (the no-scipy
early-return branch is exercised via monkeypatch).
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from transformation_portal.processors.material_response import engine as engine_module
from transformation_portal.processors.material_response.engine import (
    MaterialMask,
    MaterialResponseConfig,
    MaterialResponseEngine,
)

pytestmark = pytest.mark.unit


def _make_image(size: int = 32, fill: int = 128) -> Image.Image:
    """A small deterministic RGB PIL image."""
    return Image.fromarray(np.full((size, size, 3), fill, dtype=np.uint8), "RGB")


def _make_rgb(size: int = 32, seed: int = 0) -> np.ndarray:
    """A small deterministic RGB float32 array in [0, 1]."""
    rng = np.random.default_rng(seed)
    return rng.random((size, size, 3), dtype=np.float32)


def _assert_unit_rgb_array(arr: np.ndarray, shape: tuple[int, int, int]) -> None:
    """Assert an RGB helper result stays finite, shaped, and clipped."""
    assert arr.shape == shape
    assert np.isfinite(arr).all()
    assert float(arr.min()) >= 0.0
    assert float(arr.max()) <= 1.0


@pytest.fixture
def engine() -> MaterialResponseEngine:
    return MaterialResponseEngine(MaterialResponseConfig())


class TestMaterialResponseConfig:
    """Tests for the MaterialResponseConfig dataclass."""

    def test_defaults(self) -> None:
        cfg = MaterialResponseConfig()

        assert cfg.profile == "luxury_interior"
        assert cfg.texture_boost == pytest.approx(0.25)
        assert cfg.haze_tint == (0.82, 0.88, 0.96)

    def test_post_init_clamps_out_of_range_values(self) -> None:
        cfg = MaterialResponseConfig(
            texture_boost=1.5,
            ambient_occlusion=-0.5,
            highlight_warmth=2.0,
            haze_strength=-1.0,
            floor_plank_contrast=99.0,
            floor_specular=-99.0,
            textile_contrast=1.001,
            leather_sheen=-0.001,
            window_light_wrap=5.0,
            window_reflection=-5.0,
            wall_texture=2.0,
        )

        assert cfg.texture_boost == pytest.approx(1.0)
        assert cfg.ambient_occlusion == pytest.approx(0.0)
        assert cfg.highlight_warmth == pytest.approx(1.0)
        assert cfg.haze_strength == pytest.approx(0.0)
        assert cfg.floor_plank_contrast == pytest.approx(1.0)
        assert cfg.floor_specular == pytest.approx(0.0)
        assert cfg.textile_contrast == pytest.approx(1.0)
        assert cfg.leather_sheen == pytest.approx(0.0)
        assert cfg.window_light_wrap == pytest.approx(1.0)
        assert cfg.window_reflection == pytest.approx(0.0)
        assert cfg.wall_texture == pytest.approx(1.0)


class TestMaterialMask:
    """Tests for the MaterialMask dataclass."""

    def test_holds_named_mask_arrays(self) -> None:
        mask_shape = (4, 4)
        mask = MaterialMask(
            floor=np.zeros(mask_shape),
            wall=np.zeros(mask_shape),
            textile=np.zeros(mask_shape),
            wood=np.zeros(mask_shape),
            metal=np.zeros(mask_shape),
            highlight=np.zeros(mask_shape),
            midtone=np.ones(mask_shape),
        )

        assert mask.floor.shape == mask_shape
        assert mask.midtone.sum() == 16


class TestEngineConstructionAndFactory:
    """Tests for __init__ and the from_config classmethod."""

    def test_init_stores_config(self) -> None:
        cfg = MaterialResponseConfig(texture_boost=0.4)
        eng = MaterialResponseEngine(cfg)
        assert eng.config is cfg

    def test_from_config_merges_with_profile_defaults(self) -> None:
        eng = MaterialResponseEngine.from_config({"profile": "wood_floor_oak", "texture_boost": 0.42})

        # Explicit values override profile defaults.
        assert eng.config.texture_boost == pytest.approx(0.42)
        assert eng.config.profile == "wood_floor_oak"

        # Inherited values come from the selected profile, not dataclass defaults.
        assert eng.config.ambient_occlusion == pytest.approx(0.15)
        assert eng.config.floor_plank_contrast == pytest.approx(0.22)
        assert eng.config.haze_tint == (0.85, 0.82, 0.78)

    def test_from_config_accepts_haze_tint_as_list(self) -> None:
        eng = MaterialResponseEngine.from_config({"haze_tint": [0.9, 0.7, 0.5, 1.0]})

        # haze_tint is normalized to a 3-tuple (the 4th element is dropped).
        assert eng.config.haze_tint == (0.9, 0.7, 0.5)

    def test_from_config_rejects_unknown_profile(self) -> None:
        with pytest.raises(KeyError, match="Invalid profile name"):
            MaterialResponseEngine.from_config({"profile": "nonexistent-profile"})


class TestApplyEndToEnd:
    """Tests for MaterialResponseEngine.apply()."""

    def test_apply_returns_pil_image_with_input_size(self, engine: MaterialResponseEngine) -> None:
        image = _make_image(size=48)

        out = engine.apply(image, strength=0.5)

        assert isinstance(out, Image.Image)
        assert out.size == image.size
        assert out.mode == "RGB"

    def test_apply_converts_non_rgb_input(self, engine: MaterialResponseEngine) -> None:
        # The engine converts non-RGB modes (e.g. RGBA) to RGB before processing.
        image = Image.new("RGBA", (16, 16), color=(200, 100, 50, 255))

        out = engine.apply(image)

        assert out.mode == "RGB"

    def test_apply_returns_input_when_scipy_unavailable(
        self, engine: MaterialResponseEngine, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(engine_module, "HAS_SCIPY", False)
        image = _make_image()

        out = engine.apply(image)

        # The early-return path hands back the input unchanged (same object).
        assert out is image

    def test_apply_is_a_no_op_when_config_and_call_strength_are_zero(self) -> None:
        # All enhancement channels and the call-level strength are disabled,
        # including the metal stage that is controlled by strength alone.
        cfg = MaterialResponseConfig(
            texture_boost=0.0,
            ambient_occlusion=0.0,
            highlight_warmth=0.0,
            haze_strength=0.0,
            floor_plank_contrast=0.0,
            floor_specular=0.0,
            textile_contrast=0.0,
            leather_sheen=0.0,
            window_light_wrap=0.0,
            window_reflection=0.0,
            wall_texture=0.0,
        )
        eng = MaterialResponseEngine(cfg)
        image = _make_image()

        out = eng.apply(image, strength=0.0)

        assert isinstance(out, Image.Image)
        assert out.size == image.size
        np.testing.assert_array_equal(np.asarray(out), np.asarray(image))


class TestEnhancementStages:
    """Tests for the individual enhance_* and _* numpy helpers."""

    def test_compute_material_masks_returns_named_masks(self, engine: MaterialResponseEngine) -> None:
        rgb = _make_rgb(size=24, seed=1)

        masks = engine._compute_material_masks(rgb)

        assert isinstance(masks, MaterialMask)
        for field_name in ("floor", "wall", "textile", "wood", "metal", "highlight", "midtone"):
            arr = getattr(masks, field_name)
            assert arr.shape[:2] == rgb.shape[:2]
            assert np.all((arr >= 0.0) & (arr <= 1.0))

    def test_enhance_floor_preserves_shape_and_range(self, engine: MaterialResponseEngine) -> None:
        rgb = _make_rgb(seed=2)
        floor_mask = np.ones(rgb.shape[:2], dtype=np.float32) * 0.5
        wood_mask = np.ones(rgb.shape[:2], dtype=np.float32) * 0.3

        out = engine.enhance_floor(rgb, floor_mask, wood_mask, strength=0.7)

        _assert_unit_rgb_array(out, rgb.shape)

    def test_enhance_textiles_preserves_shape(self, engine: MaterialResponseEngine) -> None:
        rgb = _make_rgb(seed=3)
        textile_mask = np.ones(rgb.shape[:2], dtype=np.float32) * 0.4

        out = engine.enhance_textiles(rgb, textile_mask, strength=0.5)

        _assert_unit_rgb_array(out, rgb.shape)

    def test_enhance_metals_preserves_shape(self, engine: MaterialResponseEngine) -> None:
        rgb = _make_rgb(seed=4)
        metal_mask = np.ones(rgb.shape[:2], dtype=np.float32) * 0.6

        out = engine.enhance_metals(rgb, metal_mask, strength=0.5)

        _assert_unit_rgb_array(out, rgb.shape)

    def test_add_atmospheric_effects_preserves_shape(self, engine: MaterialResponseEngine) -> None:
        rgb = _make_rgb(seed=5)
        h, w = rgb.shape[:2]

        out = engine.add_atmospheric_effects(rgb, h, w, strength=0.5)

        _assert_unit_rgb_array(out, rgb.shape)
