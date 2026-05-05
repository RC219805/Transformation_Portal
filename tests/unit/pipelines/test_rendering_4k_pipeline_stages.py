"""Unit tests for rendering_4k_pipeline stage functions.

Tests the six module-level pure stage functions (apply_tone_mapping,
apply_material_response, apply_color_grading, apply_upscaling) and config
dataclass defaults — without executing the full Rendering4KPipeline or
requiring GPU/ML dependencies.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

pytestmark = [pytest.mark.unit]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _float32_image(h: int = 64, w: int = 64) -> np.ndarray:
    """Random float32 image in [0, 1]."""
    rng = np.random.default_rng(0)
    return rng.random((h, w, 3), dtype=np.float32).astype(np.float32)


def _pil_image(w: int = 64, h: int = 64) -> Image.Image:
    arr = (np.random.default_rng(1).random((h, w, 3)) * 255).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


# ---------------------------------------------------------------------------
# ToneMappingMethod enum
# ---------------------------------------------------------------------------


class TestToneMappingMethodEnum:
    def test_all_methods_exist(self):
        from transformation_portal.pipelines.rendering_4k_pipeline import ToneMappingMethod

        values = {m.value for m in ToneMappingMethod}
        for expected in ("agx", "filmic", "reinhard", "aces"):
            assert expected in values

    def test_agx_value(self):
        from transformation_portal.pipelines.rendering_4k_pipeline import ToneMappingMethod

        assert ToneMappingMethod.AGX.value == "agx"


# ---------------------------------------------------------------------------
# QualityLevel enum
# ---------------------------------------------------------------------------


class TestQualityLevelEnum:
    def test_all_levels_exist(self):
        from transformation_portal.pipelines.rendering_4k_pipeline import QualityLevel

        values = {q.value for q in QualityLevel}
        for expected in ("preview", "standard", "high", "ultra"):
            assert expected in values


# ---------------------------------------------------------------------------
# ToneMappingConfig defaults
# ---------------------------------------------------------------------------


class TestToneMappingConfigDefaults:
    def test_enabled_by_default(self):
        from transformation_portal.pipelines.rendering_4k_pipeline import ToneMappingConfig

        assert ToneMappingConfig().enabled is True

    def test_default_method_is_agx(self):
        from transformation_portal.pipelines.rendering_4k_pipeline import ToneMappingConfig, ToneMappingMethod

        assert ToneMappingConfig().method == ToneMappingMethod.AGX

    def test_default_exposure_zero(self):
        from transformation_portal.pipelines.rendering_4k_pipeline import ToneMappingConfig

        assert ToneMappingConfig().exposure == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# UpscalingConfig defaults
# ---------------------------------------------------------------------------


class TestUpscalingConfigDefaults:
    def test_enabled_by_default(self):
        from transformation_portal.pipelines.rendering_4k_pipeline import UpscalingConfig

        assert UpscalingConfig().enabled is True

    def test_default_target_4k(self):
        from transformation_portal.pipelines.rendering_4k_pipeline import UpscalingConfig

        w, h = UpscalingConfig().target_resolution
        assert w == 3840
        assert h == 2160

    def test_default_method_lanczos(self):
        from transformation_portal.pipelines.rendering_4k_pipeline import UpscalingConfig

        assert UpscalingConfig().method == "lanczos"


# ---------------------------------------------------------------------------
# apply_tone_mapping()
# ---------------------------------------------------------------------------


class TestApplyToneMapping:
    def test_output_shape_matches_input(self):
        from transformation_portal.pipelines.rendering_4k_pipeline import ToneMappingConfig, apply_tone_mapping

        image = _float32_image()
        result = apply_tone_mapping(image, ToneMappingConfig())
        assert result.shape == image.shape

    def test_disabled_returns_clipped_input(self):
        from transformation_portal.pipelines.rendering_4k_pipeline import ToneMappingConfig, apply_tone_mapping

        image = _float32_image() * 2.0  # values > 1
        config = ToneMappingConfig(enabled=False)
        result = apply_tone_mapping(image, config)
        assert result.max() <= 1.0 + 1e-6
        assert result.min() >= -1e-6

    def test_output_in_unit_range(self):
        from transformation_portal.pipelines.rendering_4k_pipeline import ToneMappingConfig, apply_tone_mapping

        image = _float32_image() * 5.0  # HDR values
        result = apply_tone_mapping(image, ToneMappingConfig())
        assert result.max() <= 1.0 + 1e-4
        assert result.min() >= -1e-4

    def test_returns_numpy_array(self):
        from transformation_portal.pipelines.rendering_4k_pipeline import ToneMappingConfig, apply_tone_mapping

        result = apply_tone_mapping(_float32_image(), ToneMappingConfig())
        assert isinstance(result, np.ndarray)

    @pytest.mark.parametrize("method", ["agx", "filmic", "reinhard", "aces"])
    def test_all_methods_produce_correct_shape(self, method):
        from transformation_portal.pipelines.rendering_4k_pipeline import (
            ToneMappingConfig,
            ToneMappingMethod,
            apply_tone_mapping,
        )

        image = _float32_image()
        config = ToneMappingConfig(method=ToneMappingMethod(method))
        result = apply_tone_mapping(image, config)
        assert result.shape == image.shape

    def test_positive_exposure_brightens(self):
        from transformation_portal.pipelines.rendering_4k_pipeline import ToneMappingConfig, apply_tone_mapping

        image = np.full((16, 16, 3), 0.2, dtype=np.float32)
        dark = apply_tone_mapping(image, ToneMappingConfig(exposure=0.0))
        bright = apply_tone_mapping(image, ToneMappingConfig(exposure=2.0))
        assert bright.mean() >= dark.mean()


# ---------------------------------------------------------------------------
# apply_material_response()
# ---------------------------------------------------------------------------


class TestApplyMaterialResponse:
    def test_output_shape_matches_input(self):
        from transformation_portal.pipelines.rendering_4k_pipeline import MaterialResponseConfig, apply_material_response

        image = _float32_image()
        result = apply_material_response(image, None, MaterialResponseConfig())
        assert result.shape == image.shape

    def test_disabled_returns_unchanged_reference(self):
        from transformation_portal.pipelines.rendering_4k_pipeline import MaterialResponseConfig, apply_material_response

        image = _float32_image()
        config = MaterialResponseConfig(enabled=False)
        result = apply_material_response(image, None, config)
        np.testing.assert_array_equal(result, image)

    def test_with_depth_map_maintains_shape(self):
        from transformation_portal.pipelines.rendering_4k_pipeline import MaterialResponseConfig, apply_material_response

        image = _float32_image(32, 32)
        depth = np.random.default_rng(2).random((32, 32)).astype(np.float32)
        result = apply_material_response(image, depth, MaterialResponseConfig())
        assert result.shape == image.shape

    def test_output_clipped_to_unit_range(self):
        from transformation_portal.pipelines.rendering_4k_pipeline import MaterialResponseConfig, apply_material_response

        image = _float32_image()
        result = apply_material_response(image, None, MaterialResponseConfig())
        assert result.min() >= -1e-6
        assert result.max() <= 1.0 + 1e-6


# ---------------------------------------------------------------------------
# apply_color_grading()
# ---------------------------------------------------------------------------


class TestApplyColorGrading:
    def test_output_shape_matches_input(self):
        from transformation_portal.pipelines.rendering_4k_pipeline import ColorGradingConfig, apply_color_grading

        image = _float32_image()
        result = apply_color_grading(image, ColorGradingConfig())
        assert result.shape == image.shape

    def test_disabled_returns_unchanged_reference(self):
        from transformation_portal.pipelines.rendering_4k_pipeline import ColorGradingConfig, apply_color_grading

        image = _float32_image()
        config = ColorGradingConfig(enabled=False)
        result = apply_color_grading(image, config)
        np.testing.assert_array_equal(result, image)

    def test_returns_numpy_array(self):
        from transformation_portal.pipelines.rendering_4k_pipeline import ColorGradingConfig, apply_color_grading

        result = apply_color_grading(_float32_image(), ColorGradingConfig())
        assert isinstance(result, np.ndarray)

    def test_temperature_shift_changes_image(self):
        """RGB multiplier shift should change the image from the neutral baseline."""
        from transformation_portal.pipelines.rendering_4k_pipeline import ColorGradingConfig, apply_color_grading

        image = np.full((16, 16, 3), 0.5, dtype=np.float32)
        # Warm shift: boost red, reduce blue
        config = ColorGradingConfig(enabled=True, temperature_shift=(1.2, 1.0, 0.8))
        result = apply_color_grading(image, config)
        assert not np.allclose(result, image)


# ---------------------------------------------------------------------------
# apply_upscaling()
# ---------------------------------------------------------------------------


class TestApplyUpscaling:
    def test_disabled_returns_original_image(self):
        from transformation_portal.pipelines.rendering_4k_pipeline import UpscalingConfig, apply_upscaling

        pil_img = _pil_image(64, 64)
        config = UpscalingConfig(enabled=False)
        result = apply_upscaling(pil_img, config)
        assert result is pil_img

    def test_returns_pil_image(self):
        from transformation_portal.pipelines.rendering_4k_pipeline import UpscalingConfig, apply_upscaling

        config = UpscalingConfig(enabled=True, target_resolution=(128, 128))
        result = apply_upscaling(_pil_image(64, 64), config)
        assert isinstance(result, Image.Image)

    def test_upscaled_image_larger_than_input(self):
        from transformation_portal.pipelines.rendering_4k_pipeline import UpscalingConfig, apply_upscaling

        small = _pil_image(32, 32)
        config = UpscalingConfig(enabled=True, target_resolution=(128, 128))
        result = apply_upscaling(small, config)
        assert result.size[0] >= small.size[0]
        assert result.size[1] >= small.size[1]

    def test_already_large_image_not_upscaled_beyond_target(self):
        """Image already larger than target should be returned as-is."""
        from transformation_portal.pipelines.rendering_4k_pipeline import UpscalingConfig, apply_upscaling

        large = _pil_image(200, 200)
        config = UpscalingConfig(enabled=True, target_resolution=(128, 128))
        result = apply_upscaling(large, config)
        # Should return original since it's already above target
        assert result.size == large.size

    def test_one_dimension_above_target_does_not_downscale(self):
        """Fitting to target must not shrink an image in the upscaling stage."""
        from transformation_portal.pipelines.rendering_4k_pipeline import UpscalingConfig, apply_upscaling

        wide = _pil_image(256, 64)
        config = UpscalingConfig(enabled=True, target_resolution=(128, 128))
        result = apply_upscaling(wide, config)

        assert result is wide
        assert result.size == wide.size
