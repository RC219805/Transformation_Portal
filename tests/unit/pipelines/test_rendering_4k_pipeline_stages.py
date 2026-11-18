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
# .cube LUT application (H-02: red/blue axis order)
# ---------------------------------------------------------------------------


def _write_cube(path, size, transfer) -> None:
    """Write a spec-conformant .cube file: red index fastest, then green, then blue."""
    lines = ['TITLE "test"', f"LUT_3D_SIZE {size}", "DOMAIN_MIN 0.0 0.0 0.0", "DOMAIN_MAX 1.0 1.0 1.0"]
    for b in range(size):
        for g in range(size):
            for r in range(size):
                out = transfer(r / (size - 1), g / (size - 1), b / (size - 1))
                lines.append(" ".join(f"{v:.6f}" for v in out))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


_LUT_PIXELS = np.array([[[1.0, 0.25, 0.0], [0.0, 0.5, 1.0], [0.2, 0.4, 0.6]]], dtype=np.float32)


class TestCubeLutApplication:
    def test_identity_lut_returns_input(self, tmp_path):
        from transformation_portal.pipelines.rendering_4k import stages

        cube = tmp_path / "identity.cube"
        _write_cube(cube, 5, lambda r, g, b: (r, g, b))

        np.testing.assert_allclose(stages._apply_lut(_LUT_PIXELS, cube, strength=1.0), _LUT_PIXELS, atol=1e-5)
        image = _float32_image(16, 16)
        np.testing.assert_allclose(stages._apply_lut(image, cube, strength=1.0), image, atol=1e-5)

    def test_red_blue_swap_lut_swaps_red_and_blue(self, tmp_path):
        from transformation_portal.pipelines.rendering_4k import stages

        cube = tmp_path / "swap_rb.cube"
        _write_cube(cube, 5, lambda r, g, b: (b, g, r))

        image = _float32_image(16, 16)
        result = stages._apply_lut(image, cube, strength=1.0)
        np.testing.assert_allclose(result, image[..., ::-1], atol=1e-5)

    def test_trilinear_interpolation_matches_multilinear_transfer(self, tmp_path):
        """A transfer that is linear in each input axis is reproduced exactly between grid nodes."""
        from transformation_portal.pipelines.rendering_4k import stages

        def transfer(r, g, b):
            return (r * g, g * (1.0 - b), 0.25 + 0.5 * b * r)

        cube = tmp_path / "multilinear.cube"
        _write_cube(cube, 3, transfer)

        image = _float32_image(16, 16)
        r, g, b = image[..., 0], image[..., 1], image[..., 2]
        expected = np.stack(transfer(r, g, b), axis=-1)
        np.testing.assert_allclose(stages._apply_lut(image, cube, strength=1.0), expected, atol=1e-5)

    def test_partial_strength_blends_lut_output_with_input(self, tmp_path):
        from transformation_portal.pipelines.rendering_4k import stages

        cube = tmp_path / "swap_rb.cube"
        _write_cube(cube, 5, lambda r, g, b: (b, g, r))

        result = stages._apply_lut(_LUT_PIXELS, cube, strength=0.25)
        np.testing.assert_allclose(result, 0.75 * _LUT_PIXELS + 0.25 * _LUT_PIXELS[..., ::-1], atol=1e-5)

    def test_color_grading_applies_lut_in_channel_order(self, tmp_path):
        from transformation_portal.pipelines.rendering_4k.stages import apply_color_grading
        from transformation_portal.pipelines.rendering_4k.types import ColorGradingConfig

        cube = tmp_path / "swap_rb.cube"
        _write_cube(cube, 5, lambda r, g, b: (b, g, r))

        config = ColorGradingConfig(lut_paths=[str(cube)], lut_strengths=[1.0], saturation=1.0, vibrance=1.0)
        result = apply_color_grading(_LUT_PIXELS, config)
        np.testing.assert_allclose(result, _LUT_PIXELS[..., ::-1], atol=1e-5)

    def test_invalid_lut_is_skipped_with_none(self, tmp_path):
        from transformation_portal.pipelines.rendering_4k import stages

        cube = tmp_path / "truncated.cube"
        cube.write_text("LUT_3D_SIZE 4\n0 0 0\n1 1 1\n", encoding="utf-8")

        assert stages._load_cube_lut(cube) is None
        assert stages._apply_lut(_LUT_PIXELS, cube, strength=1.0) is None


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
