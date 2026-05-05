"""Compatibility tests for the Phase 4B rendering_4k stage extraction."""

from __future__ import annotations

import importlib
import sys

import numpy as np
import pytest
from PIL import Image

pytestmark = [pytest.mark.unit]


PUBLIC_STAGE_SYMBOLS = (
    "apply_color_grading",
    "apply_material_response",
    "apply_tone_mapping",
    "apply_upscaling",
    "estimate_depth_simple",
)

PRIVATE_STAGE_COMPAT_SYMBOLS = (
    "_aces_approximation",
    "_agx_sigmoid",
    "_apply_local_contrast",
    "_apply_lut",
    "_apply_vibrance",
    "_filmic_hable",
    "_load_cube_lut",
    "_simple_box_blur",
    "_simple_gaussian_blur",
    "_simple_gaussian_blur_2d",
)


def _float32_image(height: int = 16, width: int = 16) -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.random((height, width, 3), dtype=np.float32)


def test_legacy_rendering_module_reexports_phase_4b_public_stages() -> None:
    legacy = importlib.import_module("transformation_portal.pipelines.rendering_4k_pipeline")
    extracted = importlib.import_module("transformation_portal.pipelines.rendering_4k.stages")

    for symbol in PUBLIC_STAGE_SYMBOLS:
        assert getattr(legacy, symbol) is getattr(extracted, symbol)


def test_legacy_rendering_module_keeps_phase_4b_private_helper_exports() -> None:
    legacy = importlib.import_module("transformation_portal.pipelines.rendering_4k_pipeline")
    extracted = importlib.import_module("transformation_portal.pipelines.rendering_4k.stages")

    for symbol in PRIVATE_STAGE_COMPAT_SYMBOLS:
        assert getattr(legacy, symbol) is getattr(extracted, symbol)


def test_extracted_stage_logger_preserves_legacy_logging_surface() -> None:
    legacy = importlib.import_module("transformation_portal.pipelines.rendering_4k_pipeline")
    extracted = importlib.import_module("transformation_portal.pipelines.rendering_4k.stages")

    assert extracted.logger.name == legacy.logger.name


def test_rendering_4k_package_reexports_phase_4b_public_stages_only() -> None:
    package = importlib.import_module("transformation_portal.pipelines.rendering_4k")
    extracted = importlib.import_module("transformation_portal.pipelines.rendering_4k.stages")

    for symbol in PUBLIC_STAGE_SYMBOLS:
        assert getattr(package, symbol) is getattr(extracted, symbol)
        assert symbol in package.__all__

    for symbol in PRIVATE_STAGE_COMPAT_SYMBOLS:
        assert not hasattr(package, symbol)
        assert symbol not in package.__all__


def test_rendering_4k_package_defers_stage_module_import(monkeypatch: pytest.MonkeyPatch) -> None:
    package_name = "transformation_portal.pipelines.rendering_4k"
    stages_name = f"{package_name}.stages"

    monkeypatch.delitem(sys.modules, stages_name, raising=False)
    monkeypatch.delitem(sys.modules, package_name, raising=False)

    package = importlib.import_module(package_name)

    assert stages_name not in sys.modules

    stages = importlib.import_module(stages_name)
    assert getattr(package, "apply_tone_mapping") is stages.apply_tone_mapping


def test_extracted_tone_mapping_smoke_preserves_shape_dtype_and_range() -> None:
    from transformation_portal.pipelines.rendering_4k.stages import apply_tone_mapping
    from transformation_portal.pipelines.rendering_4k.types import ToneMappingConfig

    image = _float32_image() * 4.0
    result = apply_tone_mapping(image, ToneMappingConfig())

    assert result.shape == image.shape
    assert result.dtype == np.float32
    assert result.min() >= -1e-6
    assert result.max() <= 1.0 + 1e-6


def test_extracted_lut_miss_falls_back_without_replacing_image(tmp_path) -> None:
    from transformation_portal.pipelines.rendering_4k import stages
    from transformation_portal.pipelines.rendering_4k.types import ColorGradingConfig

    image = _float32_image()
    missing_lut = tmp_path / "missing.cube"

    assert stages._apply_lut(image, missing_lut, strength=0.75) is None

    result = stages.apply_color_grading(
        image,
        ColorGradingConfig(lut_paths=[str(missing_lut)], lut_strengths=[0.75]),
    )

    assert result.shape == image.shape
    assert result.dtype == np.float32
    assert result.min() >= -1e-6
    assert result.max() <= 1.0 + 1e-6


def test_extracted_upscaling_smoke_preserves_aspect_ratio() -> None:
    from transformation_portal.pipelines.rendering_4k.stages import apply_upscaling
    from transformation_portal.pipelines.rendering_4k.types import UpscalingConfig

    image = Image.fromarray(np.full((4, 8, 3), 128, dtype=np.uint8), mode="RGB")
    result = apply_upscaling(
        image,
        UpscalingConfig(target_resolution=(16, 16), preserve_sharpness=False),
    )

    assert result.size == (16, 8)


def test_extracted_simple_depth_smoke_preserves_shape_dtype_and_range() -> None:
    from transformation_portal.pipelines.rendering_4k.stages import estimate_depth_simple

    x = np.linspace(0.05, 0.95, 16, dtype=np.float32)
    y = np.linspace(0.15, 0.85, 12, dtype=np.float32)[:, np.newaxis]
    image = np.stack(
        [
            np.broadcast_to(x, (12, 16)),
            np.broadcast_to(y, (12, 16)),
            np.broadcast_to((x + y) / 2.0, (12, 16)),
        ],
        axis=2,
    ).astype(np.float32)

    depth = estimate_depth_simple(image)

    assert depth.shape == image.shape[:2]
    assert depth.dtype == np.float32
    assert depth.min() >= -1e-6
    assert depth.max() <= 1.0 + 1e-6
