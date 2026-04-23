"""Regression tests for the vertical luminance banding ("paneling") fix.

These tests lock in the behavior of the banding-mitigation patches:

1. V2 tone mapping (`EnhancementStage._apply_tone_mapping`): a depth map
   with high-frequency vertical striping must not project into luminance
   bands on a smooth sky; a completely flat frame must come out essentially
   unchanged; a textured frame must still get the designed tone shift.

2. Materials V3 pixel ops (`apply_pixel_ops`): large low-texture materials
   (sky, water) must get a widened feather and a delta clamp so a flat
   per-material gain never steps at the mask boundary; small or textured
   materials must not trigger the guard.

3. Config wiring: the new `keep_intermediates` flag is on `EnhanceConfig`
   and the V2 preset dataclass carries the new tone-mapping knobs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from transformation_portal.lux_depth_v3.pixel_ops_executor import apply_pixel_ops
from transformation_portal.lux_depth_v3.pixel_ops_registry import OP_REGISTRY
from transformation_portal.lux_depth_v3.v2_presets import V2EnhancementConfig
from transformation_portal.stage_graph.stages.enhancement import EnhancementStage

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Stage 1: V2 tone-mapping regression
# ---------------------------------------------------------------------------


def _smooth_sky_image(h: int = 256, w: int = 256) -> np.ndarray:
    """Vertical top-to-bottom sky gradient, float32 in [0, 1]."""
    col = np.linspace(0.45, 0.85, h, dtype=np.float32)
    luma = np.broadcast_to(col[:, None], (h, w)).copy()
    # Tint it slightly like dusk so the tone curve has room to do something.
    img = np.stack([luma * 0.95, luma * 0.92, luma * 1.00], axis=-1)
    return np.clip(img, 0.0, 1.0).astype(np.float32)


def _striped_depth_map(h: int, w: int, stripe_amp: float = 0.08) -> np.ndarray:
    """Smooth depth + an alternating per-column ripple.

    The ripple is the upstream depth pattern that produces visible vertical
    luminance banding when the multiplier is built directly from raw depth.
    A 4-pixel cycle is used (rather than 1-pixel Nyquist alternation) so it
    resembles striping from a tiled depth backend, not pixel-grid aliasing.
    """
    base = np.linspace(0.15, 0.70, w, dtype=np.float32)  # near → far along x
    base = np.broadcast_to(base[None, :], (h, w)).copy()
    cols = np.arange(w, dtype=np.float32)
    # Square-wave-ish 4-column cycle; amplitude swings ±stripe_amp.
    ripple = (stripe_amp * np.sign(np.sin(cols * np.pi / 2.0))).astype(np.float32)
    return np.clip(base + ripple[None, :], 0.0, 1.0)


def test_v2_depth_stripe_does_not_project_into_luminance_bands():
    """The depth-smoothing guard must suppress per-column luminance ripple."""
    img = _smooth_sky_image()
    depth = _striped_depth_map(*img.shape[:2])

    # Guard ON (default).
    stage_on = EnhancementStage(
        enhancement_strength=1.0,
        clarity_strength=0.0,
        material_strength=0.0,
        tone_depth_smoothing=True,
        tone_low_tex_strength=0.6,
    )
    out_on = stage_on._apply_tone_mapping(img.copy(), depth.copy())

    # Guard OFF (simulate pre-fix behavior).
    stage_off = EnhancementStage(
        enhancement_strength=1.0,
        clarity_strength=0.0,
        material_strength=0.0,
        tone_depth_smoothing=False,
        tone_low_tex_strength=0.0,
    )
    out_off = stage_off._apply_tone_mapping(img.copy(), depth.copy())

    def column_ripple_energy(frame: np.ndarray) -> float:
        luma = 0.2126 * frame[..., 0] + 0.7152 * frame[..., 1] + 0.0722 * frame[..., 2]
        col_mean = luma.mean(axis=0)
        # High-pass: difference between adjacent columns, which is where the
        # 1-pixel depth ripple concentrates.
        return float(np.abs(np.diff(col_mean)).mean())

    ripple_on = column_ripple_energy(out_on)
    ripple_off = column_ripple_energy(out_off)

    # The guard must reduce per-column ripple by at least an order of magnitude.
    assert ripple_on < ripple_off / 10.0, (
        f"Depth-stripe guard insufficient: ripple_on={ripple_on:.6f}, " f"ripple_off={ripple_off:.6f}"
    )


@pytest.mark.parametrize(
    ("shape", "expected_sigma"),
    [
        ((512, 1), 2.0),
        ((4096, 1), 4.0),
        ((16384, 1), 8.0),
    ],
)
def test_v2_depth_smoothing_uses_bounded_sigma(monkeypatch, shape, expected_sigma):
    """Depth smoothing must not scale into large Gaussian kernels on huge frames."""
    import scipy.ndimage

    captured = {}

    def fake_gaussian_filter(depth_map: np.ndarray, sigma: float) -> np.ndarray:
        captured["sigma"] = sigma
        return depth_map

    monkeypatch.setattr(scipy.ndimage, "gaussian_filter", fake_gaussian_filter)
    stage = EnhancementStage()

    depth = np.zeros(shape, dtype=np.float32)
    out = stage._lowpass_depth_for_tone(depth)

    assert out is depth
    assert captured["sigma"] == pytest.approx(expected_sigma)


def test_v2_low_gradient_guard_preserves_flat_frame():
    """A frame with no texture must come out within a tight epsilon of the input."""
    img = np.full((128, 128, 3), 0.5, dtype=np.float32)
    depth = np.random.default_rng(0).random((128, 128), dtype=np.float32)

    stage = EnhancementStage(
        enhancement_strength=1.0,
        clarity_strength=0.0,
        material_strength=0.0,
        tone_depth_smoothing=True,
        tone_low_tex_strength=1.0,  # maximum attenuation
    )
    out = stage._apply_tone_mapping(img.copy(), depth.copy())

    # With full low-tex attenuation, a flat frame should be essentially unchanged.
    assert np.max(np.abs(out - img)) < 0.005


def test_v2_textured_frame_still_receives_tone_shift():
    """Guards must NOT suppress the tone shift on a textured frame."""
    rng = np.random.default_rng(42)
    # Heavy texture: uniform noise on [0.2, 0.8].
    img = (0.2 + 0.6 * rng.random((128, 128, 3), dtype=np.float32)).astype(np.float32)
    depth = np.linspace(0.1, 0.9, 128, dtype=np.float32)
    depth = np.broadcast_to(depth[:, None], (128, 128)).copy()

    stage = EnhancementStage(
        enhancement_strength=1.0,
        clarity_strength=0.0,
        material_strength=0.0,
        tone_depth_smoothing=True,
        tone_low_tex_strength=0.6,
    )
    out = stage._apply_tone_mapping(img.copy(), depth.copy())

    # Non-trivial shift expected.
    assert np.max(np.abs(out - img)) > 0.01


# ---------------------------------------------------------------------------
# Stage 2: Materials V3 seam-safe guard regression
# ---------------------------------------------------------------------------


@dataclass
class _PxOpsConfig:
    apply_pixel_ops: bool = True
    min_coverage_px: int = 100
    min_mean_conf: float = 0.2
    refinement_strategy: str = "canary"
    sky_response_enabled: bool = True
    water_response_enabled: bool = True
    glass_response_enabled: bool = True
    mask_feather_sigma_default: float = 3.0
    mask_feather_sigma_overrides: dict = None  # type: ignore[assignment]
    mask_feather_disabled_materials: list = None  # type: ignore[assignment]
    # Seam-safe knobs
    pixel_ops_low_grad_threshold: float = 0.01
    pixel_ops_low_tex_min_bbox_frac: float = 0.05
    pixel_ops_low_tex_feather_multiplier: float = 8.0
    pixel_ops_low_tex_delta_ceiling: float = 0.04

    def __post_init__(self) -> None:
        if self.mask_feather_sigma_overrides is None:
            self.mask_feather_sigma_overrides = {}
        if self.mask_feather_disabled_materials is None:
            self.mask_feather_disabled_materials = []


def _full_image_sky_inputs(size: int = 128) -> tuple[np.ndarray, dict, dict]:
    """Large flat sky region covering most of the frame."""
    img = np.full((size, size, 3), 180, dtype=np.uint8)  # flat bright sky
    mask = np.zeros((size, size), dtype=np.float32)
    mask[: int(size * 0.6), :] = 1.0  # covers 60% of frame vertically
    segmentation_result = {"materials": {"sky": mask}}
    response_plan = {
        "per_class": {
            "sky": {
                "coverage_px": int(mask.sum()),
                "mean_conf": 0.6,
                "edge_conf": 0.5,
            }
        }
    }
    return img, segmentation_result, response_plan


def _textured_image_sky_inputs(size: int = 128) -> tuple[np.ndarray, dict, dict]:
    """Strong checkerboard texture under the sky mask."""
    rng = np.random.default_rng(7)
    img = rng.integers(0, 255, (size, size, 3), dtype=np.uint8)
    mask = np.zeros((size, size), dtype=np.float32)
    mask[: int(size * 0.6), :] = 1.0
    segmentation_result = {"materials": {"sky": mask}}
    response_plan = {
        "per_class": {
            "sky": {
                "coverage_px": int(mask.sum()),
                "mean_conf": 0.6,
                "edge_conf": 0.5,
            }
        }
    }
    return img, segmentation_result, response_plan


def test_materials_v3_guard_fires_on_large_flat_sky():
    """Low gradient + large bbox → adaptive feather widens, telemetry records it."""
    config = _PxOpsConfig()
    image, seg, plan = _full_image_sky_inputs()

    _, telemetry = apply_pixel_ops(image, seg, plan, config, registry=OP_REGISTRY)

    applied = [a for a in telemetry["applied"] if a["material"] == "sky"]
    assert applied, "sky ops should have been applied"
    sky = applied[0]

    assert sky["low_tex_guard"]["applied"] is True
    # Feather must have grown at least 4x the default (multiplier is 8x).
    assert sky["feather_sigma"] >= config.mask_feather_sigma_default * 4.0
    # The guard's gradient probe must report a low value for a flat sky.
    assert sky["low_tex_guard"]["roi_mean_grad"] < config.pixel_ops_low_grad_threshold


def test_materials_v3_guard_does_not_fire_on_textured_sky():
    """A textured ROI must be treated normally (no feather widening, no clamp)."""
    config = _PxOpsConfig()
    image, seg, plan = _textured_image_sky_inputs()

    _, telemetry = apply_pixel_ops(image, seg, plan, config, registry=OP_REGISTRY)

    applied = [a for a in telemetry["applied"] if a["material"] == "sky"]
    assert applied
    sky = applied[0]
    assert sky["low_tex_guard"]["applied"] is False
    # Feather sigma should be the configured default, unchanged.
    assert sky["feather_sigma"] == pytest.approx(config.mask_feather_sigma_default)
    assert sky["low_tex_guard"]["delta_scale_applied"] == pytest.approx(1.0)


def test_materials_v3_delta_clamp_caps_large_flat_shift():
    """Delta ceiling must cap an aggressive op on a large flat ROI."""
    # Set an aggressive pseudo-ceiling so the clamp engages even on the
    # ordinary sky preset ops.
    config = _PxOpsConfig(pixel_ops_low_tex_delta_ceiling=0.001)
    image, seg, plan = _full_image_sky_inputs()

    _, telemetry = apply_pixel_ops(image, seg, plan, config, registry=OP_REGISTRY)

    sky = [a for a in telemetry["applied"] if a["material"] == "sky"][0]
    assert sky["low_tex_guard"]["applied"] is True
    # Clamp should have scaled the delta by <1 because the applied op
    # exceeded the aggressive ceiling.
    assert 0.0 < sky["low_tex_guard"]["delta_scale_applied"] < 1.0


# ---------------------------------------------------------------------------
# Stage 0: Config / preset wiring
# ---------------------------------------------------------------------------


def test_enhance_config_accepts_keep_intermediates():
    from transformation_portal.lux_depth_v3.config import EnhanceConfig

    cfg = EnhanceConfig(keep_intermediates=True)
    assert cfg.keep_intermediates is True

    default_cfg = EnhanceConfig()
    # Default must stay False so existing users see no behavior change.
    assert default_cfg.keep_intermediates is False


def test_v2_preset_carries_banding_knobs():
    cfg = V2EnhancementConfig.from_preset("default")
    # Guard defaults ON — the banding failure mode is strictly worse than
    # any cosmetic effect from the guard.
    assert cfg.tone_depth_smoothing is True
    assert 0.0 <= cfg.tone_low_tex_strength <= 1.0

    # Round-trip through to_dict keeps the knobs visible.
    payload = cfg.to_dict()
    assert "tone_depth_smoothing" in payload
    assert "tone_low_tex_strength" in payload
