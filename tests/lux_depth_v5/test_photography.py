"""Conservative V5 reconstruction and protected photographic response contracts."""

from dataclasses import replace

import numpy as np
import pytest

from transformation_portal.core.depth_evidence import build_depth_evidence
from transformation_portal.core.image_artifact import ImageMaster
from transformation_portal.lux_depth_v4.photography import create_proxy
from transformation_portal.lux_depth_v5.photography import align_depth, enhance_master_v5, generate_preview_maps_v5

pytestmark = pytest.mark.unit


def scene(*, sky=False, texture=False, unknown=False):
    pixels = np.full((56, 112, 3), 0.1, np.float32)
    pixels[:, 56:] = 0.8
    if texture:
        pixels = np.broadcast_to((np.indices((56, 112)).sum(axis=0) % 2)[..., None], pixels.shape).astype(np.float32)
    master = ImageMaster(pixels, "a" * 64, 16)
    proxy = create_proxy(master, 28)
    native = np.ones(proxy.transform.padded_shape, np.float32)
    native[:, 14:] = 2
    mask = np.zeros_like(native, bool)
    if sky:
        mask[:4] = True
    evidence = build_depth_evidence(native, None if unknown else mask, proxy, master.source_sha256)
    return master, proxy, evidence


def test_supported_step_boundary_improves_over_bilinear_without_changing_native():
    master, proxy, evidence = scene()
    native_bytes = evidence.native_depth.tobytes()
    baseline = align_depth(evidence, master, proxy, refinement="bilinear")
    guided = align_depth(evidence, master, proxy)
    reference = np.zeros(master.shape, np.float32)
    reference[:, 56:] = 1
    bilinear_error = float(np.mean(np.abs(baseline.relative_depth - reference)))
    guided_error = float(np.mean(np.abs(guided.relative_depth - reference)))
    baseline_transition = np.count_nonzero((baseline.relative_depth[28] > 0.1) & (baseline.relative_depth[28] < 0.9))
    guided_transition = np.count_nonzero((guided.relative_depth[28] > 0.1) & (guided.relative_depth[28] < 0.9))
    assert guided_error < bilinear_error
    assert guided_transition < baseline_transition
    assert guided.metadata["refined_pixels"] > 0
    # Antialiasing in the proxy can make one side's color match ambiguous;
    # that side must retain the baseline rather than invent exact recovery.
    assert np.argmax(guided.relative_depth[28] >= 0.5) == 56
    assert evidence.native_depth.tobytes() == native_bytes
    assert guided.metadata["quality_acceptance"] == "unestablished"


def test_high_frequency_rgb_texture_cannot_create_depth_edges_on_smooth_native_geometry():
    master, proxy, evidence = scene(texture=True)
    native = np.broadcast_to(np.linspace(1, 2, 28, dtype=np.float32), evidence.shape).copy()
    evidence = build_depth_evidence(native, evidence.sky_mask, proxy, master.source_sha256)
    bilinear = align_depth(evidence, master, proxy, refinement="bilinear")
    guided = align_depth(evidence, master, proxy)
    np.testing.assert_array_equal(guided.relative_depth, bilinear.relative_depth)
    assert guided.metadata["candidate_pixels"] == 0
    assert guided.metadata["refined_pixels"] == 0


@pytest.mark.parametrize("kind", ["affine", "steep_smooth", "narrow_valid_band"])
def test_rgb_boundary_cannot_turn_native_slope_into_surface_jump(kind):
    """Large local range alone is not evidence of a native discontinuity."""
    shape, target, split = ((518, 1036), 518, 260) if kind == "narrow_valid_band" else ((56, 112), 14, 28)
    pixels = np.full((*shape, 3), 0.1, np.float32)
    pixels[split:] = 0.8
    master = ImageMaster(pixels, "a" * 64, 16)
    proxy = create_proxy(master, target)
    height, width = proxy.transform.padded_shape
    rows = np.arange(height, dtype=np.float32)
    values = 2 + np.tanh((rows - 3) / 2) if kind == "steep_smooth" else rows + 1
    native = np.broadcast_to(values[:, None], (height, width)).copy()
    sky = np.zeros_like(native, bool)
    if kind == "narrow_valid_band":
        sky[:] = True
        sky[126:133] = False
    evidence = build_depth_evidence(native, sky, proxy, master.source_sha256)
    # These slopes exceed the original central-footprint threshold.
    relative = evidence.relative_depth()
    supported_differences = evidence.valid_mask[1:] & evidence.valid_mask[:-1]
    assert np.max(np.abs(np.diff(relative, axis=0))[supported_differences]) > 0.1
    baseline = align_depth(evidence, master, proxy, refinement="bilinear")
    guided = align_depth(evidence, master, proxy)
    np.testing.assert_array_equal(guided.relative_depth, baseline.relative_depth)
    assert guided.metadata["candidate_pixels"] == guided.metadata["refined_pixels"] == 0


@pytest.mark.parametrize("missing", ["sky", "numeric", "image_border"])
def test_guidance_abstains_when_outer_native_discontinuity_support_is_missing(missing):
    master, proxy, evidence = scene()
    native, sky = evidence.native_depth.copy(), evidence.sky_mask.copy()
    if missing == "sky":
        sky[:, 12] = True  # Central pair 13/14 remains usable.
    elif missing == "numeric":
        native[:, 12] = 0
    else:
        pixels = np.full_like(master.pixels, 0.1)
        pixels[:, 4:] = 0.8
        master = replace(master, pixels=pixels)
        proxy = create_proxy(master, 28)
        native[:] = 2
        native[:, 0] = 1  # The discontinuity has no left outer neighbor.
    evidence = build_depth_evidence(native, sky, proxy, master.source_sha256)
    baseline = align_depth(evidence, master, proxy, refinement="bilinear")
    guided = align_depth(evidence, master, proxy)
    np.testing.assert_array_equal(guided.relative_depth, baseline.relative_depth)
    assert guided.metadata["candidate_pixels"] == 0


def test_no_color_boundary_cannot_sharpen_noisy_depth():
    master, _, _ = scene()
    master = replace(master, pixels=np.full_like(master.pixels, 0.4))
    proxy = create_proxy(master, 28)
    native = np.random.default_rng(771).uniform(1, 2, proxy.transform.padded_shape).astype(np.float32)
    evidence = build_depth_evidence(native, np.zeros_like(native, bool), proxy, master.source_sha256)
    bilinear = align_depth(evidence, master, proxy, refinement="bilinear")
    guided = align_depth(evidence, master, proxy)
    np.testing.assert_array_equal(guided.relative_depth, bilinear.relative_depth)
    assert guided.metadata["refined_pixels"] == 0


def test_sky_and_invalid_holes_survive_guided_alignment_and_all_finishing():
    master, proxy, evidence = scene(sky=True)
    native = evidence.native_depth.copy()
    native[7:9, 20:22] = 0
    evidence = build_depth_evidence(native, evidence.sky_mask, proxy, master.source_sha256)
    aligned = align_depth(evidence, master, proxy)
    assert not aligned.valid_mask[:16].any()
    assert not aligned.valid_mask[28:36, 80:88].any()
    assert not aligned.relative_depth[~aligned.valid_mask].any()
    assert not aligned.support_confidence[~aligned.valid_mask].any()
    finished, report = enhance_master_v5(master, aligned, strength=1, clarity=1)
    np.testing.assert_array_equal(finished.pixels[~aligned.valid_mask], master.pixels[~aligned.valid_mask])
    assert report["protected_pixels"] == np.count_nonzero(~aligned.valid_mask)


def test_unavailable_sky_abstains_from_depth_and_clarity():
    master, proxy, evidence = scene(unknown=True)
    aligned = align_depth(evidence, master, proxy)
    result, report = enhance_master_v5(master, aligned, strength=1, clarity=1)
    np.testing.assert_array_equal(result.pixels, master.pixels)
    assert not aligned.valid_mask.any()
    assert not report["depth_applied"] and not report["clarity_applied"]
    assert report["changed_pixels"] == 0


def test_explicit_protected_regions_and_nonopaque_pixels_are_bitwise_unchanged():
    master, _, _ = scene()
    alpha = np.ones(master.shape, np.float32)
    alpha[20:25] = 0.5
    master = replace(master, alpha=alpha)
    proxy = create_proxy(master, 28)
    native = np.tile(np.linspace(1, 2, 28, dtype=np.float32), (14, 1))
    evidence = build_depth_evidence(native, np.zeros_like(native, bool), proxy, master.source_sha256)
    aligned = align_depth(evidence, master, proxy)
    protected = np.zeros(master.shape, bool)
    protected[30:40] = True
    result, report = enhance_master_v5(master, aligned, strength=1, clarity=1, protected_mask=protected)
    fixed = protected | (alpha < 1)
    assert result.pixels[fixed].tobytes() == master.pixels[fixed].tobytes()
    assert report["protected_pixels"] == fixed.sum()
    assert report["changed_pixels"] > 0


def test_response_respects_exposure_and_clarity_bounds():
    master, proxy, evidence = scene()
    aligned = align_depth(evidence, master, proxy)
    depth_only, _ = enhance_master_v5(master, aligned, strength=0.5)
    result, report = enhance_master_v5(master, aligned, strength=0.5, clarity=0.7)
    gain = depth_only.pixels / master.pixels
    assert gain.min() >= 2**-0.5 - 1e-6 and gain.max() <= 2**0.5 + 1e-6
    assert np.max(np.abs(result.pixels - depth_only.pixels)) <= 0.05 * 0.7 + 1e-7
    assert report["max_exposure_stops"] == 0.5
    assert report["max_clarity_linear_delta"] == pytest.approx(0.035)


def test_sparse_interpolation_support_attenuates_clarity():
    master, proxy, evidence = scene(texture=True)
    aligned = align_depth(evidence, master, proxy)
    supported, _ = enhance_master_v5(master, aligned, strength=0, clarity=1)
    sparse = replace(aligned, support_confidence=aligned.support_confidence * np.float32(0.25))
    attenuated, _ = enhance_master_v5(master, sparse, strength=0, clarity=1)
    full_delta = supported.pixels - master.pixels
    assert np.max(np.abs(full_delta)) > 0
    np.testing.assert_allclose(attenuated.pixels - master.pixels, full_delta * 0.25, atol=1e-7)


def test_metric_alignment_selects_the_same_native_surface_as_relative_alignment():
    master, proxy, evidence = scene()
    companion = {
        "path": "image.tif",
        "source_sha256": master.source_sha256,
        "calibration": {
            "width": 112,
            "height": 56,
            "fx": 1200,
            "fy": 1200,
            "cx": 55.5,
            "cy": 27.5,
            "source": "camera chart fixture",
            "coordinate_space": "canonical_master",
        },
    }
    evidence = build_depth_evidence(evidence.native_depth, evidence.sky_mask, proxy, master.source_sha256, companion=companion)
    aligned = align_depth(evidence, master, proxy)
    np.testing.assert_allclose(aligned.metric_map_m, 1 + aligned.relative_depth)


def test_alignment_is_independent_of_scratch_tile_size(monkeypatch):
    from transformation_portal.lux_depth_v5 import photography

    master, proxy, evidence = scene()
    normal = align_depth(evidence, master, proxy)
    monkeypatch.setattr(photography, "_MAX_CHUNK_PIXELS", 37)
    tiled = align_depth(evidence, master, proxy)
    np.testing.assert_array_equal(normal.relative_depth, tiled.relative_depth)
    np.testing.assert_array_equal(normal.valid_mask, tiled.valid_mask)


def test_alignment_rejects_other_master_source_or_geometry():
    master, proxy, evidence = scene()
    with pytest.raises(ValueError, match="same source"):
        align_depth(evidence, replace(master, source_sha256="b" * 64), proxy)
    with pytest.raises(ValueError, match="same source"):
        align_depth(evidence, replace(master, pixels=master.pixels * 2), proxy)
    with pytest.raises(ValueError, match="same source"):
        align_depth(evidence, master, replace(proxy, pixels=255 - proxy.pixels))
    with pytest.raises(ValueError, match="recipe"):
        align_depth(evidence, master, proxy, refinement="magic")
    aligned = align_depth(evidence, master, proxy)
    with pytest.raises(ValueError, match="authorize"):
        enhance_master_v5(replace(master, pixels=master.pixels * 2), aligned)


def test_aligned_arrays_are_immutable_and_support_is_never_labeled_probability():
    master, proxy, evidence = scene()
    aligned = align_depth(evidence, master, proxy)
    for array in (aligned.relative_depth, aligned.valid_mask, aligned.support_mask, aligned.support_confidence):
        with pytest.raises(ValueError):
            array.setflags(write=True)
    assert "not_accuracy_probability" in aligned.to_payload()["support_score_semantics"]
    with pytest.raises(ValueError, match="bounded"):
        replace(aligned, support_confidence=np.full(aligned.shape, 1.1, np.float32))


def test_preview_hole_neighborhood_is_unknown_without_contaminating_other_pixels():
    from scipy.ndimage import minimum_filter

    pixels = np.full((56, 84, 3), 0.2, np.float32)
    master = ImageMaster(pixels, "a" * 64, 16)
    proxy = create_proxy(master, 84)
    native = np.ones((56, 84), np.float32)
    native[:, 42:] = 2
    clean = build_depth_evidence(native, np.zeros_like(native, bool), proxy, master.source_sha256)
    invalid = native.copy()
    invalid[25:31, 60:66] = 0
    hole = build_depth_evidence(invalid, np.zeros_like(native, bool), proxy, master.source_sha256)
    clean_maps, _ = generate_preview_maps_v5(clean)
    maps, report = generate_preview_maps_v5(hole)
    for name, radius in report["support_radius_pixels"].items():
        supported = minimum_filter(hole.valid_mask, size=2 * radius + 1, mode="constant", cval=0)
        np.testing.assert_array_equal(maps[name][supported], clean_maps[name][supported])
        np.testing.assert_array_equal(
            maps[name][~supported], np.broadcast_to(report["invalid_values"][name], maps[name][~supported].shape)
        )
    assert report["physical_material_estimate"] is False
    assert report["depth_content_hash"] == hole.content_hash()


@pytest.mark.parametrize("kwargs", [{"strength": -1}, {"clarity": np.nan}, {"strength": True}])
def test_invalid_response_settings_fail_closed(kwargs):
    master, proxy, evidence = scene()
    with pytest.raises(ValueError, match="finite"):
        enhance_master_v5(master, align_depth(evidence, master, proxy), **kwargs)
