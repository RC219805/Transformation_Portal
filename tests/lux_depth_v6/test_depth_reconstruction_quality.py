"""Analytic limits and conservative topology for V6 depth reconstruction."""

from dataclasses import replace

import numpy as np
import pytest

from transformation_portal.core.depth_evidence import build_depth_evidence
from transformation_portal.core.image_artifact import ImageMaster
from transformation_portal.lux_depth_v4.photography import create_proxy
from transformation_portal.lux_depth_v5 import photography

pytestmark = pytest.mark.unit


def _scene(kind):
    rows, columns = np.indices((28, 28))
    regions = {
        "step": columns >= 14,
        "thin": columns == 14,
        "diagonal": columns >= rows,
        "checkerboard": (rows + columns) % 2 == 1,
        "slope": columns >= 14,
        "conflicting_rgb": columns >= 14,
    }[kind]
    native = (1 + regions).astype(np.float32)
    pixels = np.repeat(np.repeat(np.where(regions, 0.6, 0.15), 4, axis=0), 4, axis=1)
    if kind == "slope":
        native = np.broadcast_to(np.linspace(1, 2, 28, dtype=np.float32), (28, 28)).copy()
    elif kind == "conflicting_rgb":
        pixels[:] = 0.4
    master = ImageMaster(np.repeat(pixels[..., None], 3, axis=2).astype(np.float32), "a" * 64, 16)
    proxy = create_proxy(master, 28)
    evidence = build_depth_evidence(native, np.zeros_like(native, bool), proxy, master.source_sha256)
    return master, proxy, evidence, np.repeat(np.repeat(regions, 4, axis=0), 4, axis=1).astype(np.float32)


def _align(scene, refinement):
    master, proxy, evidence, _ = scene
    return photography.align_depth(evidence, master, proxy, refinement=refinement)


def test_disconnected_native_outliers_cannot_be_amplified_by_matching_rgb_texture():
    scene = _scene("checkerboard")
    baseline = _align(scene, "bilinear")
    legacy = _align(scene, "guided_bilinear_v3")
    connected = _align(scene, "guided_bilinear_v4")
    # This is a declared adversarial flat-surface fixture, not photographic truth.
    reference = np.full(baseline.shape, 0.5, np.float32)
    legacy_rmse = np.sqrt(np.mean((legacy.relative_depth - reference) ** 2))
    baseline_rmse = np.sqrt(np.mean((baseline.relative_depth - reference) ** 2))
    assert legacy_rmse > 2 * baseline_rmse
    assert legacy.metadata["refined_pixels"] == 5000
    np.testing.assert_array_equal(connected.relative_depth, baseline.relative_depth)
    assert connected.metadata["candidate_pixels"] == connected.metadata["refined_pixels"] == 0
    assert connected.metadata["native_connectivity"] == "four_connected_per_central_sample"
    assert connected.metadata["minimum_native_samples_per_component"] == 4
    assert connected.metadata["minimum_outer_samples_per_component"] == 2


@pytest.mark.parametrize("kind", ["step", "thin", "diagonal"])
def test_connected_surfaces_retain_measurable_boundary_refinement(kind):
    scene = _scene(kind)
    baseline = _align(scene, "bilinear")
    connected = _align(scene, "guided_bilinear_v4")
    reference = scene[3]
    assert np.mean(np.abs(connected.relative_depth - reference)) < np.mean(np.abs(baseline.relative_depth - reference))
    assert connected.metadata["refined_pixels"] > 0
    np.testing.assert_array_equal(connected.valid_mask, baseline.valid_mask)
    np.testing.assert_array_equal(connected.support_confidence, baseline.support_confidence)


@pytest.mark.parametrize("kind", ["slope", "conflicting_rgb"])
def test_geometry_and_color_ambiguity_retain_bilinear_fallback(kind):
    scene = _scene(kind)
    baseline = _align(scene, "bilinear")
    connected = _align(scene, "guided_bilinear_v4")
    np.testing.assert_array_equal(connected.relative_depth, baseline.relative_depth)
    assert connected.metadata["refined_pixels"] == 0


def test_disconnected_central_sample_is_not_authorized_by_another_large_component():
    master, proxy, evidence, _ = _scene("step")
    native = np.ones_like(evidence.native_depth)
    # A singleton at the central lower-right corner shares its depth with a
    # separate four-cell component. Global group counts authorize both in v3.
    native[9, 9:13] = 2
    native[11, 11] = 2
    evidence = build_depth_evidence(native, evidence.sky_mask, proxy, master.source_sha256)
    rows, columns = np.array([10]), np.array([10])
    assert photography._native_discontinuity_support(evidence, rows, columns, require_persistent_groups=True).item()
    assert not photography._connected_native_discontinuity_support(evidence, rows, columns).item()


@pytest.mark.parametrize("height", [1, 2, 3, 28])
def test_clamped_image_boundaries_cannot_count_repeated_native_samples(height):
    master = ImageMaster(np.full((height * 4, 112, 3), 0.4, np.float32), "a" * 64, 16)
    proxy = create_proxy(master, 28)
    native = np.ones(proxy.transform.padded_shape, np.float32)
    native[:, 14:] = 2
    evidence = build_depth_evidence(native, np.zeros_like(native, bool), proxy, master.source_sha256)
    # At the top and bottom, a clamped 4x4 patch repeats native rows. The
    # complete-neighborhood guard must reject it before connected-cell counting.
    rows, columns = np.array([0, max(0, height - 2), height - 1]), np.array([13, 13, 13])
    assert not photography._connected_native_discontinuity_support(evidence, rows, columns).any()
    if height < 4:
        aligned = _align((master, proxy, evidence, None), "guided_bilinear_v4")
        assert aligned.metadata["candidate_pixels"] == aligned.metadata["refined_pixels"] == 0


def test_bitmapped_topology_matches_independent_four_neighbor_component_search():
    master, proxy, evidence, _ = _scene("step")
    rng = np.random.default_rng(719)
    for _ in range(128):
        labels = rng.integers(0, 2, (4, 4))
        expected = bool(np.ptp(labels[1:3, 1:3]))
        for seed in ((1, 1), (1, 2), (2, 1), (2, 2)):
            seen, pending = {seed}, [seed]
            while pending:
                row, column = pending.pop()
                for neighbor in ((row - 1, column), (row + 1, column), (row, column - 1), (row, column + 1)):
                    y, x = neighbor
                    if 0 <= y < 4 and 0 <= x < 4 and neighbor not in seen and labels[y, x] == labels[seed]:
                        seen.add(neighbor)
                        pending.append(neighbor)
            outer_count = sum(row in (0, 3) or column in (0, 3) for row, column in seen)
            expected &= len(seen) >= 4 and outer_count >= 2
        native = np.ones(evidence.shape, np.float32)
        native[9:13, 9:13] = 1 + labels
        sample = build_depth_evidence(native, evidence.sky_mask, proxy, master.source_sha256)
        actual = photography._connected_native_discontinuity_support(sample, np.array([10]), np.array([10]))
        assert actual.item() == expected


@pytest.mark.parametrize("kind", ["sky", "numeric", "unknown_sky"])
def test_topology_never_fills_invalid_holes_or_changes_native_evidence(kind):
    master, proxy, evidence, reference = _scene("step")
    native, sky = evidence.native_depth.copy(), evidence.sky_mask.copy()
    if kind == "sky":
        sky[9:13, 13:15] = True
    elif kind == "numeric":
        native[9:13, 13:15] = 0
    else:
        sky = None
    evidence = build_depth_evidence(native, sky, proxy, master.source_sha256)
    scene = master, proxy, evidence, reference
    before = evidence.content_hash()
    baseline = _align(scene, "bilinear")
    connected = _align(scene, "guided_bilinear_v4")
    np.testing.assert_array_equal(connected.valid_mask, baseline.valid_mask)
    np.testing.assert_array_equal(connected.support_confidence, baseline.support_confidence)
    assert np.all(connected.relative_depth[~connected.valid_mask] == 0)
    assert evidence.content_hash() == before


def test_metric_and_relative_derivatives_choose_identical_surface_samples():
    master, proxy, evidence, reference = _scene("step")
    companion = {
        "path": "test.tif",
        "source_sha256": master.source_sha256,
        "calibration": {
            "width": 112,
            "height": 112,
            "fx": 1200,
            "fy": 1200,
            "cx": 55.5,
            "cy": 55.5,
            "source": "analytic fixture",
            "coordinate_space": "canonical_master",
        },
    }
    evidence = build_depth_evidence(evidence.native_depth, evidence.sky_mask, proxy, master.source_sha256, companion=companion)
    aligned = _align((master, proxy, evidence, reference), "guided_bilinear_v4")
    np.testing.assert_allclose(aligned.metric_map_m, 1 + aligned.relative_depth)


def test_topology_is_cached_on_native_grid_and_independent_of_scratch_tiling(monkeypatch):
    scene = _scene("diagonal")
    expected = _align(scene, "guided_bilinear_v4")
    calls = []
    original = photography._connected_native_discontinuity_support

    def measured(evidence, rows, columns):
        assert len(rows) <= 37
        calls.extend(zip(rows.tolist(), columns.tolist()))
        return original(evidence, rows, columns)

    monkeypatch.setattr(photography, "_MAX_CHUNK_PIXELS", 37)
    monkeypatch.setattr(photography, "_connected_native_discontinuity_support", measured)
    actual = _align(scene, "guided_bilinear_v4")
    np.testing.assert_array_equal(actual.relative_depth, expected.relative_depth)
    assert actual.metadata["refined_pixels"] == expected.metadata["refined_pixels"]
    assert len(calls) == len(set(calls))
    assert len(calls) <= np.prod(scene[1].transform.resized_shape)


def test_noninteger_resize_and_padding_preserve_native_geometry_without_new_support():
    master = ImageMaster(np.full((83, 137, 3), 0.4, np.float32), "a" * 64, 16)
    proxy = create_proxy(master, 28)
    shape = proxy.transform.padded_shape
    native = np.broadcast_to(np.linspace(1, 2, shape[1], dtype=np.float32), shape).copy()
    evidence = build_depth_evidence(native, np.zeros(shape, bool), proxy, master.source_sha256)
    scene = master, proxy, evidence, None
    baseline, actual = _align(scene, "bilinear"), _align(scene, "guided_bilinear_v4")
    np.testing.assert_array_equal(actual.relative_depth, baseline.relative_depth)
    assert actual.shape == master.shape
    assert actual.valid_mask.all()


@pytest.mark.parametrize("seed", [19, 37, 91])
def test_connected_recipe_can_only_reject_legacy_surface_selections(seed):
    rng = np.random.default_rng(seed)
    master, proxy, evidence, reference = _scene("step")
    master = replace(master, pixels=rng.choice([0.15, 0.6], master.pixels.shape).astype(np.float32))
    proxy = create_proxy(master, 28)
    evidence = build_depth_evidence(
        rng.choice([1, 2], evidence.shape).astype(np.float32), np.zeros(evidence.shape, bool), proxy, master.source_sha256
    )
    scene = master, proxy, evidence, reference
    baseline = _align(scene, "bilinear")
    legacy = _align(scene, "guided_bilinear_v3")
    connected = _align(scene, "guided_bilinear_v4")
    assert np.all((connected.relative_depth == baseline.relative_depth) | (connected.relative_depth == legacy.relative_depth))
