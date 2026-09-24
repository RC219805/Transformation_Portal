"""Depth exports preserve numeric authority and disclose reconstruction limits."""

from __future__ import annotations

import copy
import hashlib
import io
import json
from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest
import tifffile
from PIL import Image

from transformation_portal.core.depth_evidence import build_depth_evidence
from transformation_portal.core.image_artifact import ImageMaster
from transformation_portal.ingest.canonical_json import canonicalize_json
from transformation_portal.lux_depth_v4.photography import create_proxy
from transformation_portal.lux_depth_v5.photography import align_depth
from transformation_portal.lux_depth_v6.depth_maps import DepthMapRecipe, depth_map_products, reconstruct_depth

pytestmark = pytest.mark.unit


def scene(*, alpha=None, sky_known=True, calibrated=False, shape=(56, 112), invalid=False):
    pixels = np.full((*shape, 3), 0.1, np.float32)
    pixels[:, shape[1] // 2 :] = 0.8
    master = ImageMaster(pixels, "a" * 64, 16, alpha=alpha)
    proxy = create_proxy(master, 28)
    native = np.ones(proxy.transform.padded_shape, np.float32)
    native[:, proxy.transform.resized_shape[1] // 2 :] = 2
    if invalid:
        native[0, :4] = [np.nan, np.inf, -1, 0]
    sky = np.zeros(native.shape, bool) if sky_known else None
    companion = None
    if calibrated:
        companion = {
            "path": "source.tif",
            "source_sha256": master.source_sha256,
            "calibration": {
                "width": shape[1],
                "height": shape[0],
                "fx": 1200,
                "fy": 1200,
                "cx": (shape[1] - 1) / 2,
                "cy": (shape[0] - 1) / 2,
                "source": "synthetic camera fixture, not scene accuracy evidence",
                "coordinate_space": "canonical_master",
            },
        }
    evidence = build_depth_evidence(native, sky, proxy, master.source_sha256, companion=companion)
    return master, evidence, proxy


def products(master, evidence, proxy, recipe=None):
    aligned, receipt = reconstruct_depth(master, evidence, proxy, recipe or DepthMapRecipe("bilinear"))
    outputs = {
        name.removeprefix("input-0000/"): data for name, data in depth_map_products(evidence, aligned, receipt, "input-0000")
    }
    return aligned, receipt, outputs, json.loads(outputs["depth.json"])


def array(data):
    return np.load(io.BytesIO(data), allow_pickle=False)


def test_recipe_is_frozen_closed_and_defaults_to_explicit_v4():
    recipe = DepthMapRecipe()
    assert recipe.refinement == "guided_bilinear_v4"
    assert recipe.to_payload() == {"schema": "tp.lux.depth_maps.v1", "refinement": "guided_bilinear_v4"}
    assert DepthMapRecipe.from_payload(recipe.to_payload()) == recipe
    with pytest.raises(FrozenInstanceError):
        recipe.refinement = "bilinear"


@pytest.mark.parametrize("refinement", [True, None, [], "guided_bilinear", "nearest", ""])
def test_unsupported_refinement_fails_closed(refinement):
    with pytest.raises(ValueError, match="refinement"):
        DepthMapRecipe(refinement)


@pytest.mark.parametrize("change", ["schema", "extra", "missing"])
def test_recipe_rejects_unknown_or_incomplete_payload(change):
    payload = DepthMapRecipe().to_payload()
    if change == "schema":
        payload["schema"] = "tp.lux.depth_maps.v0"
    elif change == "extra":
        payload["sharpness"] = 1
    else:
        del payload["refinement"]
    with pytest.raises(ValueError, match="closed"):
        DepthMapRecipe.from_payload(payload)


@pytest.mark.parametrize("refinement", ["bilinear", "guided_bilinear_v3", "guided_bilinear_v4"])
def test_reconstruction_uses_selected_recipe_without_mutating_native_evidence(refinement):
    master, evidence, proxy = scene()
    digest, native = evidence.content_hash(), evidence.native_depth.tobytes()
    aligned, receipt = reconstruct_depth(master, evidence, proxy, DepthMapRecipe(refinement))
    expected = align_depth(evidence, master, proxy, refinement=refinement)
    assert aligned.content_hash() == expected.content_hash()
    assert receipt["aligned_content_sha256"] == aligned.content_hash()
    assert receipt["depth_evidence_content_sha256"] == evidence.content_hash() == digest
    assert evidence.native_depth.tobytes() == native
    assert not evidence.native_depth.flags.writeable
    assert not aligned.relative_depth.flags.writeable
    assert receipt["quality_acceptance"] == "unestablished"


def test_products_are_deterministic_and_bind_scalar_geometry_and_artifact_bytes():
    master, evidence, proxy = scene()
    aligned, receipt, outputs, descriptor = products(master, evidence, proxy)
    repeated = dict(depth_map_products(evidence, aligned, receipt, "input-0000"))
    assert all(repeated[f"input-0000/{name}"] == data for name, data in outputs.items())
    assert len(outputs) == 12
    assert "metric-depth-m.npy" not in outputs
    assert canonicalize_json(descriptor) == outputs["depth.json"]
    assert descriptor["dimensions"] == {
        "native_padded": list(evidence.shape),
        "native_unpadded": list(proxy.transform.resized_shape),
        "reconstructed_master": list(master.shape),
    }
    assert descriptor["source_sha256"] == master.source_sha256
    assert descriptor["depth_evidence_content_sha256"] == evidence.content_hash()
    assert descriptor["aligned_content_sha256"] == aligned.content_hash()
    assert descriptor["metric"]["status"] == "unavailable"
    assert descriptor["metric"]["calibration"] is None
    assert descriptor["resolution_authority"] == "reconstructed_grid_not_new_model_inference_or_recovered_native_detail"
    assert descriptor["production_acceptance"] == "not_established"
    for name, record in descriptor["artifacts"].items():
        assert record["path"] == f"input-0000/{name}"
        assert record["size_bytes"] == len(outputs[name])
        assert record["sha256"] == hashlib.sha256(outputs[name]).hexdigest()
    np.testing.assert_array_equal(array(outputs["relative-depth.npy"]), aligned.relative_depth)
    np.testing.assert_array_equal(array(outputs["depth-valid.npy"]), aligned.valid_mask)
    np.testing.assert_array_equal(array(outputs["depth-support-score.npy"]), aligned.support_confidence)


def test_native_nonfinite_and_invalid_samples_are_retained_only_in_native_authority():
    master, evidence, proxy = scene(invalid=True)
    before = evidence.native_depth.tobytes()
    aligned, _, outputs, _ = products(master, evidence, proxy)
    assert array(outputs["native-depth.npy"]).tobytes() == before
    np.testing.assert_array_equal(array(outputs["native-numeric-valid.npy"]), evidence.numeric_valid)
    np.testing.assert_array_equal(array(outputs["native-support.npy"]), evidence.support_mask)
    assert np.isfinite(aligned.relative_depth).all()
    assert not aligned.relative_depth[~aligned.valid_mask].any()
    assert evidence.native_depth.tobytes() == before


def test_subnormal_native_range_reconstructs_and_exports_finite_depth():
    master, _, proxy = scene(shape=(10, 10))
    smallest = np.nextafter(np.float32(0), np.float32(1))
    native = np.full(proxy.transform.padded_shape, smallest, np.float32)
    native[9, 9] = np.nextafter(smallest, np.float32(np.inf))
    evidence = build_depth_evidence(native, np.zeros(native.shape, bool), proxy, master.source_sha256)
    with np.errstate(divide="raise", invalid="raise"):
        aligned, _, outputs, _ = products(master, evidence, proxy, DepthMapRecipe())
    assert aligned.valid_mask.all()
    assert np.isfinite(aligned.relative_depth).all()
    assert aligned.relative_depth[9, 9] == 1
    assert np.count_nonzero(aligned.relative_depth) == 1
    assert array(outputs["native-depth.npy"]).tobytes() == native.tobytes()
    np.testing.assert_array_equal(array(outputs["relative-depth.npy"]), aligned.relative_depth)


def test_float_tiff_preserves_relative_bits_and_png_is_true_16_bit_grayscale():
    _, _, outputs, descriptor = products(*scene())
    expected = array(outputs["relative-depth.npy"])
    with tifffile.TiffFile(io.BytesIO(outputs["depth-relative.tif"])) as image:
        assert image.pages[0].dtype == np.dtype("float32")
        assert image.pages[0].sampleformat == 3
        assert image.pages[0].samplesperpixel == 1
        assert image.asarray().tobytes() == expected.tobytes()
    encoded = outputs["depth-preview.png"]
    assert encoded[:8] == b"\x89PNG\r\n\x1a\n"
    assert encoded[24] == 16 and encoded[25] == 0
    with Image.open(io.BytesIO(encoded)) as image:
        observed = np.asarray(image)
        np.testing.assert_array_equal(observed, np.rint(expected.astype(np.float64) * 65535).astype(np.uint16))
    assert descriptor["preview"]["units"] == "normalized_relative_not_meters"
    assert descriptor["preview"]["accuracy_claim"] is False


@pytest.mark.parametrize("calibrated", [False, True])
def test_unknown_sky_exports_explicit_placeholder_and_no_usable_surface(calibrated):
    master, evidence, proxy = scene(sky_known=False, calibrated=calibrated)
    aligned, _, outputs, descriptor = products(master, evidence, proxy)
    assert not aligned.valid_mask.any()
    assert not aligned.relative_depth.any()
    assert not array(outputs["native-sky.npy"]).any()
    assert descriptor["artifacts"]["native-sky.npy"]["semantics"] == "unknown_placeholder_not_non_sky_evidence"
    assert descriptor["sky_status"] == "unavailable"
    assert descriptor["surface_status"] == "unavailable"
    assert descriptor["metric"]["usable_pixels"] == 0
    if calibrated:
        assert not array(outputs["metric-depth-m.npy"]).any()


@pytest.mark.parametrize("alpha_value", [0, 0.5, np.nextafter(np.float32(1), np.float32(0))])
def test_any_nonopaque_alpha_abstains_globally_and_preserves_calibrated_native_evidence(alpha_value):
    alpha = np.ones((56, 112), np.float32)
    alpha[0, 0] = alpha_value
    master, evidence, proxy = scene(alpha=alpha, calibrated=True)
    before = evidence.content_hash()
    aligned, receipt, outputs, descriptor = products(master, evidence, proxy)
    assert receipt["alpha_abstention"]
    assert receipt["reason"] == "alpha_unaware_upstream_proxy"
    assert not aligned.valid_mask.any()
    assert not aligned.relative_depth.any()
    assert not aligned.support_confidence.any()
    assert not aligned.metric_map_m.any()
    assert aligned.support_mask.all()
    assert evidence.content_hash() == before
    assert array(outputs["native-depth.npy"]).tobytes() == evidence.native_depth.tobytes()
    assert descriptor["metric"]["usable_pixels"] == 0


def test_opaque_alpha_does_not_abstain_and_calibration_is_explicit_not_measured_accuracy():
    master, evidence, proxy = scene(alpha=np.ones((56, 112), np.float32), calibrated=True)
    aligned, receipt, outputs, descriptor = products(master, evidence, proxy)
    assert not receipt["alpha_abstention"]
    assert aligned.valid_mask.any()
    assert len(outputs) == 13
    assert aligned.metric_map_m is not None
    np.testing.assert_allclose(aligned.metric_map_m, 1 + aligned.relative_depth)
    np.testing.assert_array_equal(array(outputs["metric-depth-m.npy"]), aligned.metric_map_m)
    assert descriptor["metric"]["status"] == "inferred_with_supplied_camera_calibration"
    assert descriptor["metric"]["calibration"]["measurement_status"] == "model_inference_not_measured_scene_distance"
    assert descriptor["metric"]["measured_scene_accuracy"] == "not_established"


@pytest.mark.parametrize("alpha", [False, True])
@pytest.mark.parametrize("changed", ["source", "pixels", "metadata", "proxy", "evidence"])
def test_source_and_geometry_bindings_fail_before_alpha_abstention(alpha, changed):
    master, evidence, proxy = scene(alpha=np.full((56, 112), 0.5, np.float32) if alpha else None)
    if changed == "source":
        master = replace(master, source_sha256="b" * 64)
    elif changed == "pixels":
        master = replace(master, pixels=master.pixels * 0.5)
    elif changed == "metadata":
        master = replace(master, metadata={"decoder": "different"})
    elif changed == "proxy":
        proxy = replace(proxy, pixels=255 - proxy.pixels)
    else:
        evidence = replace(evidence, metadata={})
    with pytest.raises(ValueError, match="same source"):
        reconstruct_depth(master, evidence, proxy, DepthMapRecipe("bilinear"))


@pytest.mark.parametrize(
    "changed", ["source_sha256", "original_content_sha256", "aligned_content_sha256", "proxy_content_sha256"]
)
def test_products_reject_forged_reconstruction_receipt(changed):
    master, evidence, proxy = scene()
    aligned, receipt = reconstruct_depth(master, evidence, proxy, DepthMapRecipe("bilinear"))
    receipt[changed] = "b" * 64
    with pytest.raises(ValueError, match="bound source"):
        list(depth_map_products(evidence, aligned, receipt, "input-0000"))


def test_products_reject_different_aligned_geometry():
    master, evidence, proxy = scene()
    aligned, receipt = reconstruct_depth(master, evidence, proxy, DepthMapRecipe("bilinear"))
    changed = replace(
        aligned,
        relative_depth=aligned.relative_depth[:2].copy(),
        valid_mask=aligned.valid_mask[:2].copy(),
        support_mask=aligned.support_mask[:2].copy(),
        support_confidence=aligned.support_confidence[:2].copy(),
    )
    receipt["aligned_content_sha256"] = changed.content_hash()
    with pytest.raises(ValueError, match="bound source"):
        list(depth_map_products(evidence, changed, receipt, "input-0000"))


@pytest.mark.parametrize("identifier", ["", "../outside", "/absolute", "input\\one", "Input", "a" * 129])
def test_product_identifier_cannot_escape_its_namespace(identifier):
    master, evidence, proxy = scene()
    aligned, receipt = reconstruct_depth(master, evidence, proxy, DepthMapRecipe("bilinear"))
    with pytest.raises(ValueError, match="portable"):
        list(depth_map_products(evidence, aligned, receipt, identifier))


def test_preview_bounded_sampling_does_not_mix_invalid_depth_into_valid_pixels():
    master, evidence, proxy = scene(shape=(80, 4000))
    aligned, receipt = reconstruct_depth(master, evidence, proxy, DepthMapRecipe("bilinear"))
    # This test isolates encoding: each preview sample must select its own
    # master-grid validity, never blend an invalid zero into a valid sample.
    relative = np.full(master.shape, 0.75, np.float32)
    valid = np.ones(master.shape, bool)
    valid[:, ::3] = False
    relative[~valid] = 0
    aligned = replace(aligned, relative_depth=relative, valid_mask=valid, support_confidence=valid.astype(np.float32))
    receipt = copy.deepcopy(receipt)
    receipt["aligned_content_sha256"] = aligned.content_hash()
    outputs = dict(depth_map_products(evidence, aligned, receipt, "input-0000"))
    descriptor = json.loads(outputs["input-0000/depth.json"])
    assert descriptor["preview"]["shape"] == [32, 1600]
    assert descriptor["preview"]["sampling"] == "nearest_pixel_center_integer_floor_v1"
    with Image.open(io.BytesIO(outputs["input-0000/depth-preview.png"])) as image:
        observed = np.asarray(image)
    with Image.open(io.BytesIO(outputs["input-0000/depth-preview-valid.png"])) as image:
        mask = np.asarray(image) == 255
    assert mask.any() and not mask.all()
    assert not observed[~mask].any()
    assert np.all(observed[mask] == round(0.75 * 65535))


def test_encoded_preview_budget_fails_closed(monkeypatch):
    from transformation_portal.lux_depth_v6 import depth_maps

    monkeypatch.setattr(depth_maps, "MAX_PREVIEW_BYTES", 1)
    with pytest.raises(ValueError, match="encoded-byte budget"):
        products(*scene())
