"""Model-mask lifting agrees with independent native pixel-center coordinates."""

from __future__ import annotations

import hashlib
import math
from dataclasses import replace

import numpy as np
import pytest

from transformation_portal.core.image_artifact import ImageMaster, metadata_payload
from transformation_portal.materials_v4 import engine, inference
from transformation_portal.materials_v4.contracts import MaterialEvidence, MaterialsError, RegionEvidence
from transformation_portal.materials_v4.engine import ResponsePolicy, apply_response, plan_response
from transformation_portal.materials_v4.inference import InferenceConfig, infer_materials, prepare_inference
from transformation_portal.materials_v4.proposal import Proposal
from transformation_portal.materials_v4.semantics import SemanticObservation

pytestmark = pytest.mark.unit


@pytest.fixture
def prepared_geometry(tmp_path, monkeypatch):
    save_file = pytest.importorskip("safetensors.numpy").save_file
    sam = tmp_path / "sam.pt"
    sam.write_bytes(b"fixture-sam")
    clip = tmp_path / "clip.safetensors"
    save_file({"tensor": np.zeros((2,), np.float32)}, str(clip))
    monkeypatch.setattr(inference, "SAM2_LARGE_SHA256", hashlib.sha256(sam.read_bytes()).hexdigest())
    monkeypatch.setattr(inference, "_runtime_payload", lambda device: {"actual_device": device, "version": "geometry-fixture"})
    master = ImageMaster(np.full((73, 119, 3), 0.4, np.float32), "a" * 64, 16)
    return master, prepare_inference(master, InferenceConfig(sam, clip, proxy_longest_side=42))


def test_asymmetric_proposals_lift_without_padding_or_interpolated_labels(prepared_geometry, monkeypatch):
    master, prepared = prepared_geometry
    masks = []

    def proposals(image, *_args, **_kwargs):
        assert image.shape == (28, 42, 3)
        region = np.zeros(image.shape[:2], bool)
        region[3:9, 5:12] = True
        region[17:24, 29:40] = True
        # A deliberately different padding value must never stretch onto the
        # bottom native boundary or invent a second connected photographic area.
        region[26:] = True
        padding = np.zeros(image.shape[:2], bool)
        padding[26:] = True
        masks.extend((region, padding))
        return tuple(Proposal(mask, 0.99, 0.98) for mask in masks)

    monkeypatch.setattr(inference, "generate_proposals", proposals)
    monkeypatch.setattr(
        inference,
        "classify_proposals",
        lambda *_args, **_kwargs: tuple(SemanticObservation("water", 0.99, 0.8, 0.9, None) for _ in range(2)),
    )
    evidence = infer_materials(master, prepared)
    expected = np.zeros(master.shape, bool)
    for row in range(73):
        source_row = math.floor((row + 0.5) * 26 / 73)
        for column in range(119):
            source_column = math.floor((column + 0.5) * 42 / 119)
            expected[row, column] = (3 <= source_row < 9 and 5 <= source_column < 12) or (
                17 <= source_row < 24 and 29 <= source_column < 40
            )
    assert len(evidence.regions) == 2
    np.testing.assert_array_equal(evidence.regions[0].mask, expected)
    assert not evidence.regions[1].mask.any()
    assert set(np.unique(evidence.regions[0].mask)) == {0, 1}
    assert evidence.shape == master.shape and evidence.source_sha256 == master.source_sha256
    assert evidence.producer["mask_restore"] == "crop_padding_then_nearest_pixel_centers"
    # Experimental inferred evidence does not acquire native edit authority
    # merely because its geometric projection is correct.
    policy = ResponsePolicy(min_coverage_px=1, min_confidence=0)
    result, receipt = apply_response(master, evidence, plan_response(master, evidence, policy))
    np.testing.assert_array_equal(result.pixels, master.pixels)
    assert receipt["changed_pixels"] == 0


@pytest.mark.parametrize("mutation", ["source", "missing_transform", "wrong_transform"])
def test_prepared_geometry_drift_fails_before_models_or_native_edit(prepared_geometry, monkeypatch, mutation):
    master, prepared = prepared_geometry
    if mutation == "source":
        master = replace(master, source_sha256="f" * 64)
    else:
        transform = metadata_payload(prepared.proxy_transform)
        if mutation == "missing_transform":
            transform = {}
        else:
            transform["original_shape"] = [72, 119]
        prepared = replace(prepared, proxy_transform=transform)

    def forbidden(*_args, **_kwargs):
        pytest.fail("Stale geometry reached model execution or native editing")

    monkeypatch.setattr(inference, "generate_proposals", forbidden)
    monkeypatch.setattr(inference, "classify_proposals", forbidden)
    monkeypatch.setattr(engine, "operation_delta", forbidden)
    with pytest.raises(MaterialsError, match="master differs|proxy changed"):
        infer_materials(master, prepared)


@pytest.mark.parametrize("mutation", ["source", "shape", "mask"])
def test_native_material_plan_rejects_frame_drift_before_operation(mutation, monkeypatch):
    master = ImageMaster(np.full((73, 119, 3), 0.4, np.float32), "a" * 64, 16)
    mask = np.zeros(master.shape, np.float32)
    mask[7:21, 31:67] = 1
    evidence = MaterialEvidence(master.source_sha256, master.shape, (RegionEvidence("water", "water", mask, 0.99),))
    plan = plan_response(master, evidence, ResponsePolicy(min_coverage_px=1))
    if mutation == "source":
        master = replace(master, source_sha256="b" * 64)
    elif mutation == "shape":
        master = replace(master, pixels=np.full((119, 73, 3), 0.4, np.float32))
    else:
        changed = mask.copy()
        changed[7, 31] = 0
        evidence = replace(evidence, regions=(replace(evidence.regions[0], mask=changed),))

    def forbidden(*_args, **_kwargs):
        pytest.fail("Invalid source/frame/mask reached the native operation")

    monkeypatch.setattr(engine, "operation_delta", forbidden)
    with pytest.raises(MaterialsError, match="source or geometry|Stale or invalid"):
        apply_response(master, evidence, plan)


def test_supplied_native_mask_changes_only_its_asymmetric_photographic_coordinates():
    row, column = np.indices((73, 119))
    pixels = np.stack((0.2 + row / 1000, 0.3 + column / 1000, np.full(row.shape, 0.4)), axis=-1).astype(np.float32)
    alpha = np.ones((73, 119), np.float32)
    alpha[10:13, 37:41] = 0
    master = ImageMaster(pixels, "a" * 64, 16, alpha=alpha)
    mask = np.zeros(master.shape, np.float32)
    mask[7:21, 31:67] = 1
    mask[51:66, 9:17] = 0.5
    evidence = MaterialEvidence(master.source_sha256, master.shape, (RegionEvidence("water", "water", mask, 0.99),))
    policy = ResponsePolicy(min_coverage_px=1)
    output, receipt = apply_response(master, evidence, plan_response(master, evidence, policy))
    allowed = (mask > 0) & (alpha > 0)
    expected = pixels.copy()
    expected[allowed] += pixels[allowed] * np.float32(0.01) * mask[allowed, None]
    np.testing.assert_allclose(output.pixels, expected, atol=3e-8)
    np.testing.assert_array_equal(output.pixels[~allowed], pixels[~allowed])
    np.testing.assert_array_equal(output.alpha, alpha)
    assert receipt["changed_pixels"] == np.count_nonzero(allowed)
