"""Native human Materials regions preserve precise, explicitly display-stage output."""

from __future__ import annotations

import copy
import hashlib
import io
import json
from dataclasses import replace

import numpy as np
import pytest
import tifffile
from PIL import Image

from transformation_portal.lux_depth_v4.photography import linear_to_srgb, output_srgb_icc
from transformation_portal.lux_depth_v5.preview import preview_samples
from transformation_portal.lux_depth_v6.color import RenderRecipe
from transformation_portal.photographic_delivery import DeliveryImage, ExportPreset, NativeWorkingMaster, export_delivery
from transformation_portal.photographic_delivery.materials import (
    MATERIALS_RECIPE_SCHEMA,
    MaterialsRecipe,
    create_materials_revision,
    validate_materials_revision_receipt,
)
from transformation_portal.photographic_delivery.presentation import PresentationPolicy, revision_policy

pytestmark = pytest.mark.unit


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _image(*, orientation=1, alpha=False, bits=16, pixels=None):
    if pixels is None:
        values = np.arange(8 * 12 * 3, dtype=np.uint16).reshape(8, 12, 3)
        pixels = 18000 + values * 31
        if bits == 8:
            pixels = (pixels // 257).astype(np.uint8)
    if alpha:
        plane = np.full(pixels.shape[:2], 65535 if bits == 16 else 255, pixels.dtype)
        plane[0, :] = 0
        plane[1, :] //= 2
        pixels = np.concatenate([pixels, plane[..., None]], axis=2)
    buffer = io.BytesIO()
    tifffile.imwrite(
        buffer,
        pixels,
        photometric="rgb",
        extrasamples="unassalpha" if alpha else None,
        iccprofile=output_srgb_icc(),
        extratags=[(274, "H", 1, orientation, False)],
    )
    raw = buffer.getvalue()
    return DeliveryImage("source.tif", _sha(raw), "Room", source_bytes=raw)


def _payload(*, strength=0.03, label="glass", region=None):
    return {
        "schema": MATERIALS_RECIPE_SCHEMA,
        "input_stage": "display_referred",
        "regions": [{"id": "region-1", "label": label, "region": region or [0, 0, 1, 1], "strength": strength}],
    }


def _reopen(result):
    return DeliveryImage(
        "master.tif",
        result["receipt"]["revision_sha256"],
        "Room",
        source_bytes=result["source_bytes"],
        presentation=revision_policy(result["receipt"], result["receipt"]["revision_sha256"]),
    )


@pytest.mark.parametrize("orientation", range(1, 9))
@pytest.mark.parametrize("bits", [8, 16])
def test_native_materials_normalizes_all_orientations_and_preserves_alpha_precision(orientation, bits):
    image = _image(orientation=orientation, bits=bits, alpha=True)
    master = image.decode()
    recipe = MaterialsRecipe.from_payload(_payload())
    result = create_materials_revision(image, recipe)
    retained = _reopen(result).decode()
    receipt = result["receipt"]
    assert retained.shape == master.shape
    assert receipt["source_bit_depth"] == bits and receipt["output_bit_depth"] == 16
    assert receipt["output_master_content_sha256"] == retained.content_hash()
    assert receipt["execution"]["protected_changed_pixels"] == 0
    assert receipt["source_identity_sha256"] == image.source_identity_sha256
    np.testing.assert_allclose(retained.alpha, master.alpha, rtol=0, atol=1 / 65535)
    protected = master.alpha == 0
    before = np.rint(linear_to_srgb(master.pixels) * 65535).astype(np.int32)
    after = np.rint(linear_to_srgb(retained.pixels) * 65535).astype(np.int32)
    assert np.max(np.abs(after[protected] - before[protected]), initial=0) <= 1
    with tifffile.TiffFile(io.BytesIO(result["source_bytes"])) as output:
        assert output.pages[0].tags[274].value == 1
        assert output.pages[0].bitspersample == 16
        assert output.pages[0].extrasamples == (tifffile.EXTRASAMPLE.UNASSALPHA,)
        assert output.pages[0].tags[34675].value == output_srgb_icc()
    assert receipt["approval_transferred"] is False and receipt["review_status"] == "unreviewed"


def test_materials_zero_strength_round_trip_retains_native_detail_without_proxy_edit():
    image = _image()
    result = create_materials_revision(image, MaterialsRecipe.from_payload(_payload(strength=0)))
    receipt = result["receipt"]
    assert receipt["execution"]["status"] == "abstained"
    assert receipt["execution"]["changed_pixels"] == 0
    source = tifffile.imread(io.BytesIO(image.source_bytes)).astype(np.int32)
    retained = tifffile.imread(io.BytesIO(result["source_bytes"])).astype(np.int32)
    assert np.max(np.abs(retained - source)) <= 1
    assert len(np.unique(retained)) > 255
    repeat = create_materials_revision(image, MaterialsRecipe.from_payload(_payload(strength=0)))
    assert result == repeat


def test_materials_stone_detail_uses_full_native_samples_and_recipe_detaches_mutation():
    grid = np.indices((9, 13)).sum(axis=0) % 2
    samples = np.repeat((20000 + grid * 15000).astype(np.uint16)[..., None], 3, axis=2)
    image = _image(pixels=samples)
    payload = _payload(label="stone", region=[0, 0, 0.5, 1])
    recipe = MaterialsRecipe.from_payload(payload)
    payload["regions"][0]["region"][2] = 1
    payload["regions"][0]["strength"] = 0
    result = create_materials_revision(image, recipe)
    actual = tifffile.imread(io.BytesIO(result["source_bytes"])).astype(np.int32)
    assert np.any(actual[:, :6] != samples[:, :6])
    assert np.max(np.abs(actual[:, 6:] - samples[:, 6:])) <= 1
    assert result["receipt"]["recipe"]["regions"][0]["region"] == [0, 0, 0.5, 1]
    assert result["receipt"]["execution"]["regions"][0]["operation_id"] == "luminance_detail_gain_v1"


def test_materials_overlap_is_protected_and_execution_reports_actual_native_deltas():
    payload = _payload(region=[0, 0, 0.75, 1])
    payload["regions"].append({"id": "region-2", "label": "water", "region": [0.25, 0, 1, 1], "strength": 0.02})
    image = _image()
    result = create_materials_revision(image, MaterialsRecipe.from_payload(payload))
    before, after = image.decode(), _reopen(result).decode()
    np.testing.assert_allclose(after.pixels[:, 3:9], before.pixels[:, 3:9], rtol=0, atol=3e-5)
    assert np.any(after.pixels[:, :3] > before.pixels[:, :3])
    assert np.any(after.pixels[:, 9:] > before.pixels[:, 9:])
    execution = result["receipt"]["execution"]
    assert execution["changed_pixels"] == 8 * 6
    assert execution["protected_changed_pixels"] == 0
    assert 0 < execution["mean_abs_delta"] <= execution["max_abs_delta"] <= 0.025
    diagnostics = json.loads(result["diagnostic_files"]["materials-diagnostics.json"][0])
    assert diagnostics["execution"] == execution
    assert diagnostics["diagnostic_only"] is True and diagnostics["approval_authority"] is False
    assert diagnostics["evidence_sha256"] == result["receipt"]["evidence_sha256"]


def test_materials_display_preview_reopen_and_export_never_receive_second_tone(tmp_path, monkeypatch):
    image = _image()
    policy = PresentationPolicy(
        "display_referred",
        RenderRecipe("soft_srgb", 0.7),
        "v6_delivery",
        "b" * 64,
        image.source_sha256,
        image.source_identity_sha256,
    )
    image = replace(image, presentation=policy)

    def fail(*args, **kwargs):
        raise AssertionError("Display Materials pixels received another tone transform")

    monkeypatch.setattr("transformation_portal.photographic_delivery.presentation.render_master", fail)
    result = create_materials_revision(image, MaterialsRecipe.from_payload(_payload()))
    retained = _reopen(result)
    assert retained.presentation.authority_kind == "materials_revision_v1"
    assert retained.presentation.recipe == policy.recipe
    with Image.open(io.BytesIO(result["preview_bytes"])) as preview:
        np.testing.assert_array_equal(np.asarray(preview), preview_samples(retained.decode()))
    inventory = export_delivery(
        [retained], tmp_path / "out", property_name="House", export_presets=[ExportPreset.from_payload("print")]
    )
    assert inventory["files"][0]["render"]["operation"] == "preserve_recorded_display"
    assert result["receipt"]["input_presentation_policy_sha256"] == policy.sha256


def test_materials_overlay_is_bounded_without_reducing_native_geometry():
    pixels = np.full((3, 1701, 3), 22000, np.uint16)
    result = create_materials_revision(_image(pixels=pixels), MaterialsRecipe.from_payload(_payload(strength=0)))
    assert result["receipt"]["width"] == 1701
    with Image.open(io.BytesIO(result["diagnostic_files"]["materials-overlay.png"][0])) as overlay:
        assert overlay.size == (1600, 2)
    for ref in result["receipt"]["diagnostics"]:
        raw, mime = result["diagnostic_files"][ref["filename"]]
        assert ref["sha256"] == _sha(raw) and ref["size_bytes"] == len(raw) and ref["content_type"] == mime
    preview = result["diagnostic_files"]["materials-preview.png"]
    assert preview == (result["preview_bytes"], "image/png")
    descriptor = json.loads(result["diagnostic_files"]["materials-diagnostics.json"][0])
    assert descriptor["preview"]["sha256"] == _sha(preview[0])
    assert descriptor["preview"]["size_bytes"] == len(preview[0])
    assert descriptor["preview"]["meaning"] == "matching_retained_display_photograph"
    assert {key: descriptor["preview"][key] for key in result["receipt"]["preview"]} == result["receipt"]["preview"]


@pytest.mark.parametrize("overlap", [True, False])
def test_materials_overlay_preserves_single_native_pixel_conflicts_before_reduction(overlap):
    pixels = np.full((4, 3200, 3), 22000, np.uint16)
    first = [0, 0, 1 / 3200, 0.25]
    second = first if overlap else [1 / 3200, 0, 2 / 3200, 0.25]
    payload = _payload(strength=0, region=first)
    payload["regions"].append({"id": "region-2", "label": "water", "region": second, "strength": 0})
    result = create_materials_revision(_image(pixels=pixels), MaterialsRecipe.from_payload(payload))
    with Image.open(io.BytesIO(result["diagnostic_files"]["materials-overlay.png"][0])) as overlay:
        samples = np.asarray(overlay)
    assert samples.shape == (2, 1600, 3)
    red = np.all(samples == [255, 70, 70], axis=2)
    assert red[0, 0] == overlap
    assert np.count_nonzero(red) == int(overlap)
    assert result["receipt"]["execution"]["changed_pixels"] == 0


@pytest.mark.parametrize(
    "damage",
    [
        "schema",
        "missing_stage",
        "extra",
        "five_regions",
        "duplicate",
        "id",
        "label",
        "coordinates",
        "empty_rectangle",
        "strength",
        "bool_strength",
        "same_label_strength",
    ],
)
def test_materials_recipe_rejects_unbounded_or_ambiguous_operations(damage):
    payload = _payload()
    if damage == "schema":
        payload["schema"] = "future"
    elif damage == "missing_stage":
        del payload["input_stage"]
    elif damage == "extra":
        payload["evidence_root"] = "/tmp/untrusted"
    elif damage == "five_regions":
        payload["regions"] *= 5
    elif damage == "duplicate":
        payload["regions"] *= 2
    elif damage == "id":
        payload["regions"][0]["id"] = "../mask"
    elif damage == "label":
        payload["regions"][0]["label"] = "unknown"
    elif damage == "coordinates":
        payload["regions"][0]["region"] = [0, 0, float("nan"), 1]
    elif damage == "empty_rectangle":
        payload["regions"][0]["region"] = [1, 0, 1, 1]
    elif damage == "strength":
        payload["regions"][0]["strength"] = 0.051
    elif damage == "bool_strength":
        payload["regions"][0]["strength"] = True
    elif damage == "same_label_strength":
        payload["regions"].append({**payload["regions"][0], "id": "other", "strength": 0.01})
    with pytest.raises(ValueError):
        MaterialsRecipe.from_payload(payload)


def test_materials_rejects_native_working_float_sources_conflicting_stage_and_extended_output():
    image = _image()
    native = NativeWorkingMaster("a" * 64, descriptor_bytes=b"{}")
    with pytest.raises(ValueError, match="NPY"):
        create_materials_revision(
            replace(image, source_name="master.npy", native=native), MaterialsRecipe.from_payload(_payload())
        )
    working = PresentationPolicy(
        "working_linear", RenderRecipe(), "revision_v2", "a" * 64, image.source_sha256, image.source_identity_sha256
    )
    with pytest.raises(ValueError, match="conflicts"):
        create_materials_revision(replace(image, presentation=working), MaterialsRecipe.from_payload(_payload()))
    with pytest.raises(ValueError, match="8/16"):
        create_materials_revision(_image(pixels=np.full((4, 5, 3), 0.4, np.float32)), MaterialsRecipe.from_payload(_payload()))
    with pytest.raises(ValueError, match="unit"):
        create_materials_revision(
            _image(pixels=np.full((4, 5, 3), 65535, np.uint16)), MaterialsRecipe.from_payload(_payload())
        )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.1, 1.1])
def test_materials_rejects_nonfinite_and_extended_source_values(value):
    with pytest.raises(ValueError):
        create_materials_revision(
            _image(pixels=np.full((4, 5, 3), value, np.float32)), MaterialsRecipe.from_payload(_payload())
        )


@pytest.mark.parametrize(
    "damage",
    [
        "artifact",
        "output_color",
        "output_bits",
        "source_identity",
        "recipe",
        "policy",
        "operations",
        "master",
        "evidence",
        "stage",
        "renderer",
        "preview",
        "authority",
        "protected",
        "delta",
    ],
)
def test_materials_receipt_cannot_reinterpret_conflicting_or_forged_authority(damage):
    result = create_materials_revision(_image(), MaterialsRecipe.from_payload(_payload()))
    receipt = copy.deepcopy(result["receipt"])
    if damage == "artifact":
        receipt["revision_sha256"] = "c" * 64
    elif damage == "output_color":
        receipt["output_color"] = "extended_encoded_srgb"
    elif damage == "output_bits":
        receipt["output_bit_depth"] = 32
    elif damage == "source_identity":
        receipt["source_identity_sha256"] = "c" * 64
    elif damage == "recipe":
        receipt["recipe"]["regions"][0]["strength"] = 0.01
    elif damage == "policy":
        receipt["policy"]["max_abs_delta"] = 0.25
    elif damage == "operations":
        receipt["operations_sha256"] = "c" * 64
    elif damage == "master":
        receipt["output_master_content_sha256"] = "c" * 64
    elif damage == "evidence":
        receipt["evidence"]["shape"] = [1, 1]
    elif damage == "stage":
        receipt["output_presentation"]["stage"] = "working_linear"
    elif damage == "renderer":
        receipt["output_presentation"]["renderer_contract"] = "future"
    elif damage == "preview":
        receipt["preview_render"]["operation"] = "render_sdr"
    elif damage == "authority":
        receipt["input_presentation_policy_sha256"] = "c" * 64
    elif damage == "protected":
        receipt["execution"]["protected_changed_pixels"] = 1
    elif damage == "delta":
        receipt["execution"]["max_abs_delta"] = 0.5
    with pytest.raises(ValueError):
        revision_policy(receipt, result["receipt"]["revision_sha256"])
    assert validate_materials_revision_receipt(result["receipt"], result["receipt"]["revision_sha256"])


@pytest.mark.parametrize("schema", ["tp.photo.materials_revision.v2", "tp.photo.materials_revision.future"])
def test_unknown_materials_revision_version_never_falls_back_to_legacy_presentation(schema):
    result = create_materials_revision(_image(), MaterialsRecipe.from_payload(_payload()))
    receipt = {**result["receipt"], "schema": schema}
    with pytest.raises(ValueError, match="Unsupported native Materials revision presentation version"):
        revision_policy(receipt, result["receipt"]["revision_sha256"])
    assert revision_policy({"schema": "unknown_historical_receipt"}, result["receipt"]["revision_sha256"]) is None
