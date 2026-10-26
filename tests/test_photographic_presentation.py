"""Recorded stage authority survives containers without a second tone transform."""

from __future__ import annotations

import hashlib
import io
from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest
import tifffile
from PIL import Image

from transformation_portal.core.image_artifact import ImageMaster
from transformation_portal.ingest.canonical_json import canonicalize_json
from transformation_portal.lux_depth_v6.color import RenderRecipe, render_master
from transformation_portal.lux_depth_v6.products import _delivery
from transformation_portal.orchestrator.photographic_review_access import ReviewAccessError
from transformation_portal.orchestrator.photographic_review_service import _display, revision_refs
from transformation_portal.photographic_delivery import (
    DeliveryImage,
    ExportPreset,
    NativeWorkingMaster,
    RevisionRecipe,
    create_revision,
    export_delivery,
)
from transformation_portal.photographic_delivery.presentation import PresentationPolicy, descriptor_policy, revision_policy

pytestmark = pytest.mark.unit


def _sha(value):
    return hashlib.sha256(value).hexdigest()


def _pixels(raw):
    return np.asarray(Image.open(io.BytesIO(raw)))


def _case(*, mode="perceptual_srgb", pixels=None, alpha=None):
    if pixels is None:
        pixels = np.tile(np.array([1.6, 0.92, 0.35], np.float32), (8, 12, 1))
    master = ImageMaster(pixels, "a" * 64, 16, alpha, {"source_format": "TIFF", "input_color": "srgb"})
    recipe = RenderRecipe(mode=mode, shoulder=0.7)
    display, render = render_master(master, recipe)
    raw, delivery = _delivery(display, "input-0000/delivery.tif")
    descriptor = {
        "schema": "tp.lux.graded_photograph.v2",
        "input_id": "input-0000",
        "master": master.to_payload(),
        "display": display.to_payload(),
        "render": render,
        "delivery": delivery,
    }
    descriptor_bytes = canonicalize_json(descriptor)
    policy = descriptor_policy(
        descriptor,
        descriptor_sha256=_sha(descriptor_bytes),
        artifact_sha256=_sha(raw),
        source_identity_sha256=_sha(raw),
        delivery_path="input-0000/delivery.tif",
    )
    image = DeliveryImage("arbitrary-name.tif", _sha(raw), "Room", source_bytes=raw, presentation=policy)
    buffer = io.BytesIO()
    np.save(buffer, pixels, allow_pickle=False)
    array = buffer.getvalue()
    alpha_bytes = None
    if alpha is not None:
        buffer = io.BytesIO()
        np.save(buffer, alpha, allow_pickle=False)
        alpha_bytes = buffer.getvalue()
    native = NativeWorkingMaster(
        _sha(descriptor_bytes),
        descriptor_bytes=descriptor_bytes,
        alpha_sha256=_sha(alpha_bytes) if alpha_bytes else None,
        alpha_bytes=alpha_bytes,
    )
    working_policy = descriptor_policy(
        descriptor,
        descriptor_sha256=_sha(descriptor_bytes),
        artifact_sha256=_sha(array),
        source_identity_sha256=native.identity(_sha(array)),
    )
    working = DeliveryImage(
        "unrelated.npy", _sha(array), "Room", source_bytes=array, native=native, presentation=working_policy
    )
    return image, working, descriptor


@pytest.mark.parametrize("mode", ["perceptual_srgb", "soft_srgb", "clip_srgb"])
def test_working_revision_preview_round_trip_and_native_crop_keep_recorded_recipe(mode):
    _, working, _ = _case(mode=mode)
    result = create_revision(working, RevisionRecipe(exposure_stops=-0.1))
    receipt = result["receipt"]
    policy = revision_policy(receipt, receipt["revision_sha256"])
    reopened = DeliveryImage(
        "saved.tif", receipt["revision_sha256"], "Room", source_bytes=result["source_bytes"], presentation=policy
    )
    np.testing.assert_array_equal(_pixels(_display(reopened, view="recorded")[0]), _pixels(result["preview_bytes"]))
    crop = _pixels(_display(reopened, (2, 1, 4, 3), view="recorded")[0])
    np.testing.assert_array_equal(crop, _pixels(result["preview_bytes"])[1:4, 2:6])
    assert policy.recipe.mode == mode and policy.recipe.shoulder == 0.7
    assert receipt["output_presentation"]["stage"] == "working_linear"


def test_verified_delivery_and_its_new_revisions_exports_and_contact_receive_no_second_render(tmp_path, monkeypatch):
    delivery, _, _ = _case()

    def twice(*args, **kwargs):
        raise AssertionError("Already rendered photographic bytes received a second tone transform")

    monkeypatch.setattr("transformation_portal.photographic_delivery.presentation.render_master", twice)
    observed = _pixels(_display(delivery, view="recorded")[0])
    np.testing.assert_array_equal(observed, _pixels(_display(delivery)[0]))
    result = create_revision(delivery, RevisionRecipe())
    assert result["receipt"]["output_presentation"]["stage"] == "display_referred"
    assert result["receipt"]["preview_render"]["operation"] == "preserve_recorded_display"
    policy = revision_policy(result["receipt"], result["receipt"]["revision_sha256"])
    revision = DeliveryImage(
        "saved.tif", policy.artifact_sha256, "Room", source_bytes=result["source_bytes"], presentation=policy
    )
    np.testing.assert_array_equal(_pixels(_display(revision, view="recorded")[0]), observed)
    for index, source in enumerate((delivery, revision)):
        inventory = export_delivery(
            [source], tmp_path / str(index), property_name="House", export_presets=[ExportPreset.from_payload("print")]
        )
        assert inventory["files"][0]["render"]["operation"] == "preserve_recorded_display"
        assert inventory["contact_sheet"]["sources"][0]["render"]["operation"] == "preserve_recorded_display"
        encoded = tifffile.imread(tmp_path / str(index) / inventory["files"][0]["path"])
        original = tifffile.imread(io.BytesIO(delivery.source_bytes))
        np.testing.assert_allclose(encoded, original, atol=1, rtol=0)
    master_only = export_delivery(
        [delivery],
        tmp_path / "source-only",
        property_name="House",
        export_presets=[ExportPreset.from_payload("photographic_master")],
    )
    assert master_only["contact_sheet"]["sources"][0]["render"]["operation"] == "preserve_recorded_display"


@pytest.mark.parametrize(
    "recorded,requested",
    [("soft_srgb", "perceptual_srgb"), ("clip_srgb", "perceptual_srgb"), ("perceptual_srgb", "soft_srgb")],
)
def test_authoritative_working_export_uses_recorded_recipe_and_reports_requested_recipe(tmp_path, recorded, requested):
    _, working, _ = _case(mode=recorded)
    preset = ExportPreset.from_payload({"name": "print", "render": requested, "shoulder": 0.9})
    inventory = export_delivery([working], tmp_path / "export", property_name="House", export_presets=[preset])
    row = inventory["files"][0]
    assert row["render"]["recipe"] == working.presentation.recipe.to_payload()
    assert row["render"]["requested_recipe"] == RenderRecipe(requested, 0.9).to_payload()
    assert row["presentation_policy_sha256"] == working.presentation.sha256
    expected, _ = render_master(working.decode(), working.presentation.recipe)
    expected_bytes, _ = _delivery(expected, "expected.tif")
    np.testing.assert_array_equal(
        tifffile.imread(tmp_path / "export" / row["path"]), tifffile.imread(io.BytesIO(expected_bytes))
    )
    assert inventory["contact_sheet"]["sources"][0]["render"]["recipe"] == working.presentation.recipe.to_payload()


def test_recognized_v1_revision_preserves_its_historical_working_render_semantics():
    delivery, _, _ = _case()
    # V1 did not retain input stage, even for a rendered delivery parent.
    # Its actual behavior was the legacy default perceptual preview operation.
    result = create_revision(replace(delivery, presentation=None), RevisionRecipe())
    receipt = {
        key: value
        for key, value in result["receipt"].items()
        if key
        not in {
            "output_presentation",
            "input_presentation_policy_sha256",
            "processing_contract",
            "input_presentation_authority",
        }
    }
    receipt["schema"] = "tp.photo.revision.v1"
    policy = revision_policy(receipt, receipt["revision_sha256"])
    assert policy.stage == "working_linear" and policy.authority_kind == "revision_v1"
    assert policy.recipe == RenderRecipe()
    reopened = DeliveryImage(
        "saved.tif", policy.artifact_sha256, "Room", source_bytes=result["source_bytes"], presentation=policy
    )
    np.testing.assert_array_equal(_pixels(_display(reopened, view="recorded")[0]), _pixels(result["preview_bytes"]))
    assert "presentation_policy" not in receipt
    assert policy.authority_sha256 == _sha(canonicalize_json(receipt))


def test_unknown_files_and_receipts_never_acquire_stage_from_name_dtype_range_or_tiff_metadata():
    delivery, _, _ = _case()
    unknown = replace(delivery, presentation=None)
    for name in ("delivery.tif", "master.tif", "float-working.tif"):
        with pytest.raises(ReviewAccessError, match="Recorded presentation is unavailable"):
            _display(replace(unknown, source_name=name), view="recorded")
        assert _display(replace(unknown, source_name=name))[0]
    assert revision_policy({"schema": "unknown", "revision_sha256": delivery.source_sha256}, delivery.source_sha256) is None
    result = create_revision(unknown, RevisionRecipe())
    assert unknown.presentation is None
    assert result["receipt"]["input_presentation_authority"] == "explicit_legacy_revision_operation"
    assert result["receipt"]["input_presentation_policy_sha256"] is None
    policy = revision_policy(result["receipt"], result["receipt"]["revision_sha256"])
    assert policy.authority_kind == "revision_v2" and policy.stage == "working_linear"
    # The new receipt records the actual existing legacy operation on new bytes;
    # it makes no inference or claim about the parent's processing stage.
    with pytest.raises(ReviewAccessError, match="Recorded presentation is unavailable"):
        _display(unknown, view="recorded")


@pytest.mark.parametrize("damage", ["digest", "stage", "recipe", "delivery", "display"])
def test_policy_and_descriptor_fail_closed_on_conflicting_authority(damage):
    delivery, working, descriptor = _case()
    if damage == "digest":
        with pytest.raises(ValueError, match="different photographic bytes"):
            replace(delivery, source_sha256="0" * 64)
    elif damage == "stage":
        with pytest.raises(ValueError, match="already received"):
            replace(delivery.presentation, stage="working_linear")
    else:
        if damage == "recipe":
            descriptor["render"]["recipe"]["mode"] = "unknown"
        elif damage == "delivery":
            descriptor["delivery"]["master_content_sha256"] = "0" * 64
        else:
            descriptor["display"]["metadata"]["color_domain"] = "working"
        with pytest.raises(ValueError):
            descriptor_policy(
                descriptor,
                descriptor_sha256="1" * 64,
                artifact_sha256=delivery.source_sha256,
                source_identity_sha256=delivery.source_sha256,
                delivery_path="input-0000/delivery.tif",
            )
    with pytest.raises(FrozenInstanceError):
        working.presentation.stage = "display_referred"
    with pytest.raises(ValueError, match="working content"):
        replace(working, presentation=replace(working.presentation, input_master_sha256="0" * 64)).decode()


def test_recorded_fit_renders_before_area_reduction_and_retains_straight_alpha():
    pixels = np.array([[[4, 0.1, 0.01], [0.01, 0.1, 4]], [[0.01, 0.1, 4], [4, 0.1, 0.01]]], np.float32)
    alpha = np.array([[1, 0.3], [0.3, 1]], np.float32)
    _, working, _ = _case(pixels=pixels, alpha=alpha)
    display, _ = render_master(working.decode(), working.presentation.recipe)
    from transformation_portal.lux_depth_v5.preview import preview_samples

    expected = preview_samples(display, max_edge=1)
    observed = _pixels(_display(working, max_edge=1, view="recorded")[0])
    np.testing.assert_array_equal(observed, expected)
    assert observed.shape == (1, 1, 4)
    # Demonstrate that moving the non-linear render after reduction is observable.
    reduced = ImageMaster(np.average(pixels, axis=(0, 1), weights=alpha)[None, None].astype(np.float32), "a" * 64, 16)
    wrong, _ = render_master(reduced, working.presentation.recipe)
    assert not np.array_equal(observed[..., :3], preview_samples(wrong)[..., :3])


def test_revision_reference_policy_is_derived_from_committed_receipt_not_variant_fields():
    delivery, _, _ = _case()
    result = create_revision(delivery, RevisionRecipe())
    variant = {
        "sha256": result["receipt"]["revision_sha256"],
        "presentation_policy": {"forged": True},
        "presentation_policy_sha256": "0" * 64,
    }
    ref = revision_refs({"revisions": [{"variant": variant, "receipt": result["receipt"]}]})[0]
    policy = PresentationPolicy.from_payload(ref["presentation_policy"])
    assert policy.sha256 == ref["presentation_policy_sha256"]
    assert policy.stage == "display_referred"
    assert variant["presentation_policy"] == {"forged": True}
    assert policy.to_payload()["renderer_contract"] == "tp.photo.presentation_renderer.v1"
    assert policy.sha256 == _sha(canonicalize_json(policy.to_payload()))
    changed = {**policy.to_payload(), "renderer_contract": "tp.photo.presentation_renderer.v2"}
    with pytest.raises(ValueError, match="Unsupported recorded"):
        PresentationPolicy.from_payload(changed)


@pytest.mark.parametrize(
    "damage",
    [
        "missing_contract",
        "unknown_contract",
        "null_recorded_input",
        "legacy_input_identity",
        "legacy_display_output",
        "missing_content",
        "mismatched_preview",
    ],
)
def test_v2_receipt_requires_matching_processing_and_stage_authority(damage):
    import copy

    delivery, _, _ = _case()
    receipt = copy.deepcopy(create_revision(delivery, RevisionRecipe())["receipt"])
    if damage == "missing_contract":
        receipt.pop("processing_contract")
    elif damage == "unknown_contract":
        receipt["processing_contract"] = "tp.photo.revision_processing.future"
    elif damage == "null_recorded_input":
        receipt["input_presentation_policy_sha256"] = None
    elif damage in {"legacy_input_identity", "legacy_display_output"}:
        receipt["input_presentation_authority"] = "explicit_legacy_revision_operation"
        if damage == "legacy_display_output":
            receipt["input_presentation_policy_sha256"] = None
    elif damage == "missing_content":
        receipt.pop("output_master_content_sha256")
    else:
        receipt["preview_render"]["input_master_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        revision_policy(receipt, receipt["revision_sha256"])
