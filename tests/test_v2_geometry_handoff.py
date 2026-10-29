"""Versioned, exact-source V2 geometry without legacy resize/drop fallbacks."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageOps

from transformation_portal.lux_depth_v3.geometry import GeometryTransform
from transformation_portal.lux_depth_v3.geometry_handoff import (
    CORRECTED_GEOMETRY_RECIPE,
    GeometryBoundInputs,
    load_geometry_handoff,
    write_geometry_handoff,
)
from transformation_portal.lux_depth_v3.v2_enhance import V2EnhancementError, enhance_image
from transformation_portal.lux_depth_v3.v2_presets import V2EnhancementConfig
from transformation_portal.stage_graph.stage import StageContext
from transformation_portal.stage_graph.stages.enhancement import EnhancementStage

pytestmark = pytest.mark.unit


def _handoff(tmp_path, orientation=1, alpha=False, masks=True):
    rows, columns = np.indices((31, 45))
    pixels = np.stack((60 + rows, 70 + columns, 50 + (rows + columns) % 30), axis=-1).astype(np.uint8)
    if alpha:
        opacity = np.full(rows.shape, 255, np.uint8)
        opacity[4:9, 13:20] = 0
        opacity[11:14, 21:26] = 127
        pixels = np.dstack((pixels, opacity))
    source = tmp_path / "source.png"
    exif = Image.Exif()
    exif[274] = orientation
    with Image.fromarray(pixels) as image:
        image.save(source, exif=exif)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    transform = GeometryTransform(digest, (31, 45), orientation)
    depth = np.linspace(0, 1, np.prod(transform.normalized_shape), dtype=np.float32).reshape(transform.normalized_shape)
    depth[~transform.support()] = 0
    depth_path = tmp_path / "exact-depth.npy"
    np.save(depth_path, depth)
    masks_file = None
    if masks:
        masks_file = tmp_path / "exact-masks.npz"
        wood = np.zeros(transform.normalized_shape, np.float32)
        wood[2:16, 3:19] = 1
        np.savez_compressed(masks_file, wood=wood)
    path = tmp_path / "handoff.json"
    payload = write_geometry_handoff(
        path,
        source_sha256=digest,
        v2_input_path=source,
        plan_fingerprint_sha256=None,
        transform=transform,
        depth_map_path=depth_path,
        masks_file=masks_file,
        depth_normalization={"method": "controlled_normalized_relative_fixture"},
    )
    return source, depth_path, masks_file, path, transform, payload


def _run(source, depth, masks, handoff, tmp_path, **kwargs):
    return enhance_image(
        source,
        tmp_path / "output.png",
        depth_map_path=depth,
        geometry_recipe=CORRECTED_GEOMETRY_RECIPE,
        geometry_handoff_path=handoff,
        masks_file=masks,
        input_color="srgb",
        output_bit_depth=8,
        **kwargs,
    )


def _track_single_corrected_float_preparation(monkeypatch):
    import transformation_portal.lux_depth_v3.v2_enhance as module

    original_convert = module.bounded_encoded_srgb
    original_rint = np.rint
    original_compute = EnhancementStage.compute
    conversions = []
    stage_started = False

    def convert(master):
        pixels = original_convert(master)
        conversions.append(master.source_bit_depth)
        return pixels

    def rint(values, *args, **kwargs):
        if conversions and not stage_started and isinstance(values, np.ndarray) and values.ndim == 3:
            pytest.fail("Corrected V2 allocated a discarded integer RGB proxy before enhancement")
        return original_rint(values, *args, **kwargs)

    def compute(stage, context):
        nonlocal stage_started
        stage_started = True
        return original_compute(stage, context)

    monkeypatch.setattr(module, "bounded_encoded_srgb", convert)
    monkeypatch.setattr(np, "rint", rint)
    monkeypatch.setattr(EnhancementStage, "compute", compute)
    return conversions


@pytest.mark.parametrize("orientation", range(1, 9))
def test_corrected_v2_protects_full_pipeline_support_and_alpha(tmp_path, monkeypatch, orientation):
    source, depth, masks, handoff, transform, payload = _handoff(tmp_path, orientation, alpha=True)
    conversions = _track_single_corrected_float_preparation(monkeypatch)

    def forbidden(*_args, **_kwargs):
        pytest.fail("Corrected V2 entered a legacy resize, drop-mask or second source-load fallback")

    import transformation_portal.lux_depth_v3.v2_enhance as module

    monkeypatch.setattr(module, "load_image_preserve_bit_depth", forbidden)
    monkeypatch.setattr(EnhancementStage, "_resize_depth_map", forbidden)
    monkeypatch.setattr(EnhancementStage, "_normalize_material_masks", forbidden)
    report = _run(source, depth, masks, handoff, tmp_path)
    with Image.open(report["output"]) as result:
        actual = np.asarray(result)
    with Image.open(source) as encoded:
        with ImageOps.exif_transpose(encoded) as oriented:
            expected = np.asarray(oriented)
    protected = ~transform.support() | (expected[..., 3] != 255)
    assert protected.any() and (~protected).any()
    np.testing.assert_array_equal(actual[..., :3][protected], expected[..., :3][protected])
    np.testing.assert_array_equal(actual[..., 3], expected[..., 3])
    assert np.any(actual[..., :3][~protected] != expected[..., :3][~protected])
    assert report["processing_version"] == report["config"]["version"] == "1.3.0"
    assert report["geometry"]["handoff_identity_sha256"] == payload["identity_sha256"]
    assert report["geometry"]["handoff"] == payload
    assert report["geometry"]["source_sha256"] == transform.source_sha256
    assert report["geometry"]["shape"] == list(transform.normalized_shape)
    assert conversions == [8]
    assert report["bit_depth"]["input_dtype"] == "uint8"


@pytest.mark.parametrize("mutation", ["source", "depth", "masks", "receipt", "missing_depth", "missing_masks"])
def test_changed_or_missing_handoff_artifacts_fail_before_edit_or_output(tmp_path, monkeypatch, mutation):
    source, depth, masks, handoff, transform, payload = _handoff(tmp_path)
    if mutation == "source":
        with Image.open(source) as original:
            Image.fromarray(np.asarray(original) + np.uint8(1)).save(source)
    elif mutation == "depth":
        np.save(depth, np.ones(transform.normalized_shape, np.float32))
    elif mutation == "masks":
        np.savez_compressed(masks, wood=np.ones((1, 1), np.float32))
    elif mutation == "receipt":
        payload["source_sha256"] = "e" * 64
        handoff.write_text(json.dumps(payload))
    elif mutation == "missing_depth":
        depth.unlink()
    else:
        masks.unlink()

    def forbidden(*_args, **_kwargs):
        pytest.fail("Invalid geometry reached photographic edits")

    monkeypatch.setattr(EnhancementStage, "_enhance_image", forbidden)
    with pytest.raises(V2EnhancementError):
        _run(source, depth, masks, handoff, tmp_path)
    assert not list(tmp_path.glob("*output*"))


def test_strict_stage_requires_exact_typed_arrays_and_binds_cache_identity(tmp_path):
    source, depth, masks, handoff, transform, payload = _handoff(tmp_path)
    bound = load_geometry_handoff(handoff, input_bytes=source.read_bytes(), depth_map_path=depth, masks_file=masks)
    image = np.full((*transform.normalized_shape, 3), 0.25, np.float32)
    bound = bound.with_image(image)
    assert not bound.image.flags.writeable and not bound.depth_map.flags.writeable
    stage = EnhancementStage(geometry_recipe=CORRECTED_GEOMETRY_RECIPE, version="1.3.0")
    context = StageContext()
    context.set_artifact("image", image)
    context.set_artifact("depth_map", bound.depth_map)
    context.set_artifact("material_masks", dict(bound.masks))
    with pytest.raises(ValueError, match="verified geometry"):
        stage.compute(context)
    with pytest.raises(ValueError, match="verified geometry"):
        stage.get_cache_key(context)
    context.set_artifact("geometry_handoff", bound)
    first = stage.get_cache_key(context)
    changed_payload = write_geometry_handoff(
        handoff,
        source_sha256=transform.source_sha256,
        v2_input_path=source,
        plan_fingerprint_sha256="d" * 64,
        transform=transform,
        depth_map_path=depth,
        masks_file=masks,
    )
    changed = load_geometry_handoff(handoff, input_bytes=source.read_bytes(), depth_map_path=depth, masks_file=masks)
    assert changed_payload["identity_sha256"] != payload["identity_sha256"]
    context.set_artifact("geometry_handoff", changed.with_image(image))
    assert stage.get_cache_key(context) != first
    context.set_artifact("geometry_handoff", bound)
    context.set_artifact("material_masks", {"wood": np.zeros((1, 1), np.float32)})
    with pytest.raises(ValueError, match="mask differs"):
        stage.compute(context)
    with pytest.raises(ValueError, match="mask differs"):
        stage.get_cache_key(context)


@pytest.mark.parametrize(
    "kind", ["nan_depth", "infinite_depth", "outside_depth", "nan_mask", "missing_support", "unsupported_depth"]
)
def test_bound_arrays_fail_closed(kind):
    depth = np.full((14, 14), 0.5, np.float32)
    support = np.ones((14, 14), bool)
    masks = {"wood": np.ones((14, 14), np.float32)}
    if kind == "nan_depth":
        depth[0, 0] = np.nan
    elif kind == "infinite_depth":
        depth[0, 0] = np.inf
    elif kind == "outside_depth":
        depth[0, 0] = 2
    elif kind == "nan_mask":
        masks["wood"][0, 0] = np.nan
    elif kind == "missing_support":
        support = np.ones((1, 1), bool)
    else:
        support[0, 0] = False
    with pytest.raises(ValueError):
        GeometryBoundInputs("a" * 64, "b" * 64, "c" * 64, (14, 14), depth, support, masks)


def test_corrected_passthrough_still_validates_exact_handoff(tmp_path):
    source, depth, masks, handoff, _, payload = _handoff(tmp_path)
    report = _run(source, depth, masks, handoff, tmp_path, config=V2EnhancementConfig.from_preset("none"))
    assert report["processing_version"] == "1.3.0"
    assert report["geometry"]["handoff_identity_sha256"] == payload["identity_sha256"]
    assert report["depth_consumed"] is False


@pytest.mark.parametrize("with_masks", [False, True])
@pytest.mark.parametrize("bit_depth", [8, 16])
@pytest.mark.parametrize("preset", ["default", "none"])
def test_corrected_output_label_matches_verified_mask_inventory(tmp_path, with_masks, bit_depth, preset):
    source, depth, masks, handoff, _, payload = _handoff(tmp_path, masks=with_masks)
    report = enhance_image(
        source,
        tmp_path / "final_materials_v3_enhanced.png",
        depth_map_path=depth,
        geometry_recipe=CORRECTED_GEOMETRY_RECIPE,
        geometry_handoff_path=handoff,
        masks_file=masks,
        input_color="srgb",
        output_bit_depth=bit_depth,
        config=V2EnhancementConfig.from_preset(preset),
    )
    label = "materials_v3_enhanced" if with_masks else "v2_enhanced"
    extension = "tif" if bit_depth == 16 else "png"
    assert Path(report["output"]).name == f"final_{label}.{extension}"
    assert Path(report["output"]).is_file()
    assert report["geometry"]["handoff_identity_sha256"] == payload["identity_sha256"]
    assert report["processing_version"] == "1.3.0"
    if preset != "none":
        assert report["enhancement_metadata"]["material_masks_supplied"] is with_masks


@pytest.mark.parametrize("orientation", [6, 7])
def test_corrected_native_16bit_tiff_preserves_protected_samples_and_alpha(tmp_path, monkeypatch, orientation):
    import tifffile

    rows, columns = np.indices((31, 45))
    rgb = np.stack((10000 + rows * 177, 17000 + columns * 123, 9000 + (rows + columns) * 91), axis=-1).astype(np.uint16)
    alpha = np.full(rows.shape, 65535, np.uint16)
    alpha[5:11, 17:24] = 12345
    source = tmp_path / "native.tif"
    tifffile.imwrite(
        source,
        np.dstack((rgb, alpha)),
        photometric="rgb",
        extrasamples="unassalpha",
        extratags=[(274, "H", 1, orientation, False)],
    )
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    transform = GeometryTransform(digest, (31, 45), orientation)
    depth = np.full(transform.normalized_shape, 0.5, np.float32)
    depth[~transform.support()] = 0
    depth_path, handoff = tmp_path / "depth.npy", tmp_path / "handoff.json"
    np.save(depth_path, depth)
    write_geometry_handoff(
        handoff,
        source_sha256=digest,
        v2_input_path=source,
        plan_fingerprint_sha256=None,
        transform=transform,
        depth_map_path=depth_path,
    )
    conversions = _track_single_corrected_float_preparation(monkeypatch)
    report = enhance_image(
        source,
        tmp_path / "output.tif",
        depth_map_path=depth_path,
        output_bit_depth=16,
        input_color="srgb",
        geometry_recipe=CORRECTED_GEOMETRY_RECIPE,
        geometry_handoff_path=handoff,
    )
    actual = tifffile.imread(report["output"])
    with Image.fromarray(rgb[..., 0].astype(np.float32)) as reference:
        reference.getexif()[274] = orientation
        with ImageOps.exif_transpose(reference) as oriented:
            expected_red = np.asarray(oriented).astype(np.uint16)
    with Image.fromarray(alpha.astype(np.float32)) as reference:
        reference.getexif()[274] = orientation
        with ImageOps.exif_transpose(reference) as oriented:
            expected_alpha = np.asarray(oriented).astype(np.uint16)
    protected = ~transform.support() | (expected_alpha != 65535)
    assert actual.dtype == np.uint16 and actual.shape == (*transform.normalized_shape, 4)
    np.testing.assert_array_equal(actual[..., 0][protected], expected_red[protected])
    np.testing.assert_array_equal(actual[..., 3], expected_alpha)
    assert report["bit_depth"]["output_bits_per_sample"] == 16
    assert report["bit_depth"]["input_dtype"] == "uint16"
    assert conversions == [16]


def test_recipe_requires_explicit_geometry_and_rejects_bare_masks(tmp_path):
    source, depth, masks, handoff, _, _ = _handoff(tmp_path)
    with pytest.raises(V2EnhancementError, match="explicit geometry"):
        enhance_image(source, tmp_path / "out.png", geometry_recipe=CORRECTED_GEOMETRY_RECIPE)
    with pytest.raises(V2EnhancementError, match="bare arrays"):
        _run(source, depth, masks, handoff, tmp_path, material_masks={"wood": np.ones((31, 45), np.float32)})
    with pytest.raises(V2EnhancementError, match="corrected recipe"):
        enhance_image(source, tmp_path / "out.png", geometry_handoff_path=handoff)


def test_corrected_source_metadata_and_pixels_share_one_byte_snapshot(tmp_path, monkeypatch):
    source, depth, masks, handoff, _, payload = _handoff(tmp_path, 6)
    original_read = Path.read_bytes

    def source_reread_forbidden(path):
        if path == source:
            pytest.fail("Corrected V2 reopened its source rather than consuming the verified byte snapshot")
        return original_read(path)

    monkeypatch.setattr(Path, "read_bytes", source_reread_forbidden)
    report = _run(source, depth, masks, handoff, tmp_path)
    assert report["geometry"]["v2_input_sha256"] == payload["v2_input_sha256"]
    assert report["io"]["load_backend"] == "verified_byte_snapshot"


@pytest.mark.parametrize("with_masks", [False, True])
def test_corrected_v2_runner_roundtrip_uses_explicit_depth_and_receipt(tmp_path, with_masks):
    from transformation_portal.lux_depth_v3.v2_runner import V2Runner

    source, depth, masks, handoff, _, payload = _handoff(tmp_path, 2, masks=with_masks)
    result = V2Runner().run(
        input_path=source,
        depth_dir=None,
        output_dir=tmp_path / "subprocess-output",
        masks_file=masks,
        asset_key="unrelated-to-depth-filename",
        input_color="srgb",
        geometry_recipe=CORRECTED_GEOMETRY_RECIPE,
        geometry_handoff_path=handoff,
        depth_map_path=depth,
        timeout=30,
    )
    assert result["status"] == "success"
    assert result["processing_version"] == "1.3.0"
    assert result["geometry"]["handoff_identity_sha256"] == payload["identity_sha256"]
    assert Path(result["output"]).is_file()
    assert Path(result["report_path"]).is_file()
    assert result["depth"]["resolved_path"] == str(depth)
    label = "materials_v3_enhanced" if with_masks else "v2_enhanced"
    assert Path(result["output"]).name == f"unrelated-to-depth-filename_{label}.png"


def test_corrected_runner_forbids_filename_discovery(tmp_path):
    from transformation_portal.lux_depth_v3.v2_runner import V2Runner

    source, depth, masks, handoff, _, _ = _handoff(tmp_path)
    with pytest.raises(ValueError, match="forbids depth discovery"):
        V2Runner().run(
            input_path=source,
            depth_dir=tmp_path,
            output_dir=tmp_path / "unused-output",
            masks_file=masks,
            geometry_recipe=CORRECTED_GEOMETRY_RECIPE,
            geometry_handoff_path=handoff,
            depth_map_path=depth,
        )


def test_handoff_rejects_new_path_authority_members(tmp_path):
    source, depth, masks, handoff, _, payload = _handoff(tmp_path)
    payload["depth_path"] = "/this/must/not/authorize/a/read.npy"
    handoff.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="unsupported fields"):
        load_geometry_handoff(handoff, input_bytes=source.read_bytes(), depth_map_path=depth, masks_file=masks)


def test_handoff_rejects_oversized_npy_header_before_materialization(tmp_path, monkeypatch):
    source, depth, masks, handoff, transform, _ = _handoff(tmp_path)
    depth.write_bytes(b"\x93NUMPY\x02\x00" + (1_000_000).to_bytes(4, "little") + b" " * 200)
    monkeypatch.setattr(np, "load", lambda *_args, **_kwargs: pytest.fail("Unbounded depth reached NumPy loading"))
    with pytest.raises(ValueError, match="header exceeds"):
        load_geometry_handoff(handoff, input_bytes=source.read_bytes(), depth_map_path=depth, masks_file=masks)


@pytest.mark.parametrize("field", ["depth_map", "support", "masks"])
def test_verified_bound_inputs_cannot_be_replaced_with_different_array_authority(tmp_path, field):
    source, depth, masks, handoff, transform, _ = _handoff(tmp_path)
    bound = load_geometry_handoff(handoff, input_bytes=source.read_bytes(), depth_map_path=depth, masks_file=masks)
    changes = {}
    if field == "depth_map":
        changes["depth_map"] = np.zeros(transform.normalized_shape, np.float32)
    if field == "support":
        support = bound.support.copy()
        support[0, 0] = True
        changes["support"] = support
    elif field == "masks":
        changes["masks"] = {"wood": np.zeros(transform.normalized_shape, np.float32)}
    with pytest.raises(ValueError, match="exact validated handoff"):
        replace(bound, **changes)


def _tone_result(image, depth, support, *, alpha=None, smoothing=False, corrected=True):
    stage = EnhancementStage(
        enhancement_strength=1,
        clarity_strength=0,
        material_strength=0,
        tone_low_tex_strength=0,
        tone_depth_smoothing=smoothing,
        geometry_recipe=CORRECTED_GEOMETRY_RECIPE if corrected else "legacy_encoded_crop_stretch_v1",
        version="1.3.0" if corrected else "1.2.0",
    )
    context = StageContext()
    context.set_artifact("image", image)
    context.set_artifact("depth_map", depth)
    if corrected:
        bound = GeometryBoundInputs("a" * 64, "b" * 64, "c" * 64, support.shape, depth, support).with_image(image, alpha)
        context.set_artifact("geometry_handoff", bound)
    return stage.compute(context).artifacts["enhanced_image"]


def test_corrected_tone_statistics_exclude_unknown_native_border():
    transform = GeometryTransform("a" * 64, (27, 27), 1)
    row, column = np.indices(transform.model_shape)
    model = ((row * 14 + column) / 195).astype(np.float32)
    depth, support = transform.restore_numeric(model)
    image = np.full((27, 27, 3), 0.5, np.float32)
    actual = _tone_result(image, depth, support)
    center = float(np.percentile(depth[support], 75))
    assert center == 0.75
    assert float(np.percentile(depth, 75)) < 0.1
    factor = np.tanh(np.clip((depth[support] - center) / (1 - center + 1e-6), -1, 1) * 2)
    gain = np.where(factor < 0, 1 - factor * 0.12, 1 - factor * 0.08)
    expected = np.clip(image[support] * gain[:, None] * 255, 0, 255).astype(np.uint8)
    np.testing.assert_array_equal(actual[support], expected)
    np.testing.assert_array_equal(actual[~support], np.full((np.count_nonzero(~support), 3), 128, np.uint8))
    legacy = _tone_result(image, depth, support, corrected=False)
    assert np.count_nonzero(np.max(np.abs(legacy[support].astype(int) - expected.astype(int)), axis=1) > 1) > 100


def test_corrected_smoothing_and_statistics_ignore_protected_alpha_depth():
    transform = GeometryTransform("a" * 64, (27, 27), 1)
    support = transform.support()
    alpha = np.ones(support.shape, np.float32)
    alpha[8:12, 8:13] = 0.5
    alpha[15:18, 14:18] = 0
    image = np.full((*support.shape, 3), 0.5, np.float32)
    allowed = support & (alpha == 1)
    first = np.where(support, 0.75, 0).astype(np.float32)
    second = first.copy()
    first[support & ~allowed] = 0
    second[support & ~allowed] = 1
    actual_first = _tone_result(image, first, support, alpha=alpha, smoothing=True)
    actual_second = _tone_result(image, second, support, alpha=alpha, smoothing=True)
    np.testing.assert_array_equal(actual_first, actual_second)
    np.testing.assert_array_equal(actual_first[allowed], np.full((np.count_nonzero(allowed), 3), 127, np.uint8))
    np.testing.assert_array_equal(actual_first[~allowed], np.full((np.count_nonzero(~allowed), 3), 128, np.uint8))


def test_corrected_empty_tone_domain_preserves_source_without_percentiles(monkeypatch):
    image = np.full((14, 14, 3), 0.3, np.float32)
    support = np.ones((14, 14), bool)
    depth = np.full((14, 14), 0.5, np.float32)
    monkeypatch.setattr(
        np, "percentile", lambda *_args, **_kwargs: pytest.fail("Empty photographic domain reached percentile")
    )
    actual = _tone_result(image, depth, support, alpha=np.zeros((14, 14), np.float32), smoothing=True)
    np.testing.assert_array_equal(actual, np.rint(image * 255).astype(np.uint8))


def test_fully_supported_tone_retains_legacy_pixels():
    rng = np.random.default_rng(8)
    image = rng.uniform(0.1, 0.8, (14, 14, 3)).astype(np.float32)
    depth = rng.uniform(0, 1, (14, 14)).astype(np.float32)
    support = np.ones((14, 14), bool)
    np.testing.assert_array_equal(
        _tone_result(image, depth, support, smoothing=True),
        _tone_result(image, depth, support, smoothing=True, corrected=False),
    )


@pytest.mark.parametrize("field", ["depth_map", "support", "mask"])
def test_numpy_shape_metadata_alias_cannot_bypass_geometry_cache_validation(tmp_path, field):
    source, depth, masks, handoff, transform, _ = _handoff(tmp_path)
    image = np.full((*transform.normalized_shape, 3), 0.3, np.float32)
    bound = load_geometry_handoff(handoff, input_bytes=source.read_bytes(), depth_map_path=depth, masks_file=masks).with_image(
        image
    )
    array = bound.masks["wood"] if field == "mask" else getattr(bound, field)
    array.shape = tuple(reversed(array.shape))
    stage = EnhancementStage(geometry_recipe=CORRECTED_GEOMETRY_RECIPE)
    context = StageContext()
    context.set_artifact("image", image)
    context.set_artifact("depth_map", bound.depth_map)
    context.set_artifact("material_masks", dict(bound.masks))
    context.set_artifact("geometry_handoff", bound)
    with pytest.raises(ValueError, match="verified native geometry"):
        stage.get_cache_key(context)
    with pytest.raises(ValueError, match="verified native geometry"):
        stage.compute(context)
