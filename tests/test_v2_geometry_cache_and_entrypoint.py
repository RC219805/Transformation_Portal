"""Corrected cache inputs and script recipe validation remain fail-closed."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from transformation_portal.lux_depth_v3.geometry import LEGACY_GEOMETRY_RECIPE, NORMALIZED_GEOMETRY_RECIPE
from transformation_portal.lux_depth_v3.geometry_handoff import GeometryBoundInputs
from transformation_portal.stage_graph.stage import StageContext, StageStatus
from transformation_portal.stage_graph.stages.enhancement import EnhancementStage

pytestmark = pytest.mark.unit


def _context(cache_dir, *, mutation=None):
    image = np.full((14, 14, 3), 0.5, np.float32)
    depth = np.indices((14, 14))[1].astype(np.float32) / 13
    support = np.ones((14, 14), bool)
    masks = {"wood": np.zeros((14, 14), np.float32)}
    if mutation == "depth":
        depth = depth[:, ::-1].copy()
    elif mutation == "support":
        support[4:10, 0] = False  # The existing depth is zero at these locations.
    elif mutation == "mask":
        masks["wood"][3:11, 3:11] = 1
    bound = GeometryBoundInputs("a" * 64, "b" * 64, "c" * 64, (14, 14), depth, support, masks).with_image(image)
    return StageContext(
        artifacts={
            "image": bound.image,
            "depth_map": bound.depth_map,
            "material_masks": dict(bound.masks),
            "geometry_handoff": bound,
        },
        cache_dir=cache_dir,
    )


@pytest.mark.parametrize("mutation", ["depth", "support", "mask"])
def test_corrected_execute_cache_binds_actual_arrays_even_without_receipt(tmp_path, mutation):
    stage = EnhancementStage(geometry_recipe=NORMALIZED_GEOMETRY_RECIPE, version="1.3.0")
    original = _context(tmp_path / "cache")
    changed = _context(tmp_path / "cache", mutation=mutation)
    first_bound = original.get_artifact("geometry_handoff")
    changed_bound = changed.get_artifact("geometry_handoff")
    assert first_bound.handoff is changed_bound.handoff is None
    assert first_bound.identity_sha256 == changed_bound.identity_sha256

    first = stage.execute(original)
    assert first.status == StageStatus.COMPLETED and not first.cache_hit
    repeat = stage.execute(original)
    assert repeat.status == StageStatus.CACHED and repeat.cache_hit
    second = stage.execute(changed)
    assert second.status == StageStatus.COMPLETED and not second.cache_hit
    assert second.cache_key != first.cache_key
    assert np.any(second.artifacts["enhanced_image"] != first.artifacts["enhanced_image"])
    np.testing.assert_array_equal(second.artifacts["enhanced_image"], stage.compute(changed).artifacts["enhanced_image"])
    changed_repeat = stage.execute(changed)
    assert changed_repeat.status == StageStatus.CACHED and changed_repeat.cache_hit
    np.testing.assert_array_equal(changed_repeat.artifacts["enhanced_image"], second.artifacts["enhanced_image"])


def test_corrected_named_mask_identity_is_order_independent_but_name_sensitive(tmp_path):
    stage = EnhancementStage(geometry_recipe=NORMALIZED_GEOMETRY_RECIPE)
    context = _context(tmp_path / "cache")
    bound = context.get_artifact("geometry_handoff")
    metal = np.full((14, 14), 0.5, np.float32)

    def with_masks(masks):
        updated = GeometryBoundInputs(
            bound.identity_sha256, bound.source_sha256, bound.v2_input_sha256, (14, 14), bound.depth_map, bound.support, masks
        ).with_image(bound.image)
        return StageContext(
            artifacts={
                **context.artifacts,
                "material_masks": dict(updated.masks),
                "geometry_handoff": updated,
            }
        )

    first = with_masks({"wood": bound.masks["wood"], "metal": metal})
    reordered = with_masks({"metal": metal, "wood": bound.masks["wood"]})
    renamed = with_masks({"glass": metal, "wood": bound.masks["wood"]})
    assert stage.get_cache_key(first) == stage.get_cache_key(reordered)
    assert stage.get_cache_key(first) != stage.get_cache_key(renamed)


def _material_order_context(cache_dir, *, reverse=False):
    image = np.full((14, 14, 3), 0.51, np.float32)
    depth = np.zeros((14, 14), np.float32)
    support = np.ones((14, 14), bool)
    wood = np.zeros((14, 14), np.float32)
    wood[:, :7] = 1
    metal = np.zeros((14, 14), np.float32)
    metal[:, 7:] = 1
    masks = {"metal": metal, "wood": wood} if reverse else {"wood": wood, "metal": metal}
    bound = GeometryBoundInputs("a" * 64, "b" * 64, "c" * 64, (14, 14), depth, support, masks).with_image(image)
    return StageContext(
        artifacts={
            "image": bound.image,
            "depth_map": bound.depth_map,
            "material_masks": dict(bound.masks),
            "geometry_handoff": bound,
        },
        cache_dir=cache_dir,
    )


def _material_order_stage(recipe=NORMALIZED_GEOMETRY_RECIPE):
    return EnhancementStage(
        enhancement_strength=0,
        clarity_strength=0,
        material_strength=1,
        geometry_recipe=recipe,
        version="1.3.0",
    )


def test_corrected_nonempty_material_order_matches_compute_and_shared_cache(tmp_path):
    first = _material_order_context(tmp_path / "cache")
    reordered = _material_order_context(tmp_path / "cache", reverse=True)
    stage = _material_order_stage()
    assert list(first.get_artifact("material_masks")) != list(reordered.get_artifact("material_masks"))
    expected = stage.compute(first).artifacts["enhanced_image"]
    np.testing.assert_array_equal(expected, stage.compute(reordered).artifacts["enhanced_image"])
    assert expected[0, 10, 0] == 130
    assert stage.get_cache_key(first) == stage.get_cache_key(reordered)

    initial = stage.execute(first)
    repeat = stage.execute(reordered)
    assert initial.status == StageStatus.COMPLETED and not initial.cache_hit
    assert repeat.status == StageStatus.CACHED and repeat.cache_hit
    np.testing.assert_array_equal(initial.artifacts["enhanced_image"], expected)
    np.testing.assert_array_equal(repeat.artifacts["enhanced_image"], expected)


def test_legacy_nonempty_material_execution_retains_insertion_order(tmp_path):
    first = _material_order_context(tmp_path / "cache")
    reordered = _material_order_context(tmp_path / "cache", reverse=True)
    stage = _material_order_stage(LEGACY_GEOMETRY_RECIPE)
    initial = stage.compute(first).artifacts["enhanced_image"]
    reverse = stage.compute(reordered).artifacts["enhanced_image"]
    assert np.count_nonzero(np.any(initial != reverse, axis=-1)) == 98
    assert initial[0, 10, 0] == 129
    assert reverse[0, 10, 0] == 130


def test_corrected_canonical_material_order_does_not_reuse_pre_fix_cache(tmp_path):
    context = _material_order_context(tmp_path / "cache")
    # Captured from f02610ca before execution order was canonicalized. Those
    # experimental cache bytes can contain the insertion-ordered result.
    old_key = "enhance_geometry_e6620d4e8e6315b1f082da9532e572fae3e0395d4fed159cff5dc0ff8a6acb4d"
    stage = _material_order_stage()
    stale = _material_order_stage(LEGACY_GEOMETRY_RECIPE).compute(context)
    assert stale.artifacts["enhanced_image"][0, 10, 0] == 129
    stage._save_to_cache(stale, old_key, context.cache_dir)
    assert (context.cache_dir / f"{old_key}.json").is_file()

    result = stage.execute(context)
    assert result.status == StageStatus.COMPLETED and not result.cache_hit
    assert result.cache_key != old_key
    assert result.artifacts["enhanced_image"][0, 10, 0] == 130
    np.testing.assert_array_equal(result.artifacts["enhanced_image"], stage.compute(context).artifacts["enhanced_image"])
    assert (context.cache_dir / f"{old_key}.json").is_file()


def test_legacy_cache_key_preserves_historical_bytes():
    image = np.full((14, 14, 3), 0.5, np.float32)
    depth = np.indices((14, 14))[1].astype(np.float32) / 13
    stage = EnhancementStage(version="1.2.0")
    context = StageContext(artifacts={"image": image, "depth_map": depth})
    assert stage.get_cache_key(context) == (
        "enhance_dtype_normalization_v2_float32_14x14x3_uint8_0.70_0.50_0.60_0.60_1_1.2.0_" "770e002624f39e42_a6f0741e_"
    )


@pytest.mark.parametrize("invalid", ["depth", "mask", "support_shape", "recipe"])
def test_invalid_corrected_inputs_fail_before_existing_cache_is_read(tmp_path, monkeypatch, invalid):
    stage = EnhancementStage(geometry_recipe=NORMALIZED_GEOMETRY_RECIPE)
    context = _context(tmp_path / "cache")
    assert stage.execute(context).status == StageStatus.COMPLETED
    if invalid == "depth":
        context.set_artifact("depth_map", np.zeros((14, 14), np.float32))
    elif invalid == "mask":
        context.set_artifact("material_masks", {"wood": np.ones((14, 14), np.float32)})
    elif invalid == "support_shape":
        context.get_artifact("geometry_handoff").support.shape = (7, 28)
    else:
        stage.geometry_recipe = ""

    def forbidden(*_args):
        pytest.fail("Invalid corrected authority reached cache retrieval")

    monkeypatch.setattr(stage, "_load_from_cache", forbidden)
    result = stage.execute(context)
    assert result.status == StageStatus.FAILED and not result.cache_hit
    assert result.error


def test_corrected_missing_image_cannot_read_legacy_no_image_cache(tmp_path, monkeypatch):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    (cache_dir / "no_image.json").write_text(
        json.dumps(
            {
                "stage_name": "enhancement",
                "stage_version": "1.2.0",
                "artifacts": {"enhanced_image": "legacy no-image sentinel"},
            }
        )
    )
    context = StageContext(cache_dir=cache_dir)
    legacy = EnhancementStage().execute(context)
    assert legacy.status == StageStatus.CACHED
    assert legacy.artifacts["enhanced_image"] == "legacy no-image sentinel"
    stage = EnhancementStage(geometry_recipe=NORMALIZED_GEOMETRY_RECIPE)

    def forbidden(*_args):
        pytest.fail("Corrected missing image reached legacy no_image cache")

    monkeypatch.setattr(stage, "_load_from_cache", forbidden)
    result = stage.execute(context)
    assert result.status == StageStatus.FAILED and not result.cache_hit
    assert "verified photographic image" in result.error


@pytest.mark.parametrize("operation", ["compute", "get_cache_key"])
def test_mutated_stage_recipe_cannot_enter_legacy_path(tmp_path, operation):
    stage = EnhancementStage(geometry_recipe=NORMALIZED_GEOMETRY_RECIPE)
    stage.geometry_recipe = "unknown"
    with pytest.raises(ValueError, match="Unsupported Lux geometry recipe"):
        getattr(stage, operation)(_context(tmp_path / "cache"))


def _script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "enhance_image.py"
    spec = importlib.util.spec_from_file_location("tp_v2_geometry_entrypoint_contract", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("recipe", ["unknown", "", None, 0, False, [], {}])
def test_programmatic_script_rejects_unknown_recipe_before_any_effect(tmp_path, monkeypatch, recipe):
    module = _script()
    output_dir = tmp_path / "must-not-exist"

    def forbidden(*_args, **_kwargs):
        pytest.fail("Unknown geometry recipe reached path validation, logging, discovery, or enhancement")

    for name in ("validate_input_path", "validate_depth_dir", "validate_output_dir", "find_depth_map", "enhance_image"):
        monkeypatch.setattr(module, name, forbidden)
    monkeypatch.setattr(module.logger, "info", forbidden)
    with pytest.raises(ValueError, match="Unsupported Lux geometry recipe"):
        module.run_v2_enhancement(
            input_path=tmp_path / "missing.png",
            depth_dir=None,
            output_dir=output_dir,
            preset="default",
            device="cpu",
            upscaler="default",
            geometry_recipe=recipe,
        )
    assert not output_dir.exists()


def test_programmatic_legacy_script_keeps_child_arguments_and_output_pixels(tmp_path, monkeypatch):
    module = _script()
    source = tmp_path / "source.png"
    Image.new("RGB", (14, 14), (77, 113, 151)).save(source)
    original_enhance = module.enhance_image
    calls = []

    def enhance(**kwargs):
        calls.append(kwargs)
        return original_enhance(**kwargs)

    monkeypatch.setattr(module, "enhance_image", enhance)
    common = dict(input_path=source, depth_dir=None, preset="default", device="cpu", upscaler="default", input_color="srgb")
    default = module.run_v2_enhancement(**common, output_dir=tmp_path / "default")
    explicit = module.run_v2_enhancement(**common, output_dir=tmp_path / "explicit", geometry_recipe=LEGACY_GEOMETRY_RECIPE)
    assert all("geometry_recipe" not in call and "geometry_handoff_path" not in call for call in calls)
    assert default["processing_version"] == explicit["processing_version"] == "1.2.0"
    with Image.open(default["output"]) as first, Image.open(explicit["output"]) as second:
        np.testing.assert_array_equal(np.asarray(first), np.asarray(second))
