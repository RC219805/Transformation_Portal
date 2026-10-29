"""Successor geometry is explicit without changing legacy plan/hash contracts."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import jsonschema
import pytest
from PIL import Image

from transformation_portal.core.execution_plan import (
    CanonicalExecutionPlan,
    ExecutionPlanError,
    with_execution_plan_fingerprint,
)
from transformation_portal.lux_depth_v3.config import EnhanceConfig
from transformation_portal.lux_depth_v3.config_resolver import (
    build_depth_cache_payload,
    build_materials_fingerprint_payload,
    build_run_card_config_fingerprint,
    compute_config_fingerprint,
)
from transformation_portal.lux_depth_v3.execution_lifecycle import (
    prepare_lux_execution,
    runtime_config_from_execution_plan,
)
from transformation_portal.lux_depth_v3.geometry import LEGACY_GEOMETRY_RECIPE, NORMALIZED_GEOMETRY_RECIPE
from transformation_portal.lux_depth_v3.manifest import CombinedManifest, ConfigFingerprint
from transformation_portal.lux_depth_v3.v2_presets import processing_version_for_geometry

pytestmark = pytest.mark.unit


@pytest.fixture
def prepared_pair(tmp_path):
    root = tmp_path / "inputs"
    root.mkdir()
    source = root / "scene.jpg"
    Image.new("RGB", (45, 31), (70, 90, 110)).save(source)
    config = EnhanceConfig(
        model_key="da3-metric",
        enable_materials_v3=True,
        enable_v2=True,
        enable_parallel_processing=False,
        max_workers=1,
        max_gpu_workers=0,
    )
    return (
        prepare_lux_execution(config, root, [source]),
        prepare_lux_execution(replace(config, geometry_recipe=NORMALIZED_GEOMETRY_RECIPE), root, [source]),
    )


def test_recipe_default_and_processing_versions_are_separate():
    assert EnhanceConfig().geometry_recipe == LEGACY_GEOMETRY_RECIPE
    assert processing_version_for_geometry() == "1.2.0"
    assert processing_version_for_geometry(LEGACY_GEOMETRY_RECIPE) == "1.2.0"
    assert processing_version_for_geometry(NORMALIZED_GEOMETRY_RECIPE) == "1.3.0"


def test_geometry_selector_does_not_shift_existing_positional_configuration():
    config = EnhanceConfig(None, None, None, None, None, "cpu", "none", "architectural", "srgb")
    assert config.v2_preset == "architectural"
    assert config.v2_input_color == "srgb"
    assert config.geometry_recipe == LEGACY_GEOMETRY_RECIPE


@pytest.mark.parametrize("value", [None, True, "", "normalized", "normalized_crop_support_v2"])
def test_unknown_recipe_fails_configuration_and_processing_resolution(value):
    with pytest.raises(ValueError):
        EnhanceConfig(geometry_recipe=value)
    with pytest.raises(ValueError):
        processing_version_for_geometry(value)


@pytest.mark.parametrize("flag", ["generate_pbr", "enable_reconstruction", "spatial_ai_linear_ingest"])
def test_unsupported_corrected_modes_fail_before_preparation(flag):
    with pytest.raises(ValueError, match=flag):
        EnhanceConfig(geometry_recipe=NORMALIZED_GEOMETRY_RECIPE, **{flag: True})
    assert getattr(EnhanceConfig(**{flag: True}), flag)


@pytest.mark.parametrize("flag", ["generate_pbr", "enable_reconstruction", "spatial_ai_linear_ingest"])
def test_mutated_corrected_modes_fail_before_runtime_selection(tmp_path, monkeypatch, flag):
    from transformation_portal.lux_depth_v3 import execution_lifecycle

    config = EnhanceConfig(model_key="da3-metric", geometry_recipe=NORMALIZED_GEOMETRY_RECIPE)
    setattr(config, flag, True)
    source = tmp_path / "source.jpg"
    source.write_bytes(b"metadata-only unsupported-mode fixture")

    def forbidden(*_args, **_kwargs):
        pytest.fail("Unsupported geometry reached runtime preparation")

    monkeypatch.setattr(execution_lifecycle, "resolve_prepared_da3_python_executable", forbidden)
    with pytest.raises(ExecutionPlanError, match=flag):
        prepare_lux_execution(config, tmp_path, [source])


@pytest.mark.parametrize("extension", [".dng", ".CR2", ".arw"])
def test_corrected_raw_preparation_rejects_before_runtime_path_resolution(tmp_path, monkeypatch, extension):
    from transformation_portal.lux_depth_v3 import execution_lifecycle

    config = EnhanceConfig(model_key="da3-metric", geometry_recipe=NORMALIZED_GEOMETRY_RECIPE)
    source = tmp_path / f"unmapped{extension}"
    original = b"RAW fixture; no runtime is authorized to inspect its pixels"
    source.write_bytes(original)

    def forbidden(*_args, **_kwargs):
        pytest.fail("Unmapped corrected RAW reached runtime path resolution")

    for name in (
        "resolve_prepared_da3_python_executable",
        "resolve_prepared_raw_python_executable",
        "resolve_prepared_depth_pro_python_executable",
        "resolve_prepared_depth_pro_checkpoint_path",
    ):
        monkeypatch.setattr(execution_lifecycle, name, forbidden)
    with pytest.raises(ExecutionPlanError, match="RAW original-raster mapping"):
        prepare_lux_execution(config, tmp_path, [source])
    assert source.read_bytes() == original


def test_frozen_legacy_fingerprint_hash_and_omission_are_unchanged():
    # Recorded from main before this migration; independent of the new helper.
    fingerprint = ConfigFingerprint("fixed", "none", "cpu")
    assert fingerprint.to_sha256() == "27dfec8dd534995d83accde01c78cc6006dbb2a92a6fe3a5365151a5efdd0dab"
    assert "geometry_recipe" not in fingerprint.to_payload()
    explicit_legacy = replace(fingerprint, geometry_recipe=LEGACY_GEOMETRY_RECIPE)
    assert explicit_legacy.to_payload() == fingerprint.to_payload()
    assert explicit_legacy.to_sha256() == fingerprint.to_sha256()


def test_corrected_recipe_separates_all_fingerprint_and_cache_projections():
    legacy = EnhanceConfig(model_key="da3-metric", enable_materials_v3=True)
    corrected = replace(legacy, geometry_recipe=NORMALIZED_GEOMETRY_RECIPE)
    old, new = compute_config_fingerprint(legacy), compute_config_fingerprint(corrected)
    for projection in [lambda value: value, lambda value: value.depth_only(), lambda value: value.v2_only()]:
        assert projection(old).to_sha256() != projection(new).to_sha256()
        assert "geometry_recipe" not in projection(old).to_payload()
        assert projection(new).to_payload()["geometry_recipe"] == NORMALIZED_GEOMETRY_RECIPE
    for project in [build_depth_cache_payload, build_materials_fingerprint_payload, build_run_card_config_fingerprint]:
        assert "geometry_recipe" not in project(legacy)
        assert project(corrected)["geometry_recipe"] == NORMALIZED_GEOMETRY_RECIPE
    assert new.v2_processing_version == "1.3.0"
    assert build_materials_fingerprint_payload(corrected)["processing_version"] == "1.3.0"


@pytest.mark.parametrize("suffix", ["json", "msgpack"])
@pytest.mark.parametrize("recipe", [LEGACY_GEOMETRY_RECIPE, NORMALIZED_GEOMETRY_RECIPE])
def test_saved_fingerprint_preserves_legacy_omission_and_corrected_identity(tmp_path, recipe, suffix):
    if suffix == "msgpack":
        pytest.importorskip("msgpack")
    config = EnhanceConfig(geometry_recipe=recipe)
    fingerprint = compute_config_fingerprint(config)
    manifest = CombinedManifest(config_fingerprint=fingerprint)
    path = tmp_path / f"manifest.{suffix}"
    if suffix == "json":
        manifest.save(path)
        stored = json.loads(path.read_bytes())["config_fingerprint"]
        loaded = CombinedManifest.load(path)
    else:
        import msgpack

        manifest.save_msgpack(path)
        stored = msgpack.unpackb(path.read_bytes(), raw=False)["config_fingerprint"]
        loaded = CombinedManifest.load_msgpack(path)
    assert ("geometry_recipe" in stored) is (recipe == NORMALIZED_GEOMETRY_RECIPE)
    assert loaded.config_fingerprint.to_payload() == fingerprint.to_payload()
    assert loaded.config_fingerprint.to_sha256() == fingerprint.to_sha256()


@pytest.mark.parametrize(
    "recipe,version",
    [(NORMALIZED_GEOMETRY_RECIPE, "1.2.0"), (None, "1.3.0"), (LEGACY_GEOMETRY_RECIPE, "1.3.0")],
)
def test_fingerprint_rejects_mixed_recipe_processing_pair(recipe, version):
    with pytest.raises(ValueError):
        ConfigFingerprint("fixed", "none", "cpu", geometry_recipe=recipe, v2_processing_version=version)


def test_corrected_plan_carries_coherent_recipe_and_runtime_projection(prepared_pair):
    legacy, corrected = prepared_pair
    relevant = {"lux.preprocess", "lux.depth", "lux.materials_v3", "lux.v2"}
    for prepared, recipe in [(legacy, None), (corrected, NORMALIZED_GEOMETRY_RECIPE)]:
        configurations = {node.node_id: dict(node.configuration) for node in prepared.plan.nodes}
        assert relevant <= configurations.keys()
        for name in relevant:
            assert configurations[name].get("geometry_recipe") == recipe
        version = "1.2.0" if recipe is None else "1.3.0"
        assert configurations["lux.materials_v3"]["processing_version"] == version
        assert configurations["lux.v2"]["processing_version"] == version
        runtime = runtime_config_from_execution_plan(prepared.plan)
        assert runtime.geometry_recipe == (recipe or LEGACY_GEOMETRY_RECIPE)
        assert runtime.execution_plan_canonical_bytes == prepared.canonical_plan_bytes


def test_carried_corrected_raw_plan_is_rejected_without_runtime_availability(prepared_pair):
    payload = prepared_pair[1].plan.to_payload()
    payload["input_selection"]["files"][0]["path"] = "unmapped.DNG"
    with pytest.raises(ExecutionPlanError, match="RAW original-raster mapping"):
        CanonicalExecutionPlan.from_payload(with_execution_plan_fingerprint(payload))


@pytest.mark.parametrize("node_id", ["lux.preprocess", "lux.depth", "lux.materials_v3", "lux.v2"])
def test_mixed_stage_geometry_is_rejected_even_with_fresh_plan_fingerprint(prepared_pair, node_id):
    payload = prepared_pair[1].plan.to_payload()
    node = next(node for node in payload["nodes"] if node["id"] == node_id)
    node["configuration"].pop("geometry_recipe")
    if node_id in {"lux.materials_v3", "lux.v2"}:
        node["configuration"]["processing_version"] = "1.2.0"
    with pytest.raises(ExecutionPlanError, match="coherent"):
        CanonicalExecutionPlan.from_payload(with_execution_plan_fingerprint(payload))


@pytest.mark.parametrize("node_id", ["lux.materials_v3", "lux.v2"])
def test_plan_schema_rejects_recipe_version_mismatch(prepared_pair, node_id):
    payload = prepared_pair[1].plan.to_payload()
    node = next(node for node in payload["nodes"] if node["id"] == node_id)
    node["configuration"]["processing_version"] = "1.2.0"
    with pytest.raises(ExecutionPlanError):
        CanonicalExecutionPlan.from_payload(with_execution_plan_fingerprint(payload))


@pytest.mark.parametrize("schema_name", ["run_card.v1.schema.json", "run_card.v2.schema.json"])
def test_run_card_schema_requires_corrected_recipe_with_processing_version(schema_name):
    root = Path(__file__).resolve().parents[2]
    schema = json.loads((root / "src/transformation_portal/schemas/run_card" / schema_name).read_bytes())["properties"][
        "config_fingerprint"
    ]
    backend = SimpleNamespace(requested_backend="da3", resolved_backend="da3", device="cpu")
    legacy = build_run_card_config_fingerprint(EnhanceConfig(), backend_metadata=backend)
    corrected = build_run_card_config_fingerprint(
        EnhanceConfig(geometry_recipe=NORMALIZED_GEOMETRY_RECIPE), backend_metadata=backend
    )
    jsonschema.validate(legacy, schema)
    jsonschema.validate(corrected, schema)
    corrected.pop("geometry_recipe")
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(corrected, schema)


def test_cli_advertises_explicit_geometry_recipe_without_changing_default():
    from typer.testing import CliRunner

    from transformation_portal.lux_depth_v3.__main__ import app

    result = CliRunner().invoke(app, ["--help"], color=False)
    assert result.exit_code == 0
    assert "--geometry-recipe" in result.output
