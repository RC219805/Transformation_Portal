"""Tests for :mod:`material_response_optimizer`."""
# pylint: disable=redefined-outer-name  # pytest fixtures

from __future__ import annotations

from pathlib import Path

import pytest

from material_response_optimizer import MaterialAwareEnhancementPlanner, RenderEnhancementPlanner


@pytest.fixture(scope="module")
def blueprint() -> dict:
    planner = RenderEnhancementPlanner.from_json(Path("material_response_report.json"))
    return planner.build_blueprint()


@pytest.fixture(scope="module")
def material_blueprint() -> dict:
    planner = MaterialAwareEnhancementPlanner.from_json(Path("material_response_report.json"))
    return planner.build_blueprint()


def test_pool_requires_targeted_luminance(blueprint: dict) -> None:
    pool_entry = next(
        target for target in blueprint["luminance_strategy"]["targets"] if target["scene"] == "pool"
    )
    assert pytest.approx(pool_entry["current"], rel=1e-3) == 0.23
    assert pool_entry["target"] <= 0.32
    assert "specular_pool_reflections" in pool_entry["focus_areas"]
    assert "architectural_whites" in pool_entry["focus_areas"]


def test_awe_alignment_sets_explicit_targets(blueprint: dict) -> None:
    actions = {action["scene"]: action for action in blueprint["awe_alignment"]["actions"]}
    assert actions["great_room"]["target"] == 0.85
    assert actions["pool"]["target"] == 0.75


def test_comfort_reduction_defined_for_primary_suite(blueprint: dict) -> None:
    comfort = blueprint["comfort_realignment"]
    assert comfort["scene"] == "primary_bedroom"
    assert comfort["target"] == 0.85
    assert any("shadow" in move for move in comfort["moves"])


def test_hero_surface_texture_targets_present(blueprint: dict) -> None:
    hero_targets = blueprint["texture_dimension_strategy"]["hero_targets"]
    surfaces = {entry["surface"] for entry in hero_targets}
    assert {"island_waterfall_edge", "stone_feature_wall", "headboard_textile_panel"} <= surfaces
    assert all(entry["target"] == 2.25 for entry in hero_targets)


def test_lux_strategy_detects_low_deltas(blueprint: dict) -> None:
    lux_entries = blueprint["lux_version_strategy"]["remedy"]
    assert any(entry["scene"] == "aerial" for entry in lux_entries)
    aerial_entry = next(entry for entry in lux_entries if entry["scene"] == "aerial")
    assert aerial_entry["delta"] == pytest.approx(0.011, rel=1e-3)
    assert "golden hour" in " ".join(aerial_entry["actions"])


def test_scene_specific_targets_raise_luxury_indices(blueprint: dict) -> None:
    aerial_plan = blueprint["scene_specific_enhancements"]["aerial"]
    assert aerial_plan["target"] == pytest.approx(0.7, rel=1e-3)
    assert any("coastline" in move for move in aerial_plan["moves"])


def test_material_integration_targets_material_specifics(material_blueprint: dict) -> None:
    wood_strategy = material_blueprint["material_integration"]["great_room"]
    assert wood_strategy["material"] == "herringbone_oak"
    assert wood_strategy["target_texture_dimension"] == pytest.approx(2.25)
    assert wood_strategy["rendering_params"]["mapping"]["type"] == "uv"
    assert "displacement" in wood_strategy["rendering_params"]


def test_exposure_zones_follow_luminance(material_blueprint: dict) -> None:
    zones = {zone["scene"]: zone for zone in material_blueprint["exposure_zones"]["zones"]}
    pool_zone = zones["pool"]
    assert pool_zone["ev_adjustment"] == pytest.approx(0.43, abs=1e-2)
    assert any(adj["area"] == "water_surface" for adj in pool_zone["local_adjustments"])
    assert material_blueprint["exposure_zones"]["global_reference"] == pytest.approx(0.31)


def test_shader_settings_include_procedural_variation(material_blueprint: dict) -> None:
    stone_settings = material_blueprint["shader_settings"]["stone"]["procedural_variation"]
    assert stone_settings["count"] == 12
    assert stone_settings["mortar_depth"] == pytest.approx(3.0)
