"""Regression coverage for core config preset behavior."""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path

import pytest
from pydantic import ValidationError

# Pytest markers
pytestmark = [
    pytest.mark.unit,
]

from transformation_portal.core.config.presets import Preset, PresetRegistry, list_presets, load_preset, register_preset
from transformation_portal.core.config.schemas import PerformanceConfig
from transformation_portal.core.config.validation import validate_config


def _base_paths(tmp_path: Path) -> dict[str, dict[str, Path]]:
    return {
        "paths": {
            "input_dir": tmp_path,
            "output_dir": tmp_path / "output",
        }
    }


@pytest.fixture(autouse=True)
def _restore_preset_registry() -> None:
    original_presets = PresetRegistry._presets.copy()
    yield
    PresetRegistry._presets = original_presets


def test_fast_preview_validates_against_config_schema(tmp_path: Path) -> None:
    config = {
        **_base_paths(tmp_path),
        **load_preset("fast_preview"),
    }

    validated = validate_config(config)

    assert validated.performance.tile_size == 0
    assert validated.paths.output_dir.exists()


def test_nonzero_tile_size_boundary_matches_schema() -> None:
    config = PerformanceConfig(tile_size=256)

    assert config.tile_size == 256


def test_load_preset_returns_deep_copy_for_nested_overrides() -> None:
    first = load_preset("production")
    second = load_preset("production")

    first["output"]["quality"] = 1

    reloaded = load_preset("production")

    assert first["output"] is not second["output"]
    assert second["output"]["quality"] == 92
    assert reloaded["output"]["quality"] == 92


def test_load_preset_unknown_name_raises_keyerror() -> None:
    with pytest.raises(KeyError, match="Preset 'does_not_exist' not found"):
        load_preset("does_not_exist")


def test_tile_size_zero_is_allowed_and_sub_256_nonzero_values_are_rejected() -> None:
    assert PerformanceConfig(tile_size=0).tile_size == 0

    for invalid_tile_size in (1, 255):
        with pytest.raises(ValidationError, match="tile_size"):
            PerformanceConfig(tile_size=invalid_tile_size)


def test_register_and_load_custom_preset_without_parent_semantics() -> None:
    register_preset(
        Preset(
            name="custom_preview",
            description="Custom preset used to verify registration and loading.",
            overrides={
                "performance": {"batch_size": 2, "tile_size": 512},
                "output": {"format": "png", "quality": 100},
            },
        )
    )

    loaded = load_preset("custom_preview")

    assert loaded["performance"]["batch_size"] == 2
    assert loaded["output"]["format"] == "png"
    assert "parent" not in {field.name for field in fields(Preset)}


def test_list_presets_contains_builtin_presets() -> None:
    preset_names = list_presets()

    assert {"fast_preview", "production", "archival"}.issubset(set(preset_names))
