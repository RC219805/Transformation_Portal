"""Regression coverage for core config preset behavior."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from transformation_portal.core.config.presets import load_preset
from transformation_portal.core.config.schemas import PerformanceConfig
from transformation_portal.core.config.validation import validate_config


def _base_paths(tmp_path: Path) -> dict[str, dict[str, Path]]:
    return {
        "paths": {
            "input_dir": tmp_path,
            "output_dir": tmp_path / "output",
        }
    }


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
