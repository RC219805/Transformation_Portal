import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

import Transformation_Portal as pll

# ---- Fixtures ----
@pytest.fixture
def fake_dirs(tmp_path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()
    return input_dir, output_dir

# ---- 1. CLI vector generation tests ----
@pytest.mark.parametrize("recursive,overwrite,dry_run", [(True, True, True), (False, False, False)])
def test_build_cli_vector_flags(recursive, overwrite, dry_run, fake_dirs):
    from picacho_lane_luts.golden_hour_courtyard import _build_cli_vector, _ADJUSTMENT_FLAGS

    input_dir, output_dir = fake_dirs
    cli_vector = _build_cli_vector(
        input_dir,
        output_dir,
        recursive=recursive,
        overwrite=overwrite,
        dry_run=dry_run,
        suffix="_lux",
        compression="tiff_lzw",
        resize_long_edge=None,
        log_level="INFO",
        overrides=None,
    )

    # Base paths present
    assert str(input_dir) in cli_vector
    assert str(output_dir) in cli_vector

    # Preset always included
    assert "--preset" in cli_vector
    assert "golden_hour_courtyard" in cli_vector

    # Flags
    if recursive:
        assert "--recursive" in cli_vector
    if overwrite:
        assert "--overwrite" in cli_vector
    if dry_run:
        assert "--dry-run" in cli_vector

    # Default overrides included
    for flag in [entry.flag for entry in _ADJUSTMENT_FLAGS]:
        assert flag in cli_vector

# ---- 2. Overrides merging ----
def test_merge_overrides_removal_and_override():
    from picacho_lane_luts.golden_hour_courtyard import _merge_overrides

    overrides = {"vibrance": 0.5, "clarity": None}
    merged = _merge_overrides(overrides)
    # Confirm vibrance overridden
    assert merged["vibrance"] == 0.5
    # Confirm clarity removed
    assert "clarity" not in merged

# ---- 3. Integration test for process_courtyard_scene ----
def test_process_courtyard_scene(fake_dirs, monkeypatch):
    input_dir, output_dir = fake_dirs

    # Mock luxury_tiff_batch_processor
    fake_ltiff = MagicMock()
    fake_ltiff.parse_args.return_value = ["parsed_args"]
    fake_ltiff.run_pipeline.return_value = 42
    fake_ltiff.ProcessingCapabilities.return_value.assert_luxury_grade = lambda: None

    monkeypatch.setitem(sys.modules, "picacho_lane_luts.luxury_tiff_batch_processor", fake_ltiff)

    result = pll.process_courtyard_scene(
        input_dir, output_dir,
        recursive=True, overwrite=True, dry_run=True,
        suffix="_custom", compression="tiff_deflate",
        resize_long_edge=2048, log_level="DEBUG",
        overrides={"vibrance": 0.5, "glow": None}
    )

    # Validate mocked run_pipeline is called with parsed args
    fake_ltiff.parse_args.assert_called_once()
    fake_ltiff.run_pipeline.assert_called_once()
    assert result == 42

# ---- 4. Resize, compression, suffix in CLI vector ----
def test_cli_vector_custom_parameters(fake_dirs):
    from picacho_lane_luts.golden_hour_courtyard import _build_cli_vector

    input_dir, output_dir = fake_dirs
    cli_vector = _build_cli_vector(
        input_dir, output_dir,
        recursive=False,
        overwrite=False,
        dry_run=False,
        suffix="_test",
        compression="tiff_deflate",
        resize_long_edge=1024,
        log_level="WARNING",
        overrides=None
    )

    assert "--suffix" in cli_vector
    assert "_test" in cli_vector
    assert "--compression" in cli_vector
    assert "tiff_deflate" in cli_vector
    assert "--resize-long-edge" in cli_vector
    assert "1024" in cli_vector
    assert "--log-level" in cli_vector
    assert "WARNING" in cli_vector

# ---- 5. Exception handling for invalid overrides ----
def test_merge_overrides_invalid_key_raises():
    from picacho_lane_luts.golden_hour_courtyard import _merge_overrides
    with pytest.raises(ValueError):
        _merge_overrides({"invalid_attribute": 1.0})

# ---- 6. End-to-end dry-run vector sanity ----
def test_end_to_end_vector_sanity(fake_dirs, monkeypatch):
    input_dir, output_dir = fake_dirs
    fake_ltiff = MagicMock()
    fake_ltiff.parse_args.side_effect = lambda vec: vec
    fake_ltiff.run_pipeline.side_effect = lambda vec: vec
    fake_ltiff.ProcessingCapabilities.return_value.assert_luxury_grade = lambda: None
    monkeypatch.setitem(sys.modules, "picacho_lane_luts.luxury_tiff_batch_processor", fake_ltiff)

    result_vector = pll.process_courtyard_scene(
        input_dir, output_dir, dry_run=True
    )

    # Dry-run: returned vector contains input path
    assert str(input_dir) in result_vector
    # Preset included
    assert "golden_hour_courtyard" in result_vector