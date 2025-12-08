import pytest

from lux_depth_v2.validation.benchmark.dataset_registry import (
    BASELINE_IDS,
    baseline_dir,
    get_dataset_spec,
    list_input_images,
)


def test_validation_v1_dirs_exist():
    """Test that all expected directories and files exist for validation_v1."""
    spec = get_dataset_spec("validation_v1")

    assert spec.root_dir.exists()
    assert spec.input_dir.exists()
    assert spec.metadata_file.exists()

    assert spec.baselines_dir.exists()
    for _, p in spec.baselines.items():
        assert p.exists()


def test_list_input_images_returns_list():
    """Test list_input_images returns a list (empty or populated)."""
    images = list_input_images("validation_v1")
    # Directory exists, may be empty or contain images
    assert isinstance(images, list)


def test_list_input_images_filters_by_extension():
    """Test that list_input_images only returns TIFF files."""
    images = list_input_images("validation_v1")
    # If images exist, all should have .tif or .tiff extension
    if len(images) == 0:
        pytest.skip("No test images available to verify filtering behavior")
    for img in images:
        assert img.suffix.lower() in [".tif", ".tiff"]


def test_baseline_dir_valid_ids():
    """Test baseline_dir returns correct path for all valid baseline_ids."""
    for baseline_id in BASELINE_IDS:
        path = baseline_dir("validation_v1", baseline_id)
        assert path.exists()
        assert path.name == baseline_id
        assert "baselines" in str(path)


def test_baseline_dir_invalid_id_raises_error():
    """Test that baseline_dir raises KeyError for invalid baseline_id."""
    with pytest.raises(KeyError, match="Unknown baseline_id"):
        baseline_dir("validation_v1", "invalid_baseline")


def test_baseline_dir_empty_string_raises_error():
    """Test that baseline_dir raises KeyError for empty string baseline_id."""
    with pytest.raises(KeyError, match="Unknown baseline_id"):
        baseline_dir("validation_v1", "")


def test_get_dataset_spec_structure():
    """Test that get_dataset_spec returns properly structured DatasetSpec."""
    spec = get_dataset_spec("validation_v1")

    assert spec.dataset_id == "validation_v1"
    assert spec.root_dir.name == "validation_v1"
    assert spec.input_dir.name == "input"
    assert spec.metadata_file.name == "metadata.json"
    assert spec.baselines_dir.name == "baselines"
    assert len(spec.baselines) == len(BASELINE_IDS)
    assert set(spec.baselines.keys()) == set(BASELINE_IDS)
