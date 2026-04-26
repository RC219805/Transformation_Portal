"""Tests for APEX image metadata and 16-bit I/O helpers."""

from __future__ import annotations

import numpy as np
import pytest

# Skip the entire module when tifffile isn't installed; every test here exercises
# 16-bit TIFF I/O directly.
pytest.importorskip("tifffile")

import tifffile
from PIL import Image

from transformation_portal.evals.image_io import (
    ARTIFACT_DELIVERY_8BIT,
    ARTIFACT_MODEL_INPUT,
    ARTIFACT_WORKING_16,
    derive_model_input_metadata,
    load_16bit_tiff,
    write_16bit_master,
    write_delivery_srgb8,
)
from transformation_portal.evals.image_metadata import inspect_reference_image

pytestmark = pytest.mark.unit


def test_16bit_tiff_metadata_detection_passes(tmp_path):
    path = tmp_path / "reference16.tif"
    Image.fromarray(np.zeros((8, 8), dtype=np.uint16), mode="I;16").save(path)

    metadata = inspect_reference_image(path)

    assert metadata["observable_reference_metadata_status"] == "ok"
    assert metadata["detected_reference_format"] == "tiff"
    assert metadata["detected_reference_bit_depth"] == 16
    assert metadata["detected_reference_dimensions"] == [8, 8]


def test_8bit_tiff_claiming_16bit_is_unsupported_for_16bit_load(tmp_path):
    path = tmp_path / "reference8.tif"
    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(path, format="TIFF")

    array, metadata = load_16bit_tiff(path)

    assert array is None
    assert metadata["status"] == "unsupported_bit_depth"
    assert metadata["reason"] == "reference_bit_depth_below_16"
    assert metadata["detected_reference_bit_depth"] == 8


def test_jpeg_remains_noncanonical_for_16bit_load(tmp_path):
    path = tmp_path / "delivery.jpg"
    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(path, format="JPEG")

    array, metadata = load_16bit_tiff(path)

    assert array is None
    assert metadata["status"] == "invalid_input"
    assert metadata["reason"] == "non_tiff_reference"


def test_model_input_metadata_is_distinct_from_reference_metadata(tmp_path):
    path = tmp_path / "reference16.tif"
    Image.fromarray(np.zeros((12, 16), dtype=np.uint16), mode="I;16").save(path)
    reference_metadata = inspect_reference_image(path)

    model_input = derive_model_input_metadata(reference_metadata, downsampled_for_inference=True, input_dimensions=[8, 6])

    assert model_input["artifact_role"] == ARTIFACT_MODEL_INPUT
    assert model_input["derived_from_role"] == "reference_16bit"
    assert model_input["input_bit_depth"] == 8
    assert model_input["input_dimensions"] == [8, 6]
    assert model_input["reference_dimensions"] == [16, 12]


def test_working_16_and_delivery8_write_helpers(tmp_path):
    arr = np.zeros((8, 8), dtype=np.uint16)
    arr[2:6, 2:6] = 32768
    master_path = tmp_path / "out" / "master16.tif"
    delivery_path = tmp_path / "out" / "delivery.jpg"

    master = write_16bit_master(arr, master_path)
    delivery = write_delivery_srgb8(arr, delivery_path)

    assert master["artifact_role"] == ARTIFACT_WORKING_16
    assert master["bit_depth"] == 16
    assert master_path.is_file()
    assert delivery["artifact_role"] == ARTIFACT_DELIVERY_8BIT
    assert delivery["bit_depth"] == 8
    assert delivery_path.is_file()


def test_16bit_rgb_tiff_round_trip_preserves_uint16_precision(tmp_path):
    arr = np.zeros((8, 8, 3), dtype=np.uint16)
    arr[..., 0] = 1024
    arr[..., 1] = 32768
    arr[..., 2] = 65535
    path = tmp_path / "rgb_master16.tif"

    master = write_16bit_master(arr, path)
    loaded, metadata = load_16bit_tiff(path)

    assert master["bit_depth"] == 16
    assert metadata["status"] == "ok"
    assert loaded is not None
    assert loaded.dtype == np.uint16
    assert loaded.shape == arr.shape
    np.testing.assert_array_equal(loaded, arr)
    np.testing.assert_array_equal(tifffile.imread(path), arr)


def test_write_16bit_master_rejects_8bit_input(tmp_path):
    with pytest.raises(ValueError, match="16-bit master"):
        write_16bit_master(np.zeros((8, 8), dtype=np.uint8), tmp_path / "master.tif")
