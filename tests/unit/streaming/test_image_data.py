"""Unit coverage for streaming ImageData contracts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from transformation_portal.streaming.stages import ImageData

pytestmark = pytest.mark.unit


def test_image_data_shape_and_dtype_for_valid_array(tmp_path: Path) -> None:
    array = np.zeros((8, 6, 3), dtype=np.uint8)
    image_data = ImageData(array=array, path=tmp_path / "frame.png")

    assert image_data.shape == (8, 6, 3)
    assert image_data.dtype == np.uint8
    assert image_data.depth_map is None
    assert image_data.metadata == {}


def test_image_data_shape_and_dtype_for_none_array(tmp_path: Path) -> None:
    image_data = ImageData(
        array=None,
        path=tmp_path / "missing.png",
        metadata={"source": "fixture"},
    )

    assert image_data.shape == ()
    assert image_data.dtype is None
    assert image_data.metadata == {"source": "fixture"}
