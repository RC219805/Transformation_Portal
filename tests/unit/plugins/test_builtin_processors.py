"""Built-in ResizeProcessor target-size tests."""

from __future__ import annotations

import pytest
from PIL import Image

from transformation_portal.plugins.builtin.processors import ResizeProcessor

pytestmark = [pytest.mark.unit, pytest.mark.regression]


def _processor(**config) -> ResizeProcessor:
    processor = ResizeProcessor()
    processor.initialize(config)
    return processor


def test_resize_width_and_height_with_aspect_fits_within_bounds():
    """T-22: both bounds with maintain_aspect must fit within the box, not follow width alone."""
    processor = _processor(width=500, height=500, maintain_aspect=True)

    assert processor._calculate_target_size((1000, 4000), 500, 500, None) == (125, 500)
    assert processor._calculate_target_size((4000, 1000), 500, 500, None) == (500, 125)


def test_resize_process_honours_fit_within_bounds():
    processor = _processor(width=50, height=50)

    resized = processor.process(Image.new("RGB", (100, 400)))

    assert resized.size == (12, 50)


def test_resize_single_dimension_without_aspect_is_not_ignored():
    """T-22: a single requested dimension with maintain_aspect=False must still be applied."""
    width_only = _processor(width=500, maintain_aspect=False)
    height_only = _processor(height=300, maintain_aspect=False)

    assert width_only._calculate_target_size((1000, 4000), 500, None, None) == (500, 4000)
    assert height_only._calculate_target_size((1000, 4000), None, 300, None) == (1000, 300)


def test_resize_existing_branches_unchanged():
    exact = _processor(width=500, height=200, maintain_aspect=False)
    width_aspect = _processor(width=500)
    height_aspect = _processor(height=1000)
    scaled = _processor(scale=0.5)
    untouched = _processor()

    assert exact._calculate_target_size((1000, 4000), 500, 200, None) == (500, 200)
    assert width_aspect._calculate_target_size((1000, 4000), 500, None, None) == (500, 2000)
    assert height_aspect._calculate_target_size((1000, 4000), None, 1000, None) == (250, 1000)
    assert scaled._calculate_target_size((1000, 4000), None, None, 0.5) == (500, 2000)
    assert untouched._calculate_target_size((1000, 4000), None, None, None) == (1000, 4000)
