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


def test_resize_fit_within_meets_the_binding_bound_exactly():
    """T-22: float truncation must not shave a pixel off the bound that limits the fit."""
    processor = _processor(width=1, height=1)

    # Width binds: identical to what the width-only branch always returned.
    assert processor._calculate_target_size((5545, 1793), 3117, 1883, None) == (3117, 1007)
    # Height binds.
    assert processor._calculate_target_size((311, 4839), 3357, 892, None) == (57, 892)
    # The full width is kept when its proportional height (546.97) rounds down into the box.
    assert processor._calculate_target_size((4196, 1226), 1872, 546, None) == (1872, 546)
    # Square into a square box hits both bounds; 49 * (1 / 49) truncates to 0 in floats.
    assert processor._calculate_target_size((49, 49), 1, 1, None) == (1, 1)
    assert processor._calculate_target_size((19, 19), 500, 500, None) == (500, 500)


def test_resize_fit_within_invariants_hold_over_a_size_grid():
    """Fit-within stays inside the box, touches one bound and keeps the aspect to a pixel."""
    processor = _processor(width=1, height=1)
    sources = (1, 2, 3, 7, 11, 19, 49, 100, 311, 1080, 1793, 1920, 4839, 5545)
    bounds = (1, 2, 3, 5, 13, 64, 500, 892, 1024, 1883, 3117, 3840)

    for orig_w in sources:
        for orig_h in sources:
            for width in bounds:
                for height in bounds:
                    got_w, got_h = processor._calculate_target_size((orig_w, orig_h), width, height, None)
                    case = ((orig_w, orig_h), (width, height), (got_w, got_h))

                    assert got_w <= width and got_h <= height, case
                    width_binds = got_w == width and 0 <= orig_h * got_w - got_h * orig_w < orig_w
                    height_binds = got_h == height and 0 <= orig_w * got_h - got_w * orig_h < orig_h
                    assert width_binds or height_binds, case


def test_resize_single_dimension_keeps_exactly_proportional_results():
    """An exactly integral proportional size must not lose a pixel to float rounding."""
    width_aspect = _processor(width=500)
    height_aspect = _processor(height=500)

    assert width_aspect._calculate_target_size((19, 19), 500, None, None) == (500, 500)
    assert width_aspect._calculate_target_size((6108, 6108), 6455, None, None) == (6455, 6455)
    assert height_aspect._calculate_target_size((19, 19), None, 500, None) == (500, 500)
    # Non-integral proportions still round down, as before.
    assert width_aspect._calculate_target_size((1000, 333), 500, None, None) == (500, 166)
    assert height_aspect._calculate_target_size((333, 1000), None, 500, None) == (166, 500)


def test_resize_process_fits_square_image_into_square_box():
    processor = _processor(width=500, height=500)

    resized = processor.process(Image.new("RGB", (19, 19)))

    assert resized.size == (500, 500)


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
