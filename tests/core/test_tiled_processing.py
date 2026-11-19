"""Regression tests for tiled processing's batch and image-boundary contracts."""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.ml]

torch = pytest.importorskip("torch")


@pytest.fixture(name="tiling")
def tiling_module():
    """Import the implementation only after the optional Torch dependency gate."""
    from transformation_portal.core import processing

    return processing


@pytest.mark.parametrize("blend_mode", ["linear", "gaussian"])
@pytest.mark.parametrize("shape", [(2, 3, 8, 8), (2, 3, 13, 17), (1, 3, 1, 1), (1, 3, 2, 3), (1, 3, 5, 11)])
def test_identity_preserves_every_image_and_pixel(shape, blend_mode, tiling) -> None:
    image = torch.arange(torch.tensor(shape).prod().item(), dtype=torch.float32).reshape(shape)
    image /= image.numel()
    processor = tiling.TiledProcessor(tiling.TileConfig(tile_size=8, tile_overlap=3, batch_size=2, blend_mode=blend_mode))

    result = processor.process_image(image, lambda batch: batch)

    assert result.shape == image.shape
    torch.testing.assert_close(result, image, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("blend_mode", ["linear", "gaussian"])
def test_identity_preserves_partial_final_tile_batch(blend_mode, tiling) -> None:
    image = torch.linspace(0, 1, 2 * 3 * 8 * 17).reshape(2, 3, 8, 17)
    processor = tiling.TiledProcessor(tiling.TileConfig(tile_size=8, tile_overlap=3, batch_size=2, blend_mode=blend_mode))
    callback_batch_sizes = []

    def identity(batch):
        callback_batch_sizes.append(batch.shape[0])
        return batch

    result = processor.process_image(image, identity)

    # Three spatial tiles across two images require one full and one tail batch.
    assert callback_batch_sizes == [4, 2]
    torch.testing.assert_close(result, image, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"tile_size": 0},
        {"tile_size": 4, "tile_overlap": 4},
        {"tile_overlap": -1},
        {"batch_size": 0},
        {"blend_mode": "unsupported"},
    ],
)
def test_invalid_configuration_fails_before_processing(kwargs, tiling) -> None:
    with pytest.raises(ValueError):
        tiling.TileConfig(**kwargs)


@pytest.mark.parametrize("shape", [(0, 3, 8, 8), (1, 3, 0, 8), (3, 8, 8)])
def test_empty_or_non_batched_input_is_rejected(shape, tiling) -> None:
    processor = tiling.TiledProcessor(tiling.TileConfig(tile_size=8, tile_overlap=3))

    with pytest.raises(ValueError, match="non-empty.*B, C, H, W"):
        processor.process_image(torch.empty(shape), lambda batch: batch)


def test_callback_cannot_silently_drop_batch_members(tiling) -> None:
    image = torch.ones((2, 3, 8, 8))
    processor = tiling.TiledProcessor(tiling.TileConfig(tile_size=8, tile_overlap=3))

    with pytest.raises(ValueError, match="preserve.*batch.*spatial"):
        processor.process_image(image, lambda batch: batch[:1])


@pytest.mark.parametrize("blend_mode", ["linear", "gaussian"])
def test_uint8_image_survives_an_identity_processor_exactly(blend_mode, tiling) -> None:
    """The /255, blend, *255 round trip must round, not truncate one level down."""
    import numpy as np

    ramp = np.tile(np.arange(256, dtype=np.uint8).reshape(1, 256, 1), (40, 1, 3))
    processor = tiling.TiledProcessor(tiling.TileConfig(tile_size=16, tile_overlap=4, batch_size=4, blend_mode=blend_mode))

    result = processor.process_image(ramp, lambda batch: batch)

    assert result.dtype == np.uint8
    assert np.array_equal(result, ramp)


@pytest.mark.parametrize("blend_mode", ["linear", "gaussian"])
def test_disagreeing_tiles_blend_without_a_step_at_the_tile_borders(blend_mode, tiling) -> None:
    """A tile must fade out towards its border, not stop at a sizeable weight."""
    size, overlap = 64, 8
    stride = size - overlap
    processor = tiling.TiledProcessor(
        tiling.TileConfig(tile_size=size, tile_overlap=overlap, batch_size=1, blend_mode=blend_mode)
    )
    tiles_seen = []

    def left_predicts_zero_right_predicts_one(batch):
        tiles_seen.append(batch.shape)
        return torch.full_like(batch, float(len(tiles_seen) - 1))

    blended = processor.process_image(torch.zeros((1, 1, size, stride + size)), left_predicts_zero_right_predicts_one)

    assert len(tiles_seen) == 2
    row = blended[0, 0, size // 2]
    steps = row[1:] - row[:-1]
    entering_overlap = float(steps[stride - 1])  # last pixel only the left tile covers -> first shared pixel
    leaving_overlap = float(steps[size - 1])  # last shared pixel -> first pixel only the right tile covers
    largest_step_inside_overlap = float(steps[stride : size - 1].max())
    assert float(row[stride - 1]) == 0.0 and float(row[size]) == 1.0
    assert bool((steps >= 0).all())
    assert entering_overlap < largest_step_inside_overlap
    assert leaving_overlap < largest_step_inside_overlap


def test_gaussian_tile_weight_is_centred_and_falls_to_the_floor_at_every_border(tiling) -> None:
    processor = tiling.TiledProcessor(tiling.TileConfig(tile_size=32, tile_overlap=8, blend_mode="gaussian"))

    weight = processor._create_tile_weight(32, 8, torch.device("cpu"))[0, 0]

    floor = float(weight.min())
    assert floor == pytest.approx(1e-3)
    for border in (weight[0, :], weight[-1, :], weight[:, 0], weight[:, -1]):
        assert float(border.max()) == pytest.approx(floor)
    assert float(weight.max()) == pytest.approx(float(weight[15:17, 15:17].min()))
    torch.testing.assert_close(weight, weight.flip(0))
    torch.testing.assert_close(weight, weight.flip(1))


@pytest.mark.parametrize("blend_mode", ["linear", "gaussian"])
def test_smallest_overlapping_tile_has_finite_weights(blend_mode, tiling) -> None:
    processor = tiling.TiledProcessor(tiling.TileConfig(tile_size=2, tile_overlap=1, blend_mode=blend_mode))

    weight = processor._create_tile_weight(2, 1, torch.device("cpu"))

    assert bool(torch.isfinite(weight).all()) and float(weight.min()) > 0.0
    image = torch.linspace(0, 1, 15).reshape(1, 1, 3, 5)
    torch.testing.assert_close(processor.process_image(image, lambda batch: batch), image, rtol=1e-5, atol=1e-6)
