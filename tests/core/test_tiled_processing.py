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
