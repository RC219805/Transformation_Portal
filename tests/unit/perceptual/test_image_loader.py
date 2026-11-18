"""Unit tests for perceptual.image_loader.

Regression coverage for M-20: metadata used to be read after ``convert("RGB")``
(so bit depth, colour space and format were always 8 / RGB / unknown), 16-bit
greyscale data saturated to white, and ``normalize=False`` handed uint8 tensors
to float-only downstream metrics.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from PIL import Image  # noqa: E402

pytestmark = [pytest.mark.unit, pytest.mark.ml]


def _substrate():
    return SimpleNamespace(get_device=lambda: torch.device("cpu"), to_device=lambda t: t)


def _loader(**kwargs):
    from transformation_portal.perceptual.image_loader import ImageLoader

    return ImageLoader(_substrate(), **kwargs)


def _gray16(tmp_path: Path) -> tuple[Path, np.ndarray]:
    rng = np.random.default_rng(0)
    samples = (rng.random((32, 40)) * 65535).astype(np.uint16)
    path = tmp_path / "kitchen16.tif"
    Image.fromarray(samples).save(path)
    return path, samples


def test_sixteen_bit_gray_tiff_keeps_source_metadata_and_scale(tmp_path: Path) -> None:
    path, samples = _gray16(tmp_path)

    tensor, metadata = _loader().load(path)

    assert metadata.bit_depth == 16
    assert metadata.color_space.startswith("I")
    assert metadata.channels == 1
    assert metadata.format == "TIFF"
    assert (metadata.width, metadata.height) == (40, 32)

    expected_mean = samples.mean() / 65535.0
    assert tensor.dtype == torch.float32
    assert tuple(tensor.shape) == (3, 32, 40)
    assert float(tensor.mean()) == pytest.approx(expected_mean, abs=1e-3)
    assert float(tensor.max()) <= 1.0
    assert metadata.mean_intensity == pytest.approx(expected_mean, abs=1e-3)


def test_sixteen_bit_gray_resize_stays_in_range(tmp_path: Path) -> None:
    path, samples = _gray16(tmp_path)

    tensor, _ = _loader(target_size=(16, 20)).load(path)

    assert tuple(tensor.shape) == (3, 16, 20)
    assert float(tensor.mean()) == pytest.approx(samples.mean() / 65535.0, abs=0.02)
    assert 0.0 <= float(tensor.min()) and float(tensor.max()) <= 1.0


def test_sixteen_bit_step_edge_resize_does_not_ring_out_of_range(tmp_path: Path) -> None:
    # A hard black/white step makes LANCZOS overshoot (about -0.08 .. 1.17 here);
    # the float resize path must clamp like the 8-bit path does.
    samples = np.zeros((64, 64), dtype=np.uint16)
    samples[:, 32:] = 65535
    samples[20:24, :] = 0
    path = tmp_path / "edge16.tif"
    Image.fromarray(samples).save(path)

    tensor, _ = _loader(target_size=(23, 23)).load(path)
    raw, _ = _loader(target_size=(23, 23), normalize=False).load(path)

    assert float(tensor.min()) >= 0.0
    assert float(tensor.max()) <= 1.0
    assert float(raw.min()) >= 0.0
    assert float(raw.max()) <= 255.0
    assert float(tensor.max()) == pytest.approx(1.0, abs=1e-6)


def test_eight_bit_rgb_png_metadata_and_values(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    rgb = (rng.random((24, 24, 3)) * 255).astype(np.uint8)
    path = tmp_path / "pool.png"
    Image.fromarray(rgb).save(path)

    tensor, metadata = _loader().load(path)

    assert (metadata.bit_depth, metadata.color_space, metadata.channels, metadata.format) == (8, "RGB", 3, "PNG")
    expected = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    assert torch.equal(tensor, expected)


def test_palette_image_reports_source_mode(tmp_path: Path) -> None:
    path = tmp_path / "aerial.png"
    Image.new("P", (8, 8), color=3).save(path)

    tensor, metadata = _loader().load(path)

    assert metadata.color_space == "P"
    assert metadata.bit_depth == 8
    assert metadata.channels == 1
    assert tuple(tensor.shape) == (3, 8, 8)


def test_normalize_false_returns_float_on_8bit_scale(tmp_path: Path) -> None:
    rgb = np.full((8, 8, 3), 200, dtype=np.uint8)
    path = tmp_path / "bath.png"
    Image.fromarray(rgb).save(path)

    tensor, _ = _loader(normalize=False).load(path)

    assert tensor.is_floating_point()
    assert float(tensor.max()) == pytest.approx(200.0)


def test_normalize_false_baseline_calibration_does_not_crash(tmp_path: Path) -> None:
    from transformation_portal.perceptual.baseline import BaselineConfig, PerceptualBaseline

    rgb = (np.random.default_rng(2).random((32, 32, 3)) * 255).astype(np.uint8)
    path = tmp_path / "pool.png"
    Image.fromarray(rgb).save(path)
    config = BaselineConfig(normalize=False, save_visualizations=False, save_reports=False, output_dir=tmp_path / "out")

    # Used to raise "expected scalar type Byte but found Float" from a uint8 tensor.
    results = PerceptualBaseline(_substrate(), config).calibrate([path])
    assert len(results) == 1
