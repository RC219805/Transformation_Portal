"""Precision and failure contracts for the optional PIL TIFF comparison writer."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

pytestmark = [pytest.mark.unit]


@pytest.fixture
def optimizer():
    pytest.importorskip("tifffile")
    pytest.importorskip("imagecodecs")
    source = Path(__file__).resolve().parents[2] / "scripts/utilities/tiff_quality_optimizer.py"
    spec = importlib.util.spec_from_file_location("tiff_quality_optimizer", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.TIFFQualityOptimizer(verbose=False)


def test_pil_grayscale_tiff_roundtrips_uint16_samples(optimizer, tmp_path):
    samples = np.array([[0, 1, 257], [32767, 32768, 65535]], dtype=np.uint16)
    output = optimizer.save_tiff_method_1_pil(samples, tmp_path / "grayscale.tiff")

    with Image.open(output) as image:
        assert image.mode == "I;16"
        np.testing.assert_array_equal(np.array(image), samples)


@pytest.mark.parametrize("dtype", [np.uint16, np.float32])
@pytest.mark.parametrize("channels", [3, 4])
def test_pil_rejects_multichannel_uint16_without_publishing(optimizer, tmp_path, dtype, channels):
    samples = np.ones((2, 3, channels), dtype=dtype)
    output = tmp_path / "unsupported.tiff"

    with pytest.raises(ValueError, match="use save_tiff_method_2_tifffile"):
        optimizer.save_tiff_method_1_pil(samples, output)

    assert not output.exists()


def test_tifffile_rgb_tiff_roundtrips_uint16_samples(optimizer, tmp_path):
    import tifffile

    samples = np.arange(18, dtype=np.uint16).reshape(2, 3, 3) * 3001
    output = optimizer.save_tiff_method_2_tifffile(samples, tmp_path / "rgb.tiff")

    np.testing.assert_array_equal(tifffile.imread(output), samples)
