from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import pytest

from transformation_portal.core.batch import BatchJob, JobItem, JobStatus
from transformation_portal.core.device import DeviceDetector, DeviceType
from transformation_portal.core.observability import create_logger, setup_metrics
from transformation_portal.core.observability.integration import MetricsRegistry
from transformation_portal.core.storage import ExportConfig, ExportManager, autotune_export_config, compute_image_stats
from transformation_portal.core.validation import BaselineComparator, MetricsComputer
from transformation_portal.core.validation.report import DeviceInfo as ReportDeviceInfo
from transformation_portal.core.validation.report import GitInfo, ProcessingReport

pytestmark = pytest.mark.unit


def test_batch_job_checkpoint_round_trip(tmp_path) -> None:
    checkpoint_path = tmp_path / "batch.json"
    job = BatchJob(
        name="smoke",
        output_dir=str(tmp_path / "outputs"),
        items=[
            JobItem(
                id="image-1",
                input_path="input.tif",
                output_path="output.tif",
                status=JobStatus.COMPLETED,
                execution_time=1.25,
            )
        ],
    )

    job.save(checkpoint_path)
    loaded = BatchJob.load(checkpoint_path)

    assert loaded.get_item("image-1") is not None
    assert loaded.progress == 1.0
    assert loaded.stats["COMPLETED"] == 1


def test_device_package_cpu_fallback_smoke() -> None:
    if DeviceDetector is None:
        pytest.skip("device optional dependencies unavailable")

    info = DeviceDetector.get_optimal_device(force_cpu=True)

    assert info.type == DeviceType.CPU
    assert info.capabilities.device_name


def test_storage_autotune_and_export_smoke(tmp_path) -> None:
    image = np.zeros((4, 4, 4), dtype=np.uint8)
    image[..., 3] = 255

    config = ExportConfig(format="auto")
    tuned = autotune_export_config(image, config)
    saved_path = ExportManager.save_image(image, tmp_path / "artifact.bin", config)

    assert tuned.format == "png"
    assert saved_path.suffix == ".png"
    assert saved_path.exists()


def test_export_scales_unit_float_images_for_jpeg(tmp_path) -> None:
    """A 0-1 float image must not be cast straight to 8-bit integers (all black)."""
    import cv2

    grey = np.full((32, 32, 3), 0.5, dtype=np.float32)

    saved_path = ExportManager.save_image(grey, tmp_path / "mid_grey.jpg", ExportConfig(format="jpg"))

    decoded = cv2.imread(str(saved_path), cv2.IMREAD_UNCHANGED)
    assert decoded.dtype == np.uint8
    assert 126 <= int(decoded.min()) <= int(decoded.max()) <= 130


def test_export_keeps_unit_float_precision_in_16_bit_png(tmp_path) -> None:
    import cv2

    image = np.zeros((16, 16, 4), dtype=np.float32)
    image[..., 0] = 0.25  # red
    image[..., 2] = 1.0  # blue
    image[..., 3] = 0.5  # translucent, so autotune selects PNG

    saved_path = ExportManager.save_image(image, tmp_path / "overlay.bin", ExportConfig(format="auto"))

    decoded = cv2.imread(str(saved_path), cv2.IMREAD_UNCHANGED)
    assert saved_path.suffix == ".png"
    assert decoded.dtype == np.uint16 and decoded.shape == (16, 16, 4)
    blue, green, red, alpha = (int(decoded[0, 0, channel]) for channel in range(4))
    assert (blue, green, red, alpha) == (65535, 0, 16384, 32768)


def test_export_accepts_single_channel_images(tmp_path) -> None:
    """A 2-D depth map has no channel axis to subsample."""
    import cv2

    depth = np.linspace(0.0, 1.0, 64, dtype=np.float32).reshape(8, 8)

    stats = compute_image_stats(depth, sample_stride=1)
    saved_path = ExportManager.save_image(depth, tmp_path / "depth.bin", ExportConfig(format="auto"))

    assert (stats.has_alpha, stats.is_hdr, stats.bit_depth_hint) == (False, False, 16)
    assert stats.max_val == pytest.approx(1.0)
    assert saved_path.suffix == ".tiff"
    assert np.array_equal(cv2.imread(str(saved_path), cv2.IMREAD_UNCHANGED), depth)
    assert compute_image_stats(np.zeros((8, 8), dtype=np.uint16)).bit_depth_hint == 16


@pytest.mark.ml
def test_processing_linear_blend_mode_is_real() -> None:
    torch = pytest.importorskip("torch")
    from transformation_portal.core.processing import TileConfig, TiledProcessor

    linear_processor = TiledProcessor(TileConfig(tile_size=8, tile_overlap=3, blend_mode="linear"))
    gaussian_processor = TiledProcessor(TileConfig(tile_size=8, tile_overlap=3, blend_mode="gaussian"))

    linear_weight = linear_processor._create_tile_weight(8, 3, torch.device("cpu"))
    gaussian_weight = gaussian_processor._create_tile_weight(8, 3, torch.device("cpu"))

    assert linear_weight.shape == (1, 1, 8, 8)
    assert float(linear_weight[0, 0, 0, 0]) > 0.0
    assert float(linear_weight[0, 0, 4, 4]) > float(linear_weight[0, 0, 0, 0])
    assert torch.allclose(linear_weight, gaussian_weight) is False


def test_observability_setup_metrics_is_explicit_no_op() -> None:
    MetricsRegistry._metrics.clear()

    assert setup_metrics() is None

    logger = create_logger("transformation_portal.tests.core")
    logger.debug("metrics hook retained")
    MetricsRegistry.increment("jobs")

    assert MetricsRegistry.get_all() == {"jobs": 1}


def test_validation_package_import_is_torch_optional() -> None:
    test_script = """
import sys

class TorchBlocker:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise ImportError(f"Torch import blocked for test: {fullname}")
        return None

sys.meta_path.insert(0, TorchBlocker())

import transformation_portal.core.validation as validation
from transformation_portal.core.validation.report import DeviceInfo, ProcessingReport

assert validation.MetricsComputer is not None
device = DeviceInfo.capture()
assert device.pytorch_version == "not-installed"
assert device.cuda_version is None
assert device.gpu_name is None
report = ProcessingReport(job_id="job-1")
assert report.device.pytorch_version == "not-installed"
print("SUCCESS")
"""
    result = subprocess.run(
        [sys.executable, "-c", test_script],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )

    if result.returncode != 0:
        pytest.fail(f"validation import should not require torch\nstdout: {result.stdout}\nstderr: {result.stderr}")

    assert "SUCCESS" in result.stdout


def test_validation_helpers_smoke(tmp_path) -> None:
    image = np.full((8, 8, 3), 32, dtype=np.uint8)
    metrics = MetricsComputer.compute(image, image.copy())
    comparison = BaselineComparator().compare(image, image.copy())

    assert metrics.psnr >= 100.0
    assert metrics.mse == pytest.approx(0.0)
    assert metrics.ssim == pytest.approx(1.0)
    assert comparison.passed is True

    report = ProcessingReport(
        job_id="job-1",
        git=GitInfo(commit_hash="abc123", branch="main", is_dirty=False),
        device=ReportDeviceInfo(
            system="darwin",
            python_version="3.11.0",
            pytorch_version="2.x",
            cuda_version=None,
            gpu_name=None,
        ),
        metrics={"psnr": float(metrics.psnr)},
    )
    report_path = tmp_path / "report.json"
    report.save(str(report_path))

    payload = json.loads(report_path.read_text())
    assert payload["job_id"] == "job-1"
    assert payload["metrics"]["psnr"] == float(metrics.psnr)


@pytest.mark.parametrize("use_skimage", [False, True])
def test_validation_uint8_mse_does_not_overflow(monkeypatch, use_skimage) -> None:
    from transformation_portal.core.validation import metrics as metrics_module

    monkeypatch.setattr(metrics_module, "SKIMAGE_AVAIL", use_skimage)
    if use_skimage:
        # Exercise our arithmetic without requiring the optional backend in
        # core CI. Distinct sentinels also verify its metrics pass through.
        monkeypatch.setattr(metrics_module, "psnr", lambda *args, **kwargs: 12.5, raising=False)
        monkeypatch.setattr(metrics_module, "ssim", lambda *args, **kwargs: 0.25, raising=False)
    black = np.zeros((8, 8, 3), dtype=np.uint8)
    white = np.full_like(black, 255)

    metrics = MetricsComputer.compute(black, white)

    assert metrics.mse == pytest.approx(65025.0)
    assert metrics.psnr == pytest.approx(12.5 if use_skimage else 0.0)
    assert metrics.ssim == pytest.approx(0.25 if use_skimage else 0.0)


@pytest.mark.parametrize("use_skimage", [False, True])
@pytest.mark.parametrize(
    ("dtype", "value", "expected_range"),
    [(np.uint8, 120, 255.0), (np.uint16, 30000, 65535.0), (np.int16, 1000, 65535.0), (np.float32, 0.5, 1.0)],
    ids=["uint8", "uint16", "int16", "float32"],
)
def test_validation_psnr_uses_the_data_range_of_the_dtype(monkeypatch, use_skimage, dtype, value, expected_range) -> None:
    """One 16-bit code value of error is not a 29 dB image."""
    import math

    from transformation_portal.core.validation import metrics as metrics_module

    seen_ranges = []

    def recording_metric(_target, _prediction, *, data_range, **_kwargs):
        seen_ranges.append(data_range)
        return 42.0

    monkeypatch.setattr(metrics_module, "SKIMAGE_AVAIL", use_skimage)
    monkeypatch.setattr(metrics_module, "psnr", recording_metric, raising=False)
    monkeypatch.setattr(metrics_module, "ssim", recording_metric, raising=False)
    reference = np.full((16, 16, 3), value, dtype=dtype)
    changed = reference.copy()
    changed[0, 0, 0] += dtype(1) if np.issubdtype(dtype, np.integer) else dtype(0.25)

    metrics = MetricsComputer.compute(changed, reference)

    if use_skimage:
        assert seen_ranges == [expected_range, expected_range]
    else:
        assert metrics.psnr == pytest.approx(20 * math.log10(expected_range / math.sqrt(metrics.mse)))
        assert seen_ranges == []
    if dtype is np.uint16 and not use_skimage:
        assert metrics.psnr > 120.0
