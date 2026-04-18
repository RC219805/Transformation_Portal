from __future__ import annotations

import json
import math

import numpy as np
import pytest

from transformation_portal.core.batch import BatchJob, JobItem, JobStatus
from transformation_portal.core.device import DeviceDetector, DeviceType
from transformation_portal.core.observability import create_logger, setup_metrics
from transformation_portal.core.observability.integration import MetricsRegistry
from transformation_portal.core.storage import ExportConfig, ExportManager, autotune_export_config
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


def test_validation_helpers_smoke(tmp_path) -> None:
    image = np.full((8, 8, 3), 32, dtype=np.uint8)
    metrics = MetricsComputer.compute(image, image.copy())
    comparison = BaselineComparator().compare(image, image.copy())

    assert math.isinf(metrics.psnr)
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
