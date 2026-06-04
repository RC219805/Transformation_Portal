from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "tools" / "benchmark_unified_luxury_batch_io.py"


def _load_harness() -> ModuleType:
    spec = importlib.util.spec_from_file_location("benchmark_unified_luxury_batch_io", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_generate_synthetic_inputs_are_deterministic_tiffs(tmp_path: Path) -> None:
    harness = _load_harness()

    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first = harness.generate_synthetic_inputs(first_dir, count=2, width=16, height=12, image_format="tif")
    second = harness.generate_synthetic_inputs(second_dir, count=2, width=16, height=12, image_format="tif")

    assert [path.name for path in first] == ["synthetic_0000.tif", "synthetic_0001.tif"]
    assert [path.read_bytes() for path in first] == [path.read_bytes() for path in second]


def test_default_candidate_requires_memory_limit_and_speedup() -> None:
    harness = _load_harness()
    serial = {
        "wall_time_s": {"mean": 10.0, "p95": 10.5},
        "peak_rss_mib": {"max": 100.0},
        "images_failed": 0,
        "output_files": 12,
    }
    parallel = {
        "wall_time_s": {"mean": 5.0, "p95": 8.0},
        "peak_rss_mib": {"max": 120.0},
        "images_failed": 0,
        "output_files": 12,
    }

    missing_limit = harness.evaluate_parallel_io_default(
        serial_summary=serial,
        parallel_summary=parallel,
        min_speedup=1.10,
        memory_limit_mib=None,
        representative_input_set=True,
    )
    not_representative = harness.evaluate_parallel_io_default(
        serial_summary=serial,
        parallel_summary=parallel,
        min_speedup=1.10,
        memory_limit_mib=256.0,
        representative_input_set=False,
    )
    accepted = harness.evaluate_parallel_io_default(
        serial_summary=serial,
        parallel_summary=parallel,
        min_speedup=1.10,
        memory_limit_mib=256.0,
        representative_input_set=True,
    )

    assert missing_limit["decision"] == "keep_false"
    assert "no memory limit supplied" in missing_limit["reason"]
    assert not_representative["decision"] == "keep_false"
    assert "not marked representative" in not_representative["reason"]
    assert accepted["decision"] == "candidate_after_representative_runs"
    assert accepted["parallel_io_default_candidate"] is True


def test_default_candidate_rejects_output_count_changes() -> None:
    harness = _load_harness()
    serial = {
        "wall_time_s": {"mean": 10.0, "p95": 10.5},
        "peak_rss_mib": {"max": 100.0},
        "images_failed": 0,
        "output_files": 12,
    }
    parallel = {
        "wall_time_s": {"mean": 5.0, "p95": 8.0},
        "peak_rss_mib": {"max": 120.0},
        "images_failed": 0,
        "output_files": 10,
    }

    result = harness.evaluate_parallel_io_default(
        serial_summary=serial,
        parallel_summary=parallel,
        min_speedup=1.10,
        memory_limit_mib=256.0,
        representative_input_set=True,
    )

    assert result["decision"] == "keep_false"
    assert "output file count changed: serial=12, parallel_io=10" in result["reason"]


def test_default_candidate_rejects_p95_regression() -> None:
    harness = _load_harness()
    serial = {
        "wall_time_s": {"mean": 10.0, "p95": 10.5},
        "peak_rss_mib": {"max": 100.0},
        "images_failed": 0,
        "output_files": 12,
    }
    parallel = {
        "wall_time_s": {"mean": 5.0, "p95": 11.0},
        "peak_rss_mib": {"max": 120.0},
        "images_failed": 0,
        "output_files": 12,
    }

    result = harness.evaluate_parallel_io_default(
        serial_summary=serial,
        parallel_summary=parallel,
        min_speedup=1.10,
        memory_limit_mib=256.0,
        representative_input_set=True,
    )

    assert result["decision"] == "keep_false"
    assert "parallel_io p95 11.000s regressed versus serial p95 10.500s" in result["reason"]


def test_reuse_assessment_keeps_adjacent_pipeline_changes_evidence_driven() -> None:
    harness = _load_harness()

    assessment = harness.reuse_assessment()

    assert "ThreadPoolExecutor" in assessment["recipe_pipeline"]["current_control"]
    assert "ProcessPoolExecutor" in assessment["tiff_batch_processor"]["current_control"]
    assert "GPU/backend acceleration is out of scope" in assessment["acceleration_scope"]


def test_harness_emits_batch_io_report(tmp_path: Path) -> None:
    harness = _load_harness()
    output_json = tmp_path / "report.json"
    work_dir = tmp_path / "work"
    args = harness.build_parser().parse_args(
        [
            "--work-dir",
            str(work_dir),
            "--output-json",
            str(output_json),
            "--runs",
            "1",
            "--warmup-runs",
            "0",
            "--image-count",
            "2",
            "--width",
            "32",
            "--height",
            "24",
            "--input-format",
            "jpg",
            "--output-formats",
            "social",
        ]
    )

    written = harness.run_from_args(args)
    report = json.loads(output_json.read_text(encoding="utf-8"))

    assert written == output_json
    assert report["schema"] == harness.SCHEMA
    assert report["summary"]["serial"]["runs"] == 1
    assert report["summary"]["parallel_io"]["runs"] == 1
    assert report["trials"]["serial"][0]["images_succeeded"] == 2
    assert report["trials"]["parallel_io"][0]["images_succeeded"] == 2
    assert report["comparison"]["decision"] == "keep_false"
