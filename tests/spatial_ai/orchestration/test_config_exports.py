"""Compatibility tests for the Spatial AI PipelineConfig extraction."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from transformation_portal.spatial_ai.orchestration.error_handler import ErrorRecoveryStrategy

pytestmark = pytest.mark.unit


def test_pipeline_config_identity_is_preserved_across_import_surfaces() -> None:
    config_mod = importlib.import_module("transformation_portal.spatial_ai.orchestration.config")
    pipeline_mod = importlib.import_module("transformation_portal.spatial_ai.orchestration.pipeline")
    package_mod = importlib.import_module("transformation_portal.spatial_ai.orchestration")

    assert pipeline_mod.PipelineConfig is config_mod.PipelineConfig
    assert package_mod.PipelineConfig is config_mod.PipelineConfig
    assert "PipelineConfig" in package_mod.__all__


def test_pipeline_result_identity_is_preserved_across_import_surfaces() -> None:
    results_mod = importlib.import_module("transformation_portal.spatial_ai.orchestration.results")
    pipeline_mod = importlib.import_module("transformation_portal.spatial_ai.orchestration.pipeline")
    package_mod = importlib.import_module("transformation_portal.spatial_ai.orchestration")

    assert pipeline_mod.PipelineResult is results_mod.PipelineResult
    assert pipeline_mod.MultiViewReconstructionResult is results_mod.MultiViewReconstructionResult
    assert package_mod.PipelineResult is results_mod.PipelineResult
    assert "PipelineResult" in package_mod.__all__


def test_pipeline_result_identity_survives_pipeline_reload() -> None:
    pipeline_mod = importlib.import_module("transformation_portal.spatial_ai.orchestration.pipeline")
    PipelineResult = pipeline_mod.PipelineResult
    MultiViewReconstructionResult = pipeline_mod.MultiViewReconstructionResult

    reloaded = importlib.reload(pipeline_mod)

    assert reloaded.PipelineResult is PipelineResult
    assert reloaded.MultiViewReconstructionResult is MultiViewReconstructionResult


def test_spatial_ai_pipeline_accepts_extracted_pipeline_config() -> None:
    config_mod = importlib.import_module("transformation_portal.spatial_ai.orchestration.config")
    pipeline_mod = importlib.import_module("transformation_portal.spatial_ai.orchestration.pipeline")

    config = config_mod.PipelineConfig(tier="standard", stages=["ingest"])
    pipeline = pipeline_mod.SpatialAIPipeline(config)

    assert pipeline.config is config


def test_spatial_ai_pipeline_keeps_reload_safe_pipeline_config_contract() -> None:
    pipeline_mod = importlib.import_module("transformation_portal.spatial_ai.orchestration.pipeline")

    class PipelineConfig:
        tier = "standard"
        stages = ["ingest"]
        ingest = {}
        segmentation = {"cache_policy": "read_write"}
        materials = {}
        reconstruction = {}
        resource_limits = None
        error_strategy = ErrorRecoveryStrategy.RETRY
        use_execution_graph = False

    reload_safe_config = PipelineConfig()
    pipeline = pipeline_mod.SpatialAIPipeline(reload_safe_config)

    assert pipeline.config is reload_safe_config


def test_extracted_pipeline_config_preserves_validation_contracts(monkeypatch: pytest.MonkeyPatch) -> None:
    config_mod = importlib.import_module("transformation_portal.spatial_ai.orchestration.config")

    config = config_mod.PipelineConfig(
        tier="apex_research",
        stages=["reconstruct", "materials"],
        segmentation={"cache_policy": "off"},
        materials={"backend": "heuristic", "strict_backend": False},
        error_strategy="retry_cpu_fallback",
    )

    assert config.stages == ["reconstruction", "materials"]
    assert config.segmentation["cache_policy"] == "off"
    assert config.error_strategy is ErrorRecoveryStrategy.RETRY_WITH_CPU_FALLBACK

    with pytest.raises(ValueError, match="segmentation.cache_policy"):
        config_mod.PipelineConfig(tier="standard", stages=["ingest"], segmentation={"cache_policy": "invalid"})

    with pytest.raises(ValueError, match="research tier.*3DGS"):
        config_mod.PipelineConfig(tier="standard", stages=["reconstruct"])

    monkeypatch.delenv("PBRFUSION_PATH", raising=False)
    with pytest.raises(ValueError, match="runtime_missing"):
        config_mod.PipelineConfig(
            tier="standard",
            stages=["ingest", "segment", "materials"],
            materials={"backend": "pbr_fusion", "strict_backend": True},
        )


def test_extracted_pipeline_result_save_summary_preserves_single_image_shape(tmp_path: Path) -> None:
    results_mod = importlib.import_module("transformation_portal.spatial_ai.orchestration.results")

    result = results_mod.PipelineResult(
        input_path=Path("input.tiff"),
        output_dir=Path("output"),
        stages_completed=["ingest", "segmentation", "materials", "reconstruction"],
        linear_image=object(),
        segmentation=SimpleNamespace(masks=[object(), object()]),
        materials={"seg_0": object(), "seg_1": object()},
        scene_3d=SimpleNamespace(splats=SimpleNamespace(num_gaussians=10000), rmse=0.015),
        execution_time=123.4,
        peak_memory_mb=4096.0,
        errors=["Error 1", "Error 2"],
        warnings=["Warning 1"],
        metadata={"custom": "data"},
        stage_reports=[
            {"stage": "ingest", "status": "completed", "capability": None, "quality_gate": None},
            {"stage": "segmentation", "status": "completed", "capability": None, "quality_gate": None},
            {"stage": "materials", "status": "completed", "capability": None, "quality_gate": None},
            {"stage": "reconstruction", "status": "completed", "capability": None, "quality_gate": None},
        ],
    )

    summary_path = tmp_path / "summary.json"
    result.save_summary(summary_path)

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary == {
        "input": "input.tiff",
        "output_dir": "output",
        "stages_completed": ["ingest", "segmentation", "materials", "reconstruction"],
        "execution_time": 123.4,
        "peak_memory_mb": 4096.0,
        "errors": ["Error 1", "Error 2"],
        "warnings": ["Warning 1"],
        "stage_reports": [
            {"stage": "ingest", "status": "completed", "capability": None, "quality_gate": None},
            {"stage": "segmentation", "status": "completed", "capability": None, "quality_gate": None},
            {"stage": "materials", "status": "completed", "capability": None, "quality_gate": None},
            {"stage": "reconstruction", "status": "completed", "capability": None, "quality_gate": None},
        ],
        "results": {
            "linear_image": True,
            "segmentation": {"completed": True, "num_masks": 2},
            "materials": {"completed": True, "num_segments": 2},
            "scene_3d": {"completed": True, "num_gaussians": 10000, "rmse": 0.015},
        },
        "metadata": {"custom": "data"},
    }


def test_extracted_pipeline_result_save_summary_uses_atomic_writer(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    results_mod = importlib.import_module("transformation_portal.spatial_ai.orchestration.results")
    calls = []

    def fake_write_json_atomic(path: Path, payload: dict) -> None:
        calls.append((path, payload))

    monkeypatch.setattr(results_mod, "_write_json_atomic", fake_write_json_atomic)
    result = results_mod.PipelineResult(
        input_path=Path("input.tiff"),
        output_dir=Path("output"),
        stages_completed=["ingest"],
    )

    summary_path = tmp_path / "summary.json"
    result.save_summary(summary_path)

    assert len(calls) == 1
    assert calls[0][0] == summary_path
    assert calls[0][1]["input"] == "input.tiff"


def test_extracted_result_atomic_writer_preserves_existing_file_on_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    results_mod = importlib.import_module("transformation_portal.spatial_ai.orchestration.results")
    summary_path = tmp_path / "summary.json"
    summary_path.write_text('{"existing": true}', encoding="utf-8")

    def fail_dump(*args: object, **kwargs: object) -> None:
        raise RuntimeError("serialization failed")

    monkeypatch.setattr(results_mod.json, "dump", fail_dump)

    with pytest.raises(RuntimeError, match="serialization failed"):
        results_mod._write_json_atomic(summary_path, {"existing": False})

    assert summary_path.read_text(encoding="utf-8") == '{"existing": true}'


def test_extracted_multiview_result_save_summary_preserves_reconstruction_shape(tmp_path: Path) -> None:
    results_mod = importlib.import_module("transformation_portal.spatial_ai.orchestration.results")
    scene = SimpleNamespace(
        splats=SimpleNamespace(num_gaussians=100),
        rmse=0.02,
        convergence="converged",
        quality_score=0.91,
        iteration=20,
    )

    result = results_mod.MultiViewReconstructionResult(
        scene=scene,
        ply_path=Path("output/reconstruction.ply"),
        sidecar_path=Path("output/reconstruction.provenance.json"),
        output_dir=Path("output"),
        execution_time=12.5,
        peak_memory_mb=512.0,
        stages_completed=["reconstruction", "export"],
        request_metadata={"camera_source_summary": {"explicit": 2}},
        errors=[],
        warnings=["warn"],
        stage_reports=[
            {"stage": "reconstruction", "status": "completed", "capability": None, "quality_gate": None},
            {"stage": "export", "status": "completed", "capability": None, "quality_gate": None},
        ],
    )

    summary_path = tmp_path / "reconstruction_summary.json"
    result.save_summary(summary_path)

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary == {
        "output_dir": "output",
        "ply_path": "output/reconstruction.ply",
        "sidecar_path": "output/reconstruction.provenance.json",
        "stages_completed": ["reconstruction", "export"],
        "execution_time": 12.5,
        "peak_memory_mb": 512.0,
        "errors": [],
        "warnings": ["warn"],
        "stage_reports": [
            {"stage": "reconstruction", "status": "completed", "capability": None, "quality_gate": None},
            {"stage": "export", "status": "completed", "capability": None, "quality_gate": None},
        ],
        "scene": {
            "num_gaussians": 100,
            "rmse": 0.02,
            "convergence": "converged",
            "quality_score": 0.91,
            "iteration": 20,
        },
        "request_metadata": {"camera_source_summary": {"explicit": 2}},
    }


def test_extracted_multiview_result_save_summary_uses_atomic_writer(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    results_mod = importlib.import_module("transformation_portal.spatial_ai.orchestration.results")
    calls = []
    scene = SimpleNamespace(
        splats=SimpleNamespace(num_gaussians=100),
        rmse=0.02,
        convergence="converged",
        quality_score=0.91,
        iteration=20,
    )

    def fake_write_json_atomic(path: Path, payload: dict) -> None:
        calls.append((path, payload))

    monkeypatch.setattr(results_mod, "_write_json_atomic", fake_write_json_atomic)
    result = results_mod.MultiViewReconstructionResult(
        scene=scene,
        ply_path=Path("output/reconstruction.ply"),
        sidecar_path=Path("output/reconstruction.provenance.json"),
        output_dir=Path("output"),
    )

    summary_path = tmp_path / "reconstruction_summary.json"
    result.save_summary(summary_path)

    assert len(calls) == 1
    assert calls[0][0] == summary_path
    assert calls[0][1]["ply_path"] == "output/reconstruction.ply"
