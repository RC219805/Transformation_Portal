from __future__ import annotations

import importlib
import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pytest

pytestmark = pytest.mark.unit

PINNED_REVISION = "a" * 40
SAM2_REPO_ID = "facebook/sam2.1-hiera-large"


class _FakeTilingConfig:
    @staticmethod
    def from_dict(config: Any) -> Any:
        return config


def _install_material_classifier_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    module = ModuleType("transformation_portal.spatial_ai.segmentation.material_classifier")

    class DummyMaterialClassifier:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.confidence_threshold = kwargs.get("confidence_threshold", 0.3)

        def is_available(self) -> bool:
            return False

        def classify_masks(self, image_uint8: np.ndarray, masks: np.ndarray) -> list[tuple[str, float]]:
            del image_uint8, masks
            return []

    setattr(module, "MaterialClassifier", DummyMaterialClassifier)
    monkeypatch.setitem(sys.modules, "transformation_portal.spatial_ai.segmentation.material_classifier", module)


def _import_pipeline_module(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    _install_material_classifier_stub(monkeypatch)
    module = importlib.import_module("transformation_portal.spatial_ai.orchestration.pipeline")
    return importlib.reload(module)


def _import_error_handler_module(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    _install_material_classifier_stub(monkeypatch)
    module = importlib.import_module("transformation_portal.spatial_ai.orchestration.error_handler")
    return importlib.reload(module)


def _import_sam2_backend(monkeypatch: pytest.MonkeyPatch) -> Any:
    _install_material_classifier_stub(monkeypatch)
    module = importlib.import_module("transformation_portal.spatial_ai.segmentation.sam2_backend")
    return importlib.reload(module).SAM2Backend


def _import_material_backend() -> Any:
    module = importlib.import_module("transformation_portal.spatial_ai.materials.material_backend")
    return importlib.reload(module).MaterialBackend


def _make_pipeline(config_dict: dict[str, Any], pipeline_mod: ModuleType, monkeypatch: pytest.MonkeyPatch) -> Any:
    pipeline = pipeline_mod.SpatialAIPipeline(config_dict)
    pipeline.error_handler = pipeline_mod.ErrorHandler(max_retries=2, initial_delay=0.0, max_delay=0.0)

    pipeline.resource_manager.select_device = lambda: "cuda"
    pipeline.resource_manager.register_model = lambda *args, **kwargs: None
    pipeline.resource_manager.unload_model = lambda *args, **kwargs: None

    pipeline.progress_tracker.start_stage = lambda *args, **kwargs: None
    pipeline.progress_tracker.complete_stage = lambda *args, **kwargs: None
    pipeline.progress_tracker.update_stage = lambda *args, **kwargs: None

    return pipeline


def _make_ingest_result(*, with_depth: bool = False) -> SimpleNamespace:
    result = SimpleNamespace(
        linear_rgb=np.zeros((2, 2, 3), dtype=np.float32),
        gamma=1.0,
    )
    if with_depth:
        result.depth = np.ones((2, 2), dtype=np.float32)
    return result


def test_sam2_backend_accepts_repo_id_revision_without_local_checkpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    SAM2Backend = _import_sam2_backend(monkeypatch)
    backend = SAM2Backend(
        model_size="large",
        device="cpu",
        repo_id=SAM2_REPO_ID,
        revision=PINNED_REVISION,
        prefer_hf_pipeline=True,
    )

    assert backend.repo_id == SAM2_REPO_ID
    assert backend.revision == PINNED_REVISION
    assert backend.prefer_hf_pipeline is True


def test_sam2_backend_repo_id_path_requires_revision(monkeypatch: pytest.MonkeyPatch) -> None:
    SAM2Backend = _import_sam2_backend(monkeypatch)
    with pytest.raises(ValueError, match="pinned revision|40-char commit SHA|unpinned"):
        SAM2Backend(
            model_size="large",
            device="cpu",
            repo_id=SAM2_REPO_ID,
            prefer_hf_pipeline=True,
        )


def test_sam2_backend_repo_id_does_not_switch_loader_without_opt_in(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    SAM2Backend = _import_sam2_backend(monkeypatch)
    checkpoint = tmp_path / "sam2_hiera_large.pt"
    checkpoint.write_bytes(b"stub")
    backend = SAM2Backend(
        model_size="large",
        device="cpu",
        checkpoint_path=str(checkpoint),
        repo_id=SAM2_REPO_ID,
        revision=PINNED_REVISION,
        prefer_hf_pipeline=False,
    )

    assert backend.prefer_hf_pipeline is False
    assert backend.checkpoint_path == checkpoint


def test_error_handler_invokes_on_device_change_for_oom(monkeypatch: pytest.MonkeyPatch) -> None:
    error_handler_mod = _import_error_handler_module(monkeypatch)
    ErrorHandler = error_handler_mod.ErrorHandler
    ErrorRecoveryStrategy = error_handler_mod.ErrorRecoveryStrategy
    handler = ErrorHandler(max_retries=2, initial_delay=0.0, max_delay=0.0)

    attempts = {"count": 0}
    callbacks: list[tuple[str, int, str]] = []

    def flaky() -> str:
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise RuntimeError("CUDA out of memory")
        return "ok"

    result = handler.execute_with_retry(
        func=flaky,
        stage="segmentation",
        strategy=ErrorRecoveryStrategy.RETRY_WITH_CPU_FALLBACK,
        device="cuda",
        on_device_change=lambda new_device, attempt, exc: callbacks.append((new_device, attempt, str(exc))),
    )

    assert result == "ok"
    assert attempts["count"] == 2
    assert callbacks == [("cpu", 1, "CUDA out of memory")]


def test_pipeline_rebuilds_segmentation_backend_on_cpu_fallback(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    pipeline_mod = _import_pipeline_module(monkeypatch)
    monkeypatch.setattr(pipeline_mod, "SegmentationTilingConfig", _FakeTilingConfig)

    builds: list[Any] = []

    class FakeSAM2Backend:
        def __init__(
            self,
            *,
            model_size: Any,
            device: Any,
            checkpoint_path: Any = None,
            repo_id: Any = None,
            revision: Any = None,
            prefer_hf_pipeline: Any = None,
            generator_kwargs: Any = None,
            enable_material_classification: bool = False,
            material_confidence_threshold: float = 0.3,
            tiling: Any = None,
        ) -> None:
            self.model_size = model_size
            self.device = device
            self.checkpoint_path = checkpoint_path
            self.repo_id = repo_id
            self.revision = revision
            self.prefer_hf_pipeline = prefer_hf_pipeline
            self.generator_kwargs = dict(generator_kwargs or {})
            self.enable_material_classification = enable_material_classification
            self.material_confidence_threshold = material_confidence_threshold
            self.tiling = tiling
            builds.append(self)

        def segment(self, seg_input: Any) -> SimpleNamespace:
            del seg_input
            if self.device == "cuda":
                raise RuntimeError("CUDA out of memory")
            return SimpleNamespace(
                masks=np.zeros((1, 2, 2), dtype=bool),
                scores=np.array([0.95], dtype=np.float32),
                metadata=[SimpleNamespace()],
            )

    monkeypatch.setattr(pipeline_mod, "SAM2Backend", FakeSAM2Backend)

    pipeline = _make_pipeline(
        {
            "tier": "standard",
            "pipeline": {
                "segmentation": {
                    "backend": "sam2",
                    "model": {
                        "size": "large",
                        "repo_id": SAM2_REPO_ID,
                        "revision": PINNED_REVISION,
                        "prefer_hf_pipeline": True,
                    },
                    "generator": {
                        "points_per_batch": 64,
                        "pred_iou_thresh": 0.88,
                    },
                }
            },
            "error_strategy": "retry_cpu_fallback",
        },
        pipeline_mod,
        monkeypatch,
    )

    result = pipeline._run_segmentation(
        ingest_result=_make_ingest_result(),
        output_dir=tmp_path,
        save_intermediates=False,
    )

    assert len(builds) == 2
    assert [backend.device for backend in builds] == ["cuda", "cpu"]
    assert all(backend.repo_id == SAM2_REPO_ID for backend in builds)
    assert all(backend.revision == PINNED_REVISION for backend in builds)
    assert all(backend.prefer_hf_pipeline is True for backend in builds)
    assert builds[1].generator_kwargs == {
        "points_per_batch": 64,
        "pred_iou_thresh": 0.88,
    }
    assert float(result.scores[0]) == pytest.approx(0.95)


def test_pipeline_rebuilds_material_backend_on_cpu_fallback(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    pipeline_mod = _import_pipeline_module(monkeypatch)
    builds: list[Any] = []
    generate_calls: list[dict[str, Any]] = []

    class FakeMaterialBackend:
        def __init__(
            self,
            *,
            backend: Any,
            device: Any,
            model_repo_id: Any = None,
            model_revision: Any = None,
            generation_config_overrides: Any = None,
        ) -> None:
            self.backend = backend
            self.device = device
            self.model_repo_id = model_repo_id
            self.model_revision = model_revision
            self.generation_config_overrides = dict(generation_config_overrides or {})
            builds.append(self)

        def generate(self, mat_input: Any) -> SimpleNamespace:
            generate_calls.append(
                {
                    "device": self.device,
                    "overrides": dict(self.generation_config_overrides),
                    "depth": mat_input.depth,
                    "material_hint": mat_input.material_hint,
                }
            )
            if self.device == "cuda":
                raise RuntimeError("CUDA out of memory")
            return SimpleNamespace(height=None)

    monkeypatch.setattr(pipeline_mod, "MaterialBackend", FakeMaterialBackend)

    pipeline = _make_pipeline(
        {
            "tier": "standard",
            "pipeline": {
                "materials": {
                    "backend": "heuristic",
                    "model_repo_id": "acme/pbr-model",
                    "model_revision": PINNED_REVISION,
                    "material_hints": True,
                    "resolution": 2048,
                    "optimize_iterations": 200,
                    "use_depth": True,
                    "normal_strength": 1.2,
                    "ao_intensity": 0.8,
                }
            },
            "error_strategy": "retry_cpu_fallback",
        },
        pipeline_mod,
        monkeypatch,
    )

    result = pipeline._run_materials(
        ingest_result=_make_ingest_result(with_depth=True),
        seg_result=SimpleNamespace(
            masks=[np.ones((2, 2), dtype=bool)],
            metadata=[SimpleNamespace(material_label="wood")],
        ),
        output_dir=tmp_path,
        save_intermediates=False,
    )

    assert list(result.keys()) == ["segment_0"]
    assert len(builds) == 2
    assert [backend.device for backend in builds] == ["cuda", "cpu"]

    second_backend_overrides = builds[1].generation_config_overrides
    assert second_backend_overrides["backend"] == "heuristic"
    assert second_backend_overrides["device"] == "cpu"
    assert second_backend_overrides["resolution"] == 2048
    assert second_backend_overrides["optimize_iterations"] == 200
    assert second_backend_overrides["use_depth"] is True
    assert second_backend_overrides["normal_strength"] == pytest.approx(1.2)
    assert second_backend_overrides["ao_intensity"] == pytest.approx(0.8)

    assert generate_calls[1]["material_hint"] == "wood"
    assert generate_calls[1]["depth"] is not None
    assert builds[1].model_repo_id == "acme/pbr-model"
    assert builds[1].model_revision == PINNED_REVISION


def test_material_backend_builds_generation_config_from_overrides() -> None:
    MaterialBackend = _import_material_backend()
    backend = MaterialBackend(
        backend="heuristic",
        device="cpu",
        generation_config_overrides={
            "resolution": 2048,
            "optimize_iterations": 77,
            "use_depth": True,
            "normal_strength": 1.25,
            "ao_intensity": 0.65,
        },
    )

    config = backend._build_generation_config()

    assert config.resolution == 2048
    assert config.optimize_iterations == 77
    assert config.use_depth is True
    assert config.normal_strength == pytest.approx(1.25)
    assert config.ao_intensity == pytest.approx(0.65)


def test_materials_warn_when_use_depth_requested_but_unavailable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any, caplog: pytest.LogCaptureFixture
) -> None:
    pipeline_mod = _import_pipeline_module(monkeypatch)

    class FakeMaterialBackend:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

        def generate(self, mat_input: Any) -> SimpleNamespace:
            del mat_input
            return SimpleNamespace(height=None)

    monkeypatch.setattr(pipeline_mod, "MaterialBackend", FakeMaterialBackend)

    pipeline = _make_pipeline(
        {
            "tier": "standard",
            "pipeline": {
                "materials": {
                    "backend": "heuristic",
                    "use_depth": True,
                }
            },
            "error_strategy": "retry",
        },
        pipeline_mod,
        monkeypatch,
    )

    with caplog.at_level("WARNING"):
        pipeline._run_materials(
            ingest_result=_make_ingest_result(with_depth=False),
            seg_result=SimpleNamespace(
                masks=[np.ones((2, 2), dtype=bool)],
                metadata=[SimpleNamespace(material_label="wood")],
            ),
            output_dir=tmp_path,
            save_intermediates=False,
        )

    assert "use_depth=True" in caplog.text


@pytest.mark.parametrize(
    "config_dict",
    [
        {"tier": "apex_research", "pipeline": {"reconstruct": {"enabled": True}}},
        {"tier": "apex_research", "pipeline": {"reconstruction": {"enabled": True}}},
    ],
)
def test_pipeline_normalizes_reconstruct_alias(monkeypatch: pytest.MonkeyPatch, config_dict: dict[str, Any]) -> None:
    pipeline_mod = _import_pipeline_module(monkeypatch)
    PipelineConfig = pipeline_mod.PipelineConfig
    pipeline = pipeline_mod.SpatialAIPipeline(config_dict)
    assert pipeline.config.stages == ["reconstruction"]

    direct = PipelineConfig(tier="apex_research", stages=["reconstruct"])
    assert direct.stages == ["reconstruction"]


def test_pipeline_rejects_unrelated_pipelineconfig_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    pipeline_mod = _import_pipeline_module(monkeypatch)

    class PipelineConfig:
        def __init__(self) -> None:
            self.tier = "standard"
            self.stages = ["ingest"]

    with pytest.raises(TypeError, match="config must be PipelineConfig, dict, str, or Path"):
        pipeline_mod.SpatialAIPipeline(PipelineConfig())
