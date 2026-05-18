"""Focused SAM2 contract tests.

This module uses lazy imports plus a stubbed material-classifier dependency to
avoid the known local torch import abort during pytest collection while still
exercising the SAM2 backend contract surface directly.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pytest

# pylint: disable=redefined-outer-name

pytestmark = pytest.mark.unit

PINNED_REVISION = "a" * 40
SAM2_REPO_ID = "facebook/sam2.1-hiera-large"
PINNED_SAM21_LARGE_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
PINNED_SAM21_LARGE_SHA256 = "2647878d5dfa5098f2f8649825738a9345572bae2d4350a2468587ece47dd318"


@pytest.fixture
def checkpoint_path(tmp_path: Any) -> str:
    checkpoint = tmp_path / "sam2_hiera_large.pt"
    checkpoint.write_bytes(b"stub")
    return str(checkpoint)


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


@pytest.fixture
def segmentation_surface(monkeypatch: pytest.MonkeyPatch) -> tuple[Any, Any]:
    _install_material_classifier_stub(monkeypatch)
    backend_module = importlib.import_module("transformation_portal.spatial_ai.segmentation.sam2_backend")
    contracts_module = importlib.import_module("transformation_portal.spatial_ai.segmentation.contracts")
    backend_module = importlib.reload(backend_module)
    contracts_module = importlib.reload(contracts_module)
    return backend_module.SAM2Backend, contracts_module.SegmentationInput


def _auto_input(SegmentationInput: Any) -> Any:
    return SegmentationInput(
        image=np.zeros((4, 4, 3), dtype=np.float32),
        gamma=1.0,
        mode="auto",
    )


def _points_input(SegmentationInput: Any) -> Any:
    return SegmentationInput(
        image=np.zeros((4, 4, 3), dtype=np.float32),
        gamma=1.0,
        mode="points",
        prompts={"points": [[1.0, 1.0]], "labels": [1]},
    )


def test_hf_opt_in_requires_pinned_revision(segmentation_surface: tuple[Any, Any]) -> None:
    SAM2Backend, _ = segmentation_surface
    with pytest.raises(ValueError, match="pinned revision|40-char commit SHA|unpinned"):
        SAM2Backend(
            model_size="large",
            device="cpu",
            repo_id=SAM2_REPO_ID,
            prefer_hf_pipeline=True,
        )


def test_hf_opt_in_rejects_unpinned_revision(segmentation_surface: tuple[Any, Any]) -> None:
    SAM2Backend, _ = segmentation_surface
    with pytest.raises(ValueError, match="40-char commit SHA|unpinned"):
        SAM2Backend(
            model_size="large",
            device="cpu",
            repo_id=SAM2_REPO_ID,
            revision="main",
            prefer_hf_pipeline=True,
        )


def test_repo_id_metadata_alone_does_not_disable_checkpoint_enforcement(
    segmentation_surface: tuple[Any, Any], tmp_path: Any
) -> None:
    SAM2Backend, _ = segmentation_surface
    missing_checkpoint = tmp_path / "missing_sam2_hiera_large.pt"
    with pytest.raises(FileNotFoundError, match="SAM2 checkpoint not found"):
        SAM2Backend(
            model_size="large",
            device="cpu",
            checkpoint_path=str(missing_checkpoint),
            repo_id=SAM2_REPO_ID,
            revision=PINNED_REVISION,
            prefer_hf_pipeline=False,
        )


def test_hf_loader_reuses_pipeline_components_when_available(
    segmentation_surface: tuple[Any, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    SAM2Backend, _ = segmentation_surface
    backend = SAM2Backend(
        model_size="large",
        device="cpu",
        repo_id=SAM2_REPO_ID,
        revision=PINNED_REVISION,
        prefer_hf_pipeline=True,
    )
    fake_model = SimpleNamespace(to=lambda device: fake_model, eval=lambda: None)
    fake_processor = SimpleNamespace(post_process_masks=lambda *args, **kwargs: [])
    fake_pipeline_instance = SimpleNamespace(model=fake_model, image_processor=fake_processor)
    pipeline_calls: list[tuple[str, str]] = []

    transformers_module = ModuleType("transformers")

    def fake_pipeline(task: str, **kwargs: Any) -> Any:
        assert task == "mask-generation"
        assert kwargs["model"] == SAM2_REPO_ID
        assert kwargs["revision"] == PINNED_REVISION
        pipeline_calls.append((kwargs["model"], kwargs["revision"]))
        return fake_pipeline_instance

    class _ForbiddenSam2Model:
        @staticmethod
        def from_pretrained(*args: Any, **kwargs: Any) -> Any:
            raise AssertionError("Sam2Model.from_pretrained should not be called when pipeline exposes the model")

    class _ForbiddenSam2Processor:
        @staticmethod
        def from_pretrained(*args: Any, **kwargs: Any) -> Any:
            raise AssertionError("Sam2Processor.from_pretrained should not be called when pipeline exposes the processor")

    setattr(transformers_module, "pipeline", fake_pipeline)
    setattr(transformers_module, "Sam2Model", _ForbiddenSam2Model)
    setattr(transformers_module, "Sam2Processor", _ForbiddenSam2Processor)
    monkeypatch.setitem(sys.modules, "transformers", transformers_module)

    backend._load_huggingface_path()

    assert pipeline_calls == [(SAM2_REPO_ID, PINNED_REVISION)]
    assert backend._hf_mask_generator is fake_pipeline_instance
    assert backend._hf_model is fake_model
    assert backend._hf_processor is fake_processor


def test_video_mode_uses_cached_repo_checkpoint_without_loading_image_pipeline(
    segmentation_surface: tuple[Any, Any], tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    SAM2Backend, SegmentationInput = segmentation_surface
    cached_checkpoint = tmp_path / "sam2.1_hiera_large.pt"
    cached_checkpoint.write_bytes(b"stub-checkpoint")
    video_dir = tmp_path / "video_frames"
    video_dir.mkdir()
    (video_dir / "00000.jpg").write_bytes(b"frame")

    hub_module = ModuleType("huggingface_hub")
    hub_module.try_to_load_from_cache = lambda *, repo_id=None, filename=None, revision=None: str(cached_checkpoint)
    hub_module.hf_hub_download = lambda **_kwargs: (_ for _ in ()).throw(
        AssertionError("hf_hub_download should not run when cached checkpoint exists")
    )
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub_module)

    build_calls: list[dict[str, Any]] = []

    class _FakeVideoPredictor:
        def init_state(self, **kwargs: Any) -> dict[str, Any]:
            assert kwargs["video_path"] == str(video_dir)
            return {"num_frames": 3, "video_height": 8, "video_width": 8}

        def add_new_points(self, **kwargs: Any) -> tuple[None, np.ndarray, np.ndarray]:
            assert kwargs["frame_idx"] == 0
            assert kwargs["obj_id"] == 1
            return None, np.array([1], dtype=np.int32), np.ones((1, 8, 8), dtype=np.float32)

        def propagate_in_video(self, inference_state: dict[str, Any]):
            del inference_state
            yield 0, np.array([1], dtype=np.int32), np.ones((1, 8, 8), dtype=np.float32)
            yield 1, np.array([1], dtype=np.int32), np.zeros((1, 8, 8), dtype=np.float32)

        def reset_state(self, inference_state: dict[str, Any]) -> None:
            del inference_state

    sam2_module = ModuleType("sam2")
    build_module = ModuleType("sam2.build_sam")

    def _build_video_predictor(*, config_file: str, ckpt_path: str, device: str) -> _FakeVideoPredictor:
        build_calls.append({"config_file": config_file, "ckpt_path": ckpt_path, "device": device})
        return _FakeVideoPredictor()

    build_module.build_sam2_video_predictor = _build_video_predictor
    sam2_module.build_sam = build_module
    monkeypatch.setitem(sys.modules, "sam2", sam2_module)
    monkeypatch.setitem(sys.modules, "sam2.build_sam", build_module)
    monkeypatch.setattr(
        "transformation_portal.spatial_ai.segmentation.sam2_backend._compute_file_sha256",
        lambda path: PINNED_SAM21_LARGE_SHA256,
    )

    backend = SAM2Backend(
        model_size="large",
        device="cpu",
        repo_id=SAM2_REPO_ID,
        revision=PINNED_REVISION,
        prefer_hf_pipeline=True,
    )
    monkeypatch.setattr(
        backend,
        "_load_model",
        lambda: (_ for _ in ()).throw(AssertionError("_load_model should be skipped for video mode")),
    )

    seg_input = SegmentationInput(
        image=None,
        gamma=1.0,
        mode="video",
        video_path=str(video_dir),
        prompts={"frame_idx": 0, "object_id": 1, "points": [[2, 3]], "labels": [1]},
    )

    result = backend.segment(seg_input)

    assert build_calls == [
        {
            "config_file": backend.MODEL_CONFIGS["large"],
            "ckpt_path": str(cached_checkpoint),
            "device": "cpu",
        }
    ]
    assert result.masks.shape == (3, 8, 8)
    assert result.temporal_ids.tolist() == [1, 1, 1]
    assert result.metadata[0].is_empty is False
    assert result.metadata[1].is_empty is True
    assert result.metadata[2].is_empty is True
    assert int(result.masks[1].sum()) == 0
    assert int(result.masks[2].sum()) == 0


def test_clone_for_device_preserves_loading_contract(segmentation_surface: tuple[Any, Any], checkpoint_path: str) -> None:
    SAM2Backend, _ = segmentation_surface
    backend = SAM2Backend(
        model_size="large",
        device="cpu",
        checkpoint_path=checkpoint_path,
        repo_id=SAM2_REPO_ID,
        revision=PINNED_REVISION,
        prefer_hf_pipeline=False,
        generator_kwargs={"points_per_batch": 64, "pred_iou_thresh": 0.88},
    )

    clone = backend.clone_for_device("cpu")

    assert clone.model_size == backend.model_size
    assert clone.device == "cpu"
    assert clone.checkpoint_path == backend.checkpoint_path
    assert clone.repo_id == SAM2_REPO_ID
    assert clone.revision == PINNED_REVISION
    assert clone.prefer_hf_pipeline is False
    assert clone.generator_kwargs == {"points_per_batch": 64, "pred_iou_thresh": 0.88}


def test_large_defaults_match_apex_research_ultra_pin(segmentation_surface: tuple[Any, Any]) -> None:
    """The canonical SAM 2.1 large defaults must stay aligned with the governed preset."""
    import yaml

    SAM2Backend, _ = segmentation_surface

    preset_path = Path(__file__).resolve().parents[3] / "config" / "presets" / "experimental" / "apex_research_ultra.yaml"
    preset = yaml.safe_load(preset_path.read_text(encoding="utf-8"))
    model_block = preset["segmentation"]["model"]

    assert SAM2Backend.DEFAULT_CHECKPOINTS["large"] == "sam2.1_hiera_large.pt"
    assert SAM2Backend.MODEL_CONFIGS["large"] == PINNED_SAM21_LARGE_CONFIG
    assert SAM2Backend.CHECKPOINT_SHA256["large"] == PINNED_SAM21_LARGE_SHA256
    assert model_block["checkpoint"] == f"checkpoints/{SAM2Backend.DEFAULT_CHECKPOINTS['large']}"
    assert model_block["config"] == SAM2Backend.MODEL_CONFIGS["large"]
    assert model_block["expected_sha256"] == SAM2Backend.CHECKPOINT_SHA256["large"]


def test_model_configs_lock_the_intentional_base_vs_large_split(segmentation_surface: tuple[Any, Any]) -> None:
    """Base stays on the legacy Hydra short name while large uses the SAM 2.1 path."""
    SAM2Backend, _ = segmentation_surface

    assert SAM2Backend.MODEL_CONFIGS["base"] == "sam2_hiera_b+"
    assert SAM2Backend.MODEL_CONFIGS["large"] == PINNED_SAM21_LARGE_CONFIG


def test_extract_predictions_normalizes_2d_masks_and_bad_score_shapes(
    segmentation_surface: tuple[Any, Any], checkpoint_path: str
) -> None:
    SAM2Backend, _ = segmentation_surface
    backend = SAM2Backend(model_size="large", device="cpu", checkpoint_path=checkpoint_path)

    output = SimpleNamespace(
        pred_masks=np.array([[1, 0], [0, 1]], dtype=np.float32),
        iou_predictions=np.array([0.1, 0.9], dtype=np.float32),
        stability_scores=None,
    )

    masks, iou_scores, stability_scores = backend._extract_sam2_predictions(output)

    assert masks.shape == (1, 2, 2)
    assert masks.dtype == bool
    assert iou_scores.shape == (1,)
    assert stability_scores.shape == (1,)
    assert float(iou_scores[0]) == pytest.approx(1.0)
    assert float(stability_scores[0]) == pytest.approx(1.0)


def test_auto_mode_returns_empty_contract_when_no_masks_found(
    segmentation_surface: tuple[Any, Any], checkpoint_path: str
) -> None:
    SAM2Backend, SegmentationInput = segmentation_surface
    backend = SAM2Backend(model_size="large", device="cpu", checkpoint_path=checkpoint_path)
    backend._mask_generator = SimpleNamespace(generate=lambda image_uint8: [])  # type: ignore[assignment]

    result = backend._segment_auto(_auto_input(SegmentationInput))

    assert result.masks.shape == (0, 4, 4)
    assert result.scores.shape == (0,)
    assert result.metadata == []


def test_prompted_mode_accepts_tuple_predictor_output_and_orders_scores(
    segmentation_surface: tuple[Any, Any], checkpoint_path: str
) -> None:
    SAM2Backend, SegmentationInput = segmentation_surface
    backend = SAM2Backend(model_size="large", device="cpu", checkpoint_path=checkpoint_path)

    class FakePredictor:
        def set_image(self, image_uint8: np.ndarray) -> None:
            assert image_uint8.shape == (4, 4, 3)

        def predict(self, **kwargs: Any) -> tuple[np.ndarray, np.ndarray]:
            assert kwargs["multimask_output"] is True
            masks = np.array(
                [
                    [[1, 0], [0, 0]],
                    [[1, 1], [1, 0]],
                ],
                dtype=bool,
            )
            scores = np.array([0.2, 0.9], dtype=np.float32)
            return masks, scores

    backend._image_predictor = FakePredictor()

    result = backend._segment_prompted(_points_input(SegmentationInput))

    assert result.masks.shape == (2, 2, 2)
    assert result.scores.tolist() == pytest.approx([0.9, 0.2])
    assert [meta.area for meta in result.metadata] == [3, 1]


def test_hf_prompted_points_keep_sam2_processor_nesting(
    segmentation_surface: tuple[Any, Any], checkpoint_path: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    SAM2Backend, SegmentationInput = segmentation_surface
    backend = SAM2Backend(model_size="large", device="cpu", checkpoint_path=checkpoint_path)

    class _NoGrad:
        def __enter__(self) -> None:
            return None

        def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
            del exc_type, exc, tb
            return False

    class FakeTensor:
        def __init__(self, value: Any) -> None:
            self.value = value

        def to(self, device: Any) -> Any:
            del device
            return self

        def cpu(self) -> Any:
            return self

    processor_calls: list[dict[str, Any]] = []

    class FakeProcessor:
        def __call__(self, **kwargs: Any) -> dict[str, Any]:
            processor_calls.append(kwargs)
            return {
                "original_sizes": FakeTensor([(4, 4)]),
                "reshaped_input_sizes": FakeTensor([(4, 4)]),
            }

        def post_process_masks(self, pred_masks: Any, original_sizes: Any) -> list[np.ndarray]:
            del pred_masks, original_sizes
            return [np.array([[[1, 0], [0, 0]]], dtype=np.float32)]

    class FakeModel:
        device = "cpu"

        def __call__(self, **kwargs: Any) -> Any:
            del kwargs
            return SimpleNamespace(
                pred_masks=FakeTensor(np.array([[[[1, 0], [0, 0]]]], dtype=np.float32)),
                iou_scores=np.array([[0.9]], dtype=np.float32),
            )

    backend._hf_model = FakeModel()
    backend._hf_processor = FakeProcessor()

    def _no_grad() -> _NoGrad:
        return _NoGrad()

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(no_grad=_no_grad))

    result = backend._segment_prompted(_points_input(SegmentationInput))

    assert result.masks.shape == (1, 2, 2)
    assert processor_calls
    assert processor_calls[0]["input_points"] == [[[[1.0, 1.0]]]]
    assert processor_calls[0]["input_labels"] == [[[1]]]


def test_prompted_mode_returns_empty_result_for_zero_area_masks(
    segmentation_surface: tuple[Any, Any], checkpoint_path: str
) -> None:
    SAM2Backend, SegmentationInput = segmentation_surface
    backend = SAM2Backend(model_size="large", device="cpu", checkpoint_path=checkpoint_path)

    class FakePredictor:
        def set_image(self, image_uint8: np.ndarray) -> None:
            assert image_uint8.shape == (4, 4, 3)

        def predict(self, **kwargs: Any) -> tuple[np.ndarray, np.ndarray]:
            assert kwargs["multimask_output"] is True
            masks = np.zeros((1, 2, 2), dtype=bool)
            scores = np.array([0.3], dtype=np.float32)
            return masks, scores

    backend._image_predictor = FakePredictor()

    result = backend._segment_prompted(_points_input(SegmentationInput))

    assert result.masks.shape == (0, 4, 4)
    assert result.scores.shape == (0,)
    assert result.metadata == []
