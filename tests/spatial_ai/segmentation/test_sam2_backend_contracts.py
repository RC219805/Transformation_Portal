"""Focused SAM2 contract tests.

This module uses lazy imports plus a stubbed material-classifier dependency to
avoid the known local torch import abort during pytest collection while still
exercising the SAM2 backend contract surface directly.
"""

from __future__ import annotations

import importlib
import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pytest

# pylint: disable=redefined-outer-name

pytestmark = pytest.mark.unit

PINNED_REVISION = "a" * 40
SAM2_REPO_ID = "facebook/sam2.1-hiera-large"


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
    with pytest.raises(ValueError, match="pinned revision"):
        SAM2Backend(
            model_size="large",
            device="cpu",
            repo_id=SAM2_REPO_ID,
            prefer_hf_pipeline=True,
        )


def test_repo_id_metadata_alone_does_not_disable_checkpoint_enforcement(segmentation_surface: tuple[Any, Any]) -> None:
    SAM2Backend, _ = segmentation_surface
    with pytest.raises(FileNotFoundError, match="SAM2 checkpoint not found"):
        SAM2Backend(
            model_size="large",
            device="cpu",
            repo_id=SAM2_REPO_ID,
            revision=PINNED_REVISION,
            prefer_hf_pipeline=False,
        )


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
