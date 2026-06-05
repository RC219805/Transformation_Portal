"""Unit tests for legacy segmentation adapter contracts."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

torch = pytest.importorskip("torch", reason="legacy segmentation adapters require torch")

from transformation_portal.segmentation import CLIPClassifier, MaterialSegmenter, SAMSegmenter
from transformation_portal.segmentation.material_segmenter import MaterialSegment

pytestmark = [pytest.mark.unit, pytest.mark.ml]


def _clip_classifier_without_model() -> CLIPClassifier:
    return CLIPClassifier.__new__(CLIPClassifier)


def _sam_segmenter_without_model() -> SAMSegmenter:
    return SAMSegmenter.__new__(SAMSegmenter)


def _sample_rgb_image() -> np.ndarray:
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    image[:2, :2] = (200, 100, 50)
    image[2:, 2:] = (20, 120, 220)
    return image


def _mask(rows: slice, cols: slice, shape: tuple[int, int] = (4, 4)) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask[rows, cols] = True
    return mask


def _material_segment(
    material: str,
    mask: np.ndarray,
    *,
    confidence: float = 0.8,
    area: int | None = None,
) -> MaterialSegment:
    return MaterialSegment(
        mask=mask,
        material=material,
        confidence=confidence,
        area=int(mask.sum()) if area is None else area,
        bbox=(0, 0, mask.shape[1], mask.shape[0]),
        centroid=(1, 1),
        properties={"source": "unit-test"},
    )


def test_package_exports_legacy_segmentation_classes() -> None:
    import transformation_portal.segmentation as segmentation

    assert segmentation.__all__ == [
        "SAMSegmenter",
        "CLIPClassifier",
        "MaterialSegmenter",
    ]
    assert segmentation.SAMSegmenter is SAMSegmenter
    assert segmentation.CLIPClassifier is CLIPClassifier
    assert segmentation.MaterialSegmenter is MaterialSegmenter


class _FakeCLIPInputs(dict):
    def to(self, device: str) -> "_FakeCLIPInputs":
        self["device"] = device
        return self


class _FakeCLIPProcessor:
    def __call__(
        self,
        *,
        text: list[str],
        images: Image.Image,
        return_tensors: str,
        padding: bool,
    ) -> _FakeCLIPInputs:
        return _FakeCLIPInputs(
            text=text,
            image_size=images.size,
            return_tensors=return_tensors,
            padding=padding,
        )


class _FakeCLIPModel:
    def __call__(self, **inputs: object) -> SimpleNamespace:
        categories = inputs["text"]
        assert isinstance(categories, list)
        logits = torch.arange(len(categories), dtype=torch.float32).unsqueeze(0)
        return SimpleNamespace(logits_per_image=logits)


def test_clip_classify_image_uses_processor_model_and_temperature() -> None:
    classifier = _clip_classifier_without_model()
    classifier.processor = _FakeCLIPProcessor()
    classifier.model = _FakeCLIPModel()
    classifier.device = "cpu"

    probabilities = classifier.classify_image(
        _sample_rgb_image(),
        ["stone", "wood", "glass"],
        temperature=2.0,
    )

    expected = torch.nn.functional.softmax(torch.tensor([[0.0, 0.5, 1.0]]), dim=1).numpy()[0]
    np.testing.assert_allclose(probabilities, expected)
    assert probabilities.sum() == pytest.approx(1.0)


def test_clip_loads_images_from_supported_inputs_and_rejects_unknown_type(tmp_path: Path) -> None:
    classifier = _clip_classifier_without_model()
    image = _sample_rgb_image()
    pil_image = Image.fromarray(image, mode="RGB")
    image_path = tmp_path / "sample.png"
    pil_image.save(image_path)

    assert classifier._load_image(pil_image) is pil_image
    assert classifier._load_image(image).size == (4, 4)
    assert classifier._load_image(image_path).mode == "RGB"
    np.testing.assert_array_equal(classifier._load_image_np(image_path), image)

    with pytest.raises(ValueError, match="Unsupported image type"):
        classifier._load_image(object())


def test_clip_extracts_masked_regions_with_background_color() -> None:
    classifier = _clip_classifier_without_model()
    image = _sample_rgb_image()
    mask = _mask(slice(0, 2), slice(0, 2))

    masked = classifier._extract_masked_region(image, mask, background_color=(1, 2, 3))

    assert masked.dtype == np.uint8
    np.testing.assert_array_equal(masked[0, 0], image[0, 0])
    np.testing.assert_array_equal(masked[3, 3], np.array([1, 2, 3], dtype=np.uint8))


def test_clip_classifies_segments_and_finds_target_material_regions() -> None:
    classifier = _clip_classifier_without_model()
    calls: list[tuple[np.ndarray, list[str], float]] = []

    def fake_classify_image(image: np.ndarray, categories: list[str], temperature: float = 1.0) -> np.ndarray:
        calls.append((image, categories, temperature))
        return np.array([0.7, 0.2, 0.1], dtype=np.float32)[: len(categories)]

    classifier.classify_image = fake_classify_image
    masks = [
        _mask(slice(0, 2), slice(0, 2)),
        _mask(slice(2, 4), slice(2, 4)),
    ]

    results = classifier.classify_segments(
        _sample_rgb_image(),
        masks,
        ["marble", "wood", "glass"],
        background_color=(9, 8, 7),
        temperature=0.5,
    )

    assert [result["mask_index"] for result in results] == [0, 1]
    assert [result["top_category"] for result in results] == ["marble", "marble"]
    assert results[0]["confidence"] == pytest.approx(0.7)
    assert results[0]["all_categories"] == {
        "marble": pytest.approx(0.7),
        "wood": pytest.approx(0.2),
        "glass": pytest.approx(0.1),
    }
    assert len(calls) == 2
    assert all(call[2] == 0.5 for call in calls)

    assert classifier.find_material_regions(_sample_rgb_image(), masks, "marble", threshold=0.65) == [0, 1]


def test_clip_category_helpers_and_semantic_map_use_classification_contract() -> None:
    classifier = _clip_classifier_without_model()

    def fake_classify_image(image: object, categories: list[str], temperature: float = 1.0) -> np.ndarray:
        del image, temperature
        scores = np.linspace(0.1, 0.9, num=len(categories), dtype=np.float32)
        return scores / scores.sum()

    classifier.classify_image = fake_classify_image

    material_scores = classifier.classify_materials(_sample_rgb_image(), custom_materials=["marble", "wood"])
    assert set(material_scores) == {"marble", "wood"}
    assert material_scores["wood"] > material_scores["marble"]

    room_scores = classifier.classify_room_type(_sample_rgb_image())
    style_scores = classifier.classify_style(_sample_rgb_image())
    assert set(room_scores) == set(CLIPClassifier.ROOM_CATEGORIES)
    assert set(style_scores) == set(CLIPClassifier.STYLE_CATEGORIES)

    features = classifier.detect_features(_sample_rgb_image(), threshold=0.13)
    assert features == sorted(features, key=lambda item: item[1], reverse=True)
    assert all(score >= 0.13 for _, score in features)

    classifier.classify_segments = lambda image, mask_arrays, categories: [
        {"top_category": "wood"},
        {"top_category": "stone"},
        {"top_category": "wood"},
    ]
    masks = [
        {"segmentation": _mask(slice(0, 1), slice(0, 1))},
        {"segmentation": _mask(slice(1, 3), slice(1, 3))},
        {"segmentation": _mask(slice(3, 4), slice(3, 4))},
    ]

    semantic_map, labels = classifier.create_semantic_map(_sample_rgb_image(), masks, ["wood", "stone"])

    assert labels == {1: "wood", 2: "stone"}
    assert semantic_map[0, 0] == 1
    assert semantic_map[1, 1] == 2
    assert semantic_map[3, 3] == 1


def test_sam_init_fails_closed_when_optional_runtime_or_checkpoint_is_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import transformation_portal.segmentation.sam_segmenter as sam_module

    monkeypatch.setattr(sam_module, "SAM_AVAILABLE", False)
    with pytest.raises(ImportError, match="SAM required"):
        SAMSegmenter()

    monkeypatch.setattr(sam_module, "SAM_AVAILABLE", True)
    missing_checkpoint = tmp_path / "missing.pth"
    with pytest.raises(FileNotFoundError, match="SAM checkpoint not found"):
        SAMSegmenter(checkpoint_path=missing_checkpoint)


def test_sam_detects_device_preference(monkeypatch: pytest.MonkeyPatch) -> None:
    segmenter = _sam_segmenter_without_model()

    monkeypatch.setattr("transformation_portal.segmentation.sam_segmenter.torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("transformation_portal.segmentation.sam_segmenter.torch.backends.mps.is_available", lambda: False)
    assert segmenter._detect_device() == "cuda"

    monkeypatch.setattr("transformation_portal.segmentation.sam_segmenter.torch.cuda.is_available", lambda: False)
    monkeypatch.setattr("transformation_portal.segmentation.sam_segmenter.torch.backends.mps.is_available", lambda: True)
    assert segmenter._detect_device() == "mps"

    monkeypatch.setattr("transformation_portal.segmentation.sam_segmenter.torch.backends.mps.is_available", lambda: False)
    assert segmenter._detect_device() == "cpu"


def test_sam_finds_checkpoint_in_governed_search_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    segmenter = _sam_segmenter_without_model()
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    checkpoint = checkpoint_dir / SAMSegmenter.MODEL_CHECKPOINTS["vit_b"]
    checkpoint.write_bytes(b"checkpoint")

    monkeypatch.chdir(tmp_path)

    assert segmenter._find_checkpoint("vit_b").resolve() == checkpoint.resolve()

    with pytest.raises(FileNotFoundError, match="SAM checkpoint 'sam_vit_l_0b3195.pth' not found"):
        segmenter._find_checkpoint("vit_l")


def test_sam_loads_rgb_images_from_numpy_pil_and_paths(tmp_path: Path) -> None:
    segmenter = _sam_segmenter_without_model()
    gray = np.array([[0, 127], [200, 255]], dtype=np.uint8)
    rgba = np.zeros((2, 2, 4), dtype=np.uint8)
    rgba[..., :3] = (10, 20, 30)
    rgba[..., 3] = 255
    rgb = _sample_rgb_image()
    image_path = tmp_path / "rgb.png"
    Image.fromarray(rgb, mode="RGB").save(image_path)

    assert segmenter._load_image_rgb(gray).shape == (2, 2, 3)
    np.testing.assert_array_equal(segmenter._load_image_rgb(rgba), np.full((2, 2, 3), (10, 20, 30), dtype=np.uint8))
    assert segmenter._load_image_rgb(rgb) is rgb
    np.testing.assert_array_equal(segmenter._load_image_rgb(Image.fromarray(rgb, mode="RGB")), rgb)
    np.testing.assert_array_equal(segmenter._load_image_rgb(image_path), rgb)

    with pytest.raises(ValueError, match="Unsupported image type"):
        segmenter._load_image_rgb(object())


class _FakeMaskGenerator:
    def __init__(self, masks: list[dict[str, object]]) -> None:
        self._masks = masks

    def generate(self, image: np.ndarray) -> list[dict[str, object]]:
        assert image.shape == (4, 4, 3)
        return list(self._masks)


def test_sam_segment_automatic_filters_sorts_and_limits_masks() -> None:
    segmenter = _sam_segmenter_without_model()
    masks = [
        {"segmentation": _mask(slice(0, 1), slice(0, 1)), "area": 1, "predicted_iou": 0.5, "stability_score": 0.7},
        {"segmentation": _mask(slice(0, 2), slice(0, 2)), "area": 4, "predicted_iou": 0.9, "stability_score": 0.8},
        {"segmentation": _mask(slice(2, 4), slice(2, 4)), "area": 3, "predicted_iou": 0.8, "stability_score": 0.6},
    ]
    segmenter.mask_generator = _FakeMaskGenerator(masks)

    filtered = segmenter.segment_automatic(_sample_rgb_image(), min_area=2, max_masks=1)

    assert [mask["area"] for mask in filtered] == [4]
    assert segmenter.segment_automatic(_sample_rgb_image(), filter_by_area=False)[0]["area"] == 4


class _FakeSAMPredictor:
    def __init__(self) -> None:
        self.images: list[np.ndarray] = []
        self.calls: list[dict[str, object]] = []

    def set_image(self, image: np.ndarray) -> None:
        self.images.append(image)

    def predict(self, **kwargs: object) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        self.calls.append(kwargs)
        masks = np.stack(
            [
                _mask(slice(0, 1), slice(0, 1)),
                _mask(slice(1, 3), slice(1, 3)),
                _mask(slice(3, 4), slice(3, 4)),
            ]
        )
        scores = np.array([0.1, 0.9, 0.3], dtype=np.float32)
        logits = np.zeros((3, 4, 4), dtype=np.float32)
        return masks, scores, logits


def test_sam_prompted_segmentation_uses_points_and_boxes() -> None:
    segmenter = _sam_segmenter_without_model()
    segmenter.predictor = _FakeSAMPredictor()

    best_mask = segmenter.segment_from_points(_sample_rgb_image(), [[1, 1], [2, 2]], multimask_output=True)
    np.testing.assert_array_equal(best_mask, _mask(slice(1, 3), slice(1, 3)))
    point_call = segmenter.predictor.calls[-1]
    np.testing.assert_array_equal(point_call["point_labels"], np.ones(2))

    first_mask = segmenter.segment_from_points(_sample_rgb_image(), [[1, 1]], point_labels=[0], multimask_output=False)
    np.testing.assert_array_equal(first_mask, _mask(slice(0, 1), slice(0, 1)))
    np.testing.assert_array_equal(segmenter.predictor.calls[-1]["point_labels"], np.array([0]))

    box_mask = segmenter.segment_from_box(_sample_rgb_image(), [0, 0, 2, 2])
    np.testing.assert_array_equal(box_mask, _mask(slice(0, 1), slice(0, 1)))
    np.testing.assert_array_equal(segmenter.predictor.calls[-1]["box"], np.array([0, 0, 2, 2]))


def test_sam_mask_helpers_create_deterministic_contract_outputs() -> None:
    segmenter = _sam_segmenter_without_model()
    masks = [
        {
            "segmentation": _mask(slice(0, 2), slice(0, 2)),
            "area": 4,
            "predicted_iou": 0.8,
            "stability_score": 0.9,
        },
        {
            "segmentation": _mask(slice(2, 4), slice(2, 4)),
            "area": 6,
            "predicted_iou": 0.6,
            "stability_score": 0.7,
        },
    ]

    overlay = segmenter.create_colored_mask_overlay(_sample_rgb_image(), masks, alpha=0.25)
    assert overlay.shape == (4, 4, 3)
    assert overlay.dtype == np.uint8
    assert not np.array_equal(overlay[0, 0], _sample_rgb_image()[0, 0])

    assert [mask["area"] for mask in segmenter.extract_largest_segments(masks, n=1)] == [6]
    merged = segmenter.merge_masks([masks[0]["segmentation"], masks[1]["segmentation"]])
    assert merged.dtype == bool
    assert merged.sum() == 8
    assert segmenter.merge_masks([masks[0]["segmentation"]], image_shape=(4, 4)).shape == (4, 4)

    empty_stats = segmenter.get_mask_statistics([])
    assert empty_stats == {
        "num_masks": 0,
        "total_area": 0,
        "avg_area": 0,
        "median_area": 0,
        "avg_iou": 0,
    }
    stats = segmenter.get_mask_statistics(masks)
    assert stats["num_masks"] == 2
    assert stats["total_area"] == 10
    assert stats["avg_iou"] == pytest.approx(0.7)
    assert stats["avg_stability"] == pytest.approx(0.8)


class _FakeSAMForMaterials:
    def segment_automatic(self, image: object, *, min_area: int, max_masks: int) -> list[dict[str, object]]:
        assert min_area == 2
        assert max_masks == 3
        return [
            {
                "segmentation": _mask(slice(0, 2), slice(0, 2)),
                "area": 4,
                "bbox": (0, 0, 2, 2),
                "predicted_iou": 0.91,
                "stability_score": 0.93,
            },
            {
                "segmentation": _mask(slice(2, 3), slice(2, 3)),
                "area": 1,
                "bbox": (2, 2, 1, 1),
                "predicted_iou": 0.7,
                "stability_score": 0.75,
            },
            {
                "segmentation": np.zeros((4, 4), dtype=bool),
                "area": 0,
                "bbox": (0, 0, 0, 0),
                "predicted_iou": 0.5,
                "stability_score": 0.5,
            },
        ]


class _FakeCLIPForMaterials:
    MATERIAL_CATEGORIES = ["marble", "wood", "glass"]

    def classify_segments(
        self,
        image: object,
        masks: list[np.ndarray],
        materials: list[str],
    ) -> list[dict[str, object]]:
        assert materials == ["marble", "wood", "glass"]
        assert len(masks) == 3
        return [
            {"top_category": "marble", "confidence": 0.9, "all_categories": {"marble": 0.9, "wood": 0.1}},
            {"top_category": "wood", "confidence": 0.2, "all_categories": {"wood": 0.2}},
            {"top_category": "glass", "confidence": 0.8, "all_categories": {"glass": 0.8}},
        ]


def test_material_segmenter_combines_fake_sam_and_clip_segments() -> None:
    segmenter = MaterialSegmenter(
        sam_segmenter=_FakeSAMForMaterials(),
        clip_classifier=_FakeCLIPForMaterials(),
    )

    segments = segmenter.segment_materials(
        _sample_rgb_image(),
        min_segment_area=2,
        max_segments=3,
        confidence_threshold=0.3,
    )

    assert [segment.material for segment in segments] == ["marble", "glass"]
    assert segments[0].centroid == (0, 0)
    assert segments[1].centroid == (0, 0)
    assert segments[0].properties == {
        "predicted_iou": 0.91,
        "stability_score": 0.93,
        "all_material_probs": {"marble": 0.9, "wood": 0.1},
    }


def test_material_segmenter_maps_masks_recommendations_and_statistics() -> None:
    segmenter = MaterialSegmenter(
        sam_segmenter=_FakeSAMForMaterials(),
        clip_classifier=_FakeCLIPForMaterials(),
    )
    marble = _material_segment("marble", _mask(slice(0, 2), slice(0, 2)), confidence=0.9)
    wood = _material_segment("wood", _mask(slice(2, 4), slice(2, 4)), confidence=0.6, area=8)
    duplicate_wood = _material_segment("WOOD", _mask(slice(0, 1), slice(3, 4)), confidence=0.7, area=1)
    segments = [marble, wood, duplicate_wood]

    assert segmenter.get_material_masks(segments, "wood") == [wood.mask, duplicate_wood.mask]

    material_map, labels = segmenter.create_material_map((4, 4), segments)
    assert labels == {1: "WOOD", 2: "marble", 3: "wood"}
    assert material_map[0, 0] == 2
    assert material_map[2, 2] == 3
    assert material_map[0, 3] == 1

    recommendations = segmenter.get_enhancement_recommendations(segments)
    assert recommendations["materials_detected"][0] == {
        "material": "wood",
        "total_area": 8,
        "num_regions": 1,
    }
    assert recommendations["region_enhancements"]["marble"]["preserve_veining"] is True
    assert recommendations["overall_strategy"]["dominant_material"] == "wood"

    assert segmenter.get_statistics([]) == {"num_segments": 0, "num_materials": 0, "materials": []}
    stats = segmenter.get_statistics(segments)
    assert stats["num_segments"] == 3
    assert stats["num_materials"] == 3
    assert stats["total_area"] == 13
    assert stats["materials"][0]["material"] == "wood"
    assert stats["avg_confidence"] == pytest.approx(0.7333333333333334)


def test_material_segmenter_visualizations_are_shape_stable() -> None:
    segmenter = MaterialSegmenter(
        sam_segmenter=_FakeSAMForMaterials(),
        clip_classifier=_FakeCLIPForMaterials(),
    )
    image = _sample_rgb_image()
    segments = [
        _material_segment("marble", _mask(slice(0, 2), slice(0, 2))),
        _material_segment("glass", _mask(slice(2, 4), slice(2, 4)), confidence=0.95),
    ]

    overlay = segmenter.visualize_materials(image, segments, alpha=0.5)
    assert overlay.shape == image.shape
    assert overlay.dtype == np.uint8
    assert not np.array_equal(overlay[0, 0], image[0, 0])

    labels = segmenter.create_material_labels(image, segments, font_scale=0.3, thickness=1)
    assert labels.shape == image.shape
    assert labels.dtype == np.uint8
    assert not np.shares_memory(labels, image)
