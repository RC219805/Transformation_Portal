"""Unit tests for the pipelines quality-feedback bridge.

Phase 4 coverage. Exercises the deterministic pure-numpy quality
computation surfaces of QualityFeedbackBridge along with the supporting
dataclasses, dependency probes, and pipeline-integration callbacks.
LPIPS / torch are not required: the bridge degrades gracefully to the
heuristic-only mode in core CI.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Generator
from unittest.mock import patch

import numpy as np
import pytest

from transformation_portal.pipelines import quality_feedback_bridge as qfb
from transformation_portal.pipelines.quality_feedback_bridge import (
    HeuristicMetrics,
    MaterialFidelityMetrics,
    PerceptualMetrics,
    QualityFeedbackBridge,
    QualityTargets,
    UnifiedQualityMetrics,
    _check_lpips_available,
    _check_perceptual_assessor_available,
    _check_torch_available,
    _to_jsonable,
    create_quality_callback_for_pipeline,
    create_rag_indexing_callback,
    index_quality_metrics_to_rag,
)

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _reset_dep_caches() -> Generator[None, None, None]:
    """Reset dependency-probe caches and import state around each test."""
    saved = (qfb._TORCH_AVAILABLE, qfb._LPIPS_AVAILABLE, qfb._PERCEPTUAL_ASSESSOR_AVAILABLE)
    saved_path = list(sys.path)
    missing = object()
    saved_modules = {
        name: sys.modules.get(name, missing)
        for name in (
            "enhancements",
            "enhancements.perceptual_quality_assessment",
        )
    }
    qfb._TORCH_AVAILABLE = None
    qfb._LPIPS_AVAILABLE = None
    qfb._PERCEPTUAL_ASSESSOR_AVAILABLE = None
    yield
    qfb._TORCH_AVAILABLE, qfb._LPIPS_AVAILABLE, qfb._PERCEPTUAL_ASSESSOR_AVAILABLE = saved
    sys.path[:] = saved_path
    for name, module in saved_modules.items():
        if module is missing:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module


def _synthetic_image(seed: int = 0, size: int = 64) -> np.ndarray:
    """A reproducible RGB float image in [0, 1]."""
    rng = np.random.default_rng(seed)
    return rng.random((size, size, 3), dtype=np.float32)


class TestToJsonable:
    """Tests for the _to_jsonable helper."""

    def test_passes_primitives_through(self) -> None:
        assert _to_jsonable(1) == 1
        assert _to_jsonable("x") == "x"
        assert _to_jsonable(None) is None

    def test_converts_numpy_scalars(self) -> None:
        assert _to_jsonable(np.float32(0.5)) == pytest.approx(0.5)
        assert _to_jsonable(np.int64(7)) == 7

    def test_converts_paths_to_strings(self, tmp_path: Path) -> None:
        assert _to_jsonable(tmp_path) == str(tmp_path)

    def test_recurses_into_dicts_and_sequences(self) -> None:
        result = _to_jsonable({"a": [np.float32(0.25), {"b": np.int32(2)}], "c": (np.int8(1), "x")})
        assert result == {"a": [0.25, {"b": 2}], "c": [1, "x"]}


class TestDependencyProbes:
    """Tests for the lazy dependency-availability probes."""

    def test_torch_available_caches_first_import_result(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import builtins

        attempts = 0
        original_import = builtins.__import__
        fake_torch = ModuleType("torch")

        def _fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
            nonlocal attempts
            if name == "torch":
                attempts += 1
                return fake_torch
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _fake_import)

        first = _check_torch_available()
        # Second call returns the cached value rather than re-importing.
        assert _check_torch_available() is first
        assert first is True
        assert attempts == 1

    def test_torch_unavailable_when_import_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import builtins

        original_import = builtins.__import__

        def _missing(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "torch":
                raise ImportError("torch not installed")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _missing)
        assert _check_torch_available() is False

    def test_lpips_unavailable_when_import_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import builtins

        original_import = builtins.__import__

        def _missing(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "lpips":
                raise ImportError("lpips not installed")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _missing)
        assert _check_lpips_available() is False

    def test_perceptual_assessor_available_caches_fake_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import builtins

        attempts = 0
        original_import = builtins.__import__
        fake_module = ModuleType("enhancements.perceptual_quality_assessment")
        fake_module.PerceptualQualityAssessor = object

        def _fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
            nonlocal attempts
            if name == "enhancements.perceptual_quality_assessment":
                attempts += 1
                return fake_module
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _fake_import)

        first = _check_perceptual_assessor_available()
        assert _check_perceptual_assessor_available() is first
        assert first is True
        assert attempts == 1

    def test_perceptual_assessor_unavailable_when_import_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import builtins

        original_import = builtins.__import__

        def _missing(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "enhancements.perceptual_quality_assessment":
                raise ImportError("perceptual assessor not installed")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _missing)
        assert _check_perceptual_assessor_available() is False


class TestDataclasses:
    """Tests for the metrics dataclasses."""

    def test_quality_targets_defaults(self) -> None:
        targets = QualityTargets()
        assert targets.material_fidelity_target == pytest.approx(0.98)
        assert "quartzite" in targets.material_thresholds

    def test_heuristic_metrics_to_dict(self) -> None:
        h = HeuristicMetrics(sharpness=0.4, contrast=0.5)
        payload = h.to_dict()
        assert payload["sharpness"] == 0.4
        assert payload["contrast"] == 0.5

    def test_perceptual_metrics_to_dict(self) -> None:
        p = PerceptualMetrics(lpips_score=0.1, ssim_score=0.9)
        payload = p.to_dict()
        assert payload["lpips_score"] == 0.1
        assert payload["ssim_score"] == 0.9

    def test_material_fidelity_to_dict(self) -> None:
        m = MaterialFidelityMetrics(per_material={"oak": 0.97}, overall_fidelity=0.97)
        payload = m.to_dict()
        assert payload == {"per_material": {"oak": 0.97}, "overall_fidelity": 0.97}

    def test_unified_metrics_to_dict_and_to_rag_document(self) -> None:
        metrics = UnifiedQualityMetrics(image_id="img-1", pipeline_config_name="apex")
        payload = metrics.to_dict()
        assert payload["image_id"] == "img-1"
        assert "scores" in payload
        assert "metadata" in payload

        rag_doc = metrics.to_rag_document()
        assert rag_doc["_type"] == "unified_quality_metrics"
        assert rag_doc["_version"] == "1.0.0"
        assert "_indexed_at" in rag_doc


class TestQualityFeedbackBridgeInit:
    """Tests for QualityFeedbackBridge.__init__."""

    def test_defaults(self) -> None:
        bridge = QualityFeedbackBridge()
        assert bridge.hybrid_mode is True
        assert bridge.lpips_network == "alex"
        assert bridge.enable_material_fidelity is True
        assert bridge.rag_callback is None
        assert isinstance(bridge.targets, QualityTargets)

    def test_custom_targets_override(self) -> None:
        targets = QualityTargets(sharpness_target=0.99)
        bridge = QualityFeedbackBridge(targets=targets, hybrid_mode=False, lpips_network="vgg")
        assert bridge.targets.sharpness_target == pytest.approx(0.99)
        assert bridge.hybrid_mode is False
        assert bridge.lpips_network == "vgg"


class TestNumpyAndHeuristics:
    """Tests for the pure-numpy helpers."""

    def test_to_numpy_normalizes_uint8(self) -> None:
        bridge = QualityFeedbackBridge()
        arr = np.full((4, 4, 3), 200, dtype=np.uint8)

        out = bridge._to_numpy(arr)
        assert out.dtype == np.float32
        assert out.max() <= 1.0

    def test_to_numpy_returns_none_for_none(self) -> None:
        assert QualityFeedbackBridge()._to_numpy(None) is None

    def test_to_numpy_accepts_pil_image(self) -> None:
        from PIL import Image

        pil_image = Image.new("RGB", (8, 8), color=(255, 0, 0))
        out = QualityFeedbackBridge()._to_numpy(pil_image)

        assert isinstance(out, np.ndarray)
        assert out.shape[-1] == 3

    def test_compute_heuristic_metrics_returns_unit_range(self) -> None:
        bridge = QualityFeedbackBridge()
        metrics = bridge._compute_heuristic_metrics(_synthetic_image(seed=1))

        assert isinstance(metrics, HeuristicMetrics)
        for value in (
            metrics.sharpness,
            metrics.contrast,
            metrics.colorfulness,
            metrics.exposure_balance,
            metrics.noise_level,
            metrics.overall_score,
        ):
            assert 0.0 <= value <= 1.0

    def test_individual_compute_helpers_are_finite(self) -> None:
        bridge = QualityFeedbackBridge()
        image = _synthetic_image(seed=2)
        for value in (
            bridge._compute_sharpness(image),
            bridge._compute_contrast(image),
            bridge._compute_colorfulness(image),
            bridge._compute_exposure_balance(image),
            bridge._estimate_noise(image),
        ):
            assert np.isfinite(value)
            assert 0.0 <= value <= 1.0


class TestHybridScoreAndTargets:
    """Tests for _compute_hybrid_score, _check_targets, _summarize_targets."""

    def test_hybrid_falls_back_to_heuristic_without_lpips(self) -> None:
        bridge = QualityFeedbackBridge()
        metrics = UnifiedQualityMetrics(heuristic_composite=72.0, lpips_available=False)

        assert bridge._compute_hybrid_score(metrics) == pytest.approx(72.0)

    def test_hybrid_blends_perceptual_and_heuristic(self) -> None:
        bridge = QualityFeedbackBridge()
        metrics = UnifiedQualityMetrics(
            heuristic_composite=60.0,
            perceptual_composite=80.0,
            lpips_available=True,
        )

        # 0.7 * 80 + 0.3 * 60 == 74.0
        assert bridge._compute_hybrid_score(metrics) == pytest.approx(74.0)

    def test_check_targets_heuristic_only(self) -> None:
        bridge = QualityFeedbackBridge()
        metrics = UnifiedQualityMetrics(
            heuristic=HeuristicMetrics(sharpness=0.99, contrast=0.99, colorfulness=0.99, exposure_balance=0.99),
            lpips_available=False,
        )

        targets = bridge._check_targets(metrics)
        assert targets["heuristic_sharpness"] is True
        assert "perceptual_95th" not in targets  # only set when lpips_available

    def test_check_targets_includes_perceptual_when_available(self) -> None:
        bridge = QualityFeedbackBridge()
        metrics = UnifiedQualityMetrics(
            heuristic=HeuristicMetrics(),
            perceptual=PerceptualMetrics(
                lpips_score=0.05,
                lpips_percentile=99.0,
                ssim_score=0.99,
            ),
            material_fidelity=MaterialFidelityMetrics(overall_fidelity=0.99),
            lpips_available=True,
        )

        targets = bridge._check_targets(metrics)
        assert targets["lpips_excellent"] is True
        assert targets["ssim"] is True
        assert targets["material_98pct"] is True

    @pytest.mark.parametrize(
        "met,total,expected_marker",
        [
            (5, 5, "All"),
            (4, 5, "good"),
            (3, 5, "acceptable"),
            (1, 5, "needs improvement"),
        ],
    )
    def test_summarize_targets_thresholds(self, met: int, total: int, expected_marker: str) -> None:
        bridge = QualityFeedbackBridge()
        flags = {f"t{i}": i < met for i in range(total)}

        summary = bridge._summarize_targets(flags)
        assert expected_marker in summary
        # The "all met" case omits the "met/total" fraction; the others include it.
        if met != total:
            assert f"{met}/{total}" in summary


class TestAssess:
    """Tests for QualityFeedbackBridge.assess end-to-end (heuristic-only)."""

    def test_assess_returns_unified_metrics(self) -> None:
        bridge = QualityFeedbackBridge(hybrid_mode=True)
        # Force heuristic-only mode: no perceptual assessor available.
        with patch.object(bridge, "_ensure_perceptual_assessor", return_value=False):
            metrics = bridge.assess(_synthetic_image(), image_id="img-1", pipeline_config_name="apex")

        assert isinstance(metrics, UnifiedQualityMetrics)
        assert metrics.image_id == "img-1"
        assert metrics.pipeline_config_name == "apex"
        assert metrics.lpips_available is False
        assert metrics.processing_time_ms >= 0.0
        assert metrics.targets_summary  # non-empty
        assert metrics.heuristic_composite == pytest.approx(metrics.heuristic.overall_score * 100)

    def test_assess_invokes_rag_callback(self) -> None:
        captured: list[dict[str, Any]] = []

        def _capture(doc: dict[str, Any]) -> None:
            captured.append(doc)

        bridge = QualityFeedbackBridge(rag_callback=_capture)
        with patch.object(bridge, "_ensure_perceptual_assessor", return_value=False):
            bridge.assess(_synthetic_image(), image_id="img-2")

        assert len(captured) == 1
        assert captured[0]["_type"] == "unified_quality_metrics"

    def test_assess_records_warning_when_callback_raises(self) -> None:
        def _exploding_callback(_doc: dict[str, Any]) -> None:
            raise RuntimeError("indexer down")

        bridge = QualityFeedbackBridge(rag_callback=_exploding_callback)
        with patch.object(bridge, "_ensure_perceptual_assessor", return_value=False):
            metrics = bridge.assess(_synthetic_image())

        assert any("RAG callback failed" in w for w in metrics.warnings)

    def test_assess_in_non_hybrid_mode_uses_heuristic_when_lpips_absent(self) -> None:
        bridge = QualityFeedbackBridge(hybrid_mode=False)
        with patch.object(bridge, "_ensure_perceptual_assessor", return_value=False):
            metrics = bridge.assess(_synthetic_image())

        assert metrics.hybrid_score == pytest.approx(metrics.heuristic_composite)


class TestPipelineIntegrationHelpers:
    """Tests for the standalone pipeline-integration callbacks."""

    def test_create_quality_callback_logs_without_rag_path(self) -> None:
        callback = create_quality_callback_for_pipeline("apex")
        # Just exercise it; the callback is a logger sink.
        callback(UnifiedQualityMetrics(image_id="img-x"))

    def test_create_quality_callback_indexes_to_rag(self, tmp_path: Path) -> None:
        index_dir = tmp_path / "rag-index"
        callback = create_quality_callback_for_pipeline("apex", rag_index_path=str(index_dir))
        callback(UnifiedQualityMetrics(image_id="img-y"))

        # The RAG indexer writes one JSON document per call.
        assert any(p.suffix == ".json" for p in index_dir.glob("*"))

    def test_index_quality_metrics_to_rag_writes_document(self, tmp_path: Path) -> None:
        index_dir = tmp_path / "rag-index"
        ok = index_quality_metrics_to_rag(UnifiedQualityMetrics(image_id="img-z"), str(index_dir))

        assert ok is True
        files = list(index_dir.glob("quality_img-z_*.json"))
        assert len(files) == 1
        payload = json.loads(files[0].read_text())
        assert payload["image_id"] == "img-z"

    def test_index_quality_metrics_returns_false_on_failure(self, tmp_path: Path) -> None:
        # An invalid directory path (a regular file) makes mkdir raise.
        invalid = tmp_path / "blocking-file"
        invalid.write_text("not a directory")

        ok = index_quality_metrics_to_rag(
            UnifiedQualityMetrics(image_id="img-fail"),
            str(invalid / "nested"),
        )
        assert ok is False

    def test_create_rag_indexing_callback_noop_without_path(self) -> None:
        callback = create_rag_indexing_callback(index_path=None)
        callback({"image_id": "x"})  # must not raise

    def test_create_rag_indexing_callback_writes_when_path_set(self, tmp_path: Path) -> None:
        index_dir = tmp_path / "unified-index"
        callback = create_rag_indexing_callback(index_path=str(index_dir))

        callback({"image_id": "rag-x"})

        files = list(index_dir.glob("unified_quality_rag-x_*.json"))
        assert len(files) == 1
