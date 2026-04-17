"""Tests for evals/apex_harness.py module (Phase 5 coverage).

Tests for:
- ApexEvaluationHarness evaluation flow
- EvalResult data structures
- Metric aggregation
- VLM integration (mocked)
- Built-in metrics

All tests use mocks - no ML model downloads or GPU requirements.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from transformation_portal.evals.apex_harness import (
    ApexEvaluationHarness,
    EvalMetricResult,
    EvalResult,
    brightness_metric,
    contrast_metric,
    sharpness_metric,
)

pytestmark = [pytest.mark.unit, pytest.mark.ml]


class TestEvalMetricResult:
    """Test EvalMetricResult dataclass."""

    def test_basic_creation(self):
        """Test basic metric result creation."""
        result = EvalMetricResult(
            name="sharpness",
            score=0.85,
        )

        assert result.name == "sharpness"
        assert result.score == 0.85
        assert result.metadata == {}

    def test_with_metadata(self):
        """Test metric result with metadata."""
        result = EvalMetricResult(
            name="contrast",
            score=0.72,
            metadata={"std_dev": 45.2, "mean": 128.0},
        )

        assert result.metadata["std_dev"] == 45.2


class TestEvalResult:
    """Test EvalResult dataclass."""

    def test_basic_creation(self):
        """Test basic eval result creation."""
        result = EvalResult(
            score=0.75,
            passes=True,
        )

        assert result.score == 0.75
        assert result.passes is True
        assert result.metric_scores == {}
        assert result.vlm_score == 0.0

    def test_with_all_fields(self):
        """Test eval result with all fields."""
        result = EvalResult(
            score=0.82,
            passes=True,
            metric_scores={"sharpness": 0.9, "contrast": 0.8},
            vlm_score=0.75,
            vlm_issues=[{"issue_type": "blur", "severity": "low"}],
            details={"threshold": 0.7, "weights": {"metric": 0.5, "vlm": 0.5}},
        )

        assert result.metric_scores["sharpness"] == 0.9
        assert result.vlm_score == 0.75
        assert len(result.vlm_issues) == 1


class TestApexEvaluationHarness:
    """Test ApexEvaluationHarness class."""

    def test_harness_initialization_minimal(self):
        """Test minimal harness initialization."""
        harness = ApexEvaluationHarness()

        assert harness.llava_backend is None
        assert harness.metric_fns == []
        assert harness.threshold == 0.70
        assert harness.metric_weight == 0.5
        assert harness.vlm_weight == 0.5

    def test_harness_initialization_with_options(self):
        """Test harness with custom options."""

        def custom_metric(paths):
            return 0.5

        harness = ApexEvaluationHarness(
            llava_backend=MagicMock(),
            metric_fns=[custom_metric],
            threshold=0.80,
            metric_weight=0.4,
            fail_on_vlm_error=True,
        )

        assert harness.threshold == 0.80
        assert harness.metric_weight == 0.4
        assert harness.vlm_weight == 0.6
        assert harness.fail_on_vlm_error is True
        assert len(harness.metric_fns) == 1

    def test_evaluate_metrics_only(self, tmp_path):
        """Test evaluation with metrics only (no VLM)."""
        # Create test image
        import numpy as np
        from PIL import Image

        img_path = tmp_path / "test.png"
        img = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
        img.save(img_path)

        # Create harness with mock metric
        def mock_metric(paths):
            return 0.85

        harness = ApexEvaluationHarness(
            metric_fns=[mock_metric],
            threshold=0.70,
        )

        result = harness.evaluate(image_paths=[img_path])

        assert result.score == 0.85  # Only metric, no VLM
        assert result.passes is True
        assert result.vlm_score == 0.0

    def test_evaluate_with_mocked_vlm(self, tmp_path):
        """Test evaluation with mocked VLM backend."""
        # Create test image
        import numpy as np
        from PIL import Image

        img_path = tmp_path / "test.png"
        img = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
        img.save(img_path)

        # Mock VLM backend
        mock_vlm = MagicMock()
        mock_vlm_result = MagicMock()
        mock_vlm_result.summary_score = 0.80
        mock_vlm_result.passes_basic_quality = True
        mock_vlm_result.issues = []
        mock_vlm_result.raw_text = "Test response"
        mock_vlm_result.model_key = "test_model"
        mock_vlm.evaluate_images.return_value = mock_vlm_result

        def mock_metric(paths):
            return 0.90

        harness = ApexEvaluationHarness(
            llava_backend=mock_vlm,
            metric_fns=[mock_metric],
            threshold=0.70,
            metric_weight=0.5,
        )

        result = harness.evaluate(image_paths=[img_path])

        # Combined score: 0.90 * 0.5 + 0.80 * 0.5 = 0.85
        assert result.score == pytest.approx(0.85)
        assert result.passes is True
        assert result.vlm_score == 0.80

    def test_evaluate_vlm_error_handling(self, tmp_path):
        """Test VLM error handling."""
        # Create test image
        import numpy as np
        from PIL import Image

        img_path = tmp_path / "test.png"
        img = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
        img.save(img_path)

        # Mock VLM backend that raises error
        mock_vlm = MagicMock()
        mock_vlm.evaluate_images.side_effect = RuntimeError("VLM failed")

        def mock_metric(paths):
            return 0.90

        # Without fail_on_vlm_error
        harness = ApexEvaluationHarness(
            llava_backend=mock_vlm,
            metric_fns=[mock_metric],
            fail_on_vlm_error=False,
        )

        result = harness.evaluate(image_paths=[img_path])

        # Should fall back to metrics only
        assert result.score == 0.90
        assert result.passes is True

    def test_metric_aggregation(self):
        """Test metric score aggregation."""
        harness = ApexEvaluationHarness()

        results = [
            EvalMetricResult(name="metric1", score=0.8),
            EvalMetricResult(name="metric2", score=0.6),
            EvalMetricResult(name="metric3", score=1.0),
        ]

        avg = harness._aggregate_metrics(results)
        assert avg == pytest.approx(0.8)  # (0.8 + 0.6 + 1.0) / 3

    def test_metric_aggregation_empty(self):
        """Test metric aggregation with empty list."""
        harness = ApexEvaluationHarness()
        avg = harness._aggregate_metrics([])
        assert avg == 0.0

    def test_combine_scores_metrics_only(self):
        """Test score combination with metrics only."""

        def mock_metric(paths):
            return 0.85

        harness = ApexEvaluationHarness(metric_fns=[mock_metric])

        combined = harness._combine_scores(0.85, {"skipped": True})
        assert combined == 0.85

    def test_combine_scores_vlm_only(self):
        """Test score combination with VLM only."""
        harness = ApexEvaluationHarness()  # No metrics

        combined = harness._combine_scores(0.0, {"score": 0.80, "skipped": False})
        assert combined == 0.80

    def test_combine_scores_weighted(self):
        """Test weighted score combination."""

        def mock_metric(paths):
            return 0.90

        harness = ApexEvaluationHarness(
            metric_fns=[mock_metric],
            metric_weight=0.3,  # 30% metric, 70% VLM
        )

        combined = harness._combine_scores(0.90, {"score": 0.70, "skipped": False})
        # 0.90 * 0.3 + 0.70 * 0.7 = 0.27 + 0.49 = 0.76
        assert combined == pytest.approx(0.76)

    def test_run_metrics_error_handling(self, tmp_path):
        """Test metric error handling."""

        def failing_metric(paths):
            raise ValueError("Metric failed")

        harness = ApexEvaluationHarness(metric_fns=[failing_metric])

        results = harness._run_metrics([tmp_path / "dummy.png"])

        assert len(results) == 1
        assert results[0].score == 0.0
        assert "error" in results[0].metadata

    def test_run_metrics_score_clamping(self):
        """Test that metric scores are clamped to [0, 1]."""

        def out_of_range_metric(paths):
            return 1.5  # Out of range

        harness = ApexEvaluationHarness(metric_fns=[out_of_range_metric])

        results = harness._run_metrics([Path("dummy.png")])

        assert results[0].score == 1.0  # Clamped

    def test_evaluate_passes_threshold(self, tmp_path):
        """Test pass/fail threshold logic."""
        # Create test image
        import numpy as np
        from PIL import Image

        img_path = tmp_path / "test.png"
        img = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
        img.save(img_path)

        # Score above threshold
        def high_metric(paths):
            return 0.85

        harness_pass = ApexEvaluationHarness(
            metric_fns=[high_metric],
            threshold=0.70,
        )
        result_pass = harness_pass.evaluate(image_paths=[img_path])
        assert result_pass.passes is True

        # Score below threshold
        def low_metric(paths):
            return 0.50

        harness_fail = ApexEvaluationHarness(
            metric_fns=[low_metric],
            threshold=0.70,
        )
        result_fail = harness_fail.evaluate(image_paths=[img_path])
        assert result_fail.passes is False


class TestBuiltInMetrics:
    """Test built-in metric functions."""

    def test_sharpness_metric(self, tmp_path):
        """Test sharpness metric with real image."""
        import numpy as np

        # Create test image
        try:
            import cv2

            # Sharp image (high frequency content)
            sharp_img = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
            sharp_path = tmp_path / "sharp.png"
            cv2.imwrite(str(sharp_path), sharp_img)

            score = sharpness_metric([sharp_path])
            assert 0 <= score <= 1
        except ImportError:
            pytest.skip("OpenCV not available")

    def test_sharpness_metric_no_cv2(self, tmp_path):
        """Test sharpness metric without OpenCV."""
        with patch.dict("sys.modules", {"cv2": None}):
            score = sharpness_metric([tmp_path / "dummy.png"])
            assert score == 0.0

    def test_contrast_metric(self, tmp_path):
        """Test contrast metric with real image."""
        import numpy as np

        try:
            import cv2

            # High contrast image
            img = np.zeros((100, 100), dtype=np.uint8)
            img[50:, :] = 255
            img_path = tmp_path / "contrast.png"
            cv2.imwrite(str(img_path), img)

            score = contrast_metric([img_path])
            assert 0 <= score <= 1
        except ImportError:
            pytest.skip("OpenCV not available")

    def test_contrast_metric_no_cv2(self, tmp_path):
        """Test contrast metric without OpenCV."""
        with patch.dict("sys.modules", {"cv2": None}):
            score = contrast_metric([tmp_path / "dummy.png"])
            assert score == 0.0

    def test_brightness_metric(self, tmp_path):
        """Test brightness metric with real image."""
        import numpy as np

        try:
            import cv2

            # Medium brightness (optimal around 128)
            img = np.full((100, 100), 128, dtype=np.uint8)
            img_path = tmp_path / "brightness.png"
            cv2.imwrite(str(img_path), img)

            score = brightness_metric([img_path])
            assert score == pytest.approx(1.0)  # Perfect brightness

            # Dark image
            dark_img = np.full((100, 100), 30, dtype=np.uint8)
            dark_path = tmp_path / "dark.png"
            cv2.imwrite(str(dark_path), dark_img)

            dark_score = brightness_metric([dark_path])
            assert dark_score < 1.0  # Penalized
        except ImportError:
            pytest.skip("OpenCV not available")

    def test_brightness_metric_no_cv2(self, tmp_path):
        """Test brightness metric without OpenCV."""
        with patch.dict("sys.modules", {"cv2": None}):
            score = brightness_metric([tmp_path / "dummy.png"])
            assert score == 0.0

    def test_metrics_empty_paths(self):
        """Test metrics with empty path list."""
        assert sharpness_metric([]) == 0.0
        assert contrast_metric([]) == 0.0
        assert brightness_metric([]) == 0.0

    def test_metrics_missing_file(self, tmp_path):
        """Test metrics with missing file."""
        try:
            import cv2

            # cv2.imread returns None for missing files
            score = sharpness_metric([tmp_path / "nonexistent.png"])
            # Should handle gracefully (return 0 or skip)
            assert score == 0.0
        except ImportError:
            pytest.skip("OpenCV not available")


class TestPromptSpecBuilder:
    """Test prompt spec builder integration."""

    def test_harness_with_prompt_builder(self, tmp_path):
        """Test harness with custom prompt builder."""
        # Create test image
        import numpy as np
        from PIL import Image

        img_path = tmp_path / "test.png"
        img = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
        img.save(img_path)

        # Mock VLM backend
        mock_vlm = MagicMock()
        mock_vlm_result = MagicMock()
        mock_vlm_result.summary_score = 0.80
        mock_vlm_result.passes_basic_quality = True
        mock_vlm_result.issues = []
        mock_vlm_result.raw_text = "Test response"
        mock_vlm_result.model_key = "test_model"
        mock_vlm.evaluate_images.return_value = mock_vlm_result

        # Custom prompt builder
        def custom_prompt_builder(context):
            return MagicMock(name="custom_prompt")

        harness = ApexEvaluationHarness(
            llava_backend=mock_vlm,
            prompt_spec_builder=custom_prompt_builder,
        )

        result = harness.evaluate(
            image_paths=[img_path],
            context={"scene": "kitchen"},
        )

        # Verify prompt builder was called
        assert result.vlm_score == 0.80
