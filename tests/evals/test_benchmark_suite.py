"""Tests for evals/benchmark_suite.py module (Phase 5 coverage).

Tests for:
- BenchmarkResult dataclass
- BenchmarkWeights configuration
- BenchmarkSuite evaluation flow
- Metric computation helpers
- Batch benchmarking

All tests use mocks - no ML model downloads or GPU requirements.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from transformation_portal.evals.benchmark_suite import (
    BenchmarkResult,
    BenchmarkSuite,
    BenchmarkWeights,
    run_benchmark_batch,
)

pytestmark = [pytest.mark.unit, pytest.mark.ml]


class TestBenchmarkResult:
    """Test BenchmarkResult dataclass."""

    def test_default_values(self):
        """Test default result values."""
        result = BenchmarkResult()

        assert result.psnr == 0.0
        assert result.ssim == 0.0
        assert result.lpips == 0.0
        assert result.iou == 0.0
        assert result.dice == 0.0
        assert result.llava_score == 0.0
        assert result.llava_issues == []
        assert result.combined_score == 0.0
        assert result.metadata == {}

    def test_with_values(self):
        """Test result with all values."""
        result = BenchmarkResult(
            psnr=35.5,
            ssim=0.92,
            lpips=0.15,
            iou=0.85,
            dice=0.88,
            llava_score=0.80,
            llava_issues=[{"issue_type": "blur", "severity": "low"}],
            combined_score=0.82,
            metadata={"runtime_ms": 150},
        )

        assert result.psnr == 35.5
        assert result.ssim == 0.92
        assert result.combined_score == 0.82
        assert len(result.llava_issues) == 1

    def test_to_dict(self):
        """Test serialization to dictionary."""
        result = BenchmarkResult(
            psnr=30.0,
            ssim=0.85,
            lpips=0.20,
            combined_score=0.75,
        )

        d = result.to_dict()

        assert d["psnr"] == 30.0
        assert d["ssim"] == 0.85
        assert d["lpips"] == 0.20
        assert d["combined_score"] == 0.75


class TestBenchmarkWeights:
    """Test BenchmarkWeights dataclass."""

    def test_default_weights(self):
        """Test default weight values."""
        weights = BenchmarkWeights()

        assert weights.psnr == 0.2
        assert weights.ssim == 0.15
        assert weights.lpips == 0.15
        assert weights.iou == 0.2
        assert weights.llava == 0.3

    def test_custom_weights(self):
        """Test custom weight values."""
        weights = BenchmarkWeights(
            psnr=0.3,
            ssim=0.2,
            lpips=0.1,
            iou=0.1,
            llava=0.3,
        )

        assert weights.psnr == 0.3
        assert weights.llava == 0.3


class TestBenchmarkSuite:
    """Test BenchmarkSuite class."""

    @pytest.fixture
    def sample_images(self):
        """Create sample test images."""
        pred = np.random.rand(100, 100, 3).astype(np.float32)
        gt = np.random.rand(100, 100, 3).astype(np.float32)
        return pred, gt

    @pytest.fixture
    def sample_masks(self):
        """Create sample test masks."""
        pred_mask = np.zeros((100, 100), dtype=bool)
        pred_mask[30:70, 30:70] = True

        gt_mask = np.zeros((100, 100), dtype=bool)
        gt_mask[25:75, 25:75] = True

        return pred_mask, gt_mask

    def test_suite_initialization_minimal(self):
        """Test minimal suite initialization."""
        suite = BenchmarkSuite()

        assert suite.llava_backend is None
        assert suite.weights is not None
        assert suite.weights.psnr == 0.2  # Default

    def test_suite_initialization_with_options(self):
        """Test suite with custom options."""
        mock_backend = MagicMock()
        weights = BenchmarkWeights(psnr=0.4, llava=0.1)

        suite = BenchmarkSuite(
            llava_backend=mock_backend,
            weights=weights,
        )

        assert suite.llava_backend == mock_backend
        assert suite.weights.psnr == 0.4

    def test_run_prediction_only(self, sample_images):
        """Test running with prediction only (no GT)."""
        pred, _ = sample_images
        suite = BenchmarkSuite()

        result = suite.run(prediction=pred)

        assert result.psnr == 0.0  # No GT to compare
        assert result.metadata["has_ground_truth"] is False

    def test_run_with_ground_truth(self, sample_images):
        """Test running with ground truth."""
        pred, gt = sample_images
        suite = BenchmarkSuite()

        with patch.object(suite, "_compute_psnr", return_value=30.0):
            with patch.object(suite, "_compute_ssim", return_value=0.85):
                with patch.object(suite, "_compute_lpips", return_value=0.15):
                    result = suite.run(prediction=pred, ground_truth=gt)

        assert result.psnr == 30.0
        assert result.ssim == 0.85
        assert result.lpips == 0.15
        assert result.metadata["has_ground_truth"] is True

    def test_run_with_masks(self, sample_images, sample_masks):
        """Test running with segmentation masks."""
        pred, gt = sample_images
        pred_mask, gt_mask = sample_masks
        suite = BenchmarkSuite()

        with patch.object(suite, "_compute_psnr", return_value=30.0):
            with patch.object(suite, "_compute_ssim", return_value=0.85):
                with patch.object(suite, "_compute_lpips", return_value=0.15):
                    with patch.object(suite, "_compute_iou", return_value=0.75):
                        with patch.object(suite, "_compute_dice", return_value=0.80):
                            result = suite.run(
                                prediction=pred,
                                ground_truth=gt,
                                pred_mask=pred_mask,
                                gt_mask=gt_mask,
                            )

        assert result.iou == 0.75
        assert result.dice == 0.80
        assert result.metadata["has_masks"] is True

    def test_run_with_llava_backend(self, sample_images, tmp_path):
        """Test running with LLaVA backend."""
        pred, gt = sample_images

        # Create image file
        pred_path = tmp_path / "pred.png"
        from PIL import Image
        Image.fromarray((pred * 255).astype(np.uint8)).save(pred_path)

        # Mock LLaVA backend
        mock_backend = MagicMock()
        mock_result = MagicMock()
        mock_result.summary_score = 0.85
        mock_result.issues = []
        mock_backend.evaluate_images.return_value = mock_result

        suite = BenchmarkSuite(llava_backend=mock_backend)

        with patch.object(suite, "_compute_psnr", return_value=30.0):
            with patch.object(suite, "_compute_ssim", return_value=0.85):
                with patch.object(suite, "_compute_lpips", return_value=0.15):
                    result = suite.run(
                        prediction=pred,
                        ground_truth=gt,
                        prediction_path=pred_path,
                    )

        assert result.llava_score == 0.85
        assert result.metadata["has_llava"] is True

    def test_compute_psnr_error_handling(self, sample_images):
        """Test PSNR computation error handling."""
        pred, gt = sample_images
        suite = BenchmarkSuite()

        with patch("transformation_portal.evals.benchmark_suite.psnr", side_effect=ValueError("Test error")):
            result = suite._compute_psnr(pred, gt)

        assert result == 0.0

    def test_compute_ssim_error_handling(self, sample_images):
        """Test SSIM computation error handling."""
        pred, gt = sample_images
        suite = BenchmarkSuite()

        with patch("transformation_portal.evals.benchmark_suite.ssim", side_effect=ValueError("Test error")):
            result = suite._compute_ssim(pred, gt)

        assert result == 0.0

    def test_compute_lpips_error_handling(self, sample_images):
        """Test LPIPS computation error handling."""
        pred, gt = sample_images
        suite = BenchmarkSuite()

        with patch("transformation_portal.evals.benchmark_suite.lpips_score", side_effect=ValueError("Test error")):
            result = suite._compute_lpips(pred, gt)

        assert result == 0.0

    def test_compute_iou_error_handling(self, sample_masks):
        """Test IoU computation error handling."""
        pred_mask, gt_mask = sample_masks
        suite = BenchmarkSuite()

        with patch("transformation_portal.evals.benchmark_suite.segmentation_iou", side_effect=ValueError("Test error")):
            result = suite._compute_iou(pred_mask, gt_mask)

        assert result == 0.0

    def test_compute_dice_error_handling(self, sample_masks):
        """Test Dice computation error handling."""
        pred_mask, gt_mask = sample_masks
        suite = BenchmarkSuite()

        with patch("transformation_portal.evals.benchmark_suite.dice_coefficient", side_effect=ValueError("Test error")):
            result = suite._compute_dice(pred_mask, gt_mask)

        assert result == 0.0

    def test_llava_error_handling(self, tmp_path):
        """Test LLaVA evaluation error handling."""
        pred_path = tmp_path / "test.png"
        pred_path.touch()

        mock_backend = MagicMock()
        mock_backend.evaluate_images.side_effect = RuntimeError("LLaVA failed")

        suite = BenchmarkSuite(llava_backend=mock_backend)
        result = suite._run_llava(pred_path, context=None)

        assert result["score"] == 0.0
        assert result["issues"] == []

    def test_compute_combined_score_full(self):
        """Test combined score with all metrics."""
        suite = BenchmarkSuite()

        result = BenchmarkResult(
            psnr=35.0,  # Will be normalized
            ssim=0.90,
            lpips=0.10,  # Will be inverted
            iou=0.85,
            llava_score=0.80,
            metadata={
                "has_ground_truth": True,
                "has_masks": True,
                "has_llava": True,
            },
        )

        combined = suite._compute_combined_score(result)

        # Should be weighted average of normalized scores
        assert 0.0 <= combined <= 1.0

    def test_compute_combined_score_gt_only(self):
        """Test combined score with ground truth only."""
        suite = BenchmarkSuite()

        result = BenchmarkResult(
            psnr=30.0,
            ssim=0.85,
            lpips=0.20,
            metadata={
                "has_ground_truth": True,
                "has_masks": False,
                "has_llava": False,
            },
        )

        combined = suite._compute_combined_score(result)

        assert 0.0 <= combined <= 1.0

    def test_compute_combined_score_no_metrics(self):
        """Test combined score with no metrics."""
        suite = BenchmarkSuite()

        result = BenchmarkResult(
            metadata={
                "has_ground_truth": False,
                "has_masks": False,
                "has_llava": False,
            },
        )

        combined = suite._compute_combined_score(result)

        assert combined == 0.0


class TestRunBenchmarkBatch:
    """Test run_benchmark_batch function."""

    def test_empty_batch(self):
        """Test empty batch returns count 0."""
        suite = BenchmarkSuite()
        result = run_benchmark_batch(suite, [])

        assert result["count"] == 0

    def test_single_sample(self):
        """Test batch with single sample."""
        suite = BenchmarkSuite()

        with patch.object(suite, "run") as mock_run:
            mock_run.return_value = BenchmarkResult(
                psnr=30.0,
                ssim=0.85,
                lpips=0.15,
                iou=0.75,
                llava_score=0.80,
                combined_score=0.78,
            )

            samples = [{"prediction": np.zeros((10, 10, 3))}]
            result = run_benchmark_batch(suite, samples)

        assert result["count"] == 1
        assert result["mean_psnr"] == 30.0
        assert result["mean_ssim"] == 0.85
        assert result["mean_combined"] == 0.78

    def test_multiple_samples(self):
        """Test batch with multiple samples."""
        suite = BenchmarkSuite()

        results = [
            BenchmarkResult(psnr=30.0, ssim=0.80, lpips=0.20, combined_score=0.70),
            BenchmarkResult(psnr=35.0, ssim=0.90, lpips=0.10, combined_score=0.80),
            BenchmarkResult(psnr=25.0, ssim=0.75, lpips=0.25, combined_score=0.65),
        ]

        call_count = [0]

        def mock_run(**kwargs):
            r = results[call_count[0]]
            call_count[0] += 1
            return r

        with patch.object(suite, "run", side_effect=mock_run):
            samples = [
                {"prediction": np.zeros((10, 10, 3))},
                {"prediction": np.zeros((10, 10, 3))},
                {"prediction": np.zeros((10, 10, 3))},
            ]
            result = run_benchmark_batch(suite, samples)

        assert result["count"] == 3
        assert result["mean_psnr"] == pytest.approx(30.0)  # (30+35+25)/3
        assert result["mean_ssim"] == pytest.approx(0.8167, rel=0.01)  # (0.80+0.90+0.75)/3
        assert result["mean_combined"] == pytest.approx(0.7167, rel=0.01)  # (0.70+0.80+0.65)/3
