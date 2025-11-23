"""
Tests for Perceptual Baseline Calibration
"""

import pytest
import torch
from PIL import Image

from transformation_portal.foundation import ComputationalSubstrate, SubstrateConfig
from transformation_portal.perceptual import (
    PerceptualBaseline,
    BaselineConfig,
    ImageLoader,
    PerceptualAnalyzer,
    EnhancementTracker,
    QualityMetrics,
    MetricType,
)


class TestImageLoader:
    """Test suite for image loader."""

    @pytest.fixture
    def substrate(self):
        """Create substrate for testing."""
        config = SubstrateConfig.for_development()
        return ComputationalSubstrate(config)

    @pytest.fixture
    def image_loader(self, substrate):
        """Create image loader."""
        return ImageLoader(substrate, normalize=True)

    @pytest.fixture
    def test_image_path(self, tmp_path):
        """Create a test image."""
        img = Image.new('RGB', (512, 512), color='red')
        path = tmp_path / "test_image.jpg"
        img.save(path)
        return path

    def test_initialization(self, image_loader):
        """Test image loader initialization."""
        assert image_loader is not None
        assert image_loader.substrate is not None

    def test_load_image(self, image_loader, test_image_path):
        """Test loading a single image."""
        tensor, metadata = image_loader.load(test_image_path)

        assert tensor is not None
        assert tensor.ndim == 3  # (C, H, W)
        assert metadata is not None
        assert metadata.width == 512
        assert metadata.height == 512

    def test_image_normalization(self, image_loader, test_image_path):
        """Test image normalization."""
        tensor, _ = image_loader.load(test_image_path)

        # Should be normalized to [0, 1]
        assert tensor.min() >= 0.0
        assert tensor.max() <= 1.0

    def test_thumbnail_creation(self, image_loader, test_image_path):
        """Test thumbnail creation."""
        tensor, _ = image_loader.load(test_image_path)
        thumbnail = image_loader.create_thumbnail(tensor, size=(128, 128))

        assert thumbnail.shape[-2:] == (128, 128)

    def test_image_stats(self, image_loader, test_image_path):
        """Test image statistics."""
        tensor, _ = image_loader.load(test_image_path)
        stats = image_loader.get_image_stats(tensor)

        assert "mean" in stats
        assert "std" in stats
        assert "min" in stats
        assert "max" in stats


class TestQualityMetrics:
    """Test suite for quality metrics."""

    @pytest.fixture
    def substrate(self):
        """Create substrate."""
        config = SubstrateConfig.for_development()
        return ComputationalSubstrate(config)

    @pytest.fixture
    def quality_metrics(self, substrate):
        """Create quality metrics calculator."""
        return QualityMetrics(substrate, cache_models=False)

    @pytest.fixture
    def test_tensors(self, substrate):
        """Create test tensors."""
        device = substrate.get_device()
        image = torch.rand(3, 256, 256, device=device)
        reference = torch.rand(3, 256, 256, device=device)
        return image, reference

    def test_initialization(self, quality_metrics):
        """Test quality metrics initialization."""
        assert quality_metrics is not None

    def test_compute_psnr(self, quality_metrics, test_tensors):
        """Test PSNR computation."""
        image, reference = test_tensors
        score = quality_metrics.compute_psnr(image, reference)

        assert score is not None
        assert score.metric_type == MetricType.PSNR
        assert score.score > 0

    def test_compute_ssim(self, quality_metrics, test_tensors):
        """Test SSIM computation."""
        image, reference = test_tensors
        score = quality_metrics.compute_ssim(image, reference)

        assert score is not None
        assert score.metric_type == MetricType.SSIM
        assert 0 <= score.score <= 1

    def test_compute_mse(self, quality_metrics, test_tensors):
        """Test MSE computation."""
        image, reference = test_tensors
        score = quality_metrics.compute_mse(image, reference)

        assert score is not None
        assert score.metric_type == MetricType.MSE
        assert score.score >= 0

    def test_identical_images_psnr(self, quality_metrics, substrate):
        """Test PSNR for identical images should be infinity."""
        device = substrate.get_device()
        image = torch.rand(3, 256, 256, device=device)
        score = quality_metrics.compute_psnr(image, image)

        assert score.score == float('inf')

    def test_identical_images_ssim(self, quality_metrics, substrate):
        """Test SSIM for identical images should be 1.0."""
        device = substrate.get_device()
        image = torch.rand(3, 256, 256, device=device)
        score = quality_metrics.compute_ssim(image, image)

        assert abs(score.score - 1.0) < 0.01

    def test_compute_brisque(self, quality_metrics, substrate):
        """Test BRISQUE computation."""
        device = substrate.get_device()
        image = torch.rand(3, 256, 256, device=device)
        score = quality_metrics.compute_brisque(image)

        assert score is not None
        assert score.metric_type == MetricType.BRISQUE

    def test_compute_all(self, quality_metrics, test_tensors):
        """Test computing all metrics."""
        image, reference = test_tensors
        scores = quality_metrics.compute_all(image, reference)

        assert len(scores) > 0
        assert MetricType.PSNR in scores
        assert MetricType.SSIM in scores


class TestPerceptualAnalyzer:
    """Test suite for perceptual analyzer."""

    @pytest.fixture
    def substrate(self):
        """Create substrate."""
        config = SubstrateConfig.for_development()
        return ComputationalSubstrate(config)

    @pytest.fixture
    def analyzer(self, substrate):
        """Create analyzer."""
        return PerceptualAnalyzer(substrate)

    @pytest.fixture
    def test_image_and_metadata(self, substrate, tmp_path):
        """Create test image with metadata."""
        # Create test image
        img = Image.new('RGB', (256, 256), color='blue')
        path = tmp_path / "test.jpg"
        img.save(path)

        # Load with image loader
        loader = ImageLoader(substrate)
        tensor, metadata = loader.load(path)

        return tensor, metadata

    def test_initialization(self, analyzer):
        """Test analyzer initialization."""
        assert analyzer is not None

    def test_analyze(self, analyzer, test_image_and_metadata):
        """Test image analysis."""
        tensor, metadata = test_image_and_metadata
        result = analyzer.analyze(tensor, metadata)

        assert result is not None
        assert result.overall_quality >= 0
        assert result.sharpness >= 0
        assert result.contrast >= 0
        assert len(result.quality_scores) > 0

    def test_analysis_time_recorded(self, analyzer, test_image_and_metadata):
        """Test that analysis time is recorded."""
        tensor, metadata = test_image_and_metadata
        result = analyzer.analyze(tensor, metadata)

        assert result.analysis_time > 0

    def test_get_summary(self, analyzer, test_image_and_metadata):
        """Test getting result summary."""
        tensor, metadata = test_image_and_metadata
        result = analyzer.analyze(tensor, metadata)
        summary = result.get_summary()

        assert "path" in summary
        assert "overall_quality" in summary
        assert "scores" in summary


class TestEnhancementTracker:
    """Test suite for enhancement tracker."""

    @pytest.fixture
    def tracker(self):
        """Create tracker."""
        return EnhancementTracker(target_quality_multiplier=1.3)

    @pytest.fixture
    def mock_analysis_results(self, tmp_path):
        """Create mock analysis results."""
        from transformation_portal.perceptual.analyzer import AnalysisResult
        from transformation_portal.perceptual.image_loader import ImageMetadata, ImageType
        from transformation_portal.perceptual.metrics import PerceptualScore, MetricType

        results = []
        for i, name in enumerate(["pool", "bedroom", "kitchen"]):
            path = tmp_path / f"{name}.jpg"

            metadata = ImageMetadata(
                path=path,
                image_type=ImageType.POOL,
                width=512,
                height=512,
                channels=3,
                format="JPEG",
                size_bytes=100000,
                bit_depth=8,
                color_space="RGB",
                mean_intensity=0.5,
                std_intensity=0.2,
                dynamic_range=1.0,
                tags={}
            )

            scores = {
                MetricType.PSNR: PerceptualScore(
                    metric_type=MetricType.PSNR,
                    score=30.0,
                    higher_is_better=True,
                    normalized_score=0.7,
                    metadata={}
                )
            }

            result = AnalysisResult(
                image_path=path,
                image_metadata=metadata,
                quality_scores=scores,
                overall_quality=0.6 + i * 0.1,
                analysis_time=1.0,
                timestamp=1234567890.0
            )

            results.append(result)

        return results

    def test_initialization(self, tracker):
        """Test tracker initialization."""
        assert tracker is not None
        assert tracker.target_quality_multiplier == 1.3

    def test_establish_baseline(self, tracker, mock_analysis_results):
        """Test establishing baseline."""
        tracker.establish_baseline(mock_analysis_results)

        assert tracker.baseline_established
        assert len(tracker.trajectories) == len(mock_analysis_results)

    def test_get_trajectory(self, tracker, mock_analysis_results):
        """Test getting specific trajectory."""
        tracker.establish_baseline(mock_analysis_results)
        trajectory = tracker.get_trajectory("pool")

        assert trajectory is not None
        assert trajectory.image_name == "pool"
        assert len(trajectory.points) == 1  # Baseline point

    def test_track_enhancement(self, tracker, mock_analysis_results):
        """Test tracking enhancement."""
        tracker.establish_baseline(mock_analysis_results)

        # Create enhanced result
        result = mock_analysis_results[0]
        result.overall_quality = 0.75  # Improved

        tracker.track_enhancement(result, step=1, description="Test enhancement")

        trajectory = tracker.get_trajectory("pool")
        assert len(trajectory.points) == 2
        assert trajectory.get_improvement() > 0

    def test_get_summary(self, tracker, mock_analysis_results):
        """Test getting tracker summary."""
        tracker.establish_baseline(mock_analysis_results)
        summary = tracker.get_summary()

        assert "num_images" in summary
        assert "avg_improvement" in summary
        assert summary["num_images"] == len(mock_analysis_results)


class TestPerceptualBaseline:
    """Test suite for perceptual baseline system."""

    @pytest.fixture
    def substrate(self):
        """Create substrate."""
        config = SubstrateConfig.for_development()
        return ComputationalSubstrate(config)

    @pytest.fixture
    def baseline(self, substrate, tmp_path):
        """Create baseline system."""
        config = BaselineConfig.default()
        config.output_dir = tmp_path / "outputs"
        config.save_visualizations = False  # Skip for tests
        config.save_reports = False
        return PerceptualBaseline(substrate, config)

    @pytest.fixture
    def test_images(self, tmp_path):
        """Create test images."""
        paths = []
        for i, name in enumerate(["pool", "bedroom", "kitchen"]):
            img = Image.new('RGB', (256, 256), color=(i*80, i*80, i*80))
            path = tmp_path / f"{name}.jpg"
            img.save(path)
            paths.append(path)
        return paths

    def test_initialization(self, baseline):
        """Test baseline initialization."""
        assert baseline is not None
        assert not baseline.calibrated

    def test_calibrate(self, baseline, test_images):
        """Test baseline calibration."""
        results = baseline.calibrate(test_images)

        assert len(results) == len(test_images)
        assert baseline.calibrated
        assert len(baseline.baseline_results) == len(test_images)

    def test_get_baseline_metrics(self, baseline, test_images):
        """Test getting baseline metrics."""
        baseline.calibrate(test_images)
        metrics = baseline.get_baseline_metrics()

        assert len(metrics) == len(test_images)
        for name, metric_values in metrics.items():
            assert "overall_quality" in metric_values
            assert "sharpness" in metric_values

    def test_uncalibrated_error(self, baseline):
        """Test that operations fail before calibration."""
        with pytest.raises(RuntimeError):
            baseline.get_baseline_metrics()

    def test_export_baseline_data(self, baseline, test_images, tmp_path):
        """Test exporting baseline data."""
        baseline.calibrate(test_images)
        export_path = tmp_path / "baseline_data.json"
        result_path = baseline.export_baseline_data(export_path)

        assert result_path.exists()
        assert result_path.stat().st_size > 0

    def test_generate_report(self, baseline, test_images):
        """Test generating report."""
        baseline.calibrate(test_images)
        report = baseline.generate_report()

        assert report is not None
        assert len(report) > 0
        assert "PHASE 2" in report
        assert "PERCEPTUAL BASELINE" in report


class TestIntegration:
    """Integration tests for full perceptual baseline workflow."""

    def test_end_to_end_calibration(self, tmp_path):
        """Test complete calibration workflow."""
        # Initialize substrate
        substrate = ComputationalSubstrate(SubstrateConfig.for_development())

        # Create test images
        image_paths = []
        for name in ["pool", "bedroom", "kitchen"]:
            img = Image.new('RGB', (256, 256), color='green')
            path = tmp_path / f"{name}.jpg"
            img.save(path)
            image_paths.append(path)

        # Initialize baseline
        config = BaselineConfig.default()
        config.output_dir = tmp_path / "outputs"
        config.save_visualizations = False
        config.save_reports = False
        baseline = PerceptualBaseline(substrate, config)

        # Calibrate
        results = baseline.calibrate(image_paths)

        # Verify results
        assert len(results) == 3
        assert baseline.calibrated

        # Get metrics
        metrics = baseline.get_baseline_metrics()
        assert len(metrics) == 3

        # Generate report
        report = baseline.generate_report()
        assert "pool" in report or "bedroom" in report or "kitchen" in report


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
