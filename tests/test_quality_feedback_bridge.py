"""
Tests for the Quality Feedback Bridge.

Tests cover:
- QualityFeedbackBridge initialization and configuration
- LPIPS/heuristic fallback transitions
- Hybrid mode metric aggregation
- RAG callback invocation verification
- Error handling (CUDA OOM, network failures)
- 750 Picacho preset recognition
- UnifiedQualityMetrics document structure
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from transformation_portal.pipelines.quality_feedback_bridge import (
    HeuristicMetrics,
    PerceptualMetrics,
    QualityFeedbackBridge,
    QualityTargets,
    UnifiedQualityMetrics,
    create_quality_callback_for_pipeline,
    create_rag_indexing_callback,
    index_quality_metrics_to_rag,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_image_np():
    """Create a sample RGB image as numpy array."""
    h, w = 256, 384
    r = np.linspace(0.2, 0.8, w)
    g = np.linspace(0.3, 0.7, h)[:, np.newaxis]
    b = np.ones((h, w)) * 0.5

    image = np.stack([
        np.broadcast_to(r, (h, w)),
        np.broadcast_to(g, (h, w)),
        b,
    ], axis=2).astype(np.float32)

    return image


@pytest.fixture
def sample_image_pil(sample_image_np):
    """Create a sample PIL Image."""
    img_uint8 = (sample_image_np * 255).astype(np.uint8)
    return Image.fromarray(img_uint8, mode='RGB')


@pytest.fixture
def modified_image_np(sample_image_np):
    """Create a modified version of the sample image."""
    # Add slight modifications to simulate enhancement
    modified = sample_image_np.copy()
    modified = np.clip(modified * 1.1 + 0.02, 0, 1)  # Slight brightness increase
    return modified


@pytest.fixture
def temp_output_dir():
    """Create a temporary output directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def quality_targets():
    """Create default quality targets."""
    return QualityTargets()


@pytest.fixture
def quality_bridge():
    """Create a QualityFeedbackBridge with default settings."""
    return QualityFeedbackBridge(hybrid_mode=True)


# =============================================================================
# Quality Targets Tests
# =============================================================================

class TestQualityTargets:
    """Tests for QualityTargets configuration."""

    def test_default_targets(self):
        """Test default quality target values."""
        targets = QualityTargets()

        assert targets.perceptual_percentile_target == 95.0
        assert targets.material_fidelity_target == 0.98
        assert targets.lpips_threshold_excellent == 0.10
        assert targets.ssim_target == 0.92

    def test_material_thresholds(self, quality_targets):
        """Test per-material threshold values."""
        assert "quartzite" in quality_targets.material_thresholds
        assert "oak" in quality_targets.material_thresholds
        assert "glass" in quality_targets.material_thresholds
        assert quality_targets.material_thresholds["quartzite"] == 0.96

    def test_custom_targets(self):
        """Test custom quality targets."""
        targets = QualityTargets(
            perceptual_percentile_target=90.0,
            material_fidelity_target=0.95,
        )

        assert targets.perceptual_percentile_target == 90.0
        assert targets.material_fidelity_target == 0.95


# =============================================================================
# Metrics Dataclass Tests
# =============================================================================

class TestHeuristicMetrics:
    """Tests for HeuristicMetrics dataclass."""

    def test_default_values(self):
        """Test default heuristic metric values."""
        metrics = HeuristicMetrics()

        assert metrics.sharpness == 0.0
        assert metrics.contrast == 0.0
        assert metrics.colorfulness == 0.0
        assert metrics.exposure_balance == 0.0
        assert metrics.noise_level == 0.0
        assert metrics.overall_score == 0.0

    def test_to_dict(self):
        """Test conversion to dictionary."""
        metrics = HeuristicMetrics(
            sharpness=0.8,
            contrast=0.7,
            colorfulness=0.6,
            exposure_balance=0.9,
            noise_level=0.1,
            overall_score=0.75,
        )

        d = metrics.to_dict()

        assert d["sharpness"] == 0.8
        assert d["contrast"] == 0.7
        assert d["overall_score"] == 0.75


class TestPerceptualMetrics:
    """Tests for PerceptualMetrics dataclass."""

    def test_default_values(self):
        """Test default perceptual metric values."""
        metrics = PerceptualMetrics()

        assert metrics.lpips_score == 0.0
        assert metrics.lpips_percentile == 0.0
        assert metrics.ssim_score == 0.0
        assert metrics.composite_score == 0.0

    def test_to_dict(self):
        """Test conversion to dictionary."""
        metrics = PerceptualMetrics(
            lpips_score=0.15,
            lpips_percentile=80.0,
            ssim_score=0.92,
            composite_score=85.0,
        )

        d = metrics.to_dict()

        assert d["lpips_score"] == 0.15
        assert d["lpips_percentile"] == 80.0
        assert d["ssim_score"] == 0.92


class TestUnifiedQualityMetrics:
    """Tests for UnifiedQualityMetrics dataclass."""

    def test_default_values(self):
        """Test default unified metric values."""
        metrics = UnifiedQualityMetrics()

        assert metrics.image_id == ""
        assert metrics.perceptual_composite == 0.0
        assert metrics.hybrid_mode is False
        assert isinstance(metrics.heuristic, HeuristicMetrics)
        assert isinstance(metrics.perceptual, PerceptualMetrics)

    def test_to_dict(self):
        """Test conversion to dictionary."""
        metrics = UnifiedQualityMetrics(
            image_id="test_001",
            pipeline_config_name="luxury_estate",
            perceptual_composite=85.0,
            heuristic_composite=78.0,
            hybrid_score=82.0,
        )

        d = metrics.to_dict()

        assert d["image_id"] == "test_001"
        assert d["pipeline_config_name"] == "luxury_estate"
        assert d["scores"]["perceptual_composite"] == 85.0
        assert d["scores"]["hybrid_score"] == 82.0

    def test_to_rag_document(self):
        """Test conversion to RAG-indexable document."""
        metrics = UnifiedQualityMetrics(
            image_id="test_001",
            pipeline_config_name="750_picacho",
        )

        doc = metrics.to_rag_document()

        assert doc["_type"] == "unified_quality_metrics"
        assert doc["_version"] == "1.0.0"
        assert "_indexed_at" in doc
        assert doc["image_id"] == "test_001"


# =============================================================================
# QualityFeedbackBridge Tests
# =============================================================================

class TestQualityFeedbackBridge:
    """Tests for QualityFeedbackBridge class."""

    def test_initialization(self):
        """Test bridge initialization."""
        bridge = QualityFeedbackBridge()

        assert bridge.hybrid_mode is True
        assert bridge.lpips_network == 'alex'
        assert bridge.enable_material_fidelity is True
        assert bridge._perceptual_assessor is None

    def test_initialization_with_custom_targets(self, quality_targets):
        """Test bridge initialization with custom targets."""
        bridge = QualityFeedbackBridge(targets=quality_targets)

        assert bridge.targets == quality_targets

    def test_initialization_hybrid_mode_false(self):
        """Test bridge initialization with hybrid mode disabled."""
        bridge = QualityFeedbackBridge(hybrid_mode=False)

        assert bridge.hybrid_mode is False

    def test_initialization_with_rag_callback(self):
        """Test bridge initialization with RAG callback."""
        callback = MagicMock()
        bridge = QualityFeedbackBridge(rag_callback=callback)

        assert bridge.rag_callback == callback


class TestQualityAssessment:
    """Tests for quality assessment functionality."""

    def test_assess_heuristic_only(self, quality_bridge, sample_image_np):
        """Test assessment with heuristic metrics only (no original)."""
        metrics = quality_bridge.assess(
            enhanced=sample_image_np,
            image_id="test_001",
        )

        assert isinstance(metrics, UnifiedQualityMetrics)
        assert metrics.image_id == "test_001"
        assert metrics.heuristic.sharpness >= 0
        assert metrics.heuristic.contrast >= 0
        assert metrics.heuristic_composite > 0

    def test_assess_with_pil_input(self, quality_bridge, sample_image_pil):
        """Test assessment with PIL Image input."""
        metrics = quality_bridge.assess(
            enhanced=sample_image_pil,
            image_id="test_002",
        )

        assert isinstance(metrics, UnifiedQualityMetrics)
        assert metrics.heuristic_composite > 0

    def test_assess_with_reference(
        self, quality_bridge, sample_image_np, modified_image_np
    ):
        """Test assessment with reference image for LPIPS comparison."""
        metrics = quality_bridge.assess(
            enhanced=modified_image_np,
            original=sample_image_np,
            image_id="test_003",
        )

        assert isinstance(metrics, UnifiedQualityMetrics)
        # LPIPS may or may not be available depending on environment
        if metrics.lpips_available:
            assert metrics.perceptual.lpips_score >= 0

    def test_assess_records_timing(self, quality_bridge, sample_image_np):
        """Test that assessment records processing time."""
        metrics = quality_bridge.assess(enhanced=sample_image_np)

        assert metrics.processing_time_ms > 0

    def test_assess_records_pipeline_config(
        self, quality_bridge, sample_image_np
    ):
        """Test that assessment records pipeline configuration."""
        metrics = quality_bridge.assess(
            enhanced=sample_image_np,
            pipeline_config_name="750_picacho",
        )

        assert metrics.pipeline_config_name == "750_picacho"


class TestHybridMode:
    """Tests for hybrid mode functionality."""

    def test_hybrid_score_calculation(self, quality_bridge, sample_image_np):
        """Test hybrid score is calculated in hybrid mode."""
        metrics = quality_bridge.assess(
            enhanced=sample_image_np,
            image_id="test_hybrid",
        )

        assert metrics.hybrid_mode is True
        assert metrics.hybrid_score >= 0

    def test_hybrid_score_fallback_no_lpips(self, sample_image_np):
        """Test hybrid score falls back to heuristic when LPIPS unavailable."""
        bridge = QualityFeedbackBridge(hybrid_mode=True)

        metrics = bridge.assess(enhanced=sample_image_np)

        # Without LPIPS, hybrid score should be based on heuristics
        if not metrics.lpips_available:
            # Hybrid score should be close to heuristic composite
            assert abs(metrics.hybrid_score - metrics.heuristic_composite) < 5.0


class TestTargetChecking:
    """Tests for target achievement checking."""

    def test_check_heuristic_targets(self, quality_bridge, sample_image_np):
        """Test checking of heuristic targets."""
        metrics = quality_bridge.assess(enhanced=sample_image_np)

        assert "heuristic_sharpness" in metrics.targets_met
        assert "heuristic_contrast" in metrics.targets_met
        assert "heuristic_colorfulness" in metrics.targets_met

    def test_targets_summary_generation(self, quality_bridge, sample_image_np):
        """Test generation of targets summary."""
        metrics = quality_bridge.assess(enhanced=sample_image_np)

        assert metrics.targets_summary != ""
        assert any(marker in metrics.targets_summary for marker in ["✓", "○", "✗"])


class TestHeuristicMetricComputation:
    """Tests for individual heuristic metric computation."""

    def test_sharpness_computation(self, quality_bridge, sample_image_np):
        """Test sharpness metric computation."""
        sharpness = quality_bridge._compute_sharpness(sample_image_np)

        assert 0 <= sharpness <= 1

    def test_contrast_computation(self, quality_bridge, sample_image_np):
        """Test contrast metric computation."""
        contrast = quality_bridge._compute_contrast(sample_image_np)

        assert 0 <= contrast <= 1

    def test_colorfulness_computation(self, quality_bridge, sample_image_np):
        """Test colorfulness metric computation."""
        colorfulness = quality_bridge._compute_colorfulness(sample_image_np)

        assert 0 <= colorfulness <= 1

    def test_exposure_balance_computation(self, quality_bridge, sample_image_np):
        """Test exposure balance metric computation."""
        exposure = quality_bridge._compute_exposure_balance(sample_image_np)

        assert 0 <= exposure <= 1

    def test_noise_estimation(self, quality_bridge, sample_image_np):
        """Test noise level estimation."""
        noise = quality_bridge._estimate_noise(sample_image_np)

        assert 0 <= noise <= 1


# =============================================================================
# RAG Callback Tests
# =============================================================================

class TestRAGCallback:
    """Tests for RAG callback functionality."""

    def test_callback_invocation(self, sample_image_np):
        """Test that RAG callback is invoked during assessment."""
        callback = MagicMock()
        bridge = QualityFeedbackBridge(rag_callback=callback)

        bridge.assess(enhanced=sample_image_np, image_id="test_rag")

        callback.assert_called_once()

    def test_callback_receives_document(self, sample_image_np):
        """Test that callback receives proper document structure."""
        received_doc = None

        def capture_callback(doc):
            nonlocal received_doc
            received_doc = doc

        bridge = QualityFeedbackBridge(rag_callback=capture_callback)
        bridge.assess(enhanced=sample_image_np, image_id="test_doc")

        assert received_doc is not None
        assert received_doc["_type"] == "unified_quality_metrics"
        assert received_doc["image_id"] == "test_doc"

    def test_callback_error_handling(self, sample_image_np):
        """Test that callback errors don't crash assessment and callback was invoked."""
        callback = MagicMock(side_effect=RuntimeError("Callback failed"))

        bridge = QualityFeedbackBridge(rag_callback=callback)

        # Should not raise, just warn
        metrics = bridge.assess(enhanced=sample_image_np)

        assert metrics is not None
        # Verify the callback was actually invoked (even though it failed)
        assert callback.call_count == 1
        assert any("RAG callback failed" in w for w in metrics.warnings)


class TestRAGIndexing:
    """Tests for RAG indexing utilities."""

    def test_create_rag_indexing_callback(self, temp_output_dir):
        """Test creation of RAG indexing callback."""
        callback = create_rag_indexing_callback(str(temp_output_dir))

        assert callable(callback)

    def test_rag_indexing_callback_writes_file(self, temp_output_dir):
        """Test that RAG indexing callback writes JSON file."""
        callback = create_rag_indexing_callback(str(temp_output_dir))

        doc = {
            "image_id": "test_write",
            "_type": "unified_quality_metrics",
        }
        callback(doc)

        json_files = list(temp_output_dir.glob("*.json"))
        assert len(json_files) >= 1

    def test_index_quality_metrics_to_rag(self, temp_output_dir):
        """Test direct RAG indexing function."""
        metrics = UnifiedQualityMetrics(
            image_id="test_index",
            pipeline_config_name="test_config",
        )

        result = index_quality_metrics_to_rag(metrics, str(temp_output_dir))

        assert result is True
        json_files = list(temp_output_dir.glob("quality_*.json"))
        assert len(json_files) >= 1

    def test_index_quality_metrics_invalid_path(self):
        """Test RAG indexing with invalid path."""
        metrics = UnifiedQualityMetrics(image_id="test_fail")

        # Mock Path.mkdir to simulate a permission error (more robust than relying on OS)
        with patch('transformation_portal.pipelines.quality_feedback_bridge.Path.mkdir') as mock_mkdir:
            mock_mkdir.side_effect = PermissionError("Permission denied")

            result = index_quality_metrics_to_rag(
                metrics, "/mock/path/that/fails"
            )

        # Should return False on failure
        assert result is False


# =============================================================================
# Pipeline Integration Tests
# =============================================================================

class TestPipelineCallback:
    """Tests for pipeline callback creation."""

    def test_create_quality_callback(self):
        """Test creation of quality callback for pipeline."""
        callback = create_quality_callback_for_pipeline("luxury_estate")

        assert callable(callback)

    def test_quality_callback_logs_metrics(self, caplog):
        """Test that quality callback logs metrics."""
        import logging

        with caplog.at_level(logging.INFO):
            callback = create_quality_callback_for_pipeline("test_pipeline")

            metrics = UnifiedQualityMetrics(
                image_id="test_log",
                perceptual_composite=85.0,
                heuristic_composite=78.0,
                hybrid_score=82.0,
                targets_summary="✓ All targets met",
            )
            callback(metrics)

        # Should log quality summary
        assert any("Quality Assessment" in record.message for record in caplog.records)


# =============================================================================
# 750 Picacho Preset Tests
# =============================================================================

class TestPicachoPreset:
    """Tests for 750 Picacho preset recognition."""

    def test_picacho_preset_recognition(self, sample_image_np):
        """Test that 750 Picacho preset name is recognized."""
        bridge = QualityFeedbackBridge()

        metrics = bridge.assess(
            enhanced=sample_image_np,
            pipeline_config_name="750_picacho",
        )

        assert metrics.pipeline_config_name == "750_picacho"

    def test_picacho_material_targets(self):
        """Test that Picacho-relevant materials are in targets."""
        targets = QualityTargets()

        # 750 Picacho estate has specific materials
        picacho_materials = ["quartzite", "oak", "metal", "glass", "stucco"]
        for material in picacho_materials:
            assert material in targets.material_thresholds


# =============================================================================
# Error Handling Tests
# =============================================================================

class TestErrorHandling:
    """Tests for error handling scenarios."""

    def test_invalid_image_type(self, quality_bridge):
        """Test handling of invalid image type."""
        with pytest.raises(ValueError, match="Unsupported image type"):
            quality_bridge._to_numpy("not_an_image")

    def test_none_image_conversion(self, quality_bridge):
        """Test that None image returns None."""
        result = quality_bridge._to_numpy(None)

        assert result is None

    def test_uint8_image_normalization(self, quality_bridge, sample_image_np):
        """Test that uint8 images are normalized to [0, 1]."""
        uint8_image = (sample_image_np * 255).astype(np.uint8)

        result = quality_bridge._to_numpy(uint8_image)

        assert result.max() <= 1.0
        assert result.dtype == np.float32

    @patch('transformation_portal.pipelines.quality_feedback_bridge._check_torch_available')
    def test_torch_unavailable_fallback(self, mock_torch, sample_image_np):
        """Test graceful fallback when torch is unavailable."""
        mock_torch.return_value = False

        bridge = QualityFeedbackBridge()
        metrics = bridge.assess(enhanced=sample_image_np)

        # Should still produce heuristic metrics
        assert metrics.heuristic_composite > 0


# =============================================================================
# Integration Tests
# =============================================================================

class TestBridgeIntegration:
    """Integration tests for QualityFeedbackBridge."""

    def test_full_assessment_workflow(
        self, sample_image_np, modified_image_np, temp_output_dir
    ):
        """Test complete assessment workflow."""
        # Create bridge with RAG indexing
        callback = create_rag_indexing_callback(str(temp_output_dir))
        bridge = QualityFeedbackBridge(
            hybrid_mode=True,
            enable_material_fidelity=True,
            rag_callback=callback,
        )

        # Perform assessment
        metrics = bridge.assess(
            enhanced=modified_image_np,
            original=sample_image_np,
            image_id="integration_test",
            pipeline_config_name="luxury_estate",
        )

        # Verify metrics
        assert metrics.image_id == "integration_test"
        assert metrics.pipeline_config_name == "luxury_estate"
        assert metrics.heuristic_composite > 0
        assert metrics.hybrid_score > 0
        assert metrics.processing_time_ms > 0

        # Verify RAG document was written
        json_files = list(temp_output_dir.glob("*.json"))
        assert len(json_files) >= 1

    def test_assessment_preserves_image_data(
        self, quality_bridge, sample_image_np
    ):
        """Test that assessment doesn't modify input image."""
        original_copy = sample_image_np.copy()

        quality_bridge.assess(enhanced=sample_image_np)

        np.testing.assert_array_equal(sample_image_np, original_copy)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
