"""
CI guardrail tests for depth contract enforcement.

These tests ensure the depth contract (REQUIRED/AUTO/OPTIONAL) cannot regress.
They run fast by stubbing depth inference to avoid loading ML models.
"""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch
from lux_depth_v2.config import PipelineConfig, Preset, DepthMode
from lux_depth_v2.pipeline import LuxPipelineV2


class TestDepthContractGuardrails:
    """CI guardrails to prevent depth contract regression."""

    def test_required_mode_fails_without_depth(self, temp_dir, sample_image_file):
        """REQUIRED mode must raise FileNotFoundError when depth is missing."""
        # Setup config with REQUIRED mode (APEX preset)
        config = PipelineConfig(
            preset=Preset.INTERIOR_LUXURY_APEX_QUALITY,
            input_dir=temp_dir,
            output_dir=temp_dir,
        )
        # Verify preset applied REQUIRED mode
        assert config.depth.mode == DepthMode.REQUIRED

        pipeline = LuxPipelineV2(config)

        # Must raise when depth is missing
        with pytest.raises(FileNotFoundError, match="Depth required but missing"):
            pipeline.process_one(sample_image_file, depth_path=None)

    def test_auto_mode_generates_depth_with_provenance(self, temp_dir, sample_image_file):
        """AUTO mode must generate depth and report source='generated'."""
        # Setup config with AUTO mode
        config = PipelineConfig(
            preset=Preset.PRODUCTION_STANDARD,
            input_dir=temp_dir,
            output_dir=temp_dir,
            write_outputs=False,  # Skip file writes for speed
        )
        assert config.depth.mode == DepthMode.AUTO

        # Stub depth inference to avoid loading ML models
        mock_estimator = Mock()
        mock_estimator.estimate_depth.return_value = np.random.rand(64, 64).astype(np.float32)
        mock_estimator.compute_edge_alignment.return_value = 0.85

        with patch(
            "lux_depth_v2.depth_inference.create_tiled_estimator",
            return_value=mock_estimator,
        ):
            pipeline = LuxPipelineV2(config)
            result = pipeline.process_one(sample_image_file, depth_path=None)

        # Must succeed and report generated provenance
        assert result["status"] == "ok"
        assert result["depth"]["source"] == "generated"
        assert result["depth"]["model"] == config.depth.auto_model
        assert result["depth"]["confidence_proxy"] is not None

    def test_optional_mode_allows_missing_with_provenance(self, temp_dir, sample_image_file):
        """OPTIONAL mode must proceed without depth and report source='missing'."""
        # Setup config with OPTIONAL mode
        config = PipelineConfig(
            preset=Preset.CI_BASELINE,
            input_dir=temp_dir,
            output_dir=temp_dir,
            write_outputs=False,
        )
        assert config.depth.mode == DepthMode.OPTIONAL

        pipeline = LuxPipelineV2(config)
        result = pipeline.process_one(sample_image_file, depth_path=None)

        # Must succeed and report missing provenance
        assert result["status"] == "ok"
        assert result["depth"]["source"] == "missing"
        assert result["depth"]["path"] is None

    def test_strict_depth_flag_overrides_mode(self, temp_dir, sample_image_file):
        """strict_depth=True must enforce REQUIRED behavior regardless of mode."""
        # Setup with AUTO mode but strict_depth=True
        config = PipelineConfig(
            preset=Preset.PRODUCTION_STANDARD,
            input_dir=temp_dir,
            output_dir=temp_dir,
            strict_depth=True,  # Override AUTO mode
        )
        assert config.depth.mode == DepthMode.AUTO
        assert config.strict_depth is True

        pipeline = LuxPipelineV2(config)

        # Must raise even though mode is AUTO
        with pytest.raises(FileNotFoundError, match="strict_depth=True"):
            pipeline.process_one(sample_image_file, depth_path=None)

    def test_auto_model_is_plumbed_to_estimator(self, temp_dir, sample_image_file):
        """Regression test: cfg.depth.auto_model must reach TiledDepthEstimator."""
        custom_model = "depth-anything/Depth-Anything-V2-Small-hf"
        config = PipelineConfig(
            preset=Preset.PRODUCTION_STANDARD,
            input_dir=temp_dir,
            output_dir=temp_dir,
            write_outputs=False,
        )
        config.depth.auto_model = custom_model  # Override default

        captured_kwargs = {}

        def spy_factory(**kwargs):
            captured_kwargs.update(kwargs)
            mock_estimator = Mock()
            mock_estimator.estimate_depth.return_value = np.random.rand(64, 64).astype(np.float32)
            mock_estimator.compute_edge_alignment.return_value = 0.80
            return mock_estimator

        with patch(
            "lux_depth_v2.depth_inference.create_tiled_estimator",
            side_effect=spy_factory,
        ):
            pipeline = LuxPipelineV2(config)
            result = pipeline.process_one(sample_image_file, depth_path=None)

        # Verify pipeline completed successfully
        assert result["status"] == "ok", f"process_one failed: {result}"

        # Verify model_name was passed to factory
        assert "model_name" in captured_kwargs, "model_name not passed to create_tiled_estimator"
        assert captured_kwargs["model_name"] == custom_model, f"Expected {custom_model}, got {captured_kwargs['model_name']}"

    def test_auto_mode_importerror_does_not_raise_when_not_strict(self, temp_dir, sample_image_file):
        """AUTO mode: if depth auto-generation fails and strict_depth=False, pipeline continues with depth=None."""
        config = PipelineConfig(
            preset=Preset.PRODUCTION_STANDARD,  # DepthMode.AUTO
            input_dir=temp_dir,
            output_dir=temp_dir,
        )
        assert config.depth.mode == DepthMode.AUTO
        config.strict_depth = False

        def mock_factory_failure(**kwargs):
            raise ImportError("transformers not installed")

        with patch(
            "lux_depth_v2.depth_inference.create_tiled_estimator",
            side_effect=mock_factory_failure,
        ):
            pipeline = LuxPipelineV2(config)
            result = pipeline.process_one(sample_image_file, depth_path=None)

        assert result["status"] == "ok"
        assert result["depth"]["source"] == "error"
        assert "ImportError" in result["depth"]["error"]

    def test_auto_mode_missing_depth_fails_fast_when_strict_depth_true(self, temp_dir, sample_image_file):
        """AUTO mode: if strict_depth=True, depth must be provided (fail-fast before auto-generation)."""
        config = PipelineConfig(
            preset=Preset.PRODUCTION_STANDARD,  # DepthMode.AUTO
            input_dir=temp_dir,
            output_dir=temp_dir,
        )
        assert config.depth.mode == DepthMode.AUTO
        config.strict_depth = True

        pipeline = LuxPipelineV2(config)

        # strict_depth=True forces fail-fast at depth resolution (before auto-generation attempt)
        with pytest.raises(FileNotFoundError, match="Depth required but missing.*strict_depth=True"):
            pipeline.process_one(sample_image_file, depth_path=None)
