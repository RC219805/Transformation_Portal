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
        
        with patch('lux_depth_v2.depth_inference.create_tiled_estimator', return_value=mock_estimator):
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
