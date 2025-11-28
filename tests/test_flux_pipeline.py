"""Tests for the FLUX diffusion pipeline.

Tests the FLUXPipeline class including the enhance_with_controlnet method
that integrates with FLUXControlNet for structure-preserving enhancement.

Note: These tests mock torch and diffusers to run without ML dependencies.
"""

import sys
from contextlib import contextmanager
import pytest
from unittest.mock import MagicMock, patch


# Create mock torch module before importing flux_pipeline
def create_mock_torch():
    """Create a mock torch module with necessary attributes."""
    mock_torch = MagicMock()
    mock_torch.dtype = type('dtype', (), {})
    mock_torch.bfloat16 = MagicMock()
    mock_torch.cuda.is_available = MagicMock(return_value=False)
    mock_torch.backends.mps.is_available = MagicMock(return_value=False)
    mock_torch.inference_mode = MagicMock(
        return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock())
    )
    mock_torch.Generator = MagicMock()
    return mock_torch


@contextmanager
def mock_ml_environment():
    """Context manager to set up mocked ML environment for testing.

    This centralizes the mock setup to avoid duplication across tests.
    """
    mock_torch = create_mock_torch()
    with patch.dict(sys.modules, {
        'torch': mock_torch,
        'torch.cuda': mock_torch.cuda,
        'torch.backends': mock_torch.backends,
        'torch.backends.mps': mock_torch.backends.mps,
        'diffusers': MagicMock(),
        'cv2': MagicMock(),
        'controlnet_aux': MagicMock(),
    }):
        yield


# Skip all tests if we can't mock torch properly
pytestmark = pytest.mark.skipif(
    'torch' in sys.modules,
    reason="Tests require mocked torch environment"
)


@pytest.fixture(scope="module", autouse=True)
def mock_ml_modules():
    """Mock ML modules before any flux_pipeline imports."""
    mock_torch = create_mock_torch()
    mock_diffusers = MagicMock()
    mock_cv2 = MagicMock()
    mock_controlnet_aux = MagicMock()

    with patch.dict(sys.modules, {
        'torch': mock_torch,
        'torch.cuda': mock_torch.cuda,
        'torch.backends': mock_torch.backends,
        'torch.backends.mps': mock_torch.backends.mps,
        'diffusers': mock_diffusers,
        'cv2': mock_cv2,
        'controlnet_aux': mock_controlnet_aux,
    }):
        # Force reload of diffusion modules
        if 'transformation_portal.diffusion' in sys.modules:
            del sys.modules['transformation_portal.diffusion']
        if 'transformation_portal.diffusion.flux_pipeline' in sys.modules:
            del sys.modules['transformation_portal.diffusion.flux_pipeline']
        if 'transformation_portal.diffusion.flux_controlnet' in sys.modules:
            del sys.modules['transformation_portal.diffusion.flux_controlnet']

        yield


class TestBuildControlNetPrompt:
    """Test the _build_controlnet_prompt helper method."""

    def test_build_controlnet_prompt_depth(self, mock_ml_modules):
        """Test prompt building for depth control type."""
        with mock_ml_environment():
            from transformation_portal.diffusion.flux_pipeline import FLUXPipeline

            # Create mock instance to call instance method
            mock_self = MagicMock(spec=FLUXPipeline)

            prompt = FLUXPipeline._build_controlnet_prompt(
                mock_self,
                "luxury kitchen",
                "depth",
                0.7
            )

            assert "luxury kitchen" in prompt
            assert "preserve spatial depth" in prompt
            assert "maintain perspective" in prompt

    def test_build_controlnet_prompt_canny(self, mock_ml_modules):
        """Test prompt building for canny control type."""
        with mock_ml_environment():
            from transformation_portal.diffusion.flux_pipeline import FLUXPipeline
            mock_self = MagicMock(spec=FLUXPipeline)

            prompt = FLUXPipeline._build_controlnet_prompt(
                mock_self,
                "modern bathroom",
                "canny",
                0.7
            )

            assert "modern bathroom" in prompt
            assert "preserve edges" in prompt
            assert "architectural lines" in prompt

    def test_build_controlnet_prompt_normal(self, mock_ml_modules):
        """Test prompt building for normal map control type."""
        with mock_ml_environment():
            from transformation_portal.diffusion.flux_pipeline import FLUXPipeline
            mock_self = MagicMock(spec=FLUXPipeline)

            prompt = FLUXPipeline._build_controlnet_prompt(
                mock_self,
                "luxury bedroom",
                "normal",
                0.7
            )

            assert "luxury bedroom" in prompt
            assert "preserve surface geometry" in prompt
            assert "material details" in prompt

    def test_build_controlnet_prompt_high_scale(self, mock_ml_modules):
        """Test prompt building with high conditioning scale (strict)."""
        with mock_ml_environment():
            from transformation_portal.diffusion.flux_pipeline import FLUXPipeline
            mock_self = MagicMock(spec=FLUXPipeline)

            prompt = FLUXPipeline._build_controlnet_prompt(
                mock_self,
                "test prompt",
                "depth",
                0.9  # High scale
            )

            assert "strictly" in prompt

    def test_build_controlnet_prompt_low_scale(self, mock_ml_modules):
        """Test prompt building with low conditioning scale (subtle)."""
        with mock_ml_environment():
            from transformation_portal.diffusion.flux_pipeline import FLUXPipeline
            mock_self = MagicMock(spec=FLUXPipeline)

            prompt = FLUXPipeline._build_controlnet_prompt(
                mock_self,
                "test prompt",
                "depth",
                0.3  # Low scale
            )

            assert "subtly" in prompt

    def test_build_controlnet_prompt_unknown_type(self, mock_ml_modules):
        """Test prompt building falls back for unknown control types."""
        with mock_ml_environment():
            from transformation_portal.diffusion.flux_pipeline import FLUXPipeline
            mock_self = MagicMock(spec=FLUXPipeline)

            prompt = FLUXPipeline._build_controlnet_prompt(
                mock_self,
                "test prompt",
                "unknown_type",
                0.7
            )

            assert "test prompt" in prompt
            assert "preserve structure" in prompt


class TestFLUXPipelineModuleStructure:
    """Test FLUX pipeline module structure and attributes."""

    def test_flux_available_flag_exists(self, mock_ml_modules):
        """Verify FLUX_AVAILABLE flag is defined."""
        with mock_ml_environment():
            from transformation_portal.diffusion import flux_pipeline
            assert hasattr(flux_pipeline, 'FLUX_AVAILABLE')

    def test_flux_pipeline_class_exists(self, mock_ml_modules):
        """Verify FLUXPipeline class is defined."""
        with mock_ml_environment():
            from transformation_portal.diffusion import flux_pipeline
            assert hasattr(flux_pipeline, 'FLUXPipeline')

    def test_enhance_with_controlnet_method_exists(self, mock_ml_modules):
        """Verify enhance_with_controlnet method is defined."""
        with mock_ml_environment():
            from transformation_portal.diffusion.flux_pipeline import FLUXPipeline
            assert hasattr(FLUXPipeline, 'enhance_with_controlnet')
            assert callable(getattr(FLUXPipeline, 'enhance_with_controlnet'))

    def test_variants_dict_exists(self, mock_ml_modules):
        """Verify VARIANTS dict is defined."""
        with mock_ml_environment():
            from transformation_portal.diffusion.flux_pipeline import FLUXPipeline
            assert hasattr(FLUXPipeline, 'VARIANTS')
            assert 'dev' in FLUXPipeline.VARIANTS
            assert 'schnell' in FLUXPipeline.VARIANTS

    def test_default_prompts_exist(self, mock_ml_modules):
        """Verify default prompts are defined."""
        with mock_ml_environment():
            from transformation_portal.diffusion.flux_pipeline import FLUXPipeline
            assert hasattr(FLUXPipeline, 'DEFAULT_ARCHITECTURAL_PROMPT')
            assert hasattr(FLUXPipeline, 'DEFAULT_NEGATIVE_PROMPT')
            assert len(FLUXPipeline.DEFAULT_ARCHITECTURAL_PROMPT) > 0
            assert len(FLUXPipeline.DEFAULT_NEGATIVE_PROMPT) > 0
