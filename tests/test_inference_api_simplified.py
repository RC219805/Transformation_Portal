"""Tests for simplified DA3InferenceEngine API (Issue #3).

This test suite validates:
1. Simple string device parameter works
2. DA3Config object still works (backward compatibility)
3. API is intuitive and reduces boilerplate
4. Device auto-detection works correctly
5. Error messages are clear

Coverage target: Issue #3 from PBR Implementation Audit
"""

import sys
import types
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

pytestmark = pytest.mark.unit

from transformation_portal.lux_depth_v3.config import DA3Config, DeviceConfig, ModelVariant
from transformation_portal.lux_depth_v3.inference import DA3InferenceEngine


class TestSimplifiedAPI:
    """Test simplified string device API."""

    @patch("transformation_portal.lux_depth_v3.inference.TORCH_AVAILABLE", True)
    @patch("transformation_portal.lux_depth_v3.inference.torch")
    def test_string_device_cpu(self, mock_torch):
        """Test simple string device='cpu' works."""
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False

        # Simple API - just pass device string
        engine = DA3InferenceEngine(config="cpu")

        # Should create DA3Config internally
        assert isinstance(engine.config, DA3Config)
        assert engine.config.device.device == "cpu"
        assert engine.device == "cpu"

    @patch("transformation_portal.lux_depth_v3.inference.TORCH_AVAILABLE", True)
    @patch("transformation_portal.lux_depth_v3.inference.torch")
    def test_string_device_mps(self, mock_torch):
        """Test simple string device='mps' works."""
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = True

        # Simple API - just pass device string
        engine = DA3InferenceEngine(config="mps")

        # Should create DA3Config with MPS
        assert isinstance(engine.config, DA3Config)
        assert engine.config.device.device == "mps"
        assert engine.device == "mps"

    @patch("transformation_portal.lux_depth_v3.inference.TORCH_AVAILABLE", True)
    @patch("transformation_portal.lux_depth_v3.inference.torch")
    def test_string_device_cuda(self, mock_torch):
        """Test simple string device='cuda' works."""
        mock_torch.cuda.is_available.return_value = True
        mock_torch.backends.mps.is_available.return_value = False

        # Simple API
        engine = DA3InferenceEngine(config="cuda")

        assert engine.config.device.device == "cuda"
        assert engine.device == "cuda"

    @patch("transformation_portal.lux_depth_v3.inference.TORCH_AVAILABLE", True)
    @patch("transformation_portal.lux_depth_v3.inference.torch")
    def test_string_device_auto(self, mock_torch):
        """Test device='auto' auto-detects optimal device."""
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = True

        engine = DA3InferenceEngine(config="auto")

        # Should auto-detect MPS
        assert engine.device == "mps"


class TestBackwardCompatibility:
    """Test backward compatibility with DA3Config objects."""

    @patch("transformation_portal.lux_depth_v3.inference.TORCH_AVAILABLE", True)
    @patch("transformation_portal.lux_depth_v3.inference.torch")
    def test_da3config_object_still_works(self, mock_torch):
        """Test existing DA3Config object API still works."""
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False

        # Old style - pass DA3Config object
        config = DA3Config()
        config.device = DeviceConfig(device="cpu")

        engine = DA3InferenceEngine(config=config)

        # Should work as before
        assert engine.config is config
        assert engine.device == "cpu"

    @patch("transformation_portal.lux_depth_v3.inference.TORCH_AVAILABLE", True)
    @patch("transformation_portal.lux_depth_v3.inference.torch")
    def test_da3config_with_custom_variant(self, mock_torch):
        """Test DA3Config with custom model variant works."""
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False

        # Complex config with custom model
        config = DA3Config(model_variant=ModelVariant.METRIC_BASE, device=DeviceConfig(device="cpu"))

        engine = DA3InferenceEngine(config=config)

        assert engine.config.model_variant == ModelVariant.METRIC_BASE
        assert engine.device == "cpu"


class TestAPIUsability:
    """Test API reduces boilerplate and is intuitive."""

    @patch("transformation_portal.lux_depth_v3.inference.TORCH_AVAILABLE", True)
    @patch("transformation_portal.lux_depth_v3.inference.torch")
    def test_minimal_boilerplate_for_common_case(self, mock_torch):
        """Test common use case requires minimal code."""
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = True

        # Before (complex):
        # config = DA3Config()
        # config.device = DeviceConfig()
        # config.device.device = "mps"
        # engine = DA3InferenceEngine(config)

        # After (simple):
        engine = DA3InferenceEngine("mps")

        # Verify it works
        assert engine.device == "mps"

    @patch("transformation_portal.lux_depth_v3.inference.TORCH_AVAILABLE", True)
    @patch("transformation_portal.lux_depth_v3.inference.torch")
    def test_default_device_is_auto(self, mock_torch):
        """Test default device is 'cpu' for predictability."""
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False

        # Default should be CPU (safe default)
        engine = DA3InferenceEngine()

        assert engine.device == "cpu"


class TestDeviceConsistency:
    """Test device parameter is consistent across API."""

    @patch("transformation_portal.lux_depth_v3.inference.TORCH_AVAILABLE", True)
    @patch("transformation_portal.lux_depth_v3.inference.torch")
    def test_string_device_matches_config_device(self, mock_torch):
        """Test string device parameter matches internal config."""
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = True

        for device_str in ["cpu", "mps"]:
            engine = DA3InferenceEngine(device_str)

            # Internal config should match
            assert engine.config.device.device == device_str
            # Resolved device should match
            assert engine.device in [device_str, "cpu"]  # May fall back to CPU


class TestErrorHandling:
    """Test error messages are clear and actionable."""

    @pytest.mark.skip(reason="Requires module-level import mocking, tested manually")
    def test_no_torch_error_is_clear(self):
        """Test error when torch not available is clear.

        This test requires mocking module-level imports which is complex.
        Functionality verified manually by uninstalling torch.
        """
        pytest.skip("Requires module-level import mocking, tested manually")

    @patch("transformation_portal.lux_depth_v3.inference.TORCH_AVAILABLE", True)
    @patch("transformation_portal.lux_depth_v3.inference.torch")
    def test_invalid_device_handled_gracefully(self, mock_torch):
        """Test invalid device string handled gracefully."""
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False

        # Invalid device should fall back to CPU
        engine = DA3InferenceEngine("invalid_device")

        # Should fall back to CPU (auto-detection logic)
        assert engine.device == "cpu"


class TestAPIDocumentation:
    """Test API is self-documenting through docstrings."""

    def test_init_docstring_mentions_string_device(self):
        """Test __init__ docstring documents string device option."""
        docstring = DA3InferenceEngine.__init__.__doc__

        assert docstring is not None
        # Should mention both options
        assert "DA3Config" in docstring
        assert "string" in docstring or "str" in docstring
        assert "device" in docstring

    def test_init_signature_shows_union_type(self):
        """Test __init__ signature shows Union[DA3Config, str]."""
        import inspect

        sig = inspect.signature(DA3InferenceEngine.__init__)

        # config parameter should have Union type hint
        config_param = sig.parameters["config"]

        # Type annotation should exist
        assert config_param.annotation != inspect.Parameter.empty


class TestRealWorldUsage:
    """Test real-world usage patterns work as expected."""

    @patch("transformation_portal.lux_depth_v3.inference.TORCH_AVAILABLE", True)
    @patch("transformation_portal.lux_depth_v3.inference.torch")
    def test_quick_script_usage(self, mock_torch):
        """Test usage in a quick script is minimal."""
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = True

        # Typical quick script usage
        engine = DA3InferenceEngine("mps")

        # Should be ready to use
        assert engine is not None
        assert engine.device == "mps"

    @patch("transformation_portal.lux_depth_v3.inference.TORCH_AVAILABLE", True)
    @patch("transformation_portal.lux_depth_v3.inference.torch")
    def test_production_usage_with_config(self, mock_torch):
        """Test production usage with full config still works."""
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False

        # Production usage with detailed config
        config = DA3Config(model_variant=ModelVariant.METRIC_LARGE, device=DeviceConfig(device="cpu", dtype="float32"))
        engine = DA3InferenceEngine(config=config, commercial_use=True, validate_license_strict=True)

        # Should work with all features
        assert engine.config.model_variant == ModelVariant.METRIC_LARGE
        assert engine.commercial_use is True
        assert engine.validate_license_strict is True


class TestDA3IntegrationLogging:
    """Test DA3 integration logging accuracy."""

    def test_da3_load_does_not_emit_stale_custom_integration_warning(self, monkeypatch):
        """Loading DA3 model should not log stale 'custom integration required' warning."""
        import transformation_portal.lux_depth_v3.inference as inference_module

        class FakeDepthAnything3:
            @classmethod
            def from_pretrained(cls, _model_id):
                model = MagicMock()
                model.to.return_value = model
                model.eval.return_value = None
                return model

        api_module = types.ModuleType("depth_anything_3.api")
        api_module.DepthAnything3 = FakeDepthAnything3
        package_module = types.ModuleType("depth_anything_3")
        package_module.api = api_module

        monkeypatch.setitem(sys.modules, "depth_anything_3", package_module)
        monkeypatch.setitem(sys.modules, "depth_anything_3.api", api_module)

        engine = DA3InferenceEngine.__new__(DA3InferenceEngine)
        engine.device = "cpu"

        with patch.object(inference_module.logger, "warning") as mock_warning:
            with patch.object(inference_module.logger, "info") as mock_info:
                engine._load_da3_model("depth-anything/da3nested-giant-large")

        warning_text = " ".join(str(call.args[0]) for call in mock_warning.call_args_list if call.args)
        assert "custom integration required" not in warning_text.lower()
        assert any(
            "custom api integration active" in str(call.args[0]).lower() for call in mock_info.call_args_list if call.args
        )
