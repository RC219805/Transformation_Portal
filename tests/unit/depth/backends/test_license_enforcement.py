"""Unit tests for depth backend license enforcement.

Tests multi-layer license enforcement (config, registry, runtime).
See ADR-019 for architectural context.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from transformation_portal.depth.backends import DepthBackendRegistry, DepthResult, LicenseRestrictionError, LicenseType


class MockEnhanceConfig:
    """Mock EnhanceConfig for testing."""

    def __init__(
        self,
        non_commercial_ok: bool = False,
        accept_apple_depth_pro_research_license: bool = False,
        depth_device: str = "cpu",
        depth_pro_checkpoint_path: str = None,
        depth_pro_python_executable: str = None,
    ):
        self.non_commercial_ok = non_commercial_ok
        self.accept_apple_depth_pro_research_license = accept_apple_depth_pro_research_license
        self.depth_device = depth_device
        self.depth_pro_checkpoint_path = depth_pro_checkpoint_path
        self.depth_pro_python_executable = depth_pro_python_executable


@pytest.mark.unit
class TestLicenseEnforcement:
    """Test multi-layer license enforcement for Depth Pro."""

    def test_depth_pro_requires_non_commercial_ok(self):
        """Layer 2: Registry should reject depth_pro without non_commercial_ok."""
        registry = DepthBackendRegistry()
        config = MockEnhanceConfig(
            non_commercial_ok=False,
            accept_apple_depth_pro_research_license=True,
        )

        with pytest.raises(LicenseRestrictionError) as exc_info:
            registry.get_backend("depth_pro", config)

        assert "non_commercial_ok=True" in str(exc_info.value)
        assert "research-only" in str(exc_info.value).lower()

    def test_depth_pro_requires_explicit_license_acceptance(self):
        """Layer 2: Registry should reject depth_pro without explicit license flag."""
        registry = DepthBackendRegistry()
        config = MockEnhanceConfig(
            non_commercial_ok=True,
            accept_apple_depth_pro_research_license=False,
        )

        with pytest.raises(LicenseRestrictionError) as exc_info:
            registry.get_backend("depth_pro", config)

        assert "accept_apple_depth_pro_research_license=True" in str(exc_info.value)
        assert "Apple Machine Learning Research License" in str(exc_info.value)

    def test_depth_pro_requires_both_flags(self):
        """Layer 2: Registry requires BOTH license flags."""
        registry = DepthBackendRegistry()

        # Neither flag set
        config = MockEnhanceConfig(
            non_commercial_ok=False,
            accept_apple_depth_pro_research_license=False,
        )
        with pytest.raises(LicenseRestrictionError):
            registry.get_backend("depth_pro", config)

        # Only non_commercial_ok
        config = MockEnhanceConfig(
            non_commercial_ok=True,
            accept_apple_depth_pro_research_license=False,
        )
        with pytest.raises(LicenseRestrictionError):
            registry.get_backend("depth_pro", config)

        # Only apple license flag (non_commercial_ok=False)
        config = MockEnhanceConfig(
            non_commercial_ok=False,
            accept_apple_depth_pro_research_license=True,
        )
        with pytest.raises(LicenseRestrictionError):
            registry.get_backend("depth_pro", config)

    def test_depth_pro_accepts_both_flags(self):
        """Layer 2: Registry accepts depth_pro with both flags True."""
        registry = DepthBackendRegistry()
        config = MockEnhanceConfig(
            non_commercial_ok=True,
            accept_apple_depth_pro_research_license=True,
        )

        # Should not raise (backend instantiation may fail for other reasons)
        try:
            backend = registry.get_backend("depth_pro", config)
            assert backend.name == "depth_pro"
            assert backend.license_type == LicenseType.RESEARCH_ONLY
        except (ImportError, FileNotFoundError):
            # Expected if depth_pro package or checkpoint not installed
            pass

    def test_depth_pro_requires_config(self):
        """Layer 2: Registry requires config for research-only backends."""
        registry = DepthBackendRegistry()

        with pytest.raises(LicenseRestrictionError) as exc_info:
            registry.get_backend("depth_pro", None)

        assert "EnhanceConfig" in str(exc_info.value)

    def test_unknown_backend_error(self):
        """Registry should provide helpful error for unknown backends."""
        registry = DepthBackendRegistry()
        config = MockEnhanceConfig()

        with pytest.raises(ValueError) as exc_info:
            registry.get_backend("nonexistent_backend", config)

        assert "Unknown depth backend" in str(exc_info.value)
        assert "nonexistent_backend" in str(exc_info.value)


@pytest.mark.unit
class TestDepthProBackendUnit:
    """Unit tests for DepthProBackend (mocked, no model download)."""

    def test_backend_attributes(self):
        """Verify DepthProBackend protocol attributes."""
        from transformation_portal.depth.backends.depth_pro import DepthProBackend

        assert DepthProBackend.name == "depth_pro"
        assert DepthProBackend.license_type == LicenseType.RESEARCH_ONLY
        assert DepthProBackend.requires_checkpoint is True

    def test_checkpoint_path_resolution_from_config(self):
        """Checkpoint path should be resolved from config."""
        from transformation_portal.depth.backends.depth_pro import DepthProBackend

        config = MockEnhanceConfig(
            non_commercial_ok=True,
            accept_apple_depth_pro_research_license=True,
            depth_pro_checkpoint_path="/custom/path/checkpoint.pt",
        )

        backend = DepthProBackend(config)
        assert str(backend._checkpoint_path) == "/custom/path/checkpoint.pt"

    def test_checkpoint_path_resolution_from_env(self):
        """Checkpoint path should be resolved from environment variable."""
        import os

        from transformation_portal.depth.backends.depth_pro import DepthProBackend

        config = MockEnhanceConfig(
            non_commercial_ok=True,
            accept_apple_depth_pro_research_license=True,
        )

        with patch.dict(os.environ, {"TRANSFORMATION_PORTAL_DEPTH_PRO_CHECKPOINT": "/env/checkpoint.pt"}):
            backend = DepthProBackend(config)
            assert str(backend._checkpoint_path) == "/env/checkpoint.pt"

    def test_checkpoint_path_default(self):
        """Checkpoint path should default to checkpoints/depth_pro.pt."""
        import os

        from transformation_portal.depth.backends.depth_pro import DepthProBackend

        config = MockEnhanceConfig(
            non_commercial_ok=True,
            accept_apple_depth_pro_research_license=True,
        )

        # Clear env var if set
        with patch.dict(os.environ, {}, clear=True):
            backend = DepthProBackend(config)
            assert backend._checkpoint_path.name == "depth_pro.pt"
            assert "checkpoints" in str(backend._checkpoint_path)

    def test_python_executable_resolution_from_config(self, tmp_path):
        """Dedicated Depth Pro Python path should be resolved from config."""
        from transformation_portal.depth.backends.depth_pro import DepthProBackend

        python_executable = tmp_path / ".venv-depth-pro" / "bin" / "python"
        python_executable.parent.mkdir(parents=True)
        python_executable.write_text("#!/bin/sh\n", encoding="utf-8")

        config = MockEnhanceConfig(
            non_commercial_ok=True,
            accept_apple_depth_pro_research_license=True,
            depth_pro_python_executable=str(python_executable),
        )

        backend = DepthProBackend(config)
        assert backend._python_executable == str(python_executable.resolve())

    def test_ensure_available_missing_package(self):
        """Should raise ImportError if depth_pro package not installed."""
        from transformation_portal.depth.backends.depth_pro import DepthProBackend

        config = MockEnhanceConfig(
            non_commercial_ok=True,
            accept_apple_depth_pro_research_license=True,
        )
        backend = DepthProBackend(config)

        with patch.dict("sys.modules", {"depth_pro": None}):
            with pytest.raises(ImportError) as exc_info:
                backend.ensure_available()

            assert "pip install depth-pro" in str(exc_info.value)

    def test_ensure_available_missing_checkpoint(self):
        """Should raise FileNotFoundError if checkpoint missing."""
        from pathlib import Path

        from transformation_portal.depth.backends.depth_pro import DepthProBackend

        config = MockEnhanceConfig(
            non_commercial_ok=True,
            accept_apple_depth_pro_research_license=True,
            depth_pro_checkpoint_path="/nonexistent/checkpoint.pt",
        )
        backend = DepthProBackend(config)

        with patch.dict("sys.modules", {"depth_pro": MagicMock()}):
            with pytest.raises(FileNotFoundError) as exc_info:
                backend.ensure_available()

            assert "not found" in str(exc_info.value).lower()
            assert "curl" in str(exc_info.value)  # Download instructions

    def test_subprocess_ensure_available_skips_local_package_import(self, tmp_path):
        """Dedicated subprocess mode should not require local depth_pro imports."""
        from transformation_portal.depth.backends.depth_pro import DepthProBackend

        checkpoint_path = tmp_path / "depth_pro.pt"
        checkpoint_path.write_bytes(b"checkpoint")
        python_executable = tmp_path / ".venv-depth-pro" / "bin" / "python"
        python_executable.parent.mkdir(parents=True)
        python_executable.write_text("#!/bin/sh\n", encoding="utf-8")

        config = MockEnhanceConfig(
            non_commercial_ok=True,
            accept_apple_depth_pro_research_license=True,
            depth_pro_checkpoint_path=str(checkpoint_path),
            depth_pro_python_executable=str(python_executable),
        )
        backend = DepthProBackend(config)

        with patch.object(
            backend,
            "_ensure_local_package_available",
            side_effect=AssertionError("local package import should not be used"),
        ):
            with patch("transformation_portal.depth.backends.depth_pro.subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
                backend.ensure_available()

        command = mock_run.call_args.args[0]
        assert "--check" in command
        assert str(checkpoint_path.resolve()) in command

    def test_subprocess_compute_returns_depth_result(self, tmp_path):
        """Subprocess worker output should map back to the DepthResult contract."""
        from transformation_portal.depth.backends.depth_pro import DepthProBackend

        checkpoint_path = tmp_path / "depth_pro.pt"
        checkpoint_path.write_bytes(b"checkpoint")
        python_executable = tmp_path / ".venv-depth-pro" / "bin" / "python"
        python_executable.parent.mkdir(parents=True)
        python_executable.write_text("#!/bin/sh\n", encoding="utf-8")

        config = MockEnhanceConfig(
            non_commercial_ok=True,
            accept_apple_depth_pro_research_license=True,
            depth_pro_checkpoint_path=str(checkpoint_path),
            depth_pro_python_executable=str(python_executable),
        )
        backend = DepthProBackend(config)

        def fake_run(command, **kwargs):
            if "--check" in command:
                return MagicMock(returncode=0, stdout="", stderr="")

            output_depth_path = Path(command[command.index("--output-depth") + 1])
            output_json_path = Path(command[command.index("--output-json") + 1])
            np.save(output_depth_path, np.full((4, 5), 2.5, dtype=np.float32), allow_pickle=False)
            output_json_path.write_text(
                json.dumps(
                    {
                        "depth_units": "meters",
                        "device": "cpu",
                        "dtype": "float32",
                        "input_size": [4, 5],
                        "focal_length_px": 525.0,
                        "field_of_view_deg": 65.0,
                        "provenance": {
                            "checkpoint": {
                                "path": str(checkpoint_path),
                                "sha256": "a" * 64,
                                "bytes": checkpoint_path.stat().st_size,
                            }
                        },
                        "warnings": ["isolated depth env"],
                    }
                ),
                encoding="utf-8",
            )
            return MagicMock(returncode=0, stdout="", stderr="")

        with patch("transformation_portal.depth.backends.depth_pro.subprocess.run", side_effect=fake_run):
            result = backend.compute(np.zeros((4, 5, 3), dtype=np.uint8))

        assert result.depth_units == "meters"
        assert result.depth_map.shape == (4, 5)
        assert result.focal_length_px == 525.0
        assert result.field_of_view_deg == 65.0
        assert result.metadata["runner"]["mode"] == "subprocess"
        assert result.metadata["checkpoint"]["sha256"] == "a" * 64
        assert result.warnings == ["isolated depth env"]

    def test_runtime_license_validation(self):
        """Layer 3: Runtime validation should reject missing flags."""
        from transformation_portal.depth.backends.depth_pro import DepthProBackend

        # Config without flags
        config_no_flags = MockEnhanceConfig(
            non_commercial_ok=False,
            accept_apple_depth_pro_research_license=False,
        )

        backend = DepthProBackend.__new__(DepthProBackend)
        backend._config = config_no_flags

        with pytest.raises(LicenseRestrictionError):
            backend._validate_license_runtime()

    def test_runtime_license_validation_passes(self):
        """Layer 3: Runtime validation should pass with both flags."""
        from transformation_portal.depth.backends.depth_pro import DepthProBackend

        config = MockEnhanceConfig(
            non_commercial_ok=True,
            accept_apple_depth_pro_research_license=True,
        )

        backend = DepthProBackend.__new__(DepthProBackend)
        backend._config = config

        # Should not raise
        backend._validate_license_runtime()


@pytest.mark.unit
class TestDepthResult:
    """Test DepthResult dataclass."""

    def test_depth_result_defaults(self):
        """DepthResult should have backward-compatible defaults."""
        result = DepthResult(
            depth_map=np.zeros((100, 100), dtype=np.float32),
            original_image=np.zeros((100, 100, 3), dtype=np.uint8),
            metadata={},
        )

        assert result.depth_units == "relative"
        assert result.focal_length_px is None
        assert result.field_of_view_deg is None
        assert result.is_metric is False

    def test_depth_result_metric(self):
        """DepthResult with metric depth."""
        result = DepthResult(
            depth_map=np.random.rand(100, 100).astype(np.float32) * 10,  # 0-10 meters
            original_image=np.zeros((100, 100, 3), dtype=np.uint8),
            metadata={"engine": "depth_pro"},
            depth_units="meters",
            focal_length_px=525.0,
            field_of_view_deg=65.0,
            backend_id="depth_pro",
            device="mps",
        )

        assert result.depth_units == "meters"
        assert result.is_metric is True
        assert result.focal_length_px == 525.0
        assert result.field_of_view_deg == 65.0
        assert result.backend_id == "depth_pro"

    def test_depth_result_to_relative(self):
        """Converting metric depth to relative should normalize values."""
        # Create metric depth with known values
        depth_map = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

        result = DepthResult(
            depth_map=depth_map,
            original_image=np.zeros((2, 2, 3), dtype=np.uint8),
            metadata={},
            depth_units="meters",
        )

        relative = result.to_relative()

        assert relative.depth_units == "relative"
        assert relative.is_metric is False
        assert relative.depth_map.min() >= 0.0
        assert relative.depth_map.max() <= 1.0
        assert "converted from metric to relative" in relative.warnings

    def test_depth_result_to_relative_already_relative(self):
        """Converting relative depth should return same result."""
        result = DepthResult(
            depth_map=np.random.rand(10, 10).astype(np.float32),
            original_image=np.zeros((10, 10, 3), dtype=np.uint8),
            metadata={},
            depth_units="relative",
        )

        relative = result.to_relative()

        assert relative is result  # Same object

    def test_depth_property_alias(self):
        """depth property should be alias for depth_map."""
        depth_map = np.random.rand(50, 50).astype(np.float32)
        result = DepthResult(
            depth_map=depth_map,
            original_image=np.zeros((50, 50, 3), dtype=np.uint8),
            metadata={},
        )

        assert np.array_equal(result.depth, result.depth_map)


@pytest.mark.unit
class TestDepthBackendRegistry:
    """Test DepthBackendRegistry."""

    def test_list_backends(self):
        """Registry should list available backends."""
        registry = DepthBackendRegistry()
        backends = registry.list_backends()

        # At minimum, depth_pro should be registered
        assert "depth_pro" in backends
        assert backends["depth_pro"]["license_type"] == "research_only"
        assert backends["depth_pro"]["requires_checkpoint"] is True

    def test_register_custom_backend(self):
        """Registry should allow custom backend registration."""

        class CustomBackend:
            name = "custom_test_backend"
            license_type = LicenseType.COMMERCIAL
            requires_checkpoint = False

            def __init__(self, config=None):
                pass

            def compute(self, image, device=None):
                return DepthResult(
                    depth_map=np.zeros((10, 10), dtype=np.float32),
                    original_image=np.array(image) if hasattr(image, "__array__") else image,
                    metadata={},
                )

            def get_cache_key(self, image):
                return "custom_key"

            def ensure_available(self):
                pass

        registry = DepthBackendRegistry()
        registry.register_backend(CustomBackend)

        backends = registry.list_backends()
        assert "custom_test_backend" in backends
        assert backends["custom_test_backend"]["license_type"] == "commercial"
